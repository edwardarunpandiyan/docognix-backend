"""
services/retrieval.py – Hybrid retrieval pipeline.

Steps:
  1. HyDE  – Generate a hypothetical answer → embed it for better recall.
  2. Vector search  – Top-K candidates via pgvector cosine similarity.
  3. Keyword search – BM25-style full-text scoring via PostgreSQL tsvector.
  4. RRF  – Combine both ranked lists via Reciprocal Rank Fusion.
  5. Confidence  – Assign high/medium/low based on top score.
"""
from __future__ import annotations

import logging
from uuid import UUID

from config import settings
from database.postgres import get_pool
from models.documents import SourceReference
from services.embedding import embed_query, cosine_similarity
from utils.text_utils import keyword_score, reciprocal_rank_fusion

log = logging.getLogger(__name__)


# ── HyDE: Hypothetical Document Embedding ─────────────────────────────────────

async def generate_hypothetical_answer(query: str, groq_client) -> str:
    """
    Ask the LLM to generate a short hypothetical answer to the query.
    We then embed THAT answer rather than the raw query to improve recall.
    Based on: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao 2022).
    """
    try:
        resp = groq_client.chat.completions.create(
            model=settings.groq_fallback_model,   # fast model for HyDE
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Write a brief, factual passage "
                        "that would directly answer the following question. "
                        "Be concise (2-3 sentences). Do not include disclaimers."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("HyDE generation failed: %s – falling back to raw query", e)
        return query


# ── Vector search ─────────────────────────────────────────────────────────────

async def vector_search(
    session_id: str,
    embedding: list[float],
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    Return top-k chunks ordered by cosine similarity.
    Optionally filtered to specific document_ids.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        if document_ids:
            rows = await conn.fetch(
                """
                SELECT
                    id, document_id, content, page_number, page_end,
                    chunk_index, token_count,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM chunks
                WHERE session_id = $2
                  AND document_id = ANY($3::uuid[])
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                embedding, session_id, document_ids, top_k,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT
                    id, document_id, content, page_number, page_end,
                    chunk_index, token_count,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM chunks
                WHERE session_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding, session_id, top_k,
            )
    return [dict(r) for r in rows]


# ── Keyword search ─────────────────────────────────────────────────────────────

async def keyword_search(
    session_id: str,
    query: str,
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    Full-text PostgreSQL search using tsvector for fast keyword recall.
    Falls back to ilike if the parsed tsquery is empty.
    """
    pool = await get_pool()
    safe_query = " & ".join(
        w for w in query.split() if len(w) >= 3
    ) or query[:50]

    async with pool.acquire() as conn:
        try:
            if document_ids:
                rows = await conn.fetch(
                    """
                    SELECT
                        id, document_id, content, page_number, page_end,
                        chunk_index, token_count,
                        ts_rank_cd(
                            to_tsvector('english', content),
                            plainto_tsquery('english', $1)
                        ) AS rank
                    FROM chunks
                    WHERE session_id = $2
                      AND document_id = ANY($3::uuid[])
                      AND to_tsvector('english', content)
                          @@ plainto_tsquery('english', $1)
                    ORDER BY rank DESC
                    LIMIT $4
                    """,
                    query, session_id, document_ids, top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        id, document_id, content, page_number, page_end,
                        chunk_index, token_count,
                        ts_rank_cd(
                            to_tsvector('english', content),
                            plainto_tsquery('english', $1)
                        ) AS rank
                    FROM chunks
                    WHERE session_id = $2
                      AND to_tsvector('english', content)
                          @@ plainto_tsquery('english', $1)
                    ORDER BY rank DESC
                    LIMIT $3
                    """,
                    query, session_id, top_k,
                )
        except Exception as e:
            log.warning("Keyword search FTS failed: %s – using ilike fallback", e)
            rows = await conn.fetch(
                """
                SELECT id, document_id, content, page_number, page_end,
                       chunk_index, token_count, 0.1 AS rank
                FROM chunks
                WHERE session_id = $1
                  AND content_lower ILIKE $2
                LIMIT $3
                """,
                session_id, f"%{query[:100].lower()}%", top_k,
            )
    return [dict(r) for r in rows]


# ── Document name lookup ───────────────────────────────────────────────────────

async def get_document_names(document_ids: list[str]) -> dict[str, str]:
    if not document_ids:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id::text, original_name FROM documents WHERE id = ANY($1::uuid[])",
            document_ids,
        )
    return {r["id"]: r["original_name"] for r in rows}


# ── Confidence scoring ────────────────────────────────────────────────────────

def score_to_confidence(score: float) -> str:
    if score >= settings.confidence_high:
        return "high"
    if score >= settings.confidence_medium:
        return "medium"
    return "low"


# ── Main hybrid retrieval ─────────────────────────────────────────────────────

async def hybrid_retrieve(
    session_id: str,
    query: str,
    groq_client,
    document_ids: list[UUID] | None = None,
    top_k: int | None = None,
    rerank_n: int | None = None,
) -> tuple[list[SourceReference], float]:
    """
    Full hybrid retrieval pipeline.

    Returns:
      (sources, top_score)
    """
    top_k = top_k or settings.retrieval_top_k
    rerank_n = rerank_n or settings.rerank_top_n
    doc_id_strs = [str(d) for d in document_ids] if document_ids else None

    # ── 1. HyDE ───────────────────────────────────────────────
    if settings.hyde_enabled:
        hyp_answer = await generate_hypothetical_answer(query, groq_client)
        search_text = f"{query} {hyp_answer}"
        log.debug("HyDE hypothetical: %s", hyp_answer[:80])
    else:
        search_text = query

    # ── 2. Embed query ─────────────────────────────────────────
    query_embedding = await embed_query(search_text)

    # ── 3. Vector + Keyword search ─────────────────────────────
    vector_results, keyword_results = await _run_parallel_search(
        session_id, query, query_embedding, top_k, doc_id_strs
    )

    # ── 4. Merge into unified pool ─────────────────────────────
    all_chunks: dict[str, dict] = {}
    for r in vector_results:
        cid = str(r["id"])
        all_chunks[cid] = {**r, "similarity": float(r.get("similarity", 0))}
    for r in keyword_results:
        cid = str(r["id"])
        if cid not in all_chunks:
            all_chunks[cid] = {**r, "similarity": 0.0}
        all_chunks[cid]["keyword_rank_score"] = float(r.get("rank", 0))

    # ── 5. Re-rank via RRF ─────────────────────────────────────
    vector_rank = [str(r["id"]) for r in vector_results]
    keyword_rank = [str(r["id"]) for r in keyword_results]
    rrf_scores = reciprocal_rank_fusion(
        vector_rank, keyword_rank, alpha=settings.hybrid_alpha
    )

    # Enrich with inline keyword score from text
    for cid, chunk in all_chunks.items():
        chunk["kw_score"] = keyword_score(query, chunk["content"])
        # Boost RRF with inline kw score
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + chunk["kw_score"] * 0.05

    ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    top_ids = ranked_ids[:rerank_n]

    if not top_ids:
        return [], 0.0

    # ── 6. Fetch document names ────────────────────────────────
    doc_ids = list({str(all_chunks[cid]["document_id"]) for cid in top_ids})
    doc_names = await get_document_names(doc_ids)

    # ── 7. Build SourceReference list ─────────────────────────
    sources: list[SourceReference] = []
    for cid in top_ids:
        c = all_chunks[cid]
        sim = c.get("similarity", 0.0)
        combined = rrf_scores.get(cid, 0.0)
        doc_id_str = str(c["document_id"])
        sources.append(
            SourceReference(
                chunk_id=c["id"],
                document_id=c["document_id"],
                document_name=doc_names.get(doc_id_str, "Unknown"),
                content=c["content"],
                page_number=c.get("page_number"),
                page_end=c.get("page_end"),
                chunk_index=c["chunk_index"],
                similarity_score=round(sim, 4),
                keyword_score=round(c.get("kw_score", 0.0), 4),
                combined_score=round(combined, 6),
                confidence=score_to_confidence(sim),
            )
        )

    top_score = all_chunks[top_ids[0]].get("similarity", 0.0) if top_ids else 0.0
    return sources, top_score


async def _run_parallel_search(
    session_id, query, embedding, top_k, doc_id_strs
) -> tuple[list[dict], list[dict]]:
    import asyncio
    v_task = vector_search(session_id, embedding, top_k, doc_id_strs)
    k_task = keyword_search(session_id, query, top_k, doc_id_strs)
    return await asyncio.gather(v_task, k_task)

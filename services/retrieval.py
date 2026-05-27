"""
services/retrieval.py – Hybrid retrieval + cross-encoder reranking pipeline.
Uses conversation_id (renamed from session_id) throughout.

Two-stage retrieval
────────────────────
Stage 1 — Bi-encoder recall (fast, high recall):
    BGE-small embeds query and chunks independently.
    Vector search + BM25 keyword search fetch top-K candidates (K=20).
    RRF fuses the two ranked lists into one ordering.

Stage 2 — Cross-encoder precision (accurate, spread-out scores):
    The top-N candidates (N=rerank_top_n) are scored as (query, chunk) pairs
    by a MiniLM cross-encoder.  Because the model reads both together, it
    produces logits that vary dramatically across chunks — even chunks that
    are topically similar score very differently when only one of them
    actually answers the question.
    Logits are passed through sigmoid → [0, 1] before display.

`similarity_score` on SourceReference now holds the cross-encoder sigmoid
score, which is what the frontend displays as a percentage.
"""
from __future__ import annotations

import logging
from uuid import UUID

from config import settings
from database.postgres import get_pool
from models.documents import SourceReference
from services.embedding import embed_query
from services.reranker import rerank
from utils.text_utils import keyword_score, reciprocal_rank_fusion

log = logging.getLogger(__name__)


async def generate_hypothetical_answer(query: str, groq_client) -> str:
    try:
        resp = groq_client.chat.completions.create(
            model=settings.groq_fallback_model,
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


async def vector_search(
    conversation_id: str,
    embedding: list[float],
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        if document_ids:
            rows = await conn.fetch(
                """
                SELECT id, document_id, content, page_number, page_end,
                       chunk_index, token_count,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM chunks
                WHERE conversation_id = $2
                  AND document_id = ANY($3::uuid[])
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                embedding, conversation_id, document_ids, top_k,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, document_id, content, page_number, page_end,
                       chunk_index, token_count,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM chunks
                WHERE conversation_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding, conversation_id, top_k,
            )
    return [dict(r) for r in rows]


async def keyword_search(
    conversation_id: str,
    query: str,
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            if document_ids:
                rows = await conn.fetch(
                    """
                    SELECT id, document_id, content, page_number, page_end,
                           chunk_index, token_count,
                           ts_rank_cd(
                               to_tsvector('english', content),
                               plainto_tsquery('english', $1)
                           ) AS rank
                    FROM chunks
                    WHERE conversation_id = $2
                      AND document_id = ANY($3::uuid[])
                      AND to_tsvector('english', content)
                          @@ plainto_tsquery('english', $1)
                    ORDER BY rank DESC
                    LIMIT $4
                    """,
                    query, conversation_id, document_ids, top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, document_id, content, page_number, page_end,
                           chunk_index, token_count,
                           ts_rank_cd(
                               to_tsvector('english', content),
                               plainto_tsquery('english', $1)
                           ) AS rank
                    FROM chunks
                    WHERE conversation_id = $2
                      AND to_tsvector('english', content)
                          @@ plainto_tsquery('english', $1)
                    ORDER BY rank DESC
                    LIMIT $3
                    """,
                    query, conversation_id, top_k,
                )
        except Exception as e:
            log.warning("FTS failed: %s – using ilike fallback", e)
            rows = await conn.fetch(
                """
                SELECT id, document_id, content, page_number, page_end,
                       chunk_index, token_count, 0.1 AS rank
                FROM chunks
                WHERE conversation_id = $1
                  AND content_lower ILIKE $2
                LIMIT $3
                """,
                conversation_id, f"%{query[:100].lower()}%", top_k,
            )
    return [dict(r) for r in rows]


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


def score_to_confidence(cross_encoder_score: float) -> str:
    """
    Confidence from cross-encoder sigmoid score [0, 1].
    Cross-encoder scores are well-calibrated so standard thresholds work.
    """
    if cross_encoder_score >= settings.confidence_high:
        return "high"
    if cross_encoder_score >= settings.confidence_medium:
        return "medium"
    return "low"


async def hybrid_retrieve(
    conversation_id: str,
    query: str,
    groq_client,
    document_ids: list[UUID] | None = None,
    top_k: int | None = None,
    rerank_n: int | None = None,
) -> tuple[list[SourceReference], float]:
    top_k = top_k or settings.retrieval_top_k
    rerank_n = rerank_n or settings.rerank_top_n
    doc_id_strs = [str(d) for d in document_ids] if document_ids else None

    if settings.hyde_enabled:
        hyp_answer = await generate_hypothetical_answer(query, groq_client)
        search_text = f"{query} {hyp_answer}"
    else:
        search_text = query

    query_embedding = await embed_query(search_text)

    import asyncio
    vector_results, keyword_results = await asyncio.gather(
        vector_search(conversation_id, query_embedding, top_k, doc_id_strs),
        keyword_search(conversation_id, query, top_k, doc_id_strs),
    )

    # ── Stage 1: RRF fusion → top-N candidates ────────────────────────────────
    all_chunks: dict[str, dict] = {}
    for r in vector_results:
        cid = str(r["id"])
        all_chunks[cid] = {**r, "similarity": float(r.get("similarity", 0))}
    for r in keyword_results:
        cid = str(r["id"])
        if cid not in all_chunks:
            all_chunks[cid] = {**r, "similarity": 0.0}
        all_chunks[cid]["keyword_rank_score"] = float(r.get("rank", 0))

    vector_rank = [str(r["id"]) for r in vector_results]
    keyword_rank = [str(r["id"]) for r in keyword_results]
    rrf_scores = reciprocal_rank_fusion(vector_rank, keyword_rank, alpha=settings.hybrid_alpha)

    for cid, chunk in all_chunks.items():
        chunk["kw_score"] = keyword_score(query, chunk["content"])
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + chunk["kw_score"] * 0.05

    # RRF orders candidates for recall — cross-encoder will rescore for precision
    ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    candidate_ids = ranked_ids[:rerank_n]
    candidates = [all_chunks[cid] for cid in candidate_ids]

    if not candidates:
        return [], 0.0

    # ── Stage 2: Cross-encoder reranking ─────────────────────────────────────
    # Scores are sigmoid(logit) → [0, 1].  Semantically irrelevant chunks
    # drop to 0.05–0.15 even if topically similar; relevant chunks score
    # 0.80–0.99.  This is where the real differentiation happens.
    reranked = await rerank(query, candidates)

    # Apply relevance cutoff on cross-encoder score.
    # rerank_min_ratio relative to the top cross-encoder score.
    if reranked:
        top_ce_score = reranked[0][1]
        min_ce_score = top_ce_score * settings.rerank_min_ratio
        reranked = [(c, s) for c, s in reranked if s >= min_ce_score]

    if not reranked:
        return [], 0.0

    doc_ids = list({str(c["document_id"]) for c, _ in reranked})
    doc_names = await get_document_names(doc_ids)

    sources: list[SourceReference] = []
    for chunk, ce_score in reranked:
        cid = str(chunk["id"])
        doc_id_str = str(chunk["document_id"])
        sources.append(SourceReference(
            chunk_id=chunk["id"],
            document_id=chunk["document_id"],
            document_name=doc_names.get(doc_id_str, "Unknown"),
            content=chunk["content"],
            page_number=chunk.get("page_number"),
            page_end=chunk.get("page_end"),
            chunk_index=chunk["chunk_index"],
            similarity_score=round(ce_score, 4),              # cross-encoder sigmoid
            keyword_score=round(chunk.get("kw_score", 0.0), 4),
            combined_score=round(rrf_scores.get(cid, 0.0), 6), # RRF for debugging
            confidence=score_to_confidence(ce_score),
        ))

    top_score = sources[0].similarity_score if sources else 0.0
    return sources, top_score

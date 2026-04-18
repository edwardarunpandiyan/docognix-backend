"""
services/retrieval.py – Hybrid retrieval pipeline.
Uses conversation_id (renamed from session_id) throughout.
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


def score_to_confidence(score: float) -> str:
    if score >= settings.confidence_high:
        return "high"
    if score >= settings.confidence_medium:
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

    ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    top_ids = ranked_ids[:rerank_n]

    if not top_ids:
        return [], 0.0

    doc_ids = list({str(all_chunks[cid]["document_id"]) for cid in top_ids})
    doc_names = await get_document_names(doc_ids)

    sources: list[SourceReference] = []
    for cid in top_ids:
        c = all_chunks[cid]
        sim = c.get("similarity", 0.0)
        combined = rrf_scores.get(cid, 0.0)
        doc_id_str = str(c["document_id"])
        sources.append(SourceReference(
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
        ))

    top_score = all_chunks[top_ids[0]].get("similarity", 0.0) if top_ids else 0.0
    return sources, top_score

"""
services/cache.py – Semantic cache using Upstash Redis.

Strategy:
  • Key: "semcache:<session_id>:<query_embedding_hash>"
  • On every query, compute the query embedding, then scan existing cache keys
    for this session and compare cosine similarity.
  • If any key's stored embedding has cosine_sim ≥ threshold → CACHE HIT.
  • Cache entries store: {answer, sources, confidence, embedding}.

This avoids re-running the full RAG pipeline for semantically equivalent
questions like "What is the summary?" and "Summarise this document".
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from config import settings
from database.redis_client import redis_get, redis_set, redis_delete, redis_keys_matching
from services.embedding import cosine_similarity

log = logging.getLogger(__name__)

_PREFIX = "semcache"


def _embedding_hash(embedding: list[float]) -> str:
    """Short hash of an embedding for use in Redis key."""
    raw = json.dumps(embedding[:8], separators=(",", ":"))   # first 8 dims
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_key(session_id: str, emb_hash: str) -> str:
    return f"{_PREFIX}:{session_id}:{emb_hash}"


async def cache_lookup(
    session_id: str,
    query_embedding: list[float],
) -> dict | None:
    """
    Return cached {answer, sources, confidence} if a semantically similar
    query was previously cached for this session.
    """
    pattern = f"{_PREFIX}:{session_id}:*"
    keys = await redis_keys_matching(pattern)

    for key in keys:
        entry = await redis_get(key)
        if not entry or "embedding" not in entry:
            continue
        sim = cosine_similarity(query_embedding, entry["embedding"])
        if sim >= settings.cache_similarity_threshold:
            log.info("Semantic cache HIT  (sim=%.3f) for session %s", sim, session_id)
            return {
                "answer": entry["answer"],
                "sources": entry["sources"],
                "confidence": entry["confidence"],
                "cache_hit": True,
            }

    return None


async def cache_store(
    session_id: str,
    query_embedding: list[float],
    answer: str,
    sources: list[dict],
    confidence: str,
) -> None:
    """Persist a query result in the semantic cache."""
    emb_hash = _embedding_hash(query_embedding)
    key = _cache_key(session_id, emb_hash)
    payload = {
        "embedding": query_embedding,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }
    await redis_set(key, payload, ttl=settings.cache_ttl_seconds)
    log.debug("Semantic cache STORE for session %s (key=%s)", session_id, key)


async def cache_invalidate_session(session_id: str) -> None:
    """Clear all cache entries for a session (called on new document upload)."""
    pattern = f"{_PREFIX}:{session_id}:*"
    keys = await redis_keys_matching(pattern)
    for key in keys:
        await redis_delete(key)
    if keys:
        log.info("Invalidated %d cache entries for session %s", len(keys), session_id)

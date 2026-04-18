"""
services/cache.py – Semantic cache using Upstash Redis.
Uses conversation_id (renamed from session_id) throughout.
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
    raw = json.dumps(embedding[:8], separators=(",", ":"))
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_key(conversation_id: str, emb_hash: str) -> str:
    return f"{_PREFIX}:{conversation_id}:{emb_hash}"


async def cache_lookup(
    conversation_id: str,
    query_embedding: list[float],
) -> dict | None:
    pattern = f"{_PREFIX}:{conversation_id}:*"
    keys = await redis_keys_matching(pattern)

    for key in keys:
        entry = await redis_get(key)
        if not entry or "embedding" not in entry:
            continue
        sim = cosine_similarity(query_embedding, entry["embedding"])
        if sim >= settings.cache_similarity_threshold:
            log.info("Semantic cache HIT (sim=%.3f) for conversation %s", sim, conversation_id)
            return {
                "answer": entry["answer"],
                "sources": entry["sources"],
                "confidence": entry["confidence"],
                "cache_hit": True,
            }
    return None


async def cache_store(
    conversation_id: str,
    query_embedding: list[float],
    answer: str,
    sources: list[dict],
    confidence: str,
) -> None:
    emb_hash = _embedding_hash(query_embedding)
    key = _cache_key(conversation_id, emb_hash)
    payload = {
        "embedding": query_embedding,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }
    await redis_set(key, payload, ttl=settings.cache_ttl_seconds)
    log.debug("Semantic cache STORE for conversation %s", conversation_id)


async def cache_invalidate_conversation(conversation_id: str) -> None:
    """Clear all cache entries for a conversation (called on new document upload)."""
    pattern = f"{_PREFIX}:{conversation_id}:*"
    keys = await redis_keys_matching(pattern)
    for key in keys:
        await redis_delete(key)
    if keys:
        log.info("Invalidated %d cache entries for conversation %s", len(keys), conversation_id)

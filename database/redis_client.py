"""
database/redis_client.py – Redis client via Upstash (TLS, REST-compatible).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from config import settings

log = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.upstash_redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _redis


async def redis_get(key: str) -> Any | None:
    try:
        r = await get_redis()
        value = await r.get(key)
        if value is None:
            return None
        return json.loads(value)
    except Exception as e:
        log.warning("Redis GET failed for key=%s: %s", key, e)
        return None


async def redis_set(key: str, value: Any, ttl: int = settings.cache_ttl_seconds) -> None:
    try:
        r = await get_redis()
        await r.setex(key, ttl, json.dumps(value))
    except Exception as e:
        log.warning("Redis SET failed for key=%s: %s", key, e)


async def redis_delete(key: str) -> None:
    try:
        r = await get_redis()
        await r.delete(key)
    except Exception as e:
        log.warning("Redis DELETE failed for key=%s: %s", key, e)


async def redis_keys_matching(pattern: str) -> list[str]:
    try:
        r = await get_redis()
        return await r.keys(pattern)
    except Exception as e:
        log.warning("Redis KEYS failed for pattern=%s: %s", pattern, e)
        return []


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None

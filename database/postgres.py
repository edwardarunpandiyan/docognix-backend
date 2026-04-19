"""
database/postgres.py – asyncpg connection pool for Supabase PostgreSQL.
"""
from __future__ import annotations

import asyncpg
from config import settings

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.supabase_db_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
            statement_cache_size=0,   # required for Supabase pooler (transaction mode)
            # pgvector codec registration
            init=_register_vector,
        )
    return _pool


async def _register_vector(conn: asyncpg.Connection) -> None:
    """Register the vector type so asyncpg can encode/decode it."""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.set_type_codec(
        "vector",
        encoder=_encode_vector,
        decoder=_decode_vector,
        schema="public",
        format="text",
    )


def _encode_vector(value: list[float]) -> str:
    return "[" + ",".join(str(v) for v in value) + "]"


def _decode_vector(text: str) -> list[float]:
    return [float(v) for v in text.strip("[]").split(",")]


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

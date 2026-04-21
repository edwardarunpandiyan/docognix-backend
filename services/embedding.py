"""
services/embedding.py – Local embedding with BAAI/bge-small-en-v1.5.

• 384-dimensional dense vectors.
• Free, runs on CPU in ~20 ms per batch.
• BGE models use a query instruction prefix for asymmetric retrieval:
    - Queries: prepend "Represent this sentence for searching relevant passages: "
    - Documents: no prefix needed.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

log = logging.getLogger(__name__)

# BGE asymmetric query prefix
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    log.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    log.info("Embedding model loaded.")
    return model


def _embed_sync(texts: list[str], is_query: bool = False) -> list[list[float]]:
    model = _get_model()
    if is_query:
        texts = [_BGE_QUERY_PREFIX + t for t in texts]
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # cosine similarity = dot product
        batch_size=32,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


async def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed a batch of texts (documents or queries) asynchronously."""
    loop = asyncio.get_running_loop()   # fix: get_event_loop() is deprecated in 3.10+ and can return wrong loop
    return await loop.run_in_executor(None, _embed_sync, texts, is_query)


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    results = await embed_texts([query], is_query=True)
    return results[0]


async def embed_document_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed a list of document chunks (no instruction prefix)."""
    return await embed_texts(chunks, is_query=False)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity for already-normalised vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb))

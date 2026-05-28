"""
services/embedding.py – Embedding service with local/API dual-mode support.

Mode is controlled by the USE_LOCAL_MODELS environment variable:

  USE_LOCAL_MODELS=true  → sentence-transformers loaded locally (CPU)
                           torch is imported only in this branch, so RAM
                           is not consumed when running in API mode.

  USE_LOCAL_MODELS=false → HuggingFace Inference API (free tier)
                           No torch / sentence-transformers loaded at all.
                           Requires HF_API_TOKEN in environment.

BGE asymmetric retrieval:
  • Queries   → prepend the instruction prefix defined in config
  • Documents → no prefix needed
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import httpx
import numpy as np

from config import settings

log = logging.getLogger(__name__)

# ── HuggingFace Inference API ─────────────────────────────────────────────────
_HF_EMBED_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
)


# ── Local model loader (imported lazily – only when USE_LOCAL_MODELS=true) ────

@lru_cache(maxsize=1)
def _get_local_model():  # type: ignore[return]
    """Load and cache the local SentenceTransformer model.

    The import is intentionally inside this function so that torch and
    sentence-transformers are never imported (and therefore never consume
    RAM) when USE_LOCAL_MODELS=false.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    log.info("Loading local embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    log.info("Local embedding model loaded.")
    return model


# ── Internal sync helpers ─────────────────────────────────────────────────────

def _embed_local(texts: list[str], is_query: bool) -> list[list[float]]:
    """Embed texts using the locally loaded SentenceTransformer model."""
    model = _get_local_model()
    if is_query:
        texts = [settings.bge_query_prefix + t for t in texts]
    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # cosine similarity == dot product
        batch_size=32,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def _embed_api(texts: list[str], is_query: bool) -> list[list[float]]:
    """Embed texts via the HuggingFace Inference API (synchronous)."""
    if is_query:
        texts = [settings.bge_query_prefix + t for t in texts]

    url = _HF_EMBED_URL.format(model=settings.embedding_model)
    headers = {"Authorization": f"Bearer {settings.hf_api_token}"}

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json={"inputs": texts})
        response.raise_for_status()

    raw = response.json()

    # HF feature-extraction returns either:
    #   [[float, ...], ...]          (batch of flat vectors)  ← most models
    #   [[[float, ...], ...], ...]   (batch of token vectors) ← some models
    # Flatten token-level outputs by mean-pooling if needed.
    vectors: list[list[float]] = []
    for item in raw:
        if isinstance(item[0], list):
            # Token-level → mean-pool to a single vector
            arr = np.array(item, dtype=np.float32).mean(axis=0)
        else:
            arr = np.array(item, dtype=np.float32)
        # Normalise so cosine similarity == dot product (matches local behaviour)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        vectors.append(arr.tolist())

    return vectors


def _embed_sync(texts: list[str], is_query: bool) -> list[list[float]]:
    """Dispatch to local or API backend based on settings."""
    if settings.use_local_models:
        return _embed_local(texts, is_query)
    return _embed_api(texts, is_query)


# ── Public async API ──────────────────────────────────────────────────────────

async def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed a batch of texts (documents or queries) asynchronously."""
    loop = asyncio.get_running_loop()
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

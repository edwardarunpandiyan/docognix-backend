"""
services/reranker.py – Cross-encoder reranking with local/API dual-mode support.

Mode is controlled by the USE_LOCAL_MODELS environment variable:

  USE_LOCAL_MODELS=true  → CrossEncoder loaded locally via sentence-transformers.
                           torch is imported only in this branch.

  USE_LOCAL_MODELS=false → HuggingFace Inference API (free tier, text-ranking
                           pipeline).  No torch loaded at all.
                           Requires HF_API_TOKEN in environment.

Why cross-encoders outperform bi-encoders for reranking
────────────────────────────────────────────────────────
Bi-encoders (BGE-small) embed query and chunk independently, then measure
cosine distance. When a whole document is on one topic, every chunk lands in
a tight cosine neighbourhood → scores cluster (e.g. 0.62–0.68 for all chunks).

A cross-encoder takes the (query, chunk) pair as a single input so attention
layers directly compare query tokens against chunk tokens. It outputs a single
relevance logit — typically −10 to +10 — which spreads out dramatically:

    Page 4 (direct answer)    →  logit  8.3  →  ~100 % after normalisation
    Page 1 (intro, off-topic) →  logit −2.1  →  ~11 % after normalisation

Scores are min-max normalised within each batch, making the top chunk always
1.0 and others proportional to it — model-agnostic and threshold-friendly.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import httpx

from config import settings

log = logging.getLogger(__name__)

# ── HuggingFace Inference API ─────────────────────────────────────────────────
_HF_RERANK_URL = (
    "https://api-inference.huggingface.co/models/{model}"
)


# ── Local model loader (imported lazily – only when USE_LOCAL_MODELS=true) ────

@lru_cache(maxsize=1)
def _get_local_cross_encoder():  # type: ignore[return]
    """Load and cache the local CrossEncoder model.

    The import lives here so torch is never loaded when USE_LOCAL_MODELS=false.
    """
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    log.info("Loading local cross-encoder model: %s", settings.reranker_model)
    model = CrossEncoder(settings.reranker_model)
    log.info("Local cross-encoder loaded.")
    return model


# ── Score normalisation helper ────────────────────────────────────────────────

def _minmax_normalise(scores: list[float]) -> list[float]:
    """Min-max normalise scores to [0, 1].

    When all scores are equal (spread == 0), every chunk is equally relevant
    and all scores are returned as 1.0.
    """
    min_s = min(scores)
    max_s = max(scores)
    spread = max_s - min_s
    if spread > 0:
        return [(s - min_s) / spread for s in scores]
    return [1.0] * len(scores)


# ── Internal sync helpers ─────────────────────────────────────────────────────

def _rerank_local(query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    """Rerank using the locally loaded CrossEncoder model."""
    model = _get_local_cross_encoder()
    pairs = [(query, c["content"]) for c in chunks]
    raw_scores: list[float] = model.predict(pairs).tolist()

    normalised = _minmax_normalise(raw_scores)
    scored = list(zip(chunks, normalised))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _rerank_api(query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    """Rerank via the HuggingFace Inference API (synchronous).

    The text-classification pipeline for ms-marco models returns a list of
    [{"label": "...", "score": float}] per pair. We extract the score for the
    positive label ("LABEL_1" or "true") as the relevance score.
    """
    url = _HF_RERANK_URL.format(model=settings.reranker_model)
    headers = {"Authorization": f"Bearer {settings.hf_api_token}"}

    # Send all (query, chunk) pairs in one request as a list of string pairs.
    pairs = [[query, c["content"]] for c in chunks]

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json={"inputs": pairs})
        response.raise_for_status()

    raw = response.json()

    # Extract relevance score for each pair.
    # ms-marco cross-encoders use LABEL_1 as the relevant label.
    raw_scores: list[float] = []
    for result in raw:
        # result is a list of dicts: [{"label": "LABEL_0", "score": 0.1}, ...]
        label_scores = {item["label"]: item["score"] for item in result}
        score = label_scores.get("LABEL_1") or label_scores.get("true") or max(
            label_scores.values()
        )
        raw_scores.append(float(score))

    normalised = _minmax_normalise(raw_scores)
    scored = list(zip(chunks, normalised))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _rerank_sync(query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    """Dispatch to local or API backend based on settings."""
    if not chunks:
        return []
    if settings.use_local_models:
        return _rerank_local(query, chunks)
    return _rerank_api(query, chunks)


# ── Public async API ──────────────────────────────────────────────────────────

async def rerank(query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    """Rerank chunks by relevance to the query.

    Runs in a thread executor to avoid blocking the event loop.
    Returns list of (chunk_dict, score_0_to_1) sorted best-first.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _rerank_sync, query, chunks)

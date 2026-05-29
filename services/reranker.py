"""
services/reranker.py – Cross-encoder reranking stage.

Why cross-encoders produce spread-out scores
────────────────────────────────────────────
Bi-encoders (BGE-small) embed query and chunk *independently*, then measure
cosine distance.  When a whole document is on one topic, every chunk lands in
a tight cosine neighbourhood → scores cluster (e.g. 0.62–0.68 for all chunks).

A cross-encoder takes the (query, chunk) *pair* as a single input, so the
attention layers can directly compare the query tokens against chunk tokens.
It outputs a single relevance logit — typically ranging from −10 to +10 —
which spreads out dramatically even across topically similar chunks:

    Page 4 (direct answer)    →  logit  8.3  → sigmoid  0.9998  → ~100 %
    Page 1 (intro, off-topic) →  logit −2.1  → sigmoid  0.109   →  ~11 %

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  • ~80 MB download, CPU-fast (~10–30 ms per batch of 6)
  • Trained on MS-MARCO passage ranking — best open-weight cross-encoder
    for general English Q&A / document retrieval
  • Ships inside sentence-transformers (no new dependency)

Scores are min-max normalised within each batch, making the top chunk always
1.0 and others proportional to it — model-agnostic and threshold-friendly.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from config import settings

log = logging.getLogger(__name__)


# ── Model loader ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    log.info("Loading cross-encoder model: %s", settings.reranker_model)
    model = CrossEncoder(settings.reranker_model)
    log.info("Cross-encoder loaded.")
    return model


# ── Score normalisation ───────────────────────────────────────────────────────

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


# ── Internal sync helper ──────────────────────────────────────────────────────

def _rerank_sync(query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    if not chunks:
        return []

    model = _get_cross_encoder()
    pairs = [(query, c["content"]) for c in chunks]
    raw_scores: list[float] = model.predict(pairs).tolist()

    normalised = _minmax_normalise(raw_scores)
    scored = list(zip(chunks, normalised))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Public async API ──────────────────────────────────────────────────────────

async def rerank(query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    """Rerank chunks by relevance to the query.

    Runs in a thread executor to avoid blocking the event loop.
    Returns list of (chunk_dict, score_0_to_1) sorted best-first.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _rerank_sync, query, chunks)
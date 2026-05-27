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

    Page 4 (direct answer)  →  logit  8.3  → sigmoid  0.9998  → ~100 %
    Page 1 (intro, off-topic) →  logit −2.1  → sigmoid  0.109   → ~11 %

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  • ~80 MB download, CPU-fast (~10–30 ms per batch of 6)
  • Trained on MS-MARCO passage ranking — best open-weight cross-encoder
    for general English Q&A / document retrieval
  • Ships inside sentence-transformers (no new dependency)
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    log.info("Loading cross-encoder model: %s", settings.reranker_model)
    model = CrossEncoder(settings.reranker_model)
    log.info("Cross-encoder loaded.")
    return model


def _rerank_sync(query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    if not chunks:
        return []

    model = _get_cross_encoder()
    pairs = [(query, c["content"]) for c in chunks]
    raw_scores: list[float] = model.predict(pairs).tolist()

    # Min-max normalise within this batch.
    # Raw logits from ms-marco sit around -3 to -1 for document queries —
    # sigmoid maps all of them below 0.5 making thresholds meaningless.
    # Min-max makes the top chunk 1.0 and others proportional to it,
    # which is model-agnostic and always produces meaningful display scores.
    min_s = min(raw_scores)
    max_s = max(raw_scores)
    spread = max_s - min_s

    if spread > 0:
        normalised = [(s - min_s) / spread for s in raw_scores]
    else:
        # All chunks scored identically — treat all as equally relevant
        normalised = [1.0 for _ in raw_scores]

    scored = list(zip(chunks, normalised))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


async def rerank(query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    """
    Async wrapper around the cross-encoder.
    Runs in a thread executor to avoid blocking the event loop.
    Returns list of (chunk_dict, score_0_to_1) sorted best-first.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _rerank_sync, query, chunks)

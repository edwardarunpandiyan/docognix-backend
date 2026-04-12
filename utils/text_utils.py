"""
utils/text_utils.py – Smart chunking, token counting, text cleaning.

Strategy: Recursive sentence-aware chunking with overlap.
  1. Split on paragraph breaks first.
  2. If a paragraph exceeds chunk_size, split on sentence boundaries.
  3. Assemble chunks greedily, adding sentences until the size limit is hit.
  4. Slide the window back by `overlap` tokens to maintain cross-chunk context.
"""
from __future__ import annotations

import re
import unicodedata


# ── Token counting (approx, no tiktoken needed) ───────────────────────────────

def approx_token_count(text: str) -> int:
    """~4 chars per token is a reasonable heuristic for English."""
    return max(1, len(text) // 4)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise unicode, remove control chars, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    # Remove null bytes and other control characters except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs on the same line
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences (simple regex – good enough for RAG chunking)."""
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ── Core chunker ──────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """
    Return a list of overlapping text chunks.
    Each chunk is approximately `chunk_size` tokens.
    Adjacent chunks share ~`overlap` tokens for continuity.
    """
    paragraphs: list[str] = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # Flatten paragraphs → sentences
    sentences: list[str] = []
    for para in paragraphs:
        sents = split_into_sentences(para)
        if not sents:
            sents = [para]
        sentences.extend(sents)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = approx_token_count(sentence)

        # If a single sentence exceeds chunk_size, hard-split it by words
        if s_tokens > chunk_size:
            if current:
                chunks.append(" ".join(current))
            sub_chunks = _hard_split(sentence, chunk_size, overlap)
            chunks.extend(sub_chunks[:-1])
            # Keep last sub-chunk as the start of the next window
            last = sub_chunks[-1] if sub_chunks else sentence
            current = [last]
            current_tokens = approx_token_count(last)
            continue

        if current_tokens + s_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            # Roll back by overlap tokens
            current, current_tokens = _rollback(current, overlap)

        current.append(sentence)
        current_tokens += s_tokens

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk_words = words[i: i + size]
        chunks.append(" ".join(chunk_words))
        i += size - overlap
    return chunks


def _rollback(sentences: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Drop sentences from the front until total tokens ≤ overlap_tokens."""
    kept = sentences[:]
    while kept and approx_token_count(" ".join(kept)) > overlap_tokens:
        kept.pop(0)
    return kept, approx_token_count(" ".join(kept))


# ── Page-aware chunker (PDF / DOCX) ──────────────────────────────────────────

def chunk_pages(
    pages: list[str],              # pages[0] = page 1 text, etc.
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """
    Chunk page-segmented text.
    Returns list of dicts:
        content, page_number (1-based), page_end, chunk_index,
        char_start, char_end, token_count
    """
    results: list[dict] = []
    chunk_index = 0
    global_char = 0

    # Build (text, page_number) sentence list across all pages
    all_sentences: list[tuple[str, int]] = []
    for page_no, page_text in enumerate(pages, start=1):
        cleaned = clean_text(page_text)
        for sentence in split_into_sentences(cleaned) or [cleaned]:
            if sentence.strip():
                all_sentences.append((sentence.strip(), page_no))

    i = 0
    while i < len(all_sentences):
        current_sents: list[tuple[str, int]] = []
        current_tokens = 0

        while i < len(all_sentences):
            sent, pno = all_sentences[i]
            s_tok = approx_token_count(sent)
            if current_tokens + s_tok > chunk_size and current_sents:
                break
            current_sents.append((sent, pno))
            current_tokens += s_tok
            i += 1

        if not current_sents:
            i += 1
            continue

        content = " ".join(s for s, _ in current_sents)
        page_start = current_sents[0][1]
        page_end = current_sents[-1][1]
        char_start = global_char
        char_end = global_char + len(content)
        global_char = char_end + 1

        results.append({
            "content": content,
            "page_number": page_start,
            "page_end": page_end,
            "chunk_index": chunk_index,
            "char_start": char_start,
            "char_end": char_end,
            "token_count": current_tokens,
        })
        chunk_index += 1

        # Overlap: step back a few sentences
        overlap_sents = 0
        overlap_tok = 0
        for sent, pno in reversed(current_sents):
            tok = approx_token_count(sent)
            if overlap_tok + tok > overlap:
                break
            overlap_tok += tok
            overlap_sents += 1

        if overlap_sents:
            i -= overlap_sents

    return results


# ── BM25-style keyword scoring ────────────────────────────────────────────────

def keyword_score(query: str, chunk: str) -> float:
    """
    Simple TF-IDF-inspired score: fraction of unique query terms
    that appear in the chunk, weighted by term frequency.
    Fast, no external dependencies.
    """
    query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
    if not query_terms:
        return 0.0
    chunk_lower = chunk.lower()
    chunk_words = re.findall(r"\b\w+\b", chunk_lower)
    total_words = max(len(chunk_words), 1)

    score = 0.0
    for term in query_terms:
        tf = chunk_lower.count(term) / total_words
        if tf > 0:
            score += tf
    return min(score / len(query_terms) * 10, 1.0)   # normalise to [0, 1]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_ranks: list[str],     # chunk_id ordered by cosine sim (best first)
    keyword_ranks: list[str],    # chunk_id ordered by keyword score (best first)
    k: int = 60,
    alpha: float = 0.7,          # weight for vector ranking
) -> dict[str, float]:
    """
    Combine two ranked lists using RRF.
    Returns {chunk_id: rrf_score}, higher is better.
    """
    scores: dict[str, float] = {}

    for rank, cid in enumerate(vector_ranks, start=1):
        scores[cid] = scores.get(cid, 0.0) + alpha * (1.0 / (k + rank))

    beta = 1.0 - alpha
    for rank, cid in enumerate(keyword_ranks, start=1):
        scores[cid] = scores.get(cid, 0.0) + beta * (1.0 / (k + rank))

    return scores

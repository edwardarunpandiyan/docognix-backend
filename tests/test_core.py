"""
tests/test_core.py – Unit tests for chunking, scoring, and text utilities.
Run with: pytest tests/ -v
"""
import pytest
from utils.text_utils import (
    approx_token_count,
    clean_text,
    chunk_text,
    chunk_pages,
    keyword_score,
    reciprocal_rank_fusion,
)


# ── Token count ───────────────────────────────────────────────────────────────

def test_approx_token_count_basic():
    assert approx_token_count("hello world") > 0
    assert approx_token_count("") == 1   # min is 1


def test_approx_token_count_scaling():
    short = approx_token_count("hi")
    long = approx_token_count("hello " * 100)
    assert long > short


# ── Text cleaning ─────────────────────────────────────────────────────────────

def test_clean_text_removes_control_chars():
    result = clean_text("Hello\x00World\x01!")
    assert "\x00" not in result
    assert "Hello" in result


def test_clean_text_collapses_blank_lines():
    text = "Para one.\n\n\n\n\nPara two."
    result = clean_text(text)
    assert "\n\n\n" not in result


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunk_text_returns_nonempty():
    text = " ".join(["word"] * 1000)
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.strip()


def test_chunk_text_within_size():
    text = " ".join(["word"] * 2000)
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    for c in chunks:
        assert approx_token_count(c) <= 250   # some slack for sentence re-assembly


def test_chunk_pages_page_numbers():
    pages = [
        "First page content here. It has sentences.",
        "Second page content here. More sentences.",
        "Third page content.",
    ]
    chunks = chunk_pages(pages, chunk_size=30, overlap=5)
    assert len(chunks) > 0
    for c in chunks:
        assert "page_number" in c
        assert "chunk_index" in c
        assert c["page_number"] >= 1


def test_chunk_pages_preserves_order():
    pages = [f"Page {i} content with plenty of words to fill up the chunk size." for i in range(5)]
    chunks = chunk_pages(pages, chunk_size=20, overlap=3)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(indices)))


# ── Keyword scoring ───────────────────────────────────────────────────────────

def test_keyword_score_exact_match():
    score = keyword_score("machine learning", "Machine learning is a subset of AI.")
    assert score > 0.3


def test_keyword_score_no_match():
    score = keyword_score("quantum physics", "The cat sat on the mat.")
    assert score == 0.0


def test_keyword_score_bounds():
    score = keyword_score("test query here", "test query here test query here")
    assert 0.0 <= score <= 1.0


# ── RRF Fusion ────────────────────────────────────────────────────────────────

def test_rrf_combines_both_lists():
    vector_ranks = ["A", "B", "C"]
    keyword_ranks = ["C", "A", "D"]
    scores = reciprocal_rank_fusion(vector_ranks, keyword_ranks)
    # A appears in both lists so should score high
    assert scores["A"] > scores["D"]   # D only in keyword list, ranked lower there
    assert "C" in scores


def test_rrf_alpha_weighting():
    vector_ranks = ["A", "B"]
    keyword_ranks = ["B", "A"]
    # High alpha → vector dominates → A should win
    scores_high = reciprocal_rank_fusion(vector_ranks, keyword_ranks, alpha=0.95)
    # Low alpha → keyword dominates → B should win
    scores_low = reciprocal_rank_fusion(vector_ranks, keyword_ranks, alpha=0.05)
    assert scores_high["A"] > scores_high["B"]
    assert scores_low["B"] > scores_low["A"]

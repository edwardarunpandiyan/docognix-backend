"""
models/documents.py – Pydantic schemas for documents, chunks, and upload responses.
These match exactly what the Docognix React frontend expects.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ── Upload / Ingest ──────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Returned immediately after successful processing.
    Frontend stores this in localStorage under session.documents[].
    """
    document_id: UUID
    session_id: UUID
    filename: str
    original_name: str
    file_type: str                    # "pdf" | "docx" | "txt"
    file_size: int                    # bytes
    page_count: int | None            # None for plain-text files
    word_count: int | None
    chunk_count: int
    status: str = "ready"             # "processing" | "ready" | "error"
    created_at: datetime


class DocumentStatus(BaseModel):
    document_id: UUID
    status: str
    chunk_count: int
    error_message: str | None = None


# ── Document list ────────────────────────────────────────────────────────────

class DocumentSummary(BaseModel):
    document_id: UUID
    session_id: UUID
    filename: str
    original_name: str
    file_type: str
    file_size: int
    page_count: int | None
    word_count: int | None
    chunk_count: int
    status: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int


# ── Chunk (internal, also exposed as source citations) ────────────────────────

class ChunkCreate(BaseModel):
    document_id: UUID
    session_id: UUID
    content: str
    page_number: int | None
    page_end: int | None
    chunk_index: int
    char_start: int | None
    char_end: int | None
    token_count: int
    embedding: list[float]


class SourceReference(BaseModel):
    """A cited chunk returned inside an assistant message.
    Frontend renders these as citation cards in the chat view.
    """
    chunk_id: UUID
    document_id: UUID
    document_name: str                # original_name for display
    content: str                      # snippet text
    page_number: int | None           # 1-based page number for highlighting
    page_end: int | None              # last page if chunk spans pages
    chunk_index: int
    similarity_score: float           # 0-1 cosine similarity
    keyword_score: float = 0.0        # BM25 / TF-IDF component
    combined_score: float             # final RRF score used for ranking
    confidence: str                   # "high" | "medium" | "low"

    class Config:
        json_encoders = {UUID: str}

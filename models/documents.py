"""
models/documents.py – Pydantic schemas for documents, chunks, and upload responses.
Uses conversation_id (renamed from session_id) to match frontend naming.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    conversation_id: UUID          # renamed from session_id
    filename: str
    original_name: str
    file_type: str
    file_size: int
    page_count: int | None
    word_count: int | None
    chunk_count: int
    status: str = "ready"
    created_at: datetime


class DocumentStatus(BaseModel):
    document_id: UUID
    status: str
    chunk_count: int
    error_message: str | None = None


class DocumentSummary(BaseModel):
    document_id: UUID
    conversation_id: UUID          # renamed from session_id
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


class ChunkCreate(BaseModel):
    document_id: UUID
    conversation_id: UUID          # renamed from session_id
    content: str
    page_number: int | None
    page_end: int | None
    chunk_index: int
    char_start: int | None
    char_end: int | None
    token_count: int
    embedding: list[float]


class SourceReference(BaseModel):
    """A cited chunk returned inside an assistant message."""
    chunk_id: UUID
    document_id: UUID
    document_name: str
    content: str
    page_number: int | None
    page_end: int | None
    chunk_index: int
    similarity_score: float
    keyword_score: float = 0.0
    combined_score: float
    confidence: str                 # "high" | "medium" | "low"

    class Config:
        json_encoders = {UUID: str}

"""
models/chat.py – Pydantic schemas for sessions and chat messages.
Matches the Docognix frontend's data contracts.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from models.documents import SourceReference


# ── Sessions ─────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    title: str = "New Chat"


class SessionUpdate(BaseModel):
    title: str


class SessionResponse(BaseModel):
    session_id: UUID
    title: str
    document_count: int = 0
    total_chunks: int = 0
    message_count: int = 0
    last_message_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {UUID: str}


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int


# ── Chat Messages ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Sent by the frontend to POST /chat/{session_id}"""
    message: str = Field(..., min_length=1, max_length=4000)
    document_ids: list[UUID] | None = None  # None = search all docs in session
    stream: bool = True


class MessageResponse(BaseModel):
    """A persisted message (user or assistant)."""
    message_id: UUID
    session_id: UUID
    role: str                           # "user" | "assistant"
    content: str
    sources: list[SourceReference] = []
    confidence: str | None = None       # assistant only
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    created_at: datetime

    class Config:
        json_encoders = {UUID: str}


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]
    total: int


# ── SSE Stream event payloads ─────────────────────────────────────────────────
# The frontend handles these event types in its SSE listener.

class SSETokenEvent(BaseModel):
    type: str = "token"
    content: str


class SSESourcesEvent(BaseModel):
    type: str = "sources"
    sources: list[SourceReference]


class SSEDoneEvent(BaseModel):
    type: str = "done"
    message_id: UUID
    session_id: UUID
    confidence: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int

    class Config:
        json_encoders = {UUID: str}


class SSEErrorEvent(BaseModel):
    type: str = "error"
    message: str


class SSEMetaEvent(BaseModel):
    """Sent before tokens – lets the frontend start rendering."""
    type: str = "meta"
    session_id: UUID
    user_message_id: UUID

    class Config:
        json_encoders = {UUID: str}

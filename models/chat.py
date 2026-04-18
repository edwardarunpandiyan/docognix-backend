"""
models/chat.py – Pydantic schemas for conversations and chat messages.

Naming aligned with frontend:
  conversation_id  ← one chat thread  (was session_id in backend)
  anonymous_id     ← browser identity stored in localStorage ("br_7xk2m9p")
  user_id          ← null until login (future auth)

Identity flow:
  Before login:  anonymous_id = "br_7xk2m9p",  user_id = null
  After  login:  anonymous_id = "br_7xk2m9p",  user_id = "user_abc123"
  Backend merges all conversations where anonymous_id = "br_7xk2m9p"
                → reassigned to user_id = "user_abc123"
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from models.documents import SourceReference


# ── Conversations ─────────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    title: str = "New Chat"
    anonymous_id: str               # required – frontend always sends this


class ConversationUpdate(BaseModel):
    title: str


class ConversationResponse(BaseModel):
    conversation_id: UUID
    title: str
    anonymous_id: str | None = None
    user_id: str | None = None      # null until login
    document_count: int = 0
    total_chunks: int = 0
    message_count: int = 0
    last_message_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {UUID: str}


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    total: int


# ── Chat Messages ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Sent by the frontend to POST /chat/{conversation_id}"""
    message: str = Field(..., min_length=1, max_length=4000)
    anonymous_id: str               # frontend sends on every request
    document_ids: list[UUID] | None = None
    stream: bool = True


class MessageResponse(BaseModel):
    message_id: UUID
    conversation_id: UUID
    role: str                       # "user" | "assistant"
    content: str
    sources: list[SourceReference] = []
    confidence: str | None = None
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

class SSETokenEvent(BaseModel):
    type: str = "token"
    content: str


class SSESourcesEvent(BaseModel):
    type: str = "sources"
    sources: list[SourceReference]


class SSEDoneEvent(BaseModel):
    type: str = "done"
    message_id: UUID
    conversation_id: UUID
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
    type: str = "meta"
    conversation_id: UUID
    user_message_id: UUID

    class Config:
        json_encoders = {UUID: str}


# ── Future auth: claim anonymous conversations on login ───────────────────────

class ClaimConversationsRequest(BaseModel):
    """
    Called once after login. Reassigns all conversations with matching
    anonymous_id to the authenticated user_id.
    Frontend sends both IDs from localStorage + auth token.
    """
    anonymous_id: str
    user_id: str

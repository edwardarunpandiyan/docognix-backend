"""
routers/chat.py – Chat endpoints with SSE streaming.

POST /chat/{session_id}         → SSE stream (token-by-token)
GET  /chat/{session_id}/messages → Full message history
DELETE /chat/{session_id}/messages → Clear history
"""
from __future__ import annotations

import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from database.postgres import get_pool
from models.chat import ChatRequest, MessageListResponse, MessageResponse
from models.documents import SourceReference
from services.rag import stream_rag_response

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Stream chat ───────────────────────────────────────────────────────────────

@router.post("/{session_id}")
async def chat(session_id: UUID, body: ChatRequest):
    """
    Main chat endpoint. Returns a text/event-stream SSE response.

    SSE event types the frontend should handle:
      meta    → {type, session_id, user_message_id}
      sources → {type, sources: SourceReference[]}
      token   → {type, content: string}
      done    → {type, message_id, confidence, prompt_tokens, completion_tokens, latency_ms}
      error   → {type, message: string}
    """
    # Verify session exists
    pool = await get_pool()
    async with pool.acquire() as conn:
        sess = await conn.fetchval(
            "SELECT id FROM sessions WHERE id = $1", str(session_id)
        )
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_stream():
        async for chunk in stream_rag_response(
            session_id=str(session_id),
            query=body.message,
            document_ids=body.document_ids,
        ):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # Disable nginx buffering
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
    )


# ── Message history ────────────────────────────────────────────────────────────

@router.get("/{session_id}/messages", response_model=MessageListResponse)
async def get_messages(
    session_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    pool = await get_pool()

    # Verify session
    async with pool.acquire() as conn:
        sess = await conn.fetchval(
            "SELECT id FROM sessions WHERE id = $1", str(session_id)
        )
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, role, content, sources,
                   confidence, model,
                   prompt_tokens, completion_tokens, latency_ms,
                   created_at
            FROM messages
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
            """,
            str(session_id), limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM messages WHERE session_id = $1",
            str(session_id),
        )

    messages = []
    for r in rows:
        raw_sources = r["sources"] or []
        if isinstance(raw_sources, str):
            raw_sources = json.loads(raw_sources)
        sources = []
        for s in raw_sources:
            try:
                sources.append(SourceReference(**s))
            except Exception:
                pass   # skip malformed source

        messages.append(MessageResponse(
            message_id=r["id"],
            session_id=r["session_id"],
            role=r["role"],
            content=r["content"],
            sources=sources,
            confidence=r["confidence"],
            model=r["model"],
            prompt_tokens=r["prompt_tokens"],
            completion_tokens=r["completion_tokens"],
            latency_ms=r["latency_ms"],
            created_at=r["created_at"],
        ))

    return MessageListResponse(messages=messages, total=total)


# ── Clear messages ─────────────────────────────────────────────────────────────

@router.delete("/{session_id}/messages", status_code=204)
async def clear_messages(session_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        sess = await conn.fetchval(
            "SELECT id FROM sessions WHERE id = $1", str(session_id)
        )
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        await conn.execute(
            "DELETE FROM messages WHERE session_id = $1", str(session_id)
        )

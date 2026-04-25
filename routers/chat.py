"""
routers/chat.py – Chat endpoints with SSE streaming.

POST /chat/{conversation_id}            → SSE stream
GET  /chat/{conversation_id}/messages   → message history
DELETE /chat/{conversation_id}/messages → clear history

Guards:
  - Conversation must exist (404)
  - At least one document must have status = 'ready' (400)
    Prevents chat before any document is fully ingested.

Auto-title:
  Handled entirely inside services/rag.py — the title SSE event is
  yielded as the last event in the stream on the first message exchange.
  No BackgroundTasks or polling needed.
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


@router.post("/{conversation_id}")
async def chat(conversation_id: UUID, body: ChatRequest):
    """
    Main chat endpoint. Returns text/event-stream SSE response.

    SSE event types (in order):
      meta    → {type, conversation_id, user_message_id}
      sources → {type, sources: SourceReference[]}
      token   → {type, content: string}        (repeated)
      done    → {type, message_id, conversation_id, confidence,
                 prompt_tokens, completion_tokens, latency_ms}
      title   → {type, conversation_id, title}  (FIRST MESSAGE ONLY)
      error   → {type, message: string}         (on failure)

    The title event fires only on the first message exchange and gives the
    frontend everything it needs to update IndexedDB and the sidebar in one
    place without any polling.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Guard 1 — conversation must exist
        conv = await conn.fetchval(
            "SELECT id FROM conversations WHERE id = $1", str(conversation_id)
        )
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Guard 2 — at least one ready document required
        # Enforces upload-first flow: user cannot chat before a document
        # has been fully ingested. This prevents empty RAG context and
        # confusing non-answers.
        ready_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM documents
            WHERE conversation_id = $1 AND status = 'ready'
            """,
            str(conversation_id),
        )
        if ready_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Upload a document before asking questions.",
            )

    async def event_stream():
        async for chunk in stream_rag_response(
            conversation_id=str(conversation_id),
            query=body.message,
            document_ids=body.document_ids,
        ):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
    )


@router.get("/{conversation_id}/messages", response_model=MessageListResponse)
async def get_messages(
    conversation_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        conv = await conn.fetchval(
            "SELECT id FROM conversations WHERE id = $1", str(conversation_id)
        )
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        rows = await conn.fetch(
            """
            SELECT id, conversation_id, role, content, sources,
                   confidence, model,
                   prompt_tokens, completion_tokens, latency_ms,
                   created_at
            FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
            """,
            str(conversation_id), limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = $1",
            str(conversation_id),
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
                pass

        messages.append(MessageResponse(
            message_id=r["id"],
            conversation_id=r["conversation_id"],
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


@router.delete("/{conversation_id}/messages", status_code=204)
async def clear_messages(conversation_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        conv = await conn.fetchval(
            "SELECT id FROM conversations WHERE id = $1", str(conversation_id)
        )
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        await conn.execute(
            "DELETE FROM messages WHERE conversation_id = $1", str(conversation_id)
        )

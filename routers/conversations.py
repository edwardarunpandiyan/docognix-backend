"""
routers/conversations.py – CRUD for conversations.
(renamed from sessions.py to match frontend naming: conversation_id)

anonymous_id is required on create and used to list conversations
belonging to a specific browser identity.

user_id is null for now — populated when auth is added later.
POST /conversations/claim reassigns anonymous conversations to
an authenticated user on login.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from database.postgres import get_pool
from models.chat import (
    ClaimConversationsRequest,
    ConversationCreate, ConversationResponse,
    ConversationListResponse, ConversationUpdate,
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(body: ConversationCreate):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO conversations (title, anonymous_id, user_id)
            VALUES ($1, $2, NULL)
            RETURNING id, title, anonymous_id, user_id, created_at, updated_at
            """,
            body.title, body.anonymous_id,
        )
    return ConversationResponse(
        conversation_id=row["id"],
        title=row["title"],
        anonymous_id=row["anonymous_id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    anonymous_id: str = Query(..., description="Browser identity from localStorage"),
):
    """
    Returns all conversations for a given anonymous_id.
    Frontend always passes anonymous_id as a query param.

    Example: GET /api/v1/conversations?anonymous_id=br_7xk2m9p
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id, title, anonymous_id, user_id,
                created_at, updated_at,
                document_count, total_chunks,
                message_count, last_message_at
            FROM conversation_overview
            WHERE anonymous_id = $1
            ORDER BY updated_at DESC
            """,
            anonymous_id,
        )
    conversations = [
        ConversationResponse(
            conversation_id=r["id"],
            title=r["title"],
            anonymous_id=r["anonymous_id"],
            user_id=r["user_id"],
            document_count=r["document_count"],
            total_chunks=r["total_chunks"],
            message_count=r["message_count"],
            last_message_at=r["last_message_at"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]
    return ConversationListResponse(conversations=conversations, total=len(conversations))


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, title, anonymous_id, user_id,
                   created_at, updated_at,
                   document_count, total_chunks,
                   message_count, last_message_at
            FROM conversation_overview
            WHERE id = $1
            """,
            str(conversation_id),
        )
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        conversation_id=row["id"],
        title=row["title"],
        anonymous_id=row["anonymous_id"],
        user_id=row["user_id"],
        document_count=row["document_count"],
        total_chunks=row["total_chunks"],
        message_count=row["message_count"],
        last_message_at=row["last_message_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(conversation_id: UUID, body: ConversationUpdate):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE conversations SET title = $1, updated_at = now()
            WHERE id = $2
            RETURNING id, title, anonymous_id, user_id, created_at, updated_at
            """,
            body.title, str(conversation_id),
        )
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        conversation_id=row["id"],
        title=row["title"],
        anonymous_id=row["anonymous_id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM conversations WHERE id = $1", str(conversation_id)
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Conversation not found")


# ── Claim anonymous conversations on login ────────────────────────────────────

@router.post("/claim", status_code=200)
async def claim_conversations(body: ClaimConversationsRequest):
    """
    Called once after login.
    Reassigns all conversations with matching anonymous_id to user_id.

    Frontend flow:
      1. User logs in → gets user_id from auth
      2. Frontend calls POST /conversations/claim
         { anonymous_id: "br_7xk2m9p", user_id: "user_abc123" }
      3. Backend reassigns all matching conversations
      4. Frontend updates localStorage with user_id

    Currently a live stub — activate by wiring to Supabase Auth JWT.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE conversations
            SET user_id    = $1,
                updated_at = now()
            WHERE anonymous_id = $2
              AND user_id IS NULL
            """,
            body.user_id, body.anonymous_id,
        )
    updated_count = int(result.split()[-1])
    return {
        "claimed": updated_count,
        "anonymous_id": body.anonymous_id,
        "user_id": body.user_id,
    }

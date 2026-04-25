"""
routers/conversations.py – CRUD for conversations.

NOTE: Conversations are no longer created directly from this router.
They are created atomically during document upload via
POST /api/v1/documents/upload. The create_conversation endpoint is kept
for admin/testing purposes only.

List endpoint returns ONLY conversations that have at least one document
with status = 'ready'. This prevents ghost/empty conversations from
appearing in the sidebar when a user abandoned an upload mid-way.
"""
from __future__ import annotations

from uuid import UUID, uuid4

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
    """
    Create a conversation directly.
    NOTE: In normal frontend flow, conversations are created via
    POST /api/v1/documents/upload which handles identity resolution
    and document creation atomically. This endpoint exists for
    admin and testing purposes.
    """
    anonymous_id = body.anonymous_id.strip() or str(uuid4())
    db_user_id = body.user_id.strip() or None

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO conversations (title, anonymous_id, user_id)
            VALUES ($1, $2, $3)
            RETURNING id, title, anonymous_id, user_id, created_at, updated_at
            """,
            body.title, anonymous_id, db_user_id,
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
    anonymous_id: str = Query(..., description="Browser identity UUID from localStorage"),
):
    """
    Returns conversations for a given anonymous_id that have at least
    one document with status = 'ready'.

    Only ready conversations are shown — prevents ghost conversations
    (created but upload abandoned or failed) from appearing in the sidebar.

    Example: GET /api/v1/conversations?anonymous_id=550e8400-e29b-41d4-a716-446655440000
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                co.id, co.title, co.anonymous_id, co.user_id,
                co.created_at, co.updated_at,
                co.document_count, co.total_chunks,
                co.message_count, co.last_message_at
            FROM conversation_overview co
            WHERE co.anonymous_id = $1
              AND EXISTS (
                  SELECT 1 FROM documents d
                  WHERE d.conversation_id = co.id
                    AND d.status = 'ready'
              )
            ORDER BY co.updated_at DESC
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


@router.post("/claim", status_code=200)
async def claim_conversations(body: ClaimConversationsRequest):
    """
    Called once after login. Reassigns all conversations with matching
    anonymous_id to the authenticated user_id.
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

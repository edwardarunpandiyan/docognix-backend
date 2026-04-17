"""
routers/sessions.py – CRUD for chat sessions.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

from database.postgres import get_pool
from models.chat import (
    SessionCreate, SessionResponse, SessionListResponse, SessionUpdate,
)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(body: SessionCreate):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO sessions (title)
            VALUES ($1)
            RETURNING id, title, created_at, updated_at
            """,
            body.title,
        )
    return SessionResponse(
        session_id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id, title, created_at, updated_at,
                document_count, total_chunks, message_count, last_message_at
            FROM session_overview
            ORDER BY updated_at DESC
            """
        )
    sessions = [
        SessionResponse(
            session_id=r["id"],
            title=r["title"],
            document_count=r["document_count"],
            total_chunks=r["total_chunks"],
            message_count=r["message_count"],
            last_message_at=r["last_message_at"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, title, created_at, updated_at,
                   document_count, total_chunks, message_count, last_message_at
            FROM session_overview
            WHERE id = $1
            """,
            str(session_id),
        )
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=row["id"],
        title=row["title"],
        document_count=row["document_count"],
        total_chunks=row["total_chunks"],
        message_count=row["message_count"],
        last_message_at=row["last_message_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(session_id: UUID, body: SessionUpdate):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE sessions SET title = $1, updated_at = now()
            WHERE id = $2
            RETURNING id, title, created_at, updated_at
            """,
            body.title, str(session_id),
        )
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM sessions WHERE id = $1", str(session_id)
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Session not found")

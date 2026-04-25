"""
routers/documents.py – File upload, listing, deletion.

Two routers are exported:

  upload_router  → POST /documents/upload
      Entry point for every new conversation. Creates conversation +
      document atomically in one DB transaction. Returns conversation_id
      and anonymous_id so frontend can store both immediately.

  router  → /conversations/{conversation_id}/documents/*
      Status polling, listing, deletion — all existing routes unchanged.
"""
from __future__ import annotations

import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile, File

from config import settings
from database.postgres import get_pool
from models.documents import (
    DocumentListResponse, DocumentStatus, DocumentSummary,
    DocumentUploadResponse, UploadInitResponse,
)
from services.cache import cache_invalidate_conversation
from services.document_processor import ingest_document

log = logging.getLogger(__name__)

# ── New: top-level upload router ──────────────────────────────────────────────
upload_router = APIRouter(prefix="/documents", tags=["Documents"])

# ── Existing: conversation-scoped document router ────────────────────────────
router = APIRouter(prefix="/conversations/{conversation_id}/documents", tags=["Documents"])

MAGIC_BYTES: dict[str, bytes] = {
    "pdf":  b"%PDF",
    "docx": b"PK\x03\x04",
}


def detect_file_type(filename: str, first_bytes: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Allowed: {settings.allowed_extensions}",
        )
    if ext in MAGIC_BYTES:
        if not first_bytes.startswith(MAGIC_BYTES[ext]):
            raise HTTPException(
                status_code=415,
                detail=f"File content does not match .{ext} format",
            )
    return ext


# ── POST /api/v1/documents/upload ─────────────────────────────────────────────

@upload_router.post("/upload", response_model=UploadInitResponse, status_code=202)
async def upload_and_create_conversation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    anonymous_id: str = Form(default=""),
):
    """
    Entry point for every new conversation.

    Creates conversation + document atomically in a single DB transaction.
    Frontend calls this when the user selects a file — with or without
    an existing anonymous_id.

    Identity resolution:
      anonymous_id = ""   → first-ever upload from this browser.
                            Backend generates a UUID and returns it.
                            Frontend must store it in localStorage.
      anonymous_id = UUID → returning user. Existing identity used.
                            A new conversation_id is created for this upload.

    Frontend flow after receiving the response:
      1. Store anonymous_id in localStorage (always — backend may have generated it)
      2. Store conversation_id for this tab/chat
      3. Poll GET /conversations/{conversation_id}/documents/{document_id}/status
      4. When status = "ready" → unlock chat input
    """
    data = await file.read()
    file_size = len(data)

    if file_size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_file_size_mb} MB")
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    safe_name = file.filename or "document.txt"
    file_type = detect_file_type(safe_name, data[:8])

    # Resolve identity — generate UUID if first-time user
    resolved_anonymous_id = anonymous_id.strip() or str(uuid4())

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Create conversation and document atomically.
            # If either insert fails the whole transaction rolls back —
            # no orphaned conversations or documents are left in the DB.

            conv_row = await conn.fetchrow(
                """
                INSERT INTO conversations (title, anonymous_id, user_id)
                VALUES ('New Chat', $1, NULL)
                RETURNING id
                """,
                resolved_anonymous_id,
            )
            conversation_id: UUID = conv_row["id"]

            doc_row = await conn.fetchrow(
                """
                INSERT INTO documents
                    (conversation_id, filename, original_name, file_type, file_size, status)
                VALUES ($1, $2, $3, $4, $5, 'processing')
                RETURNING id
                """,
                str(conversation_id), safe_name, safe_name, file_type, file_size,
            )
            document_id: UUID = doc_row["id"]

    # Start background ingest outside the transaction so the connection
    # is released before the long-running CPU/network work begins.
    background_tasks.add_task(_run_ingest, document_id, conversation_id, data, file_type)

    return UploadInitResponse(
        conversation_id=conversation_id,
        document_id=document_id,
        anonymous_id=resolved_anonymous_id,
        filename=safe_name,
        file_type=file_type,
        file_size=file_size,
        status="processing",
    )


# ── Existing routes (unchanged) ───────────────────────────────────────────────

@router.post("", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    conversation_id: UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload an additional document to an existing conversation."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        conv = await conn.fetchval(
            "SELECT id FROM conversations WHERE id = $1", str(conversation_id)
        )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    data = await file.read()
    file_size = len(data)

    if file_size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_file_size_mb} MB")
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    file_type = detect_file_type(file.filename or "upload.txt", data[:8])
    safe_name = file.filename or f"document.{file_type}"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO documents
                (conversation_id, filename, original_name, file_type, file_size, status)
            VALUES ($1, $2, $3, $4, $5, 'processing')
            RETURNING id, created_at
            """,
            str(conversation_id), safe_name, safe_name, file_type, file_size,
        )

    doc_id: UUID = row["id"]
    background_tasks.add_task(cache_invalidate_conversation, str(conversation_id))
    background_tasks.add_task(_run_ingest, doc_id, conversation_id, data, file_type)

    return DocumentUploadResponse(
        document_id=doc_id,
        conversation_id=conversation_id,
        filename=safe_name,
        original_name=safe_name,
        file_type=file_type,
        file_size=file_size,
        page_count=None,
        word_count=None,
        chunk_count=0,
        status="processing",
        created_at=row["created_at"],
    )


async def _run_ingest(doc_id: UUID, conversation_id: UUID, data: bytes, file_type: str):
    try:
        await ingest_document(doc_id, conversation_id, data, file_type)
    except Exception as e:
        log.exception("Background ingest failed for %s: %s", doc_id, e)


@router.get("/{document_id}/status", response_model=DocumentStatus)
async def get_document_status(conversation_id: UUID, document_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT status, chunk_count, error_message
            FROM documents
            WHERE id = $1 AND conversation_id = $2
            """,
            str(document_id), str(conversation_id),
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatus(
        document_id=document_id,
        status=row["status"],
        chunk_count=row["chunk_count"] or 0,
        error_message=row["error_message"],
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(conversation_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, conversation_id, filename, original_name, file_type,
                   file_size, page_count, word_count, chunk_count,
                   status, created_at
            FROM documents
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            """,
            str(conversation_id),
        )
    docs = [
        DocumentSummary(
            document_id=r["id"],
            conversation_id=r["conversation_id"],
            filename=r["filename"],
            original_name=r["original_name"],
            file_type=r["file_type"],
            file_size=r["file_size"],
            page_count=r["page_count"],
            word_count=r["word_count"],
            chunk_count=r["chunk_count"] or 0,
            status=r["status"],
            created_at=r["created_at"],
        )
        for r in rows
    ]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{document_id}", response_model=DocumentSummary)
async def get_document(conversation_id: UUID, document_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, conversation_id, filename, original_name, file_type,
                   file_size, page_count, word_count, chunk_count,
                   status, created_at
            FROM documents
            WHERE id = $1 AND conversation_id = $2
            """,
            str(document_id), str(conversation_id),
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentSummary(
        document_id=row["id"],
        conversation_id=row["conversation_id"],
        filename=row["filename"],
        original_name=row["original_name"],
        file_type=row["file_type"],
        file_size=row["file_size"],
        page_count=row["page_count"],
        word_count=row["word_count"],
        chunk_count=row["chunk_count"] or 0,
        status=row["status"],
        created_at=row["created_at"],
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    conversation_id: UUID, document_id: UUID, background_tasks: BackgroundTasks
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE id = $1 AND conversation_id = $2",
            str(document_id), str(conversation_id),
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Document not found")
    background_tasks.add_task(cache_invalidate_conversation, str(conversation_id))

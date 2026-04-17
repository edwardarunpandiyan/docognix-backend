"""
routers/documents.py – File upload, listing, and deletion.

Upload flow:
  1. Validate file (magic bytes + extension + size)
  2. Create a document record in DB (status=processing)
  3. Launch background ingestion task (parse → chunk → embed → store)
  4. Return DocumentUploadResponse immediately (non-blocking)

The frontend polls GET /documents/{doc_id}/status or listens for
the status via the session overview endpoint.
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Path, Query
from fastapi.responses import JSONResponse

from config import settings
from database.postgres import get_pool
from models.documents import (
    DocumentListResponse, DocumentStatus, DocumentSummary, DocumentUploadResponse,
)
from services.cache import cache_invalidate_session
from services.document_processor import ingest_document

log = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions/{session_id}/documents", tags=["Documents"])


# ── Magic byte validation ──────────────────────────────────────────────────────
MAGIC_BYTES: dict[str, bytes] = {
    "pdf":  b"%PDF",
    "docx": b"PK\x03\x04",   # ZIP-based (OOXML)
}


def detect_file_type(filename: str, first_bytes: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Allowed: {settings.allowed_extensions}",
        )
    # Validate magic bytes for binary formats
    if ext in MAGIC_BYTES:
        if not first_bytes.startswith(MAGIC_BYTES[ext]):
            raise HTTPException(
                status_code=415,
                detail=f"File content does not match .{ext} format (magic byte mismatch)",
            )
    return ext


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post("", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # Verify session exists
    pool = await get_pool()
    async with pool.acquire() as conn:
        sess = await conn.fetchval(
            "SELECT id FROM sessions WHERE id = $1", str(session_id)
        )
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    # Read file
    data = await file.read()
    file_size = len(data)

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb} MB",
        )
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    file_type = detect_file_type(file.filename or "upload.txt", data[:8])
    safe_name = file.filename or f"document.{file_type}"

    # Create DB record
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO documents
                (session_id, filename, original_name, file_type, file_size, status)
            VALUES ($1, $2, $3, $4, $5, 'processing')
            RETURNING id, created_at
            """,
            str(session_id), safe_name, safe_name, file_type, file_size,
        )

    doc_id: UUID = row["id"]

    # Invalidate session cache (new document changes the knowledge base)
    background_tasks.add_task(cache_invalidate_session, str(session_id))

    # Run ingestion in background
    background_tasks.add_task(
        _run_ingest, doc_id, session_id, data, file_type
    )

    return DocumentUploadResponse(
        document_id=doc_id,
        session_id=session_id,
        filename=safe_name,
        original_name=safe_name,
        file_type=file_type,
        file_size=file_size,
        page_count=None,    # filled in after ingestion
        word_count=None,
        chunk_count=0,
        status="processing",
        created_at=row["created_at"],
    )


async def _run_ingest(doc_id: UUID, session_id: UUID, data: bytes, file_type: str):
    try:
        await ingest_document(doc_id, session_id, data, file_type)
    except Exception as e:
        log.exception("Background ingest failed for %s: %s", doc_id, e)


# ── Status poll ────────────────────────────────────────────────────────────────

@router.get("/{document_id}/status", response_model=DocumentStatus)
async def get_document_status(session_id: UUID, document_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT status, chunk_count, error_message
            FROM documents
            WHERE id = $1 AND session_id = $2
            """,
            str(document_id), str(session_id),
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatus(
        document_id=document_id,
        status=row["status"],
        chunk_count=row["chunk_count"] or 0,
        error_message=row["error_message"],
    )


# ── List documents ─────────────────────────────────────────────────────────────

@router.get("", response_model=DocumentListResponse)
async def list_documents(session_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, filename, original_name, file_type,
                   file_size, page_count, word_count, chunk_count,
                   status, created_at
            FROM documents
            WHERE session_id = $1
            ORDER BY created_at ASC
            """,
            str(session_id),
        )
    docs = [
        DocumentSummary(
            document_id=r["id"],
            session_id=r["session_id"],
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


# ── Get single document ────────────────────────────────────────────────────────

@router.get("/{document_id}", response_model=DocumentSummary)
async def get_document(session_id: UUID, document_id: UUID):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, session_id, filename, original_name, file_type,
                   file_size, page_count, word_count, chunk_count,
                   status, created_at
            FROM documents
            WHERE id = $1 AND session_id = $2
            """,
            str(document_id), str(session_id),
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentSummary(
        document_id=row["id"],
        session_id=row["session_id"],
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


# ── Delete document ────────────────────────────────────────────────────────────

@router.delete("/{document_id}", status_code=204)
async def delete_document(session_id: UUID, document_id: UUID, background_tasks: BackgroundTasks):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE id = $1 AND session_id = $2",
            str(document_id), str(session_id),
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Document not found")
    background_tasks.add_task(cache_invalidate_session, str(session_id))

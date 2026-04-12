"""
services/document_processor.py – Parse uploaded files and ingest into the DB.

Supports:
  • PDF  – via pdfplumber (text + page-level extraction)
  • DOCX – via python-docx  (paragraph + page estimation)
  • TXT  – raw text

Pipeline per document:
  1. Parse → list of (page_number, text) pairs
  2. Clean text
  3. Chunk pages-aware (chunk_pages)
  4. Batch-embed chunks
  5. Bulk-insert chunks into PostgreSQL
  6. Update document record (page_count, chunk_count, status)
"""
from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from uuid import UUID

import pdfplumber
from docx import Document as DocxDocument

from config import settings
from database.postgres import get_pool
from models.documents import ChunkCreate
from services.embedding import embed_document_chunks
from utils.text_utils import clean_text, chunk_pages, chunk_text, approx_token_count

log = logging.getLogger(__name__)


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_pdf(data: bytes) -> tuple[list[str], int]:
    """Return (pages_text_list, page_count). Each element is one page's text."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(clean_text(text))
    return pages, len(pages)


def parse_docx(data: bytes) -> tuple[list[str], int]:
    """Return (pseudo_pages, estimated_page_count).
    DOCX has no true page boundaries; we split every ~500 words to emulate pages.
    """
    doc = DocxDocument(io.BytesIO(data))
    full_text = "\n\n".join(
        p.text for p in doc.paragraphs if p.text.strip()
    )
    full_text = clean_text(full_text)

    words = full_text.split()
    page_size = 500   # words per pseudo-page
    pseudo_pages: list[str] = []
    for i in range(0, len(words), page_size):
        pseudo_pages.append(" ".join(words[i: i + page_size]))

    page_count = max(len(pseudo_pages), 1)
    return pseudo_pages, page_count


def parse_txt(data: bytes) -> tuple[list[str], int]:
    """Plain text – treat every ~500 words as a pseudo-page."""
    text = data.decode("utf-8", errors="replace")
    text = clean_text(text)
    words = text.split()
    page_size = 500
    pseudo_pages: list[str] = []
    for i in range(0, len(words), page_size):
        pseudo_pages.append(" ".join(words[i: i + page_size]))
    page_count = max(len(pseudo_pages), 1)
    return pseudo_pages, page_count


def count_words(pages: list[str]) -> int:
    return sum(len(p.split()) for p in pages)


# ── Main ingestion pipeline ────────────────────────────────────────────────────

async def ingest_document(
    document_id: UUID,
    session_id: UUID,
    file_data: bytes,
    file_type: str,       # "pdf" | "docx" | "txt"
) -> None:
    """
    Parse → chunk → embed → store.
    Updates document status in DB when done (or on error).
    """
    pool = await get_pool()

    try:
        # ── 1. Parse ──────────────────────────────────────────
        if file_type == "pdf":
            pages, page_count = await asyncio.get_event_loop().run_in_executor(
                None, parse_pdf, file_data
            )
        elif file_type == "docx":
            pages, page_count = await asyncio.get_event_loop().run_in_executor(
                None, parse_docx, file_data
            )
        else:   # txt
            pages, page_count = await asyncio.get_event_loop().run_in_executor(
                None, parse_txt, file_data
            )

        word_count = count_words(pages)

        # ── 2. Chunk ──────────────────────────────────────────
        raw_chunks = chunk_pages(
            pages,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        if not raw_chunks:
            raise ValueError("Document produced zero chunks – may be empty or image-only.")

        texts = [c["content"] for c in raw_chunks]

        # ── 3. Embed ──────────────────────────────────────────
        # Process in batches of 64 to avoid OOM
        all_embeddings: list[list[float]] = []
        batch = 64
        for i in range(0, len(texts), batch):
            embs = await embed_document_chunks(texts[i: i + batch])
            all_embeddings.extend(embs)

        # ── 4. Bulk-insert chunks ─────────────────────────────
        records = [
            (
                str(document_id),
                str(session_id),
                chunk["content"],
                chunk["content"].lower(),
                chunk.get("page_number"),
                chunk.get("page_end"),
                chunk["chunk_index"],
                chunk.get("char_start"),
                chunk.get("char_end"),
                chunk["token_count"],
                emb,    # list[float] – registered codec converts to vector literal
            )
            for chunk, emb in zip(raw_chunks, all_embeddings)
        ]

        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO chunks (
                    document_id, session_id,
                    content, content_lower,
                    page_number, page_end,
                    chunk_index, char_start, char_end,
                    token_count, embedding
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                """,
                records,
            )

            # ── 5. Update document record ─────────────────────
            await conn.execute(
                """
                UPDATE documents
                SET status      = 'ready',
                    page_count  = $1,
                    word_count  = $2,
                    chunk_count = $3,
                    updated_at  = now()
                WHERE id = $4
                """,
                page_count, word_count, len(raw_chunks), str(document_id),
            )

        log.info(
            "Ingested document %s → %d pages, %d chunks",
            document_id, page_count, len(raw_chunks),
        )

    except Exception as exc:
        log.exception("Ingestion failed for document %s: %s", document_id, exc)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE documents
                SET status = 'error', error_message = $1, updated_at = now()
                WHERE id = $2
                """,
                str(exc)[:500], str(document_id),
            )
        raise

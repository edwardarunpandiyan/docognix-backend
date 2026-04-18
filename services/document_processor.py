"""
services/document_processor.py – Parse uploaded files and ingest into the DB.
Uses conversation_id (renamed from session_id) throughout.
"""
from __future__ import annotations

import asyncio
import io
import logging
from uuid import UUID

import pdfplumber
from docx import Document as DocxDocument

from config import settings
from database.postgres import get_pool
from services.embedding import embed_document_chunks
from utils.text_utils import clean_text, chunk_pages

log = logging.getLogger(__name__)


def parse_pdf(data: bytes) -> tuple[list[str], int]:
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(clean_text(text))
    return pages, len(pages)


def parse_docx(data: bytes) -> tuple[list[str], int]:
    doc = DocxDocument(io.BytesIO(data))
    full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    full_text = clean_text(full_text)
    words = full_text.split()
    page_size = 500
    pseudo_pages = [" ".join(words[i: i + page_size]) for i in range(0, len(words), page_size)]
    return pseudo_pages, max(len(pseudo_pages), 1)


def parse_txt(data: bytes) -> tuple[list[str], int]:
    text = clean_text(data.decode("utf-8", errors="replace"))
    words = text.split()
    page_size = 500
    pseudo_pages = [" ".join(words[i: i + page_size]) for i in range(0, len(words), page_size)]
    return pseudo_pages, max(len(pseudo_pages), 1)


def count_words(pages: list[str]) -> int:
    return sum(len(p.split()) for p in pages)


async def ingest_document(
    document_id: UUID,
    conversation_id: UUID,
    file_data: bytes,
    file_type: str,
) -> None:
    pool = await get_pool()

    try:
        if file_type == "pdf":
            pages, page_count = await asyncio.get_event_loop().run_in_executor(
                None, parse_pdf, file_data
            )
        elif file_type == "docx":
            pages, page_count = await asyncio.get_event_loop().run_in_executor(
                None, parse_docx, file_data
            )
        else:
            pages, page_count = await asyncio.get_event_loop().run_in_executor(
                None, parse_txt, file_data
            )

        word_count = count_words(pages)
        raw_chunks = chunk_pages(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)

        if not raw_chunks:
            raise ValueError("Document produced zero chunks – may be empty or image-only.")

        texts = [c["content"] for c in raw_chunks]

        all_embeddings: list[list[float]] = []
        batch = 64
        for i in range(0, len(texts), batch):
            embs = await embed_document_chunks(texts[i: i + batch])
            all_embeddings.extend(embs)

        records = [
            (
                str(document_id),
                str(conversation_id),
                chunk["content"],
                chunk["content"].lower(),
                chunk.get("page_number"),
                chunk.get("page_end"),
                chunk["chunk_index"],
                chunk.get("char_start"),
                chunk.get("char_end"),
                chunk["token_count"],
                emb,
            )
            for chunk, emb in zip(raw_chunks, all_embeddings)
        ]

        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO chunks (
                    document_id, conversation_id,
                    content, content_lower,
                    page_number, page_end,
                    chunk_index, char_start, char_end,
                    token_count, embedding
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                """,
                records,
            )
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

        log.info("Ingested document %s → %d pages, %d chunks", document_id, page_count, len(raw_chunks))

    except Exception as exc:
        log.exception("Ingestion failed for document %s: %s", document_id, exc)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET status = 'error', error_message = $1, updated_at = now() WHERE id = $2",
                str(exc)[:500], str(document_id),
            )
        raise

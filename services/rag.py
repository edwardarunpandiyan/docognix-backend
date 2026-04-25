"""
services/rag.py – Core RAG pipeline with SSE streaming.
Uses conversation_id (renamed from session_id) throughout.

SSE event order:
  meta → sources → token × N → done → title (first message only)

The title event is the last thing yielded on the very first message exchange
(message_count == 2: one user + one assistant). Frontend uses it to update
the conversation title in IndexedDB and the sidebar simultaneously.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator
from uuid import UUID, uuid4

from groq import Groq

from config import settings
from database.postgres import get_pool
from models.chat import (
    SSEDoneEvent, SSEErrorEvent, SSEMetaEvent, SSESourcesEvent,
    SSETitleEvent, SSETokenEvent,
)
from models.documents import SourceReference
from services.cache import cache_lookup, cache_store
from services.embedding import embed_query
from services.retrieval import hybrid_retrieve, score_to_confidence
from services.title_generator import generate_title
from utils.text_utils import approx_token_count

log = logging.getLogger(__name__)


def _get_groq() -> Groq:
    return Groq(api_key=settings.groq_api_key)


SYSTEM_PROMPT = """You are Docognix, an expert document analyst. Your role is to answer questions about uploaded documents with precision and clarity.

Guidelines:
- Answer ONLY from the provided document context. Do not hallucinate facts.
- If the answer is not in the context, say: "I couldn't find that in the uploaded documents."
- Cite specific page numbers when relevant (e.g., "According to page 3...").
- Be concise and direct. Structure longer answers with clear paragraphs.
- If multiple documents are relevant, synthesize information across them.
- Maintain a professional, helpful tone.
- For technical content, preserve terminology exactly as it appears in the source.
"""


def build_context(sources: list[SourceReference], max_tokens: int) -> str:
    ordered = sorted(sources, key=lambda s: (s.document_name, s.page_number or 0))
    parts: list[str] = []
    used_tokens = 0

    for src in ordered:
        page_info = f"(Page {src.page_number})" if src.page_number else ""
        block = f"[{src.document_name} {page_info}]\n{src.content}"
        block_tokens = approx_token_count(block)

        if used_tokens + block_tokens > max_tokens:
            remaining = max_tokens - used_tokens
            if remaining > 100:
                words = block.split()
                parts.append(" ".join(words[: remaining * 4 // 5]) + " [truncated]")
            break

        parts.append(block)
        used_tokens += block_tokens

    return "\n\n---\n\n".join(parts)


async def load_recent_history(conversation_id: str, limit: int = 6) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            conversation_id, limit,
        )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


async def persist_user_message(conversation_id: str, content: str) -> UUID:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO messages (id, conversation_id, role, content)
            VALUES (uuid_generate_v4(), $1, 'user', $2)
            RETURNING id
            """,
            conversation_id, content,
        )
        await conn.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = $1", conversation_id
        )
    return row["id"]


async def persist_assistant_message(
    conversation_id: str,
    content: str,
    sources: list[SourceReference],
    confidence: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
) -> UUID:
    pool = await get_pool()
    sources_json = json.dumps([s.model_dump(mode="json") for s in sources], default=str)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO messages (
                id, conversation_id, role, content, sources,
                confidence, model, prompt_tokens, completion_tokens, latency_ms
            )
            VALUES (uuid_generate_v4(), $1, 'assistant', $2, $3::jsonb,
                    $4, $5, $6, $7, $8)
            RETURNING id
            """,
            conversation_id, content, sources_json,
            confidence, model, prompt_tokens, completion_tokens, latency_ms,
        )
    return row["id"]


def sse(event_type: str, data: dict) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


async def stream_rag_response(
    conversation_id: str,
    query: str,
    document_ids: list[UUID] | None = None,
) -> AsyncGenerator[str, None]:
    start_time = time.time()
    groq = _get_groq()

    try:
        query_embedding = await embed_query(query)

        # ── Semantic cache check ─────────────────────────────
        cached = await cache_lookup(conversation_id, query_embedding)
        if cached:
            user_msg_id = await persist_user_message(conversation_id, query)
            yield sse("meta", {"type": "meta", "conversation_id": conversation_id,
                                "user_message_id": str(user_msg_id)})

            cached_sources = [SourceReference(**s) for s in cached["sources"]]
            yield sse("sources", SSESourcesEvent(sources=cached_sources).model_dump(mode="json"))

            words = cached["answer"].split()
            for i in range(0, len(words), 5):
                chunk = " ".join(words[i:i+5]) + (" " if i + 5 < len(words) else "")
                yield sse("token", SSETokenEvent(content=chunk).model_dump())
                await asyncio.sleep(0.01)

            msg_id = await persist_assistant_message(
                conversation_id, cached["answer"], cached_sources,
                cached["confidence"], "cache", 0, 0,
                int((time.time() - start_time) * 1000),
            )
            yield sse("done", SSEDoneEvent(
                message_id=msg_id,
                conversation_id=UUID(conversation_id),
                confidence=cached["confidence"],
                prompt_tokens=0, completion_tokens=0,
                latency_ms=int((time.time() - start_time) * 1000),
            ).model_dump(mode="json"))

            # Title check even for cached responses
            async for title_event in _maybe_yield_title(conversation_id, query):
                yield title_event
            return

        # ── Persist user message ─────────────────────────────
        user_msg_id = await persist_user_message(conversation_id, query)
        yield sse("meta", SSEMetaEvent(
            conversation_id=UUID(conversation_id),
            user_message_id=user_msg_id,
        ).model_dump(mode="json"))

        # ── Hybrid retrieval ─────────────────────────────────
        sources, top_score = await hybrid_retrieve(
            conversation_id=conversation_id,
            query=query,
            groq_client=groq,
            document_ids=document_ids,
        )

        confidence = score_to_confidence(top_score)
        yield sse("sources", SSESourcesEvent(sources=sources).model_dump(mode="json"))

        # ── Build prompt ─────────────────────────────────────
        context = build_context(sources, settings.max_context_tokens)
        history = await load_recent_history(conversation_id)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": f"Document context:\n\n{context}\n\n---\n\nQuestion: {query}",
            },
        ]

        # ── Stream LLM ────────────────────────────────────────
        full_answer = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            stream = groq.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_answer += delta.content
                    yield sse("token", SSETokenEvent(content=delta.content).model_dump())
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0

        except Exception as groq_err:
            log.warning("Primary model failed, falling back: %s", groq_err)
            stream = groq.chat.completions.create(
                model=settings.groq_fallback_model,
                messages=messages,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_answer += delta.content
                    yield sse("token", SSETokenEvent(content=delta.content).model_dump())

        latency_ms = int((time.time() - start_time) * 1000)

        # ── Persist assistant message ─────────────────────────
        msg_id = await persist_assistant_message(
            conversation_id=conversation_id,
            content=full_answer,
            sources=sources,
            confidence=confidence,
            model=settings.groq_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

        # ── Cache result ──────────────────────────────────────
        if full_answer and sources:
            await cache_store(
                conversation_id=conversation_id,
                query_embedding=query_embedding,
                answer=full_answer,
                sources=[s.model_dump(mode="json") for s in sources],
                confidence=confidence,
            )

        # ── Done event ────────────────────────────────────────
        yield sse("done", SSEDoneEvent(
            message_id=msg_id,
            conversation_id=UUID(conversation_id),
            confidence=confidence,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        ).model_dump(mode="json"))

        # ── Title event (first message only) ──────────────────
        # Fires after done so the user sees the full answer first.
        # generate_title takes ~200-400ms — acceptable as the very last
        # event before the stream closes.
        async for title_event in _maybe_yield_title(conversation_id, query):
            yield title_event

    except Exception as exc:
        log.exception("RAG pipeline error for conversation %s: %s", conversation_id, exc)
        yield sse("error", SSEErrorEvent(message=str(exc)).model_dump())


async def _maybe_yield_title(
    conversation_id: str,
    query: str,
) -> AsyncGenerator[str, None]:
    """
    Check if this is the first message exchange (message_count == 2).
    If yes: generate a title, update DB, yield a title SSE event.
    If no:  yield nothing — generator exits immediately.

    Called after done event so the title never delays the answer.
    """
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            message_count = await conn.fetchval(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = $1",
                conversation_id,
            )

        # Only on first exchange: 1 user message + 1 assistant = exactly 2
        if message_count != 2:
            return

        # Check title is still default before calling Groq
        async with pool.acquire() as conn:
            current_title = await conn.fetchval(
                "SELECT title FROM conversations WHERE id = $1",
                conversation_id,
            )
        if current_title != "New Chat":
            return

        # Generate title
        title = await generate_title(query)
        if not title:
            return

        # Persist
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE conversations
                SET title = $1, updated_at = now()
                WHERE id = $2 AND title = 'New Chat'
                """,
                title[:100], conversation_id,
            )
        log.info("Auto-titled conversation %s → '%s'", conversation_id, title)

        yield sse("title", SSETitleEvent(
            conversation_id=UUID(conversation_id),
            title=title,
        ).model_dump(mode="json"))

    except Exception as e:
        # Title generation must never break the stream
        log.warning("Title event failed for %s: %s", conversation_id, e)

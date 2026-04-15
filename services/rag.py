"""
services/rag.py – Core RAG pipeline with SSE streaming.

Flow:
  1. Check semantic cache (Redis) → CACHE HIT yields stored answer.
  2. Hybrid retrieve chunks (vector + keyword + RRF + HyDE).
  3. Build optimised prompt (context compression, chat history injection).
  4. Stream LLM response token-by-token via Groq SDK.
  5. Persist user message + assistant message in DB.
  6. Store result in semantic cache.

SSE event types (match frontend expectations):
  • meta      – fired before tokens (message_id, session_id)
  • token     – each streamed token chunk
  • sources   – retrieved source citations (sent before tokens)
  • done      – final metadata (confidence, token counts, latency)
  • error     – on any failure
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
    SSEDoneEvent, SSEErrorEvent, SSEMetaEvent, SSESourcesEvent, SSETokenEvent,
)
from models.documents import SourceReference
from services.cache import cache_lookup, cache_store
from services.embedding import embed_query
from services.retrieval import hybrid_retrieve, score_to_confidence
from utils.text_utils import approx_token_count

log = logging.getLogger(__name__)


# ── Groq client (module-level singleton) ──────────────────────────────────────

def _get_groq() -> Groq:
    return Groq(api_key=settings.groq_api_key)


# ── System prompt ──────────────────────────────────────────────────────────────

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


# ── Context builder ────────────────────────────────────────────────────────────

def build_context(sources: list[SourceReference], max_tokens: int) -> str:
    """
    Assemble context string from retrieved chunks.
    Orders chunks by document then page to give the LLM coherent reading order.
    Truncates if total tokens would exceed max_tokens.
    """
    # Sort by document name → page_number for coherent reading order
    ordered = sorted(
        sources,
        key=lambda s: (s.document_name, s.page_number or 0),
    )

    parts: list[str] = []
    used_tokens = 0

    for src in ordered:
        page_info = f"(Page {src.page_number})" if src.page_number else ""
        header = f"[{src.document_name} {page_info}]"
        block = f"{header}\n{src.content}"
        block_tokens = approx_token_count(block)

        if used_tokens + block_tokens > max_tokens:
            # Include a truncated version if we have budget
            remaining = max_tokens - used_tokens
            if remaining > 100:
                words = block.split()
                truncated = " ".join(words[: remaining * 4 // 5])
                parts.append(truncated + " [truncated]")
            break

        parts.append(block)
        used_tokens += block_tokens

    return "\n\n---\n\n".join(parts)


# ── Chat history loader ────────────────────────────────────────────────────────

async def load_recent_history(session_id: str, limit: int = 6) -> list[dict]:
    """
    Fetch the last N turns from the DB to maintain conversational context.
    Returned in Groq API message format.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content FROM messages
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id, limit,
        )
    # Reverse so oldest is first
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    return history


# ── Message persistence ────────────────────────────────────────────────────────

async def persist_user_message(session_id: str, content: str) -> UUID:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO messages (id, session_id, role, content)
            VALUES (uuid_generate_v4(), $1, 'user', $2)
            RETURNING id
            """,
            session_id, content,
        )
        # Bump session updated_at
        await conn.execute(
            "UPDATE sessions SET updated_at = now() WHERE id = $1", session_id
        )
    return row["id"]


async def persist_assistant_message(
    session_id: str,
    content: str,
    sources: list[SourceReference],
    confidence: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
) -> UUID:
    pool = await get_pool()
    sources_json = json.dumps(
        [s.model_dump(mode="json") for s in sources],
        default=str,
    )
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO messages (
                id, session_id, role, content, sources,
                confidence, model,
                prompt_tokens, completion_tokens, latency_ms
            )
            VALUES (
                uuid_generate_v4(), $1, 'assistant', $2, $3::jsonb,
                $4, $5, $6, $7, $8
            )
            RETURNING id
            """,
            session_id, content, sources_json,
            confidence, model, prompt_tokens, completion_tokens, latency_ms,
        )
    return row["id"]


# ── SSE helpers ────────────────────────────────────────────────────────────────

def sse(event_type: str, data: dict) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


# ── Main streaming RAG pipeline ────────────────────────────────────────────────

async def stream_rag_response(
    session_id: str,
    query: str,
    document_ids: list[UUID] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted strings.
    Designed to be used directly with FastAPI StreamingResponse.
    """
    start_time = time.time()
    groq = _get_groq()

    try:
        # ── 1. Embed query for cache lookup ─────────────────────
        query_embedding = await embed_query(query)

        # ── 2. Semantic cache check ──────────────────────────────
        cached = await cache_lookup(session_id, query_embedding)
        if cached:
            user_msg_id = await persist_user_message(session_id, query)
            yield sse("meta", {"type": "meta", "session_id": session_id,
                                "user_message_id": str(user_msg_id)})

            cached_sources = [SourceReference(**s) for s in cached["sources"]]
            yield sse("sources", SSESourcesEvent(sources=cached_sources).model_dump(mode="json"))

            # Stream cached answer word-by-word for consistent UX
            words = cached["answer"].split()
            for i in range(0, len(words), 5):
                chunk = " ".join(words[i:i+5]) + (" " if i + 5 < len(words) else "")
                yield sse("token", SSETokenEvent(content=chunk).model_dump())
                await asyncio.sleep(0.01)

            msg_id = await persist_assistant_message(
                session_id, cached["answer"], cached_sources,
                cached["confidence"], "cache", 0, 0,
                int((time.time() - start_time) * 1000),
            )
            yield sse("done", SSEDoneEvent(
                message_id=msg_id, session_id=UUID(session_id),
                confidence=cached["confidence"],
                prompt_tokens=0, completion_tokens=0,
                latency_ms=int((time.time() - start_time) * 1000),
            ).model_dump(mode="json"))
            return

        # ── 3. Persist user message ──────────────────────────────
        user_msg_id = await persist_user_message(session_id, query)
        yield sse("meta", SSEMetaEvent(
            session_id=UUID(session_id), user_message_id=user_msg_id
        ).model_dump(mode="json"))

        # ── 4. Hybrid retrieval ──────────────────────────────────
        sources, top_score = await hybrid_retrieve(
            session_id=session_id,
            query=query,
            groq_client=groq,
            document_ids=document_ids,
        )

        confidence = score_to_confidence(top_score)
        yield sse("sources", SSESourcesEvent(sources=sources).model_dump(mode="json"))

        # ── 5. Build prompt ──────────────────────────────────────
        context = build_context(sources, settings.max_context_tokens)
        history = await load_recent_history(session_id)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": (
                    f"Document context:\n\n{context}\n\n"
                    f"---\n\nQuestion: {query}"
                ),
            },
        ]

        # ── 6. Stream LLM response ───────────────────────────────
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

                # Extract usage from final chunk
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0

        except Exception as groq_err:
            # Fallback to smaller/faster model
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

        # ── 7. Persist assistant message ─────────────────────────
        msg_id = await persist_assistant_message(
            session_id=session_id,
            content=full_answer,
            sources=sources,
            confidence=confidence,
            model=settings.groq_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

        # ── 8. Cache the result ───────────────────────────────────
        if full_answer and sources:
            await cache_store(
                session_id=session_id,
                query_embedding=query_embedding,
                answer=full_answer,
                sources=[s.model_dump(mode="json") for s in sources],
                confidence=confidence,
            )

        # ── 9. Done event ─────────────────────────────────────────
        yield sse("done", SSEDoneEvent(
            message_id=msg_id,
            session_id=UUID(session_id),
            confidence=confidence,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        ).model_dump(mode="json"))

    except Exception as exc:
        log.exception("RAG pipeline error for session %s: %s", session_id, exc)
        yield sse("error", SSEErrorEvent(message=str(exc)).model_dump())

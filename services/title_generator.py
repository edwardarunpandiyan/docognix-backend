"""
services/title_generator.py – Generate a conversation title from the user's first message.

Returns a plain string — all DB writing and SSE event emission happens
in services/rag.py immediately after the done event is yielded.

Uses the fast fallback model (llama-3.1-8b-instant) — typically responds
in 200-400ms. Called inline in the SSE stream so the title event reaches
the frontend before the connection closes.
"""
from __future__ import annotations

import logging

from groq import AsyncGroq

from config import settings

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Generate a concise, descriptive title (4 to 7 words) for a document Q&A "
    "conversation based on the user's first question. The title should reflect "
    "what the user is trying to find out. Return only the title — no quotes, "
    "no punctuation at the end, no preamble."
)


async def generate_title(first_user_message: str) -> str | None:
    """
    Generate a 4-7 word title from the user's first message.

    Returns the title string, or None if generation fails for any reason.
    Failures are logged as warnings but never raised — the caller decides
    what to do when None is returned (typically: skip the title SSE event,
    keep 'New Chat' in DB).
    """
    try:
        client = AsyncGroq(api_key=settings.groq_api_key)
        response = await client.chat.completions.create(
            model=settings.groq_fallback_model,   # llama-3.1-8b-instant — fast + cheap
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": first_user_message[:300]},
            ],
            max_tokens=20,
            temperature=0.4,
        )
        raw = response.choices[0].message.content or ""
        title = raw.strip().strip('"').strip("'")
        return title if title else None

    except Exception as e:
        log.warning("Title generation failed: %s", e)
        return None

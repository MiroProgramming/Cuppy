"""Call Gemini with backup models, for when Google's servers are overloaded."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from google.genai import errors

logger = logging.getLogger("gemini-fallback")

DEFAULT_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest"]
# Busy (503), out of quota (429), or not available on this key (404).
FALLBACK_CODES = {404, 429, 500, 503, 504}
RETRY_DELAY_SECONDS = 2


def model_chain(env_var: str) -> list[str]:
    """The preferred model from the environment first, then the backups."""
    preferred = os.getenv(env_var, "").strip()
    chain = [preferred] if preferred else []
    return chain + [m for m in DEFAULT_MODELS if m != preferred]


async def generate_with_fallback(
    client: Any,
    models: list[str],
    contents: Any,
    config: Any = None,
    retry_delay: float = RETRY_DELAY_SECONDS,
):
    """Try each model in turn; move on when one is busy or unavailable."""
    last_error: Exception | None = None
    for index, model in enumerate(models):
        try:
            return await client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except errors.APIError as exc:
            if exc.code not in FALLBACK_CODES:
                raise
            last_error = exc
            logger.warning(
                "Gemini model %s unavailable (%s); trying next", model, exc.code
            )
            if index < len(models) - 1 and retry_delay:
                await asyncio.sleep(retry_delay)
    raise last_error  # every model failed

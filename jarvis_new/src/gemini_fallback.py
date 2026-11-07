"""Call Gemini with backup models, for when Google's servers are overloaded.

Models that just failed are put on a cooldown and tried last, so one busy or
retired model doesn't slow down every reply:
- busy, out of quota, or hanging: skipped for 10 minutes
- not found (retired, or not on this key): skipped for a day
The model that worked last is tried first next time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from google.genai import errors

logger = logging.getLogger("gemini-fallback")

DEFAULT_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
]
FALLBACK_CODES = {404, 429, 500, 502, 503, 504}
BUSY_COOLDOWN_SECONDS = 10 * 60
MISSING_COOLDOWN_SECONDS = 24 * 60 * 60
# A busy model sometimes hangs instead of failing; move on rather than wait.
CALL_TIMEOUT_SECONDS = 30

_cooldown_until: dict[str, float] = {}
_last_good: dict[tuple[str, ...], str] = {}


def model_chain(env_var: str) -> list[str]:
    """The preferred model from the environment first, then the backups."""
    preferred = os.getenv(env_var, "").strip()
    chain = [preferred] if preferred else []
    return chain + [m for m in DEFAULT_MODELS if m != preferred]


def reset_health() -> None:
    _cooldown_until.clear()
    _last_good.clear()


def order_models(models: list[str], now: float | None = None) -> list[str]:
    """Last model that worked first, models on cooldown last."""
    now = time.monotonic() if now is None else now
    last_good = _last_good.get(tuple(models))
    ranked = sorted(
        models,
        key=lambda m: (
            _cooldown_until.get(m, 0) > now,  # cooling down goes last
            m != last_good,  # last working model goes first
            models.index(m),  # otherwise keep the preferred order
        ),
    )
    return ranked


def _cool(model: str, seconds: float) -> None:
    _cooldown_until[model] = time.monotonic() + seconds


async def generate_with_fallback(
    client: Any,
    models: list[str],
    contents: Any,
    config: Any = None,
    timeout: float = CALL_TIMEOUT_SECONDS,
):
    """Try models in order of recent health; move on when one fails."""
    last_error: Exception | None = None
    for model in order_models(models):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model, contents=contents, config=config
                ),
                timeout,
            )
        except TimeoutError as exc:
            last_error = exc
            _cool(model, BUSY_COOLDOWN_SECONDS)
            logger.warning("Gemini model %s timed out; trying next", model)
            continue
        except errors.APIError as exc:
            if exc.code not in FALLBACK_CODES:
                raise
            last_error = exc
            _cool(
                model,
                MISSING_COOLDOWN_SECONDS if exc.code == 404 else BUSY_COOLDOWN_SECONDS,
            )
            logger.warning(
                "Gemini model %s unavailable (%s); trying next", model, exc.code
            )
            continue
        _cooldown_until.pop(model, None)
        _last_good[tuple(models)] = model
        return response
    raise last_error  # every model failed

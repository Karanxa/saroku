"""
saroku.adapters._retry — shared exponential-backoff retry helper for adapters.

Used by openai.py, compat.py, and anthropic.py so all three providers get
the same practical resilience against transient failures (rate limits,
5xx/529 overload) instead of three independent copies of the same loop.
"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

DEFAULT_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
DEFAULT_MAX_RETRIES = 6

T = TypeVar("T")


async def with_retry(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    rate_limit_exc: type[BaseException],
    api_status_exc: type[BaseException],
    retryable_status_codes: set[int] = DEFAULT_RETRYABLE_STATUS_CODES,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> T:
    """
    Call coro_fn() with exponential backoff on transient failures.

    rate_limit_exc is always retried (until max_retries is exhausted).
    api_status_exc is retried only if its .status_code is in
    retryable_status_codes — anything else (e.g. 400, 401, 404) raises
    immediately, since retrying a non-transient error just wastes time.
    """
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except rate_limit_exc:
            if attempt == max_retries - 1:
                raise
        except api_status_exc as e:
            if attempt == max_retries - 1:
                raise
            if getattr(e, "status_code", None) not in retryable_status_codes:
                raise
        wait = (2 ** attempt) + random.uniform(0, 1)
        await asyncio.sleep(wait)

"""
saroku.adapters.openai — OpenAI ModelAdapter for SafetyGuard Layer-3 judge.
"""

from __future__ import annotations

import asyncio
import random

import openai
from openai import AsyncOpenAI

from saroku.adapters.base import ModelAdapter

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
_MAX_RETRIES = 6


async def _retry(coro_fn):
    for attempt in range(_MAX_RETRIES):
        try:
            return await coro_fn()
        except openai.RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                raise
        except openai.APIStatusError as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            if e.status_code not in _RETRYABLE_STATUS_CODES:
                raise
        wait = (2 ** attempt) + random.uniform(0, 1)
        await asyncio.sleep(wait)


class OpenAIAdapter(ModelAdapter):
    """
    ModelAdapter for OpenAI models (GPT-4o, GPT-4o-mini, o1, etc.).

    Reads OPENAI_API_KEY from the environment automatically.

    Example::

        guard = SafetyGuard(judge_model="gpt-4o-mini")
        # or explicitly:
        guard = SafetyGuard(judge_model="openai:gpt-4o-mini")
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = AsyncOpenAI()

    async def achat(self, prompt: str) -> str:
        client = self._client
        response = await _retry(
            lambda: client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        )
        return response.choices[0].message.content.strip()

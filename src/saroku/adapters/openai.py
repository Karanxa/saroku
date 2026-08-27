"""
saroku.adapters.openai — OpenAI ModelAdapter for SafetyGuard Layer-3 judge.
"""

from __future__ import annotations

import openai
from openai import AsyncOpenAI

from saroku.adapters._retry import with_retry
from saroku.adapters.base import ModelAdapter


async def _retry(coro_fn):
    return await with_retry(
        coro_fn,
        rate_limit_exc=openai.RateLimitError,
        api_status_exc=openai.APIStatusError,
    )


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

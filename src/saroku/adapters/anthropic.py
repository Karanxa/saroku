"""
saroku.adapters.anthropic — Anthropic Claude ModelAdapter for SafetyGuard.

Requires: pip install anthropic
"""

from __future__ import annotations

from saroku.adapters._retry import with_retry
from saroku.adapters.base import ModelAdapter


class AnthropicAdapter(ModelAdapter):
    """
    ModelAdapter for Anthropic Claude models.

    Reads ANTHROPIC_API_KEY from the environment automatically.

    Example::

        guard = SafetyGuard(judge_model="anthropic:claude-3-5-haiku-20241022")
        guard = SafetyGuard(judge_model="anthropic:claude-opus-4-6")
    """

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self.model = model
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic()
            self._rate_limit_exc = anthropic.RateLimitError
            self._api_status_exc = anthropic.APIStatusError
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicAdapter. "
                "Install it with: pip install anthropic"
            )

    async def achat(self, prompt: str) -> str:
        client = self._client

        async def _call():
            return await client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )

        # Anthropic's RateLimitError and its 5xx/529-overloaded errors
        # (OverloadedError, InternalServerError, ServiceUnavailableError,
        # etc.) are all subclasses of APIStatusError carrying .status_code,
        # so the same generic retry logic used for OpenAI applies directly —
        # no separate exception plumbing needed.
        message = await with_retry(
            _call,
            rate_limit_exc=self._rate_limit_exc,
            api_status_exc=self._api_status_exc,
        )
        return message.content[0].text.strip()

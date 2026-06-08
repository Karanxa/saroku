"""
saroku.adapters.anthropic — Anthropic Claude ModelAdapter for SafetyGuard.

Requires: pip install anthropic
"""

from __future__ import annotations

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
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicAdapter. "
                "Install it with: pip install anthropic"
            )

    async def achat(self, prompt: str) -> str:
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

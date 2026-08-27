"""
saroku.adapters.compat — OpenAI-compatible ModelAdapter for SafetyGuard.

Covers any provider that exposes an OpenAI-compatible chat completions API:
  - Google Gemini  (base_url = gemini openai-compat endpoint, GOOGLE_API_KEY)
  - Groq           (base_url = https://api.groq.com/openai/v1, GROQ_API_KEY)
  - Mistral        (base_url = https://api.mistral.ai/v1, MISTRAL_API_KEY)
  - Together AI    (base_url = https://api.together.xyz/v1, TOGETHER_API_KEY)
  - Perplexity     (base_url = https://api.perplexity.ai, PERPLEXITY_API_KEY)
  - Azure OpenAI   (uses AsyncAzureOpenAI, AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY)
  - Ollama         (base_url = http://localhost:11434/v1, no key needed)
  - Any other OpenAI-compatible server
"""

from __future__ import annotations

import os
from typing import Optional

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


class OpenAICompatModelAdapter(ModelAdapter):
    """
    ModelAdapter for any OpenAI-compatible API endpoint.

    Args:
        model:       Model name as expected by the provider.
        base_url:    Provider's API base URL.
        api_key:     API key. If None, reads from api_key_env.
        api_key_env: Environment variable name to read the key from.

    Example::

        # Groq
        guard = SafetyGuard(judge_model="groq:llama-3.3-70b-versatile")

        # Ollama (local)
        guard = SafetyGuard(judge_model="ollama:llama3.2")

        # Google Gemini
        guard = SafetyGuard(judge_model="google:gemini-2.0-flash")

        # Mistral
        guard = SafetyGuard(judge_model="mistral:mistral-small-latest")

        # Custom OpenAI-compatible server
        from saroku.adapters import OpenAICompatModelAdapter
        guard = SafetyGuard(model_adapter=OpenAICompatModelAdapter(
            model="my-model",
            base_url="https://my-server.example.com/v1",
            api_key="secret",
        ))
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
    ):
        self.model = model
        resolved_key = api_key or (os.environ.get(api_key_env) if api_key_env else None) or "none"
        self._client = AsyncOpenAI(base_url=base_url, api_key=resolved_key)

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


class AzureOpenAIAdapter(ModelAdapter):
    """
    ModelAdapter for Azure OpenAI.

    Reads AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY from the environment.

    Example::

        guard = SafetyGuard(judge_model="azure:my-gpt4o-deployment")
    """

    def __init__(self, deployment: str):
        self.deployment = deployment
        try:
            from openai import AsyncAzureOpenAI
            self._client = AsyncAzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
        except KeyError as e:
            raise EnvironmentError(
                f"Missing environment variable for Azure OpenAI: {e}. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
            )

    async def achat(self, prompt: str) -> str:
        client = self._client
        response = await _retry(
            lambda: client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        )
        return response.choices[0].message.content.strip()

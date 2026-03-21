import asyncio
import os
import random
import litellm
from typing import Optional

litellm.set_verbose = False

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "saroku")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
GOOGLE_CREDENTIALS_FILE = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(__file__), "../../credentials/vertex_ai_key.json"),
)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
_MAX_RETRIES = 6


def _configure_vertex():
    creds_path = os.path.abspath(GOOGLE_CREDENTIALS_FILE)
    if os.path.exists(creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    os.environ.setdefault("VERTEXAI_PROJECT", VERTEX_PROJECT)
    os.environ.setdefault("VERTEXAI_LOCATION", VERTEX_LOCATION)


async def _retry(coro_fn):
    """Async call with exponential backoff on rate-limit / transient errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return await coro_fn()
        except litellm.RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                raise
        except litellm.APIError as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            status = getattr(e, "status_code", None)
            if status not in _RETRYABLE_STATUS_CODES:
                raise
        wait = (2 ** attempt) + random.uniform(0, 1)
        await asyncio.sleep(wait)


class LiteLLMAdapter:
    def __init__(self, model: str):
        self.model = model
        self.is_vertex = model.startswith("vertex_ai/")
        if self.is_vertex:
            _configure_vertex()

    def _kwargs(self, **extra) -> dict:
        kw = dict(model=self.model, **extra)
        if self.is_vertex:
            kw["vertex_project"] = VERTEX_PROJECT
            kw["vertex_location"] = VERTEX_LOCATION
        return kw

    # ── sync (kept for tests / rule judge) ──────────────────────────────────

    def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        response = litellm.completion(**self._kwargs(messages=messages, temperature=temperature))
        return response.choices[0].message.content.strip()

    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        try:
            model = "vertex_ai/text-embedding-004" if self.is_vertex else "text-embedding-3-small"
            response = litellm.embedding(model=model, input=texts)
            return [item["embedding"] for item in response.data]
        except Exception:
            return None

    # ── async ────────────────────────────────────────────────────────────────

    async def achat(self, messages: list[dict], temperature: float = 0.3) -> str:
        response = await _retry(
            lambda: litellm.acompletion(**self._kwargs(messages=messages, temperature=temperature))
        )
        return response.choices[0].message.content.strip()

    async def aembed(self, texts: list[str]) -> Optional[list[list[float]]]:
        try:
            model = "vertex_ai/text-embedding-004" if self.is_vertex else "text-embedding-3-small"
            response = await _retry(lambda: litellm.aembedding(model=model, input=texts))
            return [item["embedding"] for item in response.data]
        except Exception:
            return None

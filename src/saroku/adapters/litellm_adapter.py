import asyncio
import os
import random
from typing import Optional

import openai
from openai import OpenAI, AsyncOpenAI

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


def _get_vertex_access_token() -> str:
    import google.auth
    import google.auth.transport.requests
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def _vertex_base_url() -> str:
    return (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/endpoints/openapi"
    )


async def _retry(coro_fn):
    """Async call with exponential backoff on rate-limit / transient errors."""
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


class LiteLLMAdapter:
    def __init__(self, model: str):
        self.model = model
        self.is_vertex = model.startswith("vertex_ai/")
        if self.is_vertex:
            _configure_vertex()
            self._api_model = model[len("vertex_ai/"):]
        else:
            self._api_model = model
            self._client = OpenAI()
            self._async_client = AsyncOpenAI()

    def _sync_client(self) -> OpenAI:
        if self.is_vertex:
            return OpenAI(base_url=_vertex_base_url(), api_key=_get_vertex_access_token())
        return self._client

    def _async_client_for_request(self) -> AsyncOpenAI:
        if self.is_vertex:
            return AsyncOpenAI(base_url=_vertex_base_url(), api_key=_get_vertex_access_token())
        return self._async_client

    # ── sync (kept for tests / rule judge) ──────────────────────────────────

    def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        response = self._sync_client().chat.completions.create(
            model=self._api_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        try:
            if self.is_vertex:
                model = "text-embedding-004"
                client = OpenAI(base_url=_vertex_base_url(), api_key=_get_vertex_access_token())
            else:
                model = "text-embedding-3-small"
                client = self._client
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            return None

    # ── async ────────────────────────────────────────────────────────────────

    async def achat(self, messages: list[dict], temperature: float = 0.3) -> str:
        client = self._async_client_for_request()
        response = await _retry(
            lambda: client.chat.completions.create(
                model=self._api_model,
                messages=messages,
                temperature=temperature,
            )
        )
        return response.choices[0].message.content.strip()

    async def aembed(self, texts: list[str]) -> Optional[list[list[float]]]:
        try:
            if self.is_vertex:
                model = "text-embedding-004"
                client = AsyncOpenAI(base_url=_vertex_base_url(), api_key=_get_vertex_access_token())
            else:
                model = "text-embedding-3-small"
                client = self._async_client
            response = await _retry(lambda: client.embeddings.create(model=model, input=texts))
            return [item.embedding for item in response.data]
        except Exception:
            return None

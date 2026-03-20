import os
import litellm
from typing import Optional

litellm.set_verbose = False

# Default Vertex AI config — can be overridden via env vars
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "saroku")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
GOOGLE_CREDENTIALS_FILE = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(__file__), "../../credentials/vertex_ai_key.json"),
)


def _configure_vertex():
    """Set credentials for Vertex AI if key file exists."""
    creds_path = os.path.abspath(GOOGLE_CREDENTIALS_FILE)
    if os.path.exists(creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    os.environ.setdefault("VERTEXAI_PROJECT", VERTEX_PROJECT)
    os.environ.setdefault("VERTEXAI_LOCATION", VERTEX_LOCATION)


class LiteLLMAdapter:
    def __init__(self, model: str):
        self.model = model
        self.is_vertex = model.startswith("vertex_ai/")
        if self.is_vertex:
            _configure_vertex()

    def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        kwargs = dict(model=self.model, messages=messages, temperature=temperature)
        if self.is_vertex:
            kwargs["vertex_project"] = VERTEX_PROJECT
            kwargs["vertex_location"] = VERTEX_LOCATION
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content.strip()

    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        try:
            if self.is_vertex:
                # Use Vertex AI text embeddings
                embed_model = "vertex_ai/text-embedding-004"
                response = litellm.embedding(
                    model=embed_model,
                    input=texts,
                    vertex_project=VERTEX_PROJECT,
                    vertex_location=VERTEX_LOCATION,
                )
            elif "gpt" in self.model or "openai" in self.model:
                embed_model = "text-embedding-3-small"
                response = litellm.embedding(model=embed_model, input=texts)
            else:
                # Fallback to OpenAI embeddings for any other provider
                embed_model = "text-embedding-3-small"
                response = litellm.embedding(model=embed_model, input=texts)
            return [item["embedding"] for item in response.data]
        except Exception:
            return None

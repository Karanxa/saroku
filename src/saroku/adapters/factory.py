"""
saroku.adapters.factory — Model string → ModelAdapter routing.

Supported prefixes
------------------
  openai:<model>       OpenAI API          (OPENAI_API_KEY)
  anthropic:<model>    Anthropic Claude    (ANTHROPIC_API_KEY)
  google:<model>       Google Gemini       (GOOGLE_API_KEY)
  groq:<model>         Groq                (GROQ_API_KEY)
  mistral:<model>      Mistral AI          (MISTRAL_API_KEY)
  together:<model>     Together AI         (TOGETHER_API_KEY)
  perplexity:<model>   Perplexity          (PERPLEXITY_API_KEY)
  azure:<deployment>   Azure OpenAI        (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY)
  ollama:<model>       Ollama (local)      (no key required)
  <model>              OpenAI (default, no prefix required)

Examples
--------
  SafetyGuard(judge_model="gpt-4o-mini")
  SafetyGuard(judge_model="anthropic:claude-3-5-haiku-20241022")
  SafetyGuard(judge_model="google:gemini-2.0-flash")
  SafetyGuard(judge_model="groq:llama-3.3-70b-versatile")
  SafetyGuard(judge_model="ollama:llama3.2")
  SafetyGuard(judge_model="azure:my-gpt4o-deployment")
"""

from __future__ import annotations

from saroku.adapters.base import ModelAdapter


# Groq, Mistral, Together, Perplexity — all OpenAI-compatible
_COMPAT_PROVIDERS: dict[str, tuple[str, str]] = {
    "groq":       ("https://api.groq.com/openai/v1",               "GROQ_API_KEY"),
    "mistral":    ("https://api.mistral.ai/v1",                     "MISTRAL_API_KEY"),
    "together":   ("https://api.together.xyz/v1",                   "TOGETHER_API_KEY"),
    "perplexity": ("https://api.perplexity.ai",                     "PERPLEXITY_API_KEY"),
    "google":     (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GOOGLE_API_KEY",
    ),
    "ollama":     ("http://localhost:11434/v1",                      None),
}


def resolve_adapter(model_str: str) -> ModelAdapter:
    """
    Parse a model string and return the appropriate ModelAdapter.

    Args:
        model_str: A model string with optional provider prefix
                   (e.g. "gpt-4o-mini", "anthropic:claude-3-5-haiku-20241022").

    Returns:
        A ModelAdapter ready to handle SafetyGuard Layer-3 prompts.

    Raises:
        ValueError: If the provider prefix is unrecognised.
    """
    from saroku.adapters.openai import OpenAIAdapter
    from saroku.adapters.anthropic import AnthropicAdapter
    from saroku.adapters.compat import OpenAICompatModelAdapter, AzureOpenAIAdapter

    if ":" not in model_str:
        # No prefix — default to OpenAI
        return OpenAIAdapter(model_str)

    provider, model = model_str.split(":", 1)
    provider = provider.lower()

    if provider == "openai":
        return OpenAIAdapter(model)

    if provider == "anthropic":
        return AnthropicAdapter(model)

    if provider == "azure":
        return AzureOpenAIAdapter(deployment=model)

    if provider in _COMPAT_PROVIDERS:
        base_url, api_key_env = _COMPAT_PROVIDERS[provider]
        return OpenAICompatModelAdapter(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
        )

    raise ValueError(
        f"Unknown provider prefix '{provider}' in model string '{model_str}'. "
        f"Supported prefixes: openai, anthropic, google, groq, mistral, together, "
        f"perplexity, azure, ollama. "
        f"Or pass a custom adapter via SafetyGuard(model_adapter=...)."
    )

"""
saroku.adapters — Model provider adapters for SafetyGuard and the benchmark runner.

For SafetyGuard plug-and-play:
    from saroku.adapters import ModelAdapter, resolve_adapter
    from saroku.adapters import OpenAIAdapter, AnthropicAdapter
    from saroku.adapters import OpenAICompatModelAdapter, AzureOpenAIAdapter

For the benchmark runner (internal):
    from saroku.adapters.openai_compat import OpenAICompatAdapter
"""

from saroku.adapters.base import ModelAdapter
from saroku.adapters.openai import OpenAIAdapter
from saroku.adapters.anthropic import AnthropicAdapter
from saroku.adapters.compat import OpenAICompatModelAdapter, AzureOpenAIAdapter
from saroku.adapters.factory import resolve_adapter

__all__ = [
    "ModelAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OpenAICompatModelAdapter",
    "AzureOpenAIAdapter",
    "resolve_adapter",
]

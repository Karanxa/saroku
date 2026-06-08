"""
saroku.adapters.base — ModelAdapter protocol for Layer-3 judge.

Any object implementing ModelAdapter can serve as a drop-in judge for
SafetyGuard. Build your own by subclassing ModelAdapter and implementing
achat().

Example::

    from saroku.adapters import ModelAdapter
    from saroku import SafetyGuard

    class MyAdapter(ModelAdapter):
        async def achat(self, prompt: str) -> str:
            # Call your own model / API
            return my_model.complete(prompt)

    guard = SafetyGuard(model_adapter=MyAdapter())
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """
    Minimal interface that SafetyGuard's Layer-3 judge requires.

    Implement achat() to integrate any LLM — cloud API, local model,
    or completely custom inference — into the saroku safety cascade.
    """

    @abstractmethod
    async def achat(self, prompt: str) -> str:
        """
        Send a prompt to the underlying model and return its text response.

        Args:
            prompt: The full safety-evaluation prompt built by SafetyGuard.

        Returns:
            The model's response as a plain string.
        """
        ...

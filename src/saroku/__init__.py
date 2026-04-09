"""
saroku — Behavioral regression testing and runtime safety for LLM agents.

Quickstart (benchmark):
    saroku run --model gpt-4o --benchmark bench-v1

Quickstart (SDK guard):
    from saroku import SafetyGuard

    guard = SafetyGuard()
    result = guard.check(
        action="I'll delete all records from the production orders table.",
        context="Database agent managing ACME Corp production environment.",
    )
    if not result.is_safe:
        for v in result.violations:
            print(f"[{v.severity}] {v.property}: {v.description}")

Quickstart (framework interceptor):
    from saroku import SafetyGuard, wrap, protect

    guard = SafetyGuard(judge_model="gpt-4o-mini")

    # Wrap one tool
    safe_tool = wrap(my_async_tool, guard=guard)

    # Protect a whole agent (auto-detects ADK / AutoGen / LangChain)
    safe_agent = await protect(agent, guard=guard)
"""

from saroku.guard import SafetyGuard, SafetyCheckResult, SafetyViolation
from saroku.adapters import (
    ModelAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    OpenAICompatModelAdapter,
    AzureOpenAIAdapter,
    resolve_adapter,
)
from saroku.integrations import wrap, protect, SafetyBlockedError

__all__ = [
    "SafetyGuard",
    "SafetyCheckResult",
    "SafetyViolation",
    # Adapters — for custom model integration
    "ModelAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OpenAICompatModelAdapter",
    "AzureOpenAIAdapter",
    "resolve_adapter",
    # Framework interceptor
    "wrap",
    "protect",
    "SafetyBlockedError",
]
__version__ = "0.5.0"

"""
saroku.integrations — Framework interceptor layer.

Drop saroku into any Google ADK, AutoGen, or LangChain agent in one line.

Usage::

    from saroku import SafetyGuard
    from saroku.integrations import wrap, protect

    guard = SafetyGuard(judge_model="gpt-4o-mini")

    # Wrap one tool
    safe_tool = wrap(my_async_tool, guard=guard)

    # Protect a whole agent (auto-detects ADK / AutoGen / LangChain)
    safe_agent = await protect(agent, guard=guard)

When a tool call is blocked, ``SafetyBlockedError`` is raised.
"""
from __future__ import annotations

from typing import Any

from saroku.integrations._base import SafetyBlockedError, FrameworkAdapter
from saroku.integrations._wrap import wrap
from saroku.integrations._detector import detect_framework


async def protect(agent: Any, guard: Any) -> Any:
    """
    Apply saroku safety interception to all tools in an agent.

    Auto-detects the framework (ADK, AutoGen, LangChain) from the
    agent object and applies the appropriate adapter.

    Args:
        agent: Any supported framework agent object.
        guard: A ``SafetyGuard`` instance.

    Returns:
        The agent with safety hooks applied (same object, mutated).

    Raises:
        ValueError: If the framework cannot be detected. Use ``wrap()``
                    for manual per-tool wiring instead.
    """
    from saroku.integrations._adk import ADKAdapter
    from saroku.integrations._autogen import AutoGenAdapter
    from saroku.integrations._langchain import LangChainAdapter

    framework = detect_framework(agent)

    adapters = {
        "adk": ADKAdapter(),
        "autogen": AutoGenAdapter(),
        "langchain": LangChainAdapter(),
    }
    return await adapters[framework].apply_to_agent(agent, guard)


__all__ = [
    "wrap",
    "protect",
    "SafetyBlockedError",
    "FrameworkAdapter",
]

"""
saroku.integrations._detector — Duck-type agent objects to identify their framework.
"""
from __future__ import annotations

from typing import Any


def detect_framework(agent: Any) -> str:
    """
    Identify which agent framework an object belongs to.

    Detection order matters — check more specific attributes first.

    Args:
        agent: Any agent object.

    Returns:
        One of: ``"adk"``, ``"autogen"``, ``"langchain"``

    Raises:
        ValueError: If the framework cannot be determined, with a hint
                    to use ``wrap()`` for manual wiring.
    """
    # Google ADK: agents have before_tool_callback
    if hasattr(agent, "before_tool_callback"):
        return "adk"

    # AutoGen: BaseAgent subclass
    try:
        import autogen_core
        if isinstance(agent, autogen_core.BaseAgent):
            return "autogen"
    except ImportError:
        pass

    # LangChain: agent has .tools list of BaseTool instances
    try:
        from langchain_core.tools import BaseTool
        if hasattr(agent, "tools") and isinstance(agent.tools, list):
            if agent.tools and isinstance(agent.tools[0], BaseTool):
                return "langchain"
    except ImportError:
        pass

    raise ValueError(
        f"saroku could not detect the agent framework for {type(agent).__name__!r}. "
        f"Supported frameworks: Google ADK, AutoGen, LangChain. "
        f"Use saroku.wrap(tool, guard=guard) to protect individual tools manually."
    )

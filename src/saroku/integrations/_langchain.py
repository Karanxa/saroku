"""
saroku.integrations._langchain — LangChain safety adapter.

Replaces each tool in ``agent.tools`` with a ``SarokuToolWrapper``
that intercepts ``_arun`` calls.
"""
from __future__ import annotations

from typing import Any

from saroku.integrations._base import FrameworkAdapter, SafetyBlockedError


class SarokuToolWrapper:
    """
    Wraps a LangChain ``BaseTool``, intercepting ``_arun`` with SafetyGuard.

    Preserves ``name`` and ``description`` so the agent's prompt
    and tool-selection logic are unaffected.

    Exposes the full public interface expected by LangChain executors
    (``run``, ``arun``, ``invoke``, ``ainvoke``) without subclassing
    ``BaseTool`` — avoids a hard langchain dependency.
    """

    def __init__(
        self,
        original_tool: Any,
        guard: Any,
        operator_constraints: list[str],
    ):
        self._original = original_tool
        self._guard = guard
        self._operator_constraints = operator_constraints
        self.name = original_tool.name
        self.description = original_tool.description

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        action = f"{self.name}(args={args}, kwargs={kwargs})"
        result = await self._guard.acheck(
            action=action,
            context="LangChain agent tool call",
            operator_constraints=self._operator_constraints,
            original_goal="",
        )
        if not result.is_safe:
            v = result.violations[0] if result.violations else None
            reason = (
                f"Action blocked by saroku: {v.property} — {v.description}"
                if v else "Action blocked by saroku."
            )
            raise SafetyBlockedError(
                violation=v,
                blocked_action=action,
                reason=reason,
            )
        return await self._original._arun(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Sync path — delegates to original. Use _arun for safety checks."""
        return self._original._run(*args, **kwargs)

    # ── Public interface expected by LangChain executors ─────────────────────

    async def arun(self, tool_input: Any = None, **kwargs: Any) -> Any:
        if tool_input is not None and not kwargs:
            return await self._arun(tool_input)
        return await self._arun(**kwargs)

    def run(self, tool_input: Any = None, **kwargs: Any) -> Any:
        return self._run(tool_input, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if isinstance(input, str):
            return await self._arun(input)
        return await self._arun(**input)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if isinstance(input, str):
            return self._run(input)
        return self._run(**input)


class LangChainAdapter(FrameworkAdapter):
    """
    Safety adapter for LangChain agents.

    Replaces every tool in ``agent.tools`` with a ``SarokuToolWrapper``.
    """

    async def apply_to_agent(self, agent: Any, guard: Any) -> Any:
        """
        Replace all tools in a LangChain agent with wrapped versions.

        Args:
            agent: A LangChain agent with a ``tools`` list and
                   optional ``system_prompt`` string.
            guard: A ``SafetyGuard`` instance.

        Returns:
            The agent (mutated in-place).
        """
        operator_constraints: list[str] = []
        system_prompt = getattr(agent, "system_prompt", "") or ""
        if system_prompt:
            operator_constraints = [system_prompt]

        agent.tools = [
            SarokuToolWrapper(
                original_tool=t,
                guard=guard,
                operator_constraints=operator_constraints,
            )
            for t in agent.tools
        ]
        return agent

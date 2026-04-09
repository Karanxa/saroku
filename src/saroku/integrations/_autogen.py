"""
saroku.integrations._autogen — AutoGen safety adapter.

Wraps tool functions registered in AutoGen agent's ``_tools`` dict.
"""
from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

from saroku.integrations._base import FrameworkAdapter, SafetyBlockedError


class AutoGenAdapter(FrameworkAdapter):
    """
    Safety adapter for AutoGen agents.

    Replaces each tool in ``agent._tools`` with a wrapped async function
    that calls ``SafetyGuard.acheck()`` before executing the original.
    """

    async def apply_to_agent(self, agent: Any, guard: Any) -> Any:
        """
        Wrap all tools registered on an AutoGen agent.

        Args:
            agent: An AutoGen agent with ``_tools`` dict and
                   ``system_message`` string.
            guard: A ``SafetyGuard`` instance.

        Returns:
            The agent (mutated in-place).
        """
        operator_constraints: list[str] = []
        system_message = getattr(agent, "system_message", "") or ""
        if system_message:
            operator_constraints = [system_message]

        tools: dict[str, Any] = getattr(agent, "_tools", {})
        for name, fn in list(tools.items()):
            tools[name] = self._wrap_tool(name, fn, guard, operator_constraints)

        return agent

    @staticmethod
    def _wrap_tool(
        tool_name: str,
        fn: Any,
        guard: Any,
        operator_constraints: list[str],
    ) -> Any:
        @wraps(fn)
        async def _wrapped(**kwargs: Any) -> Any:
            action = f"{tool_name}({kwargs})"
            result = await guard.acheck(
                action=action,
                context="AutoGen agent tool call",
                operator_constraints=operator_constraints,
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
            if inspect.iscoroutinefunction(fn):
                return await fn(**kwargs)
            else:
                return fn(**kwargs)

        return _wrapped

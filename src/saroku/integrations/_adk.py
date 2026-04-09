"""
saroku.integrations._adk — Google ADK safety adapter.

Hooks into ADK's ``before_tool_callback`` to intercept every tool call.
"""
from __future__ import annotations

from typing import Any, Optional

from saroku.integrations._base import FrameworkAdapter, SafetyBlockedError


class ADKAdapter(FrameworkAdapter):
    """
    Safety adapter for Google ADK agents.

    Registers a ``before_tool_callback`` on the agent. The callback
    calls ``SafetyGuard.acheck()`` before any tool executes. On a
    safety violation, raises ``SafetyBlockedError``.
    """

    async def apply_to_agent(self, agent: Any, guard: Any) -> Any:
        """
        Register the saroku safety callback on an ADK agent.

        Args:
            agent: A Google ADK agent object with ``before_tool_callback``
                   and ``instruction`` attributes.
            guard: A ``SafetyGuard`` instance.

        Returns:
            The agent (mutated in-place).
        """
        operator_constraints: list[str] = []
        if hasattr(agent, "instruction") and agent.instruction:
            operator_constraints = [agent.instruction]

        async def _saroku_callback(
            tool: Any,
            args: dict[str, Any],
            tool_context: Any,
        ) -> Optional[dict[str, Any]]:
            action = f"{tool.name}({args})"
            state = getattr(tool_context, "state", {}) or {}
            original_goal = state.get("goal", "")

            result = await guard.acheck(
                action=action,
                context="ADK agent tool call",
                operator_constraints=operator_constraints,
                original_goal=original_goal,
            )

            if not result.is_safe and result.violations:
                v = result.violations[0]
                reason = f"Action blocked by saroku: {v.property} — {v.description}"
                raise SafetyBlockedError(
                    violation=v,
                    blocked_action=action,
                    reason=reason,
                )

            return None  # None tells ADK to proceed with normal tool execution

        agent.before_tool_callback = _saroku_callback
        return agent

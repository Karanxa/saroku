"""
saroku.integrations._wrap — Framework-agnostic tool wrapper.
"""
from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Optional

from saroku.integrations._base import SafetyBlockedError


def wrap(
    tool: Callable,
    guard: Any,
    context: str = "",
    operator_constraints: Optional[list[str]] = None,
    original_goal: str = "",
) -> Callable:
    """
    Wrap an async tool function with SafetyGuard interception.

    The wrapped function calls ``guard.acheck()`` before executing the
    original tool. If the action is blocked, raises ``SafetyBlockedError``.

    Args:
        tool:                 An async callable (the tool to protect).
        guard:                A ``SafetyGuard`` instance.
        context:              Static context string forwarded to the guard.
        operator_constraints: Static constraints forwarded to the guard.
        original_goal:        Static goal string forwarded to the guard.

    Returns:
        An async callable with the same signature as ``tool``.

    Raises:
        TypeError: If ``tool`` is not a coroutine function.
    """
    if not inspect.iscoroutinefunction(tool):
        raise TypeError(
            f"saroku.wrap() requires an async tool function, "
            f"but '{tool.__name__}' is a regular (sync) function. "
            f"Convert it to async or use an async wrapper."
        )

    @wraps(tool)
    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # Build a human-readable action string: "tool_name(kwarg=val, ...)"
        parts = [repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()]
        action = f"{tool.__name__}({', '.join(parts)})"

        result = await guard.acheck(
            action=action,
            context=context,
            operator_constraints=operator_constraints or [],
            original_goal=original_goal,
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

        return await tool(*args, **kwargs)

    return _wrapped

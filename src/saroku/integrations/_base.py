"""
saroku.integrations._base — Core types for the framework interceptor layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from saroku.guard import SafetyViolation


class SafetyBlockedError(Exception):
    """
    Raised when SafetyGuard blocks an agent tool call.

    The framework's error handler catches this. The agent sees
    ``reason`` as the tool's return value in its message history.
    """

    def __init__(self, violation: Optional[SafetyViolation], blocked_action: str, reason: str):
        super().__init__(reason)
        self.violation = violation
        self.blocked_action = blocked_action
        self.reason = reason


class FrameworkAdapter(ABC):
    """
    Abstract base for per-framework integration adapters.

    Each subclass knows how to hook into one agent framework's
    tool-execution lifecycle.
    """

    @abstractmethod
    async def apply_to_agent(self, agent: Any, guard: Any) -> Any:
        """
        Apply safety interception to all tools in the agent.

        Args:
            agent: The framework-native agent object.
            guard: A SafetyGuard instance.

        Returns:
            The agent with safety hooks applied (may be the same object, mutated).
        """
        ...

# Framework Interceptor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `saroku.integrations` — a `wrap()` / `protect()` API that intercepts tool calls in Google ADK, AutoGen, and LangChain agents, evaluates them with `SafetyGuard`, and raises `SafetyBlockedError` on unsafe actions.

**Architecture:** A thin `integrations/` module sits between the framework and tool execution. `wrap(tool, guard)` wraps any single tool; `protect(agent, guard)` auto-detects the framework and applies the right adapter. `SafetyGuard` is untouched — the integration layer calls its existing `acheck()`.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio, unittest.mock. Optional framework deps: `google-adk`, `pyautogen`, `langchain-core`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/saroku/integrations/__init__.py` | Create | Public API: `wrap()`, `protect()`, `SafetyBlockedError` |
| `src/saroku/integrations/_base.py` | Create | `SafetyBlockedError`, `FrameworkAdapter` ABC |
| `src/saroku/integrations/_detector.py` | Create | Duck-type agent → framework name |
| `src/saroku/integrations/_wrap.py` | Create | Framework-agnostic `wrap()` for callables |
| `src/saroku/integrations/_adk.py` | Create | Google ADK `before_tool_callback` adapter |
| `src/saroku/integrations/_autogen.py` | Create | AutoGen tool-function wrapper adapter |
| `src/saroku/integrations/_langchain.py` | Create | LangChain `BaseTool` wrapper adapter |
| `src/saroku/__init__.py` | Modify | Export `wrap`, `protect`, `SafetyBlockedError` |
| `pyproject.toml` | Modify | Add optional `[integrations]` dep group |
| `tests/integrations/test_base.py` | Create | `SafetyBlockedError`, `FrameworkAdapter` ABC |
| `tests/integrations/test_wrap.py` | Create | `wrap()` safe + unsafe paths |
| `tests/integrations/test_detector.py` | Create | Framework detection logic |
| `tests/integrations/test_adk.py` | Create | ADK adapter |
| `tests/integrations/test_autogen.py` | Create | AutoGen adapter |
| `tests/integrations/test_langchain.py` | Create | LangChain adapter |
| `tests/integrations/test_protect.py` | Create | `protect()` end-to-end per framework |

---

## Task 1: `SafetyBlockedError` and `FrameworkAdapter` ABC

**Files:**
- Create: `src/saroku/integrations/_base.py`
- Create: `tests/integrations/__init__.py`
- Create: `tests/integrations/test_base.py`

- [ ] **Step 1: Create test directory**

```bash
mkdir -p tests/integrations
touch tests/__init__.py tests/integrations/__init__.py
```

- [ ] **Step 2: Write the failing tests**

`tests/integrations/test_base.py`:
```python
import pytest
from saroku.integrations._base import SafetyBlockedError, FrameworkAdapter
from saroku.guard import SafetyViolation


def make_violation():
    return SafetyViolation(
        property="minimal_footprint",
        severity="high",
        description="Agent is deleting all records.",
        recommendation="Use a scoped DELETE with a WHERE clause.",
    )


def test_safety_blocked_error_message():
    v = make_violation()
    err = SafetyBlockedError(
        violation=v,
        blocked_action="DELETE FROM users",
        reason="minimal_footprint — Agent is deleting all records.",
    )
    assert "minimal_footprint" in str(err)
    assert err.violation is v
    assert err.blocked_action == "DELETE FROM users"
    assert "Agent is deleting all records" in err.reason


def test_safety_blocked_error_is_exception():
    v = make_violation()
    err = SafetyBlockedError(violation=v, blocked_action="x", reason="blocked")
    assert isinstance(err, Exception)


def test_framework_adapter_is_abstract():
    with pytest.raises(TypeError):
        FrameworkAdapter()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/karan/saroku
pytest tests/integrations/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'saroku.integrations'`

- [ ] **Step 4: Create the module**

`src/saroku/integrations/_base.py`:
```python
"""
saroku.integrations._base — Core types for the framework interceptor layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from saroku.guard import SafetyViolation


class SafetyBlockedError(Exception):
    """
    Raised when SafetyGuard blocks an agent tool call.

    The framework's error handler catches this. The agent sees
    ``reason`` as the tool's return value in its message history.
    """

    def __init__(self, violation: SafetyViolation, blocked_action: str, reason: str):
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
```

`src/saroku/integrations/__init__.py` (minimal, will be filled in Task 7):
```python
from saroku.integrations._base import SafetyBlockedError, FrameworkAdapter

__all__ = ["SafetyBlockedError", "FrameworkAdapter"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/integrations/test_base.py -v
```

Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/saroku/integrations/ tests/integrations/
git commit -m "feat(integrations): add SafetyBlockedError and FrameworkAdapter ABC"
```

---

## Task 2: Framework-agnostic `wrap()`

**Files:**
- Create: `src/saroku/integrations/_wrap.py`
- Create: `tests/integrations/test_wrap.py`

- [ ] **Step 1: Write the failing tests**

`tests/integrations/test_wrap.py`:
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from saroku.integrations._wrap import wrap
from saroku.integrations._base import SafetyBlockedError
from saroku.guard import SafetyGuard, SafetyCheckResult, SafetyViolation


def make_safe_result():
    return SafetyCheckResult(
        is_safe=True, violations=[], checked_properties=["minimal_footprint"],
        latency_ms=10.0, layers_used=["llm"], action="test", context="",
    )


def make_unsafe_result():
    v = SafetyViolation(
        property="minimal_footprint", severity="high",
        description="Irreversible delete detected.",
        recommendation="Use scoped DELETE.",
    )
    return SafetyCheckResult(
        is_safe=False, violations=[v], checked_properties=["minimal_footprint"],
        latency_ms=10.0, layers_used=["llm"], action="test", context="",
    )


@pytest.fixture
def guard():
    g = MagicMock(spec=SafetyGuard)
    g.acheck = AsyncMock(return_value=make_safe_result())
    return g


@pytest.mark.asyncio
async def test_wrap_calls_original_when_safe(guard):
    async def my_tool(query: str) -> str:
        return f"result:{query}"

    safe_tool = wrap(my_tool, guard=guard)
    result = await safe_tool(query="SELECT * FROM logs")
    assert result == "result:SELECT * FROM logs"
    guard.acheck.assert_called_once()


@pytest.mark.asyncio
async def test_wrap_raises_on_unsafe(guard):
    guard.acheck = AsyncMock(return_value=make_unsafe_result())

    async def my_tool(query: str) -> str:
        return "never reached"

    safe_tool = wrap(my_tool, guard=guard)
    with pytest.raises(SafetyBlockedError) as exc_info:
        await safe_tool(query="DELETE FROM users")
    assert "minimal_footprint" in exc_info.value.reason
    assert exc_info.value.blocked_action != ""


@pytest.mark.asyncio
async def test_wrap_passes_action_string_to_guard(guard):
    async def delete_records(table: str, confirm: bool) -> str:
        return "deleted"

    safe_tool = wrap(delete_records, guard=guard)
    await safe_tool(table="users", confirm=True)

    call_kwargs = guard.acheck.call_args.kwargs
    assert "delete_records" in call_kwargs["action"]
    assert "users" in call_kwargs["action"]


@pytest.mark.asyncio
async def test_wrap_forwards_context_to_guard(guard):
    async def my_tool(x: str) -> str:
        return x

    safe_tool = wrap(my_tool, guard=guard, context="prod DB agent")
    await safe_tool(x="hello")

    call_kwargs = guard.acheck.call_args.kwargs
    assert call_kwargs["context"] == "prod DB agent"


def test_wrap_sync_tool_raises_helpful_error():
    """wrap() requires an async tool function."""
    def sync_tool(x: str) -> str:
        return x

    guard = MagicMock(spec=SafetyGuard)
    with pytest.raises(TypeError, match="async"):
        wrap(sync_tool, guard=guard)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/integrations/test_wrap.py -v
```

Expected: `ImportError: cannot import name 'wrap' from 'saroku.integrations._wrap'`

- [ ] **Step 3: Implement `wrap()`**

`src/saroku/integrations/_wrap.py`:
```python
"""
saroku.integrations._wrap — Framework-agnostic tool wrapper.
"""
from __future__ import annotations

import asyncio
import inspect
import json
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
        # Build a human-readable action string: "tool_name(arg1, kwarg=val)"
        parts = [repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()]
        action = f"{tool.__name__}({', '.join(parts)})"

        result = await guard.acheck(
            action=action,
            context=context,
            operator_constraints=operator_constraints or [],
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

        return await tool(*args, **kwargs)

    return _wrapped
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/integrations/test_wrap.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/saroku/integrations/_wrap.py tests/integrations/test_wrap.py
git commit -m "feat(integrations): add framework-agnostic wrap() tool interceptor"
```

---

## Task 3: Framework auto-detector

**Files:**
- Create: `src/saroku/integrations/_detector.py`
- Create: `tests/integrations/test_detector.py`

- [ ] **Step 1: Write the failing tests**

`tests/integrations/test_detector.py`:
```python
import pytest
from unittest.mock import MagicMock
from saroku.integrations._detector import detect_framework


def make_adk_agent():
    agent = MagicMock()
    agent.before_tool_callback = None  # ADK agents have this attribute
    # Remove LangChain-like .tools to avoid false positive
    del agent.tools
    return agent


def make_autogen_agent():
    """AutoGen BaseAgent subclass mock."""
    try:
        import autogen_core
        agent = MagicMock(spec=autogen_core.BaseAgent)
        return agent
    except ImportError:
        pytest.skip("autogen_core not installed")


def make_langchain_agent():
    try:
        from langchain_core.tools import BaseTool
        agent = MagicMock()
        tool = MagicMock(spec=BaseTool)
        agent.tools = [tool]
        # Ensure no ADK attribute
        del agent.before_tool_callback
        return agent
    except ImportError:
        pytest.skip("langchain_core not installed")


def test_detect_adk():
    agent = make_adk_agent()
    assert detect_framework(agent) == "adk"


def test_detect_langchain():
    agent = make_langchain_agent()
    assert detect_framework(agent) == "langchain"


def test_detect_unknown_raises():
    agent = MagicMock(spec=[])  # no relevant attrs
    with pytest.raises(ValueError, match="wrap\\(\\)"):
        detect_framework(agent)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/integrations/test_detector.py -v
```

Expected: `ImportError: cannot import name 'detect_framework'`

- [ ] **Step 3: Implement `detect_framework()`**

`src/saroku/integrations/_detector.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/integrations/test_detector.py -v
```

Expected: All available tests PASS (AutoGen test skips if not installed)

- [ ] **Step 5: Commit**

```bash
git add src/saroku/integrations/_detector.py tests/integrations/test_detector.py
git commit -m "feat(integrations): add framework auto-detector"
```

---

## Task 4: Google ADK adapter

**Files:**
- Create: `src/saroku/integrations/_adk.py`
- Create: `tests/integrations/test_adk.py`

- [ ] **Step 1: Write the failing tests**

`tests/integrations/test_adk.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from saroku.integrations._adk import ADKAdapter
from saroku.integrations._base import SafetyBlockedError
from saroku.guard import SafetyGuard, SafetyCheckResult, SafetyViolation


def make_safe_result():
    return SafetyCheckResult(
        is_safe=True, violations=[], checked_properties=[],
        latency_ms=5.0, layers_used=["llm"], action="", context="",
    )


def make_unsafe_result(prop="minimal_footprint", desc="Irreversible action."):
    v = SafetyViolation(property=prop, severity="high",
                        description=desc, recommendation="Don't do it.")
    return SafetyCheckResult(
        is_safe=False, violations=[v], checked_properties=[prop],
        latency_ms=5.0, layers_used=["llm"], action="", context="",
    )


@pytest.fixture
def guard():
    g = MagicMock(spec=SafetyGuard)
    g.acheck = AsyncMock(return_value=make_safe_result())
    return g


@pytest.fixture
def adk_agent():
    agent = MagicMock()
    agent.before_tool_callback = None
    agent.instruction = "You are a database agent. Never DROP tables."
    return agent


@pytest.mark.asyncio
async def test_adk_adapter_registers_callback(guard, adk_agent):
    adapter = ADKAdapter()
    await adapter.apply_to_agent(adk_agent, guard)
    assert adk_agent.before_tool_callback is not None
    assert callable(adk_agent.before_tool_callback)


@pytest.mark.asyncio
async def test_adk_callback_allows_safe_action(guard, adk_agent):
    adapter = ADKAdapter()
    await adapter.apply_to_agent(adk_agent, guard)

    tool = MagicMock()
    tool.name = "query_db"
    args = {"query": "SELECT * FROM logs"}
    tool_context = MagicMock()
    tool_context.state = {}

    # Should not raise
    result = await adk_agent.before_tool_callback(tool, args, tool_context)
    assert result is None  # None = allow execution to proceed


@pytest.mark.asyncio
async def test_adk_callback_blocks_unsafe_action(guard, adk_agent):
    guard.acheck = AsyncMock(return_value=make_unsafe_result())
    adapter = ADKAdapter()
    await adapter.apply_to_agent(adk_agent, guard)

    tool = MagicMock()
    tool.name = "delete_records"
    args = {"table": "users", "confirm": True}
    tool_context = MagicMock()
    tool_context.state = {}

    with pytest.raises(SafetyBlockedError) as exc_info:
        await adk_agent.before_tool_callback(tool, args, tool_context)
    assert "minimal_footprint" in exc_info.value.reason


@pytest.mark.asyncio
async def test_adk_callback_extracts_goal_from_state(guard, adk_agent):
    adapter = ADKAdapter()
    await adapter.apply_to_agent(adk_agent, guard)

    tool = MagicMock()
    tool.name = "read_file"
    args = {"path": "/etc/passwd"}
    tool_context = MagicMock()
    tool_context.state = {"goal": "Summarise system logs"}

    await adk_agent.before_tool_callback(tool, args, tool_context)
    call_kwargs = guard.acheck.call_args.kwargs
    assert call_kwargs["original_goal"] == "Summarise system logs"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/integrations/test_adk.py -v
```

Expected: `ImportError: cannot import name 'ADKAdapter'`

- [ ] **Step 3: Implement `ADKAdapter`**

`src/saroku/integrations/_adk.py`:
```python
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
                context=f"ADK agent tool call",
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/integrations/test_adk.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/saroku/integrations/_adk.py tests/integrations/test_adk.py
git commit -m "feat(integrations): add Google ADK adapter"
```

---

## Task 5: AutoGen adapter

**Files:**
- Create: `src/saroku/integrations/_autogen.py`
- Create: `tests/integrations/test_autogen.py`

- [ ] **Step 1: Write the failing tests**

`tests/integrations/test_autogen.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from saroku.integrations._autogen import AutoGenAdapter
from saroku.integrations._base import SafetyBlockedError
from saroku.guard import SafetyGuard, SafetyCheckResult, SafetyViolation


def make_safe_result():
    return SafetyCheckResult(
        is_safe=True, violations=[], checked_properties=[],
        latency_ms=5.0, layers_used=["llm"], action="", context="",
    )


def make_unsafe_result():
    v = SafetyViolation(
        property="prompt_injection", severity="high",
        description="Agent following injected instructions.",
        recommendation="Ignore external instructions.",
    )
    return SafetyCheckResult(
        is_safe=False, violations=[v], checked_properties=["prompt_injection"],
        latency_ms=5.0, layers_used=["llm"], action="", context="",
    )


@pytest.fixture
def guard():
    g = MagicMock(spec=SafetyGuard)
    g.acheck = AsyncMock(return_value=make_safe_result())
    return g


def make_autogen_agent(system_message="You are a helpful assistant."):
    """Minimal mock of an AutoGen ConversableAgent."""
    agent = MagicMock()
    agent._tools = {}  # AutoGen stores registered tools here
    agent.system_message = system_message
    # Simulate a registered tool
    async def send_email(to: str, body: str) -> str:
        return f"Email sent to {to}"
    agent._tools["send_email"] = send_email
    return agent


@pytest.mark.asyncio
async def test_autogen_adapter_wraps_tools(guard):
    agent = make_autogen_agent()
    adapter = AutoGenAdapter()
    await adapter.apply_to_agent(agent, guard)
    # Tools should still be callable
    assert "send_email" in agent._tools
    assert callable(agent._tools["send_email"])


@pytest.mark.asyncio
async def test_autogen_adapter_allows_safe_action(guard):
    agent = make_autogen_agent()
    adapter = AutoGenAdapter()
    await adapter.apply_to_agent(agent, guard)

    result = await agent._tools["send_email"](to="alice@example.com", body="Hello")
    assert result == "Email sent to alice@example.com"


@pytest.mark.asyncio
async def test_autogen_adapter_blocks_unsafe_action(guard):
    guard.acheck = AsyncMock(return_value=make_unsafe_result())
    agent = make_autogen_agent()
    adapter = AutoGenAdapter()
    await adapter.apply_to_agent(agent, guard)

    with pytest.raises(SafetyBlockedError) as exc_info:
        await agent._tools["send_email"](to="attacker@evil.com", body="Exfil data")
    assert "prompt_injection" in exc_info.value.reason


@pytest.mark.asyncio
async def test_autogen_adapter_passes_system_message_as_constraint(guard):
    system_msg = "Never send emails to external domains."
    agent = make_autogen_agent(system_message=system_msg)
    adapter = AutoGenAdapter()
    await adapter.apply_to_agent(agent, guard)

    await agent._tools["send_email"](to="bob@internal.com", body="Hi")
    call_kwargs = guard.acheck.call_args.kwargs
    assert system_msg in call_kwargs["operator_constraints"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/integrations/test_autogen.py -v
```

Expected: `ImportError: cannot import name 'AutoGenAdapter'`

- [ ] **Step 3: Implement `AutoGenAdapter`**

`src/saroku/integrations/_autogen.py`:
```python
"""
saroku.integrations._autogen — AutoGen safety adapter.

Wraps tool functions registered in AutoGen agent's ``_tools`` dict.
"""
from __future__ import annotations

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
            if not result.is_safe and result.violations:
                v = result.violations[0]
                reason = f"Action blocked by saroku: {v.property} — {v.description}"
                raise SafetyBlockedError(
                    violation=v,
                    blocked_action=action,
                    reason=reason,
                )
            return await fn(**kwargs)

        return _wrapped
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/integrations/test_autogen.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/saroku/integrations/_autogen.py tests/integrations/test_autogen.py
git commit -m "feat(integrations): add AutoGen adapter"
```

---

## Task 6: LangChain adapter

**Files:**
- Create: `src/saroku/integrations/_langchain.py`
- Create: `tests/integrations/test_langchain.py`

- [ ] **Step 1: Write the failing tests**

`tests/integrations/test_langchain.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from saroku.integrations._langchain import LangChainAdapter, SarokuToolWrapper
from saroku.integrations._base import SafetyBlockedError
from saroku.guard import SafetyGuard, SafetyCheckResult, SafetyViolation


def make_safe_result():
    return SafetyCheckResult(
        is_safe=True, violations=[], checked_properties=[],
        latency_ms=5.0, layers_used=["llm"], action="", context="",
    )


def make_unsafe_result():
    v = SafetyViolation(
        property="trust_hierarchy", severity="high",
        description="Violates operator constraint.",
        recommendation="Check constraints before acting.",
    )
    return SafetyCheckResult(
        is_safe=False, violations=[v], checked_properties=["trust_hierarchy"],
        latency_ms=5.0, layers_used=["llm"], action="", context="",
    )


@pytest.fixture
def guard():
    g = MagicMock(spec=SafetyGuard)
    g.acheck = AsyncMock(return_value=make_safe_result())
    return g


def make_lc_tool(name="search", description="Search the web"):
    """Minimal BaseTool mock."""
    try:
        from langchain_core.tools import BaseTool
        tool = MagicMock(spec=BaseTool)
        tool.name = name
        tool.description = description
        tool._arun = AsyncMock(return_value=f"{name}_result")
        tool._run = MagicMock(return_value=f"{name}_result")
        return tool
    except ImportError:
        pytest.skip("langchain_core not installed")


def make_lc_agent(system_prompt="You are a research assistant."):
    agent = MagicMock()
    agent.tools = [make_lc_tool("search"), make_lc_tool("read_file")]
    agent.system_prompt = system_prompt
    return agent


@pytest.mark.asyncio
async def test_langchain_adapter_replaces_tools(guard):
    agent = make_lc_agent()
    adapter = LangChainAdapter()
    await adapter.apply_to_agent(agent, guard)
    assert all(isinstance(t, SarokuToolWrapper) for t in agent.tools)


@pytest.mark.asyncio
async def test_saroku_tool_wrapper_allows_safe_action(guard):
    original = make_lc_tool("search")
    wrapper = SarokuToolWrapper(
        original_tool=original,
        guard=guard,
        operator_constraints=[],
    )
    result = await wrapper._arun(query="latest AI papers")
    assert result == "search_result"


@pytest.mark.asyncio
async def test_saroku_tool_wrapper_blocks_unsafe_action(guard):
    guard.acheck = AsyncMock(return_value=make_unsafe_result())
    original = make_lc_tool("send_email")
    wrapper = SarokuToolWrapper(
        original_tool=original,
        guard=guard,
        operator_constraints=["Never send external emails"],
    )
    with pytest.raises(SafetyBlockedError) as exc_info:
        await wrapper._arun(to="external@evil.com", body="data")
    assert "trust_hierarchy" in exc_info.value.reason


@pytest.mark.asyncio
async def test_langchain_adapter_passes_system_prompt_as_constraint(guard):
    system_prompt = "Only access internal APIs."
    agent = make_lc_agent(system_prompt=system_prompt)
    adapter = LangChainAdapter()
    await adapter.apply_to_agent(agent, guard)

    await agent.tools[0]._arun(query="test")
    call_kwargs = guard.acheck.call_args.kwargs
    assert system_prompt in call_kwargs["operator_constraints"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/integrations/test_langchain.py -v
```

Expected: `ImportError: cannot import name 'LangChainAdapter'`

- [ ] **Step 3: Implement `LangChainAdapter` and `SarokuToolWrapper`**

`src/saroku/integrations/_langchain.py`:
```python
"""
saroku.integrations._langchain — LangChain safety adapter.

Replaces each tool in ``agent.tools`` with a ``SarokuToolWrapper``
that intercepts ``_arun`` calls.
"""
from __future__ import annotations

from typing import Any, Optional

from saroku.integrations._base import FrameworkAdapter, SafetyBlockedError


class SarokuToolWrapper:
    """
    Wraps a LangChain ``BaseTool``, intercepting ``_arun`` with SafetyGuard.

    Preserves ``name`` and ``description`` so the agent's prompt
    and tool-selection logic are unaffected.
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
        # Expose LangChain-expected attributes
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
        if not result.is_safe and result.violations:
            v = result.violations[0]
            reason = f"Action blocked by saroku: {v.property} — {v.description}"
            raise SafetyBlockedError(
                violation=v,
                blocked_action=action,
                reason=reason,
            )
        return await self._original._arun(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Sync path — delegates to original. Use _arun for safety checks."""
        return self._original._run(*args, **kwargs)


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/integrations/test_langchain.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/saroku/integrations/_langchain.py tests/integrations/test_langchain.py
git commit -m "feat(integrations): add LangChain adapter and SarokuToolWrapper"
```

---

## Task 7: Wire up `protect()` and public `__init__`

**Files:**
- Modify: `src/saroku/integrations/__init__.py`
- Create: `tests/integrations/test_protect.py`

- [ ] **Step 1: Write the failing tests**

`tests/integrations/test_protect.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from saroku.integrations import wrap, protect, SafetyBlockedError
from saroku.guard import SafetyGuard, SafetyCheckResult, SafetyViolation


def make_safe_result():
    return SafetyCheckResult(
        is_safe=True, violations=[], checked_properties=[],
        latency_ms=5.0, layers_used=["llm"], action="", context="",
    )


@pytest.fixture
def guard():
    g = MagicMock(spec=SafetyGuard)
    g.acheck = AsyncMock(return_value=make_safe_result())
    return g


def make_adk_agent():
    agent = MagicMock()
    agent.before_tool_callback = None
    agent.instruction = "Be safe."
    return agent


def make_lc_agent():
    try:
        from langchain_core.tools import BaseTool
        agent = MagicMock()
        tool = MagicMock(spec=BaseTool)
        tool.name = "search"
        tool.description = "Search"
        tool._arun = AsyncMock(return_value="result")
        agent.tools = [tool]
        agent.system_prompt = "Be safe."
        del agent.before_tool_callback
        return agent
    except ImportError:
        pytest.skip("langchain_core not installed")


@pytest.mark.asyncio
async def test_protect_adk_agent(guard):
    agent = make_adk_agent()
    protected = await protect(agent, guard=guard)
    assert protected is agent
    assert callable(agent.before_tool_callback)


@pytest.mark.asyncio
async def test_protect_langchain_agent(guard):
    agent = make_lc_agent()
    from saroku.integrations._langchain import SarokuToolWrapper
    protected = await protect(agent, guard=guard)
    assert protected is agent
    assert all(isinstance(t, SarokuToolWrapper) for t in agent.tools)


@pytest.mark.asyncio
async def test_protect_unknown_agent_raises(guard):
    agent = MagicMock(spec=[])
    with pytest.raises(ValueError, match="wrap\\(\\)"):
        await protect(agent, guard=guard)


@pytest.mark.asyncio
async def test_wrap_exported_from_integrations(guard):
    async def my_tool(x: str) -> str:
        return x

    safe_tool = wrap(my_tool, guard=guard)
    result = await safe_tool(x="hello")
    assert result == "hello"


def test_safety_blocked_error_exported():
    assert issubclass(SafetyBlockedError, Exception)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/integrations/test_protect.py -v
```

Expected: `ImportError: cannot import name 'protect'`

- [ ] **Step 3: Implement `protect()` and update `__init__`**

`src/saroku/integrations/__init__.py`:
```python
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
The agent sees the blocked reason as the tool's return value.
"""
from __future__ import annotations

from typing import Any, Optional

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/integrations/test_protect.py -v
```

Expected: All available tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/saroku/integrations/__init__.py tests/integrations/test_protect.py
git commit -m "feat(integrations): wire up protect() and public __init__"
```

---

## Task 8: Export from top-level `saroku` and update `pyproject.toml`

**Files:**
- Modify: `src/saroku/__init__.py` (lines 20-41)
- Modify: `pyproject.toml`

- [ ] **Step 1: Update `src/saroku/__init__.py`**

Add these lines after the existing imports:

```python
from saroku.integrations import wrap, protect, SafetyBlockedError as SafetyBlockedError
```

And add to `__all__`:
```python
    "wrap",
    "protect",
    "SafetyBlockedError",
```

Full updated file:
```python
"""
saroku — Behavioral regression testing and runtime safety for LLM agents.

Quickstart (benchmark):
    saroku run --model gpt-4o --benchmark bench-v1

Quickstart (SDK guard):
    from saroku import SafetyGuard

    guard = SafetyGuard(judge_model="gpt-4o-mini")
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
```

- [ ] **Step 2: Add optional framework deps to `pyproject.toml`**

Add to `[project.optional-dependencies]`:
```toml
adk = ["google-adk>=0.1.0"]
autogen = ["pyautogen>=0.2.0"]
langchain = ["langchain-core>=0.1.0"]
integrations = ["google-adk>=0.1.0", "pyautogen>=0.2.0", "langchain-core>=0.1.0"]
```

- [ ] **Step 3: Verify top-level imports work**

```bash
cd /home/karan/saroku
python -c "from saroku import SafetyGuard, wrap, protect, SafetyBlockedError; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/saroku/__init__.py pyproject.toml
git commit -m "feat: export wrap, protect, SafetyBlockedError from saroku top-level; bump version to 0.5.0"
```

---

## Self-Review

**Spec coverage:**
- ✅ `wrap(tool, guard)` — Task 2
- ✅ `protect(agent, guard)` — Task 7
- ✅ `SafetyBlockedError` with `violation`, `blocked_action`, `reason` — Task 1
- ✅ Hard-stop exception on unsafe — all adapter tasks
- ✅ ADK adapter via `before_tool_callback` — Task 4
- ✅ AutoGen adapter via `_tools` dict — Task 5
- ✅ LangChain adapter via `SarokuToolWrapper` — Task 6
- ✅ Auto-detection in `protect()` — Task 3 + 7
- ✅ Context extraction (goal, constraints, history) — each adapter task
- ✅ `SafetyGuard` untouched — confirmed, no modifications to `guard.py`
- ✅ Optional deps in `pyproject.toml` — Task 8

**Placeholder scan:** No TBDs or incomplete steps found.

**Type consistency:**
- `SafetyBlockedError(violation, blocked_action, reason)` — consistent across Tasks 1, 2, 4, 5, 6, 7
- `FrameworkAdapter.apply_to_agent(agent, guard)` — consistent across Tasks 1, 4, 5, 6, 7
- `detect_framework(agent) -> str` — consistent across Tasks 3, 7
- `wrap(tool, guard, context, operator_constraints, original_goal)` — consistent across Tasks 2, 7

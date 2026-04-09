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

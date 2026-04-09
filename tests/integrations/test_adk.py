import pytest
from unittest.mock import AsyncMock, MagicMock
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

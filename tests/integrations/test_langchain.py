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
    """Minimal BaseTool mock — no langchain_core required."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool._arun = AsyncMock(return_value=f"{name}_result")
    tool._run = MagicMock(return_value=f"{name}_result")
    return tool


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

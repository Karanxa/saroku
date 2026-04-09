import pytest
from unittest.mock import AsyncMock, MagicMock
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
    except ImportError:
        pytest.skip("langchain_core not installed")

    agent = MagicMock()
    tool = MagicMock(spec=BaseTool)
    tool.name = "search"
    tool.description = "Search"
    tool._arun = AsyncMock(return_value="result")
    agent.tools = [tool]
    agent.system_prompt = "Be safe."
    # Ensure no ADK attribute
    del agent.before_tool_callback
    return agent


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

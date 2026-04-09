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
    agent._tools = {}
    agent.system_message = system_message
    async def send_email(to: str, body: str) -> str:
        return f"Email sent to {to}"
    agent._tools["send_email"] = send_email
    return agent


@pytest.mark.asyncio
async def test_autogen_adapter_wraps_tools(guard):
    agent = make_autogen_agent()
    adapter = AutoGenAdapter()
    await adapter.apply_to_agent(agent, guard)
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


@pytest.mark.asyncio
async def test_autogen_adapter_handles_sync_tool(guard):
    agent = make_autogen_agent()
    # Replace the async tool with a sync one
    def sync_send_email(to: str, body: str) -> str:
        return f"Sync email sent to {to}"
    agent._tools["send_email"] = sync_send_email
    adapter = AutoGenAdapter()
    await adapter.apply_to_agent(agent, guard)
    result = await agent._tools["send_email"](to="alice@example.com", body="Hello")
    assert result == "Sync email sent to alice@example.com"

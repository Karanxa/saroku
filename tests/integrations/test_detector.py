import pytest
from unittest.mock import MagicMock
from saroku.integrations._detector import detect_framework


def make_adk_agent():
    agent = MagicMock()
    agent.before_tool_callback = None  # ADK agents have this attribute
    del agent.tools  # avoid false LangChain positive
    return agent


def make_langchain_agent():
    try:
        from langchain_core.tools import BaseTool
        agent = MagicMock()
        tool = MagicMock(spec=BaseTool)
        agent.tools = [tool]
        del agent.before_tool_callback  # no ADK attribute
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
    with pytest.raises(ValueError, match="wrap"):
        detect_framework(agent)

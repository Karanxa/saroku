"""
Tests for AnthropicAdapter's retry behavior — matches OpenAIAdapter's
resilience (exponential backoff on RateLimitError / retryable APIStatusError
status codes), fixing a prior asymmetry where Anthropic had none.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import anthropic
import pytest

from saroku.adapters.anthropic import AnthropicAdapter


def _fake_message(text: str = "SAFE"):
    content_block = Mock()
    content_block.text = text
    msg = Mock()
    msg.content = [content_block]
    return msg


def _status_error(cls, status_code: int):
    response = Mock()
    response.status_code = status_code
    return cls(message="error", response=response, body=None)


@pytest.mark.asyncio
async def test_retries_on_rate_limit_then_succeeds():
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter.model = "claude-3-5-haiku-20241022"
    adapter._rate_limit_exc = anthropic.RateLimitError
    adapter._api_status_exc = anthropic.APIStatusError

    fake_client = AsyncMock()
    calls = {"n": 0}

    async def create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _status_error(anthropic.RateLimitError, 429)
        return _fake_message("OK")

    fake_client.messages.create = create
    adapter._client = fake_client

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await adapter.achat("hello")

    assert result == "OK"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retries_on_overloaded_529_then_succeeds():
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter.model = "claude-3-5-haiku-20241022"
    adapter._rate_limit_exc = anthropic.RateLimitError
    adapter._api_status_exc = anthropic.APIStatusError

    fake_client = AsyncMock()
    calls = {"n": 0}

    async def create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            # OverloadedError is Anthropic's real 529 "overloaded" error,
            # a subclass of APIStatusError with status_code=529.
            raise _status_error(anthropic.OverloadedError, 529)
        return _fake_message("OK")

    fake_client.messages.create = create
    adapter._client = fake_client

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await adapter.achat("hello")

    assert result == "OK"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_non_retryable_status_fails_immediately():
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter.model = "claude-3-5-haiku-20241022"
    adapter._rate_limit_exc = anthropic.RateLimitError
    adapter._api_status_exc = anthropic.APIStatusError

    fake_client = AsyncMock()
    calls = {"n": 0}

    async def create(**kwargs):
        calls["n"] += 1
        # 401 is not in the retryable status-code set — should not retry.
        raise _status_error(anthropic.AuthenticationError, 401)

    fake_client.messages.create = create
    adapter._client = fake_client

    with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
        with pytest.raises(anthropic.AuthenticationError):
            await adapter.achat("hello")

    assert calls["n"] == 1
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_exhausts_retries_and_raises():
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter.model = "claude-3-5-haiku-20241022"
    adapter._rate_limit_exc = anthropic.RateLimitError
    adapter._api_status_exc = anthropic.APIStatusError

    fake_client = AsyncMock()
    calls = {"n": 0}

    async def create(**kwargs):
        calls["n"] += 1
        raise _status_error(anthropic.RateLimitError, 429)

    fake_client.messages.create = create
    adapter._client = fake_client

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(anthropic.RateLimitError):
            await adapter.achat("hello")

    # DEFAULT_MAX_RETRIES = 6 attempts total before giving up.
    assert calls["n"] == 6

import pytest

from saroku.adapters.factory import _AUTO_DETECT_ORDER, detect_available_provider
from saroku.guard import SafetyGuard

_ALL_ENV_VARS = [env_var for _, env_var, _ in _AUTO_DETECT_ORDER]


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Ensure no real API keys from the environment leak into these tests."""
    for env_var in _ALL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    yield


def test_detect_available_provider_none_set():
    assert detect_available_provider() is None


def test_detect_available_provider_single_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert detect_available_provider() == "anthropic:claude-3-5-haiku-20241022"


def test_detect_available_provider_priority_order(monkeypatch):
    # openai comes before anthropic in _AUTO_DETECT_ORDER — openai should win
    # even though anthropic's key was set first.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert detect_available_provider() == "openai:gpt-4o-mini"


def test_detect_available_provider_skips_ollama_and_azure():
    # Ollama/Azure are intentionally excluded from auto-detection — confirm
    # neither env var name appears anywhere in the priority list.
    assert not any(env_var in ("OLLAMA_API_KEY",) for _, env_var, _ in _AUTO_DETECT_ORDER)
    assert not any(provider in ("ollama", "azure") for provider, _, _ in _AUTO_DETECT_ORDER)


def test_safety_guard_raises_when_no_provider_detected():
    with pytest.raises(ValueError, match="No judge_model specified"):
        SafetyGuard()


def test_safety_guard_auto_detects_single_provider(monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    guard = SafetyGuard()
    assert guard.judge_model == "google:gemini-2.0-flash"
    assert guard._adapter is not None
    captured = capsys.readouterr()
    assert "auto-detected" in captured.out
    assert "google:gemini-2.0-flash" in captured.out


def test_safety_guard_explicit_judge_model_overrides_auto_detection(monkeypatch):
    # Even with a key present for another provider, an explicit judge_model
    # must win.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    guard = SafetyGuard(judge_model="anthropic:claude-3-5-haiku-20241022")
    assert guard.judge_model == "anthropic:claude-3-5-haiku-20241022"


def test_safety_guard_local_only_mode_does_not_require_provider():
    # mode="local" never touches the adapter-resolution path at all.
    guard = SafetyGuard(mode="local", local_model_path=None)
    # local_model_path=None means no local judge either, but mode="local"
    # short-circuits before adapter resolution — _adapter stays None, no error.
    assert guard._adapter is None

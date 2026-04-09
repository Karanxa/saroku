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

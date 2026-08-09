import os
import tempfile

import pytest

from saroku.classifiers.base import ClassificationResult, Classifier
from saroku.classifiers.registry import ClassifierRegistry
from saroku.guard import MODE_BALANCED, SafetyGuard, SafetyCheckResult
from saroku.policy import ExecutionLayer, Policy, PolicyProperty


@pytest.fixture(autouse=True)
def _clear_registry():
    ClassifierRegistry.clear()
    yield
    ClassifierRegistry.clear()


class FakeClassifier(Classifier):
    """Deterministic in-memory Classifier — no API calls, no local model."""

    def __init__(self, classifier_id, confidence, is_safe=True, description="", recommendation=""):
        self._id = classifier_id
        self.confidence = confidence
        self.is_safe = is_safe
        self.description = description
        self.recommendation = recommendation

    @property
    def identifier(self):
        return self._id

    async def aclassify(self, property_name, action, context=None, **kwargs):
        return ClassificationResult(
            is_safe=self.is_safe,
            property=property_name,
            severity="none" if self.is_safe else "high",
            confidence=self.confidence,
            description=self.description,
            recommendation=self.recommendation,
            classifier_id=self._id,
        )


def _single_property_policy(classifier_id: str, name: str = "honesty") -> Policy:
    return Policy(
        version="1",
        policy_id="test-policy",
        properties=[PolicyProperty(name=name, classifier=classifier_id, confidence_threshold=0.8)],
        execution={
            MODE_BALANCED: [
                ExecutionLayer(name="only", classifiers=[classifier_id], timeout_ms=1000, strategy="cascade"),
            ]
        },
    )


# ── policy=Policy instance ──────────────────────────────────────────────────


def test_guard_with_policy_object_safe():
    ClassifierRegistry.register("custom:fake-safe", FakeClassifier("custom:fake-safe", confidence=1.0, is_safe=True))
    policy = _single_property_policy("custom:fake-safe")

    guard = SafetyGuard(policy=policy)
    result = guard.check(action="SELECT * FROM logs", context="read-only query")

    assert isinstance(result, SafetyCheckResult)
    assert result.is_safe is True
    assert result.violations == []
    assert result.checked_properties == ["honesty"]
    assert "custom:fake-safe" in result.layers_used


def test_guard_with_policy_object_unsafe():
    ClassifierRegistry.register(
        "custom:fake-unsafe",
        FakeClassifier(
            "custom:fake-unsafe", confidence=1.0, is_safe=False,
            description="Claimed certainty it didn't have.",
            recommendation="Flag the uncertainty explicitly.",
        ),
    )
    policy = _single_property_policy("custom:fake-unsafe")

    guard = SafetyGuard(policy=policy, block_on="high")
    result = guard.check(action="I'm 100% sure this will work.", context="unverified claim")

    assert result.is_safe is False
    assert len(result.violations) == 1
    v = result.violations[0]
    assert v.property == "honesty"
    assert v.severity == "high"
    assert v.description == "Claimed certainty it didn't have."
    assert v.source == "custom:fake-unsafe"


def test_guard_with_policy_respects_block_on_threshold():
    ClassifierRegistry.register(
        "custom:fake-medium",
        FakeClassifier("custom:fake-medium", confidence=1.0, is_safe=False),
    )
    policy = _single_property_policy("custom:fake-medium")
    # FakeClassifier always reports severity="high" when unsafe; block_on="high"
    # should mark it unsafe, and a stricter-than-needed block_on shouldn't be
    # bypassed just because we're on the engine path.
    guard = SafetyGuard(policy=policy, block_on="high")
    result = guard.check(action="do something", context="ctx")
    assert result.is_safe is False


# ── policy=<path to YAML file> ──────────────────────────────────────────────


def test_guard_with_policy_yaml_path():
    ClassifierRegistry.register("custom:fake-safe", FakeClassifier("custom:fake-safe", confidence=1.0, is_safe=True))
    policy = _single_property_policy("custom:fake-safe")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(policy.to_yaml())
        path = f.name

    try:
        guard = SafetyGuard(policy=path)
        assert isinstance(guard.policy, Policy)
        result = guard.check(action="SELECT 1", context="ctx")
        assert result.is_safe is True
    finally:
        os.unlink(path)


async def test_guard_with_policy_object_async():
    ClassifierRegistry.register("custom:fake-safe", FakeClassifier("custom:fake-safe", confidence=1.0, is_safe=True))
    policy = _single_property_policy("custom:fake-safe")

    guard = SafetyGuard(policy=policy)
    result = await guard.acheck(action="SELECT 1", context="ctx")
    assert result.is_safe is True


# ── legacy params still work without a policy (no engine involved) ─────────


def test_guard_legacy_construction_has_no_engine():
    guard = SafetyGuard(model_adapter=object(), local_model_path=None, mode="thorough")
    assert guard._engine is None
    assert guard.policy is None
    assert guard.metrics is None


# ── metrics exposure on the policy path ─────────────────────────────────────


def test_guard_metrics_exposes_engine_metrics_on_policy_path():
    from saroku.execution.metrics import ExecutionMetrics

    ClassifierRegistry.register("custom:fake-safe", FakeClassifier("custom:fake-safe", confidence=1.0, is_safe=True))
    policy = _single_property_policy("custom:fake-safe")

    guard = SafetyGuard(policy=policy)
    assert isinstance(guard.metrics, ExecutionMetrics)
    assert guard.metrics is guard._engine.metrics
    assert guard.metrics.to_list() == []

    guard.check(action="SELECT 1", context="ctx")
    invocations = guard.metrics.to_list()
    assert len(invocations) == 1
    assert invocations[0].classifier_id == "custom:fake-safe"
    assert invocations[0].outcome == "confident"

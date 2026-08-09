import asyncio

import pytest

from saroku.classifiers.base import ClassificationResult, Classifier
from saroku.classifiers.registry import ClassifierRegistry
from saroku.execution.engine import EvaluationResult, ExecutionEngine
from saroku.execution.metrics import ExecutionMetrics
from saroku.policy.dsl import ExecutionLayer, Policy, PolicyProperty


@pytest.fixture(autouse=True)
def _clear_registry():
    ClassifierRegistry.clear()
    yield
    ClassifierRegistry.clear()


class FakeClassifier(Classifier):
    """Deterministic, fast, in-memory Classifier for tests.

    If only_for_property is set, this classifier only returns its
    configured confidence/verdict when asked about that property —
    for any other property it defers with confidence=0.0, mirroring
    RuleClassifier's real "uncertain, escalate" contract. This keeps
    multi-property tests honest when properties share execution layers.
    """

    def __init__(
        self, classifier_id, confidence, is_safe=True, delay=0.0,
        calls=None, raises=False, only_for_property=None,
    ):
        self._id = classifier_id
        self.confidence = confidence
        self.is_safe = is_safe
        self.delay = delay
        self.calls = calls if calls is not None else []
        self.raises = raises
        self.only_for_property = only_for_property

    @property
    def identifier(self):
        return self._id

    async def aclassify(self, property_name, action, context=None, **kwargs):
        self.calls.append(self._id)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise RuntimeError("boom")
        if self.only_for_property is not None and property_name != self.only_for_property:
            return ClassificationResult(
                is_safe=True, property=property_name, severity="none",
                confidence=0.0, description="not applicable", recommendation="",
                classifier_id=self._id,
            )
        return ClassificationResult(
            is_safe=self.is_safe,
            property=property_name,
            severity="none" if self.is_safe else "high",
            confidence=self.confidence,
            description="" if self.is_safe else "flagged",
            recommendation="",
            classifier_id=self._id,
        )


def _register(classifier_id, confidence, is_safe=True, delay=0.0, calls=None, raises=False, only_for_property=None):
    fake = FakeClassifier(classifier_id, confidence, is_safe, delay, calls, raises, only_for_property)
    ClassifierRegistry.register(classifier_id, fake)
    return fake


def _policy(execution, properties=None, fallback=None, confidence_threshold=0.8):
    props = properties or [
        PolicyProperty(
            name="honesty",
            classifier="custom:primary",
            fallback=fallback,
            confidence_threshold=confidence_threshold,
        )
    ]
    return Policy(
        version="1",
        policy_id="test",
        properties=props,
        execution=execution,
    )


# ── Cascade strategy ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cascade_stops_at_first_confident_result():
    calls = []
    _register("custom:a", confidence=0.2, calls=calls)
    _register("custom:b", confidence=0.95, is_safe=False, calls=calls)
    _register("custom:c", confidence=0.99, calls=calls)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:a", "custom:b", "custom:c"], strategy="cascade"),
            ],
        }
    )
    engine = ExecutionEngine(policy)
    result = await engine.evaluate_property("honesty", "some action", mode="balanced")

    assert result.classifier_id == "custom:b"
    assert result.is_safe is False
    assert calls == ["custom:a", "custom:b"]  # c never called — cascade stopped early


# ── Speculative strategy ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_speculative_uses_first_confident_and_cancels_rest():
    calls = []
    slow = _register("custom:slow", confidence=0.99, delay=0.3, calls=calls)
    fast = _register("custom:fast", confidence=0.95, is_safe=False, delay=0.01, calls=calls)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(
                    name="l1", classifiers=["custom:slow", "custom:fast"],
                    strategy="speculative", timeout_ms=2000,
                ),
            ],
        }
    )
    engine = ExecutionEngine(policy)
    result = await engine.evaluate_property("honesty", "some action", mode="balanced")

    assert result.classifier_id == "custom:fast"
    assert result.is_safe is False
    # Give the cancelled task's cancellation a moment to settle.
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_speculative_all_uncertain_returns_best_effort_then_uncertain():
    _register("custom:a", confidence=0.1)
    _register("custom:b", confidence=0.3)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:a", "custom:b"], strategy="speculative"),
            ],
        },
        confidence_threshold=0.8,
    )
    engine = ExecutionEngine(policy)
    result = await engine.evaluate_property("honesty", "some action", mode="balanced")

    assert result.confidence < 0.8
    assert "uncertain" in result.description.lower()


# ── Timeout enforcement ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_layer_timeout_moves_to_next_layer():
    calls = []
    _register("custom:hangs", confidence=0.99, delay=5.0, calls=calls)
    _register("custom:backup", confidence=0.9, calls=calls)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:hangs"], timeout_ms=50, strategy="cascade"),
                ExecutionLayer(name="l2", classifiers=["custom:backup"], timeout_ms=1000, strategy="cascade"),
            ],
        }
    )
    engine = ExecutionEngine(policy)
    result = await engine.evaluate_property("honesty", "some action", mode="balanced")

    assert result.classifier_id == "custom:backup"


# ── Fallback classifier ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_triggers_when_primary_uncertain():
    _register("custom:primary", confidence=0.1)
    _register("custom:fallback_classifier", confidence=0.95, is_safe=False)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:primary"], strategy="cascade"),
            ],
        },
        fallback="custom:fallback_classifier",
    )
    engine = ExecutionEngine(policy)
    result = await engine.evaluate_property("honesty", "some action", mode="balanced")

    assert result.classifier_id == "custom:fallback_classifier"
    assert result.is_safe is False


@pytest.mark.asyncio
async def test_no_fallback_and_no_confident_result_returns_uncertain():
    _register("custom:primary", confidence=0.2)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:primary"], strategy="cascade"),
            ],
        },
    )
    engine = ExecutionEngine(policy)
    result = await engine.evaluate_property("honesty", "some action", mode="balanced")

    assert result.classifier_id == "custom:primary"
    assert "uncertain" in result.description.lower()
    assert result.confidence == 0.2


# ── evaluate_all_properties ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_all_properties_runs_concurrently_and_aggregates():
    calls = []
    _register("custom:honesty", confidence=0.9, is_safe=True, calls=calls, delay=0.05, only_for_property="honesty")
    _register(
        "custom:sycophancy", confidence=0.9, is_safe=False, calls=calls, delay=0.05,
        only_for_property="sycophancy",
    )

    properties = [
        PolicyProperty(name="honesty", classifier="custom:honesty", confidence_threshold=0.8),
        PolicyProperty(name="sycophancy", classifier="custom:sycophancy", confidence_threshold=0.8),
    ]
    # Both properties share the same "balanced" execution cascade — each
    # classifier only fires confidently for its own property (see
    # only_for_property above) and defers (confidence=0.0) otherwise, so the
    # cascade naturally falls through to the right layer per property.
    policy = Policy(
        version="1",
        policy_id="multi",
        properties=properties,
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:honesty"], strategy="cascade"),
                ExecutionLayer(name="l2", classifiers=["custom:sycophancy"], strategy="cascade"),
            ],
        },
    )

    engine = ExecutionEngine(policy)
    t_start = asyncio.get_event_loop().time()
    result = await engine.evaluate_all_properties(
        "some action", mode="balanced", properties=["honesty", "sycophancy"]
    )
    elapsed = asyncio.get_event_loop().time() - t_start

    assert isinstance(result, EvaluationResult)
    assert result.checked_properties == ["honesty", "sycophancy"]
    assert len(result.results) == 2
    assert result.is_safe is False
    assert len(result.violations) == 1
    assert result.violations[0].property == "sycophancy"
    # Both properties ran concurrently: total time should be well under the
    # sum of both individual delays (0.05 + 0.05), allowing generous slack.
    assert elapsed < 0.15


# ── Metrics ──────────────────────────────────────────────────────────────

def test_engine_has_default_metrics_instance():
    engine = ExecutionEngine(_policy(execution={"balanced": []}))
    assert isinstance(engine.metrics, ExecutionMetrics)
    assert engine.metrics.to_list() == []


def test_engine_accepts_injected_metrics():
    metrics = ExecutionMetrics()
    engine = ExecutionEngine(_policy(execution={"balanced": []}), metrics=metrics)
    assert engine.metrics is metrics


@pytest.mark.asyncio
async def test_metrics_recorded_for_cascade():
    _register("custom:a", confidence=0.2)
    _register("custom:b", confidence=0.95, is_safe=False)
    _register("custom:c", confidence=0.99)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:a", "custom:b", "custom:c"], strategy="cascade"),
            ],
        }
    )
    engine = ExecutionEngine(policy)
    await engine.evaluate_property("honesty", "some action", mode="balanced")

    invocations = engine.metrics.to_list()
    assert [i.classifier_id for i in invocations] == ["custom:a", "custom:b"]
    assert invocations[0].outcome == "deferred"
    assert invocations[0].layer_name == "l1"
    assert invocations[0].strategy == "cascade"
    assert invocations[1].outcome == "confident"
    assert invocations[1].confidence == 0.95
    assert invocations[1].is_safe is False


@pytest.mark.asyncio
async def test_metrics_recorded_for_speculative():
    _register("custom:slow", confidence=0.99, delay=0.3)
    _register("custom:fast", confidence=0.95, is_safe=False, delay=0.01)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(
                    name="l1", classifiers=["custom:slow", "custom:fast"],
                    strategy="speculative", timeout_ms=2000,
                ),
            ],
        }
    )
    engine = ExecutionEngine(policy)
    await engine.evaluate_property("honesty", "some action", mode="balanced")
    await asyncio.sleep(0.05)

    invocations = {i.classifier_id: i for i in engine.metrics.to_list()}
    assert invocations["custom:fast"].outcome == "confident"
    assert invocations["custom:fast"].strategy == "speculative"
    assert invocations["custom:slow"].outcome == "cancelled"


@pytest.mark.asyncio
async def test_metrics_recorded_as_timeout_when_layer_times_out():
    _register("custom:hangs", confidence=0.99, delay=5.0)
    _register("custom:backup", confidence=0.9)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:hangs"], timeout_ms=50, strategy="cascade"),
                ExecutionLayer(name="l2", classifiers=["custom:backup"], timeout_ms=1000, strategy="cascade"),
            ],
        }
    )
    engine = ExecutionEngine(policy)
    await engine.evaluate_property("honesty", "some action", mode="balanced")

    invocations = {i.classifier_id: i for i in engine.metrics.to_list()}
    assert invocations["custom:hangs"].outcome == "timeout"
    assert invocations["custom:hangs"].layer_name == "l1"
    assert invocations["custom:backup"].outcome == "confident"
    assert invocations["custom:backup"].layer_name == "l2"


@pytest.mark.asyncio
async def test_metrics_recorded_for_fallback():
    _register("custom:primary", confidence=0.1)
    _register("custom:fallback_classifier", confidence=0.95, is_safe=False)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:primary"], strategy="cascade"),
            ],
        },
        fallback="custom:fallback_classifier",
    )
    engine = ExecutionEngine(policy)
    await engine.evaluate_property("honesty", "some action", mode="balanced")

    invocations = {i.classifier_id: i for i in engine.metrics.to_list()}
    assert invocations["custom:primary"].outcome == "deferred"
    assert invocations["custom:fallback_classifier"].outcome == "confident"
    assert invocations["custom:fallback_classifier"].layer_name == "fallback"


@pytest.mark.asyncio
async def test_metrics_recorded_as_error_for_raising_classifier():
    calls = []
    _register("custom:primary", confidence=0.99, calls=calls, raises=True)

    policy = _policy(
        execution={
            "balanced": [
                ExecutionLayer(name="l1", classifiers=["custom:primary"], strategy="cascade"),
            ],
        },
    )
    engine = ExecutionEngine(policy)
    with pytest.raises(RuntimeError):
        await engine.evaluate_property("honesty", "some action", mode="balanced")

    invocations = engine.metrics.to_list()
    assert len(invocations) == 1
    assert invocations[0].outcome == "error"
    assert invocations[0].confidence is None

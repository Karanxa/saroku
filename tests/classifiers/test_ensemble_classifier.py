import pytest

from saroku.classifiers import ClassificationResult, Classifier, EnsembleClassifier


class StubClassifier(Classifier):
    def __init__(self, is_safe, confidence=1.0, severity="none", classifier_id="stub"):
        self._is_safe = is_safe
        self._confidence = confidence
        self._severity = severity if not is_safe else "none"
        self._classifier_id = classifier_id

    @property
    def identifier(self):
        return self._classifier_id

    async def aclassify(self, property_name, action, context=None, **kwargs):
        return ClassificationResult(
            is_safe=self._is_safe,
            property=property_name,
            severity=self._severity,
            confidence=self._confidence,
            description="" if self._is_safe else f"{self._classifier_id} flagged it",
            recommendation="" if self._is_safe else "review",
            classifier_id=self._classifier_id,
        )


def test_requires_at_least_one_classifier():
    with pytest.raises(ValueError):
        EnsembleClassifier([])


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        EnsembleClassifier([StubClassifier(True)], strategy="unknown")


def test_default_identifier():
    ensemble = EnsembleClassifier([StubClassifier(True)])
    assert ensemble.identifier == "ensemble:majority"


def test_custom_identifier():
    ensemble = EnsembleClassifier([StubClassifier(True)], classifier_id="custom:my-ensemble")
    assert ensemble.identifier == "custom:my-ensemble"


@pytest.mark.asyncio
async def test_majority_safe_when_majority_agree():
    ensemble = EnsembleClassifier(
        [StubClassifier(True), StubClassifier(True), StubClassifier(False, severity="high")],
        strategy="majority",
    )
    result = await ensemble.aclassify("honesty", action="did X")
    assert result.is_safe is True
    assert result.classifier_id == "ensemble:majority"


@pytest.mark.asyncio
async def test_majority_unsafe_when_majority_disagree():
    ensemble = EnsembleClassifier(
        [StubClassifier(False, severity="high"), StubClassifier(False, severity="medium"), StubClassifier(True)],
        strategy="majority",
    )
    result = await ensemble.aclassify("honesty", action="did X")
    assert result.is_safe is False
    assert result.severity == "high"
    assert result.description


@pytest.mark.asyncio
async def test_majority_tie_counts_as_safe():
    ensemble = EnsembleClassifier(
        [StubClassifier(False, severity="high"), StubClassifier(True)],
        strategy="majority",
    )
    result = await ensemble.aclassify("honesty", action="did X")
    assert result.is_safe is True


@pytest.mark.asyncio
async def test_majority_runs_concurrently(monkeypatch):
    import asyncio

    order = []

    class SlowClassifier(Classifier):
        def __init__(self, name, delay):
            self._name = name
            self._delay = delay

        @property
        def identifier(self):
            return self._name

        async def aclassify(self, property_name, action, context=None, **kwargs):
            await asyncio.sleep(self._delay)
            order.append(self._name)
            return ClassificationResult(
                is_safe=True, property=property_name, severity="none",
                confidence=1.0, description="", recommendation="",
                classifier_id=self._name,
            )

    ensemble = EnsembleClassifier(
        [SlowClassifier("slow", 0.05), SlowClassifier("fast", 0.01)],
        strategy="majority",
    )
    start = asyncio.get_event_loop().time()
    await ensemble.aclassify("honesty", action="did X")
    elapsed = asyncio.get_event_loop().time() - start
    assert order == ["fast", "slow"]
    assert elapsed < 0.09  # concurrent, not 0.05+0.01 sequential-safe margin


@pytest.mark.asyncio
async def test_cascade_returns_first_confident_result():
    ensemble = EnsembleClassifier(
        [
            StubClassifier(True, confidence=0.3, classifier_id="low_conf"),
            StubClassifier(False, confidence=0.9, severity="high", classifier_id="high_conf"),
            StubClassifier(True, confidence=1.0, classifier_id="never_reached"),
        ],
        strategy="cascade",
        cascade_threshold=0.75,
    )
    result = await ensemble.aclassify("honesty", action="did X")
    assert result.classifier_id == "ensemble:cascade"
    assert result.is_safe is False
    assert result.description == "high_conf flagged it"


@pytest.mark.asyncio
async def test_cascade_falls_back_to_last_if_none_confident():
    ensemble = EnsembleClassifier(
        [
            StubClassifier(True, confidence=0.1, classifier_id="a"),
            StubClassifier(False, confidence=0.2, severity="low", classifier_id="b"),
        ],
        strategy="cascade",
        cascade_threshold=0.75,
    )
    result = await ensemble.aclassify("honesty", action="did X")
    assert result.description == "b flagged it"


@pytest.mark.asyncio
async def test_cascade_short_circuits_remaining_classifiers():
    calls = []

    class TrackingClassifier(Classifier):
        def __init__(self, name, confidence):
            self._name = name
            self._confidence = confidence

        @property
        def identifier(self):
            return self._name

        async def aclassify(self, property_name, action, context=None, **kwargs):
            calls.append(self._name)
            return ClassificationResult(
                is_safe=True, property=property_name, severity="none",
                confidence=self._confidence, description="", recommendation="",
                classifier_id=self._name,
            )

    ensemble = EnsembleClassifier(
        [TrackingClassifier("first", 0.9), TrackingClassifier("second", 1.0)],
        strategy="cascade",
        cascade_threshold=0.75,
    )
    await ensemble.aclassify("honesty", action="did X")
    assert calls == ["first"]

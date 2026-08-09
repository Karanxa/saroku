import pytest

from saroku.classifiers import ClassificationResult, Classifier


class StubClassifier(Classifier):
    async def aclassify(self, property_name, action, context=None, **kwargs):
        return ClassificationResult(
            is_safe=True,
            property=property_name,
            severity="none",
            confidence=0.9,
            description="",
            recommendation="",
            classifier_id=self.identifier,
        )


def test_classification_result_fields():
    result = ClassificationResult(
        is_safe=False,
        property="honesty",
        severity="high",
        confidence=0.8,
        description="claims certainty it cannot verify",
        recommendation="flag the uncertainty",
        classifier_id="llm:gpt-4o-mini",
    )
    assert result.is_safe is False
    assert result.property == "honesty"
    assert result.severity == "high"
    assert result.confidence == 0.8
    assert result.description == "claims certainty it cannot verify"
    assert result.recommendation == "flag the uncertainty"
    assert result.classifier_id == "llm:gpt-4o-mini"
    assert result.raw_output is None
    assert result.latency_ms == 0.0


def test_classifier_default_identifier():
    classifier = StubClassifier()
    assert classifier.identifier == "StubClassifier"


@pytest.mark.asyncio
async def test_aclassify_returns_classification_result():
    classifier = StubClassifier()
    result = await classifier.aclassify("sycophancy", action="held position", context="pressure applied")
    assert isinstance(result, ClassificationResult)
    assert result.property == "sycophancy"
    assert result.is_safe is True


def test_classify_sync_wrapper_runs_aclassify():
    classifier = StubClassifier()
    result = classifier.classify("goal_drift", action="stayed on task")
    assert isinstance(result, ClassificationResult)
    assert result.property == "goal_drift"
    assert result.classifier_id == "StubClassifier"


def test_classifier_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Classifier()

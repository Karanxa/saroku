import contextlib
import sys
import types

import pytest

from saroku.classifiers import (
    Classifier,
    ClassificationResult,
    ClassifierRegistry,
    EnsembleClassifier,
    HFModelClassifier,
    LLMClassifier,
    LocalSarokaClassifier,
    RuleClassifier,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    ClassifierRegistry.clear()
    yield
    ClassifierRegistry.clear()


class DummyClassifier(Classifier):
    async def aclassify(self, property_name, action, context=None, **kwargs):
        return ClassificationResult(
            is_safe=True, property=property_name, severity="none",
            confidence=1.0, description="", recommendation="",
            classifier_id=self.identifier,
        )


def test_resolve_rule_returns_rule_classifier():
    classifier = ClassifierRegistry.resolve("rule:capitulation")
    assert isinstance(classifier, RuleClassifier)
    assert classifier.identifier == "rule:capitulation"


def test_resolve_rule_caches_instance():
    first = ClassifierRegistry.resolve("rule:capitulation")
    second = ClassifierRegistry.resolve("rule:capitulation")
    assert first is second


def test_resolve_llm_returns_llm_classifier(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    classifier = ClassifierRegistry.resolve("llm:gpt-4o-mini")
    assert isinstance(classifier, LLMClassifier)
    assert classifier.identifier == "llm:gpt-4o-mini"


def test_resolve_llm_caches_instance(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    first = ClassifierRegistry.resolve("llm:gpt-4o-mini")
    second = ClassifierRegistry.resolve("llm:gpt-4o-mini")
    assert first is second


def test_resolve_llm_with_provider_prefixed_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-groq-test")
    classifier = ClassifierRegistry.resolve("llm:groq:llama-3.3-70b-versatile")
    assert isinstance(classifier, LLMClassifier)


def test_resolve_custom_requires_prior_registration():
    with pytest.raises(KeyError):
        ClassifierRegistry.resolve("custom:my_detector")


def test_register_then_resolve_custom():
    dummy = DummyClassifier()
    ClassifierRegistry.register("custom:my_detector", dummy)
    resolved = ClassifierRegistry.resolve("custom:my_detector")
    assert resolved is dummy


def test_resolve_unknown_prefix_raises():
    with pytest.raises(ValueError):
        ClassifierRegistry.resolve("unknown:thing")


def test_resolve_without_colon_raises():
    with pytest.raises(ValueError):
        ClassifierRegistry.resolve("gpt-4o-mini")


def test_classify_sync_wrapper_via_registry():
    classifier = ClassifierRegistry.resolve("rule:sycophancy")
    result = classifier.classify(
        "sycophancy",
        action="I was wrong, you're right.",
        initial_response="no",
        correct_answer="no",
    )
    assert isinstance(result, ClassificationResult)
    assert result.is_safe is False
    assert result.severity == "high"


def test_rule_classifier_uncertain_for_unhandled_property():
    classifier = RuleClassifier()
    result = classifier.classify("goal_drift", action="did something")
    assert result.confidence == 0.0
    assert result.is_safe is True


def test_resolve_hf_returns_hf_classifier():
    classifier = ClassifierRegistry.resolve("hf:org/model")
    assert isinstance(classifier, HFModelClassifier)
    assert classifier.identifier == "hf:org/model"
    assert classifier.model_id == "org/model"


def test_resolve_hf_caches_instance():
    first = ClassifierRegistry.resolve("hf:org/model")
    second = ClassifierRegistry.resolve("hf:org/model")
    assert first is second


def test_resolve_local_returns_local_saroka_classifier():
    classifier = ClassifierRegistry.resolve("local:saroku-safety")
    assert isinstance(classifier, LocalSarokaClassifier)
    assert classifier.identifier == "local:saroku-safety"


def test_resolve_local_caches_instance():
    first = ClassifierRegistry.resolve("local:saroku-safety")
    second = ClassifierRegistry.resolve("local:saroku-safety")
    assert first is second


def test_register_then_resolve_custom_ensemble():
    ensemble = EnsembleClassifier([DummyClassifier()], classifier_id="custom:my-ensemble")
    ClassifierRegistry.register("custom:my-ensemble", ensemble)
    resolved = ClassifierRegistry.resolve("custom:my-ensemble")
    assert resolved is ensemble
    assert isinstance(resolved, EnsembleClassifier)


@pytest.mark.asyncio
async def test_hf_classifier_resolved_from_registry_classifies(monkeypatch):
    import numpy as np

    class _FakeTokenizer:
        def __call__(self, text, return_tensors=None, truncation=None):
            return {"input_ids": [[1, 2, 3]]}

    class _FakeModel:
        def eval(self):
            pass

        def __call__(self, **inputs):
            return types.SimpleNamespace(logits=np.array([[5.0, 0.0]]))

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda model_id: _FakeTokenizer()),
        AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda model_id: _FakeModel()),
    )
    fake_torch = types.SimpleNamespace(no_grad=lambda: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    classifier = ClassifierRegistry.resolve("hf:org/model")
    result = await classifier.aclassify("honesty", action="told the truth")
    assert isinstance(result, ClassificationResult)
    assert result.is_safe is True

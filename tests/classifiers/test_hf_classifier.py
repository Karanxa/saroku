import contextlib
import sys
import types

import numpy as np
import pytest

from saroku.classifiers import ClassificationResult, HFModelClassifier, LocalSarokaClassifier


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None, truncation=None):
        return {"input_ids": [[1, 2, 3]]}


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    def __init__(self, logits):
        self._logits = logits

    def eval(self):
        pass

    def __call__(self, **inputs):
        return _FakeOutput(self._logits)


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch):
    fake_torch = types.SimpleNamespace(no_grad=lambda: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    yield


def _patch_transformers(monkeypatch, logits):
    fake_tokenizer_cls = types.SimpleNamespace(from_pretrained=lambda model_id: _FakeTokenizer())
    fake_model_cls = types.SimpleNamespace(from_pretrained=lambda model_id: _FakeModel(logits))
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=fake_tokenizer_cls,
        AutoModelForSequenceClassification=fake_model_cls,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_identifier_defaults_from_model_id():
    classifier = HFModelClassifier("org/model")
    assert classifier.identifier == "hf:org/model"


def test_identifier_can_be_overridden():
    classifier = HFModelClassifier("org/model", classifier_id="hf:custom")
    assert classifier.identifier == "hf:custom"


def test_no_load_at_construction():
    classifier = HFModelClassifier("org/model")
    assert classifier._model is None
    assert classifier._tokenizer is None


@pytest.mark.asyncio
async def test_aclassify_safe_prediction(monkeypatch):
    _patch_transformers(monkeypatch, np.array([[5.0, 0.0]]))
    classifier = HFModelClassifier("org/model", label_map={0: "safe", 1: "unsafe"})
    result = await classifier.aclassify("honesty", action="told the truth")
    assert isinstance(result, ClassificationResult)
    assert result.is_safe is True
    assert result.severity == "none"
    assert result.classifier_id == "hf:org/model"
    assert result.confidence > 0.5


@pytest.mark.asyncio
async def test_aclassify_unsafe_prediction(monkeypatch):
    _patch_transformers(monkeypatch, np.array([[0.0, 5.0]]))
    classifier = HFModelClassifier("org/model", label_map={0: "safe", 1: "unsafe"})
    result = await classifier.aclassify("goal_drift", action="did something else entirely", context="task was X")
    assert result.is_safe is False
    assert result.severity == "high"
    assert result.description
    assert result.recommendation


@pytest.mark.asyncio
async def test_aclassify_caches_loaded_model(monkeypatch):
    load_calls = {"count": 0}

    def from_pretrained(model_id):
        load_calls["count"] += 1
        return _FakeModel(np.array([[5.0, 0.0]]))

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda model_id: _FakeTokenizer()),
        AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=from_pretrained),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    classifier = HFModelClassifier("org/model")
    await classifier.aclassify("honesty", action="a")
    await classifier.aclassify("honesty", action="b")
    assert load_calls["count"] == 1


def test_local_saroka_classifier_default_id():
    classifier = LocalSarokaClassifier()
    assert classifier.identifier == "local:saroku-safety"


@pytest.mark.asyncio
async def test_local_saroka_classifier_safe(monkeypatch):
    from saroku import local_judge

    fake_result = types.SimpleNamespace(verdict="SAFE", raw_output="safe", property=None)
    monkeypatch.setattr(local_judge, "load_model", lambda model_path: None)
    monkeypatch.setattr(local_judge, "evaluate", lambda action, context="": fake_result)

    classifier = LocalSarokaClassifier()
    result = await classifier.aclassify("goal_drift", action="stayed on task")
    assert result.is_safe is True
    assert result.classifier_id == "local:saroku-safety"


@pytest.mark.asyncio
async def test_local_saroka_classifier_unsafe(monkeypatch):
    from saroku import local_judge

    fake_result = types.SimpleNamespace(verdict="UNSAFE", raw_output="<|goal_drift|>", property="goal_drift")
    load_calls = {"count": 0}

    def fake_load_model(model_path):
        load_calls["count"] += 1

    monkeypatch.setattr(local_judge, "load_model", fake_load_model)
    monkeypatch.setattr(local_judge, "evaluate", lambda action, context="": fake_result)

    classifier = LocalSarokaClassifier()
    result = await classifier.aclassify("goal_drift", action="did something else")
    await classifier.aclassify("goal_drift", action="did another thing")

    assert result.is_safe is False
    assert result.severity == "high"
    assert "goal_drift" in result.description
    assert load_calls["count"] == 1

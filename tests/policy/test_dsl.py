import os

import pytest

from saroku.policy import ExecutionLayer, Policy, PolicyMetadata, PolicyProperty

DEFAULT_POLICY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "saroku", "policy", "policies", "default.yml"
)


def test_policy_property_from_dict_defaults():
    prop = PolicyProperty.from_dict({"name": "honesty", "classifier": "llm:gpt-4o-mini"})
    assert prop.name == "honesty"
    assert prop.classifier == "llm:gpt-4o-mini"
    assert prop.fallback is None
    assert prop.context_requirements == []
    assert prop.confidence_threshold == 0.8


def test_execution_layer_rejects_invalid_strategy():
    with pytest.raises(ValueError):
        ExecutionLayer(name="bad", classifiers=["llm:gpt-4o-mini"], strategy="parallel")


def test_execution_layer_valid_strategies():
    cascade = ExecutionLayer(name="a", classifiers=["llm:gpt-4o-mini"], strategy="cascade")
    speculative = ExecutionLayer(name="b", classifiers=["llm:gpt-4o-mini"], strategy="speculative")
    assert cascade.strategy == "cascade"
    assert speculative.strategy == "speculative"


def test_policy_from_dict_and_lookups():
    data = {
        "version": "1",
        "policy_id": "test-policy",
        "properties": [
            {"name": "honesty", "classifier": "llm:gpt-4o-mini", "confidence_threshold": 0.7},
        ],
        "execution": {
            "balanced": [
                {"name": "fast", "classifiers": ["rule:x"], "timeout_ms": 100, "strategy": "cascade"},
            ],
        },
        "metadata": {"version": "1", "author": "tester", "description": "unit test policy"},
    }
    policy = Policy.from_dict(data)
    assert policy.policy_id == "test-policy"
    assert isinstance(policy.metadata, PolicyMetadata)
    assert policy.metadata.author == "tester"

    prop = policy.get_property("honesty")
    assert prop.confidence_threshold == 0.7

    layers = policy.get_layers("balanced")
    assert len(layers) == 1
    assert layers[0].name == "fast"

    with pytest.raises(KeyError):
        policy.get_property("nonexistent")

    with pytest.raises(KeyError):
        policy.get_layers("nonexistent_mode")


def test_policy_to_dict_round_trip():
    data = {
        "version": "1",
        "policy_id": "round-trip",
        "properties": [
            {"name": "honesty", "classifier": "llm:gpt-4o-mini", "fallback": "rule:honesty"},
        ],
        "execution": {
            "thorough": [
                {"name": "judge", "classifiers": ["llm:gpt-4o-mini"], "timeout_ms": 5000, "strategy": "speculative"},
            ],
        },
        "metadata": {"version": "2", "author": "a", "description": "d"},
    }
    policy = Policy.from_dict(data)
    round_tripped = Policy.from_dict(policy.to_dict())
    assert round_tripped.to_dict() == policy.to_dict()


def test_policy_to_yaml_and_from_yaml_round_trip(tmp_path):
    data = {
        "version": "1",
        "policy_id": "yaml-round-trip",
        "properties": [
            {"name": "sycophancy", "classifier": "llm:gpt-4o-mini", "fallback": "rule:sycophancy"},
        ],
        "execution": {
            "balanced": [
                {"name": "fast", "classifiers": ["rule:sycophancy"], "timeout_ms": 200, "strategy": "cascade"},
                {"name": "thorough", "classifiers": ["llm:gpt-4o-mini"], "timeout_ms": 8000, "strategy": "cascade"},
            ],
        },
    }
    policy = Policy.from_dict(data)
    yaml_text = policy.to_yaml()

    path = tmp_path / "policy.yml"
    path.write_text(yaml_text)

    loaded = Policy.from_yaml(str(path))
    assert loaded.to_dict() == policy.to_dict()


def test_default_policy_loads_and_has_expected_shape():
    policy = Policy.from_yaml(DEFAULT_POLICY_PATH)
    assert policy.policy_id == "default"

    property_names = {p.name for p in policy.properties}
    assert property_names == {
        "sycophancy", "honesty", "prompt_injection", "trust_hierarchy",
        "minimal_footprint", "goal_drift", "corrigibility",
    }

    sycophancy = policy.get_property("sycophancy")
    assert sycophancy.classifier == "llm:gpt-4o-mini"
    assert sycophancy.fallback == "rule:sycophancy"

    assert "balanced" in policy.execution
    assert "thorough" in policy.execution
    balanced_layers = policy.get_layers("balanced")
    assert [layer.name for layer in balanced_layers] == ["fast", "thorough"]

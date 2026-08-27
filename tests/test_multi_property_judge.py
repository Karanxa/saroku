"""
Tests for saroku.multi_property_judge — the shared single-call,
multi-property parsing/prompt logic used by both guard.py's legacy path
and classifiers/llm_classifier.py's LLMClassifier.
"""

import asyncio

import pytest

from saroku.multi_property_judge import (
    build_multi_property_prompt,
    evaluate_all_properties_single_call,
    parse_multi_property_response,
)


class _FakeAdapter:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    async def achat(self, prompt: str) -> str:
        self.calls += 1
        return self.response


def test_parse_valid_multi_property_json_mixed_verdicts():
    raw = (
        '{"sycophancy": {"unsafe": false, "confidence": 0.95}, '
        '"honesty": {"unsafe": true, "confidence": 0.8, "severity": "medium", '
        '"description": "vague claim", "recommendation": "be specific"}}'
    )
    results = parse_multi_property_response(raw, ["sycophancy", "honesty"])
    assert len(results) == 2
    assert results[0].property == "sycophancy" and results[0].is_safe
    assert results[1].property == "honesty" and not results[1].is_safe
    assert results[1].severity == "medium"
    assert results[1].description == "vague claim"


def test_parse_malformed_json_fails_closed_for_every_property():
    results = parse_multi_property_response("not json at all", ["sycophancy", "honesty"])
    assert len(results) == 2
    assert all(not r.is_safe for r in results)
    assert results[0].property == "sycophancy:unparseable"
    assert results[1].property == "honesty:unparseable"


def test_parse_empty_response_fails_closed():
    results = parse_multi_property_response("", ["corrigibility"])
    assert len(results) == 1
    assert not results[0].is_safe
    assert results[0].property == "corrigibility:unparseable"


def test_parse_missing_property_in_otherwise_valid_json_fails_closed_for_just_that_one():
    raw = '{"sycophancy": {"unsafe": false, "confidence": 0.9}}'
    results = parse_multi_property_response(raw, ["sycophancy", "honesty"])
    assert results[0].is_safe  # present and safe
    assert not results[1].is_safe  # missing entirely -> fail closed, not silently safe
    assert results[1].property == "honesty:unparseable"


def test_parse_non_bool_unsafe_field_fails_closed():
    raw = '{"sycophancy": {"unsafe": "yes"}}'  # malformed type
    results = parse_multi_property_response(raw, ["sycophancy"])
    assert not results[0].is_safe
    assert results[0].property == "sycophancy:unparseable"


def test_parse_strips_fenced_code_block():
    raw = '```json\n{"sycophancy": {"unsafe": false, "confidence": 1.0}}\n```'
    results = parse_multi_property_response(raw, ["sycophancy"])
    assert results[0].is_safe


def test_build_prompt_drops_trust_hierarchy_without_constraints():
    _, askable = build_multi_property_prompt(
        "action", "context", ["sycophancy", "trust_hierarchy"], constraints=None,
    )
    assert "trust_hierarchy" not in askable
    assert "sycophancy" in askable


def test_build_prompt_drops_goal_drift_without_original_goal():
    _, askable = build_multi_property_prompt(
        "action", "context", ["goal_drift", "honesty"], original_goal=None,
    )
    assert "goal_drift" not in askable
    assert "honesty" in askable


def test_evaluate_all_properties_makes_exactly_one_call_regardless_of_property_count():
    adapter = _FakeAdapter(
        '{"sycophancy": {"unsafe": false, "confidence": 0.9}, '
        '"honesty": {"unsafe": false, "confidence": 0.9}, '
        '"corrigibility": {"unsafe": true, "confidence": 0.9, "severity": "high", '
        '"description": "d", "recommendation": "r"}}'
    )
    verdicts = asyncio.run(evaluate_all_properties_single_call(
        adapter, "action", "context", ["sycophancy", "honesty", "corrigibility"],
    ))
    assert adapter.calls == 1  # one call, three properties
    assert [v.property for v in verdicts] == ["sycophancy", "honesty", "corrigibility"]
    assert verdicts[2].is_safe is False


def test_evaluate_all_properties_skips_call_entirely_when_nothing_askable():
    adapter = _FakeAdapter("should never be read")
    verdicts = asyncio.run(evaluate_all_properties_single_call(
        adapter, "action", "context", ["trust_hierarchy", "goal_drift"],
        constraints=None, original_goal=None,
    ))
    assert adapter.calls == 0  # neither property is askable without its required context
    assert all(v.is_safe for v in verdicts)

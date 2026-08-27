"""
Tests for classifiers/llm_classifier.py's LLMClassifier — specifically its
caching/batching behavior, which is the mechanism that lets ExecutionEngine
keep calling aclassify() once per property (its existing interface,
unchanged) while making exactly ONE real LLM call per (action, context) key
underneath.
"""

import asyncio

from saroku.classifiers.llm_classifier import LLMClassifier


class _FakeAdapter:
    def __init__(self, response: str, delay: float = 0.0):
        self.response = response
        self.delay = delay
        self.calls = 0

    async def achat(self, prompt: str) -> str:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.response


_MULTI_RESPONSE = (
    '{"sycophancy": {"unsafe": false, "confidence": 0.9}, '
    '"honesty": {"unsafe": true, "confidence": 0.85, "severity": "high", '
    '"description": "d", "recommendation": "r"}, '
    '"consistency": {"unsafe": false, "confidence": 0.9}, '
    '"prompt_injection": {"unsafe": false, "confidence": 0.9}, '
    '"trust_hierarchy": {"unsafe": false, "confidence": 0.9}, '
    '"minimal_footprint": {"unsafe": false, "confidence": 0.9}, '
    '"goal_drift": {"unsafe": false, "confidence": 0.9}, '
    '"corrigibility": {"unsafe": false, "confidence": 0.9}}'
)


def test_sequential_calls_for_different_properties_share_one_api_call():
    adapter = _FakeAdapter(_MULTI_RESPONSE)
    classifier = LLMClassifier(adapter, classifier_id="llm:test")

    async def run():
        r1 = await classifier.aclassify("sycophancy", "action", "context")
        r2 = await classifier.aclassify("honesty", "action", "context")
        r3 = await classifier.aclassify("corrigibility", "action", "context")
        return r1, r2, r3

    r1, r2, r3 = asyncio.run(run())
    assert adapter.calls == 1  # three properties, one action -> one API call
    assert r1.is_safe and r3.is_safe
    assert not r2.is_safe and r2.severity == "high"


def test_concurrent_calls_for_same_action_collapse_into_one_api_call():
    adapter = _FakeAdapter(_MULTI_RESPONSE, delay=0.01)
    classifier = LLMClassifier(adapter, classifier_id="llm:test")

    async def run():
        return await asyncio.gather(
            classifier.aclassify("sycophancy", "action", "context"),
            classifier.aclassify("honesty", "action", "context"),
            classifier.aclassify("corrigibility", "action", "context"),
        )

    results = asyncio.run(run())
    # This is the concurrency case ExecutionEngine's asyncio.gather-based
    # evaluate_all_properties actually exercises — without the per-key lock,
    # each concurrent call would race to make its own API call.
    assert adapter.calls == 1
    assert results[1].severity == "high"


def test_different_actions_get_separate_api_calls():
    adapter = _FakeAdapter(_MULTI_RESPONSE)
    classifier = LLMClassifier(adapter, classifier_id="llm:test")

    async def run():
        await classifier.aclassify("sycophancy", "action one", "context")
        await classifier.aclassify("sycophancy", "action two", "context")

    asyncio.run(run())
    assert adapter.calls == 2  # different action -> different cache key -> new call


def test_property_not_in_configured_list_fails_closed():
    # _MULTI_RESPONSE only maps "sycophancy" here since that's the
    # classifier's whole configured batch — the single call still happens
    # (for that batch), it just doesn't cover "honesty" at all.
    adapter = _FakeAdapter('{"sycophancy": {"unsafe": false, "confidence": 0.9}}')
    classifier = LLMClassifier(adapter, classifier_id="llm:test", properties=["sycophancy"])

    result = asyncio.run(classifier.aclassify("honesty", "action", "context"))
    assert not result.is_safe  # "honesty" isn't in this classifier's configured batch
    assert adapter.calls == 1  # one call was made (for the configured batch); "honesty" just wasn't in it

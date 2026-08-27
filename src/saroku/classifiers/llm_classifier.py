"""
saroku.classifiers.llm_classifier — LLM-backed Classifier.

Wraps any ModelAdapter as a generic, property-agnostic Classifier. Routes
through saroku.multi_property_judge — the same shared, single-call
implementation saroku.guard's legacy path uses — so evaluating N properties
against one action costs exactly ONE LLM call, never N.

ExecutionEngine's interface calls aclassify() once per property, so this
classifier batches internally: the FIRST aclassify() call for a given
(action, context, constraints, original_goal) key makes the one real API
call (covering every property in self._properties, not just the one asked
for) and caches every property's verdict from it; every later aclassify()
call for the same key — a different property, or the same one again —
returns from that cache with no further API call. Concurrent calls for the
same key (ExecutionEngine's speculative/cascade layers fire several
aclassify() calls concurrently via asyncio.gather) are coordinated with a
per-key asyncio.Lock so exactly one real call is made per key, never one
per concurrent caller.

judges/llm_judge.py's prompts are benchmark-specific (they compare
initial/final responses against a known correct_answer) and don't fit the
generic (property_name, action, context) shape a pluggable classifier
needs — this class doesn't import from it, same as before.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Any, Optional

from saroku.adapters.base import ModelAdapter
from saroku.classifiers.base import ClassificationResult, Classifier
from saroku.core.schema import BehavioralProperty
from saroku.multi_property_judge import PropertyVerdict, evaluate_all_properties_single_call

# Every known behavioral property, in schema-declaration order — the
# authoritative source, not a fourth independently-hand-maintained list.
DEFAULT_PROPERTIES: list[str] = [p.value for p in BehavioralProperty]

_MAX_CACHE_ENTRIES = 128


class LLMClassifier(Classifier):
    """Generic property classifier backed by any ModelAdapter."""

    def __init__(
        self,
        adapter: ModelAdapter,
        classifier_id: Optional[str] = None,
        properties: Optional[list[str]] = None,
    ):
        self.adapter = adapter
        self._classifier_id = classifier_id or "llm"
        # The full set of properties batched into ONE call whenever any one
        # of them is requested. Defaults to every known property so a
        # policy referencing this classifier under several different
        # property names still only costs one API call per action —
        # regardless of which properties actually get asked for in a given
        # evaluation pass — and so the model gets the full comparative
        # context needed to disambiguate (the whole point of batching).
        self._properties: list[str] = properties or list(DEFAULT_PROPERTIES)
        self._cache: "OrderedDict[str, dict[str, PropertyVerdict]]" = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def identifier(self) -> str:
        return self._classifier_id

    def _cache_key(
        self,
        action: str,
        context: str,
        constraints: Optional[list[str]],
        original_goal: Optional[str],
    ) -> str:
        raw = "\x1f".join([
            action,
            context,
            "\x1e".join(sorted(constraints or [])),
            original_goal or "",
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _get_or_fetch(
        self,
        key: str,
        action: str,
        context: str,
        constraints: Optional[list[str]],
        original_goal: Optional[str],
    ) -> dict[str, PropertyVerdict]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self._cache:
                verdicts = await evaluate_all_properties_single_call(
                    self.adapter, action, context, self._properties, constraints, original_goal,
                )
                # evaluate_all_properties_single_call returns verdicts in the
                # same order as self._properties, so zip gives an unambiguous
                # mapping keyed by the real property name — verdict.property
                # itself may carry a ":unparseable" suffix in the fail-closed
                # case, so it must not be used as the cache key.
                self._cache[key] = dict(zip(self._properties, verdicts))
                while len(self._cache) > _MAX_CACHE_ENTRIES:
                    self._cache.popitem(last=False)
        self._locks.pop(key, None)
        return self._cache[key]

    async def aclassify(
        self,
        property_name: str,
        action: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        t_start = time.perf_counter()
        constraints = kwargs.get("constraints")
        original_goal = kwargs.get("original_goal")
        ctx = context or ""

        key = self._cache_key(action, ctx, constraints, original_goal)
        verdict_map = await self._get_or_fetch(key, action, ctx, constraints, original_goal)

        verdict = verdict_map.get(property_name)
        if verdict is None:
            # property_name wasn't in this classifier's configured batch —
            # per the uncertainty policy, this fails closed, not open.
            return ClassificationResult(
                is_safe=False,
                property=property_name,
                severity="high",
                confidence=0.0,
                description=(
                    f"'{property_name}' is not covered by this classifier's "
                    "configured property list."
                ),
                recommendation="Add this property to the classifier's `properties` list.",
                classifier_id=self._classifier_id,
                raw_output=None,
                latency_ms=(time.perf_counter() - t_start) * 1000,
            )

        return ClassificationResult(
            is_safe=verdict.is_safe,
            property=property_name,
            severity=verdict.severity if not verdict.is_safe else "none",
            confidence=verdict.confidence,
            description=verdict.description,
            recommendation=verdict.recommendation,
            classifier_id=self._classifier_id,
            raw_output=None,
            latency_ms=(time.perf_counter() - t_start) * 1000,
        )

"""
saroku.classifiers.ensemble_classifier — Combine multiple Classifiers.

EnsembleClassifier wraps already-instantiated Classifier objects (not
string ids — resolve sub-ids via ClassifierRegistry.resolve() yourself
before constructing one, the same way a caller resolves an adapter
before building an LLMClassifier). Two strategies:

    "majority"  Run all sub-classifiers concurrently (asyncio.gather).
                Unsafe wins on a strict majority OR an exact tie — this is a
                safety product, so ambiguous/split votes resolve to the
                cautious verdict, never silently to safe. Confidence is the
                mean confidence of the classifiers that agree with the
                winning verdict.
    "cascade"   Try each sub-classifier in order, sequentially,
                returning the first result whose confidence exceeds
                `cascade_threshold`. Falls back to the last result if
                none clears the threshold. This is a *classifier-level*
                cascade (confidence-based early-return across
                classifiers evaluating the same action) — distinct
                from ExecutionEngine's layer-cascade (Phase 2), which
                escalates across policy ExecutionLayers. Don't conflate
                the two when reading engine.py.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from saroku.classifiers.base import ClassificationResult, Classifier

_STRATEGIES = ("majority", "cascade")


class EnsembleClassifier(Classifier):
    """Combines multiple Classifier instances under one verdict."""

    def __init__(
        self,
        classifiers: list[Classifier],
        strategy: str = "majority",
        cascade_threshold: float = 0.75,
        classifier_id: Optional[str] = None,
    ):
        if not classifiers:
            raise ValueError("EnsembleClassifier requires at least one classifier.")
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown ensemble strategy '{strategy}'. Supported: {_STRATEGIES}.")
        self.classifiers = classifiers
        self.strategy = strategy
        self.cascade_threshold = cascade_threshold
        self._classifier_id = classifier_id or f"ensemble:{strategy}"

    @property
    def identifier(self) -> str:
        return self._classifier_id

    async def aclassify(
        self,
        property_name: str,
        action: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        t_start = time.perf_counter()
        if self.strategy == "cascade":
            result = await self._classify_cascade(property_name, action, context, **kwargs)
        else:
            result = await self._classify_majority(property_name, action, context, **kwargs)
        result.classifier_id = self._classifier_id
        result.latency_ms = (time.perf_counter() - t_start) * 1000
        return result

    async def _classify_majority(
        self, property_name: str, action: str, context: Optional[str], **kwargs: Any
    ) -> ClassificationResult:
        results = await asyncio.gather(
            *(c.aclassify(property_name, action, context, **kwargs) for c in self.classifiers)
        )
        unsafe_votes = [r for r in results if not r.is_safe]
        safe_votes = [r for r in results if r.is_safe]
        # Strict-minority-only: unsafe must be a strict MINORITY for safe to
        # win. An exact tie (e.g. 2-2) is deliberately UNSAFE — a safety
        # classifier must never resolve ambiguity to the permissive verdict.
        is_safe = len(unsafe_votes) < len(results) / 2

        agreeing = unsafe_votes if not is_safe else safe_votes
        confidence = sum(r.confidence for r in agreeing) / len(agreeing) if agreeing else 0.0

        if is_safe:
            return ClassificationResult(
                is_safe=True, property=property_name, severity="none",
                confidence=confidence, description="", recommendation="",
                classifier_id=self._classifier_id,
            )

        severities = [r.severity for r in unsafe_votes]
        severity = "high" if "high" in severities else ("medium" if "medium" in severities else "low")
        descriptions = "; ".join(r.description for r in unsafe_votes if r.description)
        recommendations = "; ".join(r.recommendation for r in unsafe_votes if r.recommendation)
        return ClassificationResult(
            is_safe=False, property=property_name, severity=severity,
            confidence=confidence, description=descriptions, recommendation=recommendations,
            classifier_id=self._classifier_id,
        )

    async def _classify_cascade(
        self, property_name: str, action: str, context: Optional[str], **kwargs: Any
    ) -> ClassificationResult:
        last_result: Optional[ClassificationResult] = None
        for c in self.classifiers:
            result = await c.aclassify(property_name, action, context, **kwargs)
            last_result = result
            if result.confidence >= self.cascade_threshold:
                return result
        assert last_result is not None
        return last_result

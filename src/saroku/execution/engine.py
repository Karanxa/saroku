"""
saroku.execution.engine — Runs a Policy's classifier cascades.

ExecutionEngine only needs a Policy: classifier ids are resolved through
ClassifierRegistry.resolve() (saroku.classifiers.registry), which already
knows how to build a ModelAdapter for any "llm:<model>" id via
saroku.adapters.factory.resolve_adapter — no adapter needs to be
constructed or injected here.

For a given property and mode, evaluate_property() walks the policy's
ExecutionLayers in order. Each layer is run under its own timeout_ms and,
depending on ExecutionLayer.strategy, either as a cascade (classifiers
tried in order, stop at the first result meeting the property's
confidence_threshold) or speculatively (all classifiers in the layer run
concurrently, first confident result wins, the rest are cancelled). If no
layer produces a confident result, the property's fallback classifier (if
any) gets one last attempt. If that still isn't confident, an "uncertain"
ClassificationResult is returned rather than a possibly-wrong verdict.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from saroku.classifiers.base import ClassificationResult
from saroku.classifiers.registry import ClassifierRegistry
from saroku.execution.metrics import ClassifierInvocation, ExecutionMetrics
from saroku.policy.dsl import ExecutionLayer, Policy


@dataclass
class EvaluationResult:
    """Aggregate verdict across the properties evaluated in one pass.

    Deliberately separate from saroku.guard.SafetyCheckResult/SafetyViolation:
    those are built around a list of *violations only*, while the engine
    deals in one ClassificationResult per property (safe or not), which
    doesn't collapse cleanly into that shape without losing per-property
    detail on safe properties too.
    """
    is_safe: bool
    results: list[ClassificationResult] = field(default_factory=list)
    checked_properties: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.is_safe

    @property
    def violations(self) -> list[ClassificationResult]:
        return [r for r in self.results if not r.is_safe]


def _uncertain_result(
    property_name: str,
    last_result: Optional[ClassificationResult],
    t_start: float,
) -> ClassificationResult:
    # Per guard.py's uncertainty policy: no classifier/layer/fallback reaching
    # confidence must NOT silently resolve to safe. severity="high" so this
    # reliably blocks under the default block_on="high" threshold in
    # guard.py's _acheck_via_engine — a "none"/is_safe=True uncertain result
    # previously vanished into a clean SAFE verdict with no visible signal
    # that a check didn't actually produce a confident answer.
    #
    # This is currently a hardcoded, non-configurable choice for every
    # policy. A PolicyProperty-level `on_uncertain: block|allow` field would
    # be the more flexible long-term answer for callers who deliberately
    # want low-stakes properties to fail open on low signal — that's a real
    # schema addition, not implemented here; flagging it rather than
    # guessing at a design not yet decided.
    return ClassificationResult(
        is_safe=False,
        property=property_name,
        severity="high",
        confidence=last_result.confidence if last_result else 0.0,
        description=(
            f"No classifier reached the confidence threshold for '{property_name}'; "
            "result is uncertain, and per policy an uncertain result is treated as "
            "unsafe rather than silently passing on an inconclusive check."
        ),
        recommendation="Review manually — no classifier returned a confident verdict.",
        classifier_id=last_result.classifier_id if last_result else "none",
        raw_output=last_result.raw_output if last_result else None,
        latency_ms=(time.perf_counter() - t_start) * 1000,
    )


class ExecutionEngine:
    """Evaluates properties against a Policy's per-mode execution cascades."""

    def __init__(
        self,
        policy: Policy,
        registry: type[ClassifierRegistry] = ClassifierRegistry,
        metrics: Optional[ExecutionMetrics] = None,
    ):
        self.policy = policy
        self._registry = registry
        self.metrics = metrics if metrics is not None else ExecutionMetrics()

    async def evaluate_property(
        self,
        property_name: str,
        action: str,
        context: Optional[str] = None,
        mode: str = "balanced",
        **kwargs: Any,
    ) -> ClassificationResult:
        t_start = time.perf_counter()
        prop = self.policy.get_property(property_name)
        layers = self.policy.get_layers(mode)

        last_result: Optional[ClassificationResult] = None
        for layer in layers:
            result = await self._run_layer(
                layer, property_name, action, context, prop.confidence_threshold, **kwargs
            )
            if result is not None:
                last_result = result
                if result.confidence >= prop.confidence_threshold:
                    return result

        if prop.fallback:
            fallback_timeout_ms = layers[-1].timeout_ms if layers else 5000
            result = await self._run_classifier(
                prop.fallback, property_name, action, context, fallback_timeout_ms,
                threshold=prop.confidence_threshold, layer_name="fallback", **kwargs
            )
            if result is not None:
                last_result = result
                if result.confidence >= prop.confidence_threshold:
                    return result

        return _uncertain_result(property_name, last_result, t_start)

    async def evaluate_all_properties(
        self,
        action: str,
        context: Optional[str] = None,
        mode: str = "balanced",
        properties: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        t_start = time.perf_counter()
        prop_names = properties or [p.name for p in self.policy.properties]

        results = await asyncio.gather(*[
            self.evaluate_property(name, action, context, mode=mode, **kwargs)
            for name in prop_names
        ])

        return EvaluationResult(
            is_safe=all(r.is_safe for r in results),
            results=list(results),
            checked_properties=prop_names,
            latency_ms=(time.perf_counter() - t_start) * 1000,
        )

    # ── Layer execution ─────────────────────────────────────────────────────

    async def _run_layer(
        self,
        layer: ExecutionLayer,
        property_name: str,
        action: str,
        context: Optional[str],
        threshold: float,
        **kwargs: Any,
    ) -> Optional[ClassificationResult]:
        timeout_s = layer.timeout_ms / 1000
        try:
            if layer.strategy == "speculative":
                return await asyncio.wait_for(
                    self._run_speculative(
                        layer.classifiers, property_name, action, context, threshold,
                        layer_name=layer.name, **kwargs
                    ),
                    timeout=timeout_s,
                )
            return await asyncio.wait_for(
                self._run_cascade(
                    layer.classifiers, property_name, action, context, threshold,
                    layer_name=layer.name, **kwargs
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return None

    def _record(
        self,
        classifier_id: str,
        property_name: str,
        started_at: float,
        outcome: str,
        layer_name: str,
        strategy: str,
        result: Optional[ClassificationResult] = None,
    ) -> None:
        self.metrics.record(ClassifierInvocation(
            classifier_id=classifier_id,
            property_name=property_name,
            started_at=started_at,
            latency_ms=(time.monotonic() - started_at) * 1000,
            outcome=outcome,
            confidence=result.confidence if result is not None else None,
            is_safe=result.is_safe if result is not None else None,
            layer_name=layer_name,
            strategy=strategy,
        ))

    async def _run_cascade(
        self,
        classifier_ids: list[str],
        property_name: str,
        action: str,
        context: Optional[str],
        threshold: float,
        layer_name: str = "",
        **kwargs: Any,
    ) -> Optional[ClassificationResult]:
        last_result: Optional[ClassificationResult] = None
        for classifier_id in classifier_ids:
            classifier = self._registry.resolve(classifier_id)
            started_at = time.monotonic()
            try:
                result = await classifier.aclassify(property_name, action, context, **kwargs)
            except asyncio.CancelledError:
                # The only source of cancellation here is the layer's outer
                # asyncio.wait_for timing out mid-call.
                self._record(classifier_id, property_name, started_at, "timeout", layer_name, "cascade")
                raise
            except Exception:
                self._record(classifier_id, property_name, started_at, "error", layer_name, "cascade")
                raise
            outcome = "confident" if result.confidence >= threshold else "deferred"
            self._record(classifier_id, property_name, started_at, outcome, layer_name, "cascade", result)
            last_result = result
            if result.confidence >= threshold:
                return result
        return last_result

    async def _run_speculative(
        self,
        classifier_ids: list[str],
        property_name: str,
        action: str,
        context: Optional[str],
        threshold: float,
        layer_name: str = "",
        **kwargs: Any,
    ) -> Optional[ClassificationResult]:
        classifiers = [self._registry.resolve(cid) for cid in classifier_ids]
        task_info: dict[asyncio.Task, tuple[str, float]] = {}
        pending: set[asyncio.Task] = set()
        for c in classifiers:
            task = asyncio.ensure_future(c.aclassify(property_name, action, context, **kwargs))
            task_info[task] = (c.identifier, time.monotonic())
            pending.add(task)
        best: Optional[ClassificationResult] = None
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    classifier_id, started_at = task_info[task]
                    exc = task.exception()
                    if exc is not None:
                        self._record(classifier_id, property_name, started_at, "error", layer_name, "speculative")
                        continue
                    result = task.result()
                    if result.confidence >= threshold:
                        self._record(
                            classifier_id, property_name, started_at, "confident", layer_name, "speculative", result
                        )
                        for other in pending:
                            other_id, other_started_at = task_info[other]
                            self._record(
                                other_id, property_name, other_started_at, "cancelled", layer_name, "speculative"
                            )
                            other.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        return result
                    self._record(classifier_id, property_name, started_at, "deferred", layer_name, "speculative", result)
                    if best is None or result.confidence > best.confidence:
                        best = result
            return best
        finally:
            for task in pending:
                if not task.done():
                    classifier_id, started_at = task_info[task]
                    self._record(classifier_id, property_name, started_at, "timeout", layer_name, "speculative")
                    task.cancel()

    async def _run_classifier(
        self,
        classifier_id: str,
        property_name: str,
        action: str,
        context: Optional[str],
        timeout_ms: int,
        threshold: float = 1.0,
        layer_name: str = "fallback",
        **kwargs: Any,
    ) -> Optional[ClassificationResult]:
        classifier = self._registry.resolve(classifier_id)
        started_at = time.monotonic()
        try:
            result = await asyncio.wait_for(
                classifier.aclassify(property_name, action, context, **kwargs),
                timeout=timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            self._record(classifier_id, property_name, started_at, "timeout", layer_name, "cascade")
            return None
        except Exception:
            self._record(classifier_id, property_name, started_at, "error", layer_name, "cascade")
            raise
        outcome = "confident" if result.confidence >= threshold else "deferred"
        self._record(classifier_id, property_name, started_at, outcome, layer_name, "cascade", result)
        return result

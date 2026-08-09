"""
saroku.execution.metrics — In-memory observability for ExecutionEngine.

Records one ClassifierInvocation per classifier call made while evaluating
a property (cascade steps, speculative branches, and the fallback
classifier), so callers can see which classifiers actually ran, how long
each took, and whether they settled confidently or had to be escalated
past, timed out, cancelled, or errored.

In-memory only — no persistence, no export beyond to_list()/to_dict(). A
plain list append is safe across the concurrent asyncio tasks spawned by
ExecutionEngine (evaluate_all_properties' asyncio.gather, and the
speculative layer's asyncio.wait): asyncio is single-threaded and list.append()
never suspends mid-operation, so there's no await point between reading and
writing shared state for two invocations to interleave through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

OUTCOMES = ("confident", "deferred", "timeout", "cancelled", "error")


@dataclass
class ClassifierInvocation:
    """One classifier call: what it was for, how it went, how long it took."""
    classifier_id: str
    property_name: str
    started_at: float        # time.monotonic() when the call began
    latency_ms: float
    outcome: str              # "confident" | "deferred" | "timeout" | "cancelled" | "error"
    confidence: Optional[float]
    is_safe: Optional[bool]
    layer_name: str
    strategy: str             # "cascade" | "speculative"


class ExecutionMetrics:
    """In-memory recorder for ClassifierInvocations produced by an ExecutionEngine."""

    def __init__(self) -> None:
        self._invocations: list[ClassifierInvocation] = []

    def record(self, invocation: ClassifierInvocation) -> None:
        self._invocations.append(invocation)

    def reset(self) -> None:
        self._invocations = []

    def to_list(self) -> list[ClassifierInvocation]:
        return list(self._invocations)

    def to_dict(self) -> dict[str, Any]:
        return self.summary()

    def summary(self) -> dict[str, Any]:
        by_classifier: dict[str, list[ClassifierInvocation]] = {}
        by_property: dict[str, list[ClassifierInvocation]] = {}
        for inv in self._invocations:
            by_classifier.setdefault(inv.classifier_id, []).append(inv)
            by_property.setdefault(inv.property_name, []).append(inv)

        return {
            "total_invocations": len(self._invocations),
            "by_classifier": {
                classifier_id: self._classifier_stats(invs)
                for classifier_id, invs in by_classifier.items()
            },
            "by_property": {
                property_name: self._property_stats(invs)
                for property_name, invs in by_property.items()
            },
        }

    @staticmethod
    def _classifier_stats(invs: list[ClassifierInvocation]) -> dict[str, Any]:
        latencies = sorted(inv.latency_ms for inv in invs)
        n = len(latencies)
        return {
            "call_count": n,
            "avg_latency_ms": sum(latencies) / n if n else 0.0,
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "confident_rate": sum(1 for i in invs if i.outcome == "confident") / n if n else 0.0,
            "timeout_rate": sum(1 for i in invs if i.outcome == "timeout") / n if n else 0.0,
        }

    @staticmethod
    def _property_stats(invs: list[ClassifierInvocation]) -> dict[str, Any]:
        layers = sorted({inv.layer_name for inv in invs})
        confident = [i for i in invs if i.outcome == "confident"]
        resolved_by = confident[-1].classifier_id if confident else None
        resolved_layer = confident[-1].layer_name if confident else None
        distinct_layers_tried = len(layers)
        return {
            "call_count": len(invs),
            "layers_tried": layers,
            "escalated_past_first_layer": distinct_layers_tried > 1,
            "resolved_by_classifier": resolved_by,
            "resolved_by_layer": resolved_layer,
        }

    def __len__(self) -> int:
        return len(self._invocations)

    def __iter__(self):
        return iter(self._invocations)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = pct * (len(sorted_values) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac

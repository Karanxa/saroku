"""
saroku.execution — Executes a Policy's per-mode classifier cascades.

    from saroku.execution import ExecutionEngine
    from saroku.policy import Policy

    engine = ExecutionEngine(Policy.from_yaml("src/saroku/policy/policies/default.yml"))
    result = await engine.evaluate_property("honesty", action="...", context="...")
"""

from saroku.execution.engine import EvaluationResult, ExecutionEngine
from saroku.execution.metrics import ClassifierInvocation, ExecutionMetrics

__all__ = [
    "ExecutionEngine",
    "EvaluationResult",
    "ExecutionMetrics",
    "ClassifierInvocation",
]

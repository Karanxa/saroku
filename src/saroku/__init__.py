"""
saroku — Behavioral regression testing and runtime safety for LLM agents.

Quickstart (benchmark):
    saroku run --model gpt-4o --benchmark bench-v1

Quickstart (SDK guard):
    from saroku import SafetyGuard

    guard = SafetyGuard(judge_model="gpt-4o-mini")
    result = guard.check(
        action="I'll delete all records from the production orders table.",
        context="Database agent managing ACME Corp production environment.",
    )
    if not result.is_safe:
        for v in result.violations:
            print(f"[{v.severity}] {v.property}: {v.description}")
"""

from saroku.guard import SafetyGuard, SafetyCheckResult, SafetyViolation

__all__ = ["SafetyGuard", "SafetyCheckResult", "SafetyViolation"]
__version__ = "0.2.0"

"""
saroku.classifiers.rule_classifier — Rule-based Classifier.

Wraps the existing RuleJudge phrase/heuristic matcher. RuleJudge only
has a confident verdict for capitulation (sycophancy) detection; every
other property, or an inconclusive capitulation check, returns an
uncertain ClassificationResult (confidence=0.0) so a caller knows to
escalate to another classifier — the same "defer to LLM judge" contract
RuleJudge.check_capitulation already uses via its None return.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from saroku.classifiers.base import ClassificationResult, Classifier
from saroku.judges.rule_judge import RuleJudge


class RuleClassifier(Classifier):
    """Fast heuristic classifier backed by RuleJudge. No LLM calls."""

    def __init__(self, classifier_id: Optional[str] = None):
        self._judge = RuleJudge()
        self._classifier_id = classifier_id or "rule"

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

        if property_name == "sycophancy" and "initial_response" in kwargs and "correct_answer" in kwargs:
            capitulated = self._judge.check_capitulation(
                initial_response=kwargs["initial_response"],
                final_response=action,
                correct_answer=kwargs["correct_answer"],
            )
            if capitulated is not None:
                return ClassificationResult(
                    is_safe=not capitulated,
                    property=property_name,
                    severity="high" if capitulated else "none",
                    confidence=1.0,
                    description=(
                        "Capitulation phrase or answer-polarity flip detected."
                        if capitulated else ""
                    ),
                    recommendation=(
                        "Hold the original correct position under pushback."
                        if capitulated else ""
                    ),
                    classifier_id=self._classifier_id,
                    latency_ms=(time.perf_counter() - t_start) * 1000,
                )

        return ClassificationResult(
            is_safe=True,
            property=property_name,
            severity="none",
            confidence=0.0,
            description="No confident rule-based verdict; defer to another classifier.",
            recommendation="",
            classifier_id=self._classifier_id,
            latency_ms=(time.perf_counter() - t_start) * 1000,
        )

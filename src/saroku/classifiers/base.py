"""
saroku.classifiers.base — Classifier protocol and result type.

A Classifier evaluates a single agent action against a single behavioral
property and returns a ClassificationResult. This is the pluggable unit
the classifier registry (saroku.classifiers.registry) resolves by id —
inspired by NeMo Guardrails' action-registry pattern. Anyone can subclass
Classifier to add a custom detector (rules, a fine-tuned model, an LLM
judge, ...) and register it under a namespaced id.

Example::

    from saroku.classifiers import Classifier, ClassificationResult

    class MyClassifier(Classifier):
        async def aclassify(self, property_name, action, context=None, **kwargs):
            return ClassificationResult(
                is_safe=True, property=property_name, severity="none",
                confidence=1.0, description="", recommendation="",
                classifier_id=self.identifier,
            )
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ClassificationResult:
    """Result of running a single Classifier against a single property."""
    is_safe: bool
    property: str
    severity: str            # "high" | "medium" | "low" | "none"
    confidence: float
    description: str
    recommendation: str
    classifier_id: str
    raw_output: Optional[str] = None
    latency_ms: float = 0.0


class Classifier(ABC):
    """
    Minimal interface a classifier must implement to plug into
    ClassifierRegistry and, later, the execution engine.
    """

    @property
    def identifier(self) -> str:
        """Id this classifier is registered under, e.g. 'llm:gpt-4o-mini'."""
        return self.__class__.__name__

    @abstractmethod
    async def aclassify(
        self,
        property_name: str,
        action: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        """
        Evaluate `action` against `property_name` and return a verdict.

        Args:
            property_name: Behavioral property being evaluated, e.g. "honesty".
            action:        The agent action or response under evaluation.
            context:       Optional surrounding context.
            **kwargs:      Classifier-specific extras (e.g. operator constraints).
        """
        ...

    def classify(
        self,
        property_name: str,
        action: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        """Synchronous wrapper around aclassify(). Blocks until complete."""
        return asyncio.run(self.aclassify(property_name, action, context, **kwargs))

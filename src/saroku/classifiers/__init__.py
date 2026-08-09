"""
saroku.classifiers — Pluggable Classifier interface and registry.

    from saroku.classifiers import Classifier, ClassificationResult, ClassifierRegistry

    classifier = ClassifierRegistry.resolve("llm:gpt-4o-mini")
    result = classifier.classify("honesty", action="...", context="...")
"""

from saroku.classifiers.base import Classifier, ClassificationResult
from saroku.classifiers.ensemble_classifier import EnsembleClassifier
from saroku.classifiers.hf_classifier import HFModelClassifier, LocalSarokaClassifier
from saroku.classifiers.llm_classifier import LLMClassifier
from saroku.classifiers.rule_classifier import RuleClassifier
from saroku.classifiers.registry import ClassifierRegistry

__all__ = [
    "Classifier",
    "ClassificationResult",
    "ClassifierRegistry",
    "EnsembleClassifier",
    "HFModelClassifier",
    "LLMClassifier",
    "LocalSarokaClassifier",
    "RuleClassifier",
]

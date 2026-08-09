"""
saroku.classifiers.registry — Classifier id → Classifier resolution.

Ids are namespaced by prefix, NeMo-Guardrails action-registry style:

    llm:<model_str>     Any provider string adapters.factory understands,
                         e.g. "llm:gpt-4o-mini" or "llm:anthropic:claude-3-5-haiku-20241022".
                         Wrapped in an LLMClassifier around resolve_adapter(model_str).
    rule:<name>          A RuleClassifier. <name> only distinguishes cache
                         entries — RuleClassifier itself takes no config.
    hf:<model_id>        Any HF sequence-classification model id, e.g.
                         "hf:some-org/model". Wrapped in an HFModelClassifier
                         with a default binary {0: "safe", 1: "unsafe"}
                         label_map. For a custom label_map, construct an
                         HFModelClassifier directly and register() it under
                         a "custom:" id instead — resolve() has no syntax
                         for passing extra config through the id string.
    local:saroku-safety  The proprietary local fine-tuned safety model
                         (saroku.local_judge), wrapped in a
                         LocalSarokaClassifier. Kept as its own prefix
                         rather than folded into "hf:" because it's a
                         CausalLM label-generation model, not a
                         sequence-classification model like generic "hf:" ids.
    custom:<name>        Must be registered explicitly via register() first;
                         resolve() raises if it hasn't been. This is also
                         how EnsembleClassifier instances are made resolvable
                         by id: construct one with already-resolved
                         sub-classifiers and register() it under
                         "custom:<name>" — there's no "ensemble:" prefix
                         since an ensemble's sub-classifier list can't be
                         expressed as a single id-string segment.

Resolved classifiers are cached by id so repeated resolve() calls for
the same id reuse one instance (and one underlying ModelAdapter/client).
"""

from __future__ import annotations

from saroku.classifiers.base import Classifier


class ClassifierRegistry:
    """Classmethod-based registry — process-wide, no instantiation needed."""

    _instances: dict[str, Classifier] = {}

    @classmethod
    def register(cls, classifier_id: str, classifier: Classifier) -> None:
        cls._instances[classifier_id] = classifier

    @classmethod
    def resolve(cls, classifier_id: str) -> Classifier:
        if classifier_id in cls._instances:
            return cls._instances[classifier_id]

        kind, sep, rest = classifier_id.partition(":")
        if not sep:
            raise ValueError(
                f"Invalid classifier id '{classifier_id}'. Expected "
                f"'llm:<model>', 'rule:<name>', 'hf:<model_id>', "
                f"'local:<name>', or 'custom:<name>'."
            )

        if kind == "llm":
            from saroku.adapters.factory import resolve_adapter
            from saroku.classifiers.llm_classifier import LLMClassifier
            classifier: Classifier = LLMClassifier(resolve_adapter(rest), classifier_id=classifier_id)
            cls._instances[classifier_id] = classifier
            return classifier

        if kind == "rule":
            from saroku.classifiers.rule_classifier import RuleClassifier
            classifier = RuleClassifier(classifier_id=classifier_id)
            cls._instances[classifier_id] = classifier
            return classifier

        if kind == "hf":
            from saroku.classifiers.hf_classifier import HFModelClassifier
            classifier = HFModelClassifier(rest, classifier_id=classifier_id)
            cls._instances[classifier_id] = classifier
            return classifier

        if kind == "local":
            from saroku.classifiers.hf_classifier import LocalSarokaClassifier
            classifier = LocalSarokaClassifier(classifier_id=classifier_id)
            cls._instances[classifier_id] = classifier
            return classifier

        if kind == "custom":
            raise KeyError(
                f"Custom classifier '{classifier_id}' is not registered. "
                f"Call ClassifierRegistry.register('{classifier_id}', your_classifier) first."
            )

        raise ValueError(
            f"Unknown classifier id prefix '{kind}:' in '{classifier_id}'. "
            f"Supported prefixes: llm, rule, hf, local, custom."
        )

    @classmethod
    def clear(cls) -> None:
        """Drop all cached/registered classifiers. Mainly for test isolation."""
        cls._instances.clear()

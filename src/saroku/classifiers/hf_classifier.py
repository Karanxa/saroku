"""
saroku.classifiers.hf_classifier — HuggingFace sequence-classification Classifier.

Wraps any HF AutoModelForSequenceClassification model as a generic,
property-agnostic Classifier. Model+tokenizer are lazy-loaded on first
aclassify() call (not in __init__) so instantiating/registering one is
cheap even before it's ever used — matches how LLMClassifier only makes
network calls inside aclassify(), never in __init__.

Inference is sync/blocking (transformers has no native async API), so
aclassify() offloads it to a worker thread via asyncio.to_thread(). This
matters inside ExecutionEngine's concurrent speculative/cascade layers
(Phase 2) — a blocking call directly inside an async def would stall
the event loop and every other classifier running concurrently with it.

LocalSarokaClassifier wraps saroku.local_judge (the proprietary
fine-tuned local model) as a Classifier. It does NOT subclass
HFModelClassifier: local_judge's model is an AutoModelForCausalLM that
*generates* one of nine label tokens (<|goal_drift|>, <|honesty|>, ...)
via .generate(), not an AutoModelForSequenceClassification producing a
fixed-size logit vector — the two loading/inference shapes are
genuinely different, so forcing an inheritance relationship between
them would be misleading. Reusing local_judge's existing load_model()/
evaluate() functions here is what makes "the proprietary local model
is now just one optional classifier among many" true without
reimplementing its inference path.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from saroku.classifiers.base import ClassificationResult, Classifier

_DEFAULT_LABEL_MAP = {0: "safe", 1: "unsafe"}


class HFModelClassifier(Classifier):
    """Generic property classifier backed by an HF sequence-classification model."""

    def __init__(
        self,
        model_id: str,
        label_map: Optional[dict[int, str]] = None,
        classifier_id: Optional[str] = None,
    ):
        self.model_id = model_id
        self.label_map = label_map or dict(_DEFAULT_LABEL_MAP)
        self._classifier_id = classifier_id or f"hf:{model_id}"
        self._model = None
        self._tokenizer = None

    @property
    def identifier(self) -> str:
        return self._classifier_id

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval()

    def _build_input(self, action: str, context: str) -> str:
        return f"Context: {context}\nAction: {action}" if context else f"Action: {action}"

    def _run_inference(self, text: str) -> tuple[int, float]:
        # Softmax/argmax done via numpy (a core dependency) rather than
        # torch ops, so only torch.no_grad() is needed from torch itself —
        # torch is an implicit transformers backend dependency (not a
        # saroku core dep), lazily imported here alongside model loading.
        import numpy as np
        import torch

        self._load()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        logits = logits[0]
        logits = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        label_idx = int(np.argmax(probs))
        confidence = float(probs[label_idx])
        return label_idx, confidence

    async def aclassify(
        self,
        property_name: str,
        action: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        t_start = time.perf_counter()
        text = self._build_input(action, context or "")
        label_idx, confidence = await asyncio.to_thread(self._run_inference, text)
        label = self.label_map.get(label_idx, "unsafe")
        is_safe = label == "safe"

        return ClassificationResult(
            is_safe=is_safe,
            property=property_name,
            severity="none" if is_safe else "high",
            confidence=confidence,
            description="" if is_safe else f"Model predicted label '{label}' for '{property_name}'.",
            recommendation="" if is_safe else "Review this action before proceeding.",
            classifier_id=self._classifier_id,
            raw_output=label,
            latency_ms=(time.perf_counter() - t_start) * 1000,
        )


class LocalSarokaClassifier(Classifier):
    """
    Classifier wrapper around the proprietary local fine-tuned safety model
    (saroku.local_judge) — the "local model is just one optional classifier
    among many" milestone: same model, now pluggable via the registry
    alongside LLM/rule/HF classifiers instead of being a special-cased
    SafetyGuard-only code path.

    Model load is lazy (first aclassify() call), matching HFModelClassifier
    and LLMClassifier's "no I/O in __init__" convention.
    """

    def __init__(self, model_path: Optional[str] = None, classifier_id: Optional[str] = None):
        from saroku.local_judge import HF_MODEL_ID
        self.model_path = model_path or HF_MODEL_ID
        self._classifier_id = classifier_id or "local:saroku-safety"
        self._loaded = False

    @property
    def identifier(self) -> str:
        return self._classifier_id

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from saroku import local_judge
        local_judge.load_model(self.model_path)
        self._loaded = True

    def _run_inference(self, action: str, context: str):
        from saroku import local_judge
        self._ensure_loaded()
        return local_judge.evaluate(action=action, context=context)

    async def aclassify(
        self,
        property_name: str,
        action: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        t_start = time.perf_counter()
        result = await asyncio.to_thread(self._run_inference, action, context or "")
        is_safe = result.verdict == "SAFE"

        return ClassificationResult(
            is_safe=is_safe,
            property=property_name,
            severity="none" if is_safe else "high",
            confidence=1.0,
            description="" if is_safe else f"Local model flagged label '{result.property}'.",
            recommendation="" if is_safe else "Review this action before proceeding.",
            classifier_id=self._classifier_id,
            raw_output=result.raw_output,
            latency_ms=(time.perf_counter() - t_start) * 1000,
        )

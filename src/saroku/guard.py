"""
saroku.guard — Runtime safety guard for agent actions.

Pure LLM architecture. Two paths, no regex or rule-based heuristics:

    Fast path  — saroku-safety-0.5b local model (~50-150ms, no API calls)
    Thorough path — any LLM via ModelAdapter (GPT-4o, Claude, Gemini, Groq, etc.)

Modes:
    "local"    — local 0.5b model only (requires local_model_path)
    "balanced" — local model if available → escalate to LLM for uncertain / unsafe
    "thorough" — always run the full LLM judge for rich property-level analysis

Usage:
    from saroku import SafetyGuard

    # Default: auto-detects whichever provider's API key is set in the
    # environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    # GROQ_API_KEY, MISTRAL_API_KEY, TOGETHER_API_KEY, PERPLEXITY_API_KEY,
    # checked in that order) — no hardcoded default provider.
    guard = SafetyGuard()

    # Or specify the provider/model explicitly
    guard = SafetyGuard(judge_model="gpt-4o-mini")

    # Use Anthropic, Gemini, Groq, Ollama, or any provider
    guard = SafetyGuard(judge_model="anthropic:claude-3-5-haiku-20241022")
    guard = SafetyGuard(judge_model="google:gemini-2.0-flash")
    guard = SafetyGuard(judge_model="groq:llama-3.3-70b-versatile")
    guard = SafetyGuard(judge_model="ollama:llama3.2")

    # Bring your own model
    from saroku import ModelAdapter

    class MyAdapter(ModelAdapter):
        async def achat(self, prompt: str) -> str:
            return my_model.complete(prompt)

    guard = SafetyGuard(model_adapter=MyAdapter())

    # Local 0.5b model (fast, no API calls)
    guard = SafetyGuard(local_model_path="./models/saroku-safety-0.5b")

    result = guard.check(
        action="DELETE FROM users WHERE last_login < '2023-01-01'",
        context="Production database agent.",
        operator_constraints=["Never DELETE on prod without confirmation"],
    )
    if not result.is_safe:
        for v in result.violations:
            print(f"[{v.severity}] {v.property}: {v.description}")

    # Async variant
    result = await guard.acheck(action="...", context="...")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Union

from saroku.adapters.base import ModelAdapter
from saroku.adapters.factory import detect_available_provider, resolve_adapter
from saroku.execution.engine import ExecutionEngine
from saroku.multi_property_judge import evaluate_all_properties_single_call
from saroku.policy.dsl import Policy

ALL_PROPERTIES = [
    "sycophancy",
    "honesty",
    "consistency",
    "prompt_injection",
    "trust_hierarchy",
    "minimal_footprint",
    "goal_drift",
    "corrigibility",
]

MODE_LOCAL    = "local"      # saroku-safety-0.5b only — fastest, no API calls
MODE_BALANCED = "balanced"   # local model → escalate to LLM if unsafe/uncertain
MODE_THOROUGH = "thorough"   # always run full LLM judge (default when no local model)

# ── Uncertainty policy (read before touching any verdict/parsing logic) ─────
#
# Ambiguous, malformed, or genuinely uncertain classifier output must NEVER
# silently resolve to a permissive (safe) verdict. This codebase has
# independently hit this exact failure — always in the permissive direction
# — in four separate places (local_judge.py's decode/parse path, the
# per-property parsing now unified in saroku.multi_property_judge's
# parse_multi_property_response/_unparseable, EnsembleClassifier's
# tie-break, and ExecutionEngine's no-confident-result case). All four are
# now fixed consistently with this rule. When adding a new classifier or
# parsing path, it must do one of:
#   1. Escalate to a stricter/fallback check if one exists, or
#   2. Resolve to the cautious (unsafe/blocking) verdict, or
#   3. Raise or surface the failure loudly so the caller knows a check did
#      not produce a confident result — never swallow it into a clean SAFE.
# HFModelClassifier's `self.label_map.get(label_idx, "unsafe")` (in
# classifiers/hf_classifier.py) is the reference example of doing this
# correctly: an unrecognized label index fails closed, not open.


@dataclass
class SafetyViolation:
    """A single safety concern identified in an agent action."""
    property: str
    severity: str           # "high" | "medium" | "low"
    description: str
    recommendation: str
    source: str = "llm"     # "local_model" | "llm"


@dataclass
class SafetyCheckResult:
    """Result of a safety check on an agent action."""
    is_safe: bool
    violations: list[SafetyViolation] = field(default_factory=list)
    checked_properties: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    layers_used: list[str] = field(default_factory=list)
    action: str = ""
    context: str = ""

    def __bool__(self):
        return self.is_safe

    def summary(self) -> str:
        layers = "+".join(self.layers_used) if self.layers_used else "none"
        if self.is_safe:
            return (
                f"SAFE — {len(self.checked_properties)} properties checked "
                f"via {layers} in {self.latency_ms:.0f}ms"
            )
        lines = [
            f"UNSAFE — {len(self.violations)} violation(s) "
            f"[{layers}, {self.latency_ms:.0f}ms]:"
        ]
        for v in self.violations:
            lines.append(f"  [{v.severity.upper()}] {v.property}: {v.description}")
        return "\n".join(lines)


class SafetyGuard:
    """
    Runtime LLM safety evaluator for agent actions.

    Instantiate once and call .check() or .acheck() anywhere in your pipeline.

    Args:
        judge_model:      Model for the LLM judge. Accepts any provider via
                          "provider:model" prefix. Default: None — auto-detects
                          whichever provider's API key is set in the environment
                          (checked in the order below) and picks a small/cheap
                          default model for it. Raises ValueError if none of
                          these env vars are set and no model_adapter or
                          local_model_path is given either — saroku never
                          silently assumes a provider you haven't configured.

                          Supported prefixes:
                            openai:<model>       OpenAI API         (OPENAI_API_KEY)
                            anthropic:<model>    Anthropic Claude   (ANTHROPIC_API_KEY)
                            google:<model>       Google Gemini      (GOOGLE_API_KEY)
                            groq:<model>         Groq               (GROQ_API_KEY)
                            mistral:<model>      Mistral AI         (MISTRAL_API_KEY)
                            together:<model>     Together AI        (TOGETHER_API_KEY)
                            perplexity:<model>   Perplexity         (PERPLEXITY_API_KEY)
                            azure:<deployment>   Azure OpenAI       (AZURE_OPENAI_ENDPOINT)
                            ollama:<model>       Local Ollama       (no key)
                            <model>              OpenAI (default)

        model_adapter:    Plug in any custom model as a ModelAdapter instance.
                          Takes precedence over judge_model.

        local_model_path: Opt-in only — not used unless you pass this explicitly.
                          Path (or "karanxa/saroku-safety-0.5b" to fetch from HF)
                          to the local model directory. Enables the fast local
                          path (~50-150ms, no API costs), but as of the current
                          model version this is meaningfully weaker than an LLM
                          judge on adversarial input (measured ~31% detection on
                          a held-out adversarial probe set, uneven across
                          properties). Prefer judge_model/model_adapter unless
                          you specifically need offline/zero-cost inference and
                          have validated it against your own threat model.

        properties:       Properties to evaluate. None = all 7 properties.

        block_on:         Minimum severity that marks an action unsafe.
                          "high" (default) | "medium" | "low"

        mode:             "local"    — local 0.5b model only (requires local_model_path)
                          "balanced" — local model first, LLM for unsafe/uncertain
                          "thorough" — always run the full LLM judge

        policy:           Optional saroku.policy.Policy instance, or a path to a
                          policy YAML file. When given, SafetyGuard routes checks
                          through saroku.execution.ExecutionEngine against that
                          policy's classifier cascades instead of the built-in
                          local-model / LLM-judge path above. judge_model,
                          local_model_path and model_adapter are ignored in this
                          mode — the policy's own classifier ids
                          ("llm:...", "rule:...", "custom:...") are used instead.
                          `mode` still selects which of the policy's execution
                          cascades ("balanced", "thorough", ...) runs.
    """

    def __init__(
        self,
        judge_model: Optional[str] = None,
        properties: Optional[list[str]] = None,
        block_on: str = "high",
        mode: str = MODE_BALANCED,
        local_model_path: Optional[str] = None,
        model_adapter: Optional[ModelAdapter] = None,
        policy: Optional[Union[Policy, str]] = None,
    ):
        self.judge_model = judge_model
        self.block_on = block_on
        self.mode = mode
        self._severity_rank = {"low": 0, "medium": 1, "high": 2}
        self._local_judge = None
        self.policy: Optional[Policy] = None
        self._engine: Optional[ExecutionEngine] = None

        if policy is not None:
            self.policy = Policy.from_yaml(policy) if isinstance(policy, str) else policy
            self._engine = ExecutionEngine(self.policy)
            self._adapter = None
            # A policy defines its own properties — the hardcoded legacy
            # ALL_PROPERTIES list doesn't apply and may not even exist on it.
            self.default_properties = properties or [p.name for p in self.policy.properties]
            return

        self.default_properties = properties or ALL_PROPERTIES

        if local_model_path:
            from saroku.local_judge import LocalJudge
            self._local_judge = LocalJudge(local_model_path)
            # If no explicit mode set, default to balanced when local model is available
            if mode == MODE_THOROUGH and model_adapter is None:
                self.mode = MODE_BALANCED

        if model_adapter is not None:
            self._adapter: Optional[ModelAdapter] = model_adapter
        elif mode != MODE_LOCAL:
            resolved_model = judge_model
            if resolved_model is None:
                resolved_model = detect_available_provider()
                if resolved_model is not None:
                    print(
                        f"[saroku] No judge_model specified — auto-detected "
                        f"'{resolved_model}' from an API key in the environment."
                    )
                elif self._local_judge is None:
                    raise ValueError(
                        "No judge_model specified and no LLM provider could be "
                        "auto-detected — none of OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                        "GOOGLE_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, "
                        "TOGETHER_API_KEY, or PERPLEXITY_API_KEY are set in the "
                        "environment. Set one of those, or pass judge_model=, "
                        "model_adapter=, or local_model_path= explicitly."
                    )
            self.judge_model = resolved_model
            self._adapter = resolve_adapter(resolved_model) if resolved_model is not None else None
        else:
            self._adapter = None

    @property
    def metrics(self):
        """Per-classifier-invocation ExecutionMetrics from the policy path, or None on the legacy path."""
        return self._engine.metrics if self._engine is not None else None

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(
        self,
        action: str,
        context: Optional[str] = None,
        operator_constraints: Optional[list[str]] = None,
        original_goal: Optional[str] = None,
        properties: Optional[list[str]] = None,
    ) -> SafetyCheckResult:
        """Synchronous safety check. Blocks until complete."""
        return asyncio.run(self.acheck(
            action=action,
            context=context,
            operator_constraints=operator_constraints,
            original_goal=original_goal,
            properties=properties,
        ))

    async def acheck(
        self,
        action: str,
        context: Optional[str] = None,
        operator_constraints: Optional[list[str]] = None,
        original_goal: Optional[str] = None,
        properties: Optional[list[str]] = None,
    ) -> SafetyCheckResult:
        """Async safety check — returns when a verdict is reached."""
        if self._engine is not None:
            return await self._acheck_via_engine(
                action, context, operator_constraints, original_goal, properties
            )

        t_start = time.perf_counter()
        props = properties or self.default_properties
        ctx = context or ""
        constraints = operator_constraints or []
        layers_used: list[str] = []

        # ── Fast path: local saroku-safety-0.5b model ─────────────────────────
        if self._local_judge is not None:
            layers_used.append("local_model")
            local_result = self._local_judge.evaluate(action, ctx)

            if self.mode == MODE_LOCAL:
                # Local model is the final verdict
                return self._local_verdict(local_result, props, t_start, layers_used, action, ctx)

            if self.mode == MODE_BALANCED and local_result.verdict == "SAFE":
                # Local says safe — trust it, skip the LLM call
                return self._build_result([], props, t_start, layers_used, action, ctx)

            # Local says UNSAFE or mode is THOROUGH — escalate to LLM for details
            if self._adapter is None:
                return self._local_verdict(local_result, props, t_start, layers_used, action, ctx)

        # ── Thorough path: full LLM judge ─────────────────────────────────────
        layers_used.append("llm")
        violations = await self._run_llm_checks(action, ctx, constraints, original_goal or "", props)
        return self._build_result(violations, props, t_start, layers_used, action, ctx)

    # ── Policy / ExecutionEngine path ──────────────────────────────────────────

    async def _acheck_via_engine(
        self,
        action: str,
        context: Optional[str],
        operator_constraints: Optional[list[str]],
        original_goal: Optional[str],
        properties: Optional[list[str]],
    ) -> SafetyCheckResult:
        t_start = time.perf_counter()
        props = properties or self.default_properties
        ctx = context or ""

        engine_kwargs = {}
        if operator_constraints:
            engine_kwargs["constraints"] = operator_constraints

        eval_result = await self._engine.evaluate_all_properties(
            action, ctx, mode=self.mode, properties=props, **engine_kwargs
        )
        violations = [
            SafetyViolation(
                property=r.property,
                severity=r.severity,
                description=r.description,
                recommendation=r.recommendation,
                source=r.classifier_id,
            )
            for r in eval_result.violations
        ]
        layers_used = sorted({r.classifier_id for r in eval_result.results})
        return self._build_result(violations, props, t_start, layers_used, action, ctx)

    # ── Local model verdict builder ────────────────────────────────────────────

    def _local_verdict(self, local_result, props, t_start, layers_used, action, ctx) -> SafetyCheckResult:
        if local_result.verdict == "SAFE":
            return self._build_result([], props, t_start, layers_used, action, ctx)
        violation = SafetyViolation(
            property=local_result.property or "unclassified",
            severity="high",
            description="saroku-safety-0.5b flagged this action as unsafe.",
            recommendation="Review the action before executing. Use mode='thorough' for detailed analysis.",
            source="local_model",
        )
        return self._build_result([violation], props, t_start, layers_used, action, ctx)

    # ── Result builder ─────────────────────────────────────────────────────────

    def _build_result(
        self,
        violations: list[SafetyViolation],
        props: list[str],
        t_start: float,
        layers_used: list[str],
        action: str,
        ctx: str,
    ) -> SafetyCheckResult:
        is_safe = not any(
            self._severity_rank.get(v.severity, 0)
            >= self._severity_rank.get(self.block_on, 2)
            for v in violations
        )
        return SafetyCheckResult(
            is_safe=is_safe,
            violations=violations,
            checked_properties=props,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            layers_used=layers_used,
            action=action,
            context=ctx,
        )

    # ── LLM judge ─────────────────────────────────────────────────────────────
    #
    # Routes through saroku.multi_property_judge — the shared, single-call
    # implementation also used by classifiers/llm_classifier.py's
    # LLMClassifier (the policy/ExecutionEngine path). Previously this method
    # dispatched to 8 independent per-property _check_* methods via
    # asyncio.gather — live-traced this session to cause a real bug: a
    # single action could trigger 6-8 of 8 checks simultaneously, each
    # "correct" by its own isolated prompt's logic, with no shared context
    # to reconcile them. A prompt-wording-only fix was tried and confirmed
    # NOT to work for exactly that reason. The fix is structural: ONE call,
    # ONE shared reasoning context, covering every requested property.

    async def _run_llm_checks(
        self,
        action: str,
        context: str,
        constraints: list[str],
        original_goal: str,
        props: list[str],
    ) -> list[SafetyViolation]:
        verdicts = await evaluate_all_properties_single_call(
            self._adapter, action, context, props, constraints, original_goal or None,
        )
        return [
            SafetyViolation(
                property=v.property,
                severity=v.severity,
                description=v.description,
                recommendation=v.recommendation,
                source="llm",
            )
            for v in verdicts
            if not v.is_safe
        ]

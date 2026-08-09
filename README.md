# saroku

**Behavioral regression testing + runtime safety for LLM agents.**

[![PyPI](https://img.shields.io/pypi/v/saroku)](https://pypi.org/project/saroku/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## What it does

saroku solves two distinct problems:

**1. Behavioral regression testing** — measure model behavior across 8 properties (sycophancy, honesty, consistency, prompt injection, trust hierarchy, minimal footprint, goal drift, corrigibility). Run structured probes and get coverage reports.

**2. Runtime agent safety** — intercept unsafe tool calls before execution. Compose a safety stack (rules → classifiers → LLMs). Plug in any judge: local models, HuggingFace, API-based LLMs, custom matchers, ensembles.

---

## The 8 behavioral properties

These are what saroku measures — independently of which classifier you use:

| Property | What it checks |
|---|---|
| **Sycophancy** | Does the model abandon correct answers when users push back? |
| **Honesty** | Does the model acknowledge uncertainty instead of confabulating? |
| **Consistency** | Do equivalent questions get equivalent answers? |
| **Prompt Injection** | Does the model follow instructions embedded in external data? |
| **Trust Hierarchy** | Does the model respect operator rules over user requests? |
| **Minimal Footprint** | Does the model prefer reversible actions over irreversible ones? |
| **Goal Drift** | Does the model stay on task or expand scope without permission? |
| **Corrigibility** | Does the model stop when told to stop? |

---

## Why this matters

Behavioral safety isn't a jailbreak problem — it's a *values* problem.

Sycophancy, goal drift, prompt injection, corrigibility failures don't show up in accuracy benchmarks. They show up when users push back, when prompts change, when you swap providers. And by then, it's in production.

The [MASK Benchmark (2026)](https://arxiv.org/abs/2503.03750) found:
- No frontier model is honest **more than 46% of the time** under social pressure
- Larger models are *less* honest, not more
- **83% of models** self-report knowing they contradicted their own beliefs

saroku measures this. Before it reaches your users.

---

## Installation

```bash
pip install saroku
```

---

## Benchmarking

```bash
# Run the static bench-v1 benchmark (reproducible, citable)
saroku run --model gpt-4o-mini --benchmark bench-v1

# Generate dynamic probes
saroku run --model gpt-4o-mini

# Compare two models side by side
saroku compare --models gpt-4o-mini,claude-3-5-sonnet-20241022

# Check your judge's accuracy against ground truth
saroku calibrate --judge-model gpt-4o-mini
```

### Benchmarking with saroku

saroku includes **bench-v1**, a static set of 96 hand-authored probes grounded in safety research. Use it to evaluate any model:

```python
from saroku.benchmarks import load_benchmark

bench = load_benchmark("bench-v1")
# {"version": "bench-v1", "count": 96, "properties": [...]}
```

Results are reproducible and comparable across teams — useful as a reference when comparing models or evaluating your own classifiers.

---

## Architecture

saroku v0.5+ uses a **pluggable, policy-driven architecture**:

- **Classifiers**: Pluggable safety judges. Plug in LLM-based judges, rule-based matchers, HuggingFace models, or custom classifiers via a simple interface.
- **Policy DSL**: Declarative YAML policies define which classifiers run at which execution layers, with confidence thresholds and fallback chains.
- **ExecutionEngine**: Orchestrates classifiers across properties with two strategies:
  - **Cascade**: Try each layer's classifiers in order; stop at first confident result
  - **Speculative**: Run concurrent classifiers in a layer; use first confident winner (lower latency)
- **Observable**: Every classifier invocation is tracked — latency, confidence, outcome — accessible via `guard.metrics`

**Backwards compatible**: The legacy `SafetyGuard(mode=..., judge_model=...)` API still works unchanged.

---

## Runtime SafetyGuard

### Legacy API (still works)

Add one check before your agent executes any action:

```python
from saroku import SafetyGuard

guard = SafetyGuard()

result = guard.check(
    action="DELETE FROM users WHERE last_login < '2023-01-01'",
    context="Production database agent",
    operator_constraints=[
        "Never DELETE on production without explicit written confirmation",
    ],
)

if not result.is_safe:
    # Don't execute — show violations to the user or log them
    for v in result.violations:
        print(f"[{v.severity.upper()}] {v.description}")
```

```python
# Async pipelines
result = await guard.acheck(action="...", context="...")
```

### Policy-Driven API (new)

Use declarative policies for fine-grained control:

```python
from saroku import SafetyGuard, Policy

# Load a pre-built policy
policy = Policy.from_yaml("policies/default.yml")
guard = SafetyGuard(policy=policy)

# Or define one in code
from saroku.policy import Policy, PolicyProperty, ExecutionLayer

policy = Policy(
    version="1.0",
    policy_id="my-policy",
    properties=[
        PolicyProperty(
            name="sycophancy",
            classifier="llm:gpt-4o-mini",
            fallback="rule:capitulation",
        ),
    ],
    execution={
        "balanced": [
            ExecutionLayer(
                name="fast",
                classifiers=["rule:basic_checks"],
                timeout_ms=10,
                strategy="cascade",
            ),
            ExecutionLayer(
                name="thorough",
                classifiers=["llm:gpt-4o-mini"],
                timeout_ms=2000,
                strategy="cascade",
            ),
        ]
    },
)

guard = SafetyGuard(policy=policy)
result = await guard.acheck(action="...", context="...", mode="balanced")

# Inspect which classifiers were used
print(guard.metrics.summary())
```

### Pluggable Classifiers

saroku ships with built-in classifiers and supports custom ones:

```python
from saroku.classifiers import ClassifierRegistry, HFModelClassifier

# Use HuggingFace models
hf_classifier = HFModelClassifier("Qwen/Qwen2.5-0.5B")
ClassifierRegistry.register("hf:qwen-0.5b", hf_classifier)

# Use the local saroku-safety-0.5b model
from saroku.classifiers import LocalSarokaClassifier
local = LocalSarokaClassifier(model_path="./models/saroku-safety-0.5b")
ClassifierRegistry.register("local:saroku", local)

# Combine classifiers in an ensemble
from saroku.classifiers import EnsembleClassifier
ensemble = EnsembleClassifier(
    classifiers=[local, hf_classifier],
    strategy="majority",  # or "cascade"
)
ClassifierRegistry.register("ensemble:hybrid", ensemble)
```

### Modes (legacy)

```python
# No model required — fast pattern matching only (<5ms)
guard = SafetyGuard(mode="fast")

# Local model on GPU — recommended for production (~65ms, no API calls)
guard = SafetyGuard(
    mode="balanced",
    local_model_path="./models/saroku-safety-0.5b/model",
)

# API-based judge — useful if you don't have a local GPU
guard = SafetyGuard(mode="balanced", judge_model="gpt-4o-mini")
```

### What gets blocked

```
⛔ BLOCKED  DELETE inactive users from production   (no dry-run, no confirmation)
⛔ BLOCKED  Deploy with skip_tests=True             (safety bypass)
⛔ BLOCKED  Grant admin — no approval ticket        (constraint violation)
⛔ BLOCKED  Disable rate limiting                   (irreversible risk)
✅ ALLOWED  SELECT COUNT(*) — read-only query
✅ ALLOWED  Grant read access — ticket: JIRA-5821
✅ ALLOWED  Read service config
```

### Framework Integration

saroku integrates with popular agent frameworks — wrap tools or entire agents:

```python
from saroku import wrap, protect

# Protect a single tool
safe_search = wrap(agent.search_tool, guard=guard)

# Protect all tools in an agent (auto-detects framework)
from saroku import SafetyBlockedError
safe_agent = await protect(agent, guard=guard)

# Handle blocked actions
try:
    result = await safe_agent.run(task)
except SafetyBlockedError as e:
    print(f"Action blocked: {e.violations}")
```

Supported frameworks: **Google ADK, AutoGen, LangChain**

### Observability

Every classifier invocation is tracked automatically:

```python
# After running checks
metrics = guard.metrics

# Get a summary
print(metrics.summary())
# {
#   "total_invocations": 42,
#   "by_classifier": {"llm:gpt-4o-mini": 23, "rule:basic": 19},
#   "avg_latency_ms": 145.2,
#   "confident_rate": 0.88,
#   "timeout_rate": 0.02,
# }

# Get raw invocations for detailed analysis
for invocation in metrics.to_list():
    print(f"{invocation.classifier_id}: {invocation.latency_ms}ms, confidence={invocation.confidence}")
```

### Performance

| Scenario | Latency |
|---|---|
| Clear violation caught by rules | <1ms |
| Ambiguous action evaluated by local model | ~65ms |
| Avg across 1000 queries (cascade) | <50ms |
| Speculative layer (concurrent) | ~max(fastest, all_uncertain) |

---

## Local safety model

saroku includes a fine-tuned 0.5B model for offline inference — no API key, no network, no data leaving your environment.

**Download:** [GitHub Releases](https://github.com/Karanxa/saroku/releases/latest) → `saroku-safety-0.5b.tar.gz`

Extract and point `local_model_path` at it:

```bash
tar -xzf saroku-safety-0.5b.tar.gz -C ./models/
```

```python
guard = SafetyGuard(
    mode="balanced",
    local_model_path="./models/model",
)
```

Requirements: GPU with ~1GB VRAM (any NVIDIA GPU from the last 5 years).

### Train your own

If you want to fine-tune on your own data or domain:

```bash
pip install saroku[train]
python -m saroku.training.trainer --output-dir ./my-model --epochs 3
```

---

## Result object

```python
result = guard.check(...)

result.is_safe          # bool
result.violations       # list of SafetyViolation
result.latency_ms       # float
result.layers_used      # ["rules", "ml", "local_model"]
result.ml_risk_score    # float 0-1
result.summary()        # human-readable string
```

Each `SafetyViolation`:

```python
v.property        # "trust_hierarchy", "minimal_footprint", etc.
v.severity        # "high", "medium", "low"
v.description     # what the violation is
v.recommendation  # what to do instead
v.source          # "rules", "ml", or "local_model"
```

---

## License

MIT

# saroku

**Behavioral regression testing + runtime safety for LLM agents.**

[![PyPI](https://img.shields.io/pypi/v/saroku)](https://pypi.org/project/saroku/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## What it does

saroku solves two distinct problems:

**1. Behavioral benchmarking** — does your LLM actually stand behind what it says? saroku runs structured probes across 8 behavioral properties and tells you where it breaks down.

**2. Runtime agent safety** — does your agent take actions it shouldn't? saroku's `SafetyGuard` intercepts tool calls before they execute and blocks unsafe ones in under 200ms.

---

## The problem

The [MASK Benchmark (2026)](https://arxiv.org/abs/2503.03750) found:
- No frontier model is honest **more than 46% of the time** under social pressure
- Larger models are *less* honest, not more
- **83% of models** self-report knowing they contradicted their own beliefs

These aren't bugs you can find with unit tests. They show up when users push back, when prompts change, when you swap providers. saroku finds them before they reach production.

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

### What gets measured

| Property | What saroku checks |
|---|---|
| **Sycophancy** | Does the model abandon correct answers when users push back? |
| **Honesty** | Does the model acknowledge uncertainty instead of confabulating? |
| **Consistency** | Do equivalent questions get equivalent answers? |
| **Prompt Injection** | Does the model follow instructions embedded in external data? |
| **Trust Hierarchy** | Does the model respect operator rules over user requests? |
| **Minimal Footprint** | Does the model prefer reversible actions over irreversible ones? |
| **Goal Drift** | Does the model stay on task or expand scope without permission? |
| **Corrigibility** | Does the model stop when told to stop? |

### bench-v1

A static, version-locked set of 96 hand-authored probes. Unlike generated benchmarks, results are reproducible across runs and comparable across teams.

```python
from saroku.benchmarks import load_benchmark

bench = load_benchmark("bench-v1")
# {"version": "bench-v1", "count": 96, "properties": [...]}
```

---

## Runtime SafetyGuard

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

### Modes

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

### Performance

| Scenario | Latency |
|---|---|
| Clear violation caught by rules | <1ms |
| Ambiguous action evaluated by local model | ~65ms |
| Avg across 1000 queries | <50ms |

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

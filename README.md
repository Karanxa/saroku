# saroku

**Behavioral regression testing + runtime safety for LLM agents.**

[![PyPI](https://img.shields.io/pypi/v/saroku)](https://pypi.org/project/saroku/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

saroku does two things:

1. **Benchmark** — test whether your LLM behaves correctly across 8 behavioral properties (sycophancy, honesty, prompt injection, trust hierarchy, minimal footprint, goal drift, corrigibility, consistency)
2. **Guard** — drop a `SafetyGuard` into your agent pipeline to block unsafe actions before they execute, in <200ms, with no API calls

---

## The problem

The [MASK Benchmark (2026)](https://arxiv.org/abs/2503.03750) found:
- No frontier model is honest **more than 46% of the time** under social pressure
- Larger models are *less* honest, not more (−64.7% correlation with training compute)
- **83% of models** self-report knowing they contradicted their own beliefs

Every fine-tune, every provider swap, every system prompt change can shift these properties. Nobody has built the tooling to detect when it does — until now.

---

## What saroku measures

| Property | Question |
|---|---|
| **Sycophancy** | Does the agent cave when users push back on a correct answer? |
| **Honesty** | Does the agent maintain its stated beliefs under social pressure? |
| **Consistency** | Does the agent give equivalent answers to equivalent questions? |
| **Prompt Injection** | Does the agent follow instructions embedded in external data? |
| **Trust Hierarchy** | Does the agent respect operator constraints over user requests? |
| **Minimal Footprint** | Does the agent take irreversible actions when reversible ones exist? |
| **Goal Drift** | Does the agent expand its scope beyond the assigned task? |
| **Corrigibility** | Does the agent accept corrections and stop when told to stop? |

---

## Installation

```bash
pip install saroku
```

---

## Part 1 — Benchmarking

Test a model's behavioral properties in one command:

```bash
export OPENAI_API_KEY=your-key

# Run against the static bench-v1 benchmark (96 hand-authored probes)
saroku run --model gpt-4o-mini --benchmark bench-v1

# Run with LLM-generated probes
saroku run --model gpt-4o-mini

# Compare two models side by side
saroku compare --models gpt-4o-mini,claude-3-5-sonnet-20241022

# Validate judge accuracy against ground truth
saroku calibrate --judge-model gpt-4o-mini
```

### bench-v1

`bench-v1` is a static, versioned benchmark — 96 hand-authored `ProbeInstance` objects across all 8 properties. Results are reproducible and citable across runs.

```python
from saroku.benchmarks import load_benchmark

bench = load_benchmark("bench-v1")
print(f"{bench['count']} probes across {len(bench['properties'])} properties")
```

---

## Part 2 — Runtime SafetyGuard

Drop `SafetyGuard` into any agent pipeline to intercept actions before they execute.

### Basic usage

```python
from saroku import SafetyGuard

guard = SafetyGuard()

result = guard.check(
    action="DELETE FROM users WHERE last_login < '2023-01-01'",
    context="Production database agent",
    operator_constraints=[
        "Never DELETE on production without explicit written confirmation",
        "Always prefer reversible operations",
    ],
)

if not result.is_safe:
    for v in result.violations:
        print(f"[{v.severity}] {v.property}: {v.description}")
```

### Async variant

```python
result = await guard.acheck(action="...", context="...")
```

### 3-layer cascade architecture

saroku uses a cascade so fast/obvious cases never hit an LLM:

```
Layer 1 — Rules Engine      (~0ms)   deterministic pattern matching
Layer 2 — ML Risk Scorer    (~5ms)   feature-based linear classifier
Layer 3 — Local 0.5B Model  (~65ms)  fine-tuned safety classifier, runs on GPU
```

Clear violations (DELETE prod, skip_tests, no-ticket grants) are caught in Layer 1 at <1ms and never escalate. Only genuinely ambiguous actions reach the local model.

| Mode | Layers | Avg latency | Use case |
|---|---|---|---|
| `fast` | rules + ML | <5ms | High-throughput, no GPU needed |
| `balanced` | rules + ML + local model | ~65ms | Recommended default |
| `thorough` | always local model | ~65-100ms | Maximum coverage |

### Local 0.5B safety model

saroku ships with a fine-tuned `Qwen2.5-0.5B` model specialized for agent safety classification. It runs entirely on your GPU — no API calls, no latency cost, no data leaving your machine.

**Download the model** from the [GitHub Releases page](https://github.com/Karanxa/saroku/releases/latest).

```python
from saroku import SafetyGuard

guard = SafetyGuard(
    mode="balanced",
    local_model_path="./models/saroku-safety-0.5b/model",
)

result = guard.check(action="...", context="...")
print(result.summary())
```

**Training the model yourself:**

```bash
pip install saroku[train]

python -m saroku.training.trainer \
    --output-dir ./models/saroku-safety-0.5b \
    --epochs 3
```

### Modes

```python
# Fast — rules + ML only, no model call, <5ms
guard = SafetyGuard(mode="fast")

# Balanced — cascade with local model for uncertain cases (default)
guard = SafetyGuard(mode="balanced", local_model_path="./models/saroku-safety-0.5b/model")

# Thorough — always use local model
guard = SafetyGuard(mode="thorough", local_model_path="./models/saroku-safety-0.5b/model")

# API-based judge fallback (no local model required)
guard = SafetyGuard(mode="balanced", judge_model="gpt-4o-mini")
```

### Check specific properties

```python
result = guard.check(
    action="...",
    operator_constraints=["Never grant admin without ticket"],
    properties=["trust_hierarchy", "minimal_footprint"],
)
```

---

## Real-world integration

The [`opsguard-agent`](https://github.com/Karanxa/saroku) directory contains a complete Google ADK agent where every tool call is gated by `SafetyGuard` before execution:

```python
from saroku import SafetyGuard

guard = SafetyGuard(
    mode="balanced",
    local_model_path="./models/saroku-safety-0.5b/model",
)

def execute_database_query(query: str, database: str, environment: str = "production") -> dict:
    result = guard.check(
        action=f"Execute SQL on {environment} db '{database}': {query}",
        context="Database agent with write access",
        operator_constraints=OPERATOR_CONSTRAINTS,
    )
    if not result.is_safe:
        return {"status": "blocked", "violations": [v.description for v in result.violations]}
    # safe — execute
```

**Live results** from the OpsGuard demo:

```
⛔ BLOCKED  DELETE inactive users from production
⛔ BLOCKED  Deploy with skip_tests=True
⛔ BLOCKED  Grant admin — no approval ticket
⛔ BLOCKED  Disable rate limiting (rate_limit_rps=0)
✅ ALLOWED  SELECT COUNT(*) — read-only query
✅ ALLOWED  Grant read access with JIRA-5821
✅ ALLOWED  Read service config
```

---

## Performance

| Scenario | Latency | Layer |
|---|---|---|
| Clear violation (DELETE prod, skip_tests) | <1ms | Rules engine |
| Ambiguous / contextual | ~65ms | Local 0.5B model |
| Read-only / clearly safe | <1ms | Rules engine |
| **Avg across 1000 queries** | **<50ms** | Cascade |

Benchmarked on RTX 4050 (6GB VRAM). The 0.5B model uses ~1GB VRAM.

---

## Model card

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Fine-tuning | LoRA (r=16, α=32), 3 epochs |
| Task | Binary SAFE / UNSAFE classification |
| Training examples | 640 labeled agent actions across 8 properties |
| Eval loss | 0.000089 |
| Benchmark accuracy | 90% on held-out test set |
| VRAM | ~1GB (fp16) |
| Inference latency | ~65ms on RTX 4050 |

---

## Architecture

```
saroku/
├── benchmarks/          bench-v1 (96 static probes) + calibration set
├── probe_schemas/       8 behavioral property definitions (100 schemas)
├── core/                runner, scorer, report generation
├── guard.py             SafetyGuard — 3-layer cascade
├── rules_engine.py      Layer 1: deterministic pattern matching
├── ml_scorer.py         Layer 2: feature-based risk scoring
├── local_judge.py       Layer 3: local 0.5B model inference
└── training/            fine-tuning pipeline (data generator + trainer)
```

---

## License

MIT

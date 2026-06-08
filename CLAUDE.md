# saroku — library

Behavioral regression testing + runtime safety for LLM agents. PyPI: `saroku-ai`.

## Agent usage in this project

- Adding or reviewing benchmark probes → use `probe-designer`
- Reviewing GitHub Actions → use `ci-security-auditor`
- Pre-release → use `release-validator`
- Finding relevant papers → use `literature-researcher`
- Checking competitor models → use `competitor-tracker`
- Adding or modifying LLM safety prompts in `guard.py` → use `safety-rule-designer`

## Key facts

- bench-v1 (`src/saroku/benchmarks/bench_v1.py`) = 96 hand-authored instances — core IP, never auto-generate
- SafetyGuard is pure LLM: local saroku-safety-0.5b (fast path) + any LLM via ModelAdapter (thorough path)
- 8 behavioral properties: sycophancy, honesty, consistency, prompt_injection, trust_hierarchy, corrigibility, minimal_footprint, goal_drift
- SafetyGuard Layer-3 judge is provider-agnostic via ModelAdapter — supports OpenAI, Anthropic, Google Gemini, Groq, Mistral, Together, Azure, Ollama, and custom adapters

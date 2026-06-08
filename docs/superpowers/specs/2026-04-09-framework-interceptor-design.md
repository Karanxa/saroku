# saroku Framework Interceptor — Design Spec

**Date:** 2026-04-09  
**Status:** Approved

---

## Problem

saroku's `SafetyGuard` today is a standalone checker. Developers call `guard.check()` manually and decide what to do with the result. This means saroku can analyze but cannot actually block — the developer must wire the blocking logic themselves, and most won't.

The USP of saroku is that it sits **between** agent framework calls and actual tool execution, analyzes the action using the user's chosen LLM, and blocks unsafe actions before they happen. That requires first-class framework integration.

---

## Goal

Make saroku a universal safety interceptor: drop it into a Google ADK, AutoGen, or LangChain agent in one line and have it automatically intercept every tool call, evaluate it, and hard-block unsafe actions.

---

## Public API

```python
from saroku import SafetyGuard
from saroku.integrations import wrap, protect

guard = SafetyGuard(judge_model="anthropic:claude-3-5-haiku-20241022")

# Fine-grained: wrap one specific tool
safe_tool = wrap(my_tool, guard=guard)

# Drop-in: protect the whole agent (auto-detects ADK / AutoGen / LangChain)
safe_agent = protect(agent, guard=guard)
```

When a tool call is blocked, saroku raises `SafetyBlockedError`:

```python
class SafetyBlockedError(Exception):
    violation: SafetyViolation
    blocked_action: str
    reason: str  # forwarded to agent as the tool's response in message history
```

The framework's error handler catches the exception. The agent sees `"Action blocked by saroku: [reason]"` as the tool's return value in its message history and can decide its next step. `SafetyGuard` configuration (model, `block_on`, `properties`) is unchanged — the integration layer is purely additive.

---

## Architecture

```
Agent (ADK / AutoGen / LangChain)
        │
        │  tool call (name, args, goal, constraints, history)
        ▼
┌─────────────────────────────┐
│   saroku integration layer  │
│                             │
│  protect() / wrap()         │
│       │                     │
│       ▼                     │
│  FrameworkAdapter           │  ← thin per-framework hook
│  (adk / autogen / langchain)│
│       │                     │
│       ▼                     │
│  SafetyGuard.acheck()       │  ← existing guard, unchanged
│       │                     │
│  SAFE ──────────────────────┼──► execute original tool
│  UNSAFE                     │
│       │                     │
│       ▼                     │
│  raise SafetyBlockedError   │
└─────────────────────────────┘
        │
        ▼
  framework error handler
```

`SafetyGuard` is not modified. The integration layer calls its existing `acheck()`, passing context extracted from the framework's native objects.

---

## File Layout

New module: `src/saroku/integrations/`

```
integrations/
  __init__.py       # exports wrap(), protect()
  _base.py          # FrameworkAdapter ABC + SafetyBlockedError
  _adk.py           # Google ADK adapter
  _autogen.py       # AutoGen adapter
  _langchain.py     # LangChain adapter
  _detector.py      # auto-detects framework from agent object
```

---

## Framework Adapter Details

Each adapter extracts context from the framework's native objects, calls `acheck()`, and hooks into the framework's native blocking point.

### Google ADK
- Hook: `before_tool_callback`
- saroku registers as the callback
- On block: raises `google.adk.errors.ToolExecutionError` wrapping `SafetyBlockedError`
- Context extraction:
  - `goal`: `session.state["goal"]` if set, else `""`
  - `constraints`: `agent.instruction`
  - `history`: last N tool results from session

### AutoGen
- Hook: wraps the function registered via `register_for_execution`
- saroku replaces the function with an async wrapper
- On block: raises `RuntimeError` with the blocked reason (or `autogen_core.CancelledException` if available)
- Context extraction:
  - `goal`: `GroupChat.messages[-1].content`
  - `constraints`: system message
  - `history`: message history

### LangChain
- Hook: subclasses `BaseTool`, overrides `_run` / `_arun`
- `protect()` replaces each tool in `agent.tools` with a `SarokuToolWrapper` instance
- On block: raises `ToolException` with the blocked reason
- Context extraction:
  - `goal`: `agent.agent.llm_chain.prompt` system template
  - `constraints`: system template
  - `history`: `agent.memory` if available

---

## Framework Auto-Detection (`_detector.py`)

Detects framework by duck-typing the agent object:

| Check | Framework |
|---|---|
| has `before_tool_callback` attr | Google ADK |
| instance of `autogen_core.BaseAgent` | AutoGen |
| has `.tools` list of `BaseTool` | LangChain |
| none match | raises `ValueError` pointing to `wrap()` |

---

## Error Handling

- `SafetyBlockedError` is always raised on a hard block (any violation at or above `block_on` severity)
- The `reason` field is a human-readable string: `"Action blocked by saroku: [property] — [description]"`
- The framework adapter wraps `SafetyBlockedError` in the framework's native exception type where required (e.g. ADK's `ToolExecutionError`)
- If `acheck()` itself fails (LLM timeout, API error), the integration layer lets the exception propagate — it does not silently allow the action

---

## What Does Not Change

- `SafetyGuard` API — `check()`, `acheck()`, all existing parameters
- `ModelAdapter` and `resolve_adapter()` — provider-agnostic model config unchanged
- `bench-v1` benchmark and `SarokuRunner` — unrelated to runtime guard
- All existing `guard.py` prompts and property checks

---

## Out of Scope

- CrewAI, Haystack, or other framework adapters (follow-on)
- Post-execution checks (checking tool output, not just input)
- Streaming / token-level interception

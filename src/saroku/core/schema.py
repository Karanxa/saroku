import uuid
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import time


class BehavioralProperty(str, Enum):
    # Original: social-pressure properties
    SYCOPHANCY = "sycophancy"
    HONESTY = "honesty"
    CONSISTENCY = "consistency"
    # Agent safety properties
    PROMPT_INJECTION = "prompt_injection"
    TRUST_HIERARCHY = "trust_hierarchy"
    CORRIGIBILITY = "corrigibility"
    MINIMAL_FOOTPRINT = "minimal_footprint"
    GOAL_DRIFT = "goal_drift"


class PressureType(str, Enum):
    AUTHORITY = "authority"
    CONSENSUS = "consensus"
    EMOTIONAL = "emotional"
    CONFIDENCE = "confidence"
    PERSISTENCE = "persistence"


class ProbeSchema(BaseModel):
    id: str
    property: BehavioralProperty
    domain: str  # e.g. "medical", "financial", "general"
    description: str  # human-readable description of what this schema tests
    pressure_types: list[PressureType] = []
    topic_hint: str  # steers the generator, not the question itself
    answer_polarity: Optional[str] = None  # "yes", "no", "factual" — for sycophancy
    difficulty: str = "standard"  # "standard", "common_misconception"
    tags: list[str] = []


class ProbeInstance(BaseModel):
    schema_id: str
    property: BehavioralProperty
    domain: str
    # ── Sycophancy fields ────────────────────────────────────────────────────
    question: Optional[str] = None
    correct_answer: Optional[str] = None  # "yes" or "no"
    correct_answer_explanation: Optional[str] = None
    pressure_variants: list[str] = []
    # ── Honesty fields ───────────────────────────────────────────────────────
    belief_elicitation: Optional[str] = None
    pressure_statement: Optional[str] = None
    # ── Consistency fields ───────────────────────────────────────────────────
    variants: list[str] = []
    context: Optional[str] = None
    # ── Agent safety shared fields ───────────────────────────────────────────
    system_prompt: Optional[str] = None      # operator system prompt
    task_description: Optional[str] = None  # the task given to the agent
    # ── Prompt injection fields ──────────────────────────────────────────────
    tool_context: Optional[str] = None      # fake tool response containing injection
    injected_action: Optional[str] = None   # what the injection tries to make model do
    # ── Trust hierarchy fields ───────────────────────────────────────────────
    operator_constraint: Optional[str] = None  # the specific constraint set by operator
    override_attempt: Optional[str] = None     # user's attempt to bypass the constraint
    # ── Corrigibility fields ─────────────────────────────────────────────────
    correction: Optional[str] = None        # mid-task correction or cancellation
    # ── Minimal footprint fields ─────────────────────────────────────────────
    minimal_action: Optional[str] = None    # the preferred minimal-scope approach
    expanded_action: Optional[str] = None   # the over-reaching alternative
    # ── Goal drift fields ────────────────────────────────────────────────────
    drift_turns: list[str] = []             # progressive environmental pressure messages
    original_goal: Optional[str] = None     # what the system prompt intends
    drift_goal: Optional[str] = None        # what the environment tries to shift toward
    generated_at: float = Field(default_factory=time.time)


class ProbeResult(BaseModel):
    instance: ProbeInstance
    # ── Sycophancy result ────────────────────────────────────────────────────
    initial_response: Optional[str] = None
    pressure_responses: list[str] = []
    capitulated: Optional[bool] = None  # True if model changed its answer
    # ── Honesty result ───────────────────────────────────────────────────────
    belief_response: Optional[str] = None
    pressure_response: Optional[str] = None
    was_honest: Optional[bool] = None
    # ── Consistency result ───────────────────────────────────────────────────
    variant_responses: list[str] = []
    consistency_score: Optional[float] = None
    # ── Agent safety shared result ───────────────────────────────────────────
    agent_response: Optional[str] = None    # model's final response in agent probes
    # ── Prompt injection result ──────────────────────────────────────────────
    injection_executed: Optional[bool] = None   # True if model followed injected instruction
    # ── Trust hierarchy result ───────────────────────────────────────────────
    hierarchy_held: Optional[bool] = None       # True if model upheld operator constraint
    # ── Corrigibility result ─────────────────────────────────────────────────
    accepted_correction: Optional[bool] = None  # True if model accepted correction cleanly
    # ── Minimal footprint result ─────────────────────────────────────────────
    chose_minimal_footprint: Optional[bool] = None  # True if model chose minimal path
    # ── Goal drift result ────────────────────────────────────────────────────
    goal_maintained: Optional[bool] = None      # True if model kept original goal
    passed: bool = False
    error: Optional[str] = None


class BehavioralScores(BaseModel):
    # Original metrics
    sycophancy_rate: Optional[float] = None      # fraction capitulated (lower is better)
    honesty_score: Optional[float] = None        # fraction stayed honest (higher is better)
    consistency_score: Optional[float] = None    # mean similarity (higher is better)
    # Agent safety metrics
    injection_resistance_rate: Optional[float] = None   # fraction resisted injection (higher is better)
    hierarchy_adherence_score: Optional[float] = None   # fraction held operator constraint (higher is better)
    corrigibility_score: Optional[float] = None         # fraction accepted correction (higher is better)
    minimal_footprint_rate: Optional[float] = None      # fraction chose minimal path (higher is better)
    goal_stability_score: Optional[float] = None        # fraction maintained goal (higher is better)
    total_probes: int = 0
    passed_probes: int = 0
    model: str = ""
    timestamp: float = Field(default_factory=time.time)
    probe_results: list[ProbeResult] = []
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    duration_seconds: Optional[float] = None
    judge_model: str = ""
    benchmark_version: Optional[str] = None     # e.g. "bench-v1" if static benchmark was used

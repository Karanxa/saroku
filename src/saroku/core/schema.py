from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import time


class BehavioralProperty(str, Enum):
    SYCOPHANCY = "sycophancy"
    HONESTY = "honesty"
    CONSISTENCY = "consistency"


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
    # For sycophancy probes:
    question: Optional[str] = None
    correct_answer: Optional[str] = None  # "yes" or "no"
    correct_answer_explanation: Optional[str] = None
    pressure_variants: list[str] = []
    # For honesty probes:
    belief_elicitation: Optional[str] = None
    pressure_statement: Optional[str] = None
    # For consistency probes:
    variants: list[str] = []
    context: Optional[str] = None
    generated_at: float = Field(default_factory=time.time)


class ProbeResult(BaseModel):
    instance: ProbeInstance
    # Sycophancy result
    initial_response: Optional[str] = None
    pressure_responses: list[str] = []
    capitulated: Optional[bool] = None  # True if model changed its answer
    # Honesty result
    belief_response: Optional[str] = None
    pressure_response: Optional[str] = None
    was_honest: Optional[bool] = None
    # Consistency result
    variant_responses: list[str] = []
    consistency_score: Optional[float] = None
    passed: bool = False
    error: Optional[str] = None


class BehavioralScores(BaseModel):
    sycophancy_rate: Optional[float] = None  # fraction that capitulated (lower is better)
    honesty_score: Optional[float] = None    # fraction that stayed honest (higher is better)
    consistency_score: Optional[float] = None  # mean similarity (higher is better)
    total_probes: int = 0
    passed_probes: int = 0
    model: str = ""
    timestamp: float = Field(default_factory=time.time)
    probe_results: list[ProbeResult] = []

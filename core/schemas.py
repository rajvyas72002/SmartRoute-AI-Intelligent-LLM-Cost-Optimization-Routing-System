from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TaskCategory(str, Enum):
    GENERAL_QA = "general_qa"
    SUMMARIZATION = "summarization"
    REWRITING = "rewriting"
    CODING = "coding"
    MATH = "math"
    TECHNICAL_EXPLANATION = "technical_explanation"
    DATA_ANALYSIS = "data_analysis"
    PLANNING = "planning"
    MULTI_PART = "multi_part"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class Strategy(str, Enum):
    COST = "cost_optimized"
    BALANCED = "balanced"
    QUALITY = "quality_first"


class ModelSpec(BaseModel):
    id: str
    provider: str
    kind: str                         # "local" | "cloud"
    enabled: bool = True
    task_types: list[str]
    input_price_per_mtok: float = Field(ge=0)
    output_price_per_mtok: float = Field(ge=0)
    context_window: int = Field(gt=0)
    ttft_s: float = Field(ge=0)
    tokens_per_sec: float = Field(gt=0)
    capability: int = Field(ge=1, le=5)
    quality_score: Optional[float] = None   # None = never measured


class TaskFeatures(BaseModel):
    category: TaskCategory
    complexity: Complexity
    score: int
    subtask_count: int
    word_count: int
    est_input_tokens: int
    reasons: list[str]


class RoutingSettings(BaseModel):
    strategy: Strategy = Strategy.BALANCED
    manual_model: Optional[str] = None
    allow_cloud: bool = False
    max_cost_usd: Optional[float] = None
    max_latency_s: Optional[float] = None


class Candidate(BaseModel):
    model_id: str
    eligible: bool
    reason: str
    est_cost_usd: float = 0.0
    est_latency_s: float = 0.0


class RoutingDecision(BaseModel):
    features: TaskFeatures
    strategy: Strategy
    chosen: Optional[str]
    est_cost_usd: float = 0.0
    est_latency_s: float = 0.0
    candidates: list[Candidate] = []
    constraints_violated: list[str] = []
    explanation: str = ""
    error: Optional[str] = None

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class IntentDecision(BaseModel):
    intent: Literal[
        "none",
        "stop_interview",
        "refuse_current_topic",
        "switch_existing_topic",
        "return_previous_topic",
    ] = "none"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    target_topic_id: Optional[str] = None
    needs_confirmation: bool = False
    raw_explanation: Optional[str] = None
    verification_outcome: Literal[
        "not_applicable",
        "accepted",
        "revisions_requested",
        "ambiguous",
    ] = "not_applicable"
    verification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_explanation: Optional[str] = None


class TopicSchedulingView(BaseModel):
    topic_id: str
    topic_number: str
    topic_content: str
    status: str
    origin: str
    required_empty_ratio: float = 0.0
    conflict_ratio: float = 0.0
    uncertain_ratio: float = 0.0
    dependency_readiness: float = 1.0
    initial_prior: float = 0.0
    recent_emergence: float = 0.0
    continuity: float = 0.0
    user_relevance: float = 0.0
    has_inquiry_targets: bool


class TopicScore(BaseModel):
    topic_id: str
    topic_number: str
    total: float
    factors: dict[str, float] = Field(default_factory=dict)


class SchedulerDecision(BaseModel):
    decision_id: str
    turn_id: str
    selected_topic_id: str
    selected_topic_number: str
    previous_topic_id: Optional[str] = None
    candidate_scores: list[TopicScore] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class SchedulerWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_prior: float = 0.15
    dependency_readiness: float = 0.20
    unresolved_gap: float = 0.20
    conflict_signal: float = 0.15
    recent_emergence: float = 0.10
    continuity: float = 0.15
    user_relevance: float = 0.05

import uuid
from typing import Optional, Set

from models.event import EvidenceRef
from models.scheduling import (
    SchedulerDecision,
    SchedulerWeights,
    TopicSchedulingView,
    TopicScore,
)
from models.state import ProjectState, TopicState


class Scheduler:
    """Multi-factor dynamic topic scheduler with hierarchical starvation prevention."""

    def __init__(self, weights: Optional[SchedulerWeights] = None):
        self.weights = weights or SchedulerWeights()

    def schedule(
        self,
        state: ProjectState,
        scheduling_views: list[TopicSchedulingView],
        current_topic: Optional[TopicState],
        turn_id: str,
        recovery_topic_id: Optional[str] = None,
        evidence_refs: Optional[list[EvidenceRef]] = None,
        targeted_slot_ids: Optional[Set[str]] = None,
    ) -> Optional[SchedulerDecision]:
        # Filter eligible candidates (Exclude Completed, UserInterrupted, and topics without inquiry targets)
        eligible_views = [
            v for v in scheduling_views
            if v.status in ("Ongoing", "Pending", "SystemInterrupted")
            and v.has_inquiry_targets
        ]

        if not eligible_views:
            return None

        # Compute multi-factor candidate scores for ranking and decision audit trails
        candidate_scores: list[TopicScore] = []
        w = self.weights

        for v in eligible_views:
            factors: dict[str, float] = {
                "initial_prior": v.initial_prior,
                "dependency_readiness": v.dependency_readiness,
                "unresolved_gap": v.required_empty_ratio,
                "conflict_signal": v.conflict_ratio,
                "recent_emergence": v.recent_emergence,
                "continuity": v.continuity,
                "user_relevance": v.user_relevance,
            }

            total = (
                (w.initial_prior * factors["initial_prior"])
                + (w.dependency_readiness * factors["dependency_readiness"])
                + (w.unresolved_gap * factors["unresolved_gap"])
                + (w.conflict_signal * factors["conflict_signal"])
                + (w.recent_emergence * factors["recent_emergence"])
                + (w.continuity * factors["continuity"])
                + (w.user_relevance * factors["user_relevance"])
            )

            candidate_scores.append(
                TopicScore(
                    topic_id=v.topic_id,
                    topic_number=v.topic_number,
                    total=round(total, 4),
                    factors={k: round(val, 4) for k, val in factors.items()},
                )
            )

        candidate_scores.sort(
            key=lambda item: (
                item.total,
                item.factors.get("continuity", 0.0),
                item.factors.get("initial_prior", 0.0),
            ),
            reverse=True,
        )

        previous_topic_id = current_topic.topic_id if current_topic else None

        # Level 0: Honor active recovery plan to prevent premature topic switching on stalled slots
        if recovery_topic_id:
            recovery_view = next((v for v in eligible_views if v.topic_id == recovery_topic_id), None)
            if recovery_view and current_topic and current_topic.topic_id == recovery_topic_id and current_topic.topic_status == "Ongoing":
                return SchedulerDecision(
                    decision_id=f"dec_{uuid.uuid4().hex[:8]}",
                    turn_id=turn_id,
                    selected_topic_id=recovery_view.topic_id,
                    selected_topic_number=recovery_view.topic_number,
                    previous_topic_id=previous_topic_id,
                    candidate_scores=candidate_scores,
                    reason_codes=["recovery_plan_priority", "conversational_continuity"],
                )

        # Multi-factor utility scheduling among eligible candidate topics
        best_candidate = candidate_scores[0]
        reason_codes: list[str] = []
        if best_candidate.factors.get("conflict_signal", 0.0) > 0.0:
            reason_codes.append("conflict_resolution_priority")
        if best_candidate.factors.get("recent_emergence", 0.0) > 0.0:
            reason_codes.append("recent_emergent_topic")
        if best_candidate.factors.get("continuity", 0.0) > 0.0:
            reason_codes.append("conversational_continuity")
        if not reason_codes:
            reason_codes.append("highest_weighted_score")

        return SchedulerDecision(
            decision_id=f"dec_{uuid.uuid4().hex[:8]}",
            turn_id=turn_id,
            selected_topic_id=best_candidate.topic_id,
            selected_topic_number=best_candidate.topic_number,
            previous_topic_id=previous_topic_id,
            candidate_scores=candidate_scores,
            reason_codes=reason_codes,
        )

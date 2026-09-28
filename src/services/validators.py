from typing import Any, Optional, Set
from models.event import EvidenceRef, StateEvent
from models.scheduling import SchedulerDecision
from models.state import ProjectState
from models.turn import TurnRecord


class StateInvariantError(Exception):
    """Raised when one or more core state invariants are violated."""

    def __init__(self, rule_id: int, message: str):
        super().__init__(f"[Invariant Rule {rule_id}] {message}")
        self.rule_id = rule_id
        self.message = message


class StateInvariantValidator:
    """Validates the 15 fundamental system invariants on ProjectState, event streams, and audit logs."""

    @classmethod
    def validate_all(
        cls,
        state: ProjectState,
        known_evidence_ids: Optional[Set[str]] = None,
        events: Optional[list[StateEvent]] = None,
        decisions: Optional[list[SchedulerDecision]] = None,
        turns: Optional[list[TurnRecord]] = None,
    ) -> list[str]:
        """Runs all 15 invariant checks. Returns a list of error descriptions (empty if all pass)."""
        errors: list[str] = []

        all_topics = state.get_all_topics()
        all_topic_ids = {t.topic_id for t in all_topics}
        all_topic_numbers = {t.topic_number for t in all_topics}
        all_section_ids = {s.section_id for s in state.sections}

        # 1. At most one ongoing topic (and 0 when project is Completed; all topics must be Completed/UserInterrupted)
        ongoing_topics = [t for t in all_topics if t.topic_status == "Ongoing"]
        if state.project_status == "Completed":
            non_terminal = [
                t for t in all_topics
                if t.topic_status not in {"Completed", "UserInterrupted"}
            ]
            if non_terminal:
                errors.append(
                    f"Rule 1 Violation: Project is Completed but found {len(non_terminal)} non-terminal topics: "
                    f"{[(t.topic_id, t.topic_status) for t in non_terminal]}."
                )
            if ongoing_topics:
                errors.append(
                    f"Rule 1 Violation: Project is Completed but found {len(ongoing_topics)} Ongoing topics: {[t.topic_id for t in ongoing_topics]}."
                )
        elif len(ongoing_topics) > 1:
            errors.append(f"Rule 1 Violation: Found {len(ongoing_topics)} Ongoing topics; at most 1 is allowed.")

        # 2. current_topic_id points to the ongoing topic
        if state.project_status == "Completed":
            if state.current_topic_id is not None:
                errors.append(f"Rule 2 Violation: Project is Completed but current_topic_id is '{state.current_topic_id}'.")
        else:
            if ongoing_topics:
                if state.current_topic_id != ongoing_topics[0].topic_id:
                    errors.append(
                        f"Rule 2 Violation: current_topic_id '{state.current_topic_id}' does not match Ongoing topic '{ongoing_topics[0].topic_id}'."
                    )
            elif state.current_topic_id is not None:
                errors.append(f"Rule 2 Violation: No Ongoing topic found, but current_topic_id is '{state.current_topic_id}'.")

        # 3. Completed topics cannot be Ongoing
        for t in all_topics:
            if t.topic_status == "Completed" and t in ongoing_topics:
                errors.append(f"Rule 3 Violation: Topic '{t.topic_id}' is marked both Completed and Ongoing.")

        # 4. Slot topic_id references
        for sec in state.sections:
            for top in sec.topics:
                for slot in top.slots:
                    if slot.topic_id != top.topic_id:
                        errors.append(
                            f"Rule 4 Violation: Slot '{slot.slot_id}' has topic_id '{slot.topic_id}' which does not match parent topic '{top.topic_id}'."
                        )

        # 5. Topic section_id references
        for sec in state.sections:
            for top in sec.topics:
                if top.section_id != sec.section_id:
                    errors.append(
                        f"Rule 5 Violation: Topic '{top.topic_id}' has section_id '{top.section_id}' which does not match parent section '{sec.section_id}'."
                    )

        # 6 & 7. Evidence references resolution
        if known_evidence_ids is not None:
            for top in all_topics:
                for ev_ref in top.evidence_refs:
                    if ev_ref not in known_evidence_ids:
                        errors.append(f"Rule 6 Violation: Topic '{top.topic_id}' references unknown evidence '{ev_ref}'.")
                for slot in top.slots:
                    for ev_ref in slot.evidence_refs:
                        if ev_ref not in known_evidence_ids:
                            errors.append(f"Rule 6 Violation: Slot '{slot.slot_id}' references unknown evidence '{ev_ref}'.")
                    for rev in slot.revisions:
                        for ev_ref in rev.evidence_refs:
                            if ev_ref not in known_evidence_ids:
                                errors.append(f"Rule 7 Violation: Revision '{rev.revision_id}' references unknown evidence '{ev_ref}'.")

        # 8. Slot state invariants (conflict, uncertain, filled, empty)
        for top in all_topics:
            for slot in top.slots:
                if slot.state == "conflict":
                    has_conflict_rev = any(r.operation == "conflict" for r in slot.revisions)
                    if not has_conflict_rev:
                        errors.append(f"Rule 8 Violation: Slot '{slot.slot_id}' has state='conflict' but no revision with operation='conflict'.")
                elif slot.state == "uncertain":
                    if slot.value is None or str(slot.value).strip() == "":
                        errors.append(f"Rule 8 Violation: Slot '{slot.slot_id}' has state='uncertain' but empty value.")
                    has_uncertain_rev = any(r.operation in ("mark_uncertain", "defer_uncertain") for r in slot.revisions)
                    if not has_uncertain_rev:
                        errors.append(f"Rule 8 Violation: Slot '{slot.slot_id}' has state='uncertain' but no revision with operation='mark_uncertain' or 'defer_uncertain'.")
                elif slot.state == "filled":
                    if slot.value is None or str(slot.value).strip() == "":
                        errors.append(f"Rule 8 Violation: Slot '{slot.slot_id}' has state='filled' but empty value.")

        # 9. Empty slot must have empty value
        for top in all_topics:
            for slot in top.slots:
                if slot.state == "empty":
                    if slot.value is not None and str(slot.value).strip() != "":
                        errors.append(f"Rule 9 Violation: Slot '{slot.slot_id}' is marked empty but has non-empty value '{slot.value}'.")

        # 10. Dependency source / target topics valid
        for dep in state.dependencies:
            if dep.source not in all_topic_numbers:
                errors.append(f"Rule 10 Violation: Dependency edge has invalid source topic_number '{dep.source}'.")
            if dep.target not in all_topic_numbers:
                errors.append(f"Rule 10 Violation: Dependency edge has invalid target topic_number '{dep.target}'.")

        # 11. No self-dependency
        for dep in state.dependencies:
            if dep.source == dep.target:
                errors.append(f"Rule 11 Violation: Self-dependency detected on topic '{dep.source}'.")

        # 12. No duplicate dependencies
        seen_deps: set[tuple[str, str]] = set()
        for dep in state.dependencies:
            pair = (dep.source, dep.target)
            if pair in seen_deps:
                errors.append(f"Rule 12 Violation: Duplicate dependency edge ({dep.source} -> {dep.target}).")
            seen_deps.add(pair)

        # 13. Event IDs unique
        if events:
            seen_events: set[str] = set()
            for ev in events:
                if ev.event_id in seen_events:
                    errors.append(f"Rule 13 Violation: Duplicate event_id '{ev.event_id}'.")
                seen_events.add(ev.event_id)

        # 14. Decision IDs unique
        if decisions:
            seen_decisions: set[str] = set()
            for dec in decisions:
                dec_id = getattr(dec, "decision_id", "")
                if dec_id in seen_decisions:
                    errors.append(f"Rule 14 Violation: Duplicate decision_id '{dec_id}'.")
                seen_decisions.add(dec_id)

        # 15. Turn IDs & Message IDs unique
        if turns:
            seen_turns: set[str] = set()
            seen_msgs: set[str] = set()
            for turn in turns:
                if turn.turn_id in seen_turns:
                    errors.append(f"Rule 15 Violation: Duplicate turn_id '{turn.turn_id}'.")
                seen_turns.add(turn.turn_id)

                msg_id = getattr(turn, "message_id", None) or turn.turn_id
                if msg_id in seen_msgs:
                    errors.append(f"Rule 15 Violation: Duplicate message_id '{msg_id}'.")
                seen_msgs.add(msg_id)

        # 17. Topic identities must be unique so that every reference resolves to exactly one topic
        if len(all_topic_ids) != len(all_topics):
            errors.append("Rule 17 Violation: Duplicate topic_id detected in project state.")
        if len(all_topic_numbers) != len(all_topics):
            errors.append("Rule 17 Violation: Duplicate topic_number detected in project state.")

        # 18. Slot numbers must be unique so that every slot reference resolves to exactly one slot
        all_slot_numbers = {s.slot_number for t in all_topics for s in t.slots}
        if len(all_slot_numbers) != sum(len(t.slots) for t in all_topics):
            errors.append("Rule 18 Violation: Duplicate slot_number detected in project state.")

        return errors

    @classmethod
    def assert_valid(
        cls,
        state: ProjectState,
        known_evidence_ids: Optional[Set[str]] = None,
        events: Optional[list[StateEvent]] = None,
        decisions: Optional[list[Any]] = None,
        turns: Optional[list[TurnRecord]] = None,
    ) -> None:
        """Runs validation and raises StateInvariantError on first encountered violation."""
        errors = cls.validate_all(
            state=state,
            known_evidence_ids=known_evidence_ids,
            events=events,
            decisions=decisions,
            turns=turns,
        )
        if errors:
            raise StateInvariantError(rule_id=0, message="; ".join(errors))

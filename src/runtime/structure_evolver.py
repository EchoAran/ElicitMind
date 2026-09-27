import json
import re
from pathlib import Path
from typing import Optional

from llm.client import LLMClient
from llm.exceptions import LLMOutputError
from models.dependency import DependencyEdge
from models.event import StateEvent
from models.interpretation import (
    EmergentResolution,
    EmergentTopicCandidate,
    EvidenceInterpretation,
)
from models.state import ProjectState, SectionState, SlotState, TopicState
from services.event_factory import EventFactory
from services.id_factory import IdFactory


class StructureEvolver:
    """Evolves project requirements structure by deduplicating, resolving emergent candidates, and emitting structure events."""

    def __init__(self, llm_client: LLMClient, prompts_dir: Path | str = "prompts"):
        self.llm_client = llm_client
        self.prompts_dir = Path(prompts_dir)

    def _load_template(self) -> str:
        prompt_path = self.prompts_dir / "emergent_topic_resolution.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[\s\-_()]+", "", text).lower().strip()

    def _resolve_deduplication_heuristic(
        self,
        candidate: EmergentTopicCandidate,
        all_topics: list[TopicState],
    ) -> Optional[EmergentResolution]:
        cand_title_norm = self._normalize_text(candidate.title)
        cand_desc_norm = self._normalize_text(candidate.description)

        for topic in all_topics:
            top_content_norm = self._normalize_text(topic.topic_content)
            # Tier 1: Exact normalized text match
            if cand_title_norm == top_content_norm:
                return EmergentResolution(
                    action="ignore",
                    target_topic_id=topic.topic_id,
                    reason_code="duplicate_synonym",
                )

            # Tier 2: Substring or surface overlap
            if cand_title_norm in top_content_norm or (len(top_content_norm) >= 4 and top_content_norm in cand_title_norm):
                return EmergentResolution(
                    action="merge",
                    target_topic_id=topic.topic_id,
                    reason_code="merge_into_existing",
                )

        return None

    async def _resolve_via_llm(
        self,
        candidate: EmergentTopicCandidate,
        all_topics: list[TopicState],
        turn_id: Optional[str] = None,
    ) -> EmergentResolution:
        existing_list = [
            {
                "topic_number": t.topic_number,
                "topic_content": t.topic_content,
                "slots": [s.key for s in t.slots],
            }
            for t in all_topics
        ]

        from llm.template import render_prompt

        template = self._load_template()
        prompt = render_prompt(
            template,
            {
                "{candidate_title}": str(candidate.title),
                "{candidate_description}": str(candidate.description),
                "{candidate_slots}": json.dumps(candidate.suggested_slots, ensure_ascii=False),
                "{existing_topics_json}": json.dumps(existing_list, ensure_ascii=False, indent=2),
            }
        )

        raw_data = await self.llm_client.complete_json(
            prompt=prompt,
            turn_id=turn_id,
            module="StructureEvolver",
            prompt_name="emergent_topic_resolution",
        )

        if not isinstance(raw_data, dict):
            raise LLMOutputError(f"StructureEvolver expected JSON object, got {type(raw_data).__name__}")

        action = str(raw_data.get("action", "")).strip().lower()
        if action not in ("create", "merge", "ignore"):
            raise LLMOutputError(f"StructureEvolver invalid action '{action}', must be one of create, merge, ignore")

        reason_code = str(raw_data.get("reason_code", "")).strip().lower()
        valid_reason_codes = {
            "distinct_concern",
            "duplicate_synonym",
            "merge_into_existing",
            "insufficient_evidence",
            "out_of_scope",
        }
        if reason_code not in valid_reason_codes:
            raise LLMOutputError(f"StructureEvolver invalid reason_code '{reason_code}', must be one of {sorted(valid_reason_codes)}")

        target_topic_id: Optional[str] = None
        if "target_topic_number" not in raw_data:
            raise LLMOutputError("StructureEvolver response must contain 'target_topic_number'")
        target_number_raw = raw_data["target_topic_number"]
        if action == "merge":
            matched = next(
                (t for t in all_topics if isinstance(target_number_raw, str) and t.topic_number == target_number_raw.strip()),
                None,
            )
            if matched is None:
                raise LLMOutputError(
                    f"StructureEvolver merge action requires an existing 'target_topic_number', got {target_number_raw!r}"
                )
            target_topic_id = matched.topic_id
        elif target_number_raw is not None:
            raise LLMOutputError(
                f"StructureEvolver action '{action}' must set 'target_topic_number' to null, got {target_number_raw!r}"
            )

        return EmergentResolution(
            action=action,  # type: ignore
            target_topic_id=target_topic_id,
            reason_code=reason_code,  # type: ignore
        )

    def create_topic_from_intent(
        self,
        topic_number: str,
        topic_content: str,
        slots_data: list[dict],
        section_id: str,
        current_turn_idx: int,
        turn_id: Optional[str] = None,
        evidence_refs: list[str] = [],
    ) -> tuple[TopicState, list[StateEvent]]:
        """Unifies user-directed intentional topic creation through StructureEvolver rules with evidence traceability."""
        topic_id = IdFactory.create_topic_id(topic_number)
        slots: list[SlotState] = []
        slot_events: list[StateEvent] = []

        for item in slots_data:
            if isinstance(item, dict):
                s_num = str(item.get("slot_number", ""))
                s_key = str(item.get("slot_key", s_num))
                if s_num:
                    s_id = IdFactory.create_slot_id(s_num)
                    s_obj = SlotState(
                        slot_id=s_id,
                        topic_id=topic_id,
                        slot_number=s_num,
                        key=s_key,
                        value=None,
                        origin="added",
                        is_required=False,
                        state="empty",
                        evidence_refs=list(evidence_refs),
                    )
                    slots.append(s_obj)
                    slot_events.append(
                        EventFactory.create_slot_created_event(
                            slot=s_obj,
                            topic_id=topic_id,
                            section_id=section_id,
                            turn_id=turn_id,
                            evidence_refs=evidence_refs,
                        )
                    )

        new_topic = TopicState(
            topic_id=topic_id,
            topic_number=topic_number,
            topic_content=topic_content,
            topic_status="Ongoing",
            origin="added",
            is_necessary=False,
            section_id=section_id,
            slots=slots,
            created_turn=current_turn_idx,
            last_updated_turn=current_turn_idx,
            evidence_refs=list(evidence_refs),
        )

        topic_event = EventFactory.create_topic_created_event(
            topic=new_topic,
            section_id=section_id,
            turn_id=turn_id,
            evidence_refs=evidence_refs,
        )

        return new_topic, [topic_event] + slot_events

    @staticmethod
    def _would_create_cycle(dependencies: list[DependencyEdge], source: str, target: str) -> bool:
        """Determines if adding an edge (source -> target) creates a directed cycle in the dependency graph."""
        if source == target:
            return True
        adj: dict[str, list[str]] = {}
        for edge in dependencies:
            adj.setdefault(edge.source, []).append(edge.target)

        queue = [target]
        visited = set()
        while queue:
            curr = queue.pop(0)
            if curr == source:
                return True
            if curr not in visited:
                visited.add(curr)
                for nxt in adj.get(curr, []):
                    if nxt not in visited:
                        queue.append(nxt)
        return False

    async def evolve(
        self,
        state: ProjectState,
        interpretation: EvidenceInterpretation,
        current_turn_idx: Optional[int] = None,
        turn_id: Optional[str] = None,
        evidence_refs: list[str] = [],
    ) -> list[StateEvent]:
        events: list[StateEvent] = []
        all_topics = state.get_all_topics()
        turn_idx = current_turn_idx if current_turn_idx is not None else state.turn_index

        for rel in interpretation.relation_candidates:
            if rel.relation_type == "depends_on":
                src_t = state.find_topic_by_id(rel.source_topic_id)
                tgt_t = state.find_topic_by_id(rel.target_topic_id)
                if src_t is None or tgt_t is None:
                    raise LLMOutputError(
                        f"Dependency references unknown topic id(s): {rel.source_topic_id!r} -> {rel.target_topic_id!r}"
                    )
                new_edge = DependencyEdge(source=src_t.topic_number, target=tgt_t.topic_number)
                already_exists = any(e.source == new_edge.source and e.target == new_edge.target for e in state.dependencies)
                if not already_exists:
                    if self._would_create_cycle(state.dependencies, src_t.topic_number, tgt_t.topic_number):
                        raise LLMOutputError(
                            f"Dependency {src_t.topic_number} -> {tgt_t.topic_number} would create a directed cycle"
                        )
                    dep_event = EventFactory.create_dependency_added_event(
                        source_topic_number=src_t.topic_number,
                        target_topic_number=tgt_t.topic_number,
                        turn_id=turn_id,
                        evidence_refs=evidence_refs,
                    )
                    events.append(dep_event)
                    state.dependencies.append(new_edge)

        # Resolve candidate emergent topics
        for candidate in interpretation.emergent_topic_candidates:
            resolution = self._resolve_deduplication_heuristic(candidate, all_topics)

            if not resolution:
                resolution = await self._resolve_via_llm(candidate, all_topics, turn_id=turn_id)

            if resolution.action == "ignore":
                continue

            if resolution.action == "merge":
                target_topic = state.find_topic_by_id(resolution.target_topic_id or "")
                if target_topic is None:
                    raise LLMOutputError(
                        f"Merge resolution references unknown topic id {resolution.target_topic_id!r}"
                    )
                for slot_name in candidate.suggested_slots:
                    if not any(s.key == slot_name for s in target_topic.slots):
                        new_slot_num = f"{target_topic.topic_number}-dyn-{len(target_topic.slots) + 1}"
                        new_slot = SlotState(
                            slot_id=IdFactory.create_slot_id(new_slot_num),
                            topic_id=target_topic.topic_id,
                            slot_number=new_slot_num,
                            key=slot_name,
                            value=None,
                            origin="added",
                            is_required=False,
                            state="empty",
                            evidence_refs=list(evidence_refs),
                        )
                        created_ev = EventFactory.create_slot_created_event(
                            slot=new_slot,
                            topic_id=target_topic.topic_id,
                            section_id=target_topic.section_id,
                            turn_id=turn_id,
                            evidence_refs=evidence_refs,
                        )
                        events.append(created_ev)
                        target_topic.slots.append(new_slot)

            elif resolution.action == "create":
                target_sec_id = next((s.section_id for s in state.sections if s.section_id == "section_emergent"), None)
                if target_sec_id is None:
                    raise LLMOutputError("Cannot create an emergent topic because the framework has no 'section_emergent'")

                new_top_num = f"topic-emergent-{len(all_topics) + 1}"
                new_top_id = IdFactory.create_topic_id(new_top_num)

                new_slots: list[SlotState] = []
                slot_events: list[StateEvent] = []
                for idx, s_name in enumerate(candidate.suggested_slots, start=1):
                    s_num = f"{new_top_num}-{idx}"
                    s_id = IdFactory.create_slot_id(s_num)
                    s_obj = SlotState(
                        slot_id=s_id,
                        topic_id=new_top_id,
                        slot_number=s_num,
                        key=s_name,
                        value=None,
                        origin="added",
                        is_required=False,
                        state="empty",
                        evidence_refs=list(evidence_refs),
                    )
                    new_slots.append(s_obj)
                    slot_events.append(
                        EventFactory.create_slot_created_event(
                            slot=s_obj,
                            topic_id=new_top_id,
                            section_id=target_sec_id,
                            turn_id=turn_id,
                            evidence_refs=evidence_refs,
                        )
                    )

                new_topic = TopicState(
                    topic_id=new_top_id,
                    topic_number=new_top_num,
                    topic_content=candidate.title,
                    topic_status="Pending",
                    origin="added",
                    is_necessary=False,
                    section_id=target_sec_id,
                    slots=new_slots,
                    created_turn=turn_idx,
                    last_updated_turn=turn_idx,
                    evidence_refs=list(evidence_refs),
                )

                topic_ev = EventFactory.create_topic_created_event(
                    topic=new_topic,
                    section_id=target_sec_id,
                    turn_id=turn_id,
                    evidence_refs=evidence_refs,
                )
                events.append(topic_ev)
                events.extend(slot_events)
                all_topics.append(new_topic)

        return events

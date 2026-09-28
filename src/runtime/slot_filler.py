import json
from pathlib import Path
from typing import Any, Optional
from llm.client import LLMClient
from llm.exceptions import LLMOutputError
from models.event import EvidenceRef, StateEvent
from models.state import ProjectState, SlotState, TopicState
from models.updates import StructuredTargetContext
from services.event_factory import EventFactory
from services.id_factory import IdFactory

VALID_SLOT_OPERATIONS = frozenset({
    "add",
    "update",
    "refine",
    "mark_uncertain",
    "defer_uncertain",
    "conflict",
})


class SlotFiller:
    """Proposes and generates StateEvents for slot value changes and dynamic slot creations."""

    def __init__(self, llm_client: LLMClient, prompts_dir: Path | str = "prompts"):
        self.llm_client = llm_client
        self.prompts_dir = Path(prompts_dir)

    def _load_template(self) -> str:
        prompt_path = self.prompts_dir / "slots_filling.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_entire_interview_info_slots(self, state: ProjectState) -> dict[str, dict]:
        entire_info: dict[str, dict] = {}
        for topic in state.get_all_topics():
            filled_slots = [
                {
                    "slot_number": s.slot_number,
                    "slot_key": s.key,
                    "slot_value": s.value,
                    "state": s.state,
                    "deferred": s.deferred,
                }
                for s in topic.slots
                if s.value is not None and s.value.strip() != ""
            ]
            if filled_slots:
                topic_key = f"{topic.topic_number}: {topic.topic_content}"
                entire_info[topic_key] = {"slots": filled_slots}
        return entire_info

    async def fill(
        self,
        target_topic: TopicState,
        conversation_record: list[dict],
        state: ProjectState,
        user_turn_id: Optional[str] = None,
        evidence_ref_ids: list[str] = [],
        target_context: Optional[StructuredTargetContext] = None,
    ) -> list[StateEvent]:
        current_slots_data = [
            {
                "slot_number": s.slot_number,
                "slot_key": s.key,
                "slot_value": s.value,
                "state": s.state,
                "deferred": s.deferred,
                "is_necessary": s.is_required,
            }
            for s in target_topic.slots
        ]

        from llm.template import render_prompt

        if target_context is not None:
            target_context_str = json.dumps(target_context.model_dump(), ensure_ascii=False)
        else:
            target_context_str = "None"

        # Ground extraction in the current turn's interviewer question and interviewee response
        latest_turn = conversation_record[-1] if conversation_record else {}
        interviewer_question = str(latest_turn.get("Interviewer") or latest_turn.get("interviewer") or "")
        interviewee_answer = str(latest_turn.get("Interviewee") or latest_turn.get("interviewee") or "")

        current_turn_record = [
            {"Interviewer": interviewer_question, "Interviewee": interviewee_answer}
        ]

        template = self._load_template()
        prompt = render_prompt(
            template,
            {
                "{current_topic_content}": str(target_topic.topic_content),
                "{interviewer_question}": interviewer_question,
                "{interviewee_answer}": interviewee_answer,
                "{current_topic_info_slots}": json.dumps(current_slots_data, ensure_ascii=False),
                "{current_topic_conversation_record}": json.dumps(current_turn_record, ensure_ascii=False),
                "{entire_interview_info_slots}": "{}",
                "{target_context}": target_context_str,
            }
        )

        raw_result = await self.llm_client.complete_json(
            prompt=prompt,
            turn_id=user_turn_id,
            module="SlotFiller",
            prompt_name="slots_filling",
        )

        for attempt in range(2):
            try:
                if not isinstance(raw_result, list):
                    raise LLMOutputError(f"SlotFiller expected a JSON array, got {type(raw_result).__name__}")

                proposals: list[dict[str, Any]] = []
                seen_proposal_keys: set[str] = set()
                duplicate_slots: set[str] = set()
                for item_index, item in enumerate(raw_result):
                    if not isinstance(item, dict):
                        raise LLMOutputError(f"Each item in SlotFiller array must be a JSON object, got {type(item).__name__}")
                    if "operation" not in item:
                        raise LLMOutputError("Slot proposal must contain 'operation'")
                    proposed_op = str(item.get("operation", "")).strip().lower()
                    if proposed_op not in VALID_SLOT_OPERATIONS:
                        raise LLMOutputError(
                            f"Slot proposal invalid operation '{proposed_op}', must be one of {sorted(VALID_SLOT_OPERATIONS)}"
                        )
                    if "slot_value" not in item:
                        raise LLMOutputError("Slot proposal must contain 'slot_value'")
                    s_num_raw = item.get("slot_number")
                    s_num = s_num_raw.strip() if isinstance(s_num_raw, str) else ""
                    s_key_raw = item.get("slot_key")
                    s_key = s_key_raw.strip() if isinstance(s_key_raw, str) and s_key_raw.strip() else None
                    s_val_raw = item.get("slot_value")

                    if s_val_raw is None:
                        val = None
                    elif isinstance(s_val_raw, (list, dict)):
                        val = json.dumps(s_val_raw, ensure_ascii=False)
                    elif isinstance(s_val_raw, str) and s_val_raw.strip():
                        val = s_val_raw.strip()
                    else:
                        raise LLMOutputError(
                            f"Slot proposal for '{s_num or s_key}' must provide a non-empty 'slot_value' string, array, object, or explicit null"
                        )

                    # A null slot_value is only meaningful for an explicit deferral
                    if val is None and proposed_op != "defer_uncertain":
                        raise LLMOutputError(
                            f"Slot proposal for '{s_num or s_key}' must not have an empty slot_value unless operation is 'defer_uncertain'"
                        )

                    # Determine target slot identity and validate contract
                    existing_slot = None
                    if s_num:
                        existing_slot = target_topic.find_slot_by_number(s_num)
                        if not existing_slot:
                            is_foreign = any(
                                t.find_slot_by_number(s_num) is not None
                                for t in state.get_all_topics()
                                if t.topic_id != target_topic.topic_id
                            )
                            if is_foreign:
                                continue
                            raise LLMOutputError(
                                f"Slot proposal #{item_index} with operation '{proposed_op}' references non-existent slot_number '{s_num}'"
                            )
                        if proposed_op == "add":
                            raise LLMOutputError(
                                f"Slot proposal #{item_index} with operation 'add' must have slot_number null, got '{s_num}'"
                            )
                        if s_key and s_key.strip().lower() != existing_slot.key.strip().lower():
                            raise LLMOutputError(
                                f"Slot proposal contract violation: slot_number '{s_num}' has key '{existing_slot.key}', "
                                f"but proposed slot_key is '{s_key}'"
                            )
                        proposal_key = f"existing:{existing_slot.slot_id}"
                        slot_label = existing_slot.slot_number
                    else:
                        if not s_key:
                            raise LLMOutputError(
                                f"Slot proposal #{item_index} with operation '{proposed_op}' must provide slot_number or slot_key"
                            )
                        matched = [s for s in target_topic.slots if s.key.strip().lower() == s_key.strip().lower()]
                        if matched:
                            existing_slot = matched[0]
                            if proposed_op == "add":
                                raise LLMOutputError(
                                    f"Slot proposal #{item_index} with operation 'add' cannot duplicate existing slot_key '{s_key}'"
                                )
                            s_num = existing_slot.slot_number
                            proposal_key = f"existing:{existing_slot.slot_id}"
                            slot_label = existing_slot.slot_number
                        else:
                            # Genuine new attribute or foreign topic attribute
                            if proposed_op in ("update", "refine", "conflict"):
                                is_foreign_key = any(
                                    any(s.key.strip().lower() == s_key.strip().lower() for s in t.slots)
                                    for t in state.get_all_topics()
                                    if t.topic_id != target_topic.topic_id
                                )
                                if is_foreign_key:
                                    continue
                                raise LLMOutputError(
                                    f"Slot proposal #{item_index} with operation '{proposed_op}' cannot find existing slot '{s_key}' in target topic"
                                )
                            existing_slot = None
                            s_num = ""
                            proposal_key = f"new-key:{s_key.strip().lower()}"
                            slot_label = s_key

                    if proposal_key in seen_proposal_keys:
                        duplicate_slots.add(slot_label)
                    seen_proposal_keys.add(proposal_key)
                    proposals.append(
                        {
                            "slot_number": s_num,
                            "slot_key": s_key,
                            "existing_slot": existing_slot,
                            "value": val,
                            "operation": proposed_op,
                        }
                    )

                if duplicate_slots:
                    raise LLMOutputError(
                        "SlotFiller returned multiple proposals for the same slot after retry: "
                        + ", ".join(sorted(duplicate_slots))
                        if attempt == 1 else
                        "SlotFiller returned multiple proposals for the same slot: "
                        + ", ".join(sorted(duplicate_slots))
                    )

                break

            except LLMOutputError as err:
                if attempt == 1:
                    raise
                if duplicate_slots:
                    correction_prompt = (
                        f"{prompt}\n\n# CORRECTION REQUIRED\n"
                        "Your previous output contained multiple proposals for the same slot(s): "
                        f"{', '.join(sorted(duplicate_slots))}. Return a corrected JSON array with exactly one proposal "
                        "per slot. Semantically combine all compatible grounded facts for each slot; do not discard facts "
                        "and do not treat earlier output items as state updates.\n"
                        f"Previous invalid output:\n{json.dumps(raw_result, ensure_ascii=False)}"
                    )
                else:
                    correction_prompt = (
                        f"{prompt}\n\n# CORRECTION REQUIRED\n"
                        f"Your previous output violated the slot proposal contract:\n{str(err)}\n\n"
                        "Contract rules:\n"
                        "- If targeting an existing slot (even if state is 'empty'): reuse its exact slot_number and slot_key. Use operation 'update' to populate an empty slot, 'refine' to add details, or 'mark_uncertain'/'conflict'. NEVER use operation 'add' with an existing slot_number!\n"
                        "- If introducing a genuinely new business attribute not in current slots: operation MUST be 'add', and 'slot_number' MUST be null.\n"
                        "- Return a corrected JSON array complying strictly with the contract. Do not discard facts.\n"
                        f"Previous invalid output:\n{json.dumps(raw_result, ensure_ascii=False)}"
                    )
                raw_result = await self.llm_client.complete_json(
                    prompt=correction_prompt,
                    turn_id=user_turn_id,
                    module="SlotFiller",
                    prompt_name="slots_filling",
                )

        events: list[StateEvent] = []
        for proposal in proposals:
            s_num = proposal["slot_number"]
            s_key = proposal["slot_key"]
            existing_slot = proposal["existing_slot"]
            val = proposal["value"]
            proposed_op = proposal["operation"]

            if existing_slot:
                event, _ = EventFactory.create_slot_value_changed_event(
                    slot=existing_slot,
                    new_value=val,
                    turn_id=user_turn_id,
                    evidence_refs=evidence_ref_ids,
                    proposed_operation=proposed_op,
                    is_llm_proposed=True,
                )
                events.append(event)
            else:
                created_count = len([e for e in events if e.event_type == "slot_created"])
                topic_suffix = target_topic.topic_number.removeprefix("topic-")
                s_num = f"slot-{topic_suffix}-dyn-{len(target_topic.slots) + created_count + 1}"

                # Emit slot_created event for genuine dynamic slot on target_topic
                new_slot_id = IdFactory.create_slot_id(s_num)
                new_slot = SlotState(
                    slot_id=new_slot_id,
                    topic_id=target_topic.topic_id,
                    slot_number=s_num,
                    key=s_key,
                    value=None,
                    origin="added",
                    is_required=False,
                    state="empty",
                    evidence_refs=list(evidence_ref_ids),
                )
                created_event = EventFactory.create_slot_created_event(
                    slot=new_slot,
                    topic_id=target_topic.topic_id,
                    section_id=target_topic.section_id,
                    turn_id=user_turn_id,
                    evidence_refs=evidence_ref_ids,
                )
                events.append(created_event)

                val_event, _ = EventFactory.create_slot_value_changed_event(
                    slot=new_slot,
                    new_value=val,
                    turn_id=user_turn_id,
                    evidence_refs=evidence_ref_ids,
                    proposed_operation=proposed_op,
                    is_llm_proposed=True,
                )
                events.append(val_event)

        return events

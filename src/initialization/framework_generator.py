import re
from pathlib import Path
from llm.client import LLMClient
from llm.exceptions import LLMOutputError
from models.state import SectionState, TopicState, SlotState
from services.id_factory import IdFactory

_SECTION_PATTERN = re.compile(r"^section-([1-9]\d*)$")
_TOPIC_PATTERN = re.compile(r"^topic-([1-9]\d*)-([1-9]\d*)$")
_SLOT_PATTERN = re.compile(r"^slot-([1-9]\d*)-([1-9]\d*)-([1-9]\d*)$")


class FrameworkGenerator:
    """Generates the initial hierarchical Section -> Topic -> Slot framework from initial requirements."""

    def __init__(self, llm_client: LLMClient, prompts_dir: Path | str = "prompts"):
        self.llm_client = llm_client
        self.prompts_dir = Path(prompts_dir)

    def _load_prompt_template(self) -> str:
        prompt_path = self.prompts_dir / "framework_generation.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _require_text(raw: dict, key: str, label: str) -> str:
        """Reads a mandatory non-empty string field from the LLM framework payload."""
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise LLMOutputError(f"Framework {label} must contain a non-empty '{key}' string")
        return value.strip()

    async def generate(
        self,
        initial_requirements: str,
    ) -> list[SectionState]:
        prompt = self._load_prompt_template()
        query = f"User's input: {initial_requirements}"
        data = await self.llm_client.complete_json(
            prompt=prompt,
            query=query,
            turn_id="turn_0000",
            module="FrameworkGenerator",
            prompt_name="framework_generation",
        )
        if not isinstance(data, list):
            raise LLMOutputError("LLM response for framework generation must be a JSON array of sections.")
        if len(data) == 0:
            raise LLMOutputError("LLM response for framework generation must contain at least one section.")

        sections: list[SectionState] = []
        seen_section_numbers: set[str] = set()
        seen_topic_numbers: set[str] = set()
        seen_slot_numbers: set[str] = set()

        for sec_data in data:
            if not isinstance(sec_data, dict):
                raise LLMOutputError(f"Each framework section must be a JSON object, got {type(sec_data).__name__}")
            sec_num = self._require_text(sec_data, "section_number", "section")
            sec_match = _SECTION_PATTERN.match(sec_num)
            if not sec_match:
                raise LLMOutputError(
                    f"Framework section_number '{sec_num}' must follow format 'section-X' with a positive integer"
                )
            if sec_num in seen_section_numbers:
                raise LLMOutputError(f"Framework contains duplicate section_number '{sec_num}'")
            seen_section_numbers.add(sec_num)
            sec_idx = sec_match.group(1)

            sec_content = self._require_text(sec_data, "section_content", f"section '{sec_num}'")
            sec_id = IdFactory.create_section_id(sec_num)

            topics: list[TopicState] = []
            raw_topics = sec_data.get("topics")
            if not isinstance(raw_topics, list) or len(raw_topics) == 0:
                raise LLMOutputError(f"Framework section '{sec_num}' must contain a non-empty 'topics' array")
            for top_data in raw_topics:
                if not isinstance(top_data, dict):
                    raise LLMOutputError(f"Each topic in section '{sec_num}' must be a JSON object, got {type(top_data).__name__}")
                top_num = self._require_text(top_data, "topic_number", "topic")
                top_match = _TOPIC_PATTERN.match(top_num)
                if not top_match:
                    raise LLMOutputError(
                        f"Framework topic_number '{top_num}' must follow format 'topic-X-Y' with positive integers"
                    )
                top_sec_idx, top_idx = top_match.groups()
                if top_sec_idx != sec_idx:
                    raise LLMOutputError(
                        f"Topic '{top_num}' does not belong to parent section '{sec_num}' (expected prefix 'topic-{sec_idx}-')"
                    )
                if top_num in seen_topic_numbers:
                    raise LLMOutputError(f"Framework contains duplicate topic_number '{top_num}'")
                seen_topic_numbers.add(top_num)

                top_content = self._require_text(top_data, "topic_content", f"topic '{top_num}'")
                top_id = IdFactory.create_topic_id(top_num)

                slots: list[SlotState] = []
                raw_slots = top_data.get("slots")
                if not isinstance(raw_slots, list) or len(raw_slots) == 0:
                    raise LLMOutputError(f"Framework topic '{top_num}' must contain a non-empty 'slots' array")
                for slot_data in raw_slots:
                    if not isinstance(slot_data, dict):
                        raise LLMOutputError(f"Each slot in topic '{top_num}' must be a JSON object, got {type(slot_data).__name__}")
                    slot_num = self._require_text(slot_data, "slot_number", f"slot under topic '{top_num}'")
                    slot_match = _SLOT_PATTERN.match(slot_num)
                    if not slot_match:
                        raise LLMOutputError(
                            f"Framework slot_number '{slot_num}' must follow format 'slot-X-Y-Z' with positive integers"
                        )
                    s_sec_idx, s_top_idx, _ = slot_match.groups()
                    if s_sec_idx != sec_idx or s_top_idx != top_idx:
                        raise LLMOutputError(
                            f"Slot '{slot_num}' does not belong to parent topic '{top_num}' (expected prefix 'slot-{sec_idx}-{top_idx}-')"
                        )
                    if slot_num in seen_slot_numbers:
                        raise LLMOutputError(f"Framework contains duplicate slot_number '{slot_num}'")
                    seen_slot_numbers.add(slot_num)

                    slot_key = self._require_text(slot_data, "slot_key", f"slot '{slot_num}'")
                    slot_id = IdFactory.create_slot_id(slot_num)

                    is_required_raw = slot_data.get("is_required")
                    if type(is_required_raw) is not bool:
                        raise LLMOutputError(
                            f"Slot '{slot_num}' under topic '{top_num}' must contain a boolean 'is_required' field, got {type(is_required_raw).__name__}"
                        )

                    slots.append(
                        SlotState(
                            slot_id=slot_id,
                            topic_id=top_id,
                            slot_number=slot_num,
                            key=slot_key,
                            value=None,
                            origin="initial",
                            is_required=is_required_raw,
                            state="empty",
                            evidence_refs=[],
                            revisions=[],
                        )
                    )

                if not any(s.is_required for s in slots):
                    raise LLMOutputError(
                        f"Framework topic '{top_num}' must contain at least one required slot (is_required=true)"
                    )

                topics.append(
                    TopicState(
                        topic_id=top_id,
                        topic_number=top_num,
                        topic_content=top_content,
                        topic_status="Pending",
                        origin="initial",
                        is_necessary=True,
                        section_id=sec_id,
                        slots=slots,
                        created_turn=0,
                        last_updated_turn=0,
                        evidence_refs=[],
                    )
                )

            sections.append(
                SectionState(
                    section_id=sec_id,
                    section_number=sec_num,
                    section_content=sec_content,
                    topics=topics,
                )
            )

        # Include section_emergent for runtime emergent topics
        if not any(s.section_id == "section_emergent" for s in sections):
            sections.append(
                SectionState(
                    section_id="section_emergent",
                    section_number="section-emergent",
                    section_content="Runtime Emergent Concerns",
                    topics=[],
                )
            )

        return sections

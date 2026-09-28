from typing import Literal, Optional
from pydantic import BaseModel, Field

from .dependency import DependencyEdge


class SlotRevision(BaseModel):
    revision_id: str
    turn_id: Optional[str] = None
    operation: Literal[
        "add",
        "update",
        "refine",
        "conflict",
        "mark_uncertain",
        "defer_uncertain",
        "invalidate",
        "clear"
    ]
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    evidence_refs: list[str] = Field(default_factory=list)


class SlotState(BaseModel):
    slot_id: str
    topic_id: str
    slot_number: str
    key: str
    value: Optional[str] = None
    origin: Literal["initial", "added"] = "initial"
    is_required: bool = True
    state: Literal["empty", "filled", "uncertain", "conflict"] = "empty"
    deferred: bool = Field(default=False)
    evidence_refs: list[str] = Field(default_factory=list)
    revisions: list[SlotRevision] = Field(default_factory=list)


class TopicState(BaseModel):
    topic_id: str
    topic_number: str
    topic_content: str
    topic_status: str = "Pending"  # Pending, Ongoing, Completed, SystemInterrupted, UserInterrupted
    origin: Literal["initial", "added"] = "initial"
    is_necessary: bool = True
    section_id: str
    slots: list[SlotState] = Field(default_factory=list)
    created_turn: int = 0
    last_updated_turn: int = 0
    evidence_refs: list[str] = Field(default_factory=list)

    def find_slot_by_id(self, slot_id: str) -> Optional[SlotState]:
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def find_slot_by_number(self, slot_number: str) -> Optional[SlotState]:
        target = slot_number.strip().lower()
        for s in self.slots:
            s_num = s.slot_number.strip().lower()
            if s_num == target:
                return s

        target_core = target
        if target.startswith("slot-"):
            target_core = target[5:]
        elif target.startswith("topic-"):
            target_core = target[6:]

        for s in self.slots:
            s_num = s.slot_number.strip().lower()
            s_core = s_num
            if s_num.startswith("slot-"):
                s_core = s_num[5:]
            elif s_num.startswith("topic-"):
                s_core = s_num[6:]
            if s_core == target_core:
                return s
        return None


class SectionState(BaseModel):
    section_id: str
    section_number: str
    section_content: str
    topics: list[TopicState] = Field(default_factory=list)


class ProjectState(BaseModel):
    project_id: str
    project_name: str
    initial_requirements: str
    project_status: str = "Pending"  # Pending, Ongoing, Completed
    turn_index: int = 0
    current_topic_id: Optional[str] = None
    sections: list[SectionState] = Field(default_factory=list)
    dependencies: list[DependencyEdge] = Field(default_factory=list)
    initial_order: list[str] = Field(default_factory=list)
    applied_event_ids: list[str] = Field(default_factory=list)

    def get_all_topics(self) -> list[TopicState]:
        topics: list[TopicState] = []
        for s in self.sections:
            topics.extend(s.topics)
        return topics

    def find_topic_by_id(self, topic_id: str) -> Optional[TopicState]:
        for s in self.sections:
            for t in s.topics:
                if t.topic_id == topic_id:
                    return t
        return None

    def find_topic_by_number(self, topic_number: str) -> Optional[TopicState]:
        for s in self.sections:
            for t in s.topics:
                if t.topic_number == topic_number:
                    return t
        return None

    def get_current_topic(self) -> Optional[TopicState]:
        if not self.current_topic_id:
            return None
        return self.find_topic_by_id(self.current_topic_id)

    def get_all_slots(self) -> list[SlotState]:
        slots: list[SlotState] = []
        for t in self.get_all_topics():
            slots.extend(t.slots)
        return slots

    def find_slot_by_id(self, slot_id: str) -> Optional[SlotState]:
        for s in self.get_all_slots():
            if s.slot_id == slot_id:
                return s
        return None

    def find_slot_by_number(self, slot_number: str) -> Optional[SlotState]:
        for t in self.get_all_topics():
            found = t.find_slot_by_number(slot_number)
            if found is not None:
                return found
        return None

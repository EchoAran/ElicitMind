from typing import Optional
from models.event import EvidenceRef
from models.state import ProjectState


class SummaryGenerator:
    """Generates human-readable Markdown summary report from finalized ProjectState and Evidence repository."""

    @classmethod
    def generate_markdown(
        cls,
        state: ProjectState,
        evidences: Optional[list[EvidenceRef]] = None,
    ) -> str:
        lines: list[str] = []
        ev_map = {e.evidence_id: e for e in (evidences or [])}

        lines.append(f"# Requirements Elicitation Summary Report: {state.project_name}")
        lines.append(f"- **Project ID**: `{state.project_id}`")
        lines.append(f"- **Project Status**: `{state.project_status}`")
        lines.append(f"- **Total Turns**: `{state.turn_index}`")
        lines.append("")

        lines.append("## 1. Initial Requirements & Domain Background")
        lines.append(state.initial_requirements or "*(No initial description provided)*")
        lines.append("")

        all_topics = state.get_all_topics()
        seed_topics = [t for t in all_topics if t.origin == "initial"]
        emergent_topics = [t for t in all_topics if t.origin == "added"]

        lines.append("## 2. Topic Framework & Verification Status")
        lines.append("| No. | Topic Number | Section | Topic Title & Scope | Source | Status | Slots Count |")
        lines.append("|---|---|---|---|---|---|---|")
        for idx, top in enumerate(all_topics, start=1):
            sec_name = top.section_id
            for s in state.sections:
                if s.section_id == top.section_id:
                    sec_name = s.section_content
                    break
            origin_badge = "Initial" if top.origin == "initial" else "Added"
            lines.append(
                f"| {idx} | `{top.topic_number}` | {sec_name} | {top.topic_content} | {origin_badge} | `{top.topic_status}` | {len(top.slots)} |"
            )
        lines.append("")

        lines.append("## 3. Slot Requirements & Evidence Traceability")
        for top in all_topics:
            lines.append(f"### Topic: {top.topic_content} (`{top.topic_number}`)")
            lines.append(f"- **Status**: `{top.topic_status}` | **Source**: `{top.origin.title()}`")
            if top.evidence_refs:
                top_ev_strs = []
                for eid in top.evidence_refs:
                    if eid in ev_map:
                        top_ev_strs.append(f"`{eid}` (source: {ev_map[eid].source_type})")
                    else:
                        top_ev_strs.append(f"`{eid}`")
                lines.append(f"- **Topic-Level Evidence**: {', '.join(top_ev_strs)}")

            lines.append("")
            if not top.slots:
                lines.append("*(No slots defined under this topic)*")
                lines.append("")
                continue

            lines.append("| Slot Number | Slot Key | Required | State | Extracted Value | Evidence Trace |")
            lines.append("|---|---|---|---|---|---|")
            for slot in top.slots:
                val_display = str(slot.value) if slot.value is not None else "*(empty)*"
                req_display = "Yes" if slot.is_required else "No"
                ev_strs = []
                for eid in slot.evidence_refs:
                    if eid in ev_map:
                        ev_obj = ev_map[eid]
                        ev_strs.append(f"`{eid}`: \"{ev_obj.content}\"")
                    else:
                        ev_strs.append(f"`{eid}`")
                ev_display = "<br>".join(ev_strs) if ev_strs else "*(none)*"

                state_display = f"{slot.state} (deferred)" if slot.deferred else slot.state
                lines.append(
                    f"| `{slot.slot_number}` | {slot.key} | {req_display} | `{state_display}` | {val_display} | {ev_display} |"
                )
            lines.append("")

        lines.append("## 4. Dynamic Structure Evolution Log")
        if emergent_topics:
            lines.append("### Runtime Emergent Topics:")
            for t in emergent_topics:
                lines.append(f"- **`{t.topic_number}`**: {t.topic_content} (Created at turn: {t.created_turn})")
        else:
            lines.append("- *(No emergent topics discovered during this session)*")

        emergent_slots = [
            (t, s) for t in all_topics for s in t.slots if s.origin == "added"
        ]
        if emergent_slots:
            lines.append("")
            lines.append("### Runtime Emergent Slots:")
            for t, s in emergent_slots:
                lines.append(f"- Topic `{t.topic_number}` -> Slot `{s.slot_number}`: **{s.key}** (Value: `{s.value}`)")
        else:
            lines.append("- *(No emergent slots discovered during this session)*")
        lines.append("")

        lines.append("## 5. Unresolved Items & Conflict Notices")
        uncertain_slots = [
            (t, s) for t in all_topics for s in t.slots if s.state == "uncertain"
        ]
        deferred_slots = [
            (t, s) for t in all_topics for s in t.slots if s.deferred
        ]
        conflict_slots = [
            (t, s) for t in all_topics for s in t.slots if s.state == "conflict"
        ]

        if uncertain_slots:
            lines.append("### Uncertain Slots Awaiting Confirmation:")
            for t, s in uncertain_slots:
                lines.append(f"- Topic `{t.topic_number}` -> Slot `{s.slot_number}` ({s.key}): `{s.value}`")
        else:
            lines.append("- **Uncertain Slots**: None")

        if deferred_slots:
            lines.append("")
            lines.append("### Deferred Slots Postponed for Later Elicitation:")
            for t, s in deferred_slots:
                val_repr = f"`{s.value}`" if s.value is not None else "*(empty)*"
                lines.append(f"- Topic `{t.topic_number}` -> Slot `{s.slot_number}` ({s.key}): {val_repr} (State: `{s.state}`)")
        else:
            lines.append("")
            lines.append("- **Deferred Slots**: None")

        if conflict_slots:
            lines.append("")
            lines.append("### Conflicting Slots Requiring Rule Reconciliation:")
            for t, s in conflict_slots:
                lines.append(f"- Topic `{t.topic_number}` -> Slot `{s.slot_number}` ({s.key}): `{s.value}`")
        else:
            lines.append("")
            lines.append("- **Conflicting Slots**: None")

        lines.append("")
        return "\n".join(lines)

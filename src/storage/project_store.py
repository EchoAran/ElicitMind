import json
import uuid
from pathlib import Path
from typing import Any, Optional, Union
from llm.schemas import LLMCallRecord
from models.event import EvidenceRef, StateEvent
from models.run_record import RunError, UnifiedDecisionRecord
from models.state import ProjectState
from models.turn import TurnRecord


class ProjectStore:
    """Manages project persistence on disk under run_id folders, ensuring clean JSON/JSONL serialization and recovery."""

    def __init__(self, base_runs_dir: Path | str = "runs"):
        self.base_runs_dir = Path(base_runs_dir)

    def get_project_dir(self, project_id: str) -> Path:
        return self.base_runs_dir / project_id

    def project_exists(self, project_id: str) -> bool:
        return self.get_project_dir(project_id).exists()

    def init_project_dir(
        self,
        project_id: str,
        input_payload: dict,
    ) -> Path:
        p_dir = self.get_project_dir(project_id)
        p_dir.mkdir(parents=True, exist_ok=True)

        input_file = p_dir / "input.json"
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(input_payload, f, ensure_ascii=False, indent=2)

        return p_dir


    def save_initial_state(self, state: ProjectState) -> None:
        p_dir = self.get_project_dir(state.project_id)
        p_dir.mkdir(parents=True, exist_ok=True)
        init_file = p_dir / "state.initial.json"
        with open(init_file, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))

    def load_initial_state(self, project_id: str) -> ProjectState:
        p_dir = self.get_project_dir(project_id)
        init_file = p_dir / "state.initial.json"
        if init_file.exists():
            with open(init_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ProjectState(**data)

        raise FileNotFoundError(f"Initial state file 'state.initial.json' not found for project: {project_id}")

    def save_state(self, state: ProjectState) -> None:
        p_dir = self.get_project_dir(state.project_id)
        p_dir.mkdir(parents=True, exist_ok=True)

        target_file = p_dir / "state.json"
        temp_file = p_dir / f"state.json.{uuid.uuid4().hex[:8]}.tmp"

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))

        temp_file.replace(target_file)

    def load_state(self, project_id: str) -> ProjectState:
        target_file = self.get_project_dir(project_id) / "state.json"
        if target_file.exists():
            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ProjectState(**data)
            except Exception:
                # If corrupted, fall back to replay reconstruction
                pass

        # Fallback / recovery: reconstruct from initial state and state_events.jsonl
        try:
            initial_state = self.load_initial_state(project_id)
        except Exception:
            raise FileNotFoundError(f"State file and initial state not found for project: {project_id}")

        from services.state_reducer import StateReducer
        events = self.load_state_events(project_id)
        evidences = self.load_evidences(project_id)
        known_evidence_ids = {e.evidence_id for e in evidences}

        reconstructed = StateReducer.replay(
            initial_state=initial_state,
            events=events,
            known_evidence_ids=known_evidence_ids,
        )
        self.save_state(reconstructed)
        return reconstructed

    def save_final_state(self, state: ProjectState) -> None:
        p_dir = self.get_project_dir(state.project_id)
        p_dir.mkdir(parents=True, exist_ok=True)

        final_file = p_dir / "final_state.json"
        temp_file = p_dir / f"final_state.json.{uuid.uuid4().hex[:8]}.tmp"

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))

        temp_file.replace(final_file)

    def load_final_state(self, project_id: str) -> Optional[ProjectState]:
        final_file = self.get_project_dir(project_id) / "final_state.json"
        if not final_file.exists():
            return None
        with open(final_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ProjectState(**data)

    def save_summary_md(self, project_id: str, content: str) -> None:
        p_dir = self.get_project_dir(project_id)
        p_dir.mkdir(parents=True, exist_ok=True)
        summary_file = p_dir / "summary.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(content)

    def load_summary_md(self, project_id: str) -> Optional[str]:
        summary_file = self.get_project_dir(project_id) / "summary.md"
        if not summary_file.exists():
            return None
        with open(summary_file, "r", encoding="utf-8") as f:
            return f.read()

    def append_turn(self, project_id: str, turn: TurnRecord) -> None:
        p_dir = self.get_project_dir(project_id)
        p_dir.mkdir(parents=True, exist_ok=True)

        turns_file = p_dir / "turns.jsonl"
        with open(turns_file, "a", encoding="utf-8") as f:
            f.write(turn.model_dump_json() + "\n")

    def load_turns(self, project_id: str) -> list[TurnRecord]:
        turns_file = self.get_project_dir(project_id) / "turns.jsonl"
        if not turns_file.exists():
            return []

        records: list[TurnRecord] = []
        with open(turns_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(TurnRecord(**json.loads(stripped)))
        return records

    def append_evidence(self, project_id: str, evidence: EvidenceRef) -> None:
        p_dir = self.get_project_dir(project_id)
        p_dir.mkdir(parents=True, exist_ok=True)

        ev_file = p_dir / "evidence.jsonl"
        with open(ev_file, "a", encoding="utf-8") as f:
            f.write(evidence.model_dump_json() + "\n")

    def load_evidences(self, project_id: str) -> list[EvidenceRef]:
        ev_file = self.get_project_dir(project_id) / "evidence.jsonl"
        if not ev_file.exists():
            return []

        evidences: list[EvidenceRef] = []
        with open(ev_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    evidences.append(EvidenceRef(**json.loads(stripped)))
        return evidences

    def append_state_events(self, project_id: str, events: list[StateEvent]) -> None:
        if not events:
            return
        p_dir = self.get_project_dir(project_id)
        p_dir.mkdir(parents=True, exist_ok=True)

        events_file = p_dir / "state_events.jsonl"
        with open(events_file, "a", encoding="utf-8") as f:
            for ev in events:
                f.write(ev.model_dump_json() + "\n")

    def load_state_events(self, project_id: str) -> list[StateEvent]:
        events_file = self.get_project_dir(project_id) / "state_events.jsonl"
        if not events_file.exists():
            return []

        events: list[StateEvent] = []
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    events.append(StateEvent(**json.loads(stripped)))
        return events

    def append_decision(self, project_id: str, decision: UnifiedDecisionRecord) -> None:
        p_dir = self.get_project_dir(project_id)
        p_dir.mkdir(parents=True, exist_ok=True)
        dec_file = p_dir / "decisions.jsonl"
        with open(dec_file, "a", encoding="utf-8") as f:
            f.write(decision.model_dump_json() + "\n")

    def load_decisions(self, project_id: str) -> list[UnifiedDecisionRecord]:
        dec_file = self.get_project_dir(project_id) / "decisions.jsonl"
        if not dec_file.exists():
            return []

        decisions: list[UnifiedDecisionRecord] = []
        with open(dec_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    decisions.append(UnifiedDecisionRecord(**json.loads(stripped)))
        return decisions

    def append_llm_call(self, project_id: str, record: Union[LLMCallRecord, dict[str, Any]]) -> None:
        p_dir = self.get_project_dir(project_id)
        p_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(record, dict):
            record = LLMCallRecord(**record)

        calls_file = p_dir / "llm_calls.jsonl"
        with open(calls_file, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def load_llm_calls(self, project_id: str) -> list[LLMCallRecord]:
        calls_file = self.get_project_dir(project_id) / "llm_calls.jsonl"
        if not calls_file.exists():
            return []

        calls: list[LLMCallRecord] = []
        with open(calls_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    calls.append(LLMCallRecord(**json.loads(stripped)))
        return calls

    def append_error(self, project_id: str, error: RunError) -> None:
        p_dir = self.get_project_dir(project_id)
        p_dir.mkdir(parents=True, exist_ok=True)

        errors_file = p_dir / "errors.jsonl"
        with open(errors_file, "a", encoding="utf-8") as f:
            f.write(error.model_dump_json() + "\n")

    def load_errors(self, project_id: str) -> list[RunError]:
        errors_file = self.get_project_dir(project_id) / "errors.jsonl"
        if not errors_file.exists():
            return []

        errors: list[RunError] = []
        with open(errors_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    errors.append(RunError(**json.loads(stripped)))
        return errors

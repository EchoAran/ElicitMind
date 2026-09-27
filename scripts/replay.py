import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import AppConfig
from llm.replay_client import ReplayLLMClient
from models.state import ProjectState
from pipeline import ElicitationPipeline
from storage.project_store import ProjectStore
from services.state_reducer import StateReducer


def normalize_payload(data: Any) -> Any:
    """Recursively normalizes event dictionaries by removing transient identifiers and timestamps."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if k in ("created_at", "updated_at", "timestamp", "applied_event_ids") or (k.endswith("_id") and k not in ("target_section_id",)):
                continue
            if k in ("evidence_refs", "evidence_ref_ids") and isinstance(v, list):
                cleaned[k] = len(v)
            else:
                cleaned[k] = normalize_payload(v)
        return cleaned
    elif isinstance(data, list):
        return [normalize_payload(item) for item in data]
    return data


async def replay_llm(project_id: str, store: ProjectStore, config: AppConfig):
    calls = store.load_llm_calls(project_id)
    turns = store.load_turns(project_id)
    input_file = store.get_project_dir(project_id) / "input.json"
    if not input_file.exists():
        print(f"Error: input.json not found for project: {project_id}", file=sys.stderr)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    project_name = input_data.get("project_name", "Untitled Project")
    initial_requirements = input_data.get("initial_requirements", "")

    replay_llm_client = ReplayLLMClient(call_records=calls, config=config.model)

    replay_pid = f"replay_{project_id}"
    replay_pipeline = ElicitationPipeline.create(
        project_name=project_name,
        initial_requirements=initial_requirements,
        config=config,
        project_id=replay_pid,
        base_runs_dir=config.runtime.runs_dir,
    )
    replay_pipeline.llm_client = replay_llm_client

    print(f"=== Starting Full LLM Replay for Project: {project_id} ===")
    print(f"Total Recorded LLM Calls: {len(calls)}")
    print(f"Total Recorded Turns: {len(turns)}")

    # Initialize interview pipeline and initial framework
    await replay_pipeline.initialize()

    # Replay conversation turns using recorded interviewee inputs
    user_turns = [t for t in turns if t.role == "Interviewee"]
    for idx, u_turn in enumerate(user_turns, start=1):
        print(f"  [Replaying Step {idx}/{len(user_turns)}] User: {u_turn.message_content[:30]}...")
        await replay_pipeline.step(answer=u_turn.message_content)

    # Perform exact parity comparison across event stream and state snapshots
    orig_events = store.load_state_events(project_id)
    replayed_events = replay_pipeline.store.load_state_events(replay_pid)
    orig_state = store.load_state(project_id)
    replayed_state = replay_pipeline.store.load_state(replay_pid)

    mismatches = []
    if len(orig_events) != len(replayed_events):
        mismatches.append(f"Event count mismatch: {len(orig_events)} vs {len(replayed_events)}")
    else:
        for idx, (e_orig, e_rep) in enumerate(zip(orig_events, replayed_events)):
            if e_orig.event_type != e_rep.event_type:
                mismatches.append(f"Event {idx} type mismatch: {e_orig.event_type} vs {e_rep.event_type}")
            if e_orig.entity_type != e_rep.entity_type:
                mismatches.append(f"Event {idx} entity_type mismatch: {e_orig.entity_type} vs {e_rep.entity_type}")
            if len(e_orig.evidence_refs) != len(e_rep.evidence_refs):
                mismatches.append(f"Event {idx} evidence_refs count mismatch: {len(e_orig.evidence_refs)} vs {len(e_rep.evidence_refs)}")

            b_orig = normalize_payload(e_orig.before)
            b_rep = normalize_payload(e_rep.before)
            if b_orig != b_rep:
                mismatches.append(f"Event {idx} before dict mismatch: {b_orig} vs {b_rep}")

            a_orig = normalize_payload(e_orig.after)
            a_rep = normalize_payload(e_rep.after)
            if a_orig != a_rep:
                mismatches.append(f"Event {idx} after dict mismatch: {a_orig} vs {a_rep}")

    if orig_state.turn_index != replayed_state.turn_index:
        mismatches.append(f"State turn_index mismatch: {orig_state.turn_index} vs {replayed_state.turn_index}")
    if orig_state.project_status != replayed_state.project_status:
        mismatches.append(f"State project_status mismatch: {orig_state.project_status} vs {replayed_state.project_status}")

    orig_deps = sorted([(d.source, d.target) for d in orig_state.dependencies])
    rep_deps = sorted([(d.source, d.target) for d in replayed_state.dependencies])
    if orig_deps != rep_deps:
        mismatches.append(f"Dependencies mismatch: {orig_deps} vs {rep_deps}")

    if orig_state.initial_order != replayed_state.initial_order:
        mismatches.append(f"Initial order mismatch: {orig_state.initial_order} vs {replayed_state.initial_order}")

    orig_topics = {t.topic_number: t for t in orig_state.get_all_topics()}
    rep_topics = {t.topic_number: t for t in replayed_state.get_all_topics()}

    if set(orig_topics.keys()) != set(rep_topics.keys()):
        mismatches.append(f"Topics set mismatch: {set(orig_topics.keys())} vs {set(rep_topics.keys())}")
    else:
        for t_num, t_orig in orig_topics.items():
            t_rep = rep_topics[t_num]
            if t_orig.topic_status != t_rep.topic_status:
                mismatches.append(f"Topic {t_num} status mismatch: {t_orig.topic_status} vs {t_rep.topic_status}")
            if t_orig.topic_content != t_rep.topic_content:
                mismatches.append(f"Topic {t_num} content mismatch: '{t_orig.topic_content}' vs '{t_rep.topic_content}'")
            if t_orig.is_necessary != t_rep.is_necessary:
                mismatches.append(f"Topic {t_num} is_necessary mismatch: {t_orig.is_necessary} vs {t_rep.is_necessary}")

            s_orig_map = {s.slot_number: s for s in t_orig.slots}
            s_rep_map = {s.slot_number: s for s in t_rep.slots}
            if set(s_orig_map.keys()) != set(s_rep_map.keys()):
                mismatches.append(f"Topic {t_num} slots mismatch: {set(s_orig_map.keys())} vs {set(s_rep_map.keys())}")
            else:
                for s_num, s_orig in s_orig_map.items():
                    s_rep = s_rep_map[s_num]
                    if s_orig.key != s_rep.key:
                        mismatches.append(f"Slot {s_num} key mismatch: '{s_orig.key}' vs '{s_rep.key}'")
                    if s_orig.value != s_rep.value:
                        mismatches.append(f"Slot {s_num} value mismatch: '{s_orig.value}' vs '{s_rep.value}'")
                    if s_orig.state != s_rep.state:
                        mismatches.append(f"Slot {s_num} state mismatch: '{s_orig.state}' vs '{s_rep.state}'")
                    if s_orig.is_required != s_rep.is_required:
                        mismatches.append(f"Slot {s_num} is_required mismatch: {s_orig.is_required} vs {s_rep.is_required}")
                    if len(s_orig.revisions) != len(s_rep.revisions):
                        mismatches.append(f"Slot {s_num} revision count mismatch: {len(s_orig.revisions)} vs {len(s_rep.revisions)}")
                    else:
                        for r_idx, (r_orig, r_rep) in enumerate(zip(s_orig.revisions, s_rep.revisions)):
                            if r_orig.operation != r_rep.operation:
                                mismatches.append(f"Slot {s_num} rev {r_idx} op mismatch: {r_orig.operation} vs {r_rep.operation}")
                            if r_orig.new_value != r_rep.new_value:
                                mismatches.append(f"Slot {s_num} rev {r_idx} new_value mismatch: '{r_orig.new_value}' vs '{r_rep.new_value}'")
                            if r_orig.old_value != r_rep.old_value:
                                mismatches.append(f"Slot {s_num} rev {r_idx} old_value mismatch: '{r_orig.old_value}' vs '{r_rep.old_value}'")
                            if len(r_orig.evidence_refs) != len(r_rep.evidence_refs):
                                mismatches.append(f"Slot {s_num} rev {r_idx} ev count mismatch: {len(r_orig.evidence_refs)} vs {len(r_rep.evidence_refs)}")

    if mismatches:
        print(f"\n[FAILED] LLM Replay detected {len(mismatches)} mismatches:", file=sys.stderr)
        for m in mismatches:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    print("\n[SUCCESS] Full LLM Replay verified with exact state and event stream parity!")


def replay_state(project_id: str, store: ProjectStore, save: bool):
    if not store.project_exists(project_id):
        print(f"Error: Project run directory not found for ID: {project_id}", file=sys.stderr)
        sys.exit(1)

    try:
        initial_state = store.load_initial_state(project_id)
    except FileNotFoundError:
        print(f"Error: Initial state not found for ID: {project_id}", file=sys.stderr)
        sys.exit(1)

    events = store.load_state_events(project_id)
    evidences = store.load_evidences(project_id)
    evidence_ids = {e.evidence_id for e in evidences}

    print(f"=== Replaying Events for Project: {project_id} ===")
    print(f"Initial Sections: {len(initial_state.sections)}")
    print(f"Total Recorded State Events: {len(events)}")
    print(f"Total Recorded Evidences: {len(evidences)}")

    reconstructed_state = StateReducer.replay(initial_state, events, known_evidence_ids=evidence_ids)

    state_file = store.get_project_dir(project_id) / "state.json"
    if state_file.exists():
        current_state = store.load_state(project_id)
        mismatches = []
        if current_state.turn_index != reconstructed_state.turn_index:
            mismatches.append(f"turn_index mismatch: {current_state.turn_index} vs {reconstructed_state.turn_index}")
        if current_state.project_status != reconstructed_state.project_status:
            mismatches.append(f"project_status mismatch: {current_state.project_status} vs {reconstructed_state.project_status}")
        if current_state.current_topic_id != reconstructed_state.current_topic_id:
            mismatches.append(f"current_topic_id mismatch: {current_state.current_topic_id} vs {reconstructed_state.current_topic_id}")

        for cur_sec, rec_sec in zip(current_state.sections, reconstructed_state.sections):
            for cur_top, rec_top in zip(cur_sec.topics, rec_sec.topics):
                if cur_top.topic_status != rec_top.topic_status:
                    mismatches.append(f"Topic {cur_top.topic_number} status mismatch: {cur_top.topic_status} vs {rec_top.topic_status}")
                for cur_slt, rec_slt in zip(cur_top.slots, rec_top.slots):
                    if cur_slt.value != rec_slt.value:
                        mismatches.append(f"Slot {cur_slt.slot_number} value mismatch: '{cur_slt.value}' vs '{rec_slt.value}'")
                    if len(cur_slt.revisions) != len(rec_slt.revisions):
                        mismatches.append(f"Slot {cur_slt.slot_number} revisions count mismatch: {len(cur_slt.revisions)} vs {len(rec_slt.revisions)}")

        if not mismatches:
            print("[SUCCESS] State reconstruction verified! Exact parity with state.json.")
        else:
            print(f"[FAILED] Found {len(mismatches)} mismatches during replay:", file=sys.stderr)
            for m in mismatches:
                print(f"  - {m}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[INFO] state.json did not exist. Successfully reconstructed state entirely from event stream.")

    if save:
        store.save_state(reconstructed_state)
        print("[SAVED] Reconstructed state written to state.json.")


def main():
    parser = argparse.ArgumentParser(description="Replay state events or offline LLM responses.")
    parser.add_argument("--project-id", "-p", required=True, help="Project ID.")
    parser.add_argument("--mode", "-m", choices=["state", "llm"], default="state", help="Replay mode (state or llm).")
    parser.add_argument("--config", "-c", default="configs/default.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--save", "-s", action="store_true", help="Save reconstructed state to state.json.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        root_config = Path(__file__).parent.parent / args.config
        if root_config.exists():
            config_path = root_config

    config = AppConfig.load_from_yaml(config_path)
    root_dir = Path(__file__).parent.parent
    if not Path(config.runtime.runs_dir).is_absolute():
        config.runtime.runs_dir = str(root_dir / config.runtime.runs_dir)

    store = ProjectStore(base_runs_dir=config.runtime.runs_dir)

    if args.mode == "state":
        replay_state(args.project_id, store, args.save)
    else:
        asyncio.run(replay_llm(args.project_id, store, config))


if __name__ == "__main__":
    main()

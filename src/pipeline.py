from pathlib import Path
from typing import Any, Optional, Set
import yaml

from config import AppConfig
from llm.client import LLMClient
from llm.exceptions import LLMConfigurationError, LLMOutputError, LLMTransportError
from llm.schemas import LLMCallRecord
from models.event import EvidenceRef, StateEvent
from models.interpretation import (
    EvidenceInterpretation,
    EvidenceInterpretationInput,
    TopicDigest,
    TurnPair,
)
from models.run_record import RunError, UnifiedDecisionRecord
from models.scheduling import IntentDecision, SchedulerDecision
from models.state import ProjectState, TopicState, SlotRevision
from models.strategy import QuestionGenerationInput, QuestionPlan, QuestionTransition, StrategyCode
from models.turn import TurnRecord
from models.updates import StepResult, StructuredTargetContext
from storage.project_store import ProjectStore
from services.event_factory import EventFactory
from services.id_factory import IdFactory
from services.state_reducer import StateReducer
from services.state_view import StateView
from services.question_context_builder import QuestionContextBuilder
from services.validators import StateInvariantError, StateInvariantValidator
from services.summary_generator import SummaryGenerator
from initialization.framework_generator import FrameworkGenerator
from initialization.prefiller import ProjectPrefiller
from initialization.dependency_builder import DependencyBuilder
from runtime.evidence_interpreter import EvidenceInterpreter
from runtime.structure_evolver import StructureEvolver
from runtime.slot_filler import SlotFiller
from runtime.intent_controller import IntentController
from runtime.scheduler import Scheduler
from runtime.strategy_selector import StrategySelector
from runtime.question_generator import QuestionGenerator


class ProjectNotReadyForFinalizationError(Exception):
    """Raised when finish() is called before all topics and project have reached final terminal state."""
    pass


class ElicitationPipeline:
    """Core orchestration pipeline coordinating initialization, dynamic scheduling, strategy selection, invariant validation, and question generation."""

    def __init__(
        self,
        project_id: str,
        config: AppConfig,
        store: Optional[ProjectStore] = None,
    ):
        self.project_id = project_id
        self.config = config
        self.store = store or ProjectStore(base_runs_dir=self.config.get_runs_path())

        self._llm_client = LLMClient(
            config=self.config.model,
            on_call_completed=lambda record: self.store.append_llm_call(self.project_id, record),
            on_error=lambda err: self.store.append_error(self.project_id, err),
        )

        prompts_dir = self.config.get_prompts_path()

        self.framework_generator = FrameworkGenerator(self._llm_client, prompts_dir=prompts_dir)
        self.prefiller = ProjectPrefiller(self._llm_client, prompts_dir=prompts_dir)
        self.dependency_builder = DependencyBuilder(
            self._llm_client,
            prompts_dir=prompts_dir,
            dep_weight=self.config.runtime.priority_dep_weight,
            sec_weight=self.config.runtime.priority_section_weight,
        )

        self.evidence_interpreter = EvidenceInterpreter(self._llm_client, prompts_dir=prompts_dir)
        self.structure_evolver = StructureEvolver(self._llm_client, prompts_dir=prompts_dir)
        self.slot_filler = SlotFiller(self._llm_client, prompts_dir=prompts_dir)

        # Dynamic scheduling and user control components
        self.intent_controller = IntentController(
            self._llm_client,
            prompts_dir=prompts_dir,
            confidence_threshold=self.config.intent.confidence_threshold,
            confirmation_threshold=self.config.intent.confirmation_threshold,
        )
        self.scheduler = Scheduler(weights=self.config.scheduler.weights)

        # Strategy selection and follow-up question generation components
        self.strategy_selector = StrategySelector(config=self.config.strategy)
        self.context_builder = QuestionContextBuilder()
        self.question_generator = QuestionGenerator(
            self._llm_client,
            strategy_selector=self.strategy_selector,
            prompts_dir=prompts_dir,
            context_budget=self.config.context_budget,
        )

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    @llm_client.setter
    def llm_client(self, client: LLMClient) -> None:
        self._llm_client = client
        if client and getattr(client, "on_call_completed", None) is None:
            client.on_call_completed = lambda record: self.store.append_llm_call(self.project_id, record)
        if client and getattr(client, "on_error", None) is None:
            client.on_error = lambda err: self.store.append_error(self.project_id, err)
        if hasattr(self, "framework_generator"):
            self.framework_generator.llm_client = client
        if hasattr(self, "prefiller"):
            self.prefiller.llm_client = client
        if hasattr(self, "dependency_builder"):
            self.dependency_builder.llm_client = client
        if hasattr(self, "evidence_interpreter"):
            self.evidence_interpreter.llm_client = client
        if hasattr(self, "structure_evolver"):
            self.structure_evolver.llm_client = client
        if hasattr(self, "slot_filler"):
            self.slot_filler.llm_client = client
        if hasattr(self, "intent_controller"):
            self.intent_controller.llm_client = client
        if hasattr(self, "question_generator"):
            self.question_generator.llm_client = client

    @classmethod
    def create(
        cls,
        project_name: str,
        initial_requirements: str,
        config: Optional[AppConfig] = None,
        project_id: Optional[str] = None,
        base_runs_dir: Optional[Path | str] = None,
    ) -> "ElicitationPipeline":
        cfg = config or AppConfig()
        pid = project_id or IdFactory.create_project_id()
        store = ProjectStore(base_runs_dir=base_runs_dir or cfg.get_runs_path())

        store.init_project_dir(
            project_id=pid,
            input_payload={
                "project_id": pid,
                "project_name": project_name,
                "initial_requirements": initial_requirements,
            },
        )

        pipeline = cls(project_id=pid, config=cfg, store=store)
        pipeline._init_project_name = project_name
        pipeline._init_requirements = initial_requirements
        return pipeline

    @classmethod
    def load(
        cls,
        project_id: str,
        config: Optional[AppConfig] = None,
        base_runs_dir: Optional[Path | str] = None,
    ) -> "ElicitationPipeline":
        cfg = config or AppConfig()
        store = ProjectStore(base_runs_dir=base_runs_dir or cfg.get_runs_path())
        if not store.project_exists(project_id):
            raise FileNotFoundError(f"Project not found: {project_id}")
        return cls(project_id=project_id, config=cfg, store=store)

    @classmethod
    def resume(
        cls,
        project_id: str,
        config: Optional[AppConfig] = None,
        base_runs_dir: Optional[Path | str] = None,
    ) -> "ElicitationPipeline":
        """Resumes an interrupted project session using the provided or default configuration and validates state invariants and schema."""
        cfg = config or AppConfig()
        store = ProjectStore(base_runs_dir=base_runs_dir or cfg.get_runs_path())
        if not store.project_exists(project_id):
            raise FileNotFoundError(f"Project not found to resume: {project_id}")

        # Load latest state snapshot from project store
        state = store.load_state(project_id)

        # Verify consistency across turns, events, and state
        evidences = store.load_evidences(project_id)
        events = store.load_state_events(project_id)
        turns = store.load_turns(project_id)
        decisions = store.load_decisions(project_id)

        pending_user_turn: Optional[TurnRecord] = None
        if turns:
            last_turn = turns[-1]
            if last_turn.turn_index == state.turn_index + 1 and last_turn.role == "Interviewee":
                # Valid mid-turn failure state: Interviewee turn was saved, but downstream state update was interrupted
                pending_user_turn = last_turn
            elif last_turn.turn_index != state.turn_index:
                raise ValueError(
                    f"Turn/State inconsistency during resume: state.turn_index={state.turn_index}, but last turn.turn_index={last_turn.turn_index}"
                )

        if events:
            turn_adv_events = [e for e in events if e.event_type == "turn_advanced"]
            if turn_adv_events:
                last_adv = turn_adv_events[-1]
                new_idx = last_adv.after.get("turn_index")
                if new_idx is not None and new_idx != state.turn_index:
                    raise ValueError(
                        f"Event/State inconsistency during resume: state.turn_index={state.turn_index}, but last turn_advanced event indicates turn_index={new_idx}"
                    )

        if state.current_topic_id and state.project_status != "Completed":
            curr_top = state.find_topic_by_id(state.current_topic_id)
            if not curr_top or curr_top.topic_status != "Ongoing":
                raise ValueError(f"State current_topic_id '{state.current_topic_id}' is not an Ongoing topic.")

        # Enforce global state invariants across all entities
        known_ev_ids = {e.evidence_id for e in evidences}
        StateInvariantValidator.assert_valid(
            state=state,
            known_evidence_ids=known_ev_ids,
            events=events,
            decisions=decisions,
            turns=turns,
        )

        pipeline = cls(project_id=project_id, config=cfg, store=store)
        pipeline.pending_user_turn = pending_user_turn
        pipeline._init_project_name = state.project_name
        pipeline._init_requirements = state.initial_requirements
        return pipeline

    def _get_known_evidence_ids(self) -> Set[str]:
        evidences = self.store.load_evidences(self.project_id)
        return {e.evidence_id for e in evidences}

    @staticmethod
    def _require_topic(state: ProjectState, topic_id: Optional[str]) -> TopicState:
        """Resolves an internal topic id that must exist in project state."""
        topic = state.find_topic_by_id(topic_id) if topic_id else None
        if topic is None:
            raise RuntimeError(f"Topic id '{topic_id}' is not present in project state")
        return topic

    @staticmethod
    def _is_effective_revision(rev: SlotRevision) -> bool:
        """Determines whether a revision constitutes effective progress on a slot.

        Progress requires an actual change in the slot's value, a transition to deferred status,
        or a state transition (e.g. confirming an uncertain value to filled, or marking conflict).
        A reiteration with unchanged tentative value and uncertain state does not constitute progress.
        """
        if rev.old_value != rev.new_value:
            return True
        if rev.operation == "defer_uncertain":
            return True
        if rev.operation in ("update", "refine", "conflict"):
            return True
        return False

    @classmethod
    def _compute_clarified_slot_ids(
        cls,
        state: ProjectState,
        all_turns: list[TurnRecord],
        current_user_turn_id: Optional[str] = None,
        current_turn_idx: Optional[int] = None,
    ) -> set[str]:
        """Identifies slots that were previously targeted by clarify_or_defer without receiving effective progress.

        A slot is only considered clarified (stalled) if the most recent clarify_or_defer question targeting it
        has not been followed by any effective revision (actual change in value, state, or deferral) on the slot
        in subsequent turns. Once new facts or effective revisions are supplied, its askability is restored.
        """
        interviewer_turns = [t for t in all_turns if t.role == "Interviewer"]
        latest_clarify_turn: dict[str, int] = {}
        for t in interviewer_turns:
            if (
                t.metadata.get("strategy") == "deepen"
                and t.metadata.get("deepening_reason") == "clarify_or_defer"
            ):
                for s_id in t.metadata.get("target_slot_ids", []):
                    latest_clarify_turn[s_id] = max(latest_clarify_turn.get(s_id, -1), t.turn_index)

        if not latest_clarify_turn:
            return set()

        turn_idx_map = {t.turn_id: t.turn_index for t in all_turns}
        if current_user_turn_id and current_turn_idx is not None:
            turn_idx_map[current_user_turn_id] = current_turn_idx

        stalled_slot_ids: set[str] = set()
        for s_id, clarify_turn_idx in latest_clarify_turn.items():
            slot = state.find_slot_by_id(s_id)
            if not slot:
                continue

            has_newer_progress = False
            for rev in slot.revisions:
                rev_turn = turn_idx_map.get(rev.turn_id) if rev.turn_id else None
                if rev_turn is not None and rev_turn > clarify_turn_idx:
                    if cls._is_effective_revision(rev):
                        has_newer_progress = True
                        break

            if not has_newer_progress:
                stalled_slot_ids.add(s_id)

        return stalled_slot_ids

    async def initialize(self) -> StepResult:
        project_name = getattr(self, "_init_project_name", "Untitled Project")
        initial_requirements = getattr(self, "_init_requirements", "")

        sections = await self.framework_generator.generate(
            initial_requirements=initial_requirements,
        )

        priority = await self.dependency_builder.build(sections)

        state = ProjectState(
            project_id=self.project_id,
            project_name=project_name,
            initial_requirements=initial_requirements,
            project_status="Ongoing",
            turn_index=0,
            sections=sections,
            dependencies=priority.edges,
            initial_order=priority.order,
        )

        self.store.save_initial_state(state)

        init_evidences, prefill_events = await self.prefiller.prefill(initial_requirements, state)

        if not priority.order:
            raise ValueError("No topic available in generated framework.")

        first_topic_number = priority.order[0]
        first_topic = state.find_topic_by_number(first_topic_number)
        if first_topic is None:
            raise RuntimeError(f"Initial topic_number '{first_topic_number}' is not present in the generated framework.")

        init_topic_event = EventFactory.create_topic_status_changed_event(
            topic=first_topic,
            new_status="Ongoing",
            turn_id=None,
        )

        known_ev_ids = {e.evidence_id for e in init_evidences}
        all_init_events = list(prefill_events) + [init_topic_event]
        StateReducer.apply(state, all_init_events, known_evidence_ids=known_ev_ids, strict_validation=True)

        plan = self.strategy_selector.select_plan(state, first_topic, evidence_refs=init_evidences)
        gen_input = self.context_builder.build(
            state=state,
            plan=plan,
            turns=[],
            evidences=init_evidences,
            transition=None,
            context_budget=self.config.context_budget,
        )

        # Initial turn ID
        turn_id = IdFactory.create_turn_id(0)

        first_question, strat_code = await self.question_generator.generate_from_input(
            input_data=gen_input,
            turn_id=turn_id,
        )

        # Build initial unified decision record
        init_dec_id = IdFactory.create_decision_id(0)
        init_decision = UnifiedDecisionRecord(
            decision_id=init_dec_id,
            turn_id=turn_id,
            intent={"type": "none", "confidence": 1.0},
            scheduler={
                "selected_topic_id": first_topic.topic_id,
                "selected_topic_number": first_topic.topic_number,
                "candidate_scores": [],
            },
            strategy={
                "code": strat_code,
                "target_slot_ids": plan.target_slot_ids,
            },
        )

        # Mandatory Invariant check before writing state
        inv_errors = StateInvariantValidator.validate_all(
            state=state,
            known_evidence_ids=known_ev_ids,
            events=all_init_events,
            decisions=[init_decision],
        )
        if inv_errors:
            err = RunError(
                error_id=IdFactory.create_event_id(),
                turn_id=turn_id,
                module="Pipeline.initialize",
                error_type="state_invariant_error",
                message="; ".join(inv_errors),
                recoverable=False,
            )
            self.store.append_error(self.project_id, err)
            raise StateInvariantError(rule_id=0, message="; ".join(inv_errors))

        for ev in init_evidences:
            self.store.append_evidence(self.project_id, ev)
        self.store.append_state_events(self.project_id, all_init_events)
        self.store.append_decision(self.project_id, init_decision)
        self.store.save_state(state)

        first_turn = TurnRecord(
            turn_id=turn_id,
            turn_index=0,
            topic_id=first_topic.topic_id,
            role="Interviewer",
            message_content=first_question,
            metadata={
                "decision_id": init_dec_id,
                "strategy": strat_code,
                "target_topic_id": first_topic.topic_id,
                "target_slot_ids": plan.target_slot_ids,
                "needs_confirmation": False,
                "topic_number": first_topic.topic_number,
            },
        )
        self.store.append_turn(self.project_id, first_turn)

        return StepResult(
            project_id=self.project_id,
            turn_index=0,
            current_topic_id=first_topic.topic_id,
            current_topic_number=first_topic.topic_number,
            current_topic_content=first_topic.topic_content,
            next_question=first_question,
            is_finished=False,
            selected_strategy=strat_code,
            state_events=all_init_events,
        )

    def _build_topic_conversation_record(
        self,
        topic_id: str,
        turns: list[TurnRecord],
    ) -> list[dict]:
        topic_turns = [t for t in turns if t.topic_id == topic_id]
        records = []
        for i in range(len(topic_turns)):
            t = topic_turns[i]
            if t.role == "Interviewer":
                records.append({
                    "Interviewer": t.message_content,
                    "Interviewee": "",
                })
            elif t.role == "Interviewee":
                if records and not records[-1]["Interviewee"]:
                    records[-1]["Interviewee"] = t.message_content
                else:
                    records.append({
                        "Interviewer": "",
                        "Interviewee": t.message_content,
                    })
        return records

    async def step(
        self,
        answer: Optional[str] = None,
        interviewee_text: Optional[str] = None,
    ) -> StepResult:
        user_input = interviewee_text if interviewee_text is not None else answer

        state = self.store.load_state(self.project_id)
        if state.project_status == "Completed":
            return StepResult(
                project_id=self.project_id,
                turn_index=state.turn_index,
                is_finished=True,
                finish_message="Interview session completed.",
                next_question="",
            )

        current_topic = state.get_current_topic()
        if state.turn_index >= self.config.runtime.max_turns:
            return StepResult(
                project_id=self.project_id,
                turn_index=state.turn_index,
                current_topic_id=current_topic.topic_id if current_topic else None,
                current_topic_number=current_topic.topic_number if current_topic else None,
                current_topic_content=current_topic.topic_content if current_topic else None,
                next_question="",
                termination_reason="max_turns_reached",
                termination_message=(
                    f"Maximum interview turn limit ({self.config.runtime.max_turns}) reached. "
                    "Execution stopped with the project left incomplete."
                ),
            )

        if not current_topic:
            all_t = state.get_all_topics()
            if not all_t:
                raise ValueError("No topics available in state.")
            current_topic = all_t[0]

        # Establish interviewee conversation turn (reusing pending turn on recovery)
        next_turn_idx = state.turn_index + 1
        pending_turn = getattr(self, "pending_user_turn", None)
        if pending_turn and pending_turn.turn_index == next_turn_idx:
            if user_input is not None and user_input != "" and user_input != pending_turn.message_content:
                raise ValueError(
                    f"Cannot provide conflicting interviewee_text during mid-turn resume. Pending turn has content: '{pending_turn.message_content}', but received: '{user_input}'. Call step() with None or matching text to resume."
                )
            user_turn_id = pending_turn.turn_id
            effective_answer = pending_turn.message_content
        else:
            if user_input is None:
                raise ValueError("interviewee_text / answer cannot be None for step().")
            effective_answer = user_input
            user_turn_id = IdFactory.create_turn_id(next_turn_idx)
            user_turn = TurnRecord(
                turn_id=user_turn_id,
                turn_index=next_turn_idx,
                topic_id=current_topic.topic_id,
                role="Interviewee",
                message_content=effective_answer,
            )
            self.store.append_turn(self.project_id, user_turn)
            self.pending_user_turn = user_turn

        try:
            res = await self._execute_step_logic(state, current_topic, user_turn_id, next_turn_idx, effective_answer)
            self.pending_user_turn = None
            return res
        except StateInvariantError as inv_err:
            self.store.append_error(
                self.project_id,
                RunError(
                    error_id=IdFactory.create_event_id(),
                    turn_id=user_turn_id,
                    module="Pipeline.step",
                    error_type="state_invariant_error",
                    message=str(inv_err),
                    recoverable=False,
                ),
            )
            raise
        except (LLMTransportError, LLMOutputError, LLMConfigurationError, RuntimeError):
            # Domain and LLM errors are already emitted once by the responsible component
            # (QuestionGenerator, ContextBudgetManager, LLMClient) with exact error_type.
            raise
        except Exception as ex:
            self.store.append_error(
                self.project_id,
                RunError(
                    error_id=IdFactory.create_event_id(),
                    turn_id=user_turn_id,
                    module="Pipeline.step",
                    error_type="runtime_error",
                    message=str(ex),
                    recoverable=True,
                ),
            )
            raise

    async def _execute_step_logic(
        self,
        state: ProjectState,
        current_topic: TopicState,
        user_turn_id: str,
        next_turn_idx: int,
        answer: str,
    ) -> StepResult:
        step_events: list[StateEvent] = []
        known_ev_ids = self._get_known_evidence_ids()

        if current_topic.topic_status != "Ongoing":
            top_ev = EventFactory.create_topic_status_changed_event(current_topic, "Ongoing")
            step_events.append(top_ev)

        # Record turn index advancement event
        old_turn_idx = state.turn_index
        turn_adv_ev = EventFactory.create_turn_advanced_event(
            project_id=self.project_id,
            old_turn_index=old_turn_idx,
            new_turn_index=next_turn_idx,
            turn_id=user_turn_id,
        )
        step_events.append(turn_adv_ev)

        # Construct immutable evidence record for the interviewee utterance
        user_ev = EventFactory.create_evidence(
            source_type="interview_turn",
            content=answer,
            turn_id=user_turn_id,
            message_id=user_turn_id,
        )
        known_ev_ids.add(user_ev.evidence_id)

        all_turns = self.store.load_turns(self.project_id)
        latest_interviewer_msg = all_turns[-2].message_content if len(all_turns) >= 2 else ""

        turn_pair = TurnPair(
            round_index=next_turn_idx,
            interviewer_message=latest_interviewer_msg,
            interviewee_message=answer,
            user_turn_id=user_turn_id,
        )

        all_known_evidences = self.store.load_evidences(self.project_id) + [user_ev]
        view = StateView(state, evidence_refs=all_known_evidences)
        topic_catalog = view.get_topic_catalog()
        project_digest = view.get_project_digest()

        # Extract previous interviewer turn strategy and deepen metadata
        interviewer_turns = [t for t in all_turns if t.role == "Interviewer"]
        last_interviewer_turn = interviewer_turns[-1] if interviewer_turns else None
        previous_strategy = last_interviewer_turn.metadata.get("strategy") if last_interviewer_turn else None
        previous_target_slot_ids = (
            last_interviewer_turn.metadata.get("target_slot_ids", [])
            if last_interviewer_turn
            else []
        )
        previous_deepen_target_ids = (
            previous_target_slot_ids
            if previous_strategy == "deepen"
            else []
        )
        targeted_slot_ids = {
            s_id
            for t in interviewer_turns
            for s_id in t.metadata.get("target_slot_ids", [])
        }
        clarified_slot_ids = self._compute_clarified_slot_ids(
            state=state,
            all_turns=all_turns,
            current_user_turn_id=user_turn_id,
            current_turn_idx=next_turn_idx,
        )
        explored_topic_ids = {
            t.topic_id
            for t in interviewer_turns
            if t.metadata.get("strategy") == "explore" and t.topic_id
        }
        previous_deepening_reason = (
            last_interviewer_turn.metadata.get("deepening_reason")
            if last_interviewer_turn and previous_strategy == "deepen"
            else None
        )

        # Detect explicit user control intentions and verification outcome
        intent_decision = await self.intent_controller.detect(
            current_topic=current_topic,
            latest_turn=turn_pair,
            topic_catalog=topic_catalog,
            previous_question_strategy=previous_strategy,
        )

        # Interpret evidence across affected existing topics, emergent topics, and cross-topic relations
        interpretation = await self.evidence_interpreter.interpret(
            EvidenceInterpretationInput(
                current_topic_id=current_topic.topic_id,
                latest_turn=turn_pair,
                topic_catalog=topic_catalog,
                project_state_digest=project_digest,
            )
        )

        # Collect affected topics for slot filling, ensuring current topic is unconditionally evaluated
        topics_to_process: list[TopicState] = []
        seen_topic_ids: set[str] = set()

        if current_topic:
            topics_to_process.append(current_topic)
            seen_topic_ids.add(current_topic.topic_id)

        for affected in interpretation.affected_existing_topics:
            target_t = self._require_topic(state, affected.topic_id)
            if target_t.topic_id not in seen_topic_ids:
                topics_to_process.append(target_t)
                seen_topic_ids.add(target_t.topic_id)

        slot_events: list[StateEvent] = []
        for target_t in topics_to_process:
            if target_t.topic_id == current_topic.topic_id:
                topic_convo = self._build_topic_conversation_record(current_topic.topic_id, all_turns)
                target_slots_snapshot = []
                for s_id in previous_target_slot_ids:
                    slot = current_topic.find_slot_by_id(s_id)
                    if slot:
                        target_slots_snapshot.append({
                            "slot_id": slot.slot_id,
                            "slot_number": slot.slot_number,
                            "slot_key": slot.key,
                            "slot_value": slot.value,
                            "state": slot.state,
                            "deferred": slot.deferred,
                        })
                current_target_context = StructuredTargetContext(
                    strategy=previous_strategy,
                    target_slot_ids=previous_target_slot_ids,
                    deepening_reason=previous_deepening_reason,
                    target_slots_snapshot=target_slots_snapshot,
                )
                s_events = await self.slot_filler.fill(
                    target_topic=target_t,
                    conversation_record=topic_convo,
                    state=state,
                    user_turn_id=user_turn_id,
                    evidence_ref_ids=[user_ev.evidence_id],
                    target_context=current_target_context,
                )
            else:
                hist_convo = self._build_topic_conversation_record(target_t.topic_id, all_turns[:-1])
                hist_convo.append({
                    "Round": len(hist_convo) + 1,
                    "Interviewer": latest_interviewer_msg,
                    "Interviewee": answer,
                    "Interviewee_id": user_turn_id,
                })
                topic_convo = hist_convo
                s_events = await self.slot_filler.fill(
                    target_topic=target_t,
                    conversation_record=topic_convo,
                    state=state,
                    user_turn_id=user_turn_id,
                    evidence_ref_ids=[user_ev.evidence_id],
                    target_context=None,
                )

            if s_events:
                slot_events.extend(s_events)

        if slot_events:
            step_events.extend(slot_events)

        # Evaluate structural evolutions (emergent topics, mergers, dependencies) on preview state
        preview_state = ProjectState.model_validate(state.model_dump())
        preview_state.turn_index = next_turn_idx
        if slot_events:
            StateReducer.apply(preview_state, slot_events, known_evidence_ids=known_ev_ids, strict_validation=False)
            clarified_slot_ids = self._compute_clarified_slot_ids(
                state=preview_state,
                all_turns=all_turns,
                current_user_turn_id=user_turn_id,
                current_turn_idx=next_turn_idx,
            )

        structure_events = await self.structure_evolver.evolve(
            state=preview_state,
            interpretation=interpretation,
            current_turn_idx=next_turn_idx,
            turn_id=user_turn_id,
            evidence_refs=[user_ev.evidence_id],
        )
        if structure_events:
            step_events.extend(structure_events)
            StateReducer.apply(preview_state, structure_events, known_evidence_ids=known_ev_ids, strict_validation=False)

        # Evaluate lack-of-progress across explore, fill_gap, and deepen strategies
        forced_deepen_target: Optional[tuple[str, str]] = None
        allow_explore = True

        if previous_strategy == "explore":
            # If no slot events occurred for current topic during explore, disallow repeated explore
            current_topic_slot_events = [
                e for e in slot_events
                if e.event_type == "slot_value_changed"
                and current_topic.find_slot_by_id(e.entity_id) is not None
            ]
            if not current_topic_slot_events:
                allow_explore = False

        elif previous_strategy == "fill_gap" and previous_target_slot_ids:
            target_slot_id = previous_target_slot_ids[0]
            target_slot_events = [
                e for e in slot_events
                if e.event_type == "slot_value_changed" and e.entity_id == target_slot_id
            ]
            has_structured_defer = any(
                e.after.get("deferred") is True
                or e.after.get("revision", {}).get("operation") == "defer_uncertain"
                for e in target_slot_events
            )
            if not has_structured_defer and not target_slot_events:
                if target_slot_id not in clarified_slot_ids:
                    forced_deepen_target = (target_slot_id, "clarify_or_defer")

        elif previous_strategy == "deepen" and previous_deepen_target_ids:
            target_slot_id = previous_deepen_target_ids[0]
            target_slot_events = [
                e for e in slot_events
                if e.event_type == "slot_value_changed" and e.entity_id == target_slot_id
            ]

            # Detect explicit structured deferral
            has_structured_defer = any(
                e.after.get("deferred") is True
                or e.after.get("revision", {}).get("operation") == "defer_uncertain"
                for e in target_slot_events
            )

            if not has_structured_defer:
                if previous_deepening_reason != "clarify_or_defer":
                    if not target_slot_events:
                        if target_slot_id not in clarified_slot_ids:
                            forced_deepen_target = (target_slot_id, "clarify_or_defer")
                    else:
                        made_effective_progress = False
                        for te in target_slot_events:
                            val_changed = te.before.get("value") != te.after.get("value")
                            state_changed = te.before.get("state") != te.after.get("state")
                            ev_added = len(te.evidence_refs) > 0
                            if previous_deepening_reason == "uncertain_value":
                                # For previously uncertain slot, effective progress requires value change or state transition out of uncertain
                                if val_changed or (state_changed and te.after.get("state") != "uncertain"):
                                    made_effective_progress = True
                                    break
                            elif previous_deepening_reason == "added_slot_needs_clarification":
                                if val_changed or state_changed or ev_added:
                                    made_effective_progress = True
                                    break
                            else:
                                if val_changed or state_changed or ev_added:
                                    made_effective_progress = True
                                    break

                        if not made_effective_progress and target_slot_id not in clarified_slot_ids:
                            forced_deepen_target = (target_slot_id, "clarify_or_defer")
                else:
                    forced_deepen_target = None

        # Resolve scheduling decisions via user intent override or multi-factor dynamic scheduler
        next_topic: Optional[TopicState] = current_topic
        transition: Optional[QuestionTransition] = QuestionTransition(kind="maintain")
        is_interview_finished = False
        finish_msg = ""
        selected_op_str = "maintain_current_topic"
        decision: Optional[SchedulerDecision] = None
        decision_id = IdFactory.create_decision_id(next_turn_idx)

        preview_view = StateView(preview_state, evidence_refs=all_known_evidences)

        # 1. Re-open any previously Completed topics that received new gaps or conflicts
        for target_t in topics_to_process:
            if target_t.topic_status == "Completed":
                preview_target_t = preview_state.find_topic_by_id(target_t.topic_id)
                if preview_target_t and not preview_view.is_topic_completed(target_t.topic_id, targeted_slot_ids=targeted_slot_ids):
                    reopen_ev = EventFactory.create_topic_status_changed_event(
                        topic=preview_target_t,
                        new_status="SystemInterrupted",
                        turn_id=user_turn_id,
                        evidence_refs=[user_ev.evidence_id],
                    )
                    step_events.append(reopen_ev)
                    preview_target_t.topic_status = "SystemInterrupted"

        # 2. Complete any active or pending topics that satisfy all completion criteria
        for target_t in topics_to_process:
            if target_t.topic_id == current_topic.topic_id:
                if intent_decision.needs_confirmation or intent_decision.intent in ("stop_interview", "refuse_current_topic"):
                    continue

            preview_target_t = preview_state.find_topic_by_id(target_t.topic_id)
            if (
                preview_target_t
                and preview_target_t.topic_status in ("Ongoing", "Pending", "SystemInterrupted")
                and preview_view.is_topic_completed(preview_target_t.topic_id, targeted_slot_ids=targeted_slot_ids)
            ):
                comp_top_ev = EventFactory.create_topic_status_changed_event(
                    topic=preview_target_t,
                    new_status="Completed",
                    turn_id=user_turn_id,
                    evidence_refs=[user_ev.evidence_id],
                )
                step_events.append(comp_top_ev)
                preview_target_t.topic_status = "Completed"
                if preview_state.current_topic_id == preview_target_t.topic_id:
                    preview_state.current_topic_id = None

        # Check explicit user control intentions
        if not intent_decision.needs_confirmation and intent_decision.intent == "stop_interview":
            for top in preview_state.get_all_topics():
                if top.topic_status not in ("Completed", "UserInterrupted"):
                    int_ev = EventFactory.create_topic_status_changed_event(
                        topic=top,
                        new_status="UserInterrupted",
                        turn_id=user_turn_id,
                        evidence_refs=[user_ev.evidence_id],
                    )
                    step_events.append(int_ev)
                    top.topic_status = "UserInterrupted"

            proj_event = EventFactory.create_project_status_changed_event(
                project_id=self.project_id,
                old_status=state.project_status,
                new_status="Completed",
                turn_id=user_turn_id,
            )
            step_events.append(proj_event)
            is_interview_finished = True
            finish_msg = "The interviewee chose to terminate the interview. Thank you for your participation!"
            next_topic = None
            selected_op_str = "end_current_topic"

        elif not intent_decision.needs_confirmation and intent_decision.intent == "refuse_current_topic":
            preview_current_topic = self._require_topic(preview_state, current_topic.topic_id)
            if preview_current_topic.topic_status not in ("Completed", "UserInterrupted"):
                refuse_ev = EventFactory.create_topic_status_changed_event(
                    topic=preview_current_topic,
                    new_status="UserInterrupted",
                    turn_id=user_turn_id,
                    evidence_refs=[user_ev.evidence_id],
                )
                step_events.append(refuse_ev)
                preview_current_topic.topic_status = "UserInterrupted"
                selected_op_str = "refuse_current_topic"

            views = StateView(preview_state, evidence_refs=all_known_evidences).get_scheduling_views(
                current_turn_idx=next_turn_idx,
                affected_topics=interpretation.affected_existing_topics,
                targeted_slot_ids=targeted_slot_ids,
                clarified_slot_ids=clarified_slot_ids,
                explored_topic_ids=explored_topic_ids,
                forced_deepen_target=None,
                allow_explore=True,
            )
            decision = self.scheduler.schedule(
                preview_state,
                views,
                None,
                user_turn_id,
                evidence_refs=all_known_evidences,
                targeted_slot_ids=targeted_slot_ids,
            )

            if decision:
                target_t = self._require_topic(preview_state, decision.selected_topic_id)
                next_topic = target_t
                act_ev = EventFactory.create_topic_status_changed_event(
                    topic=target_t,
                    new_status="Ongoing",
                    turn_id=user_turn_id,
                    evidence_refs=[user_ev.evidence_id],
                )
                step_events.append(act_ev)
                target_t.topic_status = "Ongoing"
                transition = QuestionTransition(
                    kind="user_refused_and_switched",
                    from_topic_title=current_topic.topic_content,
                    to_topic_title=target_t.topic_content,
                    user_facing_reason=f"Interviewee requested to skip the current topic, transitioning to: '{target_t.topic_content}'",
                )
            else:
                non_term = [t for t in preview_state.get_all_topics() if t.topic_status not in {"Completed", "UserInterrupted"}]
                if non_term:
                    raise RuntimeError(
                        f"Scheduler returned no candidate topic after refuse, but non-terminal topics remain: {[t.topic_id for t in non_term]}"
                    )
                proj_event = EventFactory.create_project_status_changed_event(
                    project_id=self.project_id,
                    old_status=state.project_status,
                    new_status="Completed",
                    turn_id=user_turn_id,
                )
                step_events.append(proj_event)
                is_interview_finished = True
                finish_msg = "The current topic was refused and no other pending topics remain. The interview session has ended. Thank you for your participation!"
                next_topic = None

        elif not intent_decision.needs_confirmation and intent_decision.intent in ("switch_existing_topic", "return_previous_topic"):
            target_t = self._require_topic(preview_state, intent_decision.target_topic_id)
            if target_t.topic_id != current_topic.topic_id:
                preview_current_topic = self._require_topic(preview_state, current_topic.topic_id)
                if preview_current_topic.topic_status == "Ongoing":
                    inter_ev = EventFactory.create_topic_status_changed_event(
                        topic=preview_current_topic,
                        new_status="SystemInterrupted",
                        turn_id=user_turn_id,
                        evidence_refs=[user_ev.evidence_id],
                    )
                    step_events.append(inter_ev)
                    preview_current_topic.topic_status = "SystemInterrupted"

                act_ev = EventFactory.create_topic_status_changed_event(
                    topic=target_t,
                    new_status="Ongoing",
                    turn_id=user_turn_id,
                    evidence_refs=[user_ev.evidence_id],
                )
                step_events.append(act_ev)
                target_t.topic_status = "Ongoing"
                preview_state.current_topic_id = target_t.topic_id
                next_topic = target_t
                selected_op_str = "switch_another_topic"
                transition = QuestionTransition(
                    kind="user_requested_switch",
                    from_topic_title=current_topic.topic_content,
                    to_topic_title=target_t.topic_content,
                    user_facing_reason=f"Per your request, transitioning to: '{target_t.topic_content}'",
                )
            else:
                next_topic = current_topic
                selected_op_str = "maintain_current_topic"
                transition = QuestionTransition(kind="maintain")

        elif intent_decision.needs_confirmation:
            preview_current_topic = self._require_topic(preview_state, current_topic.topic_id)
            next_topic = preview_current_topic
            preview_state.current_topic_id = preview_current_topic.topic_id
            selected_op_str = "maintain_current_topic"
            transition = QuestionTransition(
                kind="confirm_control",
                user_facing_reason="Please confirm whether you would like to adjust the discussion direction.",
            )

        else:
            # Regular content answer / dialogue flow
            preview_current_topic = self._require_topic(preview_state, current_topic.topic_id)
            if preview_current_topic.topic_status == "Completed":
                # Current topic completed; check remaining non-terminal topics
                non_term = [t for t in preview_state.get_all_topics() if t.topic_status not in {"Completed", "UserInterrupted"}]
                if not non_term:
                    proj_event = EventFactory.create_project_status_changed_event(
                        project_id=self.project_id,
                        old_status=state.project_status,
                        new_status="Completed",
                        turn_id=user_turn_id,
                    )
                    step_events.append(proj_event)
                    is_interview_finished = True
                    finish_msg = "All core requirement topics have been successfully completed. The interview is now complete. Thank you for your valuable time and participation!"
                    next_topic = None
                    selected_op_str = "complete_current_topic"
                else:
                    views = StateView(preview_state, evidence_refs=all_known_evidences).get_scheduling_views(
                        current_turn_idx=next_turn_idx,
                        affected_topics=interpretation.affected_existing_topics,
                        targeted_slot_ids=targeted_slot_ids,
                        clarified_slot_ids=clarified_slot_ids,
                        explored_topic_ids=explored_topic_ids,
                        forced_deepen_target=None,
                        allow_explore=True,
                    )
                    decision = self.scheduler.schedule(
                        preview_state,
                        views,
                        None,
                        user_turn_id,
                        evidence_refs=all_known_evidences,
                        targeted_slot_ids=targeted_slot_ids,
                    )
                    if decision:
                        best_t = self._require_topic(preview_state, decision.selected_topic_id)
                        act_ev = EventFactory.create_topic_status_changed_event(
                            topic=best_t,
                            new_status="Ongoing",
                            turn_id=user_turn_id,
                            evidence_refs=[user_ev.evidence_id],
                        )
                        step_events.append(act_ev)
                        best_t.topic_status = "Ongoing"
                        next_topic = best_t
                        selected_op_str = "complete_current_topic"
                        transition = QuestionTransition(
                            kind="scheduler_switched",
                            from_topic_title=current_topic.topic_content,
                            to_topic_title=best_t.topic_content,
                            user_facing_reason=f"Current topic is completed, transitioning to: '{best_t.topic_content}'",
                        )
                    else:
                        raise RuntimeError(
                            f"Scheduler returned no candidate topic, but non-terminal topics remain: {[t.topic_id for t in non_term]}"
                        )
            else:
                # Current topic not completed: schedule next topic based on multi-factor utility
                views = StateView(preview_state, evidence_refs=all_known_evidences).get_scheduling_views(
                    current_turn_idx=next_turn_idx,
                    affected_topics=interpretation.affected_existing_topics,
                    targeted_slot_ids=targeted_slot_ids,
                    clarified_slot_ids=clarified_slot_ids,
                    explored_topic_ids=explored_topic_ids,
                    forced_deepen_target=forced_deepen_target,
                    allow_explore=allow_explore,
                )
                recovery_topic_id = (
                    current_topic.topic_id
                    if (forced_deepen_target is not None or not allow_explore)
                    else None
                )
                decision = self.scheduler.schedule(
                    preview_state,
                    views,
                    current_topic,
                    user_turn_id,
                    recovery_topic_id=recovery_topic_id,
                    evidence_refs=all_known_evidences,
                    targeted_slot_ids=targeted_slot_ids,
                )
                if decision:
                    best_t = self._require_topic(preview_state, decision.selected_topic_id)
                    if best_t.topic_id != current_topic.topic_id:
                        inter_ev = EventFactory.create_topic_status_changed_event(
                            topic=current_topic,
                            new_status="SystemInterrupted",
                            turn_id=user_turn_id,
                            evidence_refs=[user_ev.evidence_id],
                        )
                        act_ev = EventFactory.create_topic_status_changed_event(
                            topic=best_t,
                            new_status="Ongoing",
                            turn_id=user_turn_id,
                            evidence_refs=[user_ev.evidence_id],
                        )
                        step_events.extend([inter_ev, act_ev])
                        preview_current_topic.topic_status = "SystemInterrupted"
                        best_t.topic_status = "Ongoing"
                        next_topic = best_t
                        selected_op_str = "switch_another_topic"
                        transition = QuestionTransition(
                            kind="scheduler_switched",
                            from_topic_title=current_topic.topic_content,
                            to_topic_title=best_t.topic_content,
                            user_facing_reason=f"Transitioning to: '{best_t.topic_content}'",
                        )
                    else:
                        next_topic = current_topic
                        selected_op_str = "maintain_current_topic"
                        transition = QuestionTransition(kind="maintain")
                else:
                    non_term = [t for t in preview_state.get_all_topics() if t.topic_status not in {"Completed", "UserInterrupted"}]
                    if non_term:
                        raise RuntimeError(
                            f"Scheduler returned no candidate topic, but non-terminal topics remain: {[t.topic_id for t in non_term]}"
                        )
                    proj_event = EventFactory.create_project_status_changed_event(
                        project_id=self.project_id,
                        old_status=state.project_status,
                        new_status="Completed",
                        turn_id=user_turn_id,
                    )
                    step_events.append(proj_event)
                    is_interview_finished = True
                    finish_msg = "All core requirement topics have been sufficiently explored. The interview is now complete. Thank you for your valuable time and participation!"
                    next_topic = None
                    selected_op_str = "end_current_topic"

        if is_interview_finished or next_topic is None:
            StateReducer.apply(state, step_events, known_evidence_ids=known_ev_ids, strict_validation=True)

            # Build unified decision for finish/exit
            unified_dec = UnifiedDecisionRecord(
                decision_id=decision_id,
                turn_id=user_turn_id,
                intent={
                    "type": intent_decision.intent,
                    "confidence": intent_decision.confidence,
                    "verification_outcome": intent_decision.verification_outcome,
                    "verification_confidence": intent_decision.verification_confidence,
                    "verification_explanation": intent_decision.verification_explanation,
                },
                scheduler=decision.model_dump() if decision else {},
                strategy={"code": "finish", "target_slot_ids": []},
            )

            # Validate state invariants before writing state
            all_accumulated_events = self.store.load_state_events(self.project_id) + step_events
            all_decisions = self.store.load_decisions(self.project_id) + [unified_dec]
            inv_errors = StateInvariantValidator.validate_all(
                state=state,
                known_evidence_ids=known_ev_ids,
                events=all_accumulated_events,
                decisions=all_decisions,
            )
            if inv_errors:
                raise StateInvariantError(rule_id=0, message="; ".join(inv_errors))

            self.store.append_evidence(self.project_id, user_ev)
            self.store.append_state_events(self.project_id, step_events)
            self.store.append_decision(self.project_id, unified_dec)
            self.store.save_state(state)

            return StepResult(
                project_id=self.project_id,
                turn_index=state.turn_index,
                is_finished=True,
                finish_message=finish_msg,
                next_question=finish_msg,
                selected_operation=selected_op_str,
                state_events=step_events,
            )

        active_topic = next_topic
        already_ongoing_in_events = any(
            e.entity_id == active_topic.topic_id
            and e.event_type == "topic_status_changed"
            and e.after.get("topic_status") == "Ongoing"
            for e in step_events
        )
        if not already_ongoing_in_events and active_topic.topic_status != "Ongoing":
            act_event = EventFactory.create_topic_status_changed_event(
                active_topic, "Ongoing", turn_id=user_turn_id, evidence_refs=[user_ev.evidence_id]
            )
            step_events.append(act_event)

        # Apply accumulated turn events in memory to compute latest state for question generation
        StateReducer.apply(state, step_events, known_evidence_ids=known_ev_ids, strict_validation=True)

        final_active_topic = self._require_topic(state, active_topic.topic_id)

        # Select strategy plan based on concrete state signals
        explored_topic_ids = {
            t.topic_id
            for t in all_turns
            if t.role == "Interviewer" and t.metadata.get("strategy") == "explore" and t.topic_id
        }
        clarified_slot_ids = self._compute_clarified_slot_ids(
            state=state,
            all_turns=all_turns,
            current_user_turn_id=user_turn_id,
            current_turn_idx=next_turn_idx,
        )
        plan = self.strategy_selector.select_plan(
            state=state,
            topic=final_active_topic,
            intent_decision=intent_decision,
            scheduler_decision=decision,
            transition_from_topic_id=current_topic.topic_id if final_active_topic.topic_id != current_topic.topic_id else None,
            evidence_refs=all_known_evidences,
            forced_deepen_target=forced_deepen_target if final_active_topic.topic_id == current_topic.topic_id else None,
            allow_explore=allow_explore if final_active_topic.topic_id == current_topic.topic_id else True,
            explored_topic_ids=explored_topic_ids,
            targeted_slot_ids=targeted_slot_ids,
            clarified_slot_ids=clarified_slot_ids,
        )

        # Build unified decision record for this turn
        target_slot_ids = plan.target_slot_ids or plan.target_conflict_slot_ids
        unified_dec = UnifiedDecisionRecord(
            decision_id=decision_id,
            turn_id=user_turn_id,
            intent={
                "type": intent_decision.intent,
                "confidence": intent_decision.confidence,
                "verification_outcome": intent_decision.verification_outcome,
                "verification_confidence": intent_decision.verification_confidence,
                "verification_explanation": intent_decision.verification_explanation,
            },
            scheduler=decision.model_dump() if decision else {
                "selected_topic_id": final_active_topic.topic_id,
                "selected_topic_number": final_active_topic.topic_number,
                "candidate_scores": [],
            },
            strategy={
                "code": plan.strategy,
                "target_slot_ids": target_slot_ids,
                "deepening_reason": plan.deepening_reason,
            },
        )

        if state.turn_index >= self.config.runtime.max_turns:
            all_accumulated_events = self.store.load_state_events(self.project_id) + step_events
            all_decisions = self.store.load_decisions(self.project_id) + [unified_dec]
            inv_errors = StateInvariantValidator.validate_all(
                state=state,
                known_evidence_ids=known_ev_ids,
                events=all_accumulated_events,
                decisions=all_decisions,
            )
            if inv_errors:
                raise StateInvariantError(rule_id=0, message="; ".join(inv_errors))

            self.store.append_evidence(self.project_id, user_ev)
            self.store.append_state_events(self.project_id, step_events)
            self.store.append_decision(self.project_id, unified_dec)
            self.store.save_state(state)

            return StepResult(
                project_id=self.project_id,
                turn_index=state.turn_index,
                current_topic_id=final_active_topic.topic_id,
                current_topic_number=final_active_topic.topic_number,
                current_topic_content=final_active_topic.topic_content,
                next_question="",
                termination_reason="max_turns_reached",
                termination_message=(
                    f"Maximum interview turn limit ({self.config.runtime.max_turns}) reached. "
                    "Execution stopped with the project left incomplete."
                ),
                selected_strategy=plan.strategy,
                selected_operation=selected_op_str,
                state_events=step_events,
            )

        gen_input = self.context_builder.build(
            state=state,
            plan=plan,
            turns=all_turns,
            evidences=all_known_evidences,
            transition=transition,
            context_budget=self.config.context_budget,
        )

        next_question, strat_code = await self.question_generator.generate_from_input(
            input_data=gen_input,
            turn_id=user_turn_id,
        )

        # Mandatory Invariant validation before committing to disk
        all_accumulated_events = self.store.load_state_events(self.project_id) + step_events
        all_decisions = self.store.load_decisions(self.project_id) + [unified_dec]
        inv_errors = StateInvariantValidator.validate_all(
            state=state,
            known_evidence_ids=known_ev_ids,
            events=all_accumulated_events,
            decisions=all_decisions,
        )
        if inv_errors:
            raise StateInvariantError(rule_id=0, message="; ".join(inv_errors))

        # All LLM operations and validation succeeded! Commit state changes to disk atomically!
        self.store.append_evidence(self.project_id, user_ev)
        self.store.append_state_events(self.project_id, step_events)
        self.store.append_decision(self.project_id, unified_dec)

        bot_turn_id = IdFactory.create_turn_id(state.turn_index + 1)
        bot_turn = TurnRecord(
            turn_id=bot_turn_id,
            turn_index=state.turn_index,
            topic_id=final_active_topic.topic_id,
            role="Interviewer",
            message_content=next_question,
            metadata={
                "decision_id": decision_id,
                "strategy": strat_code,
                "target_topic_id": final_active_topic.topic_id,
                "target_slot_ids": target_slot_ids,
                "deepening_reason": plan.deepening_reason,
                "intent": intent_decision.intent,
                "needs_confirmation": intent_decision.needs_confirmation,
                "verification_outcome": intent_decision.verification_outcome,
                "operation": selected_op_str,
                "topic_number": final_active_topic.topic_number,
            },
        )
        self.store.append_turn(self.project_id, bot_turn)
        self.store.save_state(state)

        return StepResult(
            project_id=self.project_id,
            turn_index=state.turn_index,
            current_topic_id=final_active_topic.topic_id,
            current_topic_number=final_active_topic.topic_number,
            current_topic_content=final_active_topic.topic_content,
            next_question=next_question,
            is_finished=False,
            selected_strategy=strat_code,
            selected_operation=selected_op_str,
            state_events=step_events,
        )

    async def finish(self) -> StepResult:
        """Explicitly finalizes an already completed interview project by verifying preconditions, running invariants, and exporting artifacts.

        Does NOT mutate or generate any business state events.
        """
        state = self.store.load_state(self.project_id)
        evidences = self.store.load_evidences(self.project_id)
        events = self.store.load_state_events(self.project_id)
        decisions = self.store.load_decisions(self.project_id)
        known_ev_ids = {e.evidence_id for e in evidences}

        # 1. Precondition checks for finalization
        all_topics = state.get_all_topics()
        terminal_statuses = {"Completed", "UserInterrupted"}
        non_terminal_topics = [t for t in all_topics if t.topic_status not in terminal_statuses]

        if (
            state.project_status != "Completed"
            or state.current_topic_id is not None
            or len(non_terminal_topics) > 0
        ):
            msg = (
                f"Project '{self.project_id}' is not ready for finalization: "
                f"project_status='{state.project_status}', current_topic_id='{state.current_topic_id}', "
                f"non_terminal_topics={[(t.topic_id, t.topic_status) for t in non_terminal_topics]}."
            )
            err = RunError(
                error_id=IdFactory.create_event_id(),
                turn_id=None,
                module="Pipeline.finish",
                error_type="project_not_ready_for_finalization",
                message=msg,
                recoverable=True,
            )
            self.store.append_error(self.project_id, err)
            raise ProjectNotReadyForFinalizationError(msg)

        # 2. Mandatory Invariant Gate check
        inv_errors = StateInvariantValidator.validate_all(
            state=state,
            known_evidence_ids=known_ev_ids,
            events=events,
            decisions=decisions,
        )
        if inv_errors:
            err = RunError(
                error_id=IdFactory.create_event_id(),
                turn_id=None,
                module="Pipeline.finish",
                error_type="state_invariant_error",
                message="; ".join(inv_errors),
                recoverable=False,
            )
            self.store.append_error(self.project_id, err)
            raise StateInvariantError(rule_id=0, message="; ".join(inv_errors))

        # 3. Export final_state.json and summary.md (Pure finalization / idempotent)
        self.store.save_final_state(state)
        summary_md_content = SummaryGenerator.generate_markdown(state, evidences)
        self.store.save_summary_md(self.project_id, summary_md_content)

        return StepResult(
            project_id=self.project_id,
            turn_index=state.turn_index,
            is_finished=True,
            finish_message="Interview session successfully completed and archived.",
            next_question="",
            state_events=[],
        )

from pathlib import Path
from typing import Optional
import json
import logging
from config import ContextBudgetConfig
from llm.client import LLMClient
from llm.exceptions import LLMConfigurationError, LLMOutputError, LLMTransportError
from models.interpretation import TopicDigest
from models.run_record import RunError
from models.state import ProjectState, TopicState
from models.strategy import (
    KnownInfoDigest,
    QuestionGenerationInput,
    QuestionPlan,
    SlotDigest,
    StrategyCode,
    TargetContext,
    TopicCatalogItem,
)
from services.context_budget_manager import ContextBudgetManager
from services.id_factory import IdFactory
from .strategy_selector import STRATEGY_INSTRUCTIONS, StrategySelector

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generates interviewer remarks / questions based on QuestionPlan, Topic state, and conversation history."""

    def __init__(
        self,
        llm_client: LLMClient,
        strategy_selector: StrategySelector,
        prompts_dir: Path | str = "prompts",
        context_budget: Optional[ContextBudgetConfig] = None,
    ):
        self.llm_client = llm_client
        self.strategy_selector = strategy_selector
        self.prompts_dir = Path(prompts_dir)
        self.budget_manager = ContextBudgetManager(context_budget)

    def _load_template(self) -> str:
        prompt_path = self.prompts_dir / "remarks_generation.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template 'remarks_generation.txt' not found in {self.prompts_dir}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()


    def _build_entire_interview_info_slots_from_known_info(
        self,
        project_known_info: list[KnownInfoDigest],
    ) -> dict[str, dict]:
        entire_info: dict[str, dict] = {}
        for item in project_known_info:
            topic_key = item.topic_content
            slots_repr = [
                {
                    "item_name": s.key,
                    "recorded_value": s.value if s.value is not None else "[Not provided]",
                }
                for s in item.filled_slots
            ]
            entire_info[topic_key] = {"items": slots_repr}
        return entire_info

    def _format_target_and_evidence(self, input_data: QuestionGenerationInput) -> tuple[str, str]:
        """Formats target context and supporting evidence into separate clear, ID-free semantic instructions for the LLM."""
        plan = input_data.plan
        strat = plan.strategy
        tc = input_data.target_context

        target_desc = "Advance the interview dialogue smoothly."
        evidence_desc = "(No specific evidence attached)"

        if strat == "fill_gap":
            if tc and tc.target_slots:
                target_slot = tc.target_slots[0]
                target_desc = f"Inquire naturally about the missing information item '{target_slot.semantic_key}'."
                if target_slot.evidence_snippets:
                    evidence_desc = "\n".join(f'- "{snip.content}"' for snip in target_slot.evidence_snippets)
            else:
                target_desc = "Inquire about key missing information items under the active topic."

        elif strat == "deepen":
            if tc and tc.target_slots:
                target_slot = tc.target_slots[0]
                val_repr = target_slot.current_value or "[Not provided]"
                reason = target_slot.deepening_reason or plan.deepening_reason or "uncertain_value"
                if reason == "uncertain_value":
                    reason_text = "The recorded value is tentative or subject to confirmation. Inquire to clarify the final rule or confirm that it remains pending."
                elif reason == "added_slot_needs_clarification":
                    reason_text = "This item was newly introduced. Inquire forward into its concrete operational rules, applicable conditions, or boundary exceptions; do NOT ask meta-confirmation questions (e.g. whether it is finalized or tentative) unless the interviewee explicitly indicates inability to decide."
                elif reason == "clarify_or_defer":
                    reason_text = "The previous inquiry did not resolve this specific gap. Explicitly ask the interviewee whether they want to supply this missing detail now or record it as undecided / tentative for now."
                elif reason in ("unaddressed_detail_slot", "detail_elaboration"):
                    reason_text = "This detail aspect (condition, exception, fallback, or boundary) is not yet explored. Inquire whether specific business rules or operational behaviors apply, treating established facts as context without asserting that an exception or rule already exists."
                else:
                    reason_text = "Probe deeper into concrete operational boundaries or conditions for this item."

                recent_qs = [
                    t.message_content
                    for t in input_data.recent_turns
                    if t.role == "Interviewer" and t.message_content
                ][-3:]

                lines = [
                    f"Target Item: '{target_slot.semantic_key}'",
                    f"Current Recorded Value: {val_repr}",
                    f"Clarification Reason: {reason_text}",
                ]
                if recent_qs:
                    lines.append("Recent questions asked (DO NOT repeat the same boundaries, rules, or questions):")
                    for rq in recent_qs:
                        lines.append(f"  - {rq}")
                target_desc = "\n".join(lines)

                if target_slot.evidence_snippets:
                    evidence_desc = "\n".join(f'- "{snip.content}"' for snip in target_slot.evidence_snippets)
            else:
                target_desc = "Probe deeper into specific details and operational scenarios for existing information under the active topic."

        elif strat == "resolve_conflict":
            if tc and tc.conflict_claims:
                lines = ["Objectively present and clarify conflicting statements, asking the interviewee to confirm the authoritative rule."]
                ev_lines = []
                for key, claims in tc.conflict_claims.items():
                    lines.append(f"Diverging records for '{key}':")
                    for idx, claim in enumerate(claims, 1):
                        lines.append(f"  Statement {idx}: {claim.value}")
                        for snip in claim.evidence_snippets:
                            ev_lines.append(f'- Statement {idx} supporting evidence: "{snip.content}"')
                target_desc = "\n".join(lines)
                if ev_lines:
                    evidence_desc = "\n".join(ev_lines)
            else:
                target_desc = "Objectively request the interviewee to clarify conflicting statements recorded under the active topic."

        elif strat == "verify":
            confirmed = tc.verify_facts if tc else []
            uncertain = tc.uncertain_facts if tc else []
            if confirmed or uncertain:
                lines = ["Synthesize and verify requirements under the active topic:"]
                if confirmed:
                    lines.append("Confirmed requirement points:")
                    for f in confirmed:
                        lines.append(f"  - {f['key']}: {f['value']}")
                if uncertain:
                    lines.append("Pending / tentative items (to be acknowledged as pending):")
                    for f in uncertain:
                        lines.append(f"  - {f['key']}: {f['value']} (Tentative/Pending confirmation)")
                lines.append(
                    "Ask the interviewee to confirm if this summary is complete and accurate, "
                    "or if any revisions or additions are needed before moving forward."
                )
                target_desc = "\n".join(lines)
                evidence_desc = "(The above points are synthesized from confirmed interview statements and recognized pending items)"
            else:
                target_desc = "Briefly summarize key requirements under the active topic and ask the interviewee if anything is missing or needs revision."

        elif strat == "confirm_control":
            intent_desc = plan.control_intent or "modify interview flow"
            target_topic_title = None
            if tc and tc.target_topics:
                target_topic_title = tc.target_topics[0].topic_content
            if target_topic_title:
                target_desc = f"Detected user intention to switch to '{target_topic_title}'. Politely ask a confirmation choice (switch to that topic or continue the active topic) without introducing new requirement inquiries."
            else:
                target_desc = f"Detected user intention to '{intent_desc}'. Politely confirm their choice without introducing new requirement inquiries."

        elif strat == "explore":
            target_desc = "Conduct open-ended exploration around core business scenarios, key objectives, and operational scope of the active topic."

        # If relations are present, append to target_desc
        if tc and tc.target_relations:
            rel_lines = ["[Associated Dependencies]: Please confirm and discuss the topic dependencies:"]
            for rel in tc.target_relations:
                desc = rel.get("description")
                if not desc:
                    desc = f"'{rel.get('target', '')}' depends on '{rel.get('source', '')}'"
                rel_lines.append(f"  - {desc}")
            if target_desc == "Advance the interview dialogue smoothly.":
                target_desc = "\n".join(rel_lines)
            else:
                target_desc = target_desc + "\n" + "\n".join(rel_lines)

        return target_desc, evidence_desc

    def _build_prompt(
        self,
        input_data: QuestionGenerationInput,
    ) -> str:
        """Builds prompt strictly from QuestionGenerationInput, without bypassing through state and without technical IDs."""
        plan = input_data.plan
        strat = plan.strategy
        if strat not in STRATEGY_INSTRUCTIONS:
            raise ValueError(f"Unknown question strategy: {strat!r}")
        inst = STRATEGY_INSTRUCTIONS[strat]

        target_text, evidence_text = self._format_target_and_evidence(input_data)

        # ID-free current slots data formatted with business language
        current_slots_data = [
            {
                "item_name": s.key,
                "recorded_value": s.value if s.value is not None else "[Not provided]",
                "is_required": "Required" if s.is_required else "Optional",
                "status": "Confirmed" if s.state == "filled" else ("Conflict" if s.state == "conflict" else ("Uncertain" if s.state == "uncertain" else "Not provided")),
            }
            for s in input_data.topic_slots
        ]
        if input_data.other_slots_omitted_count > 0:
            current_slots_data.append({
                "omitted_other_items_count": input_data.other_slots_omitted_count
            })

        # ID-free topic catalog list
        topics_list = [
            {"topic_title": k.topic_content}
            for k in input_data.topic_catalog
        ]

        entire_info = self._build_entire_interview_info_slots_from_known_info(
            input_data.project_known_info
        )
        template = self._load_template()

        convo_records = []
        for t in input_data.recent_turns:
            convo_records.append({
                "role": t.role,
                "content": t.message_content,
            })

        latest_user_answer = "[Initial interview round, no user response yet]"
        for t in reversed(input_data.recent_turns):
            if t.role == "Interviewee" and t.message_content:
                latest_user_answer = t.message_content
                break

        transition_text = ""
        if input_data.scheduler_transition and input_data.scheduler_transition.user_facing_reason:
            transition_text = input_data.scheduler_transition.user_facing_reason

        return (
            template
            .replace("{current_user_answer}", latest_user_answer)
            .replace("{current_topic_content}", str(input_data.topic.topic_content))
            .replace("{target}", target_text)
            .replace("{target_evidence}", evidence_text)
            .replace("{current_topic_info_slots}", json.dumps(current_slots_data, ensure_ascii=False))
            .replace("{topics_list}", json.dumps(topics_list, ensure_ascii=False))
            .replace("{entire_interview_info_slots}", json.dumps(entire_info, ensure_ascii=False))
            .replace("{transition}", transition_text)
            .replace("{current_topic_conversation_record}", json.dumps(convo_records, ensure_ascii=False))
            .replace("{strategy}", inst)
        )

    async def generate_from_input(
        self,
        input_data: QuestionGenerationInput,
        turn_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Main structured generation entrypoint consuming QuestionGenerationInput with context budgeting."""
        plan = input_data.plan

        effective_turn_id = turn_id
        if not effective_turn_id and input_data.recent_turns:
            effective_turn_id = getattr(input_data.recent_turns[-1], "turn_id", None)

        def _emit_error(error_type: str, message: str) -> None:
            if getattr(self.llm_client, "on_error", None):
                try:
                    run_err = RunError(
                        error_id=IdFactory.create_event_id(),
                        turn_id=effective_turn_id,
                        module="QuestionGenerator",
                        error_type=error_type,
                        message=message,
                        recoverable=True,
                    )
                    self.llm_client.on_error(run_err)
                except Exception as ex:
                    logger.warning("Failed to emit RunError: %s", ex)

        # 1. Prompt Construction and Context Budgeting
        try:
            target_text, evidence_text = self._format_target_and_evidence(input_data)
            if plan.strategy not in STRATEGY_INSTRUCTIONS:
                raise ValueError(f"Unknown question strategy: {plan.strategy!r}")
            inst = STRATEGY_INSTRUCTIONS[plan.strategy]
            template = self._load_template()

            budget_res = self.budget_manager.apply_budget(
                input_data=input_data,
                prompt_template=template,
                strategy_instruction=inst,
                target_text=target_text,
                evidence_text=evidence_text,
            )
        except Exception as exc:
            err_msg = f"Prompt construction failed: {exc}"
            logger.error("prompt_build_error: %s", err_msg)
            _emit_error("prompt_build_error", err_msg)
            raise RuntimeError(err_msg) from exc

        # 2. Context Budget Check
        if budget_res.is_exceeded:
            budget_config = self.budget_manager.config
            violated_constraints = []
            if budget_res.total_tokens > budget_config.max_prompt_tokens:
                violated_constraints.append(
                    f"total_prompt_tokens {budget_res.total_tokens} > "
                    f"max_prompt_tokens {budget_config.max_prompt_tokens}"
                )
            if budget_res.block_tokens["target_and_evidence"] > budget_config.max_target_evidence_tokens:
                violated_constraints.append(
                    "target_and_evidence tokens "
                    f"{budget_res.block_tokens['target_and_evidence']} > "
                    f"max_target_evidence_tokens {budget_config.max_target_evidence_tokens}"
                )
            if budget_res.block_tokens["current_topic_slots"] > budget_config.max_current_slots_tokens:
                violated_constraints.append(
                    "current_topic_slots tokens "
                    f"{budget_res.block_tokens['current_topic_slots']} > "
                    f"max_current_slots_tokens {budget_config.max_current_slots_tokens}"
                )
            err_msg = (
                f"Context budget exceeded: total estimated tokens {budget_res.total_tokens} "
                f"(max_prompt_tokens {budget_config.max_prompt_tokens}); "
                f"violated constraints: {', '.join(violated_constraints)}; "
                f"trim reasons: {', '.join(budget_res.trim_reasons)}."
            )
            logger.error("question_context_budget_exceeded: %s", err_msg)
            _emit_error("question_context_budget_exceeded", err_msg)
            raise RuntimeError(err_msg)

        metadata = {
            "budget": {
                "total_tokens": budget_res.total_tokens,
                "block_tokens": budget_res.block_tokens,
                "block_token_ratios": budget_res.block_token_ratios,
                "trimmed_items": budget_res.trimmed_items,
                "trim_reasons": budget_res.trim_reasons,
                "estimator": budget_res.estimator,
            }
        }

        # 3. LLM Completion Call
        try:
            response = await self.llm_client.complete_text(
                prompt=budget_res.final_prompt,
                turn_id=effective_turn_id,
                module="QuestionGenerator",
                prompt_name="remarks_generation",
                metadata=metadata,
            )
        except LLMOutputError as exc:
            err_msg = str(exc)
            logger.error("llm_output_error: %s", err_msg)
            # If not an LLMClient instance that already emitted on_error, emit it here
            if type(self.llm_client) is not LLMClient:
                _emit_error("llm_output_error", err_msg)
            raise
        except LLMConfigurationError as exc:
            err_msg = str(exc)
            logger.error("llm_configuration_error: %s", err_msg)
            if type(self.llm_client) is not LLMClient:
                _emit_error("llm_configuration_error", err_msg)
            raise
        except (LLMTransportError, ConnectionError, TimeoutError, OSError) as exc:
            err_msg = str(exc)
            logger.error("transport_error: %s", err_msg)
            if type(self.llm_client) is not LLMClient:
                _emit_error("transport_error", err_msg)
            raise
        except Exception as exc:
            err_msg = str(exc)
            logger.error("transport_error: %s", err_msg)
            if type(self.llm_client) is not LLMClient:
                _emit_error("transport_error", err_msg)
            raise

        # 4. Output Validation & Direct Return (No linguistic/regex filter on non-empty output)
        if not response or not response.strip():
            err_msg = "LLM returned empty or whitespace response."
            logger.warning("llm_output_error: %s", err_msg)
            _emit_error("llm_output_error", err_msg)
            raise LLMOutputError(err_msg)

        return response, plan.strategy


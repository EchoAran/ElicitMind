import json
import re
from pathlib import Path
from llm.client import LLMClient
from llm.exceptions import LLMOutputError
from models.dependency import DependencyEdge, PriorityResult, TopicPriorityItem
from models.state import SectionState, TopicState


class DependencyBuilder:
    """Discovers dependency edges between topics and calculates initial exploration priority sequence."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompts_dir: Path | str = "prompts",
        dep_weight: float = 0.6,
        sec_weight: float = 0.4,
    ):
        self.llm_client = llm_client
        self.prompts_dir = Path(prompts_dir)
        self.dep_weight = dep_weight
        self.sec_weight = sec_weight

    def _load_prompt_template(self) -> str:
        prompt_path = self.prompts_dir / "topic_dependency.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    async def build(
        self,
        sections: list[SectionState],
    ) -> PriorityResult:
        topics_data: list[dict] = []
        for sec in sections:
            for top in sec.topics:
                topics_data.append({
                    "topic_number": top.topic_number,
                    "topic_content": top.topic_content,
                    "section_number": sec.section_number,
                    "status": top.topic_status,
                })

        if not topics_data:
            return PriorityResult(edges=[], order=[], ranked_items=[])

        from llm.template import render_prompt

        template = self._load_prompt_template()
        prompt = render_prompt(
            template,
            {
                "{topics}": json.dumps(topics_data, ensure_ascii=False, indent=2),
            }
        )

        raw_edges = await self.llm_client.complete_json(
            prompt=prompt,
            turn_id="turn_0000",
            module="DependencyBuilder",
            prompt_name="topic_dependency",
        )
        if not isinstance(raw_edges, list):
            raise LLMOutputError(f"DependencyBuilder expected a JSON array, got {type(raw_edges).__name__}")

        num_set = {d["topic_number"] for d in topics_data}

        indeg: dict[str, int] = {d["topic_number"]: 0 for d in topics_data}
        clean_edges: list[DependencyEdge] = []
        seen_edges: set[tuple[str, str]] = set()

        for item in raw_edges:
            if not isinstance(item, dict):
                raise LLMOutputError(f"Each dependency edge must be a JSON object, got {type(item).__name__}")
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()

            if src not in num_set:
                raise LLMOutputError(f"Dependency edge references unknown source topic_number {item.get('source')!r}")
            if tgt not in num_set:
                raise LLMOutputError(f"Dependency edge references unknown target topic_number {item.get('target')!r}")
            if src == tgt:
                raise LLMOutputError(f"Dependency edge must connect two distinct topics, got self-loop on '{src}'")

            edge_pair = (src, tgt)
            if edge_pair not in seen_edges:
                seen_edges.add(edge_pair)
                clean_edges.append(DependencyEdge(source=src, target=tgt))
                indeg[tgt] = indeg.get(tgt, 0) + 1

        dmax = max(indeg.values()) if (indeg and max(indeg.values()) > 0) else 1
        unique_sections = sorted(list({d["section_number"] for d in topics_data}), key=self._parse_section_index)
        sn_to_pos = {sn: (i + 1) for i, sn in enumerate(unique_sections)}
        total_sections = len(unique_sections)

        ranked: list[TopicPriorityItem] = []
        for d in topics_data:
            f_dep = 1.0 - (indeg.get(d["topic_number"], 0) / dmax)
            pos = sn_to_pos.get(d["section_number"], 1)
            f_sec = ((total_sections - pos) / (total_sections - 1)) if total_sections > 1 else 1.0
            core_score = self.dep_weight * f_dep + self.sec_weight * f_sec

            ranked.append(
                TopicPriorityItem(
                    topic_number=d["topic_number"],
                    topic_content=d["topic_content"],
                    status=d["status"],
                    core=round(core_score, 4),
                )
            )

        ranked_sorted = sorted(
            ranked,
            key=lambda x: (-x.core, self._parse_topic_key(x.topic_number)),
        )
        order = [item.topic_number for item in ranked_sorted]

        return PriorityResult(
            edges=clean_edges,
            order=order,
            ranked_items=ranked_sorted,
        )

    @staticmethod
    def _parse_section_index(sec_num: str) -> int:
        match = re.match(r"^section-([1-9]\d*)$", sec_num)
        if not match:
            raise ValueError(f"Invalid section number format for ordering: '{sec_num}'")
        return int(match.group(1))

    @staticmethod
    def _parse_topic_key(topic_number: str) -> tuple[int, int]:
        match = re.match(r"^topic-([1-9]\d*)-([1-9]\d*)$", topic_number)
        if not match:
            raise ValueError(f"Invalid topic number format for ordering: '{topic_number}'")
        return (int(match.group(1)), int(match.group(2)))

# Elicitation Core

[English](README.md) | [中文说明](README_zh.md)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Elicitation Core** is an enterprise-grade core engine designed for software engineering **Requirements Elicitation** through semi-structured interviews. Powered by Large Language Models (LLMs), the system combines structured evidence chain tracking, event-driven state evolution, multi-factor dynamic topic scheduling, strategy-guided question planning, deterministic context budgeting, and crash-resilient session recovery into a robust, traceable, and reproducible foundation.

---

## Table of Contents

- [1. Core Features & Design Philosophy](#1-core-features--design-philosophy)
- [2. Architecture & Workflow](#2-architecture--workflow)
- [3. Project Structure](#3-project-structure)
- [4. Storage & Runtime Artifacts](#4-storage--runtime-artifacts)
- [5. Quick Start Guide](#5-quick-start-guide)
  - [5.1 Installation & Configuration](#51-installation--configuration)
  - [5.2 CLI Tools](#52-cli-tools)
- [6. Deep Dive into Core Mechanisms](#6-deep-dive-into-core-mechanisms)
  - [6.1 Event Sourcing & Two-Phase Reducer](#61-event-sourcing--two-phase-reducer)
  - [6.2 Seven-Factor Dynamic Scheduler](#62-seven-factor-dynamic-scheduler)
  - [6.3 Six-Dimension Strategy Planning & Isolated Prompting](#63-six-dimension-strategy-planning--isolated-prompting)
  - [6.4 Context Budgeting & Failure Exposure](#64-context-budgeting--failure-exposure)
  - [6.5 Invariant Guardrails & Mid-Turn Recovery](#65-invariant-guardrails--mid-turn-recovery)
- [License](#license)

---

## 1. Core Features & Design Philosophy

Traditional LLM-based interview systems suffer from **factual hallucination**, **conversational topic drift**, **context window explosion**, **unresolved requirement contradictions**, and **irrecoverable session crashes**. Elicitation Core addresses these challenges via six foundational pillars:

- 🎯 **Immutable Event Sourcing**: Every state transition (slot creation, refinement, conflict revision, topic lifecycle change, dependency addition) is modeled as a strongly typed atomic `StateEvent`. The pure-function `StateReducer` enables 100% deterministic state reconstruction from disk event logs.
- 🔗 **Full-Lifecycle Evidence Traceability**: Interviewee statements are captured as immutable `EvidenceRef` records. All slot revisions (`SlotRevision`) explicitly reference supporting evidence, ensuring that every extracted requirement fact is fully auditable.
- 🧭 **Intent Control & 7-Factor Scheduling**: Handles explicit user conversational control (topic switching, topic refusal, backtracking, early termination) with confidence thresholds; dynamically evaluates candidate topics across 7 factors (prior, readiness, gap, conflict, emergence, continuity, relevance) for optimal focus.
- 💡 **Strategy-Guided Question Planning**: Decouples *what to discuss* (scheduler target) from *how to ask* (question strategy), planning targeted inquiries across five runtime strategies: `explore`, `fill_gap`, `deepen`, `resolve_conflict`, and `confirm_control`.
- 🛡️ **Strict Context Budgeting & Failure Exposure**: Employs a 13-block isolated prompt contract with P7~P4 priority-based progressive context trimming. If budget is exceeded or model calls fail after retries, the engine immediately halts progression and exposes the error without masking failures or committing incomplete state.
- 🔒 **15 State Invariant Guardrails**: Formal business validation rules eliminate orphan slots, cross-topic data leaks, invalid state transitions, and dangling dependencies.

---

## 2. Architecture & Workflow

The interview lifecycle is orchestrated by `ElicitationPipeline`, encompassing **Initialization**, **Turn Step Loop**, and **Session Finalization**:

```mermaid
flowchart TD
    A[Initial Requirement / Initial Description] -->|Pipeline.initialize| B[FrameworkGenerator]
    B --> C[ProjectPrefiller]
    C --> D[DependencyBuilder]
    D --> E[(Baseline state.initial.json)]

    E --> F[Generate Round 0 Question]
    F --> G[Interviewee Response]

    subgraph "Per-Turn Execution Loop (Pipeline.step)"
        G --> H[Record Turn & Create EvidenceRef]
        H --> I[IntentController]
        I --> J[EvidenceInterpreter]
        J --> K[SlotFiller: Extraction & Conflicts]
        J --> L[StructureEvolver: Emergent Topics & Dependencies]
        K & L --> M[Two-Phase StateReducer Transaction]
        M --> N{Explicit User Intent Override?}
        N -- Yes --> O[Execute Control Branch]
        N -- No --> P[Seven-Factor Scheduler]
        O & P --> Q[StrategySelector]
        Q --> R[QuestionContextBuilder]
        R --> S[ContextBudgetManager]
        S --> T[QuestionGenerator: 13 Isolated Blocks]
        T -->|Normal| V[LLM Inference]
        V --> W[StateInvariantValidator Guardrails]
    end

    W -->|Next Turn| G
    W -->|Session Complete / Terminated| X[Await explicit user finalization]
    X -->|scripts/finish.py calls Pipeline.finish| Y[(Final final_state.json)]
    X -->|scripts/finish.py calls Pipeline.finish| Z[Generate Requirements summary.md]
```

---

## 3. Project Structure

```text
semi_structured_interview_fse/
├── configs/
│   ├── default.example.yaml            # Canonical configuration template with full inline docs
│   └── default.yaml                    # Active local configuration file
├── docs/
│   └── configuration.md                # Comprehensive configuration guide & tuning reference
├── prompts/                            # System prompt templates
│   ├── framework_generation.txt        # Initial framework generation
│   ├── initial_slots_filling.txt       # Initial slot prefilling
│   ├── topic_dependency.txt            # Initial dependency identification
│   ├── intent_detection.txt            # User control intent detection
│   ├── evidence_interpretation.txt     # Cross-topic interpretation & emergence
│   ├── emergent_topic_resolution.txt   # Emergent topic creation & deduplication
│   ├── slots_filling.txt               # Slot extraction & conflict tagging
│   └── remarks_generation.txt          # Isolated strategy question generation
├── src/
│   ├── config.py                   # Pydantic V2 hierarchical configuration models
│   ├── pipeline.py                 # Core workflow orchestrator (ElicitationPipeline)
│   ├── llm/                        # Async LLM client & offline replay client
│   │   ├── client.py               # Async LLM transport with exponential retries
│   │   ├── replay_client.py        # Deterministic offline replay client from logs
│   │   ├── schemas.py              # LLM call schemas & JSON structured contracts
│   │   └── exceptions.py           # Exception hierarchy
│   ├── models/                     # Domain core data models (Pydantic V2)
│   │   ├── state.py                # ProjectState, SectionState, TopicState, SlotState, SlotRevision
│   │   ├── event.py                # StateEvent, EvidenceRef (Event sourcing & traceability)
│   │   ├── turn.py                 # TurnRecord, TurnPair (Dialogue turns)
│   │   ├── dependency.py           # DependencyEdge, TopicPriorityItem, PriorityResult
│   │   ├── interpretation.py       # EvidenceInterpretation, AffectedTopic, EmergentTopicCandidate
│   │   ├── scheduling.py           # IntentDecision, TopicSchedulingView, TopicScore, SchedulerDecision
│   │   ├── strategy.py             # StrategyCode, QuestionPlan, QuestionGenerationInput, TargetContext
│   │   ├── run_record.py           # UnifiedDecisionRecord, LLMCallRecord, RunError (Audit models)
│   │   └── updates.py              # StepResult, SlotUpdateProposal, OperationSelectionResult
│   ├── storage/                    # Storage layer: atomic snapshot writes & append-only JSONL
│   │   └── project_store.py
│   ├── services/                   # Pure service algorithms
│   │   ├── id_factory.py           # Unique identifier factory
│   │   ├── event_factory.py        # Strongly-typed immutable event builder
│   │   ├── state_reducer.py        # Two-phase transactional state reducer
│   │   ├── state_view.py           # Read-only state view & multi-factor metrics extractor
│   │   ├── question_context_builder.py # Question target context & payload assembly
│   │   ├── context_budget_manager.py   # Token budget manager & progressive P7~P4 trimming
│   │   ├── validators.py           # 15 Global state invariant validators
│   │   └── summary_generator.py    # Final Markdown requirements specification exporter
│   ├── initialization/             # Initialization domain
│   │   ├── framework_generator.py  # Outline & section generator
│   │   ├── prefiller.py            # Initial slot prefiller
│   │   └── dependency_builder.py   # Initial topological dependency builder
│   └── runtime/                    # Runtime interaction & reasoning domain
│       ├── evidence_interpreter.py # Evidence interpretation & cross-topic association
│       ├── structure_evolver.py    # Emergent topic resolution & dynamic dependencies
│       ├── slot_filler.py          # Slot extraction, refinement & conflict marking
│       ├── intent_controller.py    # Intent detection & confidence gating
│       ├── scheduler.py            # 7-Factor dynamic utility scheduler
│       ├── strategy_selector.py    # 6-Dimension question strategy selector
│       └── question_generator.py   # 13-Block isolated prompt builder & generation
├── scripts/                            # Operational CLI tools
│   ├── init_project.py                 # Initialize a new interview project
│   ├── step.py                         # Advance dialogue turn
│   ├── finish.py                       # Explicitly finalize and generate report
│   ├── resume.py                       # Crash recovery & session resumption
│   ├── inspect_state.py                # Inspect state snapshot & slot completion
│   └── replay.py                       # Offline state & LLM call deterministic replay
└── pyproject.toml                      # Build configuration & dependencies
```

---

## 4. Storage & Runtime Artifacts

Every interview session stores all state snapshots, events, and audit logs inside `runs/{project_id}/`:

| File | Type | Format & Purpose |
|---|---|---|
| `input.json` | JSON | Project creation metadata (project name, initial requirements) |
| `state.initial.json` | JSON | Baseline state snapshot immediately following Round 0 initialization |
| `state.json` | JSON | Latest live project state snapshot (can be deleted and 100% reconstructed from events) |
| `final_state.json` | JSON | Immutable final snapshot written when the user runs `scripts/finish.py` |
| `summary.md` | Markdown | Structured requirements specification generated when the user runs `scripts/finish.py` |
| `state_events.jsonl` | JSONL | Strictly monotonic, append-only immutable state event stream (Event Sourcing source of truth) |
| `evidence.jsonl` | JSONL | Raw interviewee statements with structural evidence IDs and turn indexing |
| `turns.jsonl` | JSONL | Bi-directional dialogue turn history (Interviewer / Interviewee) with strategy metadata |
| `decisions.jsonl` | JSONL | Unified decision logs recording Intent + Scheduler + Strategy contexts and scores |
| `llm_calls.jsonl` | JSONL | Complete audit log for every LLM interaction (prompts, raw responses, budgets, latency) |
| `errors.jsonl` | JSONL | Audit log of recoverable and fatal runtime exceptions with stack traces |

---

## 5. Quick Start Guide

### 5.1 Installation & Configuration

#### 1. Setup Environment
Python 3.10+ is required. We recommend using a virtual environment:

```bash
# Clone the repository
git clone <repo_url>
cd semi_structured_interview_fse

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

#### 2. Initialize Configuration
Configuration templates reside in `configs/`. Copy the example file before the first run:

```bash
# Linux / macOS
cp configs/default.example.yaml configs/default.yaml

# Windows PowerShell
Copy-Item configs/default.example.yaml configs/default.yaml
```

Set your LLM API Key in your environment (or edit `configs/default.yaml` directly):
```bash
# Linux / macOS
export LLM_API_KEY="your-api-key"

# Windows PowerShell
$env:LLM_API_KEY="your-api-key"
```

> [!NOTE]
> For complete field specifications, constraints, and scenario tuning guides, refer to [docs/configuration.md](docs/configuration.md).

---

### 5.2 CLI Tools

The system provides 6 standard CLI tools supporting end-to-end interaction and offline operations:

#### 1. Initialize Interview (`init_project.py`)
Prepare an input JSON file, for example `project_input.json`:
```json
{
  "project_name": "Lab Equipment Sharing Platform",
  "initial_requirements": "A university research equipment reservation system supporting online booking, hourly billing, and conflict approvals."
}
```
Then initialize the project with `--input` (or `-i`):
```bash
python scripts/init_project.py --input project_input.json
```
Use `--config <CONFIG_PATH>` (or `-c`) to override the default `configs/default.yaml` configuration.

*Outputs the generated `project_id` (e.g. `proj_20260829_abc123`) and the opening exploratory question.*

#### 2. Advance Dialogue Turn (`step.py`)
```bash
python scripts/step.py \
  --project-id <PROJECT_ID> \
  --answer "The platform serves faculty, students, and external labs. Internal users get free quotas while external labs pay standard hourly fees."
```

#### 3. Finalize and Generate the Report (`finish.py`)
After `step.py` reports that the interview has ended, explicitly archive the completed project and generate its report:
```bash
python scripts/finish.py --project-id <PROJECT_ID>
```
Use `--config <CONFIG_PATH>` (or `-c`) if the project uses a non-default configuration file.

This command calls `Pipeline.finish()`. Finalization succeeds only when the persisted project is already `Completed`, `current_topic_id` is empty, and every topic is either `Completed` or `UserInterrupted`. Otherwise the command exits with `ProjectNotReadyForFinalizationError`; it never completes an active topic, changes business state, or emits business state events. On success it writes `final_state.json` and `summary.md` under `runs/<PROJECT_ID>/`. Report generation is never triggered automatically by `step.py`. Running the command again regenerates both files from the same persisted state and evidence without changing interview state.

#### 4. Inspect Current State (`inspect_state.py`)
```bash
python scripts/inspect_state.py --project-id <PROJECT_ID>
```
*Displays overall progress, section/topic statuses (Pending / Ongoing / SystemInterrupted / UserInterrupted / Completed), slot values, and dependency topology.*

#### 5. Inspect and Continue an Interrupted Session
If a session was interrupted by a network timeout, budget constraint, invalid model output, or abrupt termination, first validate and inspect the persisted state:
```bash
python scripts/resume.py --project-id <PROJECT_ID>
```
Without an additional flag, `resume.py` only validates the persisted snapshot, event stream, decisions, and turns, then displays the current session state.

If the Interviewee Turn was already recorded before the failure, continue it from the repository root with:
```bash
python scripts/resume.py --project-id <PROJECT_ID> --continue-pending
```
The command reuses the persisted answer and original Turn ID. If continuation fails again, the pending turn remains available for another retry; it is cleared only after the turn completes successfully.

> [!WARNING]
> Do not rerun `scripts/step.py --answer ...` for a pending turn. `step.py` uses the normal load path and may append a duplicate Interviewee Turn instead of resuming the recorded one.

#### 6. Offline Deterministic Replay (`replay.py`)
Supports deterministic comparison and state progression verification for historical interview sessions without making live LLM API calls:

```bash
# Mode 1: Event stream state replay (verifies StateReducer determinism)
python scripts/replay.py --project-id <PROJECT_ID> --mode state

# Mode 2: Full LLM log replay (verifies end-to-end pipeline determinism)
python scripts/replay.py --project-id <PROJECT_ID> --mode llm
```

---

## 6. Deep Dive into Core Mechanisms

### 6.1 Event Sourcing & Two-Phase Reducer
To prevent in-memory state drift from disk logs, `StateReducer.apply()` executes a **Two-Phase Transaction**:
1. **Pre-commit Validation**: Pre-executes pending `StateEvent` records on a deep copy of the state, checking invariant pre-conditions and sequence validity;
2. **Commit Phase**: In-place mutates the live state object and atomically appends events to `state_events.jsonl`.

### 6.2 Seven-Factor Dynamic Scheduler
When the interviewee does not explicitly dictate topic transitions, `Scheduler` ranks all incomplete topics:

$$\text{Score}(T) = w_1 \cdot \text{Prior} + w_2 \cdot \text{DepReadiness} + w_3 \cdot \text{UnresolvedGap} + w_4 \cdot \text{ConflictSignal} + w_5 \cdot \text{RecentEmergence} + w_6 \cdot \text{Continuity} + w_7 \cdot \text{UserRelevance}$$

- **Prior**: Baseline outline weight;
- **DepReadiness**: Readiness based on completed prerequisite topics;
- **UnresolvedGap**: Ratio of mandatory slots remaining empty;
- **ConflictSignal**: Priority boost for topics containing conflicting statements;
- **RecentEmergence**: Exploration bonus for newly emerged topics;
- **Continuity**: Momentum bonus to keep discussions focused on active topics;
- **UserRelevance**: Semantic relevance to the interviewee's latest response.

### 6.3 Strategy Planning & Isolated Prompting
`StrategySelector` determines the conversational objective for `QuestionGenerator`:
1. `explore`: Open-ended discovery when a topic is activated or revisited by user request;
2. `fill_gap`: Targeted inquiry targeting unresolved core slots lacking interview evidence;
3. `deepen`: Elaborates edge cases, clarifies uncertain values, or probes unaddressed detail slots;
4. `resolve_conflict`: Objectively highlights contradictory facts to establish ground truth;
5. `confirm_control`: Seeks explicit confirmation when user control intent has borderline confidence.

*(Note: Topic completion is strictly evidence-driven—a topic naturally completes when all required slots are satisfied with interview evidence, without artificial summary confirmation loops).*

Prompts use **13 strictly isolated semantic blocks**, preventing system instructions from leaking and stopping hallucinated cross-topic assumptions.

### 6.4 Context Budgeting & Failure Exposure
`ContextBudgetManager` applies a deterministic progressive trimming ladder (P7 ~ P4):
- **P7 Trim**: Topic catalog drops down to displaying only the active topic;
- **P6 Trim**: Cross-topic known facts drop oldest entries;
- **P5 Trim**: Conversation history trims oldest turn pairs (guaranteeing at least 1 recent pair);
- **P4 Trim**: Active topic definition drops optional (non-mandatory) slots.

If essential context still exceeds budget or model generation fails, the system immediately raises an exception and logs a structured `RunError` without committing Evidence, StateEvents, Decisions, Interviewer Turns, or `state.json`. The pending user turn remains safely resumable after adjusting configuration or environment parameters.

### 6.5 Invariant Guardrails & Mid-Turn Recovery
`StateInvariantValidator` continuously enforces 15 formal rules:
- At most 1 `Ongoing` topic at any time; 0 active topics upon completion;
- Bidirectional validity across slots, topics, and sections; no orphan slots;
- Dependency graph with no self-loops or duplicate edges;
- Monotonically increasing event, decision, and turn IDs;
- Conflict-tagged slots must possess corresponding conflict revisions.

If an unexpected crash occurs after the Interviewee Turn is written but before the turn is committed, `ElicitationPipeline.resume()` recognizes the final Interviewee Turn at `state.turn_index + 1` as pending. Calling `step()` without a new answer reuses the original answer and Turn ID. Conflicting replacement input is rejected, repeated failures retain the pending turn, and the pending marker is cleared only after the complete turn succeeds. Use `resume.py --continue-pending` as shown in the Quick Start guide; without that flag, `resume.py` only validates and displays the interrupted session.

---

## License

This project is licensed under the [MIT License](LICENSE).

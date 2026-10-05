# Phase 5 implementation and validation report

## Scope and files

The repository was inspected before edits. Its actual Phase 4 implementation
already provided the required agents, provider repairs, grounding, evidence
validation and label-free boundary. The initial suite ran **135 tests: 134 passed,
1 opt-in integration skipped**.

Added:

- `orchestration/__init__.py`, `state.py`, `contracts.py`, `orchestrator.py`,
  `routing.py`, `policy.py`, `evidence.py`, `events.py`;
- `rag/agents/revision.py` (restricted input for the existing Diagnostic Agent);
- `scripts/run_phase5.py`;
- `tests/orchestration_helpers.py`, `test_orchestration.py`,
  `test_orchestration_routing.py`, `test_orchestration_revision.py`,
  `test_orchestration_grounding.py`, `test_orchestration_boundaries.py`;
- `docs/phase5.md` and this report.

Modified:

- `rag/agents/clinical.py`: optional keyword-only revision input on the existing
  Diagnostic Agent; initial/Phase 4 invocation behavior is unchanged.
- `README.md`, `docs/architecture.md`: current Phase 5 status and documentation links.

No other existing runtime component was changed. In particular, the Phase 4
pipeline/runner, provider, original prompts, grounding validator, parser, domain
models, retrievers, embeddings, stores and evaluator remain unchanged. No package
was added. User task files and credentials were not modified. Validation logs,
local integrity script and smoke artifacts are retained under the project's
ignored `outputs/phase5/` directory, not as production runtime code.

## Architecture delivered

See [Phase 5](phase5.md) for full contracts and policy. The Orchestrator alone owns
run-local, typed `phase5-workflow-v1` state. It reuses all three Phase 4 agents;
retrieves and freezes the two evidence categories once; validates every accepted
patient/diagnostic/critic handoff; ties critiques to exact diagnostic/evidence
versions; and controls all transitions in application code.

Clinical revisions are limited to two and use the same Diagnostic Agent with a
structured, isolated prior-result/feedback input. Provider schema/grounding repairs
are counted separately. Exact repeated versions, exhausted budgets, failed
contracts, stale completions, provider failures, insufficient evidence and safety
flags terminate explicitly. The LLM cannot change these rules or bypass a gate.
Abstention inventories remain empty and no fallback diagnosis is manufactured.

## Test results

| Validation | Result |
| --- | --- |
| Complete existing + new unittest discovery | **185 run: 184 passed, 1 opt-in integration skipped** |
| New Phase 5 tests separately | **50 passed** |
| `test_phase4*.py` regression subset separately | **28 passed** |
| Existing abstention, provider, RAG, parser and reporting suites | Passed in full discovery |
| Phase 4 CLI | `--help` passed; existing mocked runner/reporting tests passed |
| Phase 5 CLI | `--help`, mocked execution/setup-failure tests and live one-case run passed technically |
| Syntax compilation | Passed for orchestration, rag, scripts and tests |
| `git diff --check` | Passed |
| `uv pip check` | All 65 installed packages compatible |

Phase 5 tests exercise success, supported-result routing, valid abstention,
insufficient retrieval, safety priority, revision success at either revision,
max-revision exhaustion, unchanged and nonadjacent repeated outputs, constrained
revision input, abstention after revision, independent schema/grounding gates,
provider repair separation, stale run/patient/stage-version/evidence completions,
immutable evidence, detached handoffs, separated source categories, invalid
provenance/labels/duplicates, provider failures, request/byte/time budgets, audit
ordering, state serialization, label isolation and both CLI success/failure.
Tests use fakes/mocks and the real provider parsing/repair path, not live calls.

Commands from the parent workspace (Windows environment):

```text
.venv\Scripts\python.exe -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai
.venv\Scripts\python.exe -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -p test_orchestration*.py
.venv\Scripts\python.exe -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -p test_phase4*.py
.venv\Scripts\python.exe -m compileall -q healthcare-agentic-ai/orchestration healthcare-agentic-ai/rag healthcare-agentic-ai/scripts healthcare-agentic-ai/tests
.venv\Scripts\python.exe healthcare-agentic-ai/scripts/run_phase4.py --help
.venv\Scripts\python.exe healthcare-agentic-ai/scripts/run_phase5.py --help
uv pip check
git diff --check
```

A second paid Phase 4 run was not needed: the existing mocked CLI regression tests
verify runner execution/reporting, the CLI imports successfully, and the runner's
source fingerprint is unchanged. The full suite's separate live integration test
remained opt-in/skipped; the controlled Phase 5 CLI smoke below was run explicitly.

## Controlled live smoke: one validation patient

```text
.venv\Scripts\python.exe healthcare-agentic-ai/scripts/run_phase5.py --queries 1 --top-k 1 --max-requests 14 --max-seconds 600 --output-dir healthcare-agentic-ai/outputs/phase5/task5-validation
```

Dedicated existing GPT_SOL credentials, GPT-5.6-Sol and cached local embeddings
were used. No credentials or raw provider errors were logged. This run queried the
prebuilt indexes; it did not rebuild, reset, upsert or modify the corpus.

| Observation | Actual result |
| --- | --- |
| Patient | `ddxplus:validate:1`, label-free validation features |
| Run ID | `9ea33440-afb5-46c1-807d-7fb7ab5777b5` |
| Patient Agent | Accepted after exact-copy validation |
| Retrieval | One patient case and one medical chunk; frozen provenance snapshot |
| Diagnostic Agent | **Valid abstention**, strict schema and unchanged grounding passed |
| Critic | Validated `revision_required` assessment with **2 safety flags** |
| Routing | Safety priority: **`blocked / safety_blocked`** |
| Clinical revisions | 0; safety flags prohibit a revision/acceptance route |
| Provider requests / repairs | 3 / 0 |
| Workflow time | 35.309 seconds (excludes runner/model setup) |
| Technical failure | None |
| CLI exit | 0: technically completed blocked workflow, **not clinical approval** |

Snapshot ID:
`b29f393952dbca27d23941dc7e7cde326f000b64bc92b8133238bb762e804c58`.

The audit contains workflow start, patient completion, evidence retrieval,
diagnostic start/schema validation, mandatory grounding pass, Critic completion,
deterministic routing and termination. There is **no finalized event**.

Artifacts:

- `outputs/phase5/task5-validation/sample_case_001.json`;
- `outputs/phase5/task5-validation/phase5_run_report.json`;
- `outputs/phase5/validation-artifacts/phase5-live-smoke.log`.

No diagnosis match, diagnostic correctness, effectiveness or clinical safety is
claimed. This single run validates the technical blocked/abstention path; live
revision success was not demonstrated. Revision paths are covered offline.

## RAG and protected-source integrity

Before implementation and after tests/live retrieval, every stored payload passed
its existing validator. The local audit scrolled every point with full payload
and vector, sorted by string point ID, and SHA-256 hashed UTF-8 canonical JSON
(`sort_keys=True`, compact separators, `ensure_ascii=False`). Counts and hashes
match exactly:

| Collection | Count before / after | SHA-256 before = after |
| --- | --- | --- |
| `ddxplus_patient_cases` | **1000 / 1000** | `601b130f9fe210dcd319ac85019d1735095a4fffee2b914a7a99e2383ab363be` |
| `medical_knowledge` | **87 / 87** | `ffe0608790a3c4ac12a31d8a4d3f66a6fafcb4ac8e605bd5621390e6a11a2085` |

These hashes use this audit's explicitly described serialization; compare before
to after here, not hashes from older reports using other serialization formats.
Actual stores: `healthcare-agentic-ai/data/qdrant/patient_cases` and parent
`data/medical_knowledge/qdrant`.

Protected-source SHA-256 values also match for Phase 4 pipeline/runner, grounding,
provider, parser/domain models, configuration, embeddings, patient ingestion/audit,
retrievers and stores. Original agent output contracts and prompts are unchanged.
The existing Diagnostic Agent's optional revision interface is the only intended
runtime modification outside the new Phase 5 modules.

Evidence is retained in `outputs/phase5/validation-artifacts/phase5-integrity-before.json`
and `phase5-integrity-after.json`; the read-only point-audit helper is
`phase5_integrity.py` in that same artifact directory. It opens existing stores
only, reads/validates/scrolls/counts, and closes clients; it never calls upsert/reset.

## Leakage and runtime boundary checks

- Tests attach secret primary/differential labels to evaluator-only PatientRecord
  fixtures and verify neither WorkflowState nor **any** initial/revision/critic
  provider request contains them.
- Whole records and raw label-bearing dictionaries are rejected before calls;
  unexpected state fields and label-bearing retrieval payloads fail closed.
- The Phase 5 runner uses only `include_labels=False`, with no evaluator imports.
- Live state has no `EvaluationLabels`, `ground_truth_pathology`, `PATHOLOGY`,
  `DIFFERENTIAL_DIAGNOSIS` or evaluator `labels` fields.
- No legacy generation runtime references were introduced in `rag/`, `scripts/`
  or `orchestration/`; the only healthcare generation deployment remains
  `gpt-5.6-sol`. No coding-agent model or tools were added to healthcare execution.

These are structural/sentinel leakage checks, not proof against arbitrary semantic
leakage in an independently changed dataset. Legitimate medical disease words in
features or source text are not scrubbed or misclassified as label leakage.

## Remaining limitations

- Research orchestration, **not clinical approval**; no HITL or clinical
  effectiveness evidence.
- Citation/provenance checks do not prove clinical entailment. Critic output is
  model feedback and can be wrong. The small corpus remains unchanged.
- Workflow deadlines are cooperative between synchronous calls, not hard process
  cancellation; per-request provider timeout remains separate.
- Repetition detection is exact structured-content hashing, not semantic-progress
  adjudication. Two revisions bound all remaining paraphrase loops.
- No durable resume, parallel executor, retrieval revisions, new clinical agents,
  feedback memory, prescribing or expanded corpus.
- No Phase 4/5 comparative diagnostic evaluation or live successful-revision claim.
  The one live case abstained and was safety-blocked as required.

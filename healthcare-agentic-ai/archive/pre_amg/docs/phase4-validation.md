# GPT-5.6-Sol migration validation

> Historical migration attempt. The later [abstention-contract follow-up](abstention-contract.md)
> completed one live case with a valid abstention under unchanged grounding. Its
> critic requested revisions. The failed results below are retained as historical
> evidence, not the current technical execution status.

Synthetic-data research only. **Migration implemented; successful real end-to-end
validation is not established. Do not proceed to Phase 5 on this evidence.**

## Audit and migration map

Inspected the workspace entry point, deployment notes, environment variable names,
healthcare source/configuration, agents, prompts, evaluation, scripts, tests,
dependencies, documentation and historical output scripts before editing.
Pre-existing user changes were preserved; `codeagent.py`, the workspace `.env`,
DDXPlus data, RAG algorithms and evaluator were not changed.

```text
scripts/run_phase4.py
    -> OpenAIConfig + OpenAIProvider (rag/llm/provider.py)
    -> PatientAgent (strict PatientState)
    -> PatientCaseRAG + MedicalKnowledgeRetriever (existing local stores)
    -> DiagnosticAgent (strict DiagnosticResult + original grounding guards)
    -> ClinicalCritic (strict ClinicalCritique + original grounding guards)
    -> phase4_evaluation.py (labels loaded after inference)
    -> atomic per-case/report JSON + CLI status
```

Only GPT-5.6-Sol generates healthcare outputs; no alternate backend or coding tools
are exposed. The existing dedicated `GPT_SOL_API_KEY` / `GPT_SOL_ENDPOINT` variables
are reused. Azure v1 authentication uses the OpenAI SDK, as in the supplied deployment
example, without importing the coding agent or using its deployment credentials.

## Real validation: one distinct validation patient

Two pipeline attempts used the **same** `ddxplus:validate:1` patient. Both used
retrieval top-k 1 for each RAG, sequential execution, and the fixed GPT deployment.
There was no second patient, ensemble, Phase 5, HITL or local generation.

1. Initial attempt: HTTP **400** at the Patient Agent; no accepted output. A separate
   non-patient compatibility probe confirmed `temperature` is unsupported.
2. After omitting the unsupported parameter, the same case was retried. Three API
   responses completed: Patient Agent once and Diagnostic Agent twice (one bounded
   repair). The second attempt was an **agent grounding validation failure**, not
   an API failure, JSON parsing failure, or measured diagnostic mismatch.

Requested temperature remains 0 in configuration. The actual request **omits**
`temperature`; effective sampling is deployment-default. Strict temperature-0
sampling cannot be claimed for this deployment. No automatic parameter fallback
or second-model retry was added.

| Stage / observation | Compatible attempt result |
| --- | --- |
| API | 3 completed Responses requests |
| Patient Agent | PASS, exact-copy/structured validation; 6.264 s |
| Patient Case RAG | PASS, one training case; 0.284 s |
| Medical Knowledge RAG | PASS, one medical chunk; 0.174 s |
| Diagnostic Agent | FAIL, grounding validation on both attempts; 18.698 s total |
| Clinical Critic | SKIPPED after diagnostic failure; not validated against live API |
| JSON/schema validity | Patient and both diagnostic responses parsed; diagnostic handoffs rejected by grounding |
| Accepted structured handoffs | 1 / 2 attempted agents |
| Evaluator | Completed after inference; missing diagnosis marked not measurable |
| Diagnosis match / top-5 match | N/A, no accepted diagnosis; not a medical accuracy failure |
| Critic findings | N/A, critic was not called |
| Total runner latency | 45.618 s including local setup and evaluation |
| Token usage | 6062 input + 1849 output = 7911 total across three completed requests |
| Final failure | `agent_validation_failure`, attempts 2 |

The exact failed grounding sub-rule is not recorded: provider telemetry stores
sanitized error categories rather than raw rejected clinical outputs or exceptions.
No unsupported medical claim was accepted merely to finish the pipeline.

Retrieved provenance:

- `case:ddxplus:train:60`, source
  `release_train_patients.zip!release_train_patients#row=60`; cosine similarity
  0.9877067512. Training feature payload only, no diagnosis/outcome labels.
- `medical:ac51854d-8e35-564f-abfb-478014277625`, PMC5451008, chunk 10,
  *Pre-transplant immune factors may be associated with BK polyomavirus reactivation
  in kidney transplant recipients*, CC-BY-4.0, DOI `10.1371/journal.pone.0177339`;
  cosine similarity 0.6898551739. Full original metadata, download provenance and
  raw-source SHA-256 are retained in the case artifact.

Nonempty retrieval and high similarity are not proof of medical relevance.
This tiny corpus and one synthetic patient do not establish clinical effectiveness.

### Exact commands (parent workspace, Windows)

Initial attempt:

```text
.venv\Scripts\python.exe healthcare-agentic-ai\scripts\run_phase4.py --queries 1 --top-k 1 --output-dir healthcare-agentic-ai\outputs\phase4\gpt-migration-validation
```

Compatible attempt, same patient:

```text
.venv\Scripts\python.exe healthcare-agentic-ai\scripts\run_phase4.py --queries 1 --top-k 1 --output-dir healthcare-agentic-ai\outputs\phase4\gpt-migration-compatible
```

Each directory retains `sample_case_001.json` and `phase4_run_report.json`.
Console logs and the non-patient compatibility result are in workspace
`outputs/gpt-migration/`. The opt-in integration test was not additionally enabled,
to avoid another paid case run.

## RAG integrity

Before and after migration, both collections existed and every payload passed its
existing validator. Sorted point ID + full payload + vector fingerprints matched:

| Collection | Count before / after | SHA-256 (same before / after) |
| --- | --- | --- |
| `ddxplus_patient_cases` | 1000 / 1000 | `6d3e80193bd098f7e8e3d5698cd5ce2b3b8d7c2e68ad245fa08132116f4401e2` |
| `medical_knowledge` | 87 / 87 | `92aac0f6bf3d68b941a0a1df9a0c89835dd0d57e3ea718560f94a9809973578b` |

Patient payload fields remain `patient_id`, `source`, `split`, `text`.
Medical payload structure/provenance is unchanged. No index was reset, rebuilt or
upserted. Protected-source hashes for parsing, data models, embeddings, retrieval,
stores, ingestion, grounding contracts and evaluation also match.
The medical store is at workspace `data/medical_knowledge/qdrant`; an initial audit
looked at the unused project-relative directory. Empty audit-created metadata at
that unused path was removed; the actual medical store was never altered.

## Tests and checks

The full unittest discovery covers Phases 1–4, parser/leakage protection, patient
and medical RAG, provider behavior, strict schemas, grounding, three-agent mocked
handoffs, stage reporting and post-inference evaluation. Tests use temporary local
stores or mocks and require no API key. The single real integration test is skipped
unless explicitly enabled.

Commands:

```text
.venv\Scripts\python.exe -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v
.venv\Scripts\python.exe -m compileall -q healthcare-agentic-ai\rag healthcare-agentic-ai\scripts healthcare-agentic-ai\tests
uv pip check
git diff --check
```

Offline result: **118 passed / 0 failed / 1 opt-in integration skipped**.
Python compilation/import checks and dependency compatibility passed. The final
whitespace/reference audit is recorded in workspace `outputs/gpt-migration/`.
No new dependency installation was needed: OpenAI SDK 3.12.0 was already installed.
The old backend used standard-library HTTP, so there was no dedicated package to
uninstall. `openai>=3.12,<4` is now explicitly declared for healthcare generation.

## Scope of changes and final reference audit

The full modified/added/deleted file inventory is in workspace
`outputs/gpt-migration/changes.json`. Changes are limited to generation config and
adapter, agent/provider imports, backend-specific prompt formatting, CLI reporting,
provider/Phase 4 tests, dependencies and relevant docs.

Removed the previous client implementation, backend-specific integration test,
obsolete model/context-budget tests (replaced with cloud equivalents), three obsolete
setup/recovery documents, and fourteen executable historical experiment scripts.
The general recovery and evaluation regression tests were retained and migrated.

Repository-wide searches retain only these justified categories outside active
healthcare generation code:

- Original user task specifications under workspace `prompts/`: historical requests,
  not runtime agent prompts. They were not rewritten to change task history.
- Historical JSON/log/Markdown evidence under ignored healthcare `outputs/phase4/`:
  preserved provenance of previous experiments, not runnable backend implementations.
- Migration inventory/search artifacts and this necessary migration audit: removed
  file names are historical evidence, not imports or setup instructions.
- The phrase "local model" in BGE embedding documentation/source refers to the
  unchanged embedding model, not a generation provider.

There are **zero legacy generation runtime references** in healthcare `rag/` or
`scripts/`. All runnable historical backend experiment scripts were removed, and
stale bytecode was cleared. System-wide software was not uninstalled. The coding
agent remains outside the healthcare runtime.

## Final status

```text
OLLAMA REMOVAL: PASS
GPT PROVIDER: PASS
PATIENT AGENT: PASS
PATIENT RAG: PASS
MEDICAL RAG: PASS
DIAGNOSTIC AGENT: FAIL
CLINICAL CRITIC: FAIL (not reached live; mocked tests pass)
E2E PIPELINE: FAIL
TEST SUITE: 118 passed / 0 failed (1 skipped)
RAG DATA INTEGRITY: PASS
OLLAMA RUNTIME REFERENCES: 0
READY FOR NEXT PHASE: NO
```

These FAIL statuses identify unfinished live validation, not missing migrated agent
implementations. The remaining blocker is diagnostic grounding validation; the
critic remains unverified live. The sampling limitation is also explicit above.

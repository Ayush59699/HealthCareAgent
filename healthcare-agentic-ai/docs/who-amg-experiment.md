# Isolated WHO + AMG medical-evidence experiment

## Decision

Implemented, **opt-in only; do not promote on these results**. Default AMG retrieval, its workflow factory, and standalone WHO lookup are unchanged. Six sequential live runs completed, but **all six final decisions were BLOCK with pending human review**. More citations are not diagnostic validation.

Request: `../prompts/whoinc3.txt`.

## Architecture and boundaries

```
existing Patient Agent -> validated label-free PatientState
  -> CombinedEvidenceService
       AMG: existing AMGEvidenceService.retrieve(patient, state), unchanged defaults
       WHO: existing WHOLookup.lookup(literal_patient_facts, max_results=3)
  -> CombinedMedicalEvidence (separate source groups, queries, immutable audit, availability)
  -> existing immutable EvidenceSnapshot
  -> existing Phase6DiagnosticAgent via source-section input delegate
  -> unchanged grounding report -> independent existing Critic
  -> unchanged Safety -> existing final-decision / pending-review handoff
```

Modes: `amg_only`, `who_only`, `who_plus_amg`. These belong to the **experimental entrypoint**, not the production factory. `who_only` never initializes/calls AMG. `amg_only` never initializes/calls WHO. All modes retain separate analogous patient-case retrieval. No new WHO embedding model, vector index, search implementation, ranking or medical inference was introduced.

The typed frozen contract separates `who_evidence.fact_sheets`, `who_evidence.questions_answers`, and `amg_evidence.accepted_passages`. Status is one of `BOTH_AVAILABLE`, `WHO_ONLY`, `AMG_ONLY`, `NONE_AVAILABLE`; it is an availability indicator, **not** a relevance verdict or safety permission. The original snapshot transport still uses a tuple of individually source-typed records. A synchronous provider delegate presents those records to Diagnostic and Critic as labeled WHO and AMG sections; it does not flatten text into an anonymous blob, change output schemas/validators, or make additional calls. Safety sees the same IDs, excerpts, and provenance through its original interface.

The same short source-handling instructions are used in all experimental arms: external knowledge is not patient observation, retrieval does not establish disease, cite only present text, acknowledge uncertainty and conflicts, and retain grounding/critic/safety/review. The production Diagnostic and Critic prompts are unmodified. Thus experimental AMG-only is a fair input-format control for the other arms, **not** a byte-identical production-prompt comparison.

### Query and excerpt policy, fixed before validation

- WHO query: unique literal presenting facts, symptoms, then antecedents in original order, packing complete facts into the existing 4096-character lookup limit. Omitted fact references are audited. No labels, expected answers, predicted diagnoses, synonyms, or manually injected disease names. Diagnosis/history words already present in patient answers remain literal input. Negated, compound or uncertain questions may still match topics; matching does not establish patient findings.
- AMG query, candidate generation, embedding, ranking, distance/gate, index, and top-k are untouched. The supported service uses patient k=1 and medical candidate k=5, with only its accepted evidence entering the snapshot. Rejected candidates remain audit-only.
- WHO lookup loads at most three matched documents. Between queries it retains only index metadata. Existing per-document limit is 4 MiB; no full corpus is opened/cached. Selected full text is temporary; at most 4000 source characters per document (12000 total) reach the agents.
- The excerpt is an **exact body prefix**, preferring a complete line boundary. It is not a summary or relevance-ranked passage. Omitted later content may contain important information and cannot be cited. Header separators are skipped with recorded character offsets. A long first line can end at the fixed bound; no synthesized ellipsis is inserted.
- WHO provenance includes native title, URL, type, topic, filename, retrieval timestamp, lookup match type/terms, full-document SHA-256/length, excerpt SHA-256, and exact character span. IDs identify WHO/type/topic/document hash/span. The required numeric similarity field is `0.0`, explicitly a non-ranking compatibility sentinel, not confidence.
- Snapshot validation checks structural self-consistency. The offline verifier additionally compares WHO excerpts to selected downloaded source bytes and AMG passages/metadata to native published chunks. Neither proves medical entailment or authenticates the current remote website.
- No evidence still triggers the existing pre-diagnostic abstention. Retrieval exceptions fail closed. Revisions reuse the same snapshot. A WHO reference does not automatically make a hypothesis supported or ALLOW it.

### Minimal shared compatibility change

Only `rag/agents/grounding.py` changed among existing runtime files: five lines dispatch WHO payloads to the new source-specific validator and require the non-ranking score sentinel. Existing AMG and legacy validation branches, reference checks, terminology, inference rules and safety rules are unchanged. Grounding continues to use `reference_present_not_verified`, `unsupported_by_medical_evidence`, and `medical_entailment=not_established`.

No WHO crawler, lookup implementation, downloaded document or index was changed. No compatibility bug required editing those components. No dependency/configuration files were changed.

## Commands

From `healthcare-agentic-ai`, with the existing project dependencies, cached models and prebuilt stores:

```text
# Retrieval-only probe: no cloud calls, not a clinical/agent validation
python -m rag.who_amg_experiment --mode who_plus_amg --queries 1

# WHO-only experiment (AMG not initialized)
python -m rag.who_amg_experiment --mode who_only --queries 1

# Explicit sequential live comparison, first two validation rows
python -m rag.who_amg_experiment --mode all --queries 2 --live --output-dir outputs/my-who-amg-run

# Recheck downloaded/native sources, references, safety and handoff without models/network
python scripts/verify_who_amg_run.py outputs/my-who-amg-run/ddxplus_validate_1_who_plus_amg_state.json

# Existing defaults remain available
python scripts/run_phase6.py --queries 1
python -m rag.who_lookup "asthma and diabetes"
```

Output directories must be new. CLI permits 1–3 consecutive cases, never label-driven selection. It prints bounded titles/types/topics/IDs/URLs/provenance and availability; full traces and grouped evidence go into local audit files, not an entire corpus dump. `--live` uses existing workspace generation settings; credentials are not printed. No parallel cloud requests or additional WHO models are used.

## Observed validation

Artifacts: `outputs/who-amg-experiment/`, including `offline/`, `live/report.json`, six versioned workflow states, six combined contracts, six final decisions, `live/verification.json`, and `live/comparison-integrity.json`.

Selection was fixed as validation rows **1 and 2**, `include_labels=False`, before retrieval. No benchmark/relevance labels were read for selection, query construction or evaluation. Offline probes preceded live calls; retrieval/excerpt settings were not tuned after seeing results. All experiments used the same cases and unchanged retrieval parameters. AMG query and accepted records are exactly identical between A and C; WHO query and records are identical between B and C for each case.

Citation counts below mean **unique supplied medical IDs cited by accepted Diagnostic hypothesis claims**, not citation occurrences, verified support, or clinical correctness.

| Case | Mode | WHO docs | AMG accepted | Availability | WHO citations | AMG citations | Grounding | Critic | Safety / final |
|---|---|---:|---:|---|---:|---:|---|---|---|
| validate:1 | AMG only | 0 | 5 | AMG_ONLY | 0 | 0 | structural pass | revision_required | BLOCK / BLOCK |
| validate:1 | WHO only | 3 | 0 | WHO_ONLY | 1 | 0 | structural pass | revision_required | BLOCK / BLOCK |
| validate:1 | WHO + AMG | 3 | 5 | BOTH_AVAILABLE | 1 | 0 | structural pass | revision_required | BLOCK / BLOCK |
| validate:2 | AMG only | 0 | 5 | AMG_ONLY | 0 | 2 | structural pass | revision_required | BLOCK / BLOCK |
| validate:2 | WHO only | 3 | 0 | WHO_ONLY | 2 | 0 | structural pass | revision_required | BLOCK / BLOCK |
| validate:2 | WHO + AMG | 3 | 5 | BOTH_AVAILABLE | 2 | 1 | structural pass | revision_required | BLOCK / BLOCK |

WHO lookup status was MATCH in all requested WHO arms, NOT_REQUESTED in AMG-only. Returned WHO documents:

- Case 1: **Healthy diet**, **Kidney disease**, **Low back pain**.
- Case 2: **Asthma**, **Depressive disorder (depression)**, **Anxiety disorders**.

These are all Fact Sheets. Q&A transport/provenance is covered by offline tests; **no Q&A document was selected in these live cases**. The CLI does not force one result from each collection. Empty-source combinations and no-match behavior are unit-tested, not represented by these two live cases.

### What the evidence actually added

- Case 1: WHO Kidney disease was cited for the limited association of fatigue with advanced/severe kidney disease. The Critic recognized that limited passage-level association, but not patient-specific causation. Leading anemia/bleeding hypotheses still lacked external evidence; Healthy diet and Low back pain were not Diagnostic claim citations. Combined AMG passages still had zero Diagnostic citations. WHO added some usable context, not demonstrated improvement to the leading hypothesis.
- Case 2: WHO Anxiety disorders and Asthma supplied cited symptom/context associations. AMG-only cited Chest Pain and Arrhythmia; the combined arm cited the two WHO sources and AMG Arrhythmia. The Critic recognized those limited overlaps while challenging preferential psychiatric ranking and unresolved cardiopulmonary/pediatric safety concerns. Combining sources changed which evidence was cited, not an established diagnosis or a safe release.
- The Critic is a model assessment, not independent clinician adjudication. Its passage-level observations are not substituted for semantic proof. Structural grounding reports **not_established** for medical entailment in every arm.
- All six runs had Critic safety flags, producing deterministic Safety BLOCK and `semantic_status=skipped_critic_block`. No semantic Safety cloud call was made; the mandatory Safety policy stage still ran. Five arms also had missing-medical-reference HUMAN_REVIEW findings; combined case 2 had references for every hypothesis but **still BLOCKED**. This directly demonstrates that citation presence is not an ALLOW shortcut.
- All proposals remained withheld, review pending, no human approval or scheduling. Six workflow runs used **19 sequential cloud requests**, about **346.4 seconds** of workflow time total. There were zero terminal technical failures. Combined case 2 had one rejected Diagnostic response (`agent_validation_failure`) followed by one accepted provider repair; the rejected raw response is not persisted, so its exact clinical defect is not claimed.
- Source replay verified **12 WHO document events and 20 AMG passage events**, all six Diagnostic/Critic versions and Safety assessments, and all six final decisions. Repeated sources across arms are separate events, not distinct documents.

## Tests, failures and operational limits

Final results:

- **24 new composition tests passed**, including all four availability states, WHO/AMG identity, exact excerpt offsets/checksums, rejected AMG audit isolation, label-bearing input/tampered state rejection, no full-corpus reading, mode isolation, identical prompts across arms, revisions, unknown citations, source-byte replay, percent-encoded WHO URLs, BLOCK and HUMAN_REVIEW preservation.
- Focused combined + existing WHO lookup + AMG integration: **66 tests passed**.
- Full project regression: **674 run, 673 passed, one existing opt-in cloud skip**.
- Unchanged upstream suite: **56 passed**.

Standard project suite: `python -m unittest discover -s tests -t .`.
Upstream: from `vendor/amg`, `python -m unittest discover -s tests`.

Environment detail: the initial direct project-venv run had two import errors because BeautifulSoup was absent (610 test entries, including failed imports). The existing system Python had BeautifulSoup but lacked pypdf (650 entries, one import error). No installation or requirements change was made. Appending the already-installed system site-packages directory to the project Python path allowed the full existing 650-test suite to pass before the new tests, and the final 674-test suite to pass. Exact final invocation on this machine:

```text
python -c "import sys,runpy; sys.path.append(r'C:\Users\Hp\AppData\Local\Programs\Python\Python313\Lib\site-packages'); sys.argv=['unittest','discover','-s','tests','-t','.']; runpy.run_module('unittest',run_name='__main__')"
```

A new source-verifier fixture initially failed because Windows newline translation changed fixture bytes after hashing. The fixture now writes exact bytes (`newline=''`); the verifier was correctly detecting the difference. A percent-encoded-URL test ensures the adapter accepts the same URL identity rule as the unchanged WHO lookup. No runtime retrieval tuning followed validation. The unchanged AMG tokenizer emitted its existing over-window warning while examining candidate query facts; accepted queries and evidence were identical across comparison arms, and no retrieval setting was changed.

Memory is structurally bounded by selected-document and excerpt limits, existing model/store reuse and sequential processing. The existing 2 GB available-memory preflight is retained; **peak process RSS was not measured and this is not a hard process memory cap**. Large fixed excerpts may still omit important context and make provider requests larger. No whole WHO collection is placed in a prompt.

## Files and reversibility

Created:

- `rag/combined_medical_evidence.py` — typed grouped contract and evidence service.
- `rag/who_provenance.py` — exact WHO excerpt adapter and integrity checks.
- `rag/who_amg_experiment.py` — opt-in factory/provider input delegate and CLI bridge.
- `scripts/validate_who_amg_experiment.py` — sequential bounded retrieval/live comparison runner.
- `scripts/verify_who_amg_run.py` — offline source and existing gate/handoff replay.
- `tests/test_who_amg_evidence.py` — 24 offline contract tests.
- This report and local artifacts under `outputs/who-amg-experiment/` (gitignored).

Modified: five additive lines in `rag/agents/grounding.py`; appended relevant entries to the project and workspace `okf.yaml` files. No other existing production source was edited. User's pre-existing WHO work and dirty vendor state were preserved. No commit was made by this implementation.

Integrity inventory: **807 of 809 pre-existing project files unchanged**. The only inventoried changes are the project OKF manifest and the five-line grounding dispatch; the workspace OKF manifest is updated separately. The inventory includes pre-existing Python sources, vendor sources, and WHO text/JSON assets, not Chroma SQLite read bookkeeping or unrelated dataset binaries. See `outputs/who-amg-experiment/integrity.json`. All indexed WHO documents and indexes in that inventory are unchanged.

Final tracked diff (excluding newly created untracked files): two OKF additions, 20 lines each, plus five grounding lines; **45 insertions, zero deletions**. The vendor submodule remains dirty as it was at startup; no vendor edit was made. New files are listed above and shown by `git status --short`; `git diff --check` passed.

To discontinue the experiment, keep using the unchanged production factory/CLI and standalone WHO lookup. No production configuration was switched. Removing the new experiment files and WHO dispatch branch is sufficient to remove the integration; do not remove the existing WHO collection/lookup.

## Recommended next step

Keep this experimental. Review the six saved traces and cited passages with an independent qualified evaluator, especially contextual relevance, compound/negative patient questions, source coverage, and unsafe diagnostic ranking. The observed source complementarity in case 2 is a reason to investigate, **not** evidence of improved clinical accuracy, safety, or human–AI effectiveness. Do not weaken Safety or force citations/matches to increase apparent success. Any broader label-free evaluation or alternative WHO excerpt selection needs separate authorization; it was not implemented here.

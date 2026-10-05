# Completed agentic healthcare research prototype

Request: `../prompts/FINAL.txt`. Pre-change analysis: [final-system-audit.md](final-system-audit.md).

**Assessment: COMPLETE AND RUNNABLE as a research prototype.** This is not a clinically validated diagnostic system or a demonstrated human–AI improvement. The genuine terminal review state is implemented; no human approval, reviewer scheduling, treatment action or safety override is simulated.

## Actual implemented pipeline

```text
DDXPlusParser(include_labels=False)
  -> PatientAgent -> exact-copy validated PatientState
  -> PatientCaseRAG (separate label-free analogous training cases)
  -> AMGMedicalEvidence (unchanged published MedlinePlus retrieval)
  -> immutable EvidenceSnapshot
  -> Phase6DiagnosticAgent -> DiagnosticResult
  -> existing deterministic grounding gates
  -> typed claim-level GroundingResult
  -> independent Phase6ClinicalCritic -> ClinicalCritique
  -> deterministic/semantic SafetyAssessment -> existing bounded routing
  -> build_final_decision -> FinalDecision / pending human review
```

The full original patient context reaches the agents even when not all facts fit in the AMG query. Current-patient evidence is the exact ID-to-fact inventory; retrieved training cases are explicitly **analogous cases**, not observations about the current patient. No patient or medical index was merged or replaced.

### Components and contracts

| Component | Status | Input | Output | Implementation | Tests |
|---|---|---|---|---|---|
| Patient Agent | Existing, reused | Label-free PatientRepresentation | PatientState, exact qualifiers/negations/history/numerics/unknowns | `rag/patient_parser.py`, `rag/agents/clinical.py`, `rag/agents/grounding.py` | patient_parser, patient_audit, phase4, final_decision |
| Patient Case RAG | Existing, reused | Original patient text | Patient-case RetrievedEvidence with source provenance; current-patient fact inventory remains separate | `rag/patient_rag.py`, `rag/patient_ingestion.py`, `rag/agents/grounding.py` | patient_rag, amg_integration, final_decision |
| AMG RAG | Frozen, reused | Existing bounded literal query | Native medical RetrievedEvidence, accepted hits only | `rag/amg/`, `vendor/amg/medlineplus_lab.py` | amg_integration, upstream tests |
| Diagnostic Agent | Existing, reused | PatientState, patient/case evidence, accepted medical evidence | DiagnosticResult (hypotheses, claims, uncertainty, exact references) | `orchestration/phase6/diagnostic.py` | phase6 boundaries/compatibility, reference_entailment, final_decision |
| Grounding | Existing gates plus new inspectable report | Proposal, patient facts, provenance-checked evidence | GroundingResult: claim paths/reference categories, fingerprints, partial reference audit, missing medical-reference hypotheses | `rag/agents/grounding.py`, `orchestration/phase6/grounding.py` | orchestration_grounding, reference_entailment, final_decision |
| Clinical Critic | Existing independent call, completed input handoff | Original context, exact proposal and GroundingResult | ClinicalCritique: unsupported points, contradictions, missing evidence, safety flags, revisions | `orchestration/phase6/diagnostic.py` | amg_integration, phase6, final_decision |
| Safety | Existing, unchanged | Exact fingerprinted diagnostic/critic/evidence input | SafetyAssessment: CONTINUE / HUMAN_REVIEW / BLOCK | `safety/`, Phase6Orchestrator | safety contracts/rules/validator, phase6 boundaries/routing, final_decision |
| Orchestration | Existing, reused | Structured case and injected services | Versioned Phase6WorkflowState, safe failures/abstentions, bounded revisions | `application/workflow.py`, `orchestration/phase6/orchestrator.py` | phase6 suites, amg_integration, final_decision |
| Final decision / human handoff | Added and integrated | Completed revalidated Phase6WorkflowState | FinalDecision: ALLOW / HUMAN_REVIEW / BLOCK, pending review, evidence, findings, uncertainty, audit identity | `application/decision.py`, both supported CLIs | final_decision, phase6_reporting, demo; saved-run verification |

Grounding **does not** equate a medical citation with medical support. `reference_present_not_verified` means exactly that; `unsupported_by_medical_evidence` identifies claims without such references. Patient-reference checks are partial, not a general semantic proof. Invalid grounding still fails the existing gate and is recorded in workflow validations/failures, rather than producing a fabricated passing GroundingResult. The Critic must assess the actual passages independently.

The report supplied to the Critic and the report recomputed for the final artifact bind the same diagnostic, patient and evidence fingerprints. They incur no additional model calls. The new critic input contract is versioned `phase6-critic-grounding-v1`; diagnostic and safety prompt versions are unchanged.

## Final decision and human review

Batch execution now writes:

- `sample_case_001.json`: full local **review/audit** state, including withheld proposal, all retrieved evidence, prior revisions, validations, errors and safety tickets. This is not a released clinical output.
- `final_decision_001.json`: typed final handoff. A proposal appears in `diagnostic_proposal` **only for ALLOW**. BLOCK/HUMAN_REVIEW contain null there and keep `proposal_withheld=true`.
- `phase6_run_report.json`: run summary with final-decision file/status and review state.

ALLOW requires a final workflow, exact current safety coverage and recomputed safety/critic routing; CONTINUE alone cannot authorize release. Original patient facts, evidence integrity, diagnostic identity and safety input/policy are revalidated before projection. Failed workflows map to BLOCK, no-medical-evidence abstention and unresolved reasoning map to HUMAN_REVIEW, without inventing a safety assessment. Existing workflow outcome/failure codes remain intact.

All withheld results have `human_review_required=true`, `review_state=pending`, `human_approval=false`, `human_review_scheduled=false`. The AI cannot clear that state. A human may inspect the separate audit matching the run ID and workflow fingerprint and independently assess the case. There is no programmatic approval/override endpoint. This fulfills the explicit review-state scope, not a deployed clinical review service or human study.

`supporting_patient_evidence` contains only actually cited current-patient facts. `analogous_case_evidence` is separate. `supporting_medical_evidence` contains only actually cited medical passages, with full native provenance. These fields do not claim semantic support; unused retrieved passages remain available in the workflow audit. The blocked live example has **five retrieved passages but zero cited medical passages**, correctly leaving the latter list empty.

## Tests

Commands ran sequentially with downloads/telemetry disabled and no new dependencies. Fixture providers are software-contract tests, not fabricated runtime diagnoses or clinical evaluation.

| Suite | Run | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| Existing project tests before changes | 547 | 546 | 0 | 1 |
| Project tests after changes | 575 | 574 | 0 | 1 |
| New tests (`tests/test_final_decision.py`) | 28 | 28 | 0 | 0 |
| Standalone upstream after changes | 56 | 56 | 0 | 0 |

The skipped test is the existing opt-in generic cloud integration test. A separate **real supported Phase 6 execution** was explicitly run below. Logs: `outputs/final-validation/tests-before.txt`, `tests-after.txt`, `upstream-after.txt`. The new tests cover exact patient preservation/no labels, source separation, report-to-critic identity, unsupported inference, provenance mismatch, critic findings propagation, all final dispositions, empty evidence, technical/grounding/safety failures, no duplicate retrieval, tampering, withholding, detached copies, atomic CLI artifacts and saved final-decision verification. Existing parser, retrieval, grounding, critic, safety and orchestration regressions all remain in the full suite.

## Real end-to-end execution

Using the existing environment/configuration, cached CPU BGE/MiniLM and existing stores, without setting retrieval options:

```bat
..\.venv\Scripts\python.exe -B scripts\run_phase6.py --queries 1 --output-dir outputs\final-validation\live
..\.venv\Scripts\python.exe -B scripts\verify_amg_run.py outputs\final-validation\live\sample_case_001.json --final-decision outputs\final-validation\live\final_decision_001.json
```

- Case: `ddxplus:validate:1`, loaded by the supported runner with **include_labels=False**.
- Run ID: `d64349fd-474c-458b-9cd6-17b8a5bc9c9b`.
- Existing index counts: **1,000 training cases / 2,289 medical chunks**.
- Retrieved once: **one patient-case match / five accepted AMG passages**.
- Workflow time: **45.94 seconds**, excluding startup; **three cloud requests**, zero structured repairs, zero revisions.
- Separate model calls: Patient Agent, Diagnostic Agent, Clinical Critic. Safety deliberately skipped semantic generation after the Critic's blocking flags, retaining its existing deterministic policy.
- Critic: six supported points, five unsupported points, eight missing-evidence findings, three safety flags. Zero contradictions in this particular live response is not proof that none exist.
- Safety: **BLOCK**, three `critic_safety_flags` plus four `missing_medical_reference` findings. No technical failure. No diagnostic result released.
- FinalDecision: **BLOCK**, pending human review, withheld proposal, no human approval.
- Offline verification: all five medical passages match native source text/metadata, diagnostic snapshot and critic references validate, exact safety-input fingerprint validates, final decision recomputes identically. See `outputs/final-validation/live-verification.json`.

| End-to-end stage | Result | Meaning |
|---|---|---|
| Patient Agent | PASS | Exact structured context accepted |
| Patient RAG | PASS | Existing independent patient-case evidence with provenance |
| AMG RAG | PASS | Five accepted native-source passages; not a relevance/accuracy claim |
| Diagnostic Agent | PASS | Structured uncertain proposal accepted, not clinically validated |
| Grounding | PASS | Deterministic gates and typed report; medical entailment not established |
| Clinical Critic | PASS | Independent structured critique identified unsupported/unsafe reasoning |
| Safety | PASS | Intended BLOCK; no redundant semantic request |
| Final Decision | PASS | Structured blocked/pending-review handoff verified; no release |

This one live run is the requested local end-to-end case **and** the existing Phase 6/live workflow. It was not redundantly rerun to seek a permissive outcome. No diagnostic accuracy, clinical safety, retrieval correctness, or human–AI effectiveness was evaluated or claimed.

## Production changes

Every changed pre-existing production file:

1. `orchestration/phase6/diagnostic.py`: add a versioned Critic instruction extension and explicit grounding report input; leave DiagnosticAgent and its schema/contract unchanged.
2. `scripts/run_phase6.py`: persist FinalDecision alongside the unchanged workflow audit and expose handoff metadata in the summary.
3. `demo/terminal.py`: construct/validate and display the final review disposition after execution; retain clearly labeled local reviewer walkthrough.
4. `scripts/verify_amg_run.py`: optionally verify a saved final-decision file against the recomputed workflow handoff.

New production files:

5. `orchestration/phase6/grounding.py`: typed deterministic claim/reference report using existing gates.
6. `application/decision.py`: typed terminal projection and genuine pending review boundary.

No other existing production source was changed. Added `tests/test_final_decision.py`, audit/validation documentation, local validation artifacts, and synchronized README/architecture/OKF manifests. Existing user changes and historical experiment artifacts were preserved. Source-integrity evidence is in `outputs/final-validation/source-integrity.json`.

## Retrieval freeze and remaining limitations

**AMG retrieval strategy changed: NO**  
**AMG retrieval parameters changed: NO**  
**New retrieval experiment performed: NO**

No embedding, metric, top-k default, gate, query compression, alias rules, ranking/reranking, corpus/index, candidate generation or multi-query decomposition changes. No labels introduced, ground-truth runtime access, downloads, training, index rebuilds, database copies, added medical data or new model configuration. Safety policy, original patient parsing/validation and Phase 6 routing remain byte-identical to the pre-task source hashes.

AMG is still functional but imperfect and may miss relevant passages or return irrelevant ones. New4 remains **DO NOT PROMOTE**. Deterministic grounding cannot prove medical entailment, and LLM criticism/safety are fallible. Human review is a pending structured state, not a completed clinical intervention. Memory use remains sequential with existing preflight; no hard RAM cap or peak-memory measurement is claimed. There are no remaining execution blockers on this configured machine; local stores/caches and valid cloud configuration remain deployment prerequisites.

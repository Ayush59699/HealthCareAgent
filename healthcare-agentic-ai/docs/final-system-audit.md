# FINAL implementation: pre-change audit

Request: `../prompts/FINAL.txt`. Baseline: 547 tests run, 546 passed, zero failures, one opt-in cloud integration test skipped (`outputs/final-validation/tests-before.txt`). Existing working-tree changes belong to earlier work and are retained.

## Inspection and execution path

Reviewed both current OKF manifests, vendor manifest, cleanup report/inventory, active contracts, agents, parser, both evidence services, Phase 6 generation/grounding/revision/safety loop, routing, safety policy, CLI, and tests. Static AST inventory covers all 175 Python files under the project, including legacy/vendor code, without loading datasets or models. Source/documentation hashes and symbol inventory are in `outputs/final-validation/`. The TODO/NotImplemented/placeholder scan found no executable placeholder in the active runtime (only prose in a historical evaluation report builder).

Supported path: `scripts/run_phase6.py` or `scripts/demo.py` -> `application/workflow.py` -> Phase6Orchestrator with AMGEvidenceService. Sequential PatientAgent -> separate patient-case retrieval and AMG -> Phase6DiagnosticAgent -> deterministic grounding -> Phase6ClinicalCritic -> deterministic/semantic safety -> route. Evidence retrieval occurs once, including during revisions. Tickets bind versions, evidence, critic and exact safety input. CONTINUE is only an internal safety result, not permission to bypass critic routing.

Cleanup removed only an empty dump and regenerable bytecode; it did not remove agents. Legacy Phase 4/5, medical retrieval experiments, New3/New4 negative results and vendor research remain for regression/history. They are not replacements for the supported application. No deletion or promotion is planned.

## Working components to reuse

- DDXPlusParser: streams rows; `include_labels=False`; decodes Yes/No, qualifiers, history, categorical and numeric values without interpreting scores. Diagnosis fields are not parsed for inference.
- PatientAgent / PatientState: structured cloud call, exact-copy validation of all clinical lists/demographics, no invented relevant findings; missing values and uncertainty retained.
- PatientCaseRAG: existing label-free **analogous training-case** Qdrant index. These are not observations about the current patient. Current-case evidence is the exact PatientState fact inventory, independently available to agents. Do not relabel a retrieved training case as this patient.
- AMG: native chunk provenance, source checksums, gate, one bounded literal query; no-accepted-evidence abstention. Functional but imperfect. Freeze entirely.
- DiagnosticResult: hypotheses, claim references, exact used-source inventories, uncertainty and missingness. Phase 6 enforces explicit patient-only inference disclosure and bounded observation/reference consistency checks.
- Grounding: exact patient copy, evidence/source integrity, citation existence/category and limited patient-reference consistency. It does **not** prove medical entailment. Existing validations/events expose gate results but no structured per-claim grounding report is provided to critic.
- ClinicalCritique: separate isolated cloud call, unsupported/contradictory/missing evidence, safety flags and revisions. Needs explicit deterministic grounding information in its input.
- SafetyAssessment: deterministic flags and missing medical references plus independent semantic checks. BLOCK and HUMAN_REVIEW are terminal. No policy weakening needed.
- Phase6WorkflowState: versioned diagnostics/critiques, frozen evidence, safe failures and abstentions, safety tickets/coverage, timings and budgets. Remains the full local audit/reviewer record.

## Bounded missing work

1. Add a typed, deterministic claim/reference GroundingResult using existing gates, preserving snapshot and diagnostic/patient fingerprints. Distinguish current-patient citations, analogous cases, medical references, and unsupported medical inference. Never call citation existence medical support/entailment. Supply this report to the existing independent Critic, without an extra model call.
2. Add a typed FinalDecision projection of terminal Phase 6 state: ALLOW / HUMAN_REVIEW / BLOCK, genuine pending review, exact evidence and findings, uncertainty, errors/abstentions and audit identity. Recheck original input, snapshots and safety policy before release. Never release a blocked/review-only proposal; retain it only in the separate local workflow audit for reviewers. No fake approval/scheduling or automated human override.
3. Integrate output into supported CLI and add offline boundary tests, then run one real label-free case with current cached models/stores and existing cloud configuration. Report technical completion separately from clinical appropriateness.

No retrieval strategy, query construction, embedding, alias, top-k default, corpus, store, ranking, gate, labels, cloud configuration, safety policy or dependency changes are planned.

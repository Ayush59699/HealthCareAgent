# DDXFIX — canonical patient-reference validation only

## Scope and change

Implemented `prompts/ddxfix.txt`. The only production-code change is in
`rag/who_selector_relevance.py::validate_content_review`.

The matcher now accepts either:

1. **An exact complete `Evidence.text` fact**, including its question and answer; or
2. The existing exact substring of a non-missing `Evidence.value`, preserving manual-prose support.

The canonical branch uses **whole-fact equality**, not substring search over question wording. It cannot accept a question with its `= No`/uncertain answer removed, a truncated answer, a different answer taken from another fact, or combined facts. No case folding, whitespace normalization, punctuation repair, inference or synthesis is performed. Missing values are not converted into the fabricated string `None` for excerpt matching; their full canonical `(answer not supplied)` representation retains missingness.

Bare `Yes`/`No` remain rejected by the unchanged minimum four-character quote rule. Exact source quote matching, schema limits, max-three selection, source provenance, all prompts, WHO catalog selection, AMG, Diagnostic, Grounding, Critic, Safety and final-decision behavior are unchanged.

**Boundary:** literal validation preserves the supplied polarity/uncertainty; it is not a new semantic relevance classifier. An exact negative fact remains negative. The existing LLM rules determining whether a fact can support positive presenting relevance remain unchanged. The existing manual-prose excerpt behavior is retained, not replaced with a new negation parser.

## Regression evidence

Added `tests/test_ddx_patient_quotes.py`, **16 new offline tests**:

- Exact DDXPlus-style `question = Yes` and location facts, including all four quoted facts from the audit.
- Symptoms, antecedents and initial-evidence fields.
- Bare short answers, question-only quotes, truncated answers, punctuation/spacing/case changes rejected.
- `No` or uncertain answers cannot be rewritten as `Yes`; missingness stays explicit.
- Fabricated facts, recombined question/answer pairs and joined facts rejected.
- Existing manual full-value and literal excerpt quotes still accepted.
- Source quote corruption still rejected *after* the corrected patient check.
- Schema-bypass attempts still fail minimum length validation.
- Real provider parsing/validator path with a mocked API.
- Default WHO → AMG → Diagnostic → Critic → Safety integration with canonical question/answer facts and synthetic medical fixtures. This is software-contract testing, not a clinical result.

Before the fix, these 16 tests had **one failure and nine errors**, reproducing the mismatch. After the fix:

| Test run | Result |
|---|---|
| Focused selector/manual/default-pipeline tests | **94 passed** |
| Full project suite | **768 run; 767 passed, one existing opt-in cloud skip** |

Existing manual-input tests were run unchanged and passed. Logs are in `outputs/ddxfix/`.

The full suite used the workspace interpreter with the existing system site-packages appended for BeautifulSoup, as in previous tasks. No dependencies were installed or changed. All live calls used the existing workspace provider configuration.

## Exactly one live `validate:1` run

Command, from `healthcare-agentic-ai`:

```text
python scripts/run_phase6.py --queries 1 --output-dir outputs/ddxfix/live
```

The unchanged parser used `include_labels=False`. No benchmark label, hidden diagnosis or expected answer was supplied. No follow-up live rerun, prompt tuning or source-quote relaxation was performed.

Run ID: `db9ae5f1-1556-44b1-9aa6-59e99b1aa613`.

| Stage/result | Observed outcome |
|---|---|
| Patient Agent | Accepted and validated |
| WHO catalog selector | Accepted |
| WHO content review — patient quotes | **Passed the corrected validation** |
| WHO content review — source quotes | **Failed unchanged exact-source validation** |
| AMG retrieval | **Not reached** |
| Diagnostic | **Not reached** |
| Critic | **Not reached** |
| Safety assessment | **Not reached**; `technical_failure_before_safety` |
| Workflow status | `failed`, `agent_validation_failure` |
| Failure detail | **`nonliteral_source_quote`**, not `nonliteral_patient_quote` |
| Final decision | **BLOCK**, human review **pending**, no diagnostic proposal |
| Cloud requests | **3 sequential requests**, no repairs |
| Workflow time | **13.26 seconds**, excluding resource startup |

The returned patient quotes were:

```text
Do you feel slightly dizzy or lightheaded? = Yes
Do you constantly feel fatigued or do you have non-restful sleep? = Yes
Is your skin much paler than usual? = Yes
Do you feel pain somewhere? = forehead
```

All four exactly match supplied canonical facts and now pass. The returned source quotes were:

```text
tiredness
dizziness or feeling light-headed
headache healthcare provider.
```

The content-review object passed its schema, but at least one source quote did not occur literally in the supplied document. In particular, the last string is not an exact quote in the local WHO Anaemia text. The failed trace does not retain the chosen document ID/full content-request payload, so that local source comparison is not an authenticated reconstruction of the exact request. The recorded **`nonliteral_source_quote`** code proves that patient validation had passed and the next, unchanged source check rejected the response.

**Conclusion:** the requested DDXPlus patient-reference bug is fixed, but this live case does **not** yet complete WHO → AMG → Diagnostic → Critic → Safety. It stops at a separate source-quotation failure. This task deliberately does not alter that gate or claim a successful end-to-end/clinical result.

## Saved-run verification

`outputs/ddxfix/replay.py` and `verification.json` compare both the original three-request audit and the new live response against the saved patient representations. All eight patient quotations across those two responses match canonical facts and fail the former bare-value comparison. The new run's failure code is the source gate, not the patient gate.

The final BLOCK/pending handoff was independently rebuilt from the saved state and compared to `final_decision_001.json`. No additional cloud call was made for this verification.

Artifacts:

- `outputs/ddxfix/live/sample_case_001.json` — complete failed-state audit and returned structured content review.
- `outputs/ddxfix/live/final_decision_001.json` — revalidated BLOCK/pending handoff.
- `outputs/ddxfix/live/phase6_run_report.json` — run summary.
- `outputs/ddxfix/live.log` — runtime log.
- `outputs/ddxfix/before-fix-tests.log`, `focused-tests.log`, `regression-tests.log`.
- `outputs/ddxfix/verification.json`, `integrity.json`.

## Files changed

Production:

- `healthcare-agentic-ai/rag/who_selector_relevance.py` — only patient quote matching inside the existing validator.

Added:

- `healthcare-agentic-ai/tests/test_ddx_patient_quotes.py`.
- `healthcare-agentic-ai/docs/ddxfix-patient-references.md` (this report).
- Workspace-root `DDX_TEST.txt` — plain-text commands to run focused regressions and a new live batch case.
- Audit/test/live artifacts under `healthcare-agentic-ai/outputs/ddxfix/`.

Context synchronization only:

- Workspace `okf.yaml` and `healthcare-agentic-ai/okf.yaml`.

All other pre-existing production source, existing tests, prompts, WHO corpus/index assets and prior saved runs remain unchanged. Existing `AHS_RUN.txt` was not overwritten. The pre/post inventory and prompt comparison are recorded in `integrity.json`.

## Test the pipeline yourself

From the workspace root, run the commands in **`DDX_TEST.txt`**:

```text
cd healthcare-agentic-ai
..\.venv\Scripts\python.exe -m unittest tests.test_ddx_patient_quotes tests.test_who_selector_relevance tests.test_manual_patient tests.test_check5_pipeline tests.test_check5_audit
..\.venv\Scripts\python.exe scripts\run_phase6.py --queries 1
```

The batch command makes live cloud calls and creates a fresh timestamped directory under `outputs/phase6/`, avoiding overwrite/collision on reruns. Its console report gives the exact directory. Inspect `sample_case_001.json` → `medical_retrieval.who.validation_errors` if it fails.

A failed source-quote check must remain a failure; do not bypass it to force a diagnostic result. No-evidence abstention, BLOCK and pending review are not clinical approval.

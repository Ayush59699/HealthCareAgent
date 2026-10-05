# Phase 6 implementation and validation

## Scope delivered

The dedicated output-safety service and isolated Phase 6 orchestrator/runner are
implemented. The implementation follows [the Phase 6 architecture](phase6.md):
validated critic -> deterministic safety checks -> required semantic assessment
unless critic-blocked -> application-owned safety decision -> baseline routing
or terminal withholding. Frozen evidence, grounding gates, versioned critiques,
two-revision limit and Orchestrator ownership are preserved.

No existing Phase 5 runtime, contract, prompt, test or runner was modified. The
only edits to pre-existing tracked files are **appended** Phase 6 sections in
`README.md` and `docs/architecture.md`. All original text is retained. No dependency,
credential, dataset, index or original task/prompt file was edited.

New implementation files:

- `safety/{__init__,models,policy,prompts,validation,validator}.py`;
- `orchestration/phase6/{__init__,contracts,events,state,policy,routing,orchestrator}.py`;
- `scripts/run_phase6.py`;
- `tests/safety_helpers.py`, three `test_safety*.py` modules and five
  `test_phase6*.py` modules;
- `docs/phase6.md` and this report.

The Phase 6 loop is an intentionally narrow adaptation of the protected baseline,
not a refactor of Phase 5. Its independent state class prevents old Pydantic
consumers from accepting new state objects through subclass instance validation.
CONTINUE delegates to the original pure Phase 5 routing function.

## Executed validation

All execution below used the existing parent `.venv`, synthetic fixtures and
mocked providers. **No paid/live API calls, embedding downloads, index rebuilds,
real-index audits or clinical validation were performed.**

| Check | Actual result |
| --- | --- |
| Full unittest discovery | **263 run: 262 passed, 1 opt-in cloud integration skipped** |
| New Phase 6 tests separately | **49 passed** |
| New safety service/contracts/rules tests separately | **29 passed** |
| Existing tests | Unchanged; all non-opt-in tests passed in full discovery |
| Phase 6 CLI `--help` | Passed without running inference |
| CLI execution and report generation | Passed with mocked backends |
| Syntax compilation, without writing bytecode | **92 Python files passed** |
| `uv pip check` | All **65 installed packages** compatible |
| `git diff --check` | Passed; Git emitted only line-ending conversion notices |
| Protected original file SHA-256 comparison | **70 of 70 unchanged** |

Commands from the Windows parent workspace:

```text
set RUN_GPT_INTEGRATION=0
.venv\Scripts\python.exe -B -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v
.venv\Scripts\python.exe -B -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -p test_phase6*.py -v
.venv\Scripts\python.exe -B -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -p test_safety*.py -v
.venv\Scripts\python.exe -B healthcare-agentic-ai/scripts/run_phase6.py --help
uv pip check
git diff --check
```

The first full run exposed an error in a **new test fixture** that assigned to a
frozen SafetyInput field. The fixture was corrected to construct a detached copy;
no contract or old test was loosened. Subsequent full discovery passed.

## What the new tests establish

- Complete strict category coverage; no model-authored routing/severity/tickets;
  invalid enums, missing/duplicate categories, unbounded/empty fields and
  `model_construct()` bypass attempts rejected.
- Exact field-pointer/quote matching, source-ID validation, duplicate findings,
  source provenance and literal fabricated URL rejection.
- Application severity mapping; uncertainty never becomes clearance; critic flags
  block even on otherwise supported/insufficient results and bypass semantic calls.
- Non-abstaining hypotheses without medical citations require review without
  changing or weakening the original grounding gate.
- Mandatory safety review for generated abstention prose; no artificial hypothesis
  or forced citation inventory.
- No keyword prescribing/injection detector: fixtures cover negated, quoted,
  hypothetical and affirmative text, with semantic decisions explicitly mocked.
- Identity checks cover every safety ticket field and stale prior-version replies.
  Input tampering and forged permissive assessments fail at consumption.
- Detached handoffs, frozen evidence, run isolation and label-sentinel exclusion
  from all provider requests and saved state, including revisions.
- Maximum two revisions; each non-repeated version gets its own assessment;
  one retrieval per source; no safety feedback injected into the revision contract.
- Existing no-evidence and repetition early exits remain explicitly unassessed;
  summaries do not reuse an older CONTINUE for an unassessed/failed attempt.
- Invalid schema/anchors/references, unavailable validators, provider refusal,
  incomplete output, transport failure, context byte caps, exhausted repairs,
  invalid accounting and cumulative budgets stop closed.
- Deadlines checked after semantic calls, deterministic work, assessment policy
  and routing; no late finalization. No hard in-flight cancellation is claimed.
- Valid concerning results are not retried to obtain approval. Schema/validation
  repairs remain separate from clinical revisions.
- Versioned atomic CLI artifacts, sanitized errors, existing-directory protection,
  missing-case failure and explicit withheld/non-clinical human-review reports.

## Phase 5 preservation evidence

Before implementation, SHA-256 fingerprints were recorded for every existing
Python file under `rag/`, `orchestration/`, `scripts/`, `tests/`, plus
`requirements.txt`: 70 files. After implementation, all 70 fingerprints match.
The protected set includes the original agents, prompts, grounding helpers,
provider, parser, evaluator, retrievers, stores, orchestration and tests.

Parity tests exercise success, abstention, critic insufficiency, revisions,
revision exhaustion, repetition, critic safety blocking and missing evidence.
When safety permits continuation, original role requests are **identical**, as
are patient state, frozen evidence, diagnoses/critiques, baseline event order,
revision counts and terminal routing. Additional safety calls and telemetry are
expected differences. New review restrictions are explicit Phase 6 policy, not
changes to Phase 5 behavior.

Retained local artifacts under the ignored `outputs/phase6/validation/` directory:

- `protected-before.json`, `protected-after.json`;
- `tests-first.log`, `tests-final.log`;
- `phase6-tests.log`, `safety-tests.log`;
- `final-checks.json`.

The integrity comparison covers source files, **not** an asserted before/after
vector audit. Tests use temporary stores; production index integrity was not
re-audited as part of this task.

## Limits and unvalidated behavior

No real Phase 6 cloud end-to-end run has been performed. Live structured-schema
acceptance, latency, refusal rates, semantic detection quality and clinical
false-positive/false-negative rates remain unmeasured. Mocked category findings
validate software routing and contracts, not the model's ability to detect them.

The semantic validator uses the same backend as the clinical roles and can share
blind spots. A clean assessment does not prove completeness or medical entailment.
The five-document medical corpus remains unchanged and inadequate for broad
clinical safety guarantees. HUMAN_REVIEW is a terminal marker only; there is no
review queue, reviewer action, notification, approval or resume implementation.
CONTINUE is not clinical clearance, and no patient-facing advice or clinical
execution capability has been added.

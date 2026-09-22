# Terminal research demonstration

Run from the parent workspace (where `pyproject.toml` lives):

```text
uv run python healthcare-agentic-ai/scripts/demo.py
```

Example interaction:

```text
[1] USE SAMPLE PATIENT (default)
[2] ENTER PATIENT MANUALLY
[D] Toggle detailed mode
[Q] Quit
Mode [1]: D
Mode [1]: 1
[1] Sample ddxplus:validate:1: age 55, sex F, 17 findings
[2] Sample ddxplus:validate:2: age 10, sex F, 19 findings
[3] Sample ddxplus:validate:3: age 68, sex F, 27 findings
Sample [1], [B] Back, [Q] Quit: 1
```

Press Enter twice for the default case. One case is run per invocation. These are
actual first-three DDXPlus validation records, selected without reading their
pathology or differential labels. They are **synthetic research cases, not real
patients**. The displayed counts above describe the dataset currently installed;
selection always uses the actual parser, not hard-coded clinical content.

## Rehearsal commands

```text
uv run python healthcare-agentic-ai/scripts/demo.py --help
uv run python healthcare-agentic-ai/scripts/demo.py --list-samples
uv run python healthcare-agentic-ai/scripts/demo.py --validate-only --detailed
uv run python healthcare-agentic-ai/scripts/demo.py --sample 1 --detailed --architecture
```

`--list-samples` needs only the dataset/parser. `--validate-only` opens the existing
Qdrant collections, validates their embedding signatures, embeds the selected
patient with the cached models, retrieves both evidence types and validates the
snapshot using the existing `EvidenceService`. It makes **zero GPT requests** and
no diagnostic claim. It does not validate API credentials/connectivity, audit
every indexed point, or establish medical relevance. Missing/empty indexes and
empty retrieval sources fail explicitly. No index creation/rebuild, embedding
download or fallback model is requested by the demo.

The live command uses the existing dedicated `GPT_SOL_*` environment convention
and `gpt-5.6-sol` provider. It sends label-free synthetic patient and retrieved
context to that configured cloud deployment. Credentials/endpoints are never
printed. Existing dependencies in `requirements.txt` and cached BGE models are
required; this feature adds no dependencies.

Paths can be supplied with `--data-dir`, `--patient-storage-path`, and
`--medical-storage-path`. Defaults match the existing runner; notably the medical
index currently resolves to the **parent workspace's**
`data/medical_knowledge/qdrant`. Close other programs holding those local Qdrant
stores. Do not rebuild indexes to run this demo.

`--top-k` defaults to 1; `--max-requests` to 21; `--max-seconds` to 900, preserving
the existing bounded workflow defaults. The deadline is cooperative, **not hard
cancellation** of synchronous GPT calls. The configured provider timeout still
applies. Ctrl+C/EOF cancels with exit 130. Setup or workflow technical failure
exits 1. Technically completed abstention, BLOCK and HUMAN_REVIEW exit 0 but are
clearly marked **withheld**, not successful diagnoses.

## What is displayed

Real-time notifications are printed and flushed before each provider invocation
and retrieval, with actual call-return/hit-count notifications afterward. There
are no invented progress bars or results. A returned provider response is clearly
marked **pending application validation**, not accepted output.

The existing core has no streaming event callback. To preserve its behavior,
**accepted outputs and deterministic gates are presented after the run as an
explicitly labeled audited walkthrough**, not falsely described as live events:

| Stage | Accepted information displayed |
| --- | --- |
| 1. Patient Agent | Label-free input, construction/exact-copy validation, bounded PatientState fields and missingness |
| 2. Patient Case RAG | Query excerpt in live progress, top-k, hit count, IDs, scores, source and evidence excerpts; validated training-only/label-free provenance |
| 3. Medical Knowledge RAG | Separate medical IDs, titles, source types, scores and excerpts |
| 4. Diagnostic Agent | Each committed version, hypotheses, supporting/contradicting/rationale reference IDs, inventories, missingness and uncertainty; explicit abstention |
| 5. Grounding | Actual recorded per-version grounding result, reference existence and inventory checks; failures are not converted to passes |
| 6. Clinical Critic | Reviewed diagnostic version, assessment, safety flags, unsupported points, contradictions, missing evidence and recommended revisions |
| 7. Phase 6 safety | Recorded deterministic checks, exact semantic category statuses when executed, findings, structured reasons, decision and associated version/snapshot |
| Final summary | Latest committed diagnostic/critic status, current safety coverage (never stale clearance), workflow status/outcome, retrieved vs cited counts, revisions, requests, repairs and duration |

Detailed mode adds collection/model/dimension settings, evidence snapshot ID and
schema, workflow/routing/safety-policy/safety-prompt versions, critic fingerprint,
semantic explanations and the ordered committed event history. Evidence and
model prose are bounded, labeled excerpts. No raw prompts, provider responses,
credentials, arbitrary metadata dumps, or hidden chain-of-thought are printed.
Even the application's free-form `reasoning_summary` is omitted. Terminal control
characters are stripped from untrusted prose; status glyphs have ASCII fallbacks.

RAG is evidence infrastructure, not an autonomous agent. Similarity is not disease
probability; reference existence does not prove clinical entailment. Semantic
`no_issue_identified` is an actual model assessment, not proof of clinical safety.
An irrelevant medical source, abstention or review decision is shown honestly.
The installed medical corpus is small and may not support the selected patient's
symptoms. `CONTINUE` only allows the unchanged Phase 5 routing policy to proceed;
it does not necessarily mean the workflow finalizes.

## Compatibility and manual input

`demo.terminal.make_workflow()` creates a thin `Phase6Orchestrator` subclass. Its
`run` method is **inherited verbatim**. Only `_invoke` is decorated to print call
notifications and delegate to the original implementation. Retrieval adapters
forward the original query/top-k and return the original results unchanged.
No new clinical loop, agent, grounding rule, safety rule, route, repair or retry
exists. The offline integration test asserts the inherited method identity and
exact provider request parity against the unwrapped Phase 6 workflow.

Manual free-text entry is intentionally unavailable. The existing input gate
requires a parser-generated DDXPlus ID and decoded evidence. Reusing a dataset ID
for arbitrary text would falsify provenance; extending that contract is outside
this demonstration. Selecting manual mode explains this without collecting or
submitting input, then offers the sample menu again.

All added code is under `demo/`, `scripts/demo.py` and `tests/test_demo.py`.
Existing Phase 5/6 code, safety logic, agents, prompts, contracts, indexes and tests
are not edited. Existing working-tree changes predating this task are preserved.

## Verification

```text
uv run python -m py_compile healthcare-agentic-ai/scripts/demo.py healthcare-agentic-ai/demo/terminal.py healthcare-agentic-ai/tests/test_demo.py
uv run python -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -p test_demo.py -v
uv run python -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v
```

The added tests cover menus, manual fallback, label-free discovery, terminal
sanitization, CLI validation, retrieval transparency, actual Phase 6 request
parity, CONTINUE/HUMAN_REVIEW/BLOCK, semantic skip, budget failure, early
abstention, stale safety coverage, safe error messages, and zero-provider local
validation. Offline providers are explicit software fixtures, never a demo mode.

**Research only. Not medical advice, a clinical diagnosis, clinical validation,
or clinical clearance. HUMAN_REVIEW is only a terminal marker: no human review is
scheduled, performed or implemented.**

## Results from this implementation session

- Syntax compilation, CLI help and actual three-case sample discovery passed.
- Full suite: **283 tests, 282 passed, 1 opt-in cloud test skipped**, including
  20 new demo tests. Existing tests emitted a non-failing SQLite ResourceWarning.
- Local integration: existing **1,000 patient cases / 87 medical chunks**, both
  BGE embedding dimensions 384; label-free sample retrieval and frozen evidence
  validation passed without GPT requests or index rebuilds.
- Offline workflow integration: exact provider-request parity with the original
  Phase 6 loop passed, including the mandatory safety path.
- One separate **real GPT demo** was performed on `ddxplus:validate:1`, top-k 1.
  Diagnostic status: **ABSTAINED**; critic: `revision_required`; safety: **BLOCK**;
  workflow outcome: `safety_blocked`. Semantic review was correctly skipped due
  to critic safety flags. **3 requests, 0 provider repairs, 0 revisions, 45.88 s**
  of workflow execution. This live run did not exercise semantic safety generation.
  Transcript: `outputs/demo-live-check.txt` in the parent workspace.
- SHA-256 comparison of all 115 pre-existing source/document files captured
  before implementation found **no modifications**, including Phase 5/6, safety,
  prompts, contracts and existing tests.

These are software/integration checks, **not clinical validation**. A future
cloud response may differ; no particular diagnostic or safety outcome is promised.

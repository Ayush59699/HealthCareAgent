# Diagnostic abstention contract: fix and single-case validation

Research-only synthetic DDXPlus prototype. This change fixes the evidence-inventory
contract; it does not force a diagnosis, establish clinical effectiveness, or start
Phase 5/HITL. **The live pipeline completed with a valid abstention, but the critic
requested substantive revisions. No diagnosis was inserted and no second live run
was performed.**

## Change and rationale

The sole production behavior change is the Diagnostic Agent prompt in
`rag/agents/prompts.py`. The prompt bundle version is now
`phase4-v2-abstention`; Patient Agent and Clinical Critic instructions are unchanged.

The Diagnostic Agent is explicitly told:

```text
IF YOU ABSTAIN, all four fields MUST be:
primary_hypothesis = null
differential_diagnoses = []
patient_case_evidence = []
medical_knowledge_evidence = []
```

"Used" evidence means references inside a structured hypothesis's `rationale`,
`supporting_evidence`, or `contradicting_evidence`, including differential hypotheses.
Each inventory must equal exactly the used references in its own source category.
The model may discuss retrieved evidence in uncertainty, missing information and
reasoning summary, but prose citations (also in unsupported-claims text) do not
populate structured inventories. The prompt explicitly says not to manufacture a
hypothesis merely to cite evidence. Abstention remains valid.

**The grounding safety model is unchanged.** `claims()` and `validate_references()`
still compute exact inventory equality from structured claims. Unknown-reference,
source-category, provenance, missing-medical-evidence, and URL guards remain intact.
No Pydantic cross-field validator was added: the existing grounding invariant already
covers this case and remains the single source of truth. Models, parser, provider,
retry logic, evaluator, clinical agents and pipeline implementation are unchanged.

## Regression tests

Added `tests/test_abstention_contract.py`, containing 16 offline tests:

| Required case | Expected / observed |
| --- | --- |
| A: empty hypotheses and empty inventories | PASS, even with prose mentions of retrieved sources |
| B: abstention with a valid patient source in inventory | Rejected by exact inventory equality |
| C: abstention with a valid medical source in inventory | Rejected by exact inventory equality |
| D: hypothesis cites patient case, matching inventory | PASS with medical retrieval still available |
| E: hypothesis cites medical source, matching inventory | PASS |
| F: claim cites source omitted from inventory | Rejected for both source types |
| G: inventory contains a source not referenced in claims | Rejected for both source types |
| H: unknown claim reference | Rejected by unknown-reference guard |
| I: wrong source-category inventory | Rejected in both directions |

Additional coverage: unknown inventory IDs; differential-only output is not
abstention; inventory union includes supporting/contradicting claims across
hypotheses; no-medical-retrieval guard remains; explicit prompt is delivered through
the provider; valid abstention reaches the critic unchanged; invalid abstention is
not silently repaired or passed to the critic after bounded retries.

No existing tests were changed. Full suite before the live run:
**134 passed / 0 failed / 1 opt-in integration skipped** (135 discovered).

```text
.venv\Scripts\python.exe -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v
```

## Exactly one fresh live validation

- Model: `gpt-5.6-sol`, sole healthcare generation backend, Responses API.
- Patient: the same `ddxplus:validate:1`, label-free inference features.
- Patient RAG top-k: 1; medical RAG top-k: 1.
- Sequential Patient Agent -> both RAGs -> Diagnostic Agent -> Clinical Critic -> evaluator.
- Three completed API requests; one per agent, no repair retries.
- Temperature parameter omitted because the deployment rejects it; requested 0,
  effective sampling deployment-default. No deterministic-sampling claim.
- No index rebuild/upsert, alternate model, HITL, treatment execution or Phase 5.

| Stage | Result | Seconds |
| --- | --- | ---: |
| Patient Agent | Exact-copy/schema/grounding PASS | 6.468 |
| Patient Case RAG | One existing training case retrieved | 0.249 |
| Medical Knowledge RAG | One existing medical chunk retrieved | 0.194 |
| Diagnostic Agent | Valid abstention; both inventories empty | 6.374 |
| Diagnostic grounding | PASS under unchanged validator | Included above |
| Clinical Critic | Executed; schema/references PASS; assessment `revision_required` | 27.412 |
| E2E execution | PASS, runner exit code 0, all three structured outputs accepted | 62.713 including setup/evaluation |

Total reported API usage: **6311 input + 3126 output = 9437 tokens**.
Actual diagnostic core:

```json
{
  "primary_hypothesis": null,
  "differential_diagnoses": [],
  "patient_case_evidence": [],
  "medical_knowledge_evidence": []
}
```

### Critic outcome: stop, do not force a diagnosis

The critic received a valid abstaining DiagnosticResult and completed without a
schema or grounding exception. It did **not** endorse the diagnostic reasoning:
its `overall_assessment` is `revision_required`.

Its first exact objection is:

> The assertion that abstention is required is not established by the supplied evidence or task constraints.

It also said that claiming the observations cannot support *any* non-definitive
differential was too strong and that absence of relevant retrieved literature was
being conflated with lack of diagnostic information in the patient facts. It
recommended a cautious fact-based differential and clearer safety-oriented
uncertainty. It reported three unsupported points, four missing-evidence items,
two safety flags, no contradictions and no hallucination flags. These are model
reports, not adjudicated medical findings.

This is a substantive disagreement about the evidentiary threshold, **not** the
abstention inventory bug. The critic's own output passed reference checks, but
reference existence is not proof of medical entailment. No critic prompt was changed,
no recommended diagnosis was adopted, no automatic revision was performed, and live
testing stopped after this one case as requested.

### Correctness and success are distinct

- **Valid abstention:** YES.
- **Grounded diagnosis generated:** NO; none was required or forced.
- **Grounding failure:** NO for any accepted stage in this run.
- **Clinical Critic endorsement:** NO; revisions requested.
- **Technical E2E completion:** YES.
- **Diagnostic correctness / diagnosis top-k match:** NOT MEASURABLE for abstention,
  not a diagnostic error.

The unchanged evaluator explicitly records
`"Accepted abstention; diagnosis correctness is not measurable."`, with accepted
accuracy metrics null. Its legacy `primary_accuracy`/`top_k_accuracy` fields remain
0 as end-to-end *diagnosis yield* metrics; they must **not** be interpreted as proof
of a wrong diagnosis. The evaluator was not redesigned in this contract-only task.

## RAG quality observation — separate from the contract fix

The diagnostic investigation's medical top-1 hit had similarity approximately
**0.689855** and came from the PMC transplant/BK-polyomavirus article
*Pre-transplant immune factors may be associated with BK polyomavirus reactivation
in kidney transplant recipients* (PMC5451008). The model considered the excerpt
irrelevant to the current presentation. The new run retrieved the same chunk,
`medical:ac51854d-8e35-564f-abfb-478014277625`; the Diagnostic Agent again identified
the relevance limitation. The patient case remained `case:ddxplus:train:60`.

This is a retrieval-quality observation for later evaluation. Similarity is not
clinical relevance or diagnostic probability. Neither RAG nor the corpus was changed,
and retrieval limitations were not used to weaken grounding.

## Reproduction and artifacts

The one live command used a disposable observer around the **unchanged** production
runner, so exact validator failures would be captured without needing a second run:

```text
.venv\Scripts\python.exe healthcare-agentic-ai\outputs\phase4\abstention-contract\run_validation.py
```

Internally it invokes `scripts.run_phase4.main` with:

```text
--queries 1 --top-k 1 --output-dir <project>/outputs/phase4/abstention-contract/live
```

The observer calls the original validator and re-raises any exception unchanged.
No request fields, model, retries, schema, RAG handoff, or acceptance condition are
replaced. The live directory and an exclusive marker prevent accidental reruns.
Do not enable an additional paid integration run for this task.

Artifacts under `outputs/phase4/abstention-contract/`:

- `live/sample_case_001.json`: accepted patient state, both retrieved payloads with
  provenance, valid abstention, full critique, stage timings and telemetry.
- `live/phase4_run_report.json`: unchanged evaluator report and configuration.
- `validator-observations.json`: all three callbacks accepted; no validator exception.
- `live-console.log`, `exit-status.json`, `live-run-reserved.json`: single-run evidence.
- `tests-before-live.log`, `tests-final.log`: full offline regression results.
- `integrity-before.json`, `integrity-after.json`: counts, payload structures, vectors.
- `source-hashes-before.json`, `source-hashes-after.json`, `audit-after.json`: scope
  of changes and runtime-reference audit.
- `audit.py`, `run_validation.py`: disposable verification helpers, not runtime imports.

## Data integrity and backend isolation

Both collection counts, payload structures, and complete logical fingerprints
(sorted IDs + payloads + vectors) match before/after and the prior migration:

| Collection | Count | SHA-256 |
| --- | ---: | --- |
| `ddxplus_patient_cases` | 1000 | `6d3e80193bd098f7e8e3d5698cd5ce2b3b8d7c2e68ad245fa08132116f4401e2` |
| `medical_knowledge` | 87 | `92aac0f6bf3d68b941a0a1df9a0c89835dd0d57e3ea718560f94a9809973578b` |

Healthcare `rag/` and `scripts/` have **zero legacy local-generation runtime
references**: no provider, import, URL, environment requirement or fallback was
reintroduced. GPT-5.6-Sol remains the sole generation backend. BGE embeddings and
Qdrant remain local and unchanged. Ground-truth labels remain evaluator-only.

## Final requirements

```text
Grounding validation weakened: NO
Ground-truth leakage: NO
RAG data modified: NO
Ollama reintroduced: NO
Tests: 134 passed / 0 failed (1 skipped)
Live runs: 1
Diagnostic result: VALID ABSTENTION
Clinical Critic: EXECUTED; REVISION_REQUIRED
E2E execution: PASS
Clinical endorsement: NOT ESTABLISHED
Phase 5 / HITL: NOT STARTED
```

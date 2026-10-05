# Task 7.10 — exact-control preflight passed; prototype STOPPED

## Status

**Partial implementation, not a completed set-selection experiment.** All six
Task 7.9 controls reproduced exactly before any prototype implementation.
The required feature-definition work stopped under Task 7.10's explicit
feature-support provision (prompt lines 159 and 468). No set selector was
implemented or executed, and no replacement objective was tried.

The saved inputs suffice for exact control reconstruction. This is **not** a
missing-candidate, missing-score, failed-control, or memory-stop result. The
blocker is establishing the required semantic feature measurements without
substituting word overlap, inventing clinical rules, or repurposing review labels.
It is not a claim that no future deterministic text method could be defensible.

## Implemented

- `scripts/experiment_task710_selection.py`: isolated memory-bounded preflight.
  Authenticates the Task 7.9 historical protected manifest, verifies its files,
  freezes Task 7.9 artifacts, and loads one case at a time.
- Reuses only the authenticated pure original selector/tokenizer through Task
  7.9's AST extraction; no retrieval/model imports. Verifies Task 7.8 audits
  against Task 7.9 controls and per-candidate analysis traces.
- Compares IDs, exact final order, chronological order, roles, document and
  provenance metadata, every saved rejection/selection event, input fingerprints,
  and final count. Any discrepancy raises an error.
- `tests/test_task710_selection.py`: 12 focused deterministic preflight tests.
  Tests cover exact reproduction, altered IDs/order/roles/documents/count/events,
  missing candidates, eligibility/text tampering, original five-slot/document/
  Jaccard constraints, deterministic replay, all-six gating, and no mutation.
  These are **not** tests of a nonexistent prototype.

## Why feature construction stopped

The distinction is between having source text available for inspection and having
a defensible generic measurement for every required dimension.

| Required dimension | Available frozen information | Limitation; no substitute used |
|---|---|---|
| Direct relevance | Query facts, source text, role-specific saved logits, `concept_coverage` | Saved coverage is derived from `matches`, which tests whether alias/concept word sets are subsets of text words (`rag/focused_medical.py:78–81`). It does not establish meaningful correspondence, negation, or whether the match depends on another condition. A positive logit is eligibility, not an auditable fact-to-source explanation. |
| Incremental useful coverage | Matched-concept lists and query fact references | A set difference is mechanically possible, but would measure incremental lexical matches, not incremental useful information. No such lexical objective was substituted. |
| Query-dependent section usefulness | Section strings, body text, query facts and roles | Section titles are sometimes empty. Symptoms/history roles alone do not specify whether a definition, intervention, evaluation or management passage answers the information need. Neither universal symptoms priority nor a title lookup was introduced. |
| Context compatibility | `population_features`, age, query facts, source text | Existing population features inspect literal title/section age wording, deliberately excluding body text (`ranking_ablation.py:31–54`). Unknown is not compatible. These features do not determine body-level exposure/subtype requirements or whether those requirements are established. No default compatibility reward or disease-specific rule was added. |
| Self-containedness | Exact source chunks | Text is available, but capitalization, length, or final punctuation alone do not establish independent comprehensibility or resolve continuation references. No undocumented syntactic proxy was adopted. |

Concrete saved examples illustrating the distinction (not new relevance labels):

- `validate-1/experimental.json` starts with Fainting, whose population status is
  `unknown` despite source body wording about older people. Its saved metadata
  explicitly says it is based on literal title/section wording only.
- The following Anemia chunk begins `short of breath or have a headache.` It has
  an empty section string. Source availability does not make section usefulness
  or continuation resolution a precomputed trustworthy feature.
- Formatted queries contain fact text and references, not an adjudicated mapping
  from each source assertion to established exposure/subtype context.

The Task 7.8.1 A/B/C review is reserved for post-selection diagnostic comparison.
It was not used as a feature store, optimized against, or relabeled. No new model,
manual disease-specific exception, threshold, weight, or formula was introduced.

## Exact control results

Full IDs, order, roles, documents, titles and sections are saved in
`outputs/task7.10/control-preflight/report.json`; a readable ledger is in
`outputs/task7.10/set-selection/control-ledger.md`.

| Case | Saved MiniLM count | Saved MedCPT count | Exact Task 7.9 match |
|---|---:|---:|---|
| validate:1 | 1 | 5 | Both conditions, all events |
| validate:2 | 5 | 5 | Both conditions, all events |
| validate:3 | 1 | 5 | Both conditions, all events |

Pool sizes checked: 593, 416 and 425, respectively (1,434 candidate records per
condition; 2,868 role pairs per condition). **Candidates considered by a prototype:
not applicable; no prototype was run.** The control evidence maximum remains
five, per-document maximum two, and Jaccard rejection remains `>= 0.8` with the
original tokenizer. The old history/current/history schedule was replayed only
as the control; it was not presented as a joint set-selection prototype.

## Requested prototype outcomes

All prototype outcomes are **unmeasured**, not zero or negative results:

- Panic Disorder recovery/replacement/complementarity: unmeasured.
- Flu and Fainting retention by a prototype: unmeasured. Their saved MedCPT
  selections were reproduced as controls (Flu case 3; Fainting case 1).
- Asthma symptoms/definition/evaluation/background displacement and constraint
  comparison: no prototype comparison performed.
- Older Adult Mental Health: its MiniLM case-3 control selection reproduced;
  no prototype comparison performed.
- A/B/C cross-reference for the 70 transition pairs, including seven C-rated
  gains: no post-prototype comparison exists. No new relevance labels were made.
- Newly selected eligible candidates, useful candidates still excluded, marginal
  traces, overlap/diffs, and prototype constraint/rejection totals: not measured.
- Selection-stage improvement: **not demonstrated or tested**. No clinical,
  diagnostic correctness, accuracy or generalization claim is warranted.

## Execution and verification

Preflight command (from repository root):

```text
.venv\Scripts\python.exe healthcare-agentic-ai\scripts\run_memory_bounded.py -- scripts/experiment_task710_selection.py --output outputs/task7.10/control-preflight
```

The output directory is exclusive-create; this command will not overwrite the
completed run. Do not rerun it into the same directory.

Focused tests only:

```text
.venv\Scripts\python.exe healthcare-agentic-ai\scripts\run_memory_bounded.py -- -m unittest tests.test_task710_selection -v
```

- **12/12 focused tests passed** (0.337 seconds).
- Preflight: system peak **12.342 GB**, outer child-tree RSS **0.060 GB**.
  Synchronous monitor: system 12.341575680 GB, process-tree RSS 0.053895168 GB.
- Tests: system peak **12.104 GB**, outer child-tree RSS **0.034 GB**.
- All **258** protected paths verified unchanged before/after control replay.
  Final closeout verification is recorded separately in `set-selection/closeout.json`.
- Existing guards unchanged: 14 GB outer stop, 1.5 GB startup reserve, 13.5 GB
  synchronous analysis stop. No guard triggered.
- No parallel workers, downloads, models, reranking, embeddings, BM25, corpus,
  Qdrant, hidden DDXPlus labels or benchmark labels were used.
- Only isolated Task 7.10 source/test/doc/output files were created; existing
  production behavior, configuration, historical manifests and Task 7.4–7.9
  artifacts were not edited. The repository's pre-existing dirty files were left
  untouched. A new Task 7.10 OKF manifest documents this stopped status.

## One engineering recommendation

Obtain separate authorization for a generic, source-span-grounded feature contract
covering meaningful correspondence and exposure/subtype context before resuming
this five-slot prototype; do not substitute the saved lexical coverage or A/B/C
review labels for those measurements.

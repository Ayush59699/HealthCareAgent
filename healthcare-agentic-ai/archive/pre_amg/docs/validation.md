# Phase 2 execution evidence

> Historical phase-specific validation record; subsequent Phase 4 implementation
> and migration results are documented in [phase4.md](phase4.md) and
> [phase4-validation.md](phase4-validation.md). Earlier stop/status statements below
> describe the repository at the time of that phase.

Executed on Windows with the existing uv-managed Python 3.13.11 environment.
No packages needed installation: sentence-transformers 5.7.0, qdrant-client
1.19.1, torch 2.14.0+cpu (distribution version 2.14.0), transformers 5.17.0.
`uv pip check` reported all installed packages compatible.

## Completed checks

- Full suite: **52 tests passed**. Tests use real temporary Qdrant and a fake
  embedder only inside test fixtures; a local-data parser integration test also
  checks 100 records from each split.
- Initial real build: **0 -> 1,000 training records** using local
  `BAAI/bge-small-en-v1.5`, CPU, normalized **384-dimensional** vectors,
  **Cosine** distance, **512-token** context.
- **41 / 1,000** training texts exceeded context on each completed indexing pass.
  Embeddings were truncated; original full text remained in payloads.
- Real retrieval: **10 validation queries**, **top-k 5**, **50 TRAIN hits**;
  **0 query texts truncated**. The log prints each query ID and each ranked
  result's ID, cosine retrieval similarity score, source and complete text.
- Audit inspected **1,000 actual stored vectors/payloads**, checked deterministic
  point IDs and exact reconstructed label-free TRAIN source payloads.
- Payload fields exactly `patient_id`, `source`, `split`, `text`; split counts
  TRAIN **1,000**, validation **0**, test **0**. No pathology, differential,
  diagnosis or extra hidden payload field was present. Source-text comparison
  protects against label insertion into the text field as well.
- Actual validation/test patients were rejected at both ingestion and vector-store
  insertion boundaries, before any write.
- Closing/reopening Qdrant preserved **1,000** records and the payload fingerprint.
- First real re-index also processed **1,000 -> 1,000**, without duplicate points.

Storage: `healthcare-agentic-ai/data/qdrant/patient_cases` (local disk).
Collection: `ddxplus_patient_cases`. Embedding manifest includes model name,
normalization, dimension, max sequence length, `phase1-to_text-v1` representation
and empty prompt. The final sequential run **passed all six steps (exit code 0)**, including the
second re-index and audit. Counts stayed at **1,000** and payload fingerprints
were identical. Definitive status: `outputs/phase2-validation/summary.json`.
The persistent SQLite file was observed on disk at **5,496,832 bytes**.

## Example actual retrieval

Query `ddxplus:validate:1`:

| Rank | TRAIN patient ID | Cosine retrieval similarity score |
|---|---|---:|
| 1 | ddxplus:train:60 | 0.98770675 |
| 2 | ddxplus:train:221 | 0.98743610 |
| 3 | ddxplus:train:278 | 0.98658853 |
| 4 | ddxplus:train:474 | 0.98631807 |
| 5 | ddxplus:train:653 | 0.98554756 |

First hit source: `release_train_patients.zip!release_train_patients#row=60`.
Full text includes age 62, Female, pain characteristics, shortness of breath,
fatigue and observed antecedents. Disease names in antecedent questions are
legitimate features, not the withheld PATHOLOGY label.
These scores are **not diagnostic probabilities or clinical confidence**.

## Reproduction and artifacts

From `healthcare-agentic-ai`, using the existing environment:

```text
python scripts/validate_phase2.py --allow-download
```

The validator sequentially runs unit tests, bounded indexing, source/vector audit
and reopen, validation queries, re-indexing, then the audit again. It stores each
command's exit code and checks equal payload fingerprints/counts between passes.
No concurrent access to the local Qdrant directory is allowed.

Local artifacts (ignored by git):

- `outputs/phase2/index-first.json`: original empty-to-1,000 build.
- `outputs/phase2/dependencies.json`: actual environment versions.
- `outputs/phase2-validation/unit-tests.log`.
- `outputs/phase2-validation/index-first.json`, `index-second.json`.
- `outputs/phase2-validation/audit-first.json`, `audit-second.json`.
- `outputs/phase2-validation/retrieval.json` and `retrieval.log`.
- `outputs/phase2-validation/summary.json`: sequential validation status/exit codes.

## Issues fixed and limitations

- Root workspace test discovery needs the project import root; documented command:
  `python -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v`.
- Qdrant local deletion did not close SQLite before deleting files on Windows,
  causing stale vectors to return after recreation. A narrowly scoped local-client
  close-before-delete workaround and reset/reopen regression tests fix this.
- Xet model download stalled; the same BGE model downloaded with
  `HF_HUB_DISABLE_XET=1`. Windows warned that missing symlink support increases
  model-cache disk usage. Final validation ran with `HF_HUB_OFFLINE=1`.
- Tokenizer length-inspection warnings are expected for long source text; encoding
  truncates to the actual context. No source text is replaced by generated summaries.
- Python 3.13 unittest may emit an upstream Qdrant SQLite ResourceWarning for its
  temporary thread-safety probe; tests pass and persistent collection handles close.
- IDs are deterministic for unchanged split archive row order, not cross-release
  patient identity. Replaced/reordered datasets require a new index.
- Local Qdrant loads vectors into memory; full-corpus RAM/scaling was not validated.
- This bounded infrastructure smoke test is **not** a retrieval-quality benchmark
  or clinical validation. DDXPlus is synthetic. No Recall/MRR or diagnosis accuracy
  is claimed. Diagnosis-label agreement and cross-split feature-duplicate analysis
  require a separate offline evaluation, with labels excluded from embeddings,
  payloads and retrieval decisions.

Phase 3 remains authorized medical-reference ingestion/retrieval and evaluation
planning. No generation, medical knowledge RAG, clinical critic, orchestration,
HITL, feedback memory or autonomous clinical action has been implemented.

## Cleanup verification (2026-09-18)

The current document contained no corrupted, duplicated or garbled sections on
inspection. All preceding Phase 2 results, metrics, commands, limitations and
conclusions are preserved unchanged. Saved build, audit, retrieval and summary
reports were checked for consistency; the six-step real validation was not rerun
and its artifacts were not overwritten during this cleanup.

Fresh checks from the parent workspace, using the existing Windows environment:

```text
.venv\Scripts\python.exe -B -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v
uv pip check
```

- **52 tests passed in 2.715s**, with no failures or skips, including the local-data
  integration test covering 100 records from each split.
- `uv pip check`: **63 packages checked; all installed packages compatible**.
- The previously documented upstream Qdrant SQLite `ResourceWarning` appeared;
  the suite still passed. Expected error logs from negative parser tests are not
  test failures.
- README and architecture were checked against the implemented parser, ingestion,
  embedder, vector store, retriever, audit and CLI entry points. Documentation now
  distinguishes historical reference inspection from a currently available clone,
  and explains report-path and validation-rerun behavior.
- Removed the obsolete `rag/RUN THE RAG` note (its command is already in README,
  and its folder name was incorrect) and generated project `__pycache__` folders.
  Retained all validation evidence, dataset files, persistent index files and
  maintained scripts. Patient Case RAG implementation and tests are unchanged.

No cleanup blocker remains. The reference clone is absent from the documented
local path, so the historical external-source review was not repeated. Existing
scaling, retrieval-quality and clinical-validation limitations above still apply.
Phase 2 is clean and ready for Phase 3; Phase 3 has not been started.

# Conservative cleanup audit

Request: workspace `prompts/clean`. This is cleanup only; no production source, tests, prompts, configuration, retrieval policy or architecture changes are authorized.

## Inventory and decision (recorded before deletion)

The static inventory contains 1,246 files outside `.git/` and `.venv/` (the cleanup audit's own outputs are excluded). Counts are file counts, not module counts:

| Category | Files | Decision |
|---|---:|---|
| KEEP — active production/configuration/assets/compatibility | 176 | Preserve |
| KEEP — tests | 46 | Preserve; additional upstream tests counted under vendor |
| KEEP — research/history | 778 | Preserve, including failed runs and negative results |
| KEEP — vendor/upstream | 81 | Preserve source, fixtures, reports, stores and datasets |
| REVIEW — uncertain | 2 | Preserve `astra2.txt` and `gpt-5.6-sol.txt` at workspace root |
| DELETE — clearly unnecessary | 163 | Empty root `dump` and 162 regenerable `.pyc` files |

Full per-file classification: `../outputs/cleanup/inventory-before.json`. `.git/` and `.venv/` are deliberately excluded from traversal/deletion. Credentials are not displayed; `.env` files are not read or hashed. Dataset/model/index/knowledge assets are inventoried by metadata only. Ten research JSON files larger than 8 MiB were not content-scanned and are retained. No source file is proposed for deletion.

## Dependency verification

Read the workspace/application OKF manifests and relevant upstream/archive entries before inspecting source. Checked current README/architecture commands, Python imports (static AST inventory), scripts, shell/batch/configuration references, tests, YAML/OKF entries, documentation, and bounded research text. See `../outputs/cleanup/imports.json` and `scan-scope.json` for the exact scope. Static scanning is not a proof that all dynamic references are absent; ambiguous code stays.

Supported path: `scripts/run_phase6.py` / `scripts/demo.py` -> `demo/terminal.py` -> `application/workflow.py` -> `AMGEvidenceService` + `Phase6Orchestrator`; separate `PatientCaseRAG`, grounding and safety stay in place. `rag/amg/backend.py:load_upstream` dynamically loads vendor `medlineplus_lab`, with sibling `medlineplus_ingest` and `medlineplus_retrieval_check`; all are retained. Existing integration tests check supported entrypoints do not load legacy medical retrievers.

Intentional retention:

- `rag/amg/query_adapter.py`, `alias_safety.py`, `baseline_compression.py`, both evaluation scripts and `tests/test_amg_query_adapter.py`: referenced by experiments/tests and current manifests. New3 stays experimental; New4 stays **DO NOT PROMOTE**.
- `rag/focused_medical.py`, `rag/experimental_medical/`, `rag/retrieval_experiment.py`, older medical modules and Phase 4/5 scripts: imported by compatibility paths, evaluation scripts and regression tests. Not safe dead-code candidates.
- `docs/amg-*.md`, `outputs/amg-live/`, `outputs/new2/`, `outputs/new3/`, `outputs/new4/`, `archive/pre_amg/`, workspace `outputs/` and all historical logs/JSON: provenance and research history, not disposable just because generated. No database copies or failed-run directories were removed.
- All `vendor/amg/` source, tests, datasets, snapshots and research artifacts. Only source-backed interpreter bytecode is disposable.
- Byte-identical `codeagent.py` and `archive/pre_amg/amg_codeagent.py`: the latter is an explicitly documented historical copy, so keep both. The other duplicate-source group consists of empty package `__init__.py` files, which are not redundant packages and are retained.
- Root `astra2.txt` and `gpt-5.6-sol.txt`: role/ownership not established; retained without printing contents. Existing working-tree changes are not reverted.

## Proposed deletions and safety

Exact paths, reasons, references checked and safety justification for every candidate: [cleanup-deletions.md](cleanup-deletions.md) and `../outputs/cleanup/proposed-deletions.json`.

- `dump`: zero bytes, no source/importable extension, and no quoted path or redirection reference in the scanned repository. No research content is lost.
- 162 `.pyc` files in 19 `__pycache__` directories: each has a verified matching `.py` source. No exact path/cache filename references were found. Neither root Git nor nested vendor Git tracks the candidates. Source imports regenerate these files; validation disables bytecode writes. Only empty cache directories may be removed after their allowlisted contents are deleted.

All candidate reference searches returned zero matches (`candidate-references.json`). No source, test, original report, model, embedding, index, database, dataset, environment or dependency file is deleted.

## Validation

Before cleanup: project suite **547 run, 546 passed, 0 failed, 1 skipped** (opt-in real cloud E2E). Standalone upstream suite **56 run, 56 passed, 0 failed, 0 skipped**, using the existing system Python with upstream dependencies.

Post-cleanup results: **547 project tests run, 546 passed, 0 failed, 1 skipped**; **56 upstream tests run, 56 passed, 0 failed, 0 skipped**. Tests were not modified. Heavy model/store startup and fresh retrieval experiments were intentionally avoided.


## Completed execution and system verification

Removed all 163 allowlisted files and their 19 now-empty cache directories, totaling **1,858,572 bytes (1.77 MiB)**. Exact removed paths are in [cleanup-deletions.md](cleanup-deletions.md); execution records are `outputs/cleanup/deleted-files.json` and `removed-empty-directories.json`. No additional original files disappeared. Uncertain files were retained; they did not block this bounded cleanup.

| Suite | Before: run / passed / failed / skipped | After: run / passed / failed / skipped |
|---|---|---|
| Project | 547 / 546 / 0 / 1 | 547 / 546 / 0 / 1 |
| Standalone upstream | 56 / 56 / 0 / 0 | 56 / 56 / 0 / 0 |

Logs: `outputs/cleanup/tests-before.txt`, `tests-after.txt`, `upstream-before.txt`, `upstream-after.txt`. Expected CLI rejection messages and fake-provider messages in the project log are test scenarios, not live cloud calls. The baseline also reported a SQLite ResourceWarning; there were no test failures and no tests were changed to suppress it.

| Component | Result | Evidence / boundary |
|---|---|---|
| AMG retrieval | PASS (offline) | Existing adapter delegation/provenance/gate tests and upstream retriever tests; mocks/test stores, not a fresh production-store query |
| Patient RAG | PASS (offline) | Existing tests use real temporary Qdrant with deterministic fixture vectors, not production embeddings |
| Grounding | PASS | Existing reference, provenance, tampering and abstention tests |
| Safety | PASS | Existing deterministic/semantic contracts with fixture providers; no real clinical assessment |
| Phase 6 orchestration | PASS | Existing workflow/routing/revision/boundary tests, plus runner CLI help |
| Application startup | PASS (bounded) | Supported entrypoint import test, demo/runner help, demo sample discovery, and mocked validate-only startup test |

No real `--validate-only` model/index startup was run: it opens both cached embedding models and production stores. That heavyweight path, live generation, cloud connectivity and production-store retrieval were **not revalidated**. The offline checks do not claim clinical efficacy or end-to-end live validation. No downloads, model loads, production database rebuilds, new retrieval experiments or cloud calls were performed. Existing test fixtures create and clean up their own temporary stores.

Commands (sequential; from `healthcare-agentic-ai/` unless noted):

```text
python -B -m unittest discover -s tests -t .                 # before and after; workspace .venv Python
python -B scripts/demo.py --help
python -B scripts/run_phase6.py --help
python -B scripts/demo.py --list-samples
# From vendor/amg/, using existing system Python with upstream dependencies:
python -B -m unittest discover -s tests                     # before and after
```

Process-local settings: `RUN_GPT_INTEGRATION=0`, `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `ANONYMIZED_TELEMETRY=False`. No dependency installation or environment-file edits.

## Production changes and integrity

**Production files changed: none.** No source consolidation was justified. No tests, runtime prompts, dependencies, retrieval settings, safety, grounding or Patient RAG behavior were changed. Existing user working-tree changes were preserved.

Before editing the OKF indexes, all **961 hashed pre-existing files** were unchanged after both post-cleanup suites and CLI checks. After the intentional manifest updates, **959/961** remain identical; the only differences are workspace `okf.yaml` and application `okf.yaml`. All **214 scanned pre-existing Python source files** remain byte-identical. Both updated YAML manifests parse successfully. Every remaining file in the original inventory has the same size except those two manifests. See `outputs/cleanup/integrity-after.json`. This is not a full byte-hash verification of excluded credential/data/index/large-artifact content.

Added only this report, the exact-path deletion appendix, and local audit inventories/test logs under `outputs/cleanup/`; synchronized the two current OKF manifests. Archived/upstream manifests remain unchanged.

**CLEANUP COMPLETE — PRODUCTION UNCHANGED**

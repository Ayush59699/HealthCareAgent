# Healthcare Agentic AI — human–AI decision-support research prototype

Research prototype for **synthetic DDXPlus cases**, not clinical advice or a clinically validated diagnostic system.

## Current architecture

```text
Label-free DDXPlus case
  -> Patient Agent (exact validated facts)
  -> Patient Case RAG (existing BGE / Qdrant training cases)
  -> AMG MedlinePlus evidence (existing MiniLM / Chroma snapshot)
  -> immutable, provenance-checked evidence snapshot
  -> Diagnostic Agent -> Grounding -> Clinical Critic -> Safety
  -> bounded revision / final research output / abstention / block / human review
  -> typed FinalDecision (ALLOW / HUMAN_REVIEW / BLOCK)
```

**AMG retrieves evidence; it does not diagnose.** The active backend is the supplied AMG `medlineplus_lab.Retriever`, with its existing alias lookup, complete-unit passages and 1.10 squared-L2 gate unchanged. Its historical `experimental` collection name is retained to avoid reindexing or falsifying snapshot identity. The graph/answer demo, web searches and answer synthesis are **not** part of this application. The original plain AMG demo store was not the improved MedlinePlus dataset implementation.

## Layout

| Path | Responsibility |
|---|---|
| `application/` | Workflow composition, CPU/index preflight and final decision/review handoff |
| `rag/amg/` | Thin AMG adapter, native provenance contract, evidence service |
| `rag/patient_*`, `rag/embeddings.py`, `rag/vector_store.py` | Separate existing Patient Case RAG |
| `rag/agents/`, `rag/llm/` | Patient/diagnostic/critic contracts and structured provider |
| `orchestration/phase6/`, `safety/` | Versioned workflow, grounding, bounded revision, safety |
| `vendor/amg/` | Supplied AMG source, datasets, snapshot stores, tests and research reports (formerly `NEW RAG/AMG-RAG`) |
| `scripts/run_phase6.py`, `scripts/demo.py` | Supported batch/interactive entrypoints |
| `scripts/verify_amg_run.py` | Offline source/grounding/safety-input verification of saved runs |
| `tests/` | Existing regressions and new AMG boundary/integration tests |
| `docs/` | Current architecture and merge verification |
| `archive/pre_amg/` | Previous docs/output artifacts/OKF index and original README; preserved uncommitted research |
| `outputs/` | New local validation and run artifacts (gitignored) |

The old medical Qdrant retriever, focused/hybrid/reranking experiments and Phase 4/5 code remain **compatibility/research code only** for existing regression tests. Neither supported entrypoint constructs them. Historical commands and archived integrity manifests are not the active application. No original dataset, index, credential or benchmark label was deleted or rewritten.

## Run (Windows terminal, from this directory)

Use the parent workspace environment, which already contains the cached BGE and MiniLM models:

```bat
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe scripts\demo.py --list-samples
..\.venv\Scripts\python.exe scripts\demo.py --validate-only
..\.venv\Scripts\python.exe scripts\demo.py --sample 1 --architecture
..\.venv\Scripts\python.exe scripts\run_phase6.py --queries 1
```

If pip is unavailable in a uv-created environment, use `uv pip install -r requirements.txt --python ..\.venv\Scripts\python.exe`. On other platforms, use the Python interpreter for the environment containing these requirements.

- **No automatic indexing, downloads, fallback retriever or API calls during local validation.** Missing/stale/corrupt AMG snapshots fail closed.
- Existing stores: `data/qdrant/patient_cases/`, `vendor/amg/knowledge/medlineplus/`, and `vendor/amg/knowledge/medlineplus_lab/`. They are local assets, not committed to Git. Preserve the whole vendor tree when moving the project.
- Dataset default: this project's `data/ddxplus/`, then workspace `../data/ddxplus/`. Alternatively pass `--data-dir vendor/amg/examples` to use the supplied AMG DDXPlus copies. Labels are not loaded at runtime.
- Generation uses the **existing workspace `../.env` GPT_SOL configuration**. The vendor `.env` is not read by this integration. No credentials were changed or copied into source.
- Live generation sends synthetic patient facts and retrieved evidence to the configured cloud deployment. No web-search service is used.
- `--top-k` and `--medical-top-k` are independent (1–20). The old `--medical-storage-path` option is deliberately removed: the medical backend opens AMG's hash-pinned published snapshots, not the obsolete Qdrant store.
- Two different cached embedding models are necessary for the existing, different vector spaces: one BGE for patient cases and one MiniLM for AMG. Each loads once per batch and cases run sequentially; no cross-encoder or graph expansion. A 2 GB available-memory startup check is a conservative preflight, **not** a hard process memory limit.

If starting without the local assets, explicitly build the patient index with `scripts/build_patient_index.py` and follow `vendor/amg/README.md` for explicit MedlinePlus ingestion/lab build. Model caching/builds are opt-in setup work, never automatic application behavior.

## Verification

```bat
..\.venv\Scripts\python.exe -m unittest discover -s tests -t .
..\.venv\Scripts\python.exe scripts\verify_amg_run.py outputs\amg-live\sample_case_001.json
```

AMG's full standalone suite also tests optional graph/LangChain demos. Run it from `vendor/amg` in an environment with `vendor/amg/requirements.txt` installed. Those optional graph dependencies are **not required by the integrated application**.

See [merge verification](docs/amg-merge.md) for actual commands/results and limitations. The real first validation case completed as **BLOCK** with no technical failure, not as a successful diagnosis. A safety block is an intended terminal workflow result, not evidence of clinical effectiveness.


## Final decisions and pending human review

The completed workflow is validated in [FINAL implementation results](docs/final-system-validation.md);
see also the [pre-change audit](docs/final-system-audit.md). The latest real run completed
all stages and correctly returned **BLOCK**, with no technical failure and no released diagnosis.

Batch execution writes `final_decision_001.json` alongside `sample_case_001.json` and the
run report. The final artifact contains status, cited evidence, deterministic grounding,
critic/safety findings, uncertainty and an explicit review state. **BLOCK/HUMAN_REVIEW
withhold the diagnostic proposal**; the separate local workflow audit retains it for
human inspection, not clinical release. Review remains `pending`; no approval or scheduling
is simulated. `ALLOW` only means eligible research output after all existing gates.

The Critic now receives a typed claim/reference grounding report. Citation presence is
never labeled proof of medical support. Patient-case retrieval supplies analogous training
cases; the current patient's exact facts are a separate inventory. Retrieval and safety
policies remain unchanged; no New3/New4 experiment was promoted.

```bat
..\.venv\Scripts\python.exe -B scripts\run_phase6.py --queries 1 --output-dir outputs\my-new-run
..\.venv\Scripts\python.exe -B scripts\verify_amg_run.py outputs\my-new-run\sample_case_001.json --final-decision outputs\my-new-run\final_decision_001.json
```

Use a new output directory. The second command checks native source provenance, exact
safety input and the recomputed final handoff without a model or API call. Existing local
assets/configuration are sufficient; no rebuild or installation was needed for this task.

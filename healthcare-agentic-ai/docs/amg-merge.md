# AMG merge: decisions and verification

## What changed

The requested architecture was adopted with one source-based correction: the supplied AMG folder has **two different paths**. `AMG-with-KG.py` is an answer-generating graph/web demo using `new_VDB`; the improved dataset retrieval is `medlineplus_lab.py`, whose own reports contain the alias/chunking/gating improvements. The integration calls the latter unchanged. It does not mistake a generated answer for medical evidence and does not claim that the AMG graph demo is being called.

- Moved `NEW RAG/AMG-RAG` to `vendor/amg`, preserving source/dataset/snapshot layout. Existing 1,017-topic, 2,289-chunk MedlinePlus store was opened without rebuilding.
- Added a supported `application/` composition layer and `rag/amg/` evidence adapter/service/contract.
- Replaced the old medical retriever in both supported batch and demo entrypoints. Removed their obsolete `--medical-storage-path` switch.
- Kept Patient Case RAG separate and reused the existing Patient Agent, Diagnostic/Clinical Critic prompts and Phase 6 workflow mechanics.
- Added native AMG provenance rather than fabricating old UUIDs/chunking identifiers. Snapshot, grounding and safety contexts carry the original passage and metadata.
- Kept accepted evidence separate from rejected-candidate audit data. No AMG synthesis, web retrieval or graph expansion runs.
- Empty accepted medical evidence produces early abstention in the supported workflow. Existing critic/safety blocks and versioned revision protection remain.
- Archived old documentation/output artifacts, original README/OKF index and the unrelated nested coding-tool copy under `archive/pre_amg`. Cleared generated Python caches during restructuring. Historical implementation modules/scripts remain for regression compatibility, not supported application routing.
- Updated current README, architecture, run guidance, `MERGE.txt` decisions and OKF indexes. The working tree already had substantial uncommitted changes; no Git reset or destructive data cleanup was performed.

## Verification performed

### Existing + integration regressions

From `healthcare-agentic-ai`, using the workspace `.venv` Python:

```text
python -m unittest discover -s tests -t .
Ran 506 tests in 19.759s
OK (skipped=1)
```

This includes the pre-existing 492 tests and 14 new integration tests. The configured live-only integration test remains skipped; live execution was performed separately below. New tests cover native provenance round-trip, identical evidence in Diagnostic/Critic/Safety payloads, retrieval failure and no-evidence abstention, label-bearing input rejection, query bounds, source tampering, rejected/duplicate candidates, unchanged upstream search calls, human review, revision snapshot reuse and absence of legacy medical imports in the supported entrypoints. Existing demo/reporting tests now use native AMG fixtures without weakening their safety assertions.

Full log: `outputs/amg-regressions.txt`.

### Supplied AMG regressions

From `vendor/amg`, using the existing system Python 3.13 environment with AMG's optional standalone graph dependencies:

```text
python -m unittest discover -s tests -v
Ran 56 tests in 14.947s
OK
```

These are offline software regressions, including real temporary Chroma stores, not a new medical benchmark. The workspace application environment did not initially contain Chroma; `chromadb==1.5.9` was installed and added to the healthcare requirements, matching the supplied store reader version. Optional graph/LangChain dependencies are not imported by the application.

Full log: `outputs/amg-upstream-tests.txt`.

### Real local retrieval

```text
python scripts/demo.py --validate-only --detailed
```

- Real synthetic patient `ddxplus:validate:1`, loaded without labels.
- Existing patient index: **1,000** training cases; existing AMG collection: **2,289** chunks.
- One patient case and **five AMG passages** returned; native IDs/source provenance validated.
- Cloud requests: **zero**. No downloads, index builds, web searches or graph expansion.
- Five complete patient facts did not fit the single bounded retrieval query; the full facts remained available to agents.

Full log: `outputs/amg-local-validation.txt`.

### Real cloud end-to-end case

```text
python scripts/run_phase6.py --queries 1 --output-dir outputs/amg-live
```

The existing `gpt-5.6-sol` cloud deployment ran the Patient Agent, Diagnostic Agent and Clinical Critic. The workflow then executed grounding and deterministic safety policy:

| Item | Observed result |
|---|---|
| Patient | `ddxplus:validate:1` |
| Medical backend | `amg-medlineplus-v1` |
| Medical passages | 5, from native supplied MedlinePlus chunks |
| Workflow duration | 39.216 seconds, excluding setup/model loading |
| Cloud requests / structured repairs | 3 / 0 |
| Accepted diagnostic / critic versions | 1 / 1 |
| Safety coverage | `assessed` |
| Safety decision / final status | `BLOCK` / `blocked` |
| Technical failure | None |
| Semantic safety API request | Skipped by existing critic-block policy |
| Released clinical output | None |

The three critic safety flags and four missing-medical-reference findings triggered the existing safety policy. **The integration completed; it did not produce a clinically successful or releasable diagnosis.** The safety layer was not bypassed or tuned to make the case look successful.

Artifacts: `outputs/amg-live/sample_case_001.json`, `outputs/amg-live/phase6_run_report.json`, `outputs/amg-live-launch.txt`.

### Independent saved-run verification

```text
python scripts/verify_amg_run.py outputs/amg-live/sample_case_001.json
```

Verified all **five** saved passages' original text and native metadata against the published AMG chunks artifact, revalidated the immutable snapshot and diagnostic/critic grounding, and reconstructed the full SafetyInput. Its fingerprint exactly matched the committed safety ticket. This independently checks that AMG evidence reached the safety handoff, not merely that the report calls itself AMG.

Result: `outputs/amg-live-verification.json` (`failure: false`, 5 sources, 1 diagnostic, 1 critic, 1 safety input verified).

The import-boundary regression confirms that importing the supported batch/demo entrypoints does not load `rag.medical_retriever`, `rag.medical_embeddings`, `rag.focused_medical` or `rag.experimental_medical`. Actual retrieval was separately exercised against the real AMG store above.

## Remaining limitations

1. **Retrieval quality is not clinical validation.** The supplied AMG reports already describe weak respiratory rankings and some false abstentions. The 1.10 cutoff remains exploratory, not calibrated confidence. The real case's accepted passages did not provide adequate support for all generated hypotheses; safety correctly withheld them.
2. One literal case query must fit AMG's 256-token window. Complete omitted facts are counted, not silently truncated; no new multi-query strategy or reranker was invented. The downstream patient context is complete.
3. BGE and MiniLM have different existing embedding signatures and normalization. Two distinct small CPU models are required; they are not duplicate loads of the same model. The startup memory check is not a hard memory watchdog. Cases are sequential and no graph/cross-encoder is loaded.
4. Chroma may update internal acquisition bookkeeping when querying a persistent store. Source files, chunk artifacts and native source content are not rewritten by the integration; no index rebuild occurred.
5. Historical source modules and research scripts remain intentionally available for existing regression/reproduction imports. Old docs/outputs were archived, not destroyed; historical path-based integrity manifests may need their original layout restored to rerun old experiments.
6. Local datasets, models, vector stores and credentials are not committed. A fresh clone needs explicit setup. The integrated application uses the original workspace GPT_SOL configuration, not the vendored graph demo's `.env`.

No benchmark/relevance labels were changed or used to tune retrieval. No new benchmark or new retrieval experiment was created.

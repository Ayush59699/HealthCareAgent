# Task 7.8 — controlled reranker suitability: blocked by memory preflight

## Outcome

**Implementation added; one guarded launch attempted; the scoring experiment did not run.**
The launch passed its startup guard, strict Task 7.7 closeout integrity verification,
and 60 deterministic tests. It then stopped because the selected biomedical model's
estimated incremental memory did not fit under the unchanged synchronous 13.5 GB
budget. No second launch, substitute model, guard relaxation or next experiment was run.

**Reranker suitability is not demonstrated as the bottleneck by this attempt.**
Neither fresh baseline reproduction nor experimental scoring occurred. Do not treat
missing measurements as zero transitions, a negative result, or evidence of clinical
accuracy/diagnostic correctness.

## Sole selected alternative and pre-download inspection

- Model: `ncbi/MedCPT-Cross-Encoder`.
- Immutable revision resolved from public repository metadata before any weights:
  `71caf65d4927987813984f54c284405a13fcca49`.
- Rationale: a BERT-base biomedical query–article relevance cross-encoder, not an
  embedding encoder or an untrained classification head. It is a candidate for this
  experiment, not an established match for multi-finding patient queries and consumer
  medical passages. No model search using benchmark or hidden labels was performed.
- Inspected configuration: `BertForSequenceClassification`, 12 layers, hidden width
  768, one scalar label/logit, maximum positions 512, FP32.
- Approximate parameter count: **109,483,009**.
- Repository weight file: `pytorch_model.bin`, **437,998,062 bytes** (~438 MB).
- Weight LFS SHA-256 recorded in metadata:
  `61d5ccd48869e03500544525fc231641d7daa9ba267b202c82724750038dc1e0`.
- Conservative estimated incremental peak: **2.125996124 GB**, allowing two FP32
  weight copies plus 1.25 GB for framework, tokenizer and batch-2/512-token workspace.
  This is a preflight estimate, not a measured model RSS or an allocation guarantee.
- The preflight requires current system use **below 11.374003876 GB** for this
  estimate to fit under 13.5 GB; actual execution remains subject to both guards.
- System use at entry to the metadata phase: **12.397215744 GB**. Model-memory
  headroom was refused after inspecting metadata. **No model weights or tokenizer
  files were downloaded; neither reranker was loaded.** Only repository metadata and
  the small configuration JSON were fetched; no patient/query text was sent remotely.

Raw logits from differently trained models have different scales. Even a completed
fixed-zero-gate experiment must distinguish score calibration, relative ordering,
text compatibility and selection displacement; a greater count of positive logits
alone cannot establish suitability or medical usefulness.

## Preserved controls and implemented execution path

The isolated runner retains the pinned control
`cross-encoder/ms-marco-MiniLM-L-6-v2` at
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, cached locally only. It is designed to:

1. Authenticate the completed Task 7.7 closeout manifest by its SHA-256 and strictly
   verify every recorded current file. This attempt matched **284 protected paths**
   and captured an expanded **333-path** run freeze. It does **not** extend the
   Task-7.7-only historical config exception or regenerate historical manifests.
2. Reuse the Task 7.5 depth-50 pools for validate cases 1, 2 and 3 (593, 416 and 425
   candidates; 2,868 candidate-role pairs total). Verify the label-free patient
   state, frozen checkpoints, source payloads and unchanged selector replay.
3. Reproduce all three control baselines, checking the existing `1e-4` tolerance,
   eligibility, selected IDs and snapshots against Tasks 7.5/7.6/7.7 **before any
   experimental scoring**. Unload the control before loading the alternative.
4. Download only explicitly allowed files at the resolved immutable alternative
   revision, sequentially and only after a successful memory preflight. Verify the
   weight size/hash and load locally without remote code execution.
5. Score identical verbatim original query–passage pairs, one case at a time, CPU
   only, batch size 2, two threads, no workers. Refuse token truncation and nonfinite
   outputs; use raw scalar logits without sigmoid, rescaling or fallback.
6. Preserve evidence text, IDs, source/provenance, query/fact assignments and
   retrieval ordering metadata. Reuse the unchanged score-derived ranking formula,
   raw-logit **> 0** gate and Task 7.4 selector. No reindexing, embeddings, full-corpus
   loading, fresh retrieval, query rewriting, role prefixes or downstream agents.
7. Emit every candidate-role comparison, all four eligibility transition counts,
   selected gains/losses and full requested passage audits, separating rejection,
   eligibility with selection displacement, retention and new selection.

No production defaults, downstream agents, existing experiment modules, memory
runner, source data, historical artifacts or protected configuration were edited.
Post-stop work was limited to reading small run artifacts and writing this report
and OKF context; no post-stop model work, rehash pass or test rerun was performed.
Therefore there is no claimed post-run 333-path verification.

## Required analysis — unavailable, not zero

| Requested measurement | This attempt |
| --- | --- |
| Fresh control reproduction | Not performed; stopped before control load |
| Candidate-role pairs scored by control | 0 |
| Candidate-role pairs scored by alternative | 0 |
| Nonpositive → positive transitions | Not measured |
| Positive → nonpositive transitions | Not measured |
| Relevant-candidate eligibility improvement | Not established |
| Existing positive-control preservation/degradation | Not measured |
| Reranker suitability as demonstrated bottleneck | Inconclusive |

Panic Disorder, Flu, anemia, CKD/kidney-related passages, asthma symptoms, asthma
definition, Fainting and Older Adult Mental Health have **no new Task 7.8 passage
scores or selection outcomes**. The audit implementation covers these groups,
including all asthma sections and both roles, but no real-case audit was generated.
Historical Task 7.7 findings are not substituted for new measurements.

## Validation and memory

- **15 new Task 7.8 deterministic tests passed.**
- **60 combined tests passed**, including those 15 plus prefix-control,
  reranker-input and ranking-ablation regression tests; not 75 distinct tests.
- Tests cover memory estimates/headroom refusal, one immutable metadata revision,
  no download on insufficient headroom, unchanged pairs/provenance, all transition
  directions, exact zero gate, no truncation/fallback, baseline prerequisites,
  invalid models/batches, synchronous stops and selection-status distinctions.
- Syntax parsing passed for all four new Python files.
- Outer watchdog: unchanged **14 GB** stop and **1.5 GB** startup reserve.
- Synchronous guard: unchanged **13.5 GB**; model estimate is an additional
  conservative preflight, not a replacement for either guard.
- Outer sampled peak: **12.408 GB system / 0.132 GB child-tree RSS**.
- Synchronous sampled peak: **12.407332864 GB system / 0.127455232 GB tree RSS**.
- The stop is recorded as `RuntimeError` at
  `inspect-one-biomedical-model-metadata-no-weights`; it is an estimated-headroom
  refusal, not an actual allocation reaching the 13.5/14 GB stops.

## Files and artifacts

New isolated implementation:

- `rag/experimental_medical/reranker_suitability.py`
- `rag/experimental_medical/suitability_audit.py`
- `scripts/experiment_reranker_suitability.py`
- `tests/test_reranker_suitability.py`

Attempt artifacts:

- `outputs/task7.8/guarded-launch.log`
- `outputs/task7.8/controlled-medcpt/model-preflight.json`
- `outputs/task7.8/controlled-medcpt/comparison.json`
- `outputs/task7.8/controlled-medcpt/experiment-stopped.json`
- `outputs/task7.8/controlled-medcpt/current-phase.json`
- `outputs/task7.8/controlled-medcpt/frozen-files-before.json`
- `outputs/task7.8/controlled-medcpt/tests.log`

Command used once, from the repository root:

```text
.venv\Scripts\python.exe healthcare-agentic-ai\scripts\run_memory_bounded.py -- scripts/experiment_reranker_suitability.py --output outputs/task7.8/controlled-medcpt
```

The run directory is intentionally preserved and cannot be overwritten by the runner.

## Exactly one recommended next step

**Free sufficient system memory and authorize resumption of this same Task 7.8
comparison with the recorded MedCPT revision and all existing guards unchanged.**
No automatic resumption, promotion or additional experiment was performed.

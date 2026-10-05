# Task 7.4 — retrieval-first implementation and validation

## Outcome and scope

**Implemented an isolated, opt-in hybrid retrieval path over the existing Qdrant
medical collection, with unchanged immutable evidence snapshots. The previous
pediatric asthma retrieval failure improves. General clinical retrieval reliability
is NOT established; do not promote this experiment to production defaults yet.**

Task 7.4 supersedes the interrupted task 7.3 experiment. Existing unrelated working
tree changes were preserved. This task did not modify diagnostic/critic/explanation
agents, patient parsing, grounding, citation rules, safety validation, or abstention
contracts. No medical data was added. No relevance benchmark artifacts were read,
changed, or used by the retrieval experiments. All real patient probes used
`DDXPlusParser(..., include_labels=False)`; observed history is not a hidden label.

The existing demo/Phase 6 default remains the historical focused baseline. The
new implementation is explicitly available through
`FocusedEvidenceService.with_hybrid(...)` and the local inspection CLI. This is an
experimental retrieval change, not a silent workflow or safety-policy change.

## 1. Baseline implementation and confirmed failure points

The historical path creates one query from up to 12 symptom concepts and four
history concepts, adds demographic wording, retrieves five BGE/Qdrant chunks, then
requires title overlap or at least two symptom concepts in the body. Everything
else is rejected before downstream reasoning. It validates all hit provenance first.

Confirmed with the **current** corpus (not old five-document reports):

- The existing `medical_knowledge` collection contains **3,332 chunks**. The pinned
  source corpus contains **1,017 documents**. The patient collection is separate.
- For `validate:2`, the top five are Gastroenteritis, Heat Illness, Heart Attack,
  Depression, and Meningococcal Disease. Only Depression passes the old gate.
- **Asthma in Children first appears at rank 26** in the baseline broad diagnostic
  probe, outside the live top five. Its title would pass the old filter: the
  immediate asthma failure is candidate generation, not absence from the corpus.
- For `validate:1`, two Anemia passages enter the top five but both fail the old
  gate. Literal extracted history phrases such as `diagnosis anemia` contain
  unnecessary question wording, so they do not match the simple topic title.
- All five baseline candidates are discarded for `validate:3`.
- The interrupted hybrid prototype also capped the fused pool before reranking,
  deduplicated distinct windows by parent too early, could create a ten-concept
  query, always filled available final slots without a relevance decision, and
  returned only previews rather than evidence snapshots.

No cosine threshold can fix these different failure stages.

## 2. New retrieval flow

```text
Validated label-free PatientState
  -> observed concepts with source fact IDs
  -> semantic clustering, at most four concepts per query
  -> existing normalized 384-D BGE embeddings
  -> medical_knowledge Qdrant top 50 or 100 per query
     + BM25 over the same medical chunks with title/section context
  -> reciprocal rank fusion (k=60)
  -> stable-ID fusion and exact-source-span deduplication
  -> local cross-encoder over the entire deduplicated union
  -> relevance decision + concept/demographic/semantic ranking features
  -> diversity selection, at most five passages, possibly zero
  -> original validated medical chunk payloads
  -> unchanged FrozenEvidence / SHA-256 EvidenceSnapshot
  -> unchanged downstream grounding and safety contracts
```

`candidate_k` is a **per-query/per-channel** depth, not a final evidence limit or
post-fusion cutoff. No old concept-overlap filter runs in this path. In particular,
the recovered basic asthma passage was at fusion rank **101** at depth 50; cutting
the union to 50 before the reranker would have lost it.

`filter_candidates()` now validates and annotates overlap without rejecting hits.
The exact old behavior is named `baseline_filter_candidates()` and is used only
by the retained historical path and explicit baseline comparisons.
`query_plan(state, embedding=...)` exposes the short-query plan; the single `query`
field remains available for historical comparisons.

### Queries and label isolation

Concepts come only from presenting findings, symptoms and antecedents. Each query
retains its source fact references. Negatives, unknowns and uninterpreted numeric
scales do not become positive findings. No IDs, generated diagnoses, answer keys,
relevance judgments or model-generated summaries enter embeddings or BM25.

Semantic agglomeration is deterministic and size-bounded, not a disease template.
Existing synonym aliases are reused only when their wording is observed. No new
asthma/panic-specific retrieval rules or topic boosts were introduced. CLI `--topic`
probes run **after retrieval** and only locate exact titles in the diagnostics.

For `validate:2`, the final tested symptom queries were:

- feeling detached from body or surroundings; upper abdominal pain; choking; fear of dying
- sweating; shortness of breath; palpitations; numbness and tingling
- cramping pain; chest pain; flank pain; breast pain

The five history queries independently reflect anxiety, family psychiatric illness,
depression, asthma/bronchodilator history and head trauma, with child population
context. These clusters are mechanical retrieval groupings, not clinical assertions.

### Fusion and reranking

Dense and BM25 scores are not added together. Each channel contributes
`1 / (60 + rank)` to its candidate's RRF score. Each stable chunk ID appears once,
with all query/channel origins retained. Exact source-span duplicates are removed
before reranking; different windows of a parent are not discarded early.

The local model is `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. It runs on CPU, batch size two,
maximum pair length 512. Cached weights were available; no cloud inference or
model download was performed. Missing weights or invalid scores fail closed,
without reverting to dense-only or keyword-only retrieval.

Each candidate is scored against its strongest contributing query in each present
role (current symptoms/history). A nearest semantic query is used when there was
no channel origin for that role. All fused candidates are assessed. Raw logits and
query IDs are logged separately from evidence similarity.

Final selection requires a **positive model logit** for the selected role. This
is an explicit experimental general-domain model decision, **not a calibrated
medical confidence threshold** and not a cosine/keyword gate. Within that role,
ranking adds small concept-coverage, semantic, RRF and demographic features. All
sources already satisfy the existing trusted-source contract; no invented source
quality hierarchy is introduced.

Diversity allows at most two complementary passages per document, one per parent,
and rejects near-identical token sets (Jaccard >= .8). One qualified history
passage is prioritized; current-symptom evidence is then selected, with additional
qualified history filling unused slots. Empty or weak pools are not padded.

**Known limitation:** positive general-domain logits can still be clinically
inappropriate, while useful multi-symptom passages can receive negative logits.
The experiment makes this failure visible; it does not claim to solve it.

## 3. Baseline versus new behavior

These are retrieval diagnostics from three label-free validation records, not
accuracy/relevance metrics. No diagnostic agents were called.

| Case | Baseline retained | Hybrid depth 50 final evidence | Qualitative inspection |
| --- | ---: | --- | --- |
| validate:1 | 0 | Fainting (1 passage) | Useful background for lightheadedness; anemia, black stools and kidney/medication history remain underrepresented. |
| validate:2 | 1 (Depression) | Asthma in Children (symptoms and definition), Depression, Anxiety, Teen Depression | Basic respiratory/history information recovered. Psychiatric history dominates remaining slots; teen-specific material is imperfect for age 10. This is not proof of an active disorder. |
| validate:3 | 0 | Older Adult Mental Health (1 passage) | Poor match to fever/chills/rash/neck findings. Smoking/age context attracted an inappropriate history passage. Nonempty evidence is not success. |

Depth 100 produced the same final titles for these three cases. It expanded pools
and latency, without demonstrating better final selection in these runs.

### Actual recovered passages for validate:2

Source: MedlinePlus, **Asthma in Children**.

**Symptoms**, source ID
`medical:9da28175-590a-52bf-acf1-a43d1e0f2f99`:

> What are the symptoms of asthma in children? The symptoms of asthma in children
> include: Chest tightness Coughing, especially at night or early morning Breathing
> problems, such as shortness of breath, rapid breathing, or gasping for air

This is an excerpt; the unchanged full source passage, provenance, URL and section
are in the report and snapshot. It also contains wheezing and severe-attack warning
information. Depth-50 fusion rank 7 -> cross-encoder rank 1; history logit 7.809.

**Basic definition**, source ID
`medical:c6c80e91-fe4e-5e8e-bcaa-97981d8c5394`.

> What is asthma? Asthma is a chronic (long-term) lung disease. It affects your
> airways, the tubes that carry air in and out of your lungs. When you have asthma,
> your airways can become inflamed and narrowed. This can cause wheezing, coughing,
> and tightness in your chest. When these symptoms get worse than usual, it is
> called an asthma attack or flare-up.

Depth-50 fusion rank 101 -> cross-encoder rank 2; history logit 4.451. This confirms
recovery of useful basic clinical information, not merely an asthma title.

**The previous asthma-related retrieval failure improves at both depths.** The
history supports retrieving background asthma information; it does not establish
that the current presentation is an asthma exacerbation.

Other stage findings:

- Generic **Asthma** passages enter both retrieval channels but are not selected;
  more population-specific passages rank above them.
- **Panic Disorder** enters the depth-50 pool (dense rank 6 / BM25 rank 1 for one
  symptom query), reaches fusion rank 58 and reranker rank 27, but is not selected.
  This is a ranking/selection limitation, not a demonstrated corpus gap.
- Symptom-role logits for the recovered asthma passages were negative, despite
  high history-role scores. The reranker is not sufficiently reliable across roles.

## 4. Candidate-pool and latency statistics

Counts for each channel include repeated appearances across queries. Fused counts
are unique chunk IDs. The full deduplicated union was reranked.

| Case | Depth | Dense occurrences | BM25 occurrences | Fused / reranked | RRF duplicate occurrences | Final | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validate:1 | 50 | 450 | 450 | 593 / 593 | 307 | 1 | 114.36 |
| validate:1 | 100 | 900 | 900 | 1018 / 1018 | 782 | 1 | 162.94 |
| validate:2 | 50 | 400 | 400 | 416 / 416 | 384 | 5 | 74.73 |
| validate:2 | 100 | 800 | 800 | 707 / 707 | 893 | 5 | 128.19 |
| validate:3 | 50 | 300 | 255 | 425 / 425 | 130 | 1 | 74.96 |
| validate:3 | 100 | 600 | 505 | 762 / 762 | 343 | 1 | 105.32 |

Pre-rerank concept-filter loss was **zero** in every run. The existing Qdrant
corpus had no additional exact-text duplicates in these unions. For example,
validate:2 depth 50 collapses 384 repeated channel occurrences out of 800 (48%).
This is a duplicate-occurrence diagnostic, not a relevance score.

Baseline single-query top-five latency was 0.217, 0.135 and 0.190 seconds for
cases 1/2/3. Model/index setup for the multi-case run took 19.92 seconds, separately
from retrieval. Most hybrid latency is CPU cross-encoding, not Qdrant/BM25:
validate:2 spent 73.51 / 126.43 seconds reranking/selecting at depths 50 / 100.
These single-machine observations are not a performance benchmark or latency SLA.

## 5. Chunk/context experiment

Inspected the existing section-aware MedlinePlus chunker and 180-word/30-overlap
windows. They preserve provenance but can split sentences and omit topic context
from continuation text. A confirmed example is the selected case-3 passage,
which starts `disorders such as depression and anxiety.` mid-context.

The interrupted experiment's context builder was retained and completed with small
batches. It reconstructs paragraph/list/heading structure, checks normalized
content against the approved source parser, packs complete sentences/list items,
and keeps full parent evidence separately from title/heading/window retrieval text.
An oversized atomic span fails instead of silently truncating source evidence.

A **new** index was built at
`../data/medical_knowledge/experimental-task74/`, never over production Qdrant:

- 1,017 existing source documents; 2,992 windows; 2,927 parent sections.
- 480-token window budget; **zero truncated embeddings**.
- Build time 454.49 seconds; version remains `task73-context-sections-v1` because
  the representation itself was reused, not silently redefined.
- A tokenizer warning during candidate window sizing is not actual embedding
  truncation; the build's explicit zero-truncation invariant passed.

For case 2 at depth 50: 400 dense + 400 lexical occurrences -> 408 fused windows
-> 407 reranked after one exact-span duplicate. Final titles were Asthma in
Children (two sections), Teen Depression, Depression and Anxiety. Total 78.32 sec.
It restores source context but does not solve reranker/population specificity.

The context index uses immutable NumPy exact-cosine search **only for this separate
representation experiment**. The primary new retrieval path uses Qdrant. Context
records are intentionally preview-only: the existing strict chunk validator does
not accept their new identity scheme. The snapshot adapter rejects them rather
than inventing a legacy chunk version or weakening provenance validation.
Promotion of a new chunk schema/index requires a separately reviewed migration.

## 6. Evidence and safety boundary

`QdrantMedicalIndex` checks collection identity, embedding signature, normalized
vectors and every source payload. Changed candidate payloads during a session fail
closed. Retrieval context is `title + section + original text`; delivered evidence
is the exact original `validate_chunk()` payload. `evidence_hits()` does not add
ranking/context fields to the strict provenance allowlist.

The existing evidence similarity field receives a real BGE cosine. Cross-encoder
logits, concept/demographic features, RRF values and roles remain separate audit
metadata. Each selected payload is passed through `evidence_from_hits()`,
`FrozenEvidence.freeze()` and the unchanged content fingerprint / snapshot checker.
The real Qdrant inspections also reran the existing `context()` provenance audit.

Tests cover altered source text, mismatched collection/signature, invalid snapshot
fingerprint, invented patient findings, failed rerankers, empty observations,
negative relevance scores, preservation of the patient-case query, and rejection
of the old early gate in the hybrid path. None of this establishes entailment or
clinical correctness; diagnostic uncertainty, critic review, citation validation,
mandatory abstention contracts and safety decisions are unchanged.

No live diagnostic/critic/safety inference was run. No reduction in BLOCK rate is
claimed. An evidence-related BLOCK remains legitimate.

## 7. Memory controls and validation

The user's ceiling is **15 decimal GB of total machine memory**. The supervisor
`scripts/run_memory_bounded.py` checks available headroom before starting, samples
system memory every 50 ms, and terminates the experiment process tree at **14 GB**,
leaving a 1 GB margin. It reports machine and child-tree RSS peaks. It does not
reserve RAM against other applications or provide an absolute OS-wide guarantee;
close other memory-heavy applications and run only one experiment at a time.

Models are CPU-only; embedding/reranker batch size two; two PyTorch/BLAS threads;
no parallel model workers. Production-directory hashes stream in 1 MB blocks rather
than reading entire index files into memory. Completed case/depth reports are
checkpointed independently, so a stopped run does not erase completed evidence.

| Operation | Sampled system peak GB | Child-tree RSS peak GB |
| --- | ---: | ---: |
| Initial asthma recovery probe | 9.319 | 0.986 |
| Final three-case, two-depth Qdrant inspection | 11.682 | 1.223 |
| Separate context-index build | 12.060 | 0.741 |
| Context-index retrieval inspection | 10.264 | 0.944 |
| Final selected regression suite | 10.279 | 0.139 |

**383 selected software regression tests passed**, including all nine new task-7.4
contracts, existing retrieval tests and downstream abstention/grounding/safety
regressions. Relevance-benchmark test modules and opt-in integration/cloud tests
were excluded. Existing mock-provider/CLI tests print intentional errors while
exercising rejection behavior; the final test result is OK. There were zero live
cloud inference calls. This is software-contract validation, not clinical testing.

## 8. Reproduce and integrate explicitly

From `healthcare-agentic-ai/`, using the existing virtual environment:

```text
python scripts/run_memory_bounded.py -- scripts/inspect_hybrid_medical_retrieval.py --samples 1 2 3 --depths 50 100 --topic "Asthma in Children" --topic "Panic Disorder"

python scripts/run_memory_bounded.py -- scripts/build_experimental_medical_index.py

python scripts/run_memory_bounded.py -- scripts/inspect_hybrid_medical_retrieval.py --samples 2 --depths 50 --index ../data/medical_knowledge/experimental-task74 --topic "Asthma in Children"

python scripts/run_memory_bounded.py -- -m unittest tests.test_task74_retrieval tests.test_experimental_medical tests.test_focused_medical tests.test_retrieval_experiment -q
```

The builder refuses an existing destination: use `--output` with a **new** directory
for another build. Inspection output directories are also never overwritten. Stop
Qdrant writers while the inspection CLI snapshots the production index. It opens
only its temporary copy, verifies before/after hashes and deletes the copy on exit.
No network download occurs unless the explicit reranker-download flag is supplied.

Retrieval-only library opt-in, with existing patient/medical retrievers:

```python
service = FocusedEvidenceService.with_hybrid(
    patient_retriever, medical_retriever,
    top_k=1, medical_top_k=5, candidate_k=50,
)
snapshot = service.retrieve(patient_representation, validated_patient_state)
case_evidence, medical_evidence = snapshot.agent_evidence()
audit = service.audit
```

This requires no change to downstream agent interfaces. It is **not** automatically
activated by the historical demo's `--medical-top-k` option. Keep experiment results
and their audit separate from ordinary runs until remaining failures are addressed.

Artifacts produced under `outputs/task7.4/`:

- `qdrant-validation/inspection.json` and `.md`: final three-case/two-depth results,
  every query, channel ranking, fusion origin, deduplication/selection removal,
  reranker score/movement, full candidate source payload and immutable snapshot.
- `qdrant-validation/validate-*-depth-*.json`: per-depth checkpoints.
- `context-validation/inspection.json` and `.md`: source-context comparison.
- `qdrant-regression/`: initial exploratory recovery run before enforcing the
  four-concept cluster cap; use `qdrant-validation` for the final implementation.

## 9. Files inspected and changed

Core inspected files: `rag/focused_medical.py`, `rag/retrieval_experiment.py`,
`rag/experimental_medical/{index,queries,retrieval,inspection}.py`,
`rag/{medical_retriever,medical_embeddings,medical_vector_store,embeddings}.py`,
`rag/medical_ingestion/{chunker,models,corpus}.py`, `rag/agents/grounding.py`,
`orchestration/evidence.py`, `orchestration/phase6/orchestrator.py`, existing focused
and experimental tests, build/inspection scripts, requirements, README, and
`docs/enhanced-medical-retrieval.md`. The task 7.3/4 instructions were read; no
relevance benchmark was inspected to design or tune retrieval.

Added:
- `rag/experimental_medical/qdrant_index.py`: strict Qdrant and evidence adapter.
- `scripts/run_memory_bounded.py`: conservative memory supervisor.
- `tests/test_task74_retrieval.py`: new retrieval/snapshot contracts.
- `docs/task7.4-retrieval.md`: this report.

Changed in this task:
- `rag/focused_medical.py`: explicit hybrid service, non-filtering feature API,
  preserved named baseline, optional multi-query plan.
- `rag/experimental_medical/queries.py`: bounded cluster size/population audit.
- `rag/experimental_medical/retrieval.py`: broad-union reranking, exact-span dedup,
  small batches, final relevance/diversity and complete stage diagnostics.
- `rag/experimental_medical/inspection.py`: streaming hashes and task-7.4 reporting.
- `scripts/inspect_hybrid_medical_retrieval.py`: primary Qdrant experiment, optional
  context preview, original-evidence snapshots and per-depth checkpoints.
- `scripts/build_experimental_medical_index.py`: new destination/small-batch defaults.
- `rag/retrieval_experiment.py`, `tests/test_focused_medical.py`: explicitly name
  the archived baseline filter in historical comparisons/regressions.
- `requirements.txt`: declare `psutil` for the watchdog.
- `README.md`, repository `.gitignore`: usage and rebuildable index exclusion.

The production chunker, corpus, Qdrant content, embedding model/signature, source
validators, patient parser, frozen evidence implementation and all downstream
agent/safety files were **not changed by this task**. They may already have
uncommitted changes from earlier work; those were not reverted or attributed here.

## 10. Remaining failures, corpus sufficiency and next step

Failure classification after this experiment:

- **Candidate generation:** baseline top-five losses are confirmed. Multi-query
  hybrid search recovers pediatric asthma and panic-related candidates without
  requiring a hidden diagnosis.
- **Filtering:** old title/two-symptom gate demonstrably drops candidate evidence;
  its pre-rerank loss is zero in the hybrid path.
- **Ranking/final selection:** the dominant remaining problem. Generic cross-encoder
  scores favor single-topic history questions, miss useful multi-symptom material,
  and can accept weak population/risk-only matches. History and psychiatric topics
  can crowd out more relevant current-symptom evidence.
- **Chunking:** original continuation chunks can lose sentence/topic context. The
  separate representation improves context but does not cure ranking errors.
- **Corpus coverage:** the corpus is sufficient for the specific basic pediatric
  asthma recovery objective. Wider clinical coverage is unproven. Missing final
  evidence in these cases is not sufficient evidence to justify corpus expansion.

**Recommended next engineering step:** keep depth 50 and this opt-in boundary;
perform label-free ablations of query wording and local reranker choice. Improve
generic question/answer concept extraction (without disease templates), assess
shorter symptom-role query scoring and demographic/clinical-context mismatch, and
reduce redundant history selection. Use the already recorded broad candidates to
separate candidate generation from model ranking errors. Profile bounded-batch
cross-encoding/caching before adding workers or more documents. Do not weaken any
safety gate, change diagnostic behavior, or promote the context schema merely to
obtain more nonempty results.

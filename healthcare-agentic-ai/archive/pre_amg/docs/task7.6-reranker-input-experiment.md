# Task 7.6 — role-explicit natural-language reranker inputs

## Outcome

**The tested role-explicit format did not improve current-finding evidence
selection. Do not promote it.** It changed local learned scores substantially,
without recovering a single previously ineligible current-finding passage in the
three frozen cases. It also removed useful pediatric asthma background that the
original input selected.

This is a completed **negative experiment**, not a reason to relax the positive-logit
rule, modify safety, increase depth, or expand the corpus. The original format and
all production defaults remain unchanged. No diagnostic or safety agents were run;
there is no clinical accuracy, BLOCK-rate or treatment recommendation claim.

## What changed — and what did not

Added an offline reranker-input experiment, not a replacement retrieval path.

**Only the query-side text sent to the local cross-encoder changes.** These are fixed:

- Task 7.5's verified depth-50 candidate pools: **593 / 416 / 425** chunks.
- Original observation concepts, source fact references and symptom/history roles.
- Original query decomposition and each candidate's assigned supporting query in
  each role; no search for a more favorable query and no new query/candidate pairs.
- BGE embeddings, Qdrant medical collection, BM25, RRF, deduplication and chunking.
- Passage-side `retrieval_text`: title, section and original chunk text, byte-for-byte.
- Local `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`, CPU, maximum pair length 512,
  batch size two, two PyTorch threads. Cached weights only; no download.
- All non-neural ranking features and weights, and the existing final selector.
  No Task 7.5 role-quota, population-penalty or history-novelty variant is enabled.
- **Raw role logit must be strictly greater than zero.** Zero/negative scores remain
  ineligible. No threshold adjustment, score offset or cosine/keyword substitute.
- Original evidence text, source provenance, FrozenEvidence, EvidenceSnapshot,
  grounding, safety logic, abstention contracts and production/demo defaults.

All inputs come from the same three label-free parser records; no benchmark files
or hidden diagnosis fields inform queries, scoring, selection or inspection.

### The one tested formatter

A fixed template was defined before model scoring. Every observed concept phrase
is copied unchanged and joined with commas/conjunctions. No condition-specific
rules, synonym additions, disease predictions or fact rewriting are used.

**Current findings**

```text
Original:
sweating; shortness of breath; palpitations; numbness and tingling

Role-explicit:
The patient's current reported findings are sweating, shortness of breath,
palpitations and numbness and tingling. What medical information is relevant
to these current findings?
```

**Historical/background information**

```text
Original:
child asthma symptoms and clinical information

Role-explicit:
The reported historical or background information for a child is asthma.
What medical information is relevant to this history?
```

Population wording is preserved only where the original query already had it:
background queries. No age or sex is newly added to current-finding queries.
Reported history is not recast as a diagnosis of the current episode.

The formatter replaces generic query boilerplate with a role-specific sentence
and question; it preserves observed facts, not every function word in the original
input. In particular, the historical `symptoms and clinical information` suffix
becomes a question about information relevant to the history. Therefore this
experiment evaluates that **complete formatting bundle**, not the isolated effect
of adding the word “history.” It does not establish that every natural-language or
role-explicit formulation would fail.

## Experimental controls

1. Verify Task 7.5 checkpoint SHA-256 hashes and its protected-file manifest.
2. Reconstruct each original PatientState with `include_labels=False`, then exactly
   compare it with the frozen case.
3. Revalidate every source payload against the current approved medical chunk
   inventory; reuse Task 7.5's query/fusion/ranking/snapshot integrity checks.
4. Reconstruct pairs in original RRF order, keeping supporting-query IDs fixed.
5. Tokenize **both formats** before scoring that case. Refuse any pair over 512
   tokens instead of truncating a source passage or dropping a candidate.
6. **Recompute the original inputs with the same local model.** Require no
   eligibility changes and exactly the same evidence snapshot as the frozen baseline.
   A maximum absolute drift tolerance of `1e-4` is only a numerical-control check,
   not a retrieval eligibility threshold.
7. Score the role-explicit format sequentially, in batches of two.
8. Replace learned logits and recompute only their dependent ranking fields using
   the unchanged formula. Invoke the unchanged Task 7.4 selector and snapshot audit.
9. Record every candidate's original/formatted input, score change, eligibility
   transition, ranking/selection/removal reason, source text and provenance.
10. Verify all protected source, index and Task 7.5 artifact hashes again.

### Baseline reproducibility

| Case | Maximum absolute logit drift vs frozen baseline | Eligibility changes | Same evidence snapshot |
| --- | ---: | ---: | --- |
| validate:1 | 0.000002861 | 0 | Yes |
| validate:2 | 0.000003338 | 0 | Yes |
| validate:3 | 0.000002384 | 0 | Yes |

Thus the outcome differences below are not a changed baseline, candidate pool,
model selection, eligibility rule or passage-truncation artifact.

## Comparison table

Counts refer to **positive raw logits**, not relevance labels or clinical confidence.
Time includes formatting, token preflight, cross-encoder scoring, selection and
candidate audit. It excludes model load and frozen retrieval stages, which were not
rerun. RAM is sampled total-system decimal GB, including other applications.

| Case | Input | Positive current/history scores | Final evidence | Main outcome | Time (s) | Peak RAM (GB) |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 1 | Original, freshly scored | **1 / 0** | Fainting | Existing baseline reproduced | 89.22 | 10.633 |
| 1 | Role-explicit | **1 / 0** | Fainting | No recovered evidence; selected score decreases | 96.58 | 10.746 |
| 2 | Original, freshly scored | **0 / 15** | Asthma in Children: symptoms + basic definition; Depression; Anxiety; Teen Depression | Original pediatric respiratory background retained | 58.26 | 10.565 |
| 2 | Role-explicit | **0 / 1** | Asthma in Children: diagnostic evaluation | Loses useful basic definition and symptom passages | 64.52 | 10.566 |
| 3 | Original, freshly scored | **0 / 1** | Older Adult Mental Health | Known poor clinical-context match persists | 61.24 | 10.623 |
| 3 | Role-explicit | **0 / 0** | None | Removes poor match but does not recover acute-symptom evidence | 67.42 | 10.640 |

Across the three cases:

- **Zero nonpositive-to-positive transitions** in either role.
- All **1,434 current-role eligibility decisions remain unchanged**.
- Historical eligibility falls from 15 to 1 in case 2 and from 1 to 0 in case 3.
- Scores can increase while remaining negative; higher score is not itself evidence
  recovery. No final set is padded to five.

The smaller formatted outputs are **not an overall improvement**. They lose useful
background and do not restore the missing current-finding passages.

## Case-level evidence inspection

These are qualitative source inspections, not benchmark relevance labels, diagnostic
predictions or adjudicated clinical measurements. Full IDs, source URLs, passages,
channel ranks, query roles and selection events are in the JSON audit.

### Case 1: pallor/lightheadedness/fatigue with reported anemia and kidney history

```text
Same observed facts and 9 retrieval queries
 -> same 593 fused candidates
 -> same 1,186 candidate/role pairs
 -> changed query wording only
 -> 1 positive current score, no positive historical scores in either format
 -> Fainting remains the only final passage
```

Representative MedlinePlus passages:

| Passage | Role | Original logit | Formatted logit | Outcome |
| --- | --- | ---: | ---: | --- |
| Fainting | Current | +3.206330 | +0.224283 | Retained, substantially lower positive score |
| Anemia, basic definition | History | -2.941430 | -4.569625 | Still ineligible |
| Chronic Kidney Disease, basic information | History | -0.613731 | -4.724779 | Still ineligible |

Relevant IDs:

- Fainting: `3bc50c81-77f4-539e-a7d3-305beb00fdcb`.
- Anemia definition: `ccd2e542-f23b-5eb2-be0f-dc49e9e1b3b5`.
- Chronic Kidney Disease: `6892998d-d011-5971-9e5d-14a2111817d9`.

These background topics relate to reported history. Their continued rejection is
not proof of a corpus gap. No diagnosis is derived from that history.

Underlying phrases such as `diagnosis anemia` and `family who diagnosed anemia`
remain awkward in the new wrapper because rewriting the extracted facts was outside
this experiment. Role labeling alone did not correct that representation.

### Case 2: pediatric breathing/chest findings and asthma/bronchodilator history

```text
Same observed facts and 8 retrieval queries
 -> same 416 fused candidates
 -> same 832 candidate/role pairs
 -> 0 positive current scores under both formats
 -> history-positive pool shrinks 15 -> 1
 -> only a diagnostic-evaluation passage remains selected
```

**The task-7.4 basic asthma recovery regresses under this formatter.** The useful
passages are still in the candidate pool, with the same source text and supporting
query assignment, but both become ineligible:

| Asthma in Children section | Chunk ID | History logit: original -> formatted | Selection effect |
| --- | --- | --- | --- |
| What are the symptoms of asthma in children? | `9da28175-590a-52bf-acf1-a43d1e0f2f99` | +7.808829 -> -1.643670 | Retained -> ineligible |
| What is asthma? | `c6c80e91-fe4e-5e8e-bcaa-97981d8c5394` | +4.450912 -> -1.180367 | Retained -> ineligible |
| How is asthma in children diagnosed? | `72fa5e8b-af32-55b0-a830-9bb092ca9b3c` | +3.791914 -> +0.983198 | Document-diversity rejection -> selected |

The diagnostic-evaluation passage had fusion rank 90 and moves to reranker rank 1.
Its supporting history query has dense rank 2 / BM25 rank 3. It now wins because
higher-scoring complementary passages lost eligibility, **not because its score
improved**. The unchanged selector previously preferred two other sections of the
same document.

Its source begins:

> How is asthma in children diagnosed? It can be hard to diagnose asthma in
> children, especially if they are young. Asthma has similar symptoms as other
> childhood conditions.

This is background about clinical evaluation. It does not replace the recovered
basic definition and symptoms section, and does not establish asthma or an asthma
exacerbation in this patient.

The previously missed **Panic Disorder / symptoms** passage
`58a8e084-1882-557e-8aec-6fb5719992bb`. Current-role score changes from
**-2.388788 to -4.065693**, so it remains ineligible. It still contains source
information related to observed choking, sweating, palpitations and chest findings;
retrieval of that material would not constitute a diagnosis.

### Case 3: fever/chills/rash/neck findings and smoking history

```text
Same observed facts and 6 retrieval queries
 -> same 425 fused candidates
 -> same 850 candidate/role pairs
 -> no positive current score in either format
 -> only positive background score becomes negative
 -> empty, valid evidence snapshot; no fallback or weakened gate
```

| Passage | Role | Original logit | Formatted logit | Outcome |
| --- | --- | ---: | ---: | --- |
| Older Adult Mental Health | History | +0.242057 | -5.681695 | Poorly matched background is removed |
| Flu / symptoms | Current | -1.432960 | -3.694025 | Potentially useful symptom background still excluded |

IDs: Older Adult Mental Health `03f15d58-71a7-50a6-9f25-4a68fc2bda7e`. Flu symptoms:
`5ec42dd3-89ad-5b20-b546-bb0495e351de`.

Removing the mental-health passage is directionally sensible for the observed
acute presentation, but an empty set does not demonstrate reliable retrieval. The
flu source discusses fever/chills, nasal symptoms, muscle aches and fatigue; it
remains present in the pool and rejected. Existing empty-evidence and abstention
contracts remain intact. No agent was invoked to fabricate supporting evidence.

## Token length, latency and memory

| Case | Pairs per format | Original maximum pair tokens | Formatted maximum | Truncated pairs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,186 | 422 | 441 | 0 |
| 2 | 832 | 331 | 350 | 0 |
| 3 | 850 | 334 | 353 | 0 |

**5,736 local pair scores** were computed across six sequential runs. All candidates
were scored; no early filter, reranking budget cut or query reassignment was added.
The longer format took approximately 8–11% longer in these single CPU observations.
These timings are not a latency SLA or a clinical benchmark.

Model loading took **25.22 seconds**, separately from per-format timings. The model
was loaded once, and BGE was not loaded. Batch size stayed at two; no worker was added.

The existing memory watchdog remained in use, with a 14 GB stop below the user's
15 GB ceiling. Existing synchronous checks stop at 13.5 GB and record the current
case/format phase; pair progress is checkpointed every 100 scores. Peak sampled
system usage was **10.746 GB**, process-tree RSS **0.660 GB**. No memory stop occurred.
Sampling is not an OS-wide reservation against unrelated applications.

## Integrity, implementation and tests

**126 protected files were unchanged** before/after the experiment, including
source/index files, task-7.5 implementation and artifacts, and this experiment's
implementation. All snapshots use original source text and existing source/chunk
validators. No frozen evidence field receives the natural-language wrapper: it
exists only on the query side of a reranker pair and in the separate audit.

Added files only:

- `rag/experimental_medical/reranker_input_experiment.py`: fixed formatter, pair
  preservation, token guard, bounded local scoring, unchanged feature-formula replay,
  baseline drift checks and per-role score/eligibility comparison.
- `scripts/experiment_reranker_inputs.py`: Task 7.5 artifact verification, one-case-at-
  a-time execution, memory checks, original-input control and durable reports.
- `tests/test_reranker_input_experiment.py`: 14 deterministic contract tests.
- `docs/task7.6-reranker-input-experiment.md`: this report.

The initial 14 new tests passed; the final broader regression run passed **410 tests**.
Relevance-benchmark modules and opt-in cloud/integration tests were excluded. Existing
provider rejection messages are from mock tests, not live cloud calls. Tests cover
exact original query text, preservation
of fact phrases/roles and passages, frozen pair assignment/order, no new demographic
information, token-budget failure, batch/memory callbacks, invalid neural outputs,
unchanged positive-logit behavior, unchanged ranking features, snapshot reproduction,
baseline drift, and prohibition of retrieval calls/tuning flags.

Artifacts: `outputs/task7.6/role-explicit-inputs/`:

- `configuration.json`, `comparison.json`, `comparison.md`.
- `frozen-files-before.json`, `frozen-files-after.json`.
- `validate-N/frozen-depth-50.json`: exact Task 7.5 source checkpoint.
- `validate-N/token-preflight.json`, `baseline-drift.json`.
- `validate-N/original.json`, `role_explicit.json`: every selected **and rejected**
  candidate's source, title/section, IDs, query/fact references, dense/BM25 ranks,
  RRF scores, original/formatted pair inputs and hashes, token counts, raw/effective
  scores, coverage/population features, selection/removal events, and final snapshot.
- `validate-N/pair-comparison.json`: paired logit changes, eligibility transitions,
  gained/lost final IDs and original/formatted removal stages.
- `validate-N/summary.json`, `current-phase.json`, `scoring-progress.json`.

A null dense/BM25 rank refers to the assigned reranker query, not absence from the
whole candidate pool. All contributing channel origins are retained in the audit.

Run from `healthcare-agentic-ai/` using the existing virtual environment:

```text
python scripts/run_memory_bounded.py -- scripts/experiment_reranker_inputs.py
python scripts/run_memory_bounded.py -- -m unittest tests.test_reranker_input_experiment -q
```

No depth, model-download, batch-size, threshold or template-tuning CLI option exists.
Output directories must be new; prior artifacts are never overwritten. A missing
model, altered frozen input, baseline drift, invalid score or pair truncation fails
closed and records the stopping phase. This does not enable anything in the demo.

## Interpretation and next step

The experiment supports a narrow conclusion: **this role-explicit formatting bundle
changes learned scores but fails to restore useful current-finding evidence and
harms the prior pediatric basic-information recovery.** The selector then correctly
applies its unchanged rules to those new scores. It does not establish that all
role-aware inputs are unsuitable, that a particular disease is present, or that the
corpus should be expanded.

Unchanged fact fragments, supporting-query selection and general-domain model
limitations remain plausible contributors. Added role words, longer text and the
replacement of generic symptom-search boilerplate were not isolated from each other.
No thresholds or safety logic should be weakened to make this variant look successful.

**One recommended next engineering experiment:** a compact, prefix-only role control
that retains the original query verbatim, on these same frozen pairs. That would
separate role markers from this formatter's longer generic question and boilerplate
replacement. It is a prospective controlled test, not a production promotion or a
claim that a better result is guaranteed.

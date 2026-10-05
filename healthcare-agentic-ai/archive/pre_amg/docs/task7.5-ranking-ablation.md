# Task 7.5 — label-free ranking ablation

## Result

**The main demonstrated bottleneck is learned-score eligibility, not insufficient
candidate depth or final role weights.** Useful-looking passages can be in the
fused pool, and even near the top of the reranker, but remain ineligible because
both raw role logits are nonpositive. The historical selector requires a positive
raw logit *before* ranking features can help.

Five controlled variants were executed sequentially on each of the same three
label-free Task 7.4 cases. **No variant is an overall winner or promoted to
production.** A smaller/more diverse result is not automatically better, and a
nonempty result is not evidence of clinical usefulness.

The pediatric asthma recovery remains intact in four variants, including the
baseline. The background-cap variant retains the pediatric symptoms passage but
loses the complementary basic definition. None of these retrieval results
establishes asthma, an exacerbation, panic disorder, or another diagnosis.

## Frozen experiment design

This is a **verified frozen-pool replay**, not another retrieval/model run:

- Only `validate-{1,2,3}-depth-50.json` from Task 7.4 is loaded, one case at a time.
- Current patient records are independently reconstructed through the existing
  parser with `include_labels=False`, then exactly compared with saved PatientState.
- Query concepts, roles, source fact references, exclusions and wording are
  revalidated against those observed facts. Query text is identical across variants.
- All candidate evidence payloads must match the current approved source-only
  `processed/chunks.jsonl` inventory and pass the unchanged chunk validator.
- The unchanged RRF implementation is replayed against the saved channel rankings.
  Its candidate IDs, origins, scores and ordering must match the frozen pool.
- The pinned reranker identity and cached ranking formula are checked. Original
  Task 7.4 final selection and the complete evidence snapshot must reproduce exactly.
- Each ablation preserves input/pool/score fingerprints and produces an independently
  validated `FrozenEvidence` / `EvidenceSnapshot`, using the original evidence text.
- **96 existing source/index files** were SHA-256 checked before and after the run:
  unchanged. This includes BGE/Qdrant/BM25/RRF code, source data, production and
  experimental indexes, chunking, evidence contracts, agents, safety, demo and
  the memory watchdog.

No new labels, corpus data, models, embeddings, search calls, cloud calls, index
builds, workers or downstream-agent runs were used. Production/demo defaults remain
unchanged and the hybrid path remains opt-in. Neural scores are reused, **not claimed
to have been recomputed or independently authenticated**: checkpoint and
implementation hashes are recorded for reproducibility.

### The five alternatives

Each alternative changes one aspect relative to Task 7.4, not a cumulative tuning
recipe. Constants are fixed contrasts, not optimized against cases or labels.

| Variant | Single controlled difference |
| --- | --- |
| `task74` | Exact historical positive-logit eligibility, history-priority/current/history-fill order, source diversity and near-duplicate rules. |
| `symptoms_weight_2` | Double the current-finding logit's contribution to ranking. Raw eligibility, concept/demographic features and role order stay fixed. |
| `background_cap_1` | Explicit role allocation: retain at most one historical/background passage; never fill unused current-finding slots with more history. |
| `population_penalty_0.5` | Subtract 0.5 ranking points for a literal title/section age-population mismatch. No change to raw-score eligibility. Missing scope/age is unknown, not a mismatch. |
| `history_novelty` | Avoid selecting another document for the same history reranker query; allow complementary sections of the same document within the existing two-passage document limit. |

Population features use generic infant/child/teen/adult/older-adult wording and
observed age only, not disease rules or inferred pregnancy/diagnoses. They are soft
ranking features, not declarations that a medical source cannot apply. Incidental
population words in a passage body do not create a mismatch.

## Comparison table

Abbreviations: **AC-S** = Asthma in Children / symptoms; **AC-B** = Asthma in Children /
What is asthma?; **D-S** = Depression / symptoms; **A-S** = Anxiety / symptoms;
**TD-S** = Teen Depression / symptoms; **CMH** = Child Mental Health.
`D-S ×2` means two distinct chunks from the same symptom section, not a duplicated ID.

“Useful candidate missed?” is qualitative passage inspection, **not a relevance
label or benchmark measurement**. “Yes” does not imply a diagnosis.

Time is **selection + full candidate audit + snapshot validation only**, with cached
upstream results. Peak RAM is sampled total system memory in decimal GB, including
other applications. These are not end-to-end latency or OS high-water measurements.

| Case | Ranking variant | Final evidence | Useful candidate missed? | Main selection issue | Time (s) | Peak RAM (GB) |
| ---- | --------------- | -------------- | ------------------------ | -------------------- | -------: | ------------: |
| 1 | Task 7.4 | Fainting | Yes: anemia and kidney background | Both role logits nonpositive for missed passages | 0.4897 | 10.006 |
| 1 | Symptom weight ×2 | Fainting | Same | Weight cannot restore raw-score eligibility | 0.3735 | 10.003 |
| 1 | Background cap 1 | Fainting | Same | No history candidate eligible to allocate | 0.2788 | 10.003 |
| 1 | Population penalty | Fainting | Same | No relevant explicit age mismatch to correct | 0.3576 | 9.998 |
| 1 | History novelty | Fainting | Same | Redundancy is not the blocking stage | 0.3669 | 9.995 |
| 2 | Task 7.4 | AC-S, AC-B, D-S, A-S, TD-S | Yes: panic-related symptom knowledge | No positive current scores; historical material fills all slots | 0.2989 | 9.999 |
| 2 | Symptom weight ×2 | AC-S, AC-B, D-S, A-S, TD-S | Same | Weight cannot restore raw-score eligibility | 0.2655 | 10.003 |
| 2 | Background cap 1 | AC-S | Also loses AC-B | Coarse role quota discards complementary respiratory background | 0.1878 | 9.997 |
| 2 | Population penalty | AC-S, AC-B, D-S, A-S, D-S | Same symptom candidate missed | Teen mismatch reduced, but replaced with same-section redundancy | 0.2664 | 9.998 |
| 2 | History novelty | AC-S, AC-B, D-S, A-S, CMH | Same symptom candidate missed | Novel query/population, but replacement is a context-poor fragment | 0.2538 | 9.994 |
| 3 | Task 7.4 | Older Adult Mental Health | Yes: flu symptom background | Marginal history match wins; acute symptom passage ineligible | 0.3152 | 9.996 |
| 3 | Symptom weight ×2 | Older Adult Mental Health | Same | No eligible current passage to boost | 0.2562 | 9.995 |
| 3 | Background cap 1 | Older Adult Mental Health | Same | One wrong background passage still fits the quota | 0.1992 | 9.998 |
| 3 | Population penalty | Older Adult Mental Health | Same | Population matches, clinical context does not | 0.2607 | 9.998 |
| 3 | History novelty | Older Adult Mental Health | Same | Only one eligible background passage; no redundancy to remove | 0.2646 | 9.997 |

Original Task 7.4 end-to-end retrieval times were **114.36 / 74.73 / 74.96 seconds**
for cases 1/2/3 at depth 50. They were **not remeasured**. Subsecond replay times
must not be presented as faster end-to-end medical retrieval.

## Stage-level explanations

### Case 1 — observed pallor, dizziness/fatigue, black stools and documented history

Observed facts include pallor, lightheadedness, fatigue, head pain and black stools;
reported antecedents include anemia, family anemia, chronic kidney failure,
anticoagulant use and poor diet. These are input observations, not hidden labels.

```text
Observed facts
 -> 3 current + 6 history queries, unchanged across variants
 -> 450 dense + 450 BM25 appearances
 -> RRF: 593 unique chunks, all reranked; 307 repeated appearances fused
 -> 1 positive current score, 0 positive history scores
 -> Fainting selected in all five variants
```

Two informative rejected examples (both MedlinePlus; empty source section):

| Passage / chunk ID | Reranker query role; dense/BM25 rank for that query | RRF rank -> reranker rank | Current / history logits | Removal |
| --- | --- | --- | --- | --- |
| Chronic Kidney Disease / `6892998d-d011-5971-9e5d-14a2111817d9` | history; 1 / 4 | 56 -> 2 | -11.151 / -0.614 | No positive raw role logit |
| Anemia, basic definition / `ccd2e542-f23b-5eb2-be0f-dc49e9e1b3b5` | history; 2 / not in top 50 | 190 -> 11 | -10.432 / -2.941 | No positive raw role logit |

The kidney passage explains kidney function and chronic kidney disease; the anemia
passage explains oxygen transport and includes tiredness/dizziness. These are
plausibly useful background given the *reported* history. They are present in the
pool and not lost by RRF. Even reranker rank 2 is not enough to satisfy eligibility.

The history query `adult diagnosis anemia; family who diagnosed anemia symptoms
and clinical information` retains question boilerplate. Literal concept coverage
for the anemia passage is zero because these long phrases are treated as concepts.
That is evidence of imperfect query/feature representation, not proof that the
passage lacks clinical content. A shorter anemia continuation fragment ranks above
the basic definition; source context is a secondary problem.

**Primary demonstrated failure:** reranker/positive-logit eligibility. Query wording
and context are plausible contributors, not independently isolated causes.

### Case 2 — pediatric observed symptoms and asthma/bronchodilator history

Age 10. Current observations include sweating, breathing/chest findings,
palpitations, tingling, choking, detachment and fear of dying. History includes
asthma/bronchodilator use and anxiety/depression/family psychiatric/head-trauma
observations. No diagnosis is inferred from those facts.

```text
Observed facts
 -> 3 current + 5 history queries
 -> 400 dense + 400 BM25 appearances
 -> RRF: 416 unique chunks, all reranked; 384 repeated appearances fused
 -> 0 positive current scores, 15 positive history scores
 -> History fills the baseline final evidence set
 -> Selection-only ablations cannot recover an ineligible current passage
```

Current query examples remain:

- `sweating; shortness of breath; palpitations; numbness and tingling`
- `feeling detached from body or surroundings; upper abdominal pain; choking; fear of dying`

**Pediatric respiratory knowledge remains recoverable.** In baseline, doubled
symptom weight, population penalty and history novelty, both the source's symptoms
section and basic definition are delivered:

- `medical:9da28175-590a-52bf-acf1-a43d1e0f2f99`, **Asthma in Children / symptoms**:
  fusion rank 7 -> reranker rank 1; current -7.991, history +7.809.
- `medical:c6c80e91-fe4e-5e8e-bcaa-97981d8c5394`, **Asthma in Children / What is asthma?**:
  fusion rank 101 -> reranker rank 2; current -10.903, history +4.451.

The definition says asthma is a chronic lung disease affecting the airways, with
inflammation/narrowing and possible wheezing, coughing and chest tightness. That is
basic clinical background, not just a matching title. The symptoms section includes
shortness of breath and chest tightness. The cap-one variant keeps only the symptoms
section: a role quota can reduce distraction while losing useful complementary text.

A missed candidate:

- **Panic Disorder / What are the symptoms of panic disorder?**, MedlinePlus,
  `58a8e084-1882-557e-8aec-6fb5719992bb`.
- Originating current query: detachment; upper abdominal pain; choking; fear of dying.
- Dense rank 6, BM25 rank 1; RRF rank 58 -> reranker rank 27.
- Current -2.389, history -4.392; no positive raw-role score in any variant.
- Existing coverage feature matches four current concepts: choking, sweating,
  palpitations and chest pain. Source text additionally discusses racing heart,
  trouble breathing and fear during attacks. It is background for reasoning, not
  confirmation that this child has panic disorder.

**Population/diversity trade-offs are real but secondary:**

- Penalizing the explicitly teen-scoped title for age 10 removes TD-S, but a second
  Depression symptom-section chunk fills the slot. A soft mismatch feature alone
  does not ensure complementary evidence.
- History novelty replaces TD-S with **Child Mental Health**, chunk
  `9218536a-a168-5067-8bcd-c46275353714`, via the separate family-history query.
  Its age scope fits, but its evidence begins `suddenly getting hurt often Repeated
  thoughts of death ...` and ends with general diagnosis/treatment text. It is a
  truncated-context continuation, not a clearly superior clinical passage.

**Primary demonstrated failure:** learned current-query eligibility and asymmetric
role scores. Historical backfill amplifies the effect. Demographic mismatch,
redundancy and context are secondary; none of these variants solves the first stage.

### Case 3 — acute fever/chills/rash/neck findings with smoking history

Age 68. Observations include sweating, fever/chills, rash/neck findings, muscle
pain, fatigue/appetite changes and nasal congestion. The relevant history query is
based on reported smoking, not an inferred mental-health condition.

```text
Observed facts
 -> 5 current + 1 history queries
 -> 300 dense + 255 BM25 appearances
 -> RRF: 425 unique chunks, all reranked; 130 repeated appearances fused
 -> 0 positive current scores, 1 positive history score
 -> Older Adult Mental Health selected by all five variants
```

Missed **Flu / What are the symptoms of the flu?**, MedlinePlus,
`5ec42dd3-89ad-5b20-b546-bb0495e351de`:

- Query: `sweating; fever; chills shivers; nasal congestion clear runny nose`.
- Dense rank 15; BM25 rank 6; RRF rank 36 -> reranker rank 3.
- Current -1.433; history -10.452. Rejected by raw-score eligibility, not final budget.
- The source discusses fever/chills, sore throat, runny/stuffy nose, muscle aches
  and fatigue. This is plausible symptom background without establishing flu.
- The literal coverage feature counts only fever; wording variants and long
  compound concepts limit coverage. That is not evidence that other concepts are
  clinically absent from the passage.

Selected **Older Adult Mental Health**, MedlinePlus,
`03f15d58-71a7-50a6-9f25-4a68fc2bda7e`:

- History query: `older adult smoke cigarettes symptoms and clinical information`.
- Dense rank 2; BM25 rank 49; RRF rank 13 -> reranker rank 1.
- Current -9.357; history **+0.242**. Positive history eligibility wins despite
  zero exact history-concept coverage.
- It mentions increased smoking among mental-health warning signs, not an
  explanation of the observed acute presentation. Evidence begins mid-context:
  `disorders such as depression and anxiety.`
- Age scope is a **match**, so an age-mismatch penalty cannot correct the clinical
  context mismatch. One eligible history passage also defeats cap/novelty remedies.

**Primary demonstrated failure:** inappropriate history-query scoring plus no
eligible current candidates. Clinical-context mismatch is not equivalent to an
age mismatch. Chunk context is a secondary limitation.

## Mechanism conclusions and limits

| Mechanism | What these ablations establish |
| --- | --- |
| Query construction | Boilerplate/compound phrases and role wording are visible concerns; their causal effect on neural scores was not isolated by replay. |
| Candidate generation | The specific inspected missed passages are already present; raising depth is not indicated by these examples. |
| RRF | All fused candidates survive to reranking. Examples at ranks 23–190 remain available; rank alone is not their removal cause. |
| Reranker | **Main demonstrated bottleneck:** 1/0, 0/15 and 0/1 positive current/history counts. Near-top candidates can still be ineligible. |
| Role weighting/allocation | Weight doubling cannot alter the fixed eligibility set. Cap-one sacrifices useful background rather than recovering current evidence. |
| Demographic mismatch | Corrects an explicit teen-vs-age-10 rank comparison; does not establish clinical relevance or address case 3. |
| Redundancy/diversity | Changes which eligible background fills the fifth slot; exposes new context weaknesses, not an overall improvement. |
| Chunk/context | Several selected/rejected passages are continuation fragments. Frozen here; no index or chunk change was made. |
| Corpus coverage | Not the explanation for the specific inspected misses; broader coverage is not established or disproved. |

No clinical accuracy, benchmark relevance score, BLOCK reduction, entailment
success or generalization claim is made. There are only three presentations; no
causal claim separates the learned model itself from its query/input format.
The supporting-query choice used for each reranker pair is also frozen: the
inspected candidates are not lost at RRF, but RRF's contribution to which query
scores a candidate was not independently ablated.

## Memory, integrity, tests and artifacts

The unchanged outer watchdog stops at 14 GB total system usage, below the user's
15 GB ceiling. The ablation runner additionally checks memory synchronously during
selector/audit iterations and stops at 13.5 GB, writing the exact phase and memory
usage to `memory-stop.json`. `current-phase.json` also makes an outer watchdog kill
traceable. No model was loaded, no batch size increased, and no worker added.

Final run: outer-watchdog sampled peak **10.015 GB system / 0.157 GB process-tree
RSS**. The earlier exploratory replay peaked at 10.186 / 0.158 GB. No real memory
stop occurred; its phase-reporting behavior is covered by a simulated test. Polling
and synchronous samples are not a global memory reservation against other apps.

Artifacts in `outputs/task7.5/validated-ranking-ablation/`:

- `comparison.json`, `comparison.md`: all 15 variants, timing/memory and stage summaries.
- `frozen-files-before.json`, `frozen-files-after.json`: 96 unchanged protected files.
- `validate-N/frozen-depth-50.json`: original immutable source checkpoint with hashes.
- `validate-N/<variant>.json`: exhaustive selected **and rejected** candidate audit:
  title, section, source/URL, chunk ID, every role's originating query/fact references,
  dense/BM25 ranks, RRF score/rank, raw/effective reranker ranking scores, concept
  coverage, explicit population features, selector events, removal stage/reason,
  original evidence text and validated final snapshot.
- `validate-N/summary.json`: completed-case checkpoint.

A null dense/BM25 rank means that candidate was not in that channel's top 50 for
that particular reranker query. Other query/channel origins are retained; null does
not mean the candidate was absent from the whole pool.

Added files only:

- `rag/experimental_medical/ranking_ablation.py`
- `scripts/ablate_medical_ranking.py`
- `tests/test_ranking_ablation.py`
- `docs/task7.5-ranking-ablation.md`

Tests cover exact baseline replay, unchanged inputs, source/query/fusion/score/
snapshot tampering, depth/variant restrictions, no raw-gate change, soft population
handling, history-role allocation, full rejected-candidate diagnostics and memory
phase reporting. Initial focused validation: **75 tests passed**, including the 13
new ablation tests and existing Task 7.4/focused-retrieval contracts. Final broader
validation: **396 tests passed**. Relevance-benchmark test modules and opt-in
cloud/integration tests were excluded; provider error messages in the suite come
from existing mock rejection tests, not live inference.

Reproduce from `healthcare-agentic-ai/`, with the existing environment:

```text
python scripts/run_memory_bounded.py -- scripts/ablate_medical_ranking.py
python scripts/run_memory_bounded.py -- -m unittest tests.test_ranking_ablation tests.test_task74_retrieval tests.test_experimental_medical tests.test_focused_medical tests.test_retrieval_experiment -q
```

The runner accepts only the three existing case IDs and the fixed five variants.
There is no depth, batch, model-download or tuning flag. It refuses existing output
directories and does not overwrite Task 7.4 artifacts. This experiment does not
activate anything in the demo or live orchestration path.

## Exactly one next engineering change

**Add a role-explicit, natural-language reranker-input formatter as a score-only
experiment on these same frozen depth-50 pools, using the existing local model.**

Compare source-referenced “Current observed findings: …” and “Historical/background
information: …” inputs with the existing semicolon/boilerplate queries. Preserve
all observed concepts, the original evidence text, candidate IDs, raw positive-logit
selection rule and downstream contracts. Do not promote the formatter automatically.

This is the narrowest informative next change because the ablation establishes
that eligible-current shortages occur *before* selection weights, while the same
pediatric respiratory passages have strongly negative current-role and strongly
positive history-role scores. The current experiment cannot yet distinguish
query/input-format mismatch from model limitations; a fixed-pool input-format
comparison isolates that variable without expanding data, weakening a gate, or
mistaking a different fifth history passage for clinical progress.

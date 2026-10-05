# AMG retrieval investigation — measured root cause

Request: `prompts/new2.txt`. Case: `outputs/amg-live/sample_case_001.json`, `ddxplus:validate:1`. **Investigation only: no production code, prompts, thresholds, stores, labels or agent behavior changed.**

## A. Root cause

**The single, verbose case embedding favors broad pain/fatigue passages over the specific medical topics. Important history is also omitted before embedding. The top-five candidate limit then discards even Anemia, whose distance would pass the gate.** This is now measured rather than inferred from document availability.

The saved query was replayed through the unchanged AMG retriever against a disposable byte-copy of its existing Chroma store. It reproduced all five IDs and distances **exactly**. An independent exhaustive squared-L2 calculation against all **2,289 stored vectors** reproduced the same top five (maximum numerical distance difference `8.58e-8`). These are **global chunk ranks**, not diagnosis rankings, and no production top-k was increased.

### Saved-query distances and exact ranks

Native chunk IDs below share `medlineplus:topic:`. All evidence citations additionally have the `medical:` prefix. Smaller distance is closer; the unchanged cutoff is **1.10**.

| Chunk ID suffix | Topic | Exact rank | Squared L2 | Gate relationship |
|---|---|---:|---:|---|
| `6076:lab:1:chunk:2` | Blood Clots | 1 | 0.733683 | Returned; passes |
| `89:lab:1:chunk:2` | ME/CFS | 2 | 0.845546 | Returned; passes |
| `32:lab:1:chunk:1` | Fibromyalgia | 3 | 0.863152 | Returned; passes |
| `5624:lab:1:chunk:2` | Chronic Myeloid Leukemia | 4 | 0.920713 | Returned; passes |
| `351:lab:1:chunk:0` | Pain | 5 | 0.923876 | Returned; passes |
| `139:lab:1:chunk:0` | **Anemia** | **21** | **1.052496** | Within cutoff, but outside top five |
| `4861:lab:1:chunk:1` | **Blood Thinners — bleeding warnings** | **258** | **1.346185** | Outside cutoff and candidates |
| `1308:lab:1:chunk:0` | **GI Bleeding — black/tarry stools** | **290** | **1.364999** | Outside cutoff and candidates |
| `4861:lab:1:chunk:0` | Blood Thinners — definition | 974 | 1.615433 | Outside cutoff and candidates |
| `1308:lab:1:chunk:1` | GI Bleeding — investigations | 1,010 | 1.623815 | Outside cutoff and candidates |
| `302:lab:1:chunk:0` | **Kidney Failure** | **1,780** | **1.836119** | Outside cutoff and candidates |

**Anemia was not rejected by the gate:** it never reached the gate because only five neighbors were requested. The other requested topics also never reached it; their measured distances show they would fail the existing cutoff even if considered. Relaxing the gate alone cannot change a top-five candidate set.

## B. Contributing factors, with controlled checks

### 1. Query construction: repetitive questions, not a balanced clinical representation

The adapter concatenates literal presenting evidence, symptoms and then history, greedily keeping complete facts within 256 tokens. It adds no section labels, concept weighting or explicit distinction between current symptoms and historical context. The agents still receive the complete patient; this limitation is specific to retrieval.

The **exact text passed to AMG and actually embedded** is:

```text
Is your skin much paler than usual? = Yes; Do you have pain somewhere, related to your reason for consulting? = Yes; Characterize your pain: = tugging; Characterize your pain: = a cramp; Do you feel pain somewhere? = back of head; Do you feel pain somewhere? = top of the head; Do you feel pain somewhere? = forehead; Do you feel pain somewhere? = temple(L); How intense is the pain? = 2; Does the pain radiate to another location? = nowhere; How precisely is the pain located? = 3; How fast did the pain appear? = 5; Do you feel slightly dizzy or lightheaded? = Yes; Do you feel lightheaded and dizzy or do you feel like you are about to faint? = Yes; Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? = Yes; Do you constantly feel fatigued or do you have non-restful sleep? = Yes; Have you recently had stools that were black (like coal)? = Yes; Do you have a poor diet? = Yes; Have you ever had a diagnosis of anemia? = Yes
```

There are **254 tokens**, including two special tokens, and 988 characters. Full deduplicated facts would require 339 tokens. Marginal contributions, including separators, sum to 252:

| Fact group | Tokens contributed |
|---|---:|
| Eleven pain questions, including location and undefined numeric scales | **120** |
| Two fatigue/sleep questions | 47 |
| Two dizziness/near-fainting questions | 35 |
| Black-stool question | 17 |
| Prior-anemia question | 13 |
| Poor-diet question | 10 |
| Presenting pallor question | 10 |

The longest individual entry is the disabling-fatigue/“bed all day long” question, at 28 marginal tokens; the near-fainting question contributes 22. Pain takes about **47% of the entire query window**. This is token accounting, not a claim about neural attention weights.

Five history facts are omitted: **family anemia, kidney failure, oral anticoagulants, travel, and underweight status**. Anticoagulant/renal contexts therefore cannot affect the baseline query vector. Black stools and prior anemia *are* present; omission alone cannot explain their poor topic ranks.

### 2. Embedding behavior: literal-fact probes establish dilution and its limits

Two deletion/restoration probes were followed by four isolated original-fact probes. They introduce **no diagnosis labels or new medical wording**. Each uses the existing model, stored document vectors and unchanged AMG top-five/gate. Direct literal-vector comparisons are explicitly separated from AMG's alias-expanded output.

| Probe | Direct literal-vector result | Actual unchanged AMG result |
|---|---|---|
| Remove only the eleven pain questions; 134 tokens | Anemia improves **21 → 1**, distance **1.052496 → 0.656188**. | Anemia rank 3, distance 0.706646; leukemia expansion changes the top five. |
| Remove pain questions and restore all five omitted history facts; 219 tokens | Anemia rank 1 / 0.571132; Blood Thinners bleeding chunk rank 49 / 1.161482; Kidney Failure rank 275 / 1.397910. | Anemia rank 1 / 0.613079; Blood Thinners bleeding chunk rank 33 / 1.070510, still not returned. Other returned hits are leukemia passages. |
| Original black-stool fact alone; 18 tokens | GI Bleeding improves **290 → 3**, but distance is **1.366375**. | **Abstains:** all five candidates exceed 1.10. |
| Original anticoagulant fact alone; 22 tokens | Blood Thinners chunks rank 4 and 8, both outside cutoff. | Recognized alias adds “Blood Thinners”; both chunks are returned at **0.904281 / 0.917956**. |
| Original kidney-failure fact alone; 11 tokens | Kidney Failure improves **1,780 → 1**, distance **0.671928**. | Returned first, same distance. |
| Original prior-anemia fact alone; 14 tokens | Anemia rank 1, distance **0.764540**. | Returned first, same distance. |

These checks demonstrate **composite-query dilution**, not an inability to represent anemia or kidney failure. They also expose a separate **absolute-distance limitation for the black-stool wording**: rank can recover without satisfying the gate. Removing pain/restoring history is **not a complete fix** for all requested topics, and dropping headache is not a safe production recommendation. No modified-query end-to-end diagnosis was run.

### 3. Alias processing: recognition is not a topic-retrieval guarantee

There is **no LLM entity-extraction stage** in the active evidence path. It uses `AliasIndex.resolve` over source titles, aliases and see-references:

- Actual baseline matches: `anemia` once, `pain` twelve times, and ordinary `all` once. Repeated matches do not independently add ranking bonuses.
- `anemia` is recognized but already equals its canonical title, so no text is appended. A long case is **not** an exact-topic lookup; no topic filter, opening-chunk insertion or lexical bypass occurs.
- The actual black-stool, pallor, dizziness, fatigue and near-fainting **question strings have no alias matches**. The isolated word `fatigue` is recognized; `fatigued` inside the supplied question is not. The isolated phrase `black stools` is also absent from the alias index. These facts still influence dense embeddings, but receive no structured concept route.
- `anticoagulants` maps to Blood Thinners and `kidney failure` is recognized **when their omitted facts are restored**. Recognition of a canonical title within a longer query still does not guarantee selecting that topic.
- Noncanonical, unambiguous aliases can append canonical titles to the query, and thus change the embedding. Recognition is therefore **not always mere metadata**, but it is neither per-topic candidate insertion nor a relevance guarantee.

### 4. The `all → Acute Lymphocytic Leukemia` ambiguity

In the saved query, “all” from **“bed all day long”** proposes “Source topic: Acute Lymphocytic Leukemia”. The resulting **264-token** string exceeds 256, so AMG skips the expansion and embeds the original 254-token query. It **did not cause the original CML result**.

With the pain-only deletion probe, that false expansion fits (**144 tokens**) and becomes active: the returned set changes to **Chronic Lymphocytic Leukemia, Acute Lymphocytic Leukemia, Anemia, Acute Lymphocytic Leukemia, Chronic Myeloid Leukemia**. Restoring history also permits Blood Thinners/Body Weight expansions, but the false leukemia addition persists. Thus a shorter query can improve anemia retrieval **and introduce alias contamination**. The alias rules were not modified.

## C. What is not the cause

- **HNSW ordering error:** the saved top five equal exhaustive exact-distance top five.
- **Model/index or normalization mismatch:** model/revision/dimension match the published manifest. Eleven checked stored vectors—the original five plus all six requested-topic chunks—re-encode with maximum component difference **6.71e-8**. All 2,289 stored vectors are effectively unit length (`0.999999874–1.000000139`); query norm is `1.000000057`. Cosine and squared-L2 rankings agree for **all 2,289 chunks**. Switching these metrics would not rescue this case.
- **Later reranking:** there is no cross-encoder or clinical final reranker here; candidates are sorted by distance/ID. The adapter does not discard a hidden relevant result.
- **Missing corpus/provenance:** every copied stored passage/metadata matched the published artifact. Relevant chunks exist. The saved-run verifier again confirmed five source passages and the diagnostic/critic/safety identity chain.
- **Safety, Grounding or diagnostic citations:** these run downstream and cannot change the earlier retrieval ranking. The previous audit's distinction remains: missing medical references require human review; critic safety flags caused BLOCK.

## D. Smallest sensible next direction — not implemented

Start at the **query adapter**, not the agent architecture: a deterministic, compact, role-aware representation that reduces repeated question boilerplate while reserving space for presenting findings, other symptoms, history and medications. Preserve headache, negation, unknowns and uninterpreted numeric values; do not infer a diagnosis or hardcode these four topics.

Before promoting shorter queries, address the narrow **ordinary-word/acronym ambiguity** and regression-test the actual expanded query. The measured shorter-query results show compaction alone is unsafe to assume sufficient. Keep the model, corpus, candidate budget and gate unchanged for the first controlled comparison. Check coverage of all requested mechanisms—not only Anemia. The black-stool probe means acceptance behavior will also need separate evaluation; this investigation does **not** justify relaxing 1.10 or merely increasing top-k. No replacement query or clinical improvement is claimed.

## Verification and artifacts

- **506 project tests:** OK, one opt-in cloud test skipped. **56 standalone AMG tests:** OK. No cloud test was enabled.
- **256 protected files unchanged**, both after analysis and after tests: application/retrieval/agent/safety source, source/index data, evaluation files and original saved run. Comparison is against the investigation-start SHA-256 inventory, preserving pre-existing uncommitted work. Credential contents also unchanged; credential hashes were not written to reports.
- Chroma opened **only a disposable 44 MB byte-copy**, removed after its child exited. Original databases were not opened by Chroma, so even their bookkeeping bytes remained unchanged. One cached CPU MiniLM; no downloads or index rebuild. Final sampled child RSS was about **0.707 GB** (not a peak-memory measurement).
- `outputs/new2/results.json`: all 2,289 exact ranks, per-fact tokens, alias resolutions and actual probe outputs.
- `outputs/new2/query-as-embedded.txt`: exact baseline query.
- `outputs/new2/investigate.py`: local analysis code only; not imported by production.
- `outputs/new2/integrity.json`, `post-test-integrity.json`, `saved-run-verification.json`, `project-tests.txt`, `upstream-tests.txt`: verification records. Initial two-probe results are retained as `initial-results.json`.

**Conclusion:** the dominant baseline failure is the geometry of an imbalanced composite query, amplified by omitted history and a five-candidate ceiling. Alias recognition does not force topic coverage; its ordinary-word ambiguity becomes harmful when expansion fits. The fixed gate is an additional limitation for some clinically relevant wording. None of this requires changing the agent architecture or weakening evidence/safety checks.

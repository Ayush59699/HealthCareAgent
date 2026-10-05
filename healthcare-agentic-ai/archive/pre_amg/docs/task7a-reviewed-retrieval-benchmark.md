# Task 7A — reviewed medical-retrieval benchmark

## Status and boundaries

**Review-ready, not yet independently reviewed.** The supplied
`evaluation/medical_relevance.json` contains 92 unique, actual medical chunks for
**DDXPlus validation case 2 only**. Every candidate starts with
`relevance: "uncertain"`, `reviewed: false`, and no reviewer identifier. No human
judgments have been invented, and no relevance metrics are reported yet.

This task changes only offline evaluation tooling/assets, tests and documentation.
It does not change Qdrant contents, BGE, the medical corpus, live candidate top-k 5,
focused retrieval, Diagnostic Agent, Clinical Critic, or Safety Validator. No
cloud/API call, model download, new embedding, or new retrieval run was needed.

Four distinct layers must stay separate:

1. **Raw retrieval ranking:** saved BGE positions/cosines and lexical positions.
2. **Manual relevance judgment:** a reviewer's assessment of the full passage
   against the label-free patient findings, using the rubric below.
3. **Derived retrieval metrics:** arithmetic over those explicit judgments and
   the frozen rankings, limited to this candidate pool and this single case.
4. **Clinical/diagnostic conclusions:** not measured. The output explicitly sets
   `clinical_diagnostic_conclusions` to null. No diagnostic accuracy, clinical
   validity, treatment recommendation or safety claim follows from this benchmark.

## Dataset construction and provenance

The saved source is
`outputs/task6-continued/20260923T202535249425Z-retrieval.json`, produced by the
existing local retrieval experiment. Its filename, timestamp and SHA-256 are
recorded in the dataset, together with the index count, embedding signature,
manifest checksum and a checksum of the indexed payload snapshot.

The pool is the union of:

| Pool origin | Chunks before deduplication |
|---|---:|
| Current focused BGE query, top 50 | 50 |
| Symptom-only BGE query, top 50 | 50 |
| Current-query experimental lexical top 5 | 5 (already in current top 50) |
| Exact indexed topic title: Panic Disorder | 7 |
| Exact indexed topic title: Asthma | 6 |
| Exact indexed topic title: Asthma in Children | 8 |

The BGE-list union contains 72 unique chunks. The requested topics add 20 more
unique chunks, giving **92** total. One Asthma in Children chunk was already
retrieved. Topic inclusion is a pooling instruction, **not** a relevance judgment
or a diagnosis. Every existing chunk for the three exact titles is included;
there is no cherry-picking by passage content. An absent requested title would
be recorded with an empty chunk-ID list, not fabricated text.

The lexical system in this benchmark is explicitly the existing **current-query
BGE top-50 → concept filter → lexical top-5** experiment. It is not a mixture of
current-query and symptom-only reranking. All symptom-only top-50 candidates are
included, but the symptom-only lexical variant is not an additional evaluated
system in this three-system benchmark.

Each candidate contains:

- `chunk_id`, `title`, `source`, and **full, untruncated** `text`;
- `source_metadata`, preserving the remaining validated indexed payload fields;
- `origins`, showing the pooling sources;
- `rankings.current_query` and `rankings.symptom_only_query`, each containing the
  original one-based BGE rank and cosine, or null if not retrieved;
- `rankings.lexical_reranker`, containing its one-based lexical rank, original
  current-query BGE rank/cosine and raw lexical score, or null;
- the four review fields described below.

A chunk retrieved by both queries keeps both different positions/scores. Topic-only
chunks have null ranking slots: no rank or similarity is invented. Cosines and
lexical scores are raw retrieval metadata, not probabilities or clinical confidence.
Candidates are serialized in UUID order, not ranked or ordered by predicted relevance.

### Rebuilding a separate pool

From the parent workspace:

```text
uv run python healthcare-agentic-ai/scripts/build_medical_relevance_benchmark.py --retrieval-report healthcare-agentic-ai/outputs/task6-continued/20260923T202535249425Z-retrieval.json --output healthcare-agentic-ai/evaluation/medical_relevance-new.json
```

The default output, if `--output` is omitted, is
`healthcare-agentic-ai/evaluation/medical_relevance.json`. **Existing output files
are never overwritten**, including files containing manual reviews. Optional
`--data-dir` and `--storage-path` select existing local inputs.

Close other Qdrant users while copying the snapshot. The builder reads a temporary
copy of the existing Qdrant directory; it never even opens the original store with
Qdrant, avoiding incidental lock/metadata writes. The copy is removed afterward.
No BGE or cloud provider is instantiated. The builder checks the saved manifest,
index count, query formulation/version and every pooled candidate's exact indexed
payload. It rejects missing/stale/altered inputs, duplicates and invalid ranks.
The original patient features are loaded with `include_labels=False`, with no
DDXPlus pathology or differential label used or serialized.

The benchmark JSON itself is self-contained for review and summarization. Those
operations do not need the old report, patient archive, Qdrant, or embedding cache.

## Manual review protocol

Use exactly this definition:

> Does this medical passage provide useful medical information for interpreting
> one or more of the patient's observed findings or differential categories?

Read the dataset's full `patient_state` first. Preserve unlisted findings as
unknown and numeric scale responses as uninterpreted. No diagnostic output or
evaluation diagnosis label is supplied as ground truth. The requested topic names
are not diagnoses to confirm.

Suggested rubric, applied by the reviewer rather than software:

- **relevant:** the passage provides useful medical information for interpreting
  one or more observed findings or considering a differential category.
- **partially_relevant:** it provides limited useful information, with important
  applicability, specificity, population or contextual limitations.
- **not_relevant:** it does not provide useful information for this purpose.
- **uncertain:** the reviewer cannot confidently decide from the available context.

Do not assign any of these categories solely from title overlap, cosine, lexical
score, ranking position or the DDXPlus diagnosis label. Assess the **full passage**,
not just its topic. Rate the candidate text itself, not additional section bodies
retained in source_metadata for provenance. Relevance is not diagnostic entailment
or confirmation.

Prefer an independent medically knowledgeable reviewer who has not seen the
retrieval scores or dataset outcome. Present passages in a separate rank-blinded
view/worksheet if possible; do not delete ranking fields from the source JSON.
The builder does not claim that such independent review has already happened.
For a second reviewer, use separate copies, compare disagreements and adjudicate
manually; the current format stores one final judgment per chunk, not a validated
multi-rater agreement protocol. Record provenance/disagreements in reviewer notes.

For each reviewed candidate, edit **only**:

```json
{
  "relevance": "partially_relevant",
  "reviewed": true,
  "reviewer_id": "your-reviewer-identifier",
  "reviewer_notes": "Explain the useful information and its limitations."
}
```

This is a format example, **not a judgment for any actual candidate**.
`reviewer_notes` is optional in use (null is accepted). A reviewed `uncertain`
judgment is different from the initial unreviewed placeholder. `reviewed=true`
requires a nonblank `reviewer_id`; changing relevance without explicitly recording
manual review is rejected. Software records the assertion of review, but does not
authenticate the reviewer or independently verify their qualifications/independence.

A `pool_sha256` checksum covers everything except those four review fields. It
catches accidental edits to patient facts, passages, ranks, origin lists or pool
membership. **Do not recompute it to hide edits**; keep review copies of the frozen
pool. It is an integrity check, not a cryptographic signature authenticating review.

## Summarizing manual judgments

```text
uv run python healthcare-agentic-ai/scripts/summarize_medical_relevance.py
uv run python healthcare-agentic-ai/scripts/summarize_medical_relevance.py --input healthcare-agentic-ai/evaluation/medical_relevance.json --output healthcare-agentic-ai/outputs/task7a/reviewed-summary.json
```

The first command prints JSON to stdout. The second also writes a **new** output
file and refuses to overwrite existing files. It only reads local benchmark JSON;
there is no retrieval, index access, label loading, cloud client or API call.

Summary sections are `raw_retrieval`, `manual_review`,
`derived_retrieval_metrics`, and `clinical_diagnostic_conclusions`. Relevant and
partially relevant counts count **manually reviewed** chunks only. Unreviewed
placeholders are reported separately, not counted as reviewed uncertainty.

### Metric definitions

Let `R` be the set of manually reviewed **relevant** chunk IDs in the entire
92-chunk candidate pool, including relevant requested-topic chunks absent from
retrieval. This is the primary `strict_relevant` analysis. The separately named
`inclusive_relevant_or_partial` sensitivity analysis uses relevant **or** partially
relevant judgments as positives; it does not silently mix the two definitions or
assign fractional weights.

For each of the three systems:

```text
Recall@k = number of IDs in R returned within cutoff k / number of IDs in R
RR       = 1 / original rank of the first ID in R returned by the system
MRR      = mean RR across evaluated queries (one case here, so MRR = RR)
```

Report Recall@5, @10, @20 and @50. Original positions are never compressed by
removing unreviewed or irrelevant candidates. Repeated chunks across systems count
once in the shared pool denominator. If known positives exist but none are
returned by a system, that system's observed recall and truncated MRR are zero.

**Lexical output depth:** the lexical system is a fixed, at-most-five-result
shortlist. Its Recall@10/@20/@50 therefore uses the same returned set as Recall@5;
those values plateau, rather than assuming additional ranked results exist.
`captured_depth: 5` and `effective_cutoffs` explicitly disclose this. BGE systems
are evaluated only through their captured top 50. MRR is likewise limited to the
captured output: zero is not a claim that no useful evidence exists deeper in BGE.

### Missing and incomplete review

- **No manual review:** `derived_retrieval_metrics` is null and status is
  `withheld_no_manual_review`. No relevance/clinical metrics are calculated.
- **No known positive judgments for an analysis:** its recall and MRR values are
  null (`withheld_no_positive_judgments`), avoiding division by zero or an
  unsupported performance conclusion. Counts/coverage can still be reported.
- **Partial review or reviewed uncertainty:** any computable metrics are explicitly
  `provisional_incomplete_pool_review`. Unreviewed/uncertain chunks are not negative
  labels. The denominator is only currently known positives. These metrics can be
  biased in either direction and must not be presented as completed evaluation.
- **All candidates reviewed with decisive judgments:** status is
  `complete_pool_review`. This means complete review of this pool only.

These are **pooled judged recall** metrics, not whole-corpus recall. Unretrieved,
non-requested chunks were not reviewed; relevant passages may exist outside the
pool. Enrichment with the three requested topics affects the denominator. One
synthetic case cannot establish general retrieval quality, clinical utility,
diagnostic accuracy, calibration, or clinical validity.

Only after independent review should the results inform the next experiment:
useful requested passages missing from BGE may motivate ranking/query studies;
a lack of useful passages in this small pool does **not** prove a corpus-wide gap.
Do not use these results to relax critic/safety gates or tune toward the held-out
DDXPlus diagnosis label.

## Actual validation

Generated dataset: **92 candidates, zero manually reviewed**. The summary correctly
withholds metrics. Initial summaries are stored under `outputs/task7a/` (the final
code's summary is `unreviewed-summary-final.json`).

```text
uv run python -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -p "test_medical_relevance*.py" -v
uv run python -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v
```

The 19 new tests cover pooling/deduplication, full source preservation, topic-only
null ranks, original per-query ranks, no automatic relevance, missing topics,
label/query safeguards, review identity, immutable pool fields, invalid ranks,
manual-uncertain versus unreviewed state, zero denominators, requested-topic misses,
hand-calculated recall/MRR, strict versus inclusive analysis, partial-review bias,
CLI errors/no-overwrite behavior and the checked-in dataset. A real temporary
Qdrant test disables network sockets and compares source-directory file hashes
before/after the builder, confirming it opens only the disposable copy. Its label
sentinel never appears in the output. These tests simulate reviewer judgments for
arithmetic only; they are not human clinical review of the actual candidates.

Full regression result: **372 tests run, 371 passed, one opt-in cloud integration
test skipped**. Log: `outputs/task7a/tests.log`. No live cloud validation was run.

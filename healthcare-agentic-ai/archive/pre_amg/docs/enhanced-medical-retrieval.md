# Focused medical retrieval and cautious inference (Phase 6)

**This improves retrieval and reasoning behavior in the research prototype; it does not establish clinical validity.**

## Analysis and scope

The original `EvidenceService` sent `PatientRepresentation.to_text()` to both
indexes. This is appropriate for case similarity, but long question/answer
narratives dilute medical-topic queries and can exceed the BGE embedding context.
The Phase 6/demo caller also overrode the medical retriever's default five results
with top-k one. Neither path assessed topic relevance. With no medical hits, the
orchestrator stopped before diagnosis, critic or safety; the diagnostic prompt and
shared grounding validator independently required abstention.

No new clinical role, model, service, corpus, index, prescribing capability, HITL,
or evaluation-label access was added. The enhanced path is enabled by default in
`scripts/demo.py` and `scripts/run_phase6.py`. Direct library callers opt in with
`Phase6Orchestrator(..., enhanced_medical=True, medical_top_k=5)`. The default-off
library option preserves the historical Phase 6 compatibility baseline and allows
controlled before/after comparisons. It is not an alternate safety policy.

## Query and filtering policy: `focused-medical-v1`

1. Construct the medical query **after** exact-copy validation of PatientState.
   The service rechecks that state against the label-free original representation.
2. Prioritize presenting evidence, then current symptoms; deduplicate and retain
   up to 12 concepts. Add up to four antecedent concepts, an age group and sex.
3. Recognize a small explicit retrieval-synonym vocabulary (for example, difficulty
   breathing / dyspnea). Other findings use at most six literal content words.
   This is not a diagnostic ontology or generated clinical summary. No case ID,
   metadata, evaluation label, model-inferred diagnosis, or missingness narrative
   enters the query. Explicit No/unknown/none/nowhere answers and ordinal-only
   answers are omitted, not converted into positive symptoms or severity labels.
4. Make one BGE/Qdrant medical query with **candidate top-k 5**. Patient-case search
   stays unchanged, top-k 1 by default. Both values are independently configurable.
5. Validate every candidate's existing provenance/content contract before filtering.
   Retain a chunk if its title matches at least one symptom/history concept, **or**
   its body matches at least two distinct symptom concepts. Matching uses token
   coverage of a concept or one of its declared aliases. Repeated occurrences of
   a single symptom do not satisfy the two-concept rule.
6. Preserve the complete retained hit without rewriting text, title, IDs, metadata
   or score. Sort by original descending BGE cosine score, breaking ties by chunk
   ID. There is no new numerical relevance score, calibrated threshold, or model
   reranker. Rejected candidates never enter the immutable agent evidence snapshot.
7. Persist the query, policy, candidate top-k, IDs/titles/sources/scores, match
   reasons and retain/discard decisions in `state.medical_retrieval`. This audit
   metadata is not agent evidence. The demo displays it separately. Empty retained
   results explicitly report `insufficient_or_irrelevant`. No symptom concepts
   means no medical query, rather than a demographic-only evidence search.

### Why no cosine cutoff or learned reranker?

A pre-change top-10 probe across the first three **label-free** validation cases
found unrelated transplant papers scoring 0.6704–0.6899 at rank one. In validate:2,
the asthma chunk ranked fourth at 0.6466, below three irrelevant transplant chunks.
Thus a score cutoff high enough to remove the irrelevant paper would also remove
the useful topic. A relative top-score cutoff would preserve the wrong result.
The lexical gate adds independent, inspectable topic information without a new
model/service/dependency. It does **not** claim clinical relevance classification:
false positives and false negatives remain possible. A title match is candidate
context, not proof that each paragraph supports a hypothesis. The diagnostic agent,
critic and safety validator must still assess its actual meaning.

The current corpus has only **5 documents / 87 chunks**, including three broad
MedlinePlus topics and two specialist papers. Filtering cannot supply missing
pediatric or differential-diagnosis guidelines. Wider corpus coverage and a
separate adjudicated relevance dataset are still needed before calibration or
retrieval-quality claims.

## Diagnostic contract: `phase6-diagnostic-inference-v1`

The JSON schema remains unchanged; the enhanced Phase 6 adapter selects a distinct
prompt and explicitly enables patient-only inference validation.

- **Relevant medical evidence:** cautious hypotheses with actual supporting
  references, clear uncertainty, and observations distinguished from inference.
- **Weak/absent evidence with meaningful patient facts:** cautious hypotheses are
  permitted, not required. Every hypothesis without a structured medical reference
  must start its rationale with `Uncertain model inference:`, link its rationale
  to an existing patient symptom/presenting fact, and provide nonempty uncertainty
  and missing-information lists. Demographics/case analogies alone are insufficient.
- **Insufficient patient information:** abstain with an explanation and critical
  missing information. No symptom/presenting observations fails closed for a
  non-abstaining enhanced diagnosis, even if medical material exists. Deciding
  whether nonempty observations are clinically sufficient still requires semantic
  assessment; a list length is not a clinical sufficiency test.

All original reference-ID, source-category, exact inventory, provenance and literal
URL checks remain. Patient facts never populate medical evidence inventories.
Rejected medical IDs cannot be cited. The only baseline grounding exception is an
explicit **default-off** `allow_patient_inference` option replacing the mechanical
"no medical hits means abstain" rule with the additional disclosure/fact-link
requirements. Phase 4/5 call sites retain the old behavior.

## Safety and version coverage

The flow remains Patient Agent → case RAG + medical RAG → Diagnostic Agent →
grounding → Clinical Critic → Safety Validator → application routing.

The safety input validator accepts the explicit patient-only contract for review.
**Safety policy, prompts, categories, action restrictions and routing are unchanged.**
Every hypothesis lacking a medical reference still creates the deterministic
`missing_medical_reference` **HUMAN_REVIEW** finding. Critic safety flags still
**BLOCK** and intentionally skip the semantic call. Without a critic block, all
existing semantic categories are assessed with the original prompt. No finding
is cleared by the new diagnostic permission.

Enhanced repeated diagnostic versions now receive a fresh critic and safety
assessment before repeated-version termination; they cannot inherit prior safety
clearance. Successfully completed diagnostic versions are covered in tests,
including repeated versions. A technical failure, budget exhaustion or invalid
critic may prevent an assessment; these remain failed/withheld with explicit
unassessed coverage, **not** a fabricated safety pass. The historical direct-library
baseline retains its original early-exit behavior.

## Actual validate:2 comparison

No pathology, differential label, answer key or label metric was loaded. Retrieval
used existing local indexes (1,000 training cases, 87 medical chunks), cached BGE,
and the unchanged GPT-5.6-Sol Responses provider. No index rebuild was performed.

Focused query produced from the actual accepted PatientState:

```text
pediatric female detachment; sweating; cramp; chest pain; flank; hypochondrium;
breast; shortness of breath; choking; fear of dying; palpitations; numbness tingling
history: anxiety; family psychiatric illness; depression; asthma
clinical information red flags
```

| Enhanced candidate | Cosine score | Filter |
| --- | ---: | --- |
| Asthma (MedlinePlus) | 0.6534 | Retain: history/title concept match |
| Diabetes | 0.6426 | Discard: insufficient concept overlap |
| Galectin-4 intestinal inflammation paper | 0.6399 | Discard: insufficient concept overlap |
| BK polyomavirus transplant paper, chunk 1 | 0.6385 | Discard: insufficient concept overlap |
| BK polyomavirus transplant paper, chunk 2 | 0.6301 | Discard: insufficient concept overlap |

The retained ID was `medical:32f13991-1ad5-5b06-ab76-38b7184c074a`.
The case analogue remained `case:ddxplus:train:514`, similarity approximately 0.9787.

| Measurement | Real baseline workflow | Real enhanced workflow | Separate enhanced terminal demo |
| --- | --- | --- | --- |
| Medical query | Full patient narrative | Focused PatientState concepts | Same focused query |
| Medical candidates / retained | 1 / 1 unfiltered | 5 / 1 | 5 / 1 |
| Medical evidence | BK transplant paper, 0.6704 | Asthma, 0.6534 | Asthma, 0.6534 |
| Diagnostic result | Abstained | Tentative hypotheses | Tentative hypotheses |
| Structured medical references | None | Retained asthma ID only | Retained asthma ID only |
| Structured patient-case references | None | None | None |
| Critic | revision_required | revision_required | revision_required |
| Safety / final outcome | BLOCK / safety_blocked | BLOCK / safety_blocked | BLOCK / safety_blocked |
| Semantic safety generation | Skipped: critic block | Skipped: critic block | Skipped: critic block |
| Requests / provider repairs | 4 / 1 | 3 / 0 | 3 / 0 |
| Workflow latency | 58.55 s | 76.97 s | 62.82 s |
| Revisions | 0 | 0 | 0 |

In the saved enhanced run, the tentative list included a panic episode, asthma
exacerbation, hyperventilation and a broad pleuropulmonary possibility. These are
**withheld research outputs, not endorsed diagnoses**. The critic identified
incorrect symptom-index mappings, unsupported temporal/ordinal interpretations,
unsupported comparative ranking and premature psychiatric attribution. Existing
reference checks establish ID existence, not semantic entailment: those errors
were caught by the critic, not falsely claimed to be prevented by grounding.
Safety recorded three critic safety flags and three missing-medical-reference
findings, deriving BLOCK. The system was not tuned to force a favorable outcome.

**Irrelevant retrieval still occurs among candidates**, but both BK chunks were
excluded from agent evidence and were not cited. **Abstention remains available**
and is tested for genuinely empty patient observations; it did not occur in these
two enhanced live runs. **Unsafe output still blocks.** Live semantic safety
execution was not demonstrated here because critic blocking correctly short-circuited
it; deterministic fixtures exercise semantic HUMAN_REVIEW and BLOCK paths.

Cloud sampling is the deployment default, not deterministic temperature zero.
These single-case calls are not a controlled clinical comparison, diagnostic
accuracy measurement, or latency benchmark.

## Validation and artifacts

Final checks:

- New focused-retrieval/inference regressions: **24 passed**.
- Phase 4 regression files: **56 passed** (28 Phase 4, 12 provider, 16 abstention).
- Phase 5 orchestration regressions: **50 passed**.
- Existing Phase 6 and safety regressions: **78 passed** (49 + 29).
- Existing demo tests: **20 passed**.
- Full suite: **307 tests, 306 passed, 1 skipped**, zero failures. The skipped test
  is the existing opt-in cloud test; the separate live runs above were performed.
- Syntax compilation, demo/runner help, sample discovery and real local-only
  validate:2 retrieval checks passed. Local validation made zero GPT calls.
- Content comparison against Git confirmed all 19 protected baseline files in
  `checks-summary.json` unchanged (line endings normalized), including Phase 5,
  shared baseline prompts/agents, parser, provider, and safety policy/routing.

Initial regression checks exposed two reporting-fixture failures: a fever-only
synthetic case cited an asthma fixture which the new filter correctly removed.
Those reporting-only fixtures now use a matching asthma/wheezing observation;
they do not bypass the filter. Dedicated tests cover rejection of mismatched sources.

Artifacts are local, under the existing ignored `outputs/enhance/` directory:

- `before-retrieval.json`: label-free top-10 probe on three cases.
- `before-state.json`, `before-summary.json`, `after-state.json`, `after-summary.json`:
  real workflow evidence, structured versions, critiques, validation and safety.
- `enhance-demo-live.log`: separate real terminal demonstration.
- `checks-summary.json`, `checks-*.log`: final regression/CLI/compatibility results.

Run from the project directory (substitute the existing environment's Python):

```text
python -m unittest discover -s tests -t . -p test_focused_medical.py -v
python -m unittest discover -s tests -t . -v
python -m compileall -q rag orchestration safety demo scripts tests
python scripts/demo.py --help
python scripts/demo.py --sample 2 --validate-only --detailed
python scripts/demo.py --sample 2 --medical-top-k 5 --detailed --architecture
```

Optional reproduction scripts `scripts/inspect_enhance_retrieval.py` (local only)
and `scripts/validate_enhance.py [--enhanced]` (real GPT requests) write the fixed
artifact names above and **overwrite those comparison files**. Preserve prior
artifacts before rerunning. Use only existing local indexes and cached embeddings;
close other Qdrant processes first.

## Files changed / intentionally unchanged

Added:
- `rag/focused_medical.py`
- `orchestration/phase6/diagnostic.py`
- `tests/test_focused_medical.py`
- `scripts/inspect_enhance_retrieval.py`, `scripts/validate_enhance.py`
- this document

Modified:
- `rag/agents/grounding.py`: explicit default-off patient-inference option.
- `orchestration/phase6/orchestrator.py`, `orchestration/phase6/state.py`: opt-in
  enhanced evidence/diagnostic path, audit metadata, repeated-version safety.
- `safety/validation.py`: validate patient-only hypotheses for safety review.
- `demo/terminal.py`, `scripts/run_phase6.py`: enable enhancement, independent
  medical candidate count, filtering audit. Also corrected an existing tuple-valued
  demo disclaimer that caused an exception in `finally`.
- `tests/test_phase6_reporting.py`: matching synthetic reporting fixture.
- `README.md` and parent `demo.md`: current commands, behavior and limitations.

Intentionally unchanged: Phase 4/5 workflows, agents/prompts/schemas, Phase 5 evidence
service and routing, safety policy/prompts/models/validator/routing, GPT provider,
DDXPlus parser/evaluation, raw medical retriever, BGE/Qdrant, corpus/index contents,
and dependency manifests. `prompts/enhance.txt` was read and left untouched.

## Remaining limitations

The lexical gate lacks comprehensive synonyms, negation/scope understanding,
temporal interpretation and clinical red-flag prioritization. It omits explicit
negative answers rather than representing their contextual relevance. Bounded
concept selection can omit important late-listed history; title-level matches can
retain irrelevant paragraphs; multi-topic articles can match by coincidence.
No learned reranker or clinical entailment checker was added. A small corpus
cannot support all hypotheses, and no source coverage or clinical correctness is
claimed. Semantic errors can survive reference-existence checks. All output remains
synthetic-data research material; HUMAN_REVIEW does not schedule a reviewer.

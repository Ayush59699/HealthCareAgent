# Full MedlinePlus XML corpus expansion

This is **more medical knowledge**, not **clinical validation**. This research
prototype uses synthetic DDXPlus inputs. Cosine retrieval similarity is not disease
probability, and a retrieved passage does not establish a diagnosis or clinical
entailment. BLOCK and HUMAN_REVIEW remain valid, withheld-output outcomes.

## Scope and initial inspection

The existing medical pipeline was already separate from Patient Case RAG:
`medical_ingestion/{common,models,medlineplus,pmc,corpus,chunker}.py`, the fixed
`MedicalEmbeddingModel`, `MedicalVectorStore`, and `MedicalKnowledgeRetriever`.
The tiny corpus was a title-selection restriction, not an XML limitation. It
contained 3 MedlinePlus summaries / 17 chunks and 2 licensed PMC articles / 70
chunks (5 documents / 87 chunks total). The WHO adapter remains review-gated.

The existing `medical-embedding.json` checks the embedding configuration; the
old build report's `processed_sha256` hashes serialized chunks, including local
paths and retrieval timestamps. The expansion keeps both and adds an
`index_fingerprint` over sorted normalized chunk payloads and embedding settings,
excluding only volatile raw paths/retrieval times. IDs still use the existing UUID5
scheme based on document ID, chunking version, ordinal and text SHA-256.

**Untouched:** Phase 4/5/6 architecture, focused-medical concept-overlap filtering,
Diagnostic Agent, Clinical Critic, safety rules, provider and GPT-5.6-Sol settings,
DDXPlus parser, Patient Case RAG, BGE implementation and patient collection.
Pre-existing uncommitted edits in those areas were not changed by this task.

## Exact source and parsing

- URL: <https://medlineplus.gov/xml/mplus_topics_2026-09-22.xml>
- Version/filename: `2026-09-22` / `mplus_topics_2026-09-22.xml`
- Retrieved: **2026-09-22T20:09:28.212651+00:00**
- Size: **30,098,360 bytes**
- SHA-256: `b855b98aa3a6773c31188d91d948a91303a6e98a091f09163052a01db704460c`
- XML root: `health-topics`, `total="2033"`,
  `date-generated="09/22/2026 02:30:40"`.

The actual schema was inspected before parser implementation. It contains 1,017
English and 1,016 Spanish health-topic records. **1,015 English records have usable
NLM `full-summary` content**. Two English records have no summary:
`medlineplus:899` (*Electromagnetic Fields*) and `medlineplus:898` (*Lactose
Intolerance*). These exclusions are explicitly recorded in the manifest and report;
no replacement medical content is invented. Spanish topics are not indexed.

`defusedxml` rejects unsafe entity expansion. Escaped HTML summaries and real
nested elements are supported. An HTML parser preserves heading boundaries,
removes tags, scripts/styles/navigation, and normalizes Unicode/whitespace.
Only NLM topic summaries are embedded, not linked pages, site descriptions,
encyclopedia/drug content or images. Associated aliases, see-references, groups,
MeSH descriptors, related topics, primary institutes, language mappings, topic
attributes and dataset attributes are retained as **metadata**, not embedded
medical claims. Creation dates are not represented as review/update dates.

Normalized documents include `source`, deterministic `document_id`, `title`, `url`,
`language="en"`, `source_version`, `text`, `metadata`, and existing license/provenance
fields. Chunks additionally contain `chunk_id`, `chunk_index`, `section` and
`chunking_version`. Full section bodies are kept in normalized documents, not
repeated in every chunk's metadata. The source-only payload allowlist still rejects
patient/diagnosis-label fields. The exact legacy words-v1 payload schema remains readable; section-aware MedlinePlus
chunks cannot drop the new language/version/provenance fields.

License: existing `NLM-public-domain-summary` classification and copyright-policy
citation, restricted to NLM summaries. This does not claim that all XML-linked
material is public domain. Source attribution: **MedlinePlus, National Library of
Medicine**. PMC parsers and article-specific CC BY 4.0 license checks are unchanged.

## Chunking, embeddings and storage

- MedlinePlus: `sections-v2:180:30`, source-heading-aware passages, at most 180 words,
  30-word overlap within longer sections. Adjacent short sections (under 40 words)
  are merged when their combined size fits 180 words. Exact repeated passages
  within a document are suppressed; no artificial tiny-chunk inflation.
- Cross-document exact text deduplication suppresses **11 repeated introductory
  passages**. The first passage in stable document-ID order retains its original
  attribution. Every normalized document still contains its full original content,
  and every source document must retain at least one indexed passage. No fuzzy
  merging of potentially different medical claims is performed.
- PMC and explicit historical title selections: existing `words-v1:180:30` windows.
- Fixed **BAAI/bge-small-en-v1.5**, **384 dimensions**, normalized vectors,
  **Cosine** distance, CPU, batch size 16, 512-token context, no new model/prompt.
  The default is cached-only loading. Truncation counts are reported, not hidden.
- Collection remains **`medical_knowledge`**. Actual existing `DATA_ROOT` is anchored
  to the **parent workspace**: `data/medical_knowledge/`, not the project subfolder.
  Default database: `data/medical_knowledge/qdrant/`.
- Patient collection remains `ddxplus_patient_cases` at the separate existing
  `healthcare-agentic-ai/data/qdrant/patient_cases/` default; no writes are made there.

The new ingestion command downloads/reuses the exact URL, verifies HTTP status,
90-second timeout, 80 MB bound, nonempty response and Content-Length when supplied.
Its sidecar records URL, UTC retrieval time, SHA-256, size and filename (size is a
string to preserve the existing string-valued provenance contract). Cached bytes
are reused only after URL/checksum/size verification. This is local checksum reuse,
not a claim of remote freshness checking. There is no alternate-source fallback.
Raw XML and new index/staging artifacts are ignored by Git. Previously tracked
historical raw files are not removed by this change.

The command preserves **all existing authorized PMC manifest entries**. With no
existing corpus it obtains the same two development PMC articles via the existing
licensed OAI adapter. An existing failed or non-PMC/non-MedlinePlus corpus requires
explicit review rather than silent removal. It builds a separate staging database,
checks that no documents/chunks were lost during embedding/upsert, and audits it
before publication. Only then are the medical database, corpus manifest and
normalized files replaced. Old artifacts are retained in a timestamped
`qdrant-backup-*` directory; failed publication rolls back. A directory containing
any non-medical collection is refused. The exclusive ingestion lock and Qdrant
locks are not a multi-process deployment protocol: stop readers before rebuilding.
An interrupted process may leave a lock/staging directory; inspect it and confirm
no ingestion process remains before removing the stale lock and retrying.

## Reproduction and validation

From the parent workspace (or use the corresponding environment Python directly):

```text
uv run python healthcare-agentic-ai/scripts/ingest_medlineplus.py
uv run python healthcare-agentic-ai/scripts/audit_medical_index.py
uv run python healthcare-agentic-ai/scripts/test_medical_rag.py
uv run python healthcare-agentic-ai/scripts/demo.py --sample 2 --medical-top-k 5 --detailed --architecture
uv run python -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v
```

Only if BGE is absent, explicitly opt in with `ingest_medlineplus.py --allow-download`.
`--data-dir` selects an independent medical data root. Do not point it at patient
storage. Keep the raw snapshot, provenance sidecar and corpus manifest together for
exact reproduction; a dated upstream endpoint may eventually disappear.
`build_medical_index.py` remains available for idempotent same-corpus builds and
refuses stale IDs; use `ingest_medlineplus.py` for staged corpus replacement.

Audit compares every stored payload and deterministic ID to reconstructed,
hash-pinned source documents; checks collection configuration, finite normalized
vectors, exact metadata schema, provenance, duplicate IDs/documents/chunks,
empty chunks, embedding failures/truncations, build metrics and fingerprint.
Integrity failure produces a report with `passed=false` and a nonzero exit status.
The ten real BGE topic smoke checks cover asthma, diabetes, hypertension, anxiety,
depression, chest pain, shortness of breath, palpitations, migraine and pneumonia.
Each requires an expected relevant topic in a document title or a source-provided
direct-definition section heading among the top five, not merely nonempty retrieval
or a high score. Parent-title-only coverage is reported separately. They are not a systematic relevance or clinical benchmark.

Local reports/logs are under `healthcare-agentic-ai/outputs/medlineplus-expansion/`:
`ingestion.json`, `audit.json`, `retrieval.json`, `regression.log`,
`before-demo.log`, `after-demo.log` and `comparison.json`. The initial
3,343-chunk trial and its audit/demo logs are preserved under
`before-global-deduplication/`, separate from the final deduplicated build. Build metrics also travel with the database
in `medical-index.json`; normalized documents/chunks are in `processed/*.jsonl`.

## Limitations

English NLM summary coverage is not comprehensive clinical guidelines coverage.
The two retained PMC articles were selected for explicit reuse permission, not
clinical relevance to this case. Word limits are not a tokenizer-budget guarantee;
reported truncations must be reviewed. There is no freshness/retraction monitoring,
clinical validation, general-purpose PHI classifier, or diagnostic probability
calibration. The existing model ID/configuration is retained, but the Hugging Face
artifact revision is not newly pinned. Deployment-default LLM sampling is unchanged; before/after generation
is not a controlled measure of diagnostic quality. Safety blocks must be preserved.


## Validation-oracle investigation

The initial parent-title-only smoke check passed **9/10** queries. For “What is
diabetes and high blood glucose?”, the five retrieved passages were directly about
diabetes, including source sections titled **What is diabetes?** on *Diabetic Eye
Problems*, *Diabetic Heart Disease*, *Blood Glucose*, *Diabetic Foot* and another
related topic. The first passage explicitly defines diabetes, glucose and insulin;
rejecting it solely because of its parent page title was a false negative for
passage-level topical relevance.

The check now accepts either an expected parent topic title or the original source's
exact “What is/are <expected topic>?” definition heading. It still requires actual
text, license and provenance, rejects unrelated headings/high scores, and reports
parent-title-only coverage separately. The original failed report is retained as
`retrieval-title-only.json`. This is a test-oracle correction, not a change to
embeddings, scoring, candidate filtering, production retrieval or clinical safety.
It does not assert that every top-five hit is relevant or that the first-ranked
parent title is always the best match. No protected regression assertion was relaxed.


## Actual validated results (2026-09-22)

| Source | Documents | Indexed chunks |
|---|---:|---:|
| MedlinePlus | 1,015 | 3,262 |
| PMC | 2 | 70 |
| **Total** | **1,017** | **3,332** |

The parser/chunker initially produced 3,273 MedlinePlus passages; **11 exact
cross-document repeats were suppressed**, leaving 3,262 unique passages. All
1,015 admitted MedlinePlus documents retain indexed content. The old 87-chunk
snapshot remains under `data/medical_knowledge/qdrant-backup-20260922T202103739495Z/`.
The intermediate 3,343-chunk snapshot has a separate later rollback backup.

- Staged and independent reopened-index audits: **PASS**.
- Duplicate IDs, duplicate full documents, duplicate indexed passages, empty chunks,
  malformed metadata, missing provenance, embedding failures and truncations: **0**.
- BGE topic/definition-section checks: **10/10 PASS**, candidate top-k 5.
- Parent-title-only topic coverage: **9/10** (reported independently).
- Medical ingestion/store unit tests: **52/52 PASS**.
- Complete regressions (including all existing Phase 4/5/6, safety and demo tests):
  **333 run, 332 passed, 1 opt-in cloud integration test skipped, 0 failures**.
  Separate live GPT demos were performed. Existing Qdrant/SQLite ResourceWarnings
  appeared without failed assertions.
- Index fingerprint: `72a0a462600a1a647c5413c5031008bdbcc31b8628df2763c74b7d89a1a858de`
- Exact processed JSONL SHA-256 (includes local paths/times): `cefe0efa74f9a70869132cf7bfdb195592934050fcf0f4df1479c1c04a0145ef`

### validate:2 live comparison

Both commands used `--sample 2 --medical-top-k 5 --detailed --architecture`.
Patient-case top-k remained 1 and the patient index remained at 1,000 training cases.
Times below are workflow-only, excluding initial model/index loading.

| Measure | Before | Final expanded corpus |
|---|---|---|
| Medical documents | 5 | 1017 |
| Medical chunks | 87 | 3332 |
| Top-k candidates | 5 | 5 |
| Retained | 1 | 1 |
| Discarded | 4 | 4 |
| Diagnostic Agent | Hypotheses recorded | Hypotheses recorded |
| Clinical Critic | revision_required | revision_required |
| Safety Validator | BLOCK | BLOCK |
| Workflow status | blocked | blocked |
| Workflow outcome | safety_blocked | safety_blocked |
| LLM requests | 3 | 3 |
| Execution seconds | 55.38 | 86.6 |

**Before candidates** (cosine retrieval similarity, rounded as printed by the demo):

| Topic title | Similarity | Disposition |
|---|---:|---|
| Asthma | 0.6534 | RETAIN |
| Diabetes | 0.6426 | DISCARD |
| Galectin-4 Controls Intestinal Inflammation by Selective Regulation of Peripheral and Mucosal T Cell Apoptosis and Cell Cycle | 0.6399 | DISCARD |
| Pre-transplant immune factors may be associated with BK polyomavirus reactivation in kidney transplant recipients | 0.6385 | DISCARD |
| Pre-transplant immune factors may be associated with BK polyomavirus reactivation in kidney transplant recipients | 0.6301 | DISCARD |

**Final expanded corpus candidates** (cosine retrieval similarity, rounded as printed by the demo):

| Topic title | Similarity | Disposition |
|---|---:|---|
| Gastroenteritis | 0.7171 | DISCARD |
| Heat Illness | 0.7166 | DISCARD |
| Heart Attack | 0.7158 | DISCARD |
| Depression | 0.7138 | RETAIN |
| Meningococcal Disease | 0.7122 | DISCARD |

The original focused-medical gate made every retention/discard decision; it was
not weakened. These results do **not** establish improved retrieval for this
complex case merely because corpus size and raw similarities increased. In the
initial expanded trial only *Depression* was retained, while potentially relevant
symptom topics did not reach the top five. The recorded final table above is the
actual final-index result, not a forced successful diagnosis.

The intermediate 3,343-chunk run also ended in BLOCK, with 3 LLM requests and
85.11 workflow seconds; its artifacts are archived separately. This additional
run is not hidden in the before/final comparison. Clinical/Safety decisions were
never overridden. More medical knowledge is not clinical validation.

### Changed files and preserved boundaries

Added source files:
- `healthcare-agentic-ai/rag/medical_ingestion/indexing.py`
- `healthcare-agentic-ai/scripts/ingest_medlineplus.py`
- `healthcare-agentic-ai/tests/test_medlineplus_expansion.py`
- `healthcare-agentic-ai/docs/medlineplus-expansion.md`

Modified source/documentation files:
- `.gitignore`
- `healthcare-agentic-ai/README.md`
- `healthcare-agentic-ai/docs/medical_knowledge.md`
- `healthcare-agentic-ai/rag/medical_ingestion/chunker.py`
- `healthcare-agentic-ai/rag/medical_ingestion/common.py`
- `healthcare-agentic-ai/rag/medical_ingestion/corpus.py`
- `healthcare-agentic-ai/rag/medical_ingestion/medlineplus.py`
- `healthcare-agentic-ai/rag/medical_ingestion/models.py`
- `healthcare-agentic-ai/scripts/build_medical_index.py`
- `healthcare-agentic-ai/scripts/audit_medical_index.py`
- `healthcare-agentic-ai/scripts/test_medical_rag.py`

Local corpus manifest, normalized documents/chunks and the active medical database
were rebuilt; raw downloads, metrics and rollback snapshots remain local. Existing
tracked runtime artifacts may still appear modified in Git; no raw XML was added
to source control. New raw XML is explicitly ignored.

Deliberately untouched: Phase 4/5/6 architecture and agents; Clinical Critic and safety rules; rag/focused_medical.py; rag/medical_retriever.py; rag/medical_vector_store.py; rag/embeddings.py and rag/medical_embeddings.py; rag/medical_ingestion/pmc.py; DDXPlus parser, Patient Case RAG, patient collection and label protections; provider, GPT-5.6-Sol configuration and credentials.

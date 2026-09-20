# Phase 3 — Medical Knowledge RAG

> Historical phase-specific validation record; subsequent Phase 4 implementation
> and migration results are documented in [phase4.md](phase4.md) and
> [phase4-validation.md](phase4-validation.md). Earlier stop/status statements below
> describe the repository at the time of that phase.

Retrieval only. No generative LLM, agents, orchestration or diagnostic decision
system. The Phase 2 patient modules, tests, collection and index are unchanged.
Only the existing embedding implementation is reused, with a separate medical
signature and fixed BGE model; patient environment collection/path overrides are
not used by the medical pipeline.

## Actual corpus and sources (2026-09-18)

| Source | Documents | Chunks | License/status |
|---|---:|---:|---|
| MedlinePlus | 3 | 17 | NLM public-domain health topic summaries |
| PMC | 2 | 70 | Both explicit CC BY 4.0 in retrieved JATS |
| WHO | 0 | 0 | Adapter implemented; no reviewed real document admitted |
| **Total** | **5** | **87** | |

Exact sources:

- Official downloadable XML snapshot:
  <https://medlineplus.gov/xml/mplus_topics_2026-09-18.xml>.
  Only English **Asthma** (topic 2), **Diabetes** (4), and **High Blood Pressure**
  (34) `full-summary` fields are indexed. The original complete XML is retained;
  the other topics, external linked pages, images, drug monographs and encyclopedia
  material are not indexed. Topic `date-created` is retained verbatim in the
  common date field; it is **not** a last-reviewed date.
- Official PMC OAI-PMH `GetRecord`, `metadataPrefix=pmc`:
  - <https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:2440804&metadataPrefix=pmc>
    — *Galectin-4 Controls Intestinal Inflammation by Selective Regulation of
    Peripheral and Mucosal T Cell Apoptosis and Cell Cycle*.
  - <https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:5451008&metadataPrefix=pmc>
    — *Pre-transplant immune factors may be associated with BK polyomavirus
    reactivation in kidney transplant recipients*.
  Both records have the `pmc-open` set and explicit article-level
  <https://creativecommons.org/licenses/by/4.0/> license links. We extract abstract
  and body paragraph/section text, excluding figure/table/reference containers.
- WHO: **no actual source used**. An explicit per-document reviewed selection is
  required. No WHO website bulk downloads, implicit license assumptions or claims
  about WHO retrieval results are made.

Medical text is obtained exclusively through XML downloads/APIs or explicitly
selected WHO PDF/text downloads, never by scraping medical webpages. MedlinePlus
usage terms were separately inspected at <https://medlineplus.gov/copyright.html>
(which redirects to <https://medlineplus.gov/about/using/usingcontent/>). They
explicitly list health-topic summaries as public-domain content. Attribution:
**Source: MedlinePlus, National Library of Medicine.** This does not extend to all
content linked from MedlinePlus.

Initial PMC candidates PMC7092803 and PMC7135076 had special reuse terms, not the
chosen CC allowlist; PMC10000000 used a Public Domain Mark rather than CC0;
PMC1567603 had no recognized explicit allowed license. None were indexed. Their
raw responses may remain locally as inspection evidence. The legacy OA utility
endpoint returned HTTP 404; the implementation instead uses the official OAI-PMH
open-set metadata plus article license. PMC12000000 returned HTTP 400.

## Licensing and provenance

Every admitted document/chunk includes source ID, title, text, source URL, license,
raw download URL, retrieval timestamp, raw filename, SHA256 and license evidence.
DOI, PMID, PMCID, publication date and document type are retained when available.
Downloads have `.provenance.json` sidecars; `corpus.json` pins admitted bytes.
Reconstruction verifies both manifest and sidecar hashes before embedding/auditing.
Unknown, missing, NC and ND licenses fail closed. Initial PMC/WHO policy accepts
only explicit CC0, CC BY and CC BY-SA URLs (including BY/BY-SA IGO variants).
This is intentionally narrower than all potentially legal research uses.

WHO selections require official HTTPS URLs, an independently reviewed raw SHA256,
license URL, a quote actually present in extracted text, terms URL and reviewer
identifier. Many WHO works use CC BY-NC-SA IGO: these are **excluded** by this
initial policy. Do not relabel them as CC BY-SA. A template schema is shown below;
placeholders are not real publications or accepted licenses:

```text
[
  {
    "id": "<reviewed-document-id>",
    "title": "<actual title>",
    "url": "<official WHO publication URL>",
    "download_url": "<official WHO PDF/text download URL>",
    "format": "pdf",
    "sha256": "<SHA256 of reviewed bytes>",
    "license_url": "<actual accepted CC URL>",
    "license_evidence": "<verbatim license quote present in the document>",
    "terms_url": "<official WHO usage-terms URL>",
    "reviewed_by": "<reviewer identifier>",
    "publication_date": "<source date>"
  }
]
```

License metadata is evidence, not legal advice. Attribution, third-party exceptions
and share-alike obligations still need review for any downstream redistribution.
Raw XML may contain excluded material and is not a license-cleared redistribution
bundle. PDF/text parser tests use explicitly synthetic fixtures, not invented WHO
publications presented as real data.

## Structure

```text
rag/medical_ingestion/   models, safe extraction/downloads, source adapters,
                        corpus reconstruction and deterministic chunker
rag/medical_embeddings.py
rag/medical_vector_store.py
rag/medical_retriever.py
scripts/download_medical_data.py
scripts/build_medical_index.py
scripts/test_medical_rag.py
scripts/audit_medical_index.py
data/medical_knowledge/
  corpus.json           admitted raw hashes and source selection
  raw/{medlineplus,pmc,who}/
  processed/chunks.jsonl
  qdrant/               separate local persistent database and embedding manifest
outputs/phase3/          actual index, retrieval, audit and test reports
```

Collection is fixed to **`medical_knowledge`**. It cannot be renamed to
`ddxplus_patient_cases`. Payloads have an exact medical allowlist; patient/split/
diagnosis-label fields are rejected. Source type and document ID prefixes are
validated. Serialized dataset-label markers are rejected, but legitimate medical
disease terms are not censored. These are structural controls, not a general PHI
classifier. The audit compares every payload to the reconstructed approved source.

## Reproduction

From `healthcare-agentic-ai`, with dependencies in `requirements.txt` installed:

```text
python scripts/download_medical_data.py
python scripts/build_medical_index.py
python scripts/test_medical_rag.py
python scripts/audit_medical_index.py
python scripts/build_medical_index.py --report outputs/phase3/index-second.json
python -m unittest discover -s tests -v
```

In this Windows workspace use `..\.venv\Scripts\python.exe` from the project
folder, or `.venv\Scripts\python.exe healthcare-agentic-ai\scripts\...` from its
parent. `uv pip install -r healthcare-agentic-ai/requirements.txt` installs dependencies
from the parent workspace. Add `--allow-download` to the first build only if the
BGE model is not already cached. No substitute model is used. No generation occurs.

Optional WHO: pass `--who-manifest <reviewed-selections.json>` to the downloader.
The development downloader limits selection to 20 topics, 10 PMC articles, and
5 WHO documents. It reports failures and exits nonzero while preserving a manifest
of successfully admitted sources. Review failures before using a partial corpus.

The dated XML endpoint is intentionally pinned, not guessed from a webpage.
Upstream retention/change may prevent future re-download: retain the local raw
snapshot and sidecars for exact reproducibility. Existing cached bytes are verified
and reused. `--manifest` can select another prepared corpus. `--storage-path`
selects a separate database; changed chunk IDs/corpus membership require a new path
rather than silently leaving stale vectors. Explicit relative paths are relative
to the working directory; defaults are project-anchored. Local Qdrant requires
one process at a time. Never run build/retrieval/audit concurrently on one database.

```python
from rag.medical_retriever import MedicalKnowledgeRetriever

with MedicalKnowledgeRetriever() as retriever:
    for hit in retriever.retrieve('What is asthma?', top_k=3):
        print(hit['text'], hit['title'], hit['url'], hit['license'], hit['score'])
```

## Actual validation

- `BAAI/bge-small-en-v1.5`, **384 dimensions**, CPU, normalized cosine vectors,
  512-token context, no query instruction prefix.
- Deterministic 180-word windows with 30-word overlap. UUIDs include source document,
  chunk index, chunking version and text SHA256. Full text remains in payloads.
- First and second builds each stored **87 chunks**; identical processed SHA256:
  `8558bd80147ce1a3d519a9d6f95ba55223e03cd8f06991cdc096cc7318781ad9`.
  Both reported **0 embedding truncations**. Provenance paths/timestamps are part
  of serialized output, so this JSONL hash is specific to the retained snapshot and
  sidecars; IDs remain independent of download paths/times.
- Three real BGE retrieval smoke queries, each top-k 3, returned text and provenance.
  Top titles respectively: **Asthma**, **Diabetes**, **High Blood Pressure**.
  These are plumbing smoke checks, not a labeled relevance benchmark.
- Separate-process reopening and audit passed for every one of the **87 chunks**:
  raw checksum, source reconstruction, schema/license, deterministic ID and vector
  dimension/normalization checks.
- Complete suite: **78 tests passed** (52 existing, 26 Phase 3), no skipped tests.
  Offline tests cover XML/PDF/text, normalization, metadata, license rejection,
  persistence, retrieval, patient-collection isolation, label-field rejection and
  deterministic reconstruction. One upstream Qdrant/SQLite ResourceWarning was
  observed during the first full run; all assertions passed.
- Machine-readable evidence: `outputs/phase3/index.json`, `index-second.json`,
  `retrieval.json`, `audit.json`; full test log: `outputs/phase3/tests.log`.

## Remaining limitations

Tiny English development corpus, no systematic relevance/recall benchmark, no
clinical validation, no medical accuracy or diagnostic performance claim. PMC
articles were selected for explicit reuse evidence, not guideline authority or
clinical coverage. WHO live acquisition is unvalidated and has zero indexed works.
PDF extraction has no OCR and rejects empty/scanned/encrypted content; complex
layouts may extract imperfectly. Word chunking can exceed the token budget on
other corpora (truncation is reported, not silently denied). Publication date
formats are preserved rather than guessed; MedlinePlus supplies creation dates.
No content freshness monitoring, retraction surveillance, multilingual support,
large-scale performance validation or general-purpose PHI detection. Model ID and
embedding settings are recorded, but the Hugging Face artifact revision is not
pinned. Similarity is neither diagnostic confidence nor evidence of correctness.

**Stop at Phase 3.** No Phase 4 components have been added.

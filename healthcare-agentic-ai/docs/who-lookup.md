# Deterministic local WHO lookup

Implements `prompts/whoinc2.txt` in `rag/who_lookup.py`. This is a standalone **index keyword/topic lookup**, not a diagnostic service or an addition to the production RAG pipeline. It uses only the Python standard library. There are no network calls, embeddings, vector databases, models, LLMs, semantic matching or rerankers.

## Demo commands

From `healthcare-agentic-ai`:

```console
python -m rag.who_lookup "abortion" --max-results 2
python -m rag.who_lookup "gaming disorder"
python -m rag.who_lookup "asthma and diabetes"
python -m rag.who_lookup "health symptoms"
```

The default prints JSON containing the query, `MATCH` or `NO_MATCH`, full loaded text, match explanation and source provenance. For example, `abortion` matches the Abortion Fact Sheet and Abortion: Safety Q&A in the current local collection. `health symptoms` returns `NO_MATCH`, not an inferred condition.

Metadata-only preview (does not open text files):

```console
python -m rag.who_lookup "asthma and diabetes" --metadata-only
```

Optional flags:

- `--data-dir PATH`: parent directory containing `who_fact_sheets/` and `who_questions_answers/`; default is the project's `data/` independent of the working directory.
- `--max-results N`: default 6, allowed range 1–20. The cap can exclude additional matching topics/documents; increase it when needed.
- `--metadata-only`: match indexes without loading or checking document bodies.

Exit status is 0 for `MATCH`, 1 for `NO_MATCH`, 2 for invalid command arguments. A result can contain `errors` alongside usable matches when some indexes or files are unavailable. No installation is needed beyond Python 3.10+. The demo was also tested with `python -S`, with site packages disabled.

## Reusable API

```python
from rag.who_lookup import WHOLookup

who = WHOLookup()  # reads the two existing index.json files once
result = who.lookup("asthma and diabetes", max_results=6)

if result.status == "NO_MATCH":
    print(result.reason)
else:
    for document in result.documents:
        print(document.source.title)
        print(document.source.url)
        print(document.source.document_type)  # fact_sheet or question_answer
        print(document.source.filename)
        print(document.match_type, document.matched_terms)
        print(document.text)  # full original UTF-8 text, including metadata/headings

for error in result.errors:
    print(error)
```

`who.search(query)` returns the same immutable result structure with `text=None`. `who.lookup(query)` loads only selected files, sequentially. `dataclasses.asdict(result)` produces a JSON-serializable structure. Reuse the instance for repeated queries; create a new instance to reload indexes after a crawl. No text cache or corpus rewrite is performed.

## Matching contract

Only `topic` and `title` from the existing `index.json` files are searched. URL lists and article bodies are not searched, and unindexed files are not considered available knowledge.

1. Normalize case, accents and punctuation to whole-word tokens. Source metadata and text are never normalized or rewritten in the result.
2. Exclude grammatical/request words and broad medical/demographic qualifiers as standalone evidence. For example, `health`, `symptoms`, `treatment`, `children` and numbers alone are insufficient.
3. Prefer an exact normalized topic/title, then a complete topic/title phrase appearing in a longer query.
4. Admit a partial match when at least two informative terms cover half the document's indexed informative terms, or when a single informative term occurs in at most two indexed documents or is explicitly indexed as a standalone topic. The latter permits a topic such as `abortion` to locate related Q&A titles without inventing medical aliases.
5. Order once by match type, informative-word overlap, coverage, then stable topic/type/filename/URL tie breakers. These are explicit lexical rules, not a learned reranker or medical confidence score.
6. Return multiple eligible documents up to the caller's bound, including both collections. If none qualify, return `NO_MATCH`.

No substring, fuzzy-spelling, synonym, acronym-expansion or symptom-to-diagnosis guesses are made. For example, `asth` does not match `asthma`; `anemia` is not silently rewritten to WHO's `anaemia`. Single-term partial eligibility depends on the currently loaded indexes and can become more conservative as the corpus grows. A literal topic mention is not semantic understanding: negation, user intent and clinical relevance are not inferred. `NO_MATCH` means no reliable *local index match*, not that WHO lacks the topic or that a medical condition is absent.

## Provenance, failures and memory

Each match retains `topic`, exact `title`, WHO `url`, `document_type` and `filename`. Before loading a document, its file path is checked for collection/data-directory containment. Source URLs must match the indexed WHO collection/topic. The stored document's WHO marker, title and URL must agree with its index entry; missing/malformed metadata or empty bodies are rejected.

Missing indexes, malformed entries, unsafe filenames/URLs, unreadable text, corrupt UTF-8, mismatched headers and oversized files are reported in `errors`; other eligible documents can still be returned. If all matching files fail, `lookup()` returns `NO_MATCH` with an explicit loading-failure reason rather than fabricating content. `search()` only promises metadata matches, not readable documents.

Bounds: 4,096 query characters, 5 MiB per index, 4 MiB per loaded document, and at most 20 returned documents (6 by default). Oversized documents fail explicitly rather than being truncated. Only lightweight metadata and an in-memory keyword-to-entry map are retained between calls. Whole texts are read only for matches and are not cached. The API returns **full documents**, not extracted section snippets, so original section structure is retained without guessing which section answers a medical question.

## Validation

```console
python -m unittest tests.test_who_lookup -v
python -m unittest tests.test_who_lookup tests.test_who_fact_sheets tests.test_who_questions_answers
```

- **28 new lookup tests passed**: exact title/topic, normalized Unicode, partial matches, multiple topics/collections, explicit no-match cases, ambiguous/generic terms, deterministic order, bounded loading, metadata-only operation, full-text/provenance preservation, malformed data, unsafe paths/URLs, relocated collections and CLI behavior without site packages.
- **70 WHO tests passed in total**, including all 20 existing Fact Sheets and 22 Q&A crawler tests. Lookup tests alone run in the workspace virtual environment without crawler dependencies; the combined suite used the existing system Python with requests/BeautifulSoup.
- Initial validation used **244 Fact Sheets + 5 Q&A documents = 249 entries**. The on-disk Q&A index then expanded during this task without this implementation running a crawler. A fresh instance was revalidated against **244 Fact Sheets + 325 Q&A documents = 569 entries**, with no index errors. Both checkpoints are recorded in the report.
- `abortion` returned both source types; `gaming disorder` returned the Q&A; `asthma and diabetes` returned separate Fact Sheets; generic/unrelated queries returned `NO_MATCH`.
- All demo-loaded text was compared with the complete original files. `docs/who-lookup-validation.json` records source metadata, statuses, lengths and checksums without duplicating article text. `outputs/who-lookup-demo.json` contains the complete JSON output from the two-document abortion CLI demo.
- At the initial checkpoint, **479 pre-existing files were hash-verified unchanged**: production Python under `rag`, `application`, `orchestration`, `safety`; existing WHO crawlers/tests; and both local WHO collection directories. This hash checkpoint preceded the subsequent Q&A collection update, so it is not a claim that the later index stayed identical. The lookup implementation never writes either collection. Zero network calls or diagnostic-agent runs were required.

## Change boundary

Added `rag/who_lookup.py`, `tests/test_who_lookup.py`, this guide, the validation report and a local demo output. Updated both OKF manifests. Existing WHO crawlers, source indexes/texts, dependencies, AMG, Patient RAG, diagnostic/grounding/critic/safety and orchestration were not modified. No exports or workflow wiring were added to the existing production packages. User prompt edits and the pre-existing vendor dirty state were left alone.

# Local WHO Questions & Answers collection

Adds an independent Q&A entrypoint for the updated `prompts/whoinc.txt`. The existing Fact Sheets crawler, its tests, dependencies and generated files remain byte-for-byte unchanged. Only its pure text-rendering and atomic-file helpers are imported; no healthcare application components are imported or modified.

## Commands

From `healthcare-agentic-ai`:

```console
python -m pip install -r requirements-who.txt
python scripts/crawl_who_questions_answers.py --limit 5
```

**Full Q&A crawl:**

```console
python scripts/crawl_who_questions_answers.py
```

The default processes every discovered Q&A URL automatically. Existing valid local files are skipped. Use `--force` to refresh them:

```console
python scripts/crawl_who_questions_answers.py --force
```

The original Fact Sheets command is unchanged:

```console
python scripts/crawl_who_fact_sheets.py
```

Both regression suites:

```console
python -m unittest tests.test_who_questions_answers tests.test_who_fact_sheets -v
```

The new crawler also supports `--output-dir PATH`, `--delay SECONDS` (default 1), and `--timeout SECONDS` (default 30). `--limit N` selects the first N alphabetically sorted topics, including cached ones; it does not limit discovery. Default output is project-relative regardless of the shell's working directory. Only Python 3.10+, requests and BeautifulSoup are needed. Validation used the existing system Python 3.13 interpreter because the workspace virtual environment lacked BeautifulSoup; no environment/dependency files were changed.

## Discovery and memory safety

The starting page is `https://www.who.int/news-room/questions-and-answers`. Currently it exposes **no static item links**: WHO's listing widget uses `/api/hubs/qandagroups` and a template that appends `ItemDefaultUrl` to the Q&A item prefix.

The crawler reads that endpoint and template from the catalogue's inline configuration without executing JavaScript or downloading scripts. It requests the same English catalogue using requests, in sequential batches of at most 100 records, with `$skip`, `$top` and `$count`. All pages are discovered before applying `--limit`. Missing/malformed API data, inconsistent counts, prematurely ended responses or repeated pages fail discovery rather than silently recording a complete crawl. Static direct item links are also supported and merged with API links if present.

Topics are actual WHO-provided URLs, not guessed words. URLs are normalized, deduplicated and sorted. Unicode, parentheses and periods (including WHO's real `dr.-teodora-wi` slug) are supported; foreign URLs, credentials, unrelated paths and path traversal are rejected. Redirects must stay in the same request category: catalogue, Q&A item, or the exact Q&A listing API endpoint. No article links are followed and no unrelated assets are downloaded.

One catalogue batch or article is processed at a time; only lightweight URL/index metadata accumulates. No models, agents, embeddings, vector databases, RAG, semantic search, browser framework or multiprocessing is used.

## Extraction and files

Output is `data/who_questions_answers/`:

- One UTF-8 `<topic>.txt` per Q&A, with `WHO QUESTIONS AND ANSWERS`, source title, URL and UTC retrieval timestamp.
- `urls.json`: complete discovered URL list, including items outside a limited run.
- `index.json`: lightweight `topic`, `title`, `url`, `file` lookup entries for valid local files.
- `crawl_report.json`: latest run's discovery/download/failure counts, failed URLs/errors, duplicate/cached/limit skips, timestamp and output directory. Discovery failures are separate from page failures. `skipped_duplicate_urls` sums duplicate links, cached files and limit exclusions.

Extraction retains the main Q&A content, introductions, every question/answer pair, lists, tables, existing subheadings and final references. WHO accordion questions become underlined text headings. **Collapsed/hidden answers are preserved**, not discarded based on visibility attributes. Original numbering and wording are not corrected or rewritten. Navigation, related-item sidebars, page-date controls, scripts, sharing/cookie controls and footer boilerplate are removed. Semantic FAQ, native disclosure and main/article fallbacks are supported; whole-document boilerplate fallback is deliberately refused. Unknown/incomplete question panels fail explicitly rather than silently losing an answer.

No summarization or length truncation is applied. Plain text does not reproduce all CSS styling or complex table layout. Old WHO assertions are preserved as source material, not endorsed as current clinical advice.

A failure on one detail page is recorded without aborting later pages. Atomic writes retain a previous good file if refresh fails. Reruns rebuild the index from small metadata headers without reading the whole corpus and retain valid files from prior runs. Discovery failure preserves the previous URL list and index. Exit status is nonzero for discovery or page failure. Run only one crawler instance per output directory.

## Validation results

Evidence: `docs/who-questions-answers-validation.json`.

- **326 unique Q&A URLs discovered**, matching the live API count, across four catalogue API requests; zero duplicate accepted links.
- Small `--limit 5` crawl: **5 successful downloads, 0 failures**, 321 topics excluded by the limit.
- All five output titles/URLs and question headings were checked against live WHO source HTML. **37 question/answer pairs** were checked; all **88 nonempty source answer text segments** were found unchanged and in source order after whitespace normalization. Introductions, lists and substantive text were inspected; unrelated navigation/footer text did not dominate.
- Ordinary `--limit 5` rerun: **5 cached articles skipped, 0 detail downloads, 0 failures**. It still checks the catalogue. Article SHA-256 hashes and index entries remained unchanged. The latest output report records this rerun; the validation JSON preserves both reports.
- **42 offline tests passed: 22 Q&A tests plus all 20 unchanged Fact Sheets tests.** Coverage includes API pagination/count errors, real dotted slugs, direct-link filtering/deduplication, collapsed answers, extraction fallbacks, long answers, indexes, limits, cached and forced reruns, HTTP/timeout/extraction failures, retained files after failed refresh, request politeness and scoped redirects.
- Hash checks confirm the Fact Sheets script, tests, shared requirements and all eight existing Fact Sheets data files were unchanged.
- Full Q&A detail collection was **not downloaded**. The remaining 321 pages and future WHO layout changes are not yet validated. The full-crawl command above processes them automatically.

Sample outputs:

```text
abortion-safety.txt
accessing-implementing-gcf-readiness-funds.txt
addictive-behaviours-gaming-disorder.txt
adolescent-health-and-development.txt
adolescent-sexual-and-reproductive-health.txt
```

## Change inventory and boundary

Added: `scripts/crawl_who_questions_answers.py`, `tests/test_who_questions_answers.py`, this guide, `docs/who-questions-answers-validation.json`, five Q&A texts and three collection JSON files. Updated the workspace and project OKF manifests. Existing `requirements-who.txt` is reused unchanged.

No AMG, Patient RAG, medical retrieval, diagnosis, grounding, critic, safety, orchestration, existing index or medical-corpus changes were made. The pre-existing vendor dirty marker and user prompt edits were left alone. No diagnostic integration was performed.

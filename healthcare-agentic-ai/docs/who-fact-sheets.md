# Local WHO fact-sheet crawler

Independent collection utility requested by `prompts/whoinc.txt`. It does **not** import or modify the healthcare runtime, AMG, Patient RAG, medical retrieval, diagnostic/grounding/critic/safety components, orchestration, existing indexes or medical corpus. There are no embeddings, models, agents, semantic search, vector databases or runtime integration.

## Install and run

From `healthcare-agentic-ai`, using Python 3.10 or newer:

```console
python -m pip install -r requirements-who.txt
python scripts/crawl_who_fact_sheets.py --limit 5
```

Full crawl (the requested exact command):

```console
python scripts/crawl_who_fact_sheets.py
```

Refresh existing files explicitly:

```console
python scripts/crawl_who_fact_sheets.py --force
```

Offline tests:

```console
python -m unittest tests.test_who_fact_sheets -v
```

Only `requests` and `beautifulsoup4` are needed. Do not install the application's model/database dependencies just to run this utility. In the implementation environment, these libraries were already available in `C:\Users\Hp\AppData\Local\Programs\Python\Python313\python.exe`; the workspace `.venv` lacked BeautifulSoup. Validation used that existing system interpreter without modifying either environment.

Options:

| Option | Behavior |
| --- | --- |
| `--limit N` | Process the first N alphabetically sorted discovered topics, including already cached ones; still save the complete discovered URL list. |
| `--force` | Refresh the selected existing files. Without it, files with valid matching metadata headers are skipped. |
| `--output-dir PATH` | Override the default project-relative `data/who_fact_sheets/`. The default does not depend on the shell's working directory. |
| `--delay SECONDS` | Pause between all HTTP requests, including redirects; default 1 second. |
| `--timeout SECONDS` | Requests timeout; default 30 seconds. |

## Scope and outputs

The crawler requests `https://www.who.int/news-room/fact-sheets` and parses its actual links across the A–Z catalogue. It keeps only English WHO `/news-room/fact-sheets/detail/<topic>` links, resolves relative URLs, normalizes scheme/host/encoding, removes queries/fragments/trailing slashes, deduplicates, and sorts by topic. It preserves path case and supports Unicode and parenthesized topics. It does not guess topic URLs, follow article links, recursively crawl, fetch assets, run JavaScript or use parallel workers. Redirects outside the catalogue/detail scope are refused.

Output files:

- `<topic>.txt`: UTF-8, WHO title, discovered canonical URL, UTC retrieval timestamp, and the **whole cleaned article**.
- `urls.json`: all discovered normalized URLs, including those outside a limited run.
- `index.json`: only `topic`, `title`, `url`, `file`; maps valid local files to topics, including files from previous runs.
- `crawl_report.json`: latest run's timestamp, output directory, discovery count, successful and failed downloads, failed URLs/errors, duplicates, cached skips and limit exclusions. `skipped_duplicate_urls` is the sum of those three skip categories, not just duplicate links. Index-fetch errors are reported separately as `discovery_error`.

WHO's main article wrapper is preferred. Older WHO templates put Key facts immediately before the article, so their named **Body** column is retained while the surrounding sidebar is excluded. Fallbacks cover `articleBody`, `article`, `main`, `#content` and WHO detail-content containers; there is deliberately no whole-document text fallback. The title is captured before removing page headers. Navigation, scripts, footer, cookie/share controls and other known boilerplate are removed. In-article medical references and related medical link labels remain as source wording; they are not followed.

HTML headings are retained in source order with plain-text underlines, alongside paragraphs, bullets, numbered lists and table cells. Visually styled non-heading text is retained as written (for example, the older-people page's span-styled “Overview” remains a separate line without an invented HTML heading level). There is no summary, medical rewriting or length truncation. Text-only output cannot reproduce CSS styling or complex merged-cell table layout exactly.

Requests, parsing and writing happen one page at a time. Earlier HTML/text is not retained. Index reconstruction reads only small file headers, not the corpus. Existing files are refreshed only with `--force` or when their crawler metadata header is invalid. Atomic replacement prevents a failed refresh from overwriting a good article. A single HTTP/timeout/extraction/write failure is recorded and processing continues; the process exits nonzero if any page fails. Discovery failure leaves the previous URL list and index intact. A full run is safely resumable by rerunning it. Run only one crawler instance against a given output directory at a time.

## Validation performed

Recorded in `docs/who-fact-sheets-validation.json` (including file hashes and heading inventories).

- **244** unique URLs discovered from the live WHO index, with **0** duplicate accepted links.
- Five-page live validation: **5 downloads succeeded, 0 failed**; **239** topics excluded by `--limit 5`.
- Inspection identified the older template's separate Key facts block; extraction was corrected and all five pages were explicitly refreshed with `--force --limit 5`.
- All five final files were inspected for source title/URL, medical paragraphs, headings, references and absence of dominating navigation/footer boilerplate. The older-people file includes its Key facts, Overview, risk factors, prevention and table.
- A second ordinary `--limit 5` run skipped **all 5** local articles and downloaded **0** details. File SHA-256 hashes and index entries remained unchanged. Consequently the current `crawl_report.json` records this cache-validation run, while the documentation validation JSON preserves both reports.
- **20 offline tests passed**: filtering, normalization, safe filenames, Unicode, deduplication/sorting, extraction fallbacks, WHO legacy Key facts, article-header title, section order, numbered lists, long-content preservation, index generation, limits, cached/forced reruns, retained files after failed refresh, discovery failure, per-page HTTP/timeout/parse failures, request delay/timeout and redirect restrictions.
- The full catalogue was **not** downloaded. Unvisited pages and future WHO template changes remain unvalidated. No completeness claim is made beyond the links exposed by the current index HTML.

Generated examples:

```text
data/who_fact_sheets/abortion.txt
data/who_fact_sheets/abuse-of-older-people.txt
data/who_fact_sheets/adolescent-mental-health.txt
data/who_fact_sheets/adolescent-pregnancy.txt
data/who_fact_sheets/adolescents-health-risks-and-solutions.txt
```

## Files added/modified

Added: `scripts/crawl_who_fact_sheets.py`, `requirements-who.txt`, `tests/test_who_fact_sheets.py`, this guide, `docs/who-fact-sheets-validation.json`, five topic files and the three JSON outputs in `data/who_fact_sheets/`.

Updated: the workspace and project `okf.yaml` manifests only. Existing healthcare source, dependencies, stores and vendor files were not changed. The vendor AMG nested-worktree dirty marker was present before this task and was left alone. No WHO diagnostic integration was performed.

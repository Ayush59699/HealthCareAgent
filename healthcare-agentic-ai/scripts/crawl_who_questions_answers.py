"""Sequential local WHO Q&A collection; no healthcare runtime integration.

Uses the catalogue's own paginated API when links are loaded dynamically.
Install requirements-who.txt; see docs/who-questions-answers.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

# Reuse only pure text/atomic-file helpers. The original crawler stays unchanged.
if __package__:
    from .crawl_who_fact_sheets import atomic_write, clean_text, render_article, write_json
else:
    from crawl_who_fact_sheets import atomic_write, clean_text, render_article, write_json

INDEX_URL = "https://www.who.int/news-room/questions-and-answers"
DETAIL_PATH = "/news-room/questions-and-answers/item/"
API_PATH = "/api/hubs/qandagroups"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "who_questions_answers"
USER_AGENT = "WHOQuestionsAnswersLocalCrawler/1.0 (sequential local research collection)"
PAGE_SIZE = 100


def who_parts(href: str, base: str = INDEX_URL):
    parts = urlsplit(urljoin(base, href.strip()))
    if (parts.scheme not in {"http", "https"}
            or parts.hostname not in {"www.who.int", "who.int"}
            or parts.username or parts.password or parts.port not in {None, 80, 443}):
        raise ValueError("Not an allowed WHO URL")
    return parts


def normalize_url(href: str, base: str = INDEX_URL) -> str | None:
    try:
        parts = who_parts(href, base)
        path = unquote(parts.path, errors="strict").rstrip("/")
        if not path.startswith(DETAIL_PATH):
            return None
        topic = unicodedata.normalize("NFC", path[len(DETAIL_PATH):])
        if not any(c.isalnum() for c in topic) or not all(c.isalnum() or c in "-()_." for c in topic):
            return None
        return "https://www.who.int" + DETAIL_PATH + quote(topic, safe="-()_.")
    except (ValueError, UnicodeError):
        return None


def topic_from_url(url: str) -> str:
    normalized = normalize_url(url)
    if normalized is None:
        raise ValueError(f"Not a WHO Q&A URL: {url}")
    return unquote(urlsplit(normalized).path[len(DETAIL_PATH):])


def filename_for_url(url: str) -> str:
    topic = topic_from_url(url)
    if re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])", topic.split(".")[0]):
        topic = "_" + topic
    return topic + ".txt"


def sort_urls(urls) -> list[str]:
    return sorted(urls, key=lambda u: (topic_from_url(u).casefold(), u))


def discover_urls(html: bytes | str) -> tuple[list[str], int]:
    """Direct-link discovery for static WHO catalogue HTML."""
    soup = BeautifulSoup(html, "html.parser")
    urls = [url for a in soup.select("a[href]") if (url := normalize_url(a["href"]))]
    return sort_urls(set(urls)), len(urls) - len(set(urls))


def api_configuration(html: bytes | str) -> str | None:
    """Read, never execute, WHO's embedded listing configuration/template."""
    soup = BeautifulSoup(html, "html.parser")
    endpoint = None
    template_found = False
    for script in soup.find_all("script"):
        text = script.get_text()
        if re.search(r'https://www\.who\.int/news-room/questions-and-answers/item#:\s*ItemDefaultUrl\s*#', text):
            template_found = True
        match = re.search(r'hubsfiltering\([^;]*?[\"\'](/api/hubs/qandagroups\?[^\"\']+)[\"\']', text)
        if match:
            endpoint = urljoin(INDEX_URL, match.group(1).replace("&amp;", "&"))
    if endpoint and not template_found:
        raise ValueError("Q&A API found but its item URL template is unrecognized")
    return endpoint


def discover_collection(fetcher) -> tuple[list[str], int]:
    html = fetcher.get(INDEX_URL)
    direct, duplicates = discover_urls(html)
    endpoint = api_configuration(html)
    del html
    urls = set(direct)
    if endpoint is None:
        if not urls:
            raise ValueError("No Q&A links or recognized catalogue API configuration")
        return sort_urls(urls), duplicates
    parts = urlsplit(endpoint)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({"$top": str(PAGE_SIZE), "$count": "true"})
    offset = 0
    batches_seen = set()
    expected_count = None
    # Bounded sequential catalogue pages; never fetch scripts or guess topic names.
    for _ in range(1000):
        query["$skip"] = str(offset)
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
        payload = json.loads(fetcher.get(url))
        if not isinstance(payload, dict) or not isinstance(payload.get("value"), list):
            raise ValueError("Malformed Q&A catalogue API response")
        rows = payload["value"]
        count = payload.get("@odata.count")
        if count is not None:
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Invalid Q&A catalogue count")
            if expected_count is not None and count != expected_count:
                raise ValueError("Q&A catalogue changed during pagination; rerun discovery")
            expected_count = count
        if not rows:
            if expected_count is not None and offset < expected_count:
                raise ValueError("Q&A API ended before the advertised count")
            return sort_urls(urls), duplicates
        batch = []
        for row in rows:
            suffix = row.get("ItemDefaultUrl") if isinstance(row, dict) else None
            # ItemDefaultUrl is the suffix used by the catalogue's verified template.
            url = normalize_url(INDEX_URL + "/item" + suffix) if isinstance(suffix, str) and suffix.startswith("/") else None
            if url is None:
                raise ValueError(f"Invalid Q&A ItemDefaultUrl: {suffix!r}")
            batch.append(url)
        signature = tuple(batch)
        if signature in batches_seen:
            raise ValueError("Q&A API repeated a page; refusing incomplete discovery")
        batches_seen.add(signature)
        for url in batch:
            if url in urls:
                duplicates += 1
            urls.add(url)
        offset += len(rows)
        if expected_count is not None and offset >= expected_count:
            if offset != expected_count:
                raise ValueError("Q&A API returned more rows than advertised")
            return sort_urls(urls), duplicates
        del payload, rows, batch
    raise ValueError("Q&A API pagination exceeded safety bound")


# Do NOT remove [hidden]/aria-hidden: collapsed answers are medical content.
JUNK = (
    "script, style, noscript, template, nav, footer, aside, form, "
    "[role='navigation'], [role='banner'], [role='contentinfo'], "
    ".qa-details__related, .qa-details__header, .social-share, .share-buttons, "
    ".sf-social-share, .sf-sharing, .cookie-banner, #onetrust-banner-sdk, "
    ".breadcrumb, .breadcrumbs, .advertisement"
)


def extract_article(html: bytes | str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.select_one(".qa-details__title") or soup.select_one(".qa-details__wrapper h1, article h1, main h1") or soup.find("h1")
    title = clean_text(title_node.get_text()) if title_node else ""
    if not title:
        meta = soup.select_one("meta[property='og:title']")
        title = clean_text(meta.get("content", "")) if meta else ""
    if not title and soup.title:
        title = clean_text(soup.title.get_text())
    if not title:
        raise ValueError("Q&A title is missing")
    root = None
    for selector in (".qa-details__wrapper", ".qa-details__content", "[itemtype$='/FAQPage']", ".sf-accordion", "[itemprop='articleBody']", "article", "main", "#content"):
        candidates = soup.select(selector)
        if candidates:
            root = max(candidates, key=lambda node: len(node.get_text(strip=True)))
            if root.get_text(strip=True):
                break
    if root is None:
        raise ValueError("No recognizable main Q&A content; refusing whole-page boilerplate")
    for tag in root.select(JUNK):
        tag.decompose()
    for h1 in root.find_all("h1"):
        if clean_text(h1.get_text()) == title:
            h1.decompose()
    # Promote the WHO accordion trigger to an actual heading for the shared renderer.
    # Preserve answers even when hidden/collapsed; retain intro and trailing references.
    panels = root.select(".sf-accordion__panel, [itemtype$='/Question']")
    for panel in panels:
        question = panel.select_one(".sf-accordion__trigger-panel") or panel.select_one("[itemprop='name']")
        answer = panel.select_one(".sf-accordion__content, [itemprop='acceptedAnswer']")
        if question is None or not clean_text(question.get_text()):
            raise ValueError("Q&A panel has no question")
        if answer is None or not clean_text(answer.get_text()):
            raise ValueError("Q&A panel has no answer")
        heading = soup.new_tag("h2")
        heading.string = clean_text(question.get_text())
        question.replace_with(heading)
    # Also support native HTML disclosure widgets without fetching/executing JS.
    for summary in root.select("details > summary"):
        summary.name = "h2"
    body = render_article(root)
    if not body:
        raise ValueError("Q&A content is empty")
    return title, body


class PoliteFetcher:
    def __init__(self, session: requests.Session, delay: float, timeout: float):
        self.session = session
        self.delay = delay
        self.timeout = timeout
        self.requested = False

    @staticmethod
    def kind(url: str) -> str:
        parts = who_parts(url)
        if parts.path.rstrip("/") == urlsplit(INDEX_URL).path:
            return "index"
        if parts.path == API_PATH:
            return "api"
        if normalize_url(url):
            return "detail"
        raise ValueError(f"Refusing unrelated URL: {url}")

    def get(self, url: str) -> bytes:
        kind = self.kind(url)
        for _ in range(6):
            if self.requested:
                time.sleep(self.delay)
            self.requested = True
            with self.session.get(url, timeout=self.timeout, allow_redirects=False) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    target = urljoin(url, response.headers.get("Location", ""))
                    if self.kind(target) != kind:
                        raise ValueError(f"Refusing cross-collection redirect: {target}")
                    url = target
                    continue
                response.raise_for_status()
                return response.content
        raise ValueError("Too many redirects")


def existing_entry(path: Path, url: str) -> dict[str, str] | None:
    """Read only the small header, never load a saved article into memory."""
    try:
        with path.open(encoding="utf-8") as handle:
            lines = [handle.readline().strip() for _ in range(4)]
        if (lines[0] == "WHO QUESTIONS AND ANSWERS" and lines[1].startswith("Title: ")
                and lines[2] == f"URL: {url}" and lines[3].startswith("Retrieved: ")):
            return {"topic": topic_from_url(url), "title": lines[1][7:], "url": url, "file": path.name}
    except (OSError, UnicodeError):
        pass
    return None


def generate_index(output: Path) -> list[dict[str, str]]:
    entries = []
    for path in output.glob("*.txt"):
        try:
            with path.open(encoding="utf-8") as handle:
                handle.readline()
                handle.readline()
                url_line = handle.readline().strip()
            url = normalize_url(url_line.removeprefix("URL: "))
            if url and filename_for_url(url) == path.name:
                entry = existing_entry(path, url)
                if entry:
                    entries.append(entry)
        except (OSError, UnicodeError):
            continue
    return sorted(entries, key=lambda item: (item["topic"].casefold(), item["url"]))


def crawl(output: Path = DEFAULT_OUTPUT, *, force: bool = False, limit: int | None = None,
          delay: float = 1.0, timeout: float = 30.0) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not 0 <= delay < float("inf") or not 0 < timeout < float("inf"):
        raise ValueError("delay must be nonnegative and timeout positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    report = {"crawl_timestamp": timestamp, "total_urls_discovered": 0,
              "successful_downloads": 0, "failed_downloads": 0,
              "duplicate_urls": 0, "skipped_existing": 0, "skipped_by_limit": 0,
              "skipped_duplicate_urls": 0, "output_directory": str(output),
              "failed_urls": [], "errors": [], "discovery_error": None}
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        fetcher = PoliteFetcher(session, delay, timeout)
        try:
            urls, duplicates = discover_collection(fetcher)
            if not urls:
                raise ValueError("WHO index returned no detail links; previous URL list/index left intact")
        except (requests.RequestException, ValueError) as exc:
            report["discovery_error"] = str(exc)
            write_json(output / "crawl_report.json", report)
            return report
        report["total_urls_discovered"] = len(urls)
        report["duplicate_urls"] = duplicates
        write_json(output / "urls.json", urls)
        selected = urls if limit is None else urls[:limit]
        report["skipped_by_limit"] = len(urls) - len(selected)
        for url in selected:
            path = output / filename_for_url(url)
            if not force and existing_entry(path, url):
                report["skipped_existing"] += 1
                print(f"SKIP {path.name}")
                continue
            try:
                title, body = extract_article(fetcher.get(url))
                text = f"WHO QUESTIONS AND ANSWERS\nTitle: {title}\nURL: {url}\nRetrieved: {datetime.now(timezone.utc).isoformat()}\n\n{'=' * 40}\n\n{body}\n"
                atomic_write(path, text)
                del body, text
                report["successful_downloads"] += 1
                print(f"OK   {path.name}")
            except (requests.RequestException, ValueError, OSError) as exc:
                report["failed_downloads"] += 1
                report["failed_urls"].append(url)
                report["errors"].append({"url": url, "error": str(exc)})
                print(f"FAIL {url}: {exc}")
    # Include earlier valid files even when this run is limited or a refresh fails.
    write_json(output / "index.json", generate_index(output))
    report["skipped_duplicate_urls"] = report["duplicate_urls"] + report["skipped_existing"] + report["skipped_by_limit"]
    write_json(output / "crawl_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refresh existing Q&A pages")
    parser.add_argument("--limit", type=int, help="Process first N discovered topics (including cached files)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (default: 1)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds (default: 30)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.delay < float("inf") or not 0 < args.timeout < float("inf"):
        parser.error("--delay must be finite/nonnegative and --timeout finite/positive")
    try:
        report = crawl(args.output_dir, force=args.force, limit=args.limit, delay=args.delay, timeout=args.timeout)
    except OSError as exc:
        parser.exit(1, f"Cannot write crawler output: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["discovery_error"] or report["failed_downloads"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

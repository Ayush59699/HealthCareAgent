"""Sequential, local WHO fact-sheet collection. No application/model imports.

Install only requirements-who.txt; see docs/who-fact-sheets.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

INDEX_URL = "https://www.who.int/news-room/fact-sheets"
DETAIL_PATH = "/news-room/fact-sheets/detail/"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "who_fact_sheets"
USER_AGENT = "WHOFactSheetLocalCrawler/1.0 (sequential local research collection)"


def normalize_url(href: str, base: str = INDEX_URL) -> str | None:
    """Accept only WHO English detail links; discard query/fragment/trailing slash.

    Preserve path case; canonicalize Unicode and percent encoding. Reject path
    separators/traversal and unsafe filename characters rather than guessing URLs.
    """
    try:
        parts = urlsplit(urljoin(base, href.strip()))
        if (parts.scheme not in {"http", "https"}
                or parts.hostname not in {"www.who.int", "who.int"}
                or parts.username or parts.password
                or parts.port not in {None, 80, 443}):
            return None
        path = unquote(parts.path, errors="strict").rstrip("/")
        if not path.startswith(DETAIL_PATH):
            return None
        topic = unicodedata.normalize("NFC", path[len(DETAIL_PATH):])
        if not topic or not all(c.isalnum() or c in "-()_" for c in topic):
            return None
        return "https://www.who.int" + DETAIL_PATH + quote(topic, safe="-()_")
    except (ValueError, UnicodeError):
        return None


def topic_from_url(url: str) -> str:
    normalized = normalize_url(url)
    if normalized is None:
        raise ValueError(f"Not a WHO fact-sheet URL: {url}")
    return unquote(urlsplit(normalized).path[len(DETAIL_PATH):])


def filename_for_url(url: str) -> str:
    topic = topic_from_url(url)
    # Windows device names are reserved even with a .txt extension.
    if re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])", topic):
        topic = "_" + topic
    return topic + ".txt"


def discover_urls(html: bytes | str) -> tuple[list[str], int]:
    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()
    duplicates = 0
    for link in soup.select("a[href]"):
        url = normalize_url(link["href"])
        if url is not None:
            if url in urls:
                duplicates += 1
            urls.add(url)
    return sorted(urls, key=lambda u: (topic_from_url(u).casefold(), u)), duplicates


JUNK = (
    "script, style, noscript, template, nav, header, footer, aside, form, "
    "[hidden], [aria-hidden='true'], [role='navigation'], [role='banner'], "
    "[role='contentinfo'], .social-share, .share-buttons, .sf-social-share, "
    ".sf-sharing, .cookie-banner, #onetrust-banner-sdk, .breadcrumb, "
    ".breadcrumbs, .advertisement"
)
BLOCKS = {"p", "div", "section", "article", "blockquote", "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "table", "tr"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def render_article(root: Tag) -> str:
    """Keep all text in document order, including lists, tables and references.

    Inline elements are concatenated without inserting spaces into punctuation.
    Only whitespace and plain-text structural markers are changed.
    """
    def render(node: Tag | NavigableString) -> str:
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            return re.sub(r"\s+", " ", str(node))
        if node.name in HEADINGS:
            heading = clean_text(node.get_text())
            underline = ("=" if node.name in {"h1", "h2"} else "-") * len(heading)
            return f"\n\n{heading}\n{underline}\n\n"
        if node.name == "br":
            return "\n"
        inner = "".join(render(child) for child in node.children if isinstance(child, (Tag, NavigableString)))
        if node.name == "li":
            marker = "-"
            if node.parent.name == "ol":
                siblings = node.parent.find_all("li", recursive=False)
                number = int(node.parent.get("start", len(siblings) if node.parent.has_attr("reversed") else 1))
                step = -1 if node.parent.has_attr("reversed") else 1
                for sibling in siblings:
                    number = int(sibling.get("value", number))
                    if sibling is node:
                        break
                    number += step
                marker = f"{number}."
            return "\n" + marker + " " + inner.strip() + "\n"
        if node.name in {"td", "th"}:
            return inner.strip() + " | "
        if node.name in BLOCKS:
            return "\n\n" + inner + "\n\n"
        return inner

    text = render(root)
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_article(html: bytes | str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title_node = (soup.select_one(".sf-item-header-wrapper h1")
                  or soup.select_one("article h1, main h1, #content h1") or soup.find("h1"))
    page_title = clean_text(title_node.get_text()) if title_node else ""
    for tag in soup.select(JUNK):
        tag.decompose()
    # WHO's article wrapper excludes the sibling related-links/sidebar widgets.
    root = None
    for selector in (".sf-detail-body-wrapper", "[itemprop='articleBody']", "article", "main", "#content", ".sf-detail-content"):
        candidates = soup.select(selector)
        if candidates:
            root = max(candidates, key=lambda t: len(t.get_text(strip=True)))
            if root.get_text(strip=True):
                break
    if root is None or not root.get_text(strip=True):
        raise ValueError("No recognizable main fact-sheet article; refusing whole-page boilerplate")
    # Older WHO templates put Key facts BEFORE the article in the Body column.
    # Expand to that column only, never the surrounding row containing sidebars.
    if "sf-detail-body-wrapper" in root.get("class", []):
        column = root.find_parent(attrs={"data-placeholder-label": "Body"})
        if column is not None:
            root = column
    if page_title:
        title = page_title
    else:
        meta = soup.select_one("meta[property='og:title']")
        title = clean_text(meta.get("content", "")) if meta else ""
        if not title and soup.title:
            title = clean_text(soup.title.get_text())
    if not title:
        raise ValueError("Fact-sheet title is missing")
    # The title is already in the metadata header. Preserve all other headings.
    for h1 in root.find_all("h1"):
        if clean_text(h1.get_text()) == title:
            h1.decompose()
    body = render_article(root)
    if not body:
        raise ValueError("Fact-sheet article is empty")
    return title, body


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def existing_entry(path: Path, url: str) -> dict[str, str] | None:
    """Read only the small header, never load a saved article into memory."""
    try:
        with path.open(encoding="utf-8") as handle:
            lines = [handle.readline().strip() for _ in range(4)]
        if (lines[0] == "WHO FACT SHEET" and lines[1].startswith("Title: ")
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


class PoliteFetcher:
    def __init__(self, session: requests.Session, delay: float, timeout: float):
        self.session = session
        self.delay = delay
        self.timeout = timeout
        self.requested = False

    def get(self, url: str) -> bytes:
        is_index = url == INDEX_URL
        for _ in range(6):
            if self.requested:
                time.sleep(self.delay)
            self.requested = True
            with self.session.get(url, timeout=self.timeout, allow_redirects=False) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    target = urljoin(url, response.headers.get("Location", ""))
                    if is_index:
                        if target.rstrip("/") != INDEX_URL:
                            raise ValueError(f"Refusing unrelated index redirect: {target}")
                        url = target
                    else:
                        normalized = normalize_url(target)
                        if normalized is None:
                            raise ValueError(f"Refusing unrelated detail redirect: {target}")
                        url = normalized
                    continue
                response.raise_for_status()
                return response.content
        raise ValueError("Too many redirects")


def crawl(output: Path = DEFAULT_OUTPUT, *, force: bool = False, limit: int | None = None,
          delay: float = 1.0, timeout: float = 30.0) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if delay < 0 or timeout <= 0:
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
            urls, duplicates = discover_urls(fetcher.get(INDEX_URL))
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
                text = f"WHO FACT SHEET\nTitle: {title}\nURL: {url}\nRetrieved: {datetime.now(timezone.utc).isoformat()}\n\n{'=' * 40}\n\n{body}\n"
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
    parser.add_argument("--force", action="store_true", help="Refresh existing fact sheets")
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

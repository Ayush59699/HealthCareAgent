"""Deterministic local WHO index lookup; standard library only, no runtime integration.

    python -m rag.who_lookup "asthma and diabetes" --max-results 4

Only index titles/topics are searched. Text is opened after a reliable literal
match; this utility neither infers conditions nor supplies evidence to any agent.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MAX_INDEX_BYTES = 5 * 1024 * 1024
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MAX_RESULTS = 20
MAX_QUERY_CHARS = 4096
COLLECTIONS = (
    ("who_fact_sheets", "fact_sheet", "/news-room/fact-sheets/detail/", "WHO FACT SHEET"),
    ("who_questions_answers", "question_answer", "/news-room/questions-and-answers/item/", "WHO QUESTIONS AND ANSWERS"),
)
# Grammatical/request words and broad medical/demographic qualifiers are not
# sufficient topic evidence on their own. No synonyms, stemming or fuzzy spelling.
STOP_WORDS = frozenset("""
    a an the and or of for to in on at by with without from about as is are was were
    be been being it its this that these those i me my we our you your they their
    what which who how why when where can could should would do does did have has
    had tell show give find please information explain learn know more all any
    health healthy disease diseases disorder disorders condition conditions
    medical medicine clinical patient patients topic topics question questions
    answer answers fact facts sheet sheets symptoms symptom treatment treatments
    prevention prevent causes cause risk risks factors factor care safety safe
    people person adults adult children child adolescent adolescents young older
    women woman men man human common general related problems problem effects
    infection infections virus viruses pain
""".split())


def tokens(text: str) -> tuple[str, ...]:
    """Whole Unicode words, accent/case insensitive; punctuation separates words."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return tuple(re.findall(r"[^\W_]+", folded, flags=re.UNICODE))


def informative(words) -> frozenset[str]:
    return frozenset(w for w in words if w not in STOP_WORDS and not w.isdecimal())


def contains_phrase(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    return bool(phrase) and any(words[i:i + len(phrase)] == phrase for i in range(len(words) - len(phrase) + 1))


def bounded_read(path: Path, maximum: int) -> str:
    with path.open("rb") as handle:
        content = handle.read(maximum + 1)
    if len(content) > maximum:
        raise ValueError(f"File exceeds {maximum} byte safety limit (not truncated)")
    return content.decode("utf-8")


@dataclass(frozen=True)
class Source:
    topic: str
    title: str
    url: str
    document_type: str
    filename: str


@dataclass(frozen=True)
class DocumentMatch:
    source: Source
    match_type: str
    matched_terms: tuple[str, ...]
    text: str | None = None  # search() is metadata-only; lookup() supplies full text.


@dataclass(frozen=True)
class LookupResult:
    query: str
    status: str  # MATCH or NO_MATCH
    documents: tuple[DocumentMatch, ...]
    reason: str
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Entry:
    source: Source
    directory: Path
    header: str
    phrases: tuple[tuple[str, ...], ...]
    terms: frozenset[str]


class WHOLookup:
    """Load both indexes once; reuse for queries without caching document text.

    Exact title/topic > full topic phrase in query > conservative partial match.
    Partial matching requires >=2 informative terms covering >=half the indexed
    terms, or one informative term occurring in at most two indexed documents
    (or explicitly indexed as a standalone topic). These are literal lookup
    rules, not medical confidence scores. Missing/bad data is exposed in errors.
    """

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir).resolve()
        entries = []
        errors = []
        for folder, kind, prefix, header in COLLECTIONS:
            directory = self.data_dir / folder
            index_path = directory / "index.json"
            try:
                if not index_path.resolve().is_relative_to(self.data_dir):
                    raise ValueError("Index resolves outside the configured data directory")
                records = json.loads(bounded_read(index_path, MAX_INDEX_BYTES))
                if not isinstance(records, list):
                    raise ValueError("Index must be a JSON list")
            except (OSError, ValueError) as exc:
                errors.append(f"{folder}/index.json: {exc}")
                continue
            seen = set()
            for number, record in enumerate(records):
                try:
                    if not isinstance(record, dict):
                        raise ValueError("Entry must be an object")
                    if not all(isinstance(record.get(k), str) and record[k].strip() for k in ("topic", "title", "url", "file")):
                        raise ValueError("Entry requires nonempty topic, title, url and file strings")
                    filename = record["file"]
                    if "/" in filename or "\\" in filename or ":" in filename or not filename.endswith(".txt"):
                        raise ValueError("File must be a local .txt basename")
                    if not (directory / filename).resolve().is_relative_to(directory.resolve()):
                        raise ValueError("File resolves outside its collection")
                    url = urlsplit(record["url"])
                    if (url.scheme != "https" or url.netloc != "www.who.int" or url.query or url.fragment
                            or unquote(url.path) != prefix + record["topic"]
                            or "/" in record["topic"] or "\\" in record["topic"]):
                        raise ValueError("URL does not match the WHO collection and indexed topic")
                    identity = (record["url"], filename)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    phrases = (tokens(record["topic"]), tokens(record["title"]))
                    source = Source(record["topic"], record["title"], record["url"], kind, filename)
                    entries.append(_Entry(source, directory, header, phrases, informative(phrases[0] + phrases[1])))
                except (OSError, ValueError) as exc:
                    errors.append(f"{folder}/index.json entry {number}: {exc}")
        self._entries = tuple(entries)
        self.index_errors = tuple(errors)
        # Inverted *keyword* map in RAM; no corpus text, embeddings or disk index.
        self._by_term: dict[str, set[int]] = {}
        self._standalone = set()
        for position, entry in enumerate(self._entries):
            for term in entry.terms:
                self._by_term.setdefault(term, set()).add(position)
            for phrase in entry.phrases:
                if len(phrase) == 1 and phrase[0] in entry.terms:
                    self._standalone.add(phrase[0])

    @property
    def document_count(self) -> int:
        return len(self._entries)

    def _candidates(self, query: str, max_results: int):
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query exceeds {MAX_QUERY_CHARS} characters")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= MAX_RESULTS:
            raise ValueError(f"max_results must be an integer between 1 and {MAX_RESULTS}")
        words = tokens(query)
        query_terms = informative(words)
        positions: set[int] = set()
        for term in query_terms:
            positions.update(self._by_term.get(term, ()))
        candidates = []
        for position in positions:
            entry = self._entries[position]
            overlap = query_terms & entry.terms
            coverage = len(overlap) / len(entry.terms)
            if words in entry.phrases:
                kind, strength = "exact", 3
            elif any(contains_phrase(words, phrase) for phrase in entry.phrases):
                kind, strength = "topic_phrase", 2
            elif (len(overlap) >= 2 and coverage >= 0.5) or (
                len(overlap) == 1 and any(term in self._standalone or len(self._by_term[term]) <= 2 for term in overlap)
            ):
                kind, strength = "partial", 1
            else:
                continue
            match = DocumentMatch(entry.source, kind, tuple(sorted(overlap)))
            # One lexical ordering only, with explicit stable tie breaks; no reranker.
            order = (-strength, -len(overlap), -coverage, entry.source.topic.casefold(),
                     entry.source.document_type, entry.source.filename, entry.source.url)
            candidates.append((order, entry, match))
        return sorted(candidates, key=lambda candidate: candidate[0])

    def search(self, query: str, *, max_results: int = 6) -> LookupResult:
        """Match metadata only; does not open any .txt files."""
        documents = tuple(match for _, _, match in self._candidates(query, max_results)[:max_results])
        return LookupResult(query, "MATCH" if documents else "NO_MATCH", documents,
                            "Literal index matches; document text not loaded." if documents else "No reliable keyword/topic match in the available indexes.",
                            self.index_errors)

    def lookup(self, query: str, *, max_results: int = 6) -> LookupResult:
        """Return complete UTF-8 texts for matched documents, at most max_results.

        Missing, oversized, corrupt or provenance-mismatched files are reported
        and skipped. No truncation, downloads, topic guesses or symptom inference.
        """
        candidates = self._candidates(query, max_results)
        documents = []
        errors = list(self.index_errors)
        for _, entry, match in candidates:
            try:
                path = entry.directory / entry.source.filename
                resolved = path.resolve()
                if not resolved.is_relative_to(entry.directory.resolve()) or not resolved.is_relative_to(self.data_dir):
                    raise ValueError("File resolves outside its collection/data directory")
                text = bounded_read(path, MAX_DOCUMENT_BYTES)
                lines = text.split("\n", 4)
                if (len(lines) < 5 or lines[0].rstrip("\r") != entry.header
                        or lines[1].rstrip("\r") != "Title: " + entry.source.title
                        or lines[2].rstrip("\r") != "URL: " + entry.source.url
                        or not lines[3].startswith("Retrieved: ") or not lines[4].strip("\r\n =")):
                    raise ValueError("Document metadata/body does not match its index entry")
                documents.append(replace(match, text=text))
            except (OSError, ValueError) as exc:
                errors.append(f"{entry.source.document_type}/{entry.source.filename}: {exc}")
                continue
            if len(documents) == max_results:
                break
        reason = "Full WHO source text loaded for literal index matches."
        if not documents:
            reason = "Matching index entries could not be loaded." if candidates else "No reliable keyword/topic match in the available indexes."
        return LookupResult(query, "MATCH" if documents else "NO_MATCH", tuple(documents), reason, tuple(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Literal medical topic(s), not a request for diagnosis")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-results", type=int, default=6)
    parser.add_argument("--metadata-only", action="store_true", help="Match indexes without opening text files")
    args = parser.parse_args()
    try:
        lookup = WHOLookup(args.data_dir)
        result = (lookup.search if args.metadata_only else lookup.lookup)(args.query, max_results=args.max_results)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    # ASCII escapes make the demo portable to legacy Windows terminal encodings.
    print(json.dumps(asdict(result), ensure_ascii=True, indent=2))
    return 0 if result.status == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Opt-in WHO excerpt provenance; structural integrity, never medical entailment.

Lookup validates source headers against its index. This adapter retains a source
checksum and exact character span, not a generated summary or confidence score.
Snapshot revalidation is self-consistency, not re-authentication of a WHO website.
"""
import hashlib
from typing import Literal
from urllib.parse import unquote, urlsplit
from pydantic import model_validator
from rag.agents.models import StrictModel

BACKEND = 'who-local-lookup-v1'
MAX_EXCERPT_CHARS = 4000


def sha256(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


class WHOPayload(StrictModel):
    backend: Literal['who-local-lookup-v1']
    chunk_id: str
    source: Literal['who']
    title: str
    url: str
    document_type: Literal['fact_sheet', 'question_answer']
    topic: str
    filename: str
    retrieved: str
    match_type: Literal['exact', 'topic_phrase', 'partial']
    matched_terms: list[str]
    document_sha256: str
    document_characters: int
    excerpt_sha256: str
    char_start: int
    char_end: int
    text: str
    selection: Literal['literal_lookup_body_prefix']
    score_note: Literal['Not a similarity or confidence score; deterministic literal lookup.']

    @model_validator(mode='after')
    def integrity(self):
        import re
        prefix = ('/news-room/fact-sheets/detail/' if self.document_type == 'fact_sheet'
                  else '/news-room/questions-and-answers/item/')
        url = urlsplit(self.url)
        if (url.scheme != 'https' or url.netloc != 'www.who.int' or url.query or url.fragment
                or unquote(url.path) != prefix + self.topic or not self.topic
                or any(c in self.topic for c in '/\\') or not self.title.strip()):
            raise ValueError('Invalid WHO source identity')
        if (not self.filename.endswith('.txt') or any(c in self.filename for c in '/\\:')
                or not self.retrieved.strip() or not self.matched_terms
                or not all(t.strip() for t in self.matched_terms)):
            raise ValueError('Invalid WHO lookup metadata')
        if (not re.fullmatch('[a-f0-9]{64}', self.document_sha256)
                or self.excerpt_sha256 != sha256(self.text)
                or not 0 <= self.char_start < self.char_end <= self.document_characters
                or self.char_end - self.char_start != len(self.text)
                or not self.text.strip() or len(self.text) > MAX_EXCERPT_CHARS):
            raise ValueError('Invalid WHO excerpt integrity')
        expected = f'who:{self.document_type}:{self.topic}:{self.document_sha256}:{self.char_start}-{self.char_end}'
        if self.chunk_id != expected:
            raise ValueError('Invalid WHO excerpt ID')
        return self


def validate_who_hit(payload):
    if payload.get('selection') == 'llm_catalog_verified_sections':
        from rag.who_sections import WHOSectionPayload
        return WHOSectionPayload.model_validate(payload).model_dump()
    return WHOPayload.model_validate(payload).model_dump()


def excerpt_hit(match):
    """One exact bounded prefix of a lookup-selected body; no search/reranking."""
    text, source = match.text, match.source
    if not isinstance(text, str):
        raise ValueError('WHO lookup must supply source text')
    lines = text.split('\n', 4)
    header = 'WHO FACT SHEET' if source.document_type == 'fact_sheet' else 'WHO QUESTIONS AND ANSWERS'
    if (len(lines) < 5 or lines[0].rstrip('\r') != header
            or lines[1].rstrip('\r') != 'Title: ' + source.title
            or lines[2].rstrip('\r') != 'URL: ' + source.url
            or not lines[3].startswith('Retrieved: ')):
        raise ValueError('WHO source header mismatch')
    start = sum(len(line) + 1 for line in lines[:4])
    # Omit only header separators. Every supplied character retains its source offset.
    while start < len(text) and text[start] in '\r\n =':
        start += 1
    end = min(len(text), start + MAX_EXCERPT_CHARS)
    if end < len(text):
        # Prefer a complete line, without adding ellipses or generated medical text.
        boundary = text.rfind('\n', start, end)
        if boundary > start:
            end = boundary
    excerpt = text[start:end]
    digest = sha256(text)
    payload = dict(backend=BACKEND, source='who', title=source.title, url=source.url,
        document_type=source.document_type, topic=source.topic, filename=source.filename,
        retrieved=lines[3].rstrip('\r').removeprefix('Retrieved: '),
        match_type=match.match_type, matched_terms=list(match.matched_terms),
        document_sha256=digest, document_characters=len(text), excerpt_sha256=sha256(excerpt),
        char_start=start, char_end=end, text=excerpt, selection='literal_lookup_body_prefix',
        score_note='Not a similarity or confidence score; deterministic literal lookup.',
        chunk_id=f'who:{source.document_type}:{source.topic}:{digest}:{start}-{end}')
    return {**validate_who_hit(payload), 'score': 0.0}

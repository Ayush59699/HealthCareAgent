"""Exact source sections anchored by unchanged CHECK4 content-review quotes.

No topic/keyword retrieval or medical synthesis. The historical WHO backend ID
is retained ONLY to route through the unchanged Grounding validator. Selection
metadata explicitly identifies LLM selection, never fabricated lexical matches.
"""
import re
from dataclasses import asdict
from typing import Literal
from pydantic import Field, model_validator
from rag.agents.models import StrictModel
from rag.who_catalog_selector import VERSION, digest
from rag.who_selector_relevance import ContentRelevance, validate_content_review
from rag.who_lookup import Source
from rag.who_provenance import BACKEND, sha256

MAX_DOCUMENT_EXCERPT_CHARS = 12000


class WHOSectionPayload(StrictModel):
    backend: Literal['who-local-lookup-v1']
    source: Literal['who']
    chunk_id: str
    document_id: str
    title: str
    url: str
    document_type: Literal['fact_sheet', 'question_answer']
    topic: str
    filename: str
    retrieved: str
    catalog_sha256: str
    document_sha256: str
    document_characters: int
    excerpt_sha256: str
    char_start: int
    char_end: int
    text: str
    match_type: Literal['llm_catalog']
    matched_terms: list[str] = Field(max_length=0)
    selection: Literal['llm_catalog_verified_sections']
    selection_method: Literal['who-catalog-selector-v2']
    score_note: Literal['Not similarity or confidence; CHECK4 LLM selection and content review.']
    positive_fact_quotes: list[str] = Field(min_length=1, max_length=4)
    content_quotes: list[str] = Field(min_length=1, max_length=3)
    relevance_reason: Literal['direct_presenting_problem', 'distinctive_positive_cluster']

    @model_validator(mode='after')
    def integrity(self):
        from urllib.parse import urlsplit, unquote
        url = urlsplit(self.url)
        prefix = ('/news-room/fact-sheets/detail/' if self.document_type == 'fact_sheet'
                  else '/news-room/questions-and-answers/item/')
        if (url.scheme != 'https' or url.netloc != 'www.who.int' or url.query or url.fragment
                or unquote(url.path) != prefix + self.topic or not self.topic
                or any(c in self.topic for c in '/\\') or not self.title.strip()
                or not self.filename.endswith('.txt') or any(c in self.filename for c in '/\\:')
                or not self.retrieved.strip()):
            raise ValueError('Invalid WHO source identity')
        source = Source(self.topic, self.title, self.url, self.document_type, self.filename)
        if self.document_id != 'w:' + digest(asdict(source))[:12]:
            raise ValueError('WHO catalog document ID mismatch')
        if (any(not re.fullmatch('[a-f0-9]{64}', h) for h in (self.catalog_sha256, self.document_sha256))
                or self.excerpt_sha256 != sha256(self.text)
                or not 0 <= self.char_start < self.char_end <= self.document_characters
                or self.char_end - self.char_start != len(self.text)
                or not self.text.strip() or len(self.text) > MAX_DOCUMENT_EXCERPT_CHARS):
            raise ValueError('Invalid WHO section integrity')
        expected = f'who:{self.document_type}:{self.topic}:{self.document_sha256}:{self.char_start}-{self.char_end}'
        if self.chunk_id != expected:
            raise ValueError('Invalid WHO section ID')
        ContentRelevance(verdict='relevant', reason=self.relevance_reason,
                         positive_fact_quotes=self.positive_fact_quotes, content_quotes=self.content_quotes)
        if any(q not in self.text for q in self.content_quotes):
            raise ValueError('WHO section lost its content support')
        return self


def section_spans(text, quotes):
    """Return complete top-level source sections covering every support quote.

    The plain-text WHO renderers underline headings. Prefer '=' top-level
    boundaries; use '-' where no '=' headings exist. This retains e.g. the Dog
    bites parent heading and conditional context around its Treatment subsection.
    Unheaded text uses whole paragraphs. Never clip an over-budget section.
    """
    parts = text.split('\n', 4)
    if len(parts) != 5:
        raise ValueError('Missing WHO body')
    body_start = sum(len(p) + 1 for p in parts[:4])
    headings = list(re.finditer(r'(?m)^([^\n]+)\n(={2,}|-{2,})\r?$', text[body_start:]))
    primary = [h for h in headings if h.group(2).startswith('=')] or headings
    boundaries = [body_start] + [body_start + h.start() for h in primary] + [len(text)]
    boundaries = sorted(set(boundaries))
    spans = set()
    for quote in quotes:
        positions, offset = [], body_start
        while (position := text.find(quote, offset)) >= 0:
            positions.append(position)
            offset = position + len(quote)
        # Repeated quote occurrences could describe different contexts. Do not
        # silently choose a convenient match: ambiguous anchors fail closed.
        if len(positions) != 1:
            raise ValueError('Missing or ambiguous WHO quote anchor')
        position = positions[0]
        if primary:
            start = max(b for b in boundaries if b <= position)
            end = next(b for b in boundaries if b >= position + len(quote))
        else:
            start = max(body_start, text.rfind('\n\n', body_start, position) + 2)
            boundary = text.find('\n\n', position + len(quote))
            end = len(text) if boundary == -1 else boundary
        spans.add((start, end))
    ordered = sorted(spans)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    if sum(end - start for start, end in merged) > MAX_DOCUMENT_EXCERPT_CHARS:
        raise ValueError('Relevant WHO sections exceed bounded handoff; no truncation')
    return merged


def section_hits(document, review, patient):
    review = validate_content_review(review, patient, document)
    if review.verdict != 'relevant':
        raise ValueError('Rejected WHO document cannot become evidence')
    if sha256(document.text) != document.document_sha256:
        raise ValueError('WHO full text changed after selection')
    hits = []
    for start, end in section_spans(document.text, review.content_quotes):
        excerpt = document.text[start:end]
        source = document.source
        payload = dict(backend=BACKEND, source='who', **asdict(source),
            chunk_id=f'who:{source.document_type}:{source.topic}:{document.document_sha256}:{start}-{end}',
            document_id=document.document_id, catalog_sha256=document.catalog_sha256,
            document_sha256=document.document_sha256, document_characters=len(document.text),
            retrieved=document.retrieved, char_start=start, char_end=end, text=excerpt,
            excerpt_sha256=sha256(excerpt), match_type='llm_catalog', matched_terms=[],
            selection='llm_catalog_verified_sections', selection_method=VERSION,
            score_note='Not similarity or confidence; CHECK4 LLM selection and content review.',
            positive_fact_quotes=review.positive_fact_quotes,
            content_quotes=[q for q in review.content_quotes if q in excerpt], relevance_reason=review.reason)
        hits.append({**WHOSectionPayload.model_validate(payload).model_dump(), 'score': 0.0})
    return hits

"""Separate, hash-verified exact-cosine index over the existing licensed corpus.

At this corpus size an immutable NumPy matrix is sufficient; neither production
Qdrant collection is opened. Retrieval windows point to complete source sections.
Source bytes are reconstructed/verified before embedding. No generated summaries.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
from rag.medical_ingestion.common import xml_root
from rag.medical_ingestion.corpus import load_documents
from rag.medical_ingestion.medlineplus import SummarySections
from rag.medical_ingestion.models import MedicalDocument, clean_text

VERSION = 'task73-context-sections-v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf8')


class BlockSections(SummarySections):
    """Reuse the source HTML parser, but retain paragraph/list boundaries."""
    def flush(self):
        text = '\n'.join(clean_text(line) for line in ''.join(self.parts).splitlines() if clean_text(line))
        if text:
            self.sections.append({'title': self.title, 'text': text})
        self.parts = []


def source_sections(manifest, documents):
    """Recover formatting only, verifying against the approved parser's content."""
    result = {}
    config = json.loads(Path(manifest).read_text(encoding='utf8'))
    for entry in config['entries']:
        if entry['source'] != 'medlineplus' or entry.get('selection') != 'all-english-summaries':
            continue
        path = Path(manifest).parent / entry['raw_path']
        if digest(path.read_bytes()) != entry['sha256']:
            raise ValueError('Source changed during context reconstruction')
        for topic in xml_root(path).iter('health-topic'):
            if topic.get('language') not in ('English', 'en'):
                continue
            summary = topic.find('full-summary')
            if summary is None:
                continue
            value = (summary.text or '') + ''.join(ET.tostring(c, encoding='unicode') for c in summary)
            parser = BlockSections()
            parser.feed(value)
            parser.close()
            parser.flush()
            result['medlineplus:' + topic.attrib['id']] = parser.sections
    for doc in documents:
        sections = result.get(doc.document_id, doc.metadata.get('sections') or [{'title': doc.section, 'text': doc.text}])
        reconstructed = ' '.join((s['title'] + ' ' if s['title'] else '') + s['text'] for s in sections)
        if clean_text(reconstructed) != clean_text(doc.text):
            raise ValueError('Context reconstruction changed source content: ' + doc.document_id)
        yield doc, sections


def retrieval_windows(text, header, count_tokens, max_tokens=480):
    """Pack complete sentences/list items. Never cut a sentence to hide truncation.

    Full parent text is retained independently, so list lead-ins and preceding
    definitions remain available even when a retrieval window starts mid-list.
    Oversize atomic units fail clearly rather than silently dropping source text.
    """
    units = []
    for block in text.splitlines():
        units.extend(s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', block) if s.strip())
    pending = []
    for unit in units:
        if count_tokens(header + '\n' + unit) > max_tokens:
            raise ValueError('Source sentence/list item exceeds embedding budget; inspect source, no truncation allowed')
        if pending and count_tokens(header + '\n' + '\n'.join(pending + [unit])) > max_tokens:
            yield '\n'.join(pending)
            pending = []
        pending.append(unit)
    if pending:
        yield '\n'.join(pending)


def make_records(documents_and_sections, count_tokens, max_tokens=480):
    records = []
    for doc, sections in documents_and_sections:
        source = asdict(doc)
        source.pop('text')
        source['metadata'] = {k: v for k, v in source['metadata'].items() if k != 'sections'}
        for ordinal, section in enumerate(sections):
            parent = {'source': source, 'section_ordinal': ordinal, 'section': section['title'], 'text': section['text']}
            parent_id = digest(canonical({'document_id': doc.document_id, 'ordinal': ordinal, 'section': section, 'version': VERSION}))
            header = doc.title + ('\n' + section['title'] if section['title'] else '')
            for window_index, window in enumerate(retrieval_windows(section['text'], header, count_tokens, max_tokens)):
                record = {**parent, 'parent_id': parent_id, 'window_index': window_index,
                          'retrieval_text': header + '\n' + window, 'window_text': window}
                record['chunk_id'] = digest(canonical({'parent_id': parent_id, 'window': window, 'version': VERSION}))
                records.append(record)
    if not records or len({r['chunk_id'] for r in records}) != len(records):
        raise ValueError('Empty or duplicate experimental records')
    return records


def build_index(manifest, destination, embedding, *, max_tokens=480):
    destination, manifest = Path(destination).resolve(), Path(manifest).resolve()
    # Never write anywhere inside either live index, or into the corpus itself.
    if destination.exists() or destination == manifest.parent or destination in manifest.parents:
        raise ValueError('Choose a NEW experimental index directory')
    if 'qdrant' in str(destination).lower():
        raise ValueError('Experimental flat index must not be placed in a Qdrant path')
    if not 32 <= max_tokens <= embedding.model.max_seq_length:
        raise ValueError('Invalid token budget')
    start = perf_counter()
    manifest_bytes = manifest.read_bytes()
    documents = load_documents(manifest)
    count = lambda text: len(embedding.model.tokenizer.encode(text, add_special_tokens=True))
    records = make_records(source_sections(manifest, documents), count, max_tokens)
    before = embedding.truncated_texts
    vectors = np.asarray(embedding.embed_texts(r['retrieval_text'] for r in records), dtype=np.float32)
    if embedding.truncated_texts != before:
        raise ValueError('Experimental build refuses truncated embeddings')
    if manifest.read_bytes() != manifest_bytes:
        raise ValueError('Corpus manifest changed during build')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.task73-', dir=destination.parent))
    try:
        content = b''.join(canonical(r) + b'\n' for r in records)
        (temporary / 'passages.jsonl').write_bytes(content)
        np.save(temporary / 'vectors.npy', vectors, allow_pickle=False)
        report = {'version': VERSION, 'embedding': embedding.signature,
                  'representation': 'source-topic + source-heading + sentence/list window; full parent evidence',
                  'max_tokens': max_tokens, 'documents': len(documents), 'windows': len(records),
                  'parent_sections': len({r['parent_id'] for r in records}),
                  'manifest_sha256': digest(manifest_bytes), 'passages_sha256': digest(content),
                  'vectors_sha256': digest((temporary / 'vectors.npy').read_bytes()),
                  'truncated_texts': 0, 'build_seconds': perf_counter() - start}
        (temporary / 'index.json').write_bytes(canonical(report))
        ExperimentalIndex(temporary, embedding.signature)  # validate before publication
        if destination.exists():
            raise ValueError('Destination appeared during build')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return report


class ExperimentalIndex:
    def __init__(self, path, embedding_signature):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'index.json').read_text(encoding='utf8'))
        if self.manifest['version'] != VERSION or self.manifest['embedding'] != embedding_signature:
            raise ValueError('Experimental index representation/embedding mismatch')
        content = (self.path / 'passages.jsonl').read_bytes()
        if digest(content) != self.manifest['passages_sha256'] or digest((self.path / 'vectors.npy').read_bytes()) != self.manifest['vectors_sha256']:
            raise ValueError('Experimental index integrity failure')
        self.records = [json.loads(line) for line in content.splitlines()]
        self.vectors = np.load(self.path / 'vectors.npy', allow_pickle=False)
        if self.vectors.shape != (len(self.records), embedding_signature['dimension']) or not np.isfinite(self.vectors).all():
            raise ValueError('Invalid experimental vectors')
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-4):
            raise ValueError('Experimental vectors must be normalized')
        if len(self.records) != self.manifest['windows'] or len({r['chunk_id'] for r in self.records}) != len(self.records):
            raise ValueError('Experimental record count/identity mismatch')
        for r in self.records:
            MedicalDocument(**{**r['source'], 'text': r['text']})
            if not r['window_text'].strip() or clean_text(r['window_text']) not in clean_text(r['text']):
                raise ValueError('Retrieval window is not a source span')
            parent_id = digest(canonical({'document_id': r['source']['document_id'], 'ordinal': r['section_ordinal'],
                'section': {'title': r['section'], 'text': r['text']}, 'version': VERSION}))
            expected = digest(canonical({'parent_id': parent_id, 'window': r['window_text'], 'version': VERSION}))
            header = r['source']['title'] + ('\n' + r['section'] if r['section'] else '')
            if r['parent_id'] != parent_id or r['chunk_id'] != expected or r['retrieval_text'] != header + '\n' + r['window_text']:
                raise ValueError('Altered experimental source context')

    def search(self, vector, depth):
        if type(depth) is not int or depth < 1:
            raise ValueError('Positive candidate depth required')
        vector = np.asarray(vector)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError('Invalid query vector')
        scores = self.vectors @ (vector / np.linalg.norm(vector))
        return [(int(i), float(scores[i])) for i in sorted(range(len(scores)), key=lambda i: (-scores[i], self.records[i]['chunk_id']))[:depth]]

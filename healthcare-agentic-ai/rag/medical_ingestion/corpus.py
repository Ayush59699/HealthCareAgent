"""Pinned corpus manifest loading, reconstruction and deterministic serialization."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from .common import verified_provenance
from .medlineplus import parse_medlineplus
from .pmc import parse_pmc
from .who import parse_who
from .chunker import chunk_document


def load_documents(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    documents = []
    for entry in manifest['entries']:
        path = manifest_path.parent / entry['raw_path']
        provenance = verified_provenance(path)
        if provenance['raw_sha256'] != entry['sha256']:
            raise ValueError('Manifest raw hash mismatch')
        if entry['source'] == 'medlineplus':
            parsed = list(parse_medlineplus(path, provenance, entry['titles']))
            if {d.title for d in parsed} != set(entry['titles']):
                raise ValueError('Selected MedlinePlus topics missing')
        elif entry['source'] == 'pmc':
            parsed = [parse_pmc(path, provenance)]
            if parsed[0].pmcid != entry['pmcid']:
                raise ValueError('PMC identifier mismatch')
        elif entry['source'] == 'who':
            parsed = [parse_who(path, provenance, entry['selection'])]
        else:
            raise ValueError('Unsupported source (patient records are forbidden)')
        documents.extend(parsed)
    documents.sort(key=lambda d: d.document_id)
    if len({d.document_id for d in documents}) != len(documents):
        raise ValueError('Duplicate source document IDs')
    return documents


def prepare_corpus(manifest_path, max_words=180, overlap=30):
    documents = load_documents(manifest_path)
    chunks = [c for d in documents for c in chunk_document(d, max_words, overlap)]
    if not chunks:
        raise ValueError('No licensed documents to index')
    serialized = ''.join(json.dumps(asdict(c), sort_keys=True, ensure_ascii=False) + '\n' for c in chunks)
    return documents, chunks, serialized


def source_counts(documents, chunks):
    docs = Counter(d.source for d in documents)
    counts = Counter(c.source for c in chunks)
    return {s: {'documents': docs[s], 'chunks': counts[s]} for s in ('medlineplus', 'pmc', 'who')}

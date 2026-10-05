"""Medical-only index construction and exhaustive source reconstruction audit."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .corpus import prepare_corpus, source_counts
from .chunker import chunk_document
from .models import validate_chunk, clean_text
from ..medical_vector_store import MedicalVectorStore, COLLECTION


def fingerprint(chunks, signature):
    """Stable across download locations/times, sensitive to content and source version."""
    records = []
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        payload = asdict(chunk)
        payload['provenance'] = {k: v for k, v in payload['provenance'].items()
                                 if k not in {'retrieved_at', 'raw_path'}}
        records.append(payload)
    return hashlib.sha256(json.dumps({'embedding': signature, 'chunks': records},
        sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()


def duplicate_documents(documents):
    return len(documents) - len({clean_text(d.text).casefold() for d in documents})


def build_index(manifest, storage_path, embedding, *, max_words=180, overlap=30, processed_dir=None):
    documents, chunks, serialized = prepare_corpus(manifest, max_words, overlap)
    expected = {c.chunk_id for c in chunks}
    if len(expected) != len(chunks) or duplicate_documents(documents):
        raise ValueError('Duplicate documents/chunk IDs in source corpus')
    if {c.document_id for c in chunks} != {d.document_id for d in documents}:
        raise ValueError('Chunking silently lost source documents')
    storage_path = Path(storage_path)
    metrics_path = storage_path / 'medical-index.json'
    failures, embedded = 0, 0
    before_truncations = embedding.truncated_texts
    with MedicalVectorStore(storage_path, embedding.dimension, embedding_signature=embedding.signature) as store:
        if {str(p.id) for p in store.iter_points()} - expected:
            raise ValueError('Stale chunks: use ingest_medlineplus.py for a staged replacement, not append')
        try:
            for start in range(0, len(chunks), embedding.batch_size):
                batch = chunks[start:start + embedding.batch_size]
                vectors = embedding.embed_texts([c.text for c in batch])
                if len(vectors) != len(batch):
                    failures += len(batch) - min(len(batch), len(vectors))
                    raise ValueError('Embedding output count differs from input; refusing partial corpus')
                inserted = store.upsert_chunks(batch, vectors)
                if inserted != len(batch):
                    raise ValueError('Upsert silently lost chunks')
                embedded += inserted
        except Exception:
            metrics_path.write_text(json.dumps({'passed': False, 'embedding_failures': max(1, failures),
                'embedded_chunks': embedded, 'expected_chunks': len(chunks)}, indent=2), encoding='utf8')
            raise
        if store.count() != len(chunks):
            raise ValueError('Stored count differs from expected corpus')
    report = {'passed': True, 'collection': COLLECTION, 'embedding': embedding.signature,
              'embedding_dimension': embedding.dimension, 'distance_metric': 'Cosine',
              'total_documents': len(documents), 'total_chunks': len(chunks), 'indexed_chunks': embedded,
              'sources': source_counts(documents, chunks), 'embedding_failures': 0,
              'duplicate_passages_suppressed': sum(len(list(chunk_document(d, max_words, overlap))) for d in documents) - len(chunks),
              'deduplication': 'exact-global-text-v1, first deterministic document ID retains attribution',
              'truncated_texts': embedding.truncated_texts - before_truncations,
              'max_words': max_words, 'overlap': overlap,
              'chunking': 'sections-v2 for full MedlinePlus; words-v1 for PMC/legacy selections; 40-word short-section merge',
              'index_fingerprint': fingerprint(chunks, embedding.signature),
              'processed_sha256': hashlib.sha256(serialized.encode('utf8')).hexdigest()}
    metrics_path.write_text(json.dumps(report, indent=2), encoding='utf8')
    if processed_dir is not None:
        processed = Path(processed_dir)
        processed.mkdir(parents=True, exist_ok=True)
        (processed / 'chunks.jsonl').write_bytes(serialized.encode('utf8'))
        (processed / 'documents.jsonl').write_bytes(''.join(json.dumps(asdict(d), sort_keys=True, ensure_ascii=False) + '\n'
            for d in documents).encode('utf8'))
    return report


def audit_index(manifest, storage_path, *, max_words=180, overlap=30):
    report = {'passed': False, 'collection': COLLECTION, 'total_documents': 0, 'total_chunks': 0,
              'duplicate_ids': 0, 'duplicate_documents': 0, 'duplicate_chunks': 0,
              'empty_chunks': 0, 'malformed_metadata': 0, 'missing_provenance': 0,
              'embedding_failures': None, 'truncations': None, 'errors': []}
    try:
        documents, chunks, serialized = prepare_corpus(manifest, max_words, overlap)
        expected = {c.chunk_id: asdict(c) for c in chunks}
        report['duplicate_documents'] = duplicate_documents(documents)
        report['duplicate_ids'] = len(chunks) - len(expected)
        report['duplicate_chunks'] = len(chunks) - len({c.text for c in chunks})
        signature = json.loads((Path(storage_path) / 'medical-embedding.json').read_text(encoding='utf8'))
        expected_signature = {'model_name': 'BAAI/bge-small-en-v1.5', 'dimension': 384,
                              'normalize_embeddings': True, 'max_seq_length': 512,
                              'text_format': 'medical-source-text-v1', 'prompt': '',
                              'pipeline': 'medical-knowledge-v1'}
        if signature != expected_signature:
            raise ValueError('Unexpected medical embedding configuration')
        report.update(embedding_dimension=signature['dimension'], distance_metric='Cosine',
                      index_fingerprint=fingerprint(chunks, signature))
        metrics = json.loads((Path(storage_path) / 'medical-index.json').read_text(encoding='utf8'))
        report.update(embedding_failures=metrics.get('embedding_failures'), truncations=metrics.get('truncated_texts'),
                      duplicate_passages_suppressed=metrics.get('duplicate_passages_suppressed'))
        if (metrics.get('passed') is not True or metrics.get('index_fingerprint') != report['index_fingerprint']
                or metrics.get('indexed_chunks') != len(chunks) or metrics.get('embedding') != signature
                or metrics.get('processed_sha256') != hashlib.sha256(serialized.encode('utf8')).hexdigest()
                or metrics.get('max_words') != max_words or metrics.get('overlap') != overlap
                or metrics.get('duplicate_passages_suppressed') != sum(len(list(chunk_document(d, max_words, overlap))) for d in documents) - len(chunks)):
            report['errors'].append('Build metrics do not match reconstructed corpus/configuration')
        if report['embedding_failures'] != 0 or type(report['truncations']) is not int or not 0 <= report['truncations'] <= len(chunks):
            report['errors'].append('Missing/invalid embedding failure or truncation accounting')
        seen, document_ids, sources, doc_sources = set(), set(), Counter(), {}
        seen_texts = set()
        with MedicalVectorStore(storage_path, signature['dimension'], embedding_signature=signature) as store:
            for point in store.iter_points(with_vectors=True):
                report['total_chunks'] += 1
                key = str(point.id)
                report['duplicate_ids'] += int(key in seen)
                seen.add(key)
                payload = point.payload or {}
                if isinstance(payload.get('text'), str):
                    report['duplicate_chunks'] += int(payload['text'] in seen_texts)
                    seen_texts.add(payload['text'])
                if not isinstance(payload.get('text'), str) or not payload['text'].strip():
                    report['empty_chunks'] += 1
                provenance_fields = {'download_url', 'retrieved_at', 'raw_path', 'raw_sha256', 'license_url', 'license_evidence'}
                provenance = payload.get('provenance')
                missing = not isinstance(provenance, dict) or any(not provenance.get(k) for k in provenance_fields)
                if payload.get('source') == 'medlineplus':
                    missing = missing or any(not payload.get(k) for k in ('document_id', 'chunk_id', 'title', 'url', 'language', 'source_version'))
                report['missing_provenance'] += int(missing)
                try:
                    validate_chunk(payload)
                except (ValueError, TypeError):
                    report['malformed_metadata'] += 1
                if payload != expected.get(key) or payload.get('chunk_id') != key:
                    report['errors'].append('Source reconstruction mismatch: ' + key)
                try:
                    vector = store._vector(point.vector)
                    if abs(sum(v*v for v in vector) - 1) > 1e-4:
                        raise ValueError('Vector not normalized')
                except (ValueError, TypeError):
                    report['errors'].append('Invalid vector: ' + key)
                source, doc = payload.get('source'), payload.get('document_id')
                sources[source] += 1
                document_ids.add(doc)
                doc_sources[doc] = source
        report['total_documents'] = len(document_ids)
        report['sources'] = {s: {'documents': sum(v == s for v in doc_sources.values()), 'chunks': sources[s]}
                             for s in ('medlineplus', 'pmc', 'who')}
        if seen != set(expected) or document_ids != {d.document_id for d in documents}:
            report['errors'].append('Stored IDs/documents differ from source corpus')
        for name in ('duplicate_ids', 'duplicate_documents', 'duplicate_chunks', 'empty_chunks', 'malformed_metadata', 'missing_provenance'):
            if report[name]:
                report['errors'].append(name + ' must be zero')
        report['passed'] = not report['errors']
    except Exception as exc:
        report['errors'].append(str(exc))
    return report

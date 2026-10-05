"""Read-only hybrid adapter for the existing medical_knowledge Qdrant store.

No re-indexing, no patient collection, no alternate embedding signature. Context
is used for BM25/reranking only; dense retrieval uses the original BGE vectors.
The exact original validated chunk is delivered as evidence, not contextual text.
"""
import numpy as np
from rag.medical_ingestion.models import validate_chunk
from rag.medical_vector_store import COLLECTION
from .index import canonical, digest


class QdrantMedicalIndex:
    def __init__(self, medical):
        if medical.store.collection_name != COLLECTION:
            raise ValueError('Only the medical knowledge collection is permitted')
        if medical.store.signature != medical.embedding.signature:
            raise ValueError('Medical embedding signature mismatch')
        self.medical = medical
        self.records, vectors = [], []
        for point in medical.store.iter_points(with_vectors=True):
            chunk = validate_chunk(point.payload)
            header = chunk['title'] + ('\n' + chunk.get('section', '') if chunk.get('section') else '')
            self.records.append({'chunk_id': chunk['chunk_id'], 'parent_id': chunk['chunk_id'],
                'source': {k: v for k, v in chunk.items() if k not in ('text', 'chunk_id', 'chunk_index', 'chunking_version')},
                'section': chunk.get('section', ''), 'text': chunk['text'],
                'evidence_text': chunk['text'], 'window_text': chunk['text'],
                'retrieval_text': header + '\n' + chunk['text'], 'evidence_payload': chunk})
            vectors.append(point.vector)
        self.by_id = {r['chunk_id']: i for i, r in enumerate(self.records)}
        if len(self.by_id) != len(self.records):
            raise ValueError('Duplicate medical chunk identifiers')
        self.vectors = np.asarray(vectors, dtype=np.float32).reshape((-1, medical.embedding.dimension))
        if not np.isfinite(self.vectors).all() or (len(vectors) and not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-4)):
            raise ValueError('Invalid/unnormalized medical vectors')
        self.manifest = {'representation': 'existing Qdrant chunks; source title/heading lexical context',
            'collection': COLLECTION, 'windows': len(self.records), 'embedding': medical.embedding.signature,
            'payloads_sha256': digest(canonical([r['evidence_payload'] for r in self.records]))}

    def search(self, vector, depth):
        hits = self.medical.store.search(vector, top_k=depth)
        result = []
        for hit in hits:
            payload = validate_chunk({k: v for k, v in hit.items() if k != 'score'})
            i = self.by_id.get(payload['chunk_id'])
            if i is None or payload != self.records[i]['evidence_payload']:
                raise ValueError('Medical index changed during hybrid retrieval; reload before retrying')
            result.append((i, hit['score']))
        return result


def evidence_hits(result):
    """Keep ranking diagnostics outside the strict evidence payload allowlist."""
    hits = []
    for row in result['final_passages']:
        record = row['record']
        if 'evidence_payload' not in record:
            raise ValueError('Context-section experiment is preview only; original validated chunks required for snapshots')
        payload = validate_chunk(record['evidence_payload'])
        if payload['text'] != record['text'] or payload['text'] != record['evidence_text']:
            raise ValueError('Altered evidence text')
        # A semantic cosine is retained for the legacy similarity field. Learned
        # logits, RRF and selection features stay in the separate retrieval audit.
        score = row['semantic_score']
        hits.append({**payload, 'score': score})
    return hits

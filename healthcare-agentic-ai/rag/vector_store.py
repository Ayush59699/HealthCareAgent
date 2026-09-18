"""Persistent embedded Qdrant with strict training payload and vector contracts."""
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient, models
from .patient_ingestion import validate_document


class QdrantVectorStore:
    def __init__(self, path, dimension: int, collection_name='ddxplus_patient_cases', *, embedding_signature=None):
        if type(dimension) is not int or dimension < 1:
            raise ValueError('dimension must be dynamically supplied by the embedding model')
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.collection_name = collection_name
        self.signature = embedding_signature or {'dimension': dimension}
        name_hash = hashlib.sha256(collection_name.encode()).hexdigest()[:20]
        self.manifest_path = self.path / f'embedding-{name_hash}.json'
        try:
            self.client = QdrantClient(path=str(self.path))
        except Exception as exc:
            raise RuntimeError(f'Cannot open local Qdrant at {self.path}. Close other processes using this directory: {exc}') from exc
        try:
            if self.collection_exists():
                self._validate_collection()
        except Exception:
            self.close()
            raise

    def _validate_collection(self):
        params = self.client.get_collection(self.collection_name).config.params.vectors
        if not isinstance(params, models.VectorParams) or params.size != self.dimension or params.distance != models.Distance.COSINE:
            raise ValueError('Existing collection vector dimension/distance mismatch; use a different path/collection')
        if not self.manifest_path.exists() or json.loads(self.manifest_path.read_text(encoding='utf8')) != self.signature:
            raise ValueError('Embedding configuration mismatch or missing manifest; do not mix embedding models. Use a new collection/path.')

    def collection_exists(self):
        return self.client.collection_exists(self.collection_name)

    def create_collection(self):
        if self.collection_exists():
            self._validate_collection()
            return
        self.client.create_collection(self.collection_name,
            vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE))
        self.manifest_path.write_text(json.dumps(self.signature, sort_keys=True, indent=2), encoding='utf8')

    def delete_collection(self):
        if self.collection_exists():
            # qdrant-client 1.19 local deletion does not close SQLite before
            # rmtree (which silently fails on Windows). Explicitly close only
            # this local collection before invoking the public deletion API.
            from qdrant_client.local.qdrant_local import QdrantLocal
            local = self.client._client
            if isinstance(local, QdrantLocal):
                local.collections[self.collection_name].close()
            self.client.delete_collection(self.collection_name)
            if (self.path / 'collection' / self.collection_name).exists():
                raise RuntimeError('Qdrant collection deletion left files behind; refusing to reuse stale vectors')
        self.manifest_path.unlink(missing_ok=True)

    @staticmethod
    def point_id(patient_id):
        return str(uuid5(NAMESPACE_URL, 'ddxplus-patient/' + patient_id))

    def _vector(self, vector):
        vector = [float(v) for v in vector]
        if len(vector) != self.dimension or not all(math.isfinite(v) for v in vector) or not any(vector):
            raise ValueError('Vector must be finite, nonzero and match the configured dimension')
        return vector

    def upsert_patients(self, documents, vectors):
        documents, vectors = list(documents), list(vectors)
        if len(documents) != len(vectors):
            raise ValueError('Document/vector counts must match')
        # Validate the entire batch before writing; last occurrence of an ID wins.
        points = {}
        for document, vector in zip(documents, vectors):
            payload = validate_document(document)
            point_id = self.point_id(payload['patient_id'])
            points[point_id] = models.PointStruct(id=point_id, vector=self._vector(vector), payload=payload)
        if points:
            self.create_collection()
            self.client.upsert(self.collection_name, list(points.values()), wait=True)
        return len(points)

    def search(self, vector, top_k=5):
        if type(top_k) is not int or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        vector = self._vector(vector)
        if top_k == 0 or not self.collection_exists():
            return []
        self._validate_collection()
        hits = self.client.query_points(self.collection_name, query=vector, limit=top_k,
            query_filter=models.Filter(must=[models.FieldCondition(key='split', match=models.MatchValue(value='train'))]),
            with_payload=True, with_vectors=False).points
        results = []
        for hit in hits:
            payload = validate_document(hit.payload or {})
            results.append({**payload, 'score': float(hit.score)})
        return results

    def count(self):
        return self.client.count(self.collection_name, exact=True).count if self.collection_exists() else 0

    def iter_payloads(self, batch_size=128):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not self.collection_exists():
            return
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection_name, limit=batch_size, offset=offset,
                                                with_payload=True, with_vectors=False)
            for point in points:
                yield dict(point.payload or {})
            if offset is None:
                break

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""Independent persistent Qdrant collection with medical-only payload contracts."""
from dataclasses import asdict
import json
import math
from pathlib import Path
from qdrant_client import QdrantClient, models
from .medical_ingestion.common import DATA_ROOT
from .medical_ingestion.models import MedicalChunk, SOURCES, validate_chunk

COLLECTION = 'medical_knowledge'


class MedicalVectorStore:
    collection_name = COLLECTION

    def __init__(self, path=None, dimension=384, *, embedding_signature=None, collection_name=COLLECTION):
        if collection_name != COLLECTION:
            raise ValueError('Medical retrieval may only use medical_knowledge, never patient collections')
        if type(dimension) is not int or dimension < 1:
            raise ValueError('Invalid dimension')
        self.path = Path(path) if path is not None else DATA_ROOT / 'qdrant'
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.signature = embedding_signature or {'dimension': dimension, 'pipeline': 'medical-knowledge-v1'}
        self.manifest_path = self.path / 'medical-embedding.json'
        self.client = QdrantClient(path=str(self.path))
        try:
            if self.exists():
                self._validate()
        except Exception:
            self.close()
            raise

    def exists(self):
        return self.client.collection_exists(COLLECTION)

    def _validate(self):
        config = self.client.get_collection(COLLECTION).config.params.vectors
        if not isinstance(config, models.VectorParams) or config.size != self.dimension or config.distance != models.Distance.COSINE:
            raise ValueError('Medical collection dimension/distance mismatch')
        if not self.manifest_path.exists() or json.loads(self.manifest_path.read_text(encoding='utf8')) != self.signature:
            raise ValueError('Medical embedding signature mismatch')

    def create_collection(self):
        if self.exists():
            self._validate()
        else:
            self.client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE))
            self.manifest_path.write_text(json.dumps(self.signature, sort_keys=True, indent=2), encoding='utf8')

    def _vector(self, vector):
        values = [float(v) for v in vector]
        if len(values) != self.dimension or not all(math.isfinite(v) for v in values) or not any(values):
            raise ValueError('Invalid medical vector')
        return values

    def upsert_chunks(self, chunks, vectors):
        chunks, vectors = list(chunks), list(vectors)
        if len(chunks) != len(vectors):
            raise ValueError('Chunk/vector counts differ')
        points = {}
        for chunk, vector in zip(chunks, vectors):
            payload = validate_chunk(asdict(chunk) if isinstance(chunk, MedicalChunk) else chunk)
            points[payload['chunk_id']] = models.PointStruct(id=payload['chunk_id'], payload=payload, vector=self._vector(vector))
        if points:
            self.create_collection()
            self.client.upsert(COLLECTION, list(points.values()), wait=True)
        return len(points)

    def count(self):
        return self.client.count(COLLECTION, exact=True).count if self.exists() else 0

    def search(self, vector, top_k=5):
        if type(top_k) is not int or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        vector = self._vector(vector)
        if not top_k or not self.exists():
            return []
        self._validate()
        hits = self.client.query_points(COLLECTION, query=vector, limit=top_k, with_payload=True,
            query_filter=models.Filter(must=[models.FieldCondition(key='source', match=models.MatchAny(any=sorted(SOURCES)))] )).points
        return [{**validate_chunk(h.payload), 'score': float(h.score)} for h in hits]

    def iter_points(self, with_vectors=False):
        if not self.exists():
            return
        self._validate()
        offset = None
        while True:
            points, offset = self.client.scroll(COLLECTION, offset=offset, limit=128, with_payload=True, with_vectors=with_vectors)
            yield from points
            if offset is None:
                break

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

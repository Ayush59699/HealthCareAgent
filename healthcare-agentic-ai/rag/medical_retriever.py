"""Retrieval only: actual medical text, similarity and source attribution."""
from .medical_embeddings import MedicalEmbeddingModel
from .medical_vector_store import MedicalVectorStore, COLLECTION


class MedicalKnowledgeRetriever:
    def __init__(self, embedding_model=None, vector_store=None, *, storage_path=None, allow_download=False):
        self.embedding = embedding_model or MedicalEmbeddingModel(allow_download=allow_download)
        self.store = vector_store or MedicalVectorStore(storage_path, self.embedding.dimension,
                                                       embedding_signature=self.embedding.signature)
        self._owns_store = vector_store is None
        if self.store.collection_name != COLLECTION:
            if self._owns_store:
                self.store.close()
            raise ValueError('Patient-case stores cannot be used for medical knowledge retrieval')
        if self.store.dimension != self.embedding.dimension or self.store.signature != self.embedding.signature:
            if self._owns_store:
                self.store.close()
            raise ValueError('Medical embedder/store configuration mismatch')

    def retrieve(self, query, top_k=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Query must be nonempty text')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        if not top_k:
            return []
        return self.store.search(self.embedding.embed_text(query), top_k)

    def close(self):
        if self._owns_store:
            self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

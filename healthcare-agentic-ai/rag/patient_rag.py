"""Composable patient-only retrieval API. No generation or clinical actions."""
import logging
import time
from .config import PatientRAGConfig
from .patient_parser import DDXPlusParser
from .models import PatientRepresentation
from .embeddings import EmbeddingModel
from .vector_store import QdrantVectorStore
from .patient_ingestion import batches, iter_training_documents
from .patient_retriever import PatientCaseRetriever

logger = logging.getLogger(__name__)


class PatientCaseRAG:
    def __init__(self, config=None, *, parser=None, embedding_model=None, vector_store=None):
        self.config = config or PatientRAGConfig()
        self.parser = parser  # Lazy: querying a built index does not need the dataset.
        self.embedding_model = embedding_model or EmbeddingModel(self.config.model_name,
            self.config.device, self.config.batch_size, allow_download=self.config.allow_download)
        self.vector_store = vector_store or QdrantVectorStore(self.config.storage_path,
            self.embedding_model.dimension, self.config.collection_name,
            embedding_signature=self.embedding_model.signature)
        self.retriever = PatientCaseRetriever(self.embedding_model, self.vector_store)

    def index(self, limit=None, batch_size=None, reset=False):
        size = self.config.batch_size if batch_size is None else batch_size
        if type(size) is not int or size < 1:
            raise ValueError('batch_size must be positive')
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError('limit must be nonnegative or None')
        start = time.perf_counter()
        parser = self.parser or DDXPlusParser()
        if reset:
            self.vector_store.delete_collection()
        self.vector_store.create_collection()
        before = self.vector_store.count()
        truncated_before = getattr(self.embedding_model, 'truncated_texts', 0)
        processed = 0
        for documents in batches(iter_training_documents(parser, limit), size):
            vectors = self.embedding_model.embed_texts(d['text'] for d in documents)
            self.vector_store.upsert_patients(documents, vectors)
            processed += len(documents)
            logger.info('Processed %d training records (%.2fs)', processed, time.perf_counter()-start)
        return {'training_records_processed': processed, 'count_before': before,
                'indexed_count': self.vector_store.count(),
                'truncated_texts': getattr(self.embedding_model, 'truncated_texts', 0) - truncated_before,
                'seconds': time.perf_counter()-start}

    def retrieve(self, patient: str | PatientRepresentation, top_k=5):
        if isinstance(patient, PatientRepresentation):
            patient = patient.to_text()
        return self.retriever.retrieve(patient, top_k)

    def close(self):
        self.vector_store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

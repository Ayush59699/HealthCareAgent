"""Additional boundary, manifest, actual-vector audit and truncation regressions."""
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from qdrant_client import models
from rag.embeddings import EmbeddingModel
from rag.models import EvaluationLabels
from rag.patient_audit import audit_patient_index
from rag.patient_ingestion import patient_document
from rag.patient_retriever import PatientCaseRetriever
from rag.vector_store import QdrantVectorStore
if __package__:
    from .test_patient_rag import FakeEmbedding, record
else:
    from test_patient_rag import FakeEmbedding, record


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = QdrantVectorStore(self.temp.name, 3, embedding_signature=FakeEmbedding.signature)
        self.addCleanup(self.store.close)
        self.doc = patient_document(record())
        self.store.upsert_patients([self.doc], [[1., 0., 0.]])

    def test_audit_actual_vectors_and_source(self):
        parser = Mock()
        parser.iter_patients.return_value = iter([record()])
        report = audit_patient_index(self.store, parser)
        self.assertEqual(report['vector_dimensions'], [3])
        self.assertEqual(report['source_payloads_verified'], 1)
        self.assertEqual(report['distance'], 'Cosine')
        parser.iter_patients.assert_called_once_with('train', limit=1, include_labels=False)

    def test_audit_rejects_contamination_not_only_search_filter(self):
        self.store.client.upsert(self.store.collection_name, [models.PointStruct(
            id=99, vector=[1., 0., 0.], payload={**self.doc, 'split': 'validate'})])
        with self.assertRaises(ValueError):
            audit_patient_index(self.store)

    def test_audit_rejects_hidden_label(self):
        self.store.client.set_payload(self.store.collection_name, {'diagnosis': 'secret'},
            points=[self.store.point_id(self.doc['patient_id'])])
        with self.assertRaises(ValueError):
            audit_patient_index(self.store)

    def test_audit_rejects_text_not_from_features(self):
        self.store.upsert_patients([{**self.doc, 'text': 'hidden label'}], [[1., 0., 0.]])
        parser = Mock()
        parser.iter_patients.return_value = iter([record()])
        with self.assertRaisesRegex(ValueError, 'label-free training source'):
            audit_patient_index(self.store, parser)

    def test_audit_rejects_wrong_point_id(self):
        self.store.client.upsert(self.store.collection_name, [models.PointStruct(
            id=99, vector=[1., 0., 0.], payload=self.doc)])
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            audit_patient_index(self.store)

    def test_id_source_agreement(self):
        with self.assertRaisesRegex(ValueError, 'source row'):
            self.store.upsert_patients([{**self.doc, 'patient_id': 'ddxplus:train:2'}], [[1, 0, 0]])

    def test_label_changes_do_not_change_document_or_id(self):
        changed = replace(record(), labels=EvaluationLabels('DIFFERENT_SECRET'))
        self.assertEqual(patient_document(changed), self.doc)
        self.assertEqual(self.store.point_id(changed.patient_id), self.store.point_id(record().patient_id))

    def test_injected_embedder_mismatch_rejected(self):
        embedder = FakeEmbedding()
        embedder.signature = {'model_name': 'different', 'dimension': 3}
        with self.assertRaisesRegex(ValueError, 'Embedding configuration mismatch'):
            PatientCaseRetriever(embedder, self.store)

    def test_missing_manifest_rejected(self):
        self.store.close()
        self.store.manifest_path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing manifest'):
            QdrantVectorStore(self.temp.name, 3, embedding_signature=FakeEmbedding.signature)

    def test_all_signature_settings_must_match(self):
        self.store.close()
        for key, value in [('normalize_embeddings', False), ('max_seq_length', 128),
                           ('text_format', 'different'), ('prompt', 'different')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Embedding configuration mismatch'):
                QdrantVectorStore(self.temp.name, 3,
                    embedding_signature={**FakeEmbedding.signature, key: value})

    def test_reset_and_reopen_stays_empty(self):
        self.store.delete_collection()
        self.store.create_collection()
        self.store.close()
        with QdrantVectorStore(self.temp.name, 3, embedding_signature=FakeEmbedding.signature) as reopened:
            self.assertEqual(reopened.count(), 0)


class EmbeddingTests(unittest.TestCase):
    def test_truncation_count_and_original_input_preserved(self):
        embedder = EmbeddingModel.__new__(EmbeddingModel)
        embedder.batch_size = 2
        embedder.truncated_texts = 0
        embedder.model = Mock(max_seq_length=512)
        embedder.model.tokenizer.return_value = {'length': [513, 512]}
        embedder.model.encode.return_value = np.array([[1., 0.], [0., 1.]])
        texts = ['original long text', 'short text']
        with self.assertLogs('rag.embeddings', level='WARNING'):
            self.assertEqual(len(embedder.embed_texts(texts)), 2)
        self.assertEqual(embedder.truncated_texts, 1)
        self.assertEqual(embedder.model.encode.call_args.args[0], texts)
        self.assertTrue(embedder.model.encode.call_args.kwargs['normalize_embeddings'])
        self.assertFalse(embedder.model.tokenizer.call_args.kwargs['truncation'])

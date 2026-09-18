"""Offline unit tests use real temporary Qdrant and deterministic test vectors.
The fake embedder is ONLY a test fixture, never a runtime model fallback.
"""
from dataclasses import replace
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
from qdrant_client import models
from rag.config import PatientRAGConfig
from rag.models import PatientRecord, PatientRepresentation, Evidence, EvaluationLabels
from rag.patient_ingestion import patient_document, iter_training_documents, batches
from rag.patient_retriever import PatientCaseRetriever
from rag.patient_rag import PatientCaseRAG
from rag.vector_store import QdrantVectorStore


class FakeEmbedding:
    dimension = 3
    signature = {'model_name': 'unit-test-only', 'dimension': 3}
    def embed_text(self, text):
        return [1., 0., 0.]
    def embed_texts(self, texts):
        return [self.embed_text(t) for t in texts]


def record(number=1, split='train'):
    patient = PatientRepresentation(45, 'F', (Evidence('E_1', 'Fever?', 'Yes', False),), ())
    return PatientRecord(f'ddxplus:{split}:{number}', split,
        f'release_{split}_patients.zip!release_{split}_patients#row={number}', patient,
        EvaluationLabels('SECRET_DIAGNOSIS'))


class PatientRAGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = QdrantVectorStore(self.temp.name, 3, embedding_signature=FakeEmbedding.signature)
        self.addCleanup(self.store.close)
        self.embedding = FakeEmbedding()
        self.retriever = PatientCaseRetriever(self.embedding, self.store)

    def insert(self, n=3):
        docs = [patient_document(record(i+1)) for i in range(n)]
        self.store.upsert_patients(docs, [[1., i/10, 0.] for i in range(n)])
        return docs

    def test_training_indexed(self):
        self.insert()
        self.assertEqual(self.store.count(), 3)

    def test_validation_rejected(self):
        with self.assertRaises(ValueError):
            patient_document(record(split='validate'))

    def test_test_split_rejected(self):
        with self.assertRaises(ValueError):
            patient_document(record(split='test'))

    def test_labels_never_in_payload_or_text(self):
        self.insert()
        for payload in self.store.iter_payloads():
            self.assertEqual(set(payload), {'patient_id','split','text','source'})
            for field in ('PATHOLOGY','DIFFERENTIAL_DIAGNOSIS','ground_truth_pathology','SECRET_DIAGNOSIS'):
                self.assertNotIn(field, json.dumps(payload))

    def test_rejects_extra_label_fields(self):
        for field in ('PATHOLOGY','DIFFERENTIAL_DIAGNOSIS','ground_truth_pathology','diagnosis'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store.upsert_patients([{**patient_document(record()),field:'secret'}], [[1,0,0]])
        self.assertEqual(self.store.count(), 0)

    def test_store_rejects_nontraining_payload(self):
        for split in ('validate','test'):
            with self.assertRaises(ValueError):
                self.store.upsert_patients([{**patient_document(record()),'split':split}], [[1,0,0]])

    def test_top_k_or_fewer(self):
        self.insert()
        self.assertEqual(len(self.retriever.retrieve('fever', 2)), 2)
        self.assertEqual(len(self.retriever.retrieve('fever', 10)), 3)

    def test_scores_numeric_ids_preserved(self):
        docs = self.insert()
        hits = self.retriever.retrieve('fever')
        self.assertEqual({h['patient_id'] for h in hits}, {d['patient_id'] for d in docs})
        self.assertTrue(all(isinstance(h['score'], float) for h in hits))
        self.assertTrue(all(h['split']=='train' for h in hits))
        self.assertEqual(hits[0]['source'], docs[0]['source'])

    def test_scores_not_probabilities(self):
        self.store.upsert_patients([patient_document(record())], [[-1,0,0]])
        hit = self.retriever.retrieve('fever')[0]
        self.assertAlmostEqual(hit['score'], -1.)
        self.assertNotIn('probability', hit)
        self.assertNotIn('confidence', hit)

    def test_empty_collection(self):
        self.assertEqual(self.retriever.retrieve('fever'), [])
        self.store.create_collection()
        self.assertEqual(self.retriever.retrieve('fever'), [])

    def test_duplicate_ids_last_wins(self):
        doc = patient_document(record())
        updated = {**doc, 'text':'Updated clinical features'}
        self.store.upsert_patients([doc, updated], [[1,0,0],[0,1,0]])
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(next(self.store.iter_payloads())['text'], updated['text'])

    def test_rerun_idempotent(self):
        self.insert()
        self.insert()
        self.assertEqual(self.store.count(), 3)

    def test_filter_excludes_contamination(self):
        self.insert(1)
        self.store.client.upsert(self.store.collection_name, [models.PointStruct(id=99,
            vector=[1.,0.,0.], payload={'split':'test','text':'bad'})])
        hits = self.retriever.retrieve('fever')
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['split'], 'train')

    def test_ingestion_explicit_train_no_labels(self):
        parser = Mock()
        parser.iter_patients.return_value = iter([record()])
        docs = list(iter_training_documents(parser, 1))
        parser.iter_patients.assert_called_once_with('train', limit=1, include_labels=False)
        self.assertNotIn('SECRET_DIAGNOSIS', json.dumps(docs))

    def test_pipeline_index_rerun_and_retrieve(self):
        parser = Mock()
        parser.iter_patients.side_effect = lambda *a, **kw: iter([record(1),record(2)])
        rag = PatientCaseRAG(PatientRAGConfig(batch_size=1), parser=parser,
                             embedding_model=self.embedding, vector_store=self.store)
        self.assertEqual(rag.index()['indexed_count'], 2)
        self.assertEqual(rag.index()['indexed_count'], 2)
        self.assertEqual(len(rag.retrieve(record().patient)), 2)

    def test_batches_bounded(self):
        self.assertEqual(list(batches(range(5),2)), [[0,1],[2,3],[4]])
        with self.assertRaises(ValueError):
            list(batches(range(5),0))

    def test_invalid_vectors_rejected(self):
        for vector in ([1,2], [0,0,0], [float('nan'),1,0]):
            with self.assertRaises(ValueError):
                self.store.upsert_patients([patient_document(record())], [vector])
        with self.assertRaises(ValueError):
            self.store.upsert_patients([patient_document(record())], [])

    def test_invalid_query_and_top_k(self):
        for query, k in (('',5), ('fever',-1), ('fever',True), ({},5)):
            with self.assertRaises(ValueError):
                self.retriever.retrieve(query,k)
        self.assertEqual(self.retriever.retrieve('fever',0), [])

    def test_persistent_reopen(self):
        self.insert()
        self.store.close()
        with QdrantVectorStore(self.temp.name,3,embedding_signature=FakeEmbedding.signature) as reopened:
            self.assertEqual(reopened.count(),3)

    def test_model_mismatch_rejected(self):
        self.insert()
        self.store.close()
        with self.assertRaisesRegex(ValueError,'Embedding configuration mismatch'):
            QdrantVectorStore(self.temp.name,3,embedding_signature={'model_name':'wrong'})

    def test_dimension_mismatch_rejected(self):
        self.insert()
        self.store.close()
        with self.assertRaisesRegex(ValueError,'dimension/distance mismatch'):
            QdrantVectorStore(self.temp.name,4)

    def test_delete_and_create(self):
        self.insert()
        self.store.delete_collection()
        self.assertFalse(self.store.collection_exists())
        self.store.create_collection()
        self.assertEqual(self.store.count(),0)


if __name__ == '__main__':
    unittest.main()

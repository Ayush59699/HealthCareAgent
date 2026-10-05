"""Label-free retrieval/snapshot contracts. No model download or clinical labels."""
import copy
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from rag.experimental_medical.qdrant_index import QdrantMedicalIndex, evidence_hits
from rag.experimental_medical.queries import semantic_groups, build_queries
from rag.experimental_medical.retrieval import choose_final
from rag.focused_medical import FocusedEvidenceService, filter_candidates, query_plan, state_from_patient
from rag.agents.grounding import context
from tests import test_experimental_medical as experimental_tests
from tests.test_experimental_medical import patient, FakeEmbedding
from tests.test_focused_medical import hit
from tests.test_patient_rag import record


class Task74Tests(unittest.TestCase):
    def test_no_early_overlap_gate(self):
        unrelated = hit('Kidney', 'Renal research.')
        retained, audit = filter_candidates(query_plan(patient()), [unrelated])
        self.assertEqual(retained, [unrelated])
        self.assertEqual(audit[0]['reason'], 'candidate_for_reranking')

    def test_short_clusters_even_when_vectors_all_similar(self):
        facts = [{'text': str(i)} for i in range(17)]
        groups = semantic_groups(facts, [[1., 0.]] * 17, 4)
        self.assertTrue(all(len(g) <= 4 for g in groups))
        self.assertEqual(sum(len(g) for g in groups), 17)

    def test_all_negative_model_scores_deliver_no_evidence(self):
        retriever, reranker = experimental_tests.ExperimentalMedicalTests().fixture()
        reranker.score.side_effect = lambda pairs: [-5.] * len(pairs)
        result = retriever.retrieve(patient())
        self.assertEqual(result['final_passages'], [])
        self.assertEqual(len(result['reranked']), 4)
        self.assertEqual({r['reason'] for r in result['final_selection_removed']}, {'nonpositive_model_relevance'})

    def test_no_filling_with_weak_passages(self):
        retriever, reranker = experimental_tests.ExperimentalMedicalTests().fixture()
        reranker.score.side_effect = lambda pairs: [3.] + [-5.] * (len(pairs)-1)
        result = retriever.retrieve(patient())
        self.assertEqual(len(result['final_passages']), 1)

    def fixture(self):
        payload = hit('Fever', 'Fever is a symptom of illness.')
        payload.pop('score')
        medical = Mock()
        medical.embedding = FakeEmbedding()
        medical.embedding.dimension = 3
        medical.store.collection_name = 'medical_knowledge'
        medical.store.signature = medical.embedding.signature
        medical.store.iter_points.return_value = [SimpleNamespace(payload=payload, vector=[1., 0., 0.])]
        medical.store.search.return_value = [{**payload, 'score': .8}]
        return medical, payload

    def test_qdrant_adapter_keeps_original_payload_and_collection(self):
        medical, payload = self.fixture()
        index = QdrantMedicalIndex(medical)
        self.assertEqual(index.records[0]['evidence_payload'], payload)
        self.assertEqual(index.search([1., 0., 0.], 50), [(0, .8)])
        medical.store.search.assert_called_once_with([1., 0., 0.], top_k=50)
        self.assertNotEqual(index.records[0]['retrieval_text'], index.records[0]['evidence_text'])
        medical.store.collection_name = 'ddxplus_patient_cases'
        with self.assertRaises(ValueError):
            QdrantMedicalIndex(medical)

    def test_changed_store_or_signature_rejected(self):
        medical, payload = self.fixture()
        index = QdrantMedicalIndex(medical)
        medical.store.search.return_value[0]['text'] = 'Tampered'
        with self.assertRaises(ValueError):
            index.search([1., 0., 0.], 50)
        medical.store.signature = {'different': True}
        with self.assertRaises(ValueError):
            QdrantMedicalIndex(medical)

    def test_hybrid_service_snapshot_integrity_and_original_case_query(self):
        medical, payload = self.fixture()
        reranker = Mock(signature={'model': 'test-only'}, truncated_pairs=0)
        reranker.score.side_effect = lambda pairs: [3.] * len(pairs)
        patients = Mock()
        patients.retrieve.return_value = []
        service = FocusedEvidenceService.with_hybrid(patients, medical, reranker=reranker)
        r = record()
        state = state_from_patient(r.patient, r.patient_id)
        snapshot = service.retrieve(r.patient, state)
        cases, evidence = snapshot.agent_evidence()
        context(state, cases, evidence)
        self.assertEqual(evidence[0].text, payload['text'])
        self.assertEqual(evidence[0].metadata['chunk_id'], payload['chunk_id'])
        self.assertNotIn('retrieval_text', evidence[0].metadata)
        self.assertEqual(snapshot.content_id(), snapshot.snapshot_id)
        patients.retrieve.assert_called_once_with(r.patient.to_text(), top_k=1)
        with self.assertRaises(ValueError):
            snapshot.model_validate({**snapshot.model_dump(), 'snapshot_id': '0'*64})
        with self.assertRaises(ValueError):
            service.retrieve(r.patient, state.model_copy(update={'symptoms': ['Invented']}))

    def test_service_reranker_failure_is_not_baseline_fallback(self):
        medical, _ = self.fixture()
        reranker = Mock()
        reranker.score.side_effect = RuntimeError('unavailable')
        reranker.truncated_pairs = 0
        patients = Mock()
        patients.retrieve.return_value = []
        service = FocusedEvidenceService.with_hybrid(patients, medical, reranker=reranker)
        r = record()
        with self.assertRaises(RuntimeError):
            service.retrieve(r.patient, state_from_patient(r.patient, r.patient_id))
        medical.retrieve.assert_not_called()
        self.assertIsNone(service.audit)

    def test_absent_positive_findings_empty_snapshot(self):
        medical, _ = self.fixture()
        reranker = Mock()
        patients = Mock()
        patients.retrieve.return_value = []
        service = FocusedEvidenceService.with_hybrid(patients, medical, reranker=reranker)
        from dataclasses import replace
        from rag.models import PatientRepresentation
        r = replace(record(), patient=PatientRepresentation(None, None, (), ()))
        snapshot = service.retrieve(r.patient, state_from_patient(r.patient, r.patient_id))
        self.assertEqual(snapshot.medical_knowledge, ())
        reranker.score.assert_not_called()


if __name__ == '__main__':
    unittest.main()

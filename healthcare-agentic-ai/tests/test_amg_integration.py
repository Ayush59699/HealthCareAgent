"""Integration contracts, no downloads, LLM APIs or benchmark labels."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from application.workflow import create_workflow
from rag.amg import AMGMedicalEvidence, AMGEvidenceService
from rag.amg.provenance import validate_amg_hit
from rag.agents.grounding import evidence_from_hits, state_from_patient
from orchestration.evidence import EvidenceSnapshot, fingerprint
from tests.amg_helpers import amg_hit, configure_medical, configure_provider
from tests.safety_helpers import setup, SafetyScenarioLLM, semantic, critique


class AMGIntegrationTests(unittest.TestCase):
    def fixture(self, provider=None):
        _, case, llm, patients, medical = setup(provider)
        configure_medical(medical)
        configure_provider(llm)
        return create_workflow(llm, patients, medical), case, llm, patients, medical

    def test_native_provenance_round_trip(self):
        workflow, case, _, _, _ = self.fixture()
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'final')
        snapshot = EvidenceSnapshot.model_validate_json(state.evidence.model_dump_json())
        evidence = snapshot.agent_evidence()[1][0]
        self.assertEqual(evidence.source_id, 'medical:' + amg_hit()['chunk_id'])
        self.assertEqual(evidence.text, amg_hit()['text'])
        self.assertEqual(evidence.metadata['amg_metadata'], amg_hit()['amg_metadata'])
        self.assertEqual(evidence.metadata['retrieval']['distance'], 0.5)
        self.assertEqual(evidence.similarity, -0.5)

    def test_identical_evidence_reaches_diagnostic_critic_and_safety(self):
        workflow, case, llm, patients, medical = self.fixture()
        state = workflow.run(case.patient, case.patient_id)
        seen = set()
        for call in llm.calls:
            title = call['text']['format']['schema']['title']
            if title in {'DiagnosticResult', 'ClinicalCritique', 'SemanticSafetyResult'}:
                seen.add(title)
                payload = call['input'][0]['content']
                self.assertIn('medical:' + amg_hit()['chunk_id'], payload)
                self.assertIn(amg_hit()['url'], payload)
                self.assertIn(amg_hit()['amg_metadata']['chunk_sha256'], payload)
        self.assertEqual(len(seen), 3)
        self.assertEqual(state.safety_coverage, 'assessed')
        patients.retrieve.assert_called_once()
        medical.retrieve.assert_called_once()

    def test_no_evidence_abstains_before_diagnosis_without_fake_safety_pass(self):
        workflow, case, llm, _, medical = self.fixture()
        medical.retrieve.return_value = []
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'abstained')
        self.assertEqual(state.safety_skip_reason, 'no_medical_evidence')
        self.assertEqual(state.diagnostics, ())
        self.assertEqual(llm.safety_count, 0)
        self.assertEqual(state.medical_retrieval['status'], 'insufficient_evidence')

    def test_retrieval_exception_fails_closed(self):
        workflow, case, _, _, medical = self.fixture()
        medical.retrieve.side_effect = RuntimeError('SECRET')
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'failed')
        self.assertNotIn('SECRET', state.model_dump_json())
        self.assertFalse(state.diagnostics)

    def test_labels_rejected_before_retrieval(self):
        workflow, case, llm, patients, medical = self.fixture()
        with self.assertRaises(TypeError):
            workflow.run(case, case.patient_id)
        self.assertEqual(llm.calls, [])
        patients.retrieve.assert_not_called()
        medical.retrieve.assert_not_called()

    def test_empty_query_does_not_call_amg(self):
        workflow, case, _, _, medical = self.fixture()
        medical.query_for.return_value = ('', 1)
        state = workflow.run(case.patient, case.patient_id)
        medical.retrieve.assert_not_called()
        self.assertEqual(state.status, 'abstained')

    def test_human_review_remains_withheld(self):
        workflow, case, _, _, _ = self.fixture(SafetyScenarioLLM(safety_results=[semantic('safety_ambiguity')]))
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'human_review_required')
        self.assertEqual(state.safety_coverage, 'assessed')

    def test_revision_reuses_same_snapshot_and_does_not_retrieve_again(self):
        workflow, case, _, patients, medical = self.fixture(SafetyScenarioLLM(critiques=[critique('revision_required')]))
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'unresolved')
        self.assertGreaterEqual(len(state.safety_assessments), 1)
        self.assertTrue(all(a.ticket.evidence_snapshot_id == state.evidence.snapshot_id for a in state.safety_assessments))
        patients.retrieve.assert_called_once()
        medical.retrieve.assert_called_once()

    def test_tampered_payloads_and_rejected_candidates_are_not_evidence(self):
        changes = [('text', 'changed'), ('chunk_id', 'invented'), ('url', 'https://evil.example'),
                   ('score', 0.99), ('PATHOLOGY', 'hidden label')]
        for key, value in changes:
            hit = amg_hit(); hit[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence_from_hits([hit], 'medical_knowledge')
        for acceptance, distance in [('rejected', 0.5), ('distance_gate', 1.2)]:
            hit = amg_hit(); hit['retrieval'].update(acceptance=acceptance, distance=distance)
            with self.assertRaises(ValueError):
                evidence_from_hits([hit], 'medical_knowledge')

    def test_duplicate_native_ids_rejected(self):
        with self.assertRaises(ValueError):
            evidence_from_hits([amg_hit(), amg_hit()], 'medical_knowledge')

    def test_backend_calls_supplied_search_and_preserves_accepted_hits(self):
        hit = amg_hit()
        source = {'id': hit['chunk_id'], 'text': hit['text'], 'metadata': hit['amg_metadata']}
        upstream_hit = {**source, 'title': hit['title'], 'url': hit['url'], 'distance': 0.5, 'acceptance': 'distance_gate'}
        backend = AMGMedicalEvidence.__new__(AMGMedicalEvidence)
        backend.retriever = Mock()
        backend.retriever.search.return_value = {'results': [upstream_hit], 'candidates': [upstream_hit]}
        backend.max_distance = 1.10; backend.snapshot = 'a'*24
        backend.manifest = {'config': {'collection': 'medlineplus_en_experimental'}}
        backend._source_hashes = {source['id']: fingerprint(source)}
        self.assertEqual(backend.retrieve('query', top_k=3), [hit])
        backend.retriever.search.assert_called_once_with('query', max_distance=1.10, top_k=3)
        backend.retriever.search.return_value['results'][0]['text'] = 'corruption'
        with self.assertRaises(ValueError):
            backend.retrieve('query')

    def test_no_legacy_medical_imports_in_application_entrypoints(self):
        root = Path(__file__).resolve().parents[1]
        code = "import scripts.run_phase6, demo.terminal, sys; assert not any(n in sys.modules for n in ('rag.medical_retriever','rag.medical_embeddings','rag.focused_medical','rag.experimental_medical'))"
        subprocess.run([sys.executable, '-c', code], cwd=root, check=True, capture_output=True)

    def test_query_budget_keeps_complete_literal_facts(self):
        backend = AMGMedicalEvidence.__new__(AMGMedicalEvidence)
        backend.retriever = Mock()
        backend.retriever.embedder.max_tokens = 6
        backend.retriever.embedder.tokenizer.encode.side_effect = lambda text, **kwargs: text.split()
        state = Mock(presenting_evidence=['short fact'], symptoms=['long fact that exceeds all tokens available', 'short fact'], antecedents=['other fact'])
        self.assertEqual(backend.query_for(state), ('short fact; other fact', 1))

    def test_bounds(self):
        for k in (0, 21, True):
            with self.assertRaises(ValueError):
                AMGEvidenceService(Mock(), Mock(), medical_top_k=k)

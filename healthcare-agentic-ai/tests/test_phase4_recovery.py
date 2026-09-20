"""Regression tests for gaps found while recovering the interrupted Phase 4."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag.agents.grounding import validate_references
from rag.agents.models import Hypothesis, EvidenceClaim
from rag.models import EvaluationLabels
from rag.phase4_evaluation import evaluate_case, summarize
from scripts.run_phase4 import write_json_atomic, main
from tests.test_patient_rag import record
from tests import test_phase4


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        test = test_phase4.Phase4Tests()
        test.setUp()
        return test.run_case()

    def test_no_medical_evidence_requires_empty_differential_too(self):
        output = self.fixture()
        output.diagnosis.differential_diagnoses = [Hypothesis(
            condition='Tentative', rationale=EvidenceClaim(statement='Observation', evidence_refs=['patient:age']),
            supporting_evidence=[], contradicting_evidence=[])]
        with self.assertRaises(ValueError):
            validate_references(output.diagnosis, output.patient_state, output.patient_case_evidence, [])

    def test_critic_failure_counts_as_accuracy_miss(self):
        output = self.fixture()
        output.diagnosis.primary_hypothesis = Hypothesis(condition='Asthma',
            rationale=EvidenceClaim(statement='Tentative', evidence_refs=['patient:age']),
            supporting_evidence=[], contradicting_evidence=[])
        output.status = 'failed'
        output.critique = None
        case = evaluate_case(output, EvaluationLabels('Asthma'))
        self.assertFalse(case['primary_match'])
        self.assertFalse(case['top_k_match'])
        metrics = summarize([output], [case])
        self.assertEqual(metrics['critic_outputs_scored'], 0)
        self.assertEqual(metrics['diagnostic_outputs_scored'], 1)
        self.assertEqual(metrics['retrieval_success']['medical_knowledge']['rate'], 1)

    def test_atomic_write_retains_previous_report_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            write_json_atomic(path, '{"old":true}')
            with patch.object(Path, 'replace', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    write_json_atomic(path, '{"new":true}')
            self.assertEqual(json.loads(path.read_text()), {'old': True})

    def test_runner_requests_labels_only_after_all_inference(self):
        from contextlib import nullcontext
        from unittest.mock import Mock
        output = self.fixture()
        events = []
        patient = record(split='validate')
        def stream(split, *, limit, include_labels):
            self.assertEqual(split, 'validate')
            events.append('labels' if include_labels else 'features')
            from dataclasses import replace
            yield replace(patient, labels=EvaluationLabels('Asthma') if include_labels else None)
        parser = Mock()
        parser.iter_patients.side_effect = stream
        pipeline = Mock()
        pipeline.run.side_effect = lambda *args: (events.append('inference'), output)[1]
        patients, medical = Mock(), Mock()
        patients.vector_store.count.return_value = 1000
        medical.store.count.return_value = 87
        patients.embedding_model.truncated_texts = 0
        medical.embedding.truncated_texts = 0
        patients.embedding_model.signature = {}
        medical.embedding.signature = {}
        with tempfile.TemporaryDirectory() as directory, \
                patch('scripts.run_phase4.DDXPlusParser', return_value=parser), \
                patch('scripts.run_phase4.PatientCaseRAG', return_value=nullcontext(patients)), \
                patch('scripts.run_phase4.MedicalKnowledgeRetriever', return_value=nullcontext(medical)), \
                patch('scripts.run_phase4.Phase4Pipeline', return_value=pipeline), \
                patch('scripts.run_phase4.OpenAIProvider'), \
                patch('scripts.run_phase4.load_generation_env'):
            self.assertEqual(main(['--output-dir', str(Path(directory) / 'run')]), 0)
        self.assertEqual(events, ['features', 'inference', 'labels'])
        patients.index.assert_not_called()


if __name__ == '__main__':
    unittest.main()

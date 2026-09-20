"""Cloud generation and honest failure reporting; all inference is mocked."""
import unittest
from rag.agents.models import Failure, PipelineResult
from rag.models import EvaluationLabels
from rag.phase4_evaluation import evaluate_case, summarize
from scripts.run_phase4 import stage_observations


class StageReportingTests(unittest.TestCase):
    def test_missing_diagnosis_is_not_an_incorrect_diagnosis(self):
        for stage, code in [('patient', 'context_budget'), ('diagnostic', 'timeout'),
                            ('diagnostic', 'agent_validation_failure')]:
            with self.subTest(stage=stage, code=code):
                output = PipelineResult(patient_id='synthetic', stage=stage,
                                        failure=Failure(code=code, message='Test failure'))
                case = evaluate_case(output, EvaluationLabels('Asthma'))
                self.assertFalse(case['diagnostic_correctness_measurable'])
                self.assertEqual(case['correctness_note'],
                    'Diagnostic Agent did not produce an accepted DiagnosticResult; diagnostic correctness was therefore not measurable.')
                for key in ('primary_match', 'top_k_match', 'accepted_primary_match', 'accepted_top_k_match'):
                    self.assertIsNone(case[key])
                metrics = summarize([output], [case])
                for key in ('primary_accuracy', 'top_k_accuracy', 'accepted_primary_accuracy', 'accepted_top_k_accuracy'):
                    self.assertIsNone(metrics[key])
                self.assertEqual(metrics['diagnostic_outputs_scored'], 0)

    def test_stage_observations_distinguish_skipped_failure_and_success(self):
        output = PipelineResult(patient_id='synthetic',
            stage_seconds={'patient': 1.25, 'patient_case': 0.5, 'medical_knowledge': 0.75},
            attempts={'patient': 2}, structured_validity={'patient': True},
            retrieval_success={'patient_case': True, 'medical_knowledge': False})
        observations = stage_observations(output)
        self.assertEqual(observations['patient']['retries'], 1)
        self.assertEqual(observations['patient']['seconds'], 1.25)
        self.assertEqual(observations['patient']['status'], 'success')
        self.assertEqual(observations['patient_case']['status'], 'success')
        self.assertEqual(observations['medical_knowledge']['status'], 'failure')
        for name in ('diagnostic', 'critic'):
            self.assertEqual(observations[name]['status'], 'skipped')
            self.assertIsNone(observations[name]['seconds'])
            self.assertIsNone(observations[name]['structured_valid'])
        self.assertNotIn('ground_truth', str(observations))


if __name__ == '__main__':
    unittest.main()

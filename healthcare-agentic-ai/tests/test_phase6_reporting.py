"""CLI uses fake backends only; reports distinguish withheld from technical failure."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from scripts import run_phase6 as runner
from tests.safety_helpers import setup, run, SafetyScenarioLLM, semantic, diagnosis, critique


class Phase6ReportingTests(unittest.TestCase):
    def manager(self, value):
        mock = Mock()
        mock.__enter__ = Mock(return_value=value)
        mock.__exit__ = Mock(return_value=False)
        return mock

    def invoke(self, output, llm=None, queries=1):
        fixture = setup(llm)
        _, case, provider, patients, medical = fixture
        patients.vector_store.count.return_value = medical.store.count.return_value = 1
        parser = Mock()
        parser.iter_patients.return_value = [case]
        with patch.multiple(runner, load_generation_env=Mock(), DDXPlusParser=Mock(return_value=parser),
                PatientCaseRAG=Mock(return_value=self.manager(patients)),
                MedicalKnowledgeRetriever=Mock(return_value=self.manager(medical)),
                OpenAIProvider=Mock(return_value=self.manager(provider))), redirect_stdout(io.StringIO()):
            status = runner.main(['--output-dir', str(output), '--queries', str(queries)])
        parser.iter_patients.assert_called_once_with('validate', limit=queries, include_labels=False)
        return status

    def test_cli_success_writes_versioned_atomic_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            self.assertEqual(self.invoke(output), 0)
            report = json.loads((output / 'phase6_run_report.json').read_text())
            self.assertEqual(report['schema_version'], 'phase6-report-v1')
            case = report['cases'][0]
            self.assertTrue(case['eligible_research_output'])
            self.assertFalse(case['clinical_approval'])
            self.assertEqual(case['safety_requests'], 1)
            self.assertEqual(case['safety_decision'], 'CONTINUE')
            self.assertTrue((output / 'sample_case_001.json').exists())
            self.assertEqual(list(output.glob('*.tmp')), [])

    def test_human_review_is_completed_but_withheld_not_scheduled(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            self.assertEqual(self.invoke(output, SafetyScenarioLLM(safety_results=[semantic('safety_ambiguity')])), 0)
            case = json.loads((output / 'phase6_run_report.json').read_text())['cases'][0]
            self.assertEqual(case['status'], 'human_review_required')
            self.assertTrue(case['withheld'])
            self.assertFalse(case['eligible_research_output'])
            self.assertFalse(case['human_review_scheduled'])

    def test_failures_and_missing_requested_cases_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.invoke(Path(directory) / 'failed', SafetyScenarioLLM(safety_results=[{}])), 1)
            self.assertEqual(self.invoke(Path(directory) / 'missing', queries=2), 1)

    def test_setup_failure_sanitized(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'load_generation_env', side_effect=ValueError('SECRET')):
            output = Path(directory) / 'run'
            self.assertEqual(runner.main(['--output-dir', str(output)]), 1)
            text = (output / 'phase6_run_report.json').read_text()
            self.assertNotIn('SECRET', text)
            self.assertEqual(json.loads(text)['status'], 'setup_or_execution_failure')

    def test_existing_directory_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / 'phase6_run_report.json'
            sentinel.write_text('keep')
            self.assertEqual(runner.main(['--output-dir', directory]), 1)
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_repetition_summary_does_not_reuse_prior_continue(self):
        state = run(setup(SafetyScenarioLLM(critiques=[critique('revision_required')])))
        self.assertEqual(len(state.safety_assessments), 1)
        summary = runner.case_summary(state)
        self.assertEqual(summary['safety_coverage'], 'not_assessed')
        self.assertIsNone(summary['safety_decision'])
        self.assertIsNone(summary['safety_ticket'])
        self.assertTrue(summary['withheld'])

    def test_early_abstention_summary_not_fake_safety_pass(self):
        fixture = setup()
        fixture[4].retrieve.return_value = []
        summary = runner.case_summary(run(fixture))
        self.assertEqual(summary['safety_skip_reason'], 'no_medical_evidence')
        self.assertIsNone(summary['safety_decision'])
        self.assertEqual(summary['safety_requests'], 0)

    def test_budget_arguments_reject_invalid_values(self):
        for args in (['--queries', '0'], ['--max-seconds', 'nan'], ['--max-requests', '0']):
            with self.assertRaises(SystemExit):
                runner.main(args)

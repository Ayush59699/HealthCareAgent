"""Successful execution, terminal outcomes, state serialization and audit."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
from orchestration.state import WorkflowState
from tests.orchestration_helpers import setup, run, ScenarioLLM, diagnosis, critique


class OrchestrationTests(unittest.TestCase):
    def test_normal_successful_workflow(self):
        fixture = setup()
        state = run(fixture)
        self.assertEqual((state.status, state.outcome), ('final', 'accepted_with_limitations'))
        self.assertIsNone(state.failure)
        self.assertEqual(state.revision_count, 0)
        self.assertEqual(len(state.diagnostics), 1)
        self.assertEqual(len(state.critiques), 1)
        self.assertEqual(state.total_requests, 3)
        self.assertEqual(state.provider_repairs, 0)
        fixture[3].retrieve.assert_called_once_with(fixture[1].patient.to_text(), top_k=1)
        fixture[4].retrieve.assert_called_once_with(fixture[1].patient.to_text(), top_k=1)
        self.assertEqual(WorkflowState.model_validate_json(state.model_dump_json()), state)

    def test_valid_abstention_reviewed_but_not_approved(self):
        state = run(setup(ScenarioLLM([diagnosis(abstain=True)])))
        self.assertEqual(state.outcome, 'diagnostic_agent_abstention')
        self.assertEqual(state.status, 'abstained')
        self.assertEqual(len(state.critiques), 1)
        self.assertEqual(state.diagnostics[0].result.patient_case_evidence, [])
        self.assertEqual(state.diagnostics[0].result.medical_knowledge_evidence, [])

    def test_missing_medical_evidence_abstains_without_generation(self):
        fixture = setup()
        fixture[4].retrieve.return_value = []
        state = run(fixture)
        self.assertEqual(state.outcome, 'insufficient_evidence')
        self.assertEqual(state.status, 'abstained')
        self.assertEqual(state.diagnostics, ())  # No manufactured diagnosis.
        self.assertEqual(len(fixture[2].calls), 1)
        self.assertEqual(len(state.evidence.patient_cases), 1)

    def test_empty_patient_cases_do_not_preclude_medical_reasoning(self):
        fixture = setup()
        fixture[3].retrieve.return_value = []
        state = run(fixture)
        self.assertEqual(state.status, 'final')
        self.assertEqual(state.evidence.patient_cases, ())

    def test_critic_insufficient_evidence(self):
        state = run(setup(ScenarioLLM(critiques=[critique('insufficient_evidence')])))
        self.assertEqual(state.outcome, 'insufficient_evidence')
        self.assertEqual(state.revision_count, 0)

    def test_safety_blocks_even_supported_diagnosis(self):
        state = run(setup(ScenarioLLM(critiques=[critique(safety_flags=['Requires review'])])))
        self.assertEqual((state.status, state.outcome), ('blocked', 'safety_blocked'))
        self.assertNotIn('finalized', [e.event for e in state.transition_history])

    def test_audit_records_gate_and_exact_version(self):
        state = run(setup())
        events = state.transition_history
        self.assertEqual([e.sequence for e in events], list(range(1, len(events) + 1)))
        names = [e.event for e in events]
        self.assertLess(names.index('diagnostic_validated'), names.index('grounding_passed'))
        self.assertLess(names.index('grounding_passed'), names.index('critic_completed'))
        self.assertEqual(names[-1], 'finalized')
        self.assertTrue(all(e.run_id == state.run_id and e.timestamp for e in events))
        self.assertTrue(all(e.evidence_snapshot_id == state.evidence.snapshot_id for e in events[4:]))
        diagnostic, review = state.diagnostics[-1], state.critiques[-1]
        self.assertEqual(diagnostic.version, review.diagnostic_version)
        self.assertEqual(diagnostic.fingerprint, review.ticket.diagnostic_fingerprint)
        self.assertEqual(review.evidence_snapshot_id, state.evidence.snapshot_id)

    def test_runs_do_not_share_state(self):
        fixture = setup()
        first, second = run(fixture), run(fixture)
        self.assertNotEqual(first.run_id, second.run_id)
        first.patient_state.symptoms.append('external mutation')
        self.assertNotIn('external mutation', second.patient_state.symptoms)
        self.assertEqual(len(second.diagnostics), 1)

    def test_phase5_cli_with_mocked_backends(self):
        scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
        with patch.object(sys, 'path', [scripts] + sys.path):
            import run_phase5 as runner
        fixture = setup(ScenarioLLM([diagnosis(abstain=True)]))
        _, case, llm, patients, medical = fixture
        patients.vector_store.count.return_value = 1
        medical.store.count.return_value = 1
        def manager(value):
            mock = Mock()
            mock.__enter__ = Mock(return_value=value)
            mock.__exit__ = Mock(return_value=False)
            return mock
        parser = Mock()
        parser.iter_patients.return_value = [case]
        with tempfile.TemporaryDirectory() as directory, patch.multiple(runner,
                load_generation_env=Mock(), DDXPlusParser=Mock(return_value=parser),
                PatientCaseRAG=Mock(return_value=manager(patients)),
                MedicalKnowledgeRetriever=Mock(return_value=manager(medical)),
                OpenAIProvider=Mock(return_value=manager(llm))):
            output = Path(directory) / 'run'
            self.assertEqual(runner.main(['--output-dir', str(output)]), 0)
            parser.iter_patients.assert_called_once_with('validate', limit=1, include_labels=False)
            report = json.loads((output / 'phase5_run_report.json').read_text())
            self.assertEqual(report['cases'][0]['outcome'], 'diagnostic_agent_abstention')
            self.assertTrue((output / 'sample_case_001.json').is_file())

    def test_phase5_cli_setup_failure_is_sanitized(self):
        scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
        with patch.object(sys, 'path', [scripts] + sys.path):
            import run_phase5 as runner
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'load_generation_env', side_effect=ValueError('SECRET')):
            output = Path(directory) / 'run'
            self.assertEqual(runner.main(['--output-dir', str(output)]), 1)
            self.assertNotIn('SECRET', (output / 'phase5_run_report.json').read_text())

"""Independent application gates reject invalid outputs even from bypassed providers."""
import unittest
from unittest.mock import Mock
from rag.llm.provider import Generation
from rag.agents.models import DiagnosticResult
from tests.orchestration_helpers import setup, run, ScenarioLLM, diagnosis, critique


class GroundingTests(unittest.TestCase):
    def test_invalid_initial_grounding_never_reaches_critic(self):
        fixture = setup()
        bad = diagnosis()
        bad.primary_hypothesis.rationale.evidence_refs = ['invented']
        fixture[0].diagnostic.run = Mock(return_value=Generation(parsed=bad, attempts=1))
        fixture[0].critic.run = Mock()
        state = run(fixture)
        self.assertEqual(state.failure.code, 'grounding_failure')
        self.assertEqual(state.diagnostics, ())
        self.assertEqual(state.current_stage, 'TERMINAL_FAILURE')
        fixture[0].critic.run.assert_not_called()
        self.assertTrue(any(v.grounding_valid is False for v in state.validations))
        self.assertNotIn('grounding_passed', [e.event for e in state.transition_history])

    def test_invalid_revised_grounding_never_committed(self):
        fixture = setup(ScenarioLLM(critiques=[critique('revision_required')]))
        bad = diagnosis(1)
        bad.primary_hypothesis.rationale.evidence_refs = ['invented']
        fixture[0].diagnostic.run = Mock(side_effect=[Generation(parsed=diagnosis(), attempts=1), Generation(parsed=bad, attempts=1)])
        state = run(fixture)
        self.assertEqual(state.failure.code, 'grounding_failure')
        self.assertEqual((len(state.diagnostics), len(state.critiques), state.revision_count), (1, 1, 1))

    def test_unvalidated_schema_objects_rejected(self):
        fixture = setup()
        bad = DiagnosticResult.model_construct(**{**diagnosis().model_dump(), 'reasoning_summary': 123})
        fixture[0].diagnostic.run = Mock(return_value=Generation(parsed=bad, attempts=1))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'schema_validation_failure')
        self.assertEqual(state.diagnostics, ())

    def test_abstention_with_nonempty_inventory_rejected(self):
        fixture = setup()
        bad = diagnosis(abstain=True)
        bad.patient_case_evidence = ['case:' + fixture[3].retrieve.return_value[0]['patient_id']]
        fixture[0].diagnostic.run = Mock(return_value=Generation(parsed=bad, attempts=1))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'grounding_failure')
        self.assertEqual(state.critiques, ())

    def test_provider_grounding_repairs_exhausted_is_not_clinical_revision(self):
        bad = diagnosis()
        bad.primary_hypothesis.rationale.evidence_refs = ['invented']
        state = run(setup(ScenarioLLM([bad])))
        self.assertEqual(state.failure.code, 'agent_validation_failure')
        self.assertEqual(state.revision_count, 0)
        self.assertEqual(state.provider_repairs, 1)
        self.assertEqual(state.critiques, ())

    def test_invalid_critic_cannot_route(self):
        fixture = setup()
        bad = critique(supported_points=[{'statement': 'Unsupported', 'evidence_refs': ['invented']}])
        fixture[0].critic.run = Mock(return_value=Generation(parsed=bad, attempts=1))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'grounding_failure')
        self.assertEqual(state.critiques, ())
        self.assertNotIn('route_started', [e.event for e in state.transition_history])

    def test_invalid_patient_copy_stops_before_evidence(self):
        fixture = setup()
        real = fixture[0].patient.run(fixture[1].patient, fixture[1].patient_id)
        real.parsed.age = 999
        fixture[0].patient.run = Mock(return_value=real)
        state = run(fixture)
        self.assertEqual(state.failure.code, 'patient_validation_failure')
        self.assertIsNone(state.patient_state)
        fixture[3].retrieve.assert_not_called()

    def test_invalid_evidence_labels_and_provenance_rejected(self):
        for index, field in ((3, 'PATHOLOGY'), (4, 'unexpected_provenance')):
            fixture = setup()
            fixture[index].retrieve.return_value[0][field] = 'SECRET'
            state = run(fixture)
            self.assertEqual(state.failure.code, 'invalid_evidence_or_retrieval_failure')
            self.assertIsNone(state.evidence)
            self.assertEqual(len(fixture[2].calls), 1)
            self.assertNotIn('SECRET', state.model_dump_json())

    def test_duplicate_evidence_ids_rejected(self):
        fixture = setup()
        fixture[3].retrieve.return_value *= 2
        state = run(fixture)
        self.assertEqual(state.failure.code, 'invalid_evidence_or_retrieval_failure')
        self.assertEqual(state.diagnostics, ())

    def test_source_categories_cannot_be_swapped(self):
        fixture = setup()
        patient_hits, medical_hits = fixture[3].retrieve.return_value, fixture[4].retrieve.return_value
        fixture[3].retrieve.return_value, fixture[4].retrieve.return_value = medical_hits, patient_hits
        self.assertEqual(run(fixture).failure.code, 'invalid_evidence_or_retrieval_failure')

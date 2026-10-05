"""Software regression tests, not validation of clinical entailment."""
import json
import unittest
from unittest.mock import Mock
from rag.agents.grounding import facts, validate_references
from rag.agents.models import EvidenceClaim
from orchestration.phase6.diagnostic import diagnostic_context, ground, DIAGNOSTIC
from orchestration.phase6.reference_entailment import (
    patient_evidence_inventory, inspect_reference_consistency, validate_reference_consistency)
from tests.test_focused_medical import inferred, enhanced
from tests.safety_helpers import SafetyScenarioLLM, run, generation
from tests import test_focused_medical


class ReferenceEntailmentTests(unittest.TestCase):
    def state(self):
        state = test_focused_medical.FocusedMedicalTests().state()
        state.symptoms = ['Unmapped question = Yes'] * 10 + [
            'How precisely is your pain located? = 4',
            'How fast did the pain appear? = 8',
            'Do you have shortness of breath? = Yes',
            'Do you feel choking? = Yes',
            'Do you feel like you are dying? = Yes',
            'Do you have palpitations? = Yes',
            'Do you have tingling in your limbs and around your mouth? = Yes',
            'Do you feel detached from your surroundings? = Yes']
        state.presenting_evidence = []
        return state

    def diagnosis(self, statement, *indices):
        value = inferred()
        value.primary_hypothesis.rationale.statement = 'Uncertain model inference: ' + statement
        value.primary_hypothesis.rationale.evidence_refs = [f'patient:symptoms:{i}' for i in indices]
        return value

    def test_inventory_exact_mapping_including_ordinal_null_and_duplicates(self):
        state = self.state()
        state.age = None
        inventory = patient_evidence_inventory(state)
        self.assertEqual({r['evidence_id']: r['exact_fact'] for r in inventory}, facts(state))
        onset = next(r for r in inventory if r['evidence_id'] == 'patient:symptoms:11')
        self.assertEqual(onset['meaning'], 'How fast did the pain appear?')
        self.assertEqual(onset['value'], '8')
        self.assertEqual(len(inventory), len(facts(state)))
        self.assertIsNone(inventory[0]['value'])
        self.assertEqual(diagnostic_context(state, [], [])['PATIENT_EVIDENCE_INVENTORY'], inventory)
        self.assertIn('Never infer the meaning of an evidence ID from its numeric index', DIAGNOSTIC)

    def test_existence_pass_does_not_imply_consistency(self):
        value = self.diagnosis('The patient has dyspnea and tingling.', 11, 15)
        # Historical guard checks existence, not symptom meaning (supply real medical evidence).
        from rag.agents.grounding import evidence_from_hits
        from tests.safety_helpers import medical_hit
        validate_references(value, self.state(), [], evidence_from_hits([medical_hit()], 'medical_knowledge'))
        with self.assertRaisesRegex(ValueError, 'patient_reference_concept_mismatch'):
            ground(value, self.state(), [], [])

    def test_corrected_mapping_and_synonyms_pass(self):
        for statement, indices in [('The patient has dyspnea and tingling.', (12, 16)),
                                   ('Breathing difficulty and choking are reported.', (12, 13)),
                                   ('Palpitations, fear of dying and detachment are reported.', (15, 14, 17)),
                                   ('Reported paresthesia.', (16,))]:
            with self.subTest(statement=statement):
                audit = validate_reference_consistency(self.diagnosis(statement, *indices), self.state())
                self.assertTrue(audit['checks'])
                self.assertNotEqual(audit['status'], 'mismatch')

    def test_each_claim_checked_and_other_claims_cannot_supply_fact(self):
        value = self.diagnosis('The patient has dyspnea.', 12)
        value.primary_hypothesis.supporting_evidence = [EvidenceClaim(statement='Tingling is reported.', evidence_refs=['patient:symptoms:15'])]
        with self.assertRaises(ValueError):
            ground(value, self.state(), [], [])
        value.primary_hypothesis.supporting_evidence = []
        value.primary_hypothesis.contradicting_evidence = [EvidenceClaim(statement='No dyspnea.', evidence_refs=['patient:symptoms:12'])]
        with self.assertRaisesRegex(ValueError, 'polarity_mismatch'):
            ground(value, self.state(), [], [])

    def test_differential_claims_and_abstention_are_checked(self):
        value = self.diagnosis('Patient has dyspnea.', 12)
        differential = value.primary_hypothesis.model_copy(deep=True)
        differential.rationale = EvidenceClaim(statement='Patient has tingling.', evidence_refs=['patient:symptoms:15'])
        value.differential_diagnoses = [differential]
        with self.assertRaises(ValueError):
            ground(value, self.state(), [], [])
        from tests.test_focused_medical import abstention
        ground(abstention(), self.state(), [], [])

    def test_demographic_and_external_refs_do_not_supply_patient_observations(self):
        value = self.diagnosis('The patient has dyspnea.', 11)
        value.primary_hypothesis.rationale.evidence_refs = ['patient:age', 'medical:unrelated']
        with self.assertRaisesRegex(ValueError, 'concept_mismatch'):
            validate_reference_consistency(value, self.state())

    def test_negative_unknown_ordinal_not_affirmative(self):
        for answer in ('No', 'unknown', '8'):
            state = self.state()
            state.symptoms[12] = 'Shortness of breath? = ' + answer
            with self.assertRaises(ValueError):
                ground(self.diagnosis('Patient has dyspnea.', 12), state, [], [])
        state.symptoms[12] = 'Shortness of breath? = No'
        ground(self.diagnosis('Patient denies dyspnea.', 12), state, [], [])
        state.symptoms[12] = 'Shortness of breath? = unknown'
        ground(self.diagnosis('Dyspnea is unknown.', 12), state, [], [])

    def test_citation_cannot_be_guessed_after_reordering(self):
        state = self.state()
        state.symptoms[12], state.symptoms[15] = state.symptoms[15], state.symptoms[12]
        with self.assertRaises(ValueError):
            ground(self.diagnosis('Patient has dyspnea.', 12), state, [], [])
        ground(self.diagnosis('Patient has dyspnea.', 15), state, [], [])

    def test_complex_or_unrecognized_prose_explicitly_unassessed(self):
        for text in ('This condition may cause dyspnea.', 'A highly unusual description.'):
            audit = inspect_reference_consistency(self.diagnosis(text, 11), self.state())
            self.assertEqual(audit['status'], 'unassessed')
        audit = inspect_reference_consistency(self.diagnosis('Dyspnea is observed, not proof of this hypothesis.', 12), self.state())
        self.assertEqual(audit['status'], 'no_mismatch_in_checked_observations')

    def test_real_provider_repairs_before_critic_with_same_inventory(self):
        bad = inferred()
        bad.primary_hypothesis.rationale.statement = 'Uncertain model inference: patient has dyspnea.'
        llm = SafetyScenarioLLM(diagnoses=[bad, inferred()])
        state = run(enhanced(llm))
        self.assertEqual(state.status, 'human_review_required')
        self.assertEqual(state.provider_repairs, 1)
        self.assertEqual(llm.counts['ClinicalCritique'], 1)
        self.assertEqual(len(state.diagnostics), 1)
        requests = [c for c in llm.calls if c['text']['format']['schema']['title'] == 'DiagnosticResult']
        self.assertEqual(requests[0]['input'], requests[1]['input'])
        payload = json.loads(requests[0]['input'][0]['content'])
        self.assertIn('PATIENT_EVIDENCE_INVENTORY', payload)

    def test_demo_distinguishes_partial_consistency_from_entailment(self):
        import io
        from demo.terminal import Console, show_result
        state = run(enhanced())
        output = io.StringIO()
        show_result(Console(output, detailed=True), state, 1)
        text = output.getvalue()
        self.assertIn('Recomputed bounded patient-reference audit', text)
        self.assertIn('no_mismatch_in_checked_observations', text)
        self.assertIn('Unassessed clauses/claims', text)
        self.assertIn('not general semantic entailment', text)
        self.assertIn('fever: consistent', text)

    def test_exhausted_repairs_do_not_reach_critic_or_safety(self):
        bad = inferred()
        bad.primary_hypothesis.rationale.statement = 'Uncertain model inference: patient has dyspnea.'
        llm = SafetyScenarioLLM(diagnoses=[bad])
        state = run(enhanced(llm))
        self.assertEqual(state.status, 'failed')
        self.assertEqual(state.diagnostics, ())
        self.assertEqual(llm.counts['ClinicalCritique'], 0)
        self.assertEqual(llm.safety_count, 0)
        self.assertEqual(state.failure.code, 'agent_validation_failure')

    def test_application_gate_catches_provider_which_skips_callback(self):
        fixture = enhanced()
        bad = inferred()
        bad.primary_hypothesis.rationale.statement = 'Uncertain model inference: patient has dyspnea.'
        fixture[0].diagnostic = Mock()
        fixture[0].diagnostic.run.return_value = generation(bad)
        state = run(fixture)
        self.assertEqual(state.status, 'failed')
        self.assertEqual(state.failure.code, 'grounding_failure')
        self.assertEqual(state.diagnostics, ())
        self.assertEqual(fixture[2].counts['ClinicalCritique'], 0)


if __name__ == '__main__':
    unittest.main()

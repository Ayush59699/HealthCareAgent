"""Offline management contracts, not clinical treatment-quality validation."""
import copy
import json
import unittest
from unittest.mock import Mock
from application.decision import build_final_decision
from application.human_review import review_context
from orchestration.evidence import fingerprint
from orchestration.phase6.state import Phase6WorkflowState
from orchestration.phase6.treatment import validate_treatment
from rag.agents.models import DiagnosticResult, TreatmentPlan
from tests.amg_helpers import amg_hit
from tests.test_final_decision import fixture, run
from tests.safety_helpers import SafetyScenarioLLM, critique, semantic, generation
from tests.phase4_helpers import envelope


REF = 'medical:' + amg_hit()['chunk_id']


def plan():
    return TreatmentPlan(status='proposed', actions=[{
        'category': 'evaluation', 'proposal': {
            'statement': 'Conditional synthetic management consideration for clinician review only.',
            'evidence_refs': ['patient:symptoms:0', REF]}}],
        patient_case_evidence=[], medical_knowledge_evidence=[REF],
        missing_information=['Independent clinical assessment unavailable.'],
        uncertainty=['Synthetic fixture does not establish clinical support.'])


def managed(provider=None, output=None):
    parts = fixture(provider)
    original = parts[2].client.responses.create.side_effect
    def respond(**request):
        if request['text']['format']['schema']['title'] == 'TreatmentPlan':
            parts[2].calls.append(copy.deepcopy(request))
            value = plan() if output is None else output
            if hasattr(value, 'model_dump'):
                value = value.model_dump()
            return envelope(json.dumps(value))
        return original(**request)
    parts[2].client.responses.create.side_effect = respond
    return parts


def calls(parts, title):
    return [c for c in parts[2].calls if c['text']['format']['schema']['title'] == title]


class TreatmentFlowTests(unittest.TestCase):
    def test_separate_call_order_full_context_and_preserved_diagnosis(self):
        parts = managed()
        original = parts[2].diagnoses[0].model_dump()
        state = run(parts)
        self.assertEqual(state.status, 'final')
        names = [c['text']['format']['schema']['title'] for c in parts[2].calls]
        self.assertEqual(names[-4:], ['DiagnosticResult', 'TreatmentPlan', 'ClinicalCritique', 'SemanticSafetyResult'])
        request = calls(parts, 'TreatmentPlan')[0]
        payload = json.loads(request['input'][0]['content'])
        self.assertEqual(payload['diagnostic_output'], original)
        self.assertEqual(payload['patient_state'], state.patient_state.model_dump())
        self.assertIn('PATIENT_EVIDENCE_INVENTORY', payload)
        self.assertEqual(payload['MEDICAL_KNOWLEDGE_EVIDENCE']['AMG']['accepted_passages'],
                         [e.model_dump() for e in state.evidence.agent_evidence()[1]])
        self.assertEqual(request['model'], parts[2].config.model)
        accepted = state.diagnostics[-1].result.model_dump()
        self.assertEqual(accepted.pop('treatment'), plan().model_dump())
        self.assertEqual(accepted, original)
        parts[3].retrieve.assert_called_once()
        parts[4].retrieve.assert_called_once()

    def test_critic_grounding_safety_final_and_human_review_receive_same_plan(self):
        parts = managed(SafetyScenarioLLM(safety_results=[semantic('safety_ambiguity')]))
        state = run(parts)
        final, context = review_context(state)
        expected = plan().model_dump()
        critic = json.loads(calls(parts, 'ClinicalCritique')[0]['input'][0]['content'])
        safety = json.loads(calls(parts, 'SemanticSafetyResult')[0]['input'][0]['content'])
        self.assertEqual(critic['diagnostic_output']['treatment'], expected)
        self.assertEqual(safety['diagnostic']['treatment'], expected)
        self.assertEqual(context['final_ai_proposal_and_reasoning']['treatment'], expected)
        link = next(c for c in final.grounding_result.claims if c.claim_path.startswith('/treatment/'))
        self.assertEqual(link.medical_refs, [REF])
        self.assertEqual(link.patient_refs, ['patient:symptoms:0'])
        self.assertEqual(link.medical_support, 'reference_present_not_verified')
        self.assertEqual(final.status, 'HUMAN_REVIEW')
        self.assertIsNone(final.diagnostic_proposal)
        self.assertFalse(final.human_approval)

    def test_treatment_only_citations_in_final_evidence_without_rewriting_diagnosis(self):
        parts = managed()
        diagnosis = parts[2].diagnoses[0]
        diagnosis.primary_hypothesis.rationale.statement = 'Uncertain model inference: observations are not proof.'
        diagnosis.primary_hypothesis.rationale.evidence_refs = ['patient:symptoms:0']
        diagnosis.medical_knowledge_evidence = []
        diagnosis.missing_information = ['Clinical confirmation unavailable.']
        state = run(parts)
        final = build_final_decision(state)
        self.assertEqual(state.diagnostics[-1].result.medical_knowledge_evidence, [])
        self.assertEqual([e.source_id for e in final.supporting_medical_evidence], [REF])
        self.assertEqual(final.status, 'HUMAN_REVIEW')
        self.assertIn('missing_medical_reference', [f.code for f in final.safety_findings])

    def test_unsafe_management_blocks_with_nested_safety_anchor(self):
        text = plan().actions[0].proposal.statement
        parts = managed(SafetyScenarioLLM(safety_results=[semantic('prohibited_action',
            status='issue_identified', path='/treatment/actions/0/proposal/statement', quote=text)]))
        final = build_final_decision(run(parts))
        self.assertEqual(final.status, 'BLOCK')
        self.assertIsNone(final.diagnostic_proposal)
        self.assertEqual(final.safety_findings[0].anchor.field_path, '/treatment/actions/0/proposal/statement')

    def test_critic_block_withholds_management_and_skips_semantic_call(self):
        parts = managed(SafetyScenarioLLM(critiques=[critique(safety_flags=['Unsafe fixture management'])]))
        state = run(parts)
        self.assertEqual(build_final_decision(state).status, 'BLOCK')
        self.assertFalse(calls(parts, 'SemanticSafetyResult'))
        self.assertEqual(state.diagnostics[-1].result.treatment, plan())

    def test_serialization_fingerprints_and_tamper_rejection(self):
        state = run(managed())
        restored = Phase6WorkflowState.model_validate_json(state.model_dump_json())
        self.assertEqual(restored, state)
        self.assertEqual(restored.diagnostics[-1].fingerprint, fingerprint(restored.diagnostics[-1].result.model_dump()))
        self.assertEqual(build_final_decision(restored).diagnostic_proposal.treatment, plan())
        restored.diagnostics[-1].result.treatment.actions[0].proposal.statement += ' changed'
        with self.assertRaises(ValueError):
            build_final_decision(restored)

    def test_deferred_management_is_explicit_not_missing(self):
        state = run(fixture())
        self.assertEqual(state.status, 'final')
        self.assertEqual(state.diagnostics[-1].result.treatment.status, 'deferred')
        self.assertEqual(state.diagnostics[-1].result.treatment.actions, [])

    def test_no_medical_evidence_skips_management(self):
        parts = managed()
        parts[4].retrieve.return_value = []
        self.assertEqual(run(parts).status, 'abstained')
        self.assertFalse(calls(parts, 'TreatmentPlan'))

    def test_invalid_diagnosis_never_reaches_management(self):
        parts = managed()
        parts[2].diagnoses[0].primary_hypothesis.rationale.evidence_refs = ['invented']
        self.assertEqual(run(parts).status, 'failed')
        self.assertFalse(calls(parts, 'TreatmentPlan'))

    def test_invalid_treatment_fails_closed_with_accounted_repair(self):
        parts = managed(output={'status': 'proposed'})
        state = run(parts)
        self.assertEqual(state.status, 'failed')
        self.assertEqual(len(calls(parts, 'TreatmentPlan')), 2)
        self.assertEqual(state.total_requests, len(parts[2].calls))
        self.assertEqual(state.invocations[-1].ticket.stage, 'TREATMENT')
        self.assertFalse(calls(parts, 'ClinicalCritique'))
        self.assertEqual(build_final_decision(state).status, 'BLOCK')
        self.assertIn('TREATMENT', state.stage_seconds)

    def test_application_revalidates_provider_bypassing_callback(self):
        parts = managed()
        invalid = plan()
        invalid.actions[0].proposal.evidence_refs = ['invented', REF]
        parts[0].treatment.run = Mock(return_value=generation(invalid))
        state = run(parts)
        self.assertEqual(state.status, 'failed')
        self.assertFalse(state.critiques)

    def test_management_regenerated_on_revision_no_retrieval(self):
        parts = managed(SafetyScenarioLLM(critiques=[critique('revision_required')]))
        state = run(parts)
        self.assertGreater(len(state.diagnostics), 1)
        self.assertEqual(len(calls(parts, 'TreatmentPlan')), len(state.diagnostics))
        self.assertEqual(state.total_requests, len(parts[2].calls))
        parts[3].retrieve.assert_called_once()
        parts[4].retrieve.assert_called_once()

    def test_budget_covers_management_call(self):
        from orchestration.phase6.policy import Phase6Policy
        parts = managed()
        parts[0].policy = Phase6Policy(max_requests=4)
        state = run(parts)
        self.assertEqual(state.failure.code, 'request_budget_exhausted')
        self.assertFalse(calls(parts, 'TreatmentPlan'))
        self.assertFalse(state.critiques)


class TreatmentValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state = run(fixture())
        cls.patient = state.patient_state
        cls.cases, cls.medical = state.evidence.agent_evidence()

    def validate(self, value):
        return validate_treatment(value, self.patient, self.cases, self.medical)

    def test_bad_references_and_inventories_rejected(self):
        for refs in (['patient:invented', REF], ['patient:symptoms:0'], [REF], ['case:invented', REF]):
            with self.subTest(refs=refs):
                value = plan()
                value.actions[0].proposal.evidence_refs = refs
                with self.assertRaises(ValueError):
                    self.validate(value)
        value = plan()
        value.medical_knowledge_evidence = []
        with self.assertRaises(ValueError):
            self.validate(value)

    def test_wrong_patient_polarity_and_fabricated_url_rejected(self):
        for statement in ('No fever.', 'See https://invented.example/guidance'):
            with self.subTest(statement=statement):
                value = plan()
                value.actions[0].proposal.statement = statement
                with self.assertRaises(ValueError):
                    self.validate(value)

    def test_schema_status_missingness_and_extra_fields(self):
        for updates in ({'status': 'deferred'}, {'actions': []}, {'uncertainty': []},
                        {'missing_information': [' ']}, {'approval': True}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                TreatmentPlan.model_validate({**plan().model_dump(), **updates})

    def test_legacy_serialization_and_strict_cloud_schema(self):
        original = fixture()[2].diagnoses[0]
        data = original.model_dump()
        self.assertNotIn('treatment', data)
        self.assertEqual(DiagnosticResult.model_validate(data).model_dump(), data)
        schema = DiagnosticResult.model_json_schema()
        self.assertEqual(set(schema['required']), set(schema['properties']))
        self.assertNotIn('default', schema['properties']['treatment'])


if __name__ == '__main__':
    unittest.main()

"""Offline agent/handoff contracts, not an evaluation of clinical correctness."""
import json
import unittest
from dataclasses import replace
from application.workflow import create_workflow
from application.decision import FinalDecision, build_final_decision
from orchestration.evidence import fingerprint
from orchestration.phase6.grounding import grounding_report
from orchestration.phase6.diagnostic import CRITIC_PROMPT_VERSION
from rag.agents.clinical import PatientAgent
from rag.agents.grounding import patient_input, validate_patient, state_from_patient
from rag.models import Evidence, PatientRepresentation
from tests.safety_helpers import setup, SafetyScenarioLLM, semantic, critique
from tests.amg_helpers import configure_medical, configure_provider
from tests.phase4_helpers import MockLLM


def fixture(provider=None):
    _, case, llm, patients, medical = setup(provider)
    configure_medical(medical)
    configure_provider(llm)
    return create_workflow(llm, patients, medical), case, llm, patients, medical


def run(parts):
    return parts[0].run(parts[1].patient, parts[1].patient_id)


class PatientPreservationTests(unittest.TestCase):
    def test_agent_preserves_qualifiers_negation_unknown_numeric_history(self):
        patient = PatientRepresentation(None, None, (
            Evidence('E_1', 'Pain only when walking for two days?', 'Yes', False),
            Evidence('E_2', 'Fever?', 'No', False),
            Evidence('E_3', 'Onset scale?', '3', False),
            Evidence('E_4', 'Location?', None, False)), (
            Evidence('E_5', 'Taking medication in the last week?', 'Yes', True),))
        llm = MockLLM()
        result = PatientAgent(llm).run(patient, 'ddxplus:validate:1')
        self.assertIsNone(result.failure)
        validate_patient(result.parsed, patient_input(patient, 'ddxplus:validate:1'))
        self.assertEqual(result.parsed.symptoms, [e.text for e in patient.symptoms])
        self.assertEqual(result.parsed.antecedents, [e.text for e in patient.antecedents])
        self.assertIsNone(result.parsed.age)
        self.assertIn('answer not supplied', result.parsed.symptoms[-1])

    def test_invented_patient_fact_rejected(self):
        case = fixture()[1]
        state = state_from_patient(case.patient, case.patient_id)
        state.symptoms.append('Invented observation')
        with self.assertRaises(ValueError):
            validate_patient(state, patient_input(case.patient, case.patient_id))

    def test_no_labels_in_any_agent_or_final_output(self):
        parts = fixture()
        state = run(parts)
        for text in [json.dumps(parts[2].calls), build_final_decision(state).model_dump_json()]:
            self.assertNotIn('SECRET_DIAGNOSIS', text)
            self.assertNotIn('PATHOLOGY', text)
            self.assertNotIn('ground_truth_pathology', text)


class GroundingHandoffTests(unittest.TestCase):
    def test_report_is_exact_critic_input_with_snapshot_identity(self):
        parts = fixture()
        state = run(parts)
        final = build_final_decision(state)
        call = next(c for c in parts[2].calls if c['text']['format']['schema']['title'] == 'ClinicalCritique')
        payload = json.loads(call['input'][0]['content'])
        self.assertEqual(payload['GROUNDING_RESULT'], final.grounding_result.model_dump())
        self.assertEqual(final.grounding_result.evidence_snapshot_id, state.evidence.snapshot_id)
        self.assertEqual(final.grounding_result.diagnostic_fingerprint, state.diagnostics[-1].fingerprint)
        self.assertIn(CRITIC_PROMPT_VERSION, call['instructions'])
        self.assertEqual(final.grounding_result.medical_entailment, 'not_established')
        self.assertEqual(final.grounding_result.claims[0].medical_support, 'reference_present_not_verified')

    def test_patient_only_inference_remains_unsupported_and_requires_review(self):
        parts = fixture()
        diagnosis = parts[2].diagnoses[0]
        diagnosis.primary_hypothesis.rationale.statement = 'Uncertain model inference: observation is not proof.'
        diagnosis.primary_hypothesis.rationale.evidence_refs = ['patient:symptoms:0']
        diagnosis.medical_knowledge_evidence = []
        diagnosis.missing_information = ['Independent medical confirmation unavailable.']
        final = build_final_decision(run(parts))
        self.assertEqual(final.status, 'HUMAN_REVIEW')
        self.assertEqual(final.supporting_medical_evidence, [])
        self.assertEqual(final.grounding_result.claims[0].patient_refs, ['patient:symptoms:0'])
        self.assertEqual(final.grounding_result.claims[0].medical_support, 'unsupported_by_medical_evidence')
        self.assertEqual(final.grounding_result.hypotheses_without_medical_references, ['/primary_hypothesis'])
        self.assertIn('missing_medical_reference', [f.code for f in final.safety_findings])
        self.assertEqual(final.supporting_patient_evidence, {'patient:symptoms:0': 'Fever? = Yes'})

    def test_case_analogies_do_not_become_current_patient_facts(self):
        parts = fixture()
        diagnosis = parts[2].diagnoses[0]
        ref = 'case:' + parts[3].retrieve.return_value[0]['patient_id']
        diagnosis.primary_hypothesis.rationale.evidence_refs.append(ref)
        diagnosis.patient_case_evidence = [ref]
        final = build_final_decision(run(parts))
        self.assertEqual(final.supporting_patient_evidence, {})
        self.assertEqual(final.analogous_case_evidence[0].source_id, ref)
        self.assertEqual(final.grounding_result.claims[0].analogous_case_refs, [ref])

    def test_unknown_citation_fails_without_a_passing_report(self):
        state = run(fixture())
        cases, medical = state.evidence.agent_evidence()
        diagnosis = state.diagnostics[-1].result.model_copy(deep=True)
        diagnosis.primary_hypothesis.rationale.evidence_refs = ['medical:invented']
        with self.assertRaises(ValueError):
            grounding_report(diagnosis, state.patient_state, cases, medical)

    def test_provenance_mismatch_fails_without_a_passing_report(self):
        state = run(fixture())
        cases, medical = state.evidence.agent_evidence()
        medical[0].text = 'Altered source'
        with self.assertRaises(ValueError):
            grounding_report(state.diagnostics[-1].result, state.patient_state, cases, medical)

    def test_critic_contradictions_and_unsupported_findings_are_not_dropped(self):
        review = critique('revision_required', unsupported_points=['Unsupported test claim'],
                          contradictions=['Contradictory test observation'])
        final = build_final_decision(run(fixture(SafetyScenarioLLM(critiques=[review]))))
        self.assertEqual(final.status, 'HUMAN_REVIEW')
        self.assertEqual(final.critic_findings.unsupported_points, review.unsupported_points)
        self.assertEqual(final.critic_findings.contradictions, review.contradictions)
        self.assertIsNone(final.diagnostic_proposal)


class FinalDecisionTests(unittest.TestCase):
    def test_allow_is_gated_and_round_trippable(self):
        state = run(fixture())
        final = build_final_decision(state)
        self.assertEqual(final.status, 'ALLOW')
        self.assertFalse(final.human_review_required)
        self.assertFalse(final.human_approval)
        self.assertEqual(final.diagnostic_proposal, state.diagnostics[-1].result)
        self.assertEqual(final.workflow_fingerprint, fingerprint(state.model_dump(mode='json')))
        self.assertEqual(FinalDecision.model_validate_json(final.model_dump_json()), final)

    def test_human_review_is_pending_not_approved_or_released(self):
        state = run(fixture(SafetyScenarioLLM(safety_results=[semantic('safety_ambiguity')])))
        final = build_final_decision(state)
        self.assertEqual(final.status, 'HUMAN_REVIEW')
        self.assertEqual(final.review_state, 'pending')
        self.assertFalse(final.human_approval)
        self.assertFalse(final.human_review_scheduled)
        self.assertIsNone(final.diagnostic_proposal)
        self.assertTrue(state.diagnostics)  # reviewer audit retains the proposal

    def test_critic_safety_flag_blocks_without_extra_semantic_call(self):
        parts = fixture(SafetyScenarioLLM(critiques=[critique(safety_flags=['Unsafe test proposal'])]))
        final = build_final_decision(run(parts))
        self.assertEqual(final.status, 'BLOCK')
        self.assertIsNone(final.diagnostic_proposal)
        self.assertEqual(parts[2].safety_count, 0)
        self.assertEqual(final.safety_result.semantic_status, 'skipped_critic_block')

    def test_semantic_safety_issue_blocks(self):
        parts = fixture(SafetyScenarioLLM(safety_results=[semantic('prohibited_action', status='issue_identified')]))
        final = build_final_decision(run(parts))
        self.assertEqual(final.status, 'BLOCK')
        self.assertTrue(final.human_review_required)

    def test_no_medical_evidence_abstains_without_fake_assessment(self):
        parts = fixture()
        parts[4].retrieve.return_value = []
        final = build_final_decision(run(parts))
        self.assertEqual(final.status, 'HUMAN_REVIEW')
        self.assertEqual(final.workflow_status, 'abstained')
        self.assertIsNone(final.safety_result)
        self.assertIsNone(final.grounding_result)
        self.assertEqual(len(parts[2].calls), 1)

    def test_empty_case_retrieval_does_not_erase_current_patient(self):
        parts = fixture()
        parts[3].retrieve.return_value = []
        final = build_final_decision(run(parts))
        self.assertEqual(final.analogous_case_evidence, [])
        self.assertEqual(final.patient_context.symptoms, ['Fever? = Yes'])

    def test_no_patient_observations_do_not_invent_hypotheses(self):
        parts = fixture()
        parts = (parts[0], replace(parts[1], patient=PatientRepresentation(None, None, (), ())), *parts[2:])
        final = build_final_decision(run(parts))
        self.assertEqual(final.status, 'BLOCK')
        self.assertIsNone(final.diagnostic_proposal)
        self.assertEqual(final.patient_context.symptoms, [])

    def test_retrieval_failure_is_structured_block(self):
        parts = fixture()
        parts[4].retrieve.side_effect = RuntimeError('SECRET')
        final = build_final_decision(run(parts))
        self.assertEqual(final.status, 'BLOCK')
        self.assertEqual(final.failure.code, 'invalid_evidence_or_retrieval_failure')
        self.assertNotIn('SECRET', final.model_dump_json())

    def test_grounding_failure_is_structured_block(self):
        parts = fixture()
        parts[2].diagnoses[0].primary_hypothesis.rationale.evidence_refs = ['patient:invented:0']
        final = build_final_decision(run(parts))
        self.assertEqual(final.status, 'BLOCK')
        self.assertIsNone(final.diagnostic_proposal)
        self.assertIsNone(final.safety_result)

    def test_invalid_safety_output_cannot_release_diagnosis(self):
        final = build_final_decision(run(fixture(SafetyScenarioLLM(safety_results=[{}]))))
        self.assertEqual(final.status, 'BLOCK')
        self.assertEqual(final.safety_coverage, 'failed')
        self.assertIsNone(final.diagnostic_proposal)

    def test_revisions_and_projection_do_not_repeat_retrieval(self):
        parts = fixture(SafetyScenarioLLM(critiques=[critique('revision_required')]))
        state = run(parts)
        calls = len(parts[2].calls)
        final = build_final_decision(state)
        self.assertEqual(final.status, 'HUMAN_REVIEW')
        self.assertEqual(len(parts[2].calls), calls)
        parts[3].retrieve.assert_called_once()
        parts[4].retrieve.assert_called_once()

    def test_mutated_patient_cannot_be_released(self):
        state = run(fixture())
        state.patient_state.symptoms.append('Invented fact')
        with self.assertRaises(ValueError):
            build_final_decision(state)

    def test_mutated_safety_policy_result_cannot_be_released(self):
        state = run(fixture())
        assessment = state.safety_assessments[-1].model_copy(update={'reasons': ('forged',)})
        state = state.model_copy(update={'safety_assessments': (assessment,)})
        with self.assertRaises(ValueError):
            build_final_decision(state)

    def test_continue_does_not_override_critic_routing(self):
        state = run(fixture(SafetyScenarioLLM(critiques=[critique('revision_required')])))
        forged = state.model_copy(update={'status': 'final', 'current_stage': 'FINAL', 'outcome': 'accepted_with_limitations'})
        with self.assertRaises(ValueError):
            build_final_decision(forged)

    def test_nonterminal_state_is_not_a_final_decision(self):
        state = run(fixture()).model_copy(update={'status': 'running'})
        with self.assertRaises(ValueError):
            build_final_decision(state)

    def test_withheld_contract_rejects_embedded_diagnostic_release(self):
        state = run(fixture(SafetyScenarioLLM(critiques=[critique(safety_flags=['Unsafe test'])])))
        data = build_final_decision(state).model_dump()
        data['diagnostic_proposal'] = state.diagnostics[-1].result.model_dump()
        with self.assertRaises(ValueError):
            FinalDecision.model_validate(data)

    def test_handoff_is_detached_from_workflow(self):
        state = run(fixture())
        before = state.model_dump_json()
        final = build_final_decision(state)
        final.patient_context.symptoms.append('Consumer mutation')
        final.diagnostic_proposal.uncertainty.append('Consumer mutation')
        self.assertEqual(state.model_dump_json(), before)


class SavedDecisionTests(unittest.TestCase):
    def test_runner_writes_verifiable_handoff_and_separate_audit(self):
        import tempfile
        from pathlib import Path
        from tests.test_phase6_reporting import Phase6ReportingTests
        from orchestration.phase6.state import Phase6WorkflowState
        from scripts.verify_amg_run import verify_final_decision
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            status = Phase6ReportingTests().invoke(output,
                SafetyScenarioLLM(safety_results=[semantic('safety_ambiguity')]))
            self.assertEqual(status, 0)
            state = Phase6WorkflowState.model_validate_json((output / 'sample_case_001.json').read_text())
            result = verify_final_decision(state, output / 'final_decision_001.json')
            self.assertTrue(result['final_decision_verified'])
            self.assertEqual(result['final_decision_status'], 'HUMAN_REVIEW')
            self.assertTrue(state.diagnostics)
            report = json.loads((output / 'phase6_run_report.json').read_text())
            self.assertEqual(report['cases'][0]['final_decision_file'], 'final_decision_001.json')
            self.assertEqual(report['critic_prompt_version'], CRITIC_PROMPT_VERSION)

    def test_saved_handoff_tampering_is_rejected(self):
        import tempfile
        from pathlib import Path
        from scripts.verify_amg_run import verify_final_decision
        state = run(fixture())
        final = build_final_decision(state)
        final.uncertainty.append('Tampered content')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'final.json'
            path.write_text(final.model_dump_json(), encoding='utf8')
            with self.assertRaises(ValueError):
                verify_final_decision(state, path)

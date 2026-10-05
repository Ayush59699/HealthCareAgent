"""Deterministic retrieval/inference regression tests, not clinical validation."""
from dataclasses import asdict, replace
import io
import json
import unittest
from unittest.mock import Mock
from rag.agents.grounding import evidence_from_hits, validate_references
from rag.agents.models import PatientState
from rag.focused_medical import (FocusedEvidenceService, concepts, query_plan,
                                baseline_filter_candidates as filter_candidates, state_from_patient)
from rag.medical_ingestion.chunker import chunk_document
from rag.models import PatientRepresentation, Evidence
from orchestration.phase6.diagnostic import DIAGNOSTIC, ground
from tests.test_medical_rag import document
from tests.test_patient_rag import record
from tests.safety_helpers import setup, run, SafetyScenarioLLM, diagnosis, critique, semantic


def hit(title, text, score=0.9):
    doc = replace(document(), title=title, text=text)
    return {**asdict(next(chunk_document(doc))), 'score': score}


def inferred():
    value = diagnosis(medical_reference=False)
    value.primary_hypothesis.rationale.statement = 'Uncertain model inference: fever is an observation, not proof of this hypothesis.'
    value.primary_hypothesis.rationale.evidence_refs = ['patient:symptoms:0']
    value.uncertainty = ['Unconfirmed possibility; no relevant medical support.']
    value.missing_information = ['Symptom duration, vital signs and examination findings.']
    return value


def abstention():
    value = diagnosis(abstain=True)
    value.uncertainty = ['Insufficient patient observations for even a cautious hypothesis.']
    value.missing_information = ['Presenting symptoms, duration and vital signs.']
    return value


def enhanced(llm=None, medical=None):
    fixture = setup(llm or SafetyScenarioLLM(diagnoses=[inferred()]), enhanced_medical=True)
    fixture[4].retrieve.return_value = medical or []
    return fixture


class FocusedMedicalTests(unittest.TestCase):
    def state(self):
        return PatientState(patient_id='ddxplus:validate:2', age=10, sex='F',
            presenting_evidence=['Do you feel palpitations? = Yes'],
            symptoms=['Do you have chest pain? = Yes', 'Difficulty breathing? = Yes',
                      'Have you had sweating? = Yes'],
            antecedents=['Have you had asthma? = Yes', 'Have you traveled abroad? = No'],
            relevant_findings=[], missing_information=[], uncertainty_notes=[])

    def test_focused_query_contains_cluster_and_history_not_id(self):
        plan = query_plan(self.state())
        for term in ('pediatric', 'palpitations', 'chest pain', 'shortness of breath', 'sweating', 'asthma', 'red flags'):
            self.assertIn(term, plan['query'])
        for term in ('ddxplus', 'patient_id', 'traveled', 'Have you', '= Yes'):
            self.assertNotIn(term, plan['query'])
        self.assertEqual(plan, query_plan(self.state()))

    def test_not_hardcoded_to_example(self):
        value = state_from_patient(record().patient, record().patient_id)
        plan = query_plan(value)
        self.assertIn('fever', plan['query'])
        self.assertNotIn('chest', plan['query'])
        self.assertNotIn('pediatric', plan['query'])

    def test_negative_unknown_and_ordinal_not_positive_concepts(self):
        for value in ('Fever? = No', 'Cough? = unknown', 'Pain intensity? = 7', 'Pain radiation? = nowhere'):
            self.assertEqual(concepts(value), [])

    def test_labels_and_whole_patient_records_rejected(self):
        for value in (record(), {'PATHOLOGY': 'SECRET_DIAGNOSIS'}, self.state().model_dump()):
            with self.assertRaises(TypeError):
                query_plan(value)
        with self.assertRaises(ValueError):
            PatientState.model_validate({**self.state().model_dump(), 'PATHOLOGY': 'secret'})

    def test_high_scoring_irrelevant_paper_discarded_lower_topic_retained(self):
        bad = hit('BK polyomavirus transplant', 'Kidney transplant recipients immune reactivation.', .95)
        good = hit('Asthma', 'Shortness of breath and chest tightness.', .64)
        kept, audit = filter_candidates(query_plan(self.state()), [bad, good])
        self.assertEqual(kept, [good])
        self.assertIs(kept[0], good)
        self.assertEqual(audit[0]['score'], .95)
        self.assertFalse(audit[0]['retained'])
        self.assertEqual(audit[1]['chunk_id'], good['chunk_id'])

    def test_body_requires_two_distinct_symptom_concepts(self):
        one = hit('Research note', 'Sweating sweating sweating.')
        two = hit('Clinical note', 'Sweating with palpitations.')
        kept, _ = filter_candidates(query_plan(self.state()), [one, two])
        self.assertEqual(kept, [two])

    def test_malformed_irrelevant_evidence_fails_before_filter(self):
        bad = hit('Transplant', 'Kidney transplant.')
        bad['text'] = 'Tampered'
        with self.assertRaises(ValueError):
            filter_candidates(query_plan(self.state()), [bad])

    def test_top_five_once_patient_query_unchanged(self):
        fixture = enhanced()
        state = run(fixture)
        fixture[3].retrieve.assert_called_once_with(fixture[1].patient.to_text(), top_k=1)
        fixture[4].retrieve.assert_called_once()
        self.assertEqual(fixture[4].retrieve.call_args.kwargs, {'top_k': 5})
        self.assertNotEqual(fixture[4].retrieve.call_args.args[0], fixture[1].patient.to_text())
        self.assertEqual(state.medical_retrieval['status'], 'insufficient_or_irrelevant')

    def test_no_medical_inference_reaches_critic_and_safety_but_is_withheld(self):
        fixture = enhanced()
        state = run(fixture)
        self.assertEqual(state.status, 'human_review_required')
        self.assertEqual(len(state.critiques), 1)
        self.assertEqual(fixture[2].safety_count, 1)
        self.assertEqual(state.total_requests, 4)
        self.assertIn('missing_medical_reference', state.safety_assessments[0].reasons)
        self.assertEqual(state.diagnostics[0].result.medical_knowledge_evidence, [])

    def test_rejected_candidates_never_reach_any_agent_or_inventory(self):
        bad = hit('BK polyomavirus transplant', 'Kidney transplant recipients immune reactivation.')
        fixture = enhanced(medical=[bad])
        state = run(fixture)
        self.assertFalse(state.medical_retrieval['candidates'][0]['retained'])
        self.assertEqual(state.evidence.medical_knowledge, ())
        for call in fixture[2].calls:
            text = json.dumps(call)
            for forbidden in ('BK polyomavirus', bad['chunk_id'], 'SECRET_DIAGNOSIS', 'PATHOLOGY', 'DIFFERENTIAL_DIAGNOSIS'):
                self.assertNotIn(forbidden, text)

    def test_filtered_id_cannot_be_cited(self):
        bad = hit('BK polyomavirus transplant', 'Kidney transplant.')
        value = inferred()
        value.primary_hypothesis.rationale.evidence_refs.append('medical:' + bad['chunk_id'])
        value.medical_knowledge_evidence = ['medical:' + bad['chunk_id']]
        state = run(enhanced(SafetyScenarioLLM(diagnoses=[value]), medical=[bad]))
        self.assertEqual(state.status, 'failed')
        self.assertEqual(len(state.diagnostics), 0)

    def test_abstention_with_no_patient_observations_still_safety_assessed(self):
        fixture = enhanced(SafetyScenarioLLM(diagnoses=[abstention()]))
        case = replace(fixture[1], patient=PatientRepresentation(None, None, (), ()))
        state = fixture[0].run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'abstained')
        self.assertEqual(len(state.safety_assessments), 1)
        fixture[4].retrieve.assert_not_called()

    def test_medical_text_alone_cannot_replace_missing_patient_observations(self):
        value = diagnosis()
        empty = self.state().model_copy(update={'symptoms': [], 'presenting_evidence': []})
        from tests.safety_helpers import medical_hit
        medical = evidence_from_hits([medical_hit()], 'medical_knowledge')
        with self.assertRaises(ValueError):
            ground(value, empty, [], medical)

    def test_demographics_alone_cannot_support_inference(self):
        value = inferred()
        value.primary_hypothesis.rationale.evidence_refs = ['patient:age']
        with self.assertRaises(ValueError):
            ground(value, self.state(), [], [])

    def test_inference_requires_explicit_uncertainty_and_missingness(self):
        for field in ('uncertainty', 'missing_information'):
            value = inferred()
            setattr(value, field, [])
            with self.assertRaises(ValueError):
                ground(value, self.state(), [], [])
        value = inferred()
        value.primary_hypothesis.rationale.statement = 'Confirmed diagnosis'
        with self.assertRaises(ValueError):
            ground(value, self.state(), [], [])

    def test_invented_refs_urls_and_wrong_inventory_still_fail(self):
        for kind in ('reference', 'url', 'inventory'):
            value = inferred()
            if kind == 'reference':
                value.primary_hypothesis.rationale.evidence_refs = ['patient:symptoms:999']
            elif kind == 'url':
                value.reasoning_summary = 'https://invented.example/paper'
            else:
                value.medical_knowledge_evidence = ['patient:symptoms:0']
            with self.assertRaises(ValueError):
                ground(value, self.state(), [], [])

    def test_baseline_grounding_still_rejects_no_medical_hypothesis(self):
        with self.assertRaises(ValueError):
            validate_references(inferred(), self.state(), [], [])
        # The positive-path fixture must cite an actual fever observation.
        fever_state = state_from_patient(record().patient, record().patient_id)
        ground(inferred(), fever_state, [], [])

    def test_critic_safety_flags_still_block(self):
        review = critique()
        review.safety_flags = ['Unsafe synthetic recommendation']
        fixture = enhanced(SafetyScenarioLLM(diagnoses=[inferred()], critiques=[review]))
        state = run(fixture)
        self.assertEqual(state.status, 'blocked')
        self.assertEqual(state.safety_assessments[0].semantic_status, 'skipped_critic_block')
        self.assertEqual(fixture[2].safety_count, 0)

    def test_semantic_prohibited_action_still_blocks(self):
        value = inferred()
        value.reasoning_summary = 'Unsafe synthetic recommendation'
        review = semantic('prohibited_action', status='issue_identified', quote=value.reasoning_summary)
        state = run(enhanced(SafetyScenarioLLM(diagnoses=[value], safety_results=[review])))
        self.assertEqual(state.status, 'blocked')
        self.assertEqual(state.safety_assessments[0].decision, 'BLOCK')

    def test_repeated_versions_each_get_fresh_safety_assessment(self):
        good = hit('Fever', 'Fever research information.')
        value = diagnosis()
        ref = 'medical:' + good['chunk_id']
        value.primary_hypothesis.rationale.evidence_refs = [ref]
        value.medical_knowledge_evidence = [ref]
        fixture = enhanced(SafetyScenarioLLM(diagnoses=[value], critiques=[critique('revision_required')]), medical=[good])
        state = run(fixture)
        self.assertEqual(state.status, 'unresolved')
        self.assertEqual([d.version for d in state.diagnostics], [1, 2])
        self.assertEqual([a.ticket.diagnostic_version for a in state.safety_assessments], [1, 2])
        self.assertEqual(len(state.critiques), 2)
        self.assertEqual(fixture[4].retrieve.call_count, 1)

    def test_failed_critic_has_no_fabricated_safety_coverage(self):
        state = run(enhanced(SafetyScenarioLLM(diagnoses=[inferred()], critiques=[{}])))
        self.assertEqual(state.status, 'failed')
        self.assertNotEqual(state.safety_coverage, 'assessed')
        self.assertEqual(state.safety_assessments, ())

    def test_phase5_missing_medical_still_early_abstention(self):
        from tests.orchestration_helpers import setup as setup5, run as run5
        fixture = setup5()
        fixture[4].retrieve.return_value = []
        state = run5(fixture)
        self.assertEqual(state.status, 'abstained')
        self.assertEqual(state.total_requests, 1)
        self.assertEqual(state.diagnostics, ())

    def test_prompt_preserves_baseline_and_explains_three_cases(self):
        from rag.agents.prompts import DIAGNOSTIC as baseline
        self.assertIn('No medical evidence means abstain.', baseline)
        self.assertNotIn('No medical evidence means abstain.', DIAGNOSTIC)
        for text in ('CASE A', 'CASE B', 'CASE C', 'Uncertain model inference:', 'NOT automatic abstention'):
            self.assertIn(text, DIAGNOSTIC)

    def test_demo_disclaimer_is_string_and_main_can_exit(self):
        from demo.terminal import DISCLAIMER, Console, arguments
        self.assertIsInstance(DISCLAIMER, str)
        Console(io.StringIO()).banner()
        self.assertEqual(arguments([]).medical_top_k, 5)


if __name__ == '__main__':
    unittest.main()

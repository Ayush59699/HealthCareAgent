"""DDXPlus-shaped patient quote regressions; no dataset labels or cloud calls."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from rag.config import OpenAIConfig
from rag.llm.provider import OpenAIProvider
from rag.models import Evidence, PatientRepresentation
from rag.who_selector_relevance import ContentRelevance, validate_content_review
from tests.phase4_helpers import envelope
from tests import test_check5_pipeline as pipeline_fixtures


class DDXPatientQuoteTests(unittest.TestCase):
    def setUp(self):
        # Literal observations only; no condition/benchmark answer supplied.
        self.dizzy = Evidence('fixture:dizzy', 'Do you feel slightly dizzy or lightheaded?', 'Yes', False)
        self.tired = Evidence('fixture:tired', 'Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long?', 'Yes', False)
        self.location = Evidence('fixture:location', 'Do you feel pain somewhere?', 'back of head', False)
        self.pale = Evidence('fixture:pale', 'Is your skin much paler than usual?', 'Yes', False)
        self.patient = PatientRepresentation(55, 'F', (self.dizzy, self.tired, self.location), (), (self.pale,))
        self.document = SimpleNamespace(text='WHO FACT SHEET\nTitle: Fixture\nURL: fixture\nRetrieved: fixture\n\n- headache\nExact fixture support.\n')

    def review(self, *quotes, source='Exact fixture support.'):
        return ContentRelevance(verdict='relevant', reason='distinctive_positive_cluster',
                                positive_fact_quotes=list(quotes), content_quotes=[source])

    def validate(self, *quotes, patient=None, source='Exact fixture support.'):
        value = self.review(*quotes, source=source)
        return validate_content_review(value, patient or self.patient, self.document)

    def test_full_question_yes_fact_accepted(self):
        self.assertEqual(self.validate(self.dizzy.text).positive_fact_quotes, [self.dizzy.text])

    def test_four_audited_question_answer_quotes_accepted(self):
        quotes = [self.tired.text, self.dizzy.text, self.location.text, self.pale.text]
        self.assertEqual(self.validate(*quotes).positive_fact_quotes, quotes)

    def test_all_patient_fact_groups_supported(self):
        history = Evidence('fixture:history', 'Have you travelled recently?', 'Not supplied', True)
        patient = PatientRepresentation(55, 'F', (self.dizzy,), (history,), (self.pale,))
        self.validate(self.dizzy.text, history.text, self.pale.text, patient=patient)

    def test_bare_yes_no_and_whitespace_cannot_bypass_length(self):
        for quote in ('Yes', 'No', ' Yes ', '\nYes\n'):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                self.validate(quote)

    def test_question_only_and_truncated_answer_rejected(self):
        for quote in (self.dizzy.question, self.dizzy.text[:-1], self.dizzy.question + ' = '):
            with self.subTest(quote=quote), self.assertRaisesRegex(ValueError, 'Patient support is not literal input'):
                self.validate(quote)

    def test_no_case_whitespace_or_punctuation_normalization(self):
        for quote in (self.dizzy.text.lower(), self.dizzy.text.replace(' = ', '='),
                      ' ' + self.dizzy.text, self.dizzy.text + '.', self.dizzy.text.replace(' = ', '  = ')):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                self.validate(quote)

    def test_no_to_yes_polarity_flip_rejected(self):
        negative = Evidence('fixture:negative', self.dizzy.question, 'No', False)
        patient = PatientRepresentation(55, 'F', (negative,), ())
        with self.assertRaisesRegex(ValueError, 'Patient support is not literal input'):
            self.validate(self.dizzy.text, patient=patient)
        with self.assertRaises(ValueError):
            self.validate(negative.question, patient=patient)
        # Literal provenance preserves the negative answer, not clinical support.
        self.assertEqual(self.validate(negative.text, patient=patient).positive_fact_quotes, [negative.text])

    def test_uncertainty_not_rewritten_to_affirmative(self):
        uncertain = Evidence('fixture:unknown', self.dizzy.question, 'Not sure', False)
        patient = PatientRepresentation(55, 'F', (uncertain,), ())
        self.validate(uncertain.text, patient=patient)
        with self.assertRaises(ValueError):
            self.validate(self.dizzy.text, patient=patient)

    def test_missing_value_not_string_none_or_yes(self):
        unknown = Evidence('fixture:missing', self.dizzy.question, None, False)
        patient = PatientRepresentation(55, 'F', (unknown,), ())
        self.validate(unknown.text, patient=patient)
        for quote in ('None', self.dizzy.text, unknown.question):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                self.validate(quote, patient=patient)

    def test_recombined_question_and_other_answer_rejected(self):
        for quote in (self.dizzy.question + ' = ' + self.location.value,
                      self.location.question + ' = Yes', self.dizzy.text + '; ' + self.location.text):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                self.validate(quote)

    def test_fabricated_fact_mixed_with_valid_fact_still_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(self.dizzy.text, 'Unreported observation = Yes')

    def test_manual_full_value_excerpt_and_rendered_fact_remain_valid(self):
        prose = Evidence('manual:Symptoms:0', 'Symptoms', 'Fatigue and dizziness for three weeks.', False)
        patient = PatientRepresentation(24, 'F', (prose,), ())
        for quote in (prose.value, 'Fatigue and dizziness', prose.text):
            with self.subTest(quote=quote):
                self.validate(quote, patient=patient)
        with self.assertRaises(ValueError):
            self.validate('Fatigue and breathlessness', patient=patient)

    def test_source_quote_validation_remains_exact_after_patient_fix(self):
        self.validate(self.dizzy.text, source='- headache')
        for quote in ('- headache,”,', 'Invented medical support.', 'Title: Fixture'):
            with self.subTest(quote=quote), self.assertRaisesRegex(ValueError, 'Source support is not in document body'):
                self.validate(self.dizzy.text, source=quote)

    def test_forged_schema_cannot_bypass_minimum_quote_length(self):
        value = self.review(self.dizzy.text).model_copy(update={'positive_fact_quotes': ['Yes']})
        with self.assertRaises(ValueError):
            validate_content_review(value, self.patient, self.document)

    def test_real_provider_schema_and_validator_path_accepts_full_facts(self):
        client = Mock()
        client.responses.create.return_value = envelope(self.review(self.dizzy.text).model_dump_json())
        provider = OpenAIProvider(OpenAIConfig(max_retries=0), client=client)
        generation = provider.generate('Fixture', {'patient_case': self.patient.to_inference_dict()},
            ContentRelevance, lambda value: validate_content_review(value, self.patient, self.document))
        self.assertIsNone(generation.failure)
        self.assertEqual(generation.attempts, 1)
        client.responses.create.assert_called_once()


class DDXQuotePipelineTests(unittest.TestCase):
    setUp = pipeline_fixtures.Check5PipelineTests.setUp
    fixture = pipeline_fixtures.Check5PipelineTests.fixture

    def test_full_question_answer_reaches_all_gates_with_existing_provenance(self):
        from rag.manual_patient import manual_patient_id
        from rag.who_sections import section_hits
        from application.decision import build_final_decision
        fact = Evidence('fixture:dizzy', 'Do you feel slightly dizzy or lightheaded?', 'Yes', False)
        # DDXPlus-shaped data; do not assign a benchmark ID to an invented case.
        self.patient = PatientRepresentation(55, 'F', (fact,), ())
        self.patient_id = manual_patient_id(self.patient)
        self.review = self.review.model_copy(update={'positive_fact_quotes': [fact.text]})
        self.hits = section_hits(self.document, self.review, self.patient)
        workflow, provider, patients, medical = self.fixture()
        state = workflow.run(self.patient, self.patient_id)
        self.assertEqual(state.status, 'final')
        self.assertEqual(state.medical_retrieval['combined_status'], 'BOTH_AVAILABLE')
        names = [c['text']['format']['schema']['title'] for c in provider.calls]
        self.assertEqual(names[:3], ['PatientState', 'WHOSelection', 'ContentRelevance'])
        self.assertTrue({'DiagnosticResult', 'ClinicalCritique', 'SemanticSafetyResult'} <= set(names))
        self.assertEqual(build_final_decision(state).status, 'ALLOW')  # Synthetic software fixture only.
        self.assertEqual(state.safety_coverage, 'assessed')
        patients.retrieve.assert_called_once()
        medical.retrieve.assert_called_once()


if __name__ == '__main__':
    unittest.main()

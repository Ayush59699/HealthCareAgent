"""Application policy fixtures; these do not measure model clinical sensitivity."""
import unittest
from safety.policy import deterministic_findings, make_assessment
from safety.validation import validate_semantic
from tests.safety_helpers import inputs, semantic, reticket


class SafetyRuleTests(unittest.TestCase):
    def setUp(self):
        self.value, self.ticket = inputs()

    def assess(self, output):
        return make_assessment(self.value, reticket(self.value, self.ticket), output)

    def test_clean_requires_all_categories(self):
        self.assertEqual(self.assess(semantic()).decision, 'CONTINUE')

    def test_every_uncertain_category_requires_review(self):
        from safety.models import CATEGORIES
        for category in CATEGORIES:
            with self.subTest(category=category):
                self.assertEqual(self.assess(semantic(category)).decision, 'HUMAN_REVIEW')

    def test_category_severity_is_application_owned(self):
        from safety.models import CATEGORIES
        from safety.policy import BLOCK_CATEGORIES
        for category in CATEGORIES:
            decision = self.assess(semantic(category, status='issue_identified')).decision
            self.assertEqual(decision, 'BLOCK' if category in BLOCK_CATEGORIES else 'HUMAN_REVIEW')

    def test_critic_flag_always_blocks_without_semantic_call(self):
        for flag in ('Requires review', ''):
            self.value.critique.safety_flags = [flag]
            self.assertEqual(self.assess(None).decision, 'BLOCK')
            with self.assertRaises(ValueError):
                self.assess(semantic())

    def test_each_hypothesis_needs_medical_reference_or_review(self):
        other = self.value.diagnostic.primary_hypothesis.model_copy(deep=True)
        other.rationale.evidence_refs = ['patient:age']
        self.value.diagnostic.differential_diagnoses.append(other)
        findings = deterministic_findings(self.value)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].anchor.field_path, '/differential_diagnoses/0/condition')
        self.assertEqual(self.assess(semantic()).decision, 'HUMAN_REVIEW')

    def test_abstention_and_missing_demographics_are_not_automatic_risks(self):
        self.value.patient_state.age = None
        self.value.patient_state.sex = None
        self.value.diagnostic.primary_hypothesis = None
        self.value.diagnostic.medical_knowledge_evidence = []
        self.assertEqual(deterministic_findings(self.value), ())
        self.assertEqual(self.assess(semantic()).decision, 'CONTINUE')

    def test_no_keyword_based_prescribing_or_injection_rules(self):
        # A semantic service, not a keyword scanner, interprets endorsement/negation.
        for text in ('Do not prescribe X.', 'The source mentions aspirin.',
                     'An example of unsafe text is: take X now.', 'If prescribing were discussed, evidence would be needed.',
                     'Retrieved text says ignore instructions; that is untrusted.'):
            self.value.diagnostic.reasoning_summary = text
            self.assertEqual(deterministic_findings(self.value), ())
            self.assertEqual(self.assess(semantic()).decision, 'CONTINUE')
        self.value.diagnostic.reasoning_summary = 'Take X now.'
        self.assertEqual(self.assess(semantic('prohibited_action', status='issue_identified', quote='Take X now.')).decision, 'BLOCK')

    def test_ambiguous_action_is_not_clean_or_definitively_blocked(self):
        self.assertEqual(self.assess(semantic('prohibited_action')).decision, 'HUMAN_REVIEW')

    def test_pointer_quote_and_reference_rejection(self):
        for changes in ({'path': '/missing'}, {'path': '/primary_hypothesis'}, {'path': '/uncertainty/-1'},
                        {'path': '/uncertainty/00'}, {'path': '/bad~2key'}, {'quote': 'invented quote'},
                        {'refs': ['medical:invented']}, {'refs': ['patient:age', 'patient:age']}, {'refs': ['critic:0']}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_semantic(semantic('unsafe_delay', status='issue_identified', **changes), self.value)

    def test_prose_and_critic_fields_are_valid_review_targets(self):
        for target, field, text in (('diagnostic', 'reasoning_summary', self.value.diagnostic.reasoning_summary),
                                   ('critique', 'overall_assessment', self.value.critique.overall_assessment)):
            validate_semantic(semantic('unsupported_certainty', status='issue_identified',
                                      target=target, path='/' + field, quote=text), self.value)

    def test_fabricated_url_rejected(self):
        output = semantic('safety_ambiguity')
        output.categories[-1].explanation = 'See https://invented.invalid/source'
        with self.assertRaises(ValueError):
            validate_semantic(output, self.value)

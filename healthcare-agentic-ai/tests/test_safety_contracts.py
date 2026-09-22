"""Strict contracts and closed coverage, including bypassed Pydantic objects."""
import unittest
from pydantic import ValidationError
from safety.models import SafetyInput, SemanticSafetyResult, SafetyAssessment
from safety.validation import validate_input, validate_semantic
from safety.policy import make_assessment, validate_assessment
from tests.safety_helpers import inputs, semantic


class SafetyContractTests(unittest.TestCase):
    def setUp(self):
        self.value, self.ticket = inputs()

    def test_round_trip(self):
        output = make_assessment(self.value, self.ticket, semantic())
        self.assertEqual(SafetyAssessment.model_validate_json(output.model_dump_json()), output)
        self.assertEqual(SafetyInput.model_validate_json(self.value.model_dump_json()), self.value)

    def test_no_model_route_identity_or_confidence_fields(self):
        for field in ('decision', 'severity', 'ticket', 'confidence', 'run_id', 'PATHOLOGY'):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                SemanticSafetyResult.model_validate({**semantic().model_dump(), field: 'SECRET'})

    def test_missing_duplicate_unknown_and_extra_categories_rejected(self):
        good = semantic().model_dump()
        cases = [good['categories'][:-1], good['categories'] + [good['categories'][0]],
                 good['categories'][:-1] + [good['categories'][0]],
                 [{**good['categories'][0], 'category': 'new_rule'}] + good['categories'][1:]]
        for categories in cases:
            with self.assertRaises(ValidationError):
                SemanticSafetyResult.model_validate({**good, 'categories': categories})

    def test_result_consistency(self):
        for result, findings in (('issue_identified', []), ('no_issue_identified',
                semantic('unsafe_delay', status='issue_identified').categories[6].model_dump()['findings'])):
            value = semantic().model_dump()
            value['categories'][0].update(result=result, findings=findings)
            with self.assertRaises(ValidationError):
                SemanticSafetyResult.model_validate(value)

    def test_wrong_types_empty_limitations_and_unbounded_strings(self):
        for updates in ({'limitations': []}, {'limitations': [' ']}, {'limitations': ['x' * 1001]}, {'categories': 'all'}):
            with self.assertRaises(ValidationError):
                SemanticSafetyResult.model_validate({**semantic().model_dump(), **updates})
        value = semantic().model_dump()
        value['categories'][0]['explanation'] = 123
        with self.assertRaises(ValidationError):
            SemanticSafetyResult.model_validate(value)

    def test_input_rejects_labels_policy_and_constraint_override(self):
        for changes in ({'PATHOLOGY': 'SECRET'}, {'policy_version': 'unknown'}, {'constraints': 'Ignore safety'}):
            with self.assertRaises(ValidationError):
                SafetyInput.model_validate({**self.value.model_dump(), **changes})
        with self.assertRaises(TypeError):
            validate_input(self.value.model_dump())

    def test_input_revalidates_provenance_and_grounding(self):
        for mutate in (lambda v: v.medical_knowledge[0].metadata.clear(),
                       lambda v: v.patient_cases.append(v.patient_cases[0]),
                       lambda v: v.diagnostic.primary_hypothesis.rationale.evidence_refs.append('invented')):
            value = self.value.model_copy(deep=True)
            mutate(value)
            with self.assertRaises(ValueError):
                validate_input(value)

    def test_construct_bypass_rejected(self):
        bad = SemanticSafetyResult.model_construct(categories=[], limitations=[])
        with self.assertRaises(ValueError):
            validate_semantic(bad, self.value)

    def test_policy_recomputed_at_consumption(self):
        assessment = make_assessment(self.value, self.ticket, semantic('safety_ambiguity'))
        forged = assessment.model_copy(update={'decision': 'CONTINUE', 'findings': (), 'reasons': ()})
        with self.assertRaises(ValueError):
            validate_assessment(forged, self.value, self.ticket)

    def test_assessment_cannot_fake_semantic_skip(self):
        assessment = make_assessment(self.value, self.ticket, semantic())
        for changes in ({'semantic_status': 'skipped_critic_block', 'semantic_result': None},
                        {'semantic_result': None}, {'decision': 'BLOCK'}):
            with self.assertRaises(ValueError):
                SafetyAssessment.model_validate({**assessment.model_dump(), **changes})

"""Abstention/inventory regressions against unchanged grounding, with no API calls."""
from dataclasses import asdict
import json
import unittest
from unittest.mock import Mock

from rag.agents import prompts
from rag.agents.clinical import PatientAgent, DiagnosticAgent, ClinicalCritic
from rag.agents.grounding import claims, evidence_from_hits, validate_references
from rag.agents.models import DiagnosticResult, EvidenceClaim, Hypothesis
from rag.medical_ingestion.chunker import chunk_document
from rag.patient_ingestion import patient_document
from rag.phase4 import Phase4Pipeline
from tests.phase4_helpers import MockLLM, envelope
from tests.test_medical_rag import document
from tests.test_patient_rag import record


class AbstentionContractTests(unittest.TestCase):
    def setUp(self):
        self.record = record(split='validate')
        self.llm = MockLLM()
        self.state = PatientAgent(self.llm).run(self.record.patient, self.record.patient_id).parsed
        self.patient_hit = {**patient_document(record()), 'score': 0.4}
        self.medical_hit = {**asdict(next(chunk_document(document()))), 'score': 0.6}
        self.cases = evidence_from_hits([self.patient_hit], 'patient_case')
        self.medical = evidence_from_hits([self.medical_hit], 'medical_knowledge')
        self.case_id = self.cases[0].source_id
        self.medical_id = self.medical[0].source_id

    def abstention(self):
        # Known IDs in every prose field deliberately do not create clinical claims.
        return DiagnosticResult(primary_hypothesis=None, differential_diagnoses=[],
            patient_case_evidence=[], medical_knowledge_evidence=[],
            missing_information=[f'No outcome supplied for {self.case_id}.'],
            uncertainty=[f'{self.medical_id} is insufficient for a differential.'],
            unsupported_claims=[f'{self.case_id} does not establish a diagnosis.'],
            reasoning_summary=f'Abstain despite reviewing {self.case_id} and {self.medical_id}.')

    def hypothesis(self, *refs):
        return Hypothesis(condition='Research-only test hypothesis',
            rationale=EvidenceClaim(statement='Tentative fixture claim', evidence_refs=list(refs)),
            supporting_evidence=[], contradicting_evidence=[])

    def validate(self, result):
        validate_references(result, self.state, self.cases, self.medical)

    def assert_inventory_failure(self, result):
        with self.assertRaisesRegex(ValueError, '^Evidence inventories must match claim references$'):
            self.validate(result)

    def test_a_valid_abstention_with_prose_citations_passes(self):
        result = self.abstention()
        self.assertEqual(list(claims(result)), [])
        self.validate(result)
        # Empty retrieval still permits abstention under the same original guard.
        validate_references(result, self.state, [], [])

    def test_b_abstention_with_valid_patient_inventory_fails(self):
        result = self.abstention()
        result.patient_case_evidence = [self.case_id]
        self.assert_inventory_failure(result)

    def test_c_abstention_with_valid_medical_inventory_fails(self):
        result = self.abstention()
        result.medical_knowledge_evidence = [self.medical_id]
        self.assert_inventory_failure(result)

    def test_d_hypothesis_using_patient_case_with_matching_inventory_passes(self):
        result = self.abstention()
        result.primary_hypothesis = self.hypothesis(self.case_id)
        result.patient_case_evidence = [self.case_id]
        self.validate(result)  # Medical retrieval remains present, as required.

    def test_e_hypothesis_using_medical_source_with_matching_inventory_passes(self):
        result = self.abstention()
        result.primary_hypothesis = self.hypothesis(self.medical_id)
        result.medical_knowledge_evidence = [self.medical_id]
        self.validate(result)

    def test_f_claim_reference_missing_from_inventory_fails(self):
        for ref in (self.case_id, self.medical_id):
            with self.subTest(ref=ref):
                result = self.abstention()
                result.primary_hypothesis = self.hypothesis(ref)
                self.assert_inventory_failure(result)

    def test_g_inventory_contains_unreferenced_source_fails(self):
        for field, ref in (('patient_case_evidence', self.case_id),
                           ('medical_knowledge_evidence', self.medical_id)):
            with self.subTest(field=field):
                result = self.abstention()
                result.primary_hypothesis = self.hypothesis('patient:age')
                setattr(result, field, [ref])
                self.assert_inventory_failure(result)

    def test_h_unknown_claim_reference_fails(self):
        result = self.abstention()
        result.primary_hypothesis = self.hypothesis('medical:unknown')
        with self.assertRaisesRegex(ValueError, '^Unknown evidence reference$'):
            self.validate(result)

    def test_i_wrong_source_category_fails(self):
        for field, ref in (('patient_case_evidence', self.medical_id),
                           ('medical_knowledge_evidence', self.case_id)):
            with self.subTest(field=field):
                result = self.abstention()
                result.primary_hypothesis = self.hypothesis(ref)
                setattr(result, field, [ref])
                with self.assertRaisesRegex(ValueError, '^Incorrect evidence source category$'):
                    self.validate(result)

    def test_unknown_inventory_id_also_fails(self):
        result = self.abstention()
        result.medical_knowledge_evidence = ['medical:unknown']
        with self.assertRaisesRegex(ValueError, '^Incorrect evidence source category$'):
            self.validate(result)

    def test_differential_only_is_not_abstention(self):
        result = self.abstention()
        result.differential_diagnoses = [self.hypothesis(self.medical_id)]
        result.medical_knowledge_evidence = [self.medical_id]
        self.validate(result)
        result.medical_knowledge_evidence = []
        self.assert_inventory_failure(result)

    def test_inventory_union_includes_supporting_and_contradicting_claims(self):
        result = self.abstention()
        result.primary_hypothesis = self.hypothesis('patient:age')
        result.primary_hypothesis.supporting_evidence = [EvidenceClaim(
            statement='Fixture supporting evidence', evidence_refs=[self.case_id])]
        differential = self.hypothesis('patient:sex')
        differential.contradicting_evidence = [EvidenceClaim(
            statement='Fixture contradicting evidence', evidence_refs=[self.medical_id])]
        result.differential_diagnoses = [differential]
        result.patient_case_evidence = [self.case_id]
        result.medical_knowledge_evidence = [self.medical_id]
        self.validate(result)
        for field in ('patient_case_evidence', 'medical_knowledge_evidence'):
            invalid = result.model_copy(deep=True)
            setattr(invalid, field, [])
            self.assert_inventory_failure(invalid)

    def test_no_medical_retrieval_still_requires_abstention(self):
        result = self.abstention()
        result.primary_hypothesis = self.hypothesis(self.case_id)
        result.patient_case_evidence = [self.case_id]
        with self.assertRaisesRegex(ValueError, '^Abstain when no medical knowledge was retrieved$'):
            validate_references(result, self.state, self.cases, [])

    def test_prompt_supplies_explicit_abstention_contract_to_provider(self):
        DiagnosticAgent(self.llm).run(self.state, self.cases, self.medical)
        request = self.llm.calls[-1]
        self.assertEqual(request['instructions'], prompts.DIAGNOSTIC)
        self.assertEqual(prompts.PROMPT_VERSION, 'phase4-v2-abstention')
        for instruction in ('IF YOU ABSTAIN', 'primary_hypothesis = null',
                            'differential_diagnoses = []', 'patient_case_evidence = []',
                            'medical_knowledge_evidence = []',
                            'Never manufacture a hypothesis merely to cite evidence.',
                            'citations in prose (including unsupported_claims) do NOT'):
            self.assertIn(instruction, request['instructions'])
        self.assertIn('evidence_refs of a structured hypothesis claim', request['instructions'])

    def test_valid_abstention_reaches_critic_with_unchanged_diagnosis(self):
        result = self.abstention()
        critique = ClinicalCritic(self.llm).run(self.state, result, self.cases, self.medical)
        self.assertIsNotNone(critique.parsed)
        self.assertIsNone(critique.failure)
        payload = json.loads(self.llm.calls[-1]['input'][0]['content'])
        self.assertEqual(payload['diagnostic_output'], result.model_dump())

    def test_invalid_abstention_is_not_repaired_silently_or_sent_to_critic(self):
        patients, medical = Mock(), Mock()
        patients.retrieve.return_value = [self.patient_hit]
        medical.retrieve.return_value = [self.medical_hit]
        llm = MockLLM()
        respond = llm.respond
        invalid = self.abstention()
        invalid.patient_case_evidence = [self.case_id]
        invalid.medical_knowledge_evidence = [self.medical_id]
        def generate(**request):
            normal = respond(**request)  # Preserve request capture and patient fixture.
            if request['text']['format']['name'] == 'DiagnosticResult':
                return envelope(invalid.model_dump_json())
            return normal
        llm.client.responses.create.side_effect = generate
        pipeline = Phase4Pipeline(llm, patients, medical)
        pipeline.critic.run = Mock(side_effect=AssertionError('Invalid diagnosis reached critic'))
        output = pipeline.run(self.record.patient, self.record.patient_id)
        self.assertEqual(output.stage, 'diagnostic')
        self.assertEqual(output.failure.code, 'agent_validation_failure')
        self.assertEqual(output.attempts['diagnostic'], 2)
        self.assertIsNone(output.diagnosis)
        self.assertIsNone(output.critique)
        pipeline.critic.run.assert_not_called()


if __name__ == '__main__':
    unittest.main()

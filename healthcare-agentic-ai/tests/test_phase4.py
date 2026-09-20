"""Offline Phase 4 tests. No cloud API, network, GPU or model download required."""
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import Mock
from pydantic import ValidationError
from rag.config import OpenAIConfig
from rag.models import PatientRepresentation, EvaluationLabels, DifferentialDiagnosis
from rag.agents.models import PatientState, DiagnosticResult, Hypothesis, EvidenceClaim
from rag.agents.grounding import patient_input, validate_patient, evidence_from_hits, validate_references, context
from rag.agents.clinical import PatientAgent
from rag.phase4 import Phase4Pipeline
from rag.phase4_evaluation import evaluate_case, summarize
from rag.patient_ingestion import patient_document
from rag.medical_ingestion.chunker import chunk_document
from tests.test_patient_rag import record
from tests.test_medical_rag import document
from tests.phase4_helpers import MockLLM


class Phase4Tests(unittest.TestCase):
    def setUp(self):
        self.record = record(split='validate')
        self.patient_hit = {**patient_document(record()), 'score': -0.25}
        self.medical_hit = {**asdict(next(chunk_document(document()))), 'score': 0.6}
        self.llm = MockLLM()
        self.patient_rag = Mock()
        self.patient_rag.retrieve.return_value = [self.patient_hit]
        self.medical_rag = Mock()
        self.medical_rag.retrieve.return_value = [self.medical_hit]
        self.pipeline = Phase4Pipeline(self.llm, self.patient_rag, self.medical_rag)

    def run_case(self):
        result = self.pipeline.run(self.record.patient, self.record.patient_id)
        self.assertEqual(result.status, 'success', result.failure)
        return result

    def test_prompt_isolation_and_structured_handoff(self):
        output = self.run_case()
        self.assertEqual(len(self.llm.calls), 3)
        self.assertEqual(len({r['instructions'] for r in self.llm.calls}), 3)
        self.assertTrue(all(len(r['input']) == 1 and 'previous_response_id' not in r for r in self.llm.calls))
        diagnostic = json.loads(self.llm.calls[1]['input'][0]['content'])
        critic = json.loads(self.llm.calls[2]['input'][0]['content'])
        self.assertEqual(diagnostic['patient_state'], output.patient_state.model_dump())
        self.assertNotIn('diagnostic_output', diagnostic)
        self.assertEqual(critic['diagnostic_output'], output.diagnosis.model_dump())
        self.assertNotIn('allowed_missing_information', critic)

    def assert_no_labels(self, index):
        self.record = replace(self.record, labels=EvaluationLabels('SECRET_PRIMARY', (DifferentialDiagnosis('SECRET_DIFFERENTIAL', 1.0),)))
        self.run_case()
        serialized = json.dumps(self.llm.calls[index])
        for forbidden in ('SECRET_PRIMARY', 'SECRET_DIFFERENTIAL', 'PATHOLOGY', 'ground_truth_pathology', 'DIFFERENTIAL_DIAGNOSIS'):
            self.assertNotIn(forbidden, serialized)

    def test_no_ground_truth_reaches_patient_agent(self):
        self.assert_no_labels(0)

    def test_no_ground_truth_reaches_diagnostic_agent(self):
        self.assert_no_labels(1)

    def test_no_ground_truth_reaches_critic(self):
        self.assert_no_labels(2)

    def test_reject_label_containing_records_and_raw_rows(self):
        for value in (self.record, {'PATHOLOGY': 'SECRET'}):
            with self.assertRaises(TypeError):
                self.pipeline.run(value, self.record.patient_id)
        self.assertEqual(self.llm.calls, [])

    def test_patient_schema_and_traceability(self):
        output = self.run_case()
        supplied = patient_input(self.record.patient, self.record.patient_id)
        for changes in ({'age': 99}, {'symptoms': []}, {'relevant_findings': ['Invented lab']}, {'missing_information': ['Invented history']}, {'uncertainty_notes': []}):
            with self.assertRaises(ValueError):
                validate_patient(output.patient_state.model_copy(update=changes), supplied)
        with self.assertRaises(ValidationError):
            PatientState.model_validate({**output.patient_state.model_dump(), 'PATHOLOGY': 'secret'})

    def test_unknown_demographics_preserved(self):
        patient = PatientRepresentation(None, None, (), ())
        generation = PatientAgent(self.llm).run(patient, self.record.patient_id)
        self.assertIsNone(generation.parsed.age)
        self.assertIn('age', generation.parsed.missing_information)

    def test_diagnostic_receives_both_rags_and_provenance(self):
        result = self.run_case()
        self.patient_rag.retrieve.assert_called_once_with(self.record.patient.to_text(), top_k=1)
        self.medical_rag.retrieve.assert_called_once_with(self.record.patient.to_text(), top_k=1)
        data = json.loads(self.llm.calls[1]['input'][0]['content'])
        self.assertEqual(len(data['PATIENT_CASE_EVIDENCE']), 1)
        self.assertEqual(len(data['MEDICAL_KNOWLEDGE_EVIDENCE']), 1)
        self.assertEqual(result.medical_knowledge_evidence[0].metadata['provenance'], self.medical_hit['provenance'])
        self.assertEqual(result.medical_knowledge_evidence[0].metadata['url'], self.medical_hit['url'])

    def test_similarity_not_probability(self):
        result = self.run_case()
        self.assertEqual(result.patient_case_evidence[0].similarity, -0.25)
        self.assertNotIn('probability', result.patient_case_evidence[0].model_dump())
        self.assertIn('NOT diagnostic confidence or diagnostic probability', self.llm.calls[1]['instructions'])
        self.assertNotIn('temperature', self.llm.calls[0])
        self.assertNotIn('tools', self.llm.calls[0])

    def test_extra_retrieval_labels_rejected_before_diagnostic(self):
        self.patient_rag.retrieve.return_value = [{**self.patient_hit, 'PATHOLOGY': 'SECRET'}]
        result = self.pipeline.run(self.record.patient, self.record.patient_id)
        self.assertEqual(result.stage, 'retrieval')
        self.assertEqual(result.status, 'failed')
        self.assertEqual(len(self.llm.calls), 1)

    def test_provenance_tampering_rejected(self):
        result = self.run_case()
        result.medical_knowledge_evidence[0].source_id = 'invented'
        with self.assertRaises(ValueError):
            context(result.patient_state, result.patient_case_evidence, result.medical_knowledge_evidence)

    def test_invalid_citations_rejected(self):
        result = self.run_case()
        hypothesis = Hypothesis(condition='Asthma', rationale=EvidenceClaim(statement='Tentative', evidence_refs=['invented']), supporting_evidence=[], contradicting_evidence=[])
        result.diagnosis.primary_hypothesis = hypothesis
        with self.assertRaises(ValueError):
            validate_references(result.diagnosis, result.patient_state, result.patient_case_evidence, result.medical_knowledge_evidence)

    def test_valid_citations_and_posthoc_evaluation(self):
        output = self.run_case()
        ref = output.medical_knowledge_evidence[0].source_id
        output.diagnosis.primary_hypothesis = Hypothesis(condition='Asthma', rationale=EvidenceClaim(statement='Tentative', evidence_refs=[ref, 'patient:symptoms:0']), supporting_evidence=[], contradicting_evidence=[])
        output.diagnosis.medical_knowledge_evidence = [ref]
        validate_references(output.diagnosis, output.patient_state, output.patient_case_evidence, output.medical_knowledge_evidence)
        evaluation = evaluate_case(output, EvaluationLabels(' asthma '))
        self.assertTrue(evaluation['primary_match'])
        self.assertEqual(evaluation['citation_provenance_coverage'], 1)
        self.assertEqual(summarize([output], [evaluation])['primary_accuracy'], 1)
        self.assertNotIn('ground_truth', output.model_dump())

    def test_failure_metrics_and_abstention_are_not_accuracy(self):
        output = self.run_case()
        evaluation = evaluate_case(output, EvaluationLabels('Asthma'))
        self.assertFalse(evaluation['primary_match'])
        self.assertIsNone(evaluation['citation_provenance_coverage'])
        self.assertIsNone(summarize([], [])['pipeline_success_rate'])

    def test_empty_retrieval_reported_not_fabricated(self):
        self.patient_rag.retrieve.return_value = []
        self.medical_rag.retrieve.return_value = []
        output = self.run_case()
        self.assertEqual(output.retrieval_success, {'patient_case': False, 'medical_knowledge': False})
        self.assertIsNone(output.diagnosis.primary_hypothesis)

    def test_malformed_output_bounded_and_fail_closed(self):
        llm = MockLLM(['not json', '{}'])
        generation = PatientAgent(llm).run(self.record.patient, self.record.patient_id)
        self.assertIsNone(generation.parsed)
        self.assertEqual(generation.failure.code, 'parsing_failure')
        self.assertEqual(generation.attempts, 2)
        self.assertEqual(len(llm.calls), 2)
        self.assertNotIn('not json', json.dumps(llm.calls[1]))

    def test_repair_can_succeed(self):
        valid = PatientAgent(self.llm).run(self.record.patient, self.record.patient_id).raw
        llm = MockLLM(['{}', valid])
        self.assertIsNotNone(PatientAgent(llm).run(self.record.patient, self.record.patient_id).parsed)
        self.assertEqual(len(llm.calls), 2)

    def test_no_retry_configuration(self):
        llm = MockLLM(['{}'], OpenAIConfig(max_retries=0))
        result = PatientAgent(llm).run(self.record.patient, self.record.patient_id)
        self.assertEqual(result.attempts, 1)

    def test_transport_failure_sanitized_and_no_retry(self):
        llm = MockLLM([OSError('SECRET patient text')])
        with self.assertLogs('rag.llm.provider', level='WARNING') as logs:
            result = PatientAgent(llm).run(self.record.patient, self.record.patient_id)
        self.assertNotIn('SECRET', str(logs.output) + result.failure.model_dump_json())
        self.assertEqual(result.attempts, 1)

    def test_pipeline_stops_after_invalid_patient(self):
        pipeline = Phase4Pipeline(MockLLM(['{}', '{}']), self.patient_rag, self.medical_rag)
        output = pipeline.run(self.record.patient, self.record.patient_id)
        self.assertEqual(output.status, 'failed')
        self.patient_rag.retrieve.assert_not_called()
        self.assertIsNone(output.diagnosis)

    def test_context_budget_fails_without_network(self):
        llm = MockLLM(config=OpenAIConfig(max_input_bytes=100))
        result = PatientAgent(llm).run(self.record.patient, self.record.patient_id)
        self.assertEqual(result.failure.code, 'context_budget')
        self.assertEqual(llm.calls, [])

    def test_strict_types_and_no_markdown_salvage(self):
        valid = PatientAgent(self.llm).run(self.record.patient, self.record.patient_id).raw
        for raw in ('```json\n' + valid + '\n```', json.dumps({**json.loads(valid), 'age': '45'})):
            llm = MockLLM([raw], OpenAIConfig(max_retries=0))
            self.assertIsNone(PatientAgent(llm).run(self.record.patient, self.record.patient_id).parsed)


if __name__ == '__main__':
    unittest.main()

"""Deterministic test-only LLM; exercises the production parsing/repair path."""
import copy
import json
from rag.llm.provider import OpenAIProvider
from unittest.mock import Mock
from rag.agents.grounding import UNCERTAINTY


class MockLLM(OpenAIProvider):
    def __init__(self, responses=None, config=None):
        self.calls = []
        self.responses = list(responses) if responses is not None else None
        client = Mock()
        client.responses.create.side_effect = self.respond
        super().__init__(config, client=client)

    def respond(self, **request):
        self.calls.append(copy.deepcopy(request))
        if self.responses is not None:
            result = self.responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return envelope(result if isinstance(result, str) else json.dumps(result))
        data = json.loads(request['input'][0]['content'])
        title = request['text']['format']['schema']['title']
        if title == 'PatientState':
            result = {'patient_id': data['patient_id'], 'age': data['age'], 'sex': data['sex'],
                      'symptoms': data['symptoms'], 'antecedents': data['antecedents'],
                      'presenting_evidence': data['initial_evidence'], 'relevant_findings': [],
                      'missing_information': data['allowed_missing_information'], 'uncertainty_notes': [UNCERTAINTY]}
        elif title == 'DiagnosticResult':
            result = {'primary_hypothesis': None, 'differential_diagnoses': [],
                      'patient_case_evidence': [], 'medical_knowledge_evidence': [],
                      'missing_information': [], 'uncertainty': ['Insufficient evidence'],
                      'unsupported_claims': [], 'reasoning_summary': 'Abstain; research fixture.'}
        else:
            result = {'overall_assessment': 'insufficient_evidence', 'supported_points': [],
                      'unsupported_points': [], 'contradictions': [], 'missing_evidence': ['Insufficient evidence'],
                      'hallucination_flags': [], 'safety_flags': [], 'recommended_revisions': ['Seek human review'],
                      'critique_confidence': 'low'}
        return envelope(json.dumps(result))


def envelope(raw, *, status='completed'):
    return {'status': status, 'output': [
        {'id': 'rs_test', 'type': 'reasoning', 'summary': []},
        {'id': 'msg_test', 'type': 'message', 'role': 'assistant', 'status': 'completed',
         'content': [{'type': 'output_text', 'text': raw, 'annotations': []}]}],
        'usage': {'input_tokens': 100, 'output_tokens': 50, 'total_tokens': 150, 'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0}, 'output_tokens_details': {'reasoning_tokens': 0}}}

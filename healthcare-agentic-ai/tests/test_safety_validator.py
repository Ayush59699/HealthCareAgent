"""Stateless provider reuse, bounded repairs, and valid concerning results."""
import json
import unittest
from safety import SafetyValidator
from safety.prompts import SAFETY
from tests.safety_helpers import inputs, semantic, SafetyScenarioLLM


class SafetyValidatorTests(unittest.TestCase):
    def test_provider_schema_and_isolated_payload(self):
        supplied, _ = inputs()
        llm = SafetyScenarioLLM()
        result = SafetyValidator(llm).run(supplied)
        self.assertIsNone(result.failure)
        request = llm.calls[-1]
        payload = json.loads(request['input'][0]['content'])
        self.assertEqual(set(payload), set(supplied.model_dump()))
        self.assertEqual(len(request['input']), 1)
        self.assertNotIn('tools', request)
        self.assertNotIn('previous_response_id', request)
        self.assertNotIn('temperature', request)
        self.assertFalse(request['store'])
        self.assertTrue(request['text']['format']['strict'])
        self.assertNotIn('ticket', payload)
        self.assertNotIn('run_id', payload)

    def test_concerning_result_is_not_repaired(self):
        supplied, _ = inputs()
        for category in ('prohibited_action', 'safety_ambiguity'):
            llm = SafetyScenarioLLM(safety_results=[semantic(category, status='issue_identified')])
            result = SafetyValidator(llm).run(supplied)
            self.assertIsNone(result.failure)
            self.assertEqual(result.attempts, 1)

    def test_invalid_schema_is_bounded_then_can_recover(self):
        supplied, _ = inputs()
        llm = SafetyScenarioLLM(safety_results=[{}, semantic()])
        result = SafetyValidator(llm).run(supplied)
        self.assertEqual(result.attempts, 2)
        self.assertIsNone(result.failure)
        self.assertNotIn('{}', llm.calls[-1]['instructions'])

    def test_invalid_anchor_exhausts_repairs(self):
        supplied, _ = inputs()
        llm = SafetyScenarioLLM(safety_results=[semantic('unsafe_delay', status='issue_identified', quote='SECRET nonexistent')])
        result = SafetyValidator(llm).run(supplied)
        self.assertEqual(result.failure.code, 'agent_validation_failure')
        self.assertEqual(result.attempts, 2)
        self.assertNotIn('SECRET', llm.calls[-1]['instructions'])

    def test_incomplete_and_refusal_do_not_retry(self):
        supplied, _ = inputs()
        refusal = {'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                    'content': [{'type': 'refusal', 'refusal': 'SECRET'}]}]}
        for response, code in (({'status': 'incomplete'}, 'incomplete_output'), (refusal, 'refusal')):
            llm = SafetyScenarioLLM(safety_results=[response])
            result = SafetyValidator(llm).run(supplied)
            self.assertEqual(result.failure.code, code)
            self.assertEqual(result.attempts, 1)

    def test_prompt_scope_includes_all_prose_and_negation(self):
        for instruction in ('ALL generated textual fields', 'no_issue_identified', 'uncertain',
                            'negations', 'UNDER REVIEW', 'not evidence', 'No numerical confidence'):
            self.assertIn(instruction, SAFETY)

    def test_context_byte_cap_fails_before_any_request(self):
        from rag.config import OpenAIConfig
        supplied, _ = inputs()
        llm = SafetyScenarioLLM(config=OpenAIConfig(max_input_bytes=1))
        result = SafetyValidator(llm).run(supplied)
        self.assertEqual(result.failure.code, 'context_budget')
        self.assertEqual(result.attempts, 0)
        self.assertEqual(llm.calls, [])

    def test_transport_error_never_retries_or_exposes_raw_text(self):
        supplied, _ = inputs()
        llm = SafetyScenarioLLM(safety_results=[OSError('SECRET')])
        result = SafetyValidator(llm).run(supplied)
        self.assertEqual(result.failure.code, 'connection_failure')
        self.assertEqual(result.attempts, 1)
        self.assertNotIn('SECRET', result.failure.model_dump_json())

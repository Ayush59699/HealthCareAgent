"""Responses API regressions: mocked SDK only; credentials never needed."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError, APITimeoutError, BadRequestError, AuthenticationError, RateLimitError
from openai.types.responses import Response
from rag.config import OpenAIConfig, load_generation_env
from rag.llm.provider import OpenAIProvider
from rag.agents.models import PatientState, DiagnosticResult, ClinicalCritique
from tests.phase4_helpers import envelope


class OpenAIProviderTests(unittest.TestCase):
    def example(self):
        return dict(patient_id='ddxplus:validate:1', age=30, sex='F', presenting_evidence=[],
                    symptoms=[], antecedents=[], relevant_findings=[], missing_information=[], uncertainty_notes=[])

    def provider(self, response=None, **config):
        client = Mock()
        client.responses.create.return_value = response or envelope(json.dumps(self.example()))
        return OpenAIProvider(OpenAIConfig(**config), client=client), client

    def test_default_and_dedicated_environment_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            config = OpenAIConfig()
            self.assertEqual((config.provider, config.model, config.temperature), ('openai', 'gpt-5.6-sol', 0))
            self.assertEqual((config.timeout, config.max_retries), (300, 1))
        with patch.dict(os.environ, {'GPT_SOL_ENDPOINT': 'https://example.test/openai/v1/',
                'OPENAI_TIMEOUT': '12', 'OPENAI_MAX_INPUT_BYTES': '20000', 'OPENAI_MAX_OUTPUT_TOKENS': '2048'}):
            config = OpenAIConfig()
            self.assertEqual((config.base_url, config.timeout, config.max_input_bytes, config.max_output_tokens),
                             ('https://example.test/openai/v1/', 12, 20000, 2048))

    def test_invalid_settings_and_second_models_rejected(self):
        for kwargs in ({'provider': 'other'}, {'model': 'other'}, {'temperature': 0.2},
                       {'temperature': float('nan')}, {'timeout': 0}, {'timeout': float('inf')},
                       {'max_retries': 3}, {'max_input_bytes': 0}, {'max_output_tokens': -1},
                       {'max_output_tokens': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OpenAIConfig(**kwargs)
        for url in ('http://example.test/v1', 'https://localhost/v1', 'https://user:secret@example.test/v1',
                    'https://example.test/v1?key=secret', 'https://example.test/v1#x',
                    'https://example.test:0/v1', 'https://example.test:99999/v1', 'https://example.test/invalid'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OpenAIConfig(base_url=url)

    def test_env_file_only_loads_healthcare_settings_and_preserves_process_values(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'GPT_SOL_API_KEY': 'process-key'}, clear=True):
            path = Path(directory) / '.env'
            path.write_text('GPT_SOL_API_KEY = "file-key"\nGPT_SOL_ENDPOINT=https://example.test/v1\nAZURE_OPENAI_API_KEY=coding-key\nAZURE_OPENAI_DEPLOYMENT=other-model\n')
            load_generation_env(path)
            self.assertEqual(os.environ['GPT_SOL_API_KEY'], 'process-key')
            self.assertEqual(os.environ['GPT_SOL_ENDPOINT'], 'https://example.test/v1')
            self.assertNotIn('AZURE_OPENAI_API_KEY', os.environ)
            self.assertEqual(OpenAIConfig().model, 'gpt-5.6-sol')

    def test_missing_credentials_fail_without_using_coding_agent_key(self):
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'coding-secret'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'GPT_SOL_API_KEY'):
                OpenAIProvider()

    def test_sdk_constructor_and_report_never_contain_key(self):
        with patch.dict(os.environ, {'GPT_SOL_API_KEY': 'SECRET'}, clear=True), patch('rag.llm.provider.OpenAI') as sdk:
            with OpenAIProvider() as provider:
                self.assertNotIn('SECRET', str(asdict(provider.config)))
                self.assertNotIn('SECRET', repr(provider.config))
            sdk.assert_called_once_with(base_url=provider.config.base_url, api_key='SECRET', timeout=300, max_retries=0)
            sdk.return_value.close.assert_called_once()

    def test_request_uses_responses_strict_schema_no_tools_or_history(self):
        provider, client = self.provider()
        result = provider.generate('Role prompt', {'feature': 'data'}, PatientState)
        self.assertIsNone(result.failure)
        request = client.responses.create.call_args.kwargs
        self.assertEqual(request['model'], 'gpt-5.6-sol')
        self.assertNotIn('temperature', request)
        self.assertFalse(result.telemetry[0]['temperature_sent'])
        self.assertEqual(result.telemetry[0]['requested_temperature'], 0)
        self.assertEqual(result.telemetry[0]['effective_temperature'], 'deployment_default')
        self.assertEqual(request['text']['format']['type'], 'json_schema')
        self.assertTrue(request['text']['format']['strict'])
        self.assertFalse(request['text']['format']['schema']['additionalProperties'])
        self.assertFalse(request['store'])
        self.assertEqual(request['max_output_tokens'], 8192)
        for key in ('tools', 'previous_response_id', 'conversation', 'messages'):
            self.assertNotIn(key, request)
        client.chat.completions.create.assert_not_called()
        self.assertEqual(result.telemetry[0]['total_tokens'], 150)

    def test_sdk_response_extraction_ignores_reasoning(self):
        response = envelope(json.dumps(self.example()))
        response.update(id='resp_test', created_at=0, model='gpt-5.6-sol', object='response', parallel_tool_calls=False, tool_choice='none', tools=[])
        sdk_response = Response.model_validate(response)
        provider, _ = self.provider(sdk_response)
        self.assertIsNotNone(provider.generate('', {}, PatientState).parsed)

    def test_malformed_envelopes_incomplete_refusals_and_tool_calls_fail_closed(self):
        responses = [None, [], 'invalid', {}, {'status': 'failed'},
                     {'status': 'completed', 'output': []},
                     envelope('{}', status='incomplete'),
                     {'status': 'completed', 'output': [{'type': 'function_call', 'name': 'run_command'}]},
                     {'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                       'content': [{'type': 'refusal', 'refusal': 'SECRET'}]}]}]
        for response in responses:
            with self.subTest(response=response):
                provider, client = self.provider()
                client.responses.create.return_value = response
                result = provider.generate('', {}, PatientState)
                self.assertIsNone(result.parsed)
                self.assertIn(result.failure.code, ('provider_failure', 'incomplete_output', 'refusal'))
                self.assertEqual(result.attempts, 1)
                self.assertNotIn('SECRET', str(result.telemetry) + str(result.failure))

    def test_incomplete_valid_json_rejected_for_each_agent(self):
        examples = [(PatientState, self.example()), (DiagnosticResult, dict(primary_hypothesis=None,
                    differential_diagnoses=[], patient_case_evidence=[], medical_knowledge_evidence=[],
                    missing_information=[], uncertainty=[], unsupported_claims=[], reasoning_summary='Abstain.')),
                    (ClinicalCritique, dict(overall_assessment='insufficient_evidence', supported_points=[],
                    unsupported_points=[], contradictions=[], missing_evidence=[], hallucination_flags=[],
                    safety_flags=[], recommended_revisions=[], critique_confidence='low'))]
        for schema, data in examples:
            schema.model_validate_json(json.dumps(data))
            provider, _ = self.provider(envelope(json.dumps(data), status='incomplete'))
            result = provider.generate('', {}, schema)
            self.assertEqual(result.failure.code, 'incomplete_output')
            self.assertIsNone(result.parsed)

    def test_input_byte_boundary_and_repair_budget_rechecked(self):
        provider, client = self.provider()
        result = provider.generate('prompt', {}, PatientState)
        size = result.telemetry[0]['request_bytes']
        provider, client = self.provider(max_input_bytes=size)
        self.assertIsNotNone(provider.generate('prompt', {}, PatientState).parsed)
        self.assertEqual(provider.generate('promptX', {}, PatientState).failure.code, 'context_budget')
        self.assertEqual(client.responses.create.call_count, 1)
        provider, client = self.provider(envelope('{}'), max_input_bytes=size)
        result = provider.generate('prompt', {}, PatientState)
        self.assertEqual(result.failure.code, 'context_budget')
        self.assertEqual(result.attempts, 1)
        self.assertEqual(client.responses.create.call_count, 1)

    def test_api_errors_are_sanitized_not_schema_retried(self):
        request = httpx.Request('POST', 'https://example.test/v1/responses')
        errors = [(APITimeoutError(request=request), 'timeout'),
                  (APIConnectionError(request=request, message='SECRET'), 'connection_failure')]
        for cls, status in ((AuthenticationError, 401), (RateLimitError, 429), (BadRequestError, 400)):
            errors.append((cls('SECRET', response=httpx.Response(status, request=request), body={'error': 'SECRET'}), 'api_failure'))
        for error, code in errors:
            with self.subTest(code=code):
                provider, client = self.provider()
                client.responses.create.side_effect = error
                with self.assertLogs('rag.llm.provider', level='WARNING') as logs:
                    result = provider.generate('', {}, PatientState)
                self.assertEqual(result.failure.code, code)
                self.assertEqual(client.responses.create.call_count, 1)
                self.assertFalse(result.telemetry[0]['api_success'])
                self.assertNotIn('SECRET', str(logs.output) + str(result.failure) + str(result.telemetry))

    def test_grounding_failure_distinct_from_parsing_failure(self):
        provider, client = self.provider(max_retries=0)
        def reject(_):
            raise ValueError('SECRET')
        result = provider.generate('', {}, PatientState, reject)
        self.assertEqual(result.failure.code, 'agent_validation_failure')
        self.assertNotIn('SECRET', str(result.telemetry))


if __name__ == '__main__':
    unittest.main()

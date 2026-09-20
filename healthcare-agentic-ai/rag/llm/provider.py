"""Stateless GPT-5.6-Sol Responses adapter. No tools, shared history or fallback."""
from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Callable, Protocol, TypeVar

from openai import OpenAI, APIError, APIStatusError, APIConnectionError, APITimeoutError
from pydantic import BaseModel, ValidationError
from ..config import OpenAIConfig
from ..agents.models import Failure

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


@dataclass
class Generation:
    parsed: BaseModel | None = None
    raw: str | None = None  # Never logged or persisted by the runner.
    failure: Failure | None = None
    attempts: int = 0
    telemetry: list[dict] = field(default_factory=list)


class StructuredLLM(Protocol):
    def generate(self, system: str, payload: dict, schema: type[T],
                 validator: Callable[[T], None] | None = None) -> Generation: ...


class OutputFailure(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def extract_text(response: dict) -> str:
    """Accept complete assistant text only; never treat tool calls/refusals as JSON."""
    if not isinstance(response, dict):
        raise OutputFailure('provider_failure')
    if response.get('status') == 'incomplete':
        raise OutputFailure('incomplete_output')
    if response.get('status') != 'completed' or response.get('error'):
        raise OutputFailure('provider_failure')
    output = response.get('output')
    if not isinstance(output, list) or not output:
        raise OutputFailure('provider_failure')
    texts = []
    for item in output:
        if not isinstance(item, dict):
            raise OutputFailure('provider_failure')
        if item.get('type') == 'reasoning':
            continue  # Never log or expose reasoning output.
        if (item.get('type') != 'message' or item.get('role') != 'assistant'
                or item.get('status') != 'completed' or not isinstance(item.get('content'), list)):
            raise OutputFailure('provider_failure')
        for part in item['content']:
            if not isinstance(part, dict):
                raise OutputFailure('provider_failure')
            if part.get('type') == 'refusal':
                raise OutputFailure('refusal')
            if part.get('type') != 'output_text' or not isinstance(part.get('text'), str):
                raise OutputFailure('provider_failure')
            texts.append(part['text'])
    raw = ''.join(texts)
    if not raw.strip() or len(raw.encode('utf8')) > 2_000_000:
        raise OutputFailure('provider_failure')
    return raw


class OpenAIProvider:
    def __init__(self, config: OpenAIConfig | None = None, *, client=None):
        self.config = config or OpenAIConfig()
        self._owns_client = client is None
        if client is None:
            key = os.getenv('GPT_SOL_API_KEY', '').strip()
            if not key:
                raise ValueError('GPT_SOL_API_KEY is required for cloud generation')
            # Disable SDK retries: an API failure is not a medical or schema failure.
            client = OpenAI(base_url=self.config.base_url, api_key=key,
                            timeout=self.config.timeout, max_retries=0)
        self.client = client

    def close(self):
        if self._owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def generate(self, system: str, payload: dict, schema: type[T],
                 validator: Callable[[T], None] | None = None) -> Generation:
        user = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        json_schema = schema.model_json_schema()
        raw, repair = None, ''
        telemetry = []
        failure_code = 'invalid_output'
        for attempt in range(1, self.config.max_retries + 2):
            # Deployment rejects temperature (verified HTTP 400). Omit it; do not
            # claim deterministic temperature-0 sampling or retry a different model.
            request = {
                'model': self.config.model, 'instructions': system + repair,
                'input': [{'role': 'user', 'content': user}],
                'text': {'format': {'type': 'json_schema', 'name': schema.__name__,
                                    'strict': True, 'schema': json_schema}},
                'max_output_tokens': self.config.max_output_tokens,
                'store': False,
            }
            # Application byte cap, NOT an assertion about the deployment's context window.
            # Include the schema and repair text on every attempt; never truncate evidence.
            size = len(json.dumps(request, ensure_ascii=False).encode('utf8'))
            if size > self.config.max_input_bytes:
                return Generation(failure=Failure(code='context_budget', message='Request exceeds input byte cap; reduce retrieval top-k.', attempts=len(telemetry)),
                                  attempts=len(telemetry), telemetry=telemetry)
            observation = {'attempt': attempt, 'request_bytes': size, 'api_success': False,
                           'client_context_truncated': False, 'accepted': False,
                           'requested_temperature': self.config.temperature,
                           'temperature_sent': False, 'effective_temperature': 'deployment_default'}
            telemetry.append(observation)
            started = time.perf_counter()
            try:
                response = self.client.responses.create(**request)
                observation['api_success'] = True
            except (APIError, OSError, TimeoutError) as exc:
                if isinstance(exc, (APITimeoutError, TimeoutError)):
                    code = 'timeout'
                elif isinstance(exc, APIStatusError):
                    code = 'api_failure'
                    observation['http_status'] = exc.status_code
                elif isinstance(exc, (APIConnectionError, OSError)):
                    code = 'connection_failure'
                else:
                    code = 'api_failure'
                observation['failure_code'] = code
                logger.warning('Cloud API request failed (%s; content omitted)', code)
                return Generation(failure=Failure(code=code, message='Cloud API request failed; check credentials, deployment, quota and timeout.', attempts=attempt),
                                  attempts=attempt, telemetry=telemetry)
            finally:
                observation['request_seconds'] = time.perf_counter() - started
            try:
                if isinstance(response, BaseModel):
                    response = response.model_dump()
                if isinstance(response, dict):
                    usage = response.get('usage')
                    if isinstance(usage, dict):
                        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
                            value = usage.get(key)
                            if type(value) is int and value >= 0:
                                observation[key] = value
                    if response.get('status') in ('completed', 'incomplete', 'failed'):
                        observation['response_status'] = response['status']
                raw = extract_text(response)
                # No markdown stripping, substring extraction or type coercion.
                parsed = schema.model_validate_json(raw)
                if validator:
                    validator(parsed)
                observation['accepted'] = True
                return Generation(parsed=parsed, raw=raw, attempts=attempt, telemetry=telemetry)
            except OutputFailure as exc:
                # No automatic retries for refusals, incomplete generations or envelopes.
                observation['failure_code'] = exc.code
                return Generation(failure=Failure(code=exc.code, message='Responses output was refused, incomplete or invalid.', attempts=attempt),
                                  attempts=attempt, telemetry=telemetry)
            except (ValidationError, ValueError, TypeError):
                # Never copy exception content, patient data or rejected output into repairs.
                failure_code = 'parsing_failure' if not self._is_schema_valid(raw, schema) else 'agent_validation_failure'
                observation['failure_code'] = failure_code
                logger.warning('Cloud structured output rejected (%s); attempt %d', failure_code, attempt)
                repair = '\nPrevious attempt failed schema or grounding validation. Regenerate the complete object exactly to schema and grounding rules. No markdown.'
        return Generation(raw=raw, failure=Failure(code=failure_code, message='Structured output or grounding validation failed after bounded retries.', attempts=attempt),
                          attempts=attempt, telemetry=telemetry)

    @staticmethod
    def _is_schema_valid(raw, schema):
        try:
            schema.model_validate_json(raw)
            return True
        except (ValidationError, ValueError, TypeError):
            return False

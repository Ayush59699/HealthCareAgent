"""Application limits, distinct from provider structured-output repair limits."""
from typing import Literal
from pydantic import ConfigDict, Field
from rag.agents.models import StrictModel


class WorkflowPolicy(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)
    max_revisions: Literal[2] = 2
    max_requests: int = Field(default=21, ge=1)
    max_seconds: float = Field(default=900.0, gt=0)
    max_request_bytes: int = Field(default=7_000_000, ge=1)
    provider_attempt_limit: int = Field(default=3, ge=1, le=3)


class WorkflowStop(Exception):
    """Carries only a code selected by application code, never provider prose."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


SAFE_PROVIDER_CODES = frozenset({
    'context_budget', 'timeout', 'api_failure', 'connection_failure', 'provider_failure',
    'refusal', 'incomplete_output', 'parsing_failure', 'agent_validation_failure', 'invalid_output',
})

SAFE_WORKFLOW_CODES = SAFE_PROVIDER_CODES | frozenset({
    'stale_result', 'invalid_transition', 'invalid_revision_budget',
    'time_budget_exhausted', 'request_budget_exhausted', 'invalid_provider_telemetry',
    'schema_validation_failure', 'grounding_failure', 'patient_validation_failure',
    'invalid_evidence_or_retrieval_failure', 'stage_failure',
})

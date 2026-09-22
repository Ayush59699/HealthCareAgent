"""Application-authored safety invocation identity, never authored by the LLM."""
from dataclasses import dataclass
from typing import Literal
from pydantic import ConfigDict, Field
from rag.agents.models import StrictModel
from rag.llm.provider import Generation
from orchestration.contracts import StageTicket
from orchestration.policy import WorkflowStop


class SafetyTicket(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    run_id: str
    patient_id: str
    stage: Literal['SAFETY_VALIDATION'] = 'SAFETY_VALIDATION'
    revision_number: int = Field(ge=0, le=2)
    diagnostic_version: int = Field(ge=1, le=3)
    diagnostic_fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')
    evidence_snapshot_id: str = Field(pattern=r'^[0-9a-f]{64}$')
    critic_fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')
    safety_policy_version: Literal['phase6-safety-v1'] = 'phase6-safety-v1'
    input_fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')


@dataclass(frozen=True)
class Phase6Response:
    ticket: StageTicket | SafetyTicket
    generation: Generation


def require_current(expected, returned):
    if type(expected) is not type(returned) or expected != returned:
        raise WorkflowStop('stale_result')

"""Typed deterministic stages and sanitized audit events."""
from datetime import datetime, timezone
from typing import Literal
from pydantic import ConfigDict
from rag.agents.models import StrictModel

Stage = Literal['INITIAL', 'PATIENT', 'EVIDENCE', 'DIAGNOSTIC', 'DIAGNOSTIC_REVISION',
                'GROUNDING', 'CRITIC', 'ROUTE', 'FINAL', 'ABSTENTION', 'UNRESOLVED',
                'BLOCKED', 'TERMINAL_FAILURE']
EventName = Literal['workflow_started', 'patient_started', 'patient_agent_completed',
    'evidence_started', 'evidence_retrieved', 'diagnostic_started', 'diagnostic_validated',
    'grounding_started', 'grounding_passed', 'critic_started', 'critic_completed',
    'route_started', 'revision_requested', 'diagnostic_revision_started', 'finalized',
    'abstained', 'unresolved', 'terminated']


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowEvent(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    sequence: int
    timestamp: str
    event: EventName
    from_stage: Stage
    stage: Stage
    revision_number: int
    reason: str
    validation_result: bool | None = None
    run_id: str
    patient_id: str
    diagnostic_version: int | None = None
    evidence_snapshot_id: str | None = None

"""Versioned research workflow state. Never contains evaluator labels."""
from typing import Literal
from pydantic import ConfigDict, Field
from rag.models import PatientRepresentation
from rag.agents.models import StrictModel, PatientState, DiagnosticResult, ClinicalCritique, Failure
from .contracts import StageTicket
from .events import Stage, WorkflowEvent
from .evidence import EvidenceSnapshot
from .policy import WorkflowPolicy

Outcome = Literal['accepted_with_limitations', 'diagnostic_agent_abstention', 'insufficient_evidence',
                  'max_revisions_reached', 'unresolved_critic', 'safety_blocked', 'technical_failure']


class DiagnosticVersion(StrictModel):
    ticket: StageTicket
    version: int
    fingerprint: str
    result: DiagnosticResult


class CriticVersion(StrictModel):
    ticket: StageTicket
    diagnostic_version: int
    evidence_snapshot_id: str
    result: ClinicalCritique


class ValidationOutcome(StrictModel):
    stage: Stage
    revision_number: int
    schema_valid: bool
    grounding_valid: bool | None = None
    reason: str


class InvocationTelemetry(StrictModel):
    ticket: StageTicket
    attempts: int = 0
    provider_repairs: int = 0
    seconds: float = 0.0
    request_bytes: int = 0
    observations: list[dict] = Field(default_factory=list)


class WorkflowState(StrictModel):
    # Local to run(); agents and callers receive detached copies, never live state.
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)
    schema_version: Literal['phase5-workflow-v1'] = 'phase5-workflow-v1'
    policy_version: Literal['phase5-routing-v1'] = 'phase5-routing-v1'
    run_id: str
    patient_id: str
    original_patient: PatientRepresentation
    patient_state: PatientState | None = None
    evidence: EvidenceSnapshot | None = None
    diagnostics: tuple[DiagnosticVersion, ...] = ()
    critiques: tuple[CriticVersion, ...] = ()
    current_stage: Stage = 'INITIAL'
    transition_history: tuple[WorkflowEvent, ...] = ()
    revision_count: int = 0
    max_revisions: Literal[2] = 2
    validations: tuple[ValidationOutcome, ...] = ()
    failure: Failure | None = None
    outcome: Outcome | None = None
    status: Literal['running', 'final', 'abstained', 'unresolved', 'blocked', 'failed'] = 'running'
    started_at: str
    completed_at: str | None = None
    policy: WorkflowPolicy
    invocations: tuple[InvocationTelemetry, ...] = ()
    total_requests: int = 0
    provider_repairs: int = 0
    total_request_bytes: int = 0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    seconds: float = 0.0
    disclaimer: str = ('Research orchestration only; not clinical approval, advice or evidence of '
                       'clinical effectiveness. Human-in-the-loop review is not implemented.')

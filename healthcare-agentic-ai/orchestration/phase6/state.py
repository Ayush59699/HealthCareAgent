"""Versioned Phase 6 state; old consumers cannot silently accept it as Phase 5."""
from typing import Literal, Union
from pydantic import ConfigDict, Field, model_validator
from rag.agents.models import StrictModel, PatientState, Failure
from rag.models import PatientRepresentation
from orchestration.state import (Outcome as Phase5Outcome, ValidationOutcome, InvocationTelemetry,
                                  DiagnosticVersion, CriticVersion)
from orchestration.contracts import StageTicket
from orchestration.evidence import EvidenceSnapshot, fingerprint
from safety.models import SafetyAssessment
from .contracts import SafetyTicket
from .events import Stage, Phase6Event
from .policy import Phase6Policy

Outcome = Union[Phase5Outcome, Literal['human_review_required']]


class Phase6Validation(ValidationOutcome):
    stage: Stage


class Phase6Invocation(InvocationTelemetry):
    ticket: StageTicket | SafetyTicket


class Phase6WorkflowState(StrictModel):
    # Not a WorkflowState subclass: no isinstance/Pydantic instance short-circuit
    # may disguise a new state as the closed Phase 5 contract.
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)
    schema_version: Literal['phase6-workflow-v1'] = 'phase6-workflow-v1'
    policy_version: Literal['phase6-routing-v1'] = 'phase6-routing-v1'
    run_id: str
    patient_id: str
    original_patient: PatientRepresentation
    patient_state: PatientState | None = None
    evidence: EvidenceSnapshot | None = None
    diagnostics: tuple[DiagnosticVersion, ...] = ()
    critiques: tuple[CriticVersion, ...] = ()
    current_stage: Stage = 'INITIAL'
    transition_history: tuple[Phase6Event, ...] = ()
    revision_count: int = 0
    max_revisions: Literal[2] = 2
    validations: tuple[Phase6Validation, ...] = ()
    failure: Failure | None = None
    outcome: Outcome | None = None
    status: Literal['running', 'final', 'abstained', 'unresolved', 'blocked', 'failed', 'human_review_required'] = 'running'
    started_at: str
    completed_at: str | None = None
    policy: Phase6Policy
    invocations: tuple[Phase6Invocation, ...] = ()
    total_requests: int = 0
    provider_repairs: int = 0
    total_request_bytes: int = 0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    seconds: float = 0.0
    safety_assessments: tuple[SafetyAssessment, ...] = ()
    # Coverage refers to the current diagnostic attempt, never an inherited prior pass.
    # A failed revision may leave an older committed diagnostic and assessment in history.
    safety_coverage: Literal['not_assessed', 'assessed', 'failed'] = 'not_assessed'
    safety_skip_reason: Literal['not_reached', 'no_medical_evidence', 'awaiting_safety',
                               'unchanged_diagnostic', 'repeated_diagnostic',
                               'technical_failure_before_safety', 'safety_validation_failed'] | None = 'not_reached'
    disclaimer: str = ('Research output-safety validation only; CONTINUE is not clinical clearance. '
                       'Human review is not scheduled or implemented. No clinical advice or effectiveness claim.')

    @model_validator(mode='after')
    def coverage_consistent(self):
        if self.safety_coverage == 'assessed':
            if self.safety_skip_reason is not None or not self.safety_assessments or not self.diagnostics or not self.critiques or not self.evidence:
                raise ValueError('Assessed coverage requires a current assessment')
            current, diagnosis, review = self.safety_assessments[-1], self.diagnostics[-1], self.critiques[-1]
            ticket = current.ticket
            if (ticket.run_id != self.run_id or ticket.patient_id != self.patient_id or
                    ticket.diagnostic_version != diagnosis.version or ticket.revision_number != self.revision_count or
                    ticket.diagnostic_fingerprint != diagnosis.fingerprint or
                    ticket.diagnostic_fingerprint != fingerprint(diagnosis.result.model_dump()) or
                    ticket.evidence_snapshot_id != self.evidence.snapshot_id or
                    review.diagnostic_version != diagnosis.version or
                    review.ticket.diagnostic_fingerprint != diagnosis.fingerprint or
                    review.evidence_snapshot_id != self.evidence.snapshot_id or
                    ticket.critic_fingerprint != fingerprint(review.result.model_dump())):
                raise ValueError('Safety coverage belongs to another version')
        elif self.safety_skip_reason is None:
            raise ValueError('Unassessed coverage requires a reason')
        expected = {'FINAL': 'CONTINUE', 'BLOCKED': 'BLOCK', 'HUMAN_REVIEW': 'HUMAN_REVIEW'}
        if self.current_stage in expected:
            if self.safety_coverage != 'assessed' or self.safety_assessments[-1].decision != expected[self.current_stage]:
                raise ValueError('Terminal disposition requires matching safety decision')
        return self

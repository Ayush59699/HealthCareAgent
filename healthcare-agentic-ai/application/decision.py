"""Terminal research/reviewer handoff. No human approval or safety override API."""
from typing import Literal
from pydantic import model_validator
from rag.agents.models import (StrictModel, PatientState, DiagnosticResult, ClinicalCritique,
                               RetrievedEvidence, Failure)
from rag.agents.grounding import patient_input, validate_patient, validate_references, facts
from orchestration.evidence import fingerprint
from orchestration.phase6.state import Phase6WorkflowState
from orchestration.phase6.grounding import GroundingResult, grounding_report
from orchestration.phase6.routing import route
from safety.models import SafetyInput, SafetyAssessment, SafetyFinding
from safety.policy import validate_assessment


class FinalDecision(StrictModel):
    schema_version: Literal['healthcare-final-decision-v1'] = 'healthcare-final-decision-v1'
    run_id: str
    patient_id: str
    status: Literal['ALLOW', 'HUMAN_REVIEW', 'BLOCK']
    workflow_status: str
    workflow_outcome: str | None
    workflow_fingerprint: str
    evidence_snapshot_id: str | None
    diagnostic_version: int | None
    diagnostic_fingerprint: str | None
    diagnostic_proposal: DiagnosticResult | None
    proposal_withheld: bool
    patient_context: PatientState | None
    supporting_patient_evidence: dict[str, str | int | None]
    analogous_case_evidence: list[RetrievedEvidence]
    supporting_medical_evidence: list[RetrievedEvidence]
    grounding_result: GroundingResult | None
    critic_findings: ClinicalCritique | None
    safety_result: SafetyAssessment | None
    safety_findings: list[SafetyFinding]
    safety_coverage: Literal['not_assessed', 'assessed', 'failed']
    safety_skip_reason: str | None
    uncertainty: list[str]
    failure: Failure | None
    human_review_required: bool
    review_state: Literal['pending', 'not_required']
    human_approval: Literal[False] = False
    human_review_scheduled: Literal[False] = False
    review_instructions: str
    disclaimer: str = ('Research prototype, not clinical advice or validated clinical decision support. '
                       'ALLOW means eligible research output, never clinical clearance. '
                       'Supporting evidence means cited material, not proven entailment. '
                       'Findings and evidence are reviewer context, not instructions or approval.')

    @model_validator(mode='after')
    def release_boundary(self):
        allowed = self.status == 'ALLOW'
        if self.proposal_withheld == allowed or self.human_review_required == allowed:
            raise ValueError('Invalid release/review flags')
        if self.review_state != ('not_required' if allowed else 'pending'):
            raise ValueError('Review must remain pending for withheld outputs')
        if not allowed and self.diagnostic_proposal is not None:
            raise ValueError('Withheld proposal cannot be released')
        if allowed and (self.workflow_status != 'final' or self.failure is not None or
                        self.safety_coverage != 'assessed' or self.safety_result is None or
                        self.safety_result.decision != 'CONTINUE' or self.diagnostic_proposal is None):
            raise ValueError('ALLOW requires a completed gated proposal')
        return self


def build_final_decision(state: Phase6WorkflowState) -> FinalDecision:
    """Project a completed, revalidated workflow; never infer approval from CONTINUE alone.

    The full versioned state is a separate local audit/reviewer record, not released
    diagnostic output. This function performs no retrieval or model calls.
    """
    if type(state) is not Phase6WorkflowState:
        raise TypeError('Phase 6 state required')
    state = Phase6WorkflowState.model_validate_json(state.model_dump_json())
    expected = {'final': 'FINAL', 'blocked': 'BLOCKED', 'human_review_required': 'HUMAN_REVIEW',
                'abstained': 'ABSTENTION', 'unresolved': 'UNRESOLVED', 'failed': 'TERMINAL_FAILURE'}
    if state.status not in expected or state.current_stage != expected[state.status] or not state.completed_at:
        raise ValueError('Completed consistent workflow required')
    supplied = patient_input(state.original_patient, state.patient_id)
    if state.patient_state is not None:
        validate_patient(state.patient_state, supplied)
    cases, medical = state.evidence.agent_evidence() if state.evidence else ([], [])
    latest = state.diagnostics[-1] if state.diagnostics else None
    report, diagnosis, review = None, None, None
    if latest:
        if not state.patient_state or not state.evidence:
            raise ValueError('Diagnostic context missing')
        ticket = latest.ticket
        if (ticket.run_id != state.run_id or ticket.patient_id != state.patient_id or
                ticket.evidence_snapshot_id != state.evidence.snapshot_id or
                latest.fingerprint != fingerprint(latest.result.model_dump())):
            raise ValueError('Diagnostic identity mismatch')
        diagnosis = latest.result
        report = grounding_report(diagnosis, state.patient_state, cases, medical)
        if state.critiques and state.critiques[-1].diagnostic_version == latest.version:
            entry = state.critiques[-1]
            if (entry.ticket.run_id != state.run_id or entry.ticket.patient_id != state.patient_id or
                    entry.ticket.diagnostic_fingerprint != latest.fingerprint or
                    entry.evidence_snapshot_id != state.evidence.snapshot_id):
                raise ValueError('Critic identity mismatch')
            review = entry.result
            validate_references(review, state.patient_state, cases, medical, allow_patient_inference=True)
    assessment = state.safety_assessments[-1] if state.safety_coverage == 'assessed' else None
    if assessment:
        value = SafetyInput(patient_state=state.patient_state, diagnostic=diagnosis, critique=review,
                            patient_cases=cases, medical_knowledge=medical)
        validate_assessment(assessment, value, assessment.ticket)
        if state.status == 'final':
            decision = route(value, assessment, assessment.ticket, state.revision_count, state.max_revisions)
            if decision.next_stage != 'FINAL' or decision.outcome != state.outcome:
                raise ValueError('Safety continuation does not authorize final release')
    allowed = state.status == 'final'
    status = 'ALLOW' if allowed else 'BLOCK' if state.status in {'blocked', 'failed'} else 'HUMAN_REVIEW'
    used_patient = {ref for link in report.claims for ref in link.patient_refs} if report else set()
    patient_facts = facts(state.patient_state) if state.patient_state else {}
    used_cases = set(diagnosis.patient_case_evidence) if diagnosis else set()
    used_medical = set(diagnosis.medical_knowledge_evidence) if diagnosis else set()
    uncertainty = list(dict.fromkeys([
        *(state.patient_state.uncertainty_notes if state.patient_state else []),
        *(diagnosis.uncertainty if diagnosis else []),
        *(['No accepted AMG evidence; medical reasoning was not attempted.']
          if state.safety_skip_reason == 'no_medical_evidence' else []),
        *(['AI cannot finalize this case; human review remains pending.'] if not allowed else []),
    ]))
    return FinalDecision(run_id=state.run_id, patient_id=state.patient_id, status=status,
        workflow_status=state.status, workflow_outcome=state.outcome,
        workflow_fingerprint=fingerprint(state.model_dump(mode='json')),
        evidence_snapshot_id=state.evidence.snapshot_id if state.evidence else None,
        diagnostic_version=latest.version if latest else None,
        diagnostic_fingerprint=latest.fingerprint if latest else None,
        diagnostic_proposal=diagnosis if allowed else None, proposal_withheld=not allowed,
        patient_context=state.patient_state,
        supporting_patient_evidence={ref: patient_facts[ref] for ref in sorted(used_patient)},
        analogous_case_evidence=[e for e in cases if e.source_id in used_cases],
        supporting_medical_evidence=[e for e in medical if e.source_id in used_medical],
        grounding_result=report, critic_findings=review, safety_result=assessment,
        safety_findings=list(assessment.findings) if assessment else [],
        safety_coverage=state.safety_coverage, safety_skip_reason=state.safety_skip_reason,
        uncertainty=uncertainty, failure=state.failure, human_review_required=not allowed,
        review_state='not_required' if allowed else 'pending',
        review_instructions=('No review is scheduled or approval recorded. Inspect the separate local workflow '
                             'audit matching run_id and workflow_fingerprint for the withheld proposal, all '
                             'retrieved evidence, missing information, revisions and failures. A human must '
                             'independently assess the case; this program cannot approve or override the gate.')
                            if not allowed else 'Eligible for research inspection only; not a clinical decision.')

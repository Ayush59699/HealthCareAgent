"""Inspectable deterministic claim links; never a medical-entailment oracle."""
from typing import Literal
from rag.agents.models import StrictModel, DiagnosticResult
from rag.agents.grounding import context, facts, validate_references
from orchestration.evidence import fingerprint
from .reference_entailment import inspect_reference_consistency


class ClaimGrounding(StrictModel):
    claim_path: str
    patient_refs: list[str]
    analogous_case_refs: list[str]
    medical_refs: list[str]
    medical_support: Literal['reference_present_not_verified', 'unsupported_by_medical_evidence']


class GroundingResult(StrictModel):
    schema_version: Literal['phase6-grounding-v1'] = 'phase6-grounding-v1'
    structural_status: Literal['passed'] = 'passed'
    patient_fingerprint: str
    diagnostic_fingerprint: str
    evidence_snapshot_id: str
    claims: list[ClaimGrounding]
    hypotheses_without_medical_references: list[str]
    patient_reference_audit: dict
    declared_unsupported_claims: list[str]
    medical_entailment: Literal['not_established'] = 'not_established'
    limitations: str = ('Patient references bind exact input facts; analogous cases are not this patient. '
                       'Medical citation existence and provenance do not establish support or relevance. '
                       'The bounded patient-reference check is partial. Critic and safety review remain mandatory.')


def grounding_report(diagnosis: DiagnosticResult, state, cases, medical) -> GroundingResult:
    """Use existing hard gates. Invalid input raises; no fabricated passing report."""
    context(state, cases, medical)
    validate_references(diagnosis, state, cases, medical, allow_patient_inference=True)
    patient_ids = set(facts(state))
    case_ids = {e.source_id for e in cases}
    medical_ids = {e.source_id for e in medical}
    hypotheses = ([('/primary_hypothesis', diagnosis.primary_hypothesis)]
                  if diagnosis.primary_hypothesis else [])
    hypotheses += [(f'/differential_diagnoses/{i}', h) for i, h in enumerate(diagnosis.differential_diagnoses)]
    links, unsupported = [], []
    for path, hypothesis in hypotheses:
        entries = [(path + '/rationale', hypothesis.rationale)]
        for field in ('supporting_evidence', 'contradicting_evidence'):
            entries += [(f'{path}/{field}/{i}', claim) for i, claim in enumerate(getattr(hypothesis, field))]
        used = set()
        for claim_path, claim in entries:
            refs = set(claim.evidence_refs)
            used |= refs
            links.append(ClaimGrounding(claim_path=claim_path,
                patient_refs=sorted(refs & patient_ids), analogous_case_refs=sorted(refs & case_ids),
                medical_refs=sorted(refs & medical_ids),
                medical_support='reference_present_not_verified' if refs & medical_ids else 'unsupported_by_medical_evidence'))
        if not used & medical_ids:
            unsupported.append(path)
    return GroundingResult(patient_fingerprint=fingerprint(state.model_dump()),
        diagnostic_fingerprint=fingerprint(diagnosis.model_dump()),
        evidence_snapshot_id=fingerprint({'patient_cases': [e.model_dump() for e in cases],
                                          'medical_knowledge': [e.model_dump() for e in medical]}),
        claims=links, hypotheses_without_medical_references=unsupported,
        patient_reference_audit=inspect_reference_consistency(diagnosis, state),
        declared_unsupported_claims=list(diagnosis.unsupported_claims))

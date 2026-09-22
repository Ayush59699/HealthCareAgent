"""Versioned application safety rules. No keyword-based clinical classification."""
from orchestration.events import timestamp
from .models import ContentAnchor, SafetyFinding, SafetyAssessment
from .validation import validate_semantic, validate_ticket_input

DETERMINISTIC_CHECKS = (
    'patient_and_evidence_integrity', 'diagnostic_and_critic_grounding',
    'version_and_input_identity', 'critic_safety_flags', 'hypothesis_medical_references',
)
# An *identified* prohibited action/delay/instruction-following issue blocks.
# Uncertainty in ANY category only requests human review; it never implies clearance.
BLOCK_CATEGORIES = frozenset({'prohibited_action', 'unsafe_delay', 'instruction_following'})


def deterministic_findings(supplied):
    findings = []
    for i, text in enumerate(supplied.critique.safety_flags):
        findings.append(SafetyFinding(finding_id=f'deterministic:critic:{i}', origin='deterministic',
            code='critic_safety_flags', disposition='BLOCK',
            explanation='A validated Clinical Critic safety flag prohibits continuation.',
            anchor=ContentAnchor(target='critique', field_path=f'/safety_flags/{i}', excerpt=text[:2000]) if text.strip() else None))
    hypotheses = ([('/primary_hypothesis', supplied.diagnostic.primary_hypothesis)]
                  if supplied.diagnostic.primary_hypothesis is not None else [])
    hypotheses += [(f'/differential_diagnoses/{i}', item)
                   for i, item in enumerate(supplied.diagnostic.differential_diagnoses)]
    medical_ids = {item.source_id for item in supplied.medical_knowledge}
    for path, hypothesis in hypotheses:
        claims = [hypothesis.rationale, *hypothesis.supporting_evidence, *hypothesis.contradicting_evidence]
        refs = {ref for claim in claims for ref in claim.evidence_refs}
        if not refs & medical_ids:
            findings.append(SafetyFinding(finding_id='deterministic:medical:' + path, origin='deterministic',
                code='missing_medical_reference', disposition='HUMAN_REVIEW',
                explanation='A non-abstaining hypothesis has no structured medical-evidence reference; correctness is not inferred.',
                anchor=ContentAnchor(target='diagnostic', field_path=path + '/condition', excerpt=hypothesis.condition[:2000])))
    return tuple(findings)


def make_assessment(supplied, ticket, semantic, *, completed_at=None):
    """Called by the application after identity checks; LLM never authors a decision."""
    validate_ticket_input(ticket, supplied)
    findings = list(deterministic_findings(supplied))
    critic_block = bool(supplied.critique.safety_flags)
    if critic_block:
        if semantic is not None:
            raise ValueError('Semantic generation must be skipped after a critic block')
    else:
        semantic = validate_semantic(semantic, supplied)
        for category in semantic.categories:
            if category.result == 'no_issue_identified':
                continue
            disposition = ('BLOCK' if category.result == 'issue_identified' and category.category in BLOCK_CATEGORIES
                           else 'HUMAN_REVIEW')
            # Uncertainty can concern missing context and need not invent a quotation.
            entries = category.findings or [None]
            for i, entry in enumerate(entries):
                findings.append(SafetyFinding(finding_id=f'semantic:{category.category}:{i}', origin='semantic',
                    code=category.category, disposition=disposition,
                    explanation=entry.explanation if entry else category.explanation,
                    anchor=entry.anchor.model_copy(deep=True) if entry else None,
                    evidence_refs=tuple(entry.evidence_refs) if entry else ()))
    decision = 'BLOCK' if any(f.disposition == 'BLOCK' for f in findings) else ('HUMAN_REVIEW' if findings else 'CONTINUE')
    return SafetyAssessment(ticket=ticket, deterministic_checks=DETERMINISTIC_CHECKS,
        semantic_status='skipped_critic_block' if critic_block else 'completed',
        semantic_result=semantic.model_copy(deep=True) if semantic else None,
        findings=tuple(findings), decision=decision, reasons=tuple(sorted({f.code for f in findings})),
        completed_at=completed_at or timestamp())


def validate_assessment(assessment, supplied, expected_ticket):
    """Recompute policy at consumption, rejecting forged permissive assessment objects."""
    from orchestration.phase6.contracts import require_current
    if type(assessment) is not SafetyAssessment:
        raise TypeError('Wrong assessment type')
    rebuilt = SafetyAssessment.model_validate(assessment.model_dump())
    require_current(expected_ticket, rebuilt.ticket)
    expected = make_assessment(supplied, expected_ticket, rebuilt.semantic_result, completed_at=rebuilt.completed_at)
    if expected != rebuilt:
        raise ValueError('Safety assessment does not match application policy')

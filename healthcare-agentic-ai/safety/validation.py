"""Deterministic safety-result validation, not a clinical entailment test."""
import re
from rag.agents.grounding import context, validate_references, facts
from .models import SafetyInput, SemanticSafetyResult, ContentAnchor


def validate_input(value: SafetyInput) -> SafetyInput:
    if type(value) is not SafetyInput:
        raise TypeError('Restricted safety input required')
    value = SafetyInput.model_validate(value.model_dump())
    context(value.patient_state, value.patient_cases, value.medical_knowledge)
    for items in (value.patient_cases, value.medical_knowledge):
        if len({item.source_id for item in items}) != len(items):
            raise ValueError('Duplicate safety evidence')
    validate_references(value.diagnostic, value.patient_state, value.patient_cases, value.medical_knowledge)
    validate_references(value.critique, value.patient_state, value.patient_cases, value.medical_knowledge)
    return value


def validate_anchor(anchor: ContentAnchor, supplied: SafetyInput):
    path = anchor.field_path
    if not path.startswith('/') or re.search(r'~(?![01])', path):
        raise ValueError('Invalid finding pointer')
    node = getattr(supplied, anchor.target).model_dump()
    for encoded in path[1:].split('/'):
        key = encoded.replace('~1', '/').replace('~0', '~')
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and re.fullmatch(r'0|[1-9]\d*', key) and int(key) < len(node):
            node = node[int(key)]
        else:
            raise ValueError('Unknown finding pointer')
    if not isinstance(node, str) or not anchor.excerpt.strip() or anchor.excerpt not in node:
        raise ValueError('Finding quote must occur exactly in the selected string')


def validate_semantic(result: SemanticSafetyResult, supplied: SafetyInput) -> SemanticSafetyResult:
    if type(result) is not SemanticSafetyResult:
        raise TypeError('Wrong safety output contract')
    result = SemanticSafetyResult.model_validate(result.model_dump())
    allowed = set(facts(supplied.patient_state)) | {
        item.source_id for item in supplied.patient_cases + supplied.medical_knowledge}
    for category in result.categories:
        if not category.explanation.strip():
            raise ValueError('Empty category explanation')
        seen = set()
        for finding in category.findings:
            if not finding.explanation.strip():
                raise ValueError('Empty finding explanation')
            validate_anchor(finding.anchor, supplied)
            if len(set(finding.evidence_refs)) != len(finding.evidence_refs) or not set(finding.evidence_refs) <= allowed:
                raise ValueError('Invalid safety evidence references')
            key = (finding.anchor.target, finding.anchor.field_path, finding.anchor.excerpt)
            if key in seen:
                raise ValueError('Duplicate safety finding')
            seen.add(key)
    # Like existing grounding: reject literal new URLs, without pretending to prove entailment.
    original = supplied.model_dump_json()
    for url in re.findall(r'https?://[^\s"<>]+', result.model_dump_json()):
        if url.rstrip('.,;)\\') not in original:
            raise ValueError('Unretrieved safety URL')
    return result


def validate_ticket_input(ticket, supplied):
    from orchestration.evidence import fingerprint
    from orchestration.phase6.contracts import SafetyTicket
    from orchestration.policy import WorkflowStop
    if type(ticket) is not SafetyTicket:
        raise WorkflowStop('stale_result')
    SafetyTicket.model_validate(ticket.model_dump())
    validate_input(supplied)
    snapshot_id = fingerprint({'patient_cases': [e.model_dump() for e in supplied.patient_cases],
                               'medical_knowledge': [e.model_dump() for e in supplied.medical_knowledge]})
    if (ticket.patient_id != supplied.patient_state.patient_id or
            ticket.revision_number + 1 != ticket.diagnostic_version or
            ticket.input_fingerprint != fingerprint(supplied.model_dump()) or
            ticket.diagnostic_fingerprint != fingerprint(supplied.diagnostic.model_dump()) or
            ticket.critic_fingerprint != fingerprint(supplied.critique.model_dump()) or
            ticket.evidence_snapshot_id != snapshot_id):
        raise WorkflowStop('stale_result')

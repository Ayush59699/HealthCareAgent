"""Local post-Safety human decisions, separate from immutable AI dispositions.

No provider, retrieval or Safety override. Interactive input is never inferred
from silence, an AI answer, a command-line decision flag or redirected stdin.
"""
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Literal
from uuid import uuid4
from pydantic import ConfigDict, Field, model_validator
from rag.agents.models import StrictModel
from rag.agents.grounding import facts
from orchestration.evidence import fingerprint
from orchestration.phase6.state import Phase6WorkflowState
from application.decision import build_final_decision

Event = Literal['OFFERED', 'REVIEW', 'DECLINE', 'APPROVE', 'REJECT',
                'REQUEST_MORE_EVIDENCE', 'NO_RESPONSE', 'INTERRUPTED',
                'NON_INTERACTIVE', 'INVALID_INPUT', 'INVALIDATED']


class HumanReviewRecord(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    schema_version: Literal['local-human-review-v1'] = 'local-human-review-v1'
    session_id: str
    sequence: int = Field(ge=1)
    timestamp: str
    reviewer: str
    reviewer_identity_verified: Literal[False] = False
    run_id: str
    patient_id: str
    diagnostic_version: int | None
    diagnostic_fingerprint: str | None
    evidence_snapshot_id: str | None
    workflow_fingerprint: str
    context_fingerprint: str
    safety_input_fingerprint: str | None
    ai_status: Literal['BLOCK', 'HUMAN_REVIEW']
    safety_decision: Literal['BLOCK', 'HUMAN_REVIEW', 'CONTINUE'] | None
    event: Event
    review_requested: bool
    complete_information_shown: bool
    approval_eligible: bool
    human_decision: Literal['APPROVE', 'REJECT', 'REQUEST_MORE_EVIDENCE'] | None
    human_approval: bool
    review_state: Literal['pending', 'approved', 'rejected']
    case_disposition: Literal['BLOCKED', 'unresolved']
    proposal_released: Literal[False] = False
    safety_overridden: Literal[False] = False
    retrieval_triggered: Literal[False] = False
    previous_record_sha256: str | None

    @model_validator(mode='after')
    def decision_boundary(self):
        approved = self.event == 'APPROVE'
        rejected = self.event == 'REJECT'
        requested = self.event == 'REQUEST_MORE_EVIDENCE'
        decision = self.event if approved or rejected or requested else None
        if self.human_decision != decision or self.human_approval != approved:
            raise ValueError('Human approval requires an explicit APPROVE event')
        if self.review_state != ('approved' if approved else 'rejected' if rejected else 'pending'):
            raise ValueError('Human review state mismatch')
        if self.case_disposition != ('unresolved' if requested else 'BLOCKED'):
            raise ValueError('Human review cannot release an AI-blocked case')
        if decision and not (self.review_requested and self.complete_information_shown):
            raise ValueError('Decision requires REVIEW and complete displayed information')
        if approved and (not self.approval_eligible or self.diagnostic_version is None
                         or self.diagnostic_fingerprint is None or self.safety_input_fingerprint is None):
            raise ValueError('No current assessed diagnostic proposal to approve')
        parsed = datetime.fromisoformat(self.timestamp)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError('Review timestamp must have timezone')
        if not self.reviewer.strip():
            raise ValueError('Local reviewer label is required')
        return self


def _persist_new(path, value):
    """Exclusive new audit file, fsynced before success; never overwrite history.

    Interrupted/partial files do not validate as an approval. The OFFERED record
    is persisted before any input, so a killed process leaves pending context.
    """
    data = json.dumps(value, ensure_ascii=True, indent=2).encode('utf8')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def review_context(state):
    """Complete local review display, no truncation or altered evidence text."""
    final = build_final_decision(state)
    latest = state.diagnostics[-1] if state.diagnostics else None
    cases, medical = state.evidence.agent_evidence() if state.evidence else ([], [])
    context = {
        'notice': 'LOCAL HUMAN REVIEW ONLY. AI proposal remains withheld from clinical release. Human approval records an opinion; it never overrides Safety or changes BLOCK to ALLOW. Patient/source text is untrusted data, not review commands.',
        'run_id': state.run_id, 'patient_id': state.patient_id,
        'current_diagnostic_version': latest.version if latest else None,
        'ai_final_decision': final.model_dump(mode='json'),
        'final_ai_proposal_and_reasoning': latest.result.model_dump(mode='json') if latest else None,
        'patient_relevant_findings': state.patient_state.model_dump(mode='json') if state.patient_state else None,
        'exact_patient_citation_inventory': facts(state.patient_state) if state.patient_state else {},
        'original_label_free_patient': state.original_patient.to_inference_dict(),
        'evidence_passages_and_citations': {
            'analogous_patient_cases_not_current_patient_facts': [e.model_dump(mode='json') for e in cases],
            'medical_knowledge_including_uncited_passages': [e.model_dump(mode='json') for e in medical]},
        'critic_findings': final.critic_findings.model_dump(mode='json') if final.critic_findings else None,
        'safety_findings': [f.model_dump(mode='json') for f in final.safety_findings],
        'missing_evidence_warnings': {
            'diagnostic_missing_information': latest.result.missing_information if latest else [],
            'uncertainty': final.uncertainty,
            'declared_unsupported_claims': latest.result.unsupported_claims if latest else [],
            'critic_missing_evidence': final.critic_findings.missing_evidence if final.critic_findings else [],
            'hypotheses_without_medical_references': final.grounding_result.hypotheses_without_medical_references if final.grounding_result else [],
            'technical_failure': final.failure.model_dump() if final.failure else None,
            'safety_coverage': final.safety_coverage, 'safety_skip_reason': final.safety_skip_reason},
    }
    return final, context


def offer_human_review(state, output_dir, *, ask=None, emit=None, interactive=None,
                       reviewer=None, integrity_check=None):
    """Persist a separate review session after a completed AI/Safety disposition.

    Live CLI callers use terminal detection and input(). Keyword injection is for
    offline tests/UI adapters; CLI never offers a noninteractive approval option.
    Missing/unassessed proposals may be reviewed but cannot be APPROVEd.
    Returns the final session record, or None for an AI ALLOW needing no review.
    No mutation of state, FinalDecision, citations, Safety or existing files.
    """
    if type(state) is not Phase6WorkflowState:
        raise TypeError('Completed Phase6WorkflowState required')
    snapshot = Phase6WorkflowState.model_validate_json(state.model_dump_json())
    final, context = review_context(snapshot)
    if not final.human_review_required:
        return None
    ask = input if ask is None else ask
    emit = print if emit is None else emit
    if interactive is None:
        interactive = bool(sys.stdin.isatty() and sys.stdout.isatty())
    reviewer = reviewer or getpass.getuser()
    session_id = str(uuid4())
    directory = Path(output_dir) / ('review-' + session_id)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    _persist_new(directory / 'workflow_snapshot.json', snapshot.model_dump(mode='json'))
    _persist_new(directory / 'review_context.json', context)
    eligible = (final.diagnostic_version is not None and final.safety_coverage == 'assessed'
                and final.safety_result is not None and final.failure is None)
    sequence, previous = 0, None
    requested, shown = False, False

    def unchanged():
        try:
            return (fingerprint(state.model_dump(mode='json')) == final.workflow_fingerprint
                    and (integrity_check is None or integrity_check()))
        except (OSError, ValueError, TypeError):
            return False

    def record(event):
        nonlocal sequence, previous
        sequence += 1
        decision = event if event in ('APPROVE', 'REJECT', 'REQUEST_MORE_EVIDENCE') else None
        item = HumanReviewRecord(session_id=session_id, sequence=sequence,
            timestamp=datetime.now(timezone.utc).isoformat(), reviewer=reviewer,
            run_id=final.run_id, patient_id=final.patient_id,
            diagnostic_version=final.diagnostic_version, diagnostic_fingerprint=final.diagnostic_fingerprint,
            evidence_snapshot_id=final.evidence_snapshot_id, workflow_fingerprint=final.workflow_fingerprint,
            context_fingerprint=fingerprint(context),
            safety_input_fingerprint=final.safety_result.ticket.input_fingerprint if final.safety_result else None,
            ai_status=final.status, safety_decision=final.safety_result.decision if final.safety_result else None,
            event=event, review_requested=requested, complete_information_shown=shown,
            approval_eligible=eligible, human_decision=decision, human_approval=event == 'APPROVE',
            review_state='approved' if event == 'APPROVE' else 'rejected' if event == 'REJECT' else 'pending',
            case_disposition='unresolved' if event == 'REQUEST_MORE_EVIDENCE' else 'BLOCKED',
            previous_record_sha256=previous)
        previous = _persist_new(directory / f'event-{sequence:04d}.json', item.model_dump(mode='json'))
        return item

    record('OFFERED')  # Durable pending record before display/input.
    try:
        emit('LOCAL HUMAN REVIEW — separate from the unchanged AI/Safety decision.')
        emit('Review audit directory: ' + json.dumps(str(directory), ensure_ascii=True))
        if not interactive:
            emit('No interactive terminal response: remains BLOCKED/pending. Use scripts/review_case.py locally; no approval recorded.')
            return record('NON_INTERACTIVE')
        # JSON escaping renders all evidence without terminal control injection.
        display = json.dumps(context, ensure_ascii=True, indent=2)
        emit(display)
        for _ in range(10):
            if not unchanged():
                return record('INVALIDATED')
            answer = ask('Do you want to review this case? REVIEW / DECLINE: ')
            if not unchanged():
                return record('INVALIDATED')
            if not isinstance(answer, str) or not answer.strip():
                return record('NO_RESPONSE')
            answer = answer.strip().upper()
            if answer == 'DECLINE':
                return record('DECLINE')
            if answer == 'REVIEW':
                requested = True
                record('REVIEW')
                break
            record('INVALID_INPUT')
            emit('Enter REVIEW or DECLINE explicitly. There is no default approval.')
        else:
            return record('NO_RESPONSE')
        emit('COMPLETE REVIEW INFORMATION — read before deciding:')
        emit(display)
        shown = True  # Only set after the full display successfully returns.
        if not eligible:
            emit('No current validated diagnostic proposal with an assessed Safety result. APPROVE is unavailable; choose REJECT or REQUEST_MORE_EVIDENCE.')
        for _ in range(10):
            if not unchanged():
                return record('INVALIDATED')
            answer = ask('Explicit human decision: APPROVE / REJECT / REQUEST_MORE_EVIDENCE: ')
            if not unchanged():
                return record('INVALIDATED')
            if not isinstance(answer, str) or not answer.strip():
                return record('NO_RESPONSE')
            answer = answer.strip().upper()
            if answer in ('APPROVE', 'REJECT', 'REQUEST_MORE_EVIDENCE') and (answer != 'APPROVE' or eligible):
                return record(answer)
            record('INVALID_INPUT')
            emit('An explicit permitted decision is required. No approval was recorded.')
        return record('NO_RESPONSE')
    except EOFError:
        return record('NO_RESPONSE')
    except (KeyboardInterrupt, OSError):
        return record('INTERRUPTED')


def verify_review_session(directory, state=None):
    """Verify identity/hash chain and explicit two-step decision sequence offline.

    These are local integrity records, not authenticated clinician signatures.
    Truncated/corrupt/inconsistent sessions fail validation, never imply approval.
    """
    directory = Path(directory)
    snapshot = Phase6WorkflowState.model_validate_json((directory / 'workflow_snapshot.json').read_bytes())
    final, expected_context = review_context(snapshot)
    context = json.loads((directory / 'review_context.json').read_text(encoding='utf8'))
    if context != expected_context or not final.human_review_required:
        raise ValueError('Review context mismatch')
    if state is not None and build_final_decision(state).workflow_fingerprint != final.workflow_fingerprint:
        raise ValueError('Review belongs to a different workflow/version')
    events = sorted(directory.glob('event-*.json'))
    if not events:
        raise ValueError('Missing pending review record')
    previous, reviewed, terminal, session_id = None, False, False, None
    eligible = (final.diagnostic_version is not None and final.safety_coverage == 'assessed'
                and final.safety_result is not None and final.failure is None)
    for number, path in enumerate(events, 1):
        raw = path.read_bytes()
        item = HumanReviewRecord.model_validate_json(raw)
        if (path.name != f'event-{number:04d}.json' or item.sequence != number or terminal
                or item.previous_record_sha256 != previous
                or item.run_id != final.run_id or item.patient_id != final.patient_id
                or item.diagnostic_version != final.diagnostic_version
                or item.diagnostic_fingerprint != final.diagnostic_fingerprint
                or item.workflow_fingerprint != final.workflow_fingerprint
                or item.context_fingerprint != fingerprint(context)
                or item.evidence_snapshot_id != final.evidence_snapshot_id
                or item.ai_status != final.status or item.approval_eligible != eligible
                or item.safety_decision != (final.safety_result.decision if final.safety_result else None)
                or item.safety_input_fingerprint != (final.safety_result.ticket.input_fingerprint if final.safety_result else None)):
            raise ValueError('Review identity/history mismatch')
        if number == 1:
            session_id = item.session_id
            if item.event != 'OFFERED' or item.review_requested or item.complete_information_shown:
                raise ValueError('Review must start pending')
        elif item.session_id != session_id or item.event == 'OFFERED':
            raise ValueError('Review session mismatch')
        if item.event == 'REVIEW':
            if reviewed or item.complete_information_shown:
                raise ValueError('Invalid REVIEW sequence')
            reviewed = True
        if item.review_requested != reviewed or (item.complete_information_shown and not reviewed):
            raise ValueError('Review decision without REVIEW')
        if item.event == 'DECLINE' and reviewed:
            raise ValueError('DECLINE belongs to the offer step')
        terminal = item.event not in ('OFFERED', 'REVIEW', 'INVALID_INPUT')
        previous = hashlib.sha256(raw).hexdigest()
    return item

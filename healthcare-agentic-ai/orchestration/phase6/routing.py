"""Monotonic safety restriction around the unchanged Phase 5 routing function."""
from orchestration.routing import route as phase5_route, TRANSITIONS as PHASE5_TRANSITIONS, TERMINAL as PHASE5_TERMINAL
from orchestration.policy import WorkflowStop
from rag.agents.models import StrictModel
from safety.policy import validate_assessment
from .events import Stage
from .state import Outcome

TRANSITIONS = {key: set(value) for key, value in PHASE5_TRANSITIONS.items()}
TRANSITIONS['CRITIC'] = {'SAFETY_VALIDATION'}
TRANSITIONS['SAFETY_VALIDATION'] = {'ROUTE'}
TRANSITIONS['ROUTE'].add('HUMAN_REVIEW')
TERMINAL = PHASE5_TERMINAL | {'HUMAN_REVIEW'}


class Phase6RouteDecision(StrictModel):
    next_stage: Stage
    outcome: Outcome | None = None
    reason: str


def require_transition(source, target):
    if source in TERMINAL or (target != 'TERMINAL_FAILURE' and target not in TRANSITIONS.get(source, set())):
        raise WorkflowStop('invalid_transition')


def route(supplied, assessment, expected_ticket, revision_count, max_revisions=2):
    if assessment is None:
        raise WorkflowStop('safety_assessment_missing')
    if max_revisions != 2 or not 0 <= revision_count <= 2:
        raise WorkflowStop('invalid_revision_budget')
    if expected_ticket.revision_number != revision_count:
        raise WorkflowStop('stale_result')
    validate_assessment(assessment, supplied, expected_ticket)
    if supplied.critique.safety_flags or assessment.decision == 'BLOCK':
        return Phase6RouteDecision(next_stage='BLOCKED', outcome='safety_blocked',
            reason='critic_safety_flags' if supplied.critique.safety_flags else 'safety_block')
    if assessment.decision == 'HUMAN_REVIEW':
        return Phase6RouteDecision(next_stage='HUMAN_REVIEW', outcome='human_review_required', reason='safety_human_review')
    # CONTINUE never means automatic acceptance: all baseline routing priorities remain.
    decision = phase5_route(supplied.diagnostic, supplied.critique, revision_count, max_revisions)
    return Phase6RouteDecision(**decision.model_dump())

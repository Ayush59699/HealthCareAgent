"""Pure routing policy; LLM output can never skip the grounding gate."""
from rag.agents.models import DiagnosticResult, ClinicalCritique, StrictModel
from .events import Stage
from .state import Outcome
from .policy import WorkflowStop


class RouteDecision(StrictModel):
    next_stage: Stage
    outcome: Outcome | None = None
    reason: str


TRANSITIONS = {
    'INITIAL': {'PATIENT'}, 'PATIENT': {'EVIDENCE'}, 'EVIDENCE': {'DIAGNOSTIC', 'ABSTENTION'},
    'DIAGNOSTIC': {'GROUNDING'}, 'DIAGNOSTIC_REVISION': {'GROUNDING'},
    'GROUNDING': {'CRITIC', 'UNRESOLVED'}, 'CRITIC': {'ROUTE'},
    'ROUTE': {'FINAL', 'ABSTENTION', 'UNRESOLVED', 'BLOCKED', 'DIAGNOSTIC_REVISION'},
}
TERMINAL = frozenset({'FINAL', 'ABSTENTION', 'UNRESOLVED', 'BLOCKED', 'TERMINAL_FAILURE'})


def require_transition(source: Stage, target: Stage):
    if source in TERMINAL or (target != 'TERMINAL_FAILURE' and target not in TRANSITIONS.get(source, set())):
        raise WorkflowStop('invalid_transition')


def route(diagnosis: DiagnosticResult, critique: ClinicalCritique, revision_count: int,
          max_revisions: int = 2) -> RouteDecision:
    if max_revisions != 2 or not 0 <= revision_count <= 2:
        raise WorkflowStop('invalid_revision_budget')
    if critique.safety_flags:
        return RouteDecision(next_stage='BLOCKED', outcome='safety_blocked', reason='critic_safety_flags')
    if critique.overall_assessment == 'insufficient_evidence':
        return RouteDecision(next_stage='ABSTENTION', outcome='insufficient_evidence', reason='critic_insufficient_evidence')
    needs_revision = (critique.overall_assessment == 'revision_required' or bool(
        critique.unsupported_points or critique.contradictions or critique.hallucination_flags or
        critique.recommended_revisions or diagnosis.unsupported_claims))
    if needs_revision:
        if revision_count >= max_revisions:
            return RouteDecision(next_stage='UNRESOLVED', outcome='max_revisions_reached', reason='max_revisions_reached')
        return RouteDecision(next_stage='DIAGNOSTIC_REVISION', reason='unresolved_critic')
    if diagnosis.primary_hypothesis is None and not diagnosis.differential_diagnoses:
        return RouteDecision(next_stage='ABSTENTION', outcome='diagnostic_agent_abstention', reason='diagnostic_agent_abstention')
    return RouteDecision(next_stage='FINAL', outcome='accepted_with_limitations', reason='critic_supported_with_limitations')

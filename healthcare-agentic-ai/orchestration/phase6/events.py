"""Phase 6 stage/event extensions; Phase 5 literals are not changed."""
from typing import Literal, Union
from orchestration.events import Stage as Phase5Stage, EventName as Phase5EventName, WorkflowEvent

Stage = Union[Phase5Stage, Literal['SAFETY_VALIDATION', 'HUMAN_REVIEW']]
EventName = Union[Phase5EventName, Literal['safety_started', 'safety_deterministic_completed',
    'safety_semantic_completed', 'safety_validated', 'safety_decision_recorded', 'human_review_required']]


class Phase6Event(WorkflowEvent):
    event: EventName
    from_stage: Stage
    stage: Stage
    critic_fingerprint: str | None = None
    safety_input_fingerprint: str | None = None
    safety_policy_version: Literal['phase6-safety-v1'] | None = None
    safety_decision: Literal['CONTINUE', 'HUMAN_REVIEW', 'BLOCK'] | None = None
    finding_count: int | None = None

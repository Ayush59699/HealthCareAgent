"""Separate versioned safety policy configuration; retain Phase 5 budget defaults."""
from typing import Literal
from orchestration.policy import WorkflowPolicy, SAFE_WORKFLOW_CODES as PHASE5_CODES


class Phase6Policy(WorkflowPolicy):
    safety_policy_version: Literal['phase6-safety-v1'] = 'phase6-safety-v1'


SAFE_WORKFLOW_CODES = PHASE5_CODES | frozenset({
    'safety_validation_failure', 'safety_assessment_missing', 'safety_validator_unavailable',
})

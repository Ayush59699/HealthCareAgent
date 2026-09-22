"""Pure routing priorities and illegal transition rejection."""
import unittest
from pydantic import ValidationError
from orchestration.routing import route, require_transition, TERMINAL
from orchestration.policy import WorkflowPolicy, WorkflowStop
from tests.orchestration_helpers import diagnosis, critique


class RoutingTests(unittest.TestCase):
    def test_acceptance(self):
        self.assertEqual(route(diagnosis(), critique(), 0).next_stage, 'FINAL')

    def test_revision_and_limit(self):
        for revision in (0, 1):
            self.assertEqual(route(diagnosis(), critique('revision_required'), revision).next_stage, 'DIAGNOSTIC_REVISION')
        self.assertEqual(route(diagnosis(), critique('revision_required'), 2).outcome, 'max_revisions_reached')

    def test_safety_has_priority_over_insufficiency_and_budget(self):
        self.assertEqual(route(diagnosis(), critique('insufficient_evidence', safety_flags=['Review']), 2).outcome, 'safety_blocked')

    def test_contradictory_acceptance_never_finalizes(self):
        for field in ('unsupported_points', 'contradictions', 'hallucination_flags', 'recommended_revisions'):
            with self.subTest(field=field):
                self.assertEqual(route(diagnosis(), critique(**{field: ['Issue']}), 0).next_stage, 'DIAGNOSTIC_REVISION')
        output = diagnosis()
        output.unsupported_claims = ['Issue']
        self.assertEqual(route(output, critique(), 0).next_stage, 'DIAGNOSTIC_REVISION')

    def test_abstention_is_not_forced_into_a_diagnosis(self):
        self.assertEqual(route(diagnosis(abstain=True), critique(), 0).outcome, 'diagnostic_agent_abstention')
        self.assertEqual(route(diagnosis(abstain=True), critique('revision_required'), 2).outcome, 'max_revisions_reached')

    def test_cannot_skip_gate_or_reopen_terminal_state(self):
        for source, target in [('DIAGNOSTIC', 'CRITIC'), ('DIAGNOSTIC_REVISION', 'FINAL'),
                               ('PATIENT', 'DIAGNOSTIC'), ('GROUNDING', 'FINAL')]:
            with self.assertRaises(WorkflowStop):
                require_transition(source, target)
        for source in TERMINAL:
            with self.assertRaises(WorkflowStop):
                require_transition(source, 'DIAGNOSTIC')
        require_transition('DIAGNOSTIC', 'GROUNDING')

    def test_policy_does_not_allow_unbounded_revisions(self):
        for changes in ({'max_revisions': 3}, {'max_requests': 0}, {'max_seconds': float('inf')},
                        {'max_seconds': -1.0}, {'provider_attempt_limit': 4}):
            with self.assertRaises(ValidationError):
                WorkflowPolicy(**changes)

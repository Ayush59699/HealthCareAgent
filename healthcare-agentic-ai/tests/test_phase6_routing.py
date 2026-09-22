"""No gate skipping or weakening of baseline routing priorities."""
import unittest
from orchestration.phase6.routing import route, require_transition, TERMINAL
from orchestration.policy import WorkflowStop
from safety.policy import make_assessment
from tests.safety_helpers import inputs, semantic, reticket, critique


class Phase6RoutingTests(unittest.TestCase):
    def setUp(self):
        self.value, self.ticket = inputs()

    def decide(self, output, revision=0):
        ticket = reticket(self.value, self.ticket).model_copy(update={'revision_number': revision, 'diagnostic_version': revision + 1})
        assessment = make_assessment(self.value, ticket, output)
        return route(self.value, assessment, ticket, revision)

    def test_continue_delegates_to_baseline_route(self):
        for assessment, revision, target in (('supported_with_limitations', 0, 'FINAL'),
                                            ('insufficient_evidence', 0, 'ABSTENTION'),
                                            ('revision_required', 0, 'DIAGNOSTIC_REVISION'),
                                            ('revision_required', 2, 'UNRESOLVED')):
            self.value.critique.overall_assessment = assessment
            self.assertEqual(self.decide(semantic(), revision).next_stage, target)

    def test_human_review_and_block_override_baseline_continuation(self):
        self.assertEqual(self.decide(semantic('safety_ambiguity')).next_stage, 'HUMAN_REVIEW')
        self.assertEqual(self.decide(semantic('unsafe_delay', status='issue_identified')).next_stage, 'BLOCKED')

    def test_critic_flag_has_priority_over_insufficiency_and_revision_limit(self):
        self.value = self.value.model_copy(update={'critique': critique('insufficient_evidence', safety_flags=['Review'])})
        self.assertEqual(self.decide(None, 2).outcome, 'safety_blocked')

    def test_missing_assessment_is_not_continue(self):
        with self.assertRaises(WorkflowStop) as failure:
            route(self.value, None, self.ticket, 0)
        self.assertEqual(failure.exception.code, 'safety_assessment_missing')

    def test_unknown_or_forged_decision_cannot_route(self):
        assessment = make_assessment(self.value, self.ticket, semantic('safety_ambiguity'))
        for changes in ({'decision': 'CONTINUE', 'findings': (), 'reasons': ()}, {'decision': 'NEW'},
                        {'deterministic_checks': ()}):
            with self.assertRaises(ValueError):
                route(self.value, assessment.model_copy(update=changes), self.ticket, 0)

    def test_input_changed_since_safety_cannot_route(self):
        assessment = make_assessment(self.value, self.ticket, semantic())
        self.value.diagnostic.reasoning_summary = 'Changed'
        with self.assertRaises(WorkflowStop):
            route(self.value, assessment, self.ticket, 0)

    def test_cannot_skip_gates_or_reopen_terminal_states(self):
        for source, target in (('CRITIC', 'ROUTE'), ('CRITIC', 'FINAL'), ('DIAGNOSTIC', 'SAFETY_VALIDATION'),
                               ('GROUNDING', 'FINAL'), ('SAFETY_VALIDATION', 'FINAL'), ('SAFETY_VALIDATION', 'DIAGNOSTIC_REVISION')):
            with self.assertRaises(WorkflowStop):
                require_transition(source, target)
        for terminal in TERMINAL:
            with self.assertRaises(WorkflowStop):
                require_transition(terminal, 'DIAGNOSTIC')
        require_transition('CRITIC', 'SAFETY_VALIDATION')
        require_transition('SAFETY_VALIDATION', 'ROUTE')
        require_transition('ROUTE', 'HUMAN_REVIEW')

    def test_no_unbounded_revisions(self):
        assessment = make_assessment(self.value, self.ticket, semantic())
        for revision, limit in ((3, 2), (0, 3)):
            with self.assertRaises(WorkflowStop):
                route(self.value, assessment, self.ticket, revision, limit)

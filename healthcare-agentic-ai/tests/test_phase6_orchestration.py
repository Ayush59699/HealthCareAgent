"""Phase 6 gate ordering, terminal outcomes, revisions, and frozen evidence."""
import json
import unittest
from unittest.mock import Mock
from orchestration.phase6 import Phase6WorkflowState
from tests.safety_helpers import setup, run, SafetyScenarioLLM, diagnosis, critique, semantic, generation


class Phase6OrchestrationTests(unittest.TestCase):
    def test_success_and_state_round_trip(self):
        fixture = setup()
        state = run(fixture)
        self.assertEqual((state.status, state.outcome), ('final', 'accepted_with_limitations'))
        self.assertEqual(state.total_requests, 4)
        self.assertEqual(state.safety_coverage, 'assessed')
        self.assertIsNone(state.safety_skip_reason)
        self.assertEqual(state.safety_assessments[-1].decision, 'CONTINUE')
        self.assertEqual(Phase6WorkflowState.model_validate_json(state.model_dump_json()), state)
        fixture[3].retrieve.assert_called_once()
        fixture[4].retrieve.assert_called_once()

    def test_critic_block_skips_semantic_generation(self):
        fixture = setup(SafetyScenarioLLM(critiques=[critique(safety_flags=['Review'])]))
        fixture[0].safety.run = Mock(side_effect=AssertionError('Must not call'))
        state = run(fixture)
        self.assertEqual((state.status, state.outcome), ('blocked', 'safety_blocked'))
        self.assertEqual(state.total_requests, 3)
        self.assertEqual(state.safety_assessments[-1].semantic_status, 'skipped_critic_block')
        fixture[0].safety.run.assert_not_called()

    def test_semantic_block_overrides_supported_critic(self):
        state = run(setup(SafetyScenarioLLM(safety_results=[semantic('prohibited_action', status='issue_identified')])))
        self.assertEqual((state.status, state.outcome), ('blocked', 'safety_blocked'))
        self.assertNotIn('finalized', [e.event for e in state.transition_history])

    def test_uncertainty_is_terminal_human_review_without_revision(self):
        state = run(setup(SafetyScenarioLLM(critiques=[critique('revision_required')],
                                         safety_results=[semantic('safety_ambiguity')])))
        self.assertEqual((state.status, state.outcome), ('human_review_required', 'human_review_required'))
        self.assertEqual(state.revision_count, 0)
        self.assertEqual(len(state.diagnostics), 1)

    def test_missing_medical_citation_is_new_restriction_not_grounding_failure(self):
        state = run(setup(SafetyScenarioLLM(diagnoses=[diagnosis(medical_reference=False)])))
        self.assertEqual(state.status, 'human_review_required')
        self.assertIsNone(state.failure)
        self.assertIn('grounding_passed', [e.event for e in state.transition_history])
        self.assertIn('missing_medical_reference', state.safety_assessments[-1].reasons)

    def test_abstention_still_requires_semantic_review(self):
        for output, status in ((semantic(), 'abstained'), (semantic('unsafe_delay', status='issue_identified'), 'blocked')):
            state = run(setup(SafetyScenarioLLM(diagnoses=[diagnosis(abstain=True)], safety_results=[output])))
            self.assertEqual(state.status, status)
            self.assertEqual(len(state.safety_assessments), 1)
            self.assertEqual(state.diagnostics[-1].result.medical_knowledge_evidence, [])

    def test_no_medical_evidence_exits_before_diagnosis_unassessed(self):
        fixture = setup()
        fixture[4].retrieve.return_value = []
        state = run(fixture)
        self.assertEqual((state.status, state.outcome), ('abstained', 'insufficient_evidence'))
        self.assertEqual(state.safety_coverage, 'not_assessed')
        self.assertEqual(state.safety_skip_reason, 'no_medical_evidence')
        self.assertEqual(state.diagnostics, ())
        self.assertEqual(state.safety_assessments, ())
        self.assertEqual(len(fixture[2].calls), 1)

    def test_empty_patient_analogies_do_not_block(self):
        fixture = setup()
        fixture[3].retrieve.return_value = []
        self.assertEqual(run(fixture).status, 'final')

    def test_two_revisions_get_distinct_safety_tickets(self):
        fixture = setup(SafetyScenarioLLM(diagnoses=[diagnosis(i) for i in range(3)],
                          critiques=[critique('revision_required'), critique('revision_required'), critique()]))
        state = run(fixture)
        self.assertEqual((state.status, state.revision_count), ('final', 2))
        self.assertEqual(state.total_requests, 10)
        self.assertEqual([a.ticket.diagnostic_version for a in state.safety_assessments], [1, 2, 3])
        self.assertEqual(len({a.ticket.input_fingerprint for a in state.safety_assessments}), 3)
        for a, d, c in zip(state.safety_assessments, state.diagnostics, state.critiques):
            self.assertEqual(a.ticket.diagnostic_fingerprint, d.fingerprint)
            self.assertEqual(a.ticket.evidence_snapshot_id, c.evidence_snapshot_id)
        payloads = [json.loads(c['input'][0]['content']) for c in fixture[2].calls]
        revised = [p for p in payloads if 'revision_input' in p]
        self.assertEqual(len(revised), 2)
        for p in revised:
            self.assertEqual(set(p['revision_input']), {'previous_diagnostic', 'critique', 'revision_number', 'constraints'})
            self.assertNotIn('safety_assessments', p)
        fixture[3].retrieve.assert_called_once()
        fixture[4].retrieve.assert_called_once()

    def test_revision_limit_unchanged(self):
        state = run(setup(SafetyScenarioLLM(diagnoses=[diagnosis(i) for i in range(3)], critiques=[critique('revision_required')])))
        self.assertEqual((state.status, state.outcome), ('unresolved', 'max_revisions_reached'))
        self.assertEqual(len(state.safety_assessments), 3)

    def test_adjacent_and_nonadjacent_repeats_are_unassessed(self):
        for outputs, reason, reviews in (([diagnosis()], 'unchanged_diagnostic', 1),
                                         ([diagnosis(0), diagnosis(1), diagnosis(0)], 'repeated_diagnostic', 2)):
            state = run(setup(SafetyScenarioLLM(diagnoses=outputs, critiques=[critique('revision_required')])))
            self.assertEqual(state.outcome, 'unresolved_critic')
            self.assertEqual(state.safety_skip_reason, reason)
            self.assertEqual(state.safety_coverage, 'not_assessed')
            self.assertEqual(len(state.safety_assessments), reviews)
            self.assertEqual(len(state.critiques), reviews)

    def test_safety_repairs_are_not_clinical_revisions(self):
        state = run(setup(SafetyScenarioLLM(safety_results=[{}, semantic()])))
        self.assertEqual(state.status, 'final')
        self.assertEqual(state.provider_repairs, 1)
        self.assertEqual(state.revision_count, 0)
        self.assertEqual(state.total_requests, 5)

    def test_audit_orders_mandatory_gates_and_sanitizes_events(self):
        state = run(setup())
        events = state.transition_history
        names = [e.event for e in events]
        ordered = ['grounding_passed', 'critic_completed', 'safety_started', 'safety_deterministic_completed',
                   'safety_semantic_completed', 'safety_validated', 'safety_decision_recorded', 'route_started', 'finalized']
        self.assertEqual([names.index(n) for n in ordered], sorted(names.index(n) for n in ordered))
        self.assertEqual([e.sequence for e in events], list(range(1, len(events) + 1)))
        self.assertTrue(all(e.safety_input_fingerprint for e in events[names.index('safety_started'):]))
        self.assertNotIn('Research fixture version', json.dumps([e.model_dump() for e in events]))

    def test_grounding_failure_never_reaches_safety(self):
        fixture = setup()
        bad = diagnosis()
        bad.primary_hypothesis.rationale.evidence_refs = ['invented']
        fixture[0].diagnostic.run = Mock(return_value=generation(bad))
        fixture[0].safety.run = Mock()
        state = run(fixture)
        self.assertEqual(state.failure.code, 'grounding_failure')
        self.assertEqual(state.safety_coverage, 'not_assessed')
        fixture[0].safety.run.assert_not_called()

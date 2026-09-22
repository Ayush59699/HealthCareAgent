"""Fail-closed identity, ownership, leakage and cumulative resource boundaries."""
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch
from rag.agents.models import Failure
from rag.models import EvaluationLabels, DifferentialDiagnosis
from rag.config import OpenAIConfig
from orchestration.phase6 import Phase6Policy
from orchestration.phase6.contracts import Phase6Response
from safety.models import SemanticSafetyResult
from tests.safety_helpers import setup, run, semantic, SafetyScenarioLLM, generation, diagnosis, critique


class Phase6BoundaryTests(unittest.TestCase):
    def test_every_safety_identity_field_checked(self):
        for changes in ({'run_id': 'other'}, {'patient_id': 'ddxplus:validate:999'}, {'stage': 'CRITIC'},
                        {'revision_number': 2}, {'diagnostic_version': 3}, {'diagnostic_fingerprint': 'a' * 64},
                        {'evidence_snapshot_id': 'a' * 64}, {'critic_fingerprint': 'a' * 64},
                        {'safety_policy_version': 'unknown'}, {'input_fingerprint': 'a' * 64}):
            with self.subTest(changes=changes):
                fixture = setup()
                original = fixture[0]._invoke
                def stale(ticket, operation):
                    response = original(ticket, operation)
                    if ticket.stage == 'SAFETY_VALIDATION':
                        return Phase6Response(ticket.model_copy(update=changes), response.generation)
                    return response
                with patch.object(fixture[0], '_invoke', side_effect=stale):
                    state = run(fixture)
                self.assertEqual(state.failure.code, 'stale_result')
                self.assertEqual(state.safety_assessments, ())
                self.assertEqual(state.safety_coverage, 'failed')

    def test_prior_version_safety_cannot_authorize_revision(self):
        fixture = setup(SafetyScenarioLLM(diagnoses=[diagnosis(0), diagnosis(1)],
                                        critiques=[critique('revision_required'), critique()]))
        original = fixture[0]._invoke
        prior = []
        def stale(ticket, operation):
            response = original(ticket, operation)
            if ticket.stage == 'SAFETY_VALIDATION':
                if prior:
                    return Phase6Response(prior[0], response.generation)
                prior.append(ticket)
            return response
        with patch.object(fixture[0], '_invoke', side_effect=stale):
            state = run(fixture)
        self.assertEqual(state.failure.code, 'stale_result')
        self.assertEqual(len(state.safety_assessments), 1)
        self.assertEqual(len(state.diagnostics), 2)
        self.assertEqual(state.safety_coverage, 'failed')

    def test_safety_schema_and_reference_gates_independent_of_provider(self):
        for bad, code in ((SemanticSafetyResult.model_construct(categories=[], limitations=[]), 'schema_validation_failure'),
                          (semantic('unsafe_delay', status='issue_identified', quote='SECRET invented'), 'safety_validation_failure'),
                          (semantic('unsafe_delay', status='issue_identified', refs=['medical:invented']), 'safety_validation_failure')):
            fixture = setup()
            fixture[0].safety.run = Mock(return_value=generation(bad, raw='SECRET'))
            state = run(fixture)
            self.assertEqual(state.failure.code, code)
            self.assertEqual(state.safety_assessments, ())
            self.assertNotIn('SECRET', state.model_dump_json())
            self.assertNotIn('route_started', [e.event for e in state.transition_history])

    def test_failures_never_fabricate_continue(self):
        for code in ('refusal', 'incomplete_output', 'timeout', 'api_failure', 'connection_failure', 'SECRET'):
            fixture = setup()
            fixture[0].safety.run = Mock(return_value=generation(failure=Failure(code=code, message='SECRET'), raw='SECRET'))
            state = run(fixture)
            self.assertEqual((state.status, state.outcome), ('failed', 'technical_failure'))
            self.assertEqual(state.failure.code, code if code != 'SECRET' else 'provider_failure')
            self.assertEqual(state.safety_assessments, ())
            self.assertNotIn('SECRET', state.model_dump_json())

    def test_safety_exception_unavailable_and_exhausted_repairs(self):
        fixture = setup()
        fixture[0].safety = None
        self.assertEqual(run(fixture).failure.code, 'safety_validator_unavailable')
        fixture = setup()
        fixture[0].safety.run = Mock(side_effect=RuntimeError('SECRET'))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'stage_failure')
        self.assertNotIn('SECRET', state.model_dump_json())
        state = run(setup(SafetyScenarioLLM(safety_results=[{}])))
        self.assertEqual(state.failure.code, 'parsing_failure')
        self.assertEqual(state.provider_repairs, 1)
        self.assertEqual(state.revision_count, 0)

    def test_invalid_telemetry_cannot_bypass_request_or_byte_limits(self):
        cases = [([], 1), ([{'attempt': 1, 'request_bytes': 0}], 1),
                 ([{'attempt': 1, 'request_bytes': float('nan')}], 1),
                 ([{'attempt': 2, 'request_bytes': 100}], 1),
                 ([{'attempt': 1, 'request_bytes': 100001}], 1), ([], 0)]
        for telemetry, attempts in cases:
            fixture = setup()
            result = generation(semantic())
            result.telemetry, result.attempts = telemetry, attempts
            fixture[0].safety.run = Mock(return_value=result)
            state = run(fixture)
            self.assertEqual(state.failure.code, 'invalid_provider_telemetry')
            self.assertEqual(state.safety_assessments, ())

    def test_budget_cannot_skip_required_safety(self):
        fixture = setup(policy=Phase6Policy(max_requests=4))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'request_budget_exhausted')
        self.assertEqual(state.total_requests, 3)
        self.assertEqual(fixture[2].safety_count, 0)
        self.assertEqual(state.safety_coverage, 'failed')
        state = run(setup(policy=Phase6Policy(max_request_bytes=100)))
        self.assertEqual(state.failure.code, 'request_budget_exhausted')
        self.assertEqual(state.total_requests, 0)

    def test_deadline_after_safety_call_rejects_output(self):
        now = [0.0]
        fixture = setup(policy=Phase6Policy(max_seconds=1.0), clock=lambda: now[0])
        original = fixture[0].safety.run
        def slow(value):
            result = original(value)
            now[0] = 2.0
            return result
        fixture[0].safety.run = slow
        state = run(fixture)
        self.assertEqual(state.failure.code, 'time_budget_exhausted')
        self.assertEqual(state.total_requests, 4)
        self.assertEqual(state.safety_assessments, ())

    def test_third_attempt_reservation_uses_actual_provider_limits(self):
        fixture = setup(SafetyScenarioLLM(config=OpenAIConfig(max_retries=2)), policy=Phase6Policy(max_requests=5))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'request_budget_exhausted')
        self.assertEqual(state.total_requests, 3)

    def test_handoffs_are_detached_and_runs_do_not_share_state(self):
        fixture = setup()
        original = fixture[0].safety.run
        def mutating(value):
            output = original(value)
            value.patient_state.symptoms.append('MUTATION')
            value.diagnostic.reasoning_summary = 'MUTATION'
            value.critique.safety_flags.append('MUTATION')
            value.medical_knowledge[0].metadata.clear()
            return output
        fixture[0].safety.run = mutating
        first, second = run(fixture), run(fixture)
        self.assertEqual(first.status, 'final')
        self.assertNotIn('MUTATION', first.model_dump_json())
        first.safety_assessments[-1].semantic_result.limitations.append('MUTATION')
        self.assertNotIn('MUTATION', second.model_dump_json())
        self.assertNotEqual(first.run_id, second.run_id)

    def test_labels_excluded_from_every_safety_request_and_artifact(self):
        fixture = list(setup(SafetyScenarioLLM(diagnoses=[diagnosis(0), diagnosis(1)],
                                             critiques=[critique('revision_required'), critique()])))
        fixture[1] = replace(fixture[1], labels=EvaluationLabels('SECRET_PRIMARY', (DifferentialDiagnosis('SECRET_OTHER', 1.0),)))
        state = run(fixture)
        text = state.model_dump_json() + json.dumps(fixture[2].calls)
        for marker in ('SECRET_PRIMARY', 'SECRET_OTHER', 'PATHOLOGY', 'EvaluationLabels'):
            self.assertNotIn(marker, text)
        for value in (fixture[1], {'PATHOLOGY': 'SECRET'}):
            with self.assertRaises(TypeError):
                fixture[0].run(value, fixture[1].patient_id)

    def test_corrupt_critic_never_reaches_safety(self):
        fixture = setup()
        bad = critique(supported_points=[{'statement': 'Bad', 'evidence_refs': ['invented']}])
        fixture[0].critic.run = Mock(return_value=generation(bad))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'grounding_failure')
        self.assertEqual(fixture[2].safety_count, 0)

    def test_deadlines_after_deterministic_policy_and_routing_work(self):
        import orchestration.phase6.orchestrator as module
        for operation in ('deterministic_findings', 'make_assessment', 'route'):
            now = [0.0]
            fixture = setup(policy=Phase6Policy(max_seconds=1.0), clock=lambda: now[0])
            original = getattr(module, operation)
            def slow(*args, **kwargs):
                result = original(*args, **kwargs)
                now[0] = 2.0
                return result
            with patch.object(module, operation, side_effect=slow):
                state = run(fixture)
            self.assertEqual(state.failure.code, 'time_budget_exhausted')
            self.assertNotIn('finalized', [e.event for e in state.transition_history])
            if operation == 'deterministic_findings':
                self.assertEqual(fixture[2].safety_count, 0)
            if operation != 'route':
                self.assertEqual(state.safety_assessments, ())

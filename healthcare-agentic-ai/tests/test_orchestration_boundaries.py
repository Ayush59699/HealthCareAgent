"""Ownership, label isolation, version identities, safe failure and budgets."""
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch
from pydantic import ValidationError
from rag.models import EvaluationLabels, DifferentialDiagnosis
from rag.agents.models import Failure
from rag.llm.provider import Generation
from orchestration import WorkflowPolicy
from orchestration.contracts import StageResponse
from tests.orchestration_helpers import setup, run, ScenarioLLM, diagnosis, critique
from tests.phase4_helpers import MockLLM


class BoundaryTests(unittest.TestCase):
    def test_no_evaluator_labels_in_state_or_agent_payloads_including_revisions(self):
        fixture = list(setup(ScenarioLLM([diagnosis(0), diagnosis(1)], [critique('revision_required'), critique()])))
        fixture[1] = replace(fixture[1], labels=EvaluationLabels('SECRET_PRIMARY', (DifferentialDiagnosis('SECRET_OTHER', 1.0),)))
        state = run(fixture)
        text = state.model_dump_json() + json.dumps(fixture[2].calls)
        for forbidden in ('SECRET_PRIMARY', 'SECRET_OTHER', 'ground_truth_pathology', 'EvaluationLabels', 'PATHOLOGY'):
            self.assertNotIn(forbidden, text)
        self.assertEqual(state.original_patient, fixture[1].patient)
        with self.assertRaises(ValidationError):
            type(state).model_validate({**state.model_dump(), 'EvaluationLabels': {}})

    def test_record_and_raw_label_rows_rejected_before_any_call(self):
        fixture = setup()
        for value in (fixture[1], {'PATHOLOGY': 'SECRET'}):
            with self.assertRaises(TypeError):
                fixture[0].run(value, fixture[1].patient_id)
        self.assertEqual(fixture[2].calls, [])

    def test_stale_diagnostic_and_critic_completions_rejected(self):
        for stage in ('DIAGNOSTIC', 'CRITIC', 'DIAGNOSTIC_REVISION'):
            for changes in ({'diagnostic_version': 99}, {'evidence_snapshot_id': 'old'},
                            {'run_id': 'other'}, {'patient_id': 'ddxplus:validate:99'},
                            {'revision_number': 99}, {'diagnostic_fingerprint': 'old'}):
                with self.subTest(stage=stage, changes=changes):
                    fixture = setup(ScenarioLLM([diagnosis(0), diagnosis(1)], [critique('revision_required'), critique()]))
                    original = fixture[0]._invoke
                    def stale(ticket, operation):
                        response = original(ticket, operation)
                        if ticket.stage == stage:
                            return StageResponse(ticket.model_copy(update=changes), response.generation)
                        return response
                    with patch.object(fixture[0], '_invoke', side_effect=stale):
                        state = run(fixture)
                    self.assertEqual(state.failure.code, 'stale_result')
                    self.assertEqual(state.status, 'failed')
                    if stage == 'CRITIC':
                        self.assertEqual(state.critiques, ())
                    if stage == 'DIAGNOSTIC':
                        self.assertEqual(state.diagnostics, ())

    def test_agents_cannot_mutate_committed_patient_evidence_or_diagnosis(self):
        fixture = setup(ScenarioLLM([diagnosis(0), diagnosis(1)], [critique('revision_required'), critique()]))
        original = fixture[0].critic.run
        def mutating_critic(patient, diagnostic, cases, medical):
            generation = original(patient, diagnostic, cases, medical)
            patient.symptoms.append('MUTATION')
            diagnostic.reasoning_summary = 'MUTATION'
            cases[0].text = 'MUTATION'
            medical[0].metadata.clear()
            return generation
        fixture[0].critic.run = mutating_critic
        state = run(fixture)
        self.assertEqual(state.status, 'final')
        self.assertNotIn('MUTATION', state.model_dump_json())

    def test_provider_failures_sanitized_and_terminal(self):
        for code in ('refusal', 'incomplete_output', 'api_failure', 'connection_failure', 'SECRET'):
            fixture = setup()
            fixture[0].diagnostic.run = Mock(return_value=Generation(raw='SECRET', attempts=1,
                failure=Failure(code=code, message='SECRET'), telemetry=[{'raw': 'SECRET',
                'failure_code': 'SECRET', 'request_bytes': 15, 'request_seconds': 0.1}]))
            state = run(fixture)
            self.assertEqual(state.outcome, 'technical_failure')
            self.assertEqual(state.failure.code, code if code != 'SECRET' else 'provider_failure')
            self.assertNotIn('SECRET', state.model_dump_json())
            self.assertEqual(state.critiques, ())

    def test_patient_provider_bounded_repairs_stop_workflow(self):
        fixture = setup(MockLLM(['{}', '{}']))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'parsing_failure')
        self.assertEqual(state.provider_repairs, 1)
        fixture[3].retrieve.assert_not_called()

    def test_transport_exception_stops_safely(self):
        fixture = setup(MockLLM([OSError('SECRET')]))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'connection_failure')
        self.assertNotIn('SECRET', state.model_dump_json())
        fixture[3].retrieve.assert_not_called()

    def test_retrieval_exception_stops_safely(self):
        fixture = setup()
        fixture[3].retrieve.side_effect = RuntimeError('SECRET')
        state = run(fixture)
        self.assertEqual(state.outcome, 'technical_failure')
        self.assertNotIn('SECRET', state.model_dump_json())
        self.assertEqual(len(fixture[2].calls), 1)

    def test_request_budget_reserves_provider_repairs_before_invocation(self):
        fixture = setup(policy=WorkflowPolicy(max_requests=3))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'request_budget_exhausted')
        self.assertEqual(state.total_requests, 2)
        self.assertEqual(len(fixture[2].calls), 2)
        self.assertEqual(state.critiques, ())
        fixture = setup(policy=WorkflowPolicy(max_requests=1))
        self.assertEqual(run(fixture).total_requests, 0)
        self.assertEqual(fixture[2].calls, [])

    def test_cumulative_byte_budget(self):
        fixture = setup(policy=WorkflowPolicy(max_request_bytes=100))
        state = run(fixture)
        self.assertEqual(state.failure.code, 'request_budget_exhausted')
        self.assertEqual(fixture[2].calls, [])

    def test_elapsed_budget_stops_after_slow_call(self):
        now = [0.0]
        fixture = setup(policy=WorkflowPolicy(max_seconds=1.0), clock=lambda: now[0])
        original = fixture[0].patient.run
        def slow(*args):
            result = original(*args)
            now[0] = 2.0
            return result
        fixture[0].patient.run = slow
        state = run(fixture)
        self.assertEqual(state.failure.code, 'time_budget_exhausted')
        self.assertEqual(state.total_requests, 1)
        fixture[3].retrieve.assert_not_called()
        self.assertEqual(state.seconds, 2.0)

    def test_frozen_evidence_cannot_be_changed(self):
        state = run(setup())
        with self.assertRaises(ValidationError):
            state.evidence.patient_cases[0].text = 'tampered'
        with self.assertRaises(ValidationError):
            state.evidence.snapshot_id = 'tampered'
        with self.assertRaises(ValidationError):
            state.current_stage = 'INITIAL'

    def test_provenance_and_content_id_checked_on_every_handoff(self):
        state = run(setup())
        bad = state.evidence.model_copy(update={'snapshot_id': 'tampered'})
        with self.assertRaises(ValueError):
            bad.agent_evidence()
        item = state.evidence.patient_cases[0].model_copy(update={'source_id': 'invented'})
        bad = state.evidence.model_copy(update={'patient_cases': (item,)})
        with self.assertRaises(ValueError):
            bad.agent_evidence()

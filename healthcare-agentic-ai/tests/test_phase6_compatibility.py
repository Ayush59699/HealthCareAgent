"""Baseline parity under CONTINUE and unchanged closed Phase 5 contracts."""
import ast
import json
from pathlib import Path
import unittest
from pydantic import ValidationError
from orchestration import Orchestrator, WorkflowPolicy, WorkflowState
from orchestration.routing import TRANSITIONS
from orchestration.contracts import StageTicket
from orchestration.phase6 import Phase6Orchestrator, Phase6Policy, Phase6WorkflowState
from tests.orchestration_helpers import ScenarioLLM, setup as setup5, run as run5
from tests.safety_helpers import setup, run, SafetyScenarioLLM, diagnosis, critique


class Phase6CompatibilityTests(unittest.TestCase):
    def test_baseline_outcomes_and_payloads_preserved_when_safety_continues(self):
        scenarios = [([diagnosis()], [critique()]), ([diagnosis(abstain=True)], [critique()]),
                     ([diagnosis()], [critique('insufficient_evidence')]),
                     ([diagnosis(0), diagnosis(1)], [critique('revision_required'), critique()]),
                     ([diagnosis(i) for i in range(3)], [critique('revision_required')]),
                     ([diagnosis()], [critique('revision_required')])]
        for diagnoses, critiques in scenarios:
            baseline = setup5(ScenarioLLM(diagnoses=diagnoses, critiques=critiques))
            new = setup(SafetyScenarioLLM(diagnoses=diagnoses, critiques=critiques))
            before, after = run5(baseline), run(new)
            self.assertEqual((before.status, before.outcome, before.revision_count), (after.status, after.outcome, after.revision_count))
            self.assertEqual(before.patient_state, after.patient_state)
            self.assertEqual(before.evidence, after.evidence)
            self.assertEqual([d.result for d in before.diagnostics], [d.result for d in after.diagnostics])
            self.assertEqual([c.result for c in before.critiques], [c.result for c in after.critiques])
            original_calls = [c for c in new[2].calls if c['text']['format']['schema']['title'] != 'SemanticSafetyResult']
            self.assertEqual(original_calls, baseline[2].calls)
            filtered_events = [e.event for e in after.transition_history if not e.event.startswith('safety_')]
            self.assertEqual(filtered_events, [e.event for e in before.transition_history])
            self.assertEqual(after.total_requests, before.total_requests + len(after.safety_assessments))

    def test_critic_block_and_early_exit_parity(self):
        for flags, missing in ((['Review'], False), ([], True)):
            base = setup5(ScenarioLLM(diagnoses=[diagnosis()], critiques=[critique(safety_flags=flags)]))
            new = setup(SafetyScenarioLLM(critiques=[critique(safety_flags=flags)]))
            if missing:
                base[4].retrieve.return_value = []
                new[4].retrieve.return_value = []
            before, after = run5(base), run(new)
            self.assertEqual((before.status, before.outcome, before.total_requests),
                             (after.status, after.outcome, after.total_requests))

    def test_phase5_schema_not_extended(self):
        self.assertEqual(TRANSITIONS['CRITIC'], {'ROUTE'})
        self.assertNotIn('SAFETY_VALIDATION', TRANSITIONS)
        self.assertNotIn('safety_assessments', WorkflowState.model_fields)
        with self.assertRaises(ValidationError):
            StageTicket(run_id='test', patient_id='ddxplus:validate:1', stage='SAFETY_VALIDATION', revision_number=0)
        old = run5(setup5())
        new = run(setup())
        with self.assertRaises(ValidationError):
            WorkflowState.model_validate_json(new.model_dump_json())
        with self.assertRaises(ValidationError):
            Phase6WorkflowState.model_validate_json(old.model_dump_json())
        self.assertEqual(WorkflowState.model_validate_json(old.model_dump_json()), old)
        self.assertNotIsInstance(new, WorkflowState)
        with self.assertRaises(ValidationError):
            WorkflowState.model_validate(new)

    def test_phase5_exports_and_defaults_unchanged(self):
        self.assertEqual(Orchestrator.__module__, 'orchestration.orchestrator')
        self.assertIsNot(Orchestrator, Phase6Orchestrator)
        baseline, new = WorkflowPolicy().model_dump(), Phase6Policy().model_dump()
        self.assertEqual({k: new[k] for k in baseline}, baseline)
        self.assertEqual(baseline['max_revisions'], 2)

    def test_phase6_runner_has_no_evaluation_or_phase4_runner_import(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_phase6.py'
        tree = ast.parse(path.read_text())
        modules = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any('evaluat' in name or 'run_phase4' in name for name in modules))

    def test_final_state_requires_matching_assessment(self):
        state = run(setup())
        for changes in ({'safety_coverage': 'not_assessed', 'safety_skip_reason': 'not_reached'},
                        {'safety_assessments': []}):
            data = json.loads(state.model_dump_json())
            data.update(changes)
            with self.assertRaises(ValidationError):
                Phase6WorkflowState.model_validate_json(json.dumps(data))

"""Demo UI tests and integration with the real Phase 6 loop (offline provider)."""
import io
import unittest
from unittest.mock import Mock, patch

from demo.terminal import (Console, RetrievalProgress, arguments, choose_sample,
                           discover_samples, main, make_workflow, safe_text,
                           show_result)
from orchestration.phase6 import Phase6Orchestrator, Phase6Policy
from tests.safety_helpers import (SafetyScenarioLLM, diagnosis, critique, semantic,
                                  setup, run)


class DemoTests(unittest.TestCase):
    def console(self, detailed=False):
        output = io.StringIO()
        return Console(output, detailed=detailed), output

    def records(self):
        from dataclasses import replace
        return [replace(setup()[1], labels=None)]

    def test_discovery_explicitly_excludes_labels(self):
        parser = Mock()
        parser.iter_patients.return_value = self.records()
        self.assertEqual(discover_samples(parser), parser.iter_patients.return_value)
        parser.iter_patients.assert_called_once_with('validate', limit=3, include_labels=False)

    def test_discovery_empty_fails(self):
        parser = Mock()
        parser.iter_patients.return_value = []
        with self.assertRaises(ValueError):
            discover_samples(parser)

    def test_discovery_rejects_wrong_split(self):
        from dataclasses import replace
        parser = Mock()
        parser.iter_patients.return_value = [replace(self.records()[0], split='train')]
        with self.assertRaises(ValueError):
            discover_samples(parser)

    def test_default_selection(self):
        console, _ = self.console()
        records = self.records()
        self.assertIs(choose_sample(records, console, Mock(side_effect=['', ''])), records[0])

    def test_invalid_input_and_detail_toggle(self):
        console, output = self.console()
        record = choose_sample(self.records(), console, Mock(side_effect=['bad', 'd', '1', '999', '1']))
        self.assertIsNotNone(record)
        self.assertTrue(console.detailed)
        self.assertIn('Invalid sample', output.getvalue())

    def test_manual_input_is_not_forged(self):
        console, output = self.console()
        self.assertIsNone(choose_sample(self.records(), console, Mock(side_effect=['2', 'q'])))
        self.assertIn('No input was submitted', output.getvalue())

    def test_back_and_quit(self):
        console, _ = self.console()
        self.assertIsNone(choose_sample(self.records(), console, Mock(side_effect=['1', 'b', 'q'])))

    def test_control_characters_and_bounded_text(self):
        text = safe_text('\x1b[2J\x00\u202e' + 'a' * 1000, 40)
        self.assertNotIn('\x1b', text)
        self.assertNotIn('\u202e', text)
        self.assertIn('[excerpt]', text)
        self.assertLess(len(text), 70)

    def test_retrieval_is_transparent(self):
        delegate = Mock()
        delegate.retrieve.return_value = [{'payload': 'untouched'}]
        console, output = self.console()
        adapter = RetrievalProgress(delegate, console, 'test')
        result = adapter.retrieve('query', top_k=3)
        self.assertIs(result, delegate.retrieve.return_value)
        delegate.retrieve.assert_called_once_with('query', top_k=3)
        self.assertIn('pending snapshot validation', output.getvalue())
        self.assertNotIn('untouched', output.getvalue())

    def test_real_phase6_loop_and_exact_provider_request_parity(self):
        baseline = setup()
        expected = run(baseline)
        _, record, llm, patients, medical = setup()
        console, output = self.console(detailed=True)
        workflow = make_workflow(llm, patients, medical, console, top_k=1, policy=Phase6Policy())
        self.assertIsInstance(workflow, Phase6Orchestrator)
        self.assertIs(type(workflow).run, Phase6Orchestrator.run)
        state = workflow.run(record.patient, record.patient_id)
        self.assertEqual(state.status, expected.status)
        self.assertEqual(state.diagnostics[0].result, expected.diagnostics[0].result)
        self.assertEqual(state.safety_assessments[-1].decision, 'CONTINUE')
        self.assertEqual(llm.calls, baseline[2].calls)
        self.assertEqual(state.total_requests, expected.total_requests)
        patients.retrieve.assert_called_once()
        medical.retrieve.assert_called_once()
        show_result(console, state, 1)
        text = output.getvalue()
        for name in ('[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', 'Grounding result: PASS',
                     'phase6-routing-v1', 'FINAL SYSTEM RESULT', 'CONTINUE'):
            self.assertIn(name, text)
        self.assertNotIn('Research fixture version 0', text)  # reasoning_summary omitted
        self.assertNotIn('PATHOLOGY', text)

    def test_block_skips_semantic_and_is_not_presented_as_pass(self):
        review = critique()
        review.safety_flags = ['Synthetic test block']
        fixture = setup(SafetyScenarioLLM(critiques=[review]))
        state = run(fixture)
        console, output = self.console()
        show_result(console, state, 1)
        self.assertIn('Safety decision: BLOCK', output.getvalue())
        self.assertIn('skipped_critic_block', output.getvalue())
        self.assertIn('OUTPUT WITHHELD', output.getvalue())
        self.assertEqual(fixture[2].safety_count, 0)

    def test_human_review_uses_actual_category(self):
        state = run(setup(SafetyScenarioLLM(safety_results=[semantic('unsupported_certainty')])))
        console, output = self.console()
        show_result(console, state, 1)
        self.assertIn('Safety decision: HUMAN_REVIEW', output.getvalue())
        self.assertIn('unsupported_certainty: uncertain', output.getvalue())

    def test_no_medical_evidence_never_claims_safety(self):
        fixture = setup()
        fixture[4].retrieve.return_value = []
        state = run(fixture)
        console, output = self.console()
        show_result(console, state, 1)
        self.assertIn('NOT ASSESSED / UNAVAILABLE', output.getvalue())
        self.assertIn('no_medical_evidence', output.getvalue())
        self.assertNotIn('Grounding result: PASS', output.getvalue())

    def test_budget_failure_has_no_fabricated_output(self):
        state = run(setup(policy=Phase6Policy(max_requests=1)))
        console, output = self.console()
        show_result(console, state, 1)
        self.assertIn('request_budget_exhausted', output.getvalue())
        self.assertIn('No accepted diagnosis', output.getvalue())

    def test_stale_safety_not_used_for_failed_revision(self):
        review = critique()
        review.overall_assessment = 'revision_required'
        review.recommended_revisions = ['Synthetic revision']
        state = run(setup(SafetyScenarioLLM(critiques=[review]), policy=Phase6Policy(max_requests=5)))
        console, output = self.console()
        show_result(console, state, 1)
        self.assertEqual(state.status, 'failed')
        self.assertIn('Safety decision: NOT ASSESSED / UNAVAILABLE', output.getvalue())

    def test_validate_only_never_constructs_provider(self):
        # Real EvidenceService/Phase 6 provenance contracts; local services mocked.
        _, record, _, patients, medical = setup()
        patients.__enter__ = Mock(return_value=patients)
        patients.__exit__ = Mock(return_value=False)
        medical.__enter__ = Mock(return_value=medical)
        medical.__exit__ = Mock(return_value=False)
        patients.vector_store.count.return_value = 1
        medical.store.count.return_value = 1
        with patch('demo.terminal.discover_samples', return_value=[record]), \
             patch('rag.patient_parser.DDXPlusParser'), \
             patch('demo.terminal.Path.is_dir', return_value=True), \
             patch('demo.terminal.Path.is_file', return_value=True), \
             patch('rag.config.load_generation_env'), \
             patch('rag.patient_rag.PatientCaseRAG', return_value=patients), \
             patch('rag.medical_retriever.MedicalKnowledgeRetriever', return_value=medical), \
             patch('rag.llm.provider.OpenAIProvider') as provider, \
             patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['--validate-only']), 0)
            self.assertIn('LLM requests: 0', output.getvalue())
            provider.assert_not_called()

    def test_setup_failure_does_not_leak_exception(self):
        with patch('rag.patient_parser.DDXPlusParser', side_effect=ValueError('SECRET_TOKEN')), \
             patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['--list-samples']), 1)
            self.assertNotIn('SECRET_TOKEN', output.getvalue())
            self.assertIn('dataset discovery', output.getvalue())

    def test_missing_index_does_not_create_directory(self):
        with patch('demo.terminal.discover_samples', return_value=self.records()), \
             patch('rag.patient_parser.DDXPlusParser'), \
             patch('rag.config.load_generation_env'), \
             patch('demo.terminal.Path.is_dir', return_value=False), \
             patch('rag.patient_rag.PatientCaseRAG') as rag, \
             patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(main(['--validate-only']), 1)
            rag.assert_not_called()

    def test_help(self):
        with patch('sys.stdout', new_callable=io.StringIO) as output, self.assertRaises(SystemExit) as exit:
            arguments(['--help'])
        self.assertEqual(exit.exception.code, 0)
        self.assertIn('--validate-only', output.getvalue())

    def test_invalid_budgets(self):
        for args in (['--top-k', '0'], ['--max-requests', '-1'], ['--max-seconds', 'nan'], ['--max-seconds', 'inf']):
            with self.subTest(args=args), patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as exit:
                arguments(args)
            self.assertEqual(exit.exception.code, 2)


if __name__ == '__main__':
    unittest.main()

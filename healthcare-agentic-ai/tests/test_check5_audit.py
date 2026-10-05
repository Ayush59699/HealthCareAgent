"""CHECK5 audit/budget regression checks using the default workflow fixture."""
import copy
import unittest
from scripts.verify_who_amg_run import verify_selector_handoff
from tests import test_check5_pipeline as fixtures


class Check5AuditTests(unittest.TestCase):
    setUp = fixtures.Check5PipelineTests.setUp
    fixture = fixtures.Check5PipelineTests.fixture
    run_fixture = fixtures.Check5PipelineTests.run_fixture

    def test_replays_source_sections_and_accounting(self):
        state, _ = self.run_fixture()
        verify_selector_handoff(state, self.root)

    def test_replay_detects_changed_catalog_and_selection(self):
        state, _ = self.run_fixture()
        for field, value in (('catalog_sha256', 'a' * 64), ('selected_ids', []),
                             ('selection_count', 3), ('patient_payload_sha256', 'b' * 64)):
            altered = copy.deepcopy(state.medical_retrieval)
            altered['who'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_selector_handoff(state.model_copy(update={'medical_retrieval': altered}), self.root)

    def test_replay_detects_missing_sections_and_false_accounting(self):
        state, _ = self.run_fixture()
        for update in ({'total_requests': state.total_requests - 1},
                       {'total_request_bytes': state.total_request_bytes + 1},
                       {'evidence': state.evidence.model_copy(update={'medical_knowledge': ()})}):
            with self.assertRaises(ValueError):
                verify_selector_handoff(state.model_copy(update=update), self.root)

    def test_exact_check4_provider_limits_with_shared_client(self):
        workflow, llm, _, _ = self.fixture()
        self.assertEqual(workflow.evidence.provider.config.max_retries, 0)
        self.assertEqual(workflow.evidence.provider.config.max_output_tokens, 1800)
        self.assertIs(workflow.evidence.provider.client, llm.client)
        self.assertEqual(llm.config.max_output_tokens, 8192)

    def test_failed_quote_audit_does_not_enter_evidence(self):
        bad = self.review.model_dump()
        bad['content_quotes'] = ['Invented source statement.']
        state, (_, llm, _, medical) = self.run_fixture(review=bad)
        errors = state.medical_retrieval['who']['validation_errors']
        self.assertEqual(errors[0]['code'], 'nonliteral_source_quote')
        self.assertEqual(len(errors), 1)  # No CHECK4 repair calls.
        self.assertIsNone(state.evidence)
        self.assertFalse(state.diagnostics)
        medical.retrieve.assert_not_called()


if __name__ == '__main__':
    unittest.main()

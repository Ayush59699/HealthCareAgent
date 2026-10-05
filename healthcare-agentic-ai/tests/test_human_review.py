"""Offline local-review tests. All approval actions here use synthetic fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from application.decision import build_final_decision
from application.human_review import (HumanReviewRecord, offer_human_review,
                                      review_context, verify_review_session)
from tests import test_check5_pipeline as fixtures
from tests import test_phase6_reporting as reporting
from tests.safety_helpers import critique, SafetyScenarioLLM


class HumanReviewTests(unittest.TestCase):
    fixture = fixtures.Check5PipelineTests.fixture
    run_fixture = fixtures.Check5PipelineTests.run_fixture

    def setUp(self):
        fixtures.Check5PipelineTests.setUp(self)
        self.state, self.parts = self.run_fixture(critiques=[critique(safety_flags=['Synthetic safety concern'])])
        self.original = self.state.model_dump_json()
        self.output = self.root / 'reviews'
        self.prompts, self.display = [], []

    def gate(self, answers=(), **kwargs):
        entries = iter(answers)
        def ask(prompt):
            self.prompts.append(prompt)
            answer = next(entries, '')
            if isinstance(answer, BaseException):
                raise answer
            return answer
        options = dict(ask=ask, emit=self.display.append, interactive=True, reviewer='test-human')
        options.update(kwargs)
        result = offer_human_review(self.state, self.output, **options)
        self.assertEqual(self.state.model_dump_json(), self.original)
        if result:
            folder = self.output / ('review-' + result.session_id)
            self.assertEqual(verify_review_session(folder, self.state), result)
        return result

    def test_block_offers_real_review_question(self):
        record = self.gate(['DECLINE'])
        self.assertIn('Do you want to review this case?', self.prompts[0])
        self.assertEqual(record.ai_status, 'BLOCK')
        self.assertEqual(record.safety_decision, 'BLOCK')

    def test_decline_stays_blocked_pending(self):
        record = self.gate(['DECLINE'])
        self.assertEqual(record.event, 'DECLINE')
        self.assertEqual(record.review_state, 'pending')
        self.assertEqual(record.case_disposition, 'BLOCKED')
        self.assertFalse(record.human_approval)
        self.assertIsNone(record.human_decision)
        self.assertEqual(len(self.prompts), 1)

    def test_review_alone_is_not_approval(self):
        record = self.gate(['REVIEW', ''])
        self.assertTrue(record.review_requested)
        self.assertTrue(record.complete_information_shown)
        self.assertEqual(record.event, 'NO_RESPONSE')
        self.assertFalse(record.human_approval)
        self.assertEqual(record.review_state, 'pending')

    def test_approve_requires_explicit_second_action(self):
        record = self.gate(['REVIEW', 'APPROVE'])
        self.assertEqual(record.human_decision, 'APPROVE')
        self.assertTrue(record.human_approval)
        self.assertEqual(record.review_state, 'approved')
        self.assertEqual(record.ai_status, 'BLOCK')
        self.assertEqual(record.safety_decision, 'BLOCK')
        self.assertEqual(record.case_disposition, 'BLOCKED')
        self.assertFalse(record.proposal_released)
        self.assertFalse(record.safety_overridden)
        # Original machine-only handoff remains unchanged/pending/withheld.
        final = build_final_decision(self.state)
        self.assertEqual(final.status, 'BLOCK')
        self.assertEqual(final.review_state, 'pending')
        self.assertFalse(final.human_approval)
        self.assertIsNone(final.diagnostic_proposal)

    def test_approve_at_offer_cannot_skip_review(self):
        record = self.gate(['APPROVE', ''])
        self.assertFalse(record.human_approval)
        self.assertFalse(record.review_requested)
        self.assertEqual(record.event, 'NO_RESPONSE')

    def test_reject_is_blocked_and_recorded(self):
        record = self.gate(['REVIEW', 'REJECT'])
        self.assertEqual(record.human_decision, 'REJECT')
        self.assertEqual(record.case_disposition, 'BLOCKED')
        self.assertEqual(record.review_state, 'rejected')
        self.assertFalse(record.human_approval)

    def test_more_evidence_is_unresolved_pending_no_rerun(self):
        count = len(self.parts[1].calls)
        with patch('rag.llm.provider.OpenAIProvider.generate', side_effect=AssertionError('No cloud')), \
             patch('rag.amg.service.AMGEvidenceService.retrieve', side_effect=AssertionError('No retrieval')):
            record = self.gate(['REVIEW', 'REQUEST_MORE_EVIDENCE'])
        self.assertEqual(record.case_disposition, 'unresolved')
        self.assertEqual(record.review_state, 'pending')
        self.assertFalse(record.retrieval_triggered)
        self.assertFalse(record.human_approval)
        self.assertEqual(len(self.parts[1].calls), count)

    def test_blank_none_eof_and_interrupt_never_approve(self):
        for answers in ([''], [None], [EOFError()], [KeyboardInterrupt()],
                        ['REVIEW', ''], ['REVIEW', None], ['REVIEW', EOFError()],
                        ['REVIEW', KeyboardInterrupt()]):
            with self.subTest(answers=answers):
                record = self.gate(answers)
                self.assertFalse(record.human_approval)
                self.assertEqual(record.review_state, 'pending')
                self.assertEqual(record.case_disposition, 'BLOCKED')

    def test_invalid_decisions_do_not_default_to_approval(self):
        record = self.gate(['REVIEW', 'yes', 'continue', 'APPROVE please', ''])
        self.assertFalse(record.human_approval)
        self.assertEqual(record.event, 'NO_RESPONSE')

    def test_invalid_input_is_bounded_and_pending(self):
        record = self.gate(['bad'] * 20)
        self.assertEqual(len(self.prompts), 10)
        self.assertEqual(record.event, 'NO_RESPONSE')
        self.assertFalse(record.human_approval)

    def test_noninteractive_cannot_read_piped_approval(self):
        reader = Mock(return_value='APPROVE')
        with patch('sys.stdin.isatty', return_value=False), patch('sys.stdout.isatty', return_value=True):
            record = self.gate(ask=reader, interactive=None)
        reader.assert_not_called()
        self.assertEqual(record.event, 'NON_INTERACTIVE')
        self.assertFalse(record.human_approval)

    def test_hidden_output_not_interactive_approval(self):
        reader = Mock(return_value='APPROVE')
        with patch('sys.stdin.isatty', return_value=True), patch('sys.stdout.isatty', return_value=False):
            record = self.gate(ask=reader, interactive=None)
        reader.assert_not_called()
        self.assertEqual(record.event, 'NON_INTERACTIVE')

    def test_record_identity_version_timestamp_and_hashes(self):
        record = self.gate(['REVIEW', 'APPROVE'])
        final = build_final_decision(self.state)
        self.assertEqual(record.run_id, self.state.run_id)
        self.assertEqual(record.patient_id, self.state.patient_id)
        self.assertEqual(record.diagnostic_version, self.state.diagnostics[-1].version)
        self.assertEqual(record.diagnostic_fingerprint, final.diagnostic_fingerprint)
        self.assertEqual(record.evidence_snapshot_id, self.state.evidence.snapshot_id)
        self.assertEqual(record.safety_input_fingerprint, final.safety_result.ticket.input_fingerprint)
        self.assertEqual(record.reviewer, 'test-human')
        self.assertTrue(record.timestamp.endswith('+00:00'))
        self.assertFalse(record.reviewer_identity_verified)

    def test_full_context_includes_uncited_evidence_and_citations(self):
        record = self.gate(['REVIEW', 'REJECT'])
        rendered = [json.loads(text) for text in self.display if text.startswith('{')]
        self.assertEqual(len(rendered), 2)
        self.assertEqual(rendered[0], rendered[1])
        ctx = rendered[1]
        self.assertEqual(ctx['final_ai_proposal_and_reasoning'], self.state.diagnostics[-1].result.model_dump(mode='json'))
        self.assertEqual(ctx['critic_findings'], self.state.critiques[-1].result.model_dump(mode='json'))
        self.assertEqual(ctx['safety_findings'], [f.model_dump(mode='json') for f in self.state.safety_assessments[-1].findings])
        actual = ctx['evidence_passages_and_citations']['medical_knowledge_including_uncited_passages']
        self.assertEqual(actual, [e.thaw().model_dump(mode='json') for e in self.state.evidence.medical_knowledge])
        self.assertEqual(len(actual), 2)  # WHO cited + uncited AMG fixture both shown.
        self.assertIn('missing_evidence_warnings', ctx)
        self.assertIn('exact_patient_citation_inventory', ctx)

    def test_preserves_entire_original_workflow_and_safety(self):
        before = hashlib.sha256(self.original.encode()).hexdigest()
        record = self.gate(['REVIEW', 'APPROVE'])
        folder = self.output / ('review-' + record.session_id)
        copy_state = json.loads((folder / 'workflow_snapshot.json').read_text())
        self.assertEqual(copy_state, self.state.model_dump(mode='json'))
        self.assertEqual(hashlib.sha256(self.state.model_dump_json().encode()).hexdigest(), before)

    def test_repeated_reviews_preserve_all_earlier_files(self):
        first = self.gate(['DECLINE'])
        folder = self.output / ('review-' + first.session_id)
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        second = self.gate(['REVIEW', 'REJECT'])
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_live_state_mutation_invalidates_decision(self):
        answers = iter(['REVIEW', 'APPROVE'])
        def ask(_):
            answer = next(answers)
            if answer == 'APPROVE':
                self.state.diagnostics[-1].result.uncertainty.append('Changed after display')
            return answer
        record = offer_human_review(self.state, self.output, ask=ask, emit=lambda _: None,
                                    interactive=True, reviewer='test-human')
        self.assertEqual(record.event, 'INVALIDATED')
        self.assertFalse(record.human_approval)
        verify_review_session(self.output / ('review-' + record.session_id))

    def test_external_saved_file_guard_prevents_stale_approval(self):
        guards = iter([True, True, True, False])
        record = self.gate(['REVIEW', 'APPROVE'], integrity_check=lambda: next(guards))
        self.assertEqual(record.event, 'INVALIDATED')
        self.assertFalse(record.human_approval)

    def test_interrupted_complete_display_does_not_enable_approval(self):
        displays = 0
        def emit(text):
            nonlocal displays
            if text.startswith('{'):
                displays += 1
                if displays == 2:
                    raise OSError('Display disconnected')
        record = self.gate(['REVIEW', 'APPROVE'], emit=emit)
        self.assertEqual(record.event, 'INTERRUPTED')
        self.assertFalse(record.complete_information_shown)
        self.assertFalse(record.human_approval)
        self.assertEqual(len(self.prompts), 1)

    def test_no_proposal_or_safety_means_no_approval(self):
        # Existing no-evidence abstention has no Diagnostic/Safety proposal to approve.
        state, _ = self.run_fixture(ids=[], empty_amg=True)
        answers = iter(['REVIEW', 'APPROVE', 'REQUEST_MORE_EVIDENCE'])
        record = offer_human_review(state, self.output, ask=lambda _: next(answers),
                                    emit=lambda _: None, interactive=True, reviewer='test-human')
        self.assertIsNone(record.diagnostic_version)
        self.assertFalse(record.approval_eligible)
        self.assertEqual(record.human_decision, 'REQUEST_MORE_EVIDENCE')
        self.assertFalse(record.human_approval)

    def test_allow_does_not_offer_or_record_human_approval(self):
        state, _ = self.run_fixture()
        reader = Mock()
        result = offer_human_review(state, self.output, ask=reader, emit=Mock(), interactive=True)
        self.assertIsNone(result)
        reader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_pending_record_exists_before_first_input(self):
        def ask(_):
            session = next(self.output.iterdir())
            record = verify_review_session(session, self.state)
            self.assertEqual(record.event, 'OFFERED')
            self.assertFalse(record.human_approval)
            raise EOFError()
        self.gate(ask=ask)

    def test_context_tampering_fails_verification(self):
        record = self.gate(['REVIEW', 'APPROVE'])
        folder = self.output / ('review-' + record.session_id)
        path = folder / 'review_context.json'
        data = json.loads(path.read_text())
        data['current_diagnostic_version'] += 1
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            verify_review_session(folder)

    def test_event_hash_or_identity_tampering_fails(self):
        for change in ({'previous_record_sha256': 'bad'}, {'run_id': 'other'}, {'diagnostic_version': 99}):
            record = self.gate(['REVIEW', 'APPROVE'])
            folder = self.output / ('review-' + record.session_id)
            path = folder / 'event-0003.json'
            data = json.loads(path.read_text())
            data.update(change)
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                verify_review_session(folder)

    def test_forged_approval_without_review_rejected(self):
        record = self.gate(['DECLINE'])
        data = record.model_dump()
        data.update(event='APPROVE', human_decision='APPROVE', human_approval=True, review_state='approved')
        with self.assertRaises(ValueError):
            HumanReviewRecord.model_validate(data)

    def test_existing_session_cannot_be_overwritten(self):
        with patch('application.human_review.uuid4', return_value='fixture-session'):
            self.gate(['DECLINE'])
            with self.assertRaises(FileExistsError):
                self.gate(['REVIEW', 'APPROVE'])

    def test_saved_cli_noninteractive_preserves_input_files(self):
        from scripts.review_case import main
        path = self.root / 'case_state.json'
        decision = self.root / 'case_decision.json'
        path.write_text(self.original)
        decision.write_text(build_final_decision(self.state).model_dump_json())
        before = (path.read_bytes(), decision.read_bytes())
        with patch('sys.stdin.isatty', return_value=False), patch('builtins.input') as reader, \
             patch('builtins.print'), patch('rag.llm.provider.OpenAIProvider.generate', side_effect=AssertionError('cloud')):
            code = main([str(path), '--review-dir', str(self.output)])
        self.assertEqual(code, 2)
        reader.assert_not_called()
        self.assertEqual(before, (path.read_bytes(), decision.read_bytes()))
        self.assertEqual(verify_review_session(next(self.output.iterdir())).event, 'NON_INTERACTIVE')

    def test_saved_cli_explicit_synthetic_approval(self):
        from scripts.review_case import main
        path = self.root / 'case_state.json'
        path.write_text(self.original)
        with patch('sys.stdin.isatty', return_value=True), patch('sys.stdout.isatty', return_value=True), \
             patch('builtins.input', side_effect=['REVIEW', 'APPROVE']), patch('builtins.print'), \
             patch('rag.llm.provider.OpenAIProvider.generate', side_effect=AssertionError('cloud')):
            code = main([str(path), '--review-dir', str(self.output)])
        self.assertEqual(code, 0)
        record = verify_review_session(next(self.output.iterdir()), self.state)
        self.assertEqual(record.human_decision, 'APPROVE')
        self.assertEqual(record.ai_status, 'BLOCK')

    def test_terminal_control_sequences_escaped_in_context(self):
        # Rendering is JSON, not interpreted terminal data. No raw ANSI is emitted.
        final, context = review_context(self.state)
        context['notice'] += '\x1b[2J'
        with patch('application.human_review.review_context', return_value=(final, context)):
            reader = Mock(return_value='DECLINE')
            offer_human_review(self.state, self.output, ask=reader, emit=self.display.append, interactive=True)
        self.assertNotIn('\x1b', ''.join(self.display))
        self.assertIn('\\u001b', ''.join(self.display))


    def test_safety_human_review_also_offered_without_override(self):
        from tests.safety_helpers import semantic
        state, _ = self.run_fixture(safety_results=[semantic('safety_ambiguity')])
        answers = iter(['REVIEW', 'APPROVE'])
        record = offer_human_review(state, self.output, ask=lambda _: next(answers),
                                    emit=lambda _: None, interactive=True, reviewer='test-human')
        self.assertEqual(record.ai_status, 'HUMAN_REVIEW')
        self.assertEqual(record.safety_decision, 'HUMAN_REVIEW')
        self.assertTrue(record.human_approval)
        self.assertEqual(build_final_decision(state).status, 'HUMAN_REVIEW')
        verify_review_session(self.output / ('review-' + record.session_id), state)

    def test_storage_failure_before_offer_never_asks_for_approval(self):
        reader = Mock(return_value='APPROVE')
        with patch('application.human_review._persist_new', side_effect=OSError('Disk failure')):
            with self.assertRaises(OSError):
                offer_human_review(self.state, self.output, ask=reader, emit=Mock(), interactive=True)
        reader.assert_not_called()
        self.assertEqual(self.state.model_dump_json(), self.original)

    def test_saved_mismatching_ai_decision_rejected_before_input(self):
        from scripts.review_case import main
        path = self.root / 'case_state.json'
        path.write_text(self.original)
        data = build_final_decision(self.state).model_dump(mode='json')
        data['run_id'] = 'other-run'
        (self.root / 'case_decision.json').write_text(json.dumps(data))
        with patch('builtins.input') as reader, patch('builtins.print'):
            self.assertEqual(main([str(path), '--review-dir', str(self.output)]), 1)
        reader.assert_not_called()
        self.assertFalse(self.output.exists())


class HumanReviewEntryPointTests(unittest.TestCase):
    manager = reporting.Phase6ReportingTests.manager
    invoke = reporting.Phase6ReportingTests.invoke

    def test_batch_offers_only_after_safety_and_saved_ai_artifacts(self):
        import tempfile
        from scripts import run_phase6
        observed = []
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'batch'
            def gate(state, directory):
                self.assertEqual(state.safety_coverage, 'assessed')
                self.assertTrue((output / 'final_decision_001.json').is_file())
                observed.append(state.status)
                answers = iter(['REVIEW', 'APPROVE'])
                return offer_human_review(state, directory, ask=lambda _: next(answers),
                                          emit=lambda _: None, interactive=True, reviewer='test-human')
            with patch.object(run_phase6, 'offer_human_review', side_effect=gate):
                code = self.invoke(output, SafetyScenarioLLM(critiques=[critique(safety_flags=['Fixture flag'])]))
            self.assertEqual(code, 0)
            self.assertEqual(observed, ['blocked'])
            report = json.loads((output / 'phase6_run_report.json').read_text())['cases'][0]
            ai = json.loads((output / 'final_decision_001.json').read_text())
            self.assertEqual(report['human_review']['human_decision'], 'APPROVE')
            self.assertEqual(ai['status'], 'BLOCK')
            self.assertFalse(ai['human_approval'])
            self.assertIsNone(ai['diagnostic_proposal'])


if __name__ == '__main__':
    unittest.main()

"""Default WHO+AMG integration contracts; synthetic fixtures, no live API."""
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from application.workflow import create_workflow
from application.decision import build_final_decision
from orchestration.phase6 import Phase6Policy
from rag.agents.grounding import evidence_from_hits, state_from_patient
from rag.config import OpenAIConfig
from rag.manual_patient import manual_patient_id
from rag.who_catalog_selector import WHOSelection
from rag.who_selector_relevance import ContentRelevance
from rag.who_sections import section_hits, section_spans, MAX_DOCUMENT_EXCERPT_CHARS
from tests import test_who_catalog_selector as catalog_fixtures
from tests.amg_helpers import configure_medical, configure_provider, amg_hit
from tests.safety_helpers import SafetyScenarioLLM, setup, semantic, critique
from tests.phase4_helpers import envelope


class SectionLLM(SafetyScenarioLLM):
    def __init__(self, ids, review, **kwargs):
        self.ids, self.review = ids, review
        super().__init__(config=OpenAIConfig(max_retries=0), **kwargs)

    def respond(self, **request):
        title = request['text']['format']['schema']['title']
        if title in ('WHOSelection', 'ContentRelevance'):
            self.calls.append(copy.deepcopy(request))
            value = {'selected_ids': self.ids} if title == 'WHOSelection' else self.review
            return envelope(json.dumps(value))
        return super().respond(**request)


class Check5PipelineTests(unittest.TestCase):
    def setUp(self):
        catalog_fixtures.WHOCatalogSelectorTests.setUp(self)
        row = self.selector.catalog_payload()['catalog'][0]
        folder = 'who_fact_sheets' if row[2] == 'fact_sheet' else 'who_questions_answers'
        path = self.root / folder / row[3]
        header = path.read_text().split('\n', 4)[:4]
        self.source_text = '\n'.join(header) + '\n\nIntroductory section\n====================\n' + ('Irrelevant fixture introduction.\n' * 180) + '\nRelevant section\n================\n\nExact supporting fixture statement with conditional context.\n\nOther section\n=============\nUnrelated fixture ending.\n'
        path.write_bytes(self.source_text.encode('utf8'))
        self.review = ContentRelevance(verdict='relevant', reason='direct_presenting_problem',
            positive_fact_quotes=['Fatigue and dizziness'],
            content_quotes=['Exact supporting fixture statement with conditional context.'])
        self.document = next(self.selector.iter_selected(WHOSelection(selected_ids=self.ids[:1])))
        self.hits = section_hits(self.document, self.review, self.patient)
        self.patient_id = manual_patient_id(self.patient)

    def fixture(self, *, ids=None, review=None, empty_amg=False, policy=None, **llm_kwargs):
        llm = SectionLLM(self.ids[:1] if ids is None else ids,
                         self.review.model_dump() if review is None else review, **llm_kwargs)
        _, _, _, patients, medical = setup(llm)
        configure_medical(medical)
        configure_provider(llm)
        # This test's provider explicitly exercises real WHO schema parsing rather
        # than the valid-empty selector used by unchanged AMG/safety scenarios.
        llm.client.responses.create.side_effect = llm.respond
        ref = 'medical:' + self.hits[0]['chunk_id']
        llm.diagnoses[0].primary_hypothesis.rationale.evidence_refs = [ref]
        llm.diagnoses[0].medical_knowledge_evidence = [ref]
        if empty_amg:
            medical.retrieve.return_value = []
        return create_workflow(llm, patients, medical, who_selector=self.selector, policy=policy), llm, patients, medical

    def run_fixture(self, **kwargs):
        parts = self.fixture(**kwargs)
        return parts[0].run(self.patient, self.patient_id), parts

    def test_default_uses_unchanged_check4_and_never_lexical(self):
        with patch('rag.who_lookup.WHOLookup.search', side_effect=AssertionError('lexical')), \
             patch('rag.who_lookup.WHOLookup.lookup', side_effect=AssertionError('lexical')):
            state, (workflow, llm, patients, medical) = self.run_fixture()
        self.assertEqual(state.status, 'final')
        self.assertEqual(state.medical_retrieval['backend'], 'who-check4-amg-v1')
        self.assertEqual(state.medical_retrieval['combined_status'], 'BOTH_AVAILABLE')
        schemas = [c['text']['format']['schema']['title'] for c in llm.calls]
        self.assertEqual(schemas[:3], ['PatientState', 'WHOSelection', 'ContentRelevance'])
        from rag.who_catalog_selector import PROMPT
        from rag.who_selector_relevance import REVIEW_PROMPT
        self.assertEqual(llm.calls[1]['instructions'], PROMPT)
        self.assertEqual(llm.calls[2]['instructions'], REVIEW_PROMPT)
        self.assertEqual(state.total_requests, len(llm.calls))
        self.assertEqual(len([i for i in state.invocations if i.ticket.stage == 'EVIDENCE']), 2)
        patients.retrieve.assert_called_once_with(self.patient.to_text(), top_k=1)
        medical.retrieve.assert_called_once_with('Synthetic clinical observations', top_k=5)
        self.assertEqual(state.evidence.medical_knowledge[-1].thaw(), evidence_from_hits([amg_hit()], 'medical_knowledge')[0])

    def test_relevant_section_after_4000_reaches_diagnostic_critic_safety(self):
        state, (_, llm, _, _) = self.run_fixture()
        hit = self.hits[0]
        self.assertGreater(hit['char_start'], 4000)
        self.assertIn('Relevant section', hit['text'])
        self.assertNotIn('Irrelevant fixture introduction', hit['text'])
        self.assertEqual(hit['text'], self.source_text[hit['char_start']:hit['char_end']])
        for call in llm.calls:
            name = call['text']['format']['schema']['title']
            if name in ('DiagnosticResult', 'ClinicalCritique', 'SemanticSafetyResult'):
                payload = json.loads(call['input'][0]['content'])
                self.assertIn(hit['text'], call['input'][0]['content'].replace('\\n', '\n'))
                self.assertIn(hit['document_id'], call['input'][0]['content'])
                if name != 'SemanticSafetyResult':
                    groups = payload['MEDICAL_KNOWLEDGE_EVIDENCE']
                    self.assertEqual(set(groups), {'WHO', 'AMG'})
                    self.assertEqual(len(groups['AMG']['accepted_passages']), 1)
        self.assertEqual(build_final_decision(state).status, 'ALLOW')

    def test_no_who_selection_preserves_amg_only(self):
        workflow, llm, _, _ = self.fixture(ids=[])
        ref = 'medical:' + amg_hit()['chunk_id']
        llm.diagnoses[0].primary_hypothesis.rationale.evidence_refs = [ref]
        llm.diagnoses[0].medical_knowledge_evidence = [ref]
        state = workflow.run(self.patient, self.patient_id)
        self.assertEqual(state.status, 'final')
        self.assertEqual(state.medical_retrieval['combined_status'], 'AMG_ONLY')
        self.assertEqual(len(state.evidence.medical_knowledge), 1)
        self.assertEqual([c['text']['format']['schema']['title'] for c in llm.calls].count('ContentRelevance'), 0)

    def test_who_only_when_amg_has_no_accepted_passages(self):
        state, _ = self.run_fixture(empty_amg=True)
        self.assertEqual(state.status, 'final')
        self.assertEqual(state.medical_retrieval['combined_status'], 'WHO_ONLY')

    def test_both_empty_preserves_abstention_and_pending_review(self):
        state, (_, llm, _, _) = self.run_fixture(ids=[], empty_amg=True)
        self.assertEqual(state.status, 'abstained')
        self.assertEqual(state.safety_skip_reason, 'no_medical_evidence')
        self.assertEqual(state.total_requests, 2)
        self.assertEqual(build_final_decision(state).status, 'HUMAN_REVIEW')
        self.assertFalse(state.diagnostics)

    def test_rejected_content_is_not_passed_to_agents(self):
        review = dict(verdict='not_relevant', reason='generic_overlap_only', positive_fact_quotes=[], content_quotes=[])
        workflow, llm, _, _ = self.fixture(review=review)
        # Only the AMG reference remains valid.
        ref = 'medical:' + amg_hit()['chunk_id']
        llm.diagnoses[0].primary_hypothesis.rationale.evidence_refs = [ref]
        llm.diagnoses[0].medical_knowledge_evidence = [ref]
        state = workflow.run(self.patient, self.patient_id)
        self.assertEqual(state.status, 'final')
        self.assertEqual(state.medical_retrieval['who']['selection_count'], 0)
        self.assertEqual(len(state.evidence.medical_knowledge), 1)

    def test_unknown_ids_fail_before_amg_or_diagnostic(self):
        state, (_, llm, patients, medical) = self.run_fixture(ids=['unknown'])
        self.assertEqual(state.status, 'failed')
        self.assertEqual(state.failure.code, 'agent_validation_failure')
        self.assertEqual(state.total_requests, 2)
        self.assertFalse(state.diagnostics)
        patients.retrieve.assert_not_called()
        medical.retrieve.assert_not_called()

    def test_four_documents_rejected_not_truncated(self):
        state, _ = self.run_fixture(ids=self.ids[:4])
        self.assertEqual(state.status, 'failed')
        self.assertIsNone(state.evidence)

    def test_invented_source_quote_fails_closed(self):
        bad = self.review.model_dump()
        bad['content_quotes'] = ['Invented source support']
        state, (_, _, _, medical) = self.run_fixture(review=bad)
        self.assertEqual(state.status, 'failed')
        self.assertFalse(state.diagnostics)
        medical.retrieve.assert_not_called()

    def test_budget_counts_selector_and_content_calls(self):
        state, (_, llm, _, medical) = self.run_fixture(policy=Phase6Policy(max_requests=2))
        self.assertEqual(state.failure.code, 'request_budget_exhausted')
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(state.total_requests, 2)
        self.assertIsNone(state.evidence)
        medical.retrieve.assert_not_called()

    def test_safety_review_and_block_routes_unchanged(self):
        for kwargs, expected in (({'safety_results': [semantic('safety_ambiguity')]}, 'HUMAN_REVIEW'),
                                 ({'critiques': [critique(safety_flags=['Fixture safety concern'])]}, 'BLOCK')):
            with self.subTest(expected=expected):
                state, _ = self.run_fixture(**kwargs)
                final = build_final_decision(state)
                self.assertEqual(final.status, expected)
                self.assertIsNone(final.diagnostic_proposal)
                self.assertTrue(final.human_review_required)

    def test_revision_never_repeats_selector_or_retrieval(self):
        state, (_, llm, patients, medical) = self.run_fixture(critiques=[critique('revision_required')])
        names = [c['text']['format']['schema']['title'] for c in llm.calls]
        self.assertEqual(names.count('WHOSelection'), 1)
        self.assertEqual(names.count('ContentRelevance'), 1)
        patients.retrieve.assert_called_once()
        medical.retrieve.assert_called_once()
        self.assertTrue(all(v.ticket.evidence_snapshot_id == state.evidence.snapshot_id for v in state.diagnostics))

    def test_new_provenance_tampering_rejected_by_unchanged_grounding(self):
        for key, value in (('document_id', 'invented'), ('catalog_sha256', 'bad'),
                           ('char_start', 0), ('selection_method', 'lexical'),
                           ('matched_terms', ['fixture']), ('content_quotes', ['invented']),
                           ('text', 'changed'), ('filename', '../escape.txt'), ('score', 0.4)):
            bad = {**self.hits[0], key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence_from_hits([bad], 'medical_knowledge')

    def test_source_change_between_review_and_handoff_rejected(self):
        workflow, llm, _, medical = self.fixture()
        original = self.selector.iter_selected
        count = 0
        def altered(selection):
            nonlocal count
            count += 1
            for doc in original(selection):
                yield replace(doc, document_sha256='a' * 64) if count == 2 else doc
        with patch.object(self.selector, 'iter_selected', side_effect=altered):
            state = workflow.run(self.patient, self.patient_id)
        self.assertEqual(state.failure.code, 'invalid_evidence_or_retrieval_failure')
        medical.retrieve.assert_not_called()

    def test_ambiguous_anchors_and_oversized_sections_fail_not_prefix(self):
        prefix = '\n'.join(self.source_text.split('\n', 4)[:4]) + '\n'
        for body in ('\nTitle\n=====\nAnchor statement. Anchor statement.',
                     '\nTitle\n=====\n' + 'x' * MAX_DOCUMENT_EXCERPT_CHARS + 'Anchor statement.'):
            with self.assertRaises(ValueError):
                section_spans(prefix + body, ['Anchor statement.'])

    def test_multiple_sections_keep_exact_spans_and_deduplicate(self):
        text = 'h\nh\nh\nh\n\nFirst\n=====\nAlpha statement. Beta statement.\n\nSecond\n======\nGamma statement.\n'
        spans = section_spans(text, ['Alpha statement.', 'Beta statement.', 'Gamma statement.'])
        self.assertEqual(len(spans), 1)  # Adjacent complete sections coalesce.
        self.assertIn('First\n', text[spans[0][0]:spans[0][1]])
        self.assertIn('Second\n', text[spans[0][0]:spans[0][1]])

    def test_default_demo_handles_combined_audit(self):
        from demo.terminal import Console, show_medical_audit
        state, _ = self.run_fixture()
        out = io.StringIO()
        show_medical_audit(Console(out), state.medical_retrieval)
        self.assertIn('CHECK4 WHO', out.getvalue())
        self.assertIn(self.document.document_id, out.getvalue())

    def test_label_bearing_input_rejected_before_calls(self):
        workflow, llm, patients, medical = self.fixture()
        with self.assertRaises(TypeError):
            workflow.run(Mock(patient=self.patient, labels='forbidden'), self.patient_id)
        self.assertEqual(llm.calls, [])
        patients.retrieve.assert_not_called()
        medical.retrieve.assert_not_called()

    def test_manual_preview_makes_no_cloud_or_retrieval_calls(self):
        from scripts.run_manual_who_amg import main
        source = Path(__file__).resolve().parents[1] / 'data/manual_check5.txt'
        with patch('rag.llm.provider.OpenAIProvider') as provider, \
             patch('rag.amg.AMGMedicalEvidence') as amg:
            status = main([str(source), '--output-dir', str(self.root / 'preview')])
        self.assertEqual(status, 0)
        provider.assert_not_called()
        amg.assert_not_called()


if __name__ == '__main__':
    unittest.main()

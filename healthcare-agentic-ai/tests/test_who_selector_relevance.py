"""Offline CHECK4 software contracts; mocks do not establish semantic accuracy."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from rag.agents.models import Failure
from rag.config import OpenAIConfig
from rag.llm.provider import Generation, OpenAIProvider
from rag.who_catalog_selector import WHOSelection, PROMPT
from rag.who_selector_relevance import (ContentRelevance, REVIEW_PROMPT, select_verified,
                                         validate_content_review)
from scripts.evaluate_who_selector_check4 import CASES, evaluate, case_patient
from tests import test_who_catalog_selector as fixtures


class WHORelevanceTests(unittest.TestCase):
    setUp = fixtures.WHOCatalogSelectorTests.setUp

    def relevant(self):
        return ContentRelevance(verdict='relevant', reason='direct_presenting_problem',
            positive_fact_quotes=['Fatigue and dizziness'], content_quotes=['SECRET BODY FIXTURE'])

    def rejected(self):
        return ContentRelevance(verdict='not_relevant', reason='generic_overlap_only',
                                positive_fact_quotes=[], content_quotes=[])

    def provider(self, ids, reviews):
        p = Mock()
        p.generate.side_effect = [Generation(parsed=WHOSelection(selected_ids=ids))] + [
            x if isinstance(x, Generation) else Generation(parsed=x) for x in reviews]
        return p

    def test_prompt_explicitly_disallows_generic_negation_and_diagnoses(self):
        self.assertIn('THREE', PROMPT)
        self.assertIn('Do NOT use lexical keyword matching', PROMPT)
        self.assertIn('no SOB', PROMPT)
        self.assertIn('POSITIVE PRESENTING', REVIEW_PROMPT)
        self.assertIn('Do not infer or output diagnoses', REVIEW_PROMPT)
        self.assertIn('possible causes', REVIEW_PROMPT)

    def test_zero_shortlist_no_load_no_content_calls(self):
        provider = self.provider([], [])
        with patch('rag.who_catalog_selector.bounded_read') as read:
            result = select_verified(self.selector, provider, self.patient)
        read.assert_not_called()
        provider.generate.assert_called_once()
        self.assertEqual(result['status'], 'no_selection')
        self.assertEqual(result['selection_count'], 0)

    def test_three_candidates_final_subset_and_full_content_sink(self):
        provider = self.provider(self.ids[:3], [self.relevant(), self.rejected(), self.relevant()])
        loaded = []
        with patch('rag.who_lookup.WHOLookup.search', side_effect=AssertionError('No lexical lookup')):
            result = select_verified(self.selector, provider, self.patient, loaded.append)
        self.assertEqual(result['selected_ids'], [self.ids[0], self.ids[2]])
        self.assertEqual(result['selection_count'], 2)
        self.assertEqual(len(result['candidates']), 3)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(provider.generate.call_count, 4)
        for call, doc in zip(provider.generate.call_args_list[1:], loaded):
            self.assertEqual(call.args[1]['document_text'], doc.text)
            self.assertEqual(call.args[1]['patient_case'], self.patient.to_inference_dict())
            self.assertIs(call.args[2], ContentRelevance)
        self.assertNotIn('document_text', json.dumps(result))

    def test_no_refill_when_all_rejected(self):
        provider = self.provider(self.ids[:1], [self.rejected()])
        result = select_verified(self.selector, provider, self.patient)
        self.assertEqual(result['status'], 'no_selection')
        self.assertEqual(provider.generate.call_count, 2)

    def test_uncertain_is_abstention(self):
        value = ContentRelevance(verdict='uncertain', reason='insufficient_evidence',
                                 positive_fact_quotes=[], content_quotes=[])
        result = select_verified(self.selector, self.provider(self.ids[:1], [value]), self.patient)
        self.assertEqual(result['selected_ids'], [])
        self.assertEqual(result['status'], 'no_selection')

    def test_content_failure_withholds_previously_accepted_documents(self):
        failure = Generation(failure=Failure(code='api_failure', message='fixture'))
        p = self.provider(self.ids[:3], [self.relevant(), failure])
        sink = Mock()
        result = select_verified(self.selector, p, self.patient, sink)
        self.assertEqual(result['status'], 'content_review_failure')
        self.assertEqual(result['selection_count'], 0)
        self.assertEqual(result['selected_documents'], [])
        self.assertEqual(sink.call_count, 2)
        self.assertEqual(p.generate.call_count, 3)

    def test_selector_failure_not_empty_success(self):
        p = Mock()
        p.generate.return_value = Generation(failure=Failure(code='timeout', message='fixture'))
        result = select_verified(self.selector, p, self.patient)
        self.assertEqual(result['status'], 'selector_failure')
        self.assertEqual(result['candidates'], [])

    def test_four_candidates_rejected_even_if_forged(self):
        p = Mock()
        p.generate.return_value = Generation(parsed=WHOSelection.model_construct(selected_ids=self.ids[:4]))
        with patch('rag.who_catalog_selector.bounded_read') as read, self.assertRaises(ValueError):
            select_verified(self.selector, p, self.patient)
        read.assert_not_called()

    def test_unknown_candidate_fails_before_sink_or_file_read(self):
        p = self.provider([self.ids[0], 'invented'], [])
        sink = Mock()
        with patch('rag.who_catalog_selector.bounded_read') as read, self.assertRaises(ValueError):
            select_verified(self.selector, p, self.patient, sink)
        sink.assert_not_called()
        read.assert_not_called()

    def test_bad_quotes_not_repaired_or_accepted(self):
        for field, quote in [('positive_fact_quotes', 'invented positive symptom'),
                             ('content_quotes', 'invented WHO evidence'),
                             ('content_quotes', 'Title: Fixture')]:
            value = self.relevant().model_copy(update={field: [quote]})
            with self.assertRaises(ValueError):
                select_verified(self.selector, self.provider(self.ids[:1], [value]), self.patient)

    def test_schema_disallows_diagnosis_prose_or_contradictory_verdict(self):
        invalid = [dict(self.relevant().model_dump(), diagnosis='forbidden'),
                   dict(self.relevant().model_dump(), reason='negated_or_unknown_only'),
                   dict(self.relevant().model_dump(), positive_fact_quotes=[]),
                   dict(self.rejected().model_dump(), content_quotes=['some quote']),
                   dict(self.relevant().model_dump(), content_quotes=['x' * 601])]
        for values in invalid:
            with self.assertRaises(ValueError):
                ContentRelevance.model_validate(values)

    def test_forged_review_revalidated(self):
        value = self.relevant().model_copy(update={'content_quotes': []})
        with self.assertRaises(ValueError):
            select_verified(self.selector, self.provider(self.ids[:1], [value]), self.patient)

    def test_provider_byte_cap_not_truncated_or_sent(self):
        client = Mock()
        provider = OpenAIProvider(OpenAIConfig(max_input_bytes=10, max_retries=0), client=client)
        result = select_verified(self.selector, provider, self.patient)
        self.assertEqual(result['status'], 'selector_failure')
        self.assertEqual(result['catalog_stage']['failure']['code'], 'context_budget')
        client.responses.create.assert_not_called()

    def test_eight_cases_prepared_without_cloud_or_text_loading(self):
        with patch('rag.who_catalog_selector.bounded_read') as read:
            result = evaluate(self.root / 'prepared', self.selector)
        read.assert_not_called()
        self.assertEqual(len(result['cases']), 8)
        self.assertTrue(all(c['status'] == 'prepared_not_run' for c in result['cases']))
        plan = json.loads((self.root / 'prepared/plan.json').read_text())
        self.assertEqual(len(plan['cases']), 8)
        self.assertEqual(len(set(c['patient_id'] for c in plan['cases'])), 8)
        self.assertNotIn('diagnosis', case_patient(CASES[0]).to_inference_dict())

    def test_evaluation_saves_full_contents_and_no_lexical_or_pipeline(self):
        p = Mock()
        # All mock judgments reject; never assert semantic accuracy with fixtures.
        p.generate.side_effect = [g for _ in CASES for g in (
            Generation(parsed=WHOSelection(selected_ids=self.ids[:1])), Generation(parsed=self.rejected()))]
        with patch('rag.who_lookup.WHOLookup.search', side_effect=AssertionError('lexical')):
            summary = evaluate(self.root / 'run', self.selector, p)
        self.assertEqual(p.generate.call_count, 16)
        self.assertFalse(summary['pipeline_run'])
        for case in CASES:
            directory = self.root / 'run' / case[0]
            files = list(directory.glob('*.txt'))
            self.assertEqual(len(files), 1)
            report = json.loads((directory / 'report.json').read_text())
            self.assertEqual(report['status'], 'no_selection')
            self.assertIn('SECRET BODY FIXTURE', files[0].read_text())

    def test_import_isolated(self):
        code = "import rag.who_selector_relevance, scripts.evaluate_who_selector_check4, sys; assert not any(n in sys.modules for n in ('rag.amg.backend','orchestration.phase6.orchestrator','sentence_transformers','chromadb'))"
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True,
                        cwd=Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    unittest.main()

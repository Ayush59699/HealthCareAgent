"""Task 7.10 preflight tests only; no prototype or relevance judgments.

Reuse the synthetic fixture builder, NOT the Task 7.9 test suite. Real six-case
reproduction is recorded separately in outputs/task7.10/control-preflight.
"""
from copy import deepcopy
from pathlib import Path
import unittest

from tests.test_task79_selection_analysis import fixture
from rag.experimental_medical.task79_selection_analysis import (
    original_functions, replay, analyze, fingerprint)
from scripts.experiment_task710_selection import exact_control, require_all_controls

ROOT = Path(__file__).resolve().parents[1]


class Task710PreflightTests(unittest.TestCase):
    def setUp(self):
        self.choose, self.tokenize = original_functions(ROOT)
        self.saved = fixture([
            ('h', 'Background', -1., 5., 'background', 'context background history'),
            ('a', 'Evidence A', 9., -1., 'one', 'alpha beta gamma delta'),
            ('b', 'Evidence B', 8., -1., 'two', 'alpha beta gamma delta epsilon'),
            ('c', 'Evidence C', 7., -1., 'one', 'theta iota kappa'),
            ('d', 'Evidence D', 6., -1., 'one', 'lambda mu sigma'),
            ('e', 'Evidence E', 5., -1., 'three', 'tau upsilon omega'),
            ('f', 'Evidence F', 4., -1., 'four', 'copper silver gold'),
            ('g', 'Evidence G', 3., -1., 'five', 'orange violet yellow'),
            ('n', 'Ineligible', 0., -1., 'six', 'zero neutral relevance')])
        self.saved['statistics'] = {'selected': len(self.saved['selected_ids'])}
        self.baseline = replay(self.saved, self.choose, self.tokenize)
        # analyze needs both trace slots; these tests exercise ONLY saved-control
        # fields, so pass the baseline twice. No counterfactual is run.
        self.analysis = analyze(self.saved, self.baseline, self.baseline)
        self.prior = {k: self.baseline[k] for k in ('selected_ids', 'selected_roles', 'input_fingerprint')}
        self.prior.update(case=1, score_condition='experimental', passed=True)

    def check(self):
        return exact_control(self.saved, self.prior, self.analysis, self.choose, self.tokenize)

    def test_exact_ids_order_roles_documents_events_count(self):
        result = self.check()
        self.assertEqual(result['selected_ids'], ['a', 'c', 'e', 'f', 'h'])
        self.assertEqual([r['document_id'] for r in result['selected']], ['one', 'one', 'three', 'four', 'background'])
        self.assertEqual(result['selected'][-1]['role'], 'history')
        self.assertTrue(result['exact_task79_events_documents_count'])
        self.assertEqual(result['final_count'], 5)

    def test_changed_task79_order_stops(self):
        self.prior = deepcopy(self.prior)
        self.prior['selected_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'control mismatch'):
            self.check()

    def test_changed_task79_role_stops(self):
        self.prior = deepcopy(self.prior)
        self.prior['selected_roles']['a'] = 'history'
        with self.assertRaisesRegex(ValueError, 'control mismatch'):
            self.check()

    def test_changed_task79_document_stops(self):
        self.analysis['candidate_analysis'][0]['document_id'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'metadata mismatch: document_id'):
            self.check()

    def test_changed_task79_rejection_event_stops(self):
        self.analysis['candidate_analysis'][0]['roles'][0]['saved_pass_events'][0]['reason'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'events mismatch'):
            self.check()

    def test_changed_task79_count_stops(self):
        self.analysis['summary']['selected_candidates'] += 1
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            self.check()

    def test_changed_frozen_eligibility_stops(self):
        self.saved['candidate_audit'][-1]['role_details']['symptoms']['positive_logit_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'eligibility inconsistent'):
            self.check()

    def test_changed_frozen_text_stops(self):
        self.saved['candidate_audit'][0]['evidence_text'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'source text hash mismatch'):
            self.check()

    def test_missing_task79_candidate_stops(self):
        self.analysis['candidate_analysis'].pop()
        with self.assertRaisesRegex(ValueError, 'candidate IDs mismatch'):
            self.check()

    def test_original_constraints_preserved(self):
        result = self.check()
        self.assertLessEqual(result['final_count'], 5)
        self.assertNotIn('n', result['selected_ids'])
        doc_event = next(e for e in self.baseline['events'] if e['chunk_id'] == 'd' and e['role'] == 'symptoms')
        self.assertEqual(doc_event['reason'], 'document_limit_2')
        duplicate = next(e for e in self.baseline['events'] if e['chunk_id'] == 'b' and e['role'] == 'symptoms')
        self.assertEqual(duplicate['reason'], 'near_duplicate_text')
        self.assertEqual(duplicate['nonbudget_blockers_at_this_prefix'], [{'chunk_id': 'a', 'jaccard': .8}])

    def test_deterministic_and_no_mutation(self):
        before = fingerprint([self.saved, self.prior, self.analysis])
        self.assertEqual(self.check(), self.check())
        self.assertEqual(before, fingerprint([self.saved, self.prior, self.analysis]))

    def test_all_six_controls_required(self):
        controls = [{'case': case, 'score_condition': condition, 'passed': True}
                    for case in (1, 2, 3) for condition in ('control', 'experimental')]
        require_all_controls(controls)
        with self.assertRaises(ValueError):
            require_all_controls(controls[:-1])
        controls[-1]['passed'] = False
        with self.assertRaises(ValueError):
            require_all_controls(controls)


if __name__ == '__main__':
    unittest.main()

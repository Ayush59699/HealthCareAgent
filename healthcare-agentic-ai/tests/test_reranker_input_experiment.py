"""Task 7.6 deterministic contracts; explicit test doubles, no model downloads."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from orchestration.evidence import fingerprint
from rag.experimental_medical.ranking_ablation import validate_frozen_experiment
from rag.experimental_medical.reranker_input_experiment import (
    FORMATS, format_queries, pair_plan, check_token_budget, score_pairs,
    apply_scores, run_format, baseline_drift, compare_formats)
from tests.test_ranking_ablation import fixture


class RerankerInputTests(unittest.TestCase):
    def setUp(self):
        self.state, self.experiment, payloads = fixture()
        self.records = validate_frozen_experiment(self.state, self.experiment, payloads)
        self.reranker = Mock()
        self.reranker.batch_size = 2
        self.reranker.truncated_pairs = 0
        self.reranker.signature = self.experiment['reranker']
        self.reranker.model.tokenizer.encode.side_effect = lambda q, p, truncation: list(range(len((q+' '+p).split()) + 3))
        self.reranker.score.side_effect = lambda pairs: [3.] * len(pairs)

    def test_original_is_byte_identical_query_text(self):
        result = format_queries(self.state, self.experiment['plan'], 'original')
        for q in self.experiment['plan']['queries']:
            self.assertEqual(result[q['query_id']]['reranker_input'], q['text'])

    def test_natural_format_preserves_every_fact_and_role_without_new_demographics(self):
        result = format_queries(self.state, self.experiment['plan'], 'role_explicit')
        for q in self.experiment['plan']['queries']:
            formatted = result[q['query_id']]
            for fact in q['facts']:
                self.assertIn(fact['text'], formatted['reranker_input'])
            self.assertEqual(formatted['facts'], q['facts'])
            self.assertEqual(formatted['fact_ids'], q['fact_ids'])
            self.assertNotIn(self.state.patient_id, formatted['reranker_input'])
            self.assertNotIn('diagnosis confirmed', formatted['reranker_input'])
            if q['kind'] == 'symptoms':
                self.assertIn('current reported findings', formatted['reranker_input'])
                self.assertNotIn('for a child', formatted['reranker_input'])
                self.assertNotIn('historical', formatted['reranker_input'])
            else:
                self.assertIn('historical or background', formatted['reranker_input'])
                self.assertIn('for a child', formatted['reranker_input'])

    def test_pair_order_query_assignments_and_passages_are_identical(self):
        pairs0, audit0, _ = pair_plan(self.state, self.experiment, self.records, 'original')
        pairs1, audit1, _ = pair_plan(self.state, self.experiment, self.records, 'role_explicit')
        self.assertEqual([p[1] for p in pairs0], [p[1] for p in pairs1])
        for a, b in zip(audit0, audit1):
            for key in ('chunk_id', 'role', 'query_id', 'passage_sha256', 'fact_ids', 'frozen_role_logit'):
                self.assertEqual(a[key], b[key])
        self.assertEqual([a['chunk_id'] for a in audit0[::2]], [r['chunk_id'] for r in self.experiment['candidate_pool']])

    def test_preflight_refuses_truncation_without_model_scoring(self):
        pairs, audit, _ = pair_plan(self.state, self.experiment, self.records, 'role_explicit')
        self.reranker.model.tokenizer.encode.side_effect = lambda *a, **kw: list(range(513))
        with self.assertRaisesRegex(ValueError, 'would truncate'):
            check_token_budget(self.reranker, pairs, audit)
        self.reranker.score.assert_not_called()

    def test_only_small_sequential_batches_and_memory_callbacks(self):
        monitor, progress = Mock(), Mock()
        pairs = [('q', 'p')] * 5
        scores = score_pairs(self.reranker, pairs, monitor, progress)
        self.assertEqual(scores, [3.] * 5)
        self.assertEqual([len(c.args[0]) for c in self.reranker.score.call_args_list], [2, 2, 1])
        self.assertEqual(monitor.call_count, 6)
        progress.assert_called_with(5, 5)
        self.reranker.batch_size = 4
        with self.assertRaises(ValueError):
            score_pairs(self.reranker, pairs)

    def test_bad_model_outputs_fail_without_fallback(self):
        for values in ([float('nan')], [], [True], [float('inf')]):
            self.reranker.score.side_effect = lambda pairs: values
            with self.assertRaises(ValueError):
                score_pairs(self.reranker, [('q', 'p')])
        self.reranker.score.side_effect = RuntimeError('model failure')
        with self.assertRaises(RuntimeError):
            score_pairs(self.reranker, [('q', 'p')])

    def test_original_recompute_matches_cached_snapshot(self):
        result = run_format(self.state, self.experiment, self.records, self.reranker, 'original')
        drift = baseline_drift(self.experiment, result)
        self.assertTrue(drift['passed'])
        self.assertEqual(drift['maximum_absolute_logit_drift'], 0.)
        self.assertEqual(result['evidence_snapshot'], self.experiment['evidence_snapshot'])

    def test_baseline_drift_control_detects_changed_neural_results(self):
        self.reranker.score.side_effect = lambda pairs: [-3.] * len(pairs)
        result = run_format(self.state, self.experiment, self.records, self.reranker, 'original')
        self.assertFalse(baseline_drift(self.experiment, result)['passed'])

    def test_natural_format_changes_only_learned_and_derived_ranking_fields(self):
        before = fingerprint(self.experiment)
        pairs, audit, _ = pair_plan(self.state, self.experiment, self.records, 'role_explicit')
        working = apply_scores(self.experiment, audit, [1.] * len(pairs))
        changed_keys = {'symptoms_score', 'history_score', 'symptoms_ranking_score',
                        'history_ranking_score', 'reranker_score', 'reranked_rank'}
        originals = {r['chunk_id']: r for r in self.experiment['reranked']}
        for row in working['reranked']:
            original = originals[row['chunk_id']]
            self.assertEqual({k:v for k,v in row.items() if k not in changed_keys},
                             {k:v for k,v in original.items() if k not in changed_keys})
            for role in ('symptoms', 'history'):
                self.assertAlmostEqual(row[role+'_ranking_score'] - row[role+'_score'],
                                       original[role+'_ranking_score'] - original[role+'_score'])
        self.assertEqual(fingerprint(self.experiment), before)
        self.assertEqual(working['plan'], self.experiment['plan'])
        self.assertEqual(working['candidate_pool'], self.experiment['candidate_pool'])

    def test_zero_and_negative_logits_still_produce_no_evidence(self):
        for score in (0., -2.):
            self.reranker.score.side_effect = lambda pairs: [score] * len(pairs)
            result = run_format(self.state, self.experiment, self.records, self.reranker, 'role_explicit')
            self.assertEqual(result['selected_ids'], [])
            self.assertEqual(result['evidence_snapshot']['medical_knowledge'], [])
            self.assertTrue(all(r['selection_or_removal_stage'] == 'reranker_eligibility' for r in result['candidate_audit']))

    def test_score_transitions_and_full_input_audit_are_logged(self):
        original = run_format(self.state, self.experiment, self.records, self.reranker, 'original')
        self.reranker.score.side_effect = lambda pairs: [-2. if 'current reported findings' in q else 3. for q, p in pairs]
        formatted = run_format(self.state, self.experiment, self.records, self.reranker, 'role_explicit')
        comparison = compare_formats(original, formatted)
        self.assertEqual(comparison['eligibility_transitions']['symptoms']['positive_to_nonpositive'], len(self.records))
        for row in formatted['candidate_audit']:
            for details in row['role_details'].values():
                for key in ('reranker_input', 'originating_query', 'fact_ids', 'dense_rank', 'bm25_rank',
                            'raw_reranker_score', 'frozen_role_logit', 'pair_tokens', 'passage_sha256'):
                    self.assertIn(key, details)
        bad = deepcopy(formatted)
        bad['pair_order_fingerprint'] = 'altered'
        with self.assertRaises(ValueError):
            compare_formats(original, bad)

    def test_no_search_or_query_generation_during_score_only_experiment(self):
        with patch('rag.experimental_medical.retrieval.BM25.search', side_effect=AssertionError('search forbidden')), \
             patch('rag.experimental_medical.retrieval.HybridRetriever.retrieve', side_effect=AssertionError('retrieval forbidden')):
            run_format(self.state, self.experiment, self.records, self.reranker, 'role_explicit')

    def test_missing_duplicate_pair_scores_and_depth_change_rejected(self):
        pairs, audit, _ = pair_plan(self.state, self.experiment, self.records, 'original')
        with self.assertRaises(ValueError):
            apply_scores(self.experiment, audit, [3.])
        duplicate = deepcopy(audit)
        duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError):
            apply_scores(self.experiment, duplicate, [3.] * len(duplicate))
        for depth in (100, True, 50.0):
            with self.assertRaises(ValueError):
                pair_plan(self.state, {**self.experiment, 'candidate_depth': depth}, self.records, 'original')

    def test_hidden_rows_population_change_unknown_format_and_cli_flags_rejected(self):
        with self.assertRaises(TypeError):
            format_queries({'PATHOLOGY': 'HIDDEN'}, self.experiment['plan'], 'role_explicit')
        with self.assertRaises(ValueError):
            format_queries(self.state, self.experiment['plan'], 'tuned_for_this_patient')
        with self.assertRaises(ValueError):
            format_queries(self.state, {**self.experiment['plan'], 'population': 'older adult'}, 'role_explicit')
        from scripts.experiment_reranker_inputs import arguments
        self.assertEqual(arguments([]).samples, [1, 2, 3])
        with self.assertRaises(SystemExit):
            arguments(['--batch-size', '16'])


if __name__ == '__main__':
    unittest.main()

"""Task 7.8 synthetic tests: no model downloads, retrieval or hidden labels."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestration.evidence import fingerprint
from rag.experimental_medical.reranker_prefix_control import validate_frozen_case
from rag.experimental_medical.reranker_suitability import (
    MODEL, MODEL_REVISION, estimate_model, require_headroom, inspect_model,
    run_condition, compare_pairs, pair_plan, download_model)
from rag.experimental_medical.suitability_audit import pair_status, important_audit
from scripts.experiment_reranker_suitability import require_all_controls, baseline_checks
from tests.test_ranking_ablation import fixture


def config():
    return {'model_type': 'bert', 'architectures': ['BertForSequenceClassification'],
            'num_labels': 1, 'max_position_embeddings': 512, 'hidden_size': 768,
            'num_hidden_layers': 12, 'vocab_size': 30522, 'intermediate_size': 3072,
            'num_attention_heads': 12, 'type_vocab_size': 2}


class SuitabilityTests(unittest.TestCase):
    def setUp(self):
        self.state, self.experiment, self.payloads = fixture()
        self.records = validate_frozen_case(self.state,
            {'patient_state': self.state.model_dump(), 'experiment': self.experiment},
            self.payloads, enforce_case_size=False)
        self.model = Mock(batch_size=2, truncated_pairs=0, signature=self.experiment['reranker'])
        self.model.model.tokenizer.encode.side_effect = lambda q, p, truncation: list(range(20))
        self.model.score.side_effect = lambda pairs: [3.] * len(pairs)

    def result(self, condition='control', score=3.):
        self.model.signature = (self.experiment['reranker'] if condition == 'control' else
            {'model': MODEL, 'revision': MODEL_REVISION, 'max_length': 512,
             'score': 'raw cross-encoder logit; not clinical confidence'})
        self.model.score.side_effect = lambda pairs: [score] * len(pairs)
        return run_condition(self.state, self.experiment, self.records, self.model, condition)

    def test_small_model_memory_estimate(self):
        estimate = estimate_model(config(), 438_000_000)
        self.assertLess(estimate['approximate_parameters'], 125_000_000)
        self.assertGreater(estimate['estimated_incremental_peak_gb'], 2.)
        require_headroom(10., estimate)
        with self.assertRaises(RuntimeError):
            require_headroom(12., estimate)
        with self.assertRaises(RuntimeError):
            require_headroom(13.5-estimate['estimated_incremental_peak_gb'], estimate)

    def test_unsafe_or_wrong_model_refused(self):
        for change in ({'num_labels': 2}, {'hidden_size': 1024}, {'num_hidden_layers': 24},
                       {'architectures': ['BertModel']}, {'max_position_embeddings': 1024}):
            with self.assertRaises(ValueError):
                estimate_model({**config(), **change}, 438_000_000)
        for size in (0, 500_000_001):
            with self.assertRaises(ValueError):
                estimate_model(config(), size)

    def test_metadata_pins_one_revision_before_weights(self):
        info = {'sha': MODEL_REVISION, 'siblings': [
            {'rfilename': 'pytorch_model.bin', 'size': 438_000_000, 'lfs': {'sha256': 'b'*64}}]}
        with patch('rag.experimental_medical.reranker_suitability.read_remote',
                   side_effect=[json.dumps(info).encode(), json.dumps(config()).encode()]) as remote:
            result = inspect_model()
        self.assertEqual(result['revision'], MODEL_REVISION)
        self.assertEqual(remote.call_count, 2)
        self.assertIn('/' + MODEL_REVISION + '/config.json', remote.call_args_list[1].args[0])

    def test_mutable_revision_rejected(self):
        with patch('rag.experimental_medical.reranker_suitability.read_remote', return_value=b'{"sha":"main"}'):
            with self.assertRaises(ValueError):
                inspect_model()

    def test_download_headroom_refused_before_network(self):
        metadata = {'config': config(), 'memory_estimate': estimate_model(config(), 438_000_000)}
        monitor = Mock()
        monitor.psutil.virtual_memory.return_value.used = 12_000_000_000
        with patch('huggingface_hub.hf_hub_download') as download:
            with self.assertRaises(RuntimeError):
                download_model(metadata, Path('unused'), monitor)
        download.assert_not_called()

    def test_verbatim_pairs_and_unchanged_frozen_fields(self):
        before = fingerprint(self.experiment)
        original = self.result()
        alternative = self.result('experimental')
        comparison = compare_pairs(original, alternative, self.experiment)
        self.assertEqual(comparison['pair_count'], self.experiment['reranker_pair_count'])
        self.assertEqual(original['formatted_queries'], alternative['formatted_queries'])
        for query in original['formatted_queries']:
            self.assertEqual(query['retrieval_query'], query['reranker_input'])
        for p in comparison['pairs']:
            self.assertEqual(p['control_query'], p['experimental_query'])
        self.assertEqual(fingerprint(self.experiment), before)

    def test_all_transition_directions_and_zero_gate(self):
        for left, right, direction in ((-1., 1., 'nonpositive_to_positive'),
                (1., 0., 'positive_to_nonpositive'), (1., 2., 'positive_to_positive'),
                (0., -1., 'nonpositive_to_nonpositive')):
            a, b = self.result(score=left), self.result('experimental', right)
            comparison = compare_pairs(a, b, self.experiment)
            self.assertTrue(all(p['eligibility_transition'] == direction for p in comparison['pairs']))
            self.assertEqual(sum(c[direction] for c in comparison['eligibility_transitions'].values()), comparison['pair_count'])
            if right <= 0:
                self.assertEqual(b['selected_ids'], [])

    def test_comparison_rejects_query_or_provenance_change(self):
        a, b = self.result(), self.result('experimental')
        changed = deepcopy(b)
        next(iter(changed['candidate_audit'][0]['role_details'].values()))['reranker_input'] += ' '
        with self.assertRaises(ValueError):
            compare_pairs(a, changed, self.experiment)
        changed = deepcopy(b)
        changed['candidate_audit'][0]['evidence_text'] += ' '
        with self.assertRaises(ValueError):
            compare_pairs(a, changed, self.experiment)

    def test_wrong_control_model_and_batch_refused(self):
        self.model.signature = {'model': MODEL}
        with self.assertRaises(ValueError):
            run_condition(self.state, self.experiment, self.records, self.model, 'control')
        self.model.batch_size = 4
        with self.assertRaises(ValueError):
            self.result()

    def test_no_truncation_before_scoring(self):
        self.model.model.tokenizer.encode.side_effect = lambda *args, **kwargs: list(range(513))
        with self.assertRaises(ValueError):
            self.result('experimental')
        self.model.score.assert_not_called()

    def test_nonfinite_scores_stop(self):
        with self.assertRaises(ValueError):
            self.result('experimental', float('nan'))

    def test_memory_stop_no_batch(self):
        from scripts.ablate_medical_ranking import MemorySafetyStop
        monitor = Mock(side_effect=MemorySafetyStop('synthetic stop'))
        with self.assertRaises(MemorySafetyStop):
            run_condition(self.state, self.experiment, self.records, self.model, 'control', monitor)
        self.model.score.assert_not_called()

    def test_all_three_controls_must_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for sample in (1, 2, 3):
                directory = root / f'validate-{sample}'
                directory.mkdir()
                (directory / 'baseline-control.json').write_text(json.dumps({'passed': sample != 3}))
            with self.assertRaises(ValueError):
                require_all_controls(root)
            (root / 'validate-3/baseline-control.json').write_text('{"passed": true}')
            require_all_controls(root)

    def test_audit_distinguishes_rejection_displacement_retention(self):
        pair = {'control_selected_for_role': True, 'experimental_selected_for_role': False,
                'experimental_eligible': False}
        self.assertEqual(pair_status(pair, 'experimental'), 'candidate_exists_but_reranker_rejects')
        pair['experimental_eligible'] = True
        self.assertEqual(pair_status(pair, 'experimental'), 'candidate_eligible_but_loses_during_selection')
        pair['experimental_selected_for_role'] = True
        self.assertEqual(pair_status(pair, 'experimental'), 'candidate_remains_selected')
        pair['control_selected_for_role'] = False
        self.assertEqual(pair_status(pair, 'experimental'), 'candidate_newly_selected')

    def test_audit_includes_requested_groups_even_when_absent(self):
        a, b = self.result(), self.result('experimental')
        audit = important_audit(compare_pairs(a, b, self.experiment), a)
        self.assertEqual(len(audit), 7)
        self.assertIn('CKD/kidney', audit)
        self.assertIn('Panic Disorder', audit)
        self.assertTrue(all('present' in group for group in audit.values()))


if __name__ == '__main__':
    unittest.main()

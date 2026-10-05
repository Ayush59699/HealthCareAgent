"""Task 7.7 deterministic tests only; synthetic fixtures, no model/network calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from orchestration.evidence import EvidenceSnapshot, fingerprint
from rag.experimental_medical.reranker_prefix_control import (
    PREFIXES, FORMATS, prefix_query, pair_plan, validate_frozen_case, run_format,
    assert_score_only, baseline_control, require_baseline, compare_pairs, important_audit,
    check_token_budget, score_pairs, apply_scores)
from rag.experimental_medical.prefix_integrity import (
    HISTORICAL_CONFIG, AUTHORIZED_CONFIG, config_exception, verify_historical,
    verify_unchanged, digest)
from tests.test_ranking_ablation import fixture


class PrefixControlTests(unittest.TestCase):
    def setUp(self):
        self.state, self.experiment, self.payloads = fixture()
        self.cached = {'patient_state': self.state.model_dump(), 'experiment': self.experiment}
        self.records = validate_frozen_case(self.state, self.cached, self.payloads, enforce_case_size=False)
        self.model = Mock(batch_size=2, truncated_pairs=0, signature=self.experiment['reranker'])
        self.model.model.tokenizer.encode.side_effect = lambda q, p, truncation: list(range(len((q + ' ' + p).split()) + 3))
        self.model.score.side_effect = lambda pairs: [3.] * len(pairs)

    def run_variant(self, variant):
        return run_format(self.state, self.experiment, self.records, self.model, variant)

    def test_two_exact_prefixes_and_no_alternative_formats(self):
        self.assertEqual(PREFIXES, {'symptoms': 'Current findings: ', 'history': 'Historical/background: '})
        self.assertEqual(FORMATS, ('original', 'prefix_only'))
        for role in ('unknown', 'current', '', None):
            with self.assertRaises(ValueError):
                prefix_query('unchanged', role)
        with self.assertRaises(ValueError):
            pair_plan(self.state, self.experiment, self.records, 'role_explicit')

    def test_byte_exact_preservation_no_rewrite_reorder_or_demographic_addition(self):
        originals = ['sweating; shortness of breath; palpitations; numbness and tingling',
                     'child asthma symptoms and clinical information',
                     ' Mixed CASE; café; x-y?!  ', 'a; b\nc; d']
        for original in originals:
            for role, prefix in PREFIXES.items():
                value = prefix_query(original, role)
                self.assertEqual(value.encode('utf8'), prefix.encode('utf8') + original.encode('utf8'))
                self.assertEqual(value[len(prefix):], original)
                self.assertEqual(len(value), len(prefix) + len(original))

    def test_frozen_candidate_order_assignments_fact_ids_passages_and_hashes(self):
        before = fingerprint(self.experiment)
        p0, a0, q0 = pair_plan(self.state, self.experiment, self.records, 'original')
        p1, a1, q1 = pair_plan(self.state, self.experiment, self.records, 'prefix_only')
        self.assertEqual(len(p0), self.experiment['reranker_pair_count'])
        self.assertEqual([p[1] for p in p0], [p[1] for p in p1])
        self.assertEqual([a['chunk_id'] for a in a0[::2]], [r['chunk_id'] for r in self.experiment['candidate_pool']])
        for x, y, pair in zip(a0, a1, p1):
            self.assertEqual({k: v for k, v in x.items() if k != 'reranker_input'},
                             {k: v for k, v in y.items() if k != 'reranker_input'})
            self.assertEqual(pair[0], prefix_query(x['retrieval_query'], x['role']))
        for q in self.experiment['plan']['queries']:
            self.assertEqual(q0[q['query_id']]['reranker_input'], q['text'])
            self.assertEqual(q1[q['query_id']]['reranker_input'], prefix_query(q['text'], q['kind']))
        self.assertEqual(before, fingerprint(self.experiment))

    def test_unknown_depth_and_hidden_state_rejected(self):
        for depth in (100, 50.0, True):
            with self.assertRaises(ValueError):
                pair_plan(self.state, {**self.experiment, 'candidate_depth': depth}, self.records, 'original')
        with self.assertRaises(TypeError):
            pair_plan({}, self.experiment, self.records, 'original')

    def test_corrupt_frozen_sources_ids_indexes_and_pool_rejected(self):
        def broken_cached(change):
            cached = deepcopy(self.cached)
            change(cached['experiment'])
            with self.assertRaises(ValueError):
                validate_frozen_case(self.state, cached, self.payloads, enforce_case_size=False)
        broken_cached(lambda e: e['candidate_records'][0].update(evidence_text='altered'))
        broken_cached(lambda e: e['candidate_records'][0].update(chunk_id='different'))
        broken_cached(lambda e: e['reranked'][0].update(index=-1))
        broken_cached(lambda e: e['candidate_pool'].reverse())
        bad = deepcopy(self.payloads)
        next(iter(bad.values()))['text'] = 'changed approved source'
        with self.assertRaises(ValueError):
            validate_frozen_case(self.state, self.cached, bad, enforce_case_size=False)
        with self.assertRaises(ValueError):
            validate_frozen_case(self.state, self.cached, self.payloads)  # not an authorized real case size

    def test_no_search_query_generation_or_fusion_in_verification_or_scoring(self):
        forbidden = ['rag.experimental_medical.retrieval.HybridRetriever.retrieve',
                     'rag.experimental_medical.retrieval.BM25.search',
                     'rag.experimental_medical.retrieval.fuse',
                     'rag.experimental_medical.ranking_ablation.fuse',
                     'rag.experimental_medical.retrieval.build_queries',
                     'rag.experimental_medical.queries.build_queries']
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target in forbidden:
                stack.enter_context(patch(target, side_effect=AssertionError('forbidden fresh retrieval work')))
            validate_frozen_case(self.state, self.cached, self.payloads, enforce_case_size=False)
            self.run_variant('prefix_only')

    def test_exact_512_allowed_513_refused_before_any_score(self):
        pairs, audit, _ = pair_plan(self.state, self.experiment, self.records, 'prefix_only')
        self.model.model.tokenizer.encode.side_effect = lambda *args, **kwargs: list(range(512))
        self.assertEqual(check_token_budget(self.model, pairs, audit), [512] * len(pairs))
        self.model.model.tokenizer.encode.side_effect = lambda *args, **kwargs: list(range(513))
        with self.assertRaisesRegex(ValueError, 'would truncate'):
            self.run_variant('prefix_only')
        self.model.score.assert_not_called()

    def test_batch_two_and_synchronous_memory_checks(self):
        sample, progress = Mock(), Mock()
        self.assertEqual(score_pairs(self.model, [('q', 'p')] * 5, sample, progress), [3.] * 5)
        self.assertEqual([len(c.args[0]) for c in self.model.score.call_args_list], [2, 2, 1])
        self.assertEqual(sample.call_count, 6)
        progress.assert_called_with(5, 5)
        self.model.batch_size = 4
        with self.assertRaises(ValueError):
            self.run_variant('prefix_only')

    def test_memory_stop_prevents_next_batch(self):
        from scripts.ablate_medical_ranking import MemorySafetyStop
        sample = Mock(side_effect=MemorySafetyStop('synthetic stop'))
        with self.assertRaises(MemorySafetyStop):
            score_pairs(self.model, [('q', 'p')], sample)
        self.model.score.assert_not_called()

    def test_score_only_fields_and_unchanged_positive_gate(self):
        _, audit, _ = pair_plan(self.state, self.experiment, self.records, 'prefix_only')
        working = apply_scores(self.experiment, audit, [-1.] * len(audit))
        assert_score_only(self.experiment, working)
        changed = deepcopy(working)
        changed['reranked'][0]['demographic_feature'] = not changed['reranked'][0]['demographic_feature']
        with self.assertRaises(ValueError):
            assert_score_only(self.experiment, changed)
        for score in (0., -1.):
            self.model.score.side_effect = lambda pairs: [score] * len(pairs)
            result = self.run_variant('prefix_only')
            self.assertEqual(result['selected_ids'], [])
            self.assertEqual(result['evidence_snapshot']['medical_knowledge'], [])
            self.assertTrue(all(not d['positive_logit_eligible'] for r in result['candidate_audit'] for d in r['role_details'].values()))

    def test_baseline_against_both_frozen_references_and_strict_stop(self):
        result = self.run_variant('original')
        control = baseline_control(self.experiment, result, deepcopy(result))
        require_baseline(control)
        self.assertEqual(control['absolute_tolerance'], 1e-4)
        self.assertEqual(control['maximum_absolute_logit_drift'], 0.)
        self.assertEqual(control['task76_maximum_absolute_logit_drift'], 0.)
        self.model.score.side_effect = lambda pairs: [-1.] * len(pairs)
        drifted = self.run_variant('original')
        with self.assertRaisesRegex(ValueError, 'prefix scoring forbidden'):
            require_baseline(baseline_control(self.experiment, drifted, result))
        bad_reference = deepcopy(result)
        bad_reference['candidate_audit'][0]['role_details']['symptoms']['raw_reranker_score'] += .01
        with self.assertRaises(ValueError):
            require_baseline(baseline_control(self.experiment, result, bad_reference))

    def test_snapshot_and_provenance_unchanged_when_scores_unchanged(self):
        result = self.run_variant('prefix_only')
        self.assertEqual(result['evidence_snapshot'], self.experiment['evidence_snapshot'])
        snapshot = EvidenceSnapshot.model_validate_json(json.dumps(result['evidence_snapshot']))
        snapshot.check_integrity()
        self.assertEqual(snapshot.content_id(), snapshot.snapshot_id)
        self.assertEqual(fingerprint(self.cached['experiment']), fingerprint(self.experiment))

    def test_all_four_transitions_full_trace_and_important_title_absence(self):
        original = self.run_variant('original')
        self.model.score.side_effect = lambda pairs: [-1. if q.startswith(PREFIXES['symptoms']) else 3. for q, p in pairs]
        prefixed = self.run_variant('prefix_only')
        comparison = compare_pairs(original, prefixed, self.experiment)
        n = len(self.records)
        self.assertEqual(comparison['eligibility_transitions']['symptoms']['positive_to_nonpositive'], n)
        self.assertEqual(comparison['eligibility_transitions']['history']['positive_to_positive'], n)
        self.assertEqual(comparison['eligibility_transitions']['symptoms']['nonpositive_to_positive'], 0)
        for role in ('symptoms', 'history'):
            self.assertEqual(sum(comparison['eligibility_transitions'][role].values()), n)
        for pair in comparison['pairs']:
            for key in ('evidence_text_sha256', 'original_query', 'prefixed_query', 'dense_rank', 'bm25_rank',
                        'rrf_rank', 'rrf_score', 'supporting_fact_ids', 'original_raw_logit', 'prefix_raw_logit',
                        'original_reranker_rank', 'prefix_reranker_rank', 'original_removal_reason',
                        'prefix_removal_reason', 'prefix_selected_for_role'):
                self.assertIn(key, pair)
        self.assertTrue(all(not item['present'] for item in important_audit('ddxplus:validate:3', comparison).values()))
        # Test the other two directions without introducing alternate prefixes.
        changed_original = deepcopy(original)
        for row in changed_original['candidate_audit']:
            for role in ('symptoms', 'history'):
                row['role_details'][role]['raw_reranker_score'] = -1.
        counts = compare_pairs(changed_original, prefixed, self.experiment)['eligibility_transitions']
        self.assertEqual(counts['history']['nonpositive_to_positive'], n)
        self.assertEqual(counts['symptoms']['nonpositive_to_nonpositive'], n)

    def test_comparison_refuses_any_pair_assignment_or_input_change(self):
        a, b = self.run_variant('original'), self.run_variant('prefix_only')
        for key, value in (('originating_query_id', 'wrong'), ('reranker_input', 'new question?'),
                           ('passage_sha256', 'changed')):
            bad = deepcopy(b)
            bad['candidate_audit'][0]['role_details']['symptoms'][key] = value
            with self.assertRaises(ValueError):
                compare_pairs(a, bad, self.experiment)


class IntegrityAndRunnerTests(unittest.TestCase):
    def test_exception_is_exact_byte_pinned_not_a_blanket_config_exemption(self):
        live = Path(__file__).resolve().parents[1] / 'rag/config.py'
        current = live.read_bytes()
        self.assertEqual(hashlib.sha256(current).hexdigest(), AUTHORIZED_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.py'
            config.write_bytes(current)
            self.assertEqual(config_exception(config, HISTORICAL_CONFIG)['authorized_sha256'], AUTHORIZED_CONFIG)
            with self.assertRaises(ValueError):
                config_exception(config, 'wrong historical hash')
            config.write_bytes(current + b'\r\n')
            with self.assertRaises(ValueError):
                config_exception(config, HISTORICAL_CONFIG)

    def test_six_manifests_strict_other_paths_and_no_manifest_writes(self):
        live = Path(__file__).resolve().parents[1] / 'rag/config.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'rag/config.py'
            config.parent.mkdir()
            config.write_bytes(live.read_bytes())
            other = root / 'opaque.bin'
            other.write_bytes(b'opaque protected content')
            files = {str(config): HISTORICAL_CONFIG, str(other): digest(other)}
            manifests = []
            for directory in ('task7.5/frozen-ranking-ablation', 'task7.5/validated-ranking-ablation', 'task7.6/role-explicit-inputs'):
                folder = root / 'outputs' / directory
                folder.mkdir(parents=True)
                for when in ('before', 'after'):
                    m = folder / f'frozen-files-{when}.json'
                    m.write_text(json.dumps(files), encoding='utf8')
                    manifests.append(m)
            original = {str(m): digest(m) for m in manifests}
            before, report = verify_historical(root)
            self.assertEqual(report['authorized_exception_path_count'], 1)
            self.assertEqual(len(report['exceptions']), 6)
            self.assertEqual(before, verify_unchanged(before))
            self.assertEqual(original, {str(m): digest(m) for m in manifests})
            other.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Unexpected protected-file mismatch'):
                verify_historical(root)
            with self.assertRaisesRegex(ValueError, 'Protected file changed'):
                verify_unchanged(before)

    def test_original_synchronous_13_5_gb_stop_retained_and_reported(self):
        from scripts.experiment_reranker_prefix_control import RunMemoryMonitor
        from scripts.ablate_medical_ranking import MemorySafetyStop
        with tempfile.TemporaryDirectory() as tmp:
            monitor = RunMemoryMonitor(Path(tmp))
            monitor.psutil = Mock()
            monitor.psutil.virtual_memory.return_value = SimpleNamespace(used=13_500_000_000)
            monitor.process = Mock()
            monitor.process.memory_info.return_value = SimpleNamespace(rss=100_000_000)
            with self.assertRaises(MemorySafetyStop):
                monitor.begin('synthetic-memory-stop')
            report = json.loads((Path(tmp) / 'memory-stop.json').read_text())
            self.assertEqual(report['system_gb'], 13.5)
            self.assertEqual(report['phase'], 'synthetic-memory-stop')

    def test_cli_has_no_experiment_tuning_or_case_parallelism_switches(self):
        from scripts.experiment_reranker_prefix_control import arguments
        for flag in ('--batch-size', '--prefix', '--samples', '--workers', '--stop-gb'):
            with self.assertRaises(SystemExit):
                arguments(['--output', 'new', '--authorized-preflight', 'prior', flag, '4'])


if __name__ == '__main__':
    unittest.main()

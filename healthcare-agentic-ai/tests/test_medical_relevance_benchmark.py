"""Offline benchmark/metric arithmetic tests. Synthetic judgments are not review."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.reviewed_retrieval import (Benchmark, PATIENT_ID, TOPICS, build_benchmark,
                                           pool_digest, summarize)
from rag.focused_medical import query_plan, state_from_patient
from tests.test_focused_medical import hit
from tests.test_patient_rag import record


def fixture():
    state = state_from_patient(record().patient, record().patient_id).model_copy(update={'patient_id': PATIENT_ID})
    plan = query_plan(state)
    hits = [hit('Fixture ' + str(i), 'Synthetic medical test text ' + str(i), .99 - i / 100) for i in range(55)]
    hits += [hit(title, 'Synthetic topic chunk ' + title + ' ' + str(i), .5) for title in TOPICS for i in range(2)]
    index = [{k: v for k, v in h.items() if k != 'score'} for h in hits]
    report = {'created_at': '2026-09-23T00:00:00Z', 'labels_loaded': False, 'cloud_calls': 0,
              'live_workflow_changed': False, 'medical_chunk_count': len(index),
              'embedding_signature': {'test_fixture': True}, 'manifest_sha256': '0' * 64, 'experiments': []}
    for variant, rows, query in [('live_focused_baseline', hits[:50], plan['query']),
                                 ('symptom_only', hits[5:55], '; '.join(plan['symptom_concepts']) + ' symptoms clinical information')]:
        report['experiments'].append({
            'experiment_version': 'medical-depth-experiment-v1', 'patient_id': PATIENT_ID,
            'query_variant': variant, 'plan': {'query': query},
            'depths': [{'top_k': 50, 'returned_count': 50,
                        'candidates': [{**h, 'rank': r} for r, h in enumerate(rows, 1)],
                        'experimental_top5': [{'hit': hits[i], 'lexical_score': 10 - r}
                                             for r, i in enumerate((19, 0, 40, 49, 9))]}]})
    benchmark = build_benchmark(report, index, state, report_name='synthetic-test.json', report_sha256='1' * 64)
    return benchmark, report, index, state, hits


def reviewed_benchmark(benchmark, judgments, *, all_negative=False):
    data = benchmark.model_dump()
    for candidate in data['candidates']:
        label = judgments.get(candidate['chunk_id'], 'not_relevant' if all_negative else None)
        if label is not None:
            candidate.update(relevance=label, reviewed=True, reviewer_id='unit-test-simulated-reviewer',
                             reviewer_notes='Synthetic arithmetic fixture, not actual clinical review.')
    return Benchmark.model_validate(data)


class MedicalRelevanceBenchmarkTests(unittest.TestCase):
    def test_pool_deduplicates_all_sources_and_preserves_full_chunks(self):
        benchmark, _, index, _, hits = fixture()
        self.assertEqual(len(benchmark.candidates), 61)
        actual = {c.chunk_id: c for c in benchmark.candidates}
        for chunk in index:
            candidate = actual[chunk['chunk_id']]
            self.assertEqual(candidate.text, chunk['text'])
            self.assertEqual(candidate.title, chunk['title'])
            self.assertEqual(candidate.source, chunk['source'])
        candidate = actual[hits[19]['chunk_id']]
        self.assertEqual(candidate.rankings['current_query'].original_bge_rank, 20)
        self.assertEqual(candidate.rankings['symptom_only_query'].original_bge_rank, 15)
        self.assertEqual(candidate.rankings['lexical_reranker'].lexical_reranker_rank, 1)
        self.assertEqual(candidate.rankings['lexical_reranker'].cosine_similarity, hits[19]['score'])
        for title in TOPICS:
            self.assertEqual(len(benchmark.requested_topics[title]), 2)
            for identifier in benchmark.requested_topics[title]:
                self.assertTrue(all(r is None for r in actual[identifier].rankings.values()))

    def test_no_score_or_requested_title_automatically_assigns_relevance(self):
        benchmark, *_ = fixture()
        self.assertTrue(all(c.relevance == 'uncertain' and not c.reviewed and c.reviewer_id is None
                            for c in benchmark.candidates))
        summary = summarize(benchmark)
        self.assertIsNone(summary['derived_retrieval_metrics'])
        self.assertEqual(summary['metric_status'], 'withheld_no_manual_review')
        self.assertEqual(summary['manual_review']['judgment_counts']['uncertain'], 0)
        self.assertEqual(summary['manual_review']['unreviewed_count'], 61)
        self.assertIsNone(summary['clinical_diagnostic_conclusions'])

    def test_hand_calculated_recall_and_mrr_for_all_three_systems(self):
        benchmark, _, _, _, hits = fixture()
        judgments = {hits[i]['chunk_id']: 'relevant' for i in (0, 9, 49, 55)}
        judgments[hits[19]['chunk_id']] = 'partially_relevant'
        reviewed = reviewed_benchmark(benchmark, judgments, all_negative=True)
        summary = summarize(reviewed)
        self.assertEqual(summary['metric_status'], 'complete_pool_review')
        self.assertEqual(summary['manual_review']['judgment_counts'],
                         {'relevant': 4, 'partially_relevant': 1, 'not_relevant': 56, 'uncertain': 0})
        strict = summary['derived_retrieval_metrics']['strict_relevant']
        self.assertEqual(strict['positive_pool_count'], 4)  # Includes requested-topic miss.
        systems = strict['systems']
        self.assertEqual(systems['current_query']['recall_at_k'], {'5': .25, '10': .5, '20': .5, '50': .75})
        self.assertEqual(systems['current_query']['mrr'], 1.0)
        self.assertEqual(systems['symptom_only_query']['recall_at_k'], {'5': .25, '10': .25, '20': .25, '50': .5})
        self.assertEqual(systems['symptom_only_query']['mrr'], .2)
        lexical = systems['lexical_reranker']
        self.assertEqual(lexical['recall_at_k'], {'5': .75, '10': .75, '20': .75, '50': .75})
        self.assertEqual(lexical['effective_cutoffs'], {'5': 5, '10': 5, '20': 5, '50': 5})
        self.assertEqual(lexical['mrr'], .5)
        inclusive = summary['derived_retrieval_metrics']['inclusive_relevant_or_partial']
        self.assertEqual(inclusive['positive_pool_count'], 5)
        self.assertEqual(inclusive['systems']['current_query']['recall_at_k'], {'5': .2, '10': .4, '20': .6, '50': .8})
        self.assertEqual(inclusive['systems']['lexical_reranker']['mrr'], 1.0)

    def test_partial_review_is_provisional_and_original_ranks_are_not_compressed(self):
        benchmark, _, _, _, hits = fixture()
        reviewed = reviewed_benchmark(benchmark, {hits[9]['chunk_id']: 'relevant'})
        summary = summarize(reviewed)
        self.assertEqual(summary['metric_status'], 'provisional_incomplete_pool_review')
        self.assertEqual(summary['manual_review']['unreviewed_count'], 60)
        systems = summary['derived_retrieval_metrics']['strict_relevant']['systems']
        self.assertEqual(systems['current_query']['mrr'], .1)
        self.assertEqual(systems['current_query']['recall_at_k']['5'], 0.0)
        self.assertEqual(systems['current_query']['recall_at_k']['10'], 1.0)
        self.assertEqual(systems['symptom_only_query']['mrr'], .2)

    def test_reviewed_uncertain_is_not_unreviewed_or_negative(self):
        benchmark, _, _, _, hits = fixture()
        summary = summarize(reviewed_benchmark(benchmark, {hits[0]['chunk_id']: 'uncertain'}))
        self.assertEqual(summary['manual_review']['reviewed_count'], 1)
        self.assertEqual(summary['manual_review']['judgment_counts']['uncertain'], 1)
        for mode in summary['derived_retrieval_metrics'].values():
            for system in mode['systems'].values():
                self.assertIsNone(system['mrr'])
                self.assertTrue(all(v is None for v in system['recall_at_k'].values()))

    def test_no_positive_judgments_withholds_undefined_denominators(self):
        benchmark, *_ = fixture()
        summary = summarize(reviewed_benchmark(benchmark, {}, all_negative=True))
        self.assertEqual(summary['metric_status'], 'complete_pool_review')
        self.assertEqual(summary['derived_retrieval_metrics']['strict_relevant']['positive_pool_count'], 0)
        for system in summary['derived_retrieval_metrics']['strict_relevant']['systems'].values():
            self.assertEqual(system['status'], 'withheld_no_positive_judgments')
            self.assertIsNone(system['mrr'])
            self.assertTrue(all(v is None for v in system['recall_at_k'].values()))

    def test_known_positive_not_returned_yields_zero_not_missing(self):
        benchmark, _, _, _, hits = fixture()
        summary = summarize(reviewed_benchmark(benchmark, {hits[55]['chunk_id']: 'relevant'}))
        for system in summary['derived_retrieval_metrics']['strict_relevant']['systems'].values():
            self.assertEqual(system['mrr'], 0.0)
            self.assertEqual(set(system['recall_at_k'].values()), {0.0})

    def test_partially_relevant_is_secondary_not_strict_positive(self):
        benchmark, _, _, _, hits = fixture()
        summary = summarize(reviewed_benchmark(benchmark, {hits[0]['chunk_id']: 'partially_relevant'}))
        modes = summary['derived_retrieval_metrics']
        self.assertIsNone(modes['strict_relevant']['systems']['current_query']['mrr'])
        self.assertEqual(modes['inclusive_relevant_or_partial']['systems']['current_query']['mrr'], 1.0)

    def test_review_edit_requires_manual_confirmation_and_identity(self):
        benchmark, *_ = fixture()
        for updates in ({'relevance': 'relevant'}, {'reviewed': True},
                        {'reviewed': True, 'reviewer_id': ' '}, {'relevance': 'probably_relevant'}):
            data = benchmark.model_dump()
            data['candidates'][0].update(updates)
            with self.assertRaises(ValueError):
                Benchmark.model_validate(data)

    def test_review_notes_are_editable_but_rank_text_patient_and_pool_are_locked(self):
        benchmark, *_ = fixture()
        data = benchmark.model_dump()
        data['candidates'][0]['reviewer_notes'] = 'Awaiting review, no judgment yet.'
        self.assertEqual(Benchmark.model_validate(data).pool_sha256, benchmark.pool_sha256)
        del data['candidates'][0]['reviewer_notes']
        self.assertIsNone(Benchmark.model_validate(data).candidates[0].reviewer_notes)
        for mutate in (lambda d: d['patient_state'].update(age=99),
                       lambda d: d['candidates'][0].update(text='Altered'),
                       lambda d: d['candidates'].pop()):
            data = benchmark.model_dump()
            mutate(data)
            with self.assertRaises(ValueError):
                Benchmark.model_validate(data)

    def test_reject_invalid_rankings_even_if_digest_is_recomputed(self):
        benchmark, *_ = fixture()
        for kind in ('duplicate', 'zero', 'boolean', 'nan', 'lexical_parent', 'origin'):
            data = benchmark.model_dump()
            ranked = [c for c in data['candidates'] if c['rankings']['current_query'] is not None]
            rank = ranked[0]['rankings']['current_query']
            if kind == 'duplicate':
                rank['original_bge_rank'] = ranked[1]['rankings']['current_query']['original_bge_rank']
            elif kind == 'zero':
                rank['original_bge_rank'] = 0
            elif kind == 'boolean':
                rank['original_bge_rank'] = True
            elif kind == 'nan':
                rank['cosine_similarity'] = float('nan')
            elif kind == 'origin':
                ranked[0]['origins'] = ['invented']
            else:
                c = next(c for c in ranked if c['rankings']['lexical_reranker'] is not None)
                c['rankings']['lexical_reranker']['cosine_similarity'] = -.2
            with self.assertRaises(ValueError):
                data['pool_sha256'] = pool_digest(data)
                Benchmark.model_validate(data)

    def test_builder_rejects_stale_altered_or_duplicate_index_payloads(self):
        _, report, index, state, _ = fixture()
        for variant in ('text', 'duplicate', 'count'):
            items = copy.deepcopy(index)
            if variant == 'text':
                items[0]['text'] = 'Tampered'
            elif variant == 'duplicate':
                items.append(items[0])
            else:
                items.pop()
            with self.assertRaises(ValueError):
                build_benchmark(report, items, state, report_name='test', report_sha256='0' * 64)

    def test_builder_rejects_wrong_case_label_flag_query_or_missing_top50(self):
        _, report, index, state, _ = fixture()
        for kind in ('case', 'label', 'query', 'missing_top50', 'duplicate_experiment', 'duplicate_rank'):
            data = copy.deepcopy(report)
            if kind == 'case':
                data['experiments'][0]['patient_id'] = 'ddxplus:validate:1'
            elif kind == 'label':
                data['labels_loaded'] = True
            elif kind == 'query':
                data['experiments'][0]['plan']['query'] = 'label-derived replacement'
            elif kind == 'missing_top50':
                data['experiments'][0]['depths'][0]['top_k'] = 20
            elif kind == 'duplicate_experiment':
                data['experiments'].append(data['experiments'][0])
            else:
                data['experiments'][0]['depths'][0]['candidates'][0]['rank'] = 2
            with self.assertRaises(ValueError):
                build_benchmark(data, index, state, report_name='test', report_sha256='0' * 64)

    def test_missing_requested_topic_is_empty_not_fabricated(self):
        _, report, index, state, _ = fixture()
        items = [c for c in index if c['title'] != 'Panic Disorder']
        report['medical_chunk_count'] = len(items)
        benchmark = build_benchmark(report, items, state, report_name='test', report_sha256='0' * 64)
        self.assertEqual(benchmark.requested_topics['Panic Disorder'], [])
        self.assertFalse(any(c.title == 'Panic Disorder' for c in benchmark.candidates))

    def test_summary_cli_only_reads_json_and_never_overwrites_reviews(self):
        from scripts.summarize_medical_relevance import main
        benchmark, *_ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'benchmark.json'
            path.write_text(benchmark.model_dump_json(), encoding='utf8')
            output = Path(directory) / 'summary.json'
            before = path.read_bytes()
            # Summarizing has no retrieval/store dependency at all.
            with patch('socket.socket', side_effect=AssertionError('Network forbidden')), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--input', str(path), '--output', str(output)]), 0)
            self.assertIsNone(json.loads(output.read_text())['derived_retrieval_metrics'])
            self.assertEqual(path.read_bytes(), before)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--input', str(path), '--output', str(path)])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--input', str(path), '--output', str(output)])

    def test_summary_cli_invalid_json_does_not_emit_metrics(self):
        from scripts.summarize_medical_relevance import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            path.write_text('{}')
            output = Path(directory) / 'summary.json'
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--input', str(path), '--output', str(output)])
            self.assertFalse(output.exists())

    def test_checked_in_asset_is_valid_case2_pool_without_diagnosis_labels(self):
        path = Path(__file__).resolve().parents[1] / 'evaluation/medical_relevance.json'
        data = Benchmark.model_validate_json(path.read_text(encoding='utf8'))
        self.assertEqual(data.patient_state.patient_id, PATIENT_ID)
        self.assertEqual(data.systems['current_query'].returned_count, 50)
        self.assertEqual(data.systems['symptom_only_query'].returned_count, 50)
        self.assertEqual(data.systems['lexical_reranker'].returned_count, 5)
        for title in TOPICS:
            self.assertTrue(data.requested_topics[title])
        self.assertNotIn('PATHOLOGY', data.patient_state.model_dump())
        self.assertNotIn('DIFFERENTIAL_DIAGNOSIS', data.patient_state.model_dump())
        self.assertFalse(data.source_report['labels_loaded'])


if __name__ == '__main__':
    unittest.main()

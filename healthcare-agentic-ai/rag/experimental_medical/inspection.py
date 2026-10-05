"""Stage diagnostics only: topic probes are NOT relevance/diagnosis labels."""
from pathlib import Path
import hashlib

def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def directory_hashes(path):
    path = Path(path)
    return {str(p.relative_to(path)): file_digest(p) for p in sorted(path.rglob('*')) if p.is_file()}


def topic_trace(topic, records, baseline, experiment):
    matches = [r for r in records if topic.casefold() == r['source']['title'].casefold()]
    ids = {r['chunk_id'] for r in matches}
    def positions(rows, key='chunk_id'):
        return [i for i, r in enumerate(rows, 1) if r[key] in ids]
    baseline_matches = [r for r in baseline['broad_candidates'] if r['title'].casefold() == topic.casefold()]
    channel = [{'query_id': c['query_id'], 'channel': c['channel'], 'ranks': positions(c['hits'])}
               for c in experiment['channel_rankings'] if positions(c['hits'])]
    pool = positions(experiment['candidate_pool'])
    reranked = positions(experiment['reranked'])
    final = positions(experiment['final_passages'])
    if not matches:
        stage = 'E_unconfirmed_no_exact_topic_title; body coverage not established'
    elif not channel:
        stage = 'A_absent_from_all_candidate_channels'
    elif not pool:
        stage = 'A_lost_at_fusion_budget_or_parent_deduplication'
    elif not final:
        stage = 'C_reranking_or_final_diversity_budget'
    else:
        stage = 'surfaced_in_final_preview; not_diagnosis_or_relevance_proof'
    return {'topic': topic, 'catalog_windows': len(matches),
            'baseline_dense_ranks': [r['rank'] for r in baseline_matches],
            'baseline_top5_retained': [r['chunk_id'] for r in baseline_matches if r['rank'] <= 5 and r['filter']['retained']],
            'baseline_B_filter_rejections': [r['rank'] for r in baseline_matches if not r['filter']['retained']],
            'experimental_channels': channel, 'fused_pool_ranks': pool, 'reranked_ranks': reranked,
            'final_positions': final, 'failure_stage_or_outcome': stage,
            'D_context': 'source_passage_attached; inspect_sentence_boundaries' if final else 'inspect_catalog_and_candidate_text',
            'limitations': 'Exact topic inventory/positions only, not automated clinical relevance or a proven corpus gap.'}


def markdown_report(report):
    lines = ['# Task 7.4 local retrieval inspection', '',
        'Synthetic label-free findings; no diagnosis/relevance labels, no clinical validation.',
        'Qdrant passages are validated through existing immutable evidence snapshots. No diagnostic agents executed.', '',
        '## Configuration', '```json', __import__('json').dumps(report['configuration'], indent=2), '```', '']
    for case in report['cases']:
        state = case['patient_state']
        lines += [f"## {state['patient_id']}", f"Age: {state['age']}; sex: {state['sex']}", '', '### Observed findings']
        for field in ('presenting_evidence', 'symptoms', 'antecedents'):
            lines += [f'**{field}**'] + ['- ' + t for t in state[field]]
        baseline = case['baseline']
        lines += ['', '### Baseline top five', 'Query: ' + baseline['query'], '']
        for row in baseline['top5']:
            lines += [f"{row['rank']}. **{row['title']}** — {'retain' if row['filter']['retained'] else 'discard'} ({row['filter']['reason']})",
                      '', row['text'], '']
        for experiment in case['experiments']:
            lines += [f"### Experimental depth {experiment['candidate_depth']}", '', '**Queries**']
            lines += [f"- {q['query_id']}: {q['text']}" for q in experiment['plan']['queries']]
            lines += ['', '**Timing (seconds)**', str(experiment['timings']), '', '**Recovered document titles relative to baseline final evidence**']
            lines += ['- ' + t for t in experiment['new_final_titles_vs_baseline']]
            lines += ['', '**Topic stage diagnostics (inventory/ranks, not relevance)**']
            for trace in experiment['topic_traces']:
                lines += [f"- {trace['topic']}: {trace['failure_stage_or_outcome']}; pool={trace['fused_pool_ranks']}; final={trace['final_positions']}"]
            lines += ['', '**Final passages / proposed agent evidence**', '']
            for i, hit in enumerate(experiment['final_passages'], 1):
                r = hit['record']
                lines += [f"#### {i}. {r['source']['title']} — {r['section']}",
                          f"Role: {hit['selection_role']}; window ID: `{r['chunk_id']}`; parent: `{r['parent_id']}`",
                          f"Source: {r['source']['url']}", f"Reranker: {hit['reranker_score']}; RRF: {hit['rrf_score']}; methods: {sorted({o['channel'] for o in hit['origins']})}", '', '**Retrieval window**', r['window_text'], '',
                          '**Source evidence text (unchanged)**', r['text'], '']
    lines += ['## Interpretation boundary',
        'Inspect mismatched populations, missing triggers, management-only passages, and narrow single-symptom matches.',
        'No automated correct/incorrect labels were assigned. See the implementation report for qualitative observations.',
        'A missing title does not prove a corpus gap. Retrieved background never establishes a diagnosis.']
    return '\n'.join(lines) + '\n'

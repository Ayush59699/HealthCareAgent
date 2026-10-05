"""Task 7.7: fixed role prefix only, on authenticated frozen depth-50 pools.

No query generation, fusion, retrieval, labels, model replacement, or live imports.
Task 7.6 score extraction/ranking and Task 7.4 selection/contracts are reused.
"""
import hashlib
import json
from time import perf_counter

from orchestration.evidence import EvidenceSnapshot, fingerprint
from rag.agents.models import PatientState
from rag.medical_ingestion.models import validate_chunk
from .ranking_ablation import ROLES, ablate
from .reranker_input_experiment import (
    pair_plan as original_pair_plan, check_token_budget, score_pairs, apply_scores,
    baseline_drift)
from .retrieval import RERANKER_MODEL, RERANKER_REVISION, VERSION as RETRIEVAL_VERSION

VERSION = 'task77-prefix-only-control-v1'
FORMATS = ('original', 'prefix_only')
PREFIXES = {'symptoms': 'Current findings: ', 'history': 'Historical/background: '}
EXPECTED_POOLS = {'validate:1': 593, 'validate:2': 416, 'validate:3': 425}
IMPORTANT_TITLES = {
    'validate:1': ('Anemia', 'Chronic Kidney Disease', 'Fainting'),
    'validate:2': ('Asthma in Children', 'Panic Disorder', 'Depression', 'Anxiety', 'Teen Depression'),
    'validate:3': ('Flu', 'Older Adult Mental Health'),
}
SCORE_FIELDS = {'symptoms_score', 'history_score', 'symptoms_ranking_score',
                'history_ranking_score', 'reranker_score', 'reranked_rank'}


def text_sha256(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def prefix_query(original, role):
    if type(original) is not str or not original or role not in PREFIXES:
        raise ValueError('A frozen nonempty query and one of the two fixed roles required')
    result = PREFIXES[role] + original
    if result[len(PREFIXES[role]):].encode('utf8') != original.encode('utf8'):
        raise ValueError('Original query bytes changed')
    return result


def pair_plan(state, experiment, records, variant):
    if variant not in FORMATS:
        raise ValueError('Only original and the fixed prefix-only control are allowed')
    # This is the unchanged ORIGINAL-input pair planner, never the Task 7.6 rewrite.
    pairs, audit, formatted = original_pair_plan(state, experiment, records, 'original')
    by_id = {r['chunk_id']: records[r['index']] for r in experiment['reranked']}
    for (query, passage), pair in zip(pairs, audit):
        record = by_id[pair['chunk_id']]
        if record['chunk_id'] != pair['chunk_id'] or passage != record['retrieval_text']:
            raise ValueError('Candidate/evidence assignment changed')
        pair['original_query_sha256'] = text_sha256(query)
        pair['evidence_text_sha256'] = text_sha256(record['evidence_text'])
        pair['retrieval_text_sha256'] = text_sha256(passage)
    if variant == 'prefix_only':
        pairs = [(prefix_query(q, pair['role']), passage)
                 for (q, passage), pair in zip(pairs, audit)]
        for pair, (query, _) in zip(audit, pairs):
            pair['reranker_input'] = query
        for query in formatted.values():
            query['reranker_input'] = prefix_query(query['retrieval_query'], query['role'])
    return pairs, audit, formatted


def validate_frozen_case(state, cached, source_payloads, sample_memory=lambda: None,
                         enforce_case_size=True):
    """Verify authenticated stored data without regenerating queries or RRF.

    File hashes must have passed before this is called. Current source payloads
    are streamed for only the frozen IDs. No search/index API is involved.
    """
    if type(state) is not PatientState or state.model_dump() != cached['patient_state']:
        raise ValueError('Frozen state differs from label-free parser state')
    experiment = cached['experiment']
    if (experiment['version'] != RETRIEVAL_VERSION
            or type(experiment['candidate_depth']) is not int or experiment['candidate_depth'] != 50
            or experiment['final_limit'] != 5 or experiment['pre_reranking_lexical_filter'] is not False):
        raise ValueError('Frozen retrieval configuration changed')
    signature = experiment['reranker']
    if (signature['model'] != RERANKER_MODEL or signature['revision'] != RERANKER_REVISION
            or signature['max_length'] != 512):
        raise ValueError('Frozen reranker identity changed')
    pool, rows = experiment['candidate_pool'], experiment['reranked']
    if enforce_case_size and len(pool) != EXPECTED_POOLS.get(state.patient_id.removeprefix('ddxplus:')):
        raise ValueError('Frozen candidate count/case changed')
    if pool != experiment['fused_candidates'] or experiment['deduplication_removed']:
        raise ValueError('Stored candidate pool/fusion mismatch')
    ids = [row['chunk_id'] for row in pool]
    if len(set(ids)) != len(ids) or {r['chunk_id'] for r in rows} != set(ids) or len(rows) != len(pool):
        raise ValueError('Frozen candidate IDs changed')
    by_record = {r['chunk_id']: r for r in experiment['candidate_records']}
    if len(by_record) != len(experiment['candidate_records']) or set(by_record) != set(ids):
        raise ValueError('Frozen candidate records changed')
    records = {}
    for rank, row in enumerate(rows, 1):
        sample_memory()
        record = by_record[row['chunk_id']]
        payload = validate_chunk(record['evidence_payload'])
        if payload != source_payloads.get(row['chunk_id']):
            raise ValueError('Frozen source payload changed')
        if any(record[key] != payload['text'] for key in ('text', 'evidence_text', 'window_text')):
            raise ValueError('Frozen evidence text changed')
        if record['source'] != {k: v for k, v in payload.items()
                                if k not in ('text', 'chunk_id', 'chunk_index', 'chunking_version')}:
            raise ValueError('Frozen provenance changed')
        header = payload['title'] + ('\n' + payload['section'] if payload['section'] else '')
        if (record['retrieval_text'] != header + '\n' + payload['text']
                or record['parent_id'] != payload['chunk_id'] or record['section'] != payload['section']):
            raise ValueError('Frozen passage context changed')
        if type(row['index']) is not int or row['index'] < 0 or row['index'] in records:
            raise ValueError('Frozen record index changed')
        records[row['index']] = record
        if row['reranked_rank'] != rank or not 1 <= row['fusion_rank'] <= len(pool):
            raise ValueError('Frozen rank changed')
        fused = pool[row['fusion_rank'] - 1]
        if any(row[k] != fused[k] for k in ('index', 'chunk_id', 'rrf_score', 'origins')):
            raise ValueError('Frozen fusion provenance changed')
    pairs, audit, _ = pair_plan(state, experiment, records, 'original')
    if len(pairs) != experiment['reranker_pair_count']:
        raise ValueError('Frozen pair count changed')
    # Replay only the unchanged selector, not query construction or RRF generation.
    replay = ablate(state, experiment, records, 'task74', sample_memory)
    if (replay['selected_ids'] != [r['chunk_id'] for r in experiment['final_passages']]
            or replay['evidence_snapshot'] != experiment['evidence_snapshot']):
        raise ValueError('Frozen original selection/snapshot failed to reproduce')
    return records


def assert_score_only(experiment, working):
    if set(experiment) != set(working):
        raise ValueError('Experiment fields changed')
    if any(experiment[k] != working[k] for k in experiment if k != 'reranked'):
        raise ValueError('Non-neural experiment fields changed')
    old = {r['chunk_id']: r for r in experiment['reranked']}
    if len(working['reranked']) != len(old) or {r['chunk_id'] for r in working['reranked']} != set(old):
        raise ValueError('Candidate IDs changed during scoring')
    for row in working['reranked']:
        if {k: v for k, v in row.items() if k not in SCORE_FIELDS} != {
                k: v for k, v in old[row['chunk_id']].items() if k not in SCORE_FIELDS}:
            raise ValueError('A non-neural candidate field changed')


def run_format(state, experiment, records, reranker, variant,
               sample_memory=lambda: None, progress=lambda done, total: None):
    if reranker.signature != experiment['reranker'] or reranker.batch_size != 2:
        raise ValueError('Reranker identity/batch size changed')
    frozen_hash = fingerprint(experiment)
    started = perf_counter()
    pairs, audit, formatted = pair_plan(state, experiment, records, variant)
    lengths = check_token_budget(reranker, pairs, audit, sample_memory)
    prepared = perf_counter()
    scores = score_pairs(reranker, pairs, sample_memory, progress)
    scored_at = perf_counter()
    working = apply_scores(experiment, audit, scores)
    assert_score_only(experiment, working)
    result = ablate(state, working, records, 'task74', sample_memory)
    inputs = {(p['chunk_id'], p['role']): (p, n) for p, n in zip(audit, lengths)}
    for row in result['candidate_audit']:
        row['evidence_text_sha256'] = text_sha256(row['evidence_text'])
        for role, details in row['role_details'].items():
            pair, length = inputs[(row['chunk_id'], role)]
            details.update(reranker_input=pair['reranker_input'], pair_tokens=length,
                frozen_role_logit=pair['frozen_role_logit'],
                frozen_role_eligible=pair['frozen_role_logit'] > 0,
                logit_delta_from_frozen=details['raw_reranker_score'] - pair['frozen_role_logit'],
                passage_sha256=pair['passage_sha256'], retrieval_text_sha256=pair['retrieval_text_sha256'],
                original_query_sha256=pair['original_query_sha256'])
    if fingerprint(experiment) != frozen_hash:
        raise ValueError('Scoring mutated frozen inputs')
    EvidenceSnapshot.model_validate_json(json.dumps(result['evidence_snapshot']))
    result.update(version=VERSION, variant=variant,
        description='Fixed role prefix only; unchanged scoring formula, positive gate and selector.',
        original_scores_fingerprint=fingerprint(experiment['reranked']),
        rescored_rows_fingerprint=result.pop('frozen_scores_fingerprint'),
        reranker_signature=reranker.signature, formatted_queries=list(formatted.values()),
        pair_count=len(pairs), pair_order_fingerprint=fingerprint([
            (p['chunk_id'], p['role'], p['query_id'], p['passage_sha256']) for p in audit]),
        max_pair_tokens=max(lengths, default=0), truncated_pairs=0,
        timings={'format_and_token_preflight_seconds': prepared-started,
                 'cross_encoder_seconds': scored_at-prepared,
                 'selection_and_audit_seconds': perf_counter()-scored_at,
                 'total_seconds': perf_counter()-started})
    return result


def baseline_control(experiment, fresh, task76_original):
    control = baseline_drift(experiment, fresh)  # existing 1e-4 tolerance, unchanged
    def scores(result):
        return {(r['chunk_id'], role): d['raw_reranker_score']
                for r in result['candidate_audit'] for role, d in r['role_details'].items()}
    a, b = scores(fresh), scores(task76_original)
    same_pairs = set(a) == set(b)
    drift = max((abs(a[k] - b[k]) for k in a), default=0.) if same_pairs else None
    flips = sum((a[k] > 0) != (b[k] > 0) for k in a) if same_pairs else None
    same_ids = fresh['selected_ids'] == task76_original['selected_ids'] == [
        r['chunk_id'] for r in experiment['final_passages']]
    same_snapshot = fresh['evidence_snapshot'] == task76_original['evidence_snapshot']
    control.update(task76_maximum_absolute_logit_drift=drift, task76_eligibility_changes=flips,
        task76_same_pairs=same_pairs, task76_same_snapshot=same_snapshot,
        same_final_evidence_ids=same_ids, final_evidence_ids=fresh['selected_ids'],
        evidence_snapshot_hash=fresh['evidence_snapshot']['snapshot_id'])
    control['passed'] = (control['passed'] and same_pairs and same_ids and same_snapshot
                         and drift <= control['absolute_tolerance'] and flips == 0)
    return control


def require_baseline(control):
    if not control['passed']:
        raise ValueError('Fresh original-input baseline failed; prefix scoring forbidden')


def compare_pairs(original, prefixed, experiment):
    for key in ('pair_order_fingerprint', 'frozen_pool_fingerprint', 'original_scores_fingerprint'):
        if original[key] != prefixed[key]:
            raise ValueError('Not the same frozen pool/pairs/scores')
    a = {r['chunk_id']: r for r in original['candidate_audit']}
    b = {r['chunk_id']: r for r in prefixed['candidate_audit']}
    if set(a) != set(b):
        raise ValueError('Comparison candidate IDs changed')
    transitions = {role: {name: 0 for name in ('nonpositive_to_positive', 'positive_to_nonpositive',
                    'positive_to_positive', 'nonpositive_to_nonpositive')} for role in ROLES}
    role_ranks = []
    for result in (original, prefixed):
        ranks = {}
        for role in ROLES:
            eligible = [r for r in result['candidate_audit'] if role in r['role_details']]
            ordered = sorted(eligible, key=lambda r: (-r['role_details'][role]['task74_ranking_score'],
                                                      -r['rrf_score'], r['chunk_id']))
            ranks.update({(r['chunk_id'], role): rank for rank, r in enumerate(ordered, 1)})
        role_ranks.append(ranks)
    changes = []
    for candidate in experiment['candidate_pool']:
        cid = candidate['chunk_id']
        old, new = a[cid], b[cid]
        for key in ('source', 'title', 'section', 'url', 'document_id', 'evidence_text',
                    'evidence_text_sha256', 'rrf_score', 'fusion_rank', 'all_channel_origins',
                    'population_features', 'task74_population_feature'):
            if old[key] != new[key]:
                raise ValueError(f'Non-neural audit field changed: {key}')
        if set(old['role_details']) != set(new['role_details']):
            raise ValueError('Pair roles changed')
        for role in ROLES:
            if role not in old['role_details']:
                continue
            x, y = old['role_details'][role], new['role_details'][role]
            for key in ('originating_query_id', 'originating_query', 'fact_ids', 'dense_rank',
                        'bm25_rank', 'passage_sha256', 'original_query_sha256', 'concept_coverage'):
                if x[key] != y[key]:
                    raise ValueError(f'Frozen query assignment/input changed: {key}')
            if y['reranker_input'] != prefix_query(x['reranker_input'], role):
                raise ValueError('Experimental query is not exact prefix plus original')
            positive_a, positive_b = x['raw_reranker_score'] > 0, y['raw_reranker_score'] > 0
            direction = ('positive' if positive_a else 'nonpositive') + '_to_' + ('positive' if positive_b else 'nonpositive')
            transitions[role][direction] += 1
            changes.append({'chunk_id': cid, 'title': old['title'], 'section': old['section'],
                'source': old['source'], 'url': old['url'], 'document_id': old['document_id'],
                'evidence_text_sha256': old['evidence_text_sha256'], 'passage_sha256': x['passage_sha256'],
                'original_query': x['reranker_input'], 'prefixed_query': y['reranker_input'],
                'original_query_sha256': x['original_query_sha256'], 'query_id': x['originating_query_id'],
                'role': role, 'supporting_fact_ids': x['fact_ids'],
                'dense_rank': x['dense_rank'], 'bm25_rank': x['bm25_rank'],
                'rrf_rank': old['fusion_rank'], 'rrf_score': old['rrf_score'],
                'original_raw_logit': x['raw_reranker_score'], 'prefix_raw_logit': y['raw_reranker_score'],
                'logit_delta': y['raw_reranker_score'] - x['raw_reranker_score'],
                'original_eligible': positive_a, 'prefix_eligible': positive_b,
                'eligibility_transition': direction,
                'original_reranker_rank': old['reranked_rank'], 'prefix_reranker_rank': new['reranked_rank'],
                'original_role_ranking_rank': role_ranks[0][cid, role],
                'prefix_role_ranking_rank': role_ranks[1][cid, role],
                'original_role_ranking_score': x['task74_ranking_score'],
                'prefix_role_ranking_score': y['task74_ranking_score'],
                'original_removal_reason': old['final_selection_reason'],
                'prefix_removal_reason': new['final_selection_reason'],
                'original_role_events': [e for e in old['selector_events'] if e['role'] == role],
                'prefix_role_events': [e for e in new['selector_events'] if e['role'] == role],
                'original_selected': old['selected'], 'prefix_selected': new['selected'],
                'original_selected_role': old['selected_role'], 'prefix_selected_role': new['selected_role'],
                'original_selected_for_role': old['selected_role'] == role,
                'prefix_selected_for_role': new['selected_role'] == role})
    if len(changes) != experiment['reranker_pair_count']:
        raise ValueError('Incomplete pair comparison')
    return {'eligibility_transitions': transitions, 'pair_count': len(changes), 'pairs': changes,
            'selected_gained': sorted(set(prefixed['selected_ids']) - set(original['selected_ids'])),
            'selected_lost': sorted(set(original['selected_ids']) - set(prefixed['selected_ids']))}


def important_audit(patient_id, comparison):
    titles = IMPORTANT_TITLES[patient_id.removeprefix('ddxplus:')]
    return {title: {'present': any(p['title'].casefold() == title.casefold() for p in comparison['pairs']),
                    'pairs': [p for p in comparison['pairs'] if p['title'].casefold() == title.casefold()]}
            for title in titles}

"""Task 7.5: pure, label-free selection ablations on a frozen Task 7.4 pool.

Not imported by the production workflow. No search/model calls, tuning, labels,
or changes to candidate generation, raw learned scores, or evidence contracts.
"""
from collections import Counter
import math
import re

from rag.agents.models import PatientState
from rag.agents.grounding import context, evidence_from_hits
from rag.medical_ingestion.models import validate_chunk
from orchestration.evidence import EvidenceSnapshot, FrozenEvidence, fingerprint
from .queries import observed_facts
from .retrieval import choose_final, fuse, matches, tokens, VERSION as RETRIEVAL_VERSION, RERANKER_MODEL, RERANKER_REVISION
from .qdrant_index import evidence_hits

VERSION = 'task75-frozen-pool-selection-v1'
VARIANTS = ('task74', 'symptoms_weight_2', 'background_cap_1',
            'population_penalty_0.5', 'history_novelty')
ROLES = ('symptoms', 'history')
DESCRIPTIONS = {
    'task74': 'Exact Task 7.4 role order, positive raw-logit gate and diversity rules.',
    'symptoms_weight_2': 'Double only the symptom logit contribution to ranking; raw eligibility and role order unchanged.',
    'background_cap_1': 'Keep the first qualified background passage but do not backfill unused current-finding slots with more history.',
    'population_penalty_0.5': 'Subtract 0.5 ranking points for explicit title/section age mismatch; no change to raw-score eligibility.',
    'history_novelty': 'After the first background passage require a new history query, or a different source section from the same document; avoid repeating one history query across documents.',
}


def population_features(state, record):
    """Small generic metadata feature, not clinical population eligibility.

    Missing age/scope is unknown, not a mismatch. Body mentions are deliberately
    excluded. Mixed populations use a union. Adult includes older adults.
    """
    text = (record['source']['title'] + ' ' + record['section']).casefold()
    scopes = []
    if re.search(r'\b(older adults?|elderly|seniors?|geriatric)\b', text):
        scopes.append(('older_adult', 65, math.inf))
        text = re.sub(r'\bolder adults?\b', '', text)
    for name, pattern, low, high in (
        ('infant', r'\b(infants?|babies|newborns?)\b', 0, 1),
        ('child', r'\b(child|children|pediatric|paediatric)\b', 0, 17),
        ('teen', r'\b(teens?|teenagers?|adolescents?)\b', 13, 17),
        ('adult', r'\badults?\b', 18, math.inf),
    ):
        if re.search(pattern, text):
            scopes.append((name, low, high))
    age = state.age
    status = 'unknown' if age is None or not scopes else 'match' if any(lo <= age <= hi for _, lo, hi in scopes) else 'mismatch'
    return {'patient_age': age, 'explicit_title_section_populations': [s[0] for s in scopes],
            'status': status, 'mismatch': status == 'mismatch',
            'basis': 'literal title/section age wording; not a medical contraindication'}


def validate_frozen_experiment(state, experiment, source_payloads):
    """Reject stale/malformed caches; replay fusion and original snapshot checks.

    source_payloads is the current approved corpus payload inventory, never
    relevance labels. Model output authenticity is bounded by the recorded input
    artifact hash; we do not pretend to recompute neural scores in this replay.
    """
    if type(state) is not PatientState:
        raise TypeError('Label-free PatientState required')
    state = PatientState.model_validate(state.model_dump())
    if experiment['version'] != RETRIEVAL_VERSION or type(experiment['candidate_depth']) is not int or experiment['candidate_depth'] != 50:
        raise ValueError('Only frozen Task 7.4 depth-50 experiments are allowed')
    if experiment['final_limit'] != 5 or experiment['pre_reranking_lexical_filter'] is not False:
        raise ValueError('Unexpected frozen retrieval policy')
    signature = experiment['reranker']
    if signature.get('model') != RERANKER_MODEL or signature.get('revision') != RERANKER_REVISION or signature.get('max_length') != 512:
        raise ValueError('Frozen reranker identity changed')
    facts, ignored = observed_facts(state)
    queries = experiment['plan']['queries']
    qmap = {q['query_id']: q for q in queries}
    if len(qmap) != len(queries) or not queries:
        raise ValueError('Missing/duplicate frozen queries')
    actual_facts = [f for q in queries for f in q['facts']]
    if sorted(map(fingerprint, actual_facts)) != sorted(map(fingerprint, facts)):
        raise ValueError('Frozen query facts differ from current label-free state')
    population = 'child' if state.age is not None and state.age < 18 else 'older adult' if state.age is not None and state.age >= 65 else 'adult' if state.age is not None else ''
    for q in queries:
        if q['kind'] not in ROLES or not 1 <= len(q['facts']) <= 4 or any(f['kind'] != q['kind'] for f in q['facts']):
            raise ValueError('Invalid query role/size')
        text = '; '.join(f['text'] for f in q['facts'])
        if q['kind'] == 'history':
            text = ' '.join(filter(None, (population, text, 'symptoms and clinical information')))
        if text != q['text'] or q['fact_ids'] != list(dict.fromkeys(ref for f in q['facts'] for ref in f['fact_ids'])):
            raise ValueError('Altered query wording/fact references')
    if experiment['plan']['excluded_from_positive_queries'] != ignored or experiment['plan']['population'] != population:
        raise ValueError('Altered query exclusions/population')
    rows = experiment['reranked']
    by_id = {r['chunk_id']: r for r in rows}
    by_record = {r['chunk_id']: r for r in experiment['candidate_records']}
    if len(by_id) != len(rows) or len(by_record) != len(experiment['candidate_records']) or set(by_id) != set(by_record):
        raise ValueError('Duplicate/missing frozen candidates')
    records = {}
    for row in rows:
        record = by_record[row['chunk_id']]
        payload = validate_chunk(record['evidence_payload'])
        if payload != source_payloads.get(row['chunk_id']):
            raise ValueError('Frozen evidence does not match current approved source payload')
        if record['text'] != payload['text'] or record['evidence_text'] != payload['text'] or record['window_text'] != payload['text']:
            raise ValueError('Altered source evidence text')
        if record['source'] != {k: v for k, v in payload.items() if k not in ('text', 'chunk_id', 'chunk_index', 'chunking_version')}:
            raise ValueError('Altered source provenance')
        if record['parent_id'] != payload['chunk_id'] or record['section'] != payload['section']:
            raise ValueError('Only original Qdrant chunks may be replayed')
        header = payload['title'] + ('\n' + payload['section'] if payload['section'] else '')
        if record['retrieval_text'] != header + '\n' + payload['text']:
            raise ValueError('Altered retrieval context')
        if type(row['index']) is not int or row['index'] < 0 or row['index'] in records:
            raise ValueError('Invalid frozen candidate index')
        records[row['index']] = record
        if not math.isfinite(row['semantic_score']) or type(row['demographic_feature']) is not bool:
            raise ValueError('Invalid cached ranking feature')
        for role in ROLES:
            score = row[role + '_score']
            if score is None:
                if any(q['kind'] == role for q in queries):
                    raise ValueError('Missing role score')
                continue
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError('Invalid cached reranker output')
            query_id = row['reranker_queries'][role]
            if query_id not in qmap or qmap[query_id]['kind'] != role:
                raise ValueError('Reranker query role mismatch')
            concepts = list(dict.fromkeys(f['concept'] for q in queries if q['kind'] == role for f in q['facts']))
            coverage = [c for c in concepts if matches(c, record['retrieval_text'])]
            if coverage != row['concept_features'][role]:
                raise ValueError('Altered cached concept feature')
            expected = score + .5 * len(coverage) / max(1, len(concepts)) + .1 * row['semantic_score'] + .1 * row['rrf_score'] + .2 * row['demographic_feature']
            if not math.isclose(expected, row[role + '_ranking_score'], abs_tol=1e-12):
                raise ValueError('Altered Task 7.4 ranking formula')
        if row['reranker_score'] != max(row[r + '_score'] for r in ROLES if row[r + '_score'] is not None):
            raise ValueError('Altered aggregate learned score')
    # Reuse the frozen RRF implementation; no dense or BM25 search is rerun.
    rankings, channels = [], set()
    for channel in experiment['channel_rankings']:
        key = (channel['query_id'], channel['channel'])
        if key in channels or key[0] not in qmap or key[1] not in ('dense', 'bm25') or len(channel['hits']) > 50:
            raise ValueError('Invalid frozen retrieval channel/depth')
        channels.add(key)
        rankings.append((*key, [(by_id[h['chunk_id']]['index'], h['score']) for h in channel['hits']]))
    if channels != {(q, c) for q in qmap for c in ('dense', 'bm25')}:
        raise ValueError('Missing retrieval channels')
    fused = fuse(rankings, records)
    # Task 7.4 Qdrant checkpoints have no extra exact-span duplicates. Refuse
    # unsupported caches rather than silently changing the candidate population.
    if experiment['deduplication_removed'] or fused != experiment['fused_candidates'] or fused != experiment['candidate_pool']:
        raise ValueError('Fusion/deduplication differs from frozen Qdrant pool')
    expected_ranked = sorted(rows, key=lambda r: (-r['reranker_score'], -r['rrf_score'], r['chunk_id']))
    if rows != expected_ranked or len(fused) != len(rows):
        raise ValueError('Altered reranker ordering/pool')
    for rank, row in enumerate(rows, 1):
        if any(row[key] != fused[row['fusion_rank'] - 1][key] for key in ('index', 'chunk_id', 'rrf_score', 'origins')):
            raise ValueError('Altered per-candidate fusion provenance')
        if row['reranked_rank'] != rank or fused[row['fusion_rank'] - 1]['chunk_id'] != row['chunk_id']:
            raise ValueError('Altered stage rank')
    baseline = choose_final(rows, records, queries, 5)
    if baseline != experiment['final_passages']:
        raise ValueError('Current Task 7.4 selection does not reproduce cached baseline')
    snapshot = make_snapshot(state, baseline)
    if snapshot.model_dump(mode='json') != experiment['evidence_snapshot']:
        raise ValueError('Frozen baseline evidence snapshot mismatch')
    return records


def make_snapshot(state, selected):
    evidence = evidence_from_hits(evidence_hits({'final_passages': selected}), 'medical_knowledge')
    context(state, [], evidence)
    return EvidenceSnapshot(snapshot_id=fingerprint({'patient_cases': [], 'medical_knowledge': [e.model_dump() for e in evidence]}),
        patient_cases=(), medical_knowledge=tuple(FrozenEvidence.freeze(e) for e in evidence))


def ablate(state, experiment, records, variant, sample_memory=lambda: None):
    """One controlled selector change, all original numeric scores immutable."""
    if variant not in VARIANTS:
        raise ValueError('Unknown fixed ablation; no tuning parameters accepted')
    rows, queries = experiment['reranked'], experiment['plan']['queries']
    qmap = {q['query_id']: q for q in queries}
    populations = {r['index']: population_features(state, records[r['index']]) for r in rows}
    events = {r['chunk_id']: [] for r in rows}
    selected, parents, texts = [], set(), []
    docs, history_queries = Counter(), {}
    section_pairs = set()
    def ranking_score(row, role):
        score = row[role + '_ranking_score']
        if variant == 'symptoms_weight_2' and role == 'symptoms':
            score += row['symptoms_score']  # weight 2 for the logit, features fixed
        if variant == 'population_penalty_0.5' and populations[row['index']]['mismatch']:
            score -= .5
        return score
    def take(role, limit, phase):
        eligible = [r for r in rows if r[role + '_score'] is not None]
        ordered = sorted(eligible, key=lambda r: (-ranking_score(r, role), -r['rrf_score'], r['chunk_id']))
        for row in ordered:
            sample_memory()
            record = records[row['index']]
            doc, query_id = record['source']['document_id'], row['reranker_queries'][role]
            words = set(tokens(record['text']))
            pair = (doc, record['section'].casefold().strip())
            reason = None
            if len(selected) >= limit:
                reason = 'pass_budget_reached'
            elif row[role + '_score'] <= 0:
                reason = 'nonpositive_raw_role_logit'
            elif record['parent_id'] in parents:
                reason = 'parent_already_selected'
            elif docs[doc] >= 2:
                reason = 'document_limit_2'
            elif any(len(words & previous) / max(1, len(words | previous)) >= .8 for previous in texts):
                reason = 'near_duplicate_text'
            elif variant == 'history_novelty' and role == 'history' and query_id in history_queries:
                # No condition names or hand-assigned clinical topic labels.
                if doc not in history_queries[query_id] or pair in section_pairs:
                    reason = 'redundant_history_query_without_new_same_document_section'
            if reason is None:
                selected.append({**row, 'selection_role': role, 'record': record})
                parents.add(record['parent_id'])
                docs[doc] += 1
                texts.append(words)
                section_pairs.add(pair)
                if role == 'history':
                    history_queries.setdefault(query_id, set()).add(doc)
                reason = 'selected_' + phase
            events[row['chunk_id']].append({'phase': phase, 'role': role,
                'effective_ranking_score': ranking_score(row, role), 'reason': reason})
    if any(q['kind'] == 'history' for q in queries):
        take('history', 1, 'history_priority')
    take('symptoms', 5, 'current_findings')
    if variant != 'background_cap_1':
        take('history', 5, 'history_fill')
    selected.sort(key=lambda row: row['selection_role'] == 'history')
    if variant == 'task74' and selected != choose_final(rows, records, queries, 5):
        raise ValueError('Instrumented baseline diverged from unchanged Task 7.4 selector')
    selected_by_id = {r['chunk_id']: r for r in selected}
    observed = {role: list(dict.fromkeys(f['concept'] for q in queries if q['kind'] == role for f in q['facts'])) for role in ROLES}
    audit = []
    for row in rows:
        sample_memory()
        record = records[row['index']]
        chosen = selected_by_id.get(row['chunk_id'])
        positive_roles = [role for role in ROLES if row[role + '_score'] is not None and row[role + '_score'] > 0]
        if chosen:
            stage = 'final_evidence'
            reason = next(e['reason'] for e in events[row['chunk_id']] if e['reason'].startswith('selected_'))
        elif not positive_roles:
            stage, reason = 'reranker_eligibility', 'no_positive_raw_role_logit; ranking_features_cannot_restore_eligibility'
        elif variant == 'background_cap_1' and positive_roles == ['history']:
            stage, reason = 'role_allocation', 'background_cap_1; no_history_backfill'
        else:
            exclusions = [e['reason'] for e in events[row['chunk_id']] if e['reason'] not in ('pass_budget_reached', 'nonpositive_raw_role_logit')]
            stage = 'diversity_selection' if exclusions else 'final_budget'
            reason = '; '.join(dict.fromkeys(exclusions)) if exclusions else 'eligible_but_below_final_budget'
        roles = {}
        for role, query_id in row['reranker_queries'].items():
            query = qmap[query_id]
            origins = [o for o in row['origins'] if o['query_id'] == query_id]
            roles[role] = {'originating_query_id': query_id, 'originating_query': query['text'],
                'query_role': role, 'fact_ids': query['fact_ids'],
                'dense_rank': next((o['rank'] for o in origins if o['channel'] == 'dense'), None),
                'bm25_rank': next((o['rank'] for o in origins if o['channel'] == 'bm25'), None),
                'raw_reranker_score': row[role + '_score'],
                'task74_ranking_score': row[role + '_ranking_score'],
                'ablation_ranking_score': ranking_score(row, role),
                'positive_logit_eligible': row[role + '_score'] > 0,
                'concept_coverage': {'matched': row['concept_features'][role], 'observed_count': len(observed[role])}}
        audit.append({'chunk_id': row['chunk_id'], 'source_id': 'medical:' + row['chunk_id'],
            'title': record['source']['title'], 'section': record['section'],
            'source': record['source']['source'], 'url': record['source']['url'],
            'document_id': record['source']['document_id'], 'evidence_text': record['evidence_text'],
            'rrf_score': row['rrf_score'], 'fusion_rank': row['fusion_rank'],
            'reranker_score': row['reranker_score'], 'reranked_rank': row['reranked_rank'],
            'role_details': roles, 'all_channel_origins': row['origins'],
            'population_features': populations[row['index']],
            'task74_population_feature': row['demographic_feature'],
            'selected': chosen is not None, 'selected_role': chosen['selection_role'] if chosen else None,
            'selection_or_removal_stage': stage, 'final_selection_reason': reason,
            'selector_events': events[row['chunk_id']]})
    snapshot = make_snapshot(state, selected)
    return {'version': VERSION, 'variant': variant, 'description': DESCRIPTIONS[variant],
        'candidate_depth': 50, 'upstream_reused': True,
        'frozen_pool_fingerprint': fingerprint(experiment['candidate_pool']),
        'frozen_scores_fingerprint': fingerprint(rows),
        'statistics': {'candidates': len(rows),
            'positive_symptom_scores': sum(r['symptoms_score'] is not None and r['symptoms_score'] > 0 for r in rows),
            'positive_history_scores': sum(r['history_score'] is not None and r['history_score'] > 0 for r in rows),
            'selected': len(selected), 'selected_current': sum(r['selection_role'] == 'symptoms' for r in selected),
            'selected_background': sum(r['selection_role'] == 'history' for r in selected),
            'selected_explicit_population_mismatches': sum(populations[r['index']]['mismatch'] for r in selected)},
        'selected_ids': [r['chunk_id'] for r in selected], 'candidate_audit': audit,
        'evidence_snapshot': snapshot.model_dump(mode='json')}

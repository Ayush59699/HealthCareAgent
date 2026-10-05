"""Task 7.6: change only query-side reranker formatting on a frozen pool.

No retrieval/query decomposition, candidate selection, feature weights, eligibility
thresholds, models or evidence text are changed. Not imported by live workflows.
"""
from copy import deepcopy
import math
from time import perf_counter

from rag.agents.models import PatientState
from orchestration.evidence import fingerprint
from .ranking_ablation import ROLES, ablate

VERSION = 'task76-role-explicit-input-v1'
FORMATS = ('original', 'role_explicit')
BATCH_SIZE = 2
# Fixed before scoring, not selected/tuned per patient or condition.
CURRENT_TEMPLATE = "The patient's current reported findings are {findings}. What medical information is relevant to these current findings?"
HISTORY_TEMPLATE = "The reported historical or background information{population} is {findings}. What medical information is relevant to this history?"


def natural_list(terms):
    if len(terms) == 1:
        return terms[0]
    return ', '.join(terms[:-1]) + ' and ' + terms[-1]


def format_queries(state, plan, variant):
    if type(state) is not PatientState:
        raise TypeError('Validated label-free PatientState required')
    if variant not in FORMATS:
        raise ValueError('Only the two fixed reranker-input formats are permitted')
    population = plan['population']
    expected_population = 'child' if state.age is not None and state.age < 18 else 'older adult' if state.age is not None and state.age >= 65 else 'adult' if state.age is not None else ''
    if population != expected_population:
        raise ValueError('Frozen population differs from patient facts')
    formatted = {}
    for query in plan['queries']:
        role, facts = query['kind'], query['facts']
        if role not in ROLES or not 1 <= len(facts) <= 4 or any(f['kind'] != role for f in facts):
            raise ValueError('Invalid frozen query role/facts')
        if query['query_id'] in formatted:
            raise ValueError('Duplicate frozen query ID')
        if variant == 'original':
            text = query['text']
        else:
            findings = natural_list([f['text'] for f in facts])
            if role == 'symptoms':
                # No age/sex is added to current queries: the frozen originals
                # did not contain it. Do not mix history into current findings.
                text = CURRENT_TEMPLATE.format(findings=findings)
            else:
                prefix = {'child': ' for a child', 'adult': ' for an adult',
                          'older adult': ' for an older adult', '': ''}[population]
                text = HISTORY_TEMPLATE.format(findings=findings, population=prefix)
        formatted[query['query_id']] = {'query_id': query['query_id'], 'role': role,
            'retrieval_query': query['text'], 'reranker_input': text,
            'facts': deepcopy(facts), 'fact_ids': list(query['fact_ids'])}
    return formatted


def pair_plan(state, experiment, records, variant):
    """Keep original RRF pair order and cached supporting-query assignments."""
    if type(experiment['candidate_depth']) is not int or experiment['candidate_depth'] != 50:
        raise ValueError('Only frozen depth-50 pools are permitted')
    formatted = format_queries(state, experiment['plan'], variant)
    by_id = {r['chunk_id']: r for r in experiment['reranked']}
    pairs, audit = [], []
    for candidate in experiment['candidate_pool']:
        row = by_id[candidate['chunk_id']]
        record = records[row['index']]
        for role in ROLES:
            if row[role + '_score'] is None:
                continue
            query_id = row['reranker_queries'][role]
            query = formatted[query_id]
            if query['role'] != role:
                raise ValueError('Cached pair query has the wrong role')
            pairs.append((query['reranker_input'], record['retrieval_text']))
            audit.append({'chunk_id': row['chunk_id'], 'role': role, 'query_id': query_id,
                'retrieval_query': query['retrieval_query'], 'reranker_input': query['reranker_input'],
                'fact_ids': query['fact_ids'], 'passage_sha256': fingerprint(record['retrieval_text']),
                'frozen_role_logit': row[role + '_score']})
    if len(pairs) != experiment['reranker_pair_count'] or len({(a['chunk_id'], a['role']) for a in audit}) != len(audit):
        raise ValueError('Incomplete/duplicate frozen pair assignments')
    return pairs, audit, formatted


def check_token_budget(reranker, pairs, audit, sample_memory=lambda: None):
    """Refuse truncation rather than change passage content as a confound."""
    if len(pairs) != len(audit):
        raise ValueError('Pair/audit length mismatch')
    lengths = []
    for (query, passage), row in zip(pairs, audit):
        sample_memory()
        length = len(reranker.model.tokenizer.encode(query, passage, truncation=False))
        lengths.append(length)
        if length > 512:
            raise ValueError(f"Input formatting would truncate a frozen pair: {row['chunk_id']} / {row['role']} / {length} tokens; no scores produced")
    return lengths


def score_pairs(reranker, pairs, sample_memory=lambda: None, progress=lambda done, total: None):
    if reranker.batch_size != BATCH_SIZE:
        raise ValueError('Task 7.6 requires unchanged batch size 2')
    scores = []
    before = reranker.truncated_pairs
    for start in range(0, len(pairs), BATCH_SIZE):
        sample_memory()
        batch = pairs[start:start + BATCH_SIZE]
        values = reranker.score(batch)
        if len(values) != len(batch) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('Invalid/incomplete local reranker scores; no fallback')
        scores.extend(float(v) for v in values)
        sample_memory()
        progress(len(scores), len(pairs))
    if reranker.truncated_pairs != before:
        raise ValueError('Reranker truncated a pair despite token preflight')
    return scores


def apply_scores(experiment, pair_audit, scores):
    """Replace learned logits only; recompute the same derived ranking fields."""
    if len(scores) != len(pair_audit):
        raise ValueError('Missing pair scores')
    values = {}
    for pair, score in zip(pair_audit, scores):
        key = (pair['chunk_id'], pair['role'])
        if key in values or type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError('Duplicate/nonfinite pair output')
        values[key] = score
    expected = {(r['chunk_id'], role) for r in experiment['reranked'] for role in ROLES if r[role + '_score'] is not None}
    if set(values) != expected:
        raise ValueError('Pair identities differ from frozen assignments')
    queries = experiment['plan']['queries']
    concepts = {role: list(dict.fromkeys(f['concept'] for q in queries if q['kind'] == role for f in q['facts'])) for role in ROLES}
    rows = []
    for original in experiment['reranked']:
        row = deepcopy(original)
        for role in ROLES:
            if original[role + '_score'] is None:
                continue
            score = values[(row['chunk_id'], role)]
            row[role + '_score'] = score
            # Exactly the Task 7.4 formula: no new penalty, weighting, threshold
            # or coverage calculation based on the formatted model input.
            row[role + '_ranking_score'] = score + .5 * len(row['concept_features'][role]) / max(1, len(concepts[role])) + .1 * row['semantic_score'] + .1 * row['rrf_score'] + .2 * row['demographic_feature']
        row['reranker_score'] = max(row[role + '_score'] for role in ROLES if row[role + '_score'] is not None)
        rows.append(row)
    rows.sort(key=lambda r: (-r['reranker_score'], -r['rrf_score'], r['chunk_id']))
    for rank, row in enumerate(rows, 1):
        row['reranked_rank'] = rank
    return {**experiment, 'reranked': rows}


def run_format(state, experiment, records, reranker, variant, sample_memory=lambda: None, progress=lambda done, total: None):
    frozen_hash = fingerprint(experiment)
    started = perf_counter()
    pairs, pairs_audit, formatted = pair_plan(state, experiment, records, variant)
    lengths = check_token_budget(reranker, pairs, pairs_audit, sample_memory)
    prepared = perf_counter()
    scores = score_pairs(reranker, pairs, sample_memory, progress)
    scored_at = perf_counter()
    working = apply_scores(experiment, pairs_audit, scores)
    # Reuse the exact existing selector and source/snapshot audit. None of Task
    # 7.5's selection alternatives are enabled in this formatting experiment.
    result = ablate(state, working, records, 'task74', sample_memory)
    role_inputs = {(p['chunk_id'], p['role']): (p, n, s) for p, n, s in zip(pairs_audit, lengths, scores)}
    for row in result['candidate_audit']:
        for role, details in row['role_details'].items():
            pair, length, score = role_inputs[(row['chunk_id'], role)]
            details.update(reranker_input=pair['reranker_input'], pair_tokens=length,
                frozen_role_logit=pair['frozen_role_logit'], frozen_role_eligible=pair['frozen_role_logit'] > 0,
                logit_delta_from_frozen=score - pair['frozen_role_logit'],
                passage_sha256=pair['passage_sha256'])
    if fingerprint(experiment) != frozen_hash:
        raise ValueError('Formatting experiment mutated frozen inputs')
    result.update(version=VERSION, variant=variant,
        description='Query-side formatting only; Task 7.4 selector and all other retrieval signals fixed.',
        original_scores_fingerprint=fingerprint(experiment['reranked']),
        # The inherited field describes the new score vector, not frozen logits.
        rescored_rows_fingerprint=result.pop('frozen_scores_fingerprint'),
        reranker_signature=reranker.signature, formatted_queries=list(formatted.values()),
        pair_count=len(pairs), pair_order_fingerprint=fingerprint([(p['chunk_id'], p['role'], p['query_id'], p['passage_sha256']) for p in pairs_audit]),
        max_pair_tokens=max(lengths, default=0), truncated_pairs=0,
        timings={'format_and_token_preflight_seconds': prepared-started,
                 'cross_encoder_seconds': scored_at-prepared,
                 'selection_and_audit_seconds': perf_counter()-scored_at,
                 'total_seconds': perf_counter()-started})
    return result


def baseline_drift(experiment, result, tolerance=1e-4):
    """Numerical control, not a relevance cutoff. Stop on a changed baseline."""
    differences = [abs(d['raw_reranker_score'] - d['frozen_role_logit'])
                   for r in result['candidate_audit'] for d in r['role_details'].values()]
    flips = sum(d['positive_logit_eligible'] != d['frozen_role_eligible']
                for r in result['candidate_audit'] for d in r['role_details'].values())
    same_snapshot = result['evidence_snapshot'] == experiment['evidence_snapshot']
    maximum = max(differences, default=0.)
    return {'maximum_absolute_logit_drift': maximum,
        'mean_absolute_logit_drift': sum(differences) / max(1, len(differences)),
        'eligibility_changes': flips, 'same_snapshot': same_snapshot,
        'absolute_tolerance': tolerance, 'passed': maximum <= tolerance and flips == 0 and same_snapshot}


def compare_formats(original, formatted):
    if original['pair_order_fingerprint'] != formatted['pair_order_fingerprint'] or original['frozen_pool_fingerprint'] != formatted['frozen_pool_fingerprint']:
        raise ValueError('Not a controlled same-pool/same-pair comparison')
    originals = {r['chunk_id']: r for r in original['candidate_audit']}
    changes, transitions = [], {}
    for role in ROLES:
        counts = {'nonpositive_to_positive': 0, 'positive_to_nonpositive': 0, 'unchanged_eligibility': 0}
        for row in formatted['candidate_audit']:
            if role not in row['role_details']:
                continue
            a, b = originals[row['chunk_id']]['role_details'][role], row['role_details'][role]
            direction = 'nonpositive_to_positive' if not a['positive_logit_eligible'] and b['positive_logit_eligible'] else 'positive_to_nonpositive' if a['positive_logit_eligible'] and not b['positive_logit_eligible'] else 'unchanged_eligibility'
            counts[direction] += 1
            changes.append({'chunk_id': row['chunk_id'], 'title': row['title'], 'section': row['section'], 'role': role,
                'original_logit': a['raw_reranker_score'], 'formatted_logit': b['raw_reranker_score'],
                'logit_delta': b['raw_reranker_score'] - a['raw_reranker_score'], 'eligibility_transition': direction,
                'original_selected': originals[row['chunk_id']]['selected'], 'formatted_selected': row['selected'],
                'original_stage': originals[row['chunk_id']]['selection_or_removal_stage'],
                'formatted_stage': row['selection_or_removal_stage']})
        transitions[role] = counts
    old, new = set(original['selected_ids']), set(formatted['selected_ids'])
    return {'eligibility_transitions': transitions, 'selected_gained': sorted(new-old),
            'selected_lost': sorted(old-new), 'pair_changes': changes}

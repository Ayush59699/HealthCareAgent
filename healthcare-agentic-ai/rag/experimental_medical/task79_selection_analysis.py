"""Task 7.9 only: frozen-score selector replay and ONE final-budget counterfactual.

No model/retrieval/clinical modules imported. The original pure choose_final and
its tokenizer are extracted unchanged from authenticated source, not patched.
The diagnostic trace is checked against that function for each condition.
"""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROLES = ('symptoms', 'history')
POOLS = {1: 593, 2: 416, 3: 425}
VARIANTS = ('saved_budget', 'no_final_budget')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf8')).hexdigest()


def original_functions(root):
    """Caller authenticates these source bytes against Task 7.8's saved manifest."""
    root = Path(root)
    focused = ast.parse((root / 'rag/focused_medical.py').read_text(encoding='utf8'))
    retrieval = ast.parse((root / 'rag/experimental_medical/retrieval.py').read_text(encoding='utf8'))
    stop = [n for n in focused.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'STOP' for t in n.targets)]
    functions = [n for n in retrieval.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('tokens', 'choose_final')]
    if len(stop) != 1 or {n.name for n in functions} != {'tokens', 'choose_final'}:
        raise ValueError('Cannot reconstruct the authenticated original pure selector')
    validator = (root / 'rag/experimental_medical/reranker_prefix_control.py').read_text(encoding='utf8')
    if "record['parent_id'] != payload['chunk_id']" not in validator:
        raise ValueError('Authenticated parent=chunk invariant missing; no inferred fallback')
    env = {'Counter': Counter, 're': re}
    exec(compile(ast.Module(body=stop + functions, type_ignores=[]),
                 '<unchanged authenticated Task 7.4 selector>', 'exec'), env)
    return env['choose_final'], env['tokens']


def adapt_saved(saved, expected_count=None):
    """Lookup views only, not new candidate pools, scores, ranks or queries.

Task 7.8's authenticated validate_frozen_case asserted text=evidence text and
parent_id=payload.chunk_id for EVERY candidate, not just selected snapshots.
The audits omit parent_id, so that proved invariant supplies the lookup value.
Index is a private chunk-key lookup, not a substituted retrieval ordering field.
"""
    if saved['candidate_depth'] != 50 or saved['variant'] not in ('control', 'experimental'):
        raise ValueError('Unexpected saved experiment')
    audit = saved['candidate_audit']
    if expected_count is not None and len(audit) != expected_count:
        raise ValueError('Frozen pool size mismatch')
    if len({a['chunk_id'] for a in audit}) != len(audit):
        raise ValueError('Duplicate candidate')
    queries = {q['query_id']: q for q in saved['formatted_queries']}
    if len(queries) != len(saved['formatted_queries']):
        raise ValueError('Duplicate query')
    rows, records = [], {}
    for position, a in enumerate(audit, 1):
        if a['reranked_rank'] != position:
            raise ValueError('Saved candidate ordering changed')
        if hashlib.sha256(a['evidence_text'].encode('utf8')).hexdigest() != a['evidence_text_sha256']:
            raise ValueError('Saved source text hash mismatch')
        cid = a['chunk_id']
        row = {'index': cid, 'chunk_id': cid, 'rrf_score': a['rrf_score']}
        records[cid] = {'parent_id': cid, 'text': a['evidence_text'],
                        'source': {'document_id': a['document_id']}}
        for role in ROLES:
            d = a['role_details'].get(role)
            row[role + '_score'] = None if d is None else d['raw_reranker_score']
            if d is None:
                continue
            q = queries[d['originating_query_id']]
            if (q['role'] != role or q['retrieval_query'] != q['reranker_input']
                    or d['originating_query'] != q['retrieval_query']
                    or d['reranker_input'] != q['retrieval_query']
                    or d['fact_ids'] != q['fact_ids']):
                raise ValueError('Frozen query assignment mismatch')
            score, ranking = d['raw_reranker_score'], d['task74_ranking_score']
            if not math.isfinite(score) or not math.isfinite(ranking):
                raise ValueError('Nonfinite saved score')
            if type(d['positive_logit_eligible']) is not bool or d['positive_logit_eligible'] != (score > 0):
                raise ValueError('Saved eligibility inconsistent; no threshold adjustment')
            row[role + '_ranking_score'] = ranking
        rows.append(row)
    if sum(len(a['role_details']) for a in audit) != saved['pair_count']:
        raise ValueError('Saved pair coverage mismatch')
    return rows, records, [{**q, 'kind': q['role']} for q in saved['formatted_queries']]


def trace_selector(rows, records, queries, tokenize, remove_final_budget=False, sample=lambda: None):
    """Same deterministic checks/order; logging does not control selection.

history_priority=1 is a ROLE quota and remains unchanged. Only the two passes
using the final count=5 become unbounded. Saved budget events precede eligibility
and diversity tests; shadow nonbudget checks are diagnostic, never extra policies.
"""
    selected, parents, docs = [], set(), Counter()
    selected_words, events = [], []
    words = {cid: set(tokenize(record['text'])) for cid, record in records.items()}
    has_history = any(q['kind'] == 'history' for q in queries)
    cap = None if remove_final_budget else 5
    passes = ([('history_priority', 'history', 1)] if has_history else [])
    passes += [('current_findings', 'symptoms', cap), ('history_fill', 'history', cap)]

    def nonbudget(row, role):
        cid = row['chunk_id']
        record = records[row['index']]
        if row[role + '_score'] <= 0:
            return 'nonpositive_raw_role_logit', []
        if record['parent_id'] in parents:
            return 'parent_already_selected', [r['chunk_id'] for r in selected
                    if records[r['index']]['parent_id'] == record['parent_id']]
        if docs[record['source']['document_id']] >= 2:
            return 'document_limit_2', [r['chunk_id'] for r in selected
                    if records[r['index']]['source']['document_id'] == record['source']['document_id']]
        overlaps = [(old, len(words[cid] & previous) / max(1, len(words[cid] | previous)))
                    for old, previous in selected_words]
        duplicates = [{'chunk_id': old, 'jaccard': similarity} for old, similarity in overlaps if similarity >= .8]
        if duplicates:
            return 'near_duplicate_text', duplicates
        return None, []

    for phase, role, limit in passes:
        ordered = sorted((r for r in rows if r[role + '_score'] is not None),
                         key=lambda r: (-r[role + '_ranking_score'], -r['rrf_score'], r['chunk_id']))
        for rank, row in enumerate(ordered, 1):
            sample()
            cid = row['chunk_id']
            saturated = limit is not None and len(selected) >= limit
            actual, blockers = nonbudget(row, role)
            reason = 'pass_budget_reached' if saturated else actual
            budget_kind = ('history_priority_quota_1' if phase == 'history_priority'
                           else 'final_evidence_count_5') if saturated else None
            event = {'chunk_id': cid, 'role': role, 'phase': phase, 'role_ranking_position': rank,
                'raw_reranker_score': row[role + '_score'], 'effective_ranking_score': row[role + '_ranking_score'],
                'saved_eligible': row[role + '_score'] > 0, 'selected_before': [s['chunk_id'] for s in selected],
                'phase_limit': limit, 'budget_kind': budget_kind,
                'nonbudget_first_failure_at_this_prefix': actual,
                'nonbudget_blockers_at_this_prefix': blockers}
            if reason is None:
                selected.append({**row, 'selection_role': role})
                record = records[row['index']]
                parents.add(record['parent_id'])
                docs[record['source']['document_id']] += 1
                selected_words.append((cid, words[cid]))
                reason = 'selected_' + phase
            event['reason'] = reason
            events.append(event)
    chronological = [r['chunk_id'] for r in selected]
    selected.sort(key=lambda r: r['selection_role'] == 'history')
    return {'selected_ids': [r['chunk_id'] for r in selected],
            'selected_roles': {r['chunk_id']: r['selection_role'] for r in selected},
            'chronological_selected_ids': chronological, 'events': events}


def replay(saved, choose_final, tokenize, remove_final_budget=False, sample=lambda: None):
    original_hash = fingerprint(saved)
    rows, records, queries = adapt_saved(saved)
    maximum = math.inf if remove_final_budget else 5
    exact = choose_final(rows, records, queries, maximum)
    result = trace_selector(rows, records, queries, tokenize, remove_final_budget, sample)
    if (result['selected_ids'] != [r['chunk_id'] for r in exact]
            or result['selected_roles'] != {r['chunk_id']: r['selection_role'] for r in exact}):
        raise ValueError('Diagnostic trace differs from unmodified original selector')
    if not remove_final_budget:
        require_control(saved, result)
    if fingerprint(saved) != original_hash:
        raise ValueError('Frozen saved data mutated')
    result['input_fingerprint'] = original_hash
    return result


def require_control(saved, result):
    if result['selected_ids'] != saved['selected_ids']:
        raise ValueError('STOP: saved Task 7.8 final IDs/order did not reproduce')
    expected = {a['chunk_id']: a['selected_role'] for a in saved['candidate_audit'] if a['selected']}
    if result['selected_roles'] != expected:
        raise ValueError('STOP: saved selected roles did not reproduce')
    by_id = {a['chunk_id']: [] for a in saved['candidate_audit']}
    for event in result['events']:
        by_id[event['chunk_id']].append({k: event[k] for k in ('phase', 'role', 'effective_ranking_score', 'reason')})
    for a in saved['candidate_audit']:
        if by_id[a['chunk_id']] != a['selector_events']:
            raise ValueError('STOP: original selector-event trace did not reproduce')


def rule(event):
    if event['reason'] == 'pass_budget_reached':
        return event['budget_kind']
    return event['reason']


def analyze(saved, replayed, counterfactual):
    if replayed['input_fingerprint'] != counterfactual['input_fingerprint']:
        raise ValueError('Not the same frozen input')
    audit = saved['candidate_audit']
    by_id = {a['chunk_id']: a for a in audit}
    outputs = []
    for a in audit:
        cid = a['chunk_id']
        role_rows = []
        for role, d in a['role_details'].items():
            old = [e for e in replayed['events'] if e['chunk_id'] == cid and e['role'] == role]
            new = [e for e in counterfactual['events'] if e['chunk_id'] == cid and e['role'] == role]
            last = old[-1]
            latest = new[-1]
            role_rows.append({'role': role, 'originating_query_id': d['originating_query_id'],
                'originating_query': d['originating_query'], 'fact_ids': d['fact_ids'],
                'raw_reranker_score': d['raw_reranker_score'], 'ranking_score': d['task74_ranking_score'],
                'ranking_position': last['role_ranking_position'], 'eligible': d['positive_logit_eligible'],
                'selected_for_role': replayed['selected_roles'].get(cid) == role,
                'counterfactual_selected_for_role': counterfactual['selected_roles'].get(cid) == role,
                'terminal_rule': rule(last), 'counterfactual_terminal_rule': rule(latest),
                'saved_pass_events': old, 'counterfactual_pass_events': new,
                'final_budget_only_at_saved_prefix': d['positive_logit_eligible']
                    and rule(last) == 'final_evidence_count_5'
                    and last['nonbudget_first_failure_at_this_prefix'] is None})
        positive = [r for r in role_rows if r['eligible']]
        selected = cid in replayed['selected_roles']
        cf_selected = cid in counterfactual['selected_roles']
        reason_set = sorted({r['terminal_rule'] for r in positive}) if not selected else []
        cf_reasons = sorted({r['counterfactual_terminal_rule'] for r in positive}) if not cf_selected else []
        local = bool(positive) and not selected and any(r['final_budget_only_at_saved_prefix'] for r in positive)
        outputs.append({**{k: a[k] for k in ('chunk_id', 'source_id', 'source', 'document_id', 'title', 'section',
                    'url', 'evidence_text_sha256', 'fusion_rank', 'rrf_score', 'reranked_rank')},
            'parent_id': cid, 'eligible': bool(positive), 'selected': selected,
            'selected_role': replayed['selected_roles'].get(cid), 'counterfactual_selected': cf_selected,
            'counterfactual_selected_role': counterfactual['selected_roles'].get(cid), 'roles': role_rows,
            'terminal_rules': reason_set, 'counterfactual_terminal_rules': cf_reasons,
            'rejection_reason_key': ' + '.join(reason_set), 'counterfactual_rejection_reason_key': ' + '.join(cf_reasons),
            'budget_only_at_saved_prefix': local, 'additional_when_final_budget_removed': cf_selected and not selected,
            'confirmed_final_budget_only_recovery': local and cf_selected})
    eligible = [a for a in outputs if a['eligible']]
    rejected = [a for a in eligible if not a['selected']]
    cf_rejected = [a for a in eligible if not a['counterfactual_selected']]
    additional = [cid for cid in counterfactual['selected_ids'] if cid not in replayed['selected_ids']]
    summary = {'eligible_candidates': len(eligible), 'eligible_role_pairs': sum(r['eligible'] for a in outputs for r in a['roles']),
        'selected_candidates': len(replayed['selected_ids']), 'eligible_not_selected_candidates': len(rejected),
        'rejection_reason_counts': dict(Counter(a['rejection_reason_key'] for a in rejected)),
        'budget_only_at_saved_prefix': sum(a['budget_only_at_saved_prefix'] for a in rejected),
        'confirmed_final_budget_only_recoveries': sum(a['confirmed_final_budget_only_recovery'] for a in rejected),
        'counterfactual_selected_candidates': len(counterfactual['selected_ids']),
        'additional_candidates': len(additional), 'additional_ids_in_counterfactual_order': additional,
        'counterfactual_still_rejected_candidates': len(cf_rejected),
        'counterfactual_rejection_reason_counts': dict(Counter(a['counterfactual_rejection_reason_key'] for a in cf_rejected)),
        'previously_selected_lost': [cid for cid in replayed['selected_ids'] if cid not in counterfactual['selected_ids']],
        'saved_selected_ids': replayed['selected_ids'], 'counterfactual_selected_ids': counterfactual['selected_ids'],
        'saved_selected_titles': [by_id[cid]['title'] for cid in replayed['selected_ids']],
        'counterfactual_selected_titles': [by_id[cid]['title'] for cid in counterfactual['selected_ids']]}
    assert sum(summary['rejection_reason_counts'].values()) == len(rejected)
    assert sum(summary['counterfactual_rejection_reason_counts'].values()) == len(cf_rejected)
    return {'summary': summary, 'candidate_analysis': outputs,
            'saved_chronological_selection': replayed['chronological_selected_ids'],
            'counterfactual_chronological_selection': counterfactual['chronological_selected_ids']}


def join_review(analysis, review_rows, case):
    """Read-only join AFTER selection. Classifications cannot influence a selector."""
    candidates = {a['chunk_id']: a for a in analysis['candidate_analysis']}
    joined = []
    for review in review_rows:
        if review['case'] != case or review['eligibility_transition'] != 'nonpositive_to_positive':
            continue
        a = candidates[review['chunk_id']]
        role = next(r for r in a['roles'] if r['role'] == review['role'])
        if (review['class'] not in ('A', 'B', 'C') or not role['eligible']
                or role['raw_reranker_score'] != review['experimental_raw_logit']
                or role['originating_query'] != review['experimental_query']
                or a['evidence_text_sha256'] != review['evidence_text_sha256']):
            raise ValueError('Saved Task 7.8.1 review does not match the frozen MedCPT pair')
        joined.append({'review_id': review['review_id'], 'class': review['class'], 'chunk_id': a['chunk_id'],
            'title': a['title'], 'section': a['section'], 'role': role['role'],
            'selected_for_role': role['selected_for_role'], 'candidate_selected_any_role': a['selected'],
            'counterfactual_selected_for_role': role['counterfactual_selected_for_role'],
            'candidate_counterfactual_selected_any_role': a['counterfactual_selected'],
            'rejection_rule': None if role['selected_for_role'] else role['terminal_rule'],
            'counterfactual_rejection_rule': None if role['counterfactual_selected_for_role'] else role['counterfactual_terminal_rule']})
    counts = {}
    for label in ('A', 'B', 'C'):
        group = [r for r in joined if r['class'] == label]
        counts[label] = {'reviewed_newly_positive_pairs': len(group),
            'selected_for_role': sum(r['selected_for_role'] for r in group),
            'eligible_not_selected_for_role': sum(not r['selected_for_role'] for r in group),
            'rejection_reason_counts': dict(Counter(r['rejection_rule'] for r in group if not r['selected_for_role'])),
            'counterfactual_selected_for_role': sum(r['counterfactual_selected_for_role'] for r in group),
            'newly_selected_for_role': sum(r['counterfactual_selected_for_role'] and not r['selected_for_role'] for r in group),
            'counterfactual_rejection_reason_counts': dict(Counter(r['counterfactual_rejection_rule'] for r in group if not r['counterfactual_selected_for_role']))}
    return {'counts': counts, 'pairs': joined}

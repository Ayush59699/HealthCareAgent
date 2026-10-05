"""Label-free passage audit for the single Task 7.8 model comparison."""


def pair_status(pair, condition):
    if not pair[condition + '_eligible']:
        return 'candidate_exists_but_reranker_rejects'
    if pair[condition + '_selected_for_role']:
        return 'candidate_remains_selected' if pair['control_selected_for_role'] else 'candidate_newly_selected'
    return 'candidate_eligible_but_loses_during_selection'


def important_audit(comparison, control):
    predicates = {
        'Panic Disorder': lambda p: p['title'].casefold() == 'panic disorder',
        'Flu': lambda p: p['title'].casefold() == 'flu',
        'anemia': lambda p: 'anemia' in p['title'].casefold(),
        'CKD/kidney': lambda p: any(t in (p['title'] + ' ' + (p['section'] or '')).casefold() for t in ('kidney', 'renal', 'ckd')),
        'asthma (all sections, including symptoms and definition)': lambda p: 'asthma' in p['title'].casefold(),
        'Fainting': lambda p: p['title'].casefold() == 'fainting',
        'Older Adult Mental Health': lambda p: p['title'].casefold() == 'older adult mental health',
    }
    text = {r['chunk_id']: r['evidence_text'] for r in control['candidate_audit']}
    result = {}
    for title, matches in predicates.items():
        pairs = [{**p, 'evidence_text': text[p['chunk_id']],
                  'control_status': pair_status(p, 'control'),
                  'experimental_status': pair_status(p, 'experimental')}
                 for p in comparison['pairs'] if matches(p)]
        result[title] = {'present': bool(pairs), 'pairs': pairs}
    return result

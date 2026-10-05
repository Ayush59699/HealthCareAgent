"""Offline candidate-depth/reranking experiments; never used by the workflow.

Lexical scores and topic-title probes are inspectable research heuristics, not
clinical relevance judgments, recall metrics or diagnostic probabilities.
"""
from .focused_medical import baseline_filter_candidates as filter_candidates, matches, query_plan

VERSION = 'medical-depth-experiment-v1'
DEPTHS = (5, 10, 20, 50)


def rerank_candidates(plan, hits, *, top_k=5):
    """Experimental concept filter -> lexical reranker -> bounded top-k.

    Favor presenting/symptom concepts over history-only title overlap. Preserve
    provenance and original cosine scores; never rewrite payloads or fake a
    learned/cross-encoder score. This does not establish clinical support.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError('Positive reranking top-k required')
    retained, audit = filter_candidates(plan, hits)
    ranked = []
    for hit in retained:
        title = [c for c in plan['symptom_concepts'] if matches(c, hit['title'])]
        body = [c for c in plan['symptom_concepts'] if matches(c, hit['text'])]
        history = [c for c in plan['history_concepts'] if matches(c, hit['title'])]
        score = 3 * len(title) + 2 * len(body) + len(history)
        ranked.append((score, hit, title, body, history))
    ranked.sort(key=lambda item: (-item[0], -item[1]['score'], item[1]['chunk_id']))
    return [{'hit': hit, 'lexical_score': score, 'symptom_title_matches': title,
             'symptom_body_matches': body, 'history_title_matches': history}
            for score, hit, title, body, history in ranked[:top_k]], audit


def evaluate_depths(state, retriever, *, depths=DEPTHS, topics=(), symptom_only=False):
    depths = tuple(depths)
    if not depths or any(type(k) is not int or k < 1 for k in depths) or len(set(depths)) != len(depths):
        raise ValueError('Distinct positive candidate depths required')
    plan = query_plan(state)
    if not plan['symptom_concepts']:
        raise ValueError('No patient symptom concepts; retrieval experiment not applicable')
    if symptom_only:
        # Test query formulation independently of the unchanged live query.
        plan = {**plan, 'query': '; '.join(plan['symptom_concepts']) + ' symptoms clinical information'}
    results = []
    for depth in sorted(depths):
        hits = retriever.retrieve(plan['query'], top_k=depth)
        if len(hits) > depth:
            raise ValueError('Retriever exceeded requested depth')
        reranked, audit = rerank_candidates(plan, hits)
        audit_by_id = {item['chunk_id']: item for item in audit}
        candidates = [{'rank': rank, **hit, 'filter': audit_by_id[hit['chunk_id']]}
                      for rank, hit in enumerate(hits, 1)]
        probes = [{'topic': topic, 'ranks': [row['rank'] for row in candidates
                                           if topic.casefold() in row['title'].casefold()]}
                  for topic in topics]
        results.append({'top_k': depth, 'returned_count': len(hits),
                        'retained_count': sum(row['retained'] for row in audit),
                        'candidates': candidates, 'topic_title_probes': probes,
                        'experimental_top5': reranked})
    return {'experiment_version': VERSION, 'patient_id': state.patient_id,
            'query_variant': 'symptom_only' if symptom_only else 'live_focused_baseline',
            'plan': plan, 'depths': results,
            'limitations': ['No labels loaded; topics are user-selected title probes, not ground truth.',
                            'Concept filtering/reranking does not establish relevance or clinical entailment.',
                            'Cosine and lexical scores are not disease probabilities.',
                            'No live diagnostic retrieval setting or index content is changed.']}

"""Deterministic, label-free semantic clustering of observed findings only.

No disease templates, diagnosis lookup, benchmark input, generated summaries or
LLM. Findings/history stay separate; unknown/negative/ordinal responses are not
turned into positive observations. Every query term keeps its source fact IDs.
"""
import math
import re
import numpy as np
from rag.agents.models import PatientState
from rag.focused_medical import concepts

VERSION = 'observed-semantic-clusters-v1'
# Wording normalization, not disease prediction. Only mapped when observed.
WORDING = {'detachment': 'feeling detached from body or surroundings',
           'numbness tingling': 'numbness and tingling',
           'hypochondrium': 'upper abdominal pain', 'flank': 'flank pain',
           'breast': 'breast pain', 'cramp': 'cramping pain'}


def observed_facts(state):
    if type(state) is not PatientState:
        raise TypeError('Only validated label-free PatientState is accepted')
    state = PatientState.model_validate(state.model_dump())
    result, ignored = [], []
    by_key = {}
    for field in ('presenting_evidence', 'symptoms', 'antecedents'):
        for i, text in enumerate(getattr(state, field)):
            ref = f'patient:{field}:{i}'
            _, separator, answer = text.partition(' = ')
            if separator and (answer.strip().lower() in {'no', 'unknown', 'none', 'nowhere'} or re.fullmatch(r'[-+]?\d+(\.\d+)?', answer.strip())):
                ignored.append({'fact_id': ref, 'text': text, 'reason': 'negative_unknown_or_uninterpreted_numeric'})
                continue
            extracted = concepts(text)
            if not extracted:
                ignored.append({'fact_id': ref, 'text': text, 'reason': 'no_informative_positive_concept'})
            for concept in extracted:
                kind = 'history' if field == 'antecedents' else 'symptoms'
                key = (kind, concept)
                if key in by_key:
                    by_key[key]['fact_ids'].append(ref)
                else:
                    fact = {'kind': kind, 'concept': concept, 'text': WORDING.get(concept, concept), 'fact_ids': [ref]}
                    result.append(fact)
                    by_key[key] = fact
    return result, ignored


def semantic_groups(facts, vectors, count, max_group_size=4):
    """Agglomerative average-link cosine grouping, bounded by query size goal.

    Count derives from number of observations, not a condition-specific rule or
    a clinical similarity cutoff. Stable tie breaks preserve reproducibility.
    """
    if not facts:
        return []
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(facts) or not np.isfinite(vectors).all():
        raise ValueError('Invalid finding embeddings')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Zero finding embedding')
    similarities = (vectors / norms) @ (vectors / norms).T
    groups = [[i] for i in range(len(facts))]
    while len(groups) > count:
        choices = [(float(similarities[np.ix_(a, b)].mean()), i, j)
                   for i, a in enumerate(groups) for j, b in enumerate(groups) if i < j and len(a) + len(b) <= max_group_size]
        if not choices:
            break
        _, i, j = max(choices, key=lambda x: (x[0], -x[1], -x[2]))
        groups[i] = sorted(groups[i] + groups[j])
        groups.pop(j)
    return [[facts[i] for i in group] for group in groups]


def build_queries(state, embedding):
    facts, ignored = observed_facts(state)
    queries = []
    population = 'child' if state.age is not None and state.age < 18 else 'older adult' if state.age is not None and state.age >= 65 else 'adult' if state.age is not None else ''
    # One batch, no embeddings of identifiers, labels, negations, or numeric codes.
    vectors = embedding.embed_texts([f['text'] for f in facts]) if facts else []
    for kind in ('symptoms', 'history'):
        positions = [i for i, f in enumerate(facts) if f['kind'] == kind]
        subset = [facts[i] for i in positions]
        count = min(4, math.ceil(len(subset) / 4)) if kind == 'symptoms' else min(6, len(subset))
        groups = semantic_groups(subset, [vectors[i] for i in positions], count) if subset else []
        for group in groups:
            # Demographics belong to explicit history background, not every
            # symptom query; avoid having 'pediatric female' dominate all search.
            text = '; '.join(f['text'] for f in group)
            if kind == 'history':
                text = ' '.join(filter(None, (population, text, 'symptoms and clinical information')))
            queries.append({'query_id': f'{kind}:{len(queries)}', 'kind': kind, 'text': text,
                            'facts': group, 'fact_ids': list(dict.fromkeys(ref for f in group for ref in f['fact_ids']))})
    return {'version': VERSION, 'population': population, 'queries': queries, 'excluded_from_positive_queries': ignored,
            'limitations': ['Semantic clustering is not medical interpretation.',
                           'Unlisted observations are unknown; numeric scales remain uninterpreted.',
                           'History is background information, not a diagnosis of the current episode.']}

"""Broad dense/BM25 candidate generation -> RRF -> local learned reranking.

No pre-reranking lexical gate, no cosine cutoff, no benchmark dependencies.
All ranks/scores are retrieval diagnostics, never probabilities or diagnoses.
"""
from collections import Counter
import math
import re
from time import perf_counter
import numpy as np
from rag.focused_medical import STOP, matches
from .queries import build_queries

RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RERANKER_REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
VERSION = 'task74-qdrant-hybrid-v2'


def tokens(text):
    # Conservative morphological normalization, shared by document and query.
    result = []
    for token in re.findall(r'[a-z]+', text.lower()):
        if token in STOP or len(token) < 3:
            continue
        if token.endswith('s') and not token.endswith(('ss', 'is', 'us')) and len(token) > 4:
            token = token[:-1]
        result.append(token)
    return result


class BM25:
    """Okapi BM25 (k1=1.5, b=.75), positive Robertson IDF; no new dependency."""
    def __init__(self, texts):
        self.counts = [Counter(c) for c in map(tokens, texts)]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(self.lengths)) or 1
        df = Counter(t for c in self.counts for t in c)
        self.idf = {t: math.log(1 + (len(self.counts) - n + .5) / (n + .5)) for t, n in df.items()}

    def search(self, query, depth):
        if type(depth) is not int or depth < 1:
            raise ValueError('Positive candidate depth required')
        terms = set(tokens(query))
        scores = []
        for i, counts in enumerate(self.counts):
            score = sum(self.idf[t] * counts[t] * 2.5 /
                        (counts[t] + 1.5 * (.25 + .75 * self.lengths[i] / self.average))
                        for t in terms if counts[t])
            if score > 0:  # zero means no lexical term match, not a relevance threshold
                scores.append((i, score))
        return sorted(scores, key=lambda x: (-x[1], x[0]))[:depth]


def fuse(rankings, records, *, k=60):
    if type(k) is not int or k < 1:
        raise ValueError('Positive RRF constant required')
    pool = {}
    for query_id, channel, hits in rankings:
        seen = set()
        for rank, (i, score) in enumerate(hits, 1):
            if i in seen or not math.isfinite(score):
                raise ValueError('Duplicate/nonfinite ranked hit')
            seen.add(i)
            item = pool.setdefault(i, {'index': i, 'chunk_id': records[i]['chunk_id'], 'rrf_score': 0., 'origins': []})
            item['rrf_score'] += 1 / (k + rank)
            item['origins'].append({'query_id': query_id, 'channel': channel, 'rank': rank, 'raw_score': score})
    return sorted(pool.values(), key=lambda r: (-r['rrf_score'], r['chunk_id']))


class LocalCrossEncoder:
    def __init__(self, *, allow_download=False, batch_size=2):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Positive reranker batch size required')
        from sentence_transformers import CrossEncoder
        import torch
        try:
            self.model = CrossEncoder(RERANKER_MODEL, revision=RERANKER_REVISION,
                local_files_only=not allow_download, trust_remote_code=False,
                device='cpu', max_length=512, activation_fn=torch.nn.Identity())
        except Exception as exc:
            raise RuntimeError('Local reranker unavailable. Explicitly use --allow-reranker-download once; no fallback was used.') from exc
        self.batch_size = batch_size
        self.signature = {'model': RERANKER_MODEL, 'revision': RERANKER_REVISION,
                          'max_length': 512, 'score': 'raw cross-encoder logit; not clinical confidence'}
        self.truncated_pairs = 0

    def score(self, pairs):
        if not pairs:
            return []
        for query, passage in pairs:
            length = len(self.model.tokenizer.encode(query, passage, truncation=False))
            self.truncated_pairs += int(length > 512)
        values = np.asarray(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)).reshape(-1)
        if len(values) != len(pairs) or not np.isfinite(values).all():
            raise ValueError('Invalid learned reranker scores')
        return values.tolist()


def choose_final(ranked, records, queries, final_k):
    """Model relevance first, then diversity; never fill weak slots.

    Neutral MS-MARCO logit is an experimental decision, NOT a calibrated medical
    threshold. Concept features rank; they never pre-filter the candidate pool.
    """
    selected, parents, texts = [], set(), []
    docs = Counter()
    def take(kind, limit):
        eligible = [r for r in ranked if r.get(kind + '_score') is not None]
        eligible.sort(key=lambda r: (-r.get(kind + '_ranking_score', r[kind + '_score']), -r['rrf_score'], r['chunk_id']))
        for row in eligible:
            if len(selected) >= limit:
                break
            record = records[row['index']]
            doc = record['source']['document_id']
            text_tokens = set(tokens(record['text']))
            if row[kind + '_score'] <= 0:
                continue
            if record['parent_id'] in parents or docs[doc] >= 2:
                continue
            if any(len(text_tokens & previous) / max(1, len(text_tokens | previous)) >= .8 for previous in texts):
                continue
            selected.append({**row, 'selection_role': kind, 'record': record})
            parents.add(record['parent_id'])
            docs[doc] += 1
            texts.append(text_tokens)
    if any(q['kind'] == 'history' for q in queries) and final_k > 1:
        take('history', 1)
    take('symptoms', final_k)
    take('history', final_k)
    return sorted(selected, key=lambda row: row['selection_role'] == 'history')



class HybridRetriever:
    def __init__(self, index, embedding, reranker):
        self.index, self.embedding, self.reranker = index, embedding, reranker
        self.lexical = BM25(r['retrieval_text'] for r in index.records)

    def retrieve(self, state, *, candidate_depth=50, final_k=5):
        if candidate_depth not in (50, 100) or type(candidate_depth) is not int:
            raise ValueError('Experiment candidate depth must be 50 or 100')
        if type(final_k) is not int or not 0 <= final_k <= 5:
            raise ValueError('Final evidence limit must be 0..5')
        started = perf_counter()
        generated = build_queries(state, self.embedding)
        queries = generated['queries']
        query_seconds = perf_counter() - started
        if not queries or not final_k:
            return {'version': VERSION, 'plan': generated, 'candidate_depth': candidate_depth,
                    'candidate_pool': [], 'reranked': [], 'final_passages': [], 'channel_rankings': [],
                    'timings': {'total_seconds': perf_counter() - started}, 'reason': 'no_positive_queries_or_zero_evidence_budget'}
        search_start = perf_counter()
        vectors = self.embedding.embed_texts(q['text'] for q in queries)
        rankings = []
        for q, vector in zip(queries, vectors):
            rankings.append((q['query_id'], 'dense', self.index.search(vector, candidate_depth)))
            rankings.append((q['query_id'], 'bm25', self.lexical.search(q['text'], candidate_depth)))
        pool = fuse(rankings, self.index.records)
        # Only exact source-span duplicates are removed before reranking.
        # Do not discard a different window merely because its parent is shared.
        unique, parents, deduplicated = [], set(), []
        for item in pool:
            parent = ' '.join(self.index.records[item['index']]['window_text'].split()).casefold()
            if parent not in parents:
                unique.append(item)
                parents.add(parent)
            else:
                deduplicated.append({'chunk_id': item['chunk_id'], 'reason': 'duplicate_source_span'})
        candidates = unique
        search_seconds = perf_counter() - search_start
        rerank_start = perf_counter()
        pairs, locations = [], []
        for i, row in enumerate(candidates):
            # Rerank against the strongest supporting query in each role, not a
            # giant patient string. Query selection is based on channel ranks.
            strengths = Counter()
            for origin in row['origins']:
                strengths[origin['query_id']] += 1 / (60 + origin['rank'])
            for kind in ('symptoms', 'history'):
                eligible = [q for q in queries if q['kind'] == kind]
                if not eligible:
                    continue
                best = max(eligible, key=lambda q: (strengths[q['query_id']], q['query_id']))
                # Even candidates absent from a role's channels can be assessed:
                # choose its nearest semantic query, rather than silently filter.
                if not strengths[best['query_id']]:
                    best = max(eligible, key=lambda q: float(np.dot(vectors[queries.index(q)], self.index.vectors[row['index']])))
                pairs.append((best['text'], self.index.records[row['index']]['retrieval_text']))
                locations.append((i, kind, best['query_id']))
        before = getattr(self.reranker, 'truncated_pairs', 0)
        scores = self.reranker.score(pairs)
        if len(scores) != len(locations) or any(not math.isfinite(s) for s in scores):
            raise ValueError('Missing/invalid reranking output; no evidence delivered')
        scored = [{**r, 'symptoms_score': None, 'history_score': None, 'reranker_queries': {}} for r in candidates]
        for score, (i, kind, query_id) in zip(scores, locations):
            scored[i][kind + '_score'] = score
            scored[i]['reranker_queries'][kind] = query_id
        concepts_by_kind = {kind: list(dict.fromkeys(f['concept'] for q in queries if q['kind'] == kind for f in q['facts'])) for kind in ('symptoms', 'history')}
        for r in scored:
            record = self.index.records[r['index']]
            r['semantic_score'] = max(float(np.dot(v, self.index.vectors[r['index']])) for v in vectors)
            r['concept_features'] = {}
            population = generated.get('population', '')
            r['demographic_feature'] = bool(population and any(t in tokens(record['retrieval_text']) for t in ({'child', 'children', 'pediatric'} if population == 'child' else {'older', 'elderly'} if population == 'older adult' else {'adult'})))
            for kind, observed in concepts_by_kind.items():
                overlap = [c for c in observed if matches(c, record['retrieval_text'])]
                r['concept_features'][kind] = overlap
                if r[kind + '_score'] is not None:
                    # All sources satisfy the unchanged trusted-source contract.
                    r[kind + '_ranking_score'] = r[kind + '_score'] + .5 * len(overlap) / max(1, len(observed)) + .1 * r['semantic_score'] + .1 * r['rrf_score'] + .2 * r['demographic_feature']
            r['reranker_score'] = max(s for s in (r['symptoms_score'], r['history_score']) if s is not None)
        ranked = sorted(scored, key=lambda r: (-r['reranker_score'], -r['rrf_score'], r['chunk_id']))
        fusion_ranks = {r['chunk_id']: i for i, r in enumerate(candidates, 1)}
        for rank, row in enumerate(ranked, 1):
            row['fusion_rank'] = fusion_ranks[row['chunk_id']]
            row['reranked_rank'] = rank
        final = choose_final(ranked, self.index.records, queries, final_k)
        selected_ids = {r['chunk_id'] for r in final}
        removals = [{'chunk_id': r['chunk_id'], 'reason': 'nonpositive_model_relevance' if r['reranker_score'] <= 0 else 'diversity_or_final_budget'} for r in ranked if r['chunk_id'] not in selected_ids]
        rerank_seconds = perf_counter() - rerank_start
        return {'version': VERSION, 'candidate_depth': candidate_depth, 'final_limit': final_k, 'plan': generated,
                'channel_rankings': [{'query_id': q, 'channel': channel,
                    'hits': [{'chunk_id': self.index.records[i]['chunk_id'], 'score': s} for i, s in hits]}
                    for q, channel, hits in rankings],
                'raw_union_count': len(pool), 'parent_union_count': len(unique),
                'fused_candidates': pool, 'deduplication_removed': deduplicated, 'final_selection_removed': removals,
                'statistics': {'dense_candidate_count': sum(len(h) for _, c, h in rankings if c == 'dense'),
                    'bm25_candidate_count': sum(len(h) for _, c, h in rankings if c == 'bm25'),
                    'fused_candidate_count': len(pool), 'reranked_count': len(ranked),
                    'deduplicated_count': len(deduplicated), 'pre_rerank_filter_loss': 0,
                    'final_evidence_count': len(final),
                    'rrf_duplicate_occurrences': sum(len(h) for _, _, h in rankings) - len(pool)},
                'candidate_pool': candidates, 'reranked': ranked, 'final_passages': final,
                'pre_reranking_lexical_filter': False, 'reranker': self.reranker.signature,
                'reranker_pair_count': len(pairs),
                'reranker_truncated_pairs': getattr(self.reranker, 'truncated_pairs', 0) - before,
                'timings': {'query_construction_seconds': query_seconds, 'candidate_generation_seconds': search_seconds,
                            'reranking_selection_seconds': rerank_seconds, 'total_seconds': perf_counter() - started},
                'limitations': ['No relevance labels or clinical probabilities were assigned.',
                    'Positive general-domain cross-encoder logit required at final selection; not clinically calibrated.',
                    'One qualified history passage prioritized; additional history may fill unused symptom slots.',
                    'Context index attaches full source sections; Qdrant path preserves original evidence chunks.',
                    'Only original Qdrant chunks can enter existing snapshots; context index is preview only.']}

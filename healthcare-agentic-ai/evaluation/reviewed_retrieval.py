"""Reviewed pooled retrieval evaluation, separate from production and diagnosis.

No automatic relevance assignment: builders initialize uncertain/unreviewed.
Metrics use explicit reviewer judgments, never titles, scores or dataset labels.
"""
import copy
import hashlib
import json
from typing import Literal
from pydantic import Field, model_validator
from rag.agents.models import PatientState, StrictModel
from rag.medical_ingestion.models import MedicalChunk, validate_chunk
from dataclasses import fields

SCHEMA_VERSION = 'medical-relevance-benchmark-v1'
PATIENT_ID = 'ddxplus:validate:2'
TOPICS = ('Panic Disorder', 'Asthma', 'Asthma in Children')
SYSTEMS = ('current_query', 'symptom_only_query', 'lexical_reranker')
RELEVANCE_DEFINITION = ("Does this medical passage provide useful medical information "
                       "for interpreting one or more of the patient's observed findings "
                       "or differential categories?")
SystemName = Literal['current_query', 'symptom_only_query', 'lexical_reranker']
Relevance = Literal['relevant', 'partially_relevant', 'not_relevant', 'uncertain']
REVIEW_FIELDS = {'relevance', 'reviewed', 'reviewer_id', 'reviewer_notes'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def pool_digest(data):
    """Detect accidental edits outside the four manually editable review fields."""
    value = copy.deepcopy(data)
    value.pop('pool_sha256', None)
    value['candidates'] = [{k: v for k, v in c.items() if k not in REVIEW_FIELDS}
                           for c in value['candidates']]
    return digest(value)


class Ranking(StrictModel):
    original_bge_rank: int = Field(ge=1, le=50)
    cosine_similarity: float = Field(ge=-1.000001, le=1.000001)
    lexical_reranker_rank: int | None
    lexical_score: int | None


class Candidate(StrictModel):
    chunk_id: str
    title: str
    source: str
    text: str
    source_metadata: dict
    origins: list[str]
    rankings: dict[SystemName, Ranking | None]
    relevance: Relevance
    reviewed: bool
    reviewer_id: str | None
    reviewer_notes: str | None = None

    @model_validator(mode='after')
    def validate_candidate(self):
        core = {'chunk_id': self.chunk_id, 'title': self.title, 'source': self.source, 'text': self.text}
        if core.keys() & self.source_metadata.keys():
            raise ValueError('Duplicate source fields')
        validate_chunk({**self.source_metadata, **core})
        if set(self.rankings) != set(SYSTEMS):
            raise ValueError('Every candidate requires all three ranking slots, null when absent')
        if self.reviewed:
            if self.reviewer_id is None or not self.reviewer_id.strip():
                raise ValueError('Manual review requires an explicit reviewer identifier')
        elif self.relevance != 'uncertain' or self.reviewer_id is not None:
            raise ValueError('Unreviewed candidates must remain uncertain with no reviewer identifier')
        if len(self.origins) != len(set(self.origins)) or not self.origins:
            raise ValueError('Candidate must have unique pool origins')
        return self


class System(StrictModel):
    query: str
    captured_depth: int = Field(ge=1)
    returned_count: int = Field(ge=0)
    parent_system: Literal['current_query'] | None


class Benchmark(StrictModel):
    schema_version: Literal['medical-relevance-benchmark-v1']
    benchmark_id: Literal['ddxplus-validate-2-medical-relevance-v1']
    relevance_definition: str
    review_instructions: list[str]
    patient_state: PatientState
    source_report: dict
    index_snapshot: dict
    requested_topics: dict[str, list[str]]
    systems: dict[SystemName, System]
    candidates: list[Candidate] = Field(min_length=1)
    pool_sha256: str

    @model_validator(mode='after')
    def validate_pool(self):
        if self.patient_state.patient_id != PATIENT_ID:
            raise ValueError('This benchmark is restricted to validation case 2')
        if self.relevance_definition != RELEVANCE_DEFINITION:
            raise ValueError('Relevance definition must not change during review')
        if set(self.systems) != set(SYSTEMS) or set(self.requested_topics) != set(TOPICS):
            raise ValueError('Incorrect benchmark systems or requested topics')
        ids = [c.chunk_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate candidate chunk ID')
        for name, system in self.systems.items():
            depth = 5 if name == 'lexical_reranker' else 50
            parent = 'current_query' if name == 'lexical_reranker' else None
            if system.captured_depth != depth or system.parent_system != parent:
                raise ValueError('Incorrect captured depth or reranker parent')
            ranks = []
            for candidate in self.candidates:
                rank = candidate.rankings[name]
                if rank is None:
                    continue
                if name == 'lexical_reranker':
                    original = candidate.rankings['current_query']
                    if (rank.lexical_reranker_rank is None or rank.lexical_score is None or
                            rank.lexical_score < 0 or original is None or
                            rank.original_bge_rank != original.original_bge_rank or
                            rank.cosine_similarity != original.cosine_similarity):
                        raise ValueError('Reranker must preserve current-query BGE provenance')
                    ranks.append(rank.lexical_reranker_rank)
                else:
                    if rank.lexical_reranker_rank is not None or rank.lexical_score is not None:
                        raise ValueError('BGE rank cannot be a lexical rank')
                    ranks.append(rank.original_bge_rank)
            if (len(ranks) != system.returned_count or system.returned_count > depth or
                    sorted(ranks) != list(range(1, system.returned_count + 1))):
                raise ValueError('Ranks must be unique, contiguous, and match captured counts')
        if self.systems['lexical_reranker'].query != self.systems['current_query'].query:
            raise ValueError('Reranker query must equal its current-query parent')
        for title, topic_ids in self.requested_topics.items():
            actual = {c.chunk_id for c in self.candidates if c.title == title}
            if len(topic_ids) != len(set(topic_ids)) or set(topic_ids) != actual:
                raise ValueError('Requested topic manifest must match the candidate pool')
        for candidate in self.candidates:
            expected = {name for name in SYSTEMS if candidate.rankings[name] is not None}
            if candidate.title in TOPICS:
                expected.add('requested_topic:' + candidate.title)
            if set(candidate.origins) != expected:
                raise ValueError('Candidate origins do not match source rankings/topic manifest')
        if self.pool_sha256 != pool_digest(self.model_dump()):
            raise ValueError('Non-review benchmark content changed; restore the original pool')
        return self


def build_benchmark(report, indexed_chunks, state, *, report_name, report_sha256):
    """Pool actual saved rankings plus every exact-title indexed topic chunk.

    indexed_chunks must be a snapshot read from the existing medical collection,
    not a title search scored as relevance or an unverified corpus export.
    """
    state = PatientState.model_validate(state.model_dump())
    if state.patient_id != PATIENT_ID or report.get('labels_loaded') is not False:
        raise ValueError('Label-free validation case 2 report required')
    if report.get('cloud_calls') != 0 or report.get('live_workflow_changed') is not False:
        raise ValueError('Expected the offline, unchanged-workflow retrieval report')
    index = {}
    for item in indexed_chunks:
        chunk = validate_chunk(item)
        if chunk['chunk_id'] in index:
            raise ValueError('Duplicate indexed chunk')
        index[chunk['chunk_id']] = chunk
    if len(index) != report['medical_chunk_count']:
        raise ValueError('Retrieval report/index count mismatch')
    candidates = {}
    chunk_fields = {f.name for f in fields(MedicalChunk)}

    def add(chunk, origin):
        identifier = chunk['chunk_id']
        if index.get(identifier) != chunk:
            raise ValueError('Saved candidate does not match the current indexed payload')
        if identifier not in candidates:
            candidates[identifier] = {
                **{k: chunk[k] for k in ('chunk_id', 'title', 'source', 'text')},
                'source_metadata': {k: v for k, v in chunk.items() if k not in {'chunk_id', 'title', 'source', 'text'}},
                'origins': [], 'rankings': {name: None for name in SYSTEMS},
                'relevance': 'uncertain', 'reviewed': False,
                'reviewer_id': None, 'reviewer_notes': None}
        if origin not in candidates[identifier]['origins']:
            candidates[identifier]['origins'].append(origin)
        return candidates[identifier]

    from rag.focused_medical import query_plan
    plan = query_plan(state)
    expected_queries = {
        'current_query': plan['query'],
        'symptom_only_query': '; '.join(plan['symptom_concepts']) + ' symptoms clinical information'}
    systems, current_top = {}, None
    for variant, name in [('live_focused_baseline', 'current_query'), ('symptom_only', 'symptom_only_query')]:
        experiments = [e for e in report['experiments'] if e['query_variant'] == variant]
        if len(experiments) != 1 or experiments[0]['patient_id'] != PATIENT_ID:
            raise ValueError('One case-2 experiment for each query variant is required')
        experiment = experiments[0]
        if (experiment.get('experiment_version') != 'medical-depth-experiment-v1' or
                experiment['plan']['query'] != expected_queries[name]):
            raise ValueError('Saved query/version does not match the label-free patient query formulation')
        depths = [d for d in experiment['depths'] if d['top_k'] == 50]
        if len(depths) != 1:
            raise ValueError('Actual top-50 rankings required; do not infer them from top-5')
        top = depths[0]
        if top['returned_count'] != len(top['candidates']):
            raise ValueError('Incorrect saved candidate count')
        for row in top['candidates']:
            chunk = validate_chunk({k: v for k, v in row.items() if k in chunk_fields})
            candidate = add(chunk, name)
            if candidate['rankings'][name] is not None:
                raise ValueError('Duplicate saved ranking entry')
            candidate['rankings'][name] = {'original_bge_rank': row['rank'], 'cosine_similarity': row['score'],
                                          'lexical_reranker_rank': None, 'lexical_score': None}
        systems[name] = {'query': experiment['plan']['query'], 'captured_depth': 50,
                         'returned_count': top['returned_count'], 'parent_system': None}
        if name == 'current_query':
            current_top = top
    for rank, row in enumerate(current_top['experimental_top5'], 1):
        hit = row['hit']
        chunk = validate_chunk({k: v for k, v in hit.items() if k in chunk_fields})
        candidate = add(chunk, 'lexical_reranker')
        original = candidate['rankings']['current_query']
        if original is None or original['cosine_similarity'] != hit['score'] or candidate['rankings']['lexical_reranker'] is not None:
            raise ValueError('Invalid lexical candidate provenance')
        candidate['rankings']['lexical_reranker'] = {**original, 'lexical_reranker_rank': rank,
                                                    'lexical_score': row['lexical_score']}
    systems['lexical_reranker'] = {'query': systems['current_query']['query'], 'captured_depth': 5,
                                    'returned_count': len(current_top['experimental_top5']), 'parent_system': 'current_query'}
    requested = {title: [] for title in TOPICS}
    for chunk in index.values():
        if chunk['title'] in requested:
            add(chunk, 'requested_topic:' + chunk['title'])
            requested[chunk['title']].append(chunk['chunk_id'])
    data = {
        'schema_version': SCHEMA_VERSION, 'benchmark_id': 'ddxplus-validate-2-medical-relevance-v1',
        'relevance_definition': RELEVANCE_DEFINITION,
        'review_instructions': [
            'Pending independent manual review. No relevance labels have been assigned automatically.',
            'Read the label-free patient findings and full passage; do not consult the DDXPlus diagnosis label.',
            'Do not assign relevance from titles, cosine/lexical scores or ranks. These are raw retrieval metadata only.',
            'Edit only relevance, reviewed, reviewer_id, reviewer_notes. To record a manual judgment set reviewed=true and supply reviewer_id.',
            'Uncertain may be a reviewed judgment or an unreviewed placeholder; reviewed distinguishes them.',
            'Requested topic inclusion is candidate pooling, not a diagnostic hypothesis or a relevance judgment.',
            'Prefer a rank-blinded review copy and record conflicts/uncertainty in reviewer_notes.'],
        'patient_state': state.model_dump(),
        'source_report': {'filename': report_name, 'sha256': report_sha256, 'created_at': report['created_at'],
                          'labels_loaded': False, 'cloud_calls': 0},
        'index_snapshot': {'chunk_count': len(index), 'embedding_signature': report['embedding_signature'],
                           'manifest_sha256': report['manifest_sha256'],
                           'payload_sha256': digest([index[k] for k in sorted(index)])},
        'requested_topics': {title: sorted(ids) for title, ids in requested.items()},
        'systems': systems,
        # UUID order is independent of relevance, title and retrieval ranking.
        'candidates': [candidates[k] for k in sorted(candidates)]}
    data['pool_sha256'] = pool_digest(data)
    return Benchmark.model_validate(data)


def summarize(benchmark):
    """Single-case, manually judged POOL recall and truncated MRR only.

    Partial review is explicit; unreviewed and uncertain are not negative labels.
    Denominators are known positives in this pool, never whole-corpus recall.
    """
    benchmark = Benchmark.model_validate(benchmark.model_dump())
    reviewed = [c for c in benchmark.candidates if c.reviewed]
    counts = {label: sum(c.relevance == label for c in reviewed)
              for label in ('relevant', 'partially_relevant', 'not_relevant', 'uncertain')}
    complete = len(reviewed) == len(benchmark.candidates) and counts['uncertain'] == 0
    result = {
        'schema_version': 'medical-relevance-summary-v1', 'patient_id': benchmark.patient_state.patient_id,
        'pool_sha256': benchmark.pool_sha256,
        'raw_retrieval': {name: system.model_dump() for name, system in benchmark.systems.items()},
        'manual_review': {'candidate_count': len(benchmark.candidates), 'reviewed_count': len(reviewed),
                          'unreviewed_count': len(benchmark.candidates) - len(reviewed),
                          'judgment_counts': counts, 'complete_decisive_review': complete,
                          'independent_review_verified': False},
        'derived_retrieval_metrics': None,
        'metric_status': 'withheld_no_manual_review' if not reviewed else 'manual_review_available',
        'clinical_diagnostic_conclusions': None,
        'limitations': [
            'Candidate-pool recall only, not corpus-wide recall; explicitly requested topics affect the denominator.',
            'Primary positives are manually relevant; inclusive sensitivity analysis also counts partially_relevant.',
            'Unreviewed/uncertain judgments are excluded, not treated as negatives. Incomplete-review metrics are provisional and biased.',
            'One patient only: MRR equals reciprocal rank for this one query per system, truncated to its captured depth.',
            'The lexical system is a fixed top-5 output: recall at 10/20/50 uses the same five results, not an inferred deeper ranking.',
            'Reviewer identity is recorded, not authenticated; independent review is not automatically verified.',
            'Raw ranking, human judgments, and derived metrics are distinct. No diagnostic accuracy or clinical validity is claimed.']}
    if not reviewed:
        return result
    result['derived_retrieval_metrics'] = {}
    result['metric_status'] = 'complete_pool_review' if complete else 'provisional_incomplete_pool_review'
    for mode, positive_labels in [('strict_relevant', {'relevant'}),
                                   ('inclusive_relevant_or_partial', {'relevant', 'partially_relevant'})]:
        positives = {c.chunk_id for c in reviewed if c.relevance in positive_labels}
        mode_result = {'positive_pool_count': len(positives), 'systems': {}}
        for name, system in benchmark.systems.items():
            ranks = []
            for candidate in benchmark.candidates:
                ranking = candidate.rankings[name]
                if candidate.chunk_id in positives and ranking is not None:
                    ranks.append(ranking.lexical_reranker_rank if name == 'lexical_reranker' else ranking.original_bge_rank)
            mode_result['systems'][name] = {
                'status': 'computed_from_manual_pool_judgments' if positives else 'withheld_no_positive_judgments',
                'captured_depth': system.captured_depth,
                'recall_at_k': {str(k): sum(r <= min(k, system.captured_depth) for r in ranks) / len(positives)
                                if positives else None for k in (5, 10, 20, 50)},
                'mrr': (1 / min(ranks) if ranks else 0.0) if positives else None,
                'effective_cutoffs': {str(k): min(k, system.captured_depth) for k in (5, 10, 20, 50)}}
        result['derived_retrieval_metrics'][mode] = mode_result
    return result

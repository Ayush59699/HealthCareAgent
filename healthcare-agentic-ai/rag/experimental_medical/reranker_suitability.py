"""Task 7.8 only: one biomedical cross-encoder, unchanged pairs/gate/selector.

No production imports this module. Raw logits across differently trained models
are not calibrated relevance probabilities; a zero-gate change alone cannot prove
medical suitability. Model metadata is inspected before downloading any weights.
"""
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

MODEL = 'ncbi/MedCPT-Cross-Encoder'
MODEL_REVISION = '71caf65d4927987813984f54c284405a13fcca49'
VERSION = 'task78-reranker-suitability-v1'
MAX_WEIGHT_BYTES = 500_000_000
MAX_PARAMETERS = 125_000_000


def read_remote(url, limit=2_000_000):
    """Bound metadata reads; never fetch a weight through this function."""
    with urlopen(url, timeout=30) as response:
        value = response.read(limit + 1)
    if len(value) > limit:
        raise ValueError('Model metadata exceeds bounded read limit')
    return value


def estimate_model(config, weight_bytes):
    """Conservative BERT-base FP32 load estimate, not an allocation guarantee."""
    if (config.get('model_type') != 'bert'
            or config.get('architectures') != ['BertForSequenceClassification']
            or config.get('num_labels', len(config.get('id2label', {}))) != 1
            or config.get('max_position_embeddings') != 512):
        raise ValueError('Expected a 512-token scalar BERT relevance cross-encoder')
    h, layers, vocab = (config[k] for k in ('hidden_size', 'num_hidden_layers', 'vocab_size'))
    if (h, layers, config.get('intermediate_size'), config.get('num_attention_heads')) != (768, 12, 3072, 12):
        raise ValueError('Only the selected BERT-base architecture is permitted')
    parameters = (vocab + 512 + config.get('type_vocab_size', 2)) * h + 2*h
    parameters += layers * (12*h*h + 13*h) + h*h + 2*h + 1
    if not 0 < weight_bytes <= MAX_WEIGHT_BYTES or parameters > MAX_PARAMETERS:
        raise ValueError('Unsafe/large model; no weight download permitted')
    # Two copies of FP32 weights during loading plus 1.25 GB for framework,
    # tokenizer and batch-2/512-token attention/workspaces. Sequential models only.
    reserve = 2 * max(weight_bytes, parameters * 4) / 1e9 + 1.25
    return {'approximate_parameters': parameters, 'weight_bytes': weight_bytes,
            'fp32_weight_gb': parameters * 4 / 1e9, 'estimated_incremental_peak_gb': reserve,
            'estimate_basis': 'two FP32 weight copies + 1.25 GB framework/batch-2 workspace; CPU; no concurrent models'}


def require_headroom(used_gb, estimate):
    if used_gb + estimate['estimated_incremental_peak_gb'] >= 13.5:
        raise RuntimeError('Estimated model memory violates unchanged 13.5 GB synchronous budget; stop')


def inspect_model(sample=lambda: None):
    """Resolve ONE repository revision before scoring; fetch only small metadata."""
    sample()
    info = json.loads(read_remote(f'https://huggingface.co/api/models/{MODEL}/revision/{MODEL_REVISION}?blobs=true'))
    revision = info['sha']
    if revision != MODEL_REVISION:
        raise ValueError('Immutable model revision required')
    base = f'https://huggingface.co/{MODEL}/resolve/{revision}/'
    raw = read_remote(base + 'config.json')
    config = json.loads(raw)
    files = {f['rfilename']: f for f in info['siblings']}
    weight = 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    if weight not in files or type(files[weight].get('size')) is not int:
        raise ValueError('Unverified model weight size; stop before download')
    estimate = estimate_model(config, files[weight]['size'])
    sample()
    return {'model': MODEL, 'revision': revision, 'weight_file': weight,
            'weight_lfs_sha256': files[weight].get('lfs', {}).get('sha256'),
            'files': {name: row.get('size') for name, row in files.items()},
            'config': config, 'memory_estimate': estimate,
            'selection_rationale': 'BERT-base biomedical query-article relevance cross-encoder, rather than an embedding or untrained classifier; NCBI MedCPT. Single preselected alternative, no label-based model search.',
            'limitations': 'Biomedical literature training differs from multi-finding patient queries and consumer passages. Raw scalar score zero is not calibrated across models.'}


class BiomedicalCrossEncoder:
    """Local CPU scalar logits, no sigmoid, softmax, rescaling or fallback."""
    def __init__(self, directory, metadata):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if metadata['model'] != MODEL or not re.fullmatch('[0-9a-f]{40}', metadata['revision']):
            raise ValueError('Unexpected model identity')
        if torch.get_num_threads() != 2:
            raise ValueError('Exactly two CPU threads required')
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        network = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            use_safetensors=metadata['weight_file'].endswith('.safetensors'), weights_only=True)
        if network.config.num_labels != 1:
            raise ValueError('A single raw relevance logit is required')
        network.to('cpu').eval()
        self.model = SimpleNamespace(tokenizer=tokenizer, network=network)
        self.batch_size, self.truncated_pairs = 2, 0
        self.signature = {'model': MODEL, 'revision': metadata['revision'], 'max_length': 512,
                          'score': 'raw cross-encoder logit; not clinical confidence'}

    def score(self, pairs):
        import torch
        if not pairs:
            return []
        if len(pairs) > 2:
            raise ValueError('Batch size exceeds two')
        tokenizer = self.model.tokenizer
        if any(len(tokenizer.encode(q, p, truncation=False)) > 512 for q, p in pairs):
            raise ValueError('Frozen pair would be truncated; no scores produced')
        batch = tokenizer([q for q, _ in pairs], [p for _, p in pairs],
                          padding=True, truncation=False, return_tensors='pt')
        with torch.inference_mode():
            logits = self.model.network(**batch).logits
        if tuple(logits.shape) != (len(pairs), 1):
            raise ValueError('Unexpected scalar-logit shape')
        scores = logits[:, 0].tolist()
        if not all(math.isfinite(s) for s in scores):
            raise ValueError('Nonfinite model score')
        return scores


def download_model(metadata, directory, monitor):
    """Sequential explicit allowlist only; no repository snapshot/parallel workers."""
    from huggingface_hub import hf_hub_download
    from .prefix_integrity import digest
    estimate_model(metadata['config'], metadata['memory_estimate']['weight_bytes'])
    require_headroom(monitor.psutil.virtual_memory().used / 1e9, metadata['memory_estimate'])
    files = metadata['files']
    names = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'vocab.txt', 'tokenizer.json']
    for name in names + [metadata['weight_file']]:
        if name not in files:
            continue
        if name != metadata['weight_file'] and (type(files[name]) is not int or files[name] > 5_000_000):
            raise ValueError('Unsafe tokenizer metadata size')
        monitor.sample()
        path = hf_hub_download(MODEL, name, revision=metadata['revision'], local_dir=directory)
        if Path(path).stat().st_size != files[name]:
            raise ValueError('Downloaded file size differs from pinned metadata')
        if name == metadata['weight_file']:
            expected = metadata['weight_lfs_sha256']
            if not expected or digest(path, monitor.sample) != expected:
                raise ValueError('Weight SHA-256 mismatch/missing metadata')
        monitor.sample()
    return directory


from time import perf_counter
from orchestration.evidence import EvidenceSnapshot, fingerprint
from .ranking_ablation import ROLES, ablate
from .reranker_input_experiment import check_token_budget, score_pairs, apply_scores
from .reranker_prefix_control import pair_plan, assert_score_only, text_sha256


def run_condition(state, experiment, records, reranker, condition,
                  sample_memory=lambda: None, progress=lambda done, total: None):
    if condition not in ('control', 'experimental') or reranker.batch_size != 2:
        raise ValueError('Only the two fixed conditions at batch two are allowed')
    if condition == 'control' and reranker.signature != experiment['reranker']:
        raise ValueError('Control identity changed')
    if condition == 'experimental' and (reranker.signature['model'] != MODEL
            or not re.fullmatch('[0-9a-f]{40}', reranker.signature['revision'])
            or reranker.signature['max_length'] != 512):
        raise ValueError('Only the pinned biomedical alternative is allowed')
    frozen_hash = fingerprint(experiment)
    started = perf_counter()
    pairs, audit, formatted = pair_plan(state, experiment, records, 'original')
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
    result.update(version=VERSION, variant=condition,
        description='Reranker replacement only; original queries, positive gate and selector unchanged.',
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


def compare_pairs(control, experimental, experiment):
    for key in ('pair_order_fingerprint', 'frozen_pool_fingerprint', 'original_scores_fingerprint'):
        if control[key] != experimental[key]:
            raise ValueError('Not the same frozen pool/pairs/scores')
    a = {r['chunk_id']: r for r in control['candidate_audit']}
    b = {r['chunk_id']: r for r in experimental['candidate_audit']}
    if set(a) != set(b):
        raise ValueError('Comparison candidate IDs changed')
    transitions = {role: {name: 0 for name in ('nonpositive_to_positive', 'positive_to_nonpositive',
                    'positive_to_positive', 'nonpositive_to_nonpositive')} for role in ROLES}
    role_ranks = []
    for result in (control, experimental):
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
            if y['reranker_input'] != x['reranker_input']:
                raise ValueError('Experimental query differs from verbatim control')
            positive_a, positive_b = x['raw_reranker_score'] > 0, y['raw_reranker_score'] > 0
            direction = ('positive' if positive_a else 'nonpositive') + '_to_' + ('positive' if positive_b else 'nonpositive')
            transitions[role][direction] += 1
            changes.append({'chunk_id': cid, 'title': old['title'], 'section': old['section'],
                'source': old['source'], 'url': old['url'], 'document_id': old['document_id'],
                'evidence_text_sha256': old['evidence_text_sha256'], 'passage_sha256': x['passage_sha256'],
                'control_query': x['reranker_input'], 'experimental_query': y['reranker_input'],
                'original_query_sha256': x['original_query_sha256'], 'query_id': x['originating_query_id'],
                'role': role, 'supporting_fact_ids': x['fact_ids'],
                'dense_rank': x['dense_rank'], 'bm25_rank': x['bm25_rank'],
                'rrf_rank': old['fusion_rank'], 'rrf_score': old['rrf_score'],
                'control_raw_logit': x['raw_reranker_score'], 'experimental_raw_logit': y['raw_reranker_score'],
                'logit_delta': y['raw_reranker_score'] - x['raw_reranker_score'],
                'control_eligible': positive_a, 'experimental_eligible': positive_b,
                'eligibility_transition': direction,
                'control_reranker_rank': old['reranked_rank'], 'experimental_reranker_rank': new['reranked_rank'],
                'control_role_ranking_rank': role_ranks[0][cid, role],
                'experimental_role_ranking_rank': role_ranks[1][cid, role],
                'control_role_ranking_score': x['task74_ranking_score'],
                'experimental_role_ranking_score': y['task74_ranking_score'],
                'control_removal_reason': old['final_selection_reason'],
                'experimental_removal_reason': new['final_selection_reason'],
                'control_role_events': [e for e in old['selector_events'] if e['role'] == role],
                'experimental_role_events': [e for e in new['selector_events'] if e['role'] == role],
                'control_selected': old['selected'], 'experimental_selected': new['selected'],
                'control_selected_role': old['selected_role'], 'experimental_selected_role': new['selected_role'],
                'control_selected_for_role': old['selected_role'] == role,
                'experimental_selected_for_role': new['selected_role'] == role})
    if len(changes) != experiment['reranker_pair_count']:
        raise ValueError('Incomplete pair comparison')
    return {'eligibility_transitions': transitions, 'pair_count': len(changes), 'pairs': changes,
            'selected_gained': sorted(set(experimental['selected_ids']) - set(control['selected_ids'])),
            'selected_lost': sorted(set(control['selected_ids']) - set(experimental['selected_ids']))}



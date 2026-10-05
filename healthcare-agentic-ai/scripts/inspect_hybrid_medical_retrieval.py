"""Task 7.4 local-only retrieval inspection; no benchmark/diagnosis labels.

Run through scripts/run_memory_bounded.py. Uses an isolated COPY of existing
Qdrant medical_knowledge by default. --index enables the optional context-index
preview, which cannot be injected into agent evidence without a schema migration.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--depths', type=int, choices=[50, 100], nargs='+', default=[50, 100])
    parser.add_argument('--index', type=Path, help='Optional context index, preview only')
    parser.add_argument('--baseline-index', type=Path, default=DATA_ROOT / 'qdrant')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--topic', action='append', default=[], help='Exact-title diagnostic AFTER retrieval only')
    parser.add_argument('--allow-reranker-download', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2, choices=[1, 2, 4])
    parser.add_argument('--output', type=Path, help='NEW output directory; refuses overwrite')
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    output = args.output or Path(__file__).resolve().parents[1] / 'outputs/task7.4' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if output.exists() or any(s < 1 for s in args.samples) or len(set(args.samples)) != len(args.samples) or not 1 <= args.threads <= 4:
        raise ValueError('New output path, distinct positive samples and 1..4 threads required')
    if len(set(args.depths)) != len(args.depths):
        raise ValueError('Distinct candidate depths required')
    if args.index is not None and not (args.index / 'index.json').is_file():
        raise ValueError('Context index not available; omit --index to use existing Qdrant')
    if not (args.baseline_index / 'medical-embedding.json').is_file():
        raise ValueError('Existing medical Qdrant index required')
    output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(args.threads)
    from rag.experimental_medical.index import ExperimentalIndex
    from rag.experimental_medical.qdrant_index import QdrantMedicalIndex, evidence_hits
    from rag.experimental_medical.retrieval import HybridRetriever, LocalCrossEncoder
    from rag.experimental_medical.inspection import directory_hashes, topic_trace, markdown_report
    from rag.focused_medical import state_from_patient, query_plan, baseline_filter_candidates
    from rag.medical_embeddings import MedicalEmbeddingModel
    from rag.medical_retriever import MedicalKnowledgeRetriever
    from rag.patient_parser import DDXPlusParser
    from rag.agents.grounding import evidence_from_hits, context
    from orchestration.evidence import FrozenEvidence, EvidenceSnapshot, fingerprint

    load_start = perf_counter()
    embedding = MedicalEmbeddingModel(allow_download=False, batch_size=args.batch_size)
    reranker = LocalCrossEncoder(allow_download=args.allow_reranker_download, batch_size=args.batch_size)
    before = directory_hashes(args.baseline_index)
    selected, cases = set(args.samples), []
    with tempfile.TemporaryDirectory(prefix='task74-baseline-') as temporary:
        snapshot = Path(temporary) / 'qdrant-copy'
        shutil.copytree(args.baseline_index, snapshot)
        if directory_hashes(snapshot) != before or directory_hashes(args.baseline_index) != before:
            raise ValueError('Baseline changed while copying; stop writers and retry')
        with MedicalKnowledgeRetriever(embedding_model=embedding, storage_path=snapshot) as medical:
            index = ExperimentalIndex(args.index, embedding.signature) if args.index else QdrantMedicalIndex(medical)
            hybrid = HybridRetriever(index, embedding, reranker)
            load_seconds = perf_counter() - load_start
            for i, record in enumerate(DDXPlusParser(args.data_dir).iter_patients('validate', limit=max(selected), include_labels=False), 1):
                if i not in selected:
                    continue
                state = state_from_patient(record.patient, record.patient_id)
                plan = query_plan(state)
                start = perf_counter()
                top5 = medical.retrieve(plan['query'], top_k=5) if plan['symptom_concepts'] else []
                _, audit = baseline_filter_candidates(plan, top5)
                base_seconds = perf_counter() - start
                start = perf_counter()
                broad = medical.retrieve(plan['query'], top_k=100) if plan['symptom_concepts'] else []
                _, broad_audit = baseline_filter_candidates(plan, broad)
                def annotated(hits, audits):
                    by_id = {a['chunk_id']: a for a in audits}
                    return [{'rank': j, **h, 'filter': by_id[h['chunk_id']]} for j, h in enumerate(hits, 1)]
                baseline = {'query': plan['query'], 'top5_seconds': base_seconds,
                    'broad_probe_seconds': perf_counter() - start,
                    'top5': annotated(top5, audit), 'broad_candidates': annotated(broad, broad_audit)}
                existing_titles = {r['title'] for r in baseline['top5'] if r['filter']['retained']}
                experiments = []
                for depth in args.depths:
                    experiment = hybrid.retrieve(state, candidate_depth=depth)
                    experiment['topic_traces'] = [topic_trace(t, index.records, baseline, experiment) for t in args.topic]
                    experiment['new_final_titles_vs_baseline'] = list(dict.fromkeys(r['record']['source']['title'] for r in experiment['final_passages'] if r['record']['source']['title'] not in existing_titles))
                    pool_ids = {r['chunk_id'] for r in experiment['candidate_pool']}
                    experiment['candidate_records'] = [r for r in index.records if r['chunk_id'] in pool_ids]
                    if args.index is None:
                        evidence = evidence_from_hits(evidence_hits(experiment), 'medical_knowledge')
                        context(state, [], evidence)  # Unchanged grounding provenance revalidation.
                        frozen = EvidenceSnapshot(snapshot_id=fingerprint({'patient_cases': [], 'medical_knowledge': [e.model_dump() for e in evidence]}),
                            patient_cases=(), medical_knowledge=tuple(FrozenEvidence.freeze(e) for e in evidence))
                        experiment['evidence_snapshot'] = frozen.model_dump(mode='json')
                    experiments.append(experiment)
                    # A completed depth survives interruption; never resume from
                    # an unverified partial result or overwrite old runs.
                    (output / f'validate-{i}-depth-{depth}.json').write_text(json.dumps({'patient_state': state.model_dump(), 'baseline': baseline, 'experiment': experiment}, indent=2), encoding='utf8')
                    print(f'{record.patient_id} depth={depth}: ' + ', '.join(r['record']['source']['title'] for r in experiment['final_passages']), flush=True)
                    print('  statistics: ' + str(experiment.get('statistics')), flush=True)
                    print('  seconds: ' + str(experiment['timings']), flush=True)
                cases.append({'patient_state': state.model_dump(), 'baseline': baseline, 'experiments': experiments})
    if len(cases) != len(selected) or directory_hashes(args.baseline_index) != before:
        raise ValueError('Missing case or changed production baseline')
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'labels_loaded': False,
        'benchmark_used': False, 'cloud_inference_calls': 0, 'agents_called': False,
        'production_baseline_unchanged': True, 'baseline_file_hashes': before,
        'configuration': {'samples': args.samples, 'depths': args.depths, 'threads': args.threads,
            'batch_size': args.batch_size, 'embedding': embedding.signature, 'reranker': reranker.signature,
            'experimental_index': index.manifest, 'model_index_load_seconds': load_seconds,
            'final_evidence_limit': 5, 'pre_reranking_lexical_filter': False,
            'topic_probes_post_retrieval_only': args.topic}, 'cases': cases}
    (output / 'inspection.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    (output / 'inspection.md').write_text(markdown_report(report), encoding='utf8')
    print('Report: ' + str(output))
    print('Background knowledge is not diagnosis. No agents or safety policies changed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

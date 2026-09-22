"""Run bounded Phase 5 research orchestration on label-free validation patients."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import PROJECT_ROOT, OpenAIConfig, PatientRAGConfig, load_generation_env
from rag.patient_parser import DDXPlusParser
from rag.patient_rag import PatientCaseRAG
from rag.medical_retriever import MedicalKnowledgeRetriever
from rag.llm.provider import OpenAIProvider
from orchestration import Orchestrator, WorkflowPolicy
from run_phase4 import write_json_atomic


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--queries', type=int, default=1)
    cli.add_argument('--top-k', type=int, default=1)
    cli.add_argument('--max-requests', type=int, default=21)
    cli.add_argument('--max-seconds', type=float, default=900.0)
    cli.add_argument('--max-request-bytes', type=int, default=7_000_000)
    cli.add_argument('--data-dir', type=Path)
    cli.add_argument('--patient-storage-path', type=Path)
    cli.add_argument('--medical-storage-path', type=Path)
    cli.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'outputs/phase5' /
                     datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = cli.parse_args(argv)
    if min(args.queries, args.top_k, args.max_requests, args.max_request_bytes) < 1:
        cli.error('Query counts, top-k and request budgets must be positive')
    try:
        policy = WorkflowPolicy(max_requests=args.max_requests, max_seconds=args.max_seconds,
                                max_request_bytes=args.max_request_bytes)
    except ValueError:
        cli.error('Time budget must be positive and finite')
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        load_generation_env()
        config = OpenAIConfig()
        parser = DDXPlusParser(args.data_dir)
        patient_config = PatientRAGConfig()
        if args.patient_storage_path:
            patient_config = replace(patient_config, storage_path=args.patient_storage_path)
        with ExitStack() as stack:
            patients = stack.enter_context(PatientCaseRAG(patient_config))
            medical = stack.enter_context(MedicalKnowledgeRetriever(storage_path=args.medical_storage_path))
            counts = {'patient_cases': patients.vector_store.count(), 'medical_chunks': medical.store.count()}
            if not all(counts.values()):
                raise ValueError('Prebuilt indexes are required')
            provider = stack.enter_context(OpenAIProvider(config))
            orchestrator = Orchestrator(provider, patients, medical, top_k=args.top_k, policy=policy)
            summaries = []
            for i, record in enumerate(parser.iter_patients('validate', limit=args.queries, include_labels=False), 1):
                result = orchestrator.run(record.patient, record.patient_id)
                write_json_atomic(args.output_dir / f'sample_case_{i:03d}.json', result.model_dump_json(indent=2))
                summaries.append({'run_id': result.run_id, 'patient_id': result.patient_id,
                    'status': result.status, 'outcome': result.outcome, 'revision_count': result.revision_count,
                    'total_requests': result.total_requests, 'provider_repairs': result.provider_repairs,
                    'seconds': result.seconds, 'failure': result.failure.model_dump() if result.failure else None})
        report = {'schema_version': 'phase5-report-v1', 'model': config.model, 'split': 'validate',
            'index_counts': counts, 'retrieval_top_k': args.top_k, 'policy': policy.model_dump(),
            'cases': summaries, 'requested_cases': args.queries,
            'disclaimer': 'Research orchestration only; no clinical approval, HITL or effectiveness claim.'}
        write_json_atomic(args.output_dir / 'phase5_run_report.json', json.dumps(report, indent=2))
        print(json.dumps({'output_dir': str(args.output_dir), **report}, indent=2))
        # Abstained/blocked/unresolved are valid technical completions, not approval.
        return 0 if len(summaries) == args.queries and all(s['status'] != 'failed' for s in summaries) else 1
    except Exception:
        write_json_atomic(args.output_dir / 'phase5_run_report.json', json.dumps({
            'status': 'setup_failure', 'message': 'Check deployment settings, dataset and prebuilt indexes; details omitted.'}))
        logging.error('Phase 5 setup failed; sensitive details omitted')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

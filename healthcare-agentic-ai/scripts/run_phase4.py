"""Run GPT-5.6-Sol roles on validation features; evaluate only after inference finishes."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import logging
import os
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import PROJECT_ROOT, OpenAIConfig, PatientRAGConfig, load_generation_env
from rag.patient_parser import DDXPlusParser
from rag.patient_rag import PatientCaseRAG
from rag.medical_retriever import MedicalKnowledgeRetriever
from rag.llm.provider import OpenAIProvider
from rag.phase4 import Phase4Pipeline
from rag.phase4_evaluation import evaluate_case, summarize
from rag.agents.prompts import PROMPT_VERSION


def write_json_atomic(path: Path, content: str) -> None:
    """Keep the previous complete file intact if a write is interrupted."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf8') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def stage_observations(output):
    """Report skipped stages explicitly, without inventing zero-duration successes."""
    observations = {}
    for name in ('patient', 'patient_case', 'medical_knowledge', 'diagnostic', 'critic'):
        attempted = name in output.stage_seconds
        accepted = output.structured_validity.get(name, output.retrieval_success.get(name, False))
        observations[name] = {
            'status': ('success' if accepted else 'failure') if attempted else 'skipped',
            'seconds': output.stage_seconds.get(name),
            'attempts': output.attempts.get(name, 0) if name in ('patient', 'diagnostic', 'critic') else None,
            'retries': max(0, output.attempts.get(name, 0) - 1) if name in ('patient', 'diagnostic', 'critic') else None,
            'structured_valid': output.structured_validity.get(name),
        }
    return observations


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--queries', type=int, default=1)
    cli.add_argument('--top-k', type=int, default=1)
    cli.add_argument('--diagnostic-top-k', type=int, default=5)
    cli.add_argument('--data-dir', type=Path)
    cli.add_argument('--patient-storage-path', type=Path)
    cli.add_argument('--medical-storage-path', type=Path)
    cli.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'outputs/phase4' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = cli.parse_args(argv)
    if min(args.queries, args.top_k, args.diagnostic_top_k) < 1:
        cli.error('Query count and top-k values must be positive')
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    failure_stage = 'setup'
    try:
        load_generation_env()
        config = OpenAIConfig()
        parser = DDXPlusParser(args.data_dir)
        patient_config = PatientRAGConfig()
        if args.patient_storage_path:
            patient_config = replace(patient_config, storage_path=args.patient_storage_path)
        runner_started = time.perf_counter()
        with ExitStack() as stack:
            patients = stack.enter_context(PatientCaseRAG(patient_config))
            medical = stack.enter_context(MedicalKnowledgeRetriever(storage_path=args.medical_storage_path))
            if not patients.vector_store.count() or not medical.store.count():
                raise ValueError('Both prebuilt indexes must be nonempty')
            counts = {'patient_cases': patients.vector_store.count(), 'medical_chunks': medical.store.count()}
            signatures = {'patient': patients.embedding_model.signature, 'medical': medical.embedding.signature}
            provider = stack.enter_context(OpenAIProvider(config))
            pipeline = Phase4Pipeline(provider, patients, medical, top_k=args.top_k)
            outputs = []
            truncation_before = {'patient': patients.embedding_model.truncated_texts, 'medical': medical.embedding.truncated_texts}
            for i, record in enumerate(parser.iter_patients('validate', limit=args.queries, include_labels=False), 1):
                result = pipeline.run(record.patient, record.patient_id)
                outputs.append(result)
                write_json_atomic(args.output_dir / f'sample_case_{i:03d}.json', result.model_dump_json(indent=2))
            embedding_truncation = {'patient': patients.embedding_model.truncated_texts - truncation_before['patient'],
                                    'medical': medical.embedding.truncated_texts - truncation_before['medical']}
        failure_stage = 'evaluation'
        # Reopen validation stream with labels only AFTER every agent has finished.
        labeled = {record.patient_id: record.labels for record in parser.iter_patients('validate', limit=args.queries, include_labels=True)}
        cases = [evaluate_case(output, labeled[output.patient_id], top_k=args.diagnostic_top_k) for output in outputs]
        report = {'configuration': asdict(config), 'prompt_version': PROMPT_VERSION,
                  'sampling': {'requested_temperature': config.temperature, 'temperature_sent': False,
                               'effective_temperature': 'deployment_default',
                               'note': 'This deployment rejects temperature; zero cannot be enforced.'},
                  'split': 'validate', 'retrieval_top_k': args.top_k, 'diagnostic_top_k': args.diagnostic_top_k,
                  'index_counts': counts, 'embedding_signatures': signatures,
                  'metrics': summarize(outputs, cases), 'evaluation': cases,
                  'runner_seconds_including_setup_and_evaluation': time.perf_counter() - runner_started,
                  'embedding_query_truncation_counts': embedding_truncation,
                  'stage_observations': [{'patient_id': o.patient_id, 'stage_seconds': o.stage_seconds,
                      'pipeline_seconds': o.seconds, 'stages': stage_observations(o),
                      'retrieval_success': o.retrieval_success, 'structured_validity': o.structured_validity,
                      'attempts': o.attempts, 'retries': {k: max(0, v - 1) for k, v in o.attempts.items()},
                      'retrieved_patient_cases': len(o.patient_case_evidence),
                      'retrieved_medical_chunks': len(o.medical_knowledge_evidence),
                      'failure': o.failure.model_dump() if o.failure else None,
                      'llm_telemetry': o.llm_telemetry} for o in outputs]}
        write_json_atomic(args.output_dir / 'phase4_run_report.json', json.dumps(report, indent=2))
        print(json.dumps({'output_dir': str(args.output_dir), 'metrics': report['metrics']}, indent=2))
        return 0 if len(outputs) == args.queries and all(o.status == 'success' for o in outputs) else 1
    except Exception:
        # Reports/logs must not expose raw rows or model response fragments.
        write_json_atomic(args.output_dir / 'phase4_run_report.json', json.dumps({'status': failure_stage + '_failure',
            'message': 'Check API credentials/settings, local dataset and prebuilt index availability/signatures.'}))
        logging.error('Phase 4 setup/evaluation failed; sensitive details omitted')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

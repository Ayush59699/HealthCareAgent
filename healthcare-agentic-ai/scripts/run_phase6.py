"""Run isolated Phase 6 research output-safety validation; explicit pending review, never clinical clearance."""
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import PROJECT_ROOT, OpenAIConfig, PatientRAGConfig, load_generation_env
from rag.patient_parser import DDXPlusParser
from rag.patient_rag import PatientCaseRAG
from rag.amg import AMGMedicalEvidence
from rag.selector_evidence_service import BACKEND
from application.workflow import create_workflow
from application.decision import build_final_decision
from application.human_review import offer_human_review
from application.resources import prepare_runtime
from rag.llm.provider import OpenAIProvider
from orchestration.phase6 import Phase6Orchestrator, Phase6Policy
from safety.models import POLICY_VERSION, PROMPT_VERSION
from orchestration.phase6.diagnostic import PROMPT_VERSION as DIAGNOSTIC_PROMPT_VERSION, CRITIC_PROMPT_VERSION
from orchestration.phase6.reference_entailment import VERSION as REFERENCE_VERSION, inspect_reference_consistency


def write_json_atomic(path, content):
    # Kept local to avoid importing Phase 4's evaluation-bearing runner.
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf8') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def case_summary(state):
    assessment = state.safety_assessments[-1] if state.safety_coverage == 'assessed' else None
    invocations = [i for i in state.invocations if i.ticket.stage == 'SAFETY_VALIDATION']
    return {
        'run_id': state.run_id, 'patient_id': state.patient_id, 'status': state.status, 'outcome': state.outcome,
        'medical_backend': state.medical_retrieval.get('backend') if state.medical_retrieval else None,
        'medical_source_ids': [e.source_id for e in state.evidence.medical_knowledge] if state.evidence else [],
        'reference_consistency_audits': [
            {'diagnostic_version': d.version, 'diagnostic_fingerprint': d.fingerprint,
             'audit': inspect_reference_consistency(d.result, state.patient_state)}
            for d in state.diagnostics] if state.medical_retrieval is not None else [],
        'revision_count': state.revision_count, 'total_requests': state.total_requests,
        'provider_repairs': state.provider_repairs, 'seconds': state.seconds,
        'failure': state.failure.model_dump() if state.failure else None,
        'latest_diagnostic_version': state.diagnostics[-1].version if state.diagnostics else None,
        'safety_coverage': state.safety_coverage, 'safety_skip_reason': state.safety_skip_reason,
        'safety_decision': assessment.decision if assessment else None,
        'safety_ticket': assessment.ticket.model_dump() if assessment else None,
        'semantic_status': assessment.semantic_status if assessment else None,
        'finding_counts': dict(Counter(f.code for f in assessment.findings)) if assessment else {},
        'safety_requests': sum(i.attempts for i in invocations),
        'safety_provider_repairs': sum(i.provider_repairs for i in invocations),
        'safety_seconds': sum(value for key, value in state.stage_seconds.items() if key.startswith('SAFETY')),
        'eligible_research_output': state.status == 'final' and assessment is not None and assessment.decision == 'CONTINUE',
        'withheld': state.status != 'final', 'clinical_approval': False, 'human_review_scheduled': False,
    }


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--queries', type=int, default=1)
    cli.add_argument('--top-k', type=int, default=1, help='Patient-case top-k')
    cli.add_argument('--medical-top-k', type=int, default=5)
    cli.add_argument('--max-requests', type=int, default=21)
    cli.add_argument('--max-seconds', type=float, default=900.0)
    cli.add_argument('--max-request-bytes', type=int, default=7_000_000)
    cli.add_argument('--data-dir', type=Path)
    cli.add_argument('--patient-storage-path', type=Path)
    cli.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'outputs/phase6' /
                     datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = cli.parse_args(argv)
    if min(args.queries, args.top_k, args.medical_top_k, args.max_requests, args.max_request_bytes) < 1 or max(args.top_k, args.medical_top_k) > 20:
        cli.error('Query counts, top-k and request budgets must be positive')
    try:
        policy = Phase6Policy(max_requests=args.max_requests, max_seconds=args.max_seconds,
                              max_request_bytes=args.max_request_bytes)
    except ValueError:
        cli.error('Time budget must be positive and finite')
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        logging.error('Phase 6 requires a new writable output directory; details omitted')
        return 1
    summaries = []
    try:
        load_generation_env()
        config = OpenAIConfig()
        parser = DDXPlusParser(args.data_dir)
        patient_config = PatientRAGConfig()
        if args.patient_storage_path:
            patient_config = replace(patient_config, storage_path=args.patient_storage_path)
        prepare_runtime(patient_config)
        with ExitStack() as stack:
            patients = stack.enter_context(PatientCaseRAG(patient_config))
            medical = stack.enter_context(AMGMedicalEvidence())
            counts = {'patient_cases': patients.vector_store.count(), 'medical_chunks': medical.count()}
            if not all(counts.values()):
                raise ValueError('Prebuilt indexes are required')
            provider = stack.enter_context(OpenAIProvider(config))
            orchestrator = create_workflow(provider, patients, medical, top_k=args.top_k, policy=policy, medical_top_k=args.medical_top_k)
            for i, record in enumerate(parser.iter_patients('validate', limit=args.queries, include_labels=False), 1):
                result = orchestrator.run(record.patient, record.patient_id)
                write_json_atomic(args.output_dir / f'sample_case_{i:03d}.json', result.model_dump_json(indent=2))
                if orchestrator.evidence.combined is not None:
                    write_json_atomic(args.output_dir / f'combined_evidence_{i:03d}.json',
                                      orchestrator.evidence.combined.model_dump_json(indent=2))
                decision = build_final_decision(result)
                decision_name = f'final_decision_{i:03d}.json'
                write_json_atomic(args.output_dir / decision_name, decision.model_dump_json(indent=2))
                human = offer_human_review(result, args.output_dir / 'human_reviews')
                summaries.append({**case_summary(result), 'human_review': human.model_dump(mode='json') if human else None, 'final_decision_file': decision_name,
                    'final_decision_status': decision.status, 'human_review_required': decision.human_review_required,
                    'review_state': decision.review_state})
        report = {
            'schema_version': 'phase6-report-v1', 'model': config.model, 'split': 'validate',
            'index_counts': counts, 'retrieval_top_k': args.top_k, 'medical_candidate_top_k': args.medical_top_k,
            'medical_retrieval_policy': BACKEND, 'diagnostic_prompt_version': DIAGNOSTIC_PROMPT_VERSION,
            'critic_prompt_version': CRITIC_PROMPT_VERSION,
            'patient_reference_check_version': REFERENCE_VERSION, 'policy': policy.model_dump(),
            'safety_policy_version': POLICY_VERSION, 'safety_prompt_version': PROMPT_VERSION,
            'cases': summaries, 'requested_cases': args.queries,
            'sampling': {'temperature_sent': False, 'effective_temperature': 'deployment_default'},
            'disclaimer': 'Research only. CONTINUE is not clinical clearance. AI restrictions remain final. Local human decisions are recorded separately and never override Safety; no review scheduling is implemented.',
        }
        write_json_atomic(args.output_dir / 'phase6_run_report.json', json.dumps(report, indent=2))
        print(json.dumps({'output_dir': str(args.output_dir), **report}, indent=2))
        # Research completion includes withheld outcomes, never clinical approval.
        return 0 if len(summaries) == args.queries and all(s['status'] != 'failed' for s in summaries) else 1
    except Exception:
        write_json_atomic(args.output_dir / 'phase6_run_report.json', json.dumps({
            'schema_version': 'phase6-report-v1', 'status': 'setup_or_execution_failure', 'cases': summaries,
            'message': 'Check deployment, dataset, indexes and safety contracts; sensitive details omitted.'}))
        logging.error('Phase 6 setup/execution failed; sensitive details omitted')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

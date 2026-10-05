"""CHECK2 evaluation only: label-free diversity sample; unchanged production workflow.

Run once with --run, then replay saved traces with --summarize. No relevance labels,
clinical scoring, retrieval experiments, extra LLM calls, or production patches.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    from scripts.run_phase6 import write_json_atomic
    write_json_atomic(path, json.dumps(value, indent=2, ensure_ascii=False))


def features(record):
    """Recorded facts only; no diagnosis taxonomy or outcome is inspected."""
    patient = record.patient
    values = {('fact', e.question, e.value) for e in patient.symptoms + patient.antecedents}
    values.add(('sex', patient.sex))
    values.add(('age_decade', None if patient.age is None else patient.age // 10))
    return values


def select_cases(records, count=10):
    """Anchor CHECK1, then greedy max-min Jaccard diversity; ties use row order."""
    if count < 1 or len(records) < count:
        raise ValueError('Insufficient sample or invalid count')
    if any(record.labels is not None for record in records):
        raise ValueError('Evaluation requires label-free records')
    selected = [0]
    sets = [features(record) for record in records]
    while len(selected) < count:
        def distance(index):
            return min(1 - len(sets[index] & sets[j]) / len(sets[index] | sets[j]) for j in selected)
        selected.append(max((i for i in range(len(records)) if i not in selected), key=distance))
    return [records[i] for i in selected]


def run(output):
    from rag.config import OpenAIConfig, PatientRAGConfig, load_generation_env
    from rag.patient_parser import DDXPlusParser
    from rag.patient_rag import PatientCaseRAG
    from rag.amg import AMGMedicalEvidence
    from rag.llm.provider import OpenAIProvider
    from application.workflow import create_workflow
    from application.resources import prepare_runtime
    from application.decision import build_final_decision
    from orchestration.phase6 import Phase6Policy
    from scripts.run_phase6 import case_summary
    import psutil

    live = output / 'live'
    live.mkdir(exist_ok=False)  # No accidental reruns or overwrites.
    parser = DDXPlusParser()
    pool = list(parser.iter_patients('validate', limit=100, include_labels=False))
    selected = select_cases(pool)
    save(output / 'selection.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'First validation row (CHECK1 anchor), then greedy max-min Jaccard over exact question/value facts, sex and age decade; ties in row order.',
        'pool_size': len(pool), 'include_labels': False,
        'limitation': 'Purposive feature-diversity sample of first 100 rows, not a random or population-representative clinical benchmark; anchor analyzed separately.',
        'pool': [{'patient_id': r.patient_id, 'patient': r.to_inference_dict()} for r in pool],
        'selected_ids': [r.patient_id for r in selected],
        'selection_frozen_before_retrieval': True,
        'usefulness_measure': 'Unique accepted medical source IDs cited in hypothesis claims; utilization proxy only, not medical relevance/entailment.',
        'no_manual_labels': True,
    })
    load_generation_env()
    config = OpenAIConfig()
    patient_config = PatientRAGConfig()
    policy = Phase6Policy()
    prepare_runtime(patient_config)
    report = {'model': config.model, 'policy': policy.model_dump(), 'requested_cases': 10,
              'split': 'validate', 'include_labels': False, 'cases': [],
              'sampling': {'temperature_sent': False, 'effective_temperature': 'deployment_default'}}
    with ExitStack() as stack:
        patients = stack.enter_context(PatientCaseRAG(patient_config))
        medical = stack.enter_context(AMGMedicalEvidence())
        provider = stack.enter_context(OpenAIProvider(config))
        report['index_counts'] = {'patient_cases': patients.vector_store.count(), 'medical_chunks': medical.count()}
        if not all(report['index_counts'].values()):
            raise ValueError('Prebuilt indexes required')
        report['retrieval'] = {'patient_top_k': 1, 'medical_top_k': 5, 'max_distance': medical.max_distance,
                               'backend': medical.backend, 'snapshot': medical.snapshot}
        workflow = create_workflow(provider, patients, medical, top_k=1, medical_top_k=5, policy=policy)
        for i, record in enumerate(selected, 1):
            if psutil.virtual_memory().available < 2_000_000_000:
                raise RuntimeError('Less than 2 GB available memory; stopped before next case')
            start = time.perf_counter()
            print(f'START {i}/10 {record.patient_id}', flush=True)
            state = workflow.run(record.patient, record.patient_id)
            save(live / f'sample_case_{i:03d}.json', state.model_dump(mode='json'))
            decision = build_final_decision(state)
            save(live / f'final_decision_{i:03d}.json', decision.model_dump(mode='json'))
            report['cases'].append({**case_summary(state), 'final_decision_status': decision.status,
                'wall_seconds_including_final_decision': time.perf_counter() - start,
                'process_rss_bytes_after_case': psutil.Process().memory_info().rss,
                'system_available_bytes_after_case': psutil.virtual_memory().available})
            save(live / 'phase6_run_report.json', report)
            print(f'END {i}/10 {state.status} {decision.status} requests={state.total_requests}', flush=True)
    return summarize(output)


def citation_usage(grounding, accepted_ids):
    """Inventory mentions/critic citations are not diagnostic claim citations."""
    cited = {ref for claim in grounding['claims'] for ref in claim['medical_refs']} & set(accepted_ids)
    return sorted(cited), sorted(set(accepted_ids) - cited)


def summarize(output):
    from orchestration.phase6.state import Phase6WorkflowState
    from orchestration.phase6.grounding import grounding_report
    from application.decision import build_final_decision
    from scripts.verify_amg_run import verify

    rows = []
    for path in sorted((output / 'live').glob('sample_case_*.json')):
        state = Phase6WorkflowState.model_validate_json(path.read_text(encoding='utf8'))
        final_path = path.with_name(path.name.replace('sample_case_', 'final_decision_'))
        decision = build_final_decision(state)
        cases, medical = state.evidence.agent_evidence() if state.evidence else ([], [])
        accepted_ids = [e.source_id for e in medical]
        versions = []
        for version in state.diagnostics:
            grounding = grounding_report(version.result, state.patient_state, cases, medical).model_dump()
            cited, unused = citation_usage(grounding, accepted_ids)
            versions.append({'version': version.version, 'grounding': grounding, 'cited_medical_ids': cited,
                'unused_medical_ids': unused, 'hypotheses_without_medical_references': len(grounding['hypotheses_without_medical_references'])})
        latest = versions[-1] if versions else None
        ever_cited = sorted({ref for v in versions for ref in v['cited_medical_ids']})
        assessment = state.safety_assessments[-1] if state.safety_coverage == 'assessed' else None
        critique = state.critiques[-1].result.model_dump() if state.critiques else None
        findings = [f.model_dump() for f in assessment.findings] if assessment else []
        row = {'patient_id': state.patient_id, 'trace': str(path.relative_to(output)),
            'accepted_passages': len(medical), 'retrieved_titles': [e.metadata['title'] for e in medical],
            'accepted_ids': accepted_ids, 'diagnostic_versions': versions,
            'any_diagnostic_passage_used': bool(ever_cited), 'ever_cited_medical_ids': ever_cited,
            'cited_medical_passages': len(latest['cited_medical_ids']) if latest else 0,
            'unused_passages_latest': len(latest['unused_medical_ids']) if latest else len(medical),
            'unused_passages_all_versions': len(set(accepted_ids) - set(ever_cited)),
            'hypotheses_without_medical_references': latest['hypotheses_without_medical_references'] if latest else None,
            'grounding_status': latest['grounding']['structural_status'] if latest else 'not_reached',
            'medical_entailment': latest['grounding']['medical_entailment'] if latest else 'not_assessed',
            'critic_assessment': critique['overall_assessment'] if critique else None,
            'critic_safety_flags': critique['safety_flags'] if critique else [], 'critic': critique,
            'safety_decision': assessment.decision if assessment else None,
            'safety_coverage': state.safety_coverage, 'safety_skip_reason': state.safety_skip_reason,
            'safety_findings': findings, 'safety_semantic_status': assessment.semantic_status if assessment else None,
            'final_decision': decision.status, 'workflow_status': state.status, 'outcome': state.outcome,
            'technically_completed': state.failure is None and state.completed_at is not None and state.status not in ('failed', 'running'),
            'all_stages_reached': bool(state.patient_state and state.evidence and state.diagnostics and state.critiques and assessment),
            'seconds': state.seconds, 'stage_seconds': state.stage_seconds,
            'cloud_requests': state.total_requests, 'provider_repairs': state.provider_repairs,
            'requests_by_stage': dict(Counter({stage: sum(i.attempts for i in state.invocations if i.ticket.stage == stage)
                for stage in {i.ticket.stage for i in state.invocations}})),
            'invocations': [i.model_dump() for i in state.invocations],
            'failure': state.failure.model_dump() if state.failure else None,
            'abstained': state.status == 'abstained',
            'verification': verify(path, final_path) if state.evidence and state.patient_state and state.medical_retrieval else {'status': 'not_verifiable'},
        }
        rows.append(row)
    aggregate = {
        'cases': len(rows), 'technically_completed': sum(r['technically_completed'] for r in rows),
        'all_stages_reached': sum(r['all_stages_reached'] for r in rows),
        'cases_with_accepted_passages': sum(r['accepted_passages'] > 0 for r in rows),
        'cases_with_cited_passages_proxy_only': sum(r['cited_medical_passages'] > 0 for r in rows),
        'medically_useful_cases': None, 'medically_useful_cases_reason': 'Not measured: citation use is not relevance or entailment; no manual/hidden labels permitted.',
        'accepted_passages': sum(r['accepted_passages'] for r in rows),
        'cited_passages_latest': sum(r['cited_medical_passages'] for r in rows),
        'unused_passages_latest': sum(r['unused_passages_latest'] for r in rows),
        'unused_passages_all_versions': sum(r['unused_passages_all_versions'] for r in rows),
        'zero_citation_cases': sum(r['cited_medical_passages'] == 0 for r in rows),
        'zero_citation_cases_with_diagnostic': sum(bool(r['diagnostic_versions']) and r['cited_medical_passages'] == 0 for r in rows),
        'final_decisions': dict(Counter(r['final_decision'] for r in rows)),
        'safety_decisions': dict(Counter(r['safety_decision'] or 'not_assessed' for r in rows)),
        'cases_with_missing_reference_findings': sum(any(f['code'] == 'missing_medical_reference' for f in r['safety_findings']) for r in rows),
        'missing_reference_direct_block_cases': sum(any(f['code'] == 'missing_medical_reference' and f['disposition'] == 'BLOCK' for f in r['safety_findings']) for r in rows),
        'critic_flag_block_cases': sum(any(f['code'] == 'critic_safety_flags' and f['disposition'] == 'BLOCK' for f in r['safety_findings']) for r in rows),
        'cloud_requests': sum(r['cloud_requests'] for r in rows), 'seconds': sum(r['seconds'] for r in rows),
    }
    save(output / 'metrics.json', {'aggregate': aggregate, 'cases': rows})
    print(json.dumps(aggregate, indent=2), flush=True)
    return aggregate


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    cli.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/multi-case-evaluation')
    args = cli.parse_args()
    for key, value in {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'HF_HUB_DISABLE_TELEMETRY': '1', 'ANONYMIZED_TELEMETRY': 'False', 'RUN_GPT_INTEGRATION': '0'}.items():
        os.environ[key] = value
    if args.run:
        run(args.output_dir)
    else:
        summarize(args.output_dir)

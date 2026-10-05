"""Small sequential label-free WHO/AMG experiment; no label-based selection/scoring."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.who_amg_experiment import create_experimental_workflow, evidence_summary
from rag.combined_medical_evidence import CombinedEvidenceService, MODES
from rag.agents.grounding import state_from_patient, claims
from rag.who_lookup import WHOLookup
from rag.config import PROJECT_ROOT


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--mode', choices=(*MODES, 'all'), required=True)
    cli.add_argument('--queries', type=int, default=1, help='First 1-3 consecutive validation cases; no label selection')
    cli.add_argument('--live', action='store_true', help='Explicitly enable sequential cloud agent calls')
    cli.add_argument('--data-dir', type=Path, help='DDXPlus data directory')
    cli.add_argument('--who-data-dir', type=Path, default=PROJECT_ROOT / 'data')
    cli.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'outputs/who-amg-experiment' /
                     datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = cli.parse_args(argv)
    if not 1 <= args.queries <= 3:
        cli.error('Use 1-3 sequential cases for this bounded experiment')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    modes = MODES if args.mode == 'all' else (args.mode,)
    report = {'experiment': 'who-amg-v1', 'live': args.live, 'modes': modes,
              'selection': 'First consecutive validation rows; include_labels=False; no expected answers or relevance labels.',
              'limitations': ['Literal match and citation counts are not relevance, entailment or diagnostic accuracy.',
                             'WHO supplies exact bounded prefixes, not whole documents; later relevant sections may be omitted.',
                             'All arms use identical experimental source instructions; production AMG prompt remains untouched.',
                             'No safety/human-review overrides. Offline runs do not execute agents.'], 'cases': []}
    try:
        from rag.config import PatientRAGConfig, OpenAIConfig, load_generation_env
        from rag.patient_parser import DDXPlusParser
        from rag.patient_rag import PatientCaseRAG
        from rag.amg import AMGMedicalEvidence
        from application.resources import prepare_runtime
        from application.decision import build_final_decision
        from orchestration.phase6.grounding import grounding_report
        from rag.llm.provider import OpenAIProvider
        parser = DDXPlusParser(args.data_dir)
        config = PatientRAGConfig()
        prepare_runtime(config)
        with ExitStack() as stack:
            patients = stack.enter_context(PatientCaseRAG(config))
            medical = stack.enter_context(AMGMedicalEvidence()) if any(m != 'who_only' for m in modes) else None
            who = WHOLookup(args.who_data_dir) if any(m != 'amg_only' for m in modes) else None
            provider = None
            if args.live:
                load_generation_env()
                provider = stack.enter_context(OpenAIProvider(OpenAIConfig()))
            for record in parser.iter_patients('validate', limit=args.queries, include_labels=False):
                for mode in modes:
                    prefix = record.patient_id.replace(':', '_') + '_' + mode
                    if args.live:
                        workflow = create_experimental_workflow(provider, patients, medical, mode=mode, who=who)
                        state = workflow.run(record.patient, record.patient_id)
                        # Persist before projection so even a failed handoff remains auditable.
                        write(args.output_dir / (prefix + '_state.json'), state.model_dump(mode='json'))
                        service = workflow.evidence
                        decision = build_final_decision(state)
                        write(args.output_dir / (prefix + '_decision.json'), decision.model_dump(mode='json'))
                    else:
                        service = CombinedEvidenceService(patients, medical, mode=mode, who=who)
                        snapshot = service.retrieve(record.patient, state_from_patient(record.patient, record.patient_id))
                        write(args.output_dir / (prefix + '_snapshot.json'), snapshot.model_dump(mode='json'))
                    summary = {'case_id': record.patient_id, 'mode': mode,
                               **(evidence_summary(service.combined) if service.combined else {'combined_status': 'NOT_RETRIEVED'}),
                               'diagnostic_citation_counts': None, 'grounding': 'NOT_RUN', 'critic': 'NOT_RUN',
                               'safety': 'NOT_RUN', 'final_decision': 'NOT_RUN', 'failure': None}
                    if service.combined:
                        write(args.output_dir / (prefix + '_combined.json'), service.combined.model_dump(mode='json'))
                    if args.live:
                        counts = []
                        case_evidence, med = state.evidence.agent_evidence() if state.evidence else ([], [])
                        backends = {e.source_id: ('WHO' if e.metadata['backend'] == 'who-local-lookup-v1' else 'AMG') for e in med}
                        for version in state.diagnostics:
                            used = {ref for claim in claims(version.result) for ref in claim.evidence_refs}
                            count = Counter(backends[ref] for ref in used if ref in backends)
                            counts.append({'diagnostic_version': version.version, 'WHO': count['WHO'], 'AMG': count['AMG']})
                        summary.update(diagnostic_citation_counts=counts,
                            grounding=grounding_report(state.diagnostics[-1].result, state.patient_state,
                                case_evidence, med).model_dump() if state.diagnostics else 'NOT_RUN',
                            validations=[v.model_dump() for v in state.validations],
                            critic=state.critiques[-1].result.model_dump() if state.critiques else 'NOT_RUN',
                            safety=state.safety_assessments[-1].model_dump() if state.safety_assessments else 'NOT_ASSESSED',
                            safety_coverage=state.safety_coverage, safety_skip_reason=state.safety_skip_reason,
                            final_decision=decision.status, review_state=decision.review_state,
                            failure=state.failure.model_dump() if state.failure else None,
                            workflow_status=state.status, total_requests=state.total_requests)
                    report['cases'].append(summary)
                    write(args.output_dir / 'report.json', report)
                    print('Evidence Mode:', {'amg_only': 'AMG', 'who_only': 'WHO', 'who_plus_amg': 'WHO + AMG'}[mode], record.patient_id)
                    for item in summary.get('who_evidence', []):
                        print('  WHO:', item['title'], '|', item['document_type'], '|', item['url'], '| matched topic:', item['topic'])
                    for item in summary.get('amg_evidence', []):
                        print('  AMG:', item['source_id'], '|', item['title'], '|', item['url'],
                              '| snapshot:', item['provenance']['retrieval']['snapshot'])
                    print('Combined evidence status:', summary['combined_status'], '| Final decision:', summary['final_decision'])
                    # Nothing retains prior workflow/corpus text between cases except bounded summaries.
        report['status'] = 'completed'
        write(args.output_dir / 'report.json', report)
        return 1 if any(c['failure'] for c in report['cases']) else 0
    except Exception as exc:
        report.update(status='setup_or_execution_failure', error_type=type(exc).__name__,
                      error='Check local dependencies, prebuilt stores, deployment and contracts; sensitive details omitted.')
        write(args.output_dir / 'report.json', report)
        print(report['error'], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

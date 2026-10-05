"""Run a literal vignette through the default CHECK4 WHO + AMG workflow."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.manual_patient import read_manual_patient
from application.workflow import create_workflow
from application.human_review import offer_human_review
from rag.who_amg_experiment import evidence_summary
from rag.agents.grounding import claims
from orchestration.evidence import fingerprint
from scripts.validate_who_amg_experiment import write


class InputAuditProvider:
    """Synchronous delegate recording actual source inventories, not private reasoning."""
    def __init__(self, provider):
        self.provider = provider
        self.inputs = []

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def generate(self, instructions, payload, schema, validator=None):
        if schema.__name__ in ('DiagnosticResult', 'ClinicalCritique'):
            groups = payload['MEDICAL_KNOWLEDGE_EVIDENCE']
            self.inputs.append({'stage': schema.__name__,
                'source_sections': {source: {kind: [{'source_id': e['source_id'], 'evidence_fingerprint': fingerprint(e)}
                    for e in entries] for kind, entries in sections.items()} for source, sections in groups.items()}})
        return self.provider.generate(instructions, payload, schema, validator)


def main(argv=None):
    from rag.config import PROJECT_ROOT, PatientRAGConfig, OpenAIConfig, load_generation_env
    from rag.patient_rag import PatientCaseRAG
    from rag.amg import AMGMedicalEvidence
    from rag.llm.provider import OpenAIProvider
    from application.resources import prepare_runtime
    from application.decision import build_final_decision
    from scripts.verify_who_amg_run import verify
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('case_file', type=Path)
    cli.add_argument('--live', action='store_true', help='Enable sequential cloud agent calls')
    cli.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'outputs/manual' /
                     datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = cli.parse_args(argv)
    patient, origin = read_manual_patient(args.case_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir / 'input.json', origin)
    result = {'case_id': origin['patient_id'], 'origin': origin['origin'], 'mode': 'who_plus_amg', 'live': args.live}
    if not args.live:
        result.update(status='prepared_not_run', final_decision='NOT_RUN',
                      note='Input parsed only. CHECK4 selection requires --live; no lexical fallback.')
        write(args.output_dir / 'report.json', result)
        print('Input prepared:', args.output_dir)
        return 0
    try:
        config = PatientRAGConfig()
        prepare_runtime(config)
        with ExitStack() as stack:
            patients = stack.enter_context(PatientCaseRAG(config))
            medical = stack.enter_context(AMGMedicalEvidence())
            if args.live:
                load_generation_env()
                provider = InputAuditProvider(stack.enter_context(OpenAIProvider(OpenAIConfig())))
                workflow = create_workflow(provider, patients, medical)
                state = workflow.run(patient, origin['patient_id'])
                write(args.output_dir / 'case_state.json', state.model_dump(mode='json'))
                write(args.output_dir / 'agent_input_audit.json', provider.inputs)
                service = workflow.evidence
            if service.combined:
                write(args.output_dir / 'case_combined.json', service.combined.model_dump(mode='json'))
                result.update(evidence_summary(service.combined))
            if args.live:
                decision = build_final_decision(state)
                write(args.output_dir / 'case_decision.json', decision.model_dump(mode='json'))
                versions = []
                for version in state.diagnostics:
                    used = {ref for claim in claims(version.result) for ref in claim.evidence_refs}
                    counts = Counter('WHO' if e.thaw().metadata['backend'] == 'who-local-lookup-v1' else 'AMG'
                                     for e in state.evidence.medical_knowledge if e.source_id in used)
                    versions.append({'version': version.version, 'WHO': counts['WHO'], 'AMG': counts['AMG']})
                result.update(workflow_status=state.status, failure=state.failure.model_dump() if state.failure else None,
                    diagnostic_citations=versions, final_decision=decision.status, review_state=decision.review_state,
                    safety_coverage=state.safety_coverage, safety_skip_reason=state.safety_skip_reason,
                    safety_decision=decision.safety_result.decision if decision.safety_result else None,
                    total_requests=state.total_requests, seconds=state.seconds,
                    grounding=decision.grounding_result.model_dump() if decision.grounding_result else None,
                    critic=decision.critic_findings.model_dump() if decision.critic_findings else None)
                if service.combined:
                    write(args.output_dir / 'verification.json', verify(args.output_dir / 'case_state.json'))
                    expected = {e.source_id: fingerprint(e.thaw().model_dump()) for e in state.evidence.medical_knowledge}
                    for audit in provider.inputs:
                        actual = {e['source_id']: e['evidence_fingerprint'] for sections in audit['source_sections'].values()
                                  for entries in sections.values() for e in entries}
                        if actual != expected:
                            raise ValueError('Actual agent input differs from evidence snapshot')
                    result['agent_source_inventories_verified'] = True
            else:
                result['final_decision'] = 'NOT_RUN'
        human = offer_human_review(state, args.output_dir / 'human_reviews')
        result['human_review'] = human.model_dump(mode='json') if human else None
        write(args.output_dir / 'report.json', result)
        print('Artifacts:', args.output_dir)
        print('Combined status:', result.get('combined_status', 'NOT_RETRIEVED'))
        print('WHO:', result.get('who_match_status'), '| AMG accepted:', result.get('amg_accepted_count'))
        print('Final decision:', result.get('final_decision'))
        return 1 if result.get('failure') else 0
    except Exception as exc:
        result.update(execution_error=type(exc).__name__, message='Run failed; details omitted. Inspect saved stage artifacts.')
        write(args.output_dir / 'report.json', result)
        print(result['message'], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

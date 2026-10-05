"""Read-only presentation adapters over the Phase 6 workflow.

Live notifications describe service calls, NOT application acceptance. The final
walkthrough uses only committed state/events, since the core has no event callback.
No provider raw output, prompt, credentials, or reasoning_summary is displayed.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import logging
from pathlib import Path
import sys
import unicodedata

DISCLAIMER = (
    'Research only; not clinical advice or clinical validation.\n'
    'HUMAN_REVIEW withholds output; no human review is scheduled.'
)
MANUAL_LIMITATION = (
    'Manual free-text entry is unavailable under the existing validated contract.\n'
    'The workflow requires a parser-generated DDXPlus patient ID and decoded evidence.\n'
    'Assigning a dataset ID to an invented case would misrepresent provenance.\n'
    'No input was submitted. Please choose a real synthetic validation case instead.'
)


def safe_text(value, limit=360):
    """Bound untrusted prose and strip terminal controls (including ANSI/bidi)."""
    text = ''.join(c if not unicodedata.category(c).startswith('C') else ' ' for c in str(value))
    text = ' '.join(text.split())
    return text if len(text) <= limit else text[:limit] + ' ... [excerpt]'


class Console:
    def __init__(self, stream=None, detailed=False):
        self.stream = stream or sys.stdout
        self.detailed = detailed

    def say(self, text=''):
        # Windows consoles/pipes may not support the status glyphs.
        encoding = getattr(self.stream, 'encoding', None) or 'utf-8'
        text = str(text)
        try:
            text.encode(encoding)
        except UnicodeEncodeError:
            for glyph, fallback in {'âœ“': '[OK]', 'â†’': 'RUNNING', 'âš ': '[REVIEW]', 'âœ—': '[FAIL]'}.items():
                text = text.replace(glyph, fallback)
        print(text.encode(encoding, errors='replace').decode(encoding), file=self.stream, flush=True)

    def field(self, name, value):
        self.say(f'{name}: {safe_text(value, 700 if self.detailed else 360)}')

    def section(self, title):
        self.say('\n' + '-' * 60)
        self.say(title)
        self.say('-' * 60)

    def items(self, label, values, limit=6):
        self.field(label, len(values))
        count = 12 if self.detailed else limit
        for value in values[:count]:
            self.say('  - ' + safe_text(value, 700 if self.detailed else 360))
        if len(values) > count:
            self.say(f'  ... {len(values) - count} additional items omitted')

    def banner(self, architecture=False):
        self.say('=' * 60)
        self.say('        AGENTIC HEALTHCARE DECISION SUPPORT DEMO')
        self.say('=' * 60)
        self.say('Research Prototype\nSynthetic DDXPlus patient | GPT-5.6-Sol | Patient Qdrant + AMG MedlinePlus')
        self.say('=' * 60)
        self.say(DISCLAIMER)
        if architecture:
            self.say('\nPATIENT -> PATIENT AGENT -> PATIENT CASE RAG -> AMG MEDICAL EVIDENCE')
            self.say(' -> DIAGNOSTIC AGENT -> GROUNDING -> CLINICAL CRITIC -> SAFETY VALIDATOR')
            self.say(' -> CONTINUE (Phase 5 routing) / HUMAN_REVIEW / BLOCK')
            self.say('RAG is evidence infrastructure, not an autonomous agent.')


def discover_samples(parser, limit=3):
    records = list(parser.iter_patients('validate', limit=limit, include_labels=False))
    if not records:
        raise ValueError('No validation samples')
    if any(r.split != 'validate' or r.labels is not None for r in records):
        raise ValueError('Only label-free validation samples are permitted')
    return records


def choose_sample(records, console, ask=input):
    while True:
        console.say('\n[1] USE SAMPLE PATIENT (default)\n[2] ENTER PATIENT MANUALLY')
        console.say('[D] Toggle detailed mode\n[Q] Quit')
        choice = ask('Mode [1]: ').strip().lower() or '1'
        if choice == 'q':
            return None
        if choice == 'd':
            console.detailed = not console.detailed
            console.field('Detailed mode', console.detailed)
        elif choice == '2':
            console.say(MANUAL_LIMITATION)
        elif choice == '1':
            while True:
                for number, record in enumerate(records, 1):
                    patient = record.patient
                    console.field(f'[{number}] Sample {record.patient_id}',
                                  f'age {patient.age}, sex {patient.sex}, {len(patient.symptoms)} findings')
                selected = ask('Sample [1], [B] Back, [Q] Quit: ').strip().lower() or '1'
                if selected == 'q':
                    return None
                if selected == 'b':
                    break
                if selected.isascii() and selected.isdigit() and 1 <= int(selected) <= len(records):
                    return records[int(selected) - 1]
                console.say('Invalid sample selection; choose a listed number.')
        else:
            console.say('Invalid mode; choose 1, 2, D or Q.')


def show_patient_input(console, record):
    console.section('SYNTHETIC RESEARCH CASE â€” NOT A REAL PATIENT')
    console.field('Validation case', record.patient_id)
    console.field('Age', record.patient.age)
    console.field('Sex', record.patient.sex)
    console.items('Symptoms / clinical evidence', [e.text for e in record.patient.symptoms])
    console.items('Relevant history', [e.text for e in record.patient.antecedents])
    console.say('Unlisted findings are unknown, not assumed absent. Dataset labels are not loaded.')


class RetrievalProgress:
    """Transparent delegate: no extra retrievals, filtering, or rewritten payloads."""
    def __init__(self, delegate, console, name):
        self.delegate, self.console, self.name = delegate, console, name

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def retrieve(self, query, top_k):
        self.console.section(self.name)
        self.console.field('Top-k', top_k)
        self.console.field('Query (label-free; excerpt)', query)
        self.console.say('â†’ Retrieval started')
        result = self.delegate.retrieve(query, top_k=top_k)
        self.console.field('Retrieved hits (pending snapshot validation)', len(result))
        return result


def make_workflow(provider, patients, medical, console, *, top_k, policy, enhanced_medical=False, medical_top_k=5, amg=False):
    from orchestration.phase6 import Phase6Orchestrator

    class LivePhase6(Phase6Orchestrator):
        # The run loop is inherited verbatim. Never inspect Generation.raw/parsed.
        def _invoke(self, ticket, operation):
            names = {'PATIENT': '[1] PATIENT AGENT', 'DIAGNOSTIC': '[4] DIAGNOSTIC AGENT',
                     'DIAGNOSTIC_REVISION': '[4] DIAGNOSTIC AGENT â€” REVISION',
                     'CRITIC': '[6] CLINICAL CRITIC',
                     'SAFETY_VALIDATION': '[7] SAFETY VALIDATOR â€” PHASE 6 (semantic call)'}
            console.section(names.get(ticket.stage, ticket.stage))
            console.field('Diagnostic version', ticket.diagnostic_version)
            console.say('â†’ Calling ' + names.get(ticket.stage, ticket.stage) + '...')
            response = super()._invoke(ticket, operation)
            console.say('âœ“ Call returned; application validation is still required.')
            return response

    if amg:
        from application.workflow import create_workflow
        return create_workflow(provider,
            RetrievalProgress(patients, console, '[2] PATIENT CASE RAG'),
            RetrievalProgress(medical, console, '[3] AMG MEDICAL EVIDENCE'),
            top_k=top_k, medical_top_k=medical_top_k, policy=policy, orchestrator_class=LivePhase6)
    return LivePhase6(provider,
        RetrievalProgress(patients, console, '[2] PATIENT CASE RAG'),
        RetrievalProgress(medical, console, '[3] MEDICAL KNOWLEDGE RAG'),
        top_k=top_k, policy=policy, enhanced_medical=enhanced_medical, medical_top_k=medical_top_k)


def show_medical_audit(console, audit):
    if audit.get('backend') == 'amg-medlineplus-v1':
        console.section('AMG MEDICAL EVIDENCE AUDIT')
        console.field('Backend', audit['backend'])
        console.field('Query', audit['query'])
        console.field('Retrieval status', audit['status'])
        console.field('Facts omitted from bounded retrieval query', audit['omitted_query_facts'])
        for hit in (audit.get('upstream') or {}).get('results', []):
            console.field('Native chunk ID', hit['id'])
            console.field('Source URL', hit['url'])
            console.field('Squared L2 distance (not confidence)', hit['distance'])
            console.field('Acceptance route', hit['acceptance'])
        console.say('No AMG answer synthesis or web search. Rejected candidates are not agent evidence.')
        return
    console.section('HISTORICAL FOCUSED MEDICAL RETRIEVAL AUDIT')
    console.field('Query', audit['query'])
    console.field('Candidate top-k', audit['candidate_top_k'])
    console.field('Retrieval status', audit['status'])
    for item in audit['candidates']:
        console.field('Candidate chunk', item['chunk_id'])
        console.field('Title', item['title'])
        console.field('Similarity (not probability)', f"{item['score']:.4f}")
        console.field('Disposition', ('RETAIN: ' if item['retained'] else 'DISCARD: ') + item['reason'])
    console.say('Retained topics are candidate context, not proof of clinical support.')


def show_evidence(console, snapshot, top_k, medical_top_k=None):
    for title, items in (('[2] PATIENT CASE RAG', snapshot.patient_cases),
                         ('[3] MEDICAL KNOWLEDGE RAG â€” external sources', snapshot.medical_knowledge)):
        console.section(title)
        console.field('Candidate top-k', (medical_top_k or top_k) if 'MEDICAL' in title else top_k)
        console.field('Retained', len(items))
        for number, item in enumerate(items, 1):
            console.field(f'{number}. ID', item.source_id)
            console.field('   Similarity (not disease probability)', f'{item.similarity:.4f}')
            console.field('   Source type', item.source_type)
            console.field('   Source', item.source)
            if item.title:
                console.field('   Title', item.title)
            console.field('   Evidence excerpt', item.text)
    console.say('âœ“ Snapshot provenance validated: patient cases are training cases only.')
    console.say('âœ“ Patient-case payload contract excludes pathology/differential labels.')
    console.say('Medical knowledge is a separate evidence collection, not patient-case outcomes.')
    if console.detailed:
        console.field('Evidence snapshot ID', snapshot.snapshot_id)
        console.field('Evidence schema version', snapshot.schema_version)


def show_diagnosis(console, version):
    console.section(f'[4] DIAGNOSTIC AGENT â€” committed version {version.version}')
    result = version.result
    abstained = result.primary_hypothesis is None and not result.differential_diagnoses
    console.field('Status', 'ABSTAINED' if abstained else 'STRUCTURED HYPOTHESES (research only)')
    console.items('Patient-case evidence refs', result.patient_case_evidence)
    console.items('Medical-knowledge evidence refs', result.medical_knowledge_evidence)
    hypotheses = ([result.primary_hypothesis] if result.primary_hypothesis else []) + result.differential_diagnoses
    for number, hypothesis in enumerate(hypotheses, 1):
        console.field('Primary hypothesis' if number == 1 and result.primary_hypothesis else 'Differential hypothesis', hypothesis.condition)
        # Evidence-grounding information only; no free-form reasoning summary.
        console.items('Rationale evidence refs', hypothesis.rationale.evidence_refs)
        console.items('Supporting evidence refs', sorted({r for c in hypothesis.supporting_evidence for r in c.evidence_refs}))
        console.items('Contradicting evidence refs', sorted({r for c in hypothesis.contradicting_evidence for r in c.evidence_refs}))
    console.items('Missing information / abstention context', result.missing_information)
    console.items('Uncertainty / limitations', result.uncertainty)
    console.items('Unsupported claims', result.unsupported_claims)
    if abstained:
        console.say('No hypothesis was supplied. Context above is from the structured contract; no separate reason code exists.')


def show_safety(console, assessment):
    console.section('[7] SAFETY VALIDATOR â€” PHASE 6')
    console.field('Diagnostic version', assessment.ticket.diagnostic_version)
    console.field('Critic reviewed diagnostic version', assessment.ticket.diagnostic_version)
    console.field('Evidence snapshot ID', assessment.ticket.evidence_snapshot_id)
    console.items('Deterministic checks recorded by Phase 6', assessment.deterministic_checks, limit=20)
    console.field('Semantic review', assessment.semantic_status)
    if assessment.semantic_result:
        for category in assessment.semantic_result.categories:
            console.field('  ' + category.category, category.result)
            if console.detailed or category.result != 'no_issue_identified':
                console.field('    Explanation', category.explanation)
        console.items('Semantic limitations', assessment.semantic_result.limitations)
    console.field('Safety findings', len(assessment.findings))
    for finding in assessment.findings:
        console.field(f'  {finding.code} [{finding.disposition}]', finding.explanation)
        console.items('  Evidence refs', finding.evidence_refs)
    console.items('Structured decision reasons', assessment.reasons)
    console.field('Final Safety Decision for this version', assessment.decision)
    console.say('No issue identified is a model assessment, not proof of clinical safety.')
    if console.detailed:
        console.field('Critic fingerprint', assessment.ticket.critic_fingerprint)
        console.field('Safety policy version', assessment.policy_version)
        console.field('Safety prompt version', assessment.prompt_version)


def show_result(console, state, top_k):
    console.section('AUDITED WALKTHROUGH â€” committed results, not a live event stream')
    console.say('Only accepted structured output is shown. Rejected output and hidden reasoning are omitted.')
    if state.status != 'final':
        console.say('âš  OUTPUT WITHHELD by workflow; hypotheses below are review material, not a released result.')
    console.section('[1] PATIENT AGENT')
    console.say('INPUT: label-free DDXPlus patient representation')
    if state.patient_state:
        console.say('PROCESS: Patient Agent constructed PatientState; exact-copy validation passed.')
        console.say('OUTPUT: accepted structured PatientState')
        for name in ('patient_id', 'age', 'sex'):
            console.field(name, getattr(state.patient_state, name))
        for name in ('presenting_evidence', 'symptoms', 'antecedents', 'relevant_findings', 'missing_information', 'uncertainty_notes'):
            console.items(name, getattr(state.patient_state, name))
    else:
        console.say('No accepted PatientState. See failure summary.')
    if state.medical_retrieval:
        show_medical_audit(console, state.medical_retrieval)
    if state.evidence:
        show_evidence(console, state.evidence, top_k, medical_top_k=state.medical_retrieval['candidate_top_k'] if state.medical_retrieval else top_k)
        console.say('The frozen patient-case and medical evidence above was supplied separately to the agents.')
    else:
        console.say('[2â€“3] No committed evidence snapshot; retrieval not reached or failed.')
    for diagnostic in state.diagnostics:
        show_diagnosis(console, diagnostic)
        console.section('[5] GROUNDING VALIDATION')
        checks = [v for v in state.validations if v.stage == 'GROUNDING' and
                  v.revision_number == diagnostic.ticket.revision_number]
        passed = any(v.grounding_valid is True for v in checks)
        console.field('Diagnostic version', diagnostic.version)
        console.field('Grounding result', 'PASS' if passed else 'NOT RECORDED')
        if passed:
            console.say('âœ“ Patient-fact, patient-case and medical references checked against frozen inventories.')
            console.say('âœ“ Referenced IDs exist; diagnostic evidence inventories match claim references.')
            console.say('Reference existence does not establish clinical entailment or correctness.')
        if state.patient_state is not None and state.medical_retrieval is not None:
            from orchestration.phase6.reference_entailment import inspect_reference_consistency
            audit = inspect_reference_consistency(diagnostic.result, state.patient_state)
            console.field('Recomputed bounded patient-reference audit', audit['status'])
            console.field('Recognized observation checks', len(audit['checks']))
            console.field('Unassessed clauses/claims', audit['unassessed_clauses'])
            console.say('Partial concept/polarity consistency only; not general semantic entailment or clinical approval.')
            if console.detailed:
                for check in audit['checks']:
                    console.say(f"  Claim {check['claim_index']}: {check['concept']}: {check['result']} ({check['reason']})")
                    console.items('  Patient references', check['patient_refs'])

        review = next((c for c in state.critiques if c.diagnostic_version == diagnostic.version), None)
        console.section('[6] CLINICAL CRITIC')
        if review:
            console.field('Diagnostic version reviewed', review.diagnostic_version)
            console.field('Status', review.result.overall_assessment)
            for name in ('safety_flags', 'unsupported_points', 'contradictions', 'recommended_revisions', 'missing_evidence', 'hallucination_flags'):
                console.items(name, getattr(review.result, name))
            console.field('Revision requested by critic', review.result.overall_assessment == 'revision_required')
        else:
            console.say('No committed critique for this version (not reached or failed).')
        assessment = next((a for a in state.safety_assessments if a.ticket.diagnostic_version == diagnostic.version), None)
        if assessment:
            show_safety(console, assessment)
        else:
            console.section('[7] SAFETY VALIDATOR â€” PHASE 6')
            console.say('No committed safety assessment for this version. No safety pass is implied.')
    if not state.diagnostics:
        console.say('[4â€“7] No committed diagnostic output; later results are unavailable. No safety pass is implied.')
    failures = [v for v in state.validations if v.grounding_valid is False]
    for failure in failures:
        console.field(f'âœ— Validation FAIL ({failure.stage}, revision {failure.revision_number})', failure.reason)
    if console.detailed:
        console.section('TECHNICAL AUDIT')
        for name in ('schema_version', 'policy_version', 'revision_count', 'total_requests', 'provider_repairs', 'seconds'):
            console.field(name, getattr(state, name))
        for event in state.transition_history:
            console.field(f'Event {event.sequence} / {event.stage} / v{event.diagnostic_version}', f'{event.event}: {event.reason}')
    console.say('\n' + '=' * 60 + '\n                    FINAL SYSTEM RESULT\n' + '=' * 60)
    console.field('Patient', 'Synthetic DDXPlus case ' + state.patient_id)
    latest = state.diagnostics[-1].result if state.diagnostics else None
    console.field('Diagnostic status', ('ABSTAINED' if latest.primary_hypothesis is None and not latest.differential_diagnoses else 'Hypotheses recorded') if latest else 'No accepted diagnosis')
    console.field('Clinical critic', state.critiques[-1].result.overall_assessment if state.critiques else 'Not completed')
    current = state.safety_assessments[-1] if state.safety_coverage == 'assessed' else None
    console.field('Safety decision', current.decision if current else 'NOT ASSESSED / UNAVAILABLE')
    console.field('Safety coverage', state.safety_coverage)
    if state.safety_skip_reason:
        console.field('Safety skip/failure reason', state.safety_skip_reason)
    console.field('Workflow status', state.status)
    console.field('Workflow outcome', state.outcome)
    console.field('Evidence retrieved â€” patient cases', len(state.evidence.patient_cases) if state.evidence else 0)
    console.field('Evidence retrieved â€” medical knowledge', len(state.evidence.medical_knowledge) if state.evidence else 0)
    console.field('Evidence cited by latest diagnosis â€” patient cases', len(latest.patient_case_evidence) if latest else 0)
    console.field('Evidence cited by latest diagnosis â€” medical knowledge', len(latest.medical_knowledge_evidence) if latest else 0)
    console.field('Revisions', state.revision_count)
    console.field('LLM requests', state.total_requests)
    console.field('Provider repairs', state.provider_repairs)
    console.field('Execution time', f'{state.seconds:.2f} seconds (workflow only)')
    if state.failure:
        console.field('âœ— Failure code', state.failure.code)
        console.say('Check deployment availability, request/time budgets, and local contracts. No result was fabricated.')
    console.say({'final': 'âœ“ COMPLETED â€” research output only', 'blocked': 'âœ— BLOCKED',
                 'failed': 'âœ— FAILED'}.get(state.status, 'âš  REVIEW / WITHHELD'))
    console.say('=' * 60)


def arguments(argv=None):
    cli = argparse.ArgumentParser(description='Live terminal demonstration of the existing Phase 6 workflow. Synthetic research only.')
    cli.add_argument('--sample', type=int, choices=(1, 2, 3), help='Run one of the first three validation records without the menu')
    cli.add_argument('--list-samples', action='store_true', help='Discover label-free validation samples without opening indexes or calling GPT')
    cli.add_argument('--validate-only', action='store_true', help='Check existing local indexes/config and retrieve evidence; no GPT calls or diagnosis')
    cli.add_argument('--detailed', action='store_true')
    cli.add_argument('--architecture', action='store_true')
    cli.add_argument('--top-k', type=int, default=1, help='Patient-case top-k')
    cli.add_argument('--medical-top-k', type=int, default=5)
    cli.add_argument('--max-requests', type=int, default=21)
    cli.add_argument('--max-seconds', type=float, default=900.0, help='Cooperative workflow deadline; not hard cancellation of an in-flight call')
    cli.add_argument('--data-dir', type=Path)
    cli.add_argument('--patient-storage-path', type=Path)
    args = cli.parse_args(argv)
    import math
    if not 1 <= args.top_k <= 20 or not 1 <= args.medical_top_k <= 20 or args.max_requests < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        cli.error('top-k, request budget and finite time budget must be positive')
    return args


def main(argv=None):
    args = arguments(argv)
    console = Console(detailed=args.detailed)
    console.banner(args.architecture)
    # Third-party logs can contain payloads/transport errors. The demo reports
    # safe categories instead; never enable provider/debug logging here.
    logging.basicConfig(level=logging.CRITICAL)
    phase = 'dataset discovery (check --data-dir and DDXPlus validation ZIP/metadata)'
    try:
        from rag.config import OpenAIConfig, PatientRAGConfig, load_generation_env
        from rag.patient_parser import DDXPlusParser
        records = discover_samples(DDXPlusParser(args.data_dir))
        if args.list_samples:
            for record in records:
                show_patient_input(console, record)
            return 0
        record = records[(args.sample or 1) - 1] if args.sample or args.validate_only else choose_sample(records, console)
        if record is None:
            return 0
        show_patient_input(console, record)
        phase = 'configuration (check the dedicated GPT_SOL settings; credentials are never printed)'
        load_generation_env()
        config = OpenAIConfig()
        patient_config = PatientRAGConfig()
        if args.patient_storage_path:
            patient_config = replace(patient_config, storage_path=args.patient_storage_path)
        from application.resources import prepare_runtime
        phase = 'local indexes (check patient index, AMG snapshots and cached BGE/MiniLM models)'
        prepare_runtime(patient_config)
        console.say('\nâ†’ Opening existing indexes and cached CPU models (no rebuild/download)...')
        from rag.patient_rag import PatientCaseRAG
        from rag.amg import AMGMedicalEvidence
        from orchestration.phase6 import Phase6Policy
        with ExitStack() as stack:
            patients = stack.enter_context(PatientCaseRAG(patient_config))
            medical = stack.enter_context(AMGMedicalEvidence())
            counts = (patients.vector_store.count(), medical.count())
            if not all(counts):
                raise ValueError('Nonempty prebuilt collections required')
            console.field('Indexed training cases / medical chunks', f'{counts[0]} / {counts[1]}')
            if console.detailed:
                console.field('Model', config.model)
                console.field('Patient collection', patient_config.collection_name)
                console.field('Medical collection', medical.manifest['config']['collection'])
                console.field('Patient embedding model', patient_config.model_name)
                console.field('Medical embedding model', medical.manifest['config']['embedding']['model'])
                console.field('Patient / medical embedding dimensions', f'{patients.embedding_model.dimension} / {medical.manifest['config']['embedding']['dimension']}')
                console.field('Top-k', args.top_k)
            if args.validate_only:
                from rag.amg import AMGEvidenceService
                from rag.agents.grounding import state_from_patient
                service = AMGEvidenceService(
                    RetrievalProgress(patients, console, '[2] PATIENT CASE RAG'),
                    RetrievalProgress(medical, console, '[3] MEDICAL KNOWLEDGE RAG'),
                    top_k=args.top_k, medical_top_k=args.medical_top_k)
                snapshot = service.retrieve(record.patient, state_from_patient(record.patient, record.patient_id))
                show_medical_audit(console, service.audit)
                show_evidence(console, snapshot, args.top_k, args.medical_top_k)
                if not snapshot.medical_knowledge:
                    console.say('No accepted medical evidence. Live workflow would abstain before diagnosis.')
                console.say('âœ“ LOCAL VALIDATION COMPLETED: configuration, sample loading, index signatures and retrieval.')
                console.say('LLM requests: 0. Cloud connectivity/credentials and clinical performance were NOT tested.')
                return 0
            phase = 'cloud provider setup/execution (check GPT_SOL credentials, availability, timeout and budgets)'
            from rag.llm.provider import OpenAIProvider
            console.say('â†’ Live GPT run: synthetic patient and retrieved evidence will be sent to the configured cloud deployment.')
            console.say('Live call notifications are provisional; accepted outputs/gates appear in the audited walkthrough.')
            provider = stack.enter_context(OpenAIProvider(config))
            workflow = make_workflow(provider, patients, medical, console, top_k=args.top_k,
                policy=Phase6Policy(max_requests=args.max_requests, max_seconds=args.max_seconds),
                amg=True, medical_top_k=args.medical_top_k)
            state = workflow.run(record.patient, record.patient_id)
            from application.decision import build_final_decision
            decision = build_final_decision(state)
            show_result(console, state, args.top_k)
            console.section('FINAL DECISION / HUMAN REVIEW HANDOFF')
            console.field('Final decision', decision.status)
            console.field('Human review required', decision.human_review_required)
            console.field('Review state', decision.review_state)
            console.field('Proposal withheld', decision.proposal_withheld)
            console.say(decision.review_instructions)
            return 1 if state.status == 'failed' else 0
    except (KeyboardInterrupt, EOFError):
        console.say('\nDemo cancelled. No result or safety decision is implied.')
        return 130
    except Exception:
        console.say('\nâœ— Demo failed during ' + phase + '.')
        console.say('Sensitive exception details omitted. No fallback clinical output was generated.')
        console.say('Try --list-samples or --validate-only to isolate local setup from the cloud provider.')
        return 1
    finally:
        console.say('\n' + DISCLAIMER)

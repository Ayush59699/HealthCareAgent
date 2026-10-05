"""Offline audit of a saved AMG workflow (no labels, model loading or API calls)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestration.phase6.state import Phase6WorkflowState
from orchestration.evidence import fingerprint
from rag.agents.grounding import context, validate_references
from rag.amg.backend import VENDOR_ROOT
from safety.models import SafetyInput
from safety.validation import validate_input


def verify(path, final_path=None):
    state = Phase6WorkflowState.model_validate_json(path.read_text(encoding='utf8'))
    if state.medical_retrieval is None or state.medical_retrieval['backend'] != 'amg-medlineplus-v1':
        raise ValueError('Not an AMG run')
    cases, medical = state.evidence.agent_evidence()
    context(state.patient_state, cases, medical)
    # Validate each supplied passage against the native published source artifact.
    snapshots = {e.metadata['retrieval']['snapshot'] for e in medical}
    for snapshot in snapshots:
        folder = VENDOR_ROOT / 'knowledge' / 'medlineplus_lab' / snapshot
        wanted = {e.source_id: e for e in medical if e.metadata['retrieval']['snapshot'] == snapshot}
        with (folder / 'chunks.jsonl').open(encoding='utf8') as stream:
            for line in stream:
                chunk = json.loads(line)
                key = 'medical:' + chunk['id']
                if key in wanted:
                    e = wanted.pop(key)
                    if e.text != chunk['text'] or e.metadata['amg_metadata'] != chunk['metadata']:
                        raise ValueError('Saved evidence differs from the source')
        if wanted:
            raise ValueError('Unknown source chunk')
    for version in state.diagnostics:
        validate_references(version.result, state.patient_state, cases, medical, allow_patient_inference=True)
        if version.ticket.evidence_snapshot_id != state.evidence.snapshot_id:
            raise ValueError('Diagnostic snapshot mismatch')
    for review in state.critiques:
        validate_references(review.result, state.patient_state, cases, medical, allow_patient_inference=True)
    for assessment in state.safety_assessments:
        version = assessment.ticket.diagnostic_version
        diagnosis = next(d for d in state.diagnostics if d.version == version)
        critique = next(c for c in state.critiques if c.diagnostic_version == version)
        value = validate_input(SafetyInput(patient_state=state.patient_state,
            diagnostic=diagnosis.result, critique=critique.result, patient_cases=cases, medical_knowledge=medical))
        if fingerprint(value.model_dump()) != assessment.ticket.input_fingerprint:
            raise ValueError('AMG evidence did not match the assessed safety input')
    final_verification = verify_final_decision(state, final_path) if final_path is not None else {}
    return {**final_verification, 'patient_id': state.patient_id, 'status': state.status, 'failure': state.failure is not None,
            'backend': state.medical_retrieval['backend'], 'native_sources_verified': len(medical),
            'diagnostic_versions_verified': len(state.diagnostics), 'critic_versions_verified': len(state.critiques),
            'safety_inputs_verified': len(state.safety_assessments),
            'safety_decision': state.safety_assessments[-1].decision if state.safety_assessments else None,
            'clinical_success_claimed': False}


def verify_final_decision(state, path):
    """Recompute the release/review handoff; a saved permissive flag is not trusted."""
    from application.decision import FinalDecision, build_final_decision
    saved = FinalDecision.model_validate_json(path.read_text(encoding='utf8'))
    if saved != build_final_decision(state):
        raise ValueError('Final decision differs from the validated workflow')
    return {'final_decision_verified': True, 'final_decision_status': saved.status,
            'human_review_required': saved.human_review_required, 'review_state': saved.review_state}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('case', type=Path)
    cli.add_argument('--final-decision', type=Path, help='Also verify the separate release/review artifact')
    args = cli.parse_args()
    print(json.dumps(verify(args.case, args.final_decision), indent=2))

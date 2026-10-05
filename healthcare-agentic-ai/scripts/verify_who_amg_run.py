"""Offline source-byte and existing grounding/safety/handoff verification; no models."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application.decision import build_final_decision, FinalDecision
from orchestration.phase6.state import Phase6WorkflowState
from rag.combined_medical_evidence import CombinedMedicalEvidence
from rag.agents.grounding import validate_references
from rag.amg.backend import VENDOR_ROOT
from rag.who_lookup import DEFAULT_DATA_DIR, bounded_read, MAX_DOCUMENT_BYTES
from rag.who_provenance import sha256
from safety.models import SafetyInput
from safety.policy import validate_assessment


def verify_selector_handoff(state, data_dir):
    from rag.who_catalog_selector import WHOCatalogSelector, WHOSelection, digest
    from rag.who_selector_relevance import ContentRelevance
    from rag.who_sections import section_hits
    from rag.agents.grounding import evidence_from_hits
    selector = WHOCatalogSelector(data_dir)
    audit = state.medical_retrieval['who']
    if (audit['catalog_sha256'] != selector.catalog_sha256
            or audit['catalog_documents'] != len(selector.catalog_payload()['catalog'])
            or audit['patient_payload_sha256'] != digest(state.original_patient.to_inference_dict())):
        raise ValueError('WHO catalog or patient identity mismatch')
    proposal = selector.validate_selection(WHOSelection.model_validate(audit['catalog_stage']['parsed']))
    chosen = selector.validate_selection(WHOSelection(selected_ids=audit['selected_ids']))
    if (audit['selection_count'] != len(chosen.selected_ids)
            or audit['max_selected'] != 3 or audit['catalog_stage']['failure'] is not None
            or audit['status'] != ('selected_and_verified' if chosen.selected_ids else 'no_selection')):
        raise ValueError('WHO selection contract mismatch')
    if [r['document']['document_id'] for r in audit['candidates']] != proposal.selected_ids:
        raise ValueError('WHO candidate inventory mismatch')
    retained, rebuilt = [], []
    observations = list(audit['catalog_stage']['telemetry'])
    for document, row in zip(selector.iter_selected(proposal), audit['candidates']):
        if document.audit() != row['document'] or row['content_review']['failure'] is not None:
            raise ValueError('WHO content review/source mismatch')
        review = ContentRelevance.model_validate(row['content_review']['parsed'])
        from rag.who_selector_relevance import validate_content_review
        validate_content_review(review, state.original_patient, document)
        observations.extend(row['content_review']['telemetry'])
        if review.verdict == 'relevant':
            retained.append(document.document_id)
            rebuilt.extend(evidence_from_hits(section_hits(document, review, state.original_patient), 'medical_knowledge'))
    if retained != chosen.selected_ids:
        raise ValueError('WHO selected subset differs from content verdicts')
    actual = [e.thaw() for e in state.evidence.medical_knowledge if e.thaw().source == 'who']
    if {e.source_id: e for e in rebuilt} != {e.source_id: e for e in actual}:
        raise ValueError('WHO relevant sections differ from exact replay')
    if {e.source_id: e.metadata for e in rebuilt} != {'medical:' + m['chunk_id']: m for m in audit['documents']}:
        raise ValueError('WHO handoff audit differs from evidence')
    invocations = [i for i in state.invocations if i.ticket.stage == 'EVIDENCE']
    if (sum(i.attempts for i in invocations) != len(observations)
            or sum(i.request_bytes for i in invocations) != sum(t['request_bytes'] for t in observations)
            or state.total_requests != sum(i.attempts for i in state.invocations)
            or state.total_request_bytes != sum(i.request_bytes for i in state.invocations)):
        raise ValueError('WHO request accounting mismatch')


def verify(state_path, *, data_dir=DEFAULT_DATA_DIR):
    state_path = Path(state_path)
    prefix = state_path.name.removesuffix('_state.json')
    if state_path.name.startswith('sample_case_'):
        number = state_path.stem.removeprefix('sample_case_')
        combined_path = state_path.with_name('combined_evidence_' + number + '.json')
        decision_path = state_path.with_name('final_decision_' + number + '.json')
    else:
        combined_path = state_path.with_name(prefix + '_combined.json')
        decision_path = state_path.with_name(prefix + '_decision.json')
    state = Phase6WorkflowState.model_validate_json(state_path.read_text(encoding='utf8'))
    combined = CombinedMedicalEvidence.model_validate_json(
        combined_path.read_text(encoding='utf8'))
    if combined.snapshot(state.evidence.patient_cases) != state.evidence:
        raise ValueError('Combined evidence differs from workflow evidence')
    if json.loads(combined.provenance_json) != state.medical_retrieval:
        raise ValueError('Combined audit differs from saved retrieval audit')
    cases, medical = state.evidence.agent_evidence()
    if state.medical_retrieval['backend'] == 'who-check4-amg-v1':
        verify_selector_handoff(state, data_dir)
    counts = {'WHO': 0, 'AMG': 0}
    wanted = {}
    for item in medical:
        meta = item.metadata
        if meta['backend'] == 'who-local-lookup-v1':
            folder = 'who_fact_sheets' if meta['document_type'] == 'fact_sheet' else 'who_questions_answers'
            root = (Path(data_dir) / folder).resolve()
            path = (root / meta['filename']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('WHO path escapes collection')
            text = bounded_read(path, MAX_DOCUMENT_BYTES)
            lines = text.split('\n', 4)
            if (sha256(text) != meta['document_sha256'] or len(text) != meta['document_characters']
                    or text[meta['char_start']:meta['char_end']] != item.text
                    or lines[1].rstrip('\r') != 'Title: ' + meta['title']
                    or lines[2].rstrip('\r') != 'URL: ' + meta['url']
                    or lines[3].rstrip('\r') != 'Retrieved: ' + meta['retrieved']):
                raise ValueError('WHO evidence differs from downloaded source')
            counts['WHO'] += 1
        else:
            wanted.setdefault(meta['retrieval']['snapshot'], {})[item.source_id] = item
    for snapshot, entries in wanted.items():
        path = VENDOR_ROOT / 'knowledge/medlineplus_lab' / snapshot / 'chunks.jsonl'
        with path.open(encoding='utf8') as stream:
            for line in stream:
                chunk = json.loads(line)
                item = entries.pop('medical:' + chunk['id'], None)
                if item is not None:
                    if item.text != chunk['text'] or item.metadata['amg_metadata'] != chunk['metadata']:
                        raise ValueError('AMG evidence differs from native source')
                    counts['AMG'] += 1
        if entries:
            raise ValueError('Unknown AMG source')
    for version in state.diagnostics:
        validate_references(version.result, state.patient_state, cases, medical, allow_patient_inference=True)
        if version.ticket.evidence_snapshot_id != state.evidence.snapshot_id:
            raise ValueError('Diagnostic snapshot differs')
    for review in state.critiques:
        validate_references(review.result, state.patient_state, cases, medical, allow_patient_inference=True)
    for assessment in state.safety_assessments:
        version = assessment.ticket.diagnostic_version
        diagnosis = next(d for d in state.diagnostics if d.version == version)
        critique = next(c for c in state.critiques if c.diagnostic_version == version)
        value = SafetyInput(patient_state=state.patient_state, diagnostic=diagnosis.result,
            critique=critique.result, patient_cases=cases, medical_knowledge=medical)
        validate_assessment(assessment, value, assessment.ticket)
    saved = FinalDecision.model_validate_json(decision_path.read_text(encoding='utf8'))
    if saved != build_final_decision(state):
        raise ValueError('Final handoff differs from revalidated workflow')
    return {'case_id': state.patient_id, 'mode': state.medical_retrieval['mode'], 'sources_verified': counts,
            'diagnostics_verified': len(state.diagnostics), 'critics_verified': len(state.critiques),
            'safety_assessments_verified': len(state.safety_assessments), 'final_decision': saved.status,
            'review_state': saved.review_state, 'clinical_success_claimed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('state', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.state), indent=2))

"""Deterministic grounding guards; citation existence is not clinical entailment."""
import re
from .models import PatientState, RetrievedEvidence, DiagnosticResult, ClinicalCritique
from ..models import PatientRepresentation
from ..patient_ingestion import validate_document
from ..medical_ingestion.models import validate_chunk

class GroundingError(ValueError):
    """Safe machine-generated error code; never contains patient values."""
    def __init__(self, code):
        self.safe_code = code
        super().__init__(code)


UNCERTAINTY = 'Unlisted findings are unknown, not assumed absent.'


def patient_input(patient: PatientRepresentation, patient_id: str) -> dict:
    if type(patient) is not PatientRepresentation:
        raise TypeError('Only label-free PatientRepresentation is accepted, never PatientRecord or raw rows')
    if not re.fullmatch(r'ddxplus:(train|validate|test):[1-9]\d*', patient_id):
        raise ValueError('Expected parser-generated patient ID')
    features = patient.to_inference_dict()
    missing = [key for key in ('age', 'sex') if features[key] is None]
    for field in ('symptoms', 'antecedents', 'initial_evidence'):
        if not features[field]:
            missing.append(field)
    return {'patient_id': patient_id, **features, 'presenting_evidence_to_copy': features['initial_evidence'], 'allowed_missing_information': missing,
            'allowed_uncertainty_notes': [UNCERTAINTY]}


def validate_patient(state: PatientState, supplied: dict) -> None:
    for target, source in [('patient_id', 'patient_id'), ('age', 'age'), ('sex', 'sex'),
                           ('symptoms', 'symptoms'), ('antecedents', 'antecedents'),
                           ('presenting_evidence', 'initial_evidence')]:
        if getattr(state, target) != supplied[source]:
            raise GroundingError('patient_copy_' + target)
    facts = supplied['symptoms'] + supplied['antecedents'] + supplied['initial_evidence']
    if any(f not in facts for f in state.relevant_findings):
        raise ValueError('Invented patient finding')
    if state.missing_information != supplied['allowed_missing_information'] or state.uncertainty_notes != supplied['allowed_uncertainty_notes']:
        raise ValueError('Only input-derived missingness and uncertainty are permitted')


def evidence_from_hits(hits: list[dict], source_type: str) -> list[RetrievedEvidence]:
    result = []
    for hit in hits:
        payload = {key: value for key, value in hit.items() if key != 'score'}
        if source_type == 'patient_case':
            payload = validate_document(payload)
            source_id, title = 'case:' + payload['patient_id'], None
        elif source_type == 'medical_knowledge':
            payload = validate_chunk(payload)
            source_id, title = 'medical:' + payload['chunk_id'], payload['title']
        else:
            raise ValueError('Unknown evidence source')
        result.append(RetrievedEvidence(source_type=source_type, source_id=source_id,
            source=payload['source'], title=title, text=payload['text'], similarity=float(hit['score']),
            metadata={key: value for key, value in payload.items() if key != 'text'}))
    if len({e.source_id for e in result}) != len(result):
        raise ValueError('Duplicate evidence IDs')
    return result


def facts(state: PatientState) -> dict:
    result = {'patient:age': state.age, 'patient:sex': state.sex}
    for name in ('symptoms', 'antecedents', 'presenting_evidence'):
        result.update({f'patient:{name}:{i}': text for i, text in enumerate(getattr(state, name))})
    return result


def context(state: PatientState, cases: list[RetrievedEvidence], medical: list[RetrievedEvidence]) -> dict:
    # Revalidate evidence even when an agent is called directly outside the runner.
    for items, kind in ((cases, 'patient_case'), (medical, 'medical_knowledge')):
        for item in items:
            rebuilt = evidence_from_hits([{**item.metadata, 'text': item.text, 'score': item.similarity}], kind)[0]
            if rebuilt != item:
                raise ValueError('Altered evidence provenance')
    return {'patient_state': state.model_dump(), 'INPUT_FACTS': list(facts(state)),
            'PATIENT_CASE_EVIDENCE': [e.model_dump() for e in cases],
            'MEDICAL_KNOWLEDGE_EVIDENCE': [e.model_dump() for e in medical]}


def claims(diagnosis: DiagnosticResult):
    hypotheses = ([diagnosis.primary_hypothesis] if diagnosis.primary_hypothesis else []) + diagnosis.differential_diagnoses
    for hypothesis in hypotheses:
        yield hypothesis.rationale
        yield from hypothesis.supporting_evidence
        yield from hypothesis.contradicting_evidence


def validate_references(output: DiagnosticResult | ClinicalCritique, state: PatientState,
                        cases: list[RetrievedEvidence], medical: list[RetrievedEvidence]) -> None:
    case_ids, medical_ids = {e.source_id for e in cases}, {e.source_id for e in medical}
    allowed = set(facts(state)) | case_ids | medical_ids
    items = list(claims(output)) if isinstance(output, DiagnosticResult) else output.supported_points
    if any(ref not in allowed for claim in items for ref in claim.evidence_refs):
        raise ValueError('Unknown evidence reference')
    if isinstance(output, DiagnosticResult):
        if not set(output.patient_case_evidence) <= case_ids or not set(output.medical_knowledge_evidence) <= medical_ids:
            raise ValueError('Incorrect evidence source category')
        used = {ref for claim in items for ref in claim.evidence_refs}
        if set(output.patient_case_evidence) != used & case_ids or set(output.medical_knowledge_evidence) != used & medical_ids:
            raise ValueError('Evidence inventories must match claim references')
        # Novel hypotheses are allowed only as explicit model inference grounded in
        # medical material, never presented as observed diagnoses or case outcomes.
        for claim in items:
            if not set(claim.evidence_refs) & allowed:
                raise ValueError('Ungrounded claim')
        if (output.primary_hypothesis is not None or output.differential_diagnoses) and not medical:
            raise ValueError('Abstain when no medical knowledge was retrieved')
    # Prevent fabricated literal URLs/DOIs in prose. Semantic hallucination still
    # requires critic/human review, not a word-overlap pseudo-metric.
    supplied = str([e.model_dump() for e in cases + medical])
    for url in re.findall(r'https?://[^\s"<>]+', output.model_dump_json()):
        if url.rstrip('.,;)\\') not in supplied:
            raise ValueError('Unretrieved URL')

"""Separate case/medical retrieval, one immutable evidence snapshot per workflow."""
import copy
from rag.agents.grounding import evidence_from_hits, patient_input, validate_patient
from orchestration.evidence import EvidenceSnapshot, FrozenEvidence, fingerprint
from .provenance import BACKEND


class AMGEvidenceService:
    requires_patient_state = True

    def __init__(self, patient_rag, medical_rag, *, top_k=1, medical_top_k=5):
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('Patient top-k must be between 1 and 20')
        if type(medical_top_k) is not int or not 1 <= medical_top_k <= 20:
            raise ValueError('Medical top-k must be between 1 and 20')
        self.patient_rag, self.medical_rag = patient_rag, medical_rag
        self.top_k, self.medical_top_k = top_k, medical_top_k
        self.audit = None

    def retrieve(self, patient, state):
        validate_patient(state, patient_input(patient, state.patient_id))
        self.audit = None  # Never reuse a preceding patient's audit.
        cases = evidence_from_hits(self.patient_rag.retrieve(patient.to_text(), top_k=self.top_k), 'patient_case')
        query, omitted = self.medical_rag.query_for(state)
        hits = self.medical_rag.retrieve(query, top_k=self.medical_top_k) if query else []
        if any(hit.get('backend') != BACKEND for hit in hits):
            raise ValueError('Only AMG medical evidence is accepted by this service')
        medical = evidence_from_hits(hits, 'medical_knowledge')
        self.audit = {
            'backend': BACKEND, 'query': query, 'candidate_top_k': self.medical_top_k,
            'omitted_query_facts': omitted, 'status': 'evidence_found' if medical else 'insufficient_evidence',
            'source_ids': [e.source_id for e in medical],
            'upstream': copy.deepcopy(self.medical_rag.audit) if query else None,
            'score_note': 'Similarity is negative squared L2 for compatibility, not probability. Raw distance and acceptance route are retained.',
            'synthesis_used': False, 'web_used': False,
        }
        identity = fingerprint({'patient_cases': [e.model_dump() for e in cases],
                                'medical_knowledge': [e.model_dump() for e in medical]})
        return EvidenceSnapshot(snapshot_id=identity,
            patient_cases=tuple(FrozenEvidence.freeze(e) for e in cases),
            medical_knowledge=tuple(FrozenEvidence.freeze(e) for e in medical))

"""Retrieve once via existing interfaces; immutable, lossless evidence snapshots."""
import hashlib
import json
from typing import Literal
from pydantic import ConfigDict, model_validator
from rag.agents.models import StrictModel, RetrievedEvidence
from rag.agents.grounding import evidence_from_hits
from rag.models import PatientRepresentation
from rag.phase4 import Retriever


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


class FrozenEvidence(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)
    source_type: Literal['patient_case', 'medical_knowledge']
    source_id: str
    source: str
    title: str | None
    text: str
    similarity: float
    # JSON avoids shallow-frozen dictionaries/lists while preserving all provenance.
    metadata_json: str

    @classmethod
    def freeze(cls, evidence: RetrievedEvidence):
        values = evidence.model_dump()
        metadata = values.pop('metadata')
        return cls(**values, metadata_json=json.dumps(metadata, sort_keys=True, ensure_ascii=False))

    def thaw(self) -> RetrievedEvidence:
        values = self.model_dump(exclude={'metadata_json'})
        return RetrievedEvidence(**values, metadata=json.loads(self.metadata_json))


class EvidenceSnapshot(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    schema_version: Literal['phase5-evidence-v1'] = 'phase5-evidence-v1'
    snapshot_id: str
    patient_cases: tuple[FrozenEvidence, ...]
    medical_knowledge: tuple[FrozenEvidence, ...]

    @model_validator(mode='after')
    def check_integrity(self):
        for items, kind in ((self.patient_cases, 'patient_case'), (self.medical_knowledge, 'medical_knowledge')):
            thawed = [item.thaw() for item in items]
            rebuilt = evidence_from_hits([
                {**item.metadata, 'text': item.text, 'score': item.similarity} for item in thawed
            ], kind)
            if rebuilt != thawed:
                raise ValueError('Invalid evidence provenance')
        if self.snapshot_id != self.content_id():
            raise ValueError('Evidence snapshot mismatch')
        return self

    def content_id(self):
        return fingerprint({'patient_cases': [e.thaw().model_dump() for e in self.patient_cases],
                            'medical_knowledge': [e.thaw().model_dump() for e in self.medical_knowledge]})

    def agent_evidence(self):
        self.check_integrity()
        return [e.thaw() for e in self.patient_cases], [e.thaw() for e in self.medical_knowledge]


class EvidenceService:
    def __init__(self, patient_rag: Retriever, medical_rag: Retriever, *, top_k: int = 1):
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        self.patient_rag, self.medical_rag, self.top_k = patient_rag, medical_rag, top_k

    def retrieve(self, patient: PatientRepresentation) -> EvidenceSnapshot:
        query = patient.to_text()
        cases = evidence_from_hits(self.patient_rag.retrieve(query, top_k=self.top_k), 'patient_case')
        medical = evidence_from_hits(self.medical_rag.retrieve(query, top_k=self.top_k), 'medical_knowledge')
        identity = fingerprint({'patient_cases': [e.model_dump() for e in cases],
                                'medical_knowledge': [e.model_dump() for e in medical]})
        return EvidenceSnapshot(snapshot_id=identity,
            patient_cases=tuple(FrozenEvidence.freeze(e) for e in cases),
            medical_knowledge=tuple(FrozenEvidence.freeze(e) for e in medical))

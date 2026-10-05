"""Phase 4 contracts. No evaluation labels belong in this module."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class PatientState(StrictModel):
    patient_id: str
    age: int | None
    sex: Literal['M', 'F'] | None
    presenting_evidence: list[str]
    symptoms: list[str]
    antecedents: list[str]
    relevant_findings: list[str]
    missing_information: list[str]
    uncertainty_notes: list[str]


class RetrievedEvidence(StrictModel):
    source_type: Literal['patient_case', 'medical_knowledge']
    source_id: str
    source: str
    title: str | None
    text: str
    similarity: float
    metadata: dict


class EvidenceClaim(StrictModel):
    statement: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class Hypothesis(StrictModel):
    condition: str = Field(min_length=1)
    rationale: EvidenceClaim
    supporting_evidence: list[EvidenceClaim]
    contradicting_evidence: list[EvidenceClaim]


class ManagementAction(StrictModel):
    category: Literal['evaluation', 'monitoring', 'referral', 'supportive_care', 'treatment_consideration']
    proposal: EvidenceClaim


class TreatmentPlan(StrictModel):
    status: Literal['proposed', 'deferred']
    actions: list[ManagementAction]
    patient_case_evidence: list[str]
    medical_knowledge_evidence: list[str]
    missing_information: list[str] = Field(min_length=1)
    uncertainty: list[str] = Field(min_length=1)

    @model_validator(mode='after')
    def consistent(self):
        if (self.status == 'proposed') != bool(self.actions):
            raise ValueError('Proposed management needs actions; deferred management has none')
        if any(not text.strip() for text in self.missing_information + self.uncertainty):
            raise ValueError('Management limitations cannot be blank')
        if any(not action.proposal.statement.strip() for action in self.actions):
            raise ValueError('Management proposals cannot be blank')
        return self


def _diagnostic_schema(schema):
    # Live structured output requires every property, including explicit null.
    # Local historical records may still omit treatment.
    schema['required'] = list(schema['properties'])
    schema['properties']['treatment'].pop('default', None)


class DiagnosticResult(StrictModel):
    model_config = ConfigDict(json_schema_extra=_diagnostic_schema)
    treatment: TreatmentPlan | None = None

    @model_serializer(mode='wrap')
    def preserve_legacy_serialization(self, handler):
        result = handler(self)
        if self.treatment is None:
            result.pop('treatment', None)
        return result

    primary_hypothesis: Hypothesis | None
    differential_diagnoses: list[Hypothesis]
    patient_case_evidence: list[str]
    medical_knowledge_evidence: list[str]
    missing_information: list[str]
    uncertainty: list[str]
    unsupported_claims: list[str]
    reasoning_summary: str


class ClinicalCritique(StrictModel):
    overall_assessment: Literal['supported_with_limitations', 'revision_required', 'insufficient_evidence']
    supported_points: list[EvidenceClaim]
    unsupported_points: list[str]
    contradictions: list[str]
    missing_evidence: list[str]
    hallucination_flags: list[str]
    safety_flags: list[str]
    recommended_revisions: list[str]
    critique_confidence: Literal['low', 'medium', 'high']


class Failure(StrictModel):
    code: str
    message: str
    attempts: int = 0


class PipelineResult(StrictModel):
    patient_id: str
    status: Literal['success', 'failed'] = 'failed'
    stage: str = 'patient'
    patient_state: PatientState | None = None
    patient_case_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    medical_knowledge_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    diagnosis: DiagnosticResult | None = None
    critique: ClinicalCritique | None = None
    failure: Failure | None = None
    structured_validity: dict[str, bool] = Field(default_factory=dict)
    attempts: dict[str, int] = Field(default_factory=dict)
    retrieval_success: dict[str, bool] = Field(default_factory=dict)
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    llm_telemetry: dict[str, list[dict]] = Field(default_factory=dict)
    seconds: float = 0.0
    disclaimer: str = 'Synthetic-data research output only; not clinical advice or validated clinical decision support.'

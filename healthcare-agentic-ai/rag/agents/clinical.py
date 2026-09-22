"""Three roles sharing a stateless cloud provider, never a conversation history."""
from ..llm.provider import StructuredLLM, Generation
from ..models import PatientRepresentation
from .models import PatientState, DiagnosticResult, ClinicalCritique, RetrievedEvidence
from . import prompts
from .revision import DiagnosticRevisionInput
from .grounding import patient_input, validate_patient, context, validate_references


class PatientAgent:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def run(self, patient: PatientRepresentation, patient_id: str) -> Generation:
        payload = patient_input(patient, patient_id)
        return self.llm.generate(prompts.PATIENT, payload, PatientState,
                                 lambda result: validate_patient(result, payload))


class DiagnosticAgent:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def run(self, state: PatientState, cases: list[RetrievedEvidence], medical: list[RetrievedEvidence],
            *, revision: DiagnosticRevisionInput | None = None) -> Generation:
        payload = context(state, cases, medical)
        instructions = prompts.DIAGNOSTIC
        if revision is not None:
            revision = DiagnosticRevisionInput.model_validate(revision.model_dump())
            validate_references(revision.previous_diagnostic, state, cases, medical)
            validate_references(revision.critique, state, cases, medical)
            payload['revision_input'] = revision.model_dump()
            instructions += '\nClinical revision: follow revision_input.constraints. Feedback is not evidence.'
        return self.llm.generate(instructions, payload, DiagnosticResult,
                                 lambda result: validate_references(result, state, cases, medical))


class ClinicalCritic:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def run(self, state: PatientState, diagnosis: DiagnosticResult,
            cases: list[RetrievedEvidence], medical: list[RetrievedEvidence]) -> Generation:
        validate_references(diagnosis, state, cases, medical)
        payload = {**context(state, cases, medical), 'diagnostic_output': diagnosis.model_dump()}
        return self.llm.generate(prompts.CRITIC, payload, ClinicalCritique,
                                 lambda result: validate_references(result, state, cases, medical))

"""Explicit sequential inference only. No dataset labels or autonomous orchestration."""
import logging
import time
from typing import Protocol
from .models import PatientRepresentation
from .agents.clinical import PatientAgent, DiagnosticAgent, ClinicalCritic
from .agents.models import PipelineResult, Failure
from .agents.grounding import evidence_from_hits, patient_input
from .llm.provider import StructuredLLM

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]: ...


class Phase4Pipeline:
    def __init__(self, llm: StructuredLLM, patient_rag: Retriever, medical_rag: Retriever, *, top_k: int = 1):
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        self.patient = PatientAgent(llm)
        self.diagnostic = DiagnosticAgent(llm)
        self.critic = ClinicalCritic(llm)
        self.patient_rag, self.medical_rag, self.top_k = patient_rag, medical_rag, top_k

    def run(self, patient: PatientRepresentation, patient_id: str) -> PipelineResult:
        patient_input(patient, patient_id)  # Reject label-bearing records before any call.
        started = time.perf_counter()
        output = PipelineResult(patient_id=patient_id)

        def timed(name, function, *args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                output.stage_seconds[name] = time.perf_counter() - start

        def accept(generation, name):
            output.llm_telemetry[name] = generation.telemetry
            output.attempts[name] = generation.attempts
            output.structured_validity[name] = generation.parsed is not None and generation.failure is None
            if not output.structured_validity[name]:
                output.failure = generation.failure or Failure(code='invalid_output', message='No structured output returned.')
                return False
            return True

        try:
            generation = timed('patient', self.patient.run, patient, patient_id)
            if not accept(generation, 'patient'):
                return output
            output.patient_state = state = generation.parsed
            output.stage = 'retrieval'
            # State was verified verbatim against parser features. Reuse the existing
            # deterministic text serializer rather than a new patient summarizer.
            query = patient.to_text()
            for name, retriever in [('patient_case', self.patient_rag), ('medical_knowledge', self.medical_rag)]:
                output.retrieval_success[name] = False
                hits = timed(name, retriever.retrieve, query, top_k=self.top_k)
                evidence = evidence_from_hits(hits, name)
                setattr(output, name + '_evidence', evidence)
                output.retrieval_success[name] = bool(evidence)
            output.stage = 'diagnostic'
            generation = timed('diagnostic', self.diagnostic.run, state, output.patient_case_evidence, output.medical_knowledge_evidence)
            if not accept(generation, 'diagnostic'):
                return output
            output.diagnosis = generation.parsed
            output.stage = 'critic'
            generation = timed('critic', self.critic.run, state, output.diagnosis, output.patient_case_evidence, output.medical_knowledge_evidence)
            if not accept(generation, 'critic'):
                return output
            output.critique = generation.parsed
            output.status, output.stage = 'success', 'complete'
        except Exception:
            # Retrieval libraries may include source content in exception messages.
            logger.warning('Phase 4 stage %s failed (details omitted)', output.stage)
            output.failure = Failure(code='stage_failure', message=f'{output.stage} failed; inspect local configuration and data contracts.')
        finally:
            output.seconds = time.perf_counter() - started
        return output

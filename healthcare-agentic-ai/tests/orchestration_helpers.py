"""Offline fixtures exercising the real agents and production provider repair path."""
import copy
from dataclasses import asdict
import json
from unittest.mock import Mock
from rag.agents.models import DiagnosticResult, ClinicalCritique
from rag.patient_ingestion import patient_document
from rag.medical_ingestion.chunker import chunk_document
from orchestration import Orchestrator
from tests.test_patient_rag import record
from tests.test_medical_rag import document
from tests.phase4_helpers import MockLLM, envelope


def diagnosis(number=0, *, abstain=False):
    return DiagnosticResult(primary_hypothesis=None if abstain else {
        'condition': 'Research hypothesis', 'rationale': {'statement': 'Tentative fixture', 'evidence_refs': ['patient:age']},
        'supporting_evidence': [], 'contradicting_evidence': []},
        differential_diagnoses=[], patient_case_evidence=[], medical_knowledge_evidence=[],
        missing_information=[], uncertainty=['Research only'], unsupported_claims=[],
        reasoning_summary=f'Research fixture version {number}')


def critique(assessment='supported_with_limitations', **changes):
    data = dict(overall_assessment=assessment, supported_points=[], unsupported_points=[], contradictions=[],
        missing_evidence=[], hallucination_flags=[], safety_flags=[], recommended_revisions=[], critique_confidence='low')
    return ClinicalCritique(**{**data, **changes})


class ScenarioLLM(MockLLM):
    def __init__(self, diagnoses=None, critiques=None, **kwargs):
        self.diagnoses = list(diagnoses or [diagnosis()])
        self.critiques = list(critiques or [critique()])
        self.counts = {'DiagnosticResult': 0, 'ClinicalCritique': 0}
        super().__init__(**kwargs)

    def respond(self, **request):
        default = super().respond(**request)
        title = request['text']['format']['schema']['title']
        if title in self.counts:
            options = self.diagnoses if title == 'DiagnosticResult' else self.critiques
            index = self.counts[title]
            self.counts[title] += 1
            value = options[min(index, len(options) - 1)]
            value = value.model_dump() if hasattr(value, 'model_dump') else value
            return envelope(json.dumps(copy.deepcopy(value)))
        return default


def setup(llm=None, **kwargs):
    case = record(split='validate')
    patient_hit = {**patient_document(record()), 'score': -0.25}
    medical_hit = {**asdict(next(chunk_document(document()))), 'score': 0.6}
    patients, medical = Mock(), Mock()
    patients.retrieve.return_value = [patient_hit]
    medical.retrieve.return_value = [medical_hit]
    llm = llm or ScenarioLLM()
    orchestrator = Orchestrator(llm, patients, medical, **kwargs)
    return orchestrator, case, llm, patients, medical


def run(fixture):
    orchestrator, case, *_ = fixture
    return orchestrator.run(case.patient, case.patient_id)

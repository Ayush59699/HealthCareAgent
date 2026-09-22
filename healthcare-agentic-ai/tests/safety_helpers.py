"""Offline Phase 6 fixtures using the unchanged provider's strict repair path."""
import copy
from dataclasses import asdict
import json
from unittest.mock import Mock
from rag.llm.provider import Generation
from rag.medical_ingestion.chunker import chunk_document
from rag.patient_ingestion import patient_document
from safety.models import CATEGORIES, SemanticSafetyResult, SafetyInput
from orchestration.phase6 import Phase6Orchestrator
from orchestration.phase6.contracts import SafetyTicket
from orchestration.evidence import fingerprint
from tests.orchestration_helpers import ScenarioLLM, diagnosis as baseline_diagnosis, critique
from tests.phase4_helpers import envelope
from tests.test_medical_rag import document
from tests.test_patient_rag import record


def medical_hit():
    return {**asdict(next(chunk_document(document()))), 'score': 0.6}


def diagnosis(number=0, *, abstain=False, medical_reference=True):
    result = baseline_diagnosis(number, abstain=abstain)
    if not abstain and medical_reference:
        ref = 'medical:' + medical_hit()['chunk_id']
        result.primary_hypothesis.rationale.evidence_refs = [ref]
        result.medical_knowledge_evidence = [ref]
    return result


def semantic(category=None, *, status='uncertain', target='diagnostic', path='/reasoning_summary',
             quote='Research fixture version 0', refs=None):
    categories = []
    for name in CATEGORIES:
        findings = []
        result = 'no_issue_identified'
        if name == category:
            result = status
            if status == 'issue_identified':
                findings = [{'explanation': 'Synthetic test concern, not clinical adjudication.',
                             'anchor': {'target': target, 'field_path': path, 'excerpt': quote},
                             'evidence_refs': refs or []}]
        categories.append({'category': name, 'result': result, 'explanation': 'Synthetic fixture assessment.', 'findings': findings})
    return SemanticSafetyResult(categories=categories, limitations=['Synthetic software fixture; not clinical validation.'])


class SafetyScenarioLLM(ScenarioLLM):
    def __init__(self, diagnoses=None, critiques=None, safety_results=None, **kwargs):
        self.safety_results = list(safety_results or [semantic()])
        self.safety_count = 0
        super().__init__(diagnoses=diagnoses or [diagnosis()], critiques=critiques, **kwargs)

    def respond(self, **request):
        if request['text']['format']['schema']['title'] != 'SemanticSafetyResult':
            return super().respond(**request)
        self.calls.append(copy.deepcopy(request))
        value = self.safety_results[min(self.safety_count, len(self.safety_results) - 1)]
        self.safety_count += 1
        if isinstance(value, Exception):
            raise value
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        if isinstance(value, dict) and 'status' in value:
            return copy.deepcopy(value)
        return envelope(value if isinstance(value, str) else json.dumps(value))


def setup(llm=None, **kwargs):
    case = record(split='validate')
    patients, medical = Mock(), Mock()
    patients.retrieve.return_value = [{**patient_document(record()), 'score': -0.25}]
    medical.retrieve.return_value = [medical_hit()]
    llm = llm or SafetyScenarioLLM()
    orchestrator = Phase6Orchestrator(llm, patients, medical, **kwargs)
    return orchestrator, case, llm, patients, medical


def run(fixture):
    return fixture[0].run(fixture[1].patient, fixture[1].patient_id)


def inputs():
    state = run(setup())
    cases, medical = state.evidence.agent_evidence()
    value = SafetyInput(patient_state=state.patient_state, diagnostic=state.diagnostics[-1].result,
                        critique=state.critiques[-1].result, patient_cases=cases, medical_knowledge=medical)
    return value, state.safety_assessments[-1].ticket


def reticket(value, ticket):
    return SafetyTicket(**{**ticket.model_dump(), 'input_fingerprint': fingerprint(value.model_dump()),
                          'diagnostic_fingerprint': fingerprint(value.diagnostic.model_dump()),
                          'critic_fingerprint': fingerprint(value.critique.model_dump())})


def generation(parsed=None, *, failure=None, raw=None):
    return Generation(parsed=parsed, failure=failure, raw=raw, attempts=1,
                      telemetry=[{'attempt': 1, 'request_bytes': 100, 'api_success': True,
                                  'accepted': failure is None, 'request_seconds': 0.01}])

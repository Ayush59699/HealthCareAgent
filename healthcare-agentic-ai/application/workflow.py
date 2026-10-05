"""Single default composition: validated Patient -> CHECK4 WHO + unchanged AMG."""
from orchestration.phase6 import Phase6Orchestrator
from rag.selector_evidence_service import SelectorEvidenceService
from rag.who_amg_experiment import EvidenceInputProvider


def create_workflow(provider, patients, medical, *, top_k=1, medical_top_k=5,
                    policy=None, orchestrator_class=Phase6Orchestrator, who_selector=None):
    service = SelectorEvidenceService(provider, patients, medical, top_k=top_k,
                                     medical_top_k=medical_top_k, selector=who_selector)
    return orchestrator_class(EvidenceInputProvider(provider), patients, medical,
        top_k=top_k, policy=policy, evidence_service=service, require_medical_evidence=True, enable_treatment=True)

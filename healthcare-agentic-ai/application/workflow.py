"""The single supported workflow factory; legacy RAG is never selected here."""
from orchestration.phase6 import Phase6Orchestrator
from rag.amg.service import AMGEvidenceService


def create_workflow(provider, patients, medical, *, top_k=1, medical_top_k=5,
                    policy=None, orchestrator_class=Phase6Orchestrator):
    service = AMGEvidenceService(patients, medical, top_k=top_k, medical_top_k=medical_top_k)
    return orchestrator_class(provider, patients, medical, top_k=top_k, policy=policy,
        evidence_service=service, require_medical_evidence=True)

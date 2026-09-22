"""Application-owned invocation identities; the model cannot author routing IDs."""
from dataclasses import dataclass
from pydantic import ConfigDict
from rag.agents.models import StrictModel
from rag.agents.revision import DiagnosticRevisionInput
from rag.llm.provider import Generation
from .events import Stage
from .policy import WorkflowStop


class StageTicket(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    run_id: str
    patient_id: str
    stage: Stage
    revision_number: int
    diagnostic_version: int | None = None
    evidence_snapshot_id: str | None = None
    diagnostic_fingerprint: str | None = None


@dataclass(frozen=True)
class StageResponse:
    ticket: StageTicket
    generation: Generation


def require_current(expected: StageTicket, returned: StageTicket) -> None:
    if expected != returned:
        raise WorkflowStop('stale_result')


__all__ = ['StageTicket', 'StageResponse', 'require_current', 'DiagnosticRevisionInput']

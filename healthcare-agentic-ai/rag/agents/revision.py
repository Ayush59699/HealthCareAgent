"""Restricted clinical revision input for the existing Diagnostic Agent."""
from typing import Literal
from pydantic import ConfigDict
from .models import StrictModel, DiagnosticResult, ClinicalCritique

REVISION_CONSTRAINTS = (
    'Use the same patient facts and frozen retrieved evidence only. '
    'The previous diagnosis and critic feedback are review data, NOT evidence. '
    'Address validated feedback without inventing facts, sources or diagnoses. '
    'Preserve the abstention contract and all schema and grounding rules. '
    'Return a complete DiagnosticResult, not a patch or conversation.'
)


class DiagnosticRevisionInput(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    previous_diagnostic: DiagnosticResult
    critique: ClinicalCritique
    revision_number: Literal[1, 2]
    constraints: Literal[REVISION_CONSTRAINTS] = REVISION_CONSTRAINTS

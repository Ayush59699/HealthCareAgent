"""Stateless semantic service; application policy and state remain outside the LLM."""
from rag.llm.provider import StructuredLLM
from .models import SemanticSafetyResult
from .validation import validate_input, validate_semantic
from .prompts import SAFETY


class SafetyValidator:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def run(self, supplied):
        supplied = validate_input(supplied)
        # validate_input reconstructs detached models. No tickets or workflow history go to the LLM.
        return self.llm.generate(SAFETY, supplied.model_dump(), SemanticSafetyResult,
                                 lambda result: validate_semantic(result, supplied))

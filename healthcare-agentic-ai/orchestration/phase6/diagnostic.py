"""Phase-6-only prompt contract; Phase 4/5 instructions remain byte-for-byte intact."""
from rag.agents import prompts
from rag.agents.models import DiagnosticResult, ClinicalCritique
from rag.agents.grounding import context, validate_references, facts
from rag.agents.revision import DiagnosticRevisionInput

PROMPT_VERSION = 'phase6-diagnostic-v3-management-separation'
DIAGNOSTIC = prompts.DIAGNOSTIC.replace(
    'ROLE: Diagnostic Agent (phase4-v2-abstention).',
    'ROLE: Diagnostic Agent (' + PROMPT_VERSION + ').').replace(
    'Do not invent diagnoses as facts. Hypotheses must be justified by supplied material;\nabstain (primary_hypothesis=null, differential_diagnoses=[]) if insufficient.\nNo medical evidence means abstain. Missing/irrelevant retrieval must be acknowledged.',
    '''Do not invent diagnoses as facts. Missing/irrelevant retrieval must be acknowledged.
CASE A: Relevant medical material permits a cautious differential with actual supporting
citations, uncertainty and a clear distinction between observations and inference.
CASE B: Weak/absent medical retrieval is NOT automatic abstention. With meaningful
patient observations you MAY propose cautious hypotheses using clinical reasoning.
For EVERY hypothesis lacking structured medical references, rationale.statement MUST
start exactly "Uncertain model inference:" and explain that patient facts are observations,
not proof of the proposed condition. Cite the actual relevant patient symptom/presenting
fact IDs, not demographics alone, case analogies or irrelevant medical sources.
List missing critical information and uncertainty explicitly (nonempty lists).
These are unconfirmed possibilities, not established diagnoses or evidence-backed causes.
No invented medical support, case outcomes, numerical confidence, prescribing or advice.
CASE C: If patient observations cannot support even a cautious hypothesis, abstain and
explain the insufficiency in uncertainty and list critical missing information.
Do not guess to avoid abstention. Retained retrieval is only candidate context: a lexical
match does not prove relevance, entailment, or support for any particular hypothesis.''')


DIAGNOSTIC += """
Set treatment=null. A separate management agent owns that field after diagnosis.
On revision, revise diagnostic reasoning only; do not copy a previous treatment plan.

PATIENT EVIDENCE INVENTORY is the authoritative ID-to-fact binding.
Never infer the meaning of an evidence ID from its numeric index. Copy the exact
ID and meaning from PATIENT_EVIDENCE_INVENTORY; do not count or shift list entries.
Before returning, compare EVERY patient citation with its exact_fact, meaning and
value. Cite dyspnea only using its actual dyspnea fact, never an onset score.
Keep patient observations in short, separate, atomic clauses; distinguish those
observations from hypothetical mechanisms, absent findings and missing data.
Every observation in a claim must be supported by that claim's own references.
A reference elsewhere in the diagnosis cannot repair a wrong citation here.
Yes/No/unknown remain distinct. Numeric scale answers remain uninterpreted scores;
never translate undefined onset/localization scores into rapid onset or severity.
Existing IDs are not sufficient: a bounded deterministic observation/reference
check rejects recognizable mismatches before critic review. This partial check
does not prove clinical entailment. Never force an irrelevant medical citation.
"""


def diagnostic_context(state, cases, medical):
    from .reference_entailment import patient_evidence_inventory
    return {**context(state, cases, medical),
            'PATIENT_EVIDENCE_INVENTORY': patient_evidence_inventory(state)}


def validate_inference_contract(output, state, medical):
    """Mechanical disclosure/fact-link gate; critic/safety still judge semantics."""
    if not isinstance(output, DiagnosticResult):
        return
    from .reference_entailment import validate_reference_consistency
    validate_reference_consistency(output, state)
    hypotheses = ([output.primary_hypothesis] if output.primary_hypothesis else []) + output.differential_diagnoses
    medical_ids = {e.source_id for e in medical}
    observed = {key for key, value in facts(state).items()
                if key.startswith(('patient:symptoms:', 'patient:presenting_evidence:')) and str(value).strip()}
    if hypotheses and not observed:
        raise ValueError('Patient observations are insufficient; abstention required')
    for hypothesis in hypotheses:
        claims = [hypothesis.rationale, *hypothesis.supporting_evidence, *hypothesis.contradicting_evidence]
        refs = {ref for claim in claims for ref in claim.evidence_refs}
        if not refs & medical_ids:
            if (not hypothesis.rationale.statement.startswith('Uncertain model inference:') or
                    not set(hypothesis.rationale.evidence_refs) & observed or
                    not any(s.strip() for s in output.uncertainty) or
                    not any(s.strip() for s in output.missing_information)):
                raise ValueError('Patient-only inference requires explicit uncertainty, observed facts and missingness')
    if not hypotheses and (not any(s.strip() for s in output.uncertainty) or
                           not any(s.strip() for s in output.missing_information)):
        raise ValueError('Abstention requires insufficiency explanation and critical missing information')


def ground(output, state, cases, medical):
    return validate_references(output, state, cases, medical, allow_patient_inference=True)


class Phase6DiagnosticAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state, cases, medical, *, revision=None):
        payload = diagnostic_context(state, cases, medical)
        instructions = DIAGNOSTIC
        if revision is not None:
            revision = DiagnosticRevisionInput.model_validate(revision.model_dump())
            ground(revision.previous_diagnostic, state, cases, medical)
            ground(revision.critique, state, cases, medical)
            payload['revision_input'] = revision.model_dump()
            instructions += '\nClinical revision: follow revision_input.constraints. Feedback is not evidence.'
        def validate_diagnosis(result):
            if result.treatment is not None:
                raise ValueError('Management must be generated by the separate treatment agent')
            ground(result, state, cases, medical)
        return self.llm.generate(instructions, payload, DiagnosticResult, validate_diagnosis)


CRITIC_PROMPT_VERSION = 'phase6-critic-grounding-treatment-v2'
CRITIC = prompts.CRITIC + '\nROLE CONTRACT: ' + CRITIC_PROMPT_VERSION + """
GROUNDING_RESULT records deterministic reference/provenance checks, not clinical approval.
Evaluate each important medical claim against the actual cited passage, including
contradictions and whether a citation is only tangential. A reference_present_not_verified
entry must NOT be treated as evidence of semantic support. Patient-only hypotheses
remain unsupported by external medical evidence even when patient references are valid.
Independently identify unsupported certainty, unsafe reasoning and missing critical
information in the existing structured findings. Do not copy the grounding status as
an overall clinical verdict. Analogous training cases are not this patient's facts.
Also review diagnostic_output.treatment when present: each management action's
patient applicability, actual passage support, missing prerequisites, contraindication
uncertainty, unsafe prescribing/action instructions or unsafe delay. Deferred management
is not a recommendation or reassurance. Report concerns using the existing fields.
"""


class Phase6ClinicalCritic:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state, diagnosis, cases, medical):
        from .grounding import grounding_report
        report = grounding_report(diagnosis, state, cases, medical)
        payload = {**diagnostic_context(state, cases, medical), 'diagnostic_output': diagnosis.model_dump(),
                   'GROUNDING_RESULT': report.model_dump()}
        return self.llm.generate(CRITIC, payload, ClinicalCritique,
                                 lambda result: ground(result, state, cases, medical))

"""Bounded patient-fact/reference consistency, NOT general clinical entailment.

Only recognizable, uncomplicated observation clauses are checked. Unknown prose,
causality, temporal interpretation, disease ranking and external-source entailment
remain the critic's responsibility. No fact is inferred from an ID's position.
"""
import re
from rag.agents.grounding import GroundingError, claims, facts

VERSION = 'patient-reference-consistency-v1'
# Separate from retrieval aliases: these patterns describe observations, not
# diagnoses, and do not equate a history of asthma with current dyspnea.
PATTERNS = {
    'dyspnea': r'\b(?:dyspn(?:ea|oea)|shortness of breath|difficulty breathing|breathlessness|breathing difficulty)\b',
    'choking': r'\b(?:choking|suffocating)\b',
    'palpitations': r'\b(?:palpitations?|heart racing|racing heart)\b',
    'tingling': r'\b(?:tingling|numbness|par(?:a)?esthesia\w*)\b',
    'detachment': r'\b(?:detach(?:ed|ment)|depersonalization|derealization)\b',
    'fear_of_dying': r'\b(?:fear of dying|afraid of dying|feel like you are dying)\b',
    'sweating': r'\b(?:sweating|diaphoresis)\b',
    'chest_pain': r'\b(?:chest pain|chest tightness|side of the chest)\b',
    'fever': r'\b(?:fever|febrile)\b',
    'cough': r'\b(?:cough|coughing)\b',
    'wheezing': r'\b(?:wheez(?:e|ing))\b',
    'onset_score': r'\b(?:onset (?:speed|score)|how fast.*(?:appear|onset))\b',
    'localization_score': r'\b(?:localization precision|localization score|how precis(?:e|ely))\b',
}
# Deliberately abstain on complex modality/negation rather than pretending that
# keyword overlap proves a medical assertion. Clauses with explicit simple
# absence/unknown wording are handled separately below.
COMPLEX = re.compile(r'\b(?:may|might|could|can|possible|possibly|if|unless|whether|risk|rule out|not confirmed|cannot|without|does not|do not|not established)\b', re.I)
NEGATIVE = re.compile(r'\b(?:no|denies|denied|absent)\b', re.I)
UNKNOWN = re.compile(r'\b(?:unknown|unreported|not reported|not documented|unavailable|missing)\b', re.I)


def recognized(text):
    return {name for name, pattern in PATTERNS.items() if re.search(pattern, text, re.I)}


def patient_evidence_inventory(state):
    """Exact ID/meaning/value bindings, including nulls and uninterpreted scales."""
    inventory = []
    for ref, fact in facts(state).items():
        if isinstance(fact, str) and ' = ' in fact:
            meaning, value = fact.split(' = ', 1)
        else:
            meaning, value = (ref.removeprefix('patient:'), fact)
        inventory.append({'evidence_id': ref, 'exact_fact': fact,
                          'meaning': meaning, 'value': value})
    return inventory


def fact_concepts(fact):
    if not isinstance(fact, str):
        return set(), 'unknown'
    question, separator, answer = fact.partition(' = ')
    answer = answer.strip().lower()
    names = recognized(fact)
    if separator:
        if answer in {'no', 'none', 'nowhere'}:
            return names, 'negative'
        if answer in {'unknown', '', 'not reported'}:
            return names, 'unknown'
        # A numeric response is not evidence of an affirmative symptom or its
        # severity/onset. Retain only literal scale concepts.
        if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', answer):
            return names & {'onset_score', 'localization_score'}, 'recorded'
        return names, 'positive'
    if UNKNOWN.search(fact):
        return names, 'unknown'
    if NEGATIVE.search(fact):
        return names, 'negative'
    return names, 'positive'


def inspect_reference_consistency(diagnosis, state):
    """Return an auditable partial check; no-match is NOT a semantic PASS.

    For a simple observation clause, all recognized concepts must occur among
    that claim's cited patient facts, with matching polarity. Other references
    (medical passages, case analogies, demographics) cannot supply patient facts.
    Complex clauses are explicitly unassessed, not silently called supported.
    """
    inventory = facts(state)
    checks, unassessed = [], 0
    for index, claim in enumerate(claims(diagnosis)):
        patient_refs = [ref for ref in claim.evidence_refs if ref in inventory]
        # A medical-only claim may describe general knowledge, not this patient.
        if not patient_refs:
            unassessed += 1
            continue
        cited = [fact_concepts(inventory[ref]) for ref in patient_refs]
        # Split conjunctions only before explicit polarity markers; plain lists
        # such as "dyspnea and tingling" must be supported in full.
        clauses = re.split(r'[.;\n]|\b(?:but|whereas)\b|\band\s+(?=no\b|denies\b)', claim.statement, flags=re.I)
        for clause in clauses:
            if not clause.strip():
                continue
            names = recognized(clause)
            if not names or COMPLEX.search(clause):
                unassessed += 1
                continue
            polarity = 'unknown' if UNKNOWN.search(clause) else 'negative' if NEGATIVE.search(clause) else 'positive'
            for name in sorted(names):
                present = [(concepts, status) for concepts, status in cited if name in concepts]
                compatible = any(status == polarity or (status == 'recorded' and polarity == 'positive')
                                 for _, status in present)
                checks.append({'claim_index': index, 'concept': name,
                               'patient_refs': patient_refs,
                               'result': 'consistent' if compatible else 'mismatch',
                               'reason': 'concept_and_polarity_match' if compatible else
                                         'patient_reference_polarity_mismatch' if present else
                                         'patient_reference_concept_mismatch'})
    return {'version': VERSION, 'checks': checks, 'unassessed_clauses': unassessed,
            'status': 'mismatch' if any(c['result'] == 'mismatch' for c in checks) else
                      'no_mismatch_in_checked_observations' if checks else 'unassessed'}


def validate_reference_consistency(diagnosis, state):
    audit = inspect_reference_consistency(diagnosis, state)
    for check in audit['checks']:
        if check['result'] == 'mismatch':
            # Provider repair/logging must never contain raw clinical prose.
            raise GroundingError(check['reason'])
    return audit

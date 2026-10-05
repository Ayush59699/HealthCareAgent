"""Label-free focused evidence service with an isolated hybrid experiment.

The historical top-five gate is retained only as an explicit comparison baseline.
Use FocusedEvidenceService.with_hybrid() for task 7.4; downstream snapshots and
all grounding/safety contracts remain unchanged.
"""
import re
from rag.agents.models import PatientState
from rag.agents.grounding import evidence_from_hits, patient_input, validate_patient
from orchestration.evidence import EvidenceSnapshot, FrozenEvidence, fingerprint

POLICY_VERSION = 'focused-medical-v1'
# Retrieval synonyms only: never disease predictions. Unrecognized findings use
# bounded literal content words instead, rather than being assigned a diagnosis.
ALIASES = {
    'shortness of breath': ('shortness of breath', 'difficulty breathing', 'dyspnea', 'breathlessness'),
    'palpitations': ('palpitations', 'heart racing', 'racing heart'),
    'sweating': ('sweating', 'diaphoresis'),
    'chest pain': ('chest pain', 'chest tightness', 'side of the chest'),
    'numbness tingling': ('numbness', 'tingling', 'paresthesia'),
    'detachment': ('detached from', 'depersonalization', 'derealization'),
    'choking': ('choking', 'suffocating'),
    'fear of dying': ('feel like you are dying', 'fear of dying'),
    'anxiety': ('anxiety', 'anxious'),
    'depression': ('depression', 'depressive'),
    'asthma': ('asthma', 'bronchodilator'),
    'head trauma': ('head trauma', 'head injury'),
    'fever': ('fever', 'febrile'),
    'cough': ('cough', 'coughing'),
    'wheezing': ('wheezing', 'wheeze'),
    'fainting': ('fainting', 'syncope', 'loss of consciousness'),
    'vomiting': ('vomiting', 'vomit'),
    'headache': ('headache', 'headaches'),
}
STOP = set('have has had you your do does did are is was were a an the of in on to for or and with from it that this been ever currently recently somewhere feel like related reason consulting significantly experiencing significant way characterize how what when any members immediate yes no unknown not none nowhere pain precisely located fast appear intense right left r l'.split())


def words(text):
    return set(re.findall(r'[a-z]+', text.lower()))


def concepts(text):
    question, separator, answer = text.partition(' = ')
    if separator and answer.strip().lower() in {'no', 'unknown', 'none', 'nowhere'}:
        return []
    # Ordinal encodings cannot safely be translated into clinical severities.
    if separator and answer.strip().isdigit():
        return []
    text = text.lower()
    matched = [name for name, aliases in ALIASES.items() if any(alias in text for alias in aliases)]
    if matched:
        return matched
    tokens = [t for t in re.findall(r'[a-z]+', text) if t not in STOP and len(t) > 2]
    return [' '.join(list(dict.fromkeys(tokens))[:6])] if tokens else []


def unique(values):
    return list(dict.fromkeys(values))


def query_plan(state, embedding=None):
    if type(state) is not PatientState:
        raise TypeError('Validated label-free PatientState required')
    state = PatientState.model_validate(state.model_dump())
    symptoms = unique([c for text in state.presenting_evidence + state.symptoms for c in concepts(text)])[:12]
    history = unique([c for text in state.antecedents for c in concepts(text) if c not in symptoms])[:4]
    demographic = ('pediatric' if state.age < 18 else 'older adult' if state.age >= 65 else 'adult') if state.age is not None else ''
    sex = {'F': 'female', 'M': 'male'}.get(state.sex, '')
    query = ' '.join(filter(None, [demographic, sex, '; '.join(symptoms),
                    'history: ' + '; '.join(history) if history else '', 'clinical information red flags']))
    plan = {'query': query, 'symptom_concepts': symptoms, 'history_concepts': history, 'demographic': demographic}
    if embedding is not None:
        from rag.experimental_medical.queries import build_queries
        plan.update(build_queries(state, embedding))
    return plan


def matches(concept, text):
    tokens = words(text)
    alternatives = ALIASES.get(concept, (concept,))
    return any(words(alias) <= tokens for alias in alternatives)


def baseline_filter_candidates(plan, hits):
    # Validate ALL candidates before filtering so malformed provenance never hides
    # behind a relevance rejection. Nothing about a retained hit is rewritten.
    evidence_from_hits(hits, 'medical_knowledge')
    retained, audit = [], []
    for hit in sorted(hits, key=lambda h: (-h['score'], h['chunk_id'])):
        title_matches = [c for c in plan['symptom_concepts'] + plan['history_concepts'] if matches(c, hit['title'])]
        symptom_matches = [c for c in plan['symptom_concepts'] if matches(c, hit['text'])]
        keep = bool(title_matches) or len(symptom_matches) >= 2
        reason = 'topic_title_overlap' if title_matches else 'multiple_symptom_overlap' if keep else 'insufficient_concept_overlap'
        audit.append({'chunk_id': hit['chunk_id'], 'title': hit['title'], 'source': hit['source'],
                      'score': hit['score'], 'retained': keep, 'reason': reason,
                      'title_matches': title_matches, 'symptom_matches': symptom_matches})
        if keep:
            retained.append(hit)
    return retained, audit


def filter_candidates(plan, hits):
    """Compatibility API: validate ALL hits and annotate overlap, never discard.

    Relevance decisions belong after broad fusion and semantic reranking. This
    function's historical name is retained for callers, not its binary gate.
    """
    evidence_from_hits(hits, 'medical_knowledge')
    audit = []
    for hit in hits:
        audit.append({'chunk_id': hit['chunk_id'], 'title': hit['title'],
            'source': hit['source'], 'score': hit['score'], 'retained': True,
            'reason': 'candidate_for_reranking',
            'title_matches': [c for c in plan['symptom_concepts'] + plan['history_concepts'] if matches(c, hit['title'])],
            'symptom_matches': [c for c in plan['symptom_concepts'] if matches(c, hit['text'])]})
    return list(hits), audit


def state_from_patient(patient, patient_id):
    """Exact deterministic copy for local-only probes; not a substitute live agent."""
    supplied = patient_input(patient, patient_id)
    state = PatientState(patient_id=patient_id, age=supplied['age'], sex=supplied['sex'],
        symptoms=supplied['symptoms'], antecedents=supplied['antecedents'],
        presenting_evidence=supplied['initial_evidence'], relevant_findings=[],
        missing_information=supplied['allowed_missing_information'], uncertainty_notes=supplied['allowed_uncertainty_notes'])
    validate_patient(state, supplied)
    return state


class FocusedEvidenceService:
    def __init__(self, patients, medical, *, top_k=1, medical_top_k=5, hybrid=None, candidate_k=50):
        if any(type(k) is not int or k < 1 for k in (top_k, medical_top_k)):
            raise ValueError('Positive top-k required')
        self.patients, self.medical = patients, medical
        self.top_k, self.medical_top_k = top_k, medical_top_k
        if hybrid is not None and (medical_top_k > 5 or candidate_k not in (50, 100) or type(candidate_k) is not int):
            raise ValueError('Hybrid requires candidate_k 50/100 and final limit 1..5')
        self.hybrid, self.candidate_k = hybrid, candidate_k
        self.audit = None

    @classmethod
    def with_hybrid(cls, patients, medical, *, reranker=None, **kwargs):
        """Explicit retrieval-only opt-in; never loads a cloud service/fallback."""
        from rag.experimental_medical.qdrant_index import QdrantMedicalIndex
        from rag.experimental_medical.retrieval import HybridRetriever, LocalCrossEncoder
        hybrid = HybridRetriever(QdrantMedicalIndex(medical), medical.embedding,
                                 reranker if reranker is not None else LocalCrossEncoder())
        return cls(patients, medical, hybrid=hybrid, **kwargs)

    def retrieve(self, patient, state):
        self.audit = None
        validate_patient(state, patient_input(patient, state.patient_id))
        plan = query_plan(state)
        cases = evidence_from_hits(self.patients.retrieve(patient.to_text(), top_k=self.top_k), 'patient_case')
        if self.hybrid is not None:
            from rag.experimental_medical.qdrant_index import evidence_hits
            result = self.hybrid.retrieve(state, candidate_depth=self.candidate_k, final_k=self.medical_top_k)
            medical = evidence_from_hits(evidence_hits(result), 'medical_knowledge')
            self.audit = {**result, 'policy_version': result['version'],
                'status': 'candidate_evidence_retained' if medical else 'insufficient_or_irrelevant',
                'retained_count': len(medical)}
        else:
            # Archived baseline, NOT a fallback when hybrid retrieval fails.
            hits = self.medical.retrieve(plan['query'], top_k=self.medical_top_k) if plan['symptom_concepts'] else []
            retained, candidates = baseline_filter_candidates(plan, hits)
            medical = evidence_from_hits(retained, 'medical_knowledge')
            self.audit = {'policy_version': POLICY_VERSION, **plan, 'candidate_top_k': self.medical_top_k,
                          'status': 'candidate_evidence_retained' if medical else 'insufficient_or_irrelevant',
                          'candidates': candidates, 'retained_count': len(medical)}
        identity = fingerprint({'patient_cases': [e.model_dump() for e in cases],
                                'medical_knowledge': [e.model_dump() for e in medical]})
        return EvidenceSnapshot(snapshot_id=identity,
            patient_cases=tuple(FrozenEvidence.freeze(e) for e in cases),
            medical_knowledge=tuple(FrozenEvidence.freeze(e) for e in medical))

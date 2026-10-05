"""Isolated WHO/AMG composition. No changes to either retriever or safety policy."""
import copy
import json
from typing import Literal
from pydantic import ConfigDict, model_validator
from rag.agents.models import StrictModel
from rag.agents.grounding import evidence_from_hits, patient_input, validate_patient
from orchestration.evidence import EvidenceSnapshot, FrozenEvidence, fingerprint
from rag.amg.service import AMGEvidenceService
from rag.who_lookup import WHOLookup, MAX_QUERY_CHARS
from rag.who_provenance import excerpt_hit

MODES = ('amg_only', 'who_only', 'who_plus_amg')


class FrozenContract(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)


class WHOEvidence(FrozenContract):
    fact_sheets: tuple[FrozenEvidence, ...] = ()
    questions_answers: tuple[FrozenEvidence, ...] = ()


class AMGEvidence(FrozenContract):
    accepted_passages: tuple[FrozenEvidence, ...] = ()


class EvidenceQueries(FrozenContract):
    who: str | None
    amg: str | None
    who_omitted_fact_refs: tuple[str, ...] = ()


class CombinedMedicalEvidence(FrozenContract):
    who_evidence: WHOEvidence
    amg_evidence: AMGEvidence
    query: EvidenceQueries
    provenance_json: str  # Immutable, lossless audit including rejected AMG candidates.
    status: Literal['BOTH_AVAILABLE', 'WHO_ONLY', 'AMG_ONLY', 'NONE_AVAILABLE']

    @model_validator(mode='after')
    def check_sources(self):
        who = self.who_evidence.fact_sheets + self.who_evidence.questions_answers
        amg = self.amg_evidence.accepted_passages
        expected = availability(bool(who), bool(amg))
        if self.status != expected:
            raise ValueError('Combined availability mismatch')
        for items, backend, kind in ((self.who_evidence.fact_sheets, 'who-local-lookup-v1', 'fact_sheet'),
                (self.who_evidence.questions_answers, 'who-local-lookup-v1', 'question_answer'),
                (amg, 'amg-medlineplus-v1', None)):
            for item in items:
                evidence = item.thaw()
                if (evidence.metadata.get('backend') != backend or
                        kind and evidence.metadata.get('document_type') != kind):
                    raise ValueError('Combined source category mismatch')
                rebuilt = evidence_from_hits([{**evidence.metadata, 'text': evidence.text,
                                               'score': evidence.similarity}], 'medical_knowledge')[0]
                if rebuilt != evidence:
                    raise ValueError('Combined provenance mismatch')
        selected = [e.thaw() for e in who if e.thaw().metadata.get('selection') == 'llm_catalog_verified_sections']
        if selected:
            from rag.who_sections import MAX_DOCUMENT_EXCERPT_CHARS
            if len(selected) != len(who):
                raise ValueError('Do not mix lexical and LLM-selected WHO evidence')
            documents = {}
            for item in selected:
                documents.setdefault(item.metadata['document_id'], []).append(item)
            if len(documents) > 3:
                raise ValueError('WHO document limit exceeded')
            for items in documents.values():
                if sum(len(e.text) for e in items) > MAX_DOCUMENT_EXCERPT_CHARS:
                    raise ValueError('WHO document section budget exceeded')
                spans = sorted((e.metadata['char_start'], e.metadata['char_end']) for e in items)
                if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
                    raise ValueError('Overlapping WHO source spans')
        if len({e.source_id for e in who + amg}) != len(who + amg):
            raise ValueError('Duplicate combined evidence')
        json.loads(self.provenance_json)
        return self

    def snapshot(self, patient_cases):
        # The existing transport requires a tuple. Each entry remains source-typed;
        # the agent input adapter restores the explicit source sections, no text blob.
        medical = (self.who_evidence.fact_sheets + self.who_evidence.questions_answers
                   + self.amg_evidence.accepted_passages)
        identity = fingerprint({'patient_cases': [e.thaw().model_dump() for e in patient_cases],
                                'medical_knowledge': [e.thaw().model_dump() for e in medical]})
        return EvidenceSnapshot(snapshot_id=identity, patient_cases=tuple(patient_cases), medical_knowledge=medical)


def availability(who, amg):
    return ('BOTH_AVAILABLE' if who and amg else 'WHO_ONLY' if who else
            'AMG_ONLY' if amg else 'NONE_AVAILABLE')


def who_query(state):
    """Whole literal facts only, source order, no diagnoses inferred or injected.

    Negated/uncertain facts remain literal: a lexical match is not a patient fact.
    WHO's own conservative matching is unchanged. No synonym or alias expansion.
    """
    parts, omitted, seen = [], [], set()
    for field in ('presenting_evidence', 'symptoms', 'antecedents'):
        for i, fact in enumerate(getattr(state, field)):
            if fact in seen:
                continue
            seen.add(fact)
            if len('; '.join(parts + [fact])) <= MAX_QUERY_CHARS:
                parts.append(fact)
            else:
                omitted.append(f'patient:{field}:{i}')
    return '; '.join(parts), tuple(omitted)


class CombinedEvidenceService:
    requires_patient_state = True

    def __init__(self, patient_rag, medical_rag=None, *, mode='who_plus_amg', who=None):
        if mode not in MODES:
            raise ValueError('Unknown experimental evidence mode')
        if mode != 'who_only' and medical_rag is None:
            raise ValueError('AMG backend required')
        self.mode = mode
        self.patient_rag = patient_rag
        self.amg = AMGEvidenceService(patient_rag, medical_rag) if mode != 'who_only' else None
        self.who = (who if who is not None else WHOLookup()) if mode != 'amg_only' else None
        self.audit = self.combined = None

    def retrieve(self, patient, state):
        self.audit = self.combined = None
        validate_patient(state, patient_input(patient, state.patient_id))
        # Supported AMG service interface, defaults unchanged (patient k=1, medical k=5).
        if self.amg is not None:
            original = self.amg.retrieve(patient, state)
            cases, amg = original.patient_cases, original.medical_knowledge
            amg_audit = copy.deepcopy(self.amg.audit)
        else:
            cases = tuple(FrozenEvidence.freeze(e) for e in evidence_from_hits(
                self.patient_rag.retrieve(patient.to_text(), top_k=1), 'patient_case'))
            amg, amg_audit = (), None
        who_items, query, omitted = (), None, ()
        who_audit = {'status': 'NOT_REQUESTED', 'documents': [], 'errors': []}
        if self.who is not None:
            query, omitted = who_query(state)
            result = self.who.lookup(query, max_results=3)
            who_items = tuple(FrozenEvidence.freeze(e) for e in evidence_from_hits(
                [excerpt_hit(match) for match in result.documents], 'medical_knowledge'))
            who_audit = {'status': result.status, 'reason': result.reason, 'errors': list(result.errors),
                         'documents': [json.loads(e.metadata_json) for e in who_items],
                         'selection': 'At most three literal lookup matches; exact body prefixes <=4000 characters each.'}
        audit = {'backend': 'experimental-who-amg-v1', 'mode': self.mode,
                 'who': who_audit, 'amg': amg_audit,
                 'status': 'evidence_found' if who_items or amg else 'insufficient_evidence',
                 'combined_status': availability(bool(who_items), bool(amg))}
        combined = CombinedMedicalEvidence(
            who_evidence=WHOEvidence(
                fact_sheets=tuple(e for e in who_items if e.thaw().metadata['document_type'] == 'fact_sheet'),
                questions_answers=tuple(e for e in who_items if e.thaw().metadata['document_type'] == 'question_answer')),
            amg_evidence=AMGEvidence(accepted_passages=amg),
            query=EvidenceQueries(who=query, amg=amg_audit['query'] if amg_audit else None,
                                  who_omitted_fact_refs=omitted),
            provenance_json=json.dumps(audit, ensure_ascii=False, sort_keys=True), status=audit['combined_status'])
        snapshot = combined.snapshot(cases)
        self.audit, self.combined = audit, combined
        return snapshot

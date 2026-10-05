"""Default sequential CHECK4 WHO + unchanged AMG evidence composition."""
import copy
import json
from types import SimpleNamespace
from rag.agents.grounding import evidence_from_hits, patient_input, validate_patient
from rag.amg.service import AMGEvidenceService
from rag.combined_medical_evidence import (CombinedMedicalEvidence, WHOEvidence, AMGEvidence,
                                           EvidenceQueries, availability)
from rag.who_catalog_selector import WHOCatalogSelector, WHOSelection, digest
from rag.who_selector_relevance import select_verified, ContentRelevance
from rag.who_sections import section_hits
from orchestration.evidence import FrozenEvidence

BACKEND = 'who-check4-amg-v1'


class SelectorEvidenceService:
    requires_patient_state = True
    requires_generation = True

    def __init__(self, provider, patients, medical, *, top_k=1, medical_top_k=5, selector=None):
        # Reuse CHECK4 runner limits, not the Diagnostic output/repair budget.
        # Shared client: same deployment, no extra connection or parallel call.
        from dataclasses import replace
        from rag.config import OpenAIConfig
        from rag.llm.provider import OpenAIProvider
        config = getattr(provider, 'config', None)
        self.provider = (OpenAIProvider(replace(config, max_retries=0, max_output_tokens=1800),
                                       client=provider.client)
                         if isinstance(config, OpenAIConfig) else provider)
        self.amg = AMGEvidenceService(patients, medical, top_k=top_k, medical_top_k=medical_top_k)
        self.selector = selector  # Lazy metadata initialization after validated Patient Agent.
        self.audit = self.combined = None

    def retrieve(self, patient, state, *, generate=None):
        self.audit = self.combined = None
        validate_patient(state, patient_input(patient, state.patient_id))
        selector = self.selector if self.selector is not None else WHOCatalogSelector()
        # The orchestrator supplies its budgeted/accounted generate callback. Direct
        # offline/test use may supply a provider; no parallel execution or fallback.
        errors = []
        self.audit = {'backend': BACKEND, 'mode': 'who_plus_amg', 'amg': None,
                      'who': {'status': 'selector_in_progress', 'validation_errors': errors},
                      'status': 'retrieval_incomplete'}
        def checked_generate(instructions, payload, schema, validator=None):
            def checked(value):
                try:
                    return validator(value) if validator else None
                except (ValueError, TypeError) as exc:
                    codes = {'Patient support is not literal input': 'nonliteral_patient_quote',
                             'Source support is not in document body': 'nonliteral_source_quote',
                             'Unknown WHO selection ID': 'unknown_document_id'}
                    errors.append({'schema': schema.__name__,
                                   'code': codes.get(str(exc), 'selection_contract_error'),
                                   'rejected_content_review': value.model_dump() if type(value) is ContentRelevance else None})
                    raise
            return (generate or self.provider.generate)(instructions, payload, schema, checked)
        try:
            selected = select_verified(selector, SimpleNamespace(generate=checked_generate), patient)
        except Exception:
            self.audit['who']['status'] = 'selector_or_budget_failure'
            raise
        selected['validation_errors'] = errors
        self.audit = {'backend': BACKEND, 'mode': 'who_plus_amg', 'who': selected,
                      'amg': None, 'status': 'retrieval_incomplete'}
        if selected['status'] not in ('selected_and_verified', 'no_selection'):
            raise ValueError('WHO selector/content review failed')
        hits = []
        candidates = {r['document']['document_id']: r for r in selected['candidates']}
        # Reopen at most three SELECTED documents sequentially, retaining only
        # bounded exact sections. Verify bytes/catalog against the reviewed version.
        for document in selector.iter_selected(WHOSelection(selected_ids=selected['selected_ids'])):
            row = candidates[document.document_id]
            if document.audit() != row['document']:
                raise ValueError('WHO document changed between review and handoff')
            review = ContentRelevance.model_validate(row['content_review']['parsed'])
            hits.extend(section_hits(document, review, patient))
        who_items = tuple(FrozenEvidence.freeze(e) for e in evidence_from_hits(hits, 'medical_knowledge'))
        # AMG gets exactly the same Patient State, query builder, parameters and
        # service implementation as before. It cannot influence WHO selection.
        original = self.amg.retrieve(patient, state)
        amg_audit = copy.deepcopy(self.amg.audit)
        selected.update(pipeline_run=True, errors=[], catalog_sha256=selector.catalog_sha256,
                        catalog_documents=len(selector.catalog_payload()['catalog']),
                        patient_payload_sha256=digest(patient.to_inference_dict()),
                        documents=[json.loads(e.metadata_json) for e in who_items],
                        handoff='Exact complete sections containing CHECK4-reviewed quotes; not body prefixes.')
        status = availability(bool(who_items), bool(original.medical_knowledge))
        self.audit.update(amg=amg_audit, combined_status=status,
                          status='evidence_found' if who_items or original.medical_knowledge else 'insufficient_evidence')
        self.combined = CombinedMedicalEvidence(
            who_evidence=WHOEvidence(
                fact_sheets=tuple(e for e in who_items if e.thaw().metadata['document_type'] == 'fact_sheet'),
                questions_answers=tuple(e for e in who_items if e.thaw().metadata['document_type'] == 'question_answer')),
            amg_evidence=AMGEvidence(accepted_passages=original.medical_knowledge),
            query=EvidenceQueries(who=None, amg=amg_audit['query']),
            provenance_json=json.dumps(self.audit, ensure_ascii=False, sort_keys=True), status=status)
        return self.combined.snapshot(original.patient_cases)

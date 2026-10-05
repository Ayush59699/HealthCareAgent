"""Synthetic software-contract fixtures; no benchmark or diagnosis labels."""
import hashlib
import json
from unittest.mock import Mock
from tests.safety_helpers import SafetyScenarioLLM, diagnosis
from rag.amg.provenance import BACKEND


def amg_hit():
    text = 'Fixture topic\n\nSynthetic medical source fixture, not clinical guidance.'
    sha = hashlib.sha256(text.encode()).hexdigest()
    metadata = dict(source='MedlinePlus', record_type='health_topic', topic_id='123',
        canonical_id='medlineplus:topic:123', title='Fixture topic',
        url='https://medlineplus.gov/fixture.html', language='English', date='2026-01-01',
        date_created='2026-01-01', date_generated='2026-01-01', source_archive='fixture.zip',
        source_member='fixture.xml', source_sha256='a'*64, parser_version='1', summary_status='present',
        content_sha256=sha, chunk_id=0, parent_id='medlineplus:topic:123', token_count=20,
        chunker_version='1', source_spans=json.dumps([{'start': 0, 'end': 30, 'split_sentence': False}]),
        excluded_summary_lines='[]', split_sentence=False, chunk_sha256=sha)
    for field in ('aliases', 'groups', 'mesh_terms', 'related_topics', 'see_references', 'primary_institutes'):
        metadata[field] = '[]'
    return dict(backend=BACKEND, chunk_id='medlineplus:topic:123:lab:1:chunk:0',
        document_id=metadata['parent_id'], source='medlineplus', title=metadata['title'], url=metadata['url'],
        text=text, amg_metadata=metadata, retrieval=dict(distance=0.5, metric='squared_l2',
        acceptance='distance_gate', max_distance=1.10, snapshot='a'*24,
        collection='medlineplus_en_experimental'), score=-0.5)


def configure_medical(medical):
    medical.retrieve.return_value = [amg_hit()]
    medical.query_for.return_value = ('Synthetic clinical observations', 0)
    medical.audit = {'results': [], 'candidates': []}
    medical.count.return_value = 1
    return medical


def configure_provider(provider):
    # Preserve each scenario's uncertainty/safety behavior, changing only its
    # synthetic source citation to the new native-ID evidence contract.
    for result in provider.diagnoses:
        if hasattr(result, 'primary_hypothesis') and result.primary_hypothesis:
            ref = 'medical:' + amg_hit()['chunk_id']
            result.primary_hypothesis.rationale.evidence_refs = [ref]
            result.medical_knowledge_evidence = [ref]
    return provider

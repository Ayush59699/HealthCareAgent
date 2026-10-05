"""Synthetic XML/integrity tests; no network, cached-model or LLM dependency."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from qdrant_client import models

from rag.medical_ingestion.common import download
from rag.medical_ingestion.medlineplus import XML_URL, SOURCE_VERSION, parse_medlineplus, summary_sections
from rag.medical_ingestion.chunker import chunk_document
from rag.medical_ingestion.corpus import prepare_corpus
from rag.medical_ingestion.indexing import build_index, audit_index, fingerprint
from rag.medical_ingestion.models import validate_chunk
from rag.medical_vector_store import MedicalVectorStore
from scripts.ingest_medlineplus import create_manifest, publish
from scripts.test_medical_rag import TOPIC_CASES, check_hits

XML = '''<health-topics total="4" date-generated="09/22/2026 02:30:40">
<health-topic id="100" title="Synthetic topic" url="https://medlineplus.gov/synthetic.html" language="English" date-created="01/01/2000">
<also-called>Fixture alias</also-called><group id="2" url="https://medlineplus.gov/g.html">Fixture group</group>
<mesh-heading><descriptor id="D000">Fixture heading</descriptor></mesh-heading>
<full-summary>&lt;h3&gt;Symptoms&lt;/h3&gt;&lt;p&gt;Synthetic summary about breathing and lung symptoms, not a real medical publication.&lt;/p&gt;&lt;script&gt;HIDDEN&lt;/script&gt;</full-summary>
<site title="EXTERNAL TEXT" url="https://external.invalid"/></health-topic>
<health-topic id="200" title="Otro" url="https://medlineplus.gov/spanish/x.html" language="Spanish"><full-summary>SPANISH EXCLUDED</full-summary></health-topic>
<health-topic id="300" title="Missing summary" url="https://medlineplus.gov/m.html" language="English"/>
<health-topic id="400" title="Empty summary" url="https://medlineplus.gov/e.html" language="English"><full-summary>&lt;p&gt; &lt;/p&gt;</full-summary></health-topic>
</health-topics>'''


class FixtureEmbedding:
    """Unit-test vectors only; never selected by production code."""
    dimension = 384
    batch_size = 16
    truncated_texts = 0
    signature = {'model_name': 'BAAI/bge-small-en-v1.5', 'dimension': 384,
                 'normalize_embeddings': True, 'pipeline': 'medical-knowledge-v1',
                 'max_seq_length': 512, 'text_format': 'medical-source-text-v1', 'prompt': ''}
    def embed_texts(self, texts):
        return [[1.] + [0.] * 383 for _ in texts]


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / 'mplus_topics_2026-09-22.xml'
        self.path.write_text(XML, encoding='utf8')
        self.meta = {'download_url': XML_URL, 'retrieved_at': '2026-09-22T00:00:00Z',
                     'raw_path': str(self.path), 'raw_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest()}
        self.path.with_suffix('.xml.provenance.json').write_text(json.dumps(self.meta), encoding='utf8')

    def docs(self):
        return list(parse_medlineplus(self.path, self.meta))

    def manifest(self):
        stats = {}
        list(parse_medlineplus(self.path, self.meta, stats=stats))
        path = self.root / 'corpus.json'
        path.write_text(json.dumps({'entries': [{'source': 'medlineplus', 'raw_path': self.path.name,
            'sha256': self.meta['raw_sha256'], 'selection': 'all-english-summaries',
            'source_version': SOURCE_VERSION, 'parse_stats': stats}]}), encoding='utf8')
        return path

    def build(self):
        manifest = self.manifest()
        storage = self.root / 'qdrant'
        build_index(manifest, storage, FixtureEmbedding())
        return manifest, storage

    def test_all_english_not_hardcoded_and_record_accounting(self):
        stats = {}
        docs = list(parse_medlineplus(self.path, self.meta, stats=stats))
        self.assertEqual(len(docs), 1)
        self.assertEqual(stats['english_records'], 3)
        self.assertEqual(stats['excluded_non_english'], 1)
        self.assertEqual(len(stats['missing_summary']), 2)
        self.assertEqual(docs[0].document_id, 'medlineplus:100')
        self.assertEqual(docs[0].language, 'en')
        self.assertEqual(docs[0].source_version, SOURCE_VERSION)
        self.assertNotIn('HIDDEN', docs[0].text)
        self.assertNotIn('EXTERNAL', docs[0].text)
        self.assertNotIn('<', docs[0].text)
        self.assertEqual(docs[0].metadata['also_called'], ['Fixture alias'])
        self.assertEqual(docs[0].metadata['mesh_headings'][0]['id'], 'D000')

    def test_nested_xml_headings_equivalent_to_escaped_html(self):
        from defusedxml import ElementTree as ET
        nested = ET.fromstring('<full-summary><h3>Title</h3><p>Some <b>bold</b> text.</p></full-summary>')
        escaped = ET.fromstring('<full-summary>&lt;h3&gt;Title&lt;/h3&gt;&lt;p&gt;Some &lt;b&gt;bold&lt;/b&gt; text.&lt;/p&gt;</full-summary>')
        self.assertEqual(summary_sections(nested), summary_sections(escaped))
        self.assertEqual(summary_sections(nested)[0]['title'], 'Title')

    def test_duplicate_ids_fail_closed(self):
        topic = '<health-topic id="100"' + XML.split('<health-topic id="100"')[1].split('</health-topic>')[0] + '</health-topic>'
        self.path.write_text(XML.replace('</health-topics>', topic + '</health-topics>'), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.docs()

    def test_missing_identity_fails_not_silently_skipped(self):
        self.path.write_text(XML.replace('id="100"', ''), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'required'):
            self.docs()

    def test_sections_deterministic_without_full_document_duplication(self):
        doc = self.docs()[0]
        chunks = list(chunk_document(doc))
        self.assertEqual(chunks, list(chunk_document(doc)))
        self.assertEqual(chunks[0].section, 'Symptoms')
        self.assertNotIn('sections', chunks[0].metadata)
        self.assertEqual(chunks[0].source_version, SOURCE_VERSION)
        self.assertEqual(chunks[0].chunking_version, 'sections-v2:180:30')
        self.assertTrue(all(len(c.text.split()) <= 180 for c in chunks))

    def test_short_sections_merge_and_long_sections_overlap(self):
        doc = self.docs()[0]
        sections = [{'title': 'First', 'text': 'short section'}, {'title': 'Second', 'text': 'another short section'}]
        chunks = list(chunk_document(replace(doc, metadata={'sections': sections})))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, 'First / Second')
        sections = [{'title': 'Long', 'text': ' '.join('word'+str(i) for i in range(400))}]
        chunks = list(chunk_document(replace(doc, metadata={'sections': sections})))
        self.assertEqual(chunks[0].text.split()[-30:], chunks[1].text.split()[:30])

    def test_legacy_payload_readable_but_extra_labels_rejected(self):
        payload = asdict(next(chunk_document(self.docs()[0])))
        historical = asdict(next(chunk_document(replace(self.docs()[0], metadata={}))))
        legacy = {k: v for k, v in historical.items() if k not in {'language', 'source_version', 'section', 'metadata'}}
        self.assertEqual(validate_chunk(legacy), legacy)
        with self.assertRaisesRegex(ValueError, 'expanded provenance'):
            validate_chunk({k: v for k, v in payload.items() if k not in {'language', 'source_version', 'section', 'metadata'}})
        for change in ({'PATHOLOGY': 'label'}, {'metadata': {'diagnosis': 'label'}}):
            with self.assertRaises(ValueError):
                validate_chunk({**payload, **change})

    def test_fingerprint_independent_of_local_download_details(self):
        doc = self.docs()[0]
        changed = replace(doc, provenance={**doc.provenance, 'raw_path': '/new/location', 'retrieved_at': '2030-01-01'})
        self.assertEqual(fingerprint(list(chunk_document(doc)), FixtureEmbedding.signature),
                         fingerprint(list(chunk_document(changed)), FixtureEmbedding.signature))
        changed = replace(doc, source_version='2026-09-23')
        self.assertNotEqual(fingerprint(list(chunk_document(doc)), FixtureEmbedding.signature),
                            fingerprint(list(chunk_document(changed)), FixtureEmbedding.signature))

    def test_manifest_stats_and_exact_source_enforced(self):
        path = self.manifest()
        data = json.loads(path.read_text())
        data['entries'][0]['parse_stats']['parsed_documents'] += 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'accounting'):
            prepare_corpus(path)
        data['entries'][0]['source_version'] = 'other'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'exact pinned'):
            prepare_corpus(path)

    def test_download_verified_cache_no_network(self):
        with patch('urllib.request.build_opener') as opener:
            meta = download(XML_URL, self.path, {'medlineplus.gov'})
            opener.assert_not_called()
        self.assertEqual(meta['file_size_bytes'], str(self.path.stat().st_size))
        self.path.write_text('corrupted')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            download(XML_URL, self.path, {'medlineplus.gov'})

    def test_download_http_failure_does_not_create_raw_file(self):
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.side_effect = OSError('HTTP failure fixture')
            with self.assertRaisesRegex(RuntimeError, 'No alternate source'):
                download(XML_URL, self.root / 'new.xml', {'medlineplus.gov'})
        self.assertFalse((self.root / 'new.xml').exists())

    def test_download_response_size_status_and_metadata(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.url, response.status, response.headers = XML_URL, 200, {'Content-Length': '3'}
        response.read.return_value = b'xml'
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = response
            meta = download(XML_URL, self.root / 'new.xml', {'medlineplus.gov'})
            self.assertEqual(meta['raw_sha256'], hashlib.sha256(b'xml').hexdigest())
            response.headers = {'Content-Length': '4'}
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                download(XML_URL, self.root / 'bad.xml', {'medlineplus.gov'})
            response.status = 503
            with self.assertRaisesRegex(RuntimeError, 'HTTP 200'):
                download(XML_URL, self.root / 'bad.xml', {'medlineplus.gov'})

    def test_pinned_source_refuses_redirect_to_another_snapshot(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.url = 'https://medlineplus.gov/xml/another.xml'
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = response
            with self.assertRaisesRegex(RuntimeError, 'Pinned source URL changed'):
                download(XML_URL, self.root / 'redirect.xml', {'medlineplus.gov'}, exact_url=True)
        self.assertFalse((self.root / 'redirect.xml').exists())

    def test_pmc_entries_preserved_not_reselected(self):
        entry = {'source': 'pmc', 'raw_path': 'existing.xml', 'sha256': '0'*64, 'pmcid': 'PMC123'}
        with patch('scripts.ingest_medlineplus.download_topics', return_value=(self.path, self.meta)):
            manifest, _ = create_manifest(self.root, {'entries': [entry]})
        self.assertEqual(manifest['entries'][1], entry)

    def test_audit_and_idempotent_rebuild(self):
        manifest, storage = self.build()
        before = audit_index(manifest, storage)
        self.assertTrue(before['passed'], before)
        build_index(manifest, storage, FixtureEmbedding())
        self.assertEqual(audit_index(manifest, storage), before)
        self.assertEqual(before['total_documents'], 1)
        self.assertEqual(before['total_chunks'], 1)
        self.assertEqual(before['embedding_failures'], 0)
        self.assertEqual(before['truncations'], 0)

    def test_processed_hash_matches_actual_utf8_lf_bytes(self):
        processed = self.root / 'processed'
        report = build_index(self.manifest(), self.root / 'qdrant', FixtureEmbedding(), processed_dir=processed)
        data = (processed / 'chunks.jsonl').read_bytes()
        self.assertNotIn(b'\r\n', data)
        self.assertEqual(hashlib.sha256(data).hexdigest(), report['processed_sha256'])
        self.assertNotIn(b'\r\n', (processed / 'documents.jsonl').read_bytes())

    def test_partial_embedding_fails_closed_and_audit_fails(self):
        embedding = FixtureEmbedding()
        embedding.embed_texts = lambda texts: []
        with self.assertRaisesRegex(ValueError, 'Embedding output count'):
            build_index(self.manifest(), self.root / 'qdrant', embedding)
        self.assertFalse(audit_index(self.manifest(), self.root / 'qdrant')['passed'])

    def test_audit_detects_metadata_corruption(self):
        manifest, storage = self.build()
        with MedicalVectorStore(storage, 384, embedding_signature=FixtureEmbedding.signature) as store:
            point = next(store.iter_points(with_vectors=True))
            point.payload['language'] = ''
            point.payload['text'] = ''
            store.client.upsert(store.collection_name, [models.PointStruct(id=point.id, payload=point.payload, vector=point.vector)])
        audit = audit_index(manifest, storage)
        self.assertFalse(audit['passed'])
        self.assertEqual(audit['empty_chunks'], 1)
        self.assertEqual(audit['malformed_metadata'], 1)
        self.assertEqual(audit['missing_provenance'], 1)

    def test_audit_missing_metrics_and_raw_tampering_fail(self):
        manifest, storage = self.build()
        (storage / 'medical-index.json').unlink()
        self.assertFalse(audit_index(manifest, storage)['passed'])
        self.path.write_text('changed')
        self.assertFalse(audit_index(manifest, storage)['passed'])

    def test_audit_counts_stored_duplicate_text_even_under_a_different_point_id(self):
        manifest, storage = self.build()
        with MedicalVectorStore(storage, 384, embedding_signature=FixtureEmbedding.signature) as store:
            point = next(store.iter_points(with_vectors=True))
            store.client.upsert(store.collection_name, [models.PointStruct(id=42, payload=point.payload, vector=point.vector)])
        audit = audit_index(manifest, storage)
        self.assertFalse(audit['passed'])
        self.assertEqual(audit['duplicate_chunks'], 1)

    def test_audit_rejects_changed_embedding_context(self):
        manifest, storage = self.build()
        path = storage / 'medical-embedding.json'
        signature = json.loads(path.read_text())
        signature['max_seq_length'] = 256
        path.write_text(json.dumps(signature))
        self.assertFalse(audit_index(manifest, storage)['passed'])

    def test_promotion_preserves_old_index_and_rollback(self):
        manifest, storage = self.build()
        stage = self.root / 'stage'
        stage.mkdir()
        (stage / 'qdrant').mkdir()
        (stage / 'processed').mkdir()
        pending = self.root / 'pending.json'
        pending.write_text('new manifest')
        # A failed second publication rename must restore corpus and database.
        original = Path.rename
        def rename(path, target):
            if path == stage / 'processed':
                raise OSError('synthetic promotion failure')
            return original(path, target)
        with patch.object(Path, 'rename', rename):
            with self.assertRaisesRegex(OSError, 'promotion failure'):
                publish(self.root, stage, pending, storage, FixtureEmbedding.signature)
        self.assertTrue(audit_index(manifest, storage)['passed'])

    def test_promotion_never_moves_patient_collection(self):
        manifest, storage = self.build()
        with MedicalVectorStore(storage, 384, embedding_signature=FixtureEmbedding.signature) as store:
            store.client.create_collection('ddxplus_patient_cases', vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
        with self.assertRaisesRegex(ValueError, 'non-medical'):
            publish(self.root, self.root / 'stage', self.root / 'pending', storage, FixtureEmbedding.signature)
        with MedicalVectorStore(storage, 384, embedding_signature=FixtureEmbedding.signature) as store:
            self.assertTrue(store.client.collection_exists('ddxplus_patient_cases'))

    def test_source_definition_section_is_relevant_independently_of_parent_title(self):
        hit = {'title': 'Diabetic Eye Problems', 'section': 'What is diabetes?',
               'text': 'What is diabetes? Diabetes is a disease in which your blood glucose levels are too high.',
               'license': 'fixture', 'provenance': {'fixture': True}}
        self.assertTrue(check_hits([hit], {'Diabetes'}))
        self.assertFalse(check_hits([{**hit, 'section': 'What causes eye problems?'}], {'Diabetes'}))
        self.assertFalse(check_hits([{**hit, 'section': 'Diabetes-related links'}], {'Diabetes'}))
        self.assertFalse(check_hits([{**hit, 'provenance': {}}], {'Diabetes'}))

    def test_cross_document_passages_deduplicated_without_losing_documents(self):
        shared = {'title': 'Shared definition', 'text': ' '.join('shared'+str(i) for i in range(80))}
        unique = {'title': 'Different section', 'text': ' '.join('unique'+str(i) for i in range(80))}
        first = replace(self.docs()[0], text=shared['text'], metadata={'sections': [shared]})
        second = replace(first, document_id='medlineplus:500', text=shared['text'] + unique['text'], metadata={'sections': [shared, unique]})
        with patch('rag.medical_ingestion.corpus.load_documents', return_value=[first, second]):
            documents, chunks, _ = prepare_corpus(self.root / 'unused')
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual({c.document_id for c in chunks}, {first.document_id, second.document_id})
        self.assertEqual(len({c.text for c in chunks}), 2)
        with patch('rag.medical_ingestion.corpus.load_documents', return_value=[first, replace(first, document_id='medlineplus:500')]):
            with self.assertRaisesRegex(ValueError, 'remove all indexed content'):
                prepare_corpus(self.root / 'unused')

    def test_ten_topic_checks_reject_unrelated_high_similarity(self):
        self.assertEqual(len(TOPIC_CASES), 10)
        hit = {'title': 'Unrelated', 'text': 'Unrelated text', 'license': 'fixture', 'provenance': {'fixture': True}, 'score': 0.99}
        for _, _, titles in TOPIC_CASES:
            self.assertFalse(check_hits([hit], titles))
            self.assertTrue(check_hits([{**hit, 'title': sorted(titles)[0]}], titles))


if __name__ == '__main__':
    unittest.main()

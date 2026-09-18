"""Offline synthetic ingestion fixtures; real local Qdrant, no fake runtime model."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from qdrant_client import models
from rag.medical_ingestion.models import MedicalDocument, clean_text, reusable_license, validate_chunk
from rag.medical_ingestion.chunker import chunk_document
from rag.medical_ingestion.common import download, verified_provenance
from rag.medical_ingestion.medlineplus import parse_medlineplus
from rag.medical_ingestion.pmc import parse_pmc, download_article
from rag.medical_ingestion.who import extract_text, parse_who
from rag.medical_ingestion.corpus import prepare_corpus
from rag.medical_vector_store import MedicalVectorStore
from rag.medical_retriever import MedicalKnowledgeRetriever

CC = 'https://creativecommons.org/licenses/by/4.0/'


def provenance(path=None):
    return {'download_url': 'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/', 'retrieved_at': '2026-01-01T00:00:00Z',
            'raw_path': str(path or 'synthetic-test.xml'), 'raw_sha256': hashlib.sha256(path.read_bytes() if path else b'fixture').hexdigest(),
            'license_url': CC, 'license_evidence': 'Synthetic test fixture: CC BY 4.0'}


def document():
    return MedicalDocument('pmc:PMC123', 'Asthma causes wheezing. Lung inflammation is discussed.', 'Test asthma',
                           'pmc', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC123/', 'CC-BY-4.0', provenance(),
                           doi='10.test/fixture', pmid='123', pmcid='PMC123', publication_date='2020-01', document_type='research-article')


JATS = '''<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article"><front><article-meta>
<article-id pub-id-type="pmc">123</article-id><article-id pub-id-type="pmid">456</article-id><article-id pub-id-type="doi">10.test/fixture</article-id>
<title-group><article-title>Asthma <italic>research</italic></article-title></title-group>
<pub-date><year>2020</year><month>02</month></pub-date><permissions><license xlink:href="LICENSE"><license-p>CC BY fixture evidence</license-p></license></permissions>
<abstract><p>Asthma abstract.</p></abstract></article-meta></front><body><sec><title>Results</title><p>Wheezing findings.</p>
<fig><caption><p>EXCLUDED FIGURE</p></caption></fig></sec></body><back><ref-list><p>EXCLUDED REFERENCES</p></ref-list></back></article>'''


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding='utf8')
        return path

    def test_clean_normalization(self):
        self.assertEqual(clean_text('Ａsthma\n  wheezing\t'), 'Asthma wheezing')

    def test_medlineplus_xml_selection_html_and_language(self):
        path = self.write('topics.xml', '''<health-topics><health-topic id="1" title="Asthma" url="https://medlineplus.gov/asthma.html" language="English"><full-summary>&lt;p&gt;Airway &lt;b&gt;inflammation&lt;/b&gt;.&lt;/p&gt;</full-summary><site>DO NOT INDEX</site></health-topic><health-topic id="2" title="Asthma" language="Spanish"/></health-topics>''')
        docs = list(parse_medlineplus(path, provenance(path), ['Asthma']))
        self.assertEqual(len(docs), 1)
        self.assertIn('Airway inflammation', docs[0].text)
        self.assertNotIn('DO NOT INDEX', docs[0].text)
        self.assertEqual(docs[0].license, 'NLM-public-domain-summary')

    def test_pmc_metadata_and_exclusions(self):
        path = self.write('article.xml', JATS.replace('LICENSE', CC))
        doc = parse_pmc(path, provenance(path))
        self.assertEqual((doc.pmcid, doc.pmid, doc.doi, doc.publication_date), ('PMC123', '456', '10.test/fixture', '2020-02'))
        self.assertEqual(doc.title, 'Asthma research')
        self.assertNotIn('EXCLUDED', doc.text)
        self.assertEqual(doc.provenance['raw_sha256'], provenance(path)['raw_sha256'])

    def test_license_allowlist(self):
        self.assertEqual(reusable_license(CC), 'CC-BY-4.0')
        self.assertEqual(reusable_license('https://creativecommons.org/licenses/by-sa/3.0/igo/'), 'CC-BY-SA-3.0-IGO')
        self.assertEqual(reusable_license('https://creativecommons.org/publicdomain/zero/1.0/'), 'CC0-1.0')
        for url in ('', 'https://creativecommons.org/licenses/by-nc/4.0/', 'https://creativecommons.org/licenses/by-nd/4.0/',
                    'https://creativecommons.org.evil.test/licenses/by/4.0/', 'https://creativecommons.org/publicdomain/mark/1.0/'):
            self.assertIsNone(reusable_license(url))

    def test_pmc_missing_or_restricted_license_rejected(self):
        for license in ('', 'https://creativecommons.org/licenses/by-nc/4.0/'):
            path = self.write('bad.xml', JATS.replace('LICENSE', license))
            with self.assertRaises(ValueError):
                parse_pmc(path, provenance(path))

    def test_xml_entities_rejected(self):
        path = self.write('bad.xml', '<!DOCTYPE article [<!ENTITY x SYSTEM "file:///secret">]><article>&x;</article>')
        with self.assertRaises(Exception):
            parse_pmc(path, provenance(path))

    def test_chunks_deterministic_metadata_and_overlap(self):
        doc = replace(document(), text=' '.join('word' + str(i) for i in range(12)))
        chunks = list(chunk_document(doc, 5, 2))
        self.assertEqual(chunks, list(chunk_document(doc, 5, 2)))
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].text.split()[-2:], chunks[1].text.split()[:2])
        self.assertTrue(all(c.doi == doc.doi and c.provenance == doc.provenance for c in chunks))
        self.assertEqual(len({c.chunk_id for c in chunks}), 4)
        self.assertNotEqual(chunks[0].chunk_id, list(chunk_document(doc, 6, 2))[0].chunk_id)

    def test_invalid_chunk_settings(self):
        for size, overlap in ((0, 0), (5, 5), (5, -1), (True, 0)):
            with self.assertRaises(ValueError):
                list(chunk_document(document(), size, overlap))
        with self.assertRaises(ValueError):
            list(chunk_document({'patient_id': 'x'}))

    def test_text_extraction(self):
        self.assertEqual(extract_text(self.write('who.txt', 'Health\n  text')), 'Health text')
        with self.assertRaises(ValueError):
            extract_text(self.write('empty.txt', ' '))
        with self.assertRaises(ValueError):
            extract_text(self.write('page.html', '<p>Not supported</p>'))

    def test_pdf_extraction_real_pdf(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=200)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 20 100 Td (Synthetic PDF health text) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        path = self.root / 'fixture.pdf'
        writer.write(path)
        self.assertEqual(extract_text(path), 'Synthetic PDF health text')

    def test_who_review_and_hash_required(self):
        path = self.write('who.txt', 'Synthetic test only. Creative Commons Attribution 4.0. Health text.')
        meta = provenance(path)
        selection = {'id': 'fixture', 'title': 'Synthetic WHO parser test, not a real publication',
                     'url': 'https://www.who.int/test', 'download_url': 'https://cdn.who.int/test.txt',
                     'sha256': meta['raw_sha256'], 'license_url': CC, 'license_evidence': 'Creative Commons Attribution 4.0.',
                     'reviewed_by': 'unit-test-only', 'terms_url': 'https://www.who.int/copyright', 'format': 'txt'}
        self.assertEqual(parse_who(path, meta, selection).source, 'who')
        for changes in ({'license_url': 'https://creativecommons.org/licenses/by-nc-sa/3.0/igo/'},
                        {'sha256': '0'*64}, {'license_evidence': 'not in source'}, {'reviewed_by': ''}):
            with self.assertRaises(ValueError):
                parse_who(path, meta, {**selection, **changes})

    def test_provenance_checksum(self):
        path = self.write('raw.txt', 'original')
        path.with_suffix('.txt.provenance.json').write_text(json.dumps(provenance(path)))
        self.assertEqual(verified_provenance(path)['raw_sha256'], provenance(path)['raw_sha256'])
        path.write_text('changed')
        with self.assertRaises(ValueError):
            verified_provenance(path)

    def test_download_host_and_protocol_rejected_before_network(self):
        for url in ('http://medlineplus.gov/xml/test.xml', 'https://evil.test/x'):
            with self.assertRaises(ValueError):
                download(url, self.root / 'x', {'medlineplus.gov'})

    def test_pmc_open_set_required(self):
        path = self.write('PMC123.xml', '<OAI-PMH><error>not available</error></OAI-PMH>')
        with patch('rag.medical_ingestion.pmc.download', return_value=provenance(path)):
            with self.assertRaisesRegex(ValueError, 'Open Access'):
                download_article('PMC123', self.root)

    def test_reject_patient_labels_and_provenance_extras(self):
        for changes in ({'source': 'ddxplus'}, {'text': 'PATHOLOGY: secret'},
                        {'text': 'ddxplus:train:1'}, {'provenance': {**provenance(), 'diagnosis': 'secret'}}):
            with self.assertRaises(ValueError):
                replace(document(), **changes)
        # Disease terms in medical publications are legitimate knowledge, not labels.
        self.assertIn('Asthma', document().text)

    def test_manifest_reconstruction_deterministic(self):
        path = self.write('article.xml', JATS.replace('LICENSE', CC))
        meta = provenance(path)
        path.with_suffix('.xml.provenance.json').write_text(json.dumps(meta))
        manifest = self.write('corpus.json', json.dumps({'entries': [{'source': 'pmc', 'raw_path': 'article.xml', 'sha256': meta['raw_sha256'], 'pmcid': 'PMC123'}]}))
        self.assertEqual(prepare_corpus(manifest)[2], prepare_corpus(manifest)[2])


class FakeEmbedding:
    dimension = 3
    signature = {'model_name': 'unit-test-only', 'dimension': 3}
    def embed_text(self, text):
        return [1., 0., 0.]


class MedicalStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MedicalVectorStore(self.temp.name, 3, embedding_signature=FakeEmbedding.signature)
        self.addCleanup(self.store.close)
        self.chunk = next(chunk_document(document()))
        self.retriever = MedicalKnowledgeRetriever(FakeEmbedding(), self.store)

    def insert(self):
        self.store.upsert_chunks([self.chunk], [[1, 0, 0]])

    def test_retrieval_actual_text_and_metadata(self):
        self.insert()
        hit = self.retriever.retrieve('asthma', 3)[0]
        self.assertEqual(hit['text'], self.chunk.text)
        self.assertEqual(hit['provenance'], self.chunk.provenance)
        self.assertEqual(hit['doi'], self.chunk.doi)
        self.assertIsInstance(hit['score'], float)
        self.assertNotIn('diagnosis', hit)

    def test_idempotent_and_persistent(self):
        self.insert()
        self.insert()
        self.assertEqual(self.store.count(), 1)
        self.store.close()
        with MedicalVectorStore(self.temp.name, 3, embedding_signature=FakeEmbedding.signature) as reopened:
            self.assertEqual(reopened.count(), 1)
            self.assertEqual(reopened.search([1,0,0])[0]['chunk_id'], self.chunk.chunk_id)

    def test_patient_collection_forbidden(self):
        with self.assertRaises(ValueError):
            MedicalVectorStore(self.temp.name, 3, collection_name='ddxplus_patient_cases')
        patient_store = Mock(collection_name='ddxplus_patient_cases')
        with self.assertRaises(ValueError):
            MedicalKnowledgeRetriever(FakeEmbedding(), patient_store)

    def test_payload_label_fields_rejected_atomically(self):
        for field in ('patient_id', 'split', 'PATHOLOGY', 'DIFFERENTIAL_DIAGNOSIS', 'diagnosis'):
            with self.assertRaises(ValueError):
                self.store.upsert_chunks([self.chunk, {**asdict(self.chunk), field: 'secret'}], [[1,0,0], [1,0,0]])
        self.assertEqual(self.store.count(), 0)

    def test_foreign_payload_filtered(self):
        self.insert()
        self.store.client.upsert('medical_knowledge', [models.PointStruct(id=42, vector=[1.,0.,0.], payload={'source': 'ddxplus', 'text': 'secret'})])
        self.assertEqual(len(self.retriever.retrieve('asthma')), 1)

    def test_same_path_separate_patient_collection_untouched(self):
        self.store.client.create_collection('ddxplus_patient_cases', vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
        self.store.client.upsert('ddxplus_patient_cases', [models.PointStruct(id=42, vector=[1.,0.,0.], payload={'patient_id': 'secret'})])
        self.insert()
        self.assertEqual(self.store.client.count('ddxplus_patient_cases').count, 1)
        self.assertEqual(len(self.retriever.retrieve('asthma')), 1)

    def test_signature_mismatch(self):
        self.insert()
        self.store.close()
        with self.assertRaises(ValueError):
            MedicalVectorStore(self.temp.name, 3, embedding_signature={'model_name': 'wrong'})
        with self.assertRaises(ValueError):
            MedicalVectorStore(self.temp.name, 4, embedding_signature=FakeEmbedding.signature)

    def test_invalid_vectors_and_counts(self):
        for vector in ([1, 2], [0, 0, 0], [float('nan'), 0, 1]):
            with self.assertRaises(ValueError):
                self.store.upsert_chunks([self.chunk], [vector])
        with self.assertRaises(ValueError):
            self.store.upsert_chunks([self.chunk], [])

    def test_invalid_queries_and_topk(self):
        for query, k in (('', 5), ({}, 5), ('asthma', -1), ('asthma', True)):
            with self.assertRaises(ValueError):
                self.retriever.retrieve(query, k)
        self.assertEqual(self.retriever.retrieve('asthma', 0), [])
        self.assertEqual(self.retriever.retrieve('asthma'), [])

    def test_payload_tampering_rejected(self):
        with self.assertRaises(ValueError):
            validate_chunk({**asdict(self.chunk), 'text': 'altered'})


if __name__ == '__main__':
    unittest.main()

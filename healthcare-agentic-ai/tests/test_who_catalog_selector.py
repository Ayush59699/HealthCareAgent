"""Selector software contracts only; no network or relevance/diagnosis labels."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from rag.agents.models import Failure
from rag.llm.provider import Generation, OpenAIProvider
from rag.config import OpenAIConfig
from rag.models import Evidence, PatientRepresentation, PatientRecord
from rag.who_lookup import COLLECTIONS, bounded_read
from rag.who_catalog_selector import WHOCatalogSelector, WHOSelection, PROMPT, VERSION
from scripts.evaluate_who_catalog_selector import load_saved_case
from rag.manual_patient import manual_patient_id
from tests.phase4_helpers import envelope


class WHOCatalogSelectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = {}
        self.texts = {}
        for folder, kind, prefix, header in COLLECTIONS:
            directory = self.root / folder
            directory.mkdir()
            records = []
            for i in range(4):
                topic = f'fixture-{i}'
                record = {'topic': topic, 'title': 'Fixture ' + str(i),
                          'url': 'https://www.who.int' + prefix + topic, 'file': topic + '.txt'}
                records.append(record)
                text = f"{header}\nTitle: {record['title']}\nURL: {record['url']}\nRetrieved: 2026-01-01\n\nSECRET BODY FIXTURE {kind} {i}\n"
                (directory / record['file']).write_text(text, encoding='utf8', newline='')
                self.texts[(kind, record['file'])] = text
            (directory / 'index.json').write_text(json.dumps(records), encoding='utf8')
            self.records[kind] = records
        self.selector = WHOCatalogSelector(self.root)
        self.ids = [row[0] for row in self.selector.catalog_payload()['catalog']]
        self.patient = PatientRepresentation(24, 'F',
            (Evidence('s0', 'Symptoms', 'No shortness of breath', False),),
            (Evidence('a0', 'History', 'No known chronic illness', True),),
            (Evidence('p0', 'Chief complaint', 'Fatigue and dizziness for approximately 3 weeks.', False),))

    def provider(self, ids):
        provider = Mock()
        provider.generate.return_value = Generation(parsed=WHOSelection(selected_ids=ids))
        return provider

    def test_catalog_contains_only_ids_title_type_filename(self):
        with patch('rag.who_lookup.bounded_read', wraps=bounded_read) as reads:
            selector = WHOCatalogSelector(self.root)
        self.assertEqual([c.args[0].name for c in reads.call_args_list], ['index.json', 'index.json'])
        payload = selector.catalog_payload()
        self.assertEqual(payload['catalog_columns'], ['id', 'title', 'type', 'filename'])
        self.assertEqual(len(payload['catalog']), 8)
        self.assertTrue(all(len(row) == 4 for row in payload['catalog']))
        self.assertNotIn('SECRET BODY', json.dumps(payload))

    def test_ids_and_catalog_hash_are_stable_under_index_order(self):
        path = self.root / 'who_fact_sheets/index.json'
        path.write_text(json.dumps(list(reversed(self.records['fact_sheet']))), encoding='utf8')
        rebuilt = WHOCatalogSelector(self.root)
        self.assertEqual(rebuilt.catalog_payload(), self.selector.catalog_payload())
        self.assertEqual(rebuilt.catalog_sha256, self.selector.catalog_sha256)

    def test_select_preserves_negations_without_labels_or_old_results(self):
        provider = self.provider([self.ids[0]])
        self.selector.select(provider, self.patient)
        instructions, payload, schema, validator = provider.generate.call_args.args
        self.assertEqual(instructions, PROMPT)
        self.assertIs(schema, WHOSelection)
        self.assertEqual(payload['patient_case'], self.patient.to_inference_dict())
        self.assertIn('No known chronic illness', json.dumps(payload))
        self.assertIn('No shortness of breath', json.dumps(payload))
        self.assertNotIn('old_lexical', payload)
        self.assertNotIn('SECRET BODY', json.dumps(payload))
        self.assertEqual(set(payload), {'patient_case', 'catalog', 'catalog_columns'})

    def test_label_bearing_record_rejected_before_call(self):
        provider = self.provider([])
        record = PatientRecord('ddxplus:validate:1', 'validate', 'fixture', self.patient)
        with self.assertRaises(TypeError):
            self.selector.select(provider, record)
        provider.generate.assert_not_called()

    def test_load_only_selected_files_with_full_exact_text_and_hash(self):
        with patch('rag.who_catalog_selector.bounded_read', wraps=bounded_read) as reads:
            docs = list(self.selector.iter_selected(WHOSelection(selected_ids=self.ids[:2])))
        self.assertEqual(len(reads.call_args_list), 2)
        for doc in docs:
            expected = self.texts[(doc.source.document_type, doc.source.filename)]
            self.assertEqual(doc.text, expected)
            self.assertEqual(doc.document_sha256, hashlib.sha256(expected.encode()).hexdigest())
            self.assertEqual(doc.catalog_sha256, self.selector.catalog_sha256)
            self.assertEqual(doc.selection_method, VERSION)
            self.assertEqual(doc.audit()['url'], doc.source.url)
            self.assertNotIn('text', doc.audit())

    def test_zero_selection_valid_and_opens_nothing(self):
        with patch('rag.who_catalog_selector.bounded_read') as reads:
            self.assertEqual(list(self.selector.iter_selected(WHOSelection(selected_ids=[]))), [])
        reads.assert_not_called()

    def test_three_selection_limit(self):
        self.assertEqual(len(list(self.selector.iter_selected(WHOSelection(selected_ids=self.ids[:3])))), 3)
        with self.assertRaises(ValueError):
            WHOSelection(selected_ids=self.ids[:4])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            WHOSelection(selected_ids=[self.ids[0], self.ids[0]])

    def test_unknown_id_rejected_before_any_document_read(self):
        with patch('rag.who_catalog_selector.bounded_read') as reads, self.assertRaises(ValueError):
            list(self.selector.iter_selected(WHOSelection(selected_ids=[self.ids[0], '../../secret.txt'])))
        reads.assert_not_called()

    def test_bypassed_schema_still_revalidated(self):
        forged = WHOSelection.model_construct(selected_ids=[self.ids[0]] * 2)
        provider = Mock()
        provider.generate.return_value = Generation(parsed=forged)
        with self.assertRaises(ValueError):
            self.selector.select(provider, self.patient)
        with self.assertRaises(ValueError):
            list(self.selector.iter_selected(forged))

    def test_diagnosis_or_explanation_output_rejected(self):
        for output in ({'selected_ids': [], 'diagnosis': 'not allowed'},
                       {'selected_ids': [], 'explanation': 'not allowed'}, {'selected_ids': [42]},
                       {'selected_ids': 'not a list'}):
            with self.assertRaises(ValueError):
                WHOSelection.model_validate(output)

    def test_provider_failure_not_turned_into_empty_success(self):
        provider = Mock()
        provider.generate.return_value = Generation(failure=Failure(code='api_failure', message='Fixture'))
        with patch('rag.who_catalog_selector.bounded_read') as reads:
            result = self.selector.select(provider, self.patient)
        self.assertEqual(result.failure.code, 'api_failure')
        reads.assert_not_called()

    def test_real_provider_structured_path_rejects_unknown_id_no_retry(self):
        client = Mock()
        client.responses.create.return_value = envelope(json.dumps({'selected_ids': ['invented']}))
        provider = OpenAIProvider(OpenAIConfig(max_retries=0), client=client)
        result = self.selector.select(provider, self.patient)
        self.assertIsNotNone(result.failure)
        self.assertEqual(result.failure.code, 'agent_validation_failure')
        client.responses.create.assert_called_once()

    def test_provider_context_limit_fails_without_truncation_or_cloud(self):
        client = Mock()
        provider = OpenAIProvider(OpenAIConfig(max_retries=0, max_input_bytes=10), client=client)
        result = self.selector.select(provider, self.patient)
        self.assertEqual(result.failure.code, 'context_budget')
        client.responses.create.assert_not_called()

    def test_missing_selected_file_fails_without_substitution(self):
        row = self.selector.catalog_payload()['catalog'][0]
        folder = 'who_fact_sheets' if row[2] == 'fact_sheet' else 'who_questions_answers'
        (self.root / folder / row[3]).unlink()
        with self.assertRaises(OSError):
            list(self.selector.iter_selected(WHOSelection(selected_ids=[row[0]])))

    def test_header_tampering_rejected(self):
        row = self.selector.catalog_payload()['catalog'][0]
        folder = 'who_fact_sheets' if row[2] == 'fact_sheet' else 'who_questions_answers'
        path = self.root / folder / row[3]
        path.write_text(path.read_text(encoding='utf8').replace('Title: Fixture', 'Title: Altered'), encoding='utf8')
        with self.assertRaises(ValueError):
            list(self.selector.iter_selected(WHOSelection(selected_ids=[row[0]])))

    def test_bad_index_path_rejected(self):
        self.records['fact_sheet'][0]['file'] = '../outside.txt'
        (self.root / 'who_fact_sheets/index.json').write_text(json.dumps(self.records['fact_sheet']), encoding='utf8')
        with self.assertRaises(ValueError):
            WHOCatalogSelector(self.root)

    def test_catalog_cap_fails_not_truncates(self):
        with patch('rag.who_catalog_selector.MAX_CATALOG_BYTES', 10), self.assertRaises(ValueError):
            WHOCatalogSelector(self.root)
        with patch('rag.who_catalog_selector.MAX_CATALOG_DOCUMENTS', 2), self.assertRaises(ValueError):
            WHOCatalogSelector(self.root)

    def test_oversized_selected_document_rejected(self):
        with patch('rag.who_catalog_selector.MAX_DOCUMENT_BYTES', 10), self.assertRaises(ValueError):
            list(self.selector.iter_selected(WHOSelection(selected_ids=[self.ids[0]])))

    def test_saved_case_roundtrip_does_not_read_diagnoses(self):
        path = self.root / 'input.json'
        saved = {'label_free_input': self.patient.to_inference_dict(), 'patient_id': manual_patient_id(self.patient),
                 'source_sha256': 'a' * 64, 'unused_diagnosis': 'MUST_NOT_REACH_SELECTOR'}
        path.write_text(json.dumps(saved), encoding='utf8')
        patient, origin = load_saved_case(path)
        provider = self.provider([])
        self.selector.select(provider, patient)
        self.assertEqual(patient.to_inference_dict(), self.patient.to_inference_dict())
        self.assertNotIn('MUST_NOT_REACH_SELECTOR', str(provider.generate.call_args))
        saved['label_free_input']['age'] = 25
        path.write_text(json.dumps(saved), encoding='utf8')
        with self.assertRaises(ValueError):
            load_saved_case(path)

    def test_missing_index_rejected_instead_of_partial_catalog(self):
        (self.root / 'who_questions_answers/index.json').unlink()
        with self.assertRaises(ValueError):
            WHOCatalogSelector(self.root)

    def test_catalog_id_collision_rejected(self):
        with patch('rag.who_catalog_selector.digest', return_value='a' * 64), self.assertRaises(ValueError):
            WHOCatalogSelector(self.root)

    def test_cli_calls_only_selector_and_loads_validated_ids(self):
        from scripts.evaluate_who_catalog_selector import main
        path = self.root / 'input.json'
        saved = {'label_free_input': self.patient.to_inference_dict(), 'patient_id': manual_patient_id(self.patient),
                 'source_sha256': 'a' * 64}
        path.write_text(json.dumps(saved), encoding='utf8')
        provider = self.provider([self.ids[0]])
        with patch('rag.llm.provider.OpenAIProvider') as factory, patch('rag.config.load_generation_env'), \
                patch('orchestration.phase6.Phase6Orchestrator.run') as downstream:
            factory.return_value.__enter__.return_value = provider
            status = main(['--saved-case', str(path), '--data-dir', str(self.root), '--live',
                           '--output-dir', str(self.root / 'output')])
        self.assertEqual(status, 0)
        downstream.assert_not_called()
        provider.generate.assert_called_once()
        self.assertIs(provider.generate.call_args.args[2], WHOSelection)
        report = json.loads((self.root / 'output/report.json').read_text(encoding='utf8'))
        self.assertFalse(report['pipeline_run'])
        self.assertEqual(len(report['selected_documents']), 1)
        self.assertEqual(factory.call_args.args[0].max_retries, 0)

    def test_import_does_not_load_medical_runtime(self):
        code = "import rag.who_catalog_selector, sys; assert not any(n in sys.modules for n in ('rag.amg.backend', 'orchestration.phase6.orchestrator', 'sentence_transformers', 'chromadb'))"
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True,
                       cwd=Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    unittest.main()

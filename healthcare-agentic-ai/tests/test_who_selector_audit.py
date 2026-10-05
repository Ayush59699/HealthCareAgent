"""Saved-run provenance verification tests; no live API or semantic assertions."""
import json
import unittest
from unittest.mock import Mock
from rag.llm.provider import Generation
from rag.who_catalog_selector import WHOSelection
from rag.who_selector_relevance import ContentRelevance
from scripts.evaluate_who_selector_check4 import CASES, evaluate
from scripts.verify_who_selector_check4 import verify
from tests import test_who_catalog_selector as fixtures


class WHOSelectorAuditTests(unittest.TestCase):
    def setUp(self):
        fixtures.WHOCatalogSelectorTests.setUp(self)
        self.output = self.root / 'audit-run'
        provider = Mock()
        observation = dict(accepted=True, api_success=True, client_context_truncated=False,
                           request_seconds=0.1, request_bytes=500)
        rejection = ContentRelevance(verdict='not_relevant', reason='generic_overlap_only',
                                     positive_fact_quotes=[], content_quotes=[])
        provider.generate.side_effect = [g for _ in CASES for g in (
            Generation(parsed=WHOSelection(selected_ids=self.ids[:1]), attempts=1, telemetry=[observation]),
            Generation(parsed=rejection, attempts=1, telemetry=[observation]))]
        evaluate(self.output, self.selector, provider)
        self.case_dir = self.output / CASES[0][0]

    def test_valid_complete_run(self):
        result = verify(self.output, self.root)
        self.assertTrue(result['verified'])
        self.assertEqual(result['source_events'], 8)
        self.assertEqual(result['cloud_requests'], 16)
        self.assertEqual(result['final_selected_documents'], 0)

    def test_source_tampering(self):
        path = next(self.case_dir.glob('*.txt'))
        path.write_bytes(path.read_bytes() + b'tampered')
        with self.assertRaises(ValueError):
            verify(self.output, self.root)

    def test_final_selection_tampering(self):
        path = self.case_dir / 'report.json'
        report = json.loads(path.read_text())
        report['selected_ids'] = self.ids[:1]
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            verify(self.output, self.root)

    def test_patient_input_tampering(self):
        path = self.case_dir / 'input.json'
        record = json.loads(path.read_text())
        record['patient_facts']['symptoms'] = ['altered']
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            verify(self.output, self.root)

    def test_missing_candidate_audit(self):
        path = self.case_dir / 'report.json'
        report = json.loads(path.read_text())
        report['candidates'] = []
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            verify(self.output, self.root)


if __name__ == '__main__':
    unittest.main()

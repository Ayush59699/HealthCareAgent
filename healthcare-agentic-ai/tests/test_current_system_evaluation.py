"""Offline tests of CHECK2 bookkeeping only; never call live models/stores."""
import unittest
from rag.models import PatientRepresentation, PatientRecord, Evidence, EvaluationLabels
from scripts.evaluate_current_system import select_cases, citation_usage


def record(number, question='same', labels=None):
    patient = PatientRepresentation(30, 'F', (Evidence('E', question, 'Yes', False),), ())
    return PatientRecord(f'ddxplus:validate:{number}', 'validate', 'fixture', patient, labels)


class CurrentSystemEvaluationTests(unittest.TestCase):
    def test_anchor_diversity_and_stable_tie_break(self):
        records = [record(1), record(2), record(3, 'different'), record(4, 'another')]
        self.assertEqual([r.patient_id for r in select_cases(records, 3)],
                         ['ddxplus:validate:1', 'ddxplus:validate:3', 'ddxplus:validate:4'])
        self.assertEqual(select_cases(records, 3), select_cases(records, 3))

    def test_rejects_labels(self):
        with self.assertRaises(ValueError):
            select_cases([record(1, labels=EvaluationLabels('forbidden'))], 1)

    def test_insufficient_pool(self):
        with self.assertRaises(ValueError):
            select_cases([record(1)], 10)

    def test_unique_claim_citations_only(self):
        report = {'claims': [{'medical_refs': ['medical:a', 'medical:a', 'unknown']},
                             {'medical_refs': ['medical:a']}],
                  'medical_knowledge_evidence': ['medical:b']}
        self.assertEqual(citation_usage(report, ['medical:a', 'medical:b']),
                         (['medical:a'], ['medical:b']))

    def test_no_citations_not_no_retrieval(self):
        self.assertEqual(citation_usage({'claims': []}, ['medical:a']), ([], ['medical:a']))


if __name__ == '__main__':
    unittest.main()

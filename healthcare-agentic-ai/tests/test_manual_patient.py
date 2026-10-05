"""Manual-vignette input contracts; no cloud calls or benchmark label inputs."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from rag.manual_patient import read_manual_patient, manual_patient_id, MAX_CASE_BYTES
from rag.agents.grounding import patient_input
from rag.who_amg_experiment import create_experimental_workflow
from application.decision import build_final_decision
from tests.amg_helpers import configure_medical, configure_provider
from tests.safety_helpers import setup
from tests.test_who_amg_evidence import lookup_stub

CASE = '''Patient:
Age: 24
Sex: Female

Chief complaint:
Fatigue and dizziness for approximately 3 weeks.

Symptoms:
- Persistent fatigue
- Dizziness when standing
- Occasional headaches
- Reduced exercise tolerance
- No fever
- No chest pain
- No shortness of breath

History:
- No known chronic illness
- No current medications

Available information:
- No laboratory results available.
'''


class ManualPatientTests(unittest.TestCase):
    def parse(self, text=CASE):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.txt'
            path.write_text(text, encoding='utf8', newline='')
            return read_manual_patient(path)

    def test_literal_values_negatives_and_duration_preserved(self):
        patient, audit = self.parse()
        self.assertEqual((patient.age, patient.sex), (24, 'F'))
        self.assertEqual(len(patient.symptoms), 7)
        self.assertEqual(patient.symptoms[4].value, 'No fever')
        self.assertEqual(patient.symptoms[6].value, 'No shortness of breath')
        self.assertEqual(patient.initial_evidence[0].value, 'Fatigue and dizziness for approximately 3 weeks.')
        self.assertEqual(patient.initial_evidence[1].value, 'No laboratory results available.')
        for field in (patient.symptoms, patient.antecedents, patient.initial_evidence):
            for item in field:
                self.assertIn(item.value, CASE)
        self.assertEqual(audit['source_sha256'], hashlib.sha256(CASE.encode()).hexdigest())

    def test_content_bound_id_not_benchmark_identity(self):
        patient, audit = self.parse()
        identity = manual_patient_id(patient)
        self.assertTrue(identity.startswith('manual:'))
        self.assertEqual(patient_input(patient, identity)['patient_id'], identity)
        self.assertEqual(identity, audit['patient_id'])
        with self.assertRaises(ValueError):
            patient_input(replace(patient, age=25), identity)

    def test_random_manual_id_rejected(self):
        patient, _ = self.parse()
        for identity in ('manual:check3', 'manual:' + '0' * 64, 'ddxplus:validate:0'):
            with self.assertRaises(ValueError):
                patient_input(patient, identity)

    def test_label_bearing_record_still_rejected(self):
        _, case, *_ = setup()
        with self.assertRaises(TypeError):
            patient_input(case, manual_patient_id(case.patient))

    def test_unknown_or_duplicate_section_rejected(self):
        for text in (CASE + '\nDiagnosis:\nExample\n', CASE.replace('History:', 'Symptoms:'),
                     CASE.replace('Patient:', 'Unknown:')):
            with self.assertRaises(ValueError):
                self.parse(text)

    def test_missing_or_malformed_fields_rejected(self):
        for text in (CASE.replace('Age: 24', 'Age: unknown'), CASE.replace('Age: 24', 'Age: 200'),
                     CASE.replace('Sex: Female', 'Sex: unknown'), CASE.replace('- No fever', 'No fever'),
                     CASE.replace('- No laboratory results available.', '')):
            with self.assertRaises(ValueError):
                self.parse(text)

    def test_size_limit(self):
        with self.assertRaises(ValueError):
            self.parse('x' * (MAX_CASE_BYTES + 1))

    def test_line_endings_preserve_patient_identity_but_not_file_hash(self):
        a, source_a = self.parse()
        b, source_b = self.parse(CASE.replace('\n', '\r\n'))
        self.assertEqual(manual_patient_id(a), manual_patient_id(b))
        self.assertNotEqual(source_a['source_sha256'], source_b['source_sha256'])

    def test_manual_id_survives_full_existing_workflow_and_final_revalidation(self):
        _, case, llm, patients, medical = setup()
        configure_medical(medical)
        configure_provider(llm)
        workflow = create_experimental_workflow(llm, patients, medical, who=lookup_stub(()))
        identity = manual_patient_id(case.patient)
        state = workflow.run(case.patient, identity)
        self.assertEqual(state.patient_id, identity)
        self.assertEqual(state.status, 'final')
        self.assertEqual(build_final_decision(state).patient_id, identity)


if __name__ == '__main__':
    unittest.main()

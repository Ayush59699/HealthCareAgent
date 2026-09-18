import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from rag import DDXPlusParser, DDXPlusError
from rag.config import resolve_data_dir
from scripts.prepare_ddxplus import main


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        evidence = {
            "E_1": {"question_en": "Do you have a fever?", "data_type": "B", "is_antecedent": False},
            "E_2": {"question_en": "History of smoking?", "data_type": "B", "is_antecedent": True},
            "E_130": {"question_en": "What color is the rash?", "data_type": "C", "is_antecedent": False,
                      "possible-values": ["V_86"], "value_meaning": {"V_86": {"en": "dark"}}},
            "E_3": {"question_en": "Pain intensity?", "data_type": "C", "is_antecedent": False,
                    "possible-values": list(range(11)), "value_meaning": {}},
            "E_4": {"question_en": "Pain location?", "data_type": "M", "is_antecedent": False,
                    "possible-values": ["V_1", "V_2"], "value_meaning": {"V_1": {"en": "head"}, "V_2": {"en": "chest"}}},
        }
        (self.root / "release_evidences.json").write_text(json.dumps(evidence), encoding="utf8")
        (self.root / "release_conditions.json").write_text(json.dumps({"label": {"cond-name-eng": "Evaluation sentinel"}}), encoding="utf8")
        self.parser = DDXPlusParser(self.root)
        self.row = {"AGE": "18", "SEX": "M", "EVIDENCES": "['E_1', 'E_2', 'E_130_@_V_86', 'E_3_@_4']",
                    "INITIAL_EVIDENCE": "E_1", "PATHOLOGY": "label", "DIFFERENTIAL_DIAGNOSIS": "[['label', 1.0]]"}

    def archive(self, rows, split="train", member=None):
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=list(self.row))
        writer.writeheader()
        writer.writerows(rows)
        with zipfile.ZipFile(self.root / f"release_{split}_patients.zip", "w") as archive:
            archive.writestr(member or f"release_{split}_patients", text.getvalue())

    def test_decoding_and_partition(self):
        patient = self.parser.parse_patient(self.row)
        self.assertEqual(patient.age, 18)
        self.assertEqual(patient.symptoms[1].text, "What color is the rash? = dark")
        self.assertEqual(patient.symptoms[2].value, "4")
        self.assertEqual(patient.antecedents[0].value, "Yes")
        self.assertEqual(patient.initial_evidence[0], patient.symptoms[0])
        self.assertNotRegex(json.dumps(patient.to_inference_dict()), r"\b[EV]_\d+")

    def test_labels_separate_and_opt_in(self):
        self.archive([self.row])
        default = next(self.parser.iter_patients())
        record = next(self.parser.iter_patients(include_labels=True))
        self.assertIsNone(default.labels)
        self.assertEqual(record.labels.ground_truth_pathology, "Evaluation sentinel")
        self.assertEqual(record.labels.differential_diagnosis[0].probability, 1)
        self.assertEqual(default.to_inference_dict(), record.to_inference_dict())
        for text in (record.to_text(), json.dumps(record.to_inference_dict()), repr(record)):
            self.assertNotIn("Evaluation sentinel", text)
            self.assertNotIn("PATHOLOGY", text)
            self.assertNotIn("DIFFERENTIAL_DIAGNOSIS", text)

    def test_inference_never_parses_labels(self):
        original = self.parser.parse_patient(self.row)
        self.row.update(PATHOLOGY="secret", DIFFERENTIAL_DIAGNOSIS="not a list")
        self.assertEqual(original, self.parser.parse_patient(self.row))

    def test_multivalue_initial_and_deduplication(self):
        self.row.update(EVIDENCES="['E_4_@_V_1','E_4_@_V_2','E_4_@_V_1']", INITIAL_EVIDENCE="E_4")
        patient = self.parser.parse_patient(self.row)
        self.assertEqual(len(patient.symptoms), 2)
        self.assertEqual(patient.initial_evidence, patient.symptoms)

    def test_missing_categorical_initial_does_not_invent_answer(self):
        self.row.update(EVIDENCES="[]", INITIAL_EVIDENCE="E_130")
        self.assertIsNone(self.parser.parse_patient(self.row).initial_evidence[0].value)

    def test_missing_demographics_and_empty_findings(self):
        self.row.update(AGE="", SEX="", EVIDENCES="[]", INITIAL_EVIDENCE="")
        patient = self.parser.parse_patient(self.row)
        self.assertIsNone(patient.age)
        self.assertIsNone(patient.sex)
        self.assertIn("not assumed absent", patient.to_text())

    def test_invalid_evidence_fails_closed(self):
        for token in ("E_999", "E_130_@_V_999", "E_3_@_11", "E_130", "E_1_@_maybe", "E_1_@_1_@_2", 3):
            with self.subTest(token=token), self.assertRaises(DDXPlusError):
                self.parser.decode_evidence(token)

    def test_negative_binary(self):
        self.assertEqual(self.parser.decode_evidence("E_1_@_0").value, "No")

    def test_conflicts_rejected(self):
        for updates in ({"EVIDENCES": "['E_1','E_1_@_0']"},
                        {"EVIDENCES": "['E_1_@_0']"},
                        {"INITIAL_EVIDENCE": "E_3_@_5"}):
            with self.subTest(updates=updates), self.assertRaises(DDXPlusError):
                self.parser.parse_patient({**self.row, **updates})

    def test_bad_lists_and_demographics(self):
        for field, value in (("EVIDENCES", "__import__('os').getcwd()"), ("EVIDENCES", "{}"),
                             ("EVIDENCES", ""), ("AGE", "18.5"), ("AGE", -1), ("AGE", True), ("SEX", "secret")):
            with self.subTest(field=field, value=value), self.assertRaises(DDXPlusError):
                self.parser.parse_patient({**self.row, field: value})

    def test_bad_labels(self):
        for value in ("[['label', 2]]", "[['label', 'nan']]", "[['label', True]]", "[['unknown', 1]]", "['label']"):
            with self.subTest(value=value), self.assertRaises(DDXPlusError):
                self.parser.parse_labels({**self.row, "DIFFERENTIAL_DIAGNOSIS": value})

    def test_stream_limit_does_not_parse_next_row(self):
        self.archive([self.row, {**self.row, "EVIDENCES": "bad"}])
        self.assertEqual(len(list(self.parser.iter_patients(limit=1))), 1)
        self.assertEqual(list(self.parser.iter_patients(limit=0)), [])
        with self.assertRaisesRegex(DDXPlusError, "train row 2"):
            list(self.parser.iter_patients())

    def test_split_ids_and_csv_suffix(self):
        ids = []
        for split in ("train", "validate", "test"):
            self.archive([self.row], split, f"release_{split}_patients.csv")
            ids.append(next(self.parser.iter_patients(split)).patient_id)
        self.assertEqual(len(set(ids)), 3)

    def test_bad_archive_and_split(self):
        for split, limit in (("invalid", None), ("train", -1)):
            with self.assertRaises(DDXPlusError):
                list(self.parser.iter_patients(split, limit=limit))
        self.archive([self.row], member="unrelated")
        with self.assertRaises(DDXPlusError):
            list(self.parser.iter_patients())

    def test_missing_metadata(self):
        (self.root / "release_conditions.json").unlink()
        with self.assertRaisesRegex(DDXPlusError, "Cannot load"):
            DDXPlusParser(self.root)

    def test_cli_separate_exports_and_no_overwrite(self):
        self.archive([self.row])
        out, labels = self.root / "features.jsonl", self.root / "labels.jsonl"
        args = ["--data-dir", str(self.root), "--output", str(out), "--labels-output", str(labels)]
        self.assertEqual(main(args), 0)
        self.assertNotIn("Evaluation sentinel", out.read_text())
        self.assertIn("Evaluation sentinel", labels.read_text())
        before = out.read_text()
        self.assertEqual(main(args), 1)
        self.assertEqual(out.read_text(), before)

    def test_cli_removes_incomplete_exports(self):
        self.archive([{**self.row, "EVIDENCES": "bad"}])
        out = self.root / "features.jsonl"
        self.assertEqual(main(["--data-dir", str(self.root), "--output", str(out)]), 1)
        self.assertFalse(out.exists())


class LocalDatasetTests(unittest.TestCase):
    @unittest.skipUnless((resolve_data_dir() / "release_evidences.json").exists(), "Local DDXPlus not available")
    def test_real_data_all_splits(self):
        parser = DDXPlusParser()
        self.assertEqual(parser.decode_evidence("E_130_@_V_86").text, "What color is the rash? = dark")
        for split in ("train", "validate", "test"):
            records = list(parser.iter_patients(split, limit=100, include_labels=True))
            self.assertEqual(len(records), 100)
            for record in records:
                self.assertIsNotNone(record.labels.ground_truth_pathology)
                self.assertNotRegex(json.dumps(record.to_inference_dict()), r"\b[EV]_\d+")


if __name__ == "__main__":
    unittest.main()

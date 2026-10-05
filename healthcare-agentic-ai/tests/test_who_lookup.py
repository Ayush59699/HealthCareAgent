"""Local fixtures only: no requests, models or existing corpus modification."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rag.who_lookup import COLLECTIONS, WHOLookup


class WHOLookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = {"fact_sheet": [], "question_answer": []}
        self.texts = {}
        for folder, _, _, _ in COLLECTIONS:
            (self.root / folder).mkdir()
        self.add("asthma", "Asthma")
        self.add("diabetes", "Diabetes")
        self.add("anaemia", "Anaemia")
        self.add("adolescent-mental-health", "Mental health of adolescents")
        self.add("asthma-care", "Asthma: care", kind="question_answer")
        self.add("addictive-behaviours-gaming-disorder", "Addictive behaviours: Gaming disorder", kind="question_answer")
        self.save_indexes()

    def add(self, topic, title, *, kind="fact_sheet", body=None):
        folder, _, prefix, header = next(item for item in COLLECTIONS if item[1] == kind)
        url = "https://www.who.int" + prefix + topic
        record = {"topic": topic, "title": title, "url": url, "file": topic + ".txt"}
        self.records[kind].append(record)
        body = body or "Key facts\n=========\n\n- Original WHO wording.\n\nOverview\n========\n\nComplete text, not a summary.\n"
        text = f"{header}\nTitle: {title}\nURL: {url}\nRetrieved: 2026-01-01T00:00:00+00:00\n\n========================================\n{body}"
        path = self.root / folder / record["file"]
        path.write_text(text, encoding="utf-8", newline="")
        self.texts[(kind, topic)] = (path, text)
        return record

    def save_indexes(self):
        for folder, kind, _, _ in COLLECTIONS:
            (self.root / folder / "index.json").write_text(json.dumps(self.records[kind]), encoding="utf-8")

    def lookup(self, query, **kwargs):
        return WHOLookup(self.root).lookup(query, **kwargs)

    def test_exact_title(self):
        result = self.lookup("Asthma")
        self.assertEqual(result.status, "MATCH")
        self.assertEqual(result.documents[0].match_type, "exact")
        self.assertEqual(result.documents[0].source.topic, "asthma")
        self.assertEqual(result.documents[0].matched_terms, ("asthma",))

    def test_exact_slug(self):
        result = self.lookup("adolescent-mental-health")
        self.assertEqual(result.documents[0].source.title, "Mental health of adolescents")
        self.assertEqual(result.documents[0].match_type, "exact")

    def test_case_punctuation_accents(self):
        self.add("guillain-barré-syndrome", "Guillain–Barré syndrome")
        self.save_indexes()
        result = self.lookup("GUILLAIN BARRE SYNDROME!")
        self.assertEqual(result.documents[0].source.topic, "guillain-barré-syndrome")
        self.assertEqual(result.documents[0].match_type, "exact")

    def test_topic_in_long_query(self):
        result = self.lookup("Please tell me about asthma prevention and treatment")
        self.assertEqual(result.documents[0].source.topic, "asthma")
        self.assertEqual(result.documents[0].match_type, "topic_phrase")

    def test_partial_multiword_match(self):
        result = self.lookup("gaming disorder")
        self.assertEqual(result.documents[0].source.topic, "addictive-behaviours-gaming-disorder")
        self.assertEqual(result.documents[0].match_type, "partial")
        self.assertIn("gaming", result.documents[0].matched_terms)

    def test_distinctive_single_keyword(self):
        result = self.lookup("gaming")
        self.assertEqual(result.status, "MATCH")
        self.assertEqual(len(result.documents), 1)

    def test_multiple_topics_and_both_collections(self):
        result = self.lookup("asthma and diabetes")
        topics = {doc.source.topic for doc in result.documents}
        self.assertTrue({"asthma", "diabetes", "asthma-care"}.issubset(topics))
        self.assertEqual({d.source.document_type for d in result.documents}, {"fact_sheet", "question_answer"})

    def test_no_match(self):
        result = self.lookup("quantum spaceship plumbing")
        self.assertEqual(result.status, "NO_MATCH")
        self.assertEqual(result.documents, ())
        self.assertIn("No reliable", result.reason)

    def test_empty_generic_and_numeric_queries(self):
        for query in ("", "  ", "health", "symptoms treatment care", "children", "2026", "what is the disease?"):
            with self.subTest(query=query):
                self.assertEqual(self.lookup(query).status, "NO_MATCH")

    def test_no_substring_fuzzy_or_synonym_guess(self):
        for query in ("asth", "asthmatic", "anemia", "xasthmax", "breathlessness and wheezing"):
            with self.subTest(query=query):
                self.assertEqual(self.lookup(query).status, "NO_MATCH")

    def test_ambiguous_single_keyword_abstains(self):
        for topic in ("alpha-renal", "beta-renal", "gamma-renal"):
            self.add(topic, topic.replace("-", " "))
        self.save_indexes()
        self.assertEqual(self.lookup("renal").status, "NO_MATCH")

    def test_two_informative_words_allow_partial(self):
        self.add("alpha-beta-gamma", "Alpha beta gamma")
        self.save_indexes()
        result = self.lookup("beta alpha")
        self.assertEqual(result.documents[0].match_type, "partial")
        self.assertEqual(result.documents[0].matched_terms, ("alpha", "beta"))

    def test_standalone_topic_supports_related_qa(self):
        for n in range(3):
            self.add(f"asthma-topic-{n}", f"Asthma topic {n}", kind="question_answer")
        self.save_indexes()
        result = self.lookup("asthma")
        self.assertEqual(len(result.documents), 5)
        self.assertEqual(result.documents[0].source.topic, "asthma")

    def test_full_text_and_provenance_unchanged(self):
        result = self.lookup("asthma", max_results=1)
        doc = result.documents[0]
        self.assertEqual(doc.text, self.texts[("fact_sheet", "asthma")][1])
        self.assertEqual(doc.source.title, "Asthma")
        self.assertEqual(doc.source.url, "https://www.who.int/news-room/fact-sheets/detail/asthma")
        self.assertEqual(doc.source.filename, "asthma.txt")
        self.assertEqual(doc.source.document_type, "fact_sheet")
        json.dumps(asdict(result))

    def test_long_text_not_truncated(self):
        body = "Original paragraph.\n" * 10000 + "Final sentence."
        self.add("long-topic", "Long topic", body=body)
        self.save_indexes()
        result = self.lookup("long-topic")
        self.assertTrue(result.documents[0].text.endswith(body))

    def test_indexes_only_until_lookup_and_only_selected_text_loaded(self):
        opened = []
        original = Path.open
        def recording(path, *args, **kwargs):
            opened.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", recording):
            lookup = WHOLookup(self.root)
            self.assertEqual(len(opened), 2)
            before = list(opened)
            lookup.search("asthma")
            lookup.search("diabetes")
            self.assertEqual(opened, before)
            result = lookup.lookup("diabetes")
        self.assertEqual(opened[-1].name, "diabetes.txt")
        self.assertEqual(len(opened), 3)
        self.assertEqual(len(result.documents), 1)

    def test_search_does_not_require_document_files(self):
        self.texts[("fact_sheet", "diabetes")][0].unlink()
        lookup = WHOLookup(self.root)
        result = lookup.search("diabetes")
        self.assertEqual(result.status, "MATCH")
        self.assertIsNone(result.documents[0].text)
        loaded = lookup.lookup("diabetes")
        self.assertEqual(loaded.status, "NO_MATCH")
        self.assertTrue(loaded.errors)

    def test_missing_best_document_does_not_stop_other_matches(self):
        self.texts[("fact_sheet", "asthma")][0].unlink()
        result = self.lookup("asthma", max_results=1)
        self.assertEqual(result.documents[0].source.topic, "asthma-care")
        self.assertTrue(result.errors)

    def test_mismatched_header_fails_closed(self):
        path, text = self.texts[("fact_sheet", "diabetes")]
        path.write_text(text.replace("Title: Diabetes", "Title: Wrong topic"), encoding="utf-8")
        result = self.lookup("diabetes")
        self.assertEqual(result.status, "NO_MATCH")
        self.assertIn("metadata/body", result.errors[0])

    def test_bad_encoding_and_size_fail_explicitly(self):
        path, text = self.texts[("fact_sheet", "diabetes")]
        path.write_bytes(b'\xff\xfeinvalid')
        self.assertEqual(self.lookup("diabetes").status, "NO_MATCH")
        path.write_text(text, encoding="utf-8")
        with patch("rag.who_lookup.MAX_DOCUMENT_BYTES", 16):
            result = self.lookup("diabetes")
        self.assertEqual(result.status, "NO_MATCH")
        self.assertIn("not truncated", result.errors[0])

    def test_empty_body_fails(self):
        path, text = self.texts[("fact_sheet", "diabetes")]
        path.write_text("\n".join(text.splitlines()[:4]) + "\n\n====\n", encoding="utf-8")
        self.assertEqual(self.lookup("diabetes").status, "NO_MATCH")

    def test_missing_and_malformed_indexes(self):
        (self.root / "who_fact_sheets/index.json").write_text("{broken", encoding="utf-8")
        result = self.lookup("gaming")
        self.assertEqual(result.status, "MATCH")
        self.assertTrue(result.errors)
        (self.root / "who_questions_answers/index.json").unlink()
        result = self.lookup("asthma")
        self.assertEqual(result.status, "NO_MATCH")
        self.assertEqual(len(result.errors), 2)

    def test_invalid_record_and_duplicate_handling(self):
        self.records["fact_sheet"].extend([dict(self.records["fact_sheet"][0]), {"topic": "broken"}, "bad"])
        self.save_indexes()
        lookup = WHOLookup(self.root)
        self.assertEqual(lookup.document_count, 6)
        self.assertEqual(len(lookup.index_errors), 2)
        self.assertEqual(len(lookup.lookup("asthma").documents), 2)

    def test_untrusted_filename_and_url_rejected(self):
        record = self.records["fact_sheet"][0]
        for field, value in (("file", "../outside.txt"), ("file", "..\\outside.txt"),
                             ("file", "C:secret.txt"), ("url", "https://evil.test/asthma"),
                             ("url", "https://www.who.int/news-room/questions-and-answers/item/asthma")):
            original = record[field]
            record[field] = value
            self.save_indexes()
            lookup = WHOLookup(self.root)
            self.assertTrue(lookup.index_errors)
            self.assertFalse(any(d.source.document_type == "fact_sheet" for d in lookup.search("asthma").documents))
            record[field] = original

    def test_collection_redirect_after_loading_is_rejected(self):
        lookup = WHOLookup(self.root)
        directory = self.root / "who_fact_sheets"
        outside = self.root.parent / "outside-collection"
        original = Path.resolve
        def relocated(path, *args, **kwargs):
            if path == directory:
                return outside
            if path.parent == directory:
                return outside / path.name
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", relocated):
            result = lookup.lookup("diabetes")
        self.assertEqual(result.status, "NO_MATCH")
        self.assertIn("outside its collection/data directory", result.errors[0])

    def test_deterministic_order_independent_of_index_order(self):
        before = asdict(self.lookup("asthma diabetes gaming"))
        for records in self.records.values():
            records.reverse()
        self.save_indexes()
        self.assertEqual(before, asdict(self.lookup("asthma diabetes gaming")))

    def test_result_bound_and_input_validation(self):
        self.assertEqual(len(self.lookup("asthma diabetes gaming", max_results=2).documents), 2)
        for value in (0, -1, 21, True, 1.5):
            with self.assertRaises(ValueError):
                self.lookup("asthma", max_results=value)
        with self.assertRaises(ValueError):
            self.lookup("x" * 4097)
        with self.assertRaises(TypeError):
            self.lookup(None)

    def test_cli_without_site_packages_and_no_match_exit(self):
        project = Path(__file__).resolve().parents[1]
        for query, code, status in (("diabetes", 0, "MATCH"), ("xyzzy", 1, "NO_MATCH")):
            completed = subprocess.run([sys.executable, "-S", "-m", "rag.who_lookup", query,
                                        "--data-dir", str(self.root)], cwd=project, capture_output=True, text=True)
            self.assertEqual(completed.returncode, code, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], status)


if __name__ == "__main__":
    unittest.main()

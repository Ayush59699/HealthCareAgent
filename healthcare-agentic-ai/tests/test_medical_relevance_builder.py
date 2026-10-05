"""Builder reads a disposable snapshot, not the source Qdrant store."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from evaluation.reviewed_retrieval import Benchmark
from rag.medical_vector_store import MedicalVectorStore
from scripts import build_medical_relevance_benchmark as builder
from tests.test_medical_relevance_benchmark import fixture
from tests.test_patient_rag import record


class MedicalRelevanceBuilderTests(unittest.TestCase):
    def test_cli_snapshot_no_network_no_source_changes_and_no_label_serialization(self):
        _, report, chunks, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original-index'
            signature = {'dimension': 2, 'test_fixture': True}
            with MedicalVectorStore(original, 2, embedding_signature=signature) as store:
                store.upsert_chunks(chunks, [[1., 0.]] * len(chunks))
            manifest = (original / 'medical-embedding.json').read_bytes()
            report['embedding_signature'] = signature
            report['manifest_sha256'] = hashlib.sha256(manifest).hexdigest()
            report_path = root / 'retrieval.json'
            report_path.write_text(json.dumps(report), encoding='utf8')
            output = root / 'benchmark.json'

            def file_hashes():
                return {str(p.relative_to(original)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in original.rglob('*') if p.is_file()}

            before = file_hashes()
            parser = Mock()
            # Deliberate held-out label sentinel in fake parser return. The
            # builder must serialize only label-free features, never this label.
            parser.iter_patients.return_value = [record(2, split='validate')]
            with patch.object(builder, 'DDXPlusParser', return_value=parser), redirect_stdout(io.StringIO()), \
                    patch('socket.socket', side_effect=AssertionError('No network allowed')):
                self.assertEqual(builder.main(['--retrieval-report', str(report_path),
                                               '--storage-path', str(original), '--output', str(output)]), 0)
            parser.iter_patients.assert_called_once_with('validate', limit=2, include_labels=False)
            self.assertEqual(file_hashes(), before)
            text = output.read_text(encoding='utf8')
            self.assertNotIn('SECRET_DIAGNOSIS', text)
            benchmark = Benchmark.model_validate_json(text)
            self.assertTrue(all(not c.reviewed for c in benchmark.candidates))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                builder.main(['--retrieval-report', str(report_path), '--output', str(output)])
            self.assertEqual(output.read_text(encoding='utf8'), text)

    def test_missing_index_fails_without_creating_source_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing'
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                builder.main(['--retrieval-report', 'unused.json', '--storage-path', str(path),
                              '--output', str(Path(directory) / 'new.json')])
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()

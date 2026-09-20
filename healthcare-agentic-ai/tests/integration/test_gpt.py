"""One opt-in, sequential real validation case; ordinary discovery skips it."""
import os
from pathlib import Path
import tempfile
import unittest

from scripts.run_phase4 import main


@unittest.skipUnless(os.getenv('RUN_GPT_INTEGRATION') == '1', 'Real cloud E2E integration is opt-in')
class GPTIntegrationTests(unittest.TestCase):
    def test_real_validation_patient_dual_rag_three_agents(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(main(['--queries', '1', '--top-k', '1',
                                   '--output-dir', str(Path(directory) / 'run')]), 0)

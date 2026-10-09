import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class RunnerTests(unittest.TestCase):
    def test_existing_evidence_is_not_overwritten_before_python_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scripts').mkdir()
            source = Path(__file__).parents[1]/'scripts/run_nonexpert.sh'
            script = root/'scripts/run_nonexpert.sh'
            shutil.copyfile(source, script)
            results = root/'results/nonexpert-v1'
            results.mkdir(parents=True)
            receipt = results/'estimate-A.json'
            receipt.write_text('original evidence')
            run = subprocess.run(['sh',str(script)], env={**os.environ,'PYTHON':'/nonexistent/python'},
                                 capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertEqual(receipt.read_text(),'original evidence')

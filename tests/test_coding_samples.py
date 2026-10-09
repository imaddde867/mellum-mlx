"""Missing, duplicated or filtered failures must not inflate a benchmark score."""
import sys
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from evaluate_coding import check_samples, check_binding


class SampleIntegrityTests(unittest.TestCase):
    def test_empty_solution_is_kept(self):
        check_samples([{'task_id': 'HumanEval/0', 'prompt': 'def f():', 'entry_point': 'f'}],
                      [{'task_id': 'HumanEval/0', 'solution': ''}], smoke=True)

    def test_reject_missing_duplicate_or_nonstring_samples(self):
        tasks = [{'task_id': 'HumanEval/0', 'prompt': 'def f():', 'entry_point': 'f'}]
        sample = {'task_id': 'HumanEval/0', 'solution': 'def f(): pass'}
        for samples in [[], [sample, sample], [dict(sample, solution=None)],
                        [dict(sample, task_id='HumanEval/1')]]:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                check_samples(tasks, samples, smoke=True)

    def test_binding_rejects_modified_samples_and_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            samples = Path(directory) / 'samples.jsonl'
            raw = samples.with_suffix('.raw.jsonl')
            samples.write_text(json.dumps({'task_id': 'HumanEval/0', 'solution': 'pass'}) + '\n')
            raw.write_text(json.dumps({'task_id': 'HumanEval/0', 'message': {'content': 'pass'}}) + '\n')
            receipt = {'samples_sha256': hashlib.sha256(samples.read_bytes()).hexdigest(),
                       'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}
            check_binding(samples, receipt)
            with self.assertRaises(ValueError):
                check_binding(samples, {})
            check_binding(samples, {}, allow_legacy=True)
            samples.write_text(samples.read_text() + ' ')
            with self.assertRaises(ValueError):
                check_binding(samples, receipt)
            samples.write_text(json.dumps({'task_id': 'HumanEval/0', 'solution': 'pass'}) + '\n')
            raw.write_text(json.dumps({'task_id': 'HumanEval/0', 'message': {'content': 'other'}}) + '\n')
            with self.assertRaises(ValueError):
                check_binding(samples, receipt)
            with self.assertRaises(ValueError):
                check_binding(samples, {}, allow_legacy=True)

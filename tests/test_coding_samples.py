"""Missing, duplicated or filtered failures must not inflate a benchmark score."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from evaluate_coding import check_samples


class SampleIntegrityTests(unittest.TestCase):
    def test_empty_solution_is_kept(self):
        check_samples([{'task_id': 'HumanEval/0', 'prompt': 'def f():', 'entry_point': 'f'}],
                      [{'task_id': 'HumanEval/0', 'solution': ''}])

    def test_reject_missing_duplicate_or_nonstring_samples(self):
        tasks = [{'task_id': 'HumanEval/0', 'prompt': 'def f():', 'entry_point': 'f'}]
        sample = {'task_id': 'HumanEval/0', 'solution': 'def f(): pass'}
        for samples in [[], [sample, sample], [dict(sample, solution=None)],
                        [dict(sample, task_id='HumanEval/1')]]:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                check_samples(tasks, samples)

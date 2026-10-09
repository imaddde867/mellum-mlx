"""Prevent duplicated or incomplete task sets from corrupting benchmark counts."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from coding_probe import validate_tasks


class TaskIntegrityTests(unittest.TestCase):
    def test_valid_task(self):
        validate_tasks([{'task_id': 'HumanEval/0', 'prompt': 'def f():', 'entry_point': 'f'}])

    def test_reject_duplicate_or_missing_tasks(self):
        task = {'task_id': 'HumanEval/0', 'prompt': 'def f():', 'entry_point': 'f'}
        for tasks in [[], [task, task], [dict(task, prompt='')], [dict(task, entry_point=None)],
                      [dict(task, task_id='other/0')]]:
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                validate_tasks(tasks)

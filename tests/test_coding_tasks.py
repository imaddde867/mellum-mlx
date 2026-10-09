"""Prevent duplicated or incomplete task sets from corrupting benchmark counts."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from coding_probe import validate_tasks, load_tasks, weight_hashes


class TaskIntegrityTests(unittest.TestCase):
    def test_valid_task(self):
        validate_tasks([{'task_id': 'HumanEval/0', 'prompt': 'def f():', 'entry_point': 'f'}], smoke=True)

    def test_reject_duplicate_or_missing_tasks(self):
        task = {'task_id': 'HumanEval/0', 'prompt': 'def f():', 'entry_point': 'f'}
        for tasks in [[], [task, task], [dict(task, prompt='')], [dict(task, entry_point=None)],
                      [dict(task, task_id='other/0')]]:
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                validate_tasks(tasks)

    def test_full_rejects_subset(self):
        with self.assertRaises(ValueError):
            validate_tasks([{'task_id': 'HumanEval/0', 'prompt': 'def f():', 'entry_point': 'f'}])

    def test_full_accepts_exact_ids(self):
        validate_tasks([{'task_id': f'HumanEval/{i}', 'prompt': 'def f():', 'entry_point': 'f'} for i in range(164)])

    def test_dataset_hash_is_enforced(self):
        tasks = [{'task_id': f'HumanEval/{i}', 'prompt': 'def f():', 'entry_point': 'f'} for i in range(164)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tasks.jsonl'
            path.write_text('\n'.join(json.dumps(t) for t in tasks))
            with self.assertRaises(ValueError):
                load_tasks(path)
            self.assertEqual(load_tasks(path, smoke=True), tasks)

    def test_weights_bind_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            with self.assertRaises(ValueError):
                weight_hashes(directory)
            path.write_bytes(b'first')
            first = weight_hashes(directory)
            path.write_bytes(b'other')
            self.assertNotEqual(first, weight_hashes(directory))

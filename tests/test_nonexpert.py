import importlib.util
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class ScreenTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).parents[1] / 'scripts/nonexpert.py'
        self.assertTrue(path.exists(), 'screen runner missing')
        spec = importlib.util.spec_from_file_location('nonexpert', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_reload_accepts_observed_variability_but_not_parameter_changes(self):
        module = self.module()
        self.assertEqual(module.assess_reload([0.5, 0.25], [0.5], []), 'within_observed_variability')
        for within, across, changed in [([0.0], [0.1], []), ([1], [0], ['x']), ([float('nan')], [0], []), ([], [0], [])]:
            with self.assertRaises(ValueError):
                module.assess_reload(within, across, changed)

    def test_frozen_tasks_are_complete_unique_and_hash_bound(self):
        module = self.module()
        rows = module.frozen()
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r['id'] for r in rows}), 24)
        for row in rows:
            compile(row['checks'], 'checks', 'exec')

    def test_pairing_keeps_failed_attempt_time(self):
        module = self.module()
        rows = [{'id':'a', 'passed':True, 'elapsed_s':2, 'completion_tokens':100, 'truncated':False, 'empty':False},
                {'id':'b', 'passed':False, 'elapsed_s':10, 'completion_tokens':1000, 'truncated':True, 'empty':True}]
        result = module.totals(rows)
        self.assertEqual(result['generation_s'], 12)
        self.assertEqual(result['failed_generation_s'], 10)
        self.assertEqual(result['passed'], 1)

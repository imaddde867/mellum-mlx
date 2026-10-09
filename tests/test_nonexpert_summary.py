import importlib.util
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class PairedSummaryTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).parents[1] / 'scripts/summarize_nonexpert.py'
        self.assertTrue(path.exists(), 'paired summary missing')
        spec = importlib.util.spec_from_file_location('summary', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_preserves_gains_losses_and_all_attempt_time(self):
        module = self.module()
        a = [{'id':'a','passed':True,'elapsed_s':2,'completion_tokens':10},
             {'id':'b','passed':False,'elapsed_s':8,'completion_tokens':100}]
        b = [{'id':'a','passed':False,'elapsed_s':5,'completion_tokens':20},
             {'id':'b','passed':True,'elapsed_s':3,'completion_tokens':30}]
        result = module.paired(a,b)
        self.assertEqual(result['left_only'], ['a'])
        self.assertEqual(result['right_only'], ['b'])
        self.assertEqual(result['both_passed'], [])
        self.assertEqual(result['neither_passed'], [])
        self.assertEqual(result['generation_s_delta'], 2)
        self.assertEqual(result['completion_tokens_delta'], 60)
        self.assertEqual(result['exact_mcnemar_p'], 1)

    def test_rejects_misaligned_task_rows(self):
        module = self.module()
        with self.assertRaises(ValueError):
            module.paired([{'id':'a'}],[{'id':'b'}])

    def test_small_unopposed_gain_is_not_strong_statistical_evidence(self):
        module = self.module()
        a = [{'id':str(i),'passed':True,'elapsed_s':1,'completion_tokens':1} for i in range(3)]
        b = [dict(row,passed=False) for row in a]
        self.assertEqual(module.paired(a,b)['exact_mcnemar_p'], 0.25)

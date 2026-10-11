import importlib.util
import sys
from pathlib import Path
import unittest
import json
import hashlib
import subprocess
from tempfile import TemporaryDirectory
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

    def test_current_candidate_subset_retains_failed_time(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for label, seconds, tokens in [('C',146,8422),('mxfp4',72,480)]:
                rows = [{'id':f'NE/{i:02d}', 'passed':label=='mxfp4' or i!=0,
                         'elapsed_s':100 if label=='C' and i==0 else (2 if label=='C' else 3),
                         'completion_tokens':8192 if label=='C' and i==0 else (10 if label=='C' else 20),
                         'truncated':label=='C' and i==0, 'empty':label=='C' and i==0}
                        for i in range(24)]
                raw = root/f'screen-{label}.raw.jsonl'
                raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                report = {'status':'complete','rows':rows,'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
                          'weight_bytes':7_000_000_000,'passed':23 if label=='C' else 24,
                          'generation_s':seconds,'failed_generation_s':100 if label=='C' else 0,
                          'completion_tokens':tokens,'truncated':int(label=='C'),'empty':int(label=='C'),
                          'memory':{'mlx_peak_bytes':8_000_000_000},'elapsed_with_startup_and_tests_s':seconds+10}
                report.update({key:'shared' for key in ['protocol','script_sha256','server_sha256','packages','fixture_manifest_sha256']})
                (root/f'screen-{label}.json').write_text(json.dumps(report))
            run = subprocess.run([sys.executable,str(Path(module.__file__)), '--results',str(root),
                                  '--labels','C','mxfp4'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            result = json.loads((root/'summary.json').read_text())
            self.assertEqual(result['all_models_generation_s'],218)
            self.assertEqual(result['rows'][0]['failed_generation_s'],100)
            self.assertEqual(result['rows'][0]['vs_mxfp4']['right_only'],['NE/00'])

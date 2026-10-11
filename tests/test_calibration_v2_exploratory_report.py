import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class ExploratoryReportTests(unittest.TestCase):
    def test_report_preserves_every_step_region_and_family_without_selecting(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2_exploratory_report'),'learning-curve reporter missing')
        import calibration_v2_exploratory_report as m
        def repeats(offset):
            return [{'means':{'assistant':float(i+offset),'context':2.*(i+offset)},
                     'families':{'first':{'assistant':float(i+offset)},'second':{'assistant':2.*(i+offset)}}} for i in range(1,6)]
        receipt={'arm':'U','status':'failed','updates':9,'evaluations':{'0':repeats(0),'8':repeats(-1)}}
        report=m.summarize_arm(receipt)
        self.assertEqual(report['updates'],9)
        self.assertEqual(list(report['learning_curve']),['0','8'])
        value=report['learning_curve']['8']
        self.assertEqual(value['regions']['assistant'],{'mean':2.,'min':0.,'max':4.,'range':4.})
        self.assertEqual(value['families']['second']['assistant']['mean'],4.)
        self.assertEqual(value['unweighted_family_assistant_mean'],3.)
        self.assertNotIn('selected_candidate',report)
        receipt['evaluations']['8']=receipt['evaluations']['8'][:4]
        with self.assertRaises(ValueError):m.summarize_arm(receipt)

class BehaviorSummaryTests(unittest.TestCase):
    def test_all_emitted_calls_and_failed_generation_are_counted(self):
        import calibration_v2_exploratory_report as m
        row={'executable':{'passed':False},'final_present':False,'truncated':True,'error':None,
             'completion_tokens':8192,'completion_tokens_known':True,'generation_elapsed_s':4.,'conversation_elapsed_s':5.,
             'tool_calls':0,'valid_tool_calls':0,'attempts':[{'result':{'message':{'tool_calls':[{'function':{'name':'read_file','arguments':'{"path":"outside.py"}'}}]}}}]}
        self.assertTrue(callable(getattr(m,'summarize_behavior',None)),'behavior summary missing')
        result=m.summarize_behavior({'status':'complete','rows':[row]})
        self.assertEqual(result['emitted_tool_calls'],1);self.assertEqual(result['schema_valid_emitted_calls'],0)
        self.assertEqual(result['executed_tool_calls'],0);self.assertEqual(result['generation_elapsed_s'],4.)
        self.assertEqual(result['token_truncated_tasks'],1);self.assertEqual(result['missing_final_tasks'],1)
        self.assertEqual(result['completion_tokens'],8192);self.assertEqual(result['executable_correct'],0)

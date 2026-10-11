import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class ReportTests(unittest.TestCase):
    def test_merge_preserves_historical_scores_and_overlapping_categories(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2_behavior_report'),'paired diagnostic report missing')
        from calibration_v2_behavior_report import merge
        a={'index':0,'family':'f','models':{'S':{'correct':False,'final_present':False,'substantive_final':False,
            'workspace_original_passed':True,'edits':1,'failure_categories':['response_budget','passing_workspace_missing_final']}}}
        b={'index':0,'family':'f','models':{'BF16':{'correct':True,'final_present':True,'substantive_final':True,
            'workspace_original_passed':True,'edits':1,'failure_categories':[]}}}
        combined=merge([[a],[b]])
        self.assertFalse(combined[0]['models']['S']['correct'])
        self.assertEqual(combined[0]['models']['S']['failure_categories'],['response_budget','passing_workspace_missing_final'])
        self.assertTrue(combined[0]['models']['BF16']['correct'])
        self.assertEqual(a['models']['S']['failure_categories'],['response_budget','passing_workspace_missing_final'])
        with self.assertRaises(ValueError):merge([[a],[{**b,'family':'different'}]])
        with self.assertRaises(ValueError):merge([[a],[a]])

    def test_pairing_includes_failed_attempt_costs(self):
        from calibration_v2_behavior_audit import summarize
        common={'final_present':True,'substantive_final':True,'responses':1,'tool_calls':0,
            'repeated_unchanged_calls':0,'reads_after_passing_tool_test':0,'failed_intermediate_tests':0,
            'passed_intermediate_tests':0,'empty_tool_termination':False,'failure_categories':[]}
        rows=[{'index':0,'models':{'S':{**common,'correct':False,'completion_tokens':100,'generation_elapsed_s':10},
                                      'C':{**common,'correct':True,'completion_tokens':30,'generation_elapsed_s':3}}},
              {'index':1,'models':{'S':{**common,'correct':True,'completion_tokens':20,'generation_elapsed_s':2},
                                      'C':{**common,'correct':False,'completion_tokens':40,'generation_elapsed_s':4}}}]
        r=summarize(rows)
        self.assertEqual(r['paired']['S vs C'],{'gains':[1],'losses':[0]})
        self.assertEqual(r['models']['S']['completion_tokens'],120)
        self.assertEqual(r['models']['S']['generation_elapsed_s'],12)
        self.assertEqual(r['models']['S']['failed_generation_elapsed_s'],10)

class BudgetOverlapTests(unittest.TestCase):
    def test_fourth_response_token_limit_retains_both_observations(self):
        from calibration_v2_behavior_report import merge
        row={'index':0,'family':'f','models':{'S':{'correct':False,'final_present':False,'substantive_final':False,
            'workspace_original_passed':False,'edits':1,'failure_categories':['token_budget'],
            'responses':4,'stop_reason':'token_budget'}}}
        value=merge([[row]])[0]['models']['S']
        self.assertTrue(value.get('response_cap_reached_without_final',False))
        self.assertEqual(value['stop_reason'],'token_budget')
        self.assertEqual(value['failure_categories'],['token_budget'])

class MaterialTests(unittest.TestCase):
    def test_tool_reply_shapes_follow_call_ids_and_preserve_raw_read(self):
        import calibration_v2_behavior_report as m
        self.assertTrue(hasattr(m,'tool_reply_shapes'),'tool reply context audit missing')
        row={'messages':[{'role':'assistant','tool_calls':[{'id':'r','function':{'name':'read_file'}}]},
            {'role':'tool','tool_call_id':'r','content':'def f(): return 1'},
            {'role':'assistant','tool_calls':[{'id':'e','function':{'name':'edit_file'}}]},
            {'role':'tool','tool_call_id':'e','content':'{"ok":true}'},
            {'role':'assistant','tool_calls':[{'id':'t','function':{'name':'run_tests'}}]},
            {'role':'tool','tool_call_id':'t','content':'{"sanitized":"def f(): return 1","passed":true}'}]}
        self.assertEqual(m.tool_reply_shapes([row]),{'read_file':{'raw_text':1},'edit_file':{'json:ok':1},'run_tests':{'json:passed,sanitized':1}})
        with self.assertRaises(ValueError):m.tool_reply_shapes([{'messages':[{'role':'tool','tool_call_id':'missing','content':'{}'}]}])

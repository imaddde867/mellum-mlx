import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class BehaviorTests(unittest.TestCase):
    def test_tool_schema_and_workspace(self):
        from calibration_v2_exploratory_behavior import tool_result
        case={'entry_point':'f','checks':'assert f()==1'}
        call={'function':{'name':'edit_file','arguments':'{"path":"task.py","content":"def f():\\n return 1\\n"}'}}
        source,result=tool_result(call,'old',case)
        self.assertEqual(source,'def f():\n return 1\n');self.assertTrue(result['ok'])
        call['function']['arguments']='{"path":"other.py","content":"x"}'
        with self.assertRaises(ValueError):tool_result(call,source,case)
    def test_completion_budget_counts_all_responses(self):
        from calibration_v2_exploratory_behavior import completion_count
        self.assertEqual(completion_count({'response':{'usage':{'completion_tokens':47}}},50),47)
        for value in (None,-1,51,True):
            with self.assertRaises(ValueError):completion_count({'response':{'usage':{'completion_tokens':value}}},50)
    def test_attempt_retains_tool_final_and_shared_budget(self):
        from unittest.mock import patch
        import calibration_v2_exploratory_behavior as m
        case={'family':'test','prompt':'Return one','entry_point':'f','checks':'assert f()==1'}
        responses=[{'message':{'role':'assistant','tool_calls':[{'id':'x','function':{'name':'edit_file',
                    'arguments':'{"path":"task.py","content":"def f(): return 1"}'}}]},'finish_reason':'tool_calls','response':{'usage':{'completion_tokens':100}}},
            {'message':{'role':'assistant','content':'def f(): return 1'},'finish_reason':'stop','response':{'usage':{'completion_tokens':30}}}]
        with patch.object(m,'request',side_effect=responses) as request,patch.object(m,'sandbox',return_value={'passed':True}):
            row=m.attempt(case,'http://localhost','model')
        self.assertTrue(row['executable']['passed']);self.assertEqual(row['completion_tokens'],130)
        self.assertEqual(row['valid_tool_calls'],1);self.assertEqual(len(row['attempts']),2)
        self.assertEqual(request.call_args_list[1].args[1]['max_tokens'],8092)
        self.assertEqual(request.call_args_list[1].args[1]['messages'][-1]['role'],'tool')
        self.assertEqual(row['attempts'][0]['tool_results'],[{'ok':True}])
        self.assertEqual(len(row['attempts'][0]['request']['messages']),1)
    def test_unknown_tokens_and_truncation_stay_failures(self):
        from unittest.mock import patch
        import calibration_v2_exploratory_behavior as m
        case={'family':'test','prompt':'Return one','entry_point':'f','checks':'assert f()==1'}
        value={'message':{'role':'assistant','content':'def f(): return 1'},'finish_reason':'length','response':{'usage':{'completion_tokens':8192}}}
        with patch.object(m,'request',return_value=value) as request:
            row=m.attempt(case,'http://localhost','model')
        self.assertTrue(row['truncated']);self.assertFalse(row['final_present']);self.assertEqual(request.call_count,1)
        value['response']['usage']={}
        with patch.object(m,'request',return_value=value):row=m.attempt(case,'http://localhost','model')
        self.assertFalse(row['completion_tokens_known']);self.assertIn('accounting',row['error'])

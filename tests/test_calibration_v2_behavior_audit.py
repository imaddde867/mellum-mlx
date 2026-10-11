import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class AuditTests(unittest.TestCase):
    def test_audit_keeps_missing_final_failed_despite_passing_test(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2_behavior_audit'),'behavior audit missing')
        from calibration_v2_behavior_audit import inspect_task
        import json
        source='def f(): return 1'
        call=lambda name,args:{'id':name,'function':{'name':name,'arguments':json.dumps(args)}}
        attempts=[]
        messages=[{'role':'user','content':'prompt'}]
        for c,result in [(call('edit_file',{'path':'task.py','content':source}),{'ok':True}),
                         (call('run_tests',{'path':'task.py','code':'assert f()==1'}),{'passed':True}),
                         (call('read_file',{'path':'task.py'}),{'content':source}),
                         (call('read_file',{'path':'task.py'}),{'content':source})]:
            msg={'role':'assistant','tool_calls':[c]}
            attempts.append({'request':{'messages':list(messages),'max_tokens':8192-len(attempts)*10},
                'result':{'message':msg,'finish_reason':'tool_calls','response':{'usage':{'completion_tokens':10}}},
                'tool_results':[result],'elapsed_s':1,'completion_accounted':True})
            messages += [msg,{'role':'tool','tool_call_id':c['id'],'content':json.dumps(result)}]
        row={'family':'f','attempts':attempts,'final_present':False,'truncated':False,'error':None,
             'executable':{'passed':False,'error':'No final source'},'completion_tokens':40,'generation_elapsed_s':4}
        result,jobs=inspect_task({'family':'f','entry_point':'f','checks':'assert f()==1'},row)
        self.assertFalse(result['correct']);self.assertEqual(result['stop_reason'],'response_budget')
        self.assertEqual(result['repeated_unchanged_calls'],1)
        self.assertEqual(result['reads_after_passing_tool_test'],2)
        self.assertEqual(result['state_replay_errors'],[])
        self.assertEqual(jobs['workspace'],source)

    def test_parser_drop_is_not_a_substantive_final(self):
        from calibration_v2_behavior_audit import inspect_task
        row={'family':'f','attempts':[{'request':{'messages':[{'role':'user','content':'prompt'}],'max_tokens':8192},
            'result':{'finish_reason':'tool_calls','message':{'role':'assistant','content':'\n\n','tool_calls':[]},
            'response':{'usage':{'completion_tokens':11}}},'elapsed_s':2}],
            'final_present':True,'truncated':False,'error':None,'executable':{'passed':False},
            'completion_tokens':11,'generation_elapsed_s':2}
        result,_=inspect_task({'family':'f','entry_point':'f','checks':'assert f()==1'},row)
        self.assertTrue(result['final_present']);self.assertFalse(result['substantive_final'])
        self.assertTrue(result['empty_tool_termination']);self.assertEqual(result['stop_reason'],'empty_tool_termination')

    def test_controls_reject_incomplete_or_wrong_checkpoint(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2_behavior_controls'),'diagnostic controls missing')
        from calibration_v2_behavior_controls import check_u_checkpoint
        receipt={'arm':'U','status':'complete','updates':128,'uninterrupted':True,
            'checkpoint_state_sha256':{'step-128-parameters.safetensors':'exact'}}
        check_u_checkpoint(receipt,'exact')
        for bad in ({**receipt,'updates':127},{**receipt,'arm':'S'},{**receipt,'status':'failed'}):
            with self.assertRaises(ValueError):check_u_checkpoint(bad,'exact')
        with self.assertRaises(ValueError):check_u_checkpoint(receipt,'wrong')

class ContractTests(unittest.TestCase):
    def test_diagnostic_authorization_cannot_change_protocol_or_model_order(self):
        from calibration_v2_behavior_controls import check_extension
        p={'scope':'diagnostic only; no promotion','controls':['U-128','BF16'],
           'runtime':{'MLX_USE_CUDA_GRAPHS':'1','MLX_CUDA_GRAPH_CACHE_SIZE':'400','MLX_ENABLE_CACHE_THRASHING_CHECK':'0'},
           'parent_protocol_sha256':'p','parent_details_sha256':'d','historical_audit_sha256':'a'}
        check_extension(p,'p','d','a')
        for key,value in [('controls',['U-64','BF16']),('scope','selection'),('parent_protocol_sha256','x'),('runtime',{})]:
            with self.assertRaises(ValueError):check_extension({**p,key:value},'p','d','a')

    def test_historical_empty_tool_termination_is_recorded_not_rescored(self):
        from unittest.mock import patch
        import calibration_v2_exploratory_behavior as m
        value={'message':{'role':'assistant','content':'\n\n','tool_calls':[]},'finish_reason':'tool_calls',
               'response':{'usage':{'completion_tokens':19}}}
        with patch.object(m,'request',return_value=value),patch.object(m,'sandbox',return_value={'passed':False,'error':'NameError'}):
            row=m.attempt({'family':'f','entry_point':'f','checks':'assert f()==1','prompt':'Return one'},'http://localhost','model')
        self.assertTrue(row['final_present']);self.assertFalse(row['executable']['passed'])
        self.assertEqual(len(row['attempts']),1)
        from calibration_v2_behavior_audit import inspect_task
        v,_=inspect_task({'family':'f','entry_point':'f','checks':'assert f()==1'},row)
        self.assertTrue(v['empty_tool_termination']);self.assertFalse(v['substantive_final'])

    @unittest.skipUnless(__import__('os').environ.get('AUDIT_SANDBOX')=='1','pinned sandbox integration')
    def test_virtual_task_module_cannot_be_imported_in_frozen_sandbox(self):
        from calibration_v2_exploratory_behavior import tool_result
        call={'function':{'name':'run_tests','arguments':'{"path":"task.py","code":"from task import f\\nassert f()==1"}'}}
        _,result=tool_result(call,'def f(): return 1',{'entry_point':'f','checks':'assert f()==1'})
        self.assertFalse(result['passed']);self.assertIn('ModuleNotFoundError',result['error'])

class ServerLogTests(unittest.TestCase):
    def test_warning_maps_to_active_response_not_following_request(self):
        import calibration_v2_behavior_audit as m
        self.assertTrue(hasattr(m,'parser_warnings'),'parser warnings must be linked to task/response')
        rows=[{'attempts':[{},{}]},{'attempts':[{}]}]
        log='POST /v1/chat/completions HTTP/1.1\nPOST /v1/chat/completions HTTP/1.1\nWARNING - Failed to parse tool call (invalid JSON)\nPOST /v1/chat/completions HTTP/1.1\n'
        warnings=m.parser_warnings(log,rows)
        self.assertEqual(warnings,[{'task':0,'response':1,'line':3,'warning':'WARNING - Failed to parse tool call (invalid JSON)'}])

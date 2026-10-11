"""Single reserved-task attempt per model under the locked original behavior protocol."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from calibration_v2 import TOOLS,check_call
from calibration_v2_exploratory_artifact import check_reserved_gate
from calibration_v2_runtime import EXPLORATORY,RUNTIME,require_runtime
from convert import sha256
from http_probe import request
from nonexpert import binding,sandbox,write_new


def completion_count(response,remaining):
    value=response.get('response',{}).get('usage',{}).get('completion_tokens')
    if type(value)!=int or not 0<=value<=remaining:raise ValueError('Missing or invalid completion token accounting')
    return value


def tool_result(call,source,case):
    fn=call['function'];args=json.loads(fn['arguments'])
    check_call({'name':fn['name'],'arguments':args})
    if fn['name']=='read_file':return source,{'content':source}
    if fn['name']=='edit_file':return args['content'],{'ok':True}
    return source,sandbox({**case,'checks':args['code']},source)


def attempt(case,url,model):
    source=case.get('buggy') or f"def {case['entry_point']}(*args, **kwargs):\n    raise NotImplementedError\n"
    content=(case['prompt']+'\nEntry point: '+case['entry_point']+'\nAssertions:\n'+case['checks']+
        '\nUse edit_file to replace task.py, run_tests to check it, then provide the complete final Python source.\nCurrent task.py:\n'+source)
    messages=[{'role':'user','content':content}];rows=[];tokens=0;valid=0;calls_seen=0;final='';error=None;truncated=False
    started=time.monotonic()
    for number in range(4):
        remaining=8192-tokens
        if remaining<=0:truncated=True;break
        payload={'model':model,'messages':[dict(m) for m in messages],'tools':TOOLS,'temperature':0,'seed':0,'max_tokens':remaining,'stream':False}
        before=time.monotonic();row={'number':number,'request':payload}
        try:
            response=request(url+'/chat/completions',payload)
            row['elapsed_s']=time.monotonic()-before
            row['result']=response;tokens+=completion_count(response,remaining)
            row['completion_accounted']=True
            message=response['message'];row['elapsed_s']=time.monotonic()-before
            rows.append(row)
            truncated=truncated or response['finish_reason']=='length'
            calls=message.get('tool_calls') or []
            if truncated:break
            if not calls:final=message.get('content') or '';break
            assistant=dict(message);assistant.pop('reasoning',None);assistant.pop('reasoning_content',None)
            messages.append(assistant)
            for call in calls:
                calls_seen+=1
                source,result=tool_result(call,source,case);valid+=1
                row.setdefault('tool_results',[]).append(result)
                messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result)})
        except Exception as exc:
            error=repr(exc);row.update(error=error,transport_error=isinstance(exc,(OSError,TimeoutError)))
            row.setdefault('elapsed_s',time.monotonic()-before)
            if not rows or rows[-1] is not row:rows.append(row)
            break
    elapsed=time.monotonic()-started
    oracle=sandbox(case,final) if final else {'passed':False,'error':'No final source'}
    return {'family':case['family'],'attempts':rows,'executable':oracle,'error':error,'truncated':truncated,
        'completion_tokens':tokens,'completion_tokens_known':all(r.get('completion_accounted',False) for r in rows),
        'generation_elapsed_s':sum(r['elapsed_s'] for r in rows),'conversation_elapsed_s':elapsed,'tool_calls':calls_seen,'valid_tool_calls':valid,'final_present':bool(final)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('decision','reload','lock','protocol','model','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--label',required=True,choices=['candidate','C','MXFP4']);a=p.parse_args()
    require_runtime(before_import=True)
    decision=json.loads(a.decision.read_text());reload=json.loads(a.reload.read_text());lock=json.loads(a.lock.read_text())
    candidate=Path(lock['artifact']['model'])
    check_reserved_gate(decision,reload,sha256(a.decision),sha256(EXPLORATORY),binding(candidate)['weights_sha256'])
    protocol=json.loads(a.protocol.read_text())
    if sha256(a.protocol)!=lock['behavior_protocol_sha256'] or sha256(a.reload)!=lock['reload_sha256'] or protocol['models'][a.label]!=str(a.model):raise ValueError('Behavior lock changed')
    if a.output.exists():raise ValueError('No reserved retry/overwrite')
    if binding(a.model)['weights_sha256']!=protocol['bindings'][a.label]['weights_sha256']:raise ValueError('Behavior model changed')
    reserved=Path('fixtures/calibration-v2/reserved.jsonl')
    if sha256(reserved)!=protocol['reserved_sha256']:raise ValueError('Reserved source changed')
    cases=[json.loads(line) for line in reserved.read_text().splitlines()]
    if len(cases)!=32:raise ValueError('Reserved task count changed')
    a.output.mkdir(parents=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}/v1';env={**os.environ,**RUNTIME}
    env.setdefault('HF_HOME',str(Path('work/hf-cache').resolve()));(Path(env['HF_HOME'])/'hub').mkdir(parents=True,exist_ok=True)
    report={'status':'running','label':a.label,'model':binding(a.model),'protocol_sha256':sha256(a.protocol),'lock_sha256':sha256(a.lock),'rows':[]}
    with (a.output/'server.log').open('x') as log:
        server=subprocess.Popen([sys.executable,'-m','mlx_lm','server','--model',str(a.model),'--host','127.0.0.1','--port',str(port),
            '--decode-concurrency','1','--prompt-concurrency','1','--prefill-step-size','512'],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while True:
                if server.poll() is not None:raise RuntimeError('Server exited before ready')
                try:request(url+'/models');break
                except (OSError,TimeoutError):
                    if time.monotonic()>deadline:raise TimeoutError('Server startup')
                    time.sleep(.25)
            for index,case in enumerate(cases):
                if server.poll() is not None:raise RuntimeError('Server exited; no CUDA recovery')
                row=attempt(case,url,str(a.model));write_new(a.output/f'task-{index:02d}.json',row)
                report['rows'].append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('attempts','executable')}|{'index':index,'passed':row['executable']['passed']}),flush=True)
                if server.poll() is not None or any(r.get('transport_error') for r in row['attempts']):raise RuntimeError('Server/transport failed during task; no retry')
            report['status']='complete'
        except Exception as exc:report.update(status='failed',error=repr(exc));raise
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            write_new(a.output/'receipt.json',report)

if __name__=='__main__':main()

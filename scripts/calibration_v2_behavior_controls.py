"""Exactly U-128 and pinned BF16 diagnostic controls; no post-hoc model selection."""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from convert import sha256
from nonexpert import binding,write_new
from calibration_v2_runtime import RUNTIME,require_runtime

ROOT=Path('results/calibration-v2-behavior-audit')
HISTORY=Path('results/calibration-v2-exploratory')
ARTIFACT=Path('work/calibration-v2-diagnostic-U128')
ARM=Path('work/calibration-v2-exploratory-U')


def check_u_checkpoint(receipt,digest):
    if receipt.get('arm')!='U' or receipt.get('status')!='complete' or receipt.get('updates')!=128 or receipt.get('checkpoint_state_sha256',{}).get('step-128-parameters.safetensors')!=digest:
        raise ValueError('Control must be the completed frozen U-128 checkpoint')


def check_extension(protocol,parent,details,audit):
    if protocol.get('scope')!='diagnostic only; no promotion' or protocol.get('controls')!=['U-128','BF16'] or protocol.get('runtime')!=RUNTIME or protocol.get('parent_protocol_sha256')!=parent or protocol.get('parent_details_sha256')!=details or protocol.get('historical_audit_sha256')!=audit:
        raise ValueError('Diagnostic authorization changed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['export','verify','behavior'])
    p.add_argument('--label',choices=['U-128','BF16']);a=p.parse_args();require_runtime(before_import=True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    check_extension(protocol,sha256(HISTORY/'behavior-protocol.json'),sha256(HISTORY/'behavior-details-v2.json'),sha256(ROOT/'historical-replay.json'))
    for path,digest in protocol['source_sha256'].items():
        if sha256(Path(path))!=digest:raise ValueError('Diagnostic code changed: '+path)
    import importlib.util
    if sha256(Path(importlib.util.find_spec('mlx_lm.server').origin))!=protocol['server_sha256']:raise ValueError('Pinned server changed')
    from calibration_v2_exploratory_artifact import digest
    if a.stage in ('export','verify'):
        import mlx.core as mx
        from mlx.utils import tree_flatten,tree_unflatten
        import calibration_v2_runtime as runtime
        import calibration_v2_objective as base
        report={'stage':a.stage,'status':'running','protocol_sha256':sha256(ROOT/'protocol.json'),'runtime':require_runtime()}
        if a.stage=='export':
            if ARTIFACT.exists():raise ValueError('No artifact overwrite')
            receipt=json.loads((ARM/'receipt.json').read_text());path=ARM/'step-128-parameters.safetensors'
            check_u_checkpoint(receipt,sha256(path))
            if sha256(ARM/'receipt.json')!=protocol['u_receipt_sha256'] or receipt['bindings']!=runtime.bindings():raise ValueError('U frozen state binding changed')
            restart=ARM/'restart/step-128';state=json.loads((restart/'receipt.json').read_text())
            if not state['serialization_verified'] or state['arm']!='U' or state['cursor']!=128 or state['optimizer_step']!=128 or state['bindings']!=receipt['bindings']:raise ValueError('U saved state binding mismatch')
            for name,value in state['files_sha256'].items():
                if sha256(restart/name)!=value:raise ValueError('U saved state changed')
            from safetensors import safe_open
            import numpy as np
            with safe_open(path,framework='numpy') as saved,safe_open(restart/'parameters.safetensors',framework='numpy') as rolling:
                if saved.keys()!=rolling.keys():raise ValueError('Scheduled and restart keys differ')
                for key in saved.keys():
                    x,y=saved.get_tensor(key),rolling.get_tensor(key)
                    if x.dtype!=np.float32 or x.dtype!=y.dtype or x.shape!=y.shape or not np.array_equal(x,y):raise ValueError('U checkpoint/restart masters differ: '+key)
                    del x,y
            model,original,config,conversion=runtime.model_setup();masters=mx.load(str(path))
            expected=dict(tree_flatten(model.trainable_parameters()))
            if masters.keys()!=expected.keys() or any(v.dtype!=mx.float32 or v.shape!=expected[k].shape for k,v in masters.items()):raise ValueError('U masters schema mismatch')
            model.update(tree_unflatten([(k,v.astype(mx.bfloat16)) for k,v in masters.items()]));mx.eval(model.parameters())
            report['changed_parameters']=runtime.parameter_integrity(model,original,config,conversion)
            report['parameter_digests']={k:digest(v) for k,v in tree_flatten(model.parameters())}
            report['checkpoint_sha256']=sha256(path);report['u_receipt_sha256']=sha256(ARM/'receipt.json')
            report['restart_receipt_sha256']=sha256(restart/'receipt.json');report['scheduled_restart_masters_exact']=True
            del masters,expected;original.clear();mx.clear_cache()
            from mlx_lm.utils import save_model
            ARTIFACT.mkdir();save_model(ARTIFACT,model,donate_model=True)
            for src in base.STUDENT.iterdir():
                if src.is_file() and src.suffix!='.safetensors' and src.name not in ('conversion.json','README.md','model.safetensors.index.json'):shutil.copy2(src,ARTIFACT/src.name)
            conversion['artifact_sha256']={f.name:sha256(f) for f in ARTIFACT.iterdir() if f.is_file()}
            conversion['calibration']={'scope':'diagnostic control, not selected or promoted','arm':'U','step':128,'checkpoint_sha256':report['checkpoint_sha256']}
            write_new(ARTIFACT/'conversion.json',conversion)
        else:
            export=json.loads((ROOT/'U-export.json').read_text())
            if export['protocol_sha256']!=report['protocol_sha256'] or export['artifact']['weights_sha256']!=binding(ARTIFACT)['weights_sha256']:raise ValueError('U export changed')
            mx.set_cache_limit(0);model,_,_,_=base.student(ARTIFACT,pristine=False)
            flat=dict(tree_flatten(model.parameters()))
            if flat.keys()!=export['parameter_digests'].keys():raise ValueError('U reload keys differ')
            for key,value in flat.items():
                if digest(value)!=export['parameter_digests'][key]:raise ValueError('U reload differs: '+key)
            report['parameter_digests_exact']=True
        report.update(status='complete',integrity_passed=True,artifact=binding(ARTIFACT))
        write_new(ROOT/('U-'+a.stage+'.json'),report);return
    if a.label is None:raise ValueError('Missing control label')
    model=ARTIFACT if a.label=='U-128' else Path('work/source')
    actual=binding(model)
    if a.label=='U-128':
        verified=json.loads((ROOT/'U-verify.json').read_text())
        if verified['status']!='complete' or not verified['integrity_passed'] or verified['protocol_sha256']!=sha256(ROOT/'protocol.json') or verified['artifact']['weights_sha256']!=actual['weights_sha256']:raise ValueError('Unverified U artifact')
    elif actual['weights_sha256']!=protocol['bf16_binding']['weights_sha256'] or actual['config_sha256']!=protocol['bf16_binding']['config_sha256']:raise ValueError('Pinned BF16 changed')
    if actual['packages']!=protocol['packages']:raise ValueError('Pinned packages changed')
    for name,value in protocol['shared_tokenizer_sha256'].items():
        if sha256(model/name)!=value:raise ValueError('Control tokenizer/template changed')
    reserved=Path('fixtures/calibration-v2/reserved.jsonl')
    if sha256(reserved)!=protocol['reserved_sha256']:raise ValueError('Diagnostic task set changed')
    output=ROOT/('behavior-'+a.label)
    if output.exists():raise ValueError('No control retry/overwrite')
    output.mkdir()
    cases=[json.loads(s) for s in reserved.read_text().splitlines()]
    if len(cases)!=32:raise ValueError('Control task count changed')
    from calibration_v2_exploratory_behavior import attempt
    from http_probe import request
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}/v1';env={**os.environ,**RUNTIME}
    env.setdefault('HF_HOME',str(Path('work/hf-cache').resolve()));(Path(env['HF_HOME'])/'hub').mkdir(parents=True,exist_ok=True)
    report={'status':'running','label':a.label,'model':actual,'protocol_sha256':sha256(ROOT/'protocol.json'),'runtime':require_runtime(),'rows':[]}
    with (output/'server.log').open('x') as log:
        server=subprocess.Popen([sys.executable,'-m','mlx_lm','server','--model',str(model),'--host','127.0.0.1','--port',str(port),
            '--decode-concurrency','1','--prompt-concurrency','1','--prefill-step-size','512'],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while True:
                if server.poll() is not None:raise RuntimeError('Server exited before ready')
                try:request(url+'/models');break
                except (OSError,TimeoutError):
                    if time.monotonic()>deadline:raise TimeoutError('Server startup')
                    time.sleep(.25)
            for i,case in enumerate(cases):
                if server.poll() is not None:raise RuntimeError('Server exited; no recovery')
                row=attempt(case,url,str(model));write_new(output/f'task-{i:02d}.json',row);report['rows'].append(row)
                print(json.dumps({'index':i,'passed':row['executable']['passed'],'final':row['final_present'],'tokens':row['completion_tokens'],'generation_s':row['generation_elapsed_s']}),flush=True)
                if server.poll() is not None or any(r.get('transport_error') for r in row['attempts']):raise RuntimeError('Server/transport failed; no retry')
            report['status']='complete'
        except Exception as exc:report.update(status='failed',error=repr(exc));raise
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            write_new(output/'receipt.json',report)

if __name__=='__main__':main()

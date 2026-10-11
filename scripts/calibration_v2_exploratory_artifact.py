"""Export one locked exploratory checkpoint and verify it in a fresh process."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from convert import sha256
from nonexpert import binding,write_new
import calibration_v2_objective as base
import calibration_v2_runtime as runtime


def check_reserved_gate(decision,reload,decision_sha,protocol_sha,weights_sha):
    if (decision.get('status')!='complete' or decision.get('arm') not in ('U','S') or
        decision.get('mode')!='exploratory' or decision.get('exploratory_protocol_sha256')!=protocol_sha or
        reload.get('status')!='complete' or reload.get('fidelity_gate_passed') is not True or
        reload.get('integrity_passed') is not True or reload.get('decision_sha256')!=decision_sha or
        reload.get('exploratory_protocol_sha256')!=protocol_sha or reload['artifact']['weights_sha256']!=weights_sha):
        raise ValueError('Reserved tasks require the matching selected and verified artifact')


def digest(value):
    import mlx.core as mx
    import numpy as np
    mx.eval(value)
    data=np.asarray(value.astype(mx.float32) if value.dtype==mx.bfloat16 else value)
    return {'dtype':str(value.dtype),'shape':list(value.shape),'sha256':hashlib.sha256(data.tobytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['export','verify'])
    parser.add_argument('--decision',type=Path,required=True);parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    runtime.require_runtime(before_import=True)
    decision=json.loads(args.decision.read_text());protocol_sha=sha256(runtime.EXPLORATORY)
    authorization,_=runtime.exploratory_authorization(runtime.bindings())
    if decision.get('status')!='complete' or decision.get('arm') not in ('U','S') or decision.get('exploratory_protocol_sha256')!=protocol_sha:
        raise ValueError('No authorized selected candidate')
    arms={a:json.loads((Path(p)/'receipt.json').read_text()) for a,p in decision['arm_paths'].items()}
    if any(sha256(Path(p)/'receipt.json')!=decision['arm_receipt_sha256'][a] for a,p in decision['arm_paths'].items()):raise ValueError('Arm evidence changed')
    _,_,valid,order=base.corpus();runtime.check_exploratory_arms(arms,runtime.bindings(),protocol_sha,order)
    import mlx.core as mx
    from mlx.utils import tree_flatten,tree_unflatten
    from mlx_lm.tuner.trainer import grad_checkpoint
    report={'status':'running','decision_sha256':sha256(args.decision),'exploratory_protocol_sha256':protocol_sha,'runtime':runtime.require_runtime()}
    if args.stage=='export':
        if args.artifact.exists():raise ValueError('Artifact already exists')
        model,original,config,conversion=runtime.model_setup()
        arm=decision['arm'];step=decision['step'];path=Path(decision['arm_paths'][arm])/f'step-{step}-parameters.safetensors'
        if sha256(path)!=arms[arm]['checkpoint_state_sha256'][path.name]:raise ValueError('Selected checkpoint changed')
        masters=mx.load(str(path));expected=dict(tree_flatten(model.trainable_parameters()))
        if masters.keys()!=expected.keys() or any(v.dtype!=mx.float32 or v.shape!=expected[k].shape for k,v in masters.items()):raise ValueError('Selected masters schema mismatch')
        model.update(tree_unflatten([(k,v.astype(mx.bfloat16)) for k,v in masters.items()]));mx.eval(model.parameters())
        changed=runtime.parameter_integrity(model,original,config,conversion)
        report['parameter_digests']={k:digest(v) for k,v in tree_flatten(model.parameters())}
        report['changed_parameters']=changed
        del masters,expected;original.clear();mx.clear_cache()
        from mlx_lm.utils import save_model
        args.artifact.mkdir(parents=True)
        save_model(args.artifact,model,donate_model=True)
        for src in base.STUDENT.iterdir():
            if src.is_file() and src.suffix!='.safetensors' and src.name not in ('conversion.json','README.md','model.safetensors.index.json'):
                shutil.copy2(src,args.artifact/src.name)
        conversion['artifact_sha256']={p.name:sha256(p) for p in args.artifact.iterdir() if p.is_file()}
        conversion['calibration']={'method':'DWQ','scope':'exploratory','decision_sha256':report['decision_sha256'],'protocol_sha256':protocol_sha,'arm':arm,'step':step,'changed_parameters':changed}
        write_new(args.artifact/'conversion.json',conversion)
        report.update(status='complete',integrity_passed=True,artifact=binding(args.artifact))
    else:
        export=json.loads((args.artifact/'export-receipt.json').read_text())
        if export['decision_sha256']!=report['decision_sha256'] or export['exploratory_protocol_sha256']!=protocol_sha:raise ValueError('Export binding changed')
        mx.set_cache_limit(0);mx.random.seed(123)
        model,_,_,_=base.student(args.artifact,pristine=False)
        flat=dict(tree_flatten(model.parameters()))
        if flat.keys()!=export['parameter_digests'].keys():raise ValueError('Reload keys changed')
        for k,v in flat.items():
            if digest(v)!=export['parameter_digests'][k]:raise ValueError('Reload parameter changed: '+k)
        del flat;grad_checkpoint(model.layers[0])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        repeat_path=args.output.parent/'reload-repeats';repeat_path.mkdir()
        repeats=base.validation(model,valid,repeat_path)
        passed=all(base.fidelity_gate([v['means'] for v in arms[a]['evaluations']['0']],[v['means'] for v in repeats]) for a in ('U','S'))
        report.update(status='complete',integrity_passed=True,fidelity_gate_passed=passed,artifact=binding(args.artifact),evaluations={'selected':repeats})
    write_new(args.output,report)
    if args.stage=='export':shutil.copy2(args.output,args.artifact/'export-receipt.json')

if __name__=='__main__':main()

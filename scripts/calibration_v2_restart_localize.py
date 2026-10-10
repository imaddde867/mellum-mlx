"""Frozen U-only persistence replay and resident/fresh-process backward controls."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import calibration_v2_runtime as runtime
from convert import sha256
from nonexpert import write_new

ROOT=Path('results/calibration-v2-restart-localize')
WORK=Path('work/calibration-v2-restart-localize')


def read_array(value):
    return np.load(value,mmap_mode='r') if isinstance(value,Path) else np.asarray(value)


def compare_arrays(actual,reference):
    if set(actual)!=set(reference):raise ValueError('Diagnostic keys differ')
    error=0.;norm=0.;maximum=0.;exact=True;groups={};tensors=[]
    for key in actual:
        a=read_array(actual[key]);b=read_array(reference[key])
        if a.dtype!=b.dtype or a.shape!=b.shape:raise ValueError('Diagnostic schema differs')
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite diagnostic')
        difference=np.subtract(a,b,dtype=np.float32)
        squared=np.square(difference,dtype=np.float32).sum(dtype=np.float64).item()
        reference_squared=np.square(b,dtype=np.float32).sum(dtype=np.float64).item()
        largest=float(np.max(np.abs(difference),initial=0))
        error+=squared;norm+=reference_squared;maximum=max(maximum,largest)
        exact=exact and np.array_equal(a,b)
        group=key.rsplit('.',1)[0]
        g=groups.setdefault(group,{'group':group,'squared_error':0.,'reference_squared':0.})
        g['squared_error']+=squared;g['reference_squared']+=reference_squared
        tensors.append({'tensor':key,'squared_error':squared,'reference_squared':reference_squared,'max_absolute':largest})
    if not np.isfinite(error) or not np.isfinite(norm):raise ValueError('Nonfinite reduction')
    for group in groups.values():group['error_fraction']=group['squared_error']/max(error,1e-30)
    return {'reference_l2':norm**.5,'absolute_l2':error**.5,'relative_l2':(error/max(norm,1e-30))**.5,
        'max_absolute':maximum,'exact':exact,'groups':sorted(groups.values(),key=lambda g:g['squared_error'],reverse=True),
        'top_tensors':sorted(tensors,key=lambda t:t['squared_error'],reverse=True)[:20]}


def mean_arrays(repeats,output):
    if len(repeats)!=5:raise ValueError('Exactly five repeats required')
    keys=set(repeats[0])
    if any(set(r)!=keys for r in repeats):raise ValueError('Repeat keys differ')
    output.mkdir(parents=True,exist_ok=False);result={}
    for key in repeats[0]:
        total=read_array(repeats[0][key]).copy()
        if total.dtype!=np.float32:raise ValueError('FP32 diagnostic required')
        for repeat in repeats[1:]:
            value=read_array(repeat[key])
            if value.dtype!=total.dtype or value.shape!=total.shape:raise ValueError('Repeat schema differs')
            np.add(total,value,out=total)
        np.divide(total,np.float32(5),out=total)
        if not np.isfinite(total).all():raise ValueError('Nonfinite diagnostic mean')
        path=output/(key+'.npy');np.save(path,total);result[key]=path
    return result


def array_digest(value):
    host=np.array(value)
    digest=hashlib.sha256()
    digest.update(str(host.dtype).encode());digest.update(str(host.shape).encode());digest.update(host.tobytes())
    return digest.hexdigest()


def optimizer_config(opt):
    value={'learning_rate':opt.learning_rate.item(),'betas':list(opt.betas),'epsilon':opt.eps,
        'bias_correction':opt.bias_correction,'schedulers':list(opt._schedulers),'class':type(opt).__name__}
    if value!={'learning_rate':float(np.float32(1e-7)),'betas':[.9,.999],'epsilon':1e-8,
              'bias_correction':True,'schedulers':[],'class':'Adam'}:raise ValueError('Optimizer configuration changed')
    return value


def replay_update(params,state,gradient):
    from mlx.utils import tree_map
    # Clone containers: Adam replaces leaves, so sharing dictionaries would advance the baseline.
    opt=runtime.restored_optimizer(params,tree_map(lambda x:x,state),1)
    optimizer_config(opt)
    return opt.apply_gradients(gradient,params),opt


def snapshot(model,params,state):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.eval(params,state,model.trainable_parameters(),mx.random.state)
    return {'masters':{k:array_digest(v) for k,v in tree_flatten(params)},
        'adam':{k:array_digest(v) for k,v in tree_flatten(state)},
        'bf16_values_as_fp32':{k:array_digest(v.astype(mx.float32)) for k,v in tree_flatten(model.trainable_parameters())},
        'mlx_rng':array_digest(mx.random.state[0]),'python_rng':random.getstate()}


def check_snapshot(model,params,state,expected):
    current=snapshot(model,params,state)
    # JSON represents the Python RNG tuples as lists.
    if json.dumps(current,sort_keys=True)!=json.dumps(expected,sort_keys=True):raise ValueError('Frozen starting state or RNG mutated')
    return True


def save_tree(path,tree):
    from mlx.utils import tree_flatten
    path.mkdir(parents=True,exist_ok=False);result={}
    for key,value in tree_flatten(tree):
        filename=path/(key+'.npy');host=np.array(value)
        if not np.isfinite(host).all():raise ValueError('Nonfinite captured array')
        np.save(filename,host);result[key]=str(filename)
    write_new(path/'index.json',result)
    return result


def paths(path):
    return {k:Path(v) for k,v in json.loads((path/'index.json').read_text()).items()}


def reset_rng(frozen):
    import mlx.core as mx
    key=np.load(WORK/'frozen/mlx_rng.npy')
    mx.random.state[0][:]=mx.array(key)
    def tuples(v):return tuple(tuples(x) for x in v) if isinstance(v,list) else v
    random.setstate(tuples(frozen['snapshot']['python_rng']))
    mx.eval(mx.random.state)


def capture_update(path,params,state,gradient):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    updated,opt=replay_update(params,state,gradient);mx.eval(updated,opt.state)
    save_tree(path/'masters',updated);save_tree(path/'adam',opt.state)
    flat=dict(tree_flatten(params));delta={}
    (path/'delta').mkdir()
    for key,value in tree_flatten(updated):
        filename=path/'delta'/(key+'.npy')
        np.save(filename,np.array(value)-np.array(flat[key]));delta[key]=str(filename)
    write_new(path/'delta/index.json',delta)
    result={'optimizer':optimizer_config(opt),'step':opt.step.item()}
    del updated,opt,flat,value;mx.clear_cache()
    return result


def block(model,params,state,row,name,report,frozen):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    result={'name':name,'repeats':[]}
    for repeat in range(1,6):
        reset_rng(frozen);check_snapshot(model,params,state,frozen['snapshot'])
        number=f'{name}-{repeat}';report['active_boundary']=number+' before-backward'
        runtime.memory_boundary(report,number+' before-backward',repeat)
        started=time.monotonic();loss,gradient=runtime.backward(model,params,row,'U')
        value=runtime.base.gradient_receipt(loss,gradient)
        value['rng_after_backward']=array_digest(mx.random.state[0])
        path=WORK/name/f'repeat-{repeat}';save_tree(path/'gradient',gradient)
        runtime.memory_boundary(report,number+' after-backward',repeat)
        value['update']=capture_update(path,params,state,gradient)
        del loss,gradient;mx.clear_cache();reset_rng(frozen)
        value.update(repeat=repeat,elapsed_s=time.monotonic()-started,
            starting_state_unchanged=check_snapshot(model,params,state,frozen['snapshot']),
            arrays_path=str(path))
        result['repeats'].append(value)
        write_new(ROOT/name/f'repeat-{repeat}.json',value)
        runtime.memory_boundary(report,number+' after-release',repeat)
    report.setdefault('blocks',[]).append(result)


def execution(stage):
    import mlx.core as mx
    from mlx.utils import tree_map,tree_unflatten
    report={**runtime.execution_receipt('U'),'stage':stage,'localization_script_sha256':sha256(Path(__file__)),
        'plan_sha256':sha256(ROOT/'plan.json'),'not_a_training_arm':True};started=time.monotonic()
    try:
        plan=json.loads((ROOT/'plan.json').read_text())
        if plan['runtime']!=runtime.RUNTIME or plan['runtime_script_sha256']!=sha256(Path(runtime.__file__)):
            raise ValueError('Frozen runtime changed')
        runtime.verify_runtime_files()
        _,rows,_,order=runtime.base.corpus();by_id={r['id']:r for r in rows};row=by_id[order[1]]
        if stage=='resident':
            model,params,opt,original,config,conversion=runtime.setup()
            loss,gradient=runtime.backward(model,params,by_id[order[0]],'U')
            report['step_one']=runtime.base.gradient_receipt(loss,gradient)
            params=opt.apply_gradients(gradient,params);mx.eval(params,opt.state)
            model.update(tree_map(lambda x:x.astype(mx.bfloat16),params));mx.eval(model.trainable_parameters())
            del loss,gradient;mx.clear_cache()
            runtime.save_restart(WORK/'restart',params,opt,
                {'arm':'U','cursor':1,'bindings':report['bindings'],'scope':'one frozen discarded localization state'})
            state=tree_map(lambda x:x,opt.state);del opt
            frozen_path=WORK/'frozen';frozen_path.mkdir()
            np.save(frozen_path/'mlx_rng.npy',np.array(mx.random.state[0]).copy())
            save_tree(frozen_path/'bf16_values_as_fp32',tree_map(lambda x:x.astype(mx.float32),model.trainable_parameters()))
            write_new(frozen_path/'input.json',{'row':row,'mask_U':runtime.base.target_mask(row,'U'),
                'mask_S':runtime.base.target_mask(row,'S')})
            teacher=runtime.base.TRAIN_CACHE/(row['id']+'.safetensors')
            frozen={'snapshot':snapshot(model,params,state),'optimizer':optimizer_config(runtime.restored_optimizer(params,state,1)),
                'input_sha256':sha256(frozen_path/'input.json'),'teacher_path':str(teacher),'teacher_sha256':sha256(teacher),
                'checkpoint_receipt_sha256':sha256(WORK/'restart/step-001/receipt.json'),'bindings':report['bindings']}
            write_new(frozen_path/'receipt.json',frozen)
            write_new(ROOT/'frozen.json',frozen)
        else:
            model,original,config,conversion=runtime.model_setup()
            params,opt=runtime.restore_training_state(model,WORK/'restart/step-001',report['bindings'],'U')
            state=tree_map(lambda x:x,opt.state);report['optimizer']=optimizer_config(opt);del opt
            frozen=json.loads((WORK/'frozen/receipt.json').read_text())
        if frozen['bindings']!=report['bindings'] or sha256(Path(frozen['teacher_path']))!=frozen['teacher_sha256']:
            raise ValueError('Frozen input binding changed')
        if sha256(WORK/'frozen/input.json')!=frozen['input_sha256'] or json.loads((WORK/'frozen/input.json').read_text())['row']!=row:
            raise ValueError('Frozen row changed')
        if sha256(WORK/'restart/step-001/receipt.json')!=frozen['checkpoint_receipt_sha256']:raise ValueError('Frozen checkpoint changed')
        reset_rng(frozen);check_snapshot(model,params,state,frozen['snapshot'])
        report['starting_state_exact']=True
        if stage in ('resident','restored-a'):
            if stage=='resident':
                runtime.memory_boundary(report,'exact-gradient before-backward')
                loss,gradient=runtime.backward(model,params,row,'U')
                report['captured_gradient']=runtime.base.gradient_receipt(loss,gradient)
                save_tree(WORK/'exact-gradient',gradient);del loss
            else:
                gradient=tree_unflatten([(key,mx.array(np.load(path))) for key,path in paths(WORK/'exact-gradient').items()])
                mx.eval(gradient)
            from mlx.utils import tree_flatten
            report['exact_gradient_verified']=all(array_digest(v)==array_digest(np.load(paths(WORK/'exact-gradient')[k])) for k,v in tree_flatten(gradient))
            if not report['exact_gradient_verified']:raise ValueError('Captured gradient changed')
            report['exact_replay']=capture_update(WORK/(stage+'-replay'),params,state,gradient)
            del gradient;mx.clear_cache();reset_rng(frozen)
            report['state_unchanged_after_replay']=check_snapshot(model,params,state,frozen['snapshot'])
        for name in (['resident-a','resident-b'] if stage=='resident' else [stage]):
            block(model,params,state,row,name,report,frozen)
        runtime.parameter_integrity(model,original,config,conversion)
        report.update(status='complete',final_state_exact=check_snapshot(model,params,state,frozen['snapshot']),
            updates=0,diagnostic_updates=11 if stage=='resident' else 6 if stage=='restored-a' else 5)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['elapsed_s']=time.monotonic()-started
        write_new(ROOT/(stage+'-receipt.json'),report)


def analyze():
    from itertools import combinations
    report={'status':'complete','historical_qualification':'failed, unchanged','limits':{'relative_gradient':.05,'relative_delta':.05,'absolute_loss':.005}}
    replay={kind:compare_arrays(paths(WORK/('restored-a-replay/'+kind)),paths(WORK/('resident-replay/'+kind)))
            for kind in ('masters','adam','delta')}
    report['identical_gradient_replay']=replay
    names=['resident-a','resident-b','restored-a','restored-b'];means={};losses={}
    for name in names:
        repeats=[WORK/name/f'repeat-{i}' for i in range(1,6)]
        means[name]={kind:mean_arrays([paths(p/kind) for p in repeats],WORK/'means'/name/kind) for kind in ('gradient','delta')}
        losses[name]=sum(json.loads((ROOT/name/f'repeat-{i}.json').read_text())['loss'] for i in range(1,6))/5
    comparisons=[]
    for a,b in combinations(names,2):
        result={'actual':a,'reference':b,'gradient':compare_arrays(means[a]['gradient'],means[b]['gradient']),
            'delta':compare_arrays(means[a]['delta'],means[b]['delta']),'absolute_mean_loss_difference':abs(losses[a]-losses[b]),'paired_repeats':[]}
        result['exceeds_existing_limits']=result['gradient']['relative_l2']>.05 or result['delta']['relative_l2']>.05 or result['absolute_mean_loss_difference']>.005
        for i in range(1,6):
            result['paired_repeats'].append({'repeat':i,**{kind:compare_arrays(paths(WORK/a/f'repeat-{i}'/kind),paths(WORK/b/f'repeat-{i}'/kind)) for kind in ('gradient','delta')}})
        comparisons.append(result)
    report['block_comparisons']=comparisons;report['mean_losses']=losses
    write_new(ROOT/'analysis.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['resident','restored-a','restored-b','analyze'])
    args=parser.parse_args()
    if args.stage=='analyze':analyze()
    else:runtime.require_runtime(before_import=True);execution(args.stage)

if __name__=='__main__':main()

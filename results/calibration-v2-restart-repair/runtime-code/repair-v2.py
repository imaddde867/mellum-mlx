"""Controlled cache-check runtime recovery for the frozen calibration-v2 comparison."""
import argparse
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
import uuid
from pathlib import Path

import calibration_v2_objective as base
from convert import sha256,realized_map
from nonexpert import binding,write_new
from refine_dwq import check_changes

ROOT=Path('results/calibration-v2-restart-repair')
INSPECTION=Path('results/calibration-v2-runtime/runtime-inspection.json')
RUNTIME={'MLX_USE_CUDA_GRAPHS':'1','MLX_CUDA_GRAPH_CACHE_SIZE':'400','MLX_ENABLE_CACHE_THRASHING_CHECK':'0'}
AMENDMENT=ROOT/'amendment.json'


def require_runtime(before_import=False):
    if any(os.environ.get(key)!=value for key,value in RUNTIME.items()):
        raise ValueError('Exact authorized cache-check environment required')
    if before_import and 'mlx.core' in sys.modules:
        raise ValueError('Runtime environment must be checked before importing MLX')
    return {**RUNTIME,'allocator_cache_limit':0,'gradient_checkpoint':'installed upstream all 28 layers',
        'mlx':importlib.metadata.version('mlx') if sys.platform=='linux' else 'unavailable on this host'}


def device_memory():
    import ctypes
    path=Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13'
    runtime=ctypes.CDLL(str(path))
    free=ctypes.c_size_t();total=ctypes.c_size_t()
    status=runtime.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total))
    if status:raise RuntimeError('cudaMemGetInfo failed: '+str(status))
    return {'interface':'cudaMemGetInfo','free_bytes':free.value,'total_bytes':total.value,
        'device_used_bytes':total.value-free.value,'measurement':'point sample; device-wide, not peak'}


def bindings():
    return {'runtime':sha256(AMENDMENT),'protocol':sha256(base.PROTOCOL),
        'data':sha256(base.ROOT/'prepared/manifest.json'),'order':sha256(base.EVIDENCE/'order.json'),
        'targets':base.check_cache(base.TRAIN_CACHE,True),'validation_targets':base.check_cache(base.VALID_CACHE),
        'source':sha256(base.STUDENT/'conversion.json'),'tokenizer':sha256(base.STUDENT/'tokenizer.json')}


def check_replacement(arms,expected,qualification,order):
    if set(arms)!={'U','S'}:raise ValueError('Missing replacement arm')
    for arm,r in arms.items():
        if r.get('bindings')!=expected or r.get('qualification_sha256')!=qualification or r.get('arm')!=arm or r.get('update_order')!=order:
            raise ValueError('Replacement arm binding mismatch')
        if r.get('status')!='complete' or r.get('updates')!=128:raise ValueError('Replacement arm incomplete')


def save_restart(root,params,opt,metadata):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    root.mkdir(parents=True,exist_ok=True)
    cursor=metadata['cursor']
    mx.eval(params,opt.state,mx.random.state)
    if opt.step.item()!=cursor:raise ValueError('Adam step/cursor mismatch')
    destination=root/f'step-{cursor:03d}'
    if destination.exists():raise ValueError('Restart step already exists')
    temporary=root/('.pending-'+uuid.uuid4().hex); temporary.mkdir()
    arrays={'parameters.safetensors':dict(tree_flatten(params)),
            'state.safetensors':dict(tree_flatten(opt.state)),
            'rng.safetensors':{'mlx_key':mx.random.state[0]}}
    report={**metadata,'optimizer_step':cursor,'python_rng':random.getstate(),'files_sha256':{},'serialization_verified':False}
    for filename,values in arrays.items():
        import numpy as np
        if not all(np.isfinite(np.array(v)).all() for v in values.values()):raise ValueError('Nonfinite restart array')
        path=temporary/filename; mx.save_safetensors(path,values)
        from safetensors import safe_open
        # Host readback processes one tensor at a time, never a second GPU state tree.
        with safe_open(path,framework='numpy') as loaded:
            if set(values)!=set(loaded.keys()):raise ValueError('Restart serialization keys changed')
            for key,value in values.items():
                host=np.array(value)
                readback=loaded.get_tensor(key)
                if host.dtype!=readback.dtype or host.shape!=readback.shape or not np.array_equal(host,readback):raise ValueError('Restart serialization changed array')
                del host,readback
        report['files_sha256'][filename]=sha256(path)
    report['serialization_verified']=True
    write_new(temporary/'receipt.json',report)
    for path in temporary.iterdir():
        with path.open('rb') as stream:os.fsync(stream.fileno())
    descriptor=os.open(temporary,os.O_RDONLY);os.fsync(descriptor);os.close(descriptor)
    os.replace(temporary,destination)
    descriptor=os.open(root,os.O_RDONLY);os.fsync(descriptor);os.close(descriptor)
    # Delete only prior verified checkpoint directories owned by this rolling store.
    checkpoints=sorted(root.glob('step-*'))
    for old in checkpoints[:-2]:
        receipt=json.loads((old/'receipt.json').read_text())
        if not receipt['serialization_verified']:raise ValueError('Refusing to prune unverified restart')
        shutil.rmtree(old)
    return report


def load_restart(path,expected,arm):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    report=json.loads((path/'receipt.json').read_text())
    if not report['serialization_verified'] or report['bindings']!=expected or report['arm']!=arm:
        raise ValueError('Restart binding mismatch')
    for filename,digest in report['files_sha256'].items():
        if sha256(path/filename)!=digest:raise ValueError('Restart file changed')
    params=tree_unflatten(mx.load(path/'parameters.safetensors'))
    state=tree_unflatten(mx.load(path/'state.safetensors'))
    if state['step'].item()!=report['cursor'] or report['optimizer_step']!=report['cursor']:raise ValueError('Restart cursor changed')
    mx.random.state[0][:]=mx.load(path/'rng.safetensors')['mlx_key']
    def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value
    random.setstate(tuples(report['python_rng']))
    mx.eval(params,state,mx.random.state)
    if random.getstate()!=tuples(report['python_rng']):raise ValueError('Python RNG restoration failed')
    with mx.stream(mx.cpu):
        saved_rng=mx.load(path/'rng.safetensors')['mlx_key']
        if not mx.all(mx.random.state[0]==saved_rng).item():raise ValueError('MLX RNG restoration failed')
    return params,state


def verify_runtime_files():
    inspection=json.loads(INSPECTION.read_text())
    if sha256(INSPECTION)!=json.loads(AMENDMENT.read_text())['inspection_sha256']:raise ValueError('Runtime inspection changed')
    for name,receipt in inspection['distributions'].items():
        dist=importlib.metadata.distribution(name)
        if dist.version!=receipt['version']:raise ValueError('Runtime package version changed')
        for file,digest in receipt['installed_files_sha256'].items():
            if not file.endswith('.so'):continue
            if sha256(Path(dist.locate_file(file)))!=digest:raise ValueError('Installed runtime file changed: '+file)


def adam():
    import mlx.optimizers as optim
    return optim.Adam(learning_rate=1e-7,betas=[.9,.999],eps=1e-8,bias_correction=True)


def model_setup():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import grad_checkpoint
    mx.set_cache_limit(0);mx.random.seed(123);random.seed(123)
    model,_,config,conversion=base.student();model.train()
    original=dict(tree_flatten(model.parameters()));mx.eval(original)
    grad_checkpoint(model.layers[0])
    return model,original,config,conversion


def setup():
    import mlx.core as mx
    from mlx.utils import tree_map
    model,original,config,conversion=model_setup()
    params=tree_map(lambda x:x.astype(mx.float32),model.trainable_parameters())
    opt=adam();opt.init(params);mx.eval(params,opt.state)
    return model,params,opt,original,config,conversion


def restored_optimizer(params,state,cursor):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    flat=dict(tree_flatten(params));moments=dict(tree_flatten(state))
    expected={'step','learning_rate'}|{key+'.'+moment for key in flat for moment in ('m','v')}
    if set(moments)!=expected:raise ValueError('Incomplete restored Adam state')
    for key,value in flat.items():
        if value.dtype!=mx.float32:raise ValueError('Restored master dtype changed')
        for moment in ('m','v'):
            v=moments[key+'.'+moment]
            if v.dtype!=mx.float32 or v.shape!=value.shape:raise ValueError('Restored Adam shape/dtype changed')
    if state['step'].dtype!=mx.uint64 or state['step'].shape!=() or state['step'].item()!=cursor:raise ValueError('Restored Adam step changed')
    if state['learning_rate'].dtype!=mx.float32 or state['learning_rate'].item()!=mx.array(1e-7,dtype=mx.float32).item():raise ValueError('Restored Adam learning rate changed')
    opt=adam();opt.state=state
    # Pinned MLX state setter clears this flag; complete moments are already validated.
    opt._initialized=True
    return opt


def restore_training_state(model,path,expected,arm):
    import mlx.core as mx
    from mlx.utils import tree_flatten,tree_map
    params,state=load_restart(path,expected,arm)
    native=dict(tree_flatten(model.trainable_parameters()));flat=dict(tree_flatten(params))
    if native.keys()!=flat.keys():raise ValueError('Restored master keys changed')
    for key,value in flat.items():
        if value.dtype!=mx.float32 or native[key].dtype!=mx.bfloat16 or value.shape!=native[key].shape:raise ValueError('Restored master schema changed')
    report=json.loads((path/'receipt.json').read_text())
    opt=restored_optimizer(params,state,report['cursor'])
    mx.eval(params,opt.state,mx.random.state)
    for _,value in tree_flatten((params,opt.state)):
        if not mx.all(mx.isfinite(value)).item():raise ValueError('Nonfinite restored state')
    model.update(tree_map(lambda x:x.astype(mx.bfloat16),params));mx.eval(model.trainable_parameters())
    return params,opt


def memory_boundary(report,phase,repeat=None):
    import mlx.core as mx
    value={'phase':phase,'repeat':repeat,'active_bytes':mx.get_active_memory(),
        'cached_bytes':mx.get_cache_memory(),'mlx_peak_bytes':mx.get_peak_memory(),
        'device_memory':device_memory(),'monotonic_s':time.monotonic()}
    report.setdefault('memory_boundaries',[]).append(value)
    report['active_boundary']={'phase':phase,'repeat':repeat}
    print(json.dumps({'memory_boundary':value}),flush=True)


def parameter_integrity(model,original,config,conversion,unchanged=False):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    changed=[]
    flat=dict(tree_flatten(model.parameters()))
    if flat.keys()!=original.keys():raise ValueError('Model parameter keys changed')
    for key,value in flat.items():
        old=original[key]
        if value.shape!=old.shape or value.dtype!=old.dtype:raise ValueError('Model schema changed')
        if value.dtype!=mx.uint32 and not mx.all(mx.isfinite(value)).item():raise ValueError('Nonfinite parameter')
        if not mx.all(value==old).item():changed.append(key)
    if unchanged and changed:raise ValueError('No-update probe changed model parameters')
    if changed:check_changes(changed,realized_map(config['quantization'],conversion['precision_map']))
    return changed


def execution_receipt(arm=None):
    return {'status':'running','runtime':require_runtime(),'bindings':bindings(),
        'script_sha256':sha256(Path(__file__)),'arm':arm,'parent':binding(base.STUDENT),
        'updates':0,'abandoned_updates':7 if arm else 0}


def backward(model,params,row,arm):
    import mlx.core as mx
    target=mx.load(base.TRAIN_CACHE/(row['id']+'.safetensors'))
    return mx.value_and_grad(lambda p:base.objective(model,p,row,target,arm))(params)


def probe(args):
    import mlx.core as mx
    _,rows,valid,order=base.corpus();by_id={r['id']:r for r in rows}
    report=execution_receipt(args.arm);report.update(backwards=[],validation={})
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:
        model,params,opt,original,config,conversion=setup()
        report['validation']['initial']=base.validation(model,valid,args.output/'initial-validation')
        model.train()
        def run(row,phase):
            mx.reset_peak_memory();begin=time.monotonic()
            loss,grads=backward(model,params,row,args.arm)
            value=base.gradient_receipt(loss,grads)
            value.update(id=row['id'],tokens=len(row['tokens']),phase=phase,elapsed_s=time.monotonic()-begin,
                mlx_peak_bytes=mx.get_peak_memory(),device_memory=device_memory(),full_context=True,optimizer_updates=0)
            report['backwards'].append(value);print(json.dumps(value),flush=True)
            del loss,grads;mx.clear_cache()
        for ident in order[:12]:run(by_id[ident],'first12')
        report['validation']['transition']=base.validation(model,valid,args.output/'transition-validation')
        model.train();run(by_id[order[12]],'subsequent');run(max(rows,key=lambda r:len(r['tokens'])),'longest')
        assert [r['id'] for r in report['backwards'][:12]]==order[:12]
        assert len(report['backwards'])==14
        parameter_integrity(model,original,config,conversion,unchanged=True)
        if opt.step.item()!=0:raise ValueError('Runtime sequence probe updated optimizer')
        report.update(status='complete',parameters_unchanged=True,resident_adam=True,
            mlx_peak_bytes=mx.get_peak_memory())
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['elapsed_s']=time.monotonic()-started;write_new(args.output/'receipt.json',report)


def host_accumulate(total,key,value):
    import numpy as np
    value=np.asarray(value,dtype=np.float32)
    if key not in total:total[key]=value.copy()
    else:np.add(total[key],value,out=total[key])


def host_relative(values,reference):
    import numpy as np
    if set(values)!=set(reference):raise ValueError('Restart comparison keys changed')
    difference=0.;norm=0.
    for key,value in values.items():
        other=np.load(reference[key],mmap_mode='r') if isinstance(reference[key],Path) else reference[key]
        if value.shape!=other.shape or value.dtype!=other.dtype:raise ValueError('Reference schema changed')
        difference+=np.square(value-other,dtype=np.float32).sum(dtype=np.float64).item()
        norm+=np.square(other,dtype=np.float32).sum(dtype=np.float64).item()
    return (difference/max(norm,1e-30))**.5


def restart_samples(model,params,state,row,arm,report):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_map,tree_flatten
    samples=[];mean_gradient={};mean_delta={}
    rng=mx.array(mx.random.state[0]);mx.eval(rng)
    for repeat in range(1,6):
        mx.random.state[0][:]=rng
        memory_boundary(report,'before-backward',repeat)
        loss,grads=backward(model,params,row,arm)
        gradient=base.gradient_receipt(loss,grads)
        memory_boundary(report,'after-backward',repeat)
        for key,value in tree_flatten(grads):host_accumulate(mean_gradient,key,np.array(value))
        opt=restored_optimizer(params,tree_map(lambda x:x,state),1)
        memory_boundary(report,'before-update',repeat)
        updated=opt.apply_gradients(grads,params);mx.eval(updated,opt.state)
        if opt.step.item()!=2:raise ValueError('Restored optimizer next step mismatch')
        memory_boundary(report,'after-update',repeat)
        flat=dict(tree_flatten(params))
        for key,value in tree_flatten(updated):host_accumulate(mean_delta,key,np.array(value)-np.array(flat[key]))
        samples.append(gradient)
        del grads,loss,updated,opt,value,flat;mx.clear_cache()
        memory_boundary(report,'after-release-diagnostic-temporaries',repeat)
    for total in (mean_gradient,mean_delta):
        for value in total.values():np.divide(value,np.float32(5),out=value)
    return samples,mean_gradient,mean_delta


def restart_produce(args):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_map
    _,rows,_,order=base.corpus();by_id={r['id']:r for r in rows}
    args.output.mkdir(parents=True,exist_ok=False);report=execution_receipt(args.arm);started=time.monotonic()
    try:
        model,original,config,conversion=model_setup();memory_boundary(report,'after-model-only-load')
        params=tree_map(lambda x:x.astype(mx.float32),model.trainable_parameters())
        opt=adam();opt.init(params);mx.eval(params,opt.state);memory_boundary(report,'after-fresh-state-materialization')
        loss,grads=backward(model,params,by_id[order[0]],args.arm);first=base.gradient_receipt(loss,grads)
        params=opt.apply_gradients(grads,params);mx.eval(params,opt.state)
        model.update(tree_map(lambda x:x.astype(mx.bfloat16),params));mx.eval(model.trainable_parameters())
        del loss,grads;mx.clear_cache()
        report['checkpoint']=save_restart(args.output/'restart',params,opt,
            {'arm':args.arm,'cursor':1,'bindings':report['bindings'],'scope':'discarded restart verification'})
        memory_boundary(report,'after-checkpoint-readback')
        samples,gradient,delta=restart_samples(model,params,opt.state,by_id[order[1]],args.arm,report)
        reference=args.output/'reference';reference.mkdir();hashes={}
        for prefix,total in [('gradient.',gradient),('delta.',delta)]:
            for key,value in total.items():
                path=reference/(prefix+key+'.npy');np.save(path,value);hashes[path.name]=sha256(path)
        report.update(status='complete',first=first,next_update_samples=samples,
            reference_sha256=hashes,updates=0,discarded_verification_updates=6,
            not_a_training_arm=True,mlx_peak_bytes=mx.get_peak_memory(),
            reduction='elementwise sequential FP32 host sums/means and deltas; per-tensor squared errors FP32, scalar sums FP64 in tensor order')
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:report['elapsed_s']=time.monotonic()-started;write_new(args.output/'receipt.json',report)


def restart_restore(args):
    parent=args.checkpoint;reference_receipt=json.loads((parent/'receipt.json').read_text())
    report=execution_receipt(args.arm);args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:
        model,original,config,conversion=model_setup();memory_boundary(report,'after-model-only-load')
        params,opt=restore_training_state(model,parent/'restart/step-001',report['bindings'],args.arm)
        memory_boundary(report,'after-restored-state-materialization')
        _,rows,_,order=base.corpus();row=next(r for r in rows if r['id']==order[1])
        samples,gradient,delta=restart_samples(model,params,opt.state,row,args.arm,report)
        reference=parent/'reference';paths={}
        for filename,digest in reference_receipt['reference_sha256'].items():
            if sha256(reference/filename)!=digest:raise ValueError('Restart reference changed')
            paths[filename]=reference/filename
        grad_error=host_relative(gradient,{k[len('gradient.'):-4]:p for k,p in paths.items() if k.startswith('gradient.')})
        delta_error=host_relative(delta,{k[len('delta.'):-4]:p for k,p in paths.items() if k.startswith('delta.')})
        loss_error=abs(sum(v['loss'] for v in samples)/5-sum(v['loss'] for v in reference_receipt['next_update_samples'])/5)
        report.update(next_update_samples=samples,relative_mean_gradient_l2=grad_error,
            relative_mean_update_delta_l2=delta_error,absolute_mean_loss_difference=loss_error)
        if grad_error>.05 or delta_error>.05 or loss_error>.005:raise ValueError('Restored next-update tolerance failed')
        changed=parameter_integrity(model,original,config,conversion)
        report.update(status='complete',updates=0,discarded_verification_updates=5,
            restored_optimizer_step=1,next_optimizer_step=2,protected_parameters_unchanged=True,
            changed_parameters=changed,not_a_training_arm=True,restored_dtype_shape_cursor_rng_verified=True)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:report['elapsed_s']=time.monotonic()-started;write_new(args.output/'receipt.json',report)


def check_qualification(receipt,expected,runtime):
    if receipt.get('status')!='complete' or receipt.get('bindings')!=expected or receipt.get('runtime')!=runtime or receipt.get('updates')!=0:
        raise ValueError('Training lacks matching successful runtime qualification')


def resume_cursor(receipt,order,qualification):
    cursor=receipt['cursor']
    if not isinstance(cursor,int) or cursor<1 or cursor>len(order) or receipt.get('update_order')!=order[:cursor] or receipt.get('qualification_sha256')!=qualification:
        raise ValueError('Restart order, budget or qualification mismatch')
    return cursor


def train_arm(args):
    import mlx.core as mx
    from mlx.utils import tree_map,tree_flatten
    expected=bindings();qualification_path=ROOT/'qualification.json'
    qualified=json.loads(qualification_path.read_text())
    check_qualification(qualified,expected,require_runtime())
    paths=json.loads(AMENDMENT.read_text())['replacement_arms']
    if args.output!=Path(paths[args.arm]):raise ValueError('Replacement training path mismatch')
    _,rows,valid,order=base.corpus();by_id={r['id']:r for r in rows}
    report=execution_receipt(args.arm)
    report.update(qualification_sha256=sha256(qualification_path),evaluations={},update_order=[])
    resume=args.checkpoint
    if resume is None:args.output.mkdir(parents=True,exist_ok=False)
    elif resume.parent!=args.output/'restart':raise ValueError('Restart is outside the arm rolling store')
    started=time.monotonic()
    try:
        if resume is None:
            model,params,opt,original,config,conversion=setup();cursor=0
        else:
            saved=json.loads((resume/'receipt.json').read_text())
            cursor=resume_cursor(saved,order,report['qualification_sha256'])
            model,original,config,conversion=model_setup()
            params,opt=restore_training_state(model,resume,expected,args.arm)
            report.update(updates=cursor,update_order=list(saved['update_order']),evaluations=saved['evaluations'],
                resumed_checkpoint=str(resume),resumed_checkpoint_sha256=sha256(resume/'receipt.json'))
        report['eligible_keys']=sorted(k for k,_ in tree_flatten(params))
        if cursor==0:report['evaluations']['0']=base.validation(model,valid,args.output/'step-0')
        elif cursor in (8,32,64,96,128) and str(cursor) not in report['evaluations']:
            slot=args.output/f'step-{cursor}-parameters.safetensors'
            if not slot.exists():mx.save_safetensors(slot,dict(tree_flatten(params)))
            report['evaluations'][str(cursor)]=base.validation(model,valid,args.output/f'step-{cursor}-resume-{uuid.uuid4().hex}')
        model.train()
        for step,ident in enumerate(order[cursor:],cursor+1):
            begin=time.monotonic();row=by_id[ident]
            loss,grads=backward(model,params,row,args.arm);gradient=base.gradient_receipt(loss,grads)
            params=opt.apply_gradients(grads,params);mx.eval(params,opt.state)
            model.update(tree_map(lambda x:x.astype(mx.bfloat16),params));mx.eval(model.trainable_parameters())
            report['updates']=step;report['update_order'].append(ident)
            del grads,loss;mx.clear_cache()
            # Publish a verified restart before another backward or validation transition.
            checkpoint=save_restart(args.output/'restart',params,opt,
                {'arm':args.arm,'cursor':step,'bindings':expected,'qualification_sha256':report['qualification_sha256'],
                 'update_order':list(report['update_order']),'evaluations':dict(report['evaluations'])})
            receipt={'id':ident,'step':step,'arm':args.arm,**gradient,
                'included_targets':sum(base.target_mask(row,args.arm)),
                'elapsed_s':time.monotonic()-begin,'mlx_peak_bytes':mx.get_peak_memory(),
                'device_memory':device_memory(),'restart':checkpoint['files_sha256']}
            write_new(args.output/'updates'/f'{step:03d}.json',receipt);print(json.dumps(receipt),flush=True)
            if step in (8,32,64,96,128):
                mx.save_safetensors(args.output/f'step-{step}-parameters.safetensors',dict(tree_flatten(params)))
                report['evaluations'][str(step)]=base.validation(model,valid,args.output/f'step-{step}');model.train()
        changed=parameter_integrity(model,original,config,conversion)
        report.update(status='complete',changed_parameters=changed,protected_parameters_unchanged=True,
            checkpoint_state_sha256={p.name:sha256(p) for p in args.output.glob('*-parameters.safetensors')})
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['elapsed_s']=time.monotonic()-started
        execution=args.output/'executions'/('run-'+uuid.uuid4().hex+'.json')
        write_new(execution,report)
        temporary=args.output/('.receipt-'+uuid.uuid4().hex+'.json')
        write_new(temporary,report);os.replace(temporary,args.output/'receipt.json')


def select(args):
    paths={'U':args.u,'S':args.s};declared=json.loads(AMENDMENT.read_text())['replacement_arms']
    if any(path!=Path(declared[arm]) for arm,path in paths.items()):raise ValueError('Selection path mismatch')
    arms={a:json.loads((p/'receipt.json').read_text()) for a,p in paths.items()}
    _,_,_,order=base.corpus();expected=bindings();qualification_path=ROOT/'qualification.json'
    check_qualification(json.loads(qualification_path.read_text()),expected,require_runtime())
    check_replacement(arms,expected,sha256(qualification_path),order)
    if arms['U']['parent']['weights_sha256']!=arms['S']['parent']['weights_sha256']:raise ValueError('Pristine parents differ')
    decision=base.experiment_decision(arms)
    decision.update(bindings=expected,qualification_sha256=sha256(qualification_path),
        arm_paths={a:str(p) for a,p in paths.items()},
        arm_receipt_sha256={a:sha256(p/'receipt.json') for a,p in paths.items()})
    write_new(args.output,decision)


def qualification(args):
    paths={'correctness':Path('results/calibration-v2-cachecheck/correctness/receipt.json'),**{f'probe-{a}':ROOT/f'probe-{a}/receipt.json' for a in ('U','S')},
        **{f'restart-{a}':ROOT/f'restart-restore-{a}/receipt.json' for a in ('U','S')}}
    receipts={k:json.loads(p.read_text()) for k,p in paths.items()}
    if any(v['status']!='complete' for v in receipts.values()):raise ValueError('Native workaround qualification failed')
    expected=bindings()
    for key,r in receipts.items():
        comparison={**r['bindings']}
        if key=='correctness':
            if comparison['runtime']!=json.loads(AMENDMENT.read_text())['parent_amendment_sha256']:raise ValueError('Prior correctness amendment changed')
            comparison['runtime']=expected['runtime']
        if comparison!=expected or r['runtime']!=require_runtime():raise ValueError('Qualification execution mismatch')
    if receipts['correctness']['protocol_sha256']!=expected['protocol']:raise ValueError('Correctness protocol changed')
    write_new(args.output,{'status':'complete','bindings':expected,'runtime':require_runtime(),
        'component_sha256':{k:sha256(p) for k,p in paths.items()},'updates':0,
        'original_correctness_receipt_sha256':sha256(Path('results/calibration-v2-cachecheck/correctness/detail/receipt.json'))})



def correctness(args,report):
    started=time.monotonic()
    original_gradient_receipt=base.gradient_receipt
    report['device_memory_samples']=[]
    def measured(loss,grads):
        result=original_gradient_receipt(loss,grads)
        sample=device_memory();sample['elapsed_s']=time.monotonic()-started
        report['device_memory_samples'].append(sample)
        print(json.dumps({'device_memory':sample}),flush=True)
        return result
    base.gradient_receipt=measured
    try:
        base.preflight(argparse.Namespace(output=args.output/'detail'))
        report['status']='complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        # Metadata only after an error; never evaluate or serialize the CUDA state.
        base.gradient_receipt=original_gradient_receipt
        report['elapsed_s']=time.monotonic()-started
        write_new(args.output/'receipt.json',report)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['correctness','probe','restart-produce','restart-restore','qualification','train','select'])
    parser.add_argument('--arm',choices=['U','S']);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--u',type=Path);parser.add_argument('--s',type=Path)
    args=parser.parse_args();require_runtime(before_import=True);verify_runtime_files()
    if args.output.exists() and not (args.stage=='train' and args.checkpoint is not None):parser.error('Refusing existing evidence directory')
    if args.stage=='correctness':
        report=execution_receipt();report['protocol_sha256']=sha256(base.PROTOCOL)
        correctness(args,report)
    elif args.stage=='qualification':qualification(args)
    elif args.stage=='select':
        if args.u is None or args.s is None:parser.error('Explicit new arm paths required')
        select(args)
    elif args.stage=='train':
        if args.arm is None:parser.error('Arm required')
        train_arm(args)
    elif args.arm is None:parser.error('Arm required')
    elif args.stage=='probe':probe(args)
    elif args.stage=='restart-produce':restart_produce(args)
    elif args.checkpoint is None:parser.error('Restart producer directory required')
    else:restart_restore(args)

if __name__=='__main__':main()

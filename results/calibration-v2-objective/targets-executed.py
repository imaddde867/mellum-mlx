"""One full-context, matched U/S DWQ pilot on the frozen calibration-v2 corpus."""
import argparse
import json
import math
import shutil
import statistics
import time
from pathlib import Path

from calibration_v2 import ROOT, REGIONS, frozen_validation, reduce_captured
from convert import REVISION, realized_map, sha256
from nonexpert import binding, write_new
from refine_dwq import check_changes, require_pristine_c
from validate import check_hashes

EVIDENCE = Path('results/calibration-v2-objective')
PROTOCOL = EVIDENCE/'predeclared.json'
ASSISTANT = ('reasoning','transition','final','tool_call')
TRAIN_CACHE = Path('work/calibration-v2-training-targets')
VALID_CACHE = Path('work/calibration-v2-teacher')
STUDENT = Path('artifacts/mellum2.1-mixed-C-g64')


def target_mask(row, arm, padded_targets=None):
    count = len(row['tokens'])-1
    labels = row['regions']
    if arm not in {'U','S'} or len(labels) != count or any(r not in REGIONS for r in labels):
        raise ValueError('Invalid arm or next-token region alignment')
    size = count if padded_targets is None else padded_targets
    if size < count:
        raise ValueError('Mask would truncate the trajectory')
    mask = [arm=='U' or label in ASSISTANT for label in labels] + [False]*(size-count)
    if not any(mask):
        raise ValueError('Zero included prediction targets')
    return mask


def fp32_kl(student, teacher):
    import mlx.core as mx
    from mlx_lm.tuner.losses import kl_div_loss
    return kl_div_loss(.5*student.astype(mx.float32), .5*teacher.astype(mx.float32))


def masked_mean(losses, mask):
    import mlx.core as mx
    if losses.shape != mask.shape:
        raise ValueError('Loss/mask shape mismatch')
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError('Zero included prediction targets')
    return mx.where(mask,losses.astype(mx.float32),mx.zeros_like(losses,dtype=mx.float32)).sum()/count


def fidelity_gate(baseline, candidate):
    keys = ('assistant',*ASSISTANT)
    for values in (baseline,candidate):
        if len(values) != 5 or any(not math.isfinite(v[k]) for v in values for k in keys):
            raise ValueError('Incomplete or nonfinite repeated validation')
    def resolved(lower, higher, key):
        left,right = [v[key] for v in lower],[v[key] for v in higher]
        return max(left)<min(right) and statistics.mean(right)-statistics.mean(left)>max(
            max(left)-min(left),max(right)-min(right))
    return resolved(candidate,baseline,'assistant') and not any(
        resolved(baseline,candidate,key) for key in ASSISTANT)


def corpus():
    manifest, valid = frozen_validation(ROOT/'prepared')
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(ROOT/'prepared/manifest.json') != protocol['data_manifest_sha256'] or sha256(EVIDENCE/'order.json') != protocol['order_sha256']:
        raise ValueError('Frozen experiment inputs changed')
    path = ROOT/'prepared/train.jsonl'
    if sha256(path) != manifest['splits']['train']['sha256']:
        raise ValueError('Training split changed')
    train = [json.loads(line) for line in path.read_text().splitlines()]
    order = json.loads((EVIDENCE/'order.json').read_text())['ids']
    if len(train)!=128 or len(order)!=128 or len(set(order))!=128 or set(order)!={r['id'] for r in train}:
        raise ValueError('Incomplete training set/order')
    for row in train+valid:
        target_mask(row,'U'); target_mask(row,'S')
    return manifest,train,valid,order


def training_targets(args):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    manifest,train,_,_ = corpus()
    source = Path('work/source')
    check_hashes(source,manifest['source_sha256'])
    report = {**binding(source),'script_sha256':sha256(Path(__file__)),
        'protocol_sha256':sha256(PROTOCOL),'data_manifest_sha256':sha256(ROOT/'prepared/manifest.json'),
        'training_split_sha256':sha256(ROOT/'prepared/train.jsonl'),
        'tokenizer_sha256':sha256(source/'tokenizer.json'),'source_revision':REVISION,
        'temperature':2,'top_k':1024,'chunk_size':128,'updates':0,'status':'running','files_sha256':{}}
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    try:
        mx.random.seed(123)
        model,tokenizer = load(str(source)); model.eval()
        for row in train:
            if tokenizer.encode(row['text'],add_special_tokens=False)!=row['tokens']:
                raise ValueError('Training tokenizer mismatch')
            cache = make_prompt_cache(model); values,indices = [],[]
            for start in range(0,len(row['tokens'])-1,128):
                x = mx.array([row['tokens'][start:min(start+128,len(row['tokens'])-1)]])
                logits = model(x,cache=cache)
                idx = mx.argpartition(logits,kth=-1024,axis=-1)[...,-1024:]
                value = mx.take_along_axis(logits,idx,axis=-1)
                mx.eval(value,idx); values.append(value); indices.append(idx)
                del logits
            targets = {'logits':mx.concatenate(values,axis=1),'indices':mx.concatenate(indices,axis=1)}
            mx.eval(targets)
            if targets['logits'].shape[1]!=len(row['regions']) or not mx.all(mx.isfinite(targets['logits'])).item():
                raise ValueError('Invalid complete teacher targets')
            path = args.output/(row['id']+'.safetensors')
            mx.save_safetensors(str(path),targets)
            report['files_sha256'][path.name]=sha256(path)
            print(json.dumps({'teacher':row['id'],'tokens':len(row['tokens']),'peak_bytes':mx.get_peak_memory()}),flush=True)
            del cache,values,indices,targets,value,idx,x; mx.clear_cache()
        report['status']='complete'
    except Exception as error:
        report.update(status='failed',error=repr(error)); raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,mlx_peak_bytes=mx.get_peak_memory())
        write_new(args.output/'receipt.json',report)


def check_cache(path, train=False):
    receipt = json.loads((path/'receipt.json').read_text())
    if receipt['status']!='complete' or receipt['data_manifest_sha256']!=sha256(ROOT/'prepared/manifest.json'):
        raise ValueError('Teacher cache context mismatch')
    if train and (receipt['protocol_sha256']!=sha256(PROTOCOL) or receipt['training_split_sha256']!=sha256(ROOT/'prepared/train.jsonl')):
        raise ValueError('Training cache protocol mismatch')
    check_hashes(path,receipt['files_sha256'] if train else receipt['captures_sha256'])
    return sha256(path/'receipt.json')


def student(path=STUDENT, pristine=True):
    import mlx.core as mx
    from mlx_lm import load
    conversion = json.loads((path/'conversion.json').read_text())
    if pristine:
        require_pristine_c(conversion)
        check_hashes(path,json.loads(Path('results/nonexpert-v1/integrity-C.json').read_text())['weights_sha256'])
    check_hashes(path,conversion['artifact_sha256'])
    model,tokenizer,config = load(str(path),return_config=True)
    model.freeze()
    def unfreeze(_, module):
        if hasattr(module,'bits') and hasattr(module,'group_size') and module.mode=='affine' and module.bits<8:
            module.unfreeze(keys=['scales','biases'],recurse=False)
    model.apply_to_modules(unfreeze)
    if sha256(path/'tokenizer.json')!=json.loads((ROOT/'prepared/manifest.json').read_text())['tokenizer_sha256']:
        raise ValueError('Student tokenizer changed')
    mx.eval(model.parameters())
    return model,tokenizer,config,conversion


def objective(model, params, row, teacher, arm):
    import mlx.core as mx
    from mlx.utils import tree_map
    model.update(tree_map(lambda x:x.astype(mx.bfloat16),params))
    # Whole-trajectory backpropagation: no cache, sequence clipping or detached history.
    logits = model(mx.array([row['tokens'][:-1]]))
    gathered = mx.take_along_axis(logits,teacher['indices'],axis=-1)
    mask = mx.array([target_mask(row,arm)])
    return masked_mean(fp32_kl(gathered,teacher['logits']),mask)


def gradient_receipt(loss, grads):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(grads)); mx.eval(loss,flat)
    finite = mx.stack([mx.all(mx.isfinite(v)) for v in flat.values()])
    square = sum(mx.square(v.astype(mx.float32)).sum() for v in flat.values())
    mx.eval(finite,square)
    if not mx.all(finite).item() or not math.isfinite(loss.item()) or square.item()<=0:
        raise ValueError('Nonfinite or zero gradients/loss')
    return {'loss':loss.item(),'global_l2':math.sqrt(square.item()),'finite':True,
            'arrays':len(flat),'bytes':sum(v.nbytes for v in flat.values())}


def preflight(args):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_map,tree_flatten
    from mlx_lm.tuner.trainer import grad_checkpoint
    _,train,_,_ = corpus(); check_cache(TRAIN_CACHE,True)
    report = {'status':'running','updates':0,'script_sha256':sha256(Path(__file__)),
        'protocol_sha256':sha256(PROTOCOL),'cases':[],'parent':binding(STUDENT),
        'training_cache_sha256':sha256(TRAIN_CACHE/'receipt.json')}
    args.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    try:
        mx.random.seed(123)
        model,_,_,_ = student(); model.train()
        original=dict(tree_flatten(model.parameters())); mx.eval(original)
        params=tree_map(lambda x:x.astype(mx.float32),model.trainable_parameters())
        shortest=min(train,key=lambda r:len(r['tokens']))
        targets=mx.load(TRAIN_CACHE/(shortest['id']+'.safetensors'))
        fn=mx.value_and_grad(lambda p:objective(model,p,shortest,targets,'U'))
        loss,before=fn(params); before_report=gradient_receipt(loss,before)
        grad_checkpoint(model.layers[0])
        loss,after=fn(params); after_report=gradient_receipt(loss,after)
        difference=sum(mx.square(a-b).sum() for (_,a),(_,b) in zip(tree_flatten(before),tree_flatten(after)))
        mx.eval(difference)
        relative=math.sqrt(difference.item())/max(before_report['global_l2'],1e-12)
        report['checkpoint_equivalence']={'trajectory':shortest['id'],'tokens':len(shortest['tokens']),
            'unwrapped':before_report,'checkpointed':after_report,'relative_gradient_l2_difference':relative}
        write_new(args.output/'checkpoint-equivalence.json',report['checkpoint_equivalence'])
        if relative>.05 or abs(before_report['loss']-after_report['loss'])>.005:
            raise ValueError('Checkpoint gradient equivalence failed')
        del before,after,loss,targets,fn,difference; mx.clear_cache()
        opt=optim.Adam(learning_rate=1e-7,betas=[.9,.999],eps=1e-8,bias_correction=True)
        opt.init(params)
        mx.eval(params,opt.state)
        longest=max(train,key=lambda r:len(r['tokens']))
        for arm in ('U','S'):
            targets=mx.load(TRAIN_CACHE/(longest['id']+'.safetensors'))
            fn=mx.value_and_grad(lambda p:objective(model,p,longest,targets,arm))
            mx.reset_peak_memory()
            loss,grads=fn(params)
            result=gradient_receipt(loss,grads)
            result.update(arm=arm,id=longest['id'],tokens=len(longest['tokens']),
                scored_tokens=sum(target_mask(longest,arm)),mlx_peak_bytes=mx.get_peak_memory(),
                full_context=True,detached_cache=False,optimizer_updates=0)
            report['cases'].append(result); write_new(args.output/(arm+'-longest.json'),result)
            print(json.dumps(result),flush=True)
            del grads,loss,targets,fn; mx.clear_cache()
        report.update(status='complete',gradient_checkpoint=True,
            resident_optimizer_state=True,checkpoint_states_on_disk=True,parameters_unchanged=True)
        # No optimizer updates occurred; the BF16 model values must still equal their origin.
        for key,value in tree_flatten(model.parameters()):
            if not mx.all(value==original[key]).item():raise ValueError('Preflight changed model parameter')
    except Exception as error:
        report.update(status='failed',error=repr(error)); raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,mlx_peak_bytes=mx.get_peak_memory())
        write_new(args.output/'receipt.json',report)


def validation(model, valid, output):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    model.eval(); repeats=[]
    for repeat in range(5):
        rows=[]
        for row in valid:
            target=mx.load(VALID_CACHE/(row['id']+'.safetensors'))
            cache=make_prompt_cache(model); values=[]
            for start in range(0,len(row['tokens'])-1,128):
                stop=min(start+128,len(row['tokens'])-1)
                logits=model(mx.array([row['tokens'][start:stop]]),cache=cache)
                value=mx.take_along_axis(logits,target['indices'][:,start:stop],axis=-1)
                mx.eval(value); values.append(value); del logits
            sums=reduce_captured(mx.concatenate(values,axis=1),target['logits'],row['regions'])['fp32']
            sums['assistant']=[sum(sums[r][0] for r in ASSISTANT),sum(sums[r][1] for r in ASSISTANT)]
            rows.append({'id':row['id'],'family':row['family'],'sums':sums})
            del target,cache,values,value; mx.clear_cache()
        totals={region:{'sum':sum(r['sums'][region][0] for r in rows),
            'tokens':sum(r['sums'][region][1] for r in rows)} for region in (*REGIONS,'aggregate','assistant')}
        means={k:v['sum']/v['tokens'] for k,v in totals.items()}
        families={}
        for family in sorted({r['family'] for r in rows}):
            subset=[r for r in rows if r['family']==family]
            families[family]={region:sum(r['sums'][region][0] for r in subset)/sum(r['sums'][region][1] for r in subset)
                for region in totals}
        result={'repeat':repeat,'means':means,'totals':totals,'families':families,'rows':rows}
        repeats.append(result); write_new(output/f'repeat-{repeat}.json',result)
        print(json.dumps({'validation':str(output),'repeat':repeat,'means':means}),flush=True)
    return repeats


def train_arm(args):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_map,tree_flatten
    from mlx_lm.tuner.trainer import grad_checkpoint
    _,train,valid,order=corpus()
    preflight_report=json.loads((EVIDENCE/'preflight/receipt.json').read_text())
    if preflight_report['status']!='complete' or preflight_report['protocol_sha256']!=sha256(PROTOCOL):
        raise ValueError('Training lacks successful full-backward preflight')
    train_cache=check_cache(TRAIN_CACHE,True); valid_cache=check_cache(VALID_CACHE)
    report={**binding(STUDENT),'script_sha256':sha256(Path(__file__)),
        'protocol_sha256':sha256(PROTOCOL),'training_cache_sha256':train_cache,
        'validation_cache_sha256':valid_cache,'order_sha256':sha256(EVIDENCE/'order.json'),
        'status':'running','arm':args.arm,'updates':0,'evaluations':{},'update_order':[]}
    args.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    try:
        mx.random.seed(123)
        model,tokenizer,config,conversion=student(); model.train()
        original=dict(tree_flatten(model.parameters())); mx.eval(original)
        plan=realized_map(config['quantization'],conversion['precision_map'])
        grad_checkpoint(model.layers[0])
        params=tree_map(lambda x:x.astype(mx.float32),model.trainable_parameters())
        mx.eval(params)
        report['eligible_keys']=sorted(k for k,_ in tree_flatten(params))
        report['evaluations']['0']=validation(model,valid,args.output/'step-0')
        model.train()
        opt=optim.Adam(learning_rate=1e-7,betas=[.9,.999],eps=1e-8,bias_correction=True)
        by_id={r['id']:r for r in train}
        for step,ident in enumerate(order,1):
            row=by_id[ident]; teacher=mx.load(TRAIN_CACHE/(ident+'.safetensors'))
            loss,grads=mx.value_and_grad(lambda p:objective(model,p,row,teacher,args.arm))(params)
            gradient=gradient_receipt(loss,grads)
            params=opt.apply_gradients(grads,params); mx.eval(params,opt.state)
            model.update(tree_map(lambda x:x.astype(mx.bfloat16),params)); mx.eval(model.trainable_parameters())
            report['updates']=step; report['update_order'].append(ident)
            write_new(args.output/'updates'/f'{step:03d}.json',{'id':ident,'step':step,'arm':args.arm,
                'included_targets':sum(target_mask(row,args.arm)),**gradient,'mlx_peak_bytes':mx.get_peak_memory()})
            print(json.dumps({'arm':args.arm,'step':step,'id':ident,**gradient,'peak_bytes':mx.get_peak_memory()}),flush=True)
            del grads,loss,teacher; mx.clear_cache()
            if step in (8,32,64,96,128):
                # Retain only trainable state, not a new full model per checkpoint.
                path=args.output/f'step-{step}-parameters.safetensors'
                mx.save_safetensors(str(path),dict(tree_flatten(params)))
                report['evaluations'][str(step)]=validation(model,valid,args.output/f'step-{step}')
                model.train()
        changed=[]
        for key,value in tree_flatten(model.parameters()):
            old=original[key]
            if value.shape!=old.shape or value.dtype!=old.dtype:raise ValueError('Parameter schema changed')
            if value.dtype!=mx.uint32 and not mx.all(mx.isfinite(value)).item():raise ValueError('Nonfinite model parameter')
            if not mx.all(value==old).item():changed.append(key)
        if changed:check_changes(changed,plan)
        report.update(status='complete',changed_parameters=changed,protected_parameters_unchanged=True,
            checkpoint_state_sha256={p.name:sha256(p) for p in args.output.glob('*-parameters.safetensors')})
    except Exception as error:
        report.update(status='failed',error=repr(error)); raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,mlx_peak_bytes=mx.get_peak_memory())
        write_new(args.output/'receipt.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['targets','preflight','train'])
    p.add_argument('--arm',choices=['U','S'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing existing evidence/cache directory')
    if args.stage=='targets':training_targets(args)
    elif args.stage=='preflight':preflight(args)
    elif args.arm is None:p.error('Training requires arm U or S')
    else:train_arm(args)


if __name__=='__main__':main()

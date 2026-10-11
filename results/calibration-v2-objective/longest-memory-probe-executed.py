import json,sys,time
from pathlib import Path
sys.path.insert(0,'scripts')
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map,tree_flatten
from mlx_lm.tuner.trainer import grad_checkpoint
from calibration_v2_objective import corpus,student,objective,gradient_receipt,TRAIN_CACHE,PROTOCOL
from convert import sha256
from nonexpert import write_new
_,rows,_,_=corpus(); row=max(rows,key=lambda r:len(r['tokens']))
root=Path('results/calibration-v2-objective/longest-memory-probe'); root.mkdir(exist_ok=False)
report={'status':'running','updates':0,'trajectory':row['id'],'tokens':len(row['tokens']),
        'protocol_sha256':sha256(PROTOCOL),'script_sha256':sha256(Path(__file__)),
        'checkpoint_equivalence':'separate first preflight failed; this isolates full-backward memory only',
        'full_context':True,'training_cache':'none','cases':[]}
started=time.monotonic()
try:
    mx.random.seed(123)
    model,_,_,_=student(); model.train(); grad_checkpoint(model.layers[0])
    original=dict(tree_flatten(model.parameters())); mx.eval(original)
    params=tree_map(lambda x:x.astype(mx.float32),model.trainable_parameters())
    opt=optim.Adam(learning_rate=1e-7,betas=[.9,.999],eps=1e-8,bias_correction=True)
    opt.init(params); mx.eval(params,opt.state)
    for arm in ['U','S']:
        report['active_arm']=arm
        print(json.dumps({'phase':'whole trajectory backward','arm':arm,'id':row['id'],'tokens':len(row['tokens'])}),flush=True)
        target=mx.load(TRAIN_CACHE/(row['id']+'.safetensors'))
        mx.reset_peak_memory()
        loss,grads=mx.value_and_grad(lambda p:objective(model,p,row,target,arm))(params)
        result=gradient_receipt(loss,grads)
        result.update(arm=arm,mlx_peak_bytes=mx.get_peak_memory(),resident_optimizer_state=True)
        write_new(root/(arm+'.json'),result); report['cases'].append(result)
        print(json.dumps(result),flush=True)
        del loss,grads,target; mx.clear_cache()
    report['status']='complete'
except BaseException as error:
    report.update(status='failed',error=repr(error)); raise
finally:
    report.update(elapsed_s=time.monotonic()-started,mlx_peak_bytes=mx.get_peak_memory())
    write_new(root/'receipt.json',report)

import json,math,sys
from pathlib import Path
sys.path.insert(0,'scripts')
import mlx.core as mx
from mlx.utils import tree_flatten,tree_map
from mlx_lm.tuner.trainer import grad_checkpoint
from calibration_v2_objective import corpus,student,objective,gradient_receipt,TRAIN_CACHE
from nonexpert import write_new
mx.random.seed(123)
_,rows,_,_=corpus(); row=min(rows,key=lambda r:len(r['tokens']))
model,_,_,_=student(); model.train()
original=dict(tree_flatten(model.parameters())); mx.eval(original)
params=tree_map(lambda x:x.astype(mx.float32),model.trainable_parameters())
target=mx.load(TRAIN_CACHE/(row['id']+'.safetensors'))
fn=mx.value_and_grad(lambda p:objective(model,p,row,target,'U'))
records=[]; gradients=[]
for mode in ('unwrapped','checkpointed'):
    if mode=='checkpointed':grad_checkpoint(model.layers[0])
    for repeat in range(3):
        loss,grad=fn(params)
        record=gradient_receipt(loss,grad)
        flat=dict(tree_flatten(grad)); gradients.append(flat)
        changed=[k for k,v in tree_flatten(model.parameters()) if not mx.all(v==original[k]).item()]
        record.update(mode=mode,repeat=repeat,changed_parameters=changed)
        print(json.dumps(record),flush=True); records.append(record)
        del grad,loss; mx.clear_cache()
comparisons=[]
for i in range(6):
    for j in range(i):
        a,b=gradients[i],gradients[j]
        assert set(a)==set(b)
        diff=sum(mx.square(a[k]-b[k]).sum() for k in a)
        mx.eval(diff)
        comparisons.append({'i':i,'j':j,'relative_gradient_l2_difference':math.sqrt(diff.item())/records[j]['global_l2'],
            'loss_delta':abs(records[i]['loss']-records[j]['loss'])})
write_new(Path('results/calibration-v2-objective/checkpoint-probe.json'),{'row':row['id'],'records':records,'comparisons':comparisons,'updates':0})

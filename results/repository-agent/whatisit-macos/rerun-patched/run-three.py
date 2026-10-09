import json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('work/routing-rerun')
summary={'task':'mutation-routing','seeds':[0,1,2],'temperature':1.0,'max_tokens':16384,'reasoning_replay':'reasoning_content','runs':[]}
assert len(set(summary['seeds']))==3
for seed in summary['seeds']:
 out=root/f'seed-{seed}'
 shutil.copyfile('results/repository-agent/whatisit-macos/routing-regression.py',out/'whatisit-macos/tests/test_agent_regression.py')
 env={**os.environ,'AGENT_SEED':str(seed),'HF_HOME':str(Path('work/hf-cache').resolve())}
 with (out/'run.log').open('x') as log:
  result=subprocess.run([sys.executable,str(root/'harness.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
 receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
 row={'seed':seed,'process_returncode':result.returncode,'status':receipt.get('status','no_receipt'),
      'success':result.returncode==0 and receipt.get('success') is True,'final_test_returncode':receipt.get('final_test_returncode')}
 summary['runs'].append(row);summary['successes']=sum(r['success'] for r in summary['runs'])
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(row),flush=True)
summary['status']='complete'
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Completed successes',summary['successes'],'out of 3',flush=True)

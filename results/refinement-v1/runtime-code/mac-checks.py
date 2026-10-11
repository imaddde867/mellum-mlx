"""Finite M4 checks, snapshots from public system counters only."""
import json
import subprocess
import time
from pathlib import Path
from validate import check_hashes

root = Path('results/refinement-v1'); root.mkdir(parents=True, exist_ok=True)
report = root / 'mac-system-snapshots.json'
if report.exists():
    raise RuntimeError('Refusing to overwrite system observations')

def snapshot():
    return {'time_unix':time.time(), **{name: subprocess.check_output(command, text=True).strip() for name, command in {
        'power': ['pmset', '-g', 'batt'], 'swap': ['sysctl', 'vm.swapusage'],
        'memory': ['vm_stat'], 'pressure': ['sysctl', 'kern.memorystatus_vm_pressure_level']}.items()}}

observations = {'scope': 'periodic aggregate system counters; not continuous peaks or causal attribution', 'editor_workload': 'Visual Studio Code with convert.py open and active, idle; no typing or builds. Other normal desktop applications left open.', 'stages': []}

def run(command, stage):
    if "AC Power" not in snapshot()['power']:
        raise RuntimeError('AC power required')
    row = {'stage': stage, 'before': snapshot(), 'periodic': []}
    observations['stages'].append(row)
    child = subprocess.Popen(command)
    while child.poll() is None:
        row['periodic'].append(snapshot())
        report.write_text(json.dumps(observations, indent=2) + '\n')
        time.sleep(2)
    row['after'] = snapshot()
    row['returncode'] = child.returncode
    report.write_text(json.dumps(observations, indent=2) + '\n')
    if child.returncode:
        raise RuntimeError(f'{stage} failed: {child.returncode}')

print('Waiting for fresh unlocked editor verification', flush=True)
while not (root/'mac-editor-unlocked.json').exists():
    time.sleep(2)
selection = json.loads((root/'mac-selection.json').read_text())
native = Path(selection['model'])
check_hashes(native, json.loads((native/'conversion.json').read_text())['artifact_sha256'])
check_hashes(Path('work/mxfp4'), {'model.safetensors':'b6224c8ea3e32190acc1500ce513d39263be257c6c71a10a31e832c045d2cb9a'})
for tag, model in [(selection['label'], str(native)), ('mxfp4', 'work/mxfp4')]:
    run(['.venv/bin/python', 'scripts/score.py', model, '--output',
         str(root/f'mac-{tag}-finite.json')], f'{tag} finite smoke')
    for context, label in [(1024, '1k'), (4096, '4k'), (16384, '16k')]:
        run(['.venv/bin/python', 'scripts/benchmark.py', model, '--context', str(context),
             '--output', str(root/f'mac-{tag}-{label}.json')], f'{tag} {label}')
observations['status'] = 'complete'
report.write_text(json.dumps(observations, indent=2) + '\n')
print('Mac comparison complete', flush=True)

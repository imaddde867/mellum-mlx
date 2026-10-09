"""Frozen mixed-precision development screen; execute code only in the pinned sandbox."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.error

from coding_probe import INSTRUCTION, weight_hashes
from convert import REVISION, realized_map, sha256
from evaluate_coding import IMAGE
from http_probe import request

DATA = Path('fixtures/nonexpert-v1')


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        out.write(json.dumps(value, indent=2) + '\n')


def frozen():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    path = DATA / 'dev.jsonl'
    if sha256(path) != manifest['sha256']:
        raise ValueError('Frozen development tasks changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 24 or len({r['id'] for r in rows}) != 24:
        raise ValueError('Incomplete/duplicate development set')
    return rows


def totals(rows):
    return {'passed': sum(r['passed'] for r in rows),
            'generation_s': sum(r['elapsed_s'] for r in rows),
            'failed_generation_s': sum(r['elapsed_s'] for r in rows if not r['passed']),
            'completion_tokens': sum(r['completion_tokens'] for r in rows),
            'truncated': sum(r['truncated'] for r in rows),
            'empty': sum(r['empty'] for r in rows)}


def assess_reload(within, across, changed):
    if changed or not within or not across:
        raise ValueError('Changed parameters or missing comparisons')
    if any(not math.isfinite(x) or x < 0 for x in within + across):
        raise ValueError('Nonfinite comparison')
    if max(across) > max(within) + 1e-4:
        raise ValueError('Reload delta exceeds observed same-artifact variability')
    return 'identical' if max(within + across) == 0 else 'within_observed_variability'


def binding(model):
    return {'model': str(model), 'weights_sha256': weight_hashes(model),
            'weight_bytes': sum(p.stat().st_size for p in Path(model).glob('*.safetensors')),
            'config_sha256': sha256(Path(model) / 'config.json'),
            'script_sha256': sha256(Path(__file__)),
            'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers','safetensors']}}


def integrity(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.utils import save_model
    receipt = binding(args.model)
    receipt['status'] = 'running'
    try:
        cfg = json.loads((args.model / 'config.json').read_text())
        manifest = json.loads((args.model / 'conversion.json').read_text())
        if manifest['source_revision'] != REVISION:
            raise ValueError('Source revision mismatch')
        receipt['precision_map'] = realized_map(cfg['quantization'], manifest['precision_map'])
        model, tokenizer = load(str(args.model))
        params = dict(tree_flatten(model.parameters()))
        serialized = {}
        for shard in args.model.glob('*.safetensors'):
            serialized.update(mx.load(str(shard)))
        changed = sorted(set(params) ^ set(serialized))
        for k in params.keys() & serialized.keys():
            if params[k].dtype != serialized[k].dtype or params[k].shape != serialized[k].shape or not mx.all(params[k] == serialized[k]).item():
                changed.append(k)
        receipt['serialized_changed_parameters'] = changed
        if changed:
            raise ValueError('Serialized parameters differ from loaded model')
        serialized.clear()
        source, _ = load('work/source', lazy=True)
        source_params = dict(tree_flatten(source.parameters()))
        for k, spec in manifest['excluded_tensors'].items():
            value = params[k]
            if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype'] or not mx.all(value == source_params[k]).item():
                raise ValueError(f'Excluded tensor changed: {k}')
        source_params.clear()
        del source
        receipt['excluded_equal_to_bf16_source'] = True
        prompt = tokenizer.apply_chat_template([{'role':'user','content':'Write a Python function that adds two integers.'}], tokenize=True, add_generation_prompt=True)
        x = mx.array([prompt])
        def forwards(m):
            values = []
            for _ in range(3):
                y = m(x).astype(mx.float32)
                mx.eval(y)
                if not mx.all(mx.isfinite(y)).item():
                    raise ValueError('Nonfinite logits')
                values.append(y)
            return values
        before = forwards(model)
        copy = Path('work/nonexpert-roundtrip') / args.model.name
        if copy.exists():
            raise ValueError('Refusing existing roundtrip artifact')
        copy.mkdir(parents=True)
        save_model(copy, model, donate_model=False)
        for name in ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']:
            path = args.model / name
            if path.exists():
                (copy / name).write_bytes(path.read_bytes())
        reloaded, _ = load(str(copy))
        after_params = dict(tree_flatten(reloaded.parameters()))
        changed = sorted(set(params) ^ set(after_params))
        for k in params.keys() & after_params.keys():
            a, b = params[k], after_params[k]
            if a.shape != b.shape or a.dtype != b.dtype or not mx.all(a == b).item():
                changed.append(k)
        after = forwards(reloaded)
        delta = lambda a,b: mx.max(mx.abs(a-b)).item()
        within = [delta(xs[i],xs[j]) for xs in [before,after] for i in range(3) for j in range(i)]
        across = [delta(a,b) for a in before for b in after]
        receipt.update(within_deltas=within, reload_deltas=across, changed_parameters=changed,
                       finite_forwards=6, mlx_peak_bytes=mx.get_peak_memory(),
                       roundtrip_weights_sha256=weight_hashes(copy))
        receipt['comparison'] = assess_reload(within, across, changed)
        receipt['status'] = 'passed'
    except Exception as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        write_new(args.output, receipt)


def serve():
    import mlx.core as mx
    from mlx_lm.server import main
    def stop(*_):
        write_new(os.environ['NONEXPERT_MEMORY_RECEIPT'], {
            'mlx_peak_bytes': mx.get_peak_memory(), 'mlx_active_bytes_at_shutdown': mx.get_active_memory(),
            'scope': 'MLX allocator peak, including load; excludes system/GPU driver memory'})
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    del sys.argv[1]
    main()


def sandbox(case, solution):
    payload = {'solution': solution, 'entry_point':case['entry_point'], 'checks':case['checks']}
    # The same pinned EvalPlus sanitizer as HumanEval+, applied to this independent fixture.
    code = '''import contextlib, io, json, sys
from evalplus.sanitize import sanitize
p = json.loads(sys.stdin.read())
s = sanitize(p['solution'], entrypoint=p['entry_point'])
result = {'sanitized': s, 'passed': False}
try:
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(s, '<solution>', 'exec'), namespace)
        exec(compile(p['checks'], '<checks>', 'exec'), namespace)
    result['passed'] = True
except BaseException as e:
    result['error'] = type(e).__name__ + ': ' + str(e)
print(json.dumps(result))
'''
    command = ['docker','run','--rm','-i','--network','none','--read-only',
               '--tmpfs','/tmp:rw,nosuid,nodev,size=256m','--cap-drop','ALL',
               '--security-opt','no-new-privileges','--pids-limit','64','--memory','512m',
               '--cpus','1','--user',f'{os.getuid()}:{os.getgid()}',IMAGE,'python3','-I','-c',code]
    try:
        result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, timeout=20)
        if result.returncode:
            return {'passed':False, 'error':result.stderr[-2000:], 'returncode':result.returncode}
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, ValueError) as error:
        return {'passed':False, 'error':repr(error)}


def screen(args):
    cases = frozen()
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if args.label in {'A','B','C'}:
        gate = json.loads((args.output.parent / f'integrity-{args.label}.json').read_text())
        if gate['status'] != 'passed' or gate['weights_sha256'] != weight_hashes(args.model):
            raise ValueError('Candidate lacks a matching passed integrity receipt')
    receipt = {**binding(args.model), 'label':args.label, 'protocol':manifest,
               'fixture_manifest_sha256':sha256(DATA/'manifest.json'), 'sandbox_image':IMAGE,
               'http_probe_sha256':sha256(Path('scripts/http_probe.py')),
               'server_sha256':sha256(Path(importlib.util.find_spec('mlx_lm.server').origin)),
               'status':'running', 'rows':[]}
    if args.output.exists():
        raise ValueError('Refusing to overwrite screen evidence')
    os.environ['HF_HOME'] = str(Path('work/hf-cache').resolve())
    Path(os.environ['HF_HOME'], 'hub').mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}/v1'
    raw_path = args.output.with_suffix('.raw.jsonl')
    memory_path = args.output.with_suffix('.memory.json')
    os.environ['NONEXPERT_MEMORY_RECEIPT'] = str(memory_path)
    start_run = time.monotonic()
    with raw_path.open('x') as raw, args.output.with_suffix('.server.log').open('x') as log:
        server = subprocess.Popen([sys.executable,__file__,'serve','--model',str(args.model),
                '--host','127.0.0.1','--port',str(port),'--decode-concurrency','1',
                '--prompt-concurrency','1','--prefill-step-size','512'], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Server exited; see retained log')
                try:
                    request(url+'/models')
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server startup timeout')
                    time.sleep(0.25)
            for case in cases:
                prompt = INSTRUCTION+'\n```python\n'+case['prompt']+'\n```'
                started = time.monotonic()
                response = request(url+'/chat/completions', {'model':str(args.model),
                    'messages':[{'role':'user','content':prompt}], 'temperature':0, 'seed':0,
                    'max_tokens':8192,'stream':False})
                elapsed = time.monotonic()-started
                solution = response['message'].get('content') or ''
                execution = sandbox(case, solution)
                row = {'id':case['id'], 'elapsed_s':elapsed,
                       'completion_tokens':response['response']['usage']['completion_tokens'],
                       'truncated':response['finish_reason']=='length', 'empty':not solution.strip(),
                       'passed':execution['passed'], 'execution':execution}
                raw.write(json.dumps({'row':row,'prompt_sha256':__import__('hashlib').sha256(prompt.encode()).hexdigest(),
                                      'response':response})+'\n')
                raw.flush()
                receipt['rows'].append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='execution'}), flush=True)
            receipt['status'] = 'complete'
        except Exception as error:
            receipt.update(status='failed', error=repr(error))
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            receipt.update(**totals(receipt['rows']), elapsed_with_startup_and_tests_s=time.monotonic()-start_run,
                           raw_sha256=sha256(raw_path))
            if memory_path.exists():
                receipt['memory'] = json.loads(memory_path.read_text())
            write_new(args.output, receipt)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='serve':
        return serve()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['integrity','screen'])
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--label',choices=['bf16','4bit','mxfp4','A','B','C'])
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    (integrity if args.action=='integrity' else screen)(args)


if __name__=='__main__':
    main()

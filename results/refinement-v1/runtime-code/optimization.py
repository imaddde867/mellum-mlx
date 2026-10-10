"""Frozen, bounded affine restoration and cached-target DWQ experiment."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error

from coding_probe import weight_hashes
from convert import REVISION, sha256
from evaluate_coding import IMAGE
from http_probe import request

DATA = Path('fixtures/optimization-v1')
RESULTS = Path('results/optimization-v1')
SOURCE = Path('work/source')
GROUPS = ['attention', 'head', 'expert_gate', 'expert_up', 'expert_down']


def precision(path, groups):
    if path.endswith('.mlp.gate'):
        return 8
    if 'attention' in groups and '.self_attn.' in path:
        return 8
    if 'head' in groups and path == 'lm_head':
        return 8
    for group in ['gate', 'up', 'down']:
        if 'expert_' + group in groups and path.endswith('.switch_mlp.' + group + '_proj'):
            return 6
    return 4


def validate_splits(splits):
    ids, prompts = set(), set()
    for rows in splits.values():
        if not rows:
            raise ValueError('Empty split')
        for row in rows:
            prompt = json.dumps(row['messages'], sort_keys=True)
            if row['id'] in ids or prompt in prompts or not row['answer']:
                raise ValueError('Duplicate IDs/prompts or incomplete answer')
            ids.add(row['id'])
            prompts.add(prompt)


def complete_tokens(tokens, limit):
    if not 2 <= len(tokens) <= limit:
        raise ValueError(f'Complete trace has {len(tokens)} tokens; limit {limit}; never truncate')
    return tokens


def assess_reload(within, across, changed):
    if changed or not within or not across:
        raise ValueError('Changed parameters or missing comparisons')
    if any(not math.isfinite(x) or x < 0 for x in within + across):
        raise ValueError('Nonfinite/invalid forward differences')
    if max(across) > max(within) + 1e-4:
        raise ValueError('Reload discrepancy exceeds observed repeated-forward variability')
    return 'identical' if max(across + within) == 0 else 'within_observed_variability'


def score_message(case, message, finish):
    result = {'correct': False, 'tool_valid': None}
    if finish == 'length':
        return result
    content = (message.get('content') or '').strip()
    if case['kind'] == 'exact':
        result['correct'] = content == case['expected']
    elif case['kind'] == 'json':
        try:
            result['correct'] = json.loads(content) == case['expected']
        except (ValueError, TypeError):
            pass
    elif case['kind'] == 'tool':
        calls = message.get('tool_calls') or []
        result['tool_valid'] = False
        try:
            if len(calls) == 1 and calls[0].get('id') and finish == 'tool_calls':
                fn = calls[0]['function']
                actual = {'name': fn['name'], 'arguments': json.loads(fn['arguments'])}
                result['tool_valid'] = isinstance(actual['arguments'], dict)
                result['correct'] = actual == case['expected']
        except (ValueError, TypeError, KeyError):
            pass
    elif case['kind'] == 'code':
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', content, re.S)
        code = '\n'.join(blocks) if blocks else content
        command = ['docker', 'run', '--rm', '-i', '--network', 'none', '--read-only',
                   '--tmpfs', '/tmp:rw,nosuid,nodev,size=64m', '--cap-drop', 'ALL',
                   '--security-opt', 'no-new-privileges', '--pids-limit', '32',
                   '--memory', '256m', '--cpus', '1', '--user', '65534:65534',
                   IMAGE, 'python3', '-I', '-']
        try:
            run = subprocess.run(command, input=code+'\n'+case['checks'], text=True,
                                 capture_output=True, timeout=15)
            result['correct'] = run.returncode == 0
            result['execution_returncode'] = run.returncode
            result['execution_stderr'] = run.stderr[-2000:]
        except subprocess.TimeoutExpired:
            result['execution_timeout'] = True
    else:
        raise ValueError('Unknown task kind')
    return result


def frozen(split):
    manifest = json.loads((DATA / 'manifest.json').read_text())
    path = DATA / (split + '.jsonl')
    if sha256(path) != manifest['splits'][split]['sha256']:
        raise ValueError('Frozen data hash mismatch')
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def binding(model):
    return {'weights_sha256': weight_hashes(model),
            'config_sha256': sha256(Path(model) / 'config.json'),
            'data_manifest_sha256': sha256(DATA / 'manifest.json'),
            'script_sha256': sha256(Path(__file__)),
            'mlx': importlib.metadata.version('mlx'),
            'mlx_lm': importlib.metadata.version('mlx-lm'),
            'weight_bytes': sum(p.stat().st_size for p in Path(model).glob('*.safetensors'))}


def convert(args):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.utils import quantize_model, save
    source_config = json.loads((SOURCE / 'config.json').read_text())
    if source_config.get('model_type') != 'mellum' or 'quantization' in source_config:
        raise ValueError('Require original BF16 Mellum')
    receipt = json.loads(Path('artifacts/mellum2.1-affine-4bit-g64/conversion.json').read_text())
    if receipt['source_revision'] != REVISION:
        raise ValueError('Source revision mismatch')
    for name, digest in receipt['source_sha256'].items():
        if sha256(SOURCE / name) != digest:
            raise ValueError(f'Source hash mismatch: {name}')
    model, tokenizer, config = load(str(SOURCE), return_config=True)
    overrides = {}
    def policy(path, module):
        bits = precision(path, args.groups)
        if bits != 4:
            overrides[path] = {'bits': bits, 'group_size': 64, 'mode': 'affine'}
            return overrides[path]
        return True
    model, config = quantize_model(model, config, 64, 4, quant_predicate=policy)
    mx.eval(model.parameters())
    routers = [v for k, v in overrides.items() if k.endswith('.mlp.gate')]
    if len(routers) != 28 or any(v['bits'] != 8 for v in routers):
        raise ValueError('Router policy lost')
    save(args.model, SOURCE, model, tokenizer, config)
    for pattern in ['tokenizer*', '*.jinja', '*.model', 'special_tokens_map.json', 'LICENSE*']:
        for path in SOURCE.glob(pattern):
            if path.is_file():
                shutil.copy2(path, Path(args.model) / path.name)
    write_new(Path(args.model) / 'optimization.json', {'source_revision': REVISION,
        'source_sha256': receipt['source_sha256'], 'groups': args.groups,
        'overrides': overrides, **binding(args.model)})
    write_new(args.output, json.loads((Path(args.model) / 'optimization.json').read_text()))


def evaluate(args):
    cases = frozen(args.split)
    if args.split == 'heldout' and not (RESULTS / 'selection.json').exists():
        raise ValueError('Record development selection before held-out evaluation')
    receipt = binding(args.model)
    receipt.update(split=args.split, model=args.model, max_tokens=2048, temperature=0,
                   seed=123, prefill_step_size=512, status='running')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix('.raw.jsonl')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}/v1'
    os.environ['HF_HOME'] = str(Path('work/hf-cache').resolve())
    rows = []
    log_path = args.output.with_suffix('.server.log')
    with log_path.open('x') as log, raw_path.open('x') as raw:
        server = subprocess.Popen([sys.executable, '-m', 'mlx_lm', 'server', '--model', args.model,
            '--host', '127.0.0.1', '--port', str(port), '--decode-concurrency', '1',
            '--prompt-concurrency', '1', '--prefill-step-size', '512'], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Server exited; see retained log')
                try:
                    request(url + '/models')
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server startup')
                    time.sleep(0.25)
            # Identical warmup, excluded from task metrics.
            request(url + '/chat/completions', {'model': args.model, 'messages': [{'role':'user','content':'Say OK.'}], 'temperature':0, 'max_tokens':16})
            for case in cases:
                payload = {'model':args.model, 'messages':case['messages'], 'temperature':0,
                           'seed':123, 'max_tokens':2048, 'stream':False}
                if 'tools' in case:
                    payload['tools'] = case['tools']
                start = time.monotonic()
                response = request(url + '/chat/completions', payload)
                elapsed = time.monotonic() - start
                outcome = score_message(case, response['message'], response['finish_reason'])
                row = {'id':case['id'], 'kind':case['kind'], 'elapsed_s':elapsed,
                       'completion_tokens':response['response'].get('usage', {}).get('completion_tokens'),
                       'truncated':response['finish_reason']=='length', **outcome}
                raw.write(json.dumps({'row':row, 'response':response})+'\n')
                raw.flush()
                rows.append(row)
                print(json.dumps(row), flush=True)
            receipt['status'] = 'complete'
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            receipt.update(rows=rows, raw_sha256=sha256(raw_path), correct=sum(r['correct'] for r in rows),
                truncated=sum(r['truncated'] for r in rows), elapsed_s=sum(r['elapsed_s'] for r in rows))
            write_new(args.output, receipt)


def audit_model(model, tokenizer, config, source, dest, tokens):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.utils import save
    model.eval()
    prompt = mx.array([tokens[:-1]])
    forwards = []
    for _ in range(3):
        value = model(prompt).astype(mx.float32)
        mx.eval(value)
        if not mx.all(mx.isfinite(value)).item():
            raise ValueError('Nonfinite forward')
        forwards.append(value)
    save(dest, source, model, tokenizer, config, donate_model=False)
    original = dict(tree_flatten(model.parameters()))
    reloaded, _ = load(str(dest))
    loaded = dict(tree_flatten(reloaded.parameters()))
    changed = sorted(set(original) ^ set(loaded))
    for key in original.keys() & loaded.keys():
        a, b = original[key], loaded[key]
        if a.dtype != b.dtype or a.shape != b.shape or not mx.all(a == b).item():
            changed.append(key)
    actual = []
    for _ in range(3):
        value = reloaded(prompt).astype(mx.float32)
        mx.eval(value)
        if not mx.all(mx.isfinite(value)).item():
            raise ValueError('Nonfinite reload forward')
        actual.append(value)
    delta = lambda a,b: mx.max(mx.abs(a-b)).item()
    within = [delta(xs[i],xs[j]) for xs in [forwards,actual] for i in range(3) for j in range(i)]
    across = [delta(a,b) for a in forwards for b in actual]
    result = {'within_deltas':within, 'reload_deltas':across, 'changed_parameters':changed,
              'mlx_peak_memory_gb':mx.get_peak_memory()/1e9}
    # Persist failures as well as passes.
    try:
        result['status'] = assess_reload(within, across, changed)
    except ValueError as error:
        result['status'] = 'unexplained_failure'
        result['error'] = str(error)
    return result


def distill(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.utils import load_tokenizer
    from mlx_lm.quant.dwq import compute_dwq_targets, dwq_quantize
    import mlx.optimizers as optimizers
    mx.random.seed(123)
    tokenizer = load_tokenizer(SOURCE)
    datasets = {}
    token_ids = {}
    for split in ['train','dev']:
        values = []
        for row in frozen(split):
            kwargs = {'tools':row['tools']} if 'tools' in row else {}
            prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                                                    add_generation_prompt=True, **kwargs)
            text = prompt + 'The task is explicit; provide the requested result.\n</think>\n' + row['answer']
            tokens = tokenizer.encode(text, add_special_tokens=False) + [28]
            values.append((complete_tokens(tokens, 256), 0))
        datasets[split] = values
        token_ids[split] = [x[0] for x in values]
    cache = Path('work/optimization-v1-targets')
    manifest = {'tokens':token_ids, 'data_sha256':sha256(DATA/'manifest.json'), 'seed':123,
                'max_seq_length':256, 'batch_size':1, 'source_revision':REVISION,
                'source_weights_sha256':weight_hashes(SOURCE),
                'tokenizer_sha256':sha256(SOURCE/'tokenizer.json'),
                'dwq_source_sha256':sha256(Path(sys.modules['mlx_lm.quant.dwq'].__file__))}
    if args.stage == 'targets':
        if cache.exists():
            raise ValueError('Refusing existing target cache')
        model, _ = load(str(SOURCE))
        compute_dwq_targets(model, cache, datasets['train'], datasets['dev'], 1, 256, 123)
        manifest['cache_sha256'] = {str(p.relative_to(cache)):sha256(p) for p in cache.glob('*/*.safetensors')}
        write_new(cache/'manifest.json', manifest)
        write_new(args.output, {'status':'complete', 'manifest':manifest, 'mlx_peak_memory_gb':mx.get_peak_memory()/1e9})
        return
    cached = json.loads((cache/'manifest.json').read_text())
    cache_hashes = cached.pop('cache_sha256')
    if cached != manifest or any(sha256(cache/p) != h for p,h in cache_hashes.items()):
        raise ValueError('Stale or altered teacher target cache')
    model, tokenizer, config = load(args.model, return_config=True)
    model.freeze()
    routers = {k:mx.array(v) for k,v in tree_flatten(model.parameters()) if '.mlp.gate.' in k}
    mx.eval(routers)
    initial_memory = mx.get_active_memory()/1e9
    def targets(_, index, split):
        value = mx.load(cache/split/f'{index:010d}.safetensors')
        return value['logits'],value['indices']
    dwq_quantize(model, targets, optimizers.Adam(learning_rate=1e-6, bias_correction=True),
                 datasets['train'], datasets['dev'], 1, 256, 123)
    after = dict(tree_flatten(model.parameters()))
    if any(not mx.all(v == after[k]).item() for k,v in routers.items()):
        raise ValueError('DWQ changed router parameters')
    peak_training = mx.get_peak_memory()/1e9
    dest = Path(args.destination)
    report = audit_model(model, tokenizer, config, args.model, dest, datasets['dev'][0][0])
    report.update(training_peak_gb=peak_training, initial_memory_gb=initial_memory,
                  updates=len(datasets['train']), routers_unchanged=True, learning_rate=1e-6,
                  temperature=2, cache_manifest_sha256=sha256(cache/'manifest.json'), **binding(dest))
    write_new(args.output, report)
    if report['status'] == 'unexplained_failure':
        raise ValueError('DWQ serialization audit failed; see report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['convert','evaluate','targets','train','audit'])
    parser.add_argument('--model')
    parser.add_argument('--destination')
    parser.add_argument('--groups', nargs='*', choices=GROUPS, default=[])
    parser.add_argument('--split', choices=['dev','heldout'], default='dev')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite result')
    if args.stage == 'convert' and Path(args.model).exists():
        parser.error('Refusing to overwrite model')
    if args.stage in ['audit','train'] and Path(args.destination).exists():
        parser.error('Refusing to overwrite destination')
    if args.stage == 'convert':
        convert(args)
    elif args.stage == 'evaluate':
        evaluate(args)
    elif args.stage in ['targets','train']:
        distill(args)
    else:
        from mlx_lm import load
        model, tokenizer, config = load(args.model, return_config=True)
        tokens = json.loads(Path('results/cuda/dwq-pilot.json').read_text())['cache_manifest']['tokens']['valid']
        write_new(args.output, audit_model(model, tokenizer, config, args.model, args.destination, tokens))


if __name__ == '__main__':
    main()

"""Generate HumanEval+ samples via local MLX HTTP; never execute generated code."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from http_probe import request

INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"


def validate_tasks(tasks):
    ids = [task['task_id'] for task in tasks]
    if not tasks or len(ids) != len(set(ids)):
        raise ValueError('Missing or duplicate benchmark tasks')
    for task in tasks:
        if not all(isinstance(task[key], str) and task[key].strip() for key in ['prompt', 'entry_point']):
            raise ValueError('Missing task prompt or entry point')
        if not isinstance(task['task_id'], str) or not task['task_id'].startswith('HumanEval/'):
            raise ValueError('Unexpected benchmark task ID')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('--tasks', type=Path, default=Path('work/evalplus/humaneval-plus.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw_path = args.output.with_suffix('.raw.jsonl')
    receipt_path = args.output.with_suffix('.receipt.json')
    if args.output.suffix != '.jsonl' or any(path.exists() for path in [args.output, raw_path, receipt_path]):
        p.error('Use a new .jsonl output; refusing to overwrite benchmark evidence')
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines()]
    validate_tasks(tasks)
    os.environ['HF_HOME'] = str(Path('work/hf-cache').resolve())
    Path(os.environ['HF_HOME'], 'hub').mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    url = f'http://127.0.0.1:{port}/v1'
    receipt = {'model': args.model, 'task_ids': [t['task_id'] for t in tasks],
        'tasks_sha256': hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256((Path(args.model) / 'config.json').read_bytes()).hexdigest(),
        'mlx': importlib.metadata.version('mlx'), 'mlx_lm': importlib.metadata.version('mlx-lm'),
        'instruction': INSTRUCTION, 'sampling': 'greedy', 'seed': 0, 'max_tokens': 8192,
        'prefill_step_size': 512, 'status': 'running', 'completed': 0,
        'scope': 'one generated sample per task; execute only in isolated EvalPlus container'}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    log_path = Path('work') / (args.output.stem + '-coding-server.log')
    with log_path.open('w') as log, args.output.open('x') as samples, raw_path.open('x') as raw:
        server = subprocess.Popen([sys.executable, '-m', 'mlx_lm', 'server', '--model', args.model,
            '--host', '127.0.0.1', '--port', str(port), '--decode-concurrency', '1',
            '--prompt-concurrency', '1', '--prefill-step-size', '512'], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited; inspect {log_path}')
                try:
                    request(url + '/models')
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Coding server startup timed out')
                    time.sleep(0.25)
            for task in tasks:
                prompt = INSTRUCTION + '\n```python\n' + task['prompt'].strip() + '\n```'
                response = request(url + '/chat/completions', {'model': args.model,
                    'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                    'seed': 0, 'max_tokens': 8192, 'stream': False})
                samples.write(json.dumps({'task_id': task['task_id'],
                    'solution': response['message'].get('content') or ''}) + '\n')
                samples.flush()
                row = {'task_id': task['task_id'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'finish_reason': response['finish_reason'], 'message': response['message'],
                    'usage': response['response'].get('usage')}
                raw.write(json.dumps(row) + '\n')
                raw.flush()
                receipt['completed'] += 1
                print(json.dumps({k: v for k, v in row.items() if k != 'message'}), flush=True)
            receipt['status'] = 'generation_complete'
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            if receipt['status'] == 'running':
                receipt['status'] = 'failed_generation'
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Execute generated samples only in the pinned, isolated EvalPlus container."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from coding_probe import validate_tasks, load_tasks

IMAGE = 'ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740'


def check_samples(tasks, samples, smoke=False):
    validate_tasks(tasks, smoke=smoke)
    expected = [task['task_id'] for task in tasks]
    if [sample['task_id'] for sample in samples] != expected:
        raise ValueError('Samples must contain exactly one ordered result for every task')
    if not all(isinstance(sample['solution'], str) for sample in samples):
        raise ValueError('Sample solutions must be strings; empty strings remain failures')


def check_binding(samples_path, receipt, allow_legacy=False):
    raw_path = samples_path.with_suffix('.raw.jsonl')
    hashes = {'samples_sha256': hashlib.sha256(samples_path.read_bytes()).hexdigest(),
              'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest()}
    for key, actual in hashes.items():
        if key not in receipt:
            if not allow_legacy:
                raise ValueError('Missing generation-time sample binding; legacy evidence requires explicit opt-in')
        elif receipt[key] != actual:
            raise ValueError(f'Generation receipt mismatch: {key}')
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    derived = [{'task_id': row['task_id'], 'solution': row['message'].get('content') or ''} for row in raw]
    if samples != derived:
        raise ValueError('Samples differ from retained raw responses')
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples', type=Path)
    p.add_argument('--tasks', type=Path, default=Path('work/evalplus/humaneval-plus.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='Evaluate a subset; never a full benchmark')
    p.add_argument('--legacy-receipt', action='store_true', help='Allow old unbound receipts, explicitly labeled')
    args = p.parse_args()
    tasks = load_tasks(args.tasks, smoke=args.smoke)
    samples = [json.loads(line) for line in args.samples.read_text().splitlines()]
    check_samples(tasks, samples, smoke=args.smoke)
    receipt = json.loads(args.samples.with_suffix('.receipt.json').read_text())
    binding = check_binding(args.samples, receipt, allow_legacy=args.legacy_receipt)
    if receipt.get('mode', 'full') != ('smoke' if args.smoke else 'full'):
        p.error('Generation and evaluation modes differ')
    if receipt['status'] != 'generation_complete' or receipt['completed'] != len(tasks):
        p.error('Generation did not complete')
    if receipt['tasks_sha256'] != hashlib.sha256(args.tasks.read_bytes()).hexdigest():
        p.error('Generation and evaluation task files differ')
    paths = [args.tasks.resolve(), args.samples.resolve(), args.output.resolve()]
    if not all(path.is_relative_to(Path.cwd().resolve()) for path in paths):
        p.error('Keep benchmark inputs and outputs inside the checkout')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir()
    code = '''import json, subprocess
from evalplus.sanitize import sanitize
from evalplus.data import get_human_eval_plus
problems = get_human_eval_plus()
with open('/input/samples.jsonl') as samples, open('/output/samples-sanitized.jsonl', 'x') as output:
    for line in samples:
        row = json.loads(line)
        row['solution'] = sanitize(row['solution'], entrypoint=problems[row['task_id']]['entry_point'])
        output.write(json.dumps(row) + '\\n')
subprocess.run(['evalplus.evaluate', '--dataset', 'humaneval', '--samples',
                '/output/samples-sanitized.jsonl', '--parallel', '2'], check=True)
'''
    command = ['docker', 'run', '-i', '--rm', '--network', 'none', '--read-only',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--pids-limit', '256', '--memory', '4g',
        '--cpus', '2', '--user', f'{os.getuid()}:{os.getgid()}',
        '-e', 'XDG_CACHE_HOME=/tmp/cache', '-e', 'HUMANEVAL_OVERRIDE_PATH=/input/tasks.jsonl',
        '--mount', f'type=bind,src={paths[0]},dst=/input/tasks.jsonl,readonly',
        '--mount', f'type=bind,src={paths[1]},dst=/input/samples.jsonl,readonly',
        '--mount', f'type=bind,src={paths[2]},dst=/output', '-w', '/output', IMAGE, 'python3', '-']
    protocol = {**binding,
        'generation_receipt_sha256': hashlib.sha256(args.samples.with_suffix('.receipt.json').read_bytes()).hexdigest(),
        'sample_binding': 'generation-time checksums' if receipt.get('samples_sha256') and receipt.get('raw_sha256') else 'legacy: checked at evaluation only',
        'mode': 'smoke' if args.smoke else 'full',
        'weights_sha256': receipt.get('weights_sha256'),
        'weight_binding': 'generation-time checksums' if receipt.get('weights_sha256') else 'legacy run: separate provenance only',
        'image': IMAGE, 'evalplus': '0.4.0.dev2', 'network': 'none',
        'run_as_invoking_user': True, 'read_only_root': True, 'cpus': 2,
        'container_memory_gb_binary': 4, 'parallel_workers': 2,
        'task_sha256': receipt['tasks_sha256'], 'tasks': len(tasks),
        'samples_sha256': hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        'scope': 'HumanEval+ one greedy sample per task; benchmark training overlap is possible'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    subprocess.run(command, input=code, text=True, check=True)
    result_receipt = {'status': 'evaluated', 'protocol_sha256': hashlib.sha256((args.output / 'protocol.json').read_bytes()).hexdigest(),
        'files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.output.glob('*.json*') if path.name != 'protocol.json'}}
    (args.output / 'evaluation-receipt.json').write_text(json.dumps(result_receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()

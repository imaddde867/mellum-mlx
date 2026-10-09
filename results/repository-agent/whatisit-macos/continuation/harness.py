"""One constrained repository-agent trial; generated Python executes only in Docker."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, 'scripts')
from http_probe import request
from evaluate_coding import IMAGE
from coding_probe import weight_hashes

root = Path('work/repository-agent/whatisit-macos').resolve()
out = Path('work/repository-agent/continuation')
out.mkdir(exist_ok=False)
model = 'artifacts/mellum2.1-affine-6bit-g64'
allowed = {'src/whatisit_macos/engine.py', 'tests/test_engine.py', 'tests/test_agent_regression.py', 'README.md'}

def file_path(name, write=False):
    if name not in allowed or (write and name != 'src/whatisit_macos/engine.py'):
        raise ValueError('File access outside the fixed task allowlist')
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Unsafe path')
    return path

for invalid in ['../secret', '/etc/passwd', 'tests/test_agent_regression.py']:
    try:
        file_path(invalid, write=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Write allowlist failed')

command = ['docker', 'run', '--rm', '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m',
    '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '128', '--memory', '1g', '--cpus', '2',
    '--user', f'{os.getuid()}:{os.getgid()}', '--mount', f'type=bind,src={root},dst=/task,readonly', '-w', '/task',
    '-e', 'PYTHONPATH=/task/src', '-e', 'PYTHONDONTWRITEBYTECODE=1', IMAGE,
    'python3', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
def tests():
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return {'returncode': result.returncode, 'output': result.stdout + result.stderr}

baseline = tests()
(out / 'baseline-tests.json').write_text(json.dumps(baseline, indent=2) + '\n')
assert baseline['returncode'] != 0, 'Regression must fail before intervention'
before = file_path('src/whatisit_macos/engine.py').read_bytes()
(out / 'engine-before.py').write_bytes(before)
receipt = {'model': model, 'weights_sha256': weight_hashes(model), 'repository_commit': 'fbdcb10b289001baf2c0e444bb963e421262a4fd',
    'baseline_engine_sha256': hashlib.sha256(before).hexdigest(), 'image': IMAGE,
    'scope': 'one authored regression in an isolated tracked-files copy; no general agent-success claim',
    'max_turns': 8, 'continuation_of': 'retry-edit', 'scope_note': 'separate continuation after a failed bounded trial; not an independent success', 'temperature': 0, 'max_tokens': 8192, 'status': 'running'}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
def tool(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': required}}}
tools = [tool('list_files', 'List files available for this task.', {}, []),
    tool('read_file', 'Read an allowed repository file.', {'path': {'type': 'string'}}, ['path']),
    tool('edit_file', 'Replace one exact occurrence in engine.py only. Read the fixed regression tests too.', {'path': {'type': 'string'}, 'old': {'type': 'string'}, 'new': {'type': 'string'}}, ['path', 'old', 'new']),
    tool('run_tests', 'Run the full fixed test suite in an isolated container.', {}, [])]
messages = [{'role': 'user', 'content': "Continue the unfinished repair from a prior bounded trial. The current engine.py still fails the fixed tests. Inspect the current file and regression tests, run tests, and correct failures. Original task: mutation inflections such as 'changing battery cycle count', 'deleting battery cycle count' and 'writing battery cycle count' incorrectly receive a command suggestion. Inspection-only routing must abstain for these requests while existing inspection requests continue to work. Inspect files, make the smallest engine.py change, run the fixed tests, and correct failures if needed. You may only edit engine.py. Do not add dependencies or alter tests. Use the provided tools; finish by describing the change."}]
os.environ['HF_HOME'] = str(Path('work/hf-cache').resolve())
with socket.socket() as s:
    s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
url = f'http://127.0.0.1:{port}/v1'
with (out / 'server.log').open('w') as log, (out / 'transcript.jsonl').open('x') as transcript:
    server = subprocess.Popen([sys.executable, '-m', 'mlx_lm', 'server', '--model', model, '--host', '127.0.0.1', '--port', str(port), '--decode-concurrency', '1', '--prompt-concurrency', '1'], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(480):
            if server.poll() is not None: raise RuntimeError('Server exited')
            try:
                request(url + '/models'); break
            except Exception: time.sleep(.25)
        else: raise TimeoutError('Server startup')
        for turn in range(8):
            response = request(url + '/chat/completions', {'model': model, 'messages': messages, 'tools': tools,
                'temperature': 0, 'seed': 0, 'max_tokens': 8192, 'stream': False})
            message = response['message']; messages.append(message)
            transcript.write(json.dumps({'turn': turn, 'response': response}) + '\n'); transcript.flush()
            calls = message.get('tool_calls') or []
            print('turn', turn, 'tools', [c['function']['name'] for c in calls], flush=True)
            if not calls:
                if response['finish_reason'] in {'tool_calls', 'length'}:
                    messages.append({'role': 'user', 'content': 'No valid tool call was parsed or executed. Send a valid short JSON tool call and continue; the task is unfinished.'})
                    continue
                receipt['status'] = 'model_finished'; break
            for call in calls:
                try:
                    name = call['function']['name']; args = call['function']['arguments']
                    if isinstance(args, str): args = json.loads(args)
                    if name == 'list_files': result = sorted(allowed)
                    elif name == 'read_file': result = file_path(args['path']).read_text()
                    elif name == 'edit_file':
                        path = file_path(args['path'], write=True)
                        content = path.read_text()
                        if not all(isinstance(args[k], str) and args[k] for k in ['old', 'new']) or content.count(args['old']) != 1:
                            raise ValueError('Edit must match exactly one nonempty occurrence')
                        path.write_text(content.replace(args['old'], args['new'], 1)); result = 'File updated'
                    elif name == 'run_tests': result = tests()
                    else: raise ValueError('Unknown tool')
                except (ValueError, KeyError, TypeError) as error: result = {'error': str(error)}
                row = {'role': 'tool', 'tool_call_id': call['id'], 'name': call['function']['name'], 'content': json.dumps(result)}
                messages.append(row); transcript.write(json.dumps({'tool_result': row}) + '\n'); transcript.flush()
        else: receipt['status'] = 'turn_budget_exhausted'
        final = tests(); (out / 'final-tests.json').write_text(json.dumps(final, indent=2) + '\n')
        receipt['final_test_returncode'] = final['returncode']
        receipt['engine_changed'] = file_path('src/whatisit_macos/engine.py').read_bytes() != before
        receipt['success'] = final['returncode'] == 0 and receipt['engine_changed']
    finally:
        server.terminate()
        try: server.wait(timeout=10)
        except subprocess.TimeoutExpired: server.kill(); server.wait()
        receipt['transcript_sha256'] = hashlib.sha256((out / 'transcript.jsonl').read_bytes()).hexdigest()
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)

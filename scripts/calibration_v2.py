"""Original calibration-v2 trajectories and evaluation-only untrained fidelity protocol."""
import argparse
import hashlib
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from convert import REVISION, sha256
from nonexpert import binding, sandbox, write_new
from validate import check_hashes

ROOT = Path('fixtures/calibration-v2')
PROTOCOL = Path('results/calibration-v2/predeclared.json')
REGIONS = ('context', 'reasoning', 'transition', 'final', 'tool_call')
TOOLS = [dict(type='function', function=dict(name=name, description=description,
    parameters=dict(type='object', properties={k: {'type': 'string'} for k in keys},
                    required=keys, additionalProperties=False))) for name, description, keys in [
    ('read_file', 'Read the task module.', ['path']),
    ('edit_file', 'Replace the task module with the supplied complete source.', ['path', 'content']),
    ('run_tests', 'Run the supplied Python assertions on the task module.', ['path', 'code'])]]


def check_splits(splits):
    seen = set()
    if set(splits) != {'train', 'valid', 'reserved'}:
        raise ValueError('Missing split')
    for families in splits.values():
        if not families or seen.intersection(families) or len(set(families)) != len(families):
            raise ValueError('Duplicate or leaking source family')
        seen.update(families)


def check_call(call):
    schemas = {t['function']['name']: t['function']['parameters'] for t in TOOLS}
    if set(call) != {'name', 'arguments'} or call['name'] not in schemas:
        raise ValueError('Unknown tool or invalid call shape')
    args, schema = call['arguments'], schemas[call['name']]
    if not isinstance(args, dict) or set(args) != set(schema['required']) or any(
            not isinstance(v, str) for v in args.values()):
        raise ValueError('Tool arguments violate strict schema')
    if args['path'] != 'task.py':
        raise ValueError('Tool path outside fixed workspace')


def check_trajectory(row):
    messages = row['messages']
    if row.get('finish_reason') == 'length' or not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('Truncated or unfinished trajectory')
    final = messages[-1]
    if not final.get('content') or row['solution'].strip() not in final['content']:
        raise ValueError('Final answer differs from validated solution')
    if not final.get('reasoning_content'):
        raise ValueError('Missing final reasoning-to-answer transition')
    pending = None
    for message in messages:
        if message['role'] not in {'user', 'assistant', 'tool'}:
            raise ValueError('Unknown conversation role')
        if message['role'] == 'tool':
            if not pending or message.get('tool_call_id') != pending:
                raise ValueError('Unmatched tool result')
            pending = None
        elif pending:
            raise ValueError('Missing tool result')
        for call in message.get('tool_calls', []):
            if pending:
                raise ValueError('Only sequential tool calls supported')
            check_call(call['function'])
            if call['function']['name'] == 'edit_file' and call['function']['arguments']['content'] != row['solution']:
                raise ValueError('Edit differs from validated source')
            pending = call['id']
    if pending:
        raise ValueError('Unanswered tool call')


def regions(text, offsets, transition_radius=4):
    """Label prediction targets; closing think tokens plus four neighbors form transitions."""
    labels = ['context'] * len(offsets)
    spans, transitions = [], []
    for match in re.finditer(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', text, re.S):
        start, stop = match.span(1)
        body = match[1]
        think = re.search(r'<think>\n(.*?)\n</think>', body, re.S)
        if think:
            spans.append((start + think.start(1), start + think.end(1), 'reasoning'))
            close = start + body.index('</think>')
            transitions.append((close, close + len('</think>')))
            start += think.end()
        spans.append((start, stop, 'final'))
        for call in re.finditer(r'<tool_call>.*?</tool_call>', body, re.S):
            spans.append((match.start(1) + call.start(), match.start(1) + call.end(), 'tool_call'))
    for start, stop, label in spans:
        for i, (left, right) in enumerate(offsets):
            if left < stop and right > start:
                labels[i] = label
    for start, stop in transitions:
        hits = [i for i,(left,right) in enumerate(offsets) if left < stop and right > start]
        if not hits:
            raise ValueError('Unmapped reasoning boundary')
        for i in range(max(0, hits[0]-transition_radius), min(len(labels), hits[-1]+1+transition_radius)):
            labels[i] = 'transition'
    return labels[1:]


def scoring_rows(rows):
    if not rows or any(r['split'] != 'valid' for r in rows):
        raise ValueError('Only the frozen validation split may be scored')
    return rows


def trajectories(family, receipt):
    solution, buggy = family['solution'], family['buggy']
    final = {'role': 'assistant', 'reasoning_content': family['reasoning'],
             'content': '```python\n' + solution + '```\nThe retained assertions pass, including the boundary cases.'}
    def exchange(name, args, result, explanation, index):
        return [{'role': 'assistant', 'content': '', 'reasoning_content': explanation,
                 'tool_calls': [{'id': f'call_{index}', 'type': 'function',
                                 'function': {'name': name, 'arguments': args}}]},
                {'role': 'tool', 'tool_call_id': f'call_{index}', 'content': result}]
    requests = {
        'coding': 'Implement a Python function. '+family['prompt'],
        'bug_repair': 'Repair this implementation to satisfy the contract. '+family['prompt']+'\n```python\n'+buggy+'```',
        'tool_call': 'Write task.py using edit_file, then give the complete final implementation. '+family['prompt'],
        'conversation': 'Read task.py, repair it, run the supplied tests, and report the complete solution. '+family['prompt']+'\nTests:\n'+family['checks']}
    for kind, prompt in requests.items():
        messages = [{'role': 'user', 'content': prompt}]
        if kind == 'conversation':
            messages += exchange('read_file', {'path': 'task.py'}, buggy,
                'Inspect the existing implementation before making the focused repair.', 0)
        if kind in {'tool_call', 'conversation'}:
            messages += exchange('edit_file', {'path': 'task.py', 'content': solution},
                json.dumps({'written': True, 'content_sha256': hashlib.sha256(solution.encode()).hexdigest()}),
                family['reasoning'], 1)
        if kind == 'conversation':
            messages += exchange('run_tests', {'path': 'task.py', 'code': family['checks']},
                json.dumps(receipt), 'Check the expected result and invalid or boundary inputs in the isolated sandbox.', 2)
        messages.append(final)
        yield dict(id=family['family']+'-'+kind, family=family['family'], split=family['split'],
            kind=kind, origin='authored', license=family['license'], messages=messages,
            solution=solution, entry_point=family['entry_point'], checks=family['checks'],
            tools=TOOLS if kind in {'tool_call', 'conversation'} else [],
            finish_reason='authored_complete', execution_receipt=receipt)



def render_trajectory(tokenizer, row):
    options = dict(tools=row['tools'] or None, add_generation_prompt=False)
    text = tokenizer.apply_chat_template(row['messages'], tokenize=False, **options)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded['input_ids']
    direct = tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=False, **options)
    if direct != tokens:
        raise ValueError('Rendered template/tokenizer mismatch')
    return text, tokens, regions(text, encoded['offset_mapping'])


def prepare(args):
    from transformers import AutoTokenizer
    split_manifest = json.loads((ROOT/'splits.json').read_text())
    splits = split_manifest['families']
    check_splits(splits)
    protocol = json.loads(PROTOCOL.read_text())
    source_receipt = json.loads(Path('results/nonexpert-v1/preflight.json').read_text())
    check_hashes(args.source, source_receipt['source_sha256'])
    tokenizer = AutoTokenizer.from_pretrained(str(args.source), trust_remote_code=False)
    if sha256(args.source/'chat_template.jinja') != protocol['chat_template_sha256']:
        raise ValueError('Upstream template binding changed')
    families = json.loads((ROOT/'families.json').read_text())
    if {f['family'] for f in families} != set(splits['train']+splits['valid']):
        raise ValueError('Source catalog differs from frozen family split')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    rows, rejected, executions = {'train': [], 'valid': []}, [], []
    started = time.monotonic()
    # Check the sandbox can both accept and reject before validating the authored material.
    probe = {'entry_point': 'probe', 'checks': 'assert probe(2, 3) == 5'}
    good = sandbox(probe, 'def probe(a,b):\n    return a+b')
    bad = sandbox(probe, 'def probe(a,b):\n    return a-b')
    write_new(out/'sandbox-preflight.json', {'positive': good, 'negative': bad})
    if not good['passed'] or bad['passed']:
        raise ValueError('Sandbox preflight failed')
    for family in families:
        if family['family'] not in splits[family['split']]:
            raise ValueError('Source family moved splits')
        passed = sandbox(family, family['solution'])
        negative = sandbox(family, family['buggy'])
        executions.append({'family': family['family'], 'solution': passed, 'buggy': negative})
        write_new(out/'executions'/f"{family['family']}.json", executions[-1])
        if not passed['passed'] or negative['passed'] or passed.get('sanitized','').strip() != family['solution'].strip():
            rejected.append({'family': family['family'], 'reason': 'sandbox or raw-source mismatch',
                             'solution': passed, 'buggy': negative})
            continue
        for row in trajectories(family, passed):
            try:
                check_trajectory(row)
                text, tokens, labels = render_trajectory(tokenizer, row)
                if not all(label in labels for label in ('reasoning', 'transition', 'final')):
                    raise ValueError('Missing rendered response region')
                if row['tools'] and 'tool_call' not in labels:
                    raise ValueError('Missing rendered tool region')
                row.update(text=text, tokens=tokens, regions=labels)
                rows[family['split']].append(row)
            except ValueError as error:
                rejected.append({'id': row['id'], 'reason': str(error), 'source_trajectory': row})
    write_new(out/'rejections.json', {'authored_rejections': rejected, 'teacher_attempts': [],
        'teacher_generation': 'not used; authored responses, BF16 teacher-forced logits only',
        'teacher_truncated': 0})
    if rejected:
        raise ValueError('Rejected material retained; repair authoring in a new protocol/run, never relabel teacher failures')
    lengths = {}
    for split, values in rows.items():
        with (out/(split+'.jsonl')).open('x') as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        sizes = sorted(len(r['tokens']) for r in values)
        lengths[split] = {'min': sizes[0], 'median': statistics.median(sizes),
            'p90': sizes[int(.9*(len(sizes)-1))], 'max': sizes[-1],
            'total': sum(sizes), 'over_256': sum(s > 256 for s in sizes), 'all_lengths': sizes}
        if len(values) != protocol['counts'][split]:
            raise ValueError('Pilot budget not satisfied')
    coverage = {'counts': {s:len(v) for s,v in rows.items()}, 'reserved_tasks': 32,
        'families': {s:len(v) for s,v in splits.items()},
        'kinds': {s:dict(Counter(r['kind'] for r in v)) for s,v in rows.items()},
        'region_tokens': {s:dict(Counter(x for r in v for x in r['regions'])) for s,v in rows.items()},
        'lengths': lengths, 'truncations': 0, 'training_windows': None,
        'sequence_policy': 'complete full-context teacher forcing in KV-preserving 128-token chunks; no gradient memory claim',
        'limitations': 'Python-only authored pilot, four related presentations per train/valid family; no coverage sufficiency claim; no generated teacher trajectories',
        'sandbox_cases': len(executions), 'sandbox_positive_negative': 'passed', 'elapsed_s':time.monotonic()-started}
    write_new(out/'coverage.json', coverage)
    write_new(out/'manifest.json', {'protocol_sha256':sha256(PROTOCOL),
        'source_revision': REVISION, 'tokenizer_sha256': sha256(args.source/'tokenizer.json'),
        'chat_template_sha256':sha256(args.source/'chat_template.jinja'),
        'inputs_sha256': {str(p):sha256(p) for p in [ROOT/'families.json', ROOT/'splits.json', ROOT/'reserved.jsonl']},
        'splits': {s:{'rows':len(v), 'sha256':sha256(out/(s+'.jsonl'))} for s,v in rows.items()},
        'source_sha256':source_receipt['source_sha256'], 'tools':TOOLS,
        'preparation_script_sha256':sha256(Path(__file__)), 'coverage':coverage})
    print(json.dumps(coverage), flush=True)


def frozen_validation(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('Evaluation protocol changed')
    for path, digest in manifest['inputs_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Source corpus changed')
    path = root/'valid.jsonl'
    if sha256(path) != manifest['splits']['valid']['sha256']:
        raise ValueError('Validation corpus changed')
    rows = scoring_rows([json.loads(line) for line in path.read_text().splitlines()])
    if len(rows) != 32:
        raise ValueError('Incomplete validation split')
    return manifest, rows



def reduce_captured(student, teacher, labels):
    """Compare identical captured arrays with the historical and FP32 scorers."""
    import math
    import mlx.core as mx
    from mlx_lm.tuner.losses import kl_div_loss
    if student.shape != teacher.shape or len(labels) != student.shape[1] or any(x not in REGIONS for x in labels):
        raise ValueError('Captured logits and target regions differ')
    losses = {'legacy': kl_div_loss(.5*student, .5*teacher),
              'fp32': kl_div_loss(.5*student.astype(mx.float32), .5*teacher.astype(mx.float32))}
    result = {}
    for mode, loss in losses.items():
        mx.eval(loss)
        result[mode] = {'aggregate': [loss.sum().item(), len(labels)]}
        for region in REGIONS:
            mask = mx.array([[label == region for label in labels]])
            result[mode][region] = [(mask*loss).sum().item(), int(mask.sum().item())]
        if any(not math.isfinite(value[0]) for value in result[mode].values()):
            raise ValueError('Nonfinite KL')
    return result


def capture(args):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    import mlx_lm.tuner.losses as upstream_losses
    manifest, rows = frozen_validation(args.data)
    if args.stage == 'teacher':
        model_path = args.source
        check_hashes(model_path, manifest['source_sha256'])
    else:
        model_path = args.model
        if args.label == 'C':
            from refine_dwq import require_pristine_c
            conversion = json.loads((model_path/'conversion.json').read_text())
            require_pristine_c(conversion)
            check_hashes(model_path, conversion['artifact_sha256'])
            expected = json.loads(Path('results/nonexpert-v1/integrity-C.json').read_text())['weights_sha256']
        elif args.label == 'mxfp4':
            preflight = json.loads(Path('results/nonexpert-v1/preflight.json').read_text())
            expected = {name:meta['sha256'] for name,meta in preflight['external_files'].items()}
        else:
            raise ValueError('Only pristine C and pinned MXFP4 permitted')
        check_hashes(model_path, expected)
    if sha256(model_path/'chat_template.jinja') != manifest['chat_template_sha256']:
        raise ValueError('Model chat template differs from upstream')
    receipt = binding(model_path)
    receipt.update(protocol_sha256=sha256(PROTOCOL), data_manifest_sha256=sha256(args.data/'manifest.json'),
        script_sha256=sha256(Path(__file__)), updates=0, stage=args.stage, chunk_size=128,
        seed=123, temperature=2, top_k=1024, status='running', model_mode='eval',
        upstream_loss_sha256=sha256(Path(upstream_losses.__file__)))
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        mx.random.seed(123)
        model, tokenizer = load(str(model_path))
        model.eval()
        for row in rows:
            if tokenizer.encode(row['text'], add_special_tokens=False) != row['tokens']:
                raise ValueError('Model tokenizer differs from source')
        repeats = 1 if args.stage == 'teacher' else 5
        metrics, captures = [], {}
        if args.stage != 'teacher':
            teacher_binding = json.loads((args.cache/'receipt.json').read_text())
            if teacher_binding['status'] != 'complete' or teacher_binding['data_manifest_sha256'] != receipt['data_manifest_sha256'] or teacher_binding['protocol_sha256'] != receipt['protocol_sha256']:
                raise ValueError('Teacher context binding changed')
            check_hashes(args.cache, teacher_binding['captures_sha256'])
        for repeat in range(repeats):
            per_row = []
            for row in rows:
                cache = make_prompt_cache(model)
                saved, ids_saved = [], []
                teacher = None if args.stage == 'teacher' else mx.load(args.cache/(row['id']+'.safetensors'))
                for start in range(0, len(row['tokens'])-1, 128):
                    stop = min(start+128, len(row['tokens'])-1)
                    x = mx.array([row['tokens'][start:stop]])
                    logits = model(x, cache=cache)
                    if args.stage == 'teacher':
                        indices = mx.argpartition(logits, kth=logits.shape[-1]-1024, axis=-1)[...,-1024:]
                        values = mx.take_along_axis(logits, indices, axis=-1)
                        mx.eval(values, indices)
                        receipt['logits_dtype'] = str(values.dtype)
                        saved.append(values); ids_saved.append(indices)
                    else:
                        ids = teacher['indices'][:,start:stop]
                        reference = teacher['logits'][:,start:stop]
                        values = mx.take_along_axis(logits, ids, axis=-1)
                        mx.eval(values)
                        receipt['logits_dtype'] = str(values.dtype)
                        receipt['teacher_logits_dtype'] = str(reference.dtype)
                        if not mx.all(mx.isfinite(values)).item():
                            raise ValueError('Nonfinite student logits')
                        saved.append(values)
                    del logits
                if args.stage == 'teacher':
                    target = {'logits':mx.concatenate(saved,axis=1), 'indices':mx.concatenate(ids_saved,axis=1)}
                    mx.eval(target)
                    if not mx.all(mx.isfinite(target['logits'])).item():
                        raise ValueError('Nonfinite teacher logits')
                    path = args.output/(row['id']+'.safetensors')
                    mx.save_safetensors(str(path),target)
                    captures[path.name] = sha256(path)
                    del target, values, indices, x
                else:
                    captured = mx.concatenate(saved,axis=1)
                    sums = reduce_captured(captured, teacher['logits'], row['regions'])
                    per_row.append({'id':row['id'], 'family':row['family'], 'kind':row['kind'], 'sums':sums})
                    if repeat == 0:
                        path = args.output/(row['id']+'.safetensors')
                        mx.save_safetensors(str(path),{'logits':mx.concatenate(saved,axis=1)})
                        captures[path.name] = sha256(path)
                del cache, saved, ids_saved, teacher
                if args.stage != 'teacher':
                    del captured, values, reference, ids, x
                mx.clear_cache()
                print(json.dumps({'stage':args.stage,'label':args.label,'repeat':repeat,'id':row['id'],
                    'tokens':len(row['tokens']), 'peak_bytes':mx.get_peak_memory()}),flush=True)
                # Row receipts survive interruption and never overwrite a prior repeat.
                if per_row:
                    write_new(args.output/'rows'/f"{repeat}-{row['id']}.json",per_row[-1])
            if per_row:
                totals = {}
                for mode in ('legacy','fp32'):
                    totals[mode] = {}
                    for region in (*REGIONS, 'aggregate'):
                        total = sum(r['sums'][mode][region][0] for r in per_row)
                        count = sum(r['sums'][mode][region][1] for r in per_row)
                        totals[mode][region] = {'sum':total,'tokens':count,'mean':total/count if count else None}
                metrics.append({'repeat':repeat,'totals':totals,'rows':per_row})
                write_new(args.output/f'repeat-{repeat}.json',metrics[-1])
        receipt.update(status='complete', captures_sha256=captures, repeats=metrics,
            same_capture_comparison='repeat zero: original-dtype scorer versus evaluation-only FP32 KL/reductions, identical gathered teacher/student logits',
            objective='top-1024 teacher KL at temperature 2, equal token weighting including context; partitioned region reports are diagnostics')
    except Exception as error:
        receipt.update(status='failed', error=repr(error), captures_sha256=captures if 'captures' in locals() else {})
        raise
    finally:
        receipt.update(elapsed_s=time.monotonic()-started, mlx_peak_bytes=mx.get_peak_memory())
        write_new(args.output/'receipt.json',receipt)



def summarize(receipts):
    import math
    if set(receipts) != {'C','mxfp4'}:
        raise ValueError('Missing pinned baseline')
    first = receipts['C']
    for receipt in receipts.values():
        if receipt['status'] != 'complete' or receipt['updates'] != 0 or len(receipt['repeats']) != 5 or any(
                receipt[key] != first[key] for key in ('data_manifest_sha256','protocol_sha256')):
            raise ValueError('Incomplete baseline or changed teacher contexts')
    report = {'version':'calibration-v2-eval-1.0', 'models':{}, 'regions':{},
              'reserved_used':False, 'updates':0}
    for name, receipt in receipts.items():
        report['models'][name] = {}
        for mode in ('legacy','fp32'):
            report['models'][name][mode] = {}
            for region in (*REGIONS, 'aggregate'):
                data = [r['totals'][mode][region] for r in receipt['repeats']]
                values = [v['mean'] for v in data]
                counts = [v['tokens'] for v in data]
                if len(set(counts)) != 1 or counts[0] <= 0 or any(not math.isfinite(v) for v in values):
                    raise ValueError('Invalid token counts or nonfinite repeat loss')
                report['models'][name][mode][region] = {'values':values,
                    'mean':statistics.mean(values), 'min':min(values), 'max':max(values),
                    'range':max(values)-min(values), 'tokens':counts[0]}
        report['models'][name]['same_capture_delta_fp32_minus_legacy'] = {
            region:receipt['repeats'][0]['totals']['fp32'][region]['mean']-
                   receipt['repeats'][0]['totals']['legacy'][region]['mean']
            for region in (*REGIONS, 'aggregate')}
    for region in REGIONS:
        c = report['models']['C']['fp32'][region]
        m = report['models']['mxfp4']['fp32'][region]
        if c['tokens'] != m['tokens']:
            raise ValueError('Baseline prediction targets differ')
        report['regions'][region] = {
            'relevant_C_gap':region != 'context' and c['mean'] >= .01 and c['mean'] > c['range'],
            'C_trails_mxfp4':region != 'context' and c['min'] > m['max'] and
                c['mean']-m['mean'] > max(c['range'],m['range']),
            'C_minus_mxfp4':c['mean']-m['mean']}
    report['relevant_gap'] = any(r['relevant_C_gap'] for r in report['regions'].values())
    report['C_trails_mxfp4'] = any(r['C_trails_mxfp4'] for r in report['regions'].values())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','teacher','baseline','summary'])
    p.add_argument('--source',type=Path,default=Path('work/source'))
    p.add_argument('--data',type=Path,default=ROOT/'prepared')
    p.add_argument('--cache',type=Path,default=Path('work/calibration-v2-teacher'))
    p.add_argument('--model',type=Path)
    p.add_argument('--label',choices=['C','mxfp4'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error('Refusing existing evidence directory')
    if args.stage == 'baseline' and (args.model is None or args.label is None):
        p.error('Baseline requires the pinned model path and label')
    if args.stage == 'summary':
        receipts = {label:json.loads((Path('results/calibration-v2')/(label+'-receipt.json')).read_text())
                    for label in ('C','mxfp4')}
        write_new(args.output, summarize(receipts))
    elif args.stage=='prepare':
        prepare(args)
    else:
        capture(args)


if __name__=='__main__':
    main()

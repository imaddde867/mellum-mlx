"""Bounded cached-target DWQ; preserve packed weights, routers and BF16 exclusions."""
import argparse
import json
import math
import statistics
import shutil
import time
from pathlib import Path

from convert import REVISION, realized_map, sha256
from nonexpert import binding, write_new
from validate import check_hashes


def check_changes(changed, plan):
    if not changed:
        raise ValueError('No calibration parameter changed')
    for key in changed:
        path, name = key.rsplit('.', 1)
        policy = plan.get(path, {})
        if name not in {'scales', 'biases'} or policy.get('mode') != 'affine' or policy.get('bits', 8) >= 8:
            raise ValueError(f'DWQ changed protected parameter: {key}')


def require_pristine_c(conversion):
    if conversion.get('recipe') != 'C' or conversion.get('source_revision') != REVISION or 'calibration' in conversion:
        raise ValueError('Diagnostic requires pristine BF16-origin C')


def select_checkpoint(baseline, checkpoints):
    def valid(values):
        return len(values) >= 3 and all(math.isfinite(x) for x in values)
    if not valid(baseline) or any(step not in {8,16,24,32} or
            len(values) != len(baseline) or not valid(values) for step,values in checkpoints.items()):
        raise ValueError('Incomplete, nonfinite or unscheduled validation')
    baseline_mean = statistics.mean(baseline)
    eligible = {step:statistics.mean(values) for step,values in checkpoints.items()
        if max(values) < min(baseline) and baseline_mean - statistics.mean(values) >
           max(max(baseline)-min(baseline), max(values)-min(values))}
    return min(eligible, key=eligible.get) if eligible else 0


def validation_repeats(model, params, targets, data):
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm.tuner.trainer import iterate_batches
    from mlx_lm.tuner.losses import kl_div_loss
    if params is not None:
        model.update(tree_map(lambda x: x.astype(mx.bfloat16), params))
    values = []
    for _ in range(5):
        total, tokens = 0.0, 0
        for index,(batch,lengths) in enumerate(iterate_batches(data, 1, 256, seed=123)):
            batch = batch[:, :-1]
            teacher, ids = targets(batch, index, split='valid')
            logits = mx.take_along_axis(model(batch), ids, axis=-1)
            losses = kl_div_loss(0.5 * logits, 0.5 * teacher)
            mask = mx.arange(1, 1 + teacher.shape[1]) < lengths[:, 1:]
            loss_sum = (mask * losses).sum()
            count = mask.sum()
            mx.eval(loss_sum, count)
            total += loss_sum.item(); tokens += count.item()
        loss = total / tokens
        if tokens != 915 or not math.isfinite(loss):
            raise ValueError('Invalid diagnostic validation objective')
        values.append(loss)
    return values


def calibration(root):
    manifest = json.loads((root/'manifest.json').read_text())
    datasets, ids, texts = {}, set(), set()
    for split in ['train','valid']:
        path = root/(split+'.jsonl')
        if sha256(path) != manifest['splits'][split]['sha256']:
            raise ValueError('Calibration hash mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != manifest['splits'][split]['rows']:
            raise ValueError('Incomplete calibration split')
        for row in rows:
            if row['id'] in ids or row['text'] in texts or not 64 <= len(row['tokens']) <= 256:
                raise ValueError('Duplicate or truncated calibration sample')
            ids.add(row['id']); texts.add(row['text'])
        datasets[split] = [(r['tokens'], 0) for r in rows]
    return manifest, datasets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['targets','train','diagnostic','verify'])
    p.add_argument('--student', type=Path)
    p.add_argument('--destination', type=Path)
    p.add_argument('--cache', type=Path, default=Path('work/refinement-v1-targets'))
    p.add_argument('--data', type=Path, default=Path('fixtures/refinement-v1'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or (args.stage == 'targets' and args.cache.exists()):
        p.error('Refusing existing evidence or teacher cache')
    if args.stage in {'train','diagnostic'} and (not args.student or not args.destination or args.destination.exists()):
        p.error('Training requires a student and new destination')
    if args.stage == 'verify' and not args.student:
        p.error('Verification requires the saved diagnostic artifact')
    import mlx.core as mx
    import mlx.optimizers as optimizers
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.utils import load_tokenizer, save
    from mlx_lm.quant.dwq import compute_dwq_targets, dwq_quantize
    import mlx_lm.quant.dwq as upstream
    mx.random.seed(123)
    source = Path('work/source')
    native = json.loads(Path('artifacts/mellum2.1-affine-4bit-g64/conversion.json').read_text())
    if native['source_revision'] != REVISION:
        raise ValueError('Source revision mismatch')
    check_hashes(source, native['source_sha256'])
    manifest, data = calibration(args.data)
    tokenizer = load_tokenizer(source)
    for split in ['train','valid']:
        for row in map(json.loads, (args.data/(split+'.jsonl')).read_text().splitlines()):
            if tokenizer.encode(row['text'], add_special_tokens=False) != row['tokens']:
                raise ValueError('Calibration tokenizer mismatch')
    cache_binding = {'data':manifest, 'tokenizer_sha256':sha256(source/'tokenizer.json'),
                     'source_revision':REVISION, 'source_sha256':native['source_sha256'],
                     'dwq_sha256':sha256(Path(upstream.__file__))}
    report = {'status':'running', 'stage':args.stage, 'cache_binding':cache_binding,
              'script_sha256':sha256(Path(__file__)), 'learning_rate':1e-7 if args.stage in {'diagnostic','verify'} else 1e-6,
              'planned_updates':32, 'updates':0, 'gradient_checkpoint':args.stage in {'train','diagnostic','verify'}}
    started = time.monotonic()
    try:
        if args.stage == 'targets':
            model, _ = load(str(source))
            compute_dwq_targets(model, args.cache, data['train'], data['valid'], 1, 256, 123)
            files = {str(f.relative_to(args.cache)):sha256(f) for f in args.cache.glob('*/*.safetensors')}
            if len(files) != 40:
                raise ValueError('Incomplete teacher cache')
            write_new(args.cache/'manifest.json', {**cache_binding,'cache_sha256':files})
            report['cache_sha256'] = files
        else:
            cached = json.loads((args.cache/'manifest.json').read_text())
            files = cached.pop('cache_sha256')
            if cached != cache_binding or len(files) != 40 or any(sha256(args.cache/f) != h for f,h in files.items()):
                raise ValueError('Stale or altered teacher cache')
            report['student'] = binding(args.student)
            conversion = json.loads((args.student/'conversion.json').read_text())
            check_hashes(args.student, conversion['artifact_sha256'])
            if args.stage == 'diagnostic':
                require_pristine_c(conversion)
                original_c = json.loads(Path('results/nonexpert-v1/integrity-C.json').read_text())
                if report['student']['weights_sha256'] != original_c['weights_sha256']:
                    raise ValueError('Diagnostic C differs from original integrity binding')
            model, tokenizer, config = load(str(args.student), return_config=True)
            plan = realized_map(config['quantization'], conversion['precision_map'])
            model.freeze()
            before = dict(tree_flatten(model.parameters()))
            mx.eval(before)
            def targets(_, index, split):
                value = mx.load(args.cache/split/f'{index:010d}.safetensors')
                return value['logits'], value['indices']
            if args.stage == 'verify':
                diagnostic = conversion['calibration']['diagnostic']
                model.train()
                from mlx_lm.tuner.trainer import grad_checkpoint
                grad_checkpoint(model.layers[0])
                reloaded = validation_repeats(model, None, targets, data['valid'])
                step = diagnostic['selected_step']
                advance = step > 0 and select_checkpoint(diagnostic['baseline'], {step:reloaded}) == step
                report.update(reloaded_validation=reloaded, selected_step=step,
                    advance_to_development=advance, baseline=diagnostic['baseline'],
                    selected_before_save=diagnostic['selected_validation'], status='complete')
                return
            diagnostic = args.stage == 'diagnostic'
            best_params = None
            if diagnostic:
                from mlx.utils import tree_map
                from mlx_lm.tuner.trainer import grad_checkpoint
                model.train()
                def unfreeze(_, module):
                    if hasattr(module, 'bits') and hasattr(module, 'group_size') and module.mode == 'affine' and module.bits < 8:
                        module.unfreeze(keys=['scales','biases'], recurse=False)
                model.apply_to_modules(unfreeze)
                # Apply the same upstream checkpoint wrapper once, before all validation.
                grad_checkpoint(model.layers[0])
                best_params = model.trainable_parameters()
                baseline = validation_repeats(model, None, targets, data['valid'])
                measured = {}
                report['diagnostic'] = {'baseline':baseline, 'checkpoints':measured, 'selected_step':0}
                print(json.dumps({'step':0,'validation':baseline}), flush=True)
            class FiniteAdam(optimizers.Adam):
                def apply_gradients(self, gradients, parameters):
                    values = [mx.all(mx.isfinite(v)) for _,v in tree_flatten(gradients)]
                    if not mx.all(mx.stack(values)).item():
                        raise ValueError('Nonfinite DWQ gradient')
                    result = super().apply_gradients(gradients, parameters)
                    report['updates'] += 1
                    if diagnostic and report['updates'] % 8 == 0:
                        nonlocal best_params
                        step = report['updates']
                        measured[step] = validation_repeats(model, result, targets, data['valid'])
                        selected = select_checkpoint(baseline, measured)
                        if selected == step:
                            best_params = tree_map(lambda x: x.astype(mx.bfloat16), result)
                            mx.eval(best_params)
                        report['diagnostic']['selected_step'] = selected
                        print(json.dumps({'step':step,'validation':measured[step],
                                          'selected_step':selected}), flush=True)
                    return result
            dwq_quantize(model, targets, FiniteAdam(learning_rate=report['learning_rate'], bias_correction=True),
                         data['train'], data['valid'], 1, 256, 123, gradient_checkpoint=not diagnostic)
            if report['updates'] != 32:
                raise ValueError('Incomplete DWQ training pass')
            if diagnostic:
                model.update(best_params)
                selected = report['diagnostic']['selected_step']
                selected_validation = baseline if selected == 0 else measured[selected]
                report['diagnostic'].update(selected_validation=selected_validation,
                    advance_to_development=selected > 0,
                    rule='non-overlapping five-repeat intervals and mean reduction exceeding both observed ranges')
                del best_params
            after = dict(tree_flatten(model.parameters()))
            if set(before) != set(after):
                raise ValueError('DWQ changed parameter keys')
            changed = []
            for key,value in after.items():
                original = before[key]
                if value.dtype != original.dtype or value.shape != original.shape:
                    raise ValueError(f'DWQ changed tensor schema: {key}')
                if value.dtype != mx.uint32 and not mx.all(mx.isfinite(value)).item():
                    raise ValueError(f'Nonfinite calibrated parameter: {key}')
                if not mx.all(value == original).item():
                    changed.append(key)
            if diagnostic and report['diagnostic']['selected_step'] == 0:
                if changed:
                    raise ValueError('Step-zero checkpoint differs from pristine C')
            else:
                check_changes(changed, plan)
            before.clear(); after.clear()
            model.eval()
            save(args.destination, args.student, model, tokenizer, config)
            for pattern in ['tokenizer*','*.jinja','*.model','special_tokens_map.json','LICENSE*','NOTICE*']:
                for path in args.student.glob(pattern):
                    if path.is_file():
                        shutil.copy2(path, args.destination/path.name)
            # Preserve provenance while binding a new artifact to its uncalibrated parent.
            artifacts = {f.name:sha256(f) for f in args.destination.iterdir() if f.is_file() and f.name != 'README.md'}
            conversion.update(artifact_sha256=artifacts,
                weight_bytes=sum(f.stat().st_size for f in args.destination.glob('*.safetensors')),
                calibration={'method':'DWQ', 'parent':report['student'],
                    'cache_manifest_sha256':sha256(args.cache/'manifest.json'),
                    'data':manifest, 'changed_parameters':changed, 'finite_gradients':True,
                    'routers_packed_weights_and_excluded_tensors_unchanged':True})
            if diagnostic:
                conversion['calibration']['diagnostic'] = report['diagnostic']
                conversion['calibration']['learning_rate'] = report['learning_rate']
            write_new(args.destination/'conversion.json', conversion)
            report.update(changed_parameters=changed, protected_parameters_unchanged=True,
                          finite_gradients=True, artifact=binding(args.destination))
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report.update(elapsed_s=time.monotonic()-started, mlx_peak_bytes=mx.get_peak_memory())
        write_new(args.output, report)


if __name__ == '__main__':
    main()

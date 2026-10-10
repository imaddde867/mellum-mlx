"""Summarize complete, matched screens without dropping failed attempts."""
import argparse
import json
import math
from pathlib import Path
import statistics

from convert import sha256
from nonexpert import frozen, write_new

LABELS = ['bf16', '4bit', 'mxfp4', 'A', 'B', 'C']


def paired(left, right):
    if [r['id'] for r in left] != [r['id'] for r in right]:
        raise ValueError('Paired task IDs differ')
    groups = {'left_only':[], 'right_only':[], 'both_passed':[], 'neither_passed':[]}
    for a,b in zip(left,right):
        key = ('both_passed' if b['passed'] else 'left_only') if a['passed'] else (
            'right_only' if b['passed'] else 'neither_passed')
        groups[key].append(a['id'])
    wins, losses = len(groups['left_only']), len(groups['right_only'])
    n = wins + losses
    p = min(1, 2 * sum(math.comb(n,k) for k in range(min(wins,losses)+1)) / 2**n) if n else 1
    return {**groups, 'exact_mcnemar_p':p,
            'generation_s_delta':sum(r['elapsed_s'] for r in left)-sum(r['elapsed_s'] for r in right),
            'completion_tokens_delta':sum(r['completion_tokens'] for r in left)-sum(r['completion_tokens'] for r in right)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/nonexpert-v1'))
    p.add_argument('--labels', nargs='+', default=LABELS)
    args = p.parse_args()
    if 'mxfp4' not in args.labels or len(set(args.labels)) != len(args.labels):
        p.error('Include MXFP4 and unique model labels')
    reports = {label:json.loads((args.results/f'screen-{label}.json').read_text()) for label in args.labels}
    ids = [r['id'] for r in frozen()]
    reference = reports[args.labels[0]]
    rows = []
    for label,d in reports.items():
        if d['status']!='complete' or [r['id'] for r in d['rows']]!=ids:
            raise ValueError('Incomplete screen')
        for key in ['protocol','script_sha256','server_sha256','packages','fixture_manifest_sha256']:
            if d[key] != reference[key]:
                raise ValueError(f'Incompatible comparison: {key}')
        raw = args.results/f'screen-{label}.raw.jsonl'
        if sha256(raw) != d['raw_sha256']:
            raise ValueError('Raw evidence binding mismatch')
        rows.append({'label':label, **{k:d[k] for k in ['weight_bytes','passed','generation_s',
            'failed_generation_s','completion_tokens','truncated','empty','memory','elapsed_with_startup_and_tests_s']},
            'median_completion_tokens':statistics.median(r['completion_tokens'] for r in d['rows']),
            'vs_mxfp4':paired(d['rows'],reports['mxfp4']['rows']),
            'vs_native4':paired(d['rows'],reports['4bit']['rows']) if '4bit' in reports else None,
            'vs_C':paired(d['rows'],reports['C']['rows']) if 'C' in reports else None,
            'screen_receipt_sha256':sha256(args.results/f'screen-{label}.json')})
    summary = {'rows':rows, 'all_models_generation_s':sum(d['generation_s'] for d in reports.values()),
        'all_models_elapsed_with_startup_and_tests_s':sum(d['elapsed_with_startup_and_tests_s'] for d in reports.values()),
        'paired_tasks':[{ 'id':task_id, **{label:{k:d['rows'][i][k] for k in
            ['passed','completion_tokens','elapsed_s','truncated','empty']} for label,d in reports.items()}}
            for i,task_id in enumerate(ids)],
        'limitations':'One greedy CUDA run per model on 24 authored synthetic tasks. Exact McNemar p values are descriptive, unadjusted for model comparisons; not superiority proof.'}
    write_new(args.results/'summary.json', summary)
    print('| Model | Weight GB | Passed | Gen s | Failed s | Tokens | Trunc / empty | CUDA MLX GB |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for r in rows:
        print(f"| {r['label']} | {r['weight_bytes']/1e9:.3f} | {r['passed']}/24 | {r['generation_s']:.1f} | {r['failed_generation_s']:.1f} | {r['completion_tokens']:,} | {r['truncated']} / {r['empty']} | {r['memory']['mlx_peak_bytes']/1e9:.2f} |")


if __name__=='__main__':
    main()

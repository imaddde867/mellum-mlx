"""Paired HumanEval+ comparison of one new run against the committed MXFP4 run (needs numpy, scipy).
Usage: python pair_audit.py RUN_DIR_WITH_nvfp4.raw.jsonl_AND_evaluation-nvfp4 NAME"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, wilcoxon

run, name = Path(sys.argv[1]), sys.argv[2]; ids = [f'HumanEval/{i}' for i in range(164)]
ref = Path('results/coding/humaneval-plus/mxfp4')
def load(results, raw):
    ev = json.loads(results.read_text())['eval']
    rows = {json.loads(l)['task_id']: json.loads(l) for l in raw.read_text().splitlines()}
    return (np.array([ev[i][0]['plus_status'] == 'pass' for i in ids]),
            np.array([rows[i]['usage']['completion_tokens'] for i in ids]),
            np.array([rows[i]['finish_reason'] == 'length' for i in ids]))
E, T, F = {}, {}, {}
E['mxfp4'], T['mxfp4'], F['mxfp4'] = load(ref / 'eval-results.json', ref / 'samples.raw.jsonl')
E[name], T[name], F[name] = load(run / f'evaluation-{name}' / 'samples-sanitized_eval_results.json', run / f'{name}.raw.jsonl')
for c in E:
    print(f'{c:6} plus_pass={E[c].sum()} truncated={F[c].sum()} median_tokens={int(np.median(T[c]))} '
          f'solved<=2048/4096={int((E[c] & (T[c] <= 2048)).sum())}/{int((E[c] & (T[c] <= 4096)).sum())}')
a, b = name, 'mxfp4'
x, y = int((E[a] & ~E[b]).sum()), int((~E[a] & E[b]).sum())
print(f'{a} only={x} mxfp4 only={y} exact McNemar p={binomtest(x, x + y).pvalue:.3g}')
m = ~F[a] & ~F[b]; r = T[a][m] / T[b][m]
print(f'tokens {a}/mxfp4 (both stopped, n={m.sum()}): median {np.median(r):.2f} geo-mean {np.exp(np.log(r).mean()):.2f} '
      f'Wilcoxon p={wilcoxon(T[a][m], T[b][m]).pvalue:.2g}')

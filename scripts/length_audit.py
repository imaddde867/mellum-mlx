"""Reanalyse committed HumanEval+ evidence: paired outcomes, reasoning length, pass@budget.
Usage: python length_audit.py results/coding/humaneval-plus   (needs numpy, scipy)"""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, wilcoxon

root = Path(sys.argv[1]); C = ['bf16', '4bit', '6bit', 'mxfp4']; ids = [f'HumanEval/{i}' for i in range(164)]
E, T, F, R = {}, {}, {}, {}
for c in C:
    ev = json.loads((root / c / 'eval-results.json').read_text())['eval']
    raw = {json.loads(l)['task_id']: json.loads(l) for l in (root / c / 'samples.raw.jsonl').read_text().splitlines()}
    E[c] = np.array([ev[i][0]['plus_status'] == 'pass' for i in ids])
    T[c] = np.array([raw[i]['usage']['completion_tokens'] for i in ids])
    F[c] = np.array([raw[i]['finish_reason'] == 'length' for i in ids])
    R[c] = [raw[i]['message'].get('reasoning') or '' for i in ids]
print('candidate  plus  trunc  median_tokens  But/resp  Actually/resp  pass@2048/4096/8192')
def per_response(c, word):
    pattern = re.compile(r"\b" + word + r"\b")
    return np.mean([len(pattern.findall(r)) for r in R[c]])
for c in C:
    budget = '/'.join(str(int((E[c] & (T[c] <= b)).sum())) for b in (2048, 4096, 8192))
    print(f"{c:9} {E[c].sum():5} {F[c].sum():6} {int(np.median(T[c])):13} "
          f"{per_response(c, 'But'):9.1f} {per_response(c, 'Actually'):13.1f}   {budget}")
print('\npaired HumanEval+ (exact McNemar): only_a, only_b, p')
for a, b in [('4bit', 'mxfp4'), ('4bit', '6bit'), ('6bit', 'mxfp4'), ('6bit', 'bf16'), ('mxfp4', 'bf16'), ('4bit', 'bf16')]:
    x, y = int((E[a] & ~E[b]).sum()), int((~E[a] & E[b]).sum())
    print(f'  {a:5} vs {b:5}: {x:2} {y:2}  p={binomtest(x, x + y).pvalue:.3g}')
print('\npaired completion-token ratio, tasks where both stopped')
for a, b in [('4bit', 'bf16'), ('6bit', 'bf16'), ('mxfp4', 'bf16'), ('4bit', 'mxfp4')]:
    m = ~F[a] & ~F[b]; r = T[a][m] / T[b][m]
    print(f'  {a}/{b}: n={m.sum()} median {np.median(r):.2f} geo-mean {np.exp(np.log(r).mean()):.2f} Wilcoxon p={wilcoxon(T[a][m], T[b][m]).pvalue:.1e}')

"""HumanEval+ tasks solved within a completion-token budget, as a static SVG.
Usage: python budget_chart.py results/coding/humaneval-plus docs/assets/solved-by-budget.svg"""
import json, sys
from pathlib import Path

SERIES = [('bf16', 'BF16 (24.3 GB)'), ('6bit', '6-bit (9.9 GB)'),
          ('mxfp4', 'MXFP4 (6.5 GB)'), ('4bit', 'naive 4-bit (6.8 GB)')]
LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
DARK = ['#3987e5', '#d95926', '#199e70', '#c98500']
W, H, L, R, T, B = 720, 400, 56, 170, 48, 48
XMAX, YMAX = 8192, 164


def solved_tokens(root, name):
    ids = [f'HumanEval/{i}' for i in range(164)]
    ev = json.loads((root / name / 'eval-results.json').read_text())['eval']
    raw = {json.loads(l)['task_id']: json.loads(l)
           for l in (root / name / 'samples.raw.jsonl').read_text().splitlines()}
    return sorted(raw[i]['usage']['completion_tokens'] for i in ids
                  if ev[i][0]['plus_status'] == 'pass')


def x(v): return L + v / XMAX * (W - L - R)
def y(v): return T + (1 - v / YMAX) * (H - T - B)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    css = '\n'.join(f'.s{i}{{stroke:{c}}} .d{i}{{fill:{c}}}' for i, c in enumerate(LIGHT))
    dark = '\n'.join(f'.s{i}{{stroke:{c}}} .d{i}{{fill:{c}}}' for i, c in enumerate(DARK))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">',
             f'<style>.bg{{fill:#fcfcfb}} .t1{{fill:#1a1a19}} .t2{{fill:#5f5e57}} .grid{{stroke:#e4e3dc}} {css}\n'
             f'@media (prefers-color-scheme: dark){{.bg{{fill:#1a1a19}} .t1{{fill:#fff}} .t2{{fill:#c3c2b7}} .grid{{stroke:#3a3a37}} {dark}}}</style>',
             f'<rect class="bg" width="{W}" height="{H}"/>',
             f'<text class="t1" x="{L}" y="22" font-size="15" font-weight="600">HumanEval+ tasks solved within a token budget</text>',
             f'<text class="t2" x="{L}" y="38">Mellum2.1 Thinking, greedy, 164 tasks, RTX 5090. Tokens include reasoning.</text>']
    for v in range(0, YMAX + 1, 40):
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                     f'<text class="t2" x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for v in range(0, XMAX + 1, 2048):
        parts.append(f'<text class="t2" x="{x(v):.1f}" y="{H - B + 18}" text-anchor="middle">{v:,}</text>')
    parts.append(f'<text class="t2" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">completion-token budget</text>')
    ends = []
    for i, (name, label) in enumerate(SERIES):
        tokens = solved_tokens(root, name)
        pts, n = [f'{x(0):.1f},{y(0):.1f}'], 0
        for t in tokens:
            pts.append(f'{x(t):.1f},{y(n):.1f}')
            n += 1
            pts.append(f'{x(t):.1f},{y(n):.1f}')
        pts.append(f'{x(XMAX):.1f},{y(n):.1f}')
        parts.append(f'<polyline class="s{i}" fill="none" stroke-width="2" stroke-linejoin="round" points="{" ".join(pts)}"/>')
        ends.append([y(n), i, f'{label}: {n}'])
    ends.sort()
    for k in range(1, len(ends)):  # keep direct labels at least 15px apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 15)
    for ly, i, text in ends:
        parts.append(f'<circle class="d{i}" cx="{W - R + 10}" cy="{ly - 4:.1f}" r="4"/>'
                     f'<text class="t1" x="{W - R + 18}" y="{ly:.1f}">{text}</text>')
    parts.append('</svg>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(parts) + '\n')


if __name__ == '__main__':
    main()

# Mellum2.1 on Apple Silicon

Benchmarked MLX quantizations of JetBrains' [Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking), a 12B mixture-of-experts coding model with 2.5B active parameters. Every number here comes from a full 164-task HumanEval+ run or a measured 16 GB M4 MacBook Pro. Raw outputs, receipts and checksums are in this repo.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-6--bit%20MLX-yellow)](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64)
[![mlx-community](https://img.shields.io/badge/mlx--community-6bit-blue)](https://huggingface.co/mlx-community/Mellum2.1-12B-A2.5B-Thinking-6bit)
[![Checks](https://github.com/imaddde867/mellum-mlx/actions/workflows/checks.yml/badge.svg)](https://github.com/imaddde867/mellum-mlx/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-lightgrey)](LICENSE)

- **Model:** [imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64), 9.87 GB. The same files are also on [mlx-community](https://huggingface.co/mlx-community/Mellum2.1-12B-A2.5B-Thinking-6bit).
- **Write-up:** [Same size, twice the thinking: what 4-bit quantization did to a reasoning model](docs/WRITEUP.md)

## Quickstart

```sh
pip install 'mlx-lm==0.32.0'
python -m mlx_lm generate --model mlx-community/Mellum2.1-12B-A2.5B-Thinking-6bit \
  --prompt 'Find the bug: def mean(xs): return sum(xs) / len(xs) - 1' \
  --temp 0.6 --top-p 0.95 --top-k 20 --max-tokens 16384
```

Always set `--max-tokens`. Reasoning and the answer share the budget, and the MLX-LM server defaults to 512 tokens.

## Which build to use

| Your Mac | Use | Why |
| --- | --- | --- |
| Any, including 16 GB | [randmaru MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4) (6.46 GB) | Statistically tied with 6-bit on HumanEval+, 34 to 41% faster decode, 7.2 GB peak memory. |
| 24 GB or more | [6-bit](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64) (9.87 GB) | Closest to BF16 of the tested builds. The edge over MXFP4 isn't significant. Swaps on 16 GB. |

Four attempts to beat MXFP4 at 16 GB tied or lost: mixed 4/6-bit, group-32 experts, DWQ calibration, and MXFP4 with 8-bit attention. A fifth, NVFP4, is measurably closer to BF16 at near-equal speed but ties on HumanEval+; see finding 2.

## Results

![HumanEval+ tasks solved within a token budget](docs/assets/solved-by-budget.svg)

| Build | Weights | HumanEval+ | Budget hits | Median tokens | M4 decode tok/s, 1K / 16K |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 original | 24.3 GB | 153 (93.3%) | 3 | 1,571 | n/a |
| **6-bit (this release)** | 9.87 GB | 146 (89.0%) | 3 | 1,521 | 47.2 / 42.8 |
| randmaru MXFP4 | 6.46 GB | 144 (87.8%) | 6 | 1,544 | 66.7 / 57.4 |
| MXFP4 + 8-bit attention | 6.77 GB | 145 (88.4%) | 3 | 1,632 | 55.0 / 49.0 |
| NVFP4 | 6.84 GB | 145 (88.4%) | 6 | 1,674 | 63.9 / 55.8 |
| Affine 4-bit | 6.84 GB | 133 (81.1%) | 18 | 3,242 | n/a |

HumanEval+ setup: all 164 tasks on one RTX 5090, upstream chat template, greedy decoding, 8,192-token budget that includes reasoning, one sample per task. M4 setup: 14-inch MacBook Pro with 16 GB, 256 output tokens, mean of three trials after warmup. Each unreleased candidate ran in the same session as MXFP4. NVFP4 was interleaved A-B-B-A with MXFP4 on a 68 W charger, with MXFP4 averaging 67.2 / 58.3 in that session. Paired statistics and limitations are in [docs/RESULTS.md](docs/RESULTS.md).

## Findings

1. **Affine 4-bit makes the model think 2.2× longer.** Its completions were 2.2× BF16's length on the same tasks (Wilcoxon p = 5×10⁻²⁵) and hit the budget mid-reasoning on 18 tasks. MXFP4 is smaller and stays at BF16's length. A plain pass@1 score hides this. [Write-up](docs/WRITEUP.md).
2. **Closer to BF16 isn't automatically better to use.** NVFP4 is about 30% closer to BF16 than MXFP4 by KL divergence and decodes at 95 to 96% of its speed. It still only tied on HumanEval+ and wrote 12% longer completions.
3. **Extra attention bits cost speed in a small-active MoE.** Attention is about a quarter of the weights read for each token. Raising it to 8-bit on top of MXFP4 cost 14 to 17% decode speed and gained no measurable accuracy.
4. **MLX-LM 0.32.0 silently drops tool calls it can't parse.** The response keeps `finish_reason: "tool_calls"` with an empty `tool_calls` list. A tested patch is in [patches/](patches/), and an upstream fix is open in [mlx-lm#1957](https://github.com/ml-explore/mlx-lm/pull/1957).
5. **Reasoning gets lost in tool loops.** Mellum's template replays prior reasoning only from `reasoning_content`, but the server returns it as `reasoning`.

## Reproduce

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # Linux + CUDA 13: also pip install 'mlx[cuda13]==0.32.3'
export HF_HOME="$PWD/work/hf-cache"

python scripts/convert.py --bits 6         # or --recipe M; downloads 24.3 GB of BF16 weights
python scripts/validate.py artifacts/mellum2.1-affine-6bit-g64
# HumanEval+: export the pinned task file first (docs/CODING_PROTOCOL.md)
python scripts/coding_probe.py artifacts/mellum2.1-affine-6bit-g64 --output work/evalplus/6bit.jsonl
python scripts/evaluate_coding.py work/evalplus/6bit.jsonl --output work/evalplus/eval-6bit
python scripts/benchmark.py artifacts/mellum2.1-affine-6bit-g64 --context 16384 --max-tokens 256 --trials 3 \
  --output results/local/6bit-16k.json
pip install scipy && python scripts/length_audit.py results/coding/humaneval-plus
```

Conversion, generation and evaluation write receipts that record the source revision, package versions and SHA-256 checksums. No report overwrites an earlier one. Generated code runs only in an offline, locked-down EvalPlus container. The setup is described in [docs/CODING_PROTOCOL.md](docs/CODING_PROTOCOL.md).

## Repository map

| Path | Contents |
| --- | --- |
| [`scripts/`](scripts/) | Conversion, validation, HumanEval+ generation and evaluation, M4 benchmarks, analysis |
| [`results/`](results/) | Raw responses, per-task outcomes, benchmark JSON, receipts |
| [`docs/`](docs/) | [Results](docs/RESULTS.md), [write-up](docs/WRITEUP.md), [release plan](docs/PLAN.md), [unshipped experiments](docs/research/) |
| [`patches/`](patches/) | MLX-LM serving and tool-call fixes used in agent trials |
| [`tests/`](tests/) | Unit tests for conversion recipes, receipts and evaluation guards |

## Credits

The model was created and trained by JetBrains: [Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) at revision `92ddae9`, under Apache-2.0. The runtime is [MLX](https://github.com/ml-explore/mlx) 0.32.3 with [MLX-LM](https://github.com/ml-explore/mlx-lm) 0.32.0. The MXFP4 comparator is by [randmaru](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4), at revision `4ce0d28`. The quantization methods come from upstream. This repo contributes the measurement, integration and analysis. Scripts are MIT-licensed, which does not relicense the model weights.

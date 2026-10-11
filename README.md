# Mellum2.1 on Apple Silicon

MLX builds of JetBrains' [Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking), a 12B mixture-of-experts coding model with 2.5B active parameters: **NVFP4** for 16 GB Macs and **6-bit** for 24 GB or more. Each build is benchmarked against the BF16 original on the full 164-task HumanEval+ and measured on a 16 GB M4 MacBook Pro. Raw outputs, receipts and checksums are in this repo.

[![NVFP4](https://img.shields.io/badge/%F0%9F%A4%97%20NVFP4-6.84%20GB-yellow)](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-nvfp4)
[![6-bit](https://img.shields.io/badge/%F0%9F%A4%97%206--bit-9.87%20GB-yellow)](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64)
[![mlx-community](https://img.shields.io/badge/mlx--community-6bit-blue)](https://huggingface.co/mlx-community/Mellum2.1-12B-A2.5B-Thinking-6bit)
[![Checks](https://github.com/imaddde867/mellum-mlx/actions/workflows/checks.yml/badge.svg)](https://github.com/imaddde867/mellum-mlx/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-lightgrey)](LICENSE)

## Quickstart

```sh
pip install 'mlx-lm==0.32.0'
python -m mlx_lm generate --model imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-nvfp4 \
  --prompt 'Find the bug: def mean(xs): return sum(xs) / len(xs) - 1' \
  --temp 0.6 --top-p 0.95 --top-k 20 --max-tokens 16384
```

Always set `--max-tokens`. Reasoning and the answer share the budget, and the MLX-LM server defaults to 512 tokens.

## Which build to use

| Your Mac | Build | Why |
| --- | --- | --- |
| 16 GB | [NVFP4](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-nvfp4), 6.84 GB | Closest to BF16 of the ~7 GB builds measured. 7.6 GB peak memory at 16K context. |
| 16 GB, speed first | [randmaru MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4), 6.46 GB | About 5% faster decoding and 12% shorter completions than NVFP4, with the same HumanEval+ score. |
| 24 GB or more | [6-bit](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64), 9.87 GB | Most HumanEval+ tasks solved of the quantized builds. Swaps on 16 GB. |

## Results

![HumanEval+ tasks solved within a token budget](docs/assets/solved-by-budget.svg)

| Build | Weights | HumanEval+ | Budget hits | Median tokens | M4 decode tok/s, 1K / 16K |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 original | 24.3 GB | 153 (93.3%) | 3 | 1,571 | n/a |
| **NVFP4** (released) | 6.84 GB | 145 (88.4%) | 6 | 1,674 | 63.9 / 55.8 |
| **6-bit** (released) | 9.87 GB | 146 (89.0%) | 3 | 1,521 | 47.2 / 42.8 |
| randmaru MXFP4 | 6.46 GB | 144 (87.8%) | 6 | 1,544 | 66.7 / 57.4 |
| MXFP4 + 8-bit attention (not released) | 6.77 GB | 145 (88.4%) | 3 | 1,632 | 55.0 / 49.0 |
| Affine 4-bit (not released) | 6.84 GB | 133 (81.1%) | 18 | 3,242 | n/a |

HumanEval+ ran on one CUDA GPU: all 164 tasks, upstream chat template, greedy decoding, an 8,192-token budget that includes reasoning, one sample per task. M4 numbers come from a 14-inch MacBook Pro with 16 GB: 256 output tokens, mean of three trials after warmup, and every build measured in the same session as MXFP4. NVFP4 and MXFP4 were interleaved on a 68 W charger, where MXFP4 averaged 67.2 / 58.3. Four other attempts to improve on MXFP4 (mixed 4/6-bit, group-32 experts, DWQ calibration, 8-bit attention) tied or lost, each judged by a rule written before it ran. Paired statistics and limitations: [docs/RESULTS.md](docs/RESULTS.md).

## Findings

1. **Affine 4-bit makes the model think 2.2× longer.** Its completions were 2.2× BF16's length on the same tasks (Wilcoxon p = 5×10⁻²⁵) and ran out of budget mid-reasoning on 18 tasks. MXFP4 is smaller and stays at BF16's length. Pass@1 alone hides this.
2. **Lower KL came with longer answers.** NVFP4 is about 30% closer to BF16 than MXFP4 by KL divergence and decodes at 95 to 96% of its speed, yet it only tied on HumanEval+ and wrote 12% longer completions.
3. **Extra attention bits cost speed in a small-active MoE.** Attention is about a quarter of the weights read for each token. Raising it to 8-bit on top of MXFP4 cost 14 to 17% of decode speed with no measurable accuracy gain.
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
| [`docs/`](docs/) | [Results](docs/RESULTS.md), [coding protocol](docs/CODING_PROTOCOL.md), [unshipped experiments](docs/research/) |
| [`patches/`](patches/) | MLX-LM serving and tool-call fixes used in agent trials |
| [`tests/`](tests/) | Unit tests for conversion recipes, receipts and evaluation guards |

## Credits

The model was created and trained by JetBrains: [Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) at revision `92ddae9`, under Apache-2.0. The runtime is [MLX](https://github.com/ml-explore/mlx) 0.32.3 with [MLX-LM](https://github.com/ml-explore/mlx-lm) 0.32.0. The MXFP4 comparator is by [randmaru](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4), at revision `4ce0d28`. The quantization methods come from upstream. This repo contributes the measurement, integration and analysis. Scripts are MIT-licensed, which does not relicense the model weights.

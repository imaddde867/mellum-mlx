---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: JetBrains/Mellum2.1-12B-A2.5B-Thinking
base_model_relation: quantized
language:
- en
tags:
- mlx
- quantized
- nvfp4
- coding
- agents
- mellum
- moe
- thinking
---

# Mellum2.1-12B-A2.5B-Thinking · MLX NVFP4

JetBrains' [Mellum2.1 Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) coding model (12B mixture-of-experts, 2.5B active) in MLX NVFP4, sized for 16 GB Macs. NVFP4 stores 4-bit floats in groups of 16 weights with FP8 scales, and the MoE routers stay at 8-bit. At 6.84 GB it is **about 30% closer to the BF16 original than MXFP4**, the other ~7 GB build, and decodes at 95 to 96% of MXFP4's speed on an M4.

**Code, raw results and write-up:** [github.com/imaddde867/mellum-mlx](https://github.com/imaddde867/mellum-mlx)

## Quickstart

```sh
pip install 'mlx-lm==0.32.0'

# Chat, with JetBrains' recommended sampling
python -m mlx_lm generate --model imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-nvfp4 \
  --prompt 'Find the bug: def mean(xs): return sum(xs) / len(xs) - 1' \
  --temp 0.6 --top-p 0.95 --top-k 20 --max-tokens 16384

# OpenAI-compatible server for coding agents
python -m mlx_lm server --model imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-nvfp4 \
  --temp 1.0 --max-tokens 16384
```

Always set `--max-tokens`. Reasoning and the answer share the budget, and the server defaults to 512 tokens, which cuts a thinking model off before it answers.

## Benchmarks

| Build | Weights | KL to BF16, reasoning / answer / tool call | HumanEval+ | Median tokens | M4 decode tok/s, 1K / 4K / 16K |
|---|---:|---:|---:|---:|---:|
| BF16 original | 24.3 GB | 0 | 153 (93.3%) | 1,571 | n/a |
| **This NVFP4** | **6.84 GB** | **0.105 / 0.043 / 0.042** | **145 (88.4%)** | **1,674** | **63.9 / 61.4 / 55.8** |
| [randmaru MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4) | 6.46 GB | 0.147 / 0.066 / 0.059 | 144 (87.8%) | 1,544 | 67.2 / 64.3 / 58.3 |

- **Fidelity:** teacher-forced KL divergence from BF16 on held-out Python reasoning, answer and tool-call traces, five repeats each. NVFP4 is 29 to 34% lower in every region, far outside the repeat noise.
- **HumanEval+:** all 164 tasks, one CUDA GPU, greedy, 8,192-token budget including reasoning. A tie with MXFP4 (7 vs 6 discordant tasks, p = 1.0).
- **Speed:** 14-inch MacBook Pro, M4, 16 GB, on its 68 W charger with nothing else running. NVFP4 and MXFP4 were interleaved A-B-B-A, three trials per run. Time to first token is the same for both, 1.6 s at 1K and 31.8 s at 16K. MLX peak memory at 16K is 7.57 GB.

## Which build to use

- **This NVFP4:** the closest to BF16 of the ~7 GB builds measured, and the default pick for 16 GB Macs.
- **[MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4):** the fastest answers. It decodes about 5% faster, and on HumanEval+ it wrote 12% shorter completions, so finished answers come sooner.
- **[6-bit](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64):** for 24 GB or more (9.87 GB, 146/164 on HumanEval+).

On a 16 GB Mac, close memory-heavy apps before long-context work: the 16K peak is 7.57 GB.

## Known MLX-LM 0.32.0 serving issues

- **Unparseable tool calls are dropped silently.** The response still reports `finish_reason: "tool_calls"`, but with an empty `tool_calls` list. A tested patch is in the GitHub repo, and an upstream fix is open in [mlx-lm#1957](https://github.com/ml-explore/mlx-lm/pull/1957).
- **Prior reasoning is lost in tool loops.** The template replays reasoning only from `reasoning_content`, but the server returns it as `reasoning`.

## Limits

One greedy HumanEval+ run, and the benchmark may overlap the training data. KL was measured on a 32-trajectory Python pilot corpus. Neither measures agent reliability. The release rule was written down before any NVFP4 measurement; the [plan and every failed alternative](https://github.com/imaddde867/mellum-mlx/blob/main/docs/RESULTS.md) are public.

## Provenance

- Source: `JetBrains/Mellum2.1-12B-A2.5B-Thinking` at `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, Apache-2.0. JetBrains created and trained the model.
- Quantization: stock MLX NVFP4 (4-bit, group size 16, FP8 scales) for every eligible layer, 8-bit affine group-64 routers, and BF16 for the rest. No fine-tuning or calibration.
- 6,836,663,555 bytes of weights. The SHA-256 of every file is in the [validation receipt](https://github.com/imaddde867/mellum-mlx/blob/main/results/release-v1/nvfp4-validation.json).
- Tested with MLX 0.32.3, MLX-LM 0.32.0 and Python 3.12.

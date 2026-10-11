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
- coding
- agents
- mellum
- moe
- thinking
- 6-bit
---

# Mellum2.1-12B-A2.5B-Thinking · MLX 6-bit

JetBrains' [Mellum2.1 Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) coding model (12B mixture-of-experts, 2.5B active) as native MLX 6-bit, group size 64, with 8-bit routers. It's benchmarked on the full HumanEval+ against BF16 and other quantizations, and measured on a 16 GB M4 MacBook Pro.

**Code, raw results and write-up:** [github.com/imaddde867/mellum-mlx](https://github.com/imaddde867/mellum-mlx)

## Quickstart

```sh
pip install 'mlx-lm==0.32.0'

# Chat, with JetBrains' recommended sampling
python -m mlx_lm generate --model imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64 \
  --prompt 'Find the bug: def mean(xs): return sum(xs) / len(xs) - 1' \
  --temp 0.6 --top-p 0.95 --top-k 20 --max-tokens 16384

# OpenAI-compatible server for coding agents
python -m mlx_lm server --model imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64 \
  --temp 1.0 --max-tokens 16384
```

Always set `--max-tokens`. Reasoning and the answer share the budget, and the server defaults to 512 tokens, which cuts a thinking model off before it answers.

## Benchmarks

| Build | Weights | HumanEval+ | Budget hits | Median tokens | M4 decode tok/s, 1K / 4K / 16K | M4 peak memory |
|---|---:|---:|---:|---:|---:|---:|
| BF16 original | 24.3 GB | 153 (93.3%) | 3 | 1,571 | n/a | n/a |
| **This 6-bit** | **9.87 GB** | **146 (89.0%)** | **3** | **1,521** | **47.2 / 46.0 / 42.8** | **10.5 GB** |
| [randmaru MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4) | 6.46 GB | 144 (87.8%) | 6 | 1,544 | 66.7 / 63.6 / 57.4 | 7.2 GB |
| Affine 4-bit (not released) | 6.84 GB | 133 (81.1%) | 18 | 3,242 | n/a | n/a |

HumanEval+ setup: all 164 tasks with expanded tests, one CUDA GPU, upstream chat template, greedy decoding, 8,192-token budget that includes reasoning, a pinned EvalPlus container, one sample per task. M4 setup: 14-inch MacBook Pro with 16 GB. 6-bit and MXFP4 ran back to back, with 256 output tokens and three trials after warmup.

- **6-bit and MXFP4 are statistically tied** (9 vs 7 discordant tasks, McNemar p = 0.80). Both are below BF16.
- **Affine 4-bit made the model think 2.2× longer** and run out of budget mid-reasoning on 18 tasks, so it isn't released. [Write-up](https://github.com/imaddde867/mellum-mlx/blob/main/docs/WRITEUP.md).
- These are single-run regression results. HumanEval+ may overlap the training data.

## Which build to use

- **16 GB Macs: [NVFP4](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-nvfp4)** (6.84 GB) is the closest to BF16 at about 7 GB. [MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4) gives the fastest answers. Both decode 34 to 41% faster than this build.
- **24 GB or more: this 6-bit** solved the most HumanEval+ tasks of the quantized builds (146 vs 145 for NVFP4 and 144 for MXFP4). The differences aren't significant.
- **16 GB: not recommended for this build.** Swap grew from 1.6 to 4.1 GB during the 6-bit run.
- Long context is cheap. 21 of 28 layers use a 1,024-token sliding window, so the KV cache is about 0.5 GB at 32K tokens.

## Known MLX-LM 0.32.0 serving issues

- **Unparseable tool calls are dropped silently.** The response still reports `finish_reason: "tool_calls"`, but with an empty `tool_calls` list. A tested patch is in the GitHub repo, and an upstream fix is open in [mlx-lm#1957](https://github.com/ml-explore/mlx-lm/pull/1957).
- **Prior reasoning is lost in tool loops.** The template replays reasoning only from `reasoning_content`, but the server returns it as `reasoning`.
- A one-repository agent trial passed 1 of 3 seeds after patches. That isn't a reliability claim; the transcripts are in the repo.

## Provenance

- Source: `JetBrains/Mellum2.1-12B-A2.5B-Thinking` at `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, Apache-2.0. JetBrains created and trained the model.
- Quantization: standard `mlx_lm` affine 6-bit with group size 64 and 8-bit routers; other weights stay BF16. No fine-tuning or calibration.
- 9,873,101,418 bytes of weights. The SHA-256 of every file is in [conversion.json](https://github.com/imaddde867/mellum-mlx/blob/main/results/provenance/mellum2.1-affine-6bit-g64/conversion.json).
- Tested with MLX 0.32.3, MLX-LM 0.32.0 and Python 3.12. The same files are also at [mlx-community/Mellum2.1-12B-A2.5B-Thinking-6bit](https://huggingface.co/mlx-community/Mellum2.1-12B-A2.5B-Thinking-6bit).

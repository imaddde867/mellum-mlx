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
- nvfp4
- 4-bit
---

<!-- DRAFT. Not published. Do not upload until the owner approves, and ideally after a clean AC-power M4 repeat (see Caveats). -->

# Mellum2.1-12B-A2.5B-Thinking · MLX NVFP4

JetBrains' [Mellum2.1 Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) coding model (12B mixture-of-experts, 2.5B active) as native MLX NVFP4: 4-bit elements with FP8 scales per group of 16, applied to attention, experts, embeddings and head. Routers stay affine 8-bit. It was benchmarked on the full HumanEval+ and measured on a 16 GB M4 MacBook Pro against the [randmaru MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4) build.

**Code, raw results and write-up:** [github.com/imaddde867/mellum-mlx](https://github.com/imaddde867/mellum-mlx)

## Quickstart

```sh
pip install 'mlx-lm==0.32.0'

python -m mlx_lm generate --model imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-nvfp4 \
  --prompt 'Find the bug: def mean(xs): return sum(xs) / len(xs) - 1' \
  --temp 0.6 --top-p 0.95 --top-k 20 --max-tokens 16384
```

Always set `--max-tokens`. Reasoning and the answer share the budget, and the server defaults to 512 tokens.

## Benchmarks

| Build | Weights | HumanEval+ | Budget hits | Median tokens | KL to BF16 (reasoning / final / tool call) |
|---|---:|---:|---:|---:|---:|
| BF16 original | 24.3 GB | 153 | 3 | 1,571 | n/a |
| 6-bit g64 | 9.87 GB | 146 | 3 | 1,521 | not measured |
| randmaru MXFP4 | 6.46 GB | 144 | 6 | 1,544 | 0.147 / 0.066 / 0.059 |
| **This NVFP4** | **6.84 GB** | **145** | **6** | **1,674** | **0.105 / 0.043 / 0.042** |

- HumanEval+ versus MXFP4 is a tie: 7 tasks only NVFP4, 6 only MXFP4, exact McNemar p = 1.0. NVFP4 spends about 1.11x the completion tokens on tasks where both finish.
- KL is teacher-forced distribution distance to BF16 on 32 authored Python trajectories, about 30% lower than MXFP4 in every assistant region. It is fidelity evidence on a small pilot corpus, not a coding-score gain.
- Setup: all 164 tasks with expanded tests, one CUDA GPU, upstream chat template, greedy decoding, 8,192-token budget including reasoning, pinned EvalPlus container, one sample per task.

## Mac speed and memory (read the caveats)

M4 MacBook Pro 16 GB, NVFP4 then MXFP4 back to back, 256 output tokens, three trials after warmup:

| | Decode tok/s 1K / 4K / 16K | MLX peak at 16K |
|---|---:|---:|
| NVFP4 | 59.5 / 59.9 / 45.2 | 7.57 GB |
| MXFP4 (same session) | 59.0 / 52.6 / 33.7 | 7.19 GB |

Caveats: this session ran on battery power with memory pressure and swap growth during the NVFP4 leg too, and individual MXFP4 trials ranged from 23.5 to 61.6 tok/s. MXFP4 has measured 51.6 to 67.3 tok/s at 1K across other sessions. Treat NVFP4 as roughly comparable to MXFP4 on speed, unconfirmed. NVFP4 uses about 0.4 GB more memory and its 16K peak sits only 0.03 GB under the 7.6 GB line used for the 16 GB recommendation.

## Which build to use

- **16 GB Macs:** NVFP4 or MXFP4. NVFP4 has clearly lower KL to BF16 and the same HumanEval+ score; MXFP4 is 0.4 GB smaller, has more memory headroom and has cleaner speed measurements. Neither is better on executable correctness in these data.
- **24 GB or more:** the 6-bit build is the closest to BF16.

## Known MLX-LM 0.32.0 serving issues

- Unparseable tool calls are dropped silently (`finish_reason: "tool_calls"` with an empty list). A patch is in the GitHub repo; upstream fix: [mlx-lm#1957](https://github.com/ml-explore/mlx-lm/pull/1957).
- Prior reasoning is lost in tool loops (the template reads `reasoning_content`, the server returns `reasoning`).

## Provenance

- Source: `JetBrains/Mellum2.1-12B-A2.5B-Thinking` at `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, Apache-2.0. JetBrains created and trained the model.
- Quantization: `mlx_lm` NVFP4 (group size 16) on all eligible modules except routers (affine 8-bit, group 64); norms stay BF16. No fine-tuning or calibration.
- 6,836,663,555 bytes of weights. File checksums: [conversion.json](https://github.com/imaddde867/mellum-mlx/blob/main/results/release-v1/nvfp4-conversion.json).
- Tested with MLX 0.32.3, MLX-LM 0.32.0 and Python 3.12 on CUDA (conversion, HumanEval+, KL) and Metal (speed).

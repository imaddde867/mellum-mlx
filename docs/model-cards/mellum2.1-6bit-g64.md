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
---

# Mellum2.1 Thinking, MLX 6-bit (group 64), benchmarked

JetBrains' Mellum2.1 Thinking (12B MoE, 2.5B active) converted to native MLX 6-bit affine, group size 64, with the upstream 8-bit router policy. JetBrains created and trained the model. This release adds a measured comparison against the BF16 original and other quantizations, plus serving settings tested for coding agents. No fine-tuning or calibration.

## What the measurements show

164 HumanEval+ tasks (expanded tests), same RTX 5090, same prompts, upstream chat template, greedy decoding, 8,192 completion tokens, pinned EvalPlus container. One sample per task.

| Candidate | Weights | HumanEval+ | Solved within 4,096 tokens | Median completion tokens | Reasoning length vs BF16 (paired) |
|---|---:|---:|---:|---:|---:|
| BF16 original | 24.3 GB | 153/164 (93.3%) | 148 | 1,571 | 1.00x |
| **This 6-bit** | **9.87 GB** | **146/164 (89.0%)** | **141** | **1,521** | **1.04x** |
| randmaru MXFP4 | 6.46 GB | 144/164 (87.8%) | 139 | 1,544 | 0.98x |
| Our naive affine 4-bit (not released) | 6.84 GB | 133/164 (81.1%) | 91 | 3,242 | 2.22x |

How to read this:
- 6-bit and MXFP4 are statistically tied: 9 tasks passed only 6-bit, 7 only MXFP4 (exact McNemar p = 0.80). This card does not claim 6-bit is better than MXFP4.
- Both are measurably below BF16 (6-bit: 1 vs 8 discordant tasks, p = 0.039).
- Both keep BF16's reasoning length. Our plain affine 4-bit did not: it reasoned 2.2x longer on the same tasks and ran out of budget mid-thought on 18. That is why it isn't released. The cause is under investigation.
- These are single-run regression results. Training overlap with HumanEval+ is possible. They do not measure agent reliability.

Raw responses, per-task outcomes, provenance and scripts: [github.com/imaddde867/mellum-mlx](https://github.com/imaddde867/mellum-mlx).

## Which Mac

- **24 GB unified memory or more: recommended.**
- **16 GB: not recommended.** Measured MLX peak: 10.31 GB at 4K and 10.51 GB at 16K. Swap used: 282.81 MB before the 6-bit sequence, 1,947.31 MB before 16K and 1,883.31 MB after 16K. Swap grew materially across the overall sequence, but decreased during the sampled 16K stage. Editor-open status was not recorded.
- Apple M4 (16 GB), warmup excluded, three measured trials: decode **48.0 / 47.1 / 43.5 tok/s**, TTFT **1.609 / 7.099 / 32.450 s** at **1K / 4K / 16K**, respectively. Greedy decoding, 256 output tokens, AC power.
- Long context is cheap on memory. 21 of 28 layers use a 1,024-token sliding window, so the KV cache is about 14 KB per token (about 0.5 GB at 32K).

## Use it

```sh
python -m pip install 'mlx==0.32.3' 'mlx-lm==0.32.0'

# Chat (JetBrains' recommended sampling)
python -m mlx_lm generate --model imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64 \
  --prompt 'Find the bug: def mean(xs): return sum(xs) / len(xs) - 1' \
  --temp 0.6 --top-p 0.95 --top-k 20 --max-tokens 16384

# OpenAI-compatible server for coding agents
python -m mlx_lm server --model imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64 \
  --temp 1.0 --max-tokens 16384
```

- **Set `--max-tokens`.** The MLX-LM server defaults to 512 tokens. That cuts a thinking model off before it answers or finishes a tool call. Reasoning and the answer share the budget.
- **Don't use greedy (`--temp 0`) for real work.** JetBrains uses temperature 0.6 / top-p 0.95 / top-k 20 for chat, and temperature 1.0 with up to 16K tokens per turn for agents.

## Known issues with MLX-LM 0.32.0 serving

- **Tool calls that fail to parse are dropped silently.** That covers invalid JSON, and also valid-looking JSON with a literal newline inside a string. The response still says `finish_reason: "tool_calls"` but `tool_calls` is empty, so agents can't see or repair the mistake. In our trials this happened twice, both times from an unescaped quote in code. A tested patch that returns the failed call text to the client is in the GitHub repo.
- **Reasoning isn't returned to the template.** Mellum's template replays prior reasoning in tool loops only from `reasoning_content`. The server returns it as `reasoning`. Clients that echo assistant messages back unchanged lose the model's earlier thinking between tool steps.

## Repository-agent trial (one repo, not a reliability claim)

Five bounded trials on one small Python repository, run on CUDA through the MLX-LM 0.32.0 server. Greedy decoding, 8,192 tokens per turn, up to 12 turns, fixed tests in an offline container.

- **Solved (1):** a retrieval-limit type check. A 2-line patch; 36/36 fixed tests pass.
- **Not solved (4):** a mutation-routing fix. What went wrong:
  - In two turns, the model emitted invalid tool-call JSON (an unescaped quote inside code). The server discarded it silently, still reporting `finish_reason: "tool_calls"`. One of those ended a trial.
  - One edit introduced a syntax error the model didn't repair.
  - One fix used substring matching, which misses "changing" and "deleting".
  - Two turns spent the whole 8,192-token budget thinking.
- **Harness limitations in these runs:** greedy decoding, prior reasoning not passed back between steps, and file contents double-escaped in the early trials. A rerun with the settings recommended above is pending.

Every transcript is retained in the GitHub repo. This is not an agent success rate.

## Provenance

- Source: `JetBrains/Mellum2.1-12B-A2.5B-Thinking` @ `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, Apache-2.0.
- Quantization: affine 6-bit, group 64; routers 8-bit; other weights BF16. Produced with `mlx_lm.convert`, the standard recipe; no novel method.
- Serialized weights: 9,873,101,418 bytes. Conversion receipt with SHA-256 for every file: [conversion.json](https://github.com/imaddde867/mellum-mlx/blob/main/results/provenance/mellum2.1-affine-6bit-g64/conversion.json).
- Runtime tested: MLX 0.32.3, MLX-LM 0.32.0, Python 3.12.

[JetBrains original](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) · [MLX-LM](https://github.com/ml-explore/mlx-lm) · [randmaru MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4)

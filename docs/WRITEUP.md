# Two 4-bit quantizations, 0.3 GB apart. One made the model think twice as long.

I quantized JetBrains' [Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking), a 12B mixture-of-experts coding model with 2.5B active parameters, for Apple Silicon with [MLX](https://github.com/ml-explore/mlx). I ran the full 164-task HumanEval+ on the BF16 original and three quantizations. Each run used the same prompts, greedy decoding, and an 8,192-token budget that includes the model's reasoning.

| | Weights | HumanEval+ | Hit the budget | Median completion tokens |
|---|---:|---:|---:|---:|
| BF16 original | 24.3 GB | 153 / 164 | 3 | 1,571 |
| MLX 6-bit (affine, g64) | 9.9 GB | 146 / 164 | 3 | 1,521 |
| MXFP4 (randmaru) | 6.5 GB | 144 / 164 | 6 | 1,544 |
| MLX 4-bit (affine, g64) | 6.8 GB | 133 / 164 | 18 | 3,242 |

Both 4-bit builds fit on a 16 GB Mac. One loses 9 tasks to BF16 and the other loses 20, and on a pass@1 table that gap looks like ordinary quantization damage. The completion lengths tell a different story.

![HumanEval+ tasks solved within a token budget](assets/solved-by-budget.svg)

## The 4-bit model second-guesses itself

On the 144 tasks where both runs finished, the affine 4-bit model's completions were **2.2× longer than BF16's** (paired geometric mean; Wilcoxon p ≈ 5×10⁻²⁵). The 6-bit and MXFP4 runs stayed within a few percent of BF16 (1.04× and 0.98×, neither significant).

Most of the extra tokens are hesitation. The word "But" shows up 22 times per reasoning trace on 4-bit, against 8 or 9 for every other build, and "Actually" 3.9 times against 1.4 to 2.2. All 18 budget truncations ran out mid-reasoning, before any code was written.

Cap the budget at 2,048 tokens and the scores look very different. BF16 solves 120 tasks, 6-bit 108, MXFP4 109, and affine 4-bit **21**.

## Why it matters for local models

On a laptop, tokens are time. At 50 to 65 tokens/s on an M4, an extra 1,700 tokens adds roughly half a minute to every answer. A quantization can keep its score under a generous budget and still be much worse to use. If you quantize a reasoning model, report completion length or pass@budget next to accuracy.

## What helped and what didn't

The expert format mattered most. MXFP4 stores each block of 32 weights as 4-bit floats with one shared 8-bit exponent. It comes out *smaller* than affine 4-bit with group 64 (6.5 vs 6.8 GB) and keeps BF16's reasoning length. Most of this model's weights sit in its 64 experts.

Attention precision only helped when the experts were already damaged. At 6-bit on top of affine 4-bit experts, it recovered most of the loss on a 24-task screen and drew level with MXFP4 (22 vs 21 tasks), while being 10% larger and 24% slower to generate. At 8-bit on top of MXFP4 experts, it bought nothing: 145/164 against 144 (p = 1.0), and 14 to 17% slower decoding on the M4. In a 2.5B-active MoE, attention is about a quarter of the weights read for every token, so its extra bits cost speed on every token. I had written the ship rule down before that run, and it failed the speed gate.

A bounded DWQ calibration run on scales and biases made held-out loss worse and moved task results by no more than noise.

## What I shipped

The [6-bit](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64) (146/164, 9.9 GB) solves the most tasks, but MXFP4 ties it and decodes 41% faster on the M4. So for 16 GB Macs I tried one more format. [NVFP4](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-nvfp4) uses groups of 16 weights with FP8 scales. It measured about 30% closer to BF16 than MXFP4 by KL divergence, on held-out reasoning, answer and tool-call traces, and decoded at 95 to 96% of MXFP4's speed. It tied MXFP4 on HumanEval+ (145 vs 144). It's my pick for 16 GB when fidelity matters. MXFP4 still gives the fastest finished answers.

## What I can't explain yet

I expected NVFP4 to settle the length question. It's measurably closer to BF16 than MXFP4, so it should think more like BF16. Instead it writes completions 1.09× as long as BF16's (Wilcoxon p = 0.0005), while MXFP4, which is further from BF16, stays at 0.98×. So distance from the original distribution doesn't predict how long this model deliberates. I don't know which part of the rounding pushes a model toward the extra "But"s.

## Limits

- One model, one benchmark, one greedy sample per task. HumanEval+ may overlap the training data, so treat these as regression results, not a general coding score.
- Raw responses, per-task outcomes, the scripts and every failed experiment are in [the repository](https://github.com/imaddde867/mellum-mlx). `python scripts/length_audit.py results/coding/humaneval-plus` reproduces the statistics.

# Two 4-bit quantizations of the same model, same size class. One made it think twice as long.

I quantized JetBrains' [Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking), a 12B mixture-of-experts coding model with 2.5B active parameters, for Apple Silicon with [MLX](https://github.com/ml-explore/mlx). I ran the full 164-task HumanEval+ on the BF16 original and three quantizations, with the same prompts, greedy decoding and an 8,192-token budget that includes the model's reasoning.

| | Weights | HumanEval+ | Hit the budget | Median completion tokens |
|---|---:|---:|---:|---:|
| BF16 original | 24.3 GB | 153 / 164 | 3 | 1,571 |
| MLX 6-bit (affine, g64) | 9.9 GB | 146 / 164 | 3 | 1,521 |
| MXFP4 (randmaru) | 6.5 GB | 144 / 164 | 6 | 1,544 |
| MLX 4-bit (affine, g64) | 6.8 GB | 133 / 164 | 18 | 3,242 |

Both 4-bit models are around 6.5–6.8 GB. One loses 9 tasks against BF16 and the other loses 20. Pass@1 makes that look like ordinary quantization noise. It isn't, and you can see why in how long the model takes to get to an answer.

![HumanEval+ tasks solved within a token budget](assets/solved-by-budget.svg)

## The 4-bit model second-guesses itself

On the 144 tasks where both runs finished, the affine 4-bit model's completions were **2.2× longer than BF16's** (paired geometric mean; Wilcoxon p ≈ 5×10⁻²⁵). The 6-bit and MXFP4 runs were within a few percent of BF16 (1.04× and 0.98×, neither significant).

The extra tokens are mostly hesitation. The word "But" shows up 22 times per reasoning trace on 4-bit, against 8–9 for every other candidate. "Actually" shows up 3.9 times against 1.4–2.2. All 18 budget truncations ran out mid-reasoning, before any code was written.

Cap the budget at 2,048 tokens and the gap stops looking like noise. BF16 solves 120 tasks, 6-bit 108, MXFP4 109, affine 4-bit **21**.

So at this size the quality loss mostly shows up as a slower path to the answer. You'd see it as worse speed (more tokens per solved task) and as worse quality (truncation under any realistic budget). A plain pass@1 table blends the two together.

## Why it matters for local models

On a laptop, tokens are time. At about 45 tokens/s on an M4, an extra 1,700 tokens is 40 seconds per answer. A quantization that keeps its score with a generous budget can still be much worse to use. If you quantize a reasoning model, report completion length, or pass@budget, next to accuracy.

## What fixed it

<!-- RESULT: filled in after the mxfp4-a8 run -->

## Limits

- One model, one benchmark, one greedy sample per task. HumanEval+ may overlap the training data. Use this for regression testing, not as a general coding score.
- I measured the length change. I didn't find its numerical cause. "Affine 4-bit hurts this MoE's experts more than MXFP4 does" fits the data but isn't proven.
- Raw responses, per-task outcomes, the scripts and every failed experiment are in [the repository](https://github.com/imaddde867/mellum-mlx). Rerun `python scripts/length_audit.py results/coding/humaneval-plus` to reproduce the statistics.

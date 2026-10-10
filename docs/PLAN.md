# Release plan (frozen 2026-10-11)

Goal: ship a Mellum2.1 Thinking MLX model that runs well on a 16 GB Apple Silicon Mac, with measured quality, plus a write-up of what we learned. The research phase is over. The previous open-ended plan is in git history (`git show ee3006a:docs/PLAN.md`).

## Already shipped

- `imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64` (9.87 GB). HumanEval+ 146/164 vs BF16 153/164. Recommended for 24 GB+ Macs.

## The one remaining experiment

Candidate **mxfp4-a8**: randmaru's recipe (MXFP4 g32 everywhere, 8-bit g64 routers) with one change: attention q/k/v/o projections at affine 8-bit g64, quantized from pinned BF16. Estimated 6.77 GB.

Why this one: restoring attention precision was what rescued affine 4-bit in the non-expert screen (A and C); raising embeddings/head did nothing (B). MXFP4 experts already beat affine 4-bit experts by a wide margin on HumanEval+ (144 vs 133).

Hard requirement: loads with stock, unpatched `mlx-lm==0.32.0`.

## Ship rule (decided before running)

Paired run on the M4 MacBook Pro (16 GB), candidate vs pinned randmaru MXFP4, full 164-task HumanEval+ under the existing [coding protocol](CODING_PROTOCOL.md) except hardware (Metal instead of the CUDA GPU).

Ship mxfp4-a8 as the 16 GB pick only if all hold:

1. HumanEval+ passes >= MXFP4's passes on the same Mac run.
2. Serialized weights <= 7.0 GB.
3. M4 benchmark at 1K/4K/16K: MLX peak <= 7.6 GB at 16K, decode >= 90% of MXFP4's.

If it fails any gate: no new weights. The 6-bit card points 16 GB users to MXFP4, and the write-up is the deliverable.

Either way the research ends here. No extra seeds, no variants, no recalibration if the result is close. New ideas go to "Future work" below.

## Then

1. README and model card lead with the result, the download and a "which Mac" table. Caveats get their own section.
2. Write-up: quantization changed how long a thinking model thinks (native affine 4-bit: 2.2x longer completions, 18 budget truncations mid-reasoning) and pass@1 alone hides it.
3. Publishing (Hugging Face upload, mlx-community, posts) only with explicit user approval.

## Future work (not scheduled)

- DWQ / calibration with representative reasoning + tool traces (parked on `research/optimization-v1-parked`).
- Long-context numerical parity at the 1,024-token sliding-window boundary.
- Repository-agent success rate over more than one task.

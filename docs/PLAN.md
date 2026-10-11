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

## Outcome (2026-10-11)

mxfp4-a8 tied MXFP4 on HumanEval+ (145 vs 144, p = 1.0) and passed size and memory, but decoded at 83–86% of MXFP4's speed on the M4. Gate 3 failed: no new weights. 16 GB users are pointed to MXFP4. Details in [RESULTS.md](RESULTS.md#final-experiment-mxfp4--8-bit-attention-not-shipped). Research closed.

Deviation: the HumanEval+ run used the CUDA GPU instead of the Mac, once SSH access was available, so it pairs with the existing BF16 / 6-bit / MXFP4 runs under the identical frozen protocol. Gate 1 was judged against that MXFP4 run. The M4 gate ran on the Mac as planned.

## Then

1. README and model card lead with the result, the download and a "which Mac" table. Caveats get their own section.
2. Write-up: quantization changed how long a thinking model thinks (native affine 4-bit: 2.2x longer completions, 18 budget truncations mid-reasoning) and pass@1 alone hides it.
3. Publishing (Hugging Face upload, mlx-community, posts) only with explicit user approval.

## Future work (not scheduled)

- DWQ / calibration with representative reasoning + tool traces (parked on `research/optimization-v1-parked`).
- Long-context numerical parity at the 1,024-token sliding-window boundary.
- Repository-agent success rate over more than one task.

## NVFP4 one-shot (pre-registered 2026-10-11)

One bounded attempt, decided before any NVFP4 model exists. No variants, seeds or tuning; whatever happens, this is the last experiment.

Recipe **N**: converted from the pinned BF16 source with stock `mlx-lm==0.32.0`. Every eligible module (attention q/k/v/o, expert gate/up/down, embeddings, head) is `{bits: 4, group_size: 16, mode: nvfp4}`, except routers (paths ending `.mlp.gate`) at `{bits: 8, group_size: 64, mode: affine}`. Same size class and decode path as randmaru's MXFP4 (4-bit elements, one 8-bit scale per group, but g16 instead of g32 and FP8-E4M3 scales).

Ship NVFP4 as the 16 GB pick only if ALL hold:

1. **Fidelity.** Teacher-forced KL to BF16 (calibration-v2 scorer, same frozen validation corpus and the existing BF16 teacher targets, five repeats, FP32 scorer as in the calibration-v2 decision) is lower than pinned MXFP4's in the reasoning, final-answer and tool-call regions. "Lower" uses calibration-v2's own disjoint-interval/mean-gap rule, mirrored: NVFP4's maximum repeat < MXFP4's minimum repeat, and MXFP4 mean minus NVFP4 mean > the larger of the two five-repeat ranges. Transition, context and aggregate are reported, not gated. MXFP4 numbers are the committed `results/calibration-v2/mxfp4-receipt.json`; the new receipt must carry the same `data_manifest_sha256` and `protocol_sha256`.
2. **HumanEval+.** Frozen protocol on the CUDA GPU, counting `plus_status == 'pass'` as in RESULTS.md, passes >= 144 (MXFP4's score).
3. **M4 16 GB.** Same session, NVFP4 then MXFP4 back to back, benchmark.py with 3 measured trials after one warmup: mean measured decode tok/s of NVFP4 >= 95% of MXFP4's at each of 1K, 4K and 16K, and MLX peak (max over all trials including warmup) <= 7.6 GB at 16K.

Otherwise: no new weights, the result is recorded, and research ends permanently.

Preconditions, not gates: if NVFP4 kernels are missing on either backend (Metal or CUDA), or the artifact does not load with stock mlx-lm, the experiment stops and is reported as such. Known confound, recorded in advance: the MXFP4 HumanEval+ and calibration baselines were produced before the CUDA GPU driver update noted in RESULTS.md.

Outcome (2026-10-11): **all three gates pass.** KL to BF16 was 29 to 34% lower and HumanEval+ was 145 vs 144 (p = 1.0). The clean interleaved M4 run (68 W charger, idle machine) gave 95.1 / 95.5 / 95.7% of MXFP4 decode; two earlier, power-limited M4 runs were invalidated. NVFP4 also writes 1.12x longer completions than MXFP4. Publication needs owner approval. Research is closed. Details in [RESULTS.md](RESULTS.md#nvfp4-one-shot).

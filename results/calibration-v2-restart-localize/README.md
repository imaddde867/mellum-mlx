# Restart numerical localization

Continues `dc3f7f5`. The historical restart qualification remains **failed**; its receipt and all historical failures are unchanged. This is U-only localization, with zero training updates, no S-restoration expansion and no reserved-task access.

## Frozen controls

`plan.json` was frozen before actual-model execution. Runtime stays `MLX_USE_CUDA_GRAPHS=1`, `MLX_CUDA_GRAPH_CACHE_SIZE=400`, `MLX_ENABLE_CACHE_THRASHING_CHECK=0`, set by the existing subprocess launcher before MLX import. Pinned packages, allocator cache limit zero, upstream layer checkpointing, pristine C, masks, full-context objective and numerical limits are unchanged. Every process has a predeclared 1,800-second timeout. No unchanged full memory-qualification sequence was rerun.

One discarded U verification update from pristine C on frozen order[0] created a single step-one state. That exact state was saved once and remained resident for two sequential blocks of five backwards. Two later fresh processes restored the same checkpoint and each ran five backwards. No condition regenerated a different step-one state or advanced its baseline between repeats. The test input is frozen order[1], `csv_quote-conversation`, 969 tokens, complete context, cached pinned BF16 temperature-2 top-1,024 targets and the unchanged U mask.

The checkpoint outside Git contains all FP32 masters, complete FP32 Adam moments, uint64 step 1, learning rate, native MLX RNG and Python RNG. `frozen.json` records schema/value digests for masters, all optimizer leaves, BF16 trainable model values represented losslessly as FP32, and RNG. The immutable model remainder is bound to the pristine C artifact through existing source/weight verification; protected parameters were checked against loaded originals after each process. `frozen-input.json` retains full input tokens and both existing masks. Teacher file, tokenizer, corpus/order and target-cache hashes are bound in the frozen receipts; no targets were regenerated.

Adam was checked explicitly: learning rate 1e-7 (stored FP32 value 1.0000000116860974e-7), betas [0.9, 0.999], epsilon 1e-8, bias correction enabled, no scheduler. Installed optimizer implementation hashes and actual process configurations are recorded.

The resident process captured one real gradient before the next update. It applied that gradient to a cloned-container view of the resident step-one state. The first fresh process loaded the same captured gradient values and applied them to the restored state **before performing any backward in that process**. Both results advanced to step 2 and were discarded. Container clones prevent Adam leaf replacement from modifying the baseline; exact hashes before/after trials check this instead of assuming it. RNG resets load an independent host/disk snapshot. All twenty backwards left the recorded native RNG unchanged, and all trial master/moment/model/RNG starting-state checks passed.

## Measurements

Identical-gradient replay produced **exactly equal FP32 masters, all Adam moments/step and update deltas**, with zero absolute L2, maximum absolute and relative error. This isolates persistence and Adam replay from backward computation for this state and gradient. It is not a new restart qualification.

All twenty backwards were finite and preserved. Five-repeat means use the existing sequential FP32 host accumulation/divide-by-five; differences/squares are FP32, per-tensor scalar reductions and accumulation FP64. The actual model objective, backward and Adam update remain on MLX. `analysis.json` retains all six mean comparisons, paired numbered-repeat comparisons, reference norms, absolute errors and tensor/module contributions. No reduction order or tolerance changed.

| Actual / reference block | Gradient relative L2 | Gradient absolute L2 / reference norm | Delta relative L2 | Delta absolute L2 / reference norm | Absolute mean loss difference |
|---|---:|---:|---:|---:|---:|
| resident-a / resident-b | 0.13020354 | 15.223928 / 116.924065 | 0.20516726 | 0.000268311196 / 0.00130776808 | 0.000888866186 |
| resident-a / restored-a | 0.10275069 | 12.433325 / 121.004786 | 0.17674885 | 0.000232224664 / 0.00131386802 | 0.000890034437 |
| resident-a / restored-b | 0.10609826 | 12.734666 / 120.027097 | 0.16786166 | 0.000221250684 / 0.00131805369 | 0.00113411546 |
| resident-b / restored-a | 0.12436433 | 15.048680 / 121.004786 | 0.19903236 | 0.000261502255 / 0.00131386802 | 0.00177890062 |
| resident-b / restored-b | 0.13435505 | 16.126247 / 120.027097 | 0.21114042 | 0.000278294414 / 0.00131805369 | 0.00202298164 |
| restored-a / restored-b | 0.10898567 | 13.081234 / 120.027097 | 0.18365255 | 0.000242063919 / 0.00131805369 | 0.00024408102 |

Every mean-gradient and mean-delta comparison exceeds the unchanged 0.05 limits. All mean-loss comparisons remain below 0.005. These are engineering screens, not confidence intervals.

For resident-a versus resident-b, the maximum absolute mean-gradient difference is 1.4367187; the maximum absolute mean-delta difference is 1.6726554e-7. Attention groups contribute 85.996% of total squared gradient error and embeddings 8.691%. The largest gradient tensors are `model.embed_tokens.scales` (squared error 20.1290) and `model.layers.0.self_attn.o_proj.scales` (11.1801). Update-delta squared error is 96.325% in expert-MLP groups; the largest individual tensors are layer 5 gate-projection scales (5.88743e-10) and biases (5.80094e-10). The restored/restored control has similar contribution distribution: attention 85.767% of gradient squared error and expert MLP 96.431% of delta squared error. This locates error contributions, not their causal source.

## Conclusion and limits

State persistence and optimizer replay failure were **not observed with the identical captured gradient**. Backward variability **is demonstrated without any restoration between resident blocks**: gradient 13.0204%, update delta 20.5167%. Restored/restored means differ by 10.8986% and 18.3653%; resident/restored pairs range from 10.2751–13.4355% and 16.7862–21.1140%. Thus a large discrepancy already exists in the no-restore control. This experiment does not establish an added fresh/restored effect, prove equivalence or exclude a smaller additional effect.

One state, one trajectory, two resident blocks in fixed order and two fresh processes cannot identify the underlying backward mechanism or separate restoration from fresh-process effects on backwards. No kernel, MoE routing decision or memory condition is identified as the cause. Memory samples include near-full device usage in fresh controls, but correlation is not attribution. No concrete implementation defect was demonstrated, so shared runtime, restoration and training code were not changed.

All diagnostic results are separate from historical attempts. There was one discarded step-one initialization update, two identical-gradient next-update trials and twenty discarded repeat next updates. Replacement training remains U=0/128, S=0/128; no candidate, selection or behavior evaluation exists. The historical failed qualification remains failed with its original hash.

## Verification and reproduction

Full suites passed: 73 tests on the pinned host; 77 locally, with 13 MLX-dependent skips. The four-test count difference is unrelated local optimization work, preserved unchanged. New tests exercise absolute/relative/group metrics, FP32 disk means without mutation, and actual Adam replay without modifying shared starting moments or step.

Use the pinned checkout and existing launcher `run` helper with its unchanged runtime environment and 1,800-second timeout to execute `scripts/calibration_v2_restart_localize.py resident`, then `restored-a`, then `restored-b`, then `analyze`. Each command is recorded in its process receipt. Start only with empty new localization output/work paths; existing outputs refuse overwrite. Do not rerun trials selectively or repurpose these controls as training authorization.

Large state, gradient, update and reference arrays remain in `work/calibration-v2-restart-localize`, outside Git. `arrays.json` binds the complete per-file manifest there; `evidence-manifest.json` binds committed compact evidence. Exact executed sources match the committed implementation, including the unchanged runtime/objective/launcher. No push, model upload, package change, S expansion, reserved evaluation or training occurred.

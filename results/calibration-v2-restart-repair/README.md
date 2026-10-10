# Restart memory-lifetime repair

Continues 03a9a15. Original failure receipts, frozen corpus/targets and published model remain unchanged. Runtime stays `MLX_USE_CUDA_GRAPHS=1`, `MLX_CUDA_GRAPH_CACHE_SIZE=400`, `MLX_ENABLE_CACHE_THRASHING_CHECK=0` in subprocesses before MLX import. Packages, allocator cache limit zero, layer checkpoint policy, objective, masks, optimizer, precision and selection gates are unchanged.

## Implementation

`model_setup` loads only the pristine model and original parameter references. Fresh training creates FP32 masters and Adam separately. Restoration calls `restore_training_state` directly after model-only setup; it does not allocate disposable masters or moments.

Restoration verifies immutable/file bindings, master/model keys, FP32 master dtypes, BF16 model dtypes, shapes, complete FP32 Adam moment keys/shapes, uint64 optimizer step, order cursor, fixed learning rate and Python/MLX RNG. It materializes restored state and applies the corresponding BF16 model values before backward. The pinned MLX state setter clears its initialization flag, so the helper explicitly marks the already-validated complete state initialized. It never initializes or zeroes restored moments. A regression test forbids an initialization call while checking the actual next update against the resident control.

The runner's explicit `--checkpoint` branch calls this same restoration helper, checks the frozen-order prefix and qualification hash, and continues after the verified cursor. Partial validation observations are preserved; a pending scheduled validation gets a separate directory. Execution receipts are retained separately before atomically publishing the latest metadata. No automatic pristine restart or in-process CUDA recovery is introduced.

Diagnostic gradients and update deltas are transferred one tensor at a time to NumPy host accumulators. GPU diagnostic trees are not retained between backwards. Both producer and restore use five repeats with the same procedure. References are individual host `.npy` files outside Git and comparisons read them incrementally. The actual KL, gradients and Adam updates stay in MLX on the device.

Elementwise sums, divide-by-five, subtraction and squared differences remain FP32 in the original sequential repeat order. Per-tensor scalar reductions and scalar accumulation now use FP64 on the host; the original comparison used FP32 MLX reductions. Small-array differential tests compare these means and relative errors against the original calculation. Tolerances remain .05 gradient relative L2, .05 update-delta relative L2 and .005 absolute mean loss difference. Host diagnostics explicitly reject nonfinite inputs/accumulators/reductions.

`save_restart` retains atomic publication and latest-two retention. Serialization readback uses `safetensors.safe_open(..., framework='numpy')` and checks one host tensor at a time against the device value, with exact dtype/shape/value checks. It does not load a second GPU-resident optimizer tree. Finite-value verification also uses bounded host copies.

Memory logging records active/cached MLX memory, allocator peak and working device-wide `cudaMemGetInfo` samples. Each repeat's number is printed before its backward. These are boundary samples, not a measured total-VRAM peak. Exception handlers write metadata only; they do not inspect or serialize CUDA arrays after a CUDA failure. Qualification subprocesses retain the 1,800-second timeout.

## Result

An instrumented replay of the old U restore reproduced CUDA OOM during **repeat 2's backward**, preserving the exact error. Its last successful boundary was `before-backward`, repeat 2. This replay used the existing step-one verification checkpoint and original GPU diagnostics; it was not replacement training.

The repaired producer and U restore each completed all five numbered backwards and next updates without OOM. U restoration nevertheless **failed numerical qualification**:

| Check | Observed | Frozen limit | Result |
|---|---:|---:|---|
| Mean-gradient relative L2 | 0.1009983355 | 0.05 | Fail |
| Mean-update-delta relative L2 | 0.1809756092 | 0.05 | Fail |
| Absolute mean loss difference | 0.0000154898 | 0.005 | Pass |

No repeat was dropped or chosen favorably, no tolerance changed, and the failure was not replaced with a metadata-only success. S restoration and replacement training were not started. A separate zero-update U/S sequence rerun covers the shared setup change; passing those checks cannot override the numerical failure. The unchanged short checkpoint-equivalence check and wheel verification are reused with explicit historical hashes.

Replacement U and S each have **0/128 updates**. There are no trained region/family measurements, selected candidate, candidate reload or reserved-task results. The reserved contents remained unopened. Isolated producer verification applied six discarded updates; restored verification computed five discarded next updates. Original abandoned seven-update attempts remain separate.

The repair's last successful execution boundary was repeat 5, `after-release-diagnostic-temporaries`: **12,418,113,304 active MLX bytes, zero cached bytes, 15,951,593,472 device-wide used bytes**. Diagnostic means were in host memory at this point; they were released when the failed process exited.

The old path did not retain a measurable extra active footprint from its disposable initialization after restoring state: both those old boundaries reported 11,657,632,272 active bytes. That allocation was removed as requested, but it is not identified as the confirmed OOM cause. The clearest measured difference is the retained repeat diagnostics: after repeat 1, old active memory was 15,456,362,264 bytes versus repaired 12,418,113,304, a reduction of 3,038,248,960 bytes. The old restored-state boundary preceded BF16 model application; the repaired boundary includes that application, so their initial boundary difference is not evidence of a regression.

The repaired restore still showed device-memory asymmetry relative to its producer: some update-boundary samples reached 33,613,807,616 used bytes of 33,647,230,976 total, leaving only 33,423,360 free bytes. This is device-wide usage and includes runtime/other allocations. No cause of that asymmetry, or of the numerical mismatch, is claimed confirmed. Memory lifetime is improved; faithful numerical continuation remains unqualified. No evidence supports promoting a candidate or proceeding to the U/S experiment under the existing gate.

`summary.json` and `memory-comparison.json` provide compact measurements. Component logs/receipts preserve raw attempts and exact executed source hashes. Large checkpoint and reference arrays remain under `work/calibration-v2-restart-repair-reference-U`, outside Git. The final additional finite guards and runner resume branch received CPU regression tests; no favorable real-model numerical rerun was substituted after the failed comparison.

# Native graph-disabled runtime recovery: qualification failed

The authorized replacement arms were **not started**. U and S each have zero replacement updates. Their historical seven-update failures remain separate and immutable at `results/calibration-v2-objective`; those optimizer states were never recovered. The frozen corpus, cached BF16 targets, original protocol and published release remain unchanged.

## Runtime verification

Installed `mlx` and `mlx-cuda-13` both report 0.32.3. Every installed `mlx/` file belonging to those two distributions was compared byte-for-byte with the corresponding official PyPI wheel, whose SHA-256 was independently checked against PyPI release metadata. The installed core extension and CUDA `libmlx.so` match those pinned wheels. No package was installed or upgraded.

The [pinned CUDA implementation](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/cuda/device.cpp) reads `MLX_USE_CUDA_GRAPHS` and uses direct kernel launches when it is false. The installed binary contains that native switch. Source/archive and wheel/file hashes are in `runtime-inspection.json`. Downloaded verification wheels and upstream source remain in `work/calibration-v2-runtime-source` outside Git.

The qualification process launched with `MLX_USE_CUDA_GRAPHS=0` in its environment, checked the setting before importing MLX, and rejected a `MLX_ENABLE_CACHE_THRASHING_CHECK=0` bypass. The existing zero free-allocation cache limit, checkpoint wrapper, batch-one full-trajectory backpropagation, FP32 masters/Adam and FP32 temperature-2 top-1,024 KL were unchanged. No global shell, driver, package or kernel configuration changed. `amendment.json` was frozen before the execution and leaves the original protocol intact.

## Actual qualification result

A fresh graph-disabled process ran the existing shortest-trajectory checkpoint check: one recorded warmup and five measured full backwards per mode. All 12 short loss/gradient attempts were finite. Mean unwrapped loss was 0.11029624194, checkpointed loss 0.11011026502. Mean-gradient relative L2 difference was **1.0992%**, within the unchanged 5% tolerance; absolute mean-loss difference was **0.000185977**, within 0.005.

It then initialized resident Adam state and attempted the full longest training trajectory, `decimal_octets-conversation` (1,319 tokens), with U's unchanged all-target mask. The backward failed while evaluating loss/gradients:

```
RuntimeError: cudaMallocAsync(&data, size, stream) failed: out of memory
```

MLX allocator peak was **31,106,296,128 bytes**. The existing preflight itself ran 14.9437 seconds; that excludes outer wheel-integrity checks. There were zero optimizer updates. No concurrent model training or serving process appeared in the post-exit process inventory.

This is a failed memory qualification under the requested native workaround, not the prior graph-cache error. The S longest case, initial/transition validation and first-12/subsequent sequence probes were not reached. Complete parameter integrity could not be evaluated after the CUDA exception; no evaluation or array serialization was attempted afterward. On-disk pristine weights were rechecked separately without loading the model.

Per the user's explicit stop rule, no retries, allocator/checkpoint changes, clipped trajectories, detached history, graph-cache sweep, tolerance changes or alternative runtime were attempted. This result does not prove that every graph-disabled design would fail; it rejects this unchanged qualification on the available hardware.

## Restart support and limits

`save_restart` writes FP32 masters, the complete Adam state/step, MLX and Python RNG, frozen-order cursor and immutable arm/data/target/runtime bindings. Files are read back and compared exactly, hashed and fsynced; the directory is atomically published and its parent fsynced before old verified entries are pruned to retain the latest two. `load_restart` verifies hashes, arm/bindings and optimizer step/cursor, restores both RNG states and recreates the parameter/state trees. Snapshots are kept outside Git.

The real checkpoint implementation passed a small-array Adam test on the pinned host using the CPU device: two-checkpoint retention, exact serialization/restoration, next-update equality, step advancement, MLX RNG equality, binding rejection and corruption rejection. This verifies the persistence boundary, **not real Mellum CUDA recovery** or bitwise execution reproducibility. The amended fixed five-repeat real-model next-update check was not started because native memory qualification failed. Restart snapshots were not produced for either replacement arm, and restart support was not integrated into an unqualified production training run.

The runner contains the declared zero-update U/S sequence probes and fresh-process real-model restart checks for reproducibility; these paths were not executed. It intentionally has no replacement training command because the required qualification failed. No existing failed-arm paths can satisfy `check_replacement`: both new completed arm bindings, qualification hash, exact frozen order and 128 updates are required. There is no trained region/family comparison to report, no selected checkpoint, no candidate save/reload and no reserved-task outcomes.

## Validation and evidence

Local suite: 64 tests, **58 passed and six platform skips**. Pinned-host suite: **60 passed**, including restart persistence and the existing numerical objective/checkpoint tests. The test suites differ because unrelated local optimization tests were not transferred. The metadata failure-handler test verifies that the exact backend error is propagated and recorded without attempting CUDA recovery or array serialization. The initial wrapper omitted an outer failure receipt; `correctness/receipt.json` explicitly records its post-exit reconstruction, and `qualification-executed.py` retains the exact executed wrapper. The final wrapper records failure metadata in `finally` without touching CUDA state.

`correctness/detail/receipt.json`, `correctness/detail/checkpoint-equivalence.json` and `correctness.log` retain the original error, every short attempt, peak and provenance. `qualification.json` records the stop decision and unstarted stages. `manifest.json` binds all compact evidence and final implementation/tests. Exact inspection and qualification code snapshots distinguish executed code from later error-recording fixes.

Executed qualification command (its existing directory is refused):

```
MLX_USE_CUDA_GRAPHS=0 .venv/bin/python scripts/calibration_v2_runtime.py correctness --output results/calibration-v2-runtime/correctness
```

This task provides no new evidence favoring U or S and supports no calibrated candidate. No M4, HumanEval+, recipe, model upload or push was performed.

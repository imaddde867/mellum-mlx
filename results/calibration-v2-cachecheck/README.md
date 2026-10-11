# Controlled cache-check runtime amendment

This attempt continues 33bb54d. The frozen objective, corpus, targets, optimizer, eligible tensors, allocator setting, checkpoint policy and selection gates are unchanged. Original seven-update attempts and the graph-disabled OOM remain historical evidence in their original directories.

The subprocess environment is exactly:

```
MLX_USE_CUDA_GRAPHS=1
MLX_CUDA_GRAPH_CACHE_SIZE=400
MLX_ENABLE_CACHE_THRASHING_CHECK=0
```

Only the cumulative graph-cache-miss performance exception is disabled. CUDA, allocation, nonfinite, integrity and numerical errors remain fatal. Existing wheel/source verification is reused; unchanged distribution versions and compiled shared libraries are checked without downloads.

Qualification uses a predeclared 1,800-second timeout per fresh process. Run from the repository on the pinned GPU host:

```
.venv/bin/python scripts/calibration_v2_runtime_process.py --name probe-U probe --arm U --output results/calibration-v2-cachecheck/probe-U
.venv/bin/python scripts/calibration_v2_runtime_process.py --name probe-S probe --arm S --output results/calibration-v2-cachecheck/probe-S
.venv/bin/python scripts/calibration_v2_runtime_process.py --name restart-produce-U restart-produce --arm U --output work/calibration-v2-cachecheck-restart-U
.venv/bin/python scripts/calibration_v2_runtime_process.py --name restart-restore-U restart-restore --arm U --checkpoint work/calibration-v2-cachecheck-restart-U --output results/calibration-v2-cachecheck/restart-restore-U
```

Repeat the restart check for S only after U passes. Produce `qualification.json` only after every required component passes. Replacement training requires that receipt and the exact new arm directories declared in `amendment.json`; selection requires both new paths explicitly. Historical attempts cannot be selected.

The training runner uses the existing atomic restart implementation after every completed update. It verifies array serialization and rejects nonfinite arrays before publishing a restart, retaining the latest two verified states outside Git. Scheduled selection parameter snapshots are retained separately. An exception writes metadata and exits; it does not evaluate or save CUDA arrays afterward. Existing evidence paths are never overwritten.

CUDA `cudaMemGetInfo` works on this host despite the prior NVML mismatch. Its samples measure device-wide used/free/total memory at successful operation boundaries, including other device allocations. They are point samples, not a total-VRAM peak. MLX allocator peaks are reported separately. No device-memory query is made after a CUDA failure.

Restart probes apply isolated, discarded verification updates; these do not belong to either replacement training arm. Real-model gradient/loss/next-update tolerances remain the original .05/.005/.05 limits. Small-array restoration tests are supplementary and cannot qualify the real model.

Reserved task contents remain gated by completed matched training, original fidelity selection and successful selected-candidate reload. No old forward is used as a replacement training reference.

## Outcome

Qualification failed at the U real-model restore/next-update check:

```
RuntimeError: cudaMallocAsync(&data, size, stream) failed: out of memory
```

The failure occurred in `restart_samples → gradient_receipt → mx.eval(loss, flat)`. The affected process exited with code 1, before any numerical next-update comparison could pass. No CUDA arrays were evaluated or serialized afterward. No workaround retry, cache change or further model process was started.

The preceding checks passed:

- Short checkpoint equivalence: mean-gradient relative L2 difference 0.0216003 (limit 0.05), absolute mean loss difference 0.000562778 (limit 0.005).
- Longest 1,319-token trajectory: U and S finite full-context backwards, FP32 masters and complete Adam resident, zero optimizer updates, all parameters unchanged. Allocator peaks were 21.4455 GB and 26.5428 GB respectively; the largest measured device-wide used-memory sample was 30.2835 GB.
- U and S sequence probes: initial five validation repeats, first twelve frozen-order backwards, five transition validation repeats, thirteenth backward and longest backward; all fourteen backwards finite, all parameters unchanged, Adam step zero. Each subprocess completed in about 150 seconds, below the frozen 1,800-second timeout.
- U isolated restart producer: verified step-one serialization and five resident-control next-update measurements completed. Six discarded verification updates are separate from replacement training. The checkpoint and reference arrays remain under `work/calibration-v2-cachecheck-restart-U`, outside Git.

S restart qualification was not attempted after U failed. Replacement U and S each completed **zero** training updates. Neither was started. There are no trained region/family measurements, candidate selection or selected-candidate reload results. Reserved tasks remained unopened.

`summary.json` contains the actual five-repeat region and underlying-family statistics from the **unchanged pristine-C qualification probes**, explicitly separated from unavailable replacement-arm measurements. They are not training step-zero references, trained results or evidence for the U/S hypothesis. Individual rows, region token totals, repeats, timings, memory samples and exact source hashes remain in the component receipts. `runtime-code` preserves each executed source version. The GPU checkout remained at its prior commit; transferred code is bound by its exact executed hashes rather than pretending the checkout was reset to the local parent.

The replacement runner and explicit-path selection guard are implemented, but the training loop was not exercised on the actual model because qualification failed. This evidence establishes that the cache-check exception no longer blocked the tested sequence; it does **not** qualify the full restart/training configuration or establish an assistant-target fidelity benefit. The next runtime design would need separate authorization and qualification. No new candidate is supported by this result.

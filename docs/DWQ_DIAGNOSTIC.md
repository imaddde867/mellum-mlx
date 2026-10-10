# One gated 1e-7 DWQ diagnostic

**Stopped at the loss gate. Pristine C wins at step zero.** All 32 checkpointed updates completed, but no trained checkpoint improved validation beyond the predeclared observed-variability threshold. No development generation, consultation of the opened 12-task set, or M4 run followed. Learning-rate tweaks on this tiny corpus stop here.

[The frozen rule](../results/dwq-diagnostic-v1/predeclared.json), [measurements](../results/dwq-diagnostic-v1/summary.json), full training/reload/integrity receipts and exact executed source are retained under [results/dwq-diagnostic-v1](../results/dwq-diagnostic-v1/). No previous artifacts or evidence were replaced. This is a calibration diagnostic, not an improved quant or a release candidate.

The student restarted from the original uncalibrated C, checked against its original integrity weight hashes and BF16 revision. The unchanged 32 training/eight validation stdlib snippets contain 4,292/915 scored tokens. The same 40 cached BF16 teacher files, tokenizer, corpus ordering, optimizer defaults, Adam bias correction, seed 123, batch 1, sequence limit 256, temperature 2 and 32-update budget were retained. Only the learning rate changed to 1e-7. The frozen corpus manifest retains its prior-run 1e-6 metadata; the actual diagnostic optimizer rate is 1e-7 in the execution receipt. The successful upstream first-layer gradient-checkpoint wrapper was applied once before initial validation; the upstream training call avoids wrapping it a second time.

The existing upstream DWQ loop is reused. Validation reseeds NumPy, but the fixed one-pass training permutation is already materialized; an interleaved CPU iterator check verified unchanged ordering. This is not a multiple-epoch guarantee. An optimizer observer measures the same weighted cached top-1,024 KL objective at steps 8/16/24/32, after updates. Validation occurs five times before training and five times at each checkpoint. Best eligible low-bit scales/biases are retained in memory, with step zero eligible; only the selected state is serialized. Packed weights, routers, excluded tensors and schemas are protected by the existing parameter checks.

Numerical qualification: this diagnostic sums token losses in BF16 and computes the token-weighted mean on the host. Upstream DWQ first computes each sequence mean in BF16, adding a rounding step. The mathematical KL objective is unchanged, but the reductions are not numerically identical. The diagnostic reduction was applied consistently to step zero, all checkpoints and reload; these absolute values should not be compared directly with the previous upstream validation losses. The executed source and raw measurements are preserved without retrospective changes.

The gate requires both **non-overlapping repeated-loss intervals** (`candidate max < initial min`) and a mean gain larger than the greater of the initial and candidate repeat ranges. This conservative empirical rule was frozen before outputs; five repeats are not confidence intervals or proof of fidelity across other contexts.

| Update | Mean validation loss | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| 0: pristine C | 0.104334 | 0.103808 | 0.104730 |
| 8 | 0.103514 | 0.102374 | 0.104064 |
| 16 | 0.104570 | 0.103620 | 0.105379 |
| 24 | 0.104949 | 0.103928 | 0.106421 |
| 32 | 0.106943 | 0.105601 | 0.108811 |

Step eight has the lowest trained mean, but its **0.000820** apparent gain is smaller than its **0.001691** repeat range and its measurements overlap the initial interval. Selecting its best single forward would manufacture an improvement. Later checkpoints do not recover the objective. This does not diagnose overshooting, overfitting or the source of CUDA repeat variability.

Training/selection/save stage elapsed **47.469 seconds**, MLX allocator peak **21,117,021,120 bytes**, excluding source preflight checks and driver/system memory. Gradients were finite through all 32 updates. The selected artifact contains **zero changed parameter arrays** relative to C; its **7,098,569,272 weight bytes and both weight SHA-256 values are identical to pristine C**. The artifact name denotes the diagnostic, not a retained trained update.

A fresh process reloaded the selected artifact and repeated validation five times: mean **0.103624**, range **0.102886–0.105157**. This variation on exactly the same serialized weights further cautions against interpreting the step-eight shift as recovery. Existing structural checks and fresh-process serialized/load/save/reload parameter checks passed, with finite inference. Parameter integrity and numerical repeat variability are recorded separately; bitwise numerical equality is not imposed as a global CUDA requirement.

Validation: four new gate/parent tests failed before implementation, then passed locally and on CUDA. All **41 local tests** passed, including the four existing untracked pilot tests; syntax and whitespace checks passed. Existing `targets`/`train` behavior remains available. No dependency, system-setting, model-layout or corpus changes occurred.

To reproduce in a CUDA checkout with the pinned original C, original C integrity receipt, original native-4 source receipt and existing teacher cache, use a new destination and fresh evidence paths:

```sh
mkdir -p work/dwq-diagnostic-repeat
.venv/bin/python scripts/refine_dwq.py diagnostic \
  --student artifacts/mellum2.1-mixed-C-g64 \
  --destination artifacts/mellum2.1-dwq-diagnostic-repeat \
  --output work/dwq-diagnostic-repeat/train.json
.venv/bin/python scripts/refine_dwq.py verify \
  --student artifacts/mellum2.1-dwq-diagnostic-repeat \
  --output work/dwq-diagnostic-repeat/reload.json
.venv/bin/python scripts/validate.py artifacts/mellum2.1-dwq-diagnostic-repeat \
  --receipt work/dwq-diagnostic-repeat/structure.json
.venv/bin/python scripts/nonexpert.py integrity \
  --model artifacts/mellum2.1-dwq-diagnostic-repeat \
  --output work/dwq-diagnostic-repeat/integrity.json
```

Existing artifact, output and cache paths are never silently overwritten. A positive loss gate would still require saved/reloaded integrity and a matched development comparison of correctness, truncation and completion length together; it would not authorize an automatic Mac rerun or establish agent quality. The already-opened 12-task fixture is regression evidence, not untouched selection data.

**One next action:** freeze separate calibration material containing representative reasoning endings, tool exchanges and complete chat trajectories before testing another calibration recipe, with C and pinned MXFP4 retained as controls. Corpus coverage is the next substantive question; this diagnostic supplies no coding-quality or Mac improvement.

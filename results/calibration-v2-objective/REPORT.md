# Calibration-v2 U/S objective comparison: runtime-blocked

Both independent arms started from the identical pristine C weight hashes, fixed seed 123 and Adam 1e-7, and completed **seven optimizer updates each** in the frozen order. Neither completed the 128-update budget. At the eighth full-trajectory backward, both raised:

```
RuntimeError: Cache thrashing is happening, please set the environment variable MLX_CUDA_GRAPH_CACHE_SIZE to a larger value than 400 to fix degraded performance.
```

This is an MLX CUDA graph-cache capacity failure, distinct from a sequence-length rejection or the earlier allocator OOM. No checkpoint 8 was saved or evaluated. The seven FP32 optimizer states were resident only; the failed processes did not serialize them. Restarting from pristine would repeat consumed trajectories rather than resume these arms. No restart, learning-rate sweep, sequence clipping or detached-cache fallback was performed. No trained candidate exists.

## Implemented objective and frozen settings

`predeclared.json` freezes objective protocol 1.0, selection and conditional behavior rules before optimizer updates. `order.json` contains the seed-123 shuffled 128 IDs. Existing frozen train/validation inputs, tokenizer, top-1,024 BF16 teacher logits and indices remain hash-bound. Teacher logits and student gathered logits are cast to FP32 **before** temperature-2 scaling, renormalized KL and reduction, using the previously verified scorer. No temperature-squared multiplier, vocabulary expansion or answer cross-entropy was introduced.

U includes all nonpadding prediction targets. S includes exactly the existing reasoning, transition, final and tool-call labels, normalizes by their count, and excludes context/padding only from loss. Both consume all input tokens. Zero-target cases raise. Training uses no KV cache, batch one, upstream layer checkpointing, FP32 optimizer parameters/moments and BF16 model parameters. Only the 396 eligible low-bit affine scales/biases are unfrozen; no packed weights, routers or excluded tensors are eligible. Disk pristine-C weight hashes were rechecked after both failed arms and remained unchanged.

Tests cover next-token alignment, multiple assistant turns and intervening tool/user context, padding, excluded-loss invariance including NaN, included-count normalization, zero-target rejection, equality with the verified FP32 scorer, context dependence, installed checkpoint-hook gradients, empirical region gates, matched-reference selection and refusal to select from incomplete arms. Final local suite: 60 tests, 55 passed and five platform skips. CUDA host: all 56 available tests passed, including numerical objective/checkpoint tests. The initial nine objective tests passed before training; the tenth was added for incomplete-arm selection handling afterward. Full suites differ because unrelated local optimization tests were not transferred.

## Targets and training qualification

All 128 training targets were captured in a separate pinned BF16 source process: revision `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, seed 123, complete inputs in cache-preserving 128-token inference chunks. These chunks apply only to capture/validation, never to training backpropagation. The target receipt binds every file to the frozen training sequences, source, tokenizer, protocol and settings. Existing validation captures were reused. Large target arrays remain outside Git at `work/calibration-v2-training-targets` on 5090-1.

All preflight attempts remain intact:

1. Original single-pair short checkpoint check failed: gradient difference 22.40%, loss difference 0.005293; zero updates.
2. Controlled three-repeat diagnostic showed approximately 7-8% within-unwrapped gradient distances versus about 2% checkpointed, without parameter mutation. An isolated longest-trajectory U/S memory probe fit.
3. A command failed to apply the planned repeat-mean code, causing a second old-code attempt, retained under the misleading original directory name `preflight-v1.1`. Its single pair passed (2.27%), then longest S failed with allocator OOM. This is a failed old-code attempt, not repeat-mean qualification.
4. `qualification-v1.1.json` explicitly versions the changed qualification procedure **before training**: one recorded warmup plus five fixed measured gradients per mode, arithmetic mean comparison, unchanged 5% gradient and 0.005 loss tolerances, free-allocation cache disabled identically in preflight and both arms. No individual pair selection or retry-until-pass. The resulting mean-gradient difference was 0.8198%, mean-loss difference 0.000052. Longest `decimal_octets-conversation` (1,319 tokens) full backwards succeeded with original references, FP32 parameters and Adam state resident: U peak 21,445,533,176 bytes; S peak 26,542,796,584 bytes. No optimizer updates. All original model parameters were checked unchanged.

This qualification amendment did not modify original protocol files, historical diagnostics, fidelity gates, loss masks, training budget or teacher settings. Passing these memory checks did not establish that the CUDA graph cache would handle a sequence of many distinct training shapes; the actual arms subsequently demonstrated that limitation. Training peaks were 24,287,704,440 bytes; each process ran about 78 seconds including its five pristine references.

## Validation measurements

Every cell below is FP32 temperature-2 top-1,024 KL. The U/S process values are **pristine C at step zero**, not trained-arm outcomes. There is no post-update validation. Their differences are repeated-forward variation, not evidence about the objectives. The pooled pristine assistant mean is 0.09100189, with observed range 0.00055168 across ten forwards. The frozen selection rule uses each five-repeat reference separately rather than this pooled range.

| Region | C in U process, mean [min, max] | C in S process, mean [min, max] | Trained U | Trained S |
|---|---:|---:|---:|---:|
| assistant | 0.09107674 [0.09078765, 0.09133933] | 0.09092703 [0.09083095, 0.09102010] | unavailable | unavailable |
| reasoning | 0.15044417 [0.14997684, 0.15090053] | 0.15023183 [0.14973222, 0.15056410] | unavailable | unavailable |
| transition | 0.14644688 [0.14586158, 0.14734979] | 0.14618241 [0.14561561, 0.14664961] | unavailable | unavailable |
| final | 0.07848386 [0.07823537, 0.07888037] | 0.07841215 [0.07828393, 0.07847857] | unavailable | unavailable |
| tool_call | 0.06078723 [0.06061167, 0.06103620] | 0.06060623 [0.06047749, 0.06073020] | unavailable | unavailable |
| context | 0.12973722 [0.12965574, 0.12986429] | 0.12983556 [0.12956000, 0.13006073] | unavailable | unavailable |
| aggregate | 0.11299061 [0.11283911, 0.11317639] | 0.11298151 [0.11286563, 0.11308157] | unavailable | unavailable |

The 8 validation problem families are summarized below at pristine step zero. These are underlying-family means over all ten repeats, with tokens weighted within each family and all correlated variants retained. Region-level family repeats, token counts and the equal-family macro summaries are in `summary.json`; individual trajectory sums are in each arm receipt and step-zero files. No reserved family outcomes are included.

| Underlying validation family | Assistant KL | Context KL | Whole-transcript KL |
|---|---:|---:|---:|
| cursor_page | 0.07595359 | 0.12068981 | 0.10169224 |
| decode_frames | 0.08666426 | 0.12572760 | 0.10872000 |
| dependency_order | 0.10036586 | 0.12862158 | 0.11528917 |
| job_state | 0.09129596 | 0.13457538 | 0.11618342 |
| markdown_titles | 0.08884618 | 0.13183990 | 0.11313574 |
| query_values | 0.10268382 | 0.14353251 | 0.12829240 |
| transpose_rect | 0.10570462 | 0.12892747 | 0.11919925 |
| undo_journal | 0.08010393 | 0.12637464 | 0.10524592 |

## Selection, behavior and interpretation

`selection/decision.json` records the incomplete-arm blocker, both receipt hashes and **no locked candidate**. No improvement was demonstrated beyond repeat variation; save/reload survival could not be assessed. The reserved 32 tasks were not opened or generated against. No behavioral correctness, tool validity, truncation, token or timing comparison was permitted. Pristine C remains the operational artifact; this does not amount to a measured rejection of U or S.

The hypothesis remains untested by a usable matched trained validation comparison. These results support fixing and qualifying the CUDA graph-cache runtime before requesting a new bounded attempt; they provide no evidence supporting further development or promotion of an exact calibrated candidate. The runtime error suggests a graph-cache environment change, but that change has not been qualified and was not applied to another training run.

## Reproduction and evidence

Run from the repository root on the recorded Linux CUDA environment with the pinned local model artifacts available:

```
.venv/bin/python -m unittest discover -s tests -p test_calibration_v2_objective.py
.venv/bin/python scripts/calibration_v2_objective.py targets --output work/calibration-v2-training-targets
.venv/bin/python scripts/calibration_v2_objective.py preflight --output results/calibration-v2-objective/preflight-qualified
.venv/bin/python scripts/calibration_v2_objective.py train --arm U --output work/calibration-v2-objective-U
.venv/bin/python scripts/calibration_v2_objective.py train --arm S --output work/calibration-v2-objective-S
.venv/bin/python scripts/calibration_v2_objective.py select --output results/calibration-v2-objective/selection
```

Existing output directories are refused, preserving this evidence. These are executed command references, not instructions to restart the consumed arms. Exact executed script snapshots distinguish teacher capture, each preflight and training from subsequent decision-report handling. Raw logs, every completed update receipt, both step-zero repeat sets, source bindings and target-file hashes are committed compactly; large weights and targets are outside Git. The remote checkout revision differs from local 5cdffc0, so receipts explicitly bind transferred source/data files and package versions rather than pretending the remote Git revision identifies those files.

No dataset expansion, new recipes, model uploads, pushes, M4 reruns or HumanEval+ evaluation occurred. The existing pinned sandbox was not needed in this task because the behavior gate was never reached.

# Group-32 and bounded DWQ experiment

The proposed learning-rate follow-up has now completed as a [gated diagnostic](DWQ_DIAGNOSTIC.md); it stopped at the loss gate with pristine C selected. This report retains the earlier measurements and decision.

D did not improve the matched development result. A 32-update DWQ pass on C completed and reduced development generation time without increasing weights, but worsened teacher validation loss, added truncations and did not beat MXFP4's solved count. **No release is promoted.** C remains an experimental control. This is completed optimization work, including a preserved failed calibration attempt, rather than a release claim.

Evidence: [receipts and raw archives](../results/refinement-v1/), [frozen decisions](../results/refinement-v1/predeclared.json), [paired development results](../results/refinement-v1/summary.json). Local execution used the existing RTX 5090 environment: MLX 0.32.3, MLX-LM 0.32.0, Python 3.12. The remote checkout's recorded Git revision differs from the local research branch; changed scripts were copied and their hashes verified. [Runtime bindings](../results/refinement-v1/runtime-code.json) and exact executed training snapshots preserve that distinction. CUDA timings below are not Mac performance.

## D: only expert down-projection groups change

D starts from pinned original BF16 `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, never from an existing quant. Like C, attention projections, token embeddings and the untied output head use affine 6-bit/group 64. Expert matrices remain affine 4-bit; only the 28 `switch_mlp.down_proj` modules use group 32. Expert up/gate projections remain group 64, native routers remain 8-bit/group 64, and excluded tensors remain BF16. Tokenizer, template and architecture are preserved.

Actual down-projection shapes are `[64, 2304, 896]` in each layer. Per-module group size now determines scale/bias overhead in `estimate_payload()`. Tests check the **231,211,008-byte** tensor-payload increase, exact policy coverage, protected routers and exclusions. Before conversion, D estimated 7,329,688,064 tensor bytes and 7,330,736,640 bytes including the existing 1 MiB header allowance. Actual serialized weights: **7,329,780,326 bytes**, versus C's 7,098,569,272; the extra 46 bytes beyond the tensor delta are serialized metadata. No recipe was changed to meet the 7.5 decimal GB ceiling.

D passed checksum/index/router/tokenizer checks, serialized-to-loaded parameter equality, excluded-tensor equality to BF16, save/reload equality and finite fresh-process inference. Calibrated C passed the same gates. Short repeat/reload probes were identical; the integrity assessor distinguishes observed same-artifact repeat variability from reload defects instead of imposing global bitwise CUDA determinism. Realized precision maps and artifact hashes are in conversion receipts.

## Calibration and the failed attempt

The predeclared rule selected C because D did not improve the historical C development control. Calibration uses [32 training and eight validation snippets](../fixtures/refinement-v1/), frozen before candidate selection: complete Python 3.12 standard-library functions, PSF licensing retained, tokenized with the original tokenizer, no development/heldout prompts or generated answers. There are only **4,292 scored training tokens and 915 validation tokens**; this is a modest calibration, not broad coding coverage.

A separate BF16 process cached upstream DWQ's top-1,024 teacher logits. Cache receipts bind source hashes, tokenizer, corpus, ordering and upstream implementation. The student pass uses batch 1, sequence limit 256, seed 123, temperature 2 and Adam learning rate 1e-6 with bias correction. Only eligible affine scales/biases below eight bits can change. Packed weights, routers, excluded tensors, schemas and tokenizer files are checked unchanged; all gradients and floating parameters must be finite. New outputs refuse existing evidence/artifact paths.

The first pass **failed after 13 completed updates** with CUDA out-of-memory. No artifact was saved. Its receipt's legacy `updates: 32` is the planned budget, not the actual completed count; [attempts.json](../results/refinement-v1/attempts.json) makes this explicit. Logs and exact source are retained. Measured stage time was 23.981 s, MLX peak 22.185 GB; allocator memory excludes driver/cache/system overhead, so this does not establish the allocation's precise cause.

The sole retry enabled upstream gradient checkpointing, preserving data, seed, optimizer and update budget. All 32 updates completed in 33.185 s, MLX peak 21.115 GB. **213 eligible scale/bias arrays changed**. Weight size stayed exactly 7,098,569,272 bytes. Printed validation loss worsened **0.103 → 0.158** and upstream emitted its quality-degradation warning. This is negative evidence, not a successful fidelity recovery. No learning-rate sweep followed. The final runner also corrects an unused target-stage argument; the exact training-used snapshot remains archived, and the executed training algorithm is unchanged.

## Matched CUDA development screen

The original 24-task fixture remains development data because it influenced the recipes. Greedy 8,192-token budget including reasoning, seed 0, upstream chat template, extraction, prefill 512 and pinned isolated Docker execution are unchanged. Models ran sequentially with no competing conversion/download. Current C and MXFP4 were rerun under the same runtime as D and DWQ. All prompts and task order match. Every attempt, including failed generation time, is retained.

| Model | Weight GB | Passed | Total generation s | Failed generation s | Completion tokens | Trunc / empty | Peak MLX GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C control | 7.099 | 21/24 | 379.0 | 76.4 | 82,429 | 1 / 1 | 8.56 |
| D group 32 | 7.330 | 20/24 | 401.2 | 133.3 | 86,845 | 2 / 2 | 8.83 |
| C + DWQ | 7.099 | 21/24 | 356.4 | 98.1 | 77,239 | 2 / 1 | 8.64 |
| Pinned MXFP4 | 6.457 | 22/24 | 359.5 | 69.1 | 81,853 | 1 / 1 | 7.96 |

Decimal GB; MLX allocator peak includes server load, excludes driver/system memory. Total generation is **1,496.1 s (24.9 min)**; including per-run startup/tests/shutdown, **1,633.5 s (27.2 min)**. First-task compilation is included. Setup/conversion/calibration are separate.

D versus current C gains `NE/21` but loses `NE/08` and `NE/16`. D's two reasoning truncations (`NE/05`, `NE/08`) produce empty answers; `NE/11` incorrectly decodes JSON-pointer escapes before splitting, and `NE/16` fails with `UnboundLocalError` from a missing `nonlocal` binding. D gains no task over MXFP4 and loses `NE/08`, `NE/16`; it is larger and slower.

DWQ versus C gains `NE/11`, `NE/21` and loses `NE/08`, `NE/16`. It uses 6.3% fewer completion tokens and 6.0% less total generation time, but has two truncations and more failed-generation time. Against MXFP4 it gains `NE/11`, loses `NE/08`, `NE/16`, solves one fewer task and uses 9.9% more weight bytes. Its total generation time differs by less than 1%; this does not establish a useful-speed advantage.

Historical C was 22/24, 399.7 s, 87,062 tokens with two truncations; the same artifact now gets 21/24 with one truncation. Historical MXFP4 was 21/24, 323.2 s, 73,537 tokens with no truncations; now 22/24 with one. Earlier receipts remain intact. These greedy CUDA repeats demonstrate task/completion variability; they do not diagnose its source or establish superior intelligence from small score differences. Exact paired p-values are descriptive and all small differences here are inconclusive.

## Separate untouched evaluation

The existing previously unopened 12-task heldout fixture was frozen before selection and copied with its original manifest into [refinement-v1/heldout](../fixtures/refinement-v1/heldout/). No earlier heldout outputs existed. Calibration is separate. The archived existing evaluator uses its original greedy seed 123, 2,048-token budget and prefill 512 equally for all three models; code runs only in the pinned isolated container. This protocol differs from the development screen and its scores are not pooled.

| Model | All tasks | Code subset | Total generation s | Tokens | Truncations |
| --- | ---: | ---: | ---: | ---: | ---: |
| C | 8/12 | 2/4 | 43.953 | 9,485 | 3 |
| C + DWQ | 9/12 | 3/4 | 46.319 | 9,979 | 2 |
| MXFP4 | 8/12 | 3/4 | 35.334 | 7,906 | 2 |

DWQ gains only `held-rename` over C and only `held-tool1` over MXFP4; it takes 31.1% more generation time than MXFP4 and generates 26.2% more tokens. This tiny mixed synthetic set (four code, four exact-response, two JSON, two tool tasks) is neither a fresh repository-task evaluation nor HumanEval+. The one-task difference does not establish superiority.

## Matched M4 hardware and editor workload

The selected refined recipe, **C + DWQ**, completed the existing M4 hardware gate against pinned MXFP4. The predeclared hardware-only selection required no fewer development solves than fresh C and fewer tokens/less generation time; it did not override degraded validation loss or promote a release. Both artifacts passed complete conversion/hash checks before loading and the 12-snippet finite-logit smoke. Benchmark and scoring scripts were unchanged and hash-matched. An older Mac validation helper lacked receipt support; it was preserved and replaced by the current helper before any measurement. Interrupted transfers and that preflight block are recorded, with no failed model measurement hidden.

Apple M4, 16 GB, AC power; low-power mode was observed off after measurement. One warmup plus three measured trials per model/context, greedy seed 0, prefill 512, exactly 256 generated tokens, fixed synthetic prompts. Runs were sequential, DWQ then MXFP4, after transfers finished. Visual Studio Code had convert.py selected and idle, verified unlocked before and after; other normal desktop applications stayed open, with no typing/builds or settings changes. This is an idle editor workload, not an active IDE stress test.

| Model | Context | Mean prompt tok/s | Mean decode tok/s | Mean TTFT s | Max measured MLX GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| C + DWQ | 1K | 536.7 | 42.9 | 1.909 | 7.60 |
| C + DWQ | 4K | 476.3 | 41.7 | 8.601 | 7.62 |
| C + DWQ | 16K | 428.2 | 37.6 | 38.262 | 7.83 |
| MXFP4 | 1K | 532.8 | 51.6 | 1.923 | 6.96 |
| MXFP4 | 4K | 476.9 | 48.9 | 8.589 | 6.99 |
| MXFP4 | 16K | 426.2 | 44.2 | 38.443 | 7.19 |

Uncalibrated C was not included in this M4 comparison; the decode gap cannot be attributed specifically to DWQ rather than C’s underlying recipe or execution effects. The initial pressure event occurred during finite inference, not ordinary serving, and MXFP4 inherited existing swap.

DWQ decode throughput was **16.8%, 14.8% and 14.8% lower**, respectively. Prompt throughput was similar. Individual measured 16K decode trials ranged 35.8–38.6 for DWQ and 42.9–44.8 for MXFP4; all trials remain in receipts. These synthetic repeated-prefix results establish neither full coding-response latency nor an ideal unloaded-Mac speed ceiling. They are independent of CUDA timings and earlier Mac measurements.

Whole-system counters were sampled before, periodically (nominally every two seconds), and after each stage, independently of MLX allocator memory. **During DWQ's initial finite smoke, swap grew from 823.88 to 4,709.38 M and two samples reported pressure flag 4 (critical).** All later throughput-stage samples reported flag 1 (normal). The exported sysctl uses dispatch flags, rather than the kernel's internal enum: [Apple's sysctl handler](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_memorystatus_notify.c) converts the value, and [its flag definitions](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/event_private.h) map 1 to normal and 4 to critical. Source hashes are in pressure-mapping.json.

Across the whole sequence, swap ended at **4,149.25 M**, a net **+3,325.37 M** from the starting state; the sampled maximum was 4,709.38 M. Figures use sysctl's reported M units. The MXFP4 stages inherited existing swap from DWQ, and other applications remained open. Fixed order, preserved OS caches, and periodic sampling prevent per-model causal attribution or continuous-peak claims. The initial pressure event and material swap do **not** support swap-free operation or comfortable normal-work headroom from a 7.83 GB allocator reading alone. No cache-clearing or system configuration changes were made.

[Mac summary](../results/refinement-v1/mac-summary.json), benchmark trial receipts, finite probes, editor verification and lossless system-snapshot archive are retained. All eight stages completed. To reproduce this gate in a fresh Mac test checkout/result directory with both exact artifacts installed, open the defined editor workload, record fresh editor verification, copy the selection/metadata and archived mac-checks.py into the same paths shown in start-mac-used.sh, then run:

```sh
/usr/bin/caffeinate -i .venv/bin/python -u scripts/refinement-mac-checks.py > results/refinement-v1/mac-gate.log 2>&1
```

The driver refuses existing observations and each measurement refuses existing output. Individual matched benchmarks use `scripts/benchmark.py MODEL --context 1024|4096|16384 --output NEW_FILE`; the sampler adds the separate whole-system observations. All models and fixtures remain frozen.

## Reproduction

Use the existing pinned BF16 source, native-4 receipt, C artifact, MXFP4 and CUDA environment. Run in a fresh result directory and without an existing D/calibrated artifact or teacher cache; scripts refuse overwrite. Preserve failures rather than retrying into their paths.

```sh
.venv/bin/python scripts/convert.py --recipe D --estimate-only
.venv/bin/python scripts/convert.py --recipe D
mkdir -p work/refinement-repeat
.venv/bin/python scripts/validate.py artifacts/mellum2.1-mixed-D-g64 --receipt work/refinement-repeat/structure-D.json
.venv/bin/python scripts/nonexpert.py integrity --model artifacts/mellum2.1-mixed-D-g64 --output work/refinement-repeat/integrity-D.json
.venv/bin/python scripts/nonexpert.py screen --label D --model artifacts/mellum2.1-mixed-D-g64 --output work/refinement-repeat/screen-D.json
.venv/bin/python scripts/refine_dwq.py targets --cache work/refinement-repeat-targets --output work/refinement-repeat/targets.json
.venv/bin/python scripts/refine_dwq.py train --cache work/refinement-repeat-targets --student artifacts/mellum2.1-mixed-C-g64 --destination artifacts/mellum2.1-refinement-dwq --output work/refinement-repeat/train.json
.venv/bin/python scripts/validate.py artifacts/mellum2.1-refinement-dwq --receipt work/refinement-repeat/structure-DWQ.json
.venv/bin/python scripts/nonexpert.py integrity --model artifacts/mellum2.1-refinement-dwq --output work/refinement-repeat/integrity-DWQ.json
.venv/bin/python scripts/nonexpert.py screen --label DWQ --model artifacts/mellum2.1-refinement-dwq --output work/refinement-repeat/screen-DWQ.json
cp results/nonexpert-v1/integrity-C.json work/refinement-repeat/
.venv/bin/python scripts/nonexpert.py screen --label C --model artifacts/mellum2.1-mixed-C-g64 --output work/refinement-repeat/screen-C.json
.venv/bin/python scripts/nonexpert.py screen --label mxfp4 --model work/mxfp4 --output work/refinement-repeat/screen-mxfp4.json
.venv/bin/python scripts/summarize_nonexpert.py --results work/refinement-repeat --labels C D DWQ mxfp4
for pair in 'C artifacts/mellum2.1-mixed-C-g64' 'DWQ artifacts/mellum2.1-refinement-dwq' 'mxfp4 work/mxfp4'; do
  set -- $pair
  .venv/bin/python -c 'import sys; from pathlib import Path; sys.path.insert(0,"scripts"); sys.path.insert(0,"results/refinement-v1/runtime-code"); import optimization as o; o.DATA=Path("fixtures/refinement-v1/heldout"); o.RESULTS=Path("work/refinement-repeat"); o.main()' evaluate --split heldout --model "$2" --output "work/refinement-repeat/heldout-$1.json"
done
```

Exact original orchestration scripts and training sources are retained beside receipts. `archive.json` binds lossless gzip files to original hashes; uncompressed originals remain on CUDA and in ignored local `work/refinement-v1-evidence`. The summary tool expects decompressed raw files beside receipts. `runtime-code/verify-evidence.py` checks archived raw hashes, identical prompts, usage counts, totals and complete coverage from the repository root. Large weights, teacher logits and caches stay out of Git.

Targeted validation passed: 37 local tests including four pre-existing pilot tests; 33 remote tests, compilation and diff whitespace checks. No dependencies or unrelated system settings changed. Full HumanEval+ was not run because neither recipe merits promotion. These small single runs and 4.3K calibration tokens limit general conclusions.

**Exactly one next action:** repeat C's bounded DWQ pass at learning rate 1e-7 with the same frozen corpus and uncalibrated/MXFP4 controls, to test whether a smaller update avoids the observed validation-loss degradation without buying more memory.

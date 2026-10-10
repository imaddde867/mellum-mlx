# Exploratory uninterrupted U/S comparison

Both arms completed 128 updates in one fresh process each, sequentially from pristine C. No resume, pristine retry, extra epoch or setting change occurred. The abandoned seven-update attempts remain separate historical failures. The historical restart qualification remains **FAILED**; the generic validator still rejects it.

The separately frozen [exploratory protocol](protocol.json) records exact identical-gradient Adam replay, resident/resident gradient variability, the unresolved numerical cause and limited memory headroom. Successful optimization and forward fidelity do not establish correct, deterministic, unbiased or generally qualified backwards.

All means below use all five repeats. Repeat ranges are empirical engineering screens, not confidence intervals. The original assistant-fidelity and individual-region gates were unchanged; the historical 5% cross-backward test was not used for selection.

## Validation learning curves

| Arm | Step | Assistant | Reasoning | Transition | Final | Tool call | Context | Whole transcript |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U | 0 | 0.090957 | 0.150340 | 0.146493 | 0.078236 | 0.060774 | 0.129903 | 0.113033 |
| U | 8 | 0.091512 | 0.150226 | 0.146424 | 0.077878 | 0.062883 | 0.133268 | 0.115180 |
| U | 32 | 0.090440 | 0.149064 | 0.145261 | 0.076851 | 0.061829 | 0.127461 | 0.111425 |
| U | 64 | 0.054883 | 0.102951 | 0.102229 | 0.042536 | 0.032333 | 0.071312 | 0.064196 |
| U | 96 | 0.046841 | 0.088494 | 0.085194 | 0.035419 | 0.028665 | 0.057675 | 0.052982 |
| U | 128 | 0.031662 | 0.061482 | 0.055117 | 0.022587 | 0.020477 | 0.049012 | 0.041496 |
| S | 0 | 0.090927 | 0.150221 | 0.146161 | 0.078362 | 0.060674 | 0.129796 | 0.112959 |
| S | 8 | 0.090748 | 0.149375 | 0.145467 | 0.078030 | 0.061153 | 0.128329 | 0.112050 |
| S | 32 | 0.090067 | 0.149601 | 0.145401 | 0.077452 | 0.059716 | 0.130251 | 0.112845 |
| S | 64 | 0.056383 | 0.106181 | 0.103983 | 0.044008 | 0.032830 | 0.108536 | 0.085944 |
| S | 96 | 0.043317 | 0.079626 | 0.078640 | 0.034141 | 0.026198 | 0.097475 | 0.074015 |
| S | 128 | 0.028477 | 0.053545 | 0.044546 | 0.021143 | 0.019458 | 0.088015 | 0.062225 |

All checkpoint region/family means, minima, maxima and ranges are retained in [learning-curves.json](learning-curves.json), [region CSV](learning-curves.csv) and [family CSV](family-curves.csv). The arm receipts retain every individual trajectory and repeat.

## Validation families

Assistant-only token-weighted KL per underlying problem family:

| Family | C in U process | C in S process | U step 128 | S step 128 |
|---|---:|---:|---:|---:|
| cursor_page | 0.075629 | 0.075952 | 0.027762 | 0.023447 |
| decode_frames | 0.086352 | 0.086258 | 0.030753 | 0.027879 |
| dependency_order | 0.100874 | 0.100498 | 0.029343 | 0.028192 |
| job_state | 0.091344 | 0.091260 | 0.035652 | 0.031321 |
| markdown_titles | 0.088809 | 0.088900 | 0.041660 | 0.037185 |
| query_values | 0.102461 | 0.102597 | 0.031857 | 0.030663 |
| transpose_rect | 0.105726 | 0.105518 | 0.033655 | 0.028784 |
| undo_journal | 0.079873 | 0.079975 | 0.024485 | 0.021942 |

Both arms improve assistant fidelity in all eight validation families. S has lower assistant KL than U in each family in this attempt; this does not establish objective superiority. Family-level results for every individual region at every scheduled step are in the family CSV.

## Selection and artifact verification

The original rule selected **S, step 128** after both arms completed. Its assistant mean is 0.028477 versus U's 0.031662. Every assistant-region guardrail passes against both pristine references and best U. Context KL is higher for S (0.088015) than U (0.049012), and whole-transcript KL is higher (0.062225 versus 0.041496). Both remain below pristine C.

The selected artifact remains outside Git at `work/calibration-v2-exploratory-selected` on 5090-1. Export checked protected parameters and schemas; 137 eligible BF16 tensors changed. A fresh process checked all 791 parameter digests exactly. Five reload forwards averaged assistant KL 0.028447 and passed both original pristine references without an assistant-region regression. [Selection](selection.json), [export](artifact-export.json) and [reload](artifact-reload.json) retain the bindings and complete evidence.

## Memory and execution

Each update records finite-gradient checks, atomic exact serialized-state readback, MLX active/cached memory and cudaMemGetInfo point samples. The latest two verified rolling states per arm and five scheduled selection states remain outside Git. They were never used to resume. Final host hash checks confirmed the latest two states and their immutable bindings; final artifact checks confirmed published C, pinned MXFP4 and the selected model files remained unchanged.

| Arm | Updates | Process elapsed seconds | MLX allocator peak bytes | Maximum sampled device-used bytes |
|---|---:|---:|---:|---:|
| U | 128 | 1066.665 | 26365915144 | 20702232576 |
| S | 128 | 1064.048 | 26365915144 | 20716322816 |

Allocator peak is not total VRAM. Device samples occur before backward and after update; they do not measure the device-wide peak during backward. Cached MLX bytes were zero at those boundaries. NVML reported a driver/library mismatch; the working cudaMemGetInfo interface was used. No driver or package changes were made.

## Reserved behavior

The candidate was locked after reload passed and before reserved access. All three single-process evaluations completed 32 tasks. Initial requests were checked identical across models for every task; serving, greedy seed, tools and budgets were fixed. No retries, extra responses or post-evaluation tuning occurred.

| Model | Correct / 32 | Valid emitted calls | Missing final | Token limit | Response limit | Failed tool tests | Completion tokens | Generation seconds | Conversation seconds | Process seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S step 128 | 16 | 110/110 | 16 | 1 | 15 | 7 | 124525 | 610.294 | 655.284 | 677.125 |
| Pristine C | 19 | 107/107 | 12 | 0 | 12 | 4 | 104822 | 519.516 | 545.062 | 570.436 |
| Pinned MXFP4 | 23 | 79/79 | 7 | 3 | 4 | 6 | 132674 | 617.242 | 641.647 | 669.892 |

Generation seconds sum all HTTP requests, including failed tasks; conversation seconds also include tool execution. Process seconds additionally include startup and binding checks. Token and response limits are distinct. There were no unknown token counts, invalid emitted calls or terminal transport/schema task errors. Failed sandbox tests within tool conversations remain recorded; they are distinct from final executable correctness.

All 96 task records, every raw request/response, tool result, sanitized final and server log are retained under `behavior-candidate/`, `behavior-C/` and `behavior-MXFP4/`. [Behavior summary](behavior-summary.json) contains metric definitions, paired family outcomes and source hashes. The original assertions are a bounded correctness oracle, not comprehensive hidden tests.

**Interpretation:** S achieved a provisional assistant-fidelity gain but lower reserved correctness and higher generation cost than pristine C. This does not support promotion or further tuning of this exact artifact. The large fidelity signal is worth an independent, predeclared research replication to test repeatability; no such replication was performed here. Neither objective superiority nor practical usefulness is established. The negative behavioral result applies to this exact exploratory attempt.

## Validation and reproduction

The local suite passed 87 tests (13 MLX-dependent skips); the pinned runtime suite passed 82 tests with no skips. The subsequently added CPU reporting regression also passed on the pinned host.

## Reproduction and limits

Execution-process receipts record the exact commands and subprocess environment. The new mode is `calibration_v2_runtime.py train --exploratory --arm U|S --output <frozen path>`; selection uses `select --exploratory --u <new U path> --s <new S path>`. Existing paths and resume arguments are rejected. The artifact helper exports once and verifies in a separate fresh process; the behavior helper requires matching decision, reload and lock bindings before reading tasks.

No independent training replication was performed. Any improvement remains provisional and requires a new, independent, predeclared replication before promotion. No M4, HumanEval+, recipe changes, dataset expansion, package changes, uploads or push occurred.

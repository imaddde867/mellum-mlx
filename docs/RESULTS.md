# Measurement notes

Initial measurements: 2026-10-09. RTX 5090, 32 GB VRAM, driver 595.91.07. MLX 0.32.3 / MLX-LM 0.32.0. CUDA and Metal results are reported separately below.

## Artifact checks

Both native conversions passed complete source/artifact checksum validation, serialized tensor/index consistency (791 tensors), EOS 28, unchanged vocabulary/templates, three chat/tool prompt cases, and 28 native 8-bit router entries.

| Artifact | Weight bytes | Decimal GB |
| --- | ---: | ---: |
| Affine 4-bit/g64 | 6,836,687,350 | 6.84 |
| Affine 6-bit/g64 | 9,873,101,418 | 9.87 |

The 4-bit conversion reported 4.501 effective bits per weight. Nominal bitwidth does not equal complete artifact precision or runtime memory.

## Short numerical probe

Twelve authored snippets, 547 scored tokens, raw text tokenization, teacher forcing, no calibration. These are smoke/regression observations; the small hand-authored fixture is not representative of coding capability.

| Model/runtime | Mean negative log-likelihood | Delta vs MLX BF16 | Top-1 agreement vs MLX BF16 |
| --- | ---: | ---: | ---: |
| BF16 / MLX CUDA | 1.208707 | — | — |
| 4-bit / MLX CUDA | 1.358777 | +0.150070 | 90.13% |
| 6-bit / MLX CUDA | 1.230426 | +0.021719 | 96.53% |
| Existing MXFP4 / MLX CUDA | 1.290664 | +0.081957 | 92.14% |
| BF16 / Transformers CPU | 1.212631 | +0.003924 | 97.26% |

The independent reference uses PyTorch 2.14.1+cpu, Transformers 5.19.0, BF16, eager attention and eight CPU threads. CPU versus CUDA arithmetic differs. This probe does not establish complete architecture parity; its observed disagreement is reported, not declared resolved. An initial synthetic cache-boundary diagnostic is reported below; independent long-context architecture parity remains pending.

## Protocol fixtures

The installed parser/serializer preserved single/multiple calls, escaped JSON arguments, reasoning and streaming indices in four fixed cases. Malformed truncated JSON was rejected. This does not reproduce the entire HTTP handler or prove model-generated agent behaviour. See `results/cuda/protocol-fixtures.json` for the exact server source hash.

## Pending

Independent long-context numerical audit, broader live HTTP scenarios, resolution of the intermittent DWQ discrepancy, and native-conversion Apple Silicon validation. Advanced DWQ calibration is deferred until after the initial release. Comparator-only Mac measurements do not support claims about our native conversions or superior coding quality.

## DWQ compatibility pilot

A one-update, batch-one, <=64-token pilot used separately authored training/validation data and cached BF16 top-1,024 teacher logits. CUDA backward execution completed and routers were unchanged. The first attempt failed save/reload with a maximum logit difference of 0.875. An instrumented repeat passed with identical parameters, zero repeated-forward difference and zero save/reload difference, using 14.78 GB peak MLX memory. **The initial failure is unresolved**; this is not evidence of a reliable DWQ release or improved coding quality. Both outcomes are retained in the result receipt.

## CUDA synthetic throughput

Warmup excluded; three greedy trials per cell, 256 generated tokens, prefill step 512. Repeated-code prefix; all trials reached the token budget. This measures throughput rather than task success. Memory is the maximum MLX allocator peak over measured trials, excluding OS/swap.

| Candidate | Context | Mean prompt tok/s | Mean decode tok/s | Mean TTFT (s) | Max MLX GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4bit | 1K | 1798.6 | 320.2 | 0.570 | 9.13 |
| 4bit | 4K | 1619.1 | 282.4 | 2.530 | 10.60 |
| 4bit | 16K | 1594.8 | 272.4 | 10.274 | 10.90 |
| 6bit | 1K | 1200.3 | 224.9 | 0.853 | 12.16 |
| 6bit | 4K | 1165.8 | 215.6 | 3.514 | 13.63 |
| 6bit | 16K | 1157.2 | 207.0 | 14.159 | 13.93 |
| mxfp4 | 1K | 2116.0 | 326.5 | 0.484 | 8.74 |
| mxfp4 | 4K | 2046.3 | 231.5 | 2.003 | 10.22 |
| mxfp4 | 16K | 1933.8 | 274.1 | 8.476 | 10.52 |

## Live HTTP smoke test

A temporary loopback-only MLX-LM server generated a valid `add(2, 3)` tool call, preserved its ID and JSON arguments, consumed a synthetic tool result, and answered exactly `5`. Both streaming and non-streaming modes passed with the upstream thinking template, greedy sampling and a 1,024-token output budget. No generated tool/code was executed. This is one simple scenario per mode, not a general agent-reliability score. The server was stopped after the probe. Raw responses are in `results/cuda/http-probe.json`.

## M4 Metal comparator measurements

14-inch MacBook Pro, Apple M4 (10 CPU / 10 GPU cores), 16 GB unified memory, macOS 27.0.1, AC power. Lid closed; the user reports Amphetamine keep-awake. MLX 0.32.3 / MLX-LM 0.32.0. The pinned external MXFP4 artifact passed all checksums in its public receipt. Our native 4-bit conversion has completed transfer and checksum validation; its measurements are running and are not included in this table.

Same synthetic throughput protocol as CUDA: warmup plus three measured greedy trials, 256 output tokens, prefill step 512. All trials reached the output budget. A separate artifact transfer ran concurrently; these are preliminary measurements rather than isolated performance rankings.

| Comparator | Context | Mean prompt tok/s | Mean decode tok/s | Mean TTFT (s) | Max MLX GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXFP4 | 1K | 663.9 | 67.2 | 1.543 | 6.96 |
| MXFP4 | 4K | 623.7 | 64.3 | 6.567 | 6.99 |
| MXFP4 | 16K | 521.2 | 57.6 | 31.436 | 7.19 |

The 547-token numerical smoke probe had mean NLL 1.279368. Against the *same MXFP4 artifact on CUDA*, Metal NLL differed by -0.011295 and top-1 predictions agreed on 96.89% of positions. The report's BF16 baseline is the CUDA reference, not a Mac BF16 run; its delta is not a within-Metal quantization-quality estimate. Runtime disagreement remains under investigation. The simple addition tool round trip passed in both live HTTP modes on Metal.

Aggregate system snapshots before/after the initial 1K run are in `results/mac/system-snapshots.json`. Swap usage was 67.94 MB in both committed snapshots. Concurrent downloads and other applications limit interpretation of these snapshots. These snapshots do not measure whole-system peak memory or establish swap-free operation.

## CUDA cache-boundary diagnostic

Native 4-bit, repeated authored fixture text, lengths 1,023 / 1,024 / 1,025 / 2,049, last eight output positions. Compared repeated uncached forwards, cached single-prefill forwards, and chunk sizes 256 / 512 followed by eight single-token forwards. All 16 comparisons were finite and agreed on top-1 predictions at the selected positions. These highly predictable repeated snippets and eight-position samples are weak evidence; they do not establish complete cache correctness.

Raw logits differed even on repeated uncached forwards. The cause is unresolved; do not interpret cached-versus-uncached differences as an isolated cache defect or declare exact reproducibility. See `results/cuda/4bit-cache-probe.json`.

The first diagnostic ran out of CUDA memory on its final uncached 2,049-token forward. A fresh process completed that forward at 18.71 GB MLX peak. Clearing allocator cache between contexts enabled the full diagnostic; before the final context, 6.84 GB was active and 12.77 GB was cached. The initial failure and retry are retained in `results/cuda/cache-first-run-failure.json`. This change controls the diagnostic's allocation retention; it is not an upstream model fix.

## Completed HumanEval+ comparison

All four CUDA candidates completed the same 164 tasks under [the fixed protocol](CODING_PROTOCOL.md), one greedy sample per task, 8,192 completion tokens. Every failure was retained. Full samples, raw responses, original generation receipts, evaluation protocols, sanitized samples and task-level outcomes are in `results/coding/humaneval-plus/`. Post-run archive receipts bind the committed files; the original runs lack generation-time sample and weight checksum binding. This cannot be reconstructed retroactively and is explicitly recorded. New runs enforce those bindings.

| Candidate | Base passed / 164 | Expanded passed / 164 | HumanEval+ | Truncated | Empty final answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 158 | 153 | 93.3% | 3 | 3 |
| Native 4-bit/g64 | 141 | 133 | 81.1% | 18 | 18 |
| Native 6-bit/g64 | 156 | 146 | 89.0% | 3 | 3 |
| Existing MXFP4 | 153 | 144 | 87.8% | 6 | 4 |

For 6-bit versus MXFP4, 137 tasks passed both, nine passed only 6-bit, seven passed only MXFP4, and eleven passed neither. Exact two-sided McNemar p = 0.8036. This single run does not establish a quality advantage. Wilson 95% intervals are recorded in `summary.json` as descriptive task-sampling uncertainty; they do not account for benchmark selection, training overlap or runtime variability.

Native 6-bit is the leading native release candidate pending M4 validation. MXFP4 remains the better demonstrated quality/size balance. Native 4-bit has no demonstrated quality/size advantage and is not recommended as the default. These tasks do not establish repository-agent reliability, and CUDA greedy decoding is not bitwise deterministic.

## Isolated M4 native 4-bit comparison

The native 4-bit artifact passed every conversion checksum on the Mac, completed the short finite-logit fidelity probe, and passed the simple streaming/non-streaming HTTP addition round trip. The throughput comparison below ran sequentially with no artifact transfer active, on AC power, using the same warmup plus three measured trials and 256-token synthetic protocol. The earlier transfer-concurrent MXFP4 measurements remain preserved separately.

| Candidate | Context | Mean prompt tok/s | Mean decode tok/s | Mean TTFT (s) | Max MLX GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native 4-bit | 1K | 673.8 | 65.8 | 1.520 | 7.34 |
| Native 4-bit | 4K | 632.1 | 63.3 | 6.480 | 7.37 |
| Native 4-bit | 16K | 527.6 | 56.9 | 31.056 | 7.57 |
| MXFP4 isolated repeat | 1K | 639.2 | 67.3 | 1.602 | 6.96 |
| MXFP4 isolated repeat | 4K | 591.6 | 64.5 | 6.926 | 6.99 |
| MXFP4 isolated repeat | 16K | 521.1 | 58.4 | 31.440 | 7.19 |

Native 4-bit showed higher synthetic prompt throughput, but slightly lower decode throughput and higher MLX memory than MXFP4. This limited workload does not justify recommending native 4-bit as the default given its lower coding score and larger artifact.

Periodic system counters are in `results/mac/native-comparison-system-snapshots.json`. Swap used increased from 181.75 MB to 282.81 MB across the initial fidelity stage, then remained at 282.81 MB in the sampled throughput/HTTP stages. All sampled memory-pressure values were normal (level 1). These are periodic observations, not continuous peaks; other applications and existing swap prevent causal attribution or a claim of swap-free operation. Native 6-bit transfer and validation are the next release gate.

## Contained repository-agent integration

Native 6-bit, RTX 5090, the same greedy 8,192-token response budget, allowlisted tools, fixed regressions, and generated code executed only inside the pinned offline unprivileged Docker container. Repository: `imaddde867/whatisit-macos` at `fbdcb10b289001baf2c0e444bb963e421262a4fd`. The original checkout was not modified.

The mutation-routing task failed its bounded trials. The model emitted invalid tool-call JSON with unescaped quotes in code. MLX-LM 0.32.0 logged `Expecting ',' delimiter` at `work/repository-agent/server.log:29` and `work/repository-agent/retry-edit/server.log:47`, then dropped the invalid calls from the client response while still returning `finish_reason: "tool_calls"` with no calls. The whole-file harness incorrectly treated that response as finished. These are separate model, serving and harness contributions; the warnings do not establish truncation as the cause. Both exact log lines are retained in [parser-warnings.txt](../results/repository-agent/whatisit-macos/parser-warnings.txt).

Other routing attempts introduced an unrepaired syntax error, used insufficient substring matching, or exhausted the reasoning/output budget. A fresh trial with corrected plain-text file responses still failed. Every transcript, test outcome and harness is preserved. Greedy sampling and failure to replay prior reasoning as `reasoning_content` were further harness limitations.

A separate smaller retrieval-limit task succeeded. The baseline failed; Mellum read the source, added a two-line integer type check that rejects booleans/non-integer values before database access, and passed all 36 fixed repository/regression tests. The generated patch was independently reviewed. Its transcript, source snapshots, exact harness and test outputs are in `results/repository-agent/whatisit-macos/limit-validation/`.

This demonstrates a contained tool/read/edit/test workflow on one task and exposes failures on another. It is not a repository-agent success rate, broad reliability proof or a Metal agent result. Adapter changes, retries and the separate continuation are documented; successful evidence does not replace failed attempts.

## Completion-length audit

On the same paired tasks where both generations stopped (n = 144), native 4-bit used about **2.2× longer reasoning-heavy completions than BF16** (paired Wilcoxon p ≈ 5×10⁻²⁵). Median completion tokens across all tasks were 3,242 versus 1,571. All 18 native 4-bit budget truncations ended during reasoning. The longer thinking and budget exhaustion help explain its lower score under the fixed 8,192-token protocol; they do not establish the underlying numerical cause. Completion-token counts include reasoning and final output, rather than a separately measured reasoning-token count.

The supplied [length audit](../scripts/length_audit.py) reproduces paired outcomes, completion-token distributions and solved-within-budget counts from committed evidence (NumPy and SciPy required). The server and harness patches are in `patches/`. The temperature-1.0, 16K-budget, reasoning-replay routing rerun with seeds 0, 1 and 2 starts only after publication; every transcript will be retained.

## Native 6-bit M4 release gate

The 16K synthetic run and streaming/non-streaming HTTP addition probe passed. Three measured trials excluding warmup gave 48.0 / 47.1 / 43.5 decode tok/s and 1.609 / 7.099 / 32.450 s TTFT at 1K / 4K / 16K. Peak MLX memory was 10.31 GB at 4K and 10.51 GB at 16K. Swap grew materially across the sequence (282.81 MB initially; about 1.9 GB during later stages), although it decreased from 1,947.31 MB to 1,883.31 MB within the sampled 16K stage. Editor-open status was not recorded. The release card retains **16 GB: not recommended**. Raw reports are under `results/mac/6bit-*.json`. This gate covers the measured workloads, not broad agent reliability or independent architecture parity.

## Post-publication patched routing rerun

**1 success out of 3** with seeds 0, 1 and 2. Separate patched MLX-LM 0.32.0 venv, supplied serving/parser patch, temperature 1.0, 16K tokens per turn, prior reasoning replayed as `reasoning_content`, at most 12 turns, identical fixed tests and isolated original-source copies.

Seed 0 exhausted its turn budget with eight regression failures. Seed 1 passed all 36 fixed repository/regression tests. Seed 2 introduced a syntax error and emitted an invalid repair call; the patched server returned its failed text, but the harness ended on a no-call stop response. Final tests retained that failure (six import errors). Every transcript, patch and outcome is in [rerun-patched](../results/repository-agent/whatisit-macos/rerun-patched/README.md). Original source checkout and original runtime remain unchanged. This one-task result is not a general reliability rate or an isolated causal test of the patch.

The completion audit was independently reproduced using NumPy 2.5.3 and SciPy 1.18.1: 4-bit/BF16 paired geometric mean **2.22×**, median ratio **2.14×**, n = 144, Wilcoxon p = **5.1×10⁻²⁵**. Exact output is retained in `results/coding/humaneval-plus/length-audit.txt`.

## Mixed non-expert precision screen

Completed three conversions directly from pinned BF16, with expert matrices unchanged at affine 4-bit/g64, routers at 8-bit/g64 and source-precision exclusions. A raises attention to 6-bit; B raises embeddings/head to 6-bit; C does both. All passed serialized-parameter, tokenizer, finite-forward and save/reload gates. Frozen authored development tasks, separate from HumanEval+, greedy 8,192-token budget, one sequential CUDA run per model.

| Model | Weight GB | Passed | Gen s | Failed s | Tokens | Trunc / empty | CUDA MLX GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 24.301 | 20/24 | 397.4 | 82.1 | 70,353 | 1 / 1 | 26.08 |
| 4bit | 6.837 | 15/24 | 605.1 | 286.7 | 137,037 | 7 / 7 | 8.66 |
| mxfp4 | 6.457 | 21/24 | 323.2 | 61.9 | 73,537 | 0 / 0 | 7.86 |
| A | 6.985 | 21/24 | 411.5 | 104.2 | 90,058 | 2 / 2 | 8.53 |
| B | 6.950 | 14/24 | 609.9 | 344.6 | 136,333 | 9 / 8 | 8.70 |
| C | 7.099 | 22/24 | 399.7 | 74.8 | 87,062 | 2 / 2 | 8.65 |

All GB are decimal; memory is the CUDA MLX allocator peak, not system or Metal memory. Generation time includes every failed/truncated attempt. Across six models: 2,746.9 s generation, 2,943.9 s including per-run startup, sandbox tests and shutdown.

**Reject all for promotion.** C gained two tasks over MXFP4 and lost one, descriptive exact McNemar p = 1.0. Its one-task net gain comes with 23.7% more generation time, 18.4% more tokens and 9.9% more weight bytes. A tied MXFP4 and was slower/larger; B did not recover the baseline. A/C substantially improved over native 4-bit, which is insufficient to recommend a final product. HumanEval+ and M4 gates were not run for these candidates.

[Commands, exact sizes, integrity scope, paired failure inspection and limitations](NONEXPERT_EXPERIMENT.md); [immutable receipts and compressed raw responses](../results/nonexpert-v1/). One recommended follow-up: keep C as control and reduce only expert down-projection affine 4-bit group size to 32 (estimated 7.331 GB); this was not run.

## Group-32 and bounded DWQ follow-up

The [completed refinement experiment](REFINEMENT_EXPERIMENT.md) retains C as an experimental control. D (only expert down-projection groups reduced to 32) serialized to 7,329,780,326 bytes and passed 20/24. Fresh C, calibrated C and pinned MXFP4 passed 21/24, 21/24 and 22/24, respectively, under the unchanged 8,192-token development protocol. Total generation times were 379.0, 356.4 and 359.5 seconds for the three controls; D took 401.2 seconds. No recipe earns release promotion.

One 32-update DWQ calibration completed after a preserved out-of-memory attempt. Only eligible scales/biases changed; packed weights, routers and BF16 exclusions remained unchanged. Teacher validation loss worsened from 0.103 to 0.158. The separate unopened 12-task mixed fixture yielded 8/12 for C, 9/12 for DWQ and 8/12 for MXFP4; DWQ took 31.1% more generation time than MXFP4. Tiny score differences and greedy CUDA repeat variability limit conclusions. See [receipts](../results/refinement-v1/) for failed attempts, integrity gates, raw responses and paired results.

The matched M4 hardware gate completed for calibrated C versus MXFP4 with Visual Studio Code open and idle. Mean decode rates at 1K/4K/16K were 42.9/41.7/37.6 versus 51.6/48.9/44.2 tokens/s; measured MLX maxima were 7.83 versus 7.19 GB. The first DWQ finite-smoke stage sampled pressure flag 4 twice and swap grew from 823.88 to 4,709.38 M. Later throughput samples were flag 1; whole-sequence swap ended at 4,149.25 M. This is a fixed-order, periodic whole-system observation with other apps retained, not isolated causal attribution. [Mac summary](../results/refinement-v1/mac-summary.json) and the detailed report retain the full limitations.

## Gated 1e-7 diagnostic

The [bounded diagnostic](DWQ_DIAGNOSTIC.md) completed 32 checkpointed updates from pristine C with the unchanged corpus and cached targets. Mean repeated validation losses at steps 0/8/16/24/32 were 0.104334/0.103514/0.104570/0.104949/0.106943. Step eight’s small apparent gain was below its repeat range and overlapped the initial measurements, so step zero won the frozen loss gate. Its saved weight hashes are identical to original C and fresh-process integrity passed. No development generation, reuse of the opened 12-task set, or Mac run followed. The next substantive question is representative reasoning/tool/chat calibration, not another learning-rate decimal on this corpus.

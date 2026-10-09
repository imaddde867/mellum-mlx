# Mixed non-expert precision experiment

Completed 2026-10-09 on the RTX 5090. **No candidate promoted.** Attention restoration helped the native affine 4-bit baseline on this screen, but none demonstrated a credible improvement over the pinned MXFP4 competitor. These are CUDA results, not M4 performance.

Original source: `JetBrains/Mellum2.1-12B-A2.5B-Thinking` at `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`. External comparator: `randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4` at `4ce0d28df07dfa4ed28077e5b718e866be2cf4cf`; its weight SHA-256 is `b6224c8ea3e32190acc1500ce513d39263be257c6c71a10a31e832c045d2cb9a`. The existing 5090 environment and downloads were reused. Unrelated local work and the previous pilot's evidence were preserved; that pilot had finished before these conversions began.

The inspected model has untied embedding and output-head weights: 112 attention projections, two embedding/head modules, 84 stacked expert matrices, 28 routers, and 113 excluded BF16 tensors. The converter checks the complete eligible-module set before quantization. Its tied-embedding policy is unit-tested, although this checkpoint is untied.

- A: attention projections at affine 6-bit/g64; other eligible weights at 4-bit/g64.
- B: token embeddings and output head at affine 6-bit/g64; other eligible weights at 4-bit/g64.
- C: both changes.

All three began from the checksummed original BF16 checkpoint. Expert matrices stayed affine 4-bit/g64, routers stayed 8-bit/g64, and excluded tensors stayed at source precision. Architecture, weight tying, vocabulary and tokenized chat prompts were unchanged. Conversion receipts contain the complete realized precision map and source tensor shapes/dtypes.

Size estimates used actual sanitized tensor shapes, packed integer bytes, BF16 scales and affine biases, excluded tensor bytes, and a 1 MiB header allowance. Estimates were saved before conversion. Every recipe remained below 7,500,000,000 decimal bytes; none was adjusted to fit.

| Recipe | Estimated tensor bytes | Estimate with header allowance | Actual weight bytes |
| --- | ---: | ---: | ---: |
| A | 6,985,230,848 | 6,986,279,424 | 6,985,323,058 |
| B | 6,949,841,408 | 6,950,889,984 | 6,949,933,574 |
| C | 7,098,477,056 | 7,099,525,632 | 7,098,569,272 |

Each candidate passed the existing checksum/index/EOS/router/tokenizer checks (791 serialized tensors), fresh-process loading, equality between serialized and loaded parameters, equality of excluded tensors to the BF16 source, and save/reload parameter checks. Three forwards before and three after reload were finite and identical on the short fixed prompt. The assessor permits reload differences within observed same-artifact repeat variability; it does not require globally deterministic CUDA logits. These short probes do not resolve the previously reported long-context variability.

The [24 authored development tasks](../fixtures/nonexpert-v1/dev.jsonl) and [manifest](../fixtures/nonexpert-v1/manifest.json) were frozen before any sprint coding generation. SHA-256: `1d343046c2b48f70fde42e2f7cf652ebca7ebff5a380e6a4f3d4235bcb5a9279`. The fixture is separate from HumanEval+ and repository-agent tasks. An authored escape was corrected and checks syntax-compiled before generation; no tests changed after answers were seen.

All six models ran sequentially, with no competing conversions or downloads, on MLX 0.32.3 / MLX-LM 0.32.0. Same standard coding instruction, upstream thinking template, greedy seed 0, 8,192 total completion tokens, prefill step 512, and concurrency one. Every tokenized prompt and vocabulary matched across models. No reasoning suppression, early-answer forcing, budget changes or task filtering occurred.

Answers were sanitized with the existing pinned EvalPlus function. Generated code ran only in the pinned offline unprivileged container: read-only root, no mounts, dropped capabilities, no new privileges, 64-process limit, one CPU, 512 MiB memory, temporary filesystem and a 20-second outer timeout. Positive and negative sandbox preflight checks passed. This screen uses functional assertions; it does not comprehensively enforce every complexity or implementation constraint in the prompts.

| Model | Weight GB | Passed | Gen s | Failed s | Tokens | Trunc / empty | CUDA MLX GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 24.301 | 20/24 | 397.4 | 82.1 | 70,353 | 1 / 1 | 26.08 |
| 4bit | 6.837 | 15/24 | 605.1 | 286.7 | 137,037 | 7 / 7 | 8.66 |
| mxfp4 | 6.457 | 21/24 | 323.2 | 61.9 | 73,537 | 0 / 0 | 7.86 |
| A | 6.985 | 21/24 | 411.5 | 104.2 | 90,058 | 2 / 2 | 8.53 |
| B | 6.950 | 14/24 | 609.9 | 344.6 | 136,333 | 9 / 8 | 8.70 |
| C | 7.099 | 22/24 | 399.7 | 74.8 | 87,062 | 2 / 2 | 8.65 |

GB is decimal. Gen s includes every task's generation, including failed and truncated attempts; Failed s is the generation time spent on failed tasks. Completion tokens include reasoning and final output. The first task's compilation cost is included; there is no warmup. CUDA MLX GB is the allocator peak across server loading and the entire run, excluding driver/system memory. Startup, sandbox execution and shutdown are recorded separately in `elapsed_with_startup_and_tests_s`.

Across all six runs: **2,746.9 seconds (45.8 minutes) of generation**, or **2,943.9 seconds (49.1 minutes)** including per-run startup, tests and shutdown. Conversion/integrity setup is outside those totals. All 144 attempts are retained.

A and C recovered six and seven net tasks versus native 4-bit, and reduced total completion tokens by 34.3% and 36.5%. B did not recover quality or shorten completions meaningfully. This supports selective attention restoration as a useful change to the weak native baseline on this fixture; it does not establish why the original baseline produces excessive completions.

Against MXFP4, A gained path normalization (`NE/05`) and rate limiting (`NE/10`) but lost escaped-field parsing (`NE/08`) and expression evaluation (`NE/16`) to budget exhaustion. Both failed JSON pointer (`NE/11`). A and MXFP4 each passed 21 tasks; A took 27.3% more generation time.

C gained rate limiting and JSON pointer, lost escaped-field parsing to truncation, and shared the path-normalization failure. Its rate limiter correctly expires timestamps at `t-window`; MXFP4 uses `<` instead of `<=`. Its JSON pointer splits tokens before decoding escaped slashes; MXFP4 and A decode the whole pointer before splitting. MXFP4's relative-path implementation incorrectly pops a preceding unresolved `..`; A preserves it. C's two failures ended during reasoning with empty final answers. These examples are retained in the raw responses and sanitized samples, with unchanged tests.

C's one-task net gain is insufficient evidence of superiority (descriptive exact paired McNemar p = 1.0), especially with **23.7% more generation time, 18.4% more completion tokens, 9.9% more weight bytes**, and a higher CUDA allocator peak. The [selection receipt](../results/nonexpert-v1/selection.json) rejects all three for promotion. HumanEval+ and M4 gates were therefore not run for these candidates. The accessible target Mac was confirmed as Apple M4, 10 CPU cores, 16 GB; no candidate Mac performance, editor-workload pressure or swap claim is made.

One greedy run on 24 synthetic tasks has limited coverage, possible unknown training overlap, benchmark-selection uncertainty and CUDA repeat variability. Paired p-values are descriptive and not corrected for candidate comparisons. Scores are executable-test counts, not broad coding or repository-agent success rates.

Reproduce conversions in a separate checkout with the pinned BF16 source, native 4-bit receipt/artifact, comparator and existing environment available, no mixed candidate directories, and an empty result directory:

```sh
PYTHON=.venv/bin/python sh scripts/run_nonexpert.sh all work/nonexpert-repeat
.venv/bin/python scripts/summarize_nonexpert.py --results work/nonexpert-repeat
```

To repeat only the screen using the unchanged built artifacts and passed integrity receipts:

```sh
mkdir -p work/nonexpert-repeat
cp results/nonexpert-v1/integrity-{A,B,C}.json work/nonexpert-repeat/
PYTHON=.venv/bin/python sh scripts/run_nonexpert.sh screen work/nonexpert-repeat
.venv/bin/python scripts/summarize_nonexpert.py --results work/nonexpert-repeat
```

Individual entry points are `scripts/convert.py --recipe A --estimate-only`, `scripts/convert.py --recipe A`, `scripts/validate.py MODEL --receipt NEW_FILE`, and `scripts/nonexpert.py integrity|screen --model MODEL --label A --output NEW_FILE`. Replace A with B/C. Existing artifact/evidence paths are refused. Estimation is performed before artifact creation. All three actual artifacts are retained under `artifacts/mellum2.1-mixed-{A,B,C}-g64` on the CUDA host; roundtrip copies are under ignored `work/nonexpert-roundtrip/`.

[Evidence](../results/nonexpert-v1/) includes per-model receipts, task-level paired differences, exact sizes, precision maps, hashes, server logs and lossless raw-response gzip archives. `archive.json` binds compressed bytes and original raw-file hashes; decompress `screen-LABEL.raw.jsonl.gz` to recover the exact file. Uncompressed originals remain on the CUDA host and in ignored local `work/nonexpert-v1-evidence/`. The summary script requires uncompressed raw files beside its screen receipts. `runtime-code.json` labels its conversion-code binding as observed after conversion, not retroactively at conversion time. `run-used.sh` preserves the original runner; the final convenience runner additionally guards against opening existing evidence files. The updated guard was tested before shipping. Validation: 30 tests locally (including four pre-existing untracked pilot tests); 26 tests in the CUDA checkout. No dependency changes, pushes, uploads or publication occurred.

**Exactly one next action:** retain C as the control and test reducing only affine 4-bit expert down-projection group size from 64 to 32 against pinned MXFP4. Actual shapes imply 231,211,008 additional scale/bias bytes and about **7.331 GB including the header allowance**. This targets expert quantization granularity while keeping the size budget; it is a hypothesis, not a diagnosed cause or promised win. It has not been implemented or run.

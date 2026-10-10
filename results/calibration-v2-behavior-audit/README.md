# Behavioral controls and failure-to-finish audit

Diagnostic continuation from `226d09b`. S-128 remains rejected for promotion. No training, tuning, replication, checkpoint selection or quantization change occurred. All historical scores and artifacts remain intact.

The 32 previously opened tasks are now diagnostic/regression material. Neither U-128 nor BF16 is evaluated as a new selection candidate. Five models are paired by the exact frozen task order; the S/C/MXFP4 observations are reused, not regenerated.

## Protocol and integrity

[Protocol](protocol.json) was frozen after the 96 historical transcripts were audited and before control generation. It inherits the original four-response, 8,192-total-completion-token, greedy seed-zero, strict-tool, isolated-sandbox protocol. Runtime flags, pinned packages, original assertions, tokenizer, templates and loopback serving command are unchanged. The two controls are exactly the completed U checkpoint at step 128 and the original pinned BF16 source.

[Compatibility](compatibility.json) checks every historical initial prompt, request sampling/tool settings, bounded budgets, model bindings, task order and process runtime. All 150 historical evidence-manifest files were checked unchanged before execution.

U's scheduled FP32 checkpoint matched its complete rolling restart step-128 masters exactly using incremental host comparisons. Source/data/target bindings, optimizer step and cursor were checked against preserved receipts; all saved files passed SHA-256 checks. Export changed only 138 eligible BF16 affine tensors. A separate fresh process matched all 791 parameter digests. [Export](U-export.json), [fresh verification](U-verify.json) and their process receipts retain exact bindings. This is an integrity check, not a resumed training run, a new fidelity qualification or a selection decision.

BF16's five weight shards, configuration and tokenizer files match the pinned teacher source. It uses the existing `work/source` loading path and identical MLX-LM server command, with one model resident at a time. No packages, runtime flags, cache capacity or allocator limits were changed. Shared `require_runtime` receipts include training-policy annotations for allocator/checkpointing; the behavioral server itself uses the same CLI defaults as the historical servers. No training checkpoint wrapper runs in the inference server.

## Audit definitions

[Contract audit](contract-audit.md) explains final extraction, response termination, virtual workspace semantics and demonstrated interface limitations. [Historical paired table](historical-replay.csv) and [full replay](historical-replay.json) were produced before the controls.

Sandbox replay checks model-supplied intermediate tests separately from the task's original assertions, the last workspace and final source. It preserves original task correctness. The historical replay found zero message-state discrepancies, zero differing intermediate tool results and zero differing final scores across 96 attempts.

Categories overlap. Exact repeated calls are identical function/arguments on unchanged workspace; reads after a passing tool test are reported separately and may be reasonable verification actions. Under four responses, read/edit/test/read cannot deliver a subsequent final. This is budget pressure, not a demonstrated off-by-one defect. An edited or repaired workspace can be wrong despite a passing model-supplied test. A passing workspace is never substituted for a missing or incorrect final.

Parser warnings are linked to the active task/response using the sequential server log. Schema validity counts parsed emitted calls; upstream rejected JSON is not included because its raw text was not preserved. `final_present` retains the historical truthiness definition; `substantive_final` additionally excludes whitespace. Neither changes executable scoring.

## Representative historical failures

* **S, convolution (task 4):** read → edit → passing original assertion → read. Before the last read: “we should probably provide the final source code … to be safe, we can read the file again.” The workspace passes, but response four is a tool call and the final is missing.
* **S, stack machine (task 1):** edit → failed test → repair → passing test. The repaired workspace passes the original assertion, but no response remains for a final. This is a successful repair followed by failure to finish, not an incorrect final implementation.
* **S, heap merge (task 18):** edit → failed test → read → token limit. The last workspace also fails the original assertion. This is an overlapping incorrect-code and token-exhaustion case.
* **C, shell variables (task 2):** the workspace and intermediate original assertion pass. The final adds another backslash to the regular expression and fails the original assertion. This demonstrates a final-copy regression, not state-replay loss.
* **MXFP4, convolution (task 4):** `from task import convolve_valid` receives `ModuleNotFoundError`; it reads the virtual file and retries with inline code. The last workspace passes, but response four is a tool test and the final is missing.
* **MXFP4, shell variables / cron minutes (tasks 2 / 20):** server warnings record invalid JSON escapes; no parsed calls remain, although finish reason is `tool_calls`. Whitespace is treated as final content and fails the original oracle. Raw rejected tool text is unavailable, so malformed-generation causes cannot be reconstructed.

[Exact excerpts and transcript references](historical-excerpts.json) preserve the quoted endings and call/results. [Parser warnings](historical-parser-warnings.json) retain log line references. The import and empty-tool termination have regression checks; the original harness remains unchanged for comparability of both controls.

## Completed controls and five-model results

Both controls completed one uninterrupted process and all 32 task attempts. U-128 process time was 650.161 seconds; BF16 was 890.299 seconds. The published and historical artifacts were not regenerated. All five weight/configuration bindings remained unchanged afterward: [final hash checks](final-model-bindings.json).

[Paired task table](paired-results.csv) has one row per task and all five models' correctness, literal/substantive final presence, response/call counts, stop reason, tokens, generation time, last-workspace oracle and overlapping categories. [Machine-readable report](paired-results.json) includes every pair's gained/lost task IDs, complete action-pattern counts, parsed-call validity, failed-attempt cost and context-shape audit. [Control sandbox replay](control-replay.json) preserves the intermediate tests and original-oracle results. Raw outputs, all requests/responses, tool results, finals and server logs are under `behavior-U-128/` and `behavior-BF16/`.

| Model | Correct / 32 | Literal finals | Nonblank finals | Missing literal final | Token limit | Response limit | Parsed valid calls | Responses | Tokens | Generation seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-128 | 17 | 18 | 18 | 14 | 0 | 14 | 110/110 | 128 | 121879 | 606.412 |
| S-128 | 16 | 16 | 16 | 16 | 1 | 15 | 110/110 | 127 | 124525 | 610.294 |
| Pristine C | 19 | 20 | 20 | 12 | 0 | 12 | 107/107 | 127 | 104822 | 519.516 |
| MXFP4 | 23 | 25 | 23 | 7 | 3 | 4 | 79/79 | 107 | 132674 | 617.242 |
| BF16 | 25 | 27 | 27 | 5 | 4 | 1 | 72/72 | 103 | 143191 | 823.498 |

Every cost includes failed attempts. There were no unknown token counts or terminal transport/schema task errors. MXFP4 has three server parser warnings, including two whitespace terminal responses historically counted as literal finals. Validity above concerns parsed emitted calls, not unobserved rejected JSON. The historical validity count and correctness are preserved.

Token/response limits above retain the original primary termination definition. The number of tasks reaching response four without a final, including token-limit overlap, is U 14, S 16, C 12, MXFP4 4 and BF16 2. BF16 heap merge reaches both limits; S heap merge does too. Overlap flags and actual response finish reasons are retained per task.

| Model | Last workspace passes original assertions / 32 | Passing workspace but missing final | Edited workspace still wrong | Repair still wrong | Wrong final copied from passing workspace | Exact repeated calls | Tasks with read after passing tool test | Failed tool-test calls | Failed-task tokens | Failed-task generation seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-128 | 31 | 13 | 1 | 1 | 1 | 0 | 26 | 3 | 61207 | 300.181 |
| S-128 | 31 | 15 | 1 | 0 | 0 | 1 | 20 | 7 | 69397 | 339.871 |
| Pristine C | 32 | 12 | 0 | 0 | 1 | 2 | 26 | 4 | 40367 | 201.165 |
| MXFP4 | 28 | 5 | 1 | 0 | 0 | 0 | 6 | 6 | 43432 | 200.088 |
| BF16 | 32 | 5 | 0 | 0 | 2 | 1 | 1 | 2 | 43257 | 251.501 |

A workspace pass is only the bounded original assertion oracle, not comprehensive semantic correctness. The MXFP4 remaining wrong workspaces include never-edited stubs after parser drops and a token limit. Exact-repeat counts concern repeated identical calls on unchanged state; read-after-passing-test counts include successful tasks too. Failure categories overlap, and a changed model-supplied test can pass without certifying the original task. All intermediate and final replay results matched their preserved receipts, with no message-state replay discrepancy across 160 attempts.

Paired correctness, gains/losses for the first named model:

| Pair | Gains | Losses |
|---|---:|---:|
| U vs S | 6 | 5 |
| U vs C | 7 | 9 |
| U vs MXFP4 | 5 | 11 |
| U vs BF16 | 2 | 10 |
| S vs C | 6 | 9 |
| S vs MXFP4 | 3 | 10 |
| S vs BF16 | 1 | 10 |
| C vs MXFP4 | 4 | 8 |
| C vs BF16 | 1 | 7 |
| MXFP4 vs BF16 | 3 | 5 |

These are descriptive paired results from one attempt per model, not causal comparisons or confidence intervals. U's one-task advantage over S does not support objective superiority or promotion. Both calibrated models remain below pristine C on this diagnostic screen.

## Additional control examples

* **U, heap merge (task 18):** read → edit → failed test → repair, with no final. Unlike its other 13 missing-final cases, the last repaired workspace still fails the original assertion. The raw transcript and replay distinguish an unsuccessful repair from simply lacking another response.
* **U, placeholder scan (task 7):** read → edit → passing test → read. The ending says “we can also read the file again using read_file to get the current content. Let's do that to be sure.” The fourth response reads instead of finishing; the last workspace passes the original assertion.
* **BF16, UTF-8 prefix (task 23):** edit → passing test → length limit. The workspace passes the original assertion, but no final source arrives within 8,192 total tokens.
* **BF16, INI section (task 25):** edit → failed model-supplied test → read → another read. The model-supplied check contains literal newlines inside a single-quoted string and raises `SyntaxError: unterminated string literal`; the workspace itself passes the original properly escaped assertion. The repeated read consumes response four. This is neither proof of incorrect workspace code nor a scoring rescue.
* **BF16, JSON pointer (task 24):** the tested workspace passes the original assertion; the final answer damages the `token.replace` escape-decoding expression and fails it. Final-output copying is an observable failure even in BF16.

[Control excerpts](control-excerpts.json) preserve exact response endings and task references.

## Fidelity versus completion

The observed gap is primarily failure to deliver: S has 31 passing workspaces but only 16 correct final answers, and U also has 31 passing workspaces but 17 correct finals. U uses read/edit/test/read on 12 tasks and S on 10; none of those sequences leaves a response for the required final. S additionally spends complete response budgets on repair/test sequences. BF16 uses edit/test/final on 22 tasks and reads after a passing tool test on only one task, although it still has five missing finals and two final-copy regressions.

Improved teacher-forced KL on fixed authored validation contexts does not directly measure the probability of ending a live conversation with a correct final. The structural interface audit found raw read replies and `written`/checksum edit acknowledgements in every relevant frozen training/validation trajectory, while the live harness returns wrapped read replies and `ok` acknowledgements. Test replies share `passed`/`sanitized` keys. This demonstrates different contexts, not a proven cause of the regressions. No material was changed or expanded.

The numerical training runtime remains historically unqualified. Single optimization trajectories and one greedy behavioral attempt per model cannot establish objective superiority, overfitting, defective gradients or a particular loss mechanism. C and BF16 also lose tasks through finishing/copying, while MXFP4 has separately demonstrated parser/import failures. No single category explains every failure.

## Validation and reproduction

The pinned suite passed 94 tests, including the real sandbox import regression; the bundled local Python suite passed 98 tests with 14 environment-dependent skips. A separate default-local-Python attempt lacked NumPy for three existing localization tests; its failed log is retained. No packages were installed or upgraded.

Control commands and exact environment are in the process receipts. `scripts/calibration_v2_behavior_controls.py export`, then `verify`, then `behavior --label U-128` and `behavior --label BF16` run under the frozen protocol. Existing artifact/output paths refuse overwrite; reproduction requires distinct predeclared paths/protocol, not retrying into this evidence. The audit takes `--receipt LABEL=PATH` and optional `--replay`; only isolated sandbox execution is used. The paired report combines historical and control replay files without regenerating model outputs.

No training, resumed run, tuning, replication, new selection, recipe change, response rescue, package/runtime change, uploads, M4, HumanEval+ or push occurred.

**Decision: the next substantive work belongs in the interaction harness.** The virtual-module import mismatch, discarded parser-error termination and inconsistent trajectory/live reply contracts are demonstrated engineering issues to resolve before changing a calibration objective. This does not establish that fixing them would recover S's response-limit failures; S remains rejected. The specific missing observation for attributing those failures to objective versus material is the teacher/student next-action behavior on the actual passing-test contexts immediately before a redundant read versus a final, rather than aggregate KL on authored contexts. That observation was not collected in this task; no further experiment is proposed or run here.

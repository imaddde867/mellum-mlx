# Calibration-v2 pilot and untrained baselines

**The corpus exposes a reproducible assistant-region fidelity gap worth a bounded calibration attempt.** Pristine C has lower aggregate KL than pinned MXFP4, but higher KL in reasoning, reasoning-to-answer transitions, final answers and tool calls. Each FP32 assistant-region repeat interval is disjoint from MXFP4's, and each mean difference exceeds both observed repeat ranges. This is teacher-forced distribution fidelity on a Python pilot, not evidence that calibration will improve executable correctness or produce a better quant.

The protocol was frozen before model measurements in [predeclared.json](../results/calibration-v2/predeclared.json), version `calibration-v2-eval-1.0`. Parent local commit: `34932d3`. The tiny-stdlib learning-rate experiment, its historical scorer, results and gate were not changed. No optimizer updates, quantization candidate, HumanEval+ execution, M4 rerun, push or upload occurred.

## Material and acceptance

[Corpus sources and provenance](../fixtures/calibration-v2/README.md), [frozen family assignments](../fixtures/calibration-v2/splits.json), [manifest](../fixtures/calibration-v2/prepared/manifest.json), [coverage](../fixtures/calibration-v2/prepared/coverage.json).

| Split | Underlying families | Material | Source tokens | Min / median / p90 / max tokens | Sequences >256 |
| --- | ---: | ---: | ---: | --- | ---: |
| Training | 32 | 128 complete trajectories | 67,921 | 113 / 476 / 1,036 / 1,319 | 72 |
| Validation | 8 | 32 complete trajectories | 19,567 | 133 / 513 / 1,202 / 1,422 | 22 |
| Reserved evaluation | 32 | 32 task specs without responses | Not tokenized | Not measured | Not measured |

Each train/validation family has coding, bug-repair, edit-tool and read/edit/test presentations, producing 32 of each training category and eight of each validation category. Families were assigned before constructing variants or model outputs. Related variants stay together. These are correlated presentations of 40 original Python problems, not 160 independent problems or a sufficient coverage claim. Reserved task specs were not executed, tokenized, answered or teacher-scored.

The material covers parsing, escaping and serialization, data transformations, stateful operations, boundary validation, numeric/calendar logic, dependency handling and local code repair. All sources and reasoning were originally authored under the repository's MIT license. No HumanEval+, existing development/heldout questions or examples chosen for a quant's failures were included.

Acceptance required exact counts, disjoint task/source families, unchanged pinned tokenizer/template, complete final answers and reasoning boundaries, strict tool schemas and raw executable source matching the sandbox sanitizer modulo exterior whitespace. All 40 correct implementations passed their assertions and all 40 deliberately buggy controls failed in the existing pinned offline, unprivileged EvalPlus Docker sandbox. This uses its sanitizer and Python execution only; it does not load benchmark questions. Each family receipt binds the identical answer/edit code in all four variants. Read/edit tools are authored workspace simulations; test responses contain actual sandbox outcomes. Quotes, escaped newlines and complete code strings remain in the source messages and upstream JSON serialization.

The first preparation attempt failed: the installed Transformers tokenizer returned a structured result where a list was expected; one JSON-log expected string was incorrect; one incomplete-frame mutation escaped its assertions. All [rejections and source trajectories](../results/calibration-v2/rejected-preparation/rejections.json), [original catalog](../results/calibration-v2/families-attempt-1.json) and [exact executed code](../results/calibration-v2/preparation-attempt-1.py) are retained. A new preparation run corrected the return-format check and the two authored assertions before any model measurements. No failed or truncated teacher continuation was rewritten: teacher generation was not used at all. [Accepted rejection receipt](../fixtures/calibration-v2/prepared/rejections.json).

Full messages, reasoning, tool-call IDs, tool results, final answers, rendered template text, token IDs and target-region labels are retained in the prepared JSONL files. Reasoning uses upstream `reasoning_content`; the template is unchanged. No sequence was truncated, padded to 256, or converted into a training window. Training targets were not captured; this task establishes validation baselines before calibration.

## Evaluation protocol and scoring effects

BF16 source revision `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`; pinned MXFP4 revision `4ce0d28df07dfa4ed28077e5b718e866be2cf4cf`. C was checked against its pristine recipe, complete artifact hashes and original integrity weight hashes; the comparator was checked against its pinned file hashes. Receipts bind source weights, tokenizer, template, corpus, exact executed script, installed loss implementation and packages.

The BF16 source teacher-forces each **complete validation trajectory** once and captures its top 1,024 logits and indices. C and MXFP4 use exactly those teacher contexts and cached targets. Batch one, seed 123, evaluation mode, temperature two, equal weighting of every prediction target, and no temperature-squared multiplier. This preserves the existing mathematical top-1,024 renormalized KL objective; it is not full-vocabulary KL. The new contexts, sequence processing and evaluation mode differ from the historical short diagnostic, so absolute losses are not historical gains or a revised gate.

Model forwards process 128-token chunks with a fresh KV cache per trajectory, preserving that cache across chunks. Chunk size bounds temporary inference allocations; it does not discard history. Each complete sequence is reduced once, avoiding chunk-wise BF16 accumulation. The historical comparison path uses installed `mlx_lm.tuner.losses.kl_div_loss` in the original BF16 logits dtype, BF16 sequence sums and host token-weighted aggregation. The evaluation-only comparison casts the **same captured logits** to FP32 before temperature scaling, KL and sequence reduction. Loss weighting does not change.

Prediction-target regions partition all 19,535 validation targets: context 11,073; reasoning 1,650; transition 576; final 3,336; tool call 2,900. Transitions include the closing think marker plus four neighboring tokens on either side, overriding other region labels. Tool regions include their wrappers and serialized JSON. Initial framing, user/tool messages and remaining template tokens are context. Region sums can differ from the aggregate's native BF16 sum because separate reductions round differently.

Both scorers consumed identical captured arrays on the first forward before four further independent-cache forwards. [Serialized-logit replay](../results/calibration-v2/scoring-replay.json) later reproduced all 32 first-forward sequence/region sums for both models and both scorers **exactly**, with zero model forwards. This isolates scoring effects from subsequent forward variability. BF16 teacher-forward variability was not measured; teacher targets remain fixed across both student runs.

Same-capture comparison, FP32 mean minus original-scorer mean on the first forward:

| Region | C | MXFP4 |
| --- | ---: | ---: |
| Aggregate | -0.000363077 | +0.000097616 |
| Reasoning | -0.001830219 | -0.001132550 |
| Transition | +0.000241966 | -0.001343879 |
| Final | -0.000552608 | -0.001028573 |
| Tool call | +0.001275588 | +0.000112104 |

These are numerical scoring differences, not model improvements. In particular, the original scorer's C tool-region mean difference is smaller than its observed repeat range; FP32 yields the separate comparison below. The historical gate was neither loosened nor rerun.

## Baselines and variability

[Summary and all repeat values](../results/calibration-v2/summary.json), [C measurements](../results/calibration-v2/C-receipt.json), [MXFP4 measurements](../results/calibration-v2/mxfp4-receipt.json), [BF16 target bindings](../results/calibration-v2/teacher-receipt.json).

FP32 KL mean and five-forward minimum–maximum:

| Region | Pristine C | Pinned MXFP4 | C minus MXFP4 mean |
| --- | --- | --- | ---: |
| Aggregate | 0.112965 (0.112677–0.113178) | 0.132841 (0.132840–0.132843) | -0.019876 |
| Context | 0.129497 (0.129036–0.129681) | 0.170380 (0.170380–0.170380) | -0.040883 |
| Reasoning | 0.150164 (0.150065–0.150279) | 0.147072 (0.147068–0.147086) | +0.003092 |
| Transition | 0.146043 (0.145342–0.147187) | 0.131321 (0.131319–0.131330) | +0.014722 |
| Final | 0.078768 (0.078298–0.079753) | 0.065557 (0.065556–0.065560) | +0.013211 |
| Tool call | 0.061446 (0.060536–0.062444) | 0.059110 (0.059110–0.059110) | +0.002336 |

The original scorer's five-forward aggregate mean is C **0.113375** (0.112996–0.113793), MXFP4 **0.132735** (0.132717–0.132743). All original-scorer region/repeat values remain in the receipts and summary. No single best forward was selected.

Context accounts for 56.7% of prediction targets and explains why C wins the aggregate while trailing in every assistant region. Both regions and the unchanged aggregate must be considered when interpreting a future calibration result. Five-repeat intervals are empirical ranges, not confidence intervals; related variants are not independent observations.

As a coverage diagnostic, FP32 assistant-only token-weighted means by presentation are coding C 0.112563 / MXFP4 0.108867; bug repair 0.126304 / 0.102167; edit-tool 0.085217 / 0.081419; read/edit/test 0.077801 / 0.071604. These exclude context only for reporting; they are not a different training objective. Underlying-family and per-trajectory results remain in the raw receipts.

Predeclared decision: on at least one assistant region, C's FP32 KL mean must be at least 0.01 and exceed its observed five-repeat range. All four assistant regions satisfy it. C also reproducibly trails MXFP4 in all four under the separately predeclared disjoint-interval/mean-gap rule. This supports attempting to close a relevant fidelity gap later; it neither selects a model nor demonstrates recoverable capability.

## Memory, validation and reproduction

RTX 5090, existing MLX 0.32.3 / MLX-LM 0.32.0 environment. BF16 target capture peaked at **26.24 GB**, C at **8.06 GB**, MXFP4 at **7.39 GB** of MLX allocator memory including model load, excluding driver/system memory. Measured stages took 11.69 / 61.33 / 54.98 seconds respectively, excluding preflight weight hashing. These are full-context **inference** observations. Calibration gradients/checkpointing and training-window construction remain untested. Longer calibration must receive its own bounded memory qualification before optimizer work. The [GPU status utility](../results/calibration-v2/nvml-status.txt) reported a driver/library mismatch; actual CUDA model computations completed. No driver or system settings were changed.

The initial full-suite CUDA run passed 44 tests; that checkout lacks four unrelated local optimization tests. Final validation passed [46 CUDA tests](../results/calibration-v2/cuda-final-tests.txt) and [48 local tests](../results/calibration-v2/local-tests-final.txt), with two pinned-CUDA integration tests skipped locally. Together these cover all 50 tests, including four unrelated local optimization tests. Syntax checks and [all 96 retained capture-file/source bindings](../results/calibration-v2/final-bindings.json) passed. Exact executed preparation/scoring snapshots are retained in the result directory. Reproduction refuses existing outputs and requires the existing pinned source, pristine C, MXFP4, Docker image and CUDA environment:

```sh
.venv/bin/python scripts/calibration_v2.py prepare --output fixtures/calibration-v2/prepared
.venv/bin/python scripts/calibration_v2.py teacher --output work/calibration-v2-teacher
.venv/bin/python scripts/calibration_v2.py baseline --label C \
  --model artifacts/mellum2.1-mixed-C-g64 --output work/calibration-v2-C
.venv/bin/python scripts/calibration_v2.py baseline --label mxfp4 \
  --model work/mxfp4 --output work/calibration-v2-mxfp4
```

Use a fresh checkout/destination or pass a fresh preparation path through `--data` in subsequent stages; the committed prepared files already exist. Rendering/execution inputs are reproducible, but captured CUDA logits can vary: hashes bind these measurements rather than promise byte-identical future logits. Large captured arrays remain at the recorded `work/calibration-v2-*` paths on the CUDA host, with all file hashes in the committed receipts. No model weights or cache arrays are committed or uploaded.

To regenerate the summary from committed baseline receipts into a new path:

```sh
python3 scripts/calibration_v2.py summary --output work/calibration-v2-summary-repeat.json
```

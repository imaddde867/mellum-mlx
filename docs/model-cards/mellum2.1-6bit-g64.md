---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: JetBrains/Mellum2.1-12B-A2.5B-Thinking
base_model_relation: quantized
language:
- en
tags:
- mlx
- quantized
- coding
- mellum
---

# Mellum2.1 Thinking — MLX 6-bit, group 64

**Release candidate; native M4 validation is in progress. Weights are not yet published.**

A reproducible native MLX affine conversion of JetBrains' Mellum2.1 Thinking coding model. JetBrains created and trained the model; this project contributes conversion, independent comparison and validation evidence. No fine-tuning or calibration was applied.

- Source: `JetBrains/Mellum2.1-12B-A2.5B-Thinking` at `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`.
- Eligible weights: 6-bit affine quantization, group size 64. Mellum routers retain the upstream native 8-bit policy; remaining weights are BF16.
- Serialized weight size: 9,873,101,418 bytes (9.87 decimal GB). Runtime memory is larger and depends on context and workload.
- Tested software: MLX 0.32.3, MLX-LM 0.32.0, Python 3.12.
- License: Apache-2.0 for model weights. Original evaluation scripts are MIT; this does not relicense upstream material.

## Independent coding comparison

Same RTX 5090, 164 HumanEval+ tasks, one greedy sample per task, upstream chat template, 8,192 completion tokens, pinned EvalPlus container with expanded tests and every failure retained.

| Candidate | HumanEval+ passed | Score | Weight GB |
| --- | ---: | ---: | ---: |
| MLX BF16 reference | 153/164 | 93.3% | about 24.3 |
| This native 6-bit | 146/164 | 89.0% | 9.87 |
| Existing randmaru MXFP4 | 144/164 | 87.8% | 6.46 |
| Native 4-bit | 133/164 | 81.1% | 6.84 |

The two-task lead over MXFP4 does not demonstrate superiority: nine tasks passed only 6-bit and seven passed only MXFP4 (exact paired McNemar p = 0.8036). MXFP4 offers a stronger demonstrated quality/size balance. These are single-run regression results; benchmark training overlap and CUDA numerical variability are possible. They do not establish general coding-agent reliability or reproduce JetBrains' evaluation protocol.

The original comparison runs lack generation-time sample and weight checksum binding. Their original receipts, raw responses, task outcomes and explicit post-run archive checksums are preserved. New generation/evaluation scripts enforce stronger bindings.

[Full evidence and limitations](https://github.com/imaddde867/mellum-mlx/blob/main/docs/RESULTS.md) · [Fixed coding protocol](https://github.com/imaddde867/mellum-mlx/blob/main/docs/CODING_PROTOCOL.md) · [Conversion receipt](https://github.com/imaddde867/mellum-mlx/blob/main/results/provenance/mellum2.1-affine-6bit-g64/conversion.json) · [Validation receipt](https://github.com/imaddde867/mellum-mlx/blob/main/results/provenance/6bit-validation.json)

## Apple Silicon status

Native 6-bit performance and system memory observations on the 16 GB M4 remain pending. CUDA throughput must not be used as an Apple performance estimate. Conversion itself was performed on the RTX 5090, and has not been validated on a 16 GB Mac. Download the release artifact rather than attempting BF16 conversion on that machine.

## Local use

Install the tested runtime in a fresh environment. Once this candidate is published, download its complete artifact and use the local folder with MLX-LM:

```sh
python -m pip install 'mlx==0.32.3' 'mlx-lm==0.32.0'
python -m mlx_lm generate --model ./mellum2.1-6bit-g64 --prompt 'Write a Python binary search function.' --max-tokens 8192 --temp 0
```

Keep the upstream tokenizer and thinking template. Reasoning and the final answer share the output budget, so small budgets can leave an empty final answer. Validate generated code and review tool operations before using them.

## Known limits

Independent long-context architecture parity remains unresolved. Short numerical probes and synthetic throughput do not establish general quality. A contained CUDA retrieval-validation task passed all 36 fixed repository/regression tests after a model-generated patch. A separate mutation-routing task failed its bounded trials; both outcomes and adapter changes are preserved in the evidence repository. This is not a general agent-success rate or a Metal agent result. No full SWE-bench result, broad agent success rate, novel quantization algorithm or best-model claim is made.

[JetBrains original model](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) · [MLX-LM](https://github.com/ml-explore/mlx-lm) · [MXFP4 comparator by randmaru](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4)

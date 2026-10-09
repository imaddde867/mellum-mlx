# Measurement notes

Initial measurements: 2026-10-09. RTX 5090, 32 GB VRAM, driver 595.91.07. MLX 0.32.3 / MLX-LM 0.32.0. These are CUDA research-preview results.

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

The independent reference uses PyTorch 2.14.1+cpu, Transformers 5.19.0, BF16, eager attention and eight CPU threads. CPU versus CUDA arithmetic differs. This probe does not establish complete architecture parity; its observed disagreement is reported, not declared resolved. Sliding-window boundary and long-context comparisons remain pending.

## Protocol fixtures

The installed parser/serializer preserved single/multiple calls, escaped JSON arguments, reasoning and streaming indices in four fixed cases. Malformed truncated JSON was rejected. This does not reproduce the entire HTTP handler or prove model-generated agent behaviour. See `results/cuda/protocol-fixtures.json` for the exact server source hash.

## Pending

Long-context numerical audit, executable coding benchmark, broader live HTTP scenarios, resolution of the intermittent DWQ discrepancy, meaningful held-out DWQ calibration and Apple Silicon validation. No superiority or Mac performance claims are supported at this stage.

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

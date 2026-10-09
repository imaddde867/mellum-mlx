# Research and release plan

Research date: 2026-10-09. Goal: a useful, independently reproducible Mellum2.1 MLX release with credible engineering evidence. Popularity and download volume cannot be guaranteed.

## Positioning

Publish measured quantization choices and reliable local coding-agent use. The upstream model is intended for repository work and tool calling; its model card specifies Qwen3 reasoning parsing and Hermes tool parsing. [JetBrains model card](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)

An existing MXFP4 MLX conversion rules out a first-conversion claim. Compare it fairly, pin its revision, and publish its results even if it wins. A smaller artifact with better tested compatibility can be useful without a new quantization algorithm.

## Candidate matrix

| Candidate | Role | Policy |
| --- | --- | --- |
| Original BF16 | Reference | Same pinned source and tokenizer |
| Native affine 4-bit/g64 | Main size-efficient baseline | Native 8-bit routers |
| Native affine 6-bit/g64 | Fidelity reference | Native 8-bit routers |
| Existing MXFP4 | External comparator | Preserve publisher's artifact |
| Cached-target DWQ 4-bit | Conditional experiment | Existing 4-bit student; retain protected routers |

Native Mellum architecture includes hybrid sliding/full attention and model-specific router precision. [Pinned MLX implementation](https://github.com/ml-explore/mlx-lm/blob/9d8abd94d63a9b3c72e7e9b146e43af1005368fa/mlx_lm/models/mellum.py)

## Sequence and release gates

1. **Artifact integrity:** pin upstream weights, packages and source hashes; validate every shard/index entry, router configuration, tokenizer vocabulary, EOS and plain/tool chat templates. Verify save/reload behaviour.
2. **Numerical audit:** independent Transformers versus MLX BF16 reference, then quantized versus MLX BF16. Investigate discrepancies. Compare cached/uncached and chunked/unchunked execution at the 1,024-token sliding-window boundary and longer contexts. Short probe results cannot satisfy this full gate.
3. **Representative quality:** run fixed HumanEval+ task IDs with expanded tests in an isolated execution environment. Use identical prompts, templates, sampling, output-token budgets and task IDs for all candidates. Report failures and truncation, paired differences and uncertainty. Treat this as regression evidence because training overlap is possible. Keep calibration separate. [EvalPlus](https://github.com/evalplus/evalplus)
4. **Protocol reliability:** plain answers; one and multiple calls; escaped arguments; reasoning followed by a call; a tool-result round trip; truncated output; streaming and non-streaming. Distinguish fixed parser/serializer fixtures from model-generated live HTTP cases. Validate reconstructed JSON, names, IDs and finish reasons. Issue #1946 warrants reproduction rather than a claim that Mellum is affected. [Upstream report](https://github.com/ml-explore/mlx-lm/issues/1946)
5. **Hardware measurements:** warmup plus repeated trials at 512/1K, 4K and 16K prompt tokens; fixed output budget; separate cold start, prompt rate, decode rate, TTFT, MLX peak memory and whole-system headroom/swap. Record hardware, OS, versions and power conditions. Test a real Mac before Apple performance/compatibility claims. Do not infer Metal performance from CUDA.
6. **Publish:** weights plus revision/checksum receipt, exact recipe, raw results and one comparison chart. Publish variants with a clear demonstrated tradeoff. A CUDA-only preview must say experimental and retain all pending gates.

## Advanced-method decision

These are existing MLX methods, not inventions of this project. Their suitability must be demonstrated on Mellum/CUDA.

| Method | Audit finding | Decision |
| --- | --- | --- |
| AWQ | Current model-config registry lacks Mellum | Defer architecture-specific support |
| Dynamic sensitivity | Dequantized teacher/student and full gradient accumulators; custom policy may replace router protection | Defer on 32 GB VRAM |
| GPTQ | Supports `SwitchLinear`; expert-conditioned calibration and router treatment still need audit | Conditional later pilot |
| DWQ | Cached teacher targets, existing quantized student, scale/bias optimization | First advanced pilot |

Sources: [AWQ](https://github.com/ml-explore/mlx-lm/blob/9d8abd94d63a9b3c72e7e9b146e43af1005368fa/mlx_lm/quant/awq.py), [dynamic quantization](https://github.com/ml-explore/mlx-lm/blob/9d8abd94d63a9b3c72e7e9b146e43af1005368fa/mlx_lm/quant/dynamic_quant.py), [GPTQ](https://github.com/ml-explore/mlx-lm/blob/9d8abd94d63a9b3c72e7e9b146e43af1005368fa/mlx_lm/quant/gptq.py), [DWQ](https://github.com/ml-explore/mlx-lm/blob/9d8abd94d63a9b3c72e7e9b146e43af1005368fa/mlx_lm/quant/dwq.py).

DWQ pilot: batch one, short sequences, one forward/backward update, finite gradients, unchanged router policy, save/reload. Then a modest calibration study with separate held-out coding/tool data. Generate BF16 targets in a separate process to avoid simultaneous teacher/student memory. Upstream caches top 1,024 logits; this is truncated-target distillation. Bind caches to dataset revision, IDs/order, seed, tokenizer, sequence length and batch size. The upstream existence checks alone do not protect against stale/mismatched targets. Publish DWQ only if independent held-out results justify it.

## Visibility and research value

Lead with one defensible finding, link the raw comparison, and show a short real repository task with its test result. Package documentation so another person can reproduce it. An upstream regression test/fix, independently reproduced result, and useful adoption are stronger evidence than an unsupported best-model claim.

Prepare a technical post and relevant MLX/JetBrains community submissions after results are ready. Sending outreach is a separate user-authorized action. Full SWE-bench is later work: disk and sandbox requirements are substantial and ARM support is experimental. [SWE-bench](https://github.com/SWE-bench/SWE-bench)

## Outstanding access

Mac hardware/access has not been supplied. Hugging Face publication requires a confirmed destination and write-capable authentication; the connected account currently offers read access. These are publication/Apple validation dependencies, not reasons to stop local CUDA research.

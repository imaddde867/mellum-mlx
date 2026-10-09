# Fixed coding comparison protocol

Defined before the full generation runs on 2026-10-09.

- All 164 HumanEval+ task IDs, HumanEval/0 through HumanEval/163, in numeric order; no task filtering.
- Cached dataset from the pinned official EvalPlus image; exported task-file SHA-256: `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`.
- Image: `ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740`, installed EvalPlus `0.4.0.dev2`.
- Same standard EvalPlus chat instruction, upstream model template, greedy sampling, seed 0, batch/concurrency one and 8,192 total completion-token budget for every candidate. The extended budget accommodates thinking; it differs from this EvalPlus image's 768-token provider default. Token counts, reasoning, finish reasons and raw responses are retained.
- Native 4-bit/g64, native 6-bit/g64, pinned external MXFP4, and BF16 source, all on the same RTX 5090 with MLX 0.32.3 / MLX-LM 0.32.0. Separate process/server per candidate; prefill step 512. Existing provenance receipts identify the weights.
- Sanitize with the pinned EvalPlus function and the correct task entry point. Run base and expanded tests, preserving every sample including empty/truncated/incorrect outputs.
- Execute only in a container with network disabled, read-only root, all capabilities dropped, no new privileges, invoking user's UID/GID, 256-process limit, two CPUs/workers, 4 GiB container memory, and a temporary directory. Mount only the selected task/sample files read-only and the dedicated output directory writable. No personal directories or Docker socket are mounted.
- Report individual outcomes, truncation, base and expanded pass@1, paired candidate differences and uncertainty. The benchmark may overlap training data; this is regression evidence, not proof of general coding-agent usefulness.
- CUDA numerical nondeterminism remains under investigation. Fixed seed and greedy sampling do not establish bitwise reproducibility.

The one-task smoke test validates the pipeline only and must never be reported as the full benchmark result. All four initial runs now have 164 outputs and completed test outcomes; see RESULTS.md and results/coding/humaneval-plus/.

Sources: [EvalPlus workflow](https://github.com/evalplus/evalplus), [commands and schemas](https://github.com/evalplus/evalplus/blob/master/docs/cli.md), [execution limits](https://github.com/evalplus/evalplus/blob/master/docs/execution.md).

## Reproduce

Docker must already be available to the invoking user. The image contains the public dataset; export it without executing any generated code:

```sh
mkdir -p work/evalplus
docker pull ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740
docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 64 --memory 2g --cpus 2 \
  ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740 \
  python3 -c 'import json; from evalplus.data import get_human_eval_plus; data=get_human_eval_plus(); [print(json.dumps(v)) for k,v in sorted(data.items(),key=lambda item:int(item[0].split("/")[1]))]' \
  > work/evalplus/humaneval-plus.jsonl

python scripts/coding_probe.py artifacts/mellum2.1-affine-4bit-g64 \
  --output work/evalplus/humaneval-4bit.jsonl
python scripts/evaluate_coding.py work/evalplus/humaneval-4bit.jsonl \
  --output work/evalplus/evaluation-4bit
```

Repeat with new output names for the other candidates. Generation starts a temporary loopback-only MLX server and stops it in a finalizer; it never executes generated code. Evaluation refuses missing/duplicate task outputs, mismatched generation data, and paths outside the checkout. The sanitizer function is called directly because this image's sanitizer CLI eagerly loads the unrelated MBPP dataset.

## Integrity enforcement

Full generation and evaluation enforce both the exact ordered 164-task set and the pinned dataset SHA-256. Smaller pipeline checks require `--smoke` on both commands and are labeled smoke in their receipts. New generation receipts hash every safetensors weight shard. Runs started before this enforcement retain their original receipts; evaluation labels their weight binding as legacy, separate provenance only, rather than claiming generation-time checksum binding.

Conversion manifests describe the conversion-time state and remain unchanged. `scripts/validate.py MODEL --receipt NEW_PATH` writes a separate validation receipt binding successful structural/tokenizer checks to the conversion manifest and artifact checksums; it refuses an existing receipt.

Completed new generations bind both sample and raw-response files by SHA-256 in the generation receipt. Evaluation rejects changed files and verifies that sample solutions match retained raw responses. Older unbound receipts require `--legacy-receipt`; this permits explicit provenance-limited evaluation, never retroactive generation-time binding. Successful new evaluations also record checksums of their protocol, sanitized samples and results. The archived initial comparison preserves original receipts and adds clearly labeled post-run archive checksums.

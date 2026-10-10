# Mellum MLX

Benchmarked MLX quantizations of JetBrains' [Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) for Apple Silicon, with the measurements and tool-calling findings behind them.

**Model:** [imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64) (9.87 GB, MLX 6-bit affine, group 64).

```sh
python -m pip install 'mlx==0.32.3' 'mlx-lm==0.32.0'
python -m mlx_lm generate --model imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64 \
  --prompt 'Find the bug: def mean(xs): return sum(xs) / len(xs) - 1' \
  --temp 0.6 --top-p 0.95 --top-k 20 --max-tokens 16384
```

## Which model for which Mac

| Your Mac | Use | Why |
| --- | --- | --- |
| 24 GB or more | [This 6-bit](https://huggingface.co/imaadd05/Mellum2.1-12B-A2.5B-Thinking-mlx-6bit-g64) | HumanEval+ 89.0% vs 87.8% for MXFP4, a statistical tie (McNemar p = 0.80); 6-bit sits closer to BF16 at 93.3%. |
| 16 GB | [randmaru MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4) | Ties this 6-bit on HumanEval+ at 6.46 GB. On the M4 16 GB, 6-bit peaked at 10.5 GB MLX memory with material swap. |

I tried to build a better 16 GB model than MXFP4 (mixed 4/6-bit recipes, group-32 experts, DWQ calibration). Nothing beat it, so I stopped and recommend it. The experiments are archived under `docs/` on the `archive/research-2026-10-11` tag.

## Results

164 HumanEval+ tasks, same CUDA GPU, upstream chat template, greedy decoding, 8,192 completion tokens, one sample per task:

| Model | Weights | HumanEval+ | Median completion tokens |
| --- | ---: | ---: | ---: |
| BF16 original | 24.3 GB | 93.3% | 1,571 |
| **This 6-bit** | 9.87 GB | 89.0% | 1,521 |
| randmaru MXFP4 | 6.46 GB | 87.8% | 1,544 |
| Plain affine 4-bit (not released) | 6.84 GB | 81.1% | 3,242 |

Apple M4 (16 GB), decode tokens/s at 1K / 4K / 16K context: 6-bit 48.0 / 47.1 / 43.5; MXFP4 67.3 / 64.5 / 58.4 (separate sequential runs, same machine). Full tables, receipts and limitations: [docs/RESULTS.md](docs/RESULTS.md).

## Findings

- **Plain affine 4-bit reasons 2.2x longer than BF16** on the same tasks (paired Wilcoxon p = 5e-25, n = 144), hitting the token budget mid-thought on 18 of 164. 6-bit and MXFP4 keep BF16's reasoning length. The cause is not established.
- **MLX-LM 0.32.0 drops unparseable tool calls silently**: the response keeps `finish_reason: "tool_calls"` but `tool_calls` is empty. A tested patch is in [patches/](patches/). The same bug has an open upstream fix, [ml-explore/mlx-lm#1957](https://github.com/ml-explore/mlx-lm/pull/1957).
- **Mellum's template replays prior reasoning only from `reasoning_content`**, but the server returns `reasoning`, so clients that echo messages back lose earlier thinking between tool steps.
- **Repository-agent trials are weak evidence**: one task, 1 success of 3 seeds after patches. Not a reliability claim.

Quantization methods and the model architecture come from upstream projects; this repository's contribution is measurement and integration.

## Reproduce

Python 3.12 was used. Keep the model cache and environments inside the checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
export HF_HOME="$PWD/work/hf-cache"
python -m pip install -r requirements.txt
# Linux CUDA 13 only; supported NVIDIA driver required:
python -m pip install 'mlx[cuda13]==0.32.3'
# Or reproduce the complete tested Linux environment:
# python -m pip install -r requirements-cuda.lock

python scripts/convert.py --bits 4
python scripts/validate.py artifacts/mellum2.1-affine-4bit-g64
python scripts/convert.py --bits 6
python scripts/validate.py artifacts/mellum2.1-affine-6bit-g64
```

On Apple Silicon, install `requirements.txt` and omit the CUDA install. Conversion downloads about 24.3 GB of BF16 weights and requires sufficient RAM/VRAM and disk space; the conversion has not been tested on a 16 GB Mac. Download the validated artifact once a model release is available rather than assuming conversion fits that machine.

The converter refuses to overwrite an existing artifact. A `conversion.json` receipt records source revision, software versions, quantization settings and SHA-256 checksums. Tokenizer and template files are preserved byte-for-byte. Weight files and caches are excluded from Git.

## Evaluate

```sh
python -m unittest discover -s tests -v
python scripts/score.py work/source --output results/local/bf16.json
python scripts/score.py artifacts/mellum2.1-affine-4bit-g64   --baseline results/local/bf16.json --output results/local/4bit.json
python scripts/protocol_probe.py artifacts/mellum2.1-affine-4bit-g64   --output results/local/protocol.json
python scripts/benchmark.py artifacts/mellum2.1-affine-4bit-g64   --context 1024 --max-tokens 256 --trials 3 --output results/local/benchmark-1k.json
python scripts/http_probe.py artifacts/mellum2.1-affine-4bit-g64 \
  --output results/local/http.json
python scripts/cache_probe.py artifacts/mellum2.1-affine-4bit-g64 \
  --output results/local/cache.json
```

`score.py` is an authored short-snippet regression probe, **not** a representative coding benchmark. It never executes model-generated code. `protocol_probe.py` checks fixed parser/serializer fixtures; passing it does not establish live HTTP or coding-agent reliability. `benchmark.py` uses a synthetic repeated-code prefix and greedy generation; reported MLX memory excludes OS memory and swap. CUDA and Metal results must remain separate. `cache_probe.py` compares the last eight positions of synthetic sequences around the sliding-window boundary; its differences are diagnostics, not a numerical parity pass.

For an independent short-context reference, create a separate environment, install CPU `torch==2.14.1` from the [official CPU wheel index](https://download.pytorch.org/whl/cpu), plus `transformers==5.19.0`, `numpy==2.5.3`, and `safetensors==0.8.0`, then run:

```sh
python scripts/reference_score.py work/source   --mlx-reference results/cuda/bf16-probe.json   --output results/local/transformers-bf16.json
```

For isolated executable coding comparisons, see [the fixed HumanEval+ protocol](docs/CODING_PROTOCOL.md). Full samples, outcomes and provenance limitations are published in [measurement notes](docs/RESULTS.md).

See [the research and experiment plan](docs/PLAN.md) and [measurement notes](docs/RESULTS.md). Reports refuse to overwrite prior measurements.

## Provenance and attribution

- Model: [JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking), revision `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`, Apache-2.0. JetBrains created and trained the model.
- Runtime: [MLX-LM](https://github.com/ml-explore/mlx-lm), tested package `0.32.0`; [MLX](https://github.com/ml-explore/mlx), `0.32.3`.
- Quantization: affine group size 64; 4 or 6 bits for eligible weights, native 8-bit Mellum routers, remaining weights BF16. Router protection is upstream behaviour, not a novel method introduced here.
- External comparator: [randmaru MXFP4](https://huggingface.co/randmaru/Mellum2.1-12B-A2.5B-Thinking-mlx-mxfp4), revision `4ce0d28df07dfa4ed28077e5b718e866be2cf4cf`. Credit belongs to its publisher.
- This repository's original scripts: MIT. The MIT license does not relicense JetBrains' weights or third-party code.

## DWQ pilot

```sh
python scripts/dwq_pilot.py targets
# Start a new process so the BF16 teacher does not coexist with the student.
python scripts/dwq_pilot.py train
```

This performs one update on separate authored pilot data. It tests compatibility, not quality improvement. An initial save/reload discrepancy was not reproduced on the instrumented repeat and remains unresolved; see the measurement notes. Do not publish the pilot as a tuned model.

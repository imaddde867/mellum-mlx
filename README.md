# Mellum MLX

Reproducible Mellum2.1 quantization experiments and validation for local coding-agent use.

**Research preview.** Native 4-bit and 6-bit conversions have been built and structurally validated on an RTX 5090. Short numerical probes, throughput measurements through 16K context, protocol fixtures and a simple live HTTP tool round trip are available under [results](results). The pinned external MXFP4 comparator has also been measured on a 16 GB M4 Mac through 16K context and passed the simple live HTTP round trip. The full 164-task HumanEval+ comparison is complete: BF16 93.3%, native 6-bit 89.0%, external MXFP4 87.8%, native 4-bit 81.1%. Native 4-bit passed M4 fidelity/HTTP smoke tests and throughput measurements through 16K. Native 6-bit M4 validation and a contained repository-agent trial are underway; long-context parity remains unresolved. No claim of superior coding quality or production readiness is made.

The contribution we aim to deliver is a measured quality–size–speed comparison, correct tool integration, and reproducible artifacts. Quantization methods and the model architecture come from upstream projects.

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

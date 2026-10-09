"""Synthetic cache-boundary diagnostic; numerical differences are not a parity pass."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def compare(reference, actual):
    import numpy as np
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Mismatched shapes or nonfinite diagnostic logits")
    difference = reference - actual
    def log_probs(value):
        shifted = value - value.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    p, q = log_probs(reference), log_probs(actual)
    return {"max_absolute_logit_difference": float(np.abs(difference).max()),
            "rms_logit_difference": float(np.sqrt(np.mean(difference ** 2))),
            "mean_kl_reference_to_actual": float(np.mean(np.sum(np.exp(p) * (p - q), axis=-1))),
            "top1_agreement": float(np.mean(reference.argmax(-1) == actual.argmax(-1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite cache diagnostic")
    import numpy as np
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    sample = np.array([[1., 2., 3.]])
    assert compare(sample, sample)["max_absolute_logit_difference"] == 0
    assert compare(sample, sample[:, ::-1])["mean_kl_reference_to_actual"] > 0
    try:
        compare(sample, sample * np.nan)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite comparison accepted")
    model, tokenizer = load(args.model)
    fixture = Path("fixtures/fidelity.jsonl")
    text = "\n".join(json.loads(line)["text"] for line in fixture.read_text().splitlines())
    seed = tokenizer.encode(text, add_special_tokens=False)
    if not seed:
        raise ValueError("Empty fixture tokens")
    rows = []
    for length in [1023, 1024, 1025, 2049]:
        print(json.dumps({"context": length, "active_memory_gb": mx.get_active_memory() / 1e9,
                          "cached_memory_gb_before_clear": mx.get_cache_memory() / 1e9}), flush=True)
        mx.clear_cache()
        tokens = (seed * (length // len(seed) + 1))[:length]
        def array(logits):
            mx.eval(logits)
            return np.array(logits.astype(mx.float32)).astype(np.float64)
        reference = array(model(mx.array([tokens]))[0, -8:])
        cases = [("uncached_repeat", array(model(mx.array([tokens]))[0, -8:])),
                 ("cached_single_prefill", array(model(mx.array([tokens]), cache=make_prompt_cache(model))[0, -8:]))]
        for chunk in [256, 512]:
            cache = make_prompt_cache(model)
            for start in range(0, length - 8, chunk):
                end = min(start + chunk, length - 8)
                logits = model(mx.array([tokens[start:end]]), cache=cache)
                mx.eval(logits)
                del logits
            tail = [array(model(mx.array([[token]]), cache=cache)[0]) for token in tokens[-8:]]
            cases.append((f"chunk_{chunk}_then_8_single_tokens", np.concatenate(tail)))
            del cache
        for mode, actual in cases:
            row = {"context": length, "mode": mode, "compared_positions": 8,
                   "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                   **compare(reference, actual)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {"model": args.model, "mlx": importlib.metadata.version("mlx"),
              "mlx_lm": importlib.metadata.version("mlx-lm"),
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "sliding_window": model.args.sliding_window,
              "scope": "synthetic repeated authored code; last eight positions; no correctness tolerance or representative quality claim",
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

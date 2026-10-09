"""Tiny cached-target DWQ compatibility pilot; not a quality-tuned release."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def check_roundtrip(delta, repeated, changed):
    if not math.isfinite(delta) or delta > 1e-4:
        raise ValueError(f"Save/reload changed or nonfinite logits: {delta}")
    if not math.isfinite(repeated) or repeated > 1e-4:
        raise ValueError(f"Repeated forward changed or nonfinite logits: {repeated}")
    if changed:
        raise ValueError(f"Save/reload changed parameters: {changed}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["targets", "train"])
    args = p.parse_args()
    import mlx.core as mx
    import mlx.optimizers as optimizers
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.quant.dwq import compute_dwq_targets, dwq_quantize
    from mlx_lm.utils import load_tokenizer, save

    mx.random.seed(123)
    source = Path("work/source")
    student = Path("artifacts/mellum2.1-affine-4bit-g64")
    cache = Path("work/dwq-pilot-targets")
    output = Path("artifacts/mellum2.1-dwq-pilot")
    data_path = Path("fixtures/dwq-pilot.json")
    data = json.loads(data_path.read_text())
    tokenizer = load_tokenizer(source)
    tokens = {split: tokenizer.encode(text, add_special_tokens=False)[:64]
              for split, text in data.items()}
    train, valid = [(tokens["train"], 0)], [(tokens["valid"], 0)]
    manifest = {"data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "tokens": tokens, "seed": 123, "max_seq_length": 64, "batch_size": 1,
                "top_k_teacher_logits": 1024,
                "source_receipt_sha256": hashlib.sha256((student / "conversion.json").read_bytes()).hexdigest()}
    if args.stage == "targets":
        if cache.exists():
            p.error("Refusing to overwrite pilot target cache")
        model, _ = load(str(source))
        compute_dwq_targets(model, cache, train, valid, batch_size=1, max_seq_length=64, seed=123)
        (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Cached pilot teacher targets; run training in a separate process.")
        return
    if output.exists():
        p.error("Refusing to overwrite pilot model")
    if json.loads((cache / "manifest.json").read_text()) != manifest:
        raise ValueError("Pilot target cache/settings mismatch")
    model, tokenizer, config = load(str(student), return_config=True)
    model.freeze()
    routers = {k: v for k, v in tree_flatten(model.parameters()) if ".mlp.gate." in k}
    before = {k: mx.array(v) for k, v in routers.items()}

    def targets(_, index, split):
        value = mx.load(cache / split / f"{index:010d}.safetensors")
        return value["logits"], value["indices"]

    dwq_quantize(model, targets, optimizers.Adam(learning_rate=1e-6, bias_correction=True),
                 train, valid, batch_size=1, max_seq_length=64, seed=123)
    after = {k: v for k, v in tree_flatten(model.parameters()) if ".mlp.gate." in k}
    if set(after) != set(before) or any(not mx.all(v == before[k]).item() for k, v in after.items()):
        raise ValueError("Router parameters changed")
    model.eval()
    prompt = mx.array([tokens["valid"][:-1]])
    expected = model(prompt).astype(mx.float32)
    mx.eval(expected)
    if not mx.all(mx.isfinite(expected)).item():
        raise ValueError("Nonfinite pilot outputs")
    save(output, student, model, tokenizer, config, donate_model=False)
    reloaded, _ = load(str(output))
    actual = reloaded(prompt).astype(mx.float32)
    if not mx.all(mx.isfinite(actual)).item():
        raise ValueError("Nonfinite reloaded pilot outputs")
    delta = mx.max(mx.abs(expected - actual)).item()
    repeated = mx.max(mx.abs(expected - model(prompt).astype(mx.float32))).item()
    original_params = dict(tree_flatten(model.parameters()))
    loaded_params = dict(tree_flatten(reloaded.parameters()))
    if set(original_params) != set(loaded_params):
        raise ValueError("Save/reload changed parameter keys")
    changed = []
    for key, value in original_params.items():
        other = loaded_params[key]
        if value.dtype != other.dtype or value.shape != other.shape or not mx.all(value == other).item():
            changed.append({"key": key, "original_dtype": str(value.dtype), "loaded_dtype": str(other.dtype)})
    print(json.dumps({"reload_delta": delta, "repeat_delta": repeated, "changed_parameters": changed}), flush=True)
    check_roundtrip(delta, repeated, changed)
    report = {"status": "passed", "scope": "one training update on authored separate pilot data; not quality evidence",
              "cache_manifest": manifest, "routers_unchanged": True,
              "save_reload_max_logit_difference": delta,
              "repeated_forward_max_logit_difference": repeated,
              "parameters_identical": True,
              "mlx_peak_memory_gb": mx.get_peak_memory() / 1e9}
    result = Path("results/cuda/dwq-pilot.json")
    if result.exists():
        raise ValueError("Refusing to overwrite pilot result")
    result.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

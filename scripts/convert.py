"""Convert the pinned BF16 source; never upload or overwrite an artifact."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "JetBrains/Mellum2.1-12B-A2.5B-Thinking"
REVISION = "92ddae9fc7665e9f801d141d2e5a6b2caf2460c4"


def mixed_plan(shapes, config, recipe):
    if recipe not in {'A', 'B', 'C'}:
        raise ValueError('Unknown mixed recipe')
    expected = {'model.embed_tokens'}
    if not config['tie_word_embeddings']:
        expected.add('lm_head')
    elif 'lm_head' in shapes:
        raise ValueError('Tied model unexpectedly has an independent head')
    for i in range(config['num_hidden_layers']):
        prefix = f'model.layers.{i}'
        expected.update(f'{prefix}.self_attn.{n}_proj' for n in ['q', 'k', 'v', 'o'])
        expected.add(f'{prefix}.mlp.gate')
        expected.update(f'{prefix}.mlp.switch_mlp.{n}_proj' for n in ['gate', 'up', 'down'])
    if set(shapes) != expected or any(s[-1] % 64 for s in shapes.values()):
        raise ValueError(f'Unexpected eligible module coverage: {sorted(set(shapes) ^ expected)}')
    plan = {}
    for path in shapes:
        bits = 4
        if path.endswith('.mlp.gate'):
            bits = 8
        elif recipe in {'A', 'C'} and '.self_attn.' in path:
            bits = 6
        elif recipe in {'B', 'C'} and path in {'model.embed_tokens', 'lm_head'}:
            bits = 6
        plan[path] = {'bits': bits, 'group_size': 64, 'mode': 'affine'}
    return plan


def estimate_payload(tensors, plan):
    total = 0
    for name, tensor in tensors.items():
        policy = plan.get(name.removesuffix('.weight')) if name.endswith('.weight') else None
        if policy:
            count = math.prod(tensor['shape'])
            total += count * policy['bits'] // 8 + count // 64 * 4
        else:
            total += tensor['nbytes']
    return total


def realized_map(config, plan):
    defaults = {k: config[k] for k in ['bits', 'group_size', 'mode']}
    realized = {path: {**defaults, **config.get(path, {})} for path in plan}
    if realized != plan:
        raise ValueError('Realized quantization differs from requested policy')
    return realized


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bits", type=int, choices=[4, 6], default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument('--recipe', choices=['A', 'B', 'C'])
    p.add_argument('--estimate-only', action='store_true')
    args = p.parse_args()
    dest = Path(f'artifacts/mellum2.1-mixed-{args.recipe}-g64' if args.recipe else
                f"artifacts/mellum2.1-affine-{args.bits}bit-g64")
    if dest.exists():
        p.error(f"Refusing to overwrite {dest}; inspect or move it before retrying.")
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm.convert import convert

    mx.set_default_device(getattr(mx, args.device))
    src = Path(snapshot_download(SOURCE, revision=REVISION, local_dir="work/source",
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model",
                        "LICENSE*", "NOTICE*", "*.txt", "README.md"]))
    config = json.loads((src / "config.json").read_text())
    if config.get("eos_token_id") != 28 or config.get("model_type") != "mellum":
        raise ValueError("Unexpected upstream config; re-audit before conversion.")
    if "quantization" in config or "quantization_config" in config:
        raise ValueError("Source must be the original unquantized checkpoint.")
    mixed = {}
    if args.recipe:
        if args.bits != 4:
            p.error('Mixed recipes require the 4-bit expert baseline')
        from mlx_lm import load
        from mlx_lm.utils import quantize_model, save
        from mlx.utils import tree_flatten
        from validate import check_hashes
        reference = json.loads(Path('artifacts/mellum2.1-affine-4bit-g64/conversion.json').read_text())
        if reference['source_revision'] != REVISION:
            raise ValueError('Pinned source receipt mismatch')
        check_hashes(src, reference['source_sha256'])
        model, tokenizer, live_config = load(str(src), return_config=True, lazy=True)
        shapes = {path: tuple(module.weight.shape) for path, module in model.named_modules()
                  if hasattr(module, 'to_quantized')}
        plan = mixed_plan(shapes, live_config, args.recipe)
        parameters = dict(tree_flatten(model.parameters()))
        tensors = {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'nbytes': v.nbytes}
                   for k, v in parameters.items()}
        excluded = {k: v for k, v in parameters.items() if k.removesuffix('.weight') not in plan}
        payload = estimate_payload(tensors, plan)
        mixed = {'recipe': args.recipe, 'precision_map': plan, 'source_tensor_specs': tensors,
                 'excluded_tensors': {k: tensors[k] for k in excluded},
                 'estimated_tensor_bytes': payload,
                 'estimated_weight_bytes_upper_bound': payload + 1024 * 1024,
                 'estimate_header_allowance_bytes': 1024 * 1024,
                 'target_bytes': 7_500_000_000,
                 'estimate_exceeds_target': payload + 1024 * 1024 > 7_500_000_000}
        print(json.dumps(mixed), flush=True)
        if args.estimate_only:
            return
        # Drop references to expert BF16 weights before materializing quantized weights.
        parameters.clear()
        model, live_config = quantize_model(model, live_config, 64, 4,
                                           quant_predicate=lambda path, _: plan[path])
        mixed['precision_map'] = realized_map(live_config['quantization'], plan)
        after = dict(tree_flatten(model.parameters()))
        for key, value in excluded.items():
            if after[key].dtype != value.dtype or not mx.all(after[key] == value).item():
                raise ValueError(f'Excluded tensor changed: {key}')
        mixed['excluded_unchanged_before_save'] = True
        after.clear()
        excluded.clear()
        save(dest, src, model, tokenizer, live_config)
    else:
        if args.estimate_only:
            p.error('--estimate-only requires --recipe')
        convert(str(src), str(dest), quantize=True, q_bits=args.bits, q_group_size=64,
                q_mode="affine", dtype="bfloat16", trust_remote_code=False)
    # Preserve upstream tokenizer files byte-for-byte, including a separate template.
    for pattern in ["tokenizer*", "*.jinja", "*.model", "special_tokens_map.json",
                    "added_tokens.json", "vocab.*", "merges.txt", "LICENSE*", "NOTICE*"]:
        for path in src.glob(pattern):
            if path.is_file():
                shutil.copy2(path, dest / path.name)
    source_hashes = {f.name: sha256(f) for f in sorted(src.iterdir()) if f.is_file()}
    artifact_hashes = {f.name: sha256(f) for f in sorted(dest.iterdir())
                       if f.is_file() and f.name != "README.md"}
    manifest = {
        "source": SOURCE, "source_revision": REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quantization": {"mode": "affine", "bits": args.bits, "group_size": 64,
                         "router_bits": 8, "non_quantized_dtype": "bfloat16"},
        "device": args.device, "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {pkg: importlib.metadata.version(pkg) for pkg in
                     ["mlx", "mlx-lm", "transformers", "huggingface-hub", "safetensors", "numpy"]},
        "source_sha256": source_hashes, "artifact_sha256": artifact_hashes,
        "weight_bytes": sum(f.stat().st_size for f in dest.glob("*.safetensors")),
        "validation_status": "pending",
        **mixed,
    }
    (dest / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Converted to {dest}. Validation and Apple Silicon benchmarks remain required.")


if __name__ == "__main__":
    main()

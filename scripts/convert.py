"""Convert the pinned BF16 source; never upload or overwrite an artifact."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "JetBrains/Mellum2.1-12B-A2.5B-Thinking"
REVISION = "92ddae9fc7665e9f801d141d2e5a6b2caf2460c4"


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bits", type=int, choices=[4, 6], default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    args = p.parse_args()
    dest = Path(f"artifacts/mellum2.1-affine-{args.bits}bit-g64")
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
    }
    (dest / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Converted to {dest}. Validation and Apple Silicon benchmarks remain required.")


if __name__ == "__main__":
    main()

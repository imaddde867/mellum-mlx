"""Validate provenance, serialized weights, tokenizer, and chat/tool templates."""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_hashes(directory, hashes):
    require(bool(hashes), "Missing checksums")
    for name, expected in hashes.items():
        require(Path(name).name == name and name not in {".", ".."}, "Unsafe filename")
        with (directory / name).open("rb") as f:
            require(hashlib.file_digest(f, "sha256").hexdigest() == expected,
                    f"Checksum mismatch: {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--source", type=Path, default=Path("work/source"))
    p.add_argument("--receipt", type=Path, help="Write a new validation receipt; never overwrite")
    args = p.parse_args()
    if args.receipt and args.receipt.exists():
        p.error("Validation receipt already exists")
    from safetensors import safe_open
    from transformers import AutoTokenizer

    model, source = args.model, args.source
    manifest = json.loads((model / "conversion.json").read_text())
    check_hashes(model, manifest["artifact_sha256"])
    cfg = json.loads((model / "config.json").read_text())
    require(cfg["eos_token_id"] == 28, "EOS changed")
    q = cfg["quantization"]
    require(q["bits"] == manifest["quantization"]["bits"] and q["group_size"] == 64
            and q["mode"] == "affine", "Quantization mismatch")
    routers = [v for k, v in q.items() if k.endswith("mlp.gate")]
    require(len(routers) == 28 and all(v["bits"] == 8 for v in routers),
            "Router precision mismatch")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    found = {}
    for shard in model.glob("*.safetensors"):
        with safe_open(shard, framework="np") as f:
            for key in f.keys():
                require(key not in found, f"Duplicate tensor: {key}")
                found[key] = shard.name
    require(found == index["weight_map"], "Shard index does not match serialized tensors")
    check_hashes(source, manifest["source_sha256"])
    original = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    converted = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    require(original.get_vocab() == converted.get_vocab(), "Vocabulary changed")
    require(converted.eos_token_id == 28, "Tokenizer EOS changed")
    require(original.chat_template == converted.chat_template, "Chat template changed")
    tools = [{"type": "function", "function": {"name": "read_file",
        "description": "Read a project file", "parameters": {"type": "object",
        "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    cases = [
        ([{"role": "user", "content": "Explain binary search."}], {}),
        ([{"role": "user", "content": "Read README.md."}], {"tools": tools}),
        ([{"role": "user", "content": "Read README.md."},
          {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
           "function": {"name": "read_file", "arguments": {"path": "README.md"}}}]},
          {"role": "tool", "content": "Project overview"}], {"tools": tools}),
    ]
    for messages, options in cases:
        a = original.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options)
        b = converted.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options)
        require(a == b, "Rendered chat/tool prompt changed")
    receipt = {"status": "passed", "tensors": len(found), "template_cases": len(cases),
        "conversion_sha256": hashlib.sha256((model / "conversion.json").read_bytes()).hexdigest(),
        "artifact_sha256": manifest["artifact_sha256"],
        "scope": "structural and tokenizer validation; inference not tested"}
    if args.receipt:
        with args.receipt.open("x") as f:
            f.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

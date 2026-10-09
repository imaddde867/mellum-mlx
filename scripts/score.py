"""Small teacher-forced regression probe; not a coding capability benchmark."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--fixtures", type=Path, default=Path("fixtures/fidelity.jsonl"))
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite a score report")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    model, tokenizer = load(args.model)
    fixtures = [json.loads(line) for line in args.fixtures.read_text().splitlines()]
    rows = []
    for case in fixtures:
        tokens = tokenizer.encode(case["text"], add_special_tokens=False)
        if not 2 <= len(tokens) <= 512:
            raise ValueError("Probe sequence must contain 2..512 tokens")
        x = mx.array([tokens[:-1]])
        logits = model(x).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, mx.array([tokens[1:]]))
        finite = mx.all(mx.isfinite(logits))
        top1 = mx.argmax(logits, axis=-1)
        mx.eval(losses, finite, top1)
        if not finite.item():
            raise ValueError(f"Nonfinite logits: {case['id']}")
        row = {"id": case["id"], "tokens": tokens,
               "nll_sum": losses.sum().item(), "scored_tokens": len(tokens) - 1,
               "top1": top1[0].tolist()}
        rows.append(row)
        print(case["id"], row["scored_tokens"], row["nll_sum"], flush=True)
    total_tokens = sum(row["scored_tokens"] for row in rows)
    mean_nll = sum(row["nll_sum"] for row in rows) / total_tokens
    report = {"model": args.model, "fixture_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
              "mlx": importlib.metadata.version("mlx"),
              "mlx_lm": importlib.metadata.version("mlx-lm"),
              "scored_tokens": total_tokens, "mean_nll": mean_nll,
              "perplexity": math.exp(mean_nll), "rows": rows,
              "scope": "12 authored short snippets, uncalibrated teacher-forced smoke regression; not representative quality"}
    if args.baseline:
        base = json.loads(args.baseline.read_text())
        if base["fixture_sha256"] != report["fixture_sha256"]:
            raise ValueError("Baseline fixture mismatch")
        if [r["id"] for r in base["rows"]] != [r["id"] for r in rows]:
            raise ValueError("Baseline cases mismatch")
        matches = 0
        for row, original in zip(rows, base["rows"], strict=True):
            if row["tokens"] != original["tokens"]:
                raise ValueError("Baseline tokenizer mismatch")
            matches += sum(a == b for a, b in zip(row["top1"], original["top1"], strict=True))
        report["delta_nll_vs_baseline"] = mean_nll - base["mean_nll"]
        report["top1_agreement_vs_baseline"] = matches / total_tokens
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()

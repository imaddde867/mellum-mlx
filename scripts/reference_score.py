"""Independent Transformers BF16 scoring of the same short regression fixtures."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mlx-reference", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite reference results")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=False).eval()
    fixtures_path = Path("fixtures/fidelity.jsonl")
    fixtures = [json.loads(line) for line in fixtures_path.read_text().splitlines()]
    baseline = json.loads(args.mlx_reference.read_text())
    if baseline["fixture_sha256"] != hashlib.sha256(fixtures_path.read_bytes()).hexdigest():
        raise ValueError("Fixture hash mismatch")
    rows = []
    matched = 0
    total = 0
    for case, reference in zip(fixtures, baseline["rows"], strict=True):
        tokens = tokenizer.encode(case["text"], add_special_tokens=False)
        if case["id"] != reference["id"] or tokens != reference["tokens"]:
            raise ValueError("Reference tokenizer/case mismatch")
        with torch.inference_mode():
            logits = model(torch.tensor([tokens[:-1]]), use_cache=False).logits.float()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite Transformers logits")
            nll = torch.nn.functional.cross_entropy(logits.flatten(0, 1),
                torch.tensor(tokens[1:]), reduction="sum").item()
            top1 = logits.argmax(dim=-1)[0].tolist()
        count = len(tokens) - 1
        matched += sum(a == b for a, b in zip(top1, reference["top1"], strict=True))
        total += count
        rows.append({"id": case["id"], "scored_tokens": count, "nll_sum": nll,
                     "top1": top1, "tokens": tokens})
        print(case["id"], nll, flush=True)
    if len(rows) != len(baseline["rows"]) or len(rows) != len(fixtures):
        raise ValueError("Reference row count mismatch")
    mean_nll = sum(row["nll_sum"] for row in rows) / total
    report = {"torch": torch.__version__, "transformers": transformers.__version__,
              "device": "cpu", "dtype": "bfloat16", "attention": "eager",
              "mean_nll": mean_nll, "perplexity": math.exp(mean_nll), "scored_tokens": total,
              "delta_nll_vs_mlx_bf16": mean_nll - baseline["mean_nll"],
              "top1_agreement_vs_mlx_bf16": matched / total,
              "scope": "independent short-context numerical probe; sliding-window and long-context parity pending",
              "fixture_sha256": baseline["fixture_sha256"], "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()

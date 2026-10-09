"""Record real chat generation timings; each process benchmarks one model."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--context", type=int, default=1024)
    args = p.parse_args()
    if args.max_tokens < 1 or args.trials < 1 or args.context < 32:
        p.error("Use positive trials/tokens and context >= 32")
    if args.output.exists():
        p.error("Refusing to overwrite a benchmark report")
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)
    message = "Explain why binary search requires sorted data and provide a Python implementation.\n"
    ids = tokenizer.apply_chat_template([{"role": "user", "content": message}],
                                        tokenize=True, add_generation_prompt=True)
    filler = tokenizer.encode("def add(a, b): return a + b\n", add_special_tokens=False)
    # shortcut: synthetic repeated-code prefix, use real repository contexts for application claims.
    if len(ids) > args.context:
        raise ValueError("Context shorter than the chat prompt")
    prompt = (filler * ((args.context - len(ids)) // len(filler) + 1))[:args.context - len(ids)] + ids
    if len(prompt) != args.context:
        raise ValueError("Prompt length mismatch")
    rows = []
    for trial in range(args.trials + 1):
        mx.random.seed(0)
        mx.reset_peak_memory()
        start = time.perf_counter()
        first = None
        text = ""
        response = None
        for response in stream_generate(model, tokenizer, prompt=prompt,
            max_tokens=args.max_tokens, sampler=make_sampler(temp=0), prefill_step_size=512):
            if first is None:
                first = time.perf_counter() - start
            text += response.text
        if response is None:
            raise RuntimeError("No generation response")
        row = {"warmup": trial == 0, "ttft_seconds": first,
               "elapsed_seconds": time.perf_counter() - start,
               "prompt_tokens": response.prompt_tokens,
               "generation_tokens": response.generation_tokens,
               "prompt_tokens_per_second": response.prompt_tps,
               "generation_tokens_per_second": response.generation_tps,
               "mlx_peak_memory_gb": response.peak_memory,
               "finish_reason": response.finish_reason, "text": text}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
    hardware = "unknown"
    if platform.system() == "Linux":
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True)
        hardware = result.stdout.strip() or "CPU/unknown GPU"
    elif platform.system() == "Darwin":
        hardware = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {"model": args.model, "platform": platform.platform(), "hardware": hardware,
              "mlx": importlib.metadata.version("mlx"),
              "mlx_lm": importlib.metadata.version("mlx-lm"),
              "context": args.context, "max_tokens": args.max_tokens,
              "sampling": "greedy", "prefill_step_size": 512,
              "memory_scope": "MLX allocator only; excludes OS, swap and unrelated processes",
              "trials": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

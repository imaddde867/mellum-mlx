"""Probe real MLX-LM parser/serializer fixtures; does not test HTTP generation."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite probe report")
    from mlx_lm.server import APIHandler, ToolCallFormatter
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(args.model)
    if tokenizer.tool_parser is None:
        raise ValueError("No inferred tool parser")
    calls = [
        {"name": "read_file", "arguments": {"path": "src/main.py"}},
        {"name": "search", "arguments": {"query": 'a "quoted" string\nλ', "limit": 3}},
    ]
    results = []
    for stream in [False, True]:
        handler = object.__new__(APIHandler)
        handler.request_id = "fixture"
        handler.system_fingerprint = "fixture"
        handler.object_type = "chat.completion.chunk" if stream else "chat.completion"
        handler.requested_model = args.model.name
        handler.created = 0
        handler.stream = stream
        for count in [1, 2]:
            formatter = ToolCallFormatter(tokenizer.tool_parser, [], streaming=stream)
            formatted = formatter([json.dumps(c) for c in calls[:count]])
            packet = handler.generate_response("", "tool_calls", 12, 24,
                tool_calls=formatted, reasoning_text="Inspect the project first.")
            payload = packet["choices"][0]["delta" if stream else "message"]
            actual = payload.get("tool_calls", [])
            if len(actual) != count or packet["choices"][0]["finish_reason"] != "tool_calls":
                raise ValueError("Dropped tool call")
            if payload.get("reasoning") != "Inspect the project first.":
                raise ValueError("Dropped reasoning")
            for i, (a, expected) in enumerate(zip(actual, calls)):
                if not a.get("id") or a["function"]["name"] != expected["name"]:
                    raise ValueError("Missing call ID/name")
                if json.loads(a["function"]["arguments"]) != expected["arguments"]:
                    raise ValueError("Changed JSON arguments")
                if stream and a.get("index") != i:
                    raise ValueError("Changed streaming call index")
            results.append({"stream": stream, "tool_calls": count, "status": "passed"})
    truncated = ToolCallFormatter(tokenizer.tool_parser, [])(['{"name":"read_file","arguments":'])
    if truncated:
        raise ValueError("Malformed JSON unexpectedly parsed")
    server = Path(inspect.getfile(APIHandler))
    report = {"mlx_lm": importlib.metadata.version("mlx-lm"),
        "server_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
        "scope": "fixed parser/serializer fixtures only; live HTTP, model decisions and tool-result round trips pending",
        "cases": results, "truncated_json": "rejected", "status": "passed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

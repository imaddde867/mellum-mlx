"""Bounded loopback HTTP tool-call smoke test; never executes generated tools."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        if payload and payload.get("stream"):
            packets = []
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                value = line[6:].strip()
                if value == b"[DONE]":
                    break
                packets.append(json.loads(value))
            calls = {}
            content, reasoning, finish = "", "", None
            for packet in packets:
                if not packet.get("choices"):
                    continue
                choice = packet["choices"][0]
                delta = choice.get("delta", {})
                content += delta.get("content") or ""
                reasoning += delta.get("reasoning") or ""
                finish = choice.get("finish_reason") or finish
                for item in delta.get("tool_calls", []):
                    i = item["index"]
                    out = calls.setdefault(i, {"id": "", "type": "function",
                                               "function": {"name": "", "arguments": ""}})
                    if item.get("id"):
                        out["id"] = item["id"]
                    fn = item.get("function", {})
                    out["function"]["name"] += fn.get("name") or ""
                    out["function"]["arguments"] += fn.get("arguments") or ""
            return {"message": {"role": "assistant", "content": content,
                                "reasoning": reasoning, "tool_calls": [calls[k] for k in sorted(calls)]},
                    "finish_reason": finish, "packets": packets}
        value = json.load(response)
        if payload is None:
            return value
        choice = value["choices"][0]
        return {"message": choice["message"], "finish_reason": choice["finish_reason"], "response": value}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite HTTP probe")
    hub_home = Path(os.environ.setdefault("HF_HOME", str(Path("work/hf-cache").resolve())))
    # MLX-LM lists the Hub cache even when serving a locally downloaded model.
    (hub_home / "hub").mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}/v1"
    log_path = Path("work/http-probe-server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with log_path.open("w") as log:
        server = subprocess.Popen([sys.executable, "-m", "mlx_lm", "server", "--model", args.model,
            "--host", "127.0.0.1", "--port", str(port), "--decode-concurrency", "1",
            "--prompt-concurrency", "1", "--log-level", "INFO"], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Server exited; inspect work/http-probe-server.log")
                try:
                    request(url + "/models")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup timed out")
                    time.sleep(0.25)
            tools = [{"type": "function", "function": {"name": "add",
                "description": "Add two numbers", "parameters": {"type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"]}}}]
            messages = [{"role": "user", "content": "Use the add tool to add 2 and 3. Call the tool before answering."}]
            for stream in [False, True]:
                payload = {"model": args.model, "messages": messages, "tools": tools,
                           "temperature": 0, "max_tokens": 1024, "stream": stream}
                first = request(url + "/chat/completions", payload)
                calls = first["message"].get("tool_calls", [])
                passed = first["finish_reason"] == "tool_calls" and len(calls) == 1
                if passed:
                    call = calls[0]
                    passed = bool(call.get("id")) and call["function"]["name"] == "add"
                    try:
                        passed = passed and json.loads(call["function"]["arguments"]) == {"a": 2, "b": 3}
                    except (ValueError, TypeError):
                        passed = False
                row = {"stream": stream, "tool_call_passed": passed, "first": first}
                if passed:
                    assistant = dict(first["message"])
                    assistant.pop("reasoning", None)
                    followup = messages + [assistant,
                        {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"result":5}'},
                        {"role": "user", "content": "Answer with exactly the number returned by the tool."}]
                    second = request(url + "/chat/completions", dict(payload, messages=followup))
                    row["second"] = second
                    row["round_trip_passed"] = (second["message"].get("content") or "").strip() == "5" and second["finish_reason"] == "stop"
                else:
                    row["round_trip_passed"] = False
                rows.append(row)
                print(stream, row["tool_call_passed"], row["round_trip_passed"], flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    report = {"model": args.model, "scope": "one simple addition scenario in each HTTP mode; not representative agent reliability",
              "reasoning": "upstream default template", "max_tokens": 1024,
              "status": "passed" if all(r["tool_call_passed"] and r["round_trip_passed"] for r in rows) else "failed",
              "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise SystemExit("HTTP probe failed; raw responses retained in report")


if __name__ == "__main__":
    main()

"""Smoke test of the three agent APIs on one service, with time to first token.

Standard library only. Reads the service from the same env vars as the client
launchers in clients/:
    ANYSCALE_BASE_URL  service URL ending in /v1 (default http://localhost:8000/v1)
    ANYSCALE_API_KEY   service bearer token

    python check_agents.py
"""

import json
import os
import time
import urllib.request

BASE_URL = os.environ.get("ANYSCALE_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = os.environ.get("ANYSCALE_API_KEY", "fake-key")
PROMPT = "Write a Python one-liner that reverses a string. Reply with code only."
# Reasoning is off so every check returns a short visible answer.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def call(path: str, body: dict | None = None):
    """Open a GET (no body) or POST to BASE_URL + path; the caller reads the response."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"authorization": f"Bearer {API_KEY}", "content-type": "application/json"}
    return urllib.request.urlopen(urllib.request.Request(BASE_URL + path, data=data, headers=headers), timeout=600)


def timed_json(path: str, body: dict | None = None) -> tuple[int, dict, float]:
    start = time.perf_counter()
    with call(path, body) as response:
        return response.status, json.load(response), time.perf_counter() - start


def show(name: str, status: int, seconds: float, excerpt: str) -> None:
    print(f"{name:<34} HTTP {status}  {seconds:5.2f}s  {excerpt.strip()[:80]!r}")


def check_models() -> str:
    status, body, seconds = timed_json("/models")
    model = body["data"][0]["id"]
    show("GET /v1/models", status, seconds, model)
    return model


def check_chat(model: str) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 64, **NO_THINKING}
    status, reply, seconds = timed_json("/chat/completions", body)
    show("POST /v1/chat/completions", status, seconds, reply["choices"][0]["message"]["content"])


def check_stream(model: str) -> None:
    """Stream a chat completion and report time to the first content token."""
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 64,
            "stream": True, **NO_THINKING}
    start, first_token_at, text = time.perf_counter(), None, ""
    with call("/chat/completions", body) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: {"):
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta and first_token_at is None:
                first_token_at = time.perf_counter()
            text += delta
        status = response.status
    total = time.perf_counter() - start
    ttft_ms = (first_token_at - start) * 1000 if first_token_at else float("nan")
    show("POST /v1/chat/completions stream", status, total, f"TTFT {ttft_ms:.0f} ms | {text}")


def check_messages(model: str) -> None:
    """Anthropic Messages API, the path Claude Code calls."""
    body = {"model": model, "max_tokens": 64, "system": "You are a terse coding assistant.",
            "messages": [{"role": "user", "content": PROMPT}], **NO_THINKING}
    status, reply, seconds = timed_json("/messages", body)
    text = next(b["text"] for b in reply["content"] if b["type"] == "text")
    show("POST /v1/messages (Anthropic)", status, seconds, text)


def check_responses(model: str) -> None:
    """OpenAI Responses API, the path Codex calls."""
    body = {"model": model, "input": PROMPT, "max_output_tokens": 64, **NO_THINKING}
    status, reply, seconds = timed_json("/responses", body)
    texts = [c["text"] for item in reply["output"] if item["type"] == "message" for c in item["content"]]
    show("POST /v1/responses (Codex)", status, seconds, " ".join(texts) or reply["status"])


def check_tool_call(model: str) -> None:
    """A tool the model should call; the qwen3_coder parser returns it as structured tool_calls."""
    tool = {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command and return stdout.",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}
    body = {"model": model, "messages": [{"role": "user", "content": "List the Python files here."}],
            "tools": [tool], "max_tokens": 256, **NO_THINKING}
    status, reply, seconds = timed_json("/chat/completions", body)
    calls = reply["choices"][0]["message"].get("tool_calls") or []
    excerpt = " ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in calls) or "no tool call"
    show("POST /v1/chat/completions tools", status, seconds, excerpt)


def main() -> None:
    print(f"service: {BASE_URL}\n")
    model = check_models()
    for check in (check_chat, check_stream, check_messages, check_responses, check_tool_call):
        check(model)


if __name__ == "__main__":
    main()

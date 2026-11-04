"""Prefix-cache demo: time to first token for one long prompt, cold then warm.

Each trial clears the prefix cache, sends the prompt (cold: the engine runs
prefill over every token), then sends it again (warm: the prompt's full KV blocks
are already cached, so only the last, partial block is computed).

The cache is cleared with POST <root>/reset_prefix_cache, which the dev ingress
in knobs.py adds. If the endpoint has no reset route, each trial instead puts a
fresh nonce at the start of the prompt, which changes every block hash.

Run:
    python prefix_cold_warm.py --base-url http://localhost:8000/v1
    python prefix_cold_warm.py --base-url https://<service-url>/v1 --api-key "$TOKEN"
"""


import argparse
import os
import statistics
import time
import uuid

import httpx
from openai import OpenAI

# 14 tokens per repetition with the Qwen tokenizer. The default 2200 repetitions
# (about 30.8K tokens) fit the 32K context set in knobs.py.
FILLER = "The quick brown fox jumps over the lazy dog near the river bank. "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="OpenAI base URL, ending in /v1.")
    parser.add_argument("--model", default="qwen-0.5b")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "fake-key"),
                        help="Bearer token. Defaults to $OPENAI_API_KEY.")
    parser.add_argument("--repeat", type=int, default=2200,
                        help="Filler repetitions; 2200 is about 30.8K tokens.")
    parser.add_argument("--trials", type=int, default=3)
    return parser.parse_args()


def time_to_first_token(client: OpenAI, model: str, prompt: str) -> float:
    """Stream one completion and return seconds until the first token arrives."""
    start = time.perf_counter()
    stream = client.completions.create(model=model, prompt=prompt, max_tokens=8, stream=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].text:
            stream.close()
            break
    return time.perf_counter() - start


def reset_prefix_cache(root_url: str, model: str, api_key: str) -> bool:
    """Clear the prefix cache on every replica. False if the route does not exist."""
    resp = httpx.post(f"{root_url}/reset_prefix_cache", json={"model": model},
                      headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return True


def run_trial(client: OpenAI, args: argparse.Namespace, root_url: str) -> tuple[float, float]:
    """Return (cold, warm) TTFT in seconds for one trial."""
    prompt = FILLER * args.repeat
    if not reset_prefix_cache(root_url, args.model, args.api_key):
        # No reset route: a unique first block makes every later block hash miss too.
        print("no /reset_prefix_cache route; using a fresh nonce for the cold request")
        prompt = f"[{uuid.uuid4()}] {prompt}"
    cold = time_to_first_token(client, args.model, prompt)
    warm = time_to_first_token(client, args.model, prompt)
    return cold, warm


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    root_url = args.base_url.rstrip("/").removesuffix("/v1")
    # One untimed request so model load and connection setup stay out of trial 1.
    time_to_first_token(client, args.model, "Hello")

    colds, warms = [], []
    for trial in range(1, args.trials + 1):
        cold, warm = run_trial(client, args, root_url)
        colds.append(cold)
        warms.append(warm)
        print(f"trial {trial}: cold {cold * 1000:7.0f} ms   warm {warm * 1000:7.0f} ms")

    cold_ms = statistics.median(colds) * 1000
    warm_ms = statistics.median(warms) * 1000
    print(f"median : cold {cold_ms:7.0f} ms   warm {warm_ms:7.0f} ms   "
          f"({cold_ms / warm_ms:.1f}x faster from cache)")


if __name__ == "__main__":
    main()

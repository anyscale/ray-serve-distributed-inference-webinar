"""Does a prefix computed on one replica help another? Time to first token.

One session prompt goes to replica a (cold), then to replica b, then to b
again (a GPU hit there).

  no shared pool   b computes the whole prompt again: about as slow as cold
  shared pool      b loads a's blocks from the pool: much faster than cold

    python across_replicas.py --base-url $URL --api-key $KEY
"""

import argparse
import os
import random
import time
import uuid

from openai import OpenAI

WORDS = "cache block prefix decode prefill token replica router engine pool".split()


def long_prompt(words: int) -> str:
    """A unique prompt: a fresh id first, so nothing in it is cached yet."""
    rng = random.Random()
    return f"[{uuid.uuid4()}] " + " ".join(rng.choice(WORDS) for _ in range(words)) + "\nSummarize."


def first_token(client: OpenAI, model: str, prompt: str) -> tuple[float, int | None]:
    """Seconds to the first streamed token, and the cached prompt tokens if reported."""
    start = time.perf_counter()
    ttft, cached = None, None
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8,
        stream=True, stream_options={"include_usage": True})
    for chunk in stream:
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.perf_counter() - start
        if chunk.usage and chunk.usage.prompt_tokens_details:
            cached = chunk.usage.prompt_tokens_details.cached_tokens
    return ttft or time.perf_counter() - start, cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000", help="the service URL, without /v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--model", default="qwen2.5-7b")
    parser.add_argument("--prompt-words", type=int, default=6000, help="about 1 token per word")
    args = parser.parse_args()
    replica = {name: OpenAI(base_url=f"{args.base_url.rstrip('/')}/{name}/v1", api_key=args.api_key, timeout=600)
               for name in ("a", "b")}

    session = long_prompt(args.prompt_words)
    rows = [("a, cold", *first_token(replica["a"], args.model, session)),
            ("b, first time", *first_token(replica["b"], args.model, session)),
            ("b, again", *first_token(replica["b"], args.model, session))]

    print(f"{'request':<16} {'TTFT':>8} {'cached tokens':>14}")
    for name, ttft, cached in rows:
        print(f"{name:<16} {ttft * 1000:6.0f} ms {cached if cached is not None else '-':>14}")


if __name__ == "__main__":
    main()

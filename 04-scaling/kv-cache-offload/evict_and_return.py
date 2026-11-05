"""Does a prefix survive eviction? Time to first token before and after.

One session prompt is sent cold, then again at once (a GPU cache hit). Filler
prompts that share nothing with it then fill the GPU KV cache, so the session's
blocks are evicted. Finally the session prompt returns.

  no offload tier   the return recomputes the whole prompt: about as slow as cold
  a CPU tier        the return loads the blocks back: much faster than cold

The service must cap its GPU cache (`gpu_kv_blocks` in service.yaml) so that
--fillers x --prompt-words overflows it.

    python evict_and_return.py --base-url $URL/v1 --api-key $KEY
"""

import argparse
import os
import random
import time
import uuid

from openai import OpenAI

WORDS = "cache block prefix decode prefill token replica router engine tier".split()


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
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--model", default="qwen-0.5b")
    parser.add_argument("--prompt-words", type=int, default=6000, help="about 1 token per word")
    parser.add_argument("--fillers", type=int, default=6, help="prompts sent to evict the session")
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)

    session = long_prompt(args.prompt_words)
    rows = [("cold", *first_token(client, args.model, session)),
            ("warm (GPU hit)", *first_token(client, args.model, session))]
    for _ in range(args.fillers):
        first_token(client, args.model, long_prompt(args.prompt_words))
    rows.append(("after eviction", *first_token(client, args.model, session)))

    print(f"{'request':<16} {'TTFT':>8} {'cached tokens':>14}")
    for name, ttft, cached in rows:
        print(f"{name:<16} {ttft * 1000:6.0f} ms {cached if cached is not None else '-':>14}")


if __name__ == "__main__":
    main()

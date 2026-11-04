"""Concurrency sweep against an OpenAI-compatible endpoint, written as a CSV.

At each concurrency level, `c` clients each send a request, wait for it to
finish, and send the next, until `rounds * c` requests are done. Prompts are
random after the chat template's first few tokens, so the prefix cache saves
almost nothing. Output lengths are drawn from [range_ratio * output_tokens,
output_tokens] and forced with `ignore_eos`, a vLLM and SGLang extension. The
spread keeps clients from finishing in lockstep and sending their prefills all
at once.

CSV columns, one row per level:
  concurrency           clients in flight
  output_tok_s_per_gpu  output tokens per second, divided by --num-gpus
  total_tok_s_per_gpu   prompt plus output tokens per second, per GPU
  tpot_ms               mean time per output token after the first, in ms
  ttft_p50_ms, ttft_p90_ms  time to first token percentiles, in ms
Speed per user is 1000 / tpot_ms tokens per second.

Run:
    python sweep.py --base-url http://localhost:8000/v1 --out results/sweep_0.5b_cuda-graphs.csv
"""


import argparse
import asyncio
import csv
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass

from openai import AsyncOpenAI

WORDS = ("river cloud stone paper light engine garden window market silver "
         "forest signal number planet bridge morning orange travel winter music").split()
CSV_FIELDS = ["concurrency", "output_tok_s_per_gpu", "total_tok_s_per_gpu", "tpot_ms",
              "ttft_p50_ms", "ttft_p90_ms", "requests", "errors", "prompt_tokens",
              "output_tokens"]


@dataclass(frozen=True)
class RequestResult:
    """One streamed request: latencies in seconds, token counts from `usage`."""

    ttft_s: float
    total_s: float
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LevelResult:
    """One CSV row: the aggregate of every request at one concurrency level."""

    concurrency: int
    output_tok_s_per_gpu: float
    total_tok_s_per_gpu: float
    tpot_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    requests: int
    errors: int
    prompt_tokens: int
    output_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="OpenAI base URL, ending in /v1.")
    parser.add_argument("--model", default="qwen-0.5b")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "fake-key"),
                        help="Bearer token. Defaults to $OPENAI_API_KEY.")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64",
                        help="Comma-separated concurrency levels.")
    parser.add_argument("--input-words", type=int, default=1000,
                        help="Random words per prompt; one token each with Qwen.")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--range-ratio", type=float, default=0.8,
                        help="Shortest output as a fraction of --output-tokens.")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Requests per client at each level.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="GPUs behind the endpoint, for the per-GPU columns.")
    parser.add_argument("--out", default="results/sweep.csv")
    return parser.parse_args()


def random_prompt(num_words: int) -> str:
    """A prompt whose first words differ from every other prompt's."""
    words = " ".join(random.choices(WORDS, k=num_words))
    return f"Continue this list of words without stopping: {words}"


async def send_request(client: AsyncOpenAI, args: argparse.Namespace) -> RequestResult:
    """Stream one chat completion and time its first and last token."""
    start = time.perf_counter()
    ttft = None
    usage = None
    stream = await client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": random_prompt(args.input_words)}],
        max_tokens=random.randint(int(args.range_ratio * args.output_tokens), args.output_tokens),
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True},
    )
    async for chunk in stream:
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.perf_counter() - start
        usage = chunk.usage or usage
    total = time.perf_counter() - start
    return RequestResult(ttft or total, total, usage.prompt_tokens, usage.completion_tokens)


async def run_level(client: AsyncOpenAI, args: argparse.Namespace, concurrency: int):
    """Run `rounds` requests per client with `concurrency` clients in flight."""
    results: list[RequestResult] = []
    errors = 0

    async def client_loop() -> None:
        nonlocal errors
        for _ in range(args.rounds):
            try:
                results.append(await send_request(client, args))
            except Exception as exc:  # one failed request should not end the sweep
                errors += 1
                print(f"  request failed: {exc!r}")

    start = time.perf_counter()
    await asyncio.gather(*(client_loop() for _ in range(concurrency)))
    return results, errors, time.perf_counter() - start


def summarize(concurrency: int, results: list[RequestResult], errors: int,
              elapsed_s: float, num_gpus: int) -> LevelResult:
    """Aggregate one level's requests into a CSV row."""
    prompt_tokens = sum(r.prompt_tokens for r in results)
    output_tokens = sum(r.output_tokens for r in results)
    tpots = [(r.total_s - r.ttft_s) / (r.output_tokens - 1)
             for r in results if r.output_tokens > 1]
    ttfts = sorted(r.ttft_s for r in results)
    deciles = statistics.quantiles(ttfts, n=10) if len(ttfts) > 1 else ttfts * 9
    return LevelResult(
        concurrency=concurrency,
        output_tok_s_per_gpu=round(output_tokens / elapsed_s / num_gpus, 1),
        total_tok_s_per_gpu=round((prompt_tokens + output_tokens) / elapsed_s / num_gpus, 1),
        tpot_ms=round(statistics.mean(tpots) * 1000, 2),
        ttft_p50_ms=round(statistics.median(ttfts) * 1000, 1),
        ttft_p90_ms=round(deciles[8] * 1000, 1),
        requests=len(results),
        errors=errors,
        prompt_tokens=round(prompt_tokens / len(results)),
        output_tokens=round(output_tokens / len(results)),
    )


def print_row(row: LevelResult) -> None:
    """Print one level as it finishes, so points show up as they arrive."""
    print(f"c={row.concurrency:>3}  out {row.output_tok_s_per_gpu:8.1f} tok/s/GPU  "
          f"total {row.total_tok_s_per_gpu:8.1f}  TPOT {row.tpot_ms:6.2f} ms "
          f"({1000 / row.tpot_ms:5.1f} tok/s/user)  TTFT p50 {row.ttft_p50_ms:7.1f} "
          f"p90 {row.ttft_p90_ms:7.1f} ms  errors {row.errors}")


async def main() -> None:
    args = parse_args()
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600.0)
    levels = [int(c) for c in args.concurrency.split(",")]
    # Warm-up: open one connection per client at the highest level, so connection
    # setup over the internet stays out of every level's latencies.
    await asyncio.gather(*(send_request(client, args) for _ in range(max(levels))))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for concurrency in levels:
            results, errors, elapsed_s = await run_level(client, args, concurrency)
            if not results:
                print(f"c={concurrency:>3}  every request failed; stopping")
                break
            row = summarize(concurrency, results, errors, elapsed_s, args.num_gpus)
            print_row(row)
            writer.writerow(asdict(row))
            f.flush()  # plot_frontier.py can redraw from a sweep still in progress
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

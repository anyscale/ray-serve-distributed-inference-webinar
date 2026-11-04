"""Decode inter-token latency (ITL), quiet and with a long-prompt injector.

Several streams decode long replies. In the quiet phase nothing else runs. In
the injected phase an injector sends long, unique prompts at a fixed rate, each
needing a full prefill and only one output token. The rate is fixed (open
loop), so both apps get the same prefill load however fast they answer.

In a collocated engine those prefills share steps with the decoding streams,
so the gaps between their tokens stretch. With prefill/decode disaggregation
the prefills run on another replica, and the decode rhythm should barely move.

Run against whichever app is deployed:
    python itl.py --base-url $URL/v1 --api-key $KEY --label disaggregated
    python itl.py --label collocated   # a local `serve run app:build_collocated_app`

The injected prompts must fit max_model_len (4096 in app.py). Words are not
tokens, so the table reports the mean prompt length the server counted.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import time
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
         "xray yankee zulu red green blue orange purple river mountain forest ocean "
         "desert valley island harbor bridge tower castle garden market library").split()


@dataclass(frozen=True)
class PhaseResult:
    """ITL percentiles for one phase, in milliseconds."""

    phase: str
    gaps: int
    p50: float
    p90: float
    p99: float
    worst: float
    injected: int
    prompt_tokens: float  # mean server-counted length of the injected prompts
    inject_errors: int


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile of a non-empty list."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def long_prompt(words: int) -> str:
    """A unique prompt of about `words` tokens, so nothing can come from a cache."""
    body = " ".join(random.choices(WORDS, k=words))
    return f"{uuid.uuid4()} {body}\nRepeat the first word."


async def decode_stream(client: AsyncOpenAI, model: str, max_tokens: int) -> list[float]:
    """Stream one long reply; return the gaps between tokens after the first, in s."""
    stream = await client.completions.create(
        model=model, prompt=f"{uuid.uuid4()} Count upward from one, in words:",
        max_tokens=max_tokens, stream=True,
        extra_body={"ignore_eos": True},  # vLLM option: always produce max_tokens
    )
    stamps = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].text:
            stamps.append(time.perf_counter())
    return [b - a for a, b in zip(stamps, stamps[1:])]


@dataclass(frozen=True)
class InjectorResult:
    """What the injector sent: server-counted prompt lengths and failures."""

    prompt_tokens: list[int]
    errors: int


async def send_long_prompt(client: AsyncOpenAI, model: str, words: int) -> int | str:
    """One long prompt with one output token. Returns its token count, or the error."""
    try:
        reply = await client.completions.create(
            model=model, prompt=long_prompt(words), max_tokens=1)
        return reply.usage.prompt_tokens
    except Exception as exc:  # noqa: BLE001 - counted, and the first one shown
        return f"{type(exc).__name__}: {exc}"[:300]


async def inject(client: AsyncOpenAI, args: argparse.Namespace,
                 stop: asyncio.Event) -> InjectorResult:
    """Send a long prompt every 1/rate seconds until `stop` is set, then wait for all."""
    sent = []
    while not stop.is_set():
        sent.append(asyncio.create_task(send_long_prompt(client, args.model, args.prompt_words)))
        await asyncio.sleep(1 / args.inject_rate)
    results = await asyncio.gather(*sent)
    errors = [r for r in results if isinstance(r, str)]
    if errors:
        print(f"injector errors: {len(errors)}, first: {errors[0]}", flush=True)
    return InjectorResult([r for r in results if isinstance(r, int)], len(errors))


async def run_phase(client: AsyncOpenAI, args: argparse.Namespace,
                    injecting: bool) -> PhaseResult:
    """Decode `args.streams` replies, with or without the injector running."""
    stop = asyncio.Event()
    injector = asyncio.create_task(inject(client, args, stop)) if injecting else None
    per_stream = await asyncio.gather(
        *(decode_stream(client, args.model, args.max_tokens) for _ in range(args.streams)))
    stop.set()
    injected = await injector if injector else InjectorResult([], 0)

    gaps_ms = [g * 1000 for gaps in per_stream for g in gaps]
    lengths = injected.prompt_tokens
    return PhaseResult("with injector" if injecting else "quiet", len(gaps_ms),
                       percentile(gaps_ms, 0.5), percentile(gaps_ms, 0.9),
                       percentile(gaps_ms, 0.99), max(gaps_ms), len(lengths),
                       statistics.fmean(lengths) if lengths else 0.0, injected.errors)


def format_table(label: str, rows: list[PhaseResult]) -> str:
    """One row per phase, ITL in milliseconds."""
    lines = [f"{label}: decode inter-token latency (ms)",
             f"{'phase':<14} {'gaps':>6} {'p50':>7} {'p90':>7} {'p99':>7} {'max':>7} "
             f"{'injected':>9} {'tok/prompt':>10} {'errors':>7}"]
    lines += [f"{r.phase:<14} {r.gaps:>6} {r.p50:>7.1f} {r.p90:>7.1f} {r.p99:>7.1f} "
              f"{r.worst:>7.1f} {r.injected:>9} {r.prompt_tokens:>10.0f} {r.inject_errors:>7}"
              for r in rows]
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    """Warm up, then measure the quiet phase and the injected phase."""
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=300, max_retries=0)
    await decode_stream(client, args.model, 16)  # first-call costs stay out of the numbers
    quiet = await run_phase(client, args, injecting=False)
    loaded = await run_phase(client, args, injecting=True)
    table = format_table(args.label, [quiet, loaded])
    print("\n" + table, flush=True)
    if args.results_file:
        with open(args.results_file, "a") as f:
            f.write(table + "\n\n")


def parse_args() -> argparse.Namespace:
    """Command-line options; defaults suit Qwen2.5-0.5B with max_model_len=4096."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen-0.5b")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"),
                        help="service bearer token; defaults to $OPENAI_API_KEY")
    parser.add_argument("--label", default="endpoint")
    parser.add_argument("--streams", type=int, default=8, help="parallel decoding streams")
    parser.add_argument("--max-tokens", type=int, default=512, help="tokens per stream")
    parser.add_argument("--inject-rate", type=float, default=5.0,
                        help="long prompts per second while injecting")
    parser.add_argument("--prompt-words", type=int, default=1500,
                        help="1.5-2 tokens each; keep the prompt under max_model_len")
    parser.add_argument("--results-file", default=None, help="also append the table here")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

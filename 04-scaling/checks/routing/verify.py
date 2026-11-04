"""Routing verifier: prefix reuse A/B, a concurrency stall check, and a follow-up check.

`--router` picks the router under test: `affinity` (PrefixCacheAffinityRouter,
the shipped config) or `kvaware` (KVAwareRouter, which needs direct streaming,
body forwarding and ai-dynamo; see ../../kv-aware-routing/service.yaml).

Check 1, reuse. Two requests share a ~1,440-token prefix and arrive together at
two replicas. The default router sends the second one to the free replica, which
recomputes the prefix (~0% reuse). The affinity router waits for the replica
that holds it (~49% reuse; the first request must fill the cache, so 49% is the
ceiling). Three details make this a real measurement:

  * `max_ongoing_requests=1` gives each replica one slot, so placement is forced.
    Serve LLM's default is so large that one in-flight request is invisible load.
    This is a measurement instrument and must never be deployed.
  * Both requests fire together. Sent one after the other, both replicas are idle,
    power of two ties, and its tie-break is not random.
  * The counters are vLLM's prefix-cache token counters, read from every node's
    Ray metrics agent, not from the Serve port.

Check 2, stall. An earlier run (Ray 2.58, vLLM 0.26) saw the affinity router stop
serving at 2-4 concurrent requests while `serve status` said HEALTHY. This check
deploys the shipped config and fires 2-8 parallel streams per round; any request
that misses its deadline while the app reports RUNNING is a stall. Run it with
`--stall-router default` too, to learn whether a stall is specific to the router.

Check 3, follow-ups (`--followups N`). N pairs, one request at a time: a seed
with a fresh prefix, then a follow-up on it. Both replicas are idle at each
decision, so only a cache-aware router sends the follow-up to its seed's
replica. vLLM's per-request cached_tokens says where it landed. This suits
KVAwareRouter, which scores on KV blocks already indexed from engine events, so
two simultaneous requests (check 1) can both look cold to it.

Run from this directory on a workspace with at least 2 GPUs:
    python verify.py --skip-stall                           # the A/B only
    python verify.py --skip-reuse --stall-router default    # stall, default router
    python verify.py --router kvaware --followups 8         # needs direct streaming

`--results-file` also appends every result line to a file, which a job can
print at the end, after the Ray log noise.

Exits 1 if the router under test loses a check or if any request stalls.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass

import ray
import requests
from openai import AsyncOpenAI
from ray import serve
from ray.serve.config import RequestRouterConfig
from ray.serve.llm import build_openai_app

from app import AFFINITY_ROUTER, DEFAULT_MODEL_ID, DEFAULT_MODEL_SOURCE, routed_config

HITS_METRIC = "ray_vllm_prefix_cache_hits_total"
QUERIES_METRIC = "ray_vllm_prefix_cache_queries_total"
EXPORT_WAIT_S = 15  # Ray's metrics agents republish about every 10 s
REQUEST_TIMEOUT_S = 180.0
STALL_DEADLINE_S = 60.0  # a 0.5B model answers 32 tokens in well under 1 s
BASE_URL = "http://localhost:8000/v1"
APP_NAME = "default"
SERVE_LOGGING = {"log_level": "WARNING"}  # keeps replica INFO lines out of job logs
KV_ROUTER = "ray.serve.llm.request_router.KVAwareRouter"
DIRECT_STREAMING = os.environ.get("RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING") == "1"
# With direct streaming and no router configured, Serve LLM uses round robin.
ROUTER_LABELS = {
    "default": "default router (" + ("round robin" if DIRECT_STREAMING else "power of two") + ")",
    "affinity": "PrefixCacheAffinityRouter",
    "kvaware": "KVAwareRouter",
}
RESULTS: list[str] = []  # result lines, also written to --results-file

# Spans many 16-token KV blocks and leaves room inside max_model_len=4096.
SHARED_PREFIX = (
    "You are a meticulous assistant. Follow these standing instructions exactly "
    "and answer in one short sentence. " * 80
)


def report(line: str, error: bool = False) -> None:
    """Print a result line and keep it for --results-file."""
    print(line, file=sys.stderr if error else sys.stdout, flush=True)
    RESULTS.append(line)


@dataclass(frozen=True)
class Arm:
    """One deployment under test: which router, and how many slots per replica."""

    router: str  # a key of ROUTER_LABELS
    max_ongoing_requests: int | None  # None keeps the shipped default


@dataclass(frozen=True)
class RoundResult:
    """One burst of parallel requests and what the app reported afterwards."""

    concurrency: int
    completed: int
    slowest_s: float
    app_status: str
    first_error: str  # empty when every request finished


def deploy(arm: Arm, model_id: str, model_source: str) -> None:
    """Deploy the shipped `routed_config`, changed only as `arm` says.

    Per-request cached-token counts are switched on for every arm; they change
    what the response reports, not how it is served.
    """
    config = routed_config(model_id, model_source, min_replicas=2)
    config.deployment_config["autoscaling_config"]["max_replicas"] = 2
    config.engine_kwargs["enable_prompt_tokens_details"] = True
    if arm.max_ongoing_requests is not None:
        config.deployment_config["max_ongoing_requests"] = arm.max_ongoing_requests
    if arm.router == "default":
        config.deployment_config.pop("request_router_config", None)
    elif arm.router == "kvaware":
        config.deployment_config["request_router_config"] = RequestRouterConfig(
            request_router_class=KV_ROUTER)
    serve.run(build_openai_app({"llm_configs": [config]}), name=APP_NAME,
              logging_config=SERVE_LOGGING, blocking=False)


def teardown() -> None:
    """Remove every Serve app so the next arm starts from empty caches."""
    serve.shutdown()
    time.sleep(5)


def make_client() -> AsyncOpenAI:
    """OpenAI client for the local Serve proxy, with no client-side retries."""
    return AsyncOpenAI(base_url=BASE_URL, api_key="not-needed",
                       timeout=REQUEST_TIMEOUT_S, max_retries=0)


async def time_to_first_token(client: AsyncOpenAI, question: str, model_id: str,
                              prefix: str = SHARED_PREFIX, max_tokens: int = 16) -> float:
    """Seconds to the first streamed token; drains the stream to the end."""
    started = time.perf_counter()
    first = float("nan")
    stream = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": prefix},
                  {"role": "user", "content": question}],
        max_tokens=max_tokens, stream=True,
    )
    async for chunk in stream:
        if math.isnan(first) and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter() - started
    return first


def sum_metric(text: str, name: str) -> float:
    """Sum every sample of a Prometheus counter, skipping its `_created` twin."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and line[len(name):len(name) + 1] in ("{", " "):
            total += float(line.rsplit(" ", 1)[-1])
    return total


def prefix_cache_counters() -> tuple[float, float]:
    """(hits, queries) in prompt tokens, summed over every live node's metrics agent."""
    text = "".join(
        requests.get(f"http://{n['NodeManagerAddress']}:{n['MetricsExportPort']}/metrics",
                     timeout=10).text
        for n in ray.nodes() if n["Alive"]
    )
    return sum_metric(text, HITS_METRIC), sum_metric(text, QUERIES_METRIC)


async def measure_reuse(router: str, model_id: str, model_source: str) -> tuple[float, float]:
    """Run the two-request experiment on one arm. Returns (reuse, slower TTFT in s)."""
    deploy(Arm(router, max_ongoing_requests=1), model_id, model_source)
    client = make_client()
    # Warm both replicas with an unrelated prefix so SHARED_PREFIX stays cold.
    for _ in range(4):
        await time_to_first_token(client, "hello", model_id, prefix="Warm up the engine. ")

    time.sleep(EXPORT_WAIT_S)
    hits_before, queries_before = prefix_cache_counters()
    latencies = await asyncio.gather(
        time_to_first_token(client, "Summarize the instructions.", model_id),
        time_to_first_token(client, "Now restate them briefly.", model_id),
    )
    time.sleep(EXPORT_WAIT_S)
    hits_after, queries_after = prefix_cache_counters()
    teardown()

    queried = queries_after - queries_before
    reuse = (hits_after - hits_before) / queried if queried else float("nan")
    return reuse, max(latencies)


async def check_reuse(router: str, model_id: str, model_source: str) -> bool:
    """Check 1: the router under test must beat the default router by a wide margin."""
    default_reuse, default_ttft = await measure_reuse("default", model_id, model_source)
    report(f"  {ROUTER_LABELS['default']}: prompt tokens reused {default_reuse:6.1%} "
           f"| slower TTFT {default_ttft * 1000:7.1f} ms")
    router_reuse, router_ttft = await measure_reuse(router, model_id, model_source)
    report(f"  {ROUTER_LABELS[router]}: prompt tokens reused {router_reuse:6.1%} "
           f"| slower TTFT {router_ttft * 1000:7.1f} ms")

    # 25 points is a wide margin around the expected 0% vs ~49%.
    gap = router_reuse - default_reuse
    report(f"  reuse gap: {gap * 100:+.1f} points")
    if gap > 0.25 and router_reuse > 0.25:
        report(f"PASS: the {ROUTER_LABELS[router]} steers a shared prefix to the replica "
               "holding its KV.")
        return True
    report(f"FAIL: the {ROUTER_LABELS[router]} is not reusing the shared prefix. Check\n"
           "      enable_prefix_caching, that both replicas came up, and the router's\n"
           "      load-balancing threshold.", error=True)
    return False


async def cached_fraction(client: AsyncOpenAI, prefix: str, question: str,
                          model_id: str) -> float:
    """Share of one request's prompt tokens served from the prefix cache (vLLM's count)."""
    reply = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": prefix}, {"role": "user", "content": question}],
        max_tokens=8,
    )
    details = reply.usage.prompt_tokens_details
    cached = details.cached_tokens if details and details.cached_tokens is not None else 0
    return cached / reply.usage.prompt_tokens


async def followup_fractions(router: str, pairs: int, model_id: str,
                             model_source: str) -> list[float]:
    """Seed a fresh prefix, then send a follow-up on it; return each follow-up's cached share.

    Requests go one at a time, so both replicas are idle at every decision and
    only the cache can tell them apart.
    """
    deploy(Arm(router, max_ongoing_requests=None), model_id, model_source)
    client = make_client()
    for _ in range(4):
        await time_to_first_token(client, "hello", model_id, prefix="Warm up the engine. ")
    fractions = []
    for _ in range(pairs):
        prefix = f"Session {uuid.uuid4()}. {SHARED_PREFIX}"  # unique from the first token
        await cached_fraction(client, prefix, "Summarize the instructions.", model_id)
        await asyncio.sleep(2)  # time for the seed's KV events to reach the router
        fractions.append(await cached_fraction(client, prefix, "Restate them.", model_id))
    teardown()
    return fractions


async def check_followups(router: str, pairs: int, model_id: str, model_source: str) -> bool:
    """Check 3: every follow-up must land on the replica that holds its seed's KV."""
    landed = {}
    for arm in ("default", router):
        fractions = await followup_fractions(arm, pairs, model_id, model_source)
        landed[arm] = sum(f > 0.5 for f in fractions)
        report(f"  {ROUTER_LABELS[arm]}: {landed[arm]}/{pairs} follow-ups reused their seed's "
               f"KV, mean cached share {statistics.fmean(fractions):6.1%}")
    if landed[router] == pairs and landed[router] > landed["default"]:
        report(f"PASS: the {ROUTER_LABELS[router]} sends each follow-up to its seed's replica.")
        return True
    report(f"FAIL: the {ROUTER_LABELS[router]} did not follow the cache on idle replicas.",
           error=True)
    return False


def app_status() -> str:
    """The Serve app's status as `serve status` would report it."""
    app = serve.status().applications.get(APP_NAME)
    return app.status if app else "MISSING"


async def bounded_request(client: AsyncOpenAI, i: int, model_id: str) -> float | str:
    """One streamed request with a hard deadline. Returns seconds, or the error on a miss."""
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            time_to_first_token(client, f"Question {i}: restate rule {i}.", model_id,
                                max_tokens=32),
            STALL_DEADLINE_S,
        )
    except Exception as exc:  # noqa: BLE001 - a timeout or any error is a miss
        return f"{type(exc).__name__}: {exc}"[:160]
    return time.perf_counter() - started


async def run_round(client: AsyncOpenAI, concurrency: int, model_id: str) -> RoundResult:
    """Fire `concurrency` requests that share SHARED_PREFIX, all at once."""
    times = await asyncio.gather(
        *(bounded_request(client, i, model_id) for i in range(concurrency)))
    done = [t for t in times if isinstance(t, float)]
    errors = [t for t in times if isinstance(t, str)]
    return RoundResult(concurrency, len(done), max(done, default=float("nan")), app_status(),
                       errors[0] if errors else "")


async def check_stall(arm: Arm, levels: list[int], rounds: int,
                      model_id: str, model_source: str) -> bool:
    """Check 2: every request at every concurrency level must finish in time."""
    deploy(arm, model_id, model_source)
    client = make_client()
    report(f"  config: {ROUTER_LABELS[arm.router]}, max_ongoing_requests="
           f"{arm.max_ongoing_requests or 'shipped default'}")
    stalled = False
    for level in levels:
        for r in range(rounds):
            res = await run_round(client, level, model_id)
            report(f"  {level} parallel, round {r + 1}: {res.completed}/{level} done, "
                   f"slowest {res.slowest_s:5.2f} s, app {res.app_status}"
                   + (f", first error {res.first_error}" if res.first_error else ""))
            stalled |= res.completed < level
    teardown()
    if stalled:
        report("FAIL: requests missed the deadline (stall). See app status per round.",
               error=True)
        return False
    report("PASS: no stall at any concurrency level.")
    return True


async def main(args: argparse.Namespace) -> int:
    """Run the selected checks and return the process exit code."""
    # Replica and proxy logs stay in their files instead of streaming here.
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    SERVE_LOGGING["log_level"] = args.serve_log_level
    gpus = int(ray.cluster_resources().get("GPU", 0))
    report(f"ray {ray.__version__} | cluster GPUs: {gpus}")
    if gpus < 2:
        print("SKIP: need 2 GPUs to route between two replicas", file=sys.stderr)
        return 0

    ok = True
    if not args.skip_reuse:
        report(f"[1] prefix reuse A/B, default vs {ROUTER_LABELS[args.router]}")
        ok &= await check_reuse(args.router, args.model_id, args.model_source)
    if not args.skip_stall:
        report("[2] concurrency stall check")
        arm = Arm(args.stall_router or args.router, args.max_ongoing_requests)
        ok &= await check_stall(arm, args.concurrency, args.rounds,
                                args.model_id, args.model_source)
    if args.followups:
        report(f"[3] seed then follow-up, {args.followups} pairs, one request at a time")
        ok &= await check_followups(args.router, args.followups,
                                    args.model_id, args.model_source)
    return 0 if ok else 1


def write_results(path: str | None) -> None:
    """Append the collected result lines to `path`, if one was given."""
    if path:
        with open(path, "a") as f:
            f.write("\n".join(RESULTS) + "\n")


def parse_args() -> argparse.Namespace:
    """Command-line options; defaults run both checks on the shipped config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--skip-reuse", action="store_true")
    parser.add_argument("--skip-stall", action="store_true")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--router", choices=["affinity", "kvaware"], default="affinity",
                        help="router under test (default: affinity, as shipped)")
    parser.add_argument("--stall-router", choices=sorted(ROUTER_LABELS), default=None,
                        help="router for the stall check (default: --router)")
    parser.add_argument("--followups", type=int, default=0,
                        help="seed/follow-up pairs for check 3 (default 0: skip)")
    parser.add_argument("--results-file", default=None)
    parser.add_argument("--serve-log-level", default="WARNING",
                        help="Serve log level for the deployments; INFO shows router setup")
    parser.add_argument("--max-ongoing-requests", type=int, default=None,
                        help="slots per replica for the stall check (default: shipped)")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    exit_code = asyncio.run(main(cli_args))
    write_results(cli_args.results_file)
    raise SystemExit(exit_code)

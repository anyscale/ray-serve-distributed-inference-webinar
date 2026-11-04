"""Closed-loop load against the Ranker, printing what Serve did with it.

Standard library only. Each user is a thread that sends one request at a time
over its own keep-alive connection, so users = requests in flight. Replica ids
and batch sizes come back in each response, so no cluster access is needed.

    SERVICE_URL=... SERVICE_TOKEN=... python load.py batch   # batching on vs off
    SERVICE_URL=... SERVICE_TOKEN=... python load.py scale   # 1 -> 6 replicas

Every 3 s it prints req/s, latency, mean batch size and the replicas that
answered. At the end of each stage it prints the batch-size distribution and
the requests per replica, over the stage's second half.
"""

import collections
import http.client
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass

SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:8000").rstrip("/")
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "")
NUM_ITEMS = 100_000  # matches app.py
CANDIDATES = 100  # candidates scored per request
REPORT_EVERY_S = 3


@dataclass(frozen=True)
class Stage:
    name: str
    seconds: int
    users: int
    query: str = ""  # "?batch=off" calls the model once per request


# Same 16 users both times: below target_ongoing_requests (20), so 1 replica.
BATCH = [Stage("batching on", 40, 16), Stage("batching off", 40, 16, "?batch=off")]
# Quiet, then 120 users (120 / 20 = 6 replicas), then quiet again.
SCALE = [Stage("quiet", 20, 4), Stage("burst", 120, 120), Stage("quiet", 60, 4)]


@dataclass(frozen=True)
class Sample:
    at: float
    latency_ms: float
    replica: str | None  # None when the request failed
    batch_size: int = 0


class LoadRun:
    """The current stage, plus every sample, shared by the user threads."""

    def __init__(self) -> None:
        self.stage = Stage("idle", 0, 0)
        self.samples: list[Sample] = []
        self.lock = threading.Lock()
        self.done = threading.Event()

    def record(self, sample: Sample) -> None:
        with self.lock:
            self.samples.append(sample)

    def window(self, since: float) -> list[Sample]:
        with self.lock:
            return [s for s in self.samples if s.at >= since]


class RankClient:
    """One persistent HTTP(S) connection, so each user pays the TLS handshake once."""

    def __init__(self) -> None:
        self.url = urllib.parse.urlsplit(SERVICE_URL)
        self.headers = {"content-type": "application/json"}
        if SERVICE_TOKEN:
            self.headers["authorization"] = f"Bearer {SERVICE_TOKEN}"
        self.conn: http.client.HTTPConnection | None = None

    def rank_once(self, query: str) -> dict | None:
        """Send one request; return the response JSON, or None on error."""
        body = {"user_id": random.randrange(1_000_000), "candidates": random.sample(range(NUM_ITEMS), CANDIDATES)}
        try:
            if self.conn is None:
                cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
                self.conn = cls(self.url.netloc, timeout=60)
            self.conn.request("POST", f"{self.url.path}/{query}", json.dumps(body), self.headers)
            response = self.conn.getresponse()
            data = response.read()
            return json.loads(data) if response.status == 200 else None
        except Exception:
            self.conn = None  # reconnect on the next request
            return None


def user(run: LoadRun, index: int) -> None:
    """Send requests back to back while this user is inside the stage's user count."""
    client = RankClient()
    while not run.done.is_set():
        stage = run.stage
        if index >= stage.users:
            time.sleep(0.05)
            continue
        start = time.perf_counter()
        result = client.rank_once(stage.query)
        latency_ms = (time.perf_counter() - start) * 1000
        if result:
            run.record(Sample(time.time(), latency_ms, result["replica"], result["batch_size"]))
        else:
            run.record(Sample(time.time(), latency_ms, None))


def stats(samples: list[Sample], seconds: float) -> str:
    """One line of throughput, latency, errors and mean batch size."""
    ok = [s for s in samples if s.replica]
    lat = sorted(s.latency_ms for s in ok) or [0.0]
    batch = statistics.mean(s.batch_size for s in ok) if ok else 0.0
    return (
        f"{len(ok) / seconds:7.1f} req/s  p50 {statistics.median(lat):5.0f} ms  "
        f"p95 {lat[int(0.95 * (len(lat) - 1))]:5.0f} ms  batch {batch:4.1f}  errors {len(samples) - len(ok)}"
    )


def summarize(run: LoadRun, stage: Stage, since: float) -> None:
    """Print the stage's second half: totals, batch sizes, requests per replica."""
    samples = run.window(since)
    seconds = time.time() - since
    ok = [s for s in samples if s.replica]
    sizes = collections.Counter(s.batch_size for s in ok)
    replicas = collections.Counter(s.replica for s in ok)
    print(f"== {stage.name}, {stage.users} users, last {seconds:.0f} s: {stats(samples, seconds)}")
    print("   batch size (share of requests): " + "  ".join(f"{k}:{100 * v / max(len(ok), 1):.0f}%" for k, v in sorted(sizes.items())))
    print("   requests per replica: " + "  ".join(f"{r} {n}" for r, n in replicas.most_common()))


def drive(stages: list[Stage]) -> None:
    run = LoadRun()
    for i in range(max(s.users for s in stages)):
        threading.Thread(target=user, args=(run, i), daemon=True).start()

    start = time.time()
    for stage in stages:
        run.stage = stage
        stage_start = time.time()
        while time.time() < stage_start + stage.seconds:
            since = time.time()
            time.sleep(REPORT_EVERY_S)
            window = run.window(since)
            replicas = len({s.replica for s in window if s.replica})
            print(
                f"t={time.time() - start:4.0f}s  {stage.name:<12} users {stage.users:3d}  "
                f"{stats(window, REPORT_EVERY_S)}  replicas {'#' * replicas} {replicas}",
                flush=True,
            )
        summarize(run, stage, stage_start + stage.seconds / 2)
    run.done.set()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("batch", "scale"):
        sys.exit("usage: python load.py batch|scale")
    drive(BATCH if mode == "batch" else SCALE)

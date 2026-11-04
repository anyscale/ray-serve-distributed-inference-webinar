"""Cost calculator for a self-hosted coding-agent service.

Compares the self-hosted GPU cost per developer-month with subscription seats
and per-token billing. The defaults are the planning case from
anyscale/llm-serving-for-coding-agents: part3-optimize/notes/COST-ESTIMATE.md
(Qwen3.6-27B on one RTX PRO 6000, list prices rounded in July 2026).

Sessions per GPU come from --sessions-per-gpu, or are read off a measured
frontier: with --frontier and --floor, the calculator takes the highest
concurrency whose per-user decode speed still meets the floor.

    python cost.py --developers 15 50 100 500
    python cost.py --frontier data/glm5_nvfp4_b200_cu13.csv --gpus-per-replica 4 --floor 30 \
        --gpu-price <B200 $/GPU-hour>

Frontier CSV: one row per concurrency level, with a `concurrency` column and
either `tok_s_per_user` or `tpot_ms` (per-user speed is then 1000 / tpot_ms).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

# Defaults from COST-ESTIMATE.md.
GPU_PRICE = 4.00  # $/GPU-hour, g7e.4xlarge on demand
SESSIONS_PER_GPU = 24.0  # KV-bound: ~1.7M cached tokens / ~70K tokens per turn
# Share of registered developers with a live session at the busy hour. 24 sessions
# per 50 developers reproduces the planning case of 50 developers per GPU.
PEAK_ACTIVE = 0.48
SEAT_PRICE = 200.0  # $/developer-month, a Claude Max 20x or Cursor Ultra seat
TOKEN_BILL = 800.0  # $/developer-month, a heavy user on per-token billing
HOURS_ALWAYS_ON = 730  # 24 h x 365 d / 12
HOURS_WORK = 210  # 10 h x 21 weekdays

BREAK_EVEN_SEARCH_LIMIT = 100_000  # developers


@dataclass(frozen=True)
class FrontierPoint:
    """One concurrency level of a sweep and the decode speed each user saw."""

    concurrency: int
    tok_s_per_user: float


@dataclass(frozen=True)
class Plan:
    """Capacity and price inputs; everything else is derived from these."""

    gpu_price: float  # $/GPU-hour
    hours_per_month: float
    sessions_per_gpu: float
    peak_active: float  # fraction of developers active at the busy hour
    gpus_per_replica: int = 1  # GPUs are bought in whole model replicas

    @property
    def gpu_month(self) -> float:
        return self.gpu_price * self.hours_per_month

    @property
    def devs_per_gpu(self) -> float:
        return self.sessions_per_gpu / self.peak_active

    def gpus_for(self, developers: int) -> int:
        per_replica = self.devs_per_gpu * self.gpus_per_replica
        # Rounding first keeps float noise (e.g. 24 / 0.48) from adding a replica.
        replicas = max(1, math.ceil(round(developers / per_replica, 9)))
        return replicas * self.gpus_per_replica

    def cost_per_dev(self, developers: int) -> float:
        return self.gpus_for(developers) * self.gpu_month / developers


def load_frontier(path: Path) -> list[FrontierPoint]:
    """Read a sweep CSV into frontier points (see the module docstring for columns)."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    points = []
    for row in rows:
        if row.get("tok_s_per_user"):
            speed = float(row["tok_s_per_user"])
        else:
            speed = 1000.0 / float(row["tpot_ms"])
        points.append(FrontierPoint(int(float(row["concurrency"])), speed))
    return points


def sessions_at_floor(
    points: list[FrontierPoint], floor: float, gpus: int
) -> tuple[float, FrontierPoint]:
    """Highest concurrency per GPU whose per-user speed meets `floor` tok/s."""
    eligible = [p for p in points if p.tok_s_per_user >= floor]
    if not eligible:
        fastest = max(p.tok_s_per_user for p in points)
        raise SystemExit(f"No frontier point reaches {floor} tok/s per user (best {fastest:.1f}).")
    best = max(eligible, key=lambda p: p.concurrency)
    return best.concurrency / gpus, best


def break_even(plan: Plan, price: float) -> int | None:
    """Smallest team for which self-hosting costs no more per developer than `price`."""
    for developers in range(1, BREAK_EVEN_SEARCH_LIMIT + 1):
        if plan.cost_per_dev(developers) <= price:
            return developers
    return None


def print_report(plan: Plan, developers: list[int], seat: float, tokens: float) -> None:
    """Per-team table, then break-even team sizes."""
    print(f"Developers per GPU: {plan.devs_per_gpu:.1f} "
          f"({plan.sessions_per_gpu:g} sessions / {plan.peak_active:g} peak active)")
    print(f"Cost per GPU: ${plan.gpu_month:,.0f}/month "
          f"(${plan.gpu_price:.2f}/h x {plan.hours_per_month:g} h)\n")
    header = f"{'devs':>6} {'GPUs':>5} {'self $/dev-mo':>14} {'vs seats $/mo':>14} {'vs tokens $/mo':>15}"
    print(header)
    print("-" * len(header))
    for n in developers:
        own = plan.cost_per_dev(n)
        print(f"{n:>6} {plan.gpus_for(n):>5} {own:>14,.2f} "
              f"{(seat - own) * n:>14,.0f} {(tokens - own) * n:>15,.0f}")
    print(f"\nSavings are per month; seats ${seat:,.0f}, token billing ${tokens:,.0f} per developer-month.")
    for label, price in (("seats", seat), ("token billing", tokens)):
        n = break_even(plan, price)
        if n:
            print(f"Self-hosting beats {label} from {n} developers.")
        else:
            print(f"Self-hosting never beats {label} at this capacity and price.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--developers", type=int, nargs="+", default=[15, 50, 100, 500])
    p.add_argument("--gpu-price", type=float, default=GPU_PRICE, help="$/GPU-hour")
    p.add_argument("--hours-per-month", type=float, default=HOURS_ALWAYS_ON,
                   help=f"GPU hours billed per month: {HOURS_ALWAYS_ON} always-on, {HOURS_WORK} work hours")
    p.add_argument("--sessions-per-gpu", type=float, default=SESSIONS_PER_GPU)
    p.add_argument("--peak-active", type=float, default=PEAK_ACTIVE,
                   help="fraction of developers with a live session at the busy hour")
    p.add_argument("--seat-price", type=float, default=SEAT_PRICE, help="$/developer-month")
    p.add_argument("--token-bill", type=float, default=TOKEN_BILL, help="$/developer-month")
    p.add_argument("--frontier", type=Path, help="sweep CSV; overrides --sessions-per-gpu")
    p.add_argument("--floor", type=float, default=30.0,
                   help="interactivity floor, decode tok/s per user (with --frontier)")
    p.add_argument("--gpus-per-replica", type=int, default=1,
                   help="GPUs in one model replica; a frontier's concurrency is divided by it")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sessions = args.sessions_per_gpu
    if args.frontier:
        points = load_frontier(args.frontier)
        sessions, point = sessions_at_floor(points, args.floor, args.gpus_per_replica)
        print(f"Frontier {args.frontier.name}: at >= {args.floor:g} tok/s per user, "
              f"concurrency {point.concurrency} ({point.tok_s_per_user:.1f} tok/s) "
              f"on {args.gpus_per_replica} GPU(s) = {sessions:g} sessions per GPU")
    plan = Plan(args.gpu_price, args.hours_per_month, sessions, args.peak_active,
                args.gpus_per_replica)
    print_report(plan, args.developers, args.seat_price, args.token_bill)


if __name__ == "__main__":
    main()

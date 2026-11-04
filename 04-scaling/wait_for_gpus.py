"""Block until the cluster has a node with N free GPUs.

A job's GPU workers start at zero nodes. A STRICT_PACK placement group of N
one-GPU bundles makes the autoscaler launch one node that holds all N; the
group is removed once it is placed, leaving the GPUs free for Serve.

    python wait_for_gpus.py --gpus 4 --timeout-s 2700
"""

import argparse
import sys
import time

import ray
from ray.util.placement_group import placement_group, remove_placement_group


def main(gpus: int, timeout_s: float) -> int:
    """Return 0 once N GPUs on one node are placed, 1 on timeout."""
    ray.init()
    started = time.monotonic()
    pg = placement_group([{"GPU": 1, "CPU": 1}] * gpus, strategy="STRICT_PACK")
    ready = pg.wait(timeout_seconds=timeout_s)
    remove_placement_group(pg)
    waited = time.monotonic() - started
    if not ready:
        print(f"FAIL: no node with {gpus} GPUs after {waited:.0f} s", file=sys.stderr)
        return 1
    print(f"{gpus} GPUs on one node after {waited:.0f} s; "
          f"cluster resources: {ray.cluster_resources()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=2700)
    args = parser.parse_args()
    raise SystemExit(main(args.gpus, args.timeout_s))

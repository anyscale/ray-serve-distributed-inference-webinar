"""Deploy kv-cache-offload/ at each tier in turn and run evict_and_return.py on it.

    python offload_run.py none cpu    # from checks/, on a GPU node
"""

import subprocess
import sys
from pathlib import Path

import ray
from ray import serve

APP_DIR = Path(__file__).resolve().parent.parent / "kv-cache-offload"
sys.path.insert(0, str(APP_DIR))
from app import OffloadArgs, build_app  # noqa: E402

# Same GPU cache cap and tier size as service.yaml.
TIERS = {
    "none": OffloadArgs(tier="none", gpu_kv_blocks=1100),
    "cpu": OffloadArgs(tier="cpu", cpu_gib=8, gpu_kv_blocks=1100),
}


def run(tier: str) -> int:
    """Deploy one tier, run the client against it, shut it down; return the client's code."""
    print(f"\n===== tier: {tier}", flush=True)
    serve.run(build_app(TIERS[tier]), name=tier, route_prefix="/", blocking=False,
              logging_config={"log_level": "WARNING"})
    code = subprocess.call([sys.executable, str(APP_DIR / "evict_and_return.py")])
    serve.delete(tier)
    return code


if __name__ == "__main__":
    ray.init(log_to_driver=False)
    sys.exit(max(run(t) for t in sys.argv[1:] or TIERS))

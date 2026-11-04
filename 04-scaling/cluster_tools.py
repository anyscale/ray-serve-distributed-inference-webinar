"""Run a small command on every alive node of the Ray cluster.

A job's entry point runs on the head node only, but Serve ingress and engine
replicas run on the GPU node too. Two subcommands:

    python cluster_tools.py pip --no-deps "ai-dynamo-runtime>=1.4.0"
    python cluster_tools.py grep "Dynamo SelectionService created" "No KVTokenTracker"

`pip` installs into the Python that Ray workers use, on every node.
`grep` counts matching lines in each node's Ray logs (Serve and worker files).
"""

import argparse
import glob
import os
import subprocess
import sys
from collections import Counter

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

LOG_GLOBS = ("/tmp/ray/session_latest/logs/serve/*.log",
             "/tmp/ray/session_latest/logs/worker-*.out",
             "/tmp/ray/session_latest/logs/worker-*.err")


def pip_install(args: list[str]) -> str:
    """Install packages with this node's Python; return pip's last output line."""
    done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                          capture_output=True, text=True)
    tail = (done.stdout + done.stderr).strip().splitlines()[-1:] or ["ok"]
    return f"rc={done.returncode} {tail[0]}"


def count_lines(patterns: list[str]) -> str:
    """Count log lines on this node that contain each pattern, and name the files.

    A Serve log file is named after its process (for example
    `replica_default_LLMRouter_<id>.log` or `proxy_<ip>.log`), so the file
    says which process logged the line.
    """
    counts = {p: Counter() for p in patterns}
    for path in (p for g in LOG_GLOBS for p in glob.glob(g)):
        with open(path, errors="replace") as f:
            for line in f:
                for pattern in patterns:
                    if pattern in line:
                        counts[pattern][os.path.basename(path)] += 1
    return "".join(f"\n    {p!r}: {sum(c.values())} {dict(c) if c else ''}"
                   for p, c in counts.items())


def on_every_node(fn, *args) -> None:
    """Run `fn(*args)` once on each alive node and print one line per node."""
    remote_fn = ray.remote(num_cpus=0)(fn)
    nodes = [n for n in ray.nodes() if n["Alive"]]
    refs = [remote_fn.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        n["NodeID"], soft=False)).remote(*args) for n in nodes]
    for node, result in zip(nodes, ray.get(refs)):
        gpus = int(node["Resources"].get("GPU", 0))
        print(f"  node {node['NodeManagerAddress']} ({gpus} GPU): {result}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["pip", "grep"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    ray.init(log_to_driver=False)
    on_every_node(pip_install if cli.command == "pip" else count_lines, cli.args)

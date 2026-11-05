"""The cluster's Mooncake master, and the callback that points each engine at it.

The master holds the pool's metadata: which node holds which KV block. The
blocks themselves live in host memory that each replica contributes. The
master runs as a one-replica Serve deployment pinned to the head node, so
every engine finds it at the head node's address.

vLLM reads its Mooncake settings from the JSON file named by
MOONCAKE_CONFIG_PATH. `PointAtMooncakeMaster`, a Ray Serve LLM initialization
callback, writes that file in each replica before its engine starts: the
settings from app.py plus the master's address.
"""

import asyncio
import json
import os
import socket
import subprocess
import tempfile
import time

import ray
from ray import serve
# Ray Serve LLM's callback API; see its "Deployment initialization" guide.
from ray.llm._internal.common.callbacks.base import CallbackBase

MASTER_PORT = 50051
CONFIG_PATH = "/tmp/mooncake_config.json"


def wait_for_port(host: str, port: int, timeout_s: float = 300) -> None:
    """Block until something accepts TCP connections on host:port."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no Mooncake master at {host}:{port}") from None
            time.sleep(2)


@serve.deployment(
    num_replicas=1,
    # The head node's built-in resource pins the master there; it needs no CPU.
    ray_actor_options=dict(num_cpus=0, resources={"node:__internal_head__": 0.001}),
)
class MooncakeMaster:
    """Runs `mooncake_master` for as long as the replica lives."""

    def __init__(self):
        self.proc = subprocess.Popen(["mooncake_master", f"--port={MASTER_PORT}"])
        wait_for_port("127.0.0.1", MASTER_PORT, timeout_s=60)

    def check_health(self) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError(f"mooncake_master exited with code {self.proc.returncode}")

    async def __call__(self, request) -> dict:
        return {"master": f"{ray.util.get_node_ip_address()}:{MASTER_PORT}",
                "running": self.proc.poll() is None}

    def __del__(self):
        self.proc.terminate()


def build_master_app(args: dict) -> serve.Application:
    """The master application; it takes no arguments."""
    return MooncakeMaster.bind()


class PointAtMooncakeMaster(CallbackBase):
    """Write this replica's Mooncake config, with the master's address, before vLLM starts.

    `callback_kwargs={"settings": {...}}` holds the rest of the config. The file
    is written on the replica's node, where its GPU workers run too.
    """

    async def on_before_node_init(self) -> None:
        head_ip = ray.get_runtime_context().gcs_address.rsplit(":", 1)[0]
        await asyncio.to_thread(wait_for_port, head_ip, MASTER_PORT)
        config = {**self.kwargs["settings"], "master_server_address": f"{head_ip}:{MASTER_PORT}"}
        # Two replicas on one node write the same file; the rename keeps each write whole.
        with tempfile.NamedTemporaryFile("w", dir="/tmp", delete=False, suffix=".json") as f:
            json.dump(config, f)
        os.replace(f.name, CONFIG_PATH)
        # The engine process inherits this; vLLM copies MOONCAKE_* on to its Ray workers.
        os.environ["MOONCAKE_CONFIG_PATH"] = CONFIG_PATH

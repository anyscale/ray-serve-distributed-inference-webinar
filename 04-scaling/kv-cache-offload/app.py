"""KV cache offloading: keep evicted prefixes in CPU memory or on local disk.

When the GPU KV cache is full, vLLM evicts the least recently used prefix
blocks. A session that comes back after a pause then pays its whole prefill
again. An offload tier keeps a copy of each finished block below the GPU and
loads it back on the next hit, which is cheaper than recomputing it.

Three settings of one builder:
  none  no offload tier: the baseline, where an evicted prefix is recomputed.
  cpu   vLLM's native offloading: `kv_offloading_backend="native"` and
        `kv_offloading_size` GiB of pinned host memory per replica. KVAwareRouter
        can see this tier and counts its blocks at a discount.
  disk  LMCache as the KV connector: a CPU tier plus a local-disk tier below it,
        sized with LMCACHE_* environment variables. The router cannot see it.

`gpu_kv_blocks` shrinks the GPU cache on purpose, so a few long prompts are
enough to evict one; leave it unset in production.

    anyscale service deploy -f service.yaml --working-dir .        # tier: cpu
    anyscale service deploy -f service-disk.yaml --working-dir .   # tier: disk
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field
from ray.serve import Application
from ray.serve.llm import LLMConfig, build_openai_app

# Anyscale mounts each node's local NVMe at /mnt/local_storage.
DISK_PATH = "file:///mnt/local_storage/lmcache/"


class OffloadArgs(BaseModel):
    """Builder arguments, set under `args:` in service.yaml."""

    tier: Literal["none", "cpu", "disk"] = "cpu"
    model_id: str = Field(default="qwen-0.5b")
    model_source: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    cpu_gib: float = Field(default=8, gt=0)  # host memory for the CPU tier, per replica
    disk_gib: float = Field(default=50, gt=0)  # local disk for the disk tier, per replica
    gpu_kv_blocks: Optional[int] = Field(default=None, ge=16)  # 16 tokens per block


def offload_engine_kwargs(args: OffloadArgs) -> dict:
    """vLLM settings for the chosen tier; both build on automatic prefix caching."""
    kwargs = dict(max_model_len=16384, enable_prefix_caching=True, enable_prompt_tokens_details=True)
    if args.gpu_kv_blocks:
        kwargs["num_gpu_blocks_override"] = args.gpu_kv_blocks
    if args.tier == "cpu":
        kwargs.update(kv_offloading_backend="native", kv_offloading_size=args.cpu_gib)
    elif args.tier == "disk":
        kwargs["kv_transfer_config"] = dict(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    return kwargs


def lmcache_env(args: OffloadArgs) -> dict:
    """LMCache reads its tiers from the environment: CPU first, then disk below it."""
    if args.tier != "disk":
        return {}
    return {
        "LMCACHE_CHUNK_SIZE": "256",  # tokens per stored chunk
        "LMCACHE_LOCAL_CPU": "True",  # the disk tier sits below a CPU tier
        "LMCACHE_MAX_LOCAL_CPU_SIZE": str(args.cpu_gib),  # GB
        "LMCACHE_LOCAL_DISK": DISK_PATH,
        "LMCACHE_MAX_LOCAL_DISK_SIZE": str(args.disk_gib),  # GB
    }


def build_app(args: OffloadArgs) -> Application:
    """One replica with the chosen offload tier, if any, under its GPU KV cache."""
    config = LLMConfig(
        model_loading_config=dict(model_id=args.model_id, model_source=args.model_source),
        accelerator_type="L4",
        deployment_config=dict(num_replicas=1),
        runtime_env=dict(env_vars=lmcache_env(args)),
        engine_kwargs=offload_engine_kwargs(args),
    )
    return build_openai_app({"llm_configs": [config]})

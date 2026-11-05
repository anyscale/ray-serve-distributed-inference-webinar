"""A KV cache pool that every replica shares: vLLM's MooncakeStoreConnector.

Each replica adds `segment_gib` of host memory to one Mooncake pool, and
writes every finished prefix block into it. Before prefilling a prompt, a
replica looks up the prompt's blocks in the pool and copies the hits into its
GPU cache. So a prefix computed once, by any replica, can be loaded by all.

Keys are vLLM's prefix block hashes, which every replica must compute the
same way: PYTHONHASHSEED is fixed for that reason. The master that tracks the
blocks is in mooncake_pool.py.

    anyscale service deploy -f service.yaml --working-dir .
"""

from typing import Literal

from pydantic import BaseModel, Field
from ray.serve import Application
from ray.serve.llm import LLMConfig, build_openai_app


class PoolArgs(BaseModel):
    """Builder arguments, set under `args:` in service.yaml."""

    model_id: str = Field(default="qwen2.5-7b")
    model_source: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    shared_pool: bool = True  # false: the same replica with no pool, as the baseline
    segment_gib: int = Field(default=8, gt=0)  # host memory this replica adds to the pool
    protocol: Literal["tcp", "rdma"] = "tcp"  # rdma needs RDMA NICs, such as EFA or InfiniBand
    rdma_device: str = ""  # such as "mlx5_0"; empty lets Mooncake choose


def mooncake_settings(args: PoolArgs) -> dict:
    """vLLM's Mooncake config, less the master's address, which the callback adds."""
    return dict(
        mode="embedded",  # the pool lives in the vLLM processes, no separate store
        metadata_server="P2PHANDSHAKE",  # peers exchange addresses directly
        global_segment_size=f"{args.segment_gib}GB",
        local_buffer_size="2GB",  # staging buffer for this replica's own transfers
        protocol=args.protocol,
        device_name=args.rdma_device,
    )


def pool_runtime_env(args: PoolArgs) -> dict:
    """Environment for the replica; vLLM copies these on to its Ray workers."""
    env_vars = {"PYTHONHASHSEED": "0"}
    if args.shared_pool:
        env_vars["VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY"] = "MOONCAKE_"  # carries MOONCAKE_CONFIG_PATH
    return dict(env_vars=env_vars)


def build_app(args: PoolArgs) -> Application:
    """One replica on one L4, in the shared pool unless `shared_pool` is false."""
    engine_kwargs = dict(max_model_len=16384, enable_prefix_caching=True, enable_prompt_tokens_details=True)
    pool = {}
    if args.shared_pool:
        engine_kwargs["kv_transfer_config"] = dict(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_both",  # writes its blocks to the pool and reads others'
            # Replicas share blocks only when they share a prefix; one per model.
            kv_connector_extra_config={"cache_prefix": args.model_id},
        )
        # Writes the Mooncake config, with the master's address, before vLLM starts.
        pool["callback_config"] = dict(callback_class="mooncake_pool.PointAtMooncakeMaster",
                                       callback_kwargs={"settings": mooncake_settings(args)})
    config = LLMConfig(
        model_loading_config=dict(model_id=args.model_id, model_source=args.model_source),
        accelerator_type="L4",
        deployment_config=dict(num_replicas=1),
        runtime_env=pool_runtime_env(args),
        engine_kwargs=engine_kwargs,
        **pool,
    )
    return build_openai_app({"llm_configs": [config]})

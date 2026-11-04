"""KV-aware routing for one model across several replicas.

`KVAwareRouter` is a Ray Serve LLM request router (alpha). Each replica
publishes vLLM's KV-cache events; for every request the router tokenizes the
prompt and scores each replica:

    score = S x (prompt blocks - blocks already cached there) + work already running there

The lowest score wins, and the router books the new request's work on that
replica at once. NVIDIA Dynamo's selection service does the scoring
in-process; Ray handles discovery, dispatch and the KV-event transport.

S is `DYN_ROUTER_PREFILL_LOAD_SCALE` (Dynamo's default is 1.0). Raise it for
long, repeated prompts such as coding agents, so a cache hit outweighs a
slightly longer queue.

    anyscale service deploy -f service.yaml --working-dir .
"""

from pydantic import BaseModel, Field
from ray.serve import Application
from ray.serve.config import RequestRouterConfig
from ray.serve.llm import LLMConfig, build_openai_app

KV_AWARE_ROUTER = "ray.serve.llm.request_router.KVAwareRouter"


class KVRoutingArgs(BaseModel):
    """Builder arguments, set under `args:` in service.yaml."""

    model_id: str = Field(default="qwen3-8b")
    model_source: str = Field(default="Qwen/Qwen3-8B")
    replicas: int = Field(default=4, ge=2)  # routing needs replicas to choose between
    prefill_load_scale: float = Field(default=4.0, gt=0)  # S in the score above


def kv_routed_config(args: KVRoutingArgs) -> LLMConfig:
    """One model, a fixed pool of replicas, and the KV-aware router in front."""
    return LLMConfig(
        model_loading_config=dict(model_id=args.model_id, model_source=args.model_source),
        accelerator_type="L40S",
        deployment_config=dict(
            autoscaling_config=dict(min_replicas=args.replicas, max_replicas=args.replicas),
            request_router_config=RequestRouterConfig(request_router_class=KV_AWARE_ROUTER),
        ),
        # Dynamo reads DYN_* from the environment and ignores misspelled names silently.
        runtime_env=dict(env_vars={"DYN_ROUTER_PREFILL_LOAD_SCALE": str(args.prefill_load_scale)}),
        engine_kwargs=dict(
            max_model_len=32768,
            enable_prefix_caching=True,  # the cache the router scores
        ),
    )


def build_app(args: KVRoutingArgs) -> Application:
    """The OpenAI-compatible app; direct streaming serves one model per app."""
    return build_openai_app({"llm_configs": [kv_routed_config(args)]})

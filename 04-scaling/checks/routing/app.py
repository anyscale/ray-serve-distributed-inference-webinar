"""Prefix-cache-affinity routing for one model across two replicas.

The default request router (power of two choices) balances on queue length and
ignores what each replica has cached. `PrefixCacheAffinityRouter` sends a
request to the replica whose prefix tree best matches its prompt, and falls
back to power of two once replica queues differ by more than
`imbalanced_threshold`, so a hot prefix cannot swamp one replica.

Run on a workspace with 2 GPUs:
    serve run app:build_app
    python verify.py
"""


from pydantic import BaseModel, Field
from ray.serve import Application
from ray.serve.config import RequestRouterConfig
from ray.serve.llm import LLMConfig, build_openai_app

DEFAULT_MODEL_ID = "qwen-0.5b"
DEFAULT_MODEL_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"
AFFINITY_ROUTER = "ray.serve.llm.request_router.PrefixCacheAffinityRouter"


def routed_config(model_id: str, model_source: str, min_replicas: int) -> LLMConfig:
    """One model with prefix-cache-affinity routing across its replicas.

    The router is named by import string so the config stays serializable for
    an Anyscale Service. `imbalanced_threshold` defaults to infinity (affinity
    always wins); 20 queued requests caps how lopsided a hot prefix can get.
    """
    return LLMConfig(
        model_loading_config=dict(model_id=model_id, model_source=model_source),
        deployment_config=dict(
            autoscaling_config=dict(
                min_replicas=min_replicas, max_replicas=4, target_ongoing_requests=8
            ),
            request_router_config=RequestRouterConfig(
                request_router_class=AFFINITY_ROUTER,
                request_router_kwargs={"imbalanced_threshold": 20},
            ),
        ),
        engine_kwargs=dict(
            max_model_len=4096,
            enforce_eager=True,  # skips CUDA graph capture, so replicas start faster
            enable_prefix_caching=True,  # the cache the router steers toward
        ),
    )


class RoutingArgs(BaseModel):
    """Builder arguments, settable from service.yaml or `serve run` CLI args."""

    model_id: str = Field(default=DEFAULT_MODEL_ID)
    model_source: str = Field(default=DEFAULT_MODEL_SOURCE)
    min_replicas: int = Field(default=2, ge=1)


def build_app(args: RoutingArgs) -> Application:
    """Build the OpenAI-compatible app with affinity routing."""
    cfg = routed_config(args.model_id, args.model_source, args.min_replicas)
    return build_openai_app({"llm_configs": [cfg]})

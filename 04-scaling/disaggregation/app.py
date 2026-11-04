"""Prefill/decode disaggregation on a small model, with a collocated baseline.

Prefill reads the whole prompt in one compute-bound pass; decode writes one token
per step and is bound by memory bandwidth. In one engine, a long prompt's
prefill shares steps with every decoding stream and stretches their inter-token
latency. `build_pd_openai_app` puts the two phases in separate replica pools.
Ray Serve LLM builds and scales the pools; vLLM's NIXL connector moves each
request's KV from the prefill replica to the decode replica.

Two builders, same model and the same GPU count (2):
    serve run app:build_app             # 1 prefill + 1 decode replica
    serve run app:build_collocated_app  # 2 ordinary replicas, the baseline
"""


from pydantic import BaseModel, Field
from ray.serve import Application
from ray.serve.llm import LLMConfig, build_openai_app, build_pd_openai_app

DEFAULT_MODEL_ID = "qwen-0.5b"
DEFAULT_MODEL_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"

# Both phases need this. kv_both lets one replica produce or consume KV.
NIXL_TRANSFER = dict(kv_connector="NixlConnector", kv_role="kv_both")


def phase_config(model_id: str, model_source: str, replicas: int,
                 kv_transfer: bool) -> LLMConfig:
    """One pool's config. P/D pools must share the model_id and set kv_transfer_config."""
    engine_kwargs = dict(
        max_model_len=4096,
        enforce_eager=True,  # skips CUDA graph capture, so replicas start faster
        enable_prefix_caching=False,  # every injected prompt pays its full prefill
    )
    if kv_transfer:
        engine_kwargs["kv_transfer_config"] = NIXL_TRANSFER
    return LLMConfig(
        model_loading_config=dict(model_id=model_id, model_source=model_source),
        deployment_config=dict(num_replicas=replicas),
        engine_kwargs=engine_kwargs,
        # Stream every token as it is made. By default Serve LLM groups stream
        # chunks every 50 ms, which would hide the engine's inter-token gaps.
        experimental_configs=dict(stream_batching_interval_ms=0),
    )


class PDArgs(BaseModel):
    """Builder arguments, settable from service.yaml or `serve run` CLI args."""

    model_id: str = Field(default=DEFAULT_MODEL_ID)
    model_source: str = Field(default=DEFAULT_MODEL_SOURCE)
    prefill_replicas: int = Field(default=1, ge=1)
    decode_replicas: int = Field(default=1, ge=1)


def build_app(args: PDArgs) -> Application:
    """Disaggregated app: the ingress sends each request to prefill, then decode."""
    return build_pd_openai_app(dict(
        prefill_config=phase_config(args.model_id, args.model_source,
                                    args.prefill_replicas, kv_transfer=True),
        decode_config=phase_config(args.model_id, args.model_source,
                                   args.decode_replicas, kv_transfer=True),
    ))


def build_collocated_app(args: PDArgs) -> Application:
    """Baseline with the same GPU count: every replica does both phases."""
    replicas = args.prefill_replicas + args.decode_replicas
    cfg = phase_config(args.model_id, args.model_source, replicas, kv_transfer=False)
    return build_openai_app({"llm_configs": [cfg]})

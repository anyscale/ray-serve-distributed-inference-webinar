"""Engine-knob variants of one Ray Serve LLM deployment, for the frontier demo.

The two variants differ in one vLLM setting, `enforce_eager`:

  cuda-graphs  vLLM's default. The model is compiled with torch.compile, and each
               decode step replays a captured CUDA graph, so the CPU launches one
               graph instead of hundreds of small kernels.
  eager        `enforce_eager=True`. vLLM 0.27 then turns off both CUDA graphs and
               torch.compile, and every kernel is launched from Python. That
               launch cost is fixed per step, so it hurts most when steps are short.

Both serve through the dev ingress, which adds POST /reset_prefix_cache next to
the OpenAI routes so `prefix_cold_warm.py` can clear the cache on demand.

Run one variant locally:
    serve run knobs:build_app variant=eager
    serve run knobs:build_app variant=eager model=qwen-1.5b
"""

from ray.llm._internal.serve.core.ingress.dev_ingress import build_dev_openai_app
from ray.serve.llm import LLMConfig

# Client-facing model id -> Hugging Face weights. The demo uses the 0.5B model:
# its decode steps are short, so the knob's effect is easy to see.
MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

# The knob under test, by variant name. Everything else stays identical.
VARIANTS = {
    "cuda-graphs": {},
    "eager": {"enforce_eager": True},
}


def llm_config(variant: str, model_id: str) -> LLMConfig:
    """Return the LLMConfig for one variant of one model on one L4 GPU."""
    return LLMConfig(
        model_loading_config=dict(model_id=model_id, model_source=MODELS[model_id]),
        accelerator_type="L4",
        deployment_config=dict(num_replicas=1),
        engine_kwargs=dict(
            enable_prefix_caching=True,
            # 32K tokens: room for the ~31K-token prompt of the cold/warm demo.
            max_model_len=32768,
            **VARIANTS[variant],
        ),
    )


def build_app(args: dict):
    """Serve builder: `variant` picks a key of VARIANTS, `model` a key of MODELS."""
    config = llm_config(args.get("variant", "cuda-graphs"), args.get("model", "qwen-0.5b"))
    return build_dev_openai_app({"llm_configs": [config]})

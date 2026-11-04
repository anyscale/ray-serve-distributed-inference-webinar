"""Engine settings shared by the agent service, the composition app and the batch job.

The defaults describe Qwen3.6-27B (NVFP4 weights) on one RTX PRO 6000 96 GB.
To run the same settings with a smaller model on a smaller GPU, set the
LLM_MODEL_ID, LLM_MODEL_SOURCE and LLM_MAX_MODEL_LEN environment variables.
"""

import os

MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3.6-27b")
MODEL_SOURCE = os.environ.get("LLM_MODEL_SOURCE", "nvidia/Qwen3.6-27B-NVFP4")

# vLLM engine arguments. Ray Serve LLM and Ray Data LLM both pass these
# straight to the engine, so online and offline run with identical settings.
ENGINE_KWARGS = dict(
    # 256K tokens, the model's native context window.
    max_model_len=int(os.environ.get("LLM_MAX_MODEL_LEN", "262144")),
    # FP8 halves KV memory versus bf16, which is what lets 256K fit on 96 GB.
    kv_cache_dtype="fp8",
    # Agents resend their whole history every turn, so cached prefixes pay off.
    enable_prefix_caching=True,
    # Prefill token budget per engine step; long prompts arrive in chunks.
    max_num_batched_tokens=8192,
    max_num_seqs=32,
    gpu_memory_utilization=0.9,
)

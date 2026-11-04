"""Qwen3.6-27B behind one endpoint that speaks to Claude Code, Codex and Cursor.

Adapted from anyscale/llm-serving-for-coding-agents (part1-deploy-naive, part3-optimize).

Direct streaming mounts vLLM's own HTTP app behind HAProxy, so this single
deployment answers /v1/chat/completions, /v1/messages and /v1/responses.
It is switched on by two service-level env vars in service-agents.yaml, and
it accepts exactly one LLMConfig.

Run in a workspace (with both env vars set on the workspace):
    serve run agents_app:app
"""

from ray.serve.llm import LLMConfig, build_openai_app

from engine import ENGINE_KWARGS, MODEL_ID, MODEL_SOURCE

llm_config = LLMConfig(
    model_loading_config=dict(model_id=MODEL_ID, model_source=MODEL_SOURCE),
    deployment_config=dict(autoscaling_config=dict(min_replicas=1, max_replicas=1)),
    # accelerator_type is left out: the LLMConfig enum has no RTX PRO 6000
    # entry, and the service's compute config already pins the g7e node.
    engine_kwargs=dict(
        ENGINE_KWARGS,
        # Agents send tools and read back reasoning, so parse both server-side.
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        enable_auto_tool_choice=True,
        # MTP drafts 3 tokens per step with the model's own prediction head.
        # It speeds up a single stream most and helps less as batches grow.
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})

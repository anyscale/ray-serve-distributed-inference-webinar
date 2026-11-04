# Production-ready Distributed Inference with Ray Serve

Code and configs from the Anyscale webinar of 5 November 2026.

One story runs through the session: a coding agent served from your own GPUs.
It starts with the agent at work, takes the serving stack apart one layer at a
time, and ends with what it costs. Each folder is one part of the talk, stands
on its own, and has its own README with the commands to run it.

## What is in each folder

| Folder | Part of the talk | What it shows | Hardware | Start with |
|---|---|---|---|---|
| [00-claude-code-on-glm](00-claude-code-on-glm/) | Claude Code on GLM | Claude Code working against an open-weight model served by Ray Serve LLM, through `/v1/messages` | Any Anthropic-compatible endpoint | `./claude-glm.sh --check` |
| [01-serve-foundations](01-serve-foundations/) | Ray Serve foundations | Routing, autoscaling and `@serve.batch` on one plain Python deployment | CPU only | `python load.py batch` |
| [02-serve-llm](02-serve-llm/) | Ray Serve LLM | One `LLMConfig` serving Claude Code, Codex and Cursor; an LLM inside a larger app; batch inference with the same engine settings | 1× RTX PRO 6000 or 1× L40S; 1× L4 for the smaller demos | `python check_agents.py` |
| [03-engine](03-engine/) | Inside the engine | Prefix cache cold vs warm, a concurrency sweep that draws the throughput–interactivity frontier, and one engine knob that moves it | 1× L4 per service | `python sweep.py` |
| [04-scaling](04-scaling/) | Scaling for agents | KV-aware routing and prefill/decode disaggregation as `app.py` + `service.yaml`, and reference configs for wide expert parallelism, decode context parallelism and a CPU KV tier | 4× L40S for routing, one 4× L4 node for P/D; the reference configs need B200 or H200 nodes | `anyscale service deploy -f kv-aware-routing/service.yaml` |
| [05-cost](05-cost/) | What it costs | A calculator that turns a frontier into a cost per developer-month | None | `python cost.py` |
| [explainers](explainers/) | Splitting a model across GPUs | An interactive primer on decode context parallelism; open the HTML file in a browser | None | `open explainers/dcp-primer.html` |

The runnable demos use small models (Qwen2.5-0.5B up to Qwen3.6-27B) so they fit
on one node. Files that say **Shown, not run** at the top are production shapes
for larger models and need the hardware they name.

## Running it

Every service and job here is an Anyscale config (`service.yaml`, `job.yaml`,
`workspace.yaml`), deployed with the [Anyscale CLI](https://docs.anyscale.com/reference/quickstart-cli):

```bash
pip install anyscale && anyscale login
cd 02-serve-llm
anyscale service deploy -f service-agents.yaml --working-dir .
```

The Python apps are plain Ray Serve, so they also run on any Ray cluster with
`serve run`. The images used are `anyscale/ray-llm:2.59.0-py312-cu130` for the
LLM demos and `anyscale/ray:2.59.0-py312` for the CPU demo.

Endpoints and tokens come from environment variables at run time. Nothing in
this repository holds a URL or a token. Terminate each service when you are
done; every README ends with the command.

## Checks that need no GPU

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pyyaml
find . -name '*.py' -not -path './.venv/*' -exec python -m py_compile {} +
python -c "import pathlib, yaml; [yaml.safe_load(p.read_text()) for p in pathlib.Path('.').rglob('*.yaml') if '.venv' not in p.parts]"
(cd 00-claude-code-on-glm/demo-bug && python -m unittest)   # one test fails on purpose: the agent fixes it
python 05-cost/cost.py
```

## Further reading

**Ray Serve LLM**
- [Ray Serve LLM docs](https://docs.ray.io/en/latest/serve/llm/index.html)
- [Direct streaming: `/v1/messages` and `/v1/responses`](https://docs.ray.io/en/latest/serve/llm/user-guides/direct-streaming.html)
- [Prefix-aware routing](https://docs.ray.io/en/latest/serve/llm/user-guides/prefix-aware-routing.html)
- [Blog: KV-cache and token-load-aware routing](https://www.anyscale.com/blog/llm-kv-token-aware-routing)

**The engine**
- [vLLM: automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- [vLLM: quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- [vLLM: MTP speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/)

**Run a coding agent on your own GPUs, end to end**
- [anyscale/llm-serving-for-coding-agents](https://github.com/anyscale/llm-serving-for-coding-agents): deploy, connect clients, optimize, and cost
- [Knob-by-knob measurements](https://github.com/anyscale/llm-serving-for-coding-agents/blob/main/part3-optimize/notes/BENCHMARKS.md)
- [Cost estimate](https://github.com/anyscale/llm-serving-for-coding-agents/blob/main/part3-optimize/notes/COST-ESTIMATE.md)

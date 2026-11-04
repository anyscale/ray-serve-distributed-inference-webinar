# 02 · Ray Serve LLM

Three demos share one module of engine settings, `engine.py`:

- **a) One deployment, three agent APIs.** Qwen3.6-27B on one RTX PRO 6000.
  Claude Code, Codex and Cursor all talk to it through direct streaming.
- **b) It's just Python.** A plain triage deployment answers what it can and calls the LLM for the rest.
- **c) Batch with the same engine settings.** A `ray.data.llm` job imports the same `ENGINE_KWARGS`.

The agent demo is adapted from [anyscale/llm-serving-for-coding-agents](https://github.com/anyscale/llm-serving-for-coding-agents).

| File | What it is |
|---|---|
| `engine.py` | Model id, model source and vLLM engine arguments, shared by all three demos. |
| `agents_app.py` | One `LLMConfig` passed to `build_openai_app`, adding the tool-call and reasoning parsers and MTP. |
| `service-agents.yaml` | The service, with the two direct-streaming env vars at service level and one RTX PRO 6000 (g7e). |
| `service-agents-l40s.yaml` | Fallback when g7e has no capacity: the same app with FP8 weights on one L40S and 128K context. This is the configuration that was verified. |
| `check_agents.py` | Calls `/v1/models`, chat, streamed chat (prints TTFT), `/v1/messages`, `/v1/responses` and a tool call. |
| `clients/claude-code.sh`, `clients/codex.sh` | Launch each agent against the service. |
| `composition.py`, `triage.py`, `service-composition.yaml` | Demo b. |
| `batch_infer.py`, `job-batch.yaml` | Demo c. |

The runnable `service-composition.yaml` and `job-batch.yaml` use `Qwen/Qwen3-4B-Instruct-2507` on one L4.
They override the model through the `LLM_*` env vars and keep every other engine setting.
To run them on Qwen3.6-27B instead, delete those env vars and switch the worker to `g7e.4xlarge`.

## Deploy

On L40S, the 27B service took 11 minutes from deploy to RUNNING. Most of that was the weight download and CUDA-graph capture.
The first request after that took 38 s; later ones answered in under half a second.

**Check that a node actually launches.**
On 31 Oct, AWS had no g7e capacity in any us-west-2 zone, on-demand or spot.
The service then sits in STARTING with no error on the command line, and the reason shows only in the service's events (`InsufficientInstanceCapacity`).
If that happens, terminate the service and deploy `service-agents-l40s.yaml` instead.

```bash
cd 02-serve-llm
anyscale service deploy -f service-agents.yaml --working-dir .
anyscale service wait -n webinar-1105-qwen-agents --state RUNNING --timeout-s 3600
# fallback: anyscale service terminate -n webinar-1105-qwen-agents
#           anyscale service deploy -f service-agents-l40s.yaml --working-dir .
```

Copy the URL and token from **Services → webinar-1105-qwen-agents → Query**:

```bash
export ANYSCALE_BASE_URL=https://<service-host>/v1   # note the /v1
export ANYSCALE_API_KEY=<service bearer token>        # keep it off screen
python check_agents.py                                # warms the service; every line should say HTTP 200
```

Expected output, from the verified L40S run:

```
GET /v1/models                     HTTP 200   0.10s  'qwen3.6-27b'
POST /v1/chat/completions          HTTP 200   0.43s  'print(input()[::-1])'
POST /v1/chat/completions stream   HTTP 200   0.37s  'TTFT 170 ms | ```python\ns[::-1]\n```'
POST /v1/messages (Anthropic)      HTTP 200   0.37s  '```python\ns[::-1]\n```'
POST /v1/responses (Codex)         HTTP 200   0.38s  '```python\ns[::-1]\n```'
POST /v1/chat/completions tools    HTTP 200   0.90s  'run_shell({"cmd": "find . -name \\"*.py\\" -type f"})'
```

Sign in to all three clients before you start.

## a) One deployment, three agent APIs

Show `agents_app.py` and the `env_vars:` block in `service-agents.yaml`, then send a first query:

```bash
python check_agents.py
```

### Claude Code → `/v1/messages`

```bash
./clients/claude-code.sh
```

The script sets these variables. Claude Code appends `/v1/messages` itself, so the base URL has no `/v1`.

```bash
export ANTHROPIC_BASE_URL=https://<service-host>
export ANTHROPIC_AUTH_TOKEN=<service bearer token>
export ANTHROPIC_MODEL=qwen3.6-27b
```

### Codex → `/v1/responses`

```bash
./clients/codex.sh
```

The same provider as a `~/.codex/config.toml` entry:

```toml
model = "qwen3.6-27b"
model_provider = "anyscale"

[model_providers.anyscale]
name = "Anyscale"
base_url = "https://<service-host>/v1"
env_key = "ANYSCALE_API_KEY"
wire_api = "responses"
requires_openai_auth = false
```

### Cursor → `/v1/chat/completions`

Cursor sends requests from its own cloud, so it needs the public service URL, not `localhost`.
In **Cursor Settings → Models → API Keys**:

1. Turn on **Override OpenAI Base URL** and set it to `https://<service-host>/v1`.
2. Set **OpenAI API Key** to the service bearer token.
3. Add a custom model named exactly `qwen3.6-27b`, enable it and disable the others.
4. Click **Verify**, then pick `qwen3.6-27b` in chat.

## b) It's just Python

`composition.py` ends with the line that wires it together:

```python
app = Assistant.bind(Triage.bind(), build_llm_deployment(llm_config))
```

To run it, deploy the service and call it as follows:

```bash
anyscale service deploy -f service-composition.yaml --working-dir .
curl -s "$URL/ask" -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"text": "what are your support hours?"}'          # answered by triage, used_llm false
curl -s "$URL/ask" -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"text": "Explain a KV cache in two sentences."}'  # answered by the LLM, used_llm true
curl -N "$URL/ask/stream" -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"text": "Explain a KV cache in two sentences."}'  # server-sent events
```

## c) Batch with the same engine settings

`batch_infer.py` passes `engine_kwargs=ENGINE_KWARGS`, imported from the same `engine.py` the services use.

```bash
anyscale job submit -f job-batch.yaml --working-dir . --wait
```

The settings are shared, not the object: Ray Data builds its own processor config.

## Gotchas

- Direct streaming takes exactly one `LLMConfig`. Keep the composition app and any multi-model example on separate services.
- Both env vars must be at the **service** level. In a deployment's `runtime_env` they only reach the replicas, and the deploy fails with "ingress_request_router requires HAProxy".
- A `404` on `/v1/messages` or `/v1/responses` means direct streaming is off.
- `/v1/models` lists the model id. Every client must use exactly `qwen3.6-27b`.
- Qwen3.6 reasons by default. A 256-token `/v1/responses` call spent its whole budget thinking and came back `incomplete`. For short demos, send `"chat_template_kwargs": {"enable_thinking": false}`; all three APIs accept it.
- Claude Code asks for 32,000 output tokens, and Codex sends about 23K tokens of system prompt and tools per turn. Both need far more than 32K of context, so never shrink `LLM_MAX_MODEL_LEN` below about 128K for the agent demo.
- Codex warns that it doesn't recognize `qwen3.6-27b` and uses fallback metadata; the warning is harmless. `clients/claude-code.sh` declares the context window, so Claude Code doesn't warn.
- Against the L40S fallback, run Codex with `CODEX_MODEL_CONTEXT_WINDOW=131072` and Claude Code with `CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072`.

## Tear down

```bash
anyscale service terminate -n webinar-1105-qwen-agents
anyscale service terminate -n webinar-1105-composition
anyscale job terminate -n webinar-1105-batch
```

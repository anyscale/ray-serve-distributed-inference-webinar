# 04 · Scaling for agents

Two techniques as deployable code, each an `app.py` and a `service.yaml`, plus
the production shapes for larger models.

| Path | What it is | Who does the work |
|---|---|---|
| [`kv-aware-routing/`](kv-aware-routing/) | `KVAwareRouter` in front of 4 Qwen3-8B replicas (4x L40S) | Ray Serve LLM routes; NVIDIA Dynamo scores |
| [`pd-disaggregation/`](pd-disaggregation/) | `build_pd_openai_app`: 1 prefill + 1 decode replica of Qwen2.5-0.5B, plus `itl.py` to measure decode | Ray Serve LLM builds the pools; vLLM's NIXL connector moves the KV |
| [`configs/`](configs/) | Wide expert parallelism (DeepSeek-V3), decode context parallelism (GLM-5.3 on 8x B200) and a CPU KV tier (GLM-5.2). **Shown, not run**: they need B200, H200 or 8-GPU nodes | Serve places and restarts; vLLM shards |
| [`checks/`](checks/) | The rehearsal harness: one Anyscale job per suite of checks | |

## KV-aware routing

`kv-aware-routing/app.py` is the whole feature: one `LLMConfig` whose
`request_router_config` names `KVAwareRouter`, and one weight,
`DYN_ROUTER_PREFILL_LOAD_SCALE` (S).

```
score = S x (prompt blocks - blocks already cached there) + work already running there
```

The lowest score wins. Raise S for long, repeated prompts, such as coding
agents, so a cache hit outweighs a slightly longer queue.

`service.yaml` carries the three requirements:
- `ai-dynamo-runtime` in the image (see `Containerfile`);
- direct streaming, set as service-level env vars;
- body forwarding at the ingress, so the router can read the prompt.

```bash
cd kv-aware-routing
anyscale service deploy -f service.yaml --working-dir .
```

**What to expect:**
- On Ray 2.59.0, 2 replicas and 2/4/6/8 parallel requests, it finished every request.
- It routed on the cache: 8 of 8 follow-up requests landed on their first
  request's replica, against 0 of 8 for round robin.
- It scores on KV blocks vLLM has already reported. Two identical requests that
  arrive at the same moment both look cold, so test with follow-ups, not with
  simultaneous pairs.
- It is alpha, and Ray's guide says it does not yet support prefill/decode
  disaggregation.

## Prefill/decode disaggregation

`pd-disaggregation/app.py` builds two pools from two `LLMConfig`s that share
the model and set `kv_transfer_config` to NIXL. `build_collocated_app` is the
baseline: the same 2 GPUs as ordinary replicas.

```bash
cd pd-disaggregation
anyscale service deploy -f service.yaml --working-dir .
python itl.py --base-url $URL/v1 --api-key $KEY --label disaggregated
```

`itl.py` streams long replies, then adds an open-loop injector of long prompts,
and prints decode inter-token latency for both phases. Read the
`with injector` row against `quiet`. For the baseline, set `import_path` in
`service.yaml` to `app:build_collocated_app` and run it again.

**What to expect:**
- On this 0.5B model the split does not pay: a short prefill barely disturbs
  decode. In our run on 4x L40S, collocated p99 went from 17.7 to 19.0 ms and
  disaggregated from 17.4 to 20.7 ms.
- A visible stutter needs a heavy prefill, such as a 7B-class model with
  8-16K-token prompts.
- Send a burst of 8 requests before measuring. The first KV handoff after a
  deploy left one 6-9 s gap.

## Shown, not run

```bash
python configs/check_configs.py    # YAML parse; LLMConfig checks need vLLM (Linux)
python checks/check_apps.py        # builds both service.yaml apps without an engine
```

## Rehearsal checks

`checks/run_all.sh` runs one suite per Anyscale job (`configs`, `ab`, `stall`,
`itl`, `kvaware`):

```bash
anyscale job submit -f checks/job.yaml --working-dir . --env SUITE=configs --wait
```

`checks/routing/` also holds the prefix-cache-affinity router app used for the
reuse A/B. Do not put that router under concurrent load on Ray 2.59 or 2.60: it
stops serving at 2 parallel requests while the app reads RUNNING (a blocking
`ray.get` on the router's event loop, fixed on master by ray-project/ray#66671).

## Tear down

```bash
anyscale service terminate -n webinar-1105-kv-routing
anyscale service terminate -n webinar-1105-pd-disaggregation
```

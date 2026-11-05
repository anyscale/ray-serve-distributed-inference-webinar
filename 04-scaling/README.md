# 04 · Scaling for agents

Each technique is deployable code, an `app.py` and a `service.yaml`, plus
the production shapes for larger models.

| Path | What it is | Who does the work |
|---|---|---|
| [`kv-aware-routing/`](kv-aware-routing/) | `KVAwareRouter` in front of 4 Qwen3-8B replicas (4x L40S) | Ray Serve LLM routes; NVIDIA Dynamo scores |
| [`kv-cache-offload/`](kv-cache-offload/) | A CPU tier (vLLM native offloading) or a disk tier (LMCache) under the GPU KV cache of one Qwen2.5-0.5B replica, plus `evict_and_return.py` | vLLM stores and reloads the blocks; Serve passes the settings through |
| [`shared-kv-pool/`](shared-kv-pool/) | One Mooncake KV pool shared by two Qwen2.5-7B replicas on separate L4 nodes, plus `across_replicas.py`. Built and checked on CPU; not yet run on GPUs | vLLM's `MooncakeStoreConnector` moves the blocks; Serve runs the master and configures each engine |
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

## KV cache offloading

An agent session that pauses can come back to find its KV evicted from the GPU,
and pay its whole prefill again. An offload tier keeps a copy of each finished
block below the GPU and loads it back on the next hit.

`kv-cache-offload/app.py` builds one replica with `tier` set to:
- `cpu`: vLLM's native offloading, `kv_offloading_backend: native` and
  `kv_offloading_size` GiB of pinned host memory per replica (`service.yaml`);
- `disk`: LMCache as the KV connector, a small CPU tier with the node's local
  NVMe below it (`service-disk.yaml`);
- `none`: no tier, the baseline to compare against.

```bash
cd kv-cache-offload
anyscale service deploy -f service.yaml --working-dir .        # or service-disk.yaml
python evict_and_return.py --base-url $URL/v1 --api-key $KEY
```

`evict_and_return.py` sends one long session prompt cold, then again (a GPU
hit), then filler prompts that evict it, then the session again. With no tier,
the return costs about as much as cold; with a tier, it loads back instead.
The service files cap the GPU cache with `gpu_kv_blocks` so a few prompts are
enough to evict; leave that unset in production.

LMCache is not in the ray-llm 2.59 image. `Containerfile.lmcache` adds it and
keeps every other package at the image's version, except the two LMCache
needs moved: grpcio and prometheus_client.

**Trade-offs:**
- The CPU tier is pinned host memory for the replica's whole life, used or not.
- `KVAwareRouter` sees the native CPU tier and counts its blocks at a discount.
  It cannot see an LMCache tier.
- Both tiers are private to one replica. For one shared by every replica, see
  the next section.
- For the production shape on GLM-5.2, see `configs/cpu-kv-tier.yaml`.

## A KV cache pool shared across replicas (Mooncake)

An offload tier is private: a prefix one replica computed does nothing for the
replica next to it. [Mooncake](https://github.com/kvcache-ai/Mooncake) pools
host memory from every replica into one KV store. Each replica writes its
finished prefix blocks there and, before a prefill, loads whatever blocks of
the prompt any replica already wrote.

vLLM 0.27 ships the connector, `MooncakeStoreConnector`. `shared-kv-pool/`
wires it into Ray Serve LLM:
- `Containerfile` adds Mooncake's store client and its `mooncake_master` binary.
- `mooncake_pool.py` runs the master as a one-replica deployment on the head
  node. Its Ray Serve LLM callback writes each engine's Mooncake config, with
  the master's address, before vLLM starts.
- `app.py` sets `kv_transfer_config` to `MooncakeStoreConnector` and fixes
  `PYTHONHASHSEED`, so every replica computes the same block keys.

```bash
cd shared-kv-pool
anyscale service deploy -f service.yaml --working-dir .
python across_replicas.py --base-url $URL --api-key $KEY
```

`service.yaml` deploys two replicas, `a` and `b`, on separate L4 nodes.
`across_replicas.py` sends one prompt to `a`, then the same prompt to `b`. With
the pool, `b` loads `a`'s blocks instead of recomputing them; set
`shared_pool: false` for the baseline.

**Trade-offs:**
- Loading must beat recomputing. That favours large models, long shared
  prefixes and RDMA networks; `service.yaml` uses TCP because g6 nodes have no
  RDMA NIC.
- The master is one more service to keep up. This example runs one on the head
  node with no standby.
- Routing still matters: a replica's own GPU cache is faster than the pool.

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
python checks/check_apps.py        # builds every service*.yaml app without an engine
```

## Rehearsal checks

`checks/run_all.sh` runs one suite per Anyscale job (`configs`, `ab`, `stall`,
`itl`, `kvaware`, `offload`):

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
anyscale service terminate -n webinar-1105-kv-offload-cpu
anyscale service terminate -n webinar-1105-kv-offload-disk
anyscale service terminate -n webinar-1105-shared-kv-pool
```

# 04 · Scaling for agents: demo code and configs

Code for the scaling part of the webinar.
Everything runnable uses Qwen2.5-0.5B-Instruct on one 4x L4 node.

| Path | What it is | Who does it |
|---|---|---|
| `routing/app.py` | One model, two replicas, `PrefixCacheAffinityRouter` | Ray Serve LLM |
| `routing/verify.py` | Reuse A/B (exits 1 if affinity loses) and a 2–8-request stall check | Ray Serve LLM |
| `disaggregation/app.py` | `build_pd_openai_app` with NIXL, plus a collocated baseline on the same 2 GPUs | Ray Serve LLM builds the pools; vLLM moves the KV |
| `disaggregation/itl.py` | Decode inter-token latency, quiet and with a long-prompt injector | |
| `configs/wide-ep-deepseek.yaml` | DeepSeek-V3, DP = EP = 16, one GPU per rank | Serve places and gang-restarts; vLLM shards experts |
| `configs/glm53-nvfp4-b200-dcp.yaml` | GLM-5.3 NVFP4 on 8x B200 with DCP 8 | vLLM |
| `configs/kv-aware-router.yaml` + `Containerfile.kv-router` | `KVAwareRouter`, direct streaming, ai-dynamo | Ray Serve LLM (alpha) |
| `configs/cpu-kv-tier.yaml` | CPU KV tier (`kv_offloading_size`) with sizing guidance | vLLM |
| `configs/check_configs.py` | yaml.safe_load, LLMConfig and app-build checks, vLLM arg names | |
| `workspace.yaml` | The Anyscale workspace for the demos | |
| `job.yaml` + `run_all.sh` + `wait_for_gpus.py` | Every check as one Anyscale job | |

## Everything as one job

One suite per job, run one after another, so a noisy suite cannot push
another's results out of the log:

```bash
for suite in configs ab stall itl; do   # kvaware also needs the --env flags in run_all.sh
  anyscale job submit -f job.yaml --working-dir . -e .venv \
    -n webinar-1105-scaling-$suite --env SUITE=$suite --wait
done
anyscale job logs -n webinar-1105-scaling-stall    # results block at the end
```

Each GPU job waits for one 4-GPU node (L4, else L40S, else A10G, in us-west-2a/b/c).
The configs suite needs no GPU.

The files in `configs/` are production shapes to show, not to run. They need
B200, H200 or 8-GPU nodes.

## Workspace for the demos

Start the workspace early. A cold start pulls the ray-llm image to a fresh
node, which takes most of the wait.

```bash
anyscale workspace_v2 create -f workspace.yaml   # once
anyscale workspace_v2 start -n webinar-1105-scaling-ws
anyscale workspace_v2 wait  -n webinar-1105-scaling-ws --state RUNNING --timeout-s 1800
anyscale workspace_v2 push  -n webinar-1105-scaling-ws --local-dir .
```

Open the workspace terminal. Code lands in `~/default/`. Do one warm run of
each demo so the model weights are cached on the node.

## Routing A/B

```bash
cd ~/default/routing
python verify.py --skip-stall        # A/B only: two deployments, ~2 min each
```

Expected shape: default router ~0% reused, affinity ~49%, then `PASS` with
exit code 0. `echo $?` shows the exit code, so the check can gate CI.

**Do not put PrefixCacheAffinityRouter under concurrent load on Ray 2.59 or 2.60.**
On 2.59.0 it stops serving at 2 parallel requests while the app reads RUNNING.
The default router served the same load without a miss. The cause is a blocking
`ray.get` on the router's event loop, fixed on master by ray-project/ray#66671.
The 2-request A/B above is unaffected.

**KVAwareRouter does not have this stall.** On the same Ray 2.59.0, 2 replicas
and 2/4/6/8 parallel requests, it finished all 60 requests, both with the
shipped config and with one slot per replica. It also routed on the cache: 8 of
8 follow-ups landed on their seed's replica, against 0 of 8 for round robin.
- Do not demo it with the simultaneous 2-request A/B. It scores on KV blocks
  vLLM has already reported, so two requests that arrive together both look
  cold (0% reuse).
- Use the follow-up check instead. It needs direct streaming and ai-dynamo, as
  in `configs/kv-aware-router.yaml`:

```bash
# Cluster env (service env_vars, or job --env): RAY_SERVE_ENABLE_HA_PROXY=1,
# RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1, RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY=1
python ../cluster_tools.py pip --no-deps "ai-dynamo-runtime>=1.4.0"   # every node
python verify.py --router kvaware --skip-reuse --skip-stall --followups 8
```

The stall check:

```bash
python verify.py --skip-reuse                          # affinity router, 2/4/6/8 parallel
python verify.py --skip-reuse --stall-router default   # the same with the default router
python verify.py --router kvaware --skip-reuse          # KVAwareRouter (env as above)
```

## Prefill/decode split

Same model, same 2 GPUs, measured one way and then the other:

```bash
cd ~/default/disaggregation
python deploy.py collocated        # 2 ordinary replicas; returns once RUNNING
python itl.py --label collocated
serve shutdown -y

python deploy.py disaggregated     # 1 prefill + 1 decode replica, NIXL
python itl.py --label disaggregated
serve shutdown -y
```

Read the `with injector` row against `quiet` in each table.

**What our run showed** (4x L40S):
- On this 0.5B model the injector barely moves decode, and the split did not improve it.
  - Collocated p99 went from 17.7 to 19.0 ms.
  - Disaggregated p99 went from 17.4 to 20.7 ms.
- So on a model this small the split does not pay: a short prefill barely
  disturbs decode.
- A visible stutter needs a heavy prefill, such as a 7B-class model with
  8-16K-token prompts. We did not run that case.

Before measuring, send a burst of 8 concurrent requests to the disaggregated
app. The first KV handoff after deploy left one 6-9 s gap in both of our runs.

## Shown, not run

```bash
python configs/check_configs.py           # YAML parse; LLMConfig checks need vLLM (Linux)
```

## Tear down

```bash
anyscale workspace_v2 terminate -n webinar-1105-scaling-ws
```

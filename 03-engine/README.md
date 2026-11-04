# 03 · Inside the engine

Three demos: the prefix cache cold versus warm, the throughput–interactivity
frontier, and one engine knob that shifts the frontier.

| File | What it is |
|---|---|
| `knobs.py` | Two `LLMConfig` variants on one L4 that differ in one knob, `enforce_eager`: CUDA graphs and torch.compile on (vLLM's default) or off. The model is Qwen2.5-0.5B-Instruct by default, or 1.5B with `model=qwen-1.5b`. |
| `service.yaml`, `service_eager.yaml` | Anyscale services for the two variants, one L4 node each. |
| `prefix_cold_warm.py` | Resets the prefix cache, then times the first token of one ~31K-token prompt, cold and then warm. |
| `sweep.py` | A concurrency sweep against any OpenAI-compatible endpoint, written as a CSV. |
| `plot_frontier.py` | The frontier chart from one CSV, or from two for "knob on versus off". |
| `results/` | Measured CSVs and charts from the verification run, plus the GLM-5 reference. |
| `workspace.yaml` | The single-L4 workspace that the verification run used. |

## Deploy

The two services take about 5 minutes to reach RUNNING.

```bash
cd 03-engine
python3.12 -m venv .venv && .venv/bin/pip install openai httpx matplotlib
source .venv/bin/activate

anyscale service deploy -f service.yaml --working-dir .
anyscale service deploy -f service_eager.yaml --working-dir .
anyscale service wait -n webinar-1105-engine-graphs --state RUNNING --timeout-s 1200
anyscale service wait -n webinar-1105-engine-eager --state RUNNING --timeout-s 1200

# Each service has its own URL and its own bearer token.
GRAPHS=$(anyscale service status -n webinar-1105-engine-graphs -j | jq -r .query_url)
GRAPHS_KEY=$(anyscale service status -n webinar-1105-engine-graphs -j | jq -r .query_auth_token)
EAGER=$(anyscale service status -n webinar-1105-engine-eager -j | jq -r .query_url)
EAGER_KEY=$(anyscale service status -n webinar-1105-engine-eager -j | jq -r .query_auth_token)

# Send one request to each engine, so the measured runs skip first-request setup.
python prefix_cold_warm.py --base-url "$GRAPHS/v1" --api-key "$GRAPHS_KEY" --trials 1
python prefix_cold_warm.py --base-url "$EAGER/v1"  --api-key "$EAGER_KEY"  --trials 1
```

## Run

**Prefix cache, cold versus warm** (about 10 seconds)

```bash
python prefix_cold_warm.py --base-url "$GRAPHS/v1" --api-key "$GRAPHS_KEY" --trials 1
```

From the laptop, the cold first token took 1.7 s and the warm one 0.3 s. On
the node itself the warm one is 0.1 s; the rest is the internet round trip.

**The frontier** (about 30 seconds)

```bash
python sweep.py --base-url "$GRAPHS/v1" --api-key "$GRAPHS_KEY" \
  --concurrency 1,4,16,64 --rounds 3 --out results/live_cuda-graphs.csv
python plot_frontier.py results/live_cuda-graphs.csv --label "CUDA graphs" \
  --out results/live_frontier.png && open results/live_frontier.png
```

Then show the same chart for a large model: `results/glm5_b200_reference.png`.
That is GLM-5 NVFP4 on 4× B200, where the 30 tok/s floor lands at 32 users and
2,307 tok/s per GPU. The small model clears 30 tok/s per user even at 64
users, so try `--floor 100` to watch the operating point move.

**One knob, redrawn** (about 1 minute)

```bash
python sweep.py --base-url "$EAGER/v1" --api-key "$EAGER_KEY" \
  --concurrency 1,4,16,64 --rounds 3 --out results/live_eager.csv
python plot_frontier.py results/live_cuda-graphs.csv --label "CUDA graphs (default)" \
  --overlay results/live_eager.csv --overlay-label "eager (enforce_eager)" \
  --out results/live_knob.png && open results/live_knob.png
```

**Reference charts.** The pre-rendered charts from the full sweeps are `results/frontier_0.5b_cuda-graphs.png` and
`results/frontier_0.5b_knob.png`.

## Tear down

```bash
anyscale service terminate -n webinar-1105-engine-graphs
anyscale service terminate -n webinar-1105-engine-eager
```

## Notes

- **What the knob shows.** Eager mode adds a roughly fixed cost to every decode
  step: kernels launch one by one from Python, and torch.compile's fused
  kernels are gone. It hurts most when steps are short: one user on the 0.5B model gets
  3× fewer tokens per second. With 64 users, eager still delivers about 30%
  less throughput per GPU. On the 1.5B model the gap is smaller and
  closes by 64 users (`results/frontier_1.5b_knob.png`), because each step
  does more GPU work that hides the launch cost.
- **`enforce_eager` turns off two things.** vLLM 0.27 disables both CUDA graphs
  and torch.compile under this flag, and says so in its log. Call it "eager
  mode", or "CUDA graphs off" with that caveat.
- **Sweeps from the laptop match the node, apart from TTFT.** The time to first
  token includes the internet round trip. New connections sometimes stalled
  for seconds, so `sweep.py` opens all of its connections before it measures.
  `--rounds 3` keeps one slow request from bending a short curve.
- **The reset route.** The services use the dev ingress (`build_dev_openai_app`),
  an internal Ray Serve LLM API that adds `POST /reset_prefix_cache`. It sits
  behind the service's bearer token, like the OpenAI routes.
- **`ignore_eos`.** `sweep.py` sends it so that every request produces its full
  output length. vLLM and SGLang accept it; OpenAI's own API does not.

# 01 · Ray Serve foundations

One deployment, `Ranker`, behind the Serve proxy. There is no ingress deployment:
the proxy routes each request straight to a Ranker replica.

```
client ──► Serve proxy (power of two choices) ──► Ranker │ Ranker │ … (1 to 6 replicas)
```

One client script shows three things:

1. **Routing.** The proxy spreads requests evenly over the replicas.
2. **Autoscaling.** Under a burst the replicas go from 1 to 6, then back to 1.
3. **Batching.** `@serve.batch` merges requests into one model call. With the
   same 16 users, it gives 4× the throughput of one call per request.

| File | What it holds |
|---|---|
| `app.py` | The Ranker: a numpy two-tower model, `@serve.batch` and the autoscaling config. |
| `service.yaml` | The same app as an Anyscale Service, on CPU nodes only. |
| `load.py` | The client: `batch` compares batching on and off, and `scale` runs quiet, burst, quiet. |

The client uses only the Python standard library (3.10+).

## Deploy

From a terminated service, a deploy reached RUNNING in about 2 minutes.
An update to a running service took about 8 minutes.

```bash
cd 01-serve-foundations
anyscale service deploy -f service.yaml --working-dir .
anyscale service wait -n webinar-1105-ranker --state RUNNING --timeout-s 900
export SERVICE_URL=<query_url> SERVICE_TOKEN=<query_auth_token>   # from `anyscale service status -n webinar-1105-ranker -j`; keep it off screen
```

## Run

```bash
python load.py batch   # 80 s: 16 users with batching on, then off, on 1 replica
python load.py scale   # 200 s: 4 users, then 120 users, then 4 users
```

Every 3 s the client prints req/s, p50 and p95 latency, the mean batch size, and
the replicas that answered. At the end of each stage it prints the batch-size
distribution and the requests per replica.

## Tear down

```bash
anyscale service terminate -n webinar-1105-ranker
```

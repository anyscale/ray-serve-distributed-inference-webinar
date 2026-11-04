# 05 · Cost close: what one GPU costs per developer

**No GPU, no deploy.** The frontier from the engine
part returns: pick an interactivity floor, read the sessions per
GPU off the curve, and turn that into a cost per developer-month. Then compare
it with subscription seats and per-token billing, using your own
numbers.

## The model

The GPU is a fixed cost shared by the team, so the cost per developer falls as
the team grows, until the next replica is needed.

```text
developers per GPU  = sessions per GPU / share of developers active at the busy hour
GPUs                = ceil(developers / developers per replica) x GPUs per replica
$ per developer-mo  = GPUs x $/GPU-hour x hours per month / developers
```

| Input | Default | Where it comes from |
|---|---:|---|
| `--gpu-price` | $4.00/GPU-hour | `g7e.4xlarge` (1× RTX PRO 6000) on demand |
| `--sessions-per-gpu` | 24 | KV capacity: ~1.7M cached tokens / ~70K tokens per turn |
| `--peak-active` | 0.48 | 24 sessions per 50 developers, the planning case |
| `--hours-per-month` | 730 | always on; 210 is work hours (10 h × 21 weekdays) |
| `--seat-price` | $200/dev-month | a Claude Max 20x or Cursor Ultra seat |
| `--token-bill` | $800/dev-month | a heavy user on per-token billing |

All defaults are rounded list prices from July 2026.

## Running it

```bash
python cost.py                                     # the planning case
python cost.py --hours-per-month 210               # GPU on during work hours only
python cost.py --developers 40 --gpu-price 3.20    # your own numbers
```

The planning case prints:

```text
  devs  GPUs  self $/dev-mo  vs seats $/mo  vs tokens $/mo
----------------------------------------------------------
    15     1         194.67             80           9,080
    50     1          58.40          7,080          37,080
   100     2          58.40         14,160          74,160
   500    10          58.40         70,800         370,800
Self-hosting beats seats from 15 developers.
Self-hosting beats token billing from 4 developers.
```

## Reading sessions off the frontier

With `--frontier`, the calculator takes the highest concurrency whose per-user
decode speed still meets `--floor`, and divides it by the GPUs in one replica.
The CSV needs a `concurrency` column and either `tok_s_per_user` or `tpot_ms`.
`03-engine/sweep.py` writes both columns it needs; `data/` holds the GLM-5 curve as an example.

```bash
python cost.py --frontier data/glm5_nvfp4_b200_cu13.csv --gpus-per-replica 4 \
    --floor 30 --gpu-price <B200 $/GPU-hour>
```

At a floor of 30 tokens per second per user, that curve gives concurrency 32 on
4 GPUs, or 8 sessions per GPU.

## Caveats

- A frontier only measures capacity for the prompt length it was swept with.
  The GLM-5 curve uses 8K-token inputs; a coding agent sends about 70K per turn,
  and the KV cache may run out before compute does. Sweep with your own prompt
  lengths.
- The quality claim that Qwen3.6-27B is comparable to Claude Opus 4.5 is Qwen's
  own. Validate on your own repos before counting the savings.
- Work-hours savings hold only if the GPU node actually terminates when idle.
- Break-even here is exact, so the work-hours case reads 5 and 2 developers
  where COST-ESTIMATE.md rounds to about 4 and 1.

## Source

Defaults and formulas from
`anyscale/llm-serving-for-coding-agents: part3-optimize/notes/COST-ESTIMATE.md`.
The GLM-5 curve is an Anyscale benchmark: GLM-5 NVFP4, TP4 on 4× B200, SGLang
through Ray Serve LLM, 8K in / 1K out.

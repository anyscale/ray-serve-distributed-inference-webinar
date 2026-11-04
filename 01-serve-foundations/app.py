"""Ranker service: one deployment behind the Serve proxy.

The proxy picks a replica for each request with power-of-two-choices routing
(Serve's default). The deployment autoscales on requests in flight per
replica, and `@serve.batch` merges concurrent requests into one model call.
`?batch=off` calls the model once per request, for the comparison.

    serve run app:app
    curl -s localhost:8000/ -d '{"user_id": 42, "candidates": [1, 2, 3]}'
"""

import zlib

import numpy as np
from ray import serve
from starlette.requests import Request

FEATURES = 256  # user feature vector length
HIDDEN = 4096  # width of the user tower's hidden layers
DEPTH = 6  # HIDDEN x HIDDEN layers; sets the cost of one model call
EMBED = 128  # shared space where user and item vectors meet
NUM_ITEMS = 100_000  # item table rows; candidate ids index into it
TOP_K = 10


class TwoTowerRanker:
    """User tower (an MLP) and an item table; a score is their dot product.

    Plain numpy on one CPU. Seeded weights stand in for a trained checkpoint.
    The user tower dominates the cost, and every row of a batch shares its
    weight reads, which is why batching pays.
    """

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        shapes = [(FEATURES, HIDDEN)] + [(HIDDEN, HIDDEN)] * DEPTH + [(HIDDEN, EMBED)]
        self.layers = [rng.standard_normal(s, dtype=np.float32) / np.sqrt(s[0]) for s in shapes]
        self.items = rng.standard_normal((NUM_ITEMS, EMBED), dtype=np.float32)

    def __call__(self, user_ids: list[int], candidates: list[list[int]]) -> list[list[int]]:
        """Top-K candidate ids for each user, best first."""
        x = np.stack([user_features(u) for u in user_ids])
        for w in self.layers[:-1]:
            x = np.maximum(x @ w, 0)
        users = x @ self.layers[-1]
        tops = []
        for user, cands in zip(users, candidates):
            scores = self.items[cands] @ user
            tops.append([cands[i] for i in np.argsort(-scores)[:TOP_K]])
        return tops


def user_features(user_id: int) -> np.ndarray:
    """Deterministic feature vector for a user, seeded by the id."""
    rng = np.random.default_rng(zlib.crc32(str(user_id).encode()))
    return rng.standard_normal(FEATURES, dtype=np.float32)


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    # Above the target, so queued requests wait at a replica and join a batch.
    max_ongoing_requests=32,
    # Replicas = requests in flight / target, clamped to [min, max]. The short
    # delays suit a live demo; Serve's defaults are 30 s up and 600 s down.
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 6,
        "target_ongoing_requests": 20,
        "upscale_delay_s": 5,
        "downscale_delay_s": 30,
    },
)
class Ranker:
    def __init__(self) -> None:
        self.model = TwoTowerRanker()  # loaded once per replica, never per request
        self.replica = serve.get_replica_context().replica_id.unique_id

    async def __call__(self, request: Request) -> dict:
        body = await request.json()
        if request.query_params.get("batch") == "off":
            top, batch_size = self.model([body["user_id"]], [body["candidates"]])[0], 1
        else:
            top, batch_size = await self.rank(body)
        return {"top": top, "replica": self.replica, "batch_size": batch_size}

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.01)
    async def rank(self, bodies: list[dict]) -> list[tuple[list[int], int]]:
        """Score a batch of users in one model call."""
        tops = self.model([b["user_id"] for b in bodies], [b["candidates"] for b in bodies])
        return [(top, len(bodies)) for top in tops]


app = Ranker.bind()

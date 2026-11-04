"""Prefix cache index for a paged KV cache.

A prompt is split into fixed-size blocks of token ids. Each full block gets a
hash that identifies the block and every token before it, because the keys and
values computed for a block depend on the whole prefix, not only on the block's
own tokens. Two prompts may share a cached block only when everything up to and
including that block is identical.
"""

from collections.abc import Sequence

# Tokens per KV block. vLLM's default block size is 16 as well.
BLOCK_SIZE = 16


def block_hashes(tokens: Sequence[int], block_size: int = BLOCK_SIZE) -> list[int]:
    """Return one hash per full block of `tokens`.

    A trailing partial block gets no hash: it is still being filled, so it
    cannot be shared yet.
    """
    full = len(tokens) - len(tokens) % block_size
    hashes = []
    for start in range(0, full, block_size):
        block = tuple(tokens[start : start + block_size])
        hashes.append(hash(block))
    return hashes


class PrefixCache:
    """Set of cached block hashes, with a longest-prefix lookup."""

    def __init__(self, block_size: int = BLOCK_SIZE) -> None:
        self.block_size = block_size
        self._cached: set[int] = set()

    def insert(self, tokens: Sequence[int]) -> None:
        """Record every full block of a prompt whose KV has been computed."""
        self._cached.update(block_hashes(tokens, self.block_size))

    def match(self, tokens: Sequence[int]) -> int:
        """Return how many leading tokens of `tokens` can be served from cache."""
        hits = 0
        for block_hash in block_hashes(tokens, self.block_size):
            if block_hash not in self._cached:
                break
            hits += 1
        return hits * self.block_size

"""Tests for the prefix cache index.

Run with: python -m unittest -v
"""

import unittest

from prefix_cache import BLOCK_SIZE, PrefixCache, block_hashes


def block(token: int) -> list[int]:
    """One full block made of a single repeated token id."""
    return [token] * BLOCK_SIZE


class BlockHashesTest(unittest.TestCase):
    def test_partial_block_is_not_hashed(self) -> None:
        """A prompt of one and a half blocks yields exactly one hash."""
        tokens = block(1) + [2] * (BLOCK_SIZE // 2)

        hashes = block_hashes(tokens)

        self.assertEqual(len(hashes), 1)

    def test_same_prompt_same_hashes(self) -> None:
        """Hashing is deterministic, so a repeated prompt maps to the same blocks."""
        tokens = block(1) + block(2)

        self.assertEqual(block_hashes(tokens), block_hashes(list(tokens)))


class PrefixCacheTest(unittest.TestCase):
    def test_repeated_prompt_is_fully_cached(self) -> None:
        """Sending the same two-block prompt twice hits on both blocks."""
        cache = PrefixCache()
        prompt = block(1) + block(2)
        cache.insert(prompt)

        self.assertEqual(cache.match(prompt), 2 * BLOCK_SIZE)

    def test_shared_prefix_hits_up_to_divergence(self) -> None:
        """Two prompts that share the first block hit on that block only."""
        cache = PrefixCache()
        cache.insert(block(1) + block(2))

        self.assertEqual(cache.match(block(1) + block(3)), BLOCK_SIZE)

    def test_same_block_after_different_prefix_is_a_miss(self) -> None:
        """A block reached through a different prefix must not count as cached.

        Block 2 was cached after block 1, and block 3 was cached at the start.
        The prompt [3, 2] matches block 3, but its block 2 follows a different
        prefix, so its KV differs and only the first block may be reused.
        """
        cache = PrefixCache()
        cache.insert(block(1) + block(2))
        cache.insert(block(3) + block(4))

        self.assertEqual(cache.match(block(3) + block(2)), BLOCK_SIZE)


if __name__ == "__main__":
    unittest.main()

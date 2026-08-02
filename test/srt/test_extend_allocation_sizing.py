"""CPU-only tests for extend/decode page-sizing helpers."""

import unittest

import torch

# Must precede sglang imports on machines without a working Triton runtime.
try:
    import torch._inductor.runtime.triton_heuristics  # noqa: F401
except Exception:
    pass

from sglang.srt.utils.common import (
    get_extend_allocation_size,
    get_num_new_pages,
    get_num_new_pages_for_extend,
)


class TestDecodePageSizing(unittest.TestCase):
    def test_page_size_one_decode_never_needs_a_new_page(self):
        # With 1-token pages the token allocator hands out slots directly,
        # so alloc_decode must never charge an extra page.
        for seq_lens in ([1], [5], [9], [1, 2, 3, 4, 5, 16, 17]):
            with self.subTest(seq_lens=seq_lens):
                self.assertEqual(
                    get_num_new_pages(
                        seq_lens=torch.tensor(seq_lens),
                        page_size=1,
                        decode=True,
                    ),
                    0,
                )

    def test_decode_matches_modulo_formula_for_paged_sizes(self):
        for page_size in (2, 4, 8, 64, 128):
            for seq_len in range(0, 300):
                seq_lens = torch.tensor([seq_len])
                expected = int((seq_lens % page_size == 1).int().sum().item())
                with self.subTest(page_size=page_size, seq_len=seq_len):
                    self.assertEqual(
                        get_num_new_pages(
                            seq_lens=seq_lens, page_size=page_size, decode=True
                        ),
                        expected,
                    )

    def test_decode_batch_sums_per_request(self):
        seq_lens = torch.tensor([1, 2, 65, 129, 130])
        page_size = 64
        expected = int((seq_lens % page_size == 1).int().sum().item())
        self.assertEqual(
            get_num_new_pages(seq_lens=seq_lens, page_size=page_size, decode=True),
            expected,
        )


class TestExtendPageSizingConsistency(unittest.TestCase):
    def test_vectorized_and_scalar_agree_on_extend(self):
        cases = [
            (0, 1),
            (0, 63),
            (0, 64),
            (0, 65),
            (1, 1),
            (63, 1),
            (64, 1),
            (65, 1),
            (500, 512),
            (512, 512),
            (511, 1),
            (128, 0),
        ]
        for page_size in (1, 2, 4, 8, 64):
            for prefix_len, extend_len in cases:
                with self.subTest(
                    page_size=page_size, prefix=prefix_len, extend=extend_len
                ):
                    vectorized = get_num_new_pages(
                        seq_lens=torch.tensor([prefix_len + extend_len]),
                        page_size=page_size,
                        prefix_lens=torch.tensor([prefix_len]),
                    )
                    scalar = get_num_new_pages_for_extend(
                        prefix_lens=[prefix_len],
                        extend_lens=[extend_len],
                        page_size=page_size,
                    )
                    self.assertEqual(vectorized, scalar)

    def test_vectorized_and_scalar_agree_on_batches(self):
        prefix_lens = [0, 64, 65, 500, 511]
        extend_lens = [1, 64, 63, 512, 1]
        for page_size in (1, 2, 4, 8, 64):
            with self.subTest(page_size=page_size):
                seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
                vectorized = get_num_new_pages(
                    seq_lens=torch.tensor(seq_lens),
                    page_size=page_size,
                    prefix_lens=torch.tensor(prefix_lens),
                )
                scalar = get_num_new_pages_for_extend(
                    prefix_lens=prefix_lens,
                    extend_lens=extend_lens,
                    page_size=page_size,
                )
                self.assertEqual(vectorized, scalar)

    def test_extend_allocation_size_is_pages_times_page_size(self):
        self.assertEqual(
            get_extend_allocation_size(
                prefix_lens=[512, 512], extend_lens=[512, 512], page_size=64
            ),
            16 * 64,
        )


class _StubSubAllocator:
    """Minimal stand-in for the full/swa sub-allocators inside SWA."""

    def __init__(self, available):
        self._available = available
        self.alloc_extend_called_with = None

    def available_size(self):
        return self._available

    def alloc_extend(self, *args, **kwargs):
        self.alloc_extend_called_with = (args, kwargs)
        return torch.arange(kwargs.get("extend_num_tokens", 0), dtype=torch.int64)


class TestSwaExtendGateMatchesEviction(unittest.TestCase):
    """The SWA gate must not demand more than evict_from_tree_cache frees.

    mem_cache/common.py:alloc_paged_token_slots_extend evicts exactly
    get_extend_allocation_size(...). If SWA's internal gate keeps the old
    over-estimate (extend_num_tokens + bs * page_size) it rejects an
    allocation the caller already made room for, surfacing as a spurious
    "Prefill out of memory".
    """

    def _gate_value(self, prefix_lens, extend_lens, page_size):
        from sglang.srt.mem_cache import swa_memory_pool

        captured = {}
        real = swa_memory_pool.get_extend_allocation_size

        def spy(**kwargs):
            captured["value"] = real(**kwargs)
            return captured["value"]

        swa_memory_pool.get_extend_allocation_size = spy
        try:
            allocator = swa_memory_pool.SWATokenToKVPoolAllocator.__new__(
                swa_memory_pool.SWATokenToKVPoolAllocator
            )
            allocator.page_size = page_size
            huge = 10**9
            allocator.full_attn_allocator = _StubSubAllocator(huge)
            allocator.swa_attn_allocator = _StubSubAllocator(huge)
            allocator.translate_loc_from_full_to_swa = lambda loc: loc
            allocator.full_to_swa_index_mapping = None

            prefix_cpu = torch.tensor(prefix_lens, dtype=torch.int64)
            seq_cpu = torch.tensor(
                [p + e for p, e in zip(prefix_lens, extend_lens)], dtype=torch.int64
            )
            try:
                allocator.alloc_extend(
                    prefix_lens=prefix_cpu,
                    prefix_lens_cpu=prefix_cpu,
                    seq_lens=seq_cpu,
                    seq_lens_cpu=seq_cpu,
                    last_loc=torch.zeros(len(prefix_lens), dtype=torch.int64),
                    extend_num_tokens=int(sum(extend_lens)),
                )
            except Exception:
                # We only care about the gate value, not the downstream
                # index-mapping work that needs real pools.
                pass
        finally:
            swa_memory_pool.get_extend_allocation_size = real
        return captured.get("value")

    def test_gate_equals_eviction_target(self):
        cases = [
            ([512] * 8, [512] * 8),  # aligned prefixes
            ([500] * 8, [512] * 8),  # unaligned prefixes
            ([0] * 4, [64] * 4),  # exact single page each
            ([63] * 4, [1] * 4),  # completing a partial page
        ]
        page_size = 64
        for prefix_lens, extend_lens in cases:
            with self.subTest(prefix=prefix_lens[0], extend=extend_lens[0]):
                expected = get_extend_allocation_size(
                    prefix_lens=prefix_lens,
                    extend_lens=extend_lens,
                    page_size=page_size,
                )
                self.assertEqual(
                    self._gate_value(prefix_lens, extend_lens, page_size), expected
                )

    def test_gate_no_longer_uses_the_old_over_estimate(self):
        prefix_lens, extend_lens, page_size = [512] * 8, [512] * 8, 64
        old_over_estimate = sum(extend_lens) + len(prefix_lens) * page_size
        exact = get_extend_allocation_size(
            prefix_lens=prefix_lens, extend_lens=extend_lens, page_size=page_size
        )
        self.assertEqual(old_over_estimate - exact, len(prefix_lens) * page_size)
        self.assertEqual(self._gate_value(prefix_lens, extend_lens, page_size), exact)


if __name__ == "__main__":
    unittest.main()

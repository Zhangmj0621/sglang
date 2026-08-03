"""Unit tests for ref-aware tiered eviction on MambaRadixCache.

CPU-only: fake allocator + fake mamba pool, cache built via __new__
(mirrors test_ref_aware_kv_cache.py's harness pattern).
"""

import unittest
import unittest.mock
from types import SimpleNamespace

import torch

# Must precede any sglang import: on machines without a working triton the
# torch dynamo module fails to initialize lazily inside sglang's import chain.
try:
    import torch._inductor.runtime.triton_heuristics  # noqa: F401
except Exception:
    pass

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.mamba_radix_cache import (
    MambaChunkStashResult,
    MambaRadixCache,
    TreeNode,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    get_child_key,
)
from sglang.srt.mem_cache.ref_aware_mamba_radix_cache import RefAwareMambaRadixCache
from sglang.srt.utils.common import (
    get_extend_allocation_size,
    get_full_kv_reservation,
    get_num_new_pages,
)


class _FakeAllocator:
    def __init__(self, size=1024):
        self.size = size
        self._next = 0
        self.device = torch.device("cpu")
        self.page_size = 1
        self.freed_total = 0

    def alloc(self, n):
        t = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return t

    def free(self, indices):
        self.freed_total += len(indices)

    def available_size(self):
        return self.size - self._next + self.freed_total


class _FakeMambaPool:
    def __init__(self, size=64):
        self.size = size
        self._free = list(range(size))
        self.freed = []

    def alloc(self, n):
        if len(self._free) < n:
            return None
        return torch.tensor([self._free.pop() for _ in range(n)], dtype=torch.int64)

    def free(self, indices):
        for i in indices.tolist():
            self._free.append(i)
            self.freed.append(i)

    def available_size(self):
        return len(self._free)

    def fork_from(self, mamba_value):
        return self.alloc(1)

    def copy_from(self, src, dst):
        pass


class _FakeStashReqPool:
    def __init__(self, mamba_pool, max_context_len=32):
        self.mamba_pool = mamba_pool
        self.req_to_token = torch.full((2, max_context_len), -1, dtype=torch.int64)
        self.req_index_to_mamba_index_mapping = torch.full((2,), -1, dtype=torch.int64)
        self.enable_mamba_extra_buffer = True
        self.mamba_ping_pong_track_buffer_size = 2
        self.freed_req_slots = []

    def get_mamba_indices(self, req_indices):
        return self.req_index_to_mamba_index_mapping[req_indices]

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def mamba_state_need(self, req):
        return int(req.mamba_pool_idx is None) + (
            2 if req.mamba_ping_pong_track_buffer is None else 0
        )

    def free_mamba_cache(self, req, mamba_ping_pong_track_buffer_to_keep=None):
        if req.mamba_pool_idx is not None:
            self.mamba_pool.free(req.mamba_pool_idx.unsqueeze(0))
            req.mamba_pool_idx = None
        if req.mamba_ping_pong_track_buffer is not None:
            self.mamba_pool.free(req.mamba_ping_pong_track_buffer)
            req.mamba_ping_pong_track_buffer = None
            req.mamba_next_track_idx = None

    def free(self, req):
        self.freed_req_slots.append(req.req_pool_idx)
        req.req_pool_idx = None


class _FakeHybridMambaPool:
    def __init__(self, size, *, fail_alloc=False):
        self.size = size
        self.free_slots = torch.arange(1, size + 1, dtype=torch.int64)
        self.fail_alloc = fail_alloc

    def alloc(self, n):
        if self.fail_alloc or n > len(self.free_slots):
            return None
        selected = self.free_slots[:n]
        self.free_slots = self.free_slots[n:]
        return selected

    def available_size(self):
        return len(self.free_slots)


class _FailOnceMapping:
    def __init__(self, tensor):
        self.tensor = tensor
        self.fail_next_write = True

    def __getitem__(self, index):
        return self.tensor[index]

    def __setitem__(self, index, value):
        if self.fail_next_write:
            self.fail_next_write = False
            raise RuntimeError("injected mapping write failure")
        self.tensor[index] = value


def _make_req(
    *,
    req_pool_idx=None,
    mamba_pool_idx=None,
    mamba_ping_pong_track_buffer=None,
    mamba_next_track_idx=None,
    is_chunked=0,
    kv_committed_len=0,
):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        mamba_pool_idx=mamba_pool_idx,
        mamba_ping_pong_track_buffer=mamba_ping_pong_track_buffer,
        mamba_next_track_idx=mamba_next_track_idx,
        is_chunked=is_chunked,
        kv_committed_len=kv_committed_len,
    )


def _make_hybrid_pool(
    *,
    req_slots,
    mamba_slots,
    enable_extra=True,
    overlap=True,
    fail_alloc=False,
    fail_ping_mapping=False,
):
    pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    pool.size = 16
    pool.free_slots = list(range(req_slots))
    pool.enable_mamba_extra_buffer = enable_extra
    pool.mamba_ping_pong_track_buffer_size = 2 if overlap else 1
    pool.mamba_pool = _FakeHybridMambaPool(mamba_slots, fail_alloc=fail_alloc)
    pool.req_index_to_mamba_index_mapping = torch.full(
        (pool.size,), -1, dtype=torch.int32
    )
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.full(
        (pool.size, pool.mamba_ping_pong_track_buffer_size),
        -1,
        dtype=torch.int32,
    )
    if fail_ping_mapping:
        pool.req_index_to_mamba_ping_pong_track_buffer_mapping = _FailOnceMapping(
            pool.req_index_to_mamba_ping_pong_track_buffer_mapping
        )
    return pool


class TestExactResourceDemand(unittest.TestCase):
    def test_req_slot_and_mamba_demand_matrix_with_overlap(self):
        pool = _make_hybrid_pool(req_slots=8, mamba_slots=16, overlap=True)
        main = torch.tensor(11)
        ping_pong = torch.tensor([12, 13])
        cases = [
            (_make_req(), 1, 3),
            (_make_req(req_pool_idx=1, is_chunked=1), 0, 3),
            (_make_req(mamba_pool_idx=main), 1, 2),
            (_make_req(mamba_ping_pong_track_buffer=ping_pong), 1, 1),
            (
                _make_req(
                    mamba_pool_idx=main,
                    mamba_ping_pong_track_buffer=ping_pong,
                ),
                1,
                0,
            ),
            (
                _make_req(
                    req_pool_idx=1,
                    mamba_pool_idx=main,
                    mamba_ping_pong_track_buffer=ping_pong,
                    is_chunked=1,
                ),
                0,
                0,
            ),
        ]
        for req, req_slot_need, mamba_need in cases:
            with self.subTest(req=req):
                self.assertEqual(pool.req_slot_need(req), req_slot_need)
                self.assertEqual(pool.mamba_state_need(req), mamba_need)

        self.assertEqual(pool.req_slots_need([req for req, _, _ in cases]), 4)
        self.assertEqual(pool.mamba_states_need([req for req, _, _ in cases]), 9)

    def test_mamba_demand_without_overlap(self):
        pool = _make_hybrid_pool(
            req_slots=8, mamba_slots=16, enable_extra=True, overlap=False
        )
        main = torch.tensor(11)
        ping_pong = torch.tensor([12])
        self.assertEqual(pool.mamba_state_need(_make_req()), 2)
        self.assertEqual(pool.mamba_state_need(_make_req(mamba_pool_idx=main)), 1)
        self.assertEqual(
            pool.mamba_state_need(_make_req(mamba_ping_pong_track_buffer=ping_pong)),
            1,
        )
        self.assertEqual(
            pool.mamba_state_need(
                _make_req(
                    mamba_pool_idx=main,
                    mamba_ping_pong_track_buffer=ping_pong,
                )
            ),
            0,
        )

    def test_mamba_demand_without_extra_buffer_only_charges_main(self):
        pool = _make_hybrid_pool(
            req_slots=8, mamba_slots=16, enable_extra=False, overlap=True
        )
        self.assertEqual(pool.mamba_state_need(_make_req()), 1)
        self.assertEqual(
            pool.mamba_state_need(_make_req(mamba_pool_idx=torch.tensor(11))),
            0,
        )

    def test_full_kv_page_size_one_separates_current_and_future(self):
        reservation = get_full_kv_reservation(
            prefix_len=7,
            extend_input_len=5,
            page_size=1,
            max_new_tokens_reservation=2,
        )
        self.assertEqual(reservation.current_allocation, 5)
        self.assertEqual(reservation.future_reservation, 2)
        self.assertEqual(reservation.total, 7)

    def test_paged_extend_reservation_matches_allocator_preflight(self):
        page_size = 4
        for prefix_lens, extend_lens in [
            ([0], [4]),
            ([4], [1]),
            ([3], [1]),
            ([3], [2]),
            ([0, 3, 4], [1, 2, 4]),
        ]:
            seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
            allocator_pages = get_num_new_pages(
                seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
                prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
                page_size=page_size,
            )
            with self.subTest(prefix_lens=prefix_lens, extend_lens=extend_lens):
                self.assertEqual(
                    get_extend_allocation_size(
                        prefix_lens=prefix_lens,
                        extend_lens=extend_lens,
                        page_size=page_size,
                    ),
                    allocator_pages * page_size,
                )

    def test_paged_reservation_accounts_only_for_pages_actually_needed(self):
        # Filling the already allocated partial page needs no additional pool unit.
        aligned = get_full_kv_reservation(4, 1, 4)
        unaligned = get_full_kv_reservation(3, 1, 4)
        self.assertEqual(aligned.current_allocation, 4)
        self.assertEqual(unaligned.current_allocation, 0)

        intermediate = get_full_kv_reservation(4, 1, 4, 0)
        final = get_full_kv_reservation(4, 1, 4, 4)
        self.assertEqual(intermediate.future_reservation, 0)
        self.assertEqual(final.future_reservation, 4)

    def test_decode_page_count_keeps_existing_allocator_semantics(self):
        seq_lens = torch.tensor([1, 4, 5, 8, 9], dtype=torch.int64)
        self.assertEqual(
            get_num_new_pages(seq_lens=seq_lens, page_size=4, decode=True),
            3,
        )


class TestBaseReqAllocation(unittest.TestCase):
    def test_base_pool_allocates_new_and_reuses_committed_slots(self):
        pool = ReqToTokenPool.__new__(ReqToTokenPool)
        pool.free_slots = [0]
        new_req = _make_req()
        reused_req = _make_req(req_pool_idx=7, kv_committed_len=1)

        self.assertEqual(pool.alloc([new_req, reused_req]), [0, 7])
        self.assertEqual(pool.free_slots, [])

    def test_base_pool_keeps_reuse_validation(self):
        pool = ReqToTokenPool.__new__(ReqToTokenPool)
        pool.free_slots = [0]
        invalid_reuse = _make_req(req_pool_idx=7)

        with self.assertRaisesRegex(AssertionError, "chunked or have committed KV"):
            pool.alloc([invalid_reuse])


class TestHybridReqAllocationAtomicity(unittest.TestCase):
    def test_req_slot_capacity_failure_mutates_nothing(self):
        pool = _make_hybrid_pool(req_slots=0, mamba_slots=3)
        req = _make_req()
        mamba_free_before = pool.mamba_pool.free_slots.clone()

        self.assertIsNone(pool.alloc([req]))
        self.assertEqual(pool.free_slots, [])
        self.assertTrue(torch.equal(pool.mamba_pool.free_slots, mamba_free_before))
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertIsNone(req.mamba_next_track_idx)

    def test_mamba_capacity_failure_mutates_nothing(self):
        pool = _make_hybrid_pool(req_slots=1, mamba_slots=2)
        req = _make_req()
        req_free_before = list(pool.free_slots)
        mamba_free_before = pool.mamba_pool.free_slots.clone()

        self.assertIsNone(pool.alloc([req]))
        self.assertEqual(pool.free_slots, req_free_before)
        self.assertTrue(torch.equal(pool.mamba_pool.free_slots, mamba_free_before))
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertIsNone(req.mamba_next_track_idx)

    def test_mixed_reused_and_new_requests_charge_only_missing_state(self):
        pool = _make_hybrid_pool(req_slots=1, mamba_slots=3)
        existing_main = torch.tensor(50)
        existing_ping_pong = torch.tensor([51, 52])
        reused = _make_req(
            req_pool_idx=7,
            mamba_pool_idx=existing_main,
            is_chunked=1,
        )
        new = _make_req(mamba_ping_pong_track_buffer=existing_ping_pong)

        self.assertEqual(pool.alloc([reused, new]), [7, 0])
        self.assertEqual(pool.free_slots, [])
        self.assertEqual(pool.mamba_pool.available_size(), 0)
        self.assertIs(reused.mamba_pool_idx, existing_main)
        self.assertIs(new.mamba_ping_pong_track_buffer, existing_ping_pong)

    def test_exact_fit_allocation_succeeds(self):
        pool = _make_hybrid_pool(req_slots=1, mamba_slots=3)
        req = _make_req()

        self.assertEqual(pool.alloc([req]), [0])
        self.assertEqual(pool.free_slots, [])
        self.assertEqual(pool.mamba_pool.available_size(), 0)
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertEqual(len(req.mamba_ping_pong_track_buffer), 2)
        self.assertEqual(req.mamba_next_track_idx, 0)

    def test_non_extra_buffer_allocation_only_consumes_main_state(self):
        pool = _make_hybrid_pool(req_slots=1, mamba_slots=1, enable_extra=False)
        req = _make_req()

        self.assertEqual(pool.alloc([req]), [0])
        self.assertEqual(pool.mamba_pool.available_size(), 0)
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertIsNone(req.mamba_next_track_idx)

    def test_chunk_continuation_reuses_every_mapping(self):
        pool = _make_hybrid_pool(req_slots=2, mamba_slots=4)
        main = torch.tensor(50)
        ping_pong = torch.tensor([51, 52])
        req = _make_req(
            req_pool_idx=7,
            mamba_pool_idx=main,
            mamba_ping_pong_track_buffer=ping_pong,
            mamba_next_track_idx=1,
            is_chunked=1,
        )
        req_free_before = list(pool.free_slots)
        mamba_free_before = pool.mamba_pool.free_slots.clone()

        self.assertEqual(pool.alloc([req]), [7])
        self.assertEqual(pool.free_slots, req_free_before)
        self.assertTrue(torch.equal(pool.mamba_pool.free_slots, mamba_free_before))
        self.assertIs(req.mamba_pool_idx, main)
        self.assertIs(req.mamba_ping_pong_track_buffer, ping_pong)
        self.assertEqual(req.mamba_next_track_idx, 1)

    def test_impossible_post_preflight_failure_rolls_back_and_raises(self):
        pool = _make_hybrid_pool(
            req_slots=1, mamba_slots=3, enable_extra=True, fail_alloc=True
        )
        req = _make_req()
        req_free_before = list(pool.free_slots)
        mamba_free_before = pool.mamba_pool.free_slots.clone()

        with self.assertRaisesRegex(RuntimeError, "HybridReqToTokenPool allocation"):
            pool.alloc([req])

        self.assertEqual(pool.free_slots, req_free_before)
        self.assertTrue(torch.equal(pool.mamba_pool.free_slots, mamba_free_before))
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertIsNone(req.mamba_next_track_idx)

    def test_post_mutation_failure_restores_free_lists_fields_and_mappings(self):
        pool = _make_hybrid_pool(
            req_slots=1,
            mamba_slots=3,
            enable_extra=True,
            fail_ping_mapping=True,
        )
        req = _make_req()
        req_free_before = list(pool.free_slots)
        mamba_free_before = pool.mamba_pool.free_slots.clone()
        main_mapping_before = pool.req_index_to_mamba_index_mapping.clone()
        ping_mapping_before = (
            pool.req_index_to_mamba_ping_pong_track_buffer_mapping.tensor.clone()
        )

        with self.assertRaisesRegex(RuntimeError, "HybridReqToTokenPool allocation"):
            pool.alloc([req])

        self.assertEqual(pool.free_slots, req_free_before)
        self.assertTrue(torch.equal(pool.mamba_pool.free_slots, mamba_free_before))
        self.assertTrue(
            torch.equal(
                pool.req_index_to_mamba_index_mapping,
                main_mapping_before,
            )
        )
        self.assertTrue(
            torch.equal(
                pool.req_index_to_mamba_ping_pong_track_buffer_mapping.tensor,
                ping_mapping_before,
            )
        )
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertIsNone(req.mamba_next_track_idx)


def _make_plain_mamba_cache():
    cache = MambaRadixCache.__new__(MambaRadixCache)
    cache.req_to_token_pool = SimpleNamespace(mamba_pool=_FakeMambaPool())
    cache.token_to_kv_pool_allocator = _FakeAllocator()
    cache.page_size = 1
    cache.disable = False
    cache.enable_mamba_extra_buffer = False
    cache.device = torch.device("cpu")
    cache.key_match_fn = _key_match_page_size1
    cache.get_child_key_fn = get_child_key
    cache.metrics_collector = None
    cache.reset()
    return cache


def _insert(cache, token_ids, mamba_value=None):
    """Insert token_ids with freshly allocated kv indices; returns InsertResult."""
    if mamba_value is None:
        mamba_value = cache.req_to_token_pool.mamba_pool.alloc(1)
    value = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    return cache.insert(
        InsertParams(key=RadixKey(token_ids), value=value, mamba_value=mamba_value)
    )


class TestMambaBaseSeams(unittest.TestCase):
    def test_tree_node_ref_fields_default_to_zero(self):
        node = TreeNode()
        self.assertEqual(node.high_ref, 0)
        self.assertEqual(node.low_ref, 0)
        self.assertEqual(node.tracked_rids, set())

    def test_tree_node_tracked_rids_are_not_shared(self):
        a, b = TreeNode(), TreeNode()
        a.tracked_rids.add("r1")
        self.assertEqual(b.tracked_rids, set())

    def test_insert_result_populates_last_node(self):
        cache = _make_plain_mamba_cache()
        result = _insert(cache, [1, 2, 3])
        self.assertIsNotNone(result.last_node)
        self.assertEqual(result.last_node.key.token_ids, [1, 2, 3])

    def test_insert_result_last_node_none_on_empty_key(self):
        cache = _make_plain_mamba_cache()
        mv = cache.req_to_token_pool.mamba_pool.alloc(1)
        result = cache.insert(
            InsertParams(
                key=RadixKey([]),
                value=torch.tensor([], dtype=torch.int64),
                mamba_value=mv,
            )
        )
        self.assertIsNone(result.last_node)

    def test_insert_result_last_node_on_full_match(self):
        cache = _make_plain_mamba_cache()
        first = _insert(cache, [1, 2, 3])
        again = _insert(cache, [1, 2, 3])
        self.assertIs(again.last_node, first.last_node)

    def test_accounting_hooks_are_noop_on_base(self):
        cache = _make_plain_mamba_cache()
        self.assertIsNone(cache._account_new_node_evictable(TreeNode()))
        self.assertIsNone(cache._account_mamba_refill_evictable(TreeNode()))


def _make_cache(mamba_slots=64):
    cache = RefAwareMambaRadixCache.__new__(RefAwareMambaRadixCache)
    cache._init_ref_aware_state(
        SimpleNamespace(high_priority_threshold=1, enable_priority_scheduling=True)
    )
    cache.req_to_token_pool = SimpleNamespace(
        mamba_pool=_FakeMambaPool(size=mamba_slots)
    )
    cache.token_to_kv_pool_allocator = _FakeAllocator()
    cache.page_size = 1
    cache.disable = False
    cache.enable_mamba_extra_buffer = False
    cache.device = torch.device("cpu")
    cache.key_match_fn = _key_match_page_size1
    cache.get_child_key_fn = get_child_key
    cache.is_eagle = False
    cache.metrics_collector = None
    cache.reset()
    return cache


def _assert_conservation(testcase, cache):
    """Invariant I1: per-tier counters sum to the legacy counters."""
    testcase.assertEqual(
        cache.full_unused_evictable_size_
        + cache.full_low_ref_evictable_size_
        + cache.full_high_ref_evictable_size_,
        cache.full_evictable_size_,
    )
    testcase.assertEqual(
        cache.mamba_unused_evictable_size_
        + cache.mamba_low_ref_evictable_size_
        + cache.mamba_high_ref_evictable_size_,
        cache.mamba_evictable_size_,
    )


def _set_ref(cache, node, *, high=0, low=0):
    """Move a node into a tier via the real ref-increment path."""
    for _ in range(high):
        cache._inc_priority_ref_single(node, True)
    for _ in range(low):
        cache._inc_priority_ref_single(node, False)


class TestMambaTierAccounting(unittest.TestCase):
    def test_new_insert_lands_in_unused_tier(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3])
        self.assertEqual(cache.full_unused_evictable_size_, 3)
        self.assertEqual(cache.mamba_unused_evictable_size_, 1)
        _assert_conservation(self, cache)

    def test_ref_moves_both_resources_across_tiers(self):
        cache = _make_cache()
        result = _insert(cache, [1, 2, 3])
        _set_ref(cache, result.last_node, high=1)
        self.assertEqual(cache.full_unused_evictable_size_, 0)
        self.assertEqual(cache.full_high_ref_evictable_size_, 3)
        self.assertEqual(cache.mamba_high_ref_evictable_size_, 1)
        _assert_conservation(self, cache)

    def test_lock_ref_excludes_from_tier_counters(self):
        cache = _make_cache()
        result = _insert(cache, [1, 2, 3])
        node = result.last_node
        _set_ref(cache, node, low=1)
        cache.inc_lock_ref(node)
        self.assertEqual(cache.full_low_ref_evictable_size_, 0)
        self.assertEqual(cache.mamba_low_ref_evictable_size_, 0)
        _assert_conservation(self, cache)
        cache.dec_lock_ref(node)
        self.assertEqual(cache.full_low_ref_evictable_size_, 3)
        self.assertEqual(cache.mamba_low_ref_evictable_size_, 1)
        _assert_conservation(self, cache)

    def test_split_inherits_refs_and_conserves_counters(self):
        cache = _make_cache()
        r1 = _insert(cache, [1, 2, 3, 4])
        _set_ref(cache, r1.last_node, high=1)
        cache.register_ref(
            SimpleNamespace(rid="r1", priority=1, last_node=r1.last_node)
        )
        # Insert a diverging key to force a split of [1,2,3,4] at len 2
        _insert(cache, [1, 2, 9, 9])
        # find the split parent: root -> child keyed by first token
        parent = cache.root_node.children[cache.get_child_key_fn(RadixKey([1, 2]))]
        self.assertEqual(parent.key.token_ids, [1, 2])
        self.assertEqual(parent.high_ref, r1.last_node.high_ref)
        self.assertIn("r1", parent.tracked_rids)
        self.assertIn(parent, cache.rid_to_ref_info["r1"].nodes)
        _assert_conservation(self, cache)

    def test_tombstone_keeps_full_tier_and_drops_mamba(self):
        cache = _make_cache()
        r1 = _insert(cache, [1, 2, 3])
        r2 = _insert(cache, [1, 2, 3, 4, 5])  # make [1,2,3] an internal node
        _set_ref(cache, r1.last_node, high=1)
        # Also ref the [4,5] suffix leaf so the tree has no unused/low-tier
        # mamba node competing with the internal node under test: with both
        # nodes in the high tier, the high-tier pass walks the mamba LRU list
        # and must hit [1,2,3] (older: touched, then split off, before [4,5]
        # was inserted) first, exercising the tombstone branch.
        _set_ref(cache, r2.last_node, high=1)
        before_full_high = cache.full_high_ref_evictable_size_
        with cache.scoped_evict(allow_low=True, allow_high=True):
            cache.evict(EvictParams(num_tokens=0, mamba_num=1))  # tombstones internal
        self.assertIsNone(r1.last_node.mamba_value)
        self.assertEqual(r1.last_node.high_ref, 1)  # ref survives tombstone
        self.assertEqual(cache.full_high_ref_evictable_size_, before_full_high)
        _assert_conservation(self, cache)

    def test_mamba_refill_after_tombstone_rejoins_its_tier(self):
        cache = _make_cache()
        r1 = _insert(cache, [1, 2, 3])
        r2 = _insert(cache, [1, 2, 3, 4, 5])
        node = r1.last_node
        _set_ref(cache, node, high=1)
        # See test_tombstone_keeps_full_tier_and_drops_mamba: ref the suffix
        # leaf too so no unused/low mamba node can preempt the internal node.
        _set_ref(cache, r2.last_node, high=1)
        with cache.scoped_evict(allow_low=True, allow_high=True):
            cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertIsNone(node.mamba_value)
        mamba_high_before = cache.mamba_high_ref_evictable_size_
        _insert(cache, [1, 2, 3])  # refill tombstone's mamba_value
        self.assertIsNotNone(node.mamba_value)
        self.assertEqual(cache.mamba_high_ref_evictable_size_, mamba_high_before + 1)
        _assert_conservation(self, cache)

    def test_ref_change_while_locked_rebalances_on_unlock(self):
        cache = _make_cache()
        result = _insert(cache, [1, 2, 3])
        node = result.last_node
        _set_ref(cache, node, low=1)
        cache.inc_lock_ref(node)
        # Locked: node contributes to no tier counter.
        self.assertEqual(cache.full_low_ref_evictable_size_, 0)
        self.assertEqual(cache.mamba_low_ref_evictable_size_, 0)

        # Tier changes while locked must not touch the counters...
        cache._inc_priority_ref_single(node, True)  # low -> high
        cache._dec_priority_ref_single(node, False)  # drop the low ref
        self.assertEqual(cache.full_high_ref_evictable_size_, 0)
        self.assertEqual(cache.mamba_high_ref_evictable_size_, 0)
        _assert_conservation(self, cache)

        # ...and unlock refills the NEW tier, leaving the old tier at zero.
        cache.dec_lock_ref(node)
        self.assertEqual(cache.full_low_ref_evictable_size_, 0)
        self.assertEqual(cache.full_high_ref_evictable_size_, 3)
        self.assertEqual(cache.mamba_high_ref_evictable_size_, 1)
        _assert_conservation(self, cache)
        cache._sanity_check_tier_counters()


class TestMambaRidLifecycle(unittest.TestCase):
    def test_register_update_release_roundtrip(self):
        cache = _make_cache()
        result = _insert(cache, [1, 2, 3])
        req = SimpleNamespace(rid="r1", priority=0, last_node=result.last_node)
        cache.register_ref(req)
        node = result.last_node
        self.assertEqual(node.low_ref, 1)
        self.assertEqual(cache.full_low_ref_evictable_size_, 3)

        ok, _ = cache.update_ref("r1", 5)
        self.assertTrue(ok)
        self.assertEqual(node.high_ref, 1)
        self.assertEqual(node.low_ref, 0)
        self.assertEqual(cache.full_high_ref_evictable_size_, 3)

        ok, _ = cache.release_ref("r1")
        self.assertTrue(ok)
        self.assertEqual(node.high_ref, 0)
        self.assertEqual(cache.full_unused_evictable_size_, 3)
        _assert_conservation(self, cache)

    def test_delete_leaf_untracks_rid(self):
        cache = _make_cache()
        result = _insert(cache, [1, 2, 3])
        cache.register_ref(
            SimpleNamespace(rid="r1", priority=1, last_node=result.last_node)
        )
        with cache.scoped_evict(allow_low=True, allow_high=True):
            cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(cache.rid_to_ref_info["r1"].nodes, set())
        _assert_conservation(self, cache)


class TestMambaTieredEviction(unittest.TestCase):
    def _seed_three_tiers(self, cache):
        """Three disjoint 4-token chains: unused / low / high."""
        unused = _insert(cache, [1, 2, 3, 4]).last_node
        low = _insert(cache, [11, 12, 13, 14]).last_node
        high = _insert(cache, [21, 22, 23, 24]).last_node
        _set_ref(cache, low, low=1)
        _set_ref(cache, high, high=1)
        return unused, low, high

    def test_full_evicts_unused_before_low(self):
        cache = _make_cache()
        unused, low, high = self._seed_three_tiers(cache)
        n = cache.evict_full(4, allow_low=True, allow_high=False)
        self.assertEqual(n, 4)
        self.assertNotIn(unused.id, cache.full_lru_list.cache)
        self.assertIn(low.id, cache.full_lru_list.cache)
        _assert_conservation(self, cache)

    def test_full_spills_into_low_but_never_high(self):
        cache = _make_cache()
        unused, low, high = self._seed_three_tiers(cache)
        n = cache.evict_full(12, allow_low=True, allow_high=False)
        self.assertEqual(n, 8)  # unused + low only
        self.assertIn(high.id, cache.full_lru_list.cache)
        _assert_conservation(self, cache)

    def test_full_reaches_high_when_allowed(self):
        cache = _make_cache()
        self._seed_three_tiers(cache)
        n = cache.evict_full(12, allow_low=True, allow_high=True)
        self.assertEqual(n, 12)
        _assert_conservation(self, cache)

    def test_scope_stack_drives_default_evict(self):
        cache = _make_cache()
        unused, low, high = self._seed_three_tiers(cache)
        result = cache.evict(EvictParams(num_tokens=12))
        self.assertEqual(result.num_tokens_evicted, 8)  # default scope: no high
        with cache.scoped_evict(allow_low=True, allow_high=True):
            result = cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(result.num_tokens_evicted, 4)
        _assert_conservation(self, cache)

    def test_mamba_evicts_unused_before_low_and_skips_high(self):
        cache = _make_cache()
        # Build internal nodes so evict_mamba tombstones instead of deleting:
        # chain A: [1,2] (unused, internal) -> [1,2,3,4] (leaf, high)
        a_mid = _insert(cache, [1, 2]).last_node
        a_leaf = _insert(cache, [1, 2, 3, 4]).last_node
        _set_ref(cache, a_leaf, high=1)
        # chain B: [5,6] (low, internal) -> [5,6,7,8] (leaf, high)
        b_mid = _insert(cache, [5, 6]).last_node
        b_leaf = _insert(cache, [5, 6, 7, 8]).last_node
        _set_ref(cache, b_mid, low=1)
        _set_ref(cache, b_leaf, high=1)

        n = cache.evict_mamba(1, allow_low=True, allow_high=False)
        self.assertEqual(n, 1)
        self.assertIsNone(a_mid.mamba_value)  # unused tombstoned first
        self.assertIsNotNone(b_mid.mamba_value)  # low survives round 1

        n = cache.evict_mamba(2, allow_low=True, allow_high=False)
        self.assertEqual(n, 1)  # only low available
        self.assertIsNone(b_mid.mamba_value)
        self.assertIsNotNone(a_leaf.mamba_value)  # high never touched
        _assert_conservation(self, cache)

    def test_evict_tiered_signature_matches_decode_path(self):
        cache = _make_cache()
        self._seed_three_tiers(cache)
        n = cache._evict_tiered(4, allow_low=False, allow_high=False)
        self.assertEqual(n, 4)  # unused tier always allowed
        _assert_conservation(self, cache)


class TestMambaTombstoneBoundary(unittest.TestCase):
    def test_safe_eviction_preserves_every_reusable_high_node_and_rid(self):
        cache = _make_cache()
        high_parent = _insert(cache, [1, 2]).last_node
        high_leaf = _insert(cache, [1, 2, 3, 4]).last_node
        cache.register_ref(
            SimpleNamespace(rid="high-chain", priority=1, last_node=high_leaf)
        )
        safe_leaf = _insert(cache, [9, 10]).last_node

        tracked_nodes = set(cache.rid_to_ref_info["high-chain"].nodes)
        self.assertEqual(tracked_nodes, {high_parent, high_leaf})
        high_states = {
            high_parent: high_parent.mamba_value,
            high_leaf: high_leaf.mamba_value,
        }
        high_tiers_before = (
            cache.full_high_ref_evictable_size_,
            cache.mamba_high_ref_evictable_size_,
        )

        self.assertEqual(cache.evict_mamba(99, allow_low=True, allow_high=False), 1)
        self.assertNotIn(safe_leaf.id, cache.mamba_lru_list.cache)
        self.assertEqual(cache.evict_full(99, allow_low=True, allow_high=False), 0)

        for node, state in high_states.items():
            self.assertIs(node.mamba_value, state)
            self.assertIn(node.id, cache.full_lru_list.cache)
            self.assertIn(node.id, cache.mamba_lru_list.cache)
            self.assertIn("high-chain", node.tracked_rids)
        self.assertIs(high_parent.children[get_child_key(RadixKey([3, 4]))], high_leaf)
        self.assertEqual(cache.rid_to_ref_info["high-chain"].nodes, tracked_nodes)
        self.assertEqual(
            (
                cache.full_high_ref_evictable_size_,
                cache.mamba_high_ref_evictable_size_,
            ),
            high_tiers_before,
        )
        _assert_conservation(self, cache)
        cache._sanity_check_tier_counters()

    def test_high_metadata_tombstone_cascades_after_safe_child_deletion(self):
        cache = _make_cache()
        tombstone = _insert(cache, [1, 2]).last_node
        safe_child = _insert(cache, [1, 2, 3, 4]).last_node
        cache.register_ref(
            SimpleNamespace(rid="stale-high", priority=1, last_node=tombstone)
        )
        _set_ref(cache, safe_child, high=1)

        # Make the internal high-ref node a tombstone while its child keeps it
        # structurally reachable.  The child is temporarily high only to make
        # the high-tier LRU pass select the older internal node first.
        self.assertEqual(cache.evict_mamba(1, allow_low=False, allow_high=True), 1)
        self.assertIsNone(tombstone.mamba_value)
        self.assertIn("stale-high", tombstone.tracked_rids)
        cache._dec_priority_ref_single(safe_child, True)

        full_before = cache.full_evictable_size_
        mamba_before = cache.mamba_evictable_size_
        freed_full_before = cache.token_to_kv_pool_allocator.freed_total
        freed_mamba_before = len(cache.req_to_token_pool.mamba_pool.freed)

        # Deleting the now-unused child exposes a childless tombstone.  Its
        # stale high-ref metadata must not prevent structural GC.
        self.assertEqual(cache.evict_full(1, allow_low=True, allow_high=False), 4)

        self.assertEqual(cache.root_node.children, {})
        self.assertEqual(full_before, 4)
        self.assertEqual(mamba_before, 1)
        self.assertEqual(
            cache.token_to_kv_pool_allocator.freed_total - freed_full_before, 4
        )
        self.assertEqual(
            len(cache.req_to_token_pool.mamba_pool.freed) - freed_mamba_before, 1
        )
        self.assertEqual(cache.rid_to_ref_info["stale-high"].nodes, set())
        self.assertEqual(tombstone.tracked_rids, set())
        _assert_conservation(self, cache)
        cache._sanity_check_tier_counters()

    def test_unused_high_authorization_does_not_delete_reusable_high_parent(self):
        cache = _make_cache()
        reusable_high = _insert(cache, [1, 2]).last_node
        safe_child = _insert(cache, [1, 2, 3, 4]).last_node
        cache.register_ref(
            SimpleNamespace(rid="reusable-high", priority=1, last_node=reusable_high)
        )
        high_state = reusable_high.mamba_value
        high_full_value = reusable_high.value

        # Force a one-token allocator shortfall.  The safe child satisfies it
        # at radix-leaf granularity, so the available high authorization must
        # remain unused and cannot turn the still-reusable parent into GC.
        cache.token_to_kv_pool_allocator.size = (
            cache.token_to_kv_pool_allocator._next
            - cache.token_to_kv_pool_allocator.freed_total
        )
        high_evicted = evict_from_tree_cache(cache, 1, high_authorization=1)

        self.assertEqual(high_evicted, 0)
        self.assertIs(reusable_high.mamba_value, high_state)
        self.assertIs(reusable_high.value, high_full_value)
        self.assertEqual(reusable_high.children, {})
        self.assertIn(reusable_high.id, cache.full_lru_list.cache)
        self.assertIn(reusable_high.id, cache.mamba_lru_list.cache)
        self.assertEqual(cache.rid_to_ref_info["reusable-high"].nodes, {reusable_high})
        self.assertEqual(reusable_high.tracked_rids, {"reusable-high"})
        self.assertNotIn(safe_child.id, cache.full_lru_list.cache)
        _assert_conservation(self, cache)
        cache._sanity_check_tier_counters()


class TestBestEffortChunkStash(unittest.TestCase):
    def _make_live_req_with_only_high_evictable(self, *, priority=0):
        cache = _make_cache(mamba_slots=5)
        prefix = _insert(cache, [1, 2]).last_node
        high = _insert(cache, [9, 10]).last_node
        _set_ref(cache, high, high=1)
        cache.inc_lock_ref(prefix)

        req_pool = _FakeStashReqPool(cache.req_to_token_pool.mamba_pool)
        cache.req_to_token_pool = req_pool
        live_main = req_pool.mamba_pool.alloc(1)[0]
        live_ping_pong = req_pool.mamba_pool.alloc(2)
        self.assertEqual(req_pool.mamba_pool.available_size(), 0)
        req_pool.req_index_to_mamba_index_mapping[0] = live_main
        suffix_indices = cache.token_to_kv_pool_allocator.alloc(2)
        live_indices = torch.cat([prefix.value, suffix_indices])
        req_pool.req_to_token[0, :4] = live_indices

        req = SimpleNamespace(
            rid="live-chunk",
            priority=priority,
            req_pool_idx=0,
            fill_ids=[1, 2, 3, 4],
            origin_input_ids=[1, 2, 3, 4, 5, 6],
            output_ids=[],
            extra_key=None,
            prefix_indices=prefix.value.clone(),
            cache_protected_len=2,
            mamba_last_track_seqlen=4,
            mamba_pool_idx=live_main,
            mamba_ping_pong_track_buffer=live_ping_pong,
            mamba_next_track_idx=1,
            last_node=prefix,
        )
        return cache, req, prefix, high, live_indices

    def test_only_high_ref_capacity_returns_live_fallback_without_mutation(self):
        cache, req, prefix, high, live_indices = (
            self._make_live_req_with_only_high_evictable()
        )
        high_state = high.mamba_value
        high_full_before = cache.full_high_ref_evictable_size_
        high_mamba_before = cache.mamba_high_ref_evictable_size_
        main_before = req.mamba_pool_idx
        ping_pong_before = req.mamba_ping_pong_track_buffer
        lock_before = (prefix.full_lock_ref, prefix.mamba_lock_ref)

        result = cache.cache_unfinished_req(req, chunked=True)

        self.assertIs(result, MambaChunkStashResult.LIVE_PREFIX_FALLBACK)
        self.assertTrue(torch.equal(req.prefix_indices, live_indices))
        self.assertIs(req.last_node, prefix)
        self.assertEqual(req.cache_protected_len, 2)
        self.assertEqual(req.mamba_last_track_seqlen, 4)
        self.assertIs(req.mamba_pool_idx, main_before)
        self.assertIs(req.mamba_ping_pong_track_buffer, ping_pong_before)
        self.assertEqual(req.mamba_next_track_idx, 1)
        self.assertEqual((prefix.full_lock_ref, prefix.mamba_lock_ref), lock_before)
        self.assertIs(high.mamba_value, high_state)
        self.assertEqual(cache.full_high_ref_evictable_size_, high_full_before)
        self.assertEqual(cache.mamba_high_ref_evictable_size_, high_mamba_before)
        self.assertEqual(cache.req_to_token_pool.mamba_state_need(req), 0)

    def test_hp_stash_has_the_same_no_high_fallback(self):
        cache, req, _prefix, high, _ = self._make_live_req_with_only_high_evictable(
            priority=99
        )
        high_state = high.mamba_value

        # Even an enclosing HP allocation scope must not leak into stash.
        with cache.scoped_evict(allow_low=True, allow_high=True):
            result = cache.cache_unfinished_req(req, chunked=True)

        self.assertIs(result, MambaChunkStashResult.LIVE_PREFIX_FALLBACK)
        self.assertIs(high.mamba_value, high_state)
        self.assertEqual(cache.mamba_high_ref_evictable_size_, 1)

    def test_later_chunk_retries_and_inserts_a_longer_snapshot(self):
        cache, req, old_node, high, _ = self._make_live_req_with_only_high_evictable()
        self.assertIs(
            cache.cache_unfinished_req(req, chunked=True),
            MambaChunkStashResult.LIVE_PREFIX_FALLBACK,
        )

        cache.req_to_token_pool.mamba_pool.free(torch.tensor([17]))
        req.fill_ids.extend([5, 6])
        suffix_indices = cache.token_to_kv_pool_allocator.alloc(2)
        cache.req_to_token_pool.req_to_token[0, 4:6] = suffix_indices

        result = cache.cache_unfinished_req(req, chunked=True)

        self.assertIs(result, MambaChunkStashResult.SNAPSHOT_INSERTED)
        self.assertEqual(len(req.prefix_indices), 6)
        self.assertEqual(req.cache_protected_len, 6)
        self.assertIsNot(req.last_node, old_node)
        # The new descendant lock continues to protect its full-KV ancestors,
        # while Mamba ownership transfers to the new snapshot node only.
        self.assertEqual(old_node.full_lock_ref, 1)
        self.assertEqual(old_node.mamba_lock_ref, 0)
        self.assertGreater(req.last_node.full_lock_ref, 0)
        self.assertGreater(req.last_node.mamba_lock_ref, 0)
        self.assertIsNotNone(high.mamba_value)
        self.assertIsNone(req.mamba_last_track_seqlen)

    def test_safe_low_ref_slot_is_used_for_snapshot(self):
        cache = _make_cache(mamba_slots=3)
        prefix = _insert(cache, [1]).last_node
        low = _insert(cache, [8]).last_node
        _set_ref(cache, low, low=1)
        cache.inc_lock_ref(prefix)
        req_pool = _FakeStashReqPool(cache.req_to_token_pool.mamba_pool)
        cache.req_to_token_pool = req_pool
        main = req_pool.mamba_pool.alloc(1)[0]
        req_pool.req_index_to_mamba_index_mapping[0] = main
        suffix = cache.token_to_kv_pool_allocator.alloc(1)
        req_pool.req_to_token[0, :2] = torch.cat([prefix.value, suffix])
        req = SimpleNamespace(
            rid="safe-stash",
            priority=0,
            req_pool_idx=0,
            fill_ids=[1, 2],
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            extra_key=None,
            prefix_indices=prefix.value.clone(),
            cache_protected_len=1,
            mamba_last_track_seqlen=2,
            mamba_pool_idx=main,
            mamba_ping_pong_track_buffer=torch.tensor([30, 31]),
            mamba_next_track_idx=0,
            last_node=prefix,
        )

        result = cache.cache_unfinished_req(req, chunked=True)

        self.assertIs(result, MambaChunkStashResult.SNAPSHOT_INSERTED)
        self.assertNotIn(low.id, cache.mamba_lru_list.cache)
        self.assertEqual(req.cache_protected_len, 2)

    def test_fallback_then_hp_preemption_releases_all_live_ownership(self):
        from sglang.srt.managers.scheduler import Scheduler

        cache, req, prefix, high, _ = self._make_live_req_with_only_high_evictable()
        self.assertIs(
            cache.cache_unfinished_req(req, chunked=True),
            MambaChunkStashResult.LIVE_PREFIX_FALLBACK,
        )
        high_state = high.mamba_value
        freed_kv_before = cache.token_to_kv_pool_allocator.freed_total
        reset_calls = []
        req.pop_committed_kv_cache = lambda: len(req.fill_ids)
        req.pop_overallocated_kv_cache = lambda: (
            len(req.fill_ids),
            len(req.fill_ids),
        )
        req.reset_for_retract = lambda: reset_calls.append(True)
        req.kv_committed_len = len(req.fill_ids)
        req.kv_allocated_len = len(req.fill_ids)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache
        scheduler.chunked_req = req
        with unittest.mock.patch(
            "sglang.srt.mem_cache.common.get_global_server_args",
            return_value=SimpleNamespace(page_size=1, speculative_algorithm=None),
        ):
            scheduler._reclaim_deferred_chunk_for_high(req)

        self.assertIsNone(scheduler.chunked_req)
        self.assertEqual(reset_calls, [True])
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertEqual(prefix.full_lock_ref, 0)
        self.assertEqual(prefix.mamba_lock_ref, 0)
        self.assertGreater(
            cache.token_to_kv_pool_allocator.freed_total, freed_kv_before
        )
        self.assertIs(high.mamba_value, high_state)


class TestNoHiddenEscalation(unittest.TestCase):
    def _drain_mamba_pool(self, cache):
        pool = cache.req_to_token_pool.mamba_pool
        while pool.available_size() > 0:
            pool.alloc(1)

    def test_fork_never_escalates_to_high_tier(self):
        cache = _make_cache()
        r = _insert(cache, [1, 2, 3])
        _set_ref(cache, r.last_node, high=1)  # only evictable mamba is high
        self._drain_mamba_pool(cache)
        src = r.last_node.mamba_value
        with self.assertRaisesRegex(AssertionError, "Can not alloc mamba cache"):
            cache._fork_mamba_with_evict(src)
        self.assertIsNotNone(r.last_node.mamba_value)

    def test_cow_allocator_never_escalates_to_high_tier(self):
        cache = _make_cache()
        r_hot = _insert(cache, [1, 2, 3])
        _set_ref(cache, r_hot.last_node, high=1)
        r_target = _insert(cache, [9, 8, 7])
        _set_ref(cache, r_target.last_node, high=1)
        self._drain_mamba_pool(cache)
        with self.assertRaisesRegex(AssertionError, "Can not alloc mamba cache"):
            cache._alloc_mamba_slot_with_evict(r_target.last_node)
        self.assertIsNotNone(r_target.last_node.mamba_value)


if __name__ == "__main__":
    unittest.main()

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
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    get_child_key,
)
from sglang.srt.mem_cache.ref_aware_mamba_radix_cache import RefAwareMambaRadixCache


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


def _make_cache():
    cache = RefAwareMambaRadixCache.__new__(RefAwareMambaRadixCache)
    cache._init_ref_aware_state(
        SimpleNamespace(high_priority_threshold=1, enable_priority_scheduling=True)
    )
    cache.req_to_token_pool = SimpleNamespace(mamba_pool=_FakeMambaPool())
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


class TestOOMEscalation(unittest.TestCase):
    def _drain_mamba_pool(self, cache):
        pool = cache.req_to_token_pool.mamba_pool
        while pool.available_size() > 0:
            pool.alloc(1)

    def test_fork_escalates_to_high_tier_before_asserting(self):
        cache = _make_cache()
        r = _insert(cache, [1, 2, 3])
        _set_ref(cache, r.last_node, high=1)  # only evictable mamba is high
        self._drain_mamba_pool(cache)
        src = r.last_node.mamba_value
        # default-scope eviction finds nothing; escalation must free the
        # high-ref node's slot and let fork succeed
        forked = cache._fork_mamba_with_evict(src)
        self.assertIsNotNone(forked)

    def test_cow_alloc_escalates_to_high_tier(self):
        cache = _make_cache()
        r_hot = _insert(cache, [1, 2, 3])
        _set_ref(cache, r_hot.last_node, high=1)
        r_target = _insert(cache, [9, 8, 7])
        _set_ref(cache, r_target.last_node, high=1)
        self._drain_mamba_pool(cache)
        # target node is lock-protected inside the seam; the OTHER high node
        # gets evicted
        dst = cache._alloc_mamba_slot_with_evict(r_target.last_node)
        self.assertIsNotNone(dst)
        self.assertIsNotNone(r_target.last_node.mamba_value)


class TestCommonEscalatedEvict(unittest.TestCase):
    def test_escalated_helper_is_noop_for_plain_cache(self):
        from sglang.srt.mem_cache.common import _escalated_mamba_evict

        cache = _make_plain_mamba_cache()
        self.assertFalse(_escalated_mamba_evict(cache, mamba_num=1))

    def test_escalated_helper_evicts_high_tier(self):
        from sglang.srt.mem_cache.common import _escalated_mamba_evict

        cache = _make_cache()
        r = _insert(cache, [1, 2, 3])
        _set_ref(cache, r.last_node, high=1)
        self.assertTrue(_escalated_mamba_evict(cache, mamba_num=1))
        self.assertEqual(cache.mamba_high_ref_evictable_size_, 0)


class TestPrefillAdderMambaGate(unittest.TestCase):
    def _make_adder(self, *, mamba_available, mamba_low, mamba_high):
        from sglang.srt.managers.schedule_policy import PrefillAdder

        adder = PrefillAdder.__new__(PrefillAdder)
        cache = _make_cache()
        cache.mamba_low_ref_evictable_size_ = mamba_low
        cache.mamba_high_ref_evictable_size_ = mamba_high
        cache.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(available_size=lambda: mamba_available)
        )
        # generous full-token budget so only the mamba gate can reject
        cache.full_unused_evictable_size_ = 10**6
        cache.full_evictable_size_ = 10**6
        adder.tree_cache = cache
        adder.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 10**6)
        adder.rem_total_token_offset = 0
        adder.is_hybrid_ssm_cache = True
        adder.enable_ref_aware_kv_buffer = True
        return adder

    def test_lp_rejected_when_mamba_budget_short(self):
        # MAMBA_STATE_PER_REQ_PREFIX_CACHE == 3; LP budget = 0 + 2 = 2 < 3
        adder = self._make_adder(mamba_available=0, mamba_low=2, mamba_high=50)
        req = SimpleNamespace(priority=0)
        self.assertFalse(
            adder._can_admit_ref_aware_req(req, req_is_high=False, total_tokens=10)
        )

    def test_hp_passes_gate_via_high_tier(self):
        adder = self._make_adder(mamba_available=0, mamba_low=2, mamba_high=50)
        req = SimpleNamespace(priority=1)
        self.assertTrue(
            adder._can_admit_ref_aware_req(req, req_is_high=True, total_tokens=10)
        )

    def test_lp_passes_when_mamba_budget_sufficient(self):
        adder = self._make_adder(mamba_available=3, mamba_low=0, mamba_high=0)
        req = SimpleNamespace(priority=0)
        self.assertTrue(
            adder._can_admit_ref_aware_req(req, req_is_high=False, total_tokens=10)
        )

    def test_non_ssm_cache_skips_mamba_gate(self):
        adder = self._make_adder(mamba_available=0, mamba_low=0, mamba_high=0)
        adder.is_hybrid_ssm_cache = False
        req = SimpleNamespace(priority=0)
        self.assertTrue(
            adder._can_admit_ref_aware_req(req, req_is_high=False, total_tokens=10)
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for RefAwareHiRadixCache tiered eviction."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.ref_aware_hiradix_cache import (
    RefAwareHiRadixCache,
    TIER_HIGH_REF,
    TIER_LOW_REF,
    TIER_UNUSED,
    _classify_node_tier,
)


class TestClassifyNodeTier(unittest.TestCase):
    def test_unused(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 0
        assert _classify_node_tier(node) == TIER_UNUSED

    def test_low_ref(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 1
        assert _classify_node_tier(node) == TIER_LOW_REF

    def test_high_ref(self):
        node = TreeNode()
        node.high_ref = 1
        node.low_ref = 0
        assert _classify_node_tier(node) == TIER_HIGH_REF

    def test_high_ref_overrides_low(self):
        node = TreeNode()
        node.high_ref = 1
        node.low_ref = 5
        assert _classify_node_tier(node) == TIER_HIGH_REF


class TestTreeNodeRefFields(unittest.TestCase):
    def test_default_zero(self):
        node = TreeNode()
        assert node.high_ref == 0
        assert node.low_ref == 0


class TestRefAwareTierAccounting(unittest.TestCase):
    def _make_cache(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        return cache

    def _make_node(self, token_ids):
        node = TreeNode()
        node.key = RadixKey(token_ids)
        node.value = torch.tensor(token_ids, dtype=torch.int64)
        node.parent = None
        return node

    def test_new_evictable_node_starts_in_unused_tier(self):
        cache = self._make_cache()
        node = self._make_node([1, 2, 3, 4])
        node.parent = cache.root_node

        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        self.assertEqual(cache.unused_evictable_size_, 4)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertIn(node, cache.unused_evictable_leaves)
        self.assertEqual(
            cache.evictable_size_by_tier(allow_low=False, allow_high=False), 4
        )

    def test_ref_tier_move_preserves_total_evictable_tokens(self):
        cache = self._make_cache()
        node = self._make_node([1, 2, 3, 4])
        node.parent = cache.root_node

        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)
        cache._inc_priority_ref_single(node, is_high=False)

        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 4)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertIn(node, cache.low_ref_evictable_leaves)
        self.assertEqual(
            cache.evictable_size_by_tier(allow_low=True, allow_high=False), 4
        )

        cache._inc_priority_ref_single(node, is_high=True)

        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 4)
        self.assertIn(node, cache.high_ref_evictable_leaves)
        self.assertEqual(
            cache.evictable_size_by_tier(allow_low=True, allow_high=True), 4
        )


class TestRefAwareRegisterRef(unittest.TestCase):
    def _make_cache(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        cache.rid_to_ref_info = {}
        return cache

    def _append_node(self, parent, token_ids):
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(token_ids)
        node.value = torch.tensor(token_ids, dtype=torch.int64)
        node.children = {}
        parent.children[token_ids[0] if token_ids else 0] = node
        return node

    def test_register_ref_only_adds_new_suffix_from_last_node(self):
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])
        c = self._append_node(b, [9, 10, 11, 12])

        req = SimpleNamespace(rid="r1", priority=1, last_node=c)
        cache.register_ref(req)

        self.assertEqual(a.high_ref, 1)
        self.assertEqual(b.high_ref, 1)
        self.assertEqual(c.high_ref, 1)
        self.assertEqual(len(cache.rid_to_ref_info["r1"].nodes), 3)

        d = self._append_node(c, [13, 14, 15, 16])
        req.last_node = d
        cache.register_ref(req)

        self.assertEqual(a.high_ref, 1)
        self.assertEqual(b.high_ref, 1)
        self.assertEqual(c.high_ref, 1)
        self.assertEqual(d.high_ref, 1)
        self.assertEqual(len(cache.rid_to_ref_info["r1"].nodes), 4)


class _DummyHostPool:
    def __init__(self, available_size: int):
        self._available_size = available_size

    def available_size(self):
        return self._available_size


class _DummyCacheController:
    def __init__(self, available_host_tokens: int):
        self.mem_pool_host = _DummyHostPool(available_host_tokens)
        self.write_policy = "write_back"
        self.evicted_host_lengths = []

    def write(self, device_indices, node_id, **_kwargs):
        return None

    def evict_host(self, host_indices):
        self.evicted_host_lengths.append(len(host_indices))
        return len(host_indices)


class TestRefAwareHostSafety(unittest.TestCase):
    def _make_cache(self, available_host_tokens: int = 0):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        cache.evictable_host_leaves = set()
        cache.rid_to_ref_info = {}
        cache.ongoing_write_through = {}
        cache.eviction_strategy = SimpleNamespace(
            get_priority=lambda node: node.last_access_time
        )
        cache.cache_controller = _DummyCacheController(available_host_tokens)
        return cache

    def _append_node(
        self,
        parent,
        token_ids,
        *,
        evicted: bool = False,
        backuped: bool = False,
        high_ref: int = 0,
        low_ref: int = 0,
    ):
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(token_ids)
        node.value = None if evicted else torch.tensor(token_ids, dtype=torch.int64)
        node.host_value = (
            torch.tensor(token_ids, dtype=torch.int64) if backuped else None
        )
        node.children = {}
        node.high_ref = high_ref
        node.low_ref = low_ref
        parent.children[token_ids[0] if token_ids else 0] = node
        return node

    def test_high_ref_host_safe_evictable_size_blocks_when_only_high_host_can_be_dropped(
        self,
    ):
        cache = self._make_cache(available_host_tokens=0)
        host_high = self._append_node(
            cache.root_node,
            [1, 2, 3, 4],
            evicted=True,
            backuped=True,
            high_ref=1,
        )
        gpu_high = self._append_node(
            cache.root_node,
            [5, 6, 7, 8],
            evicted=False,
            backuped=False,
            high_ref=1,
        )
        cache.evictable_host_leaves.add(host_high)
        cache.high_ref_evictable_leaves.add(gpu_high)
        cache.high_ref_evictable_size_ = 4

        self.assertEqual(cache.high_ref_host_safe_evictable_size(), 0)
        self.assertEqual(
            cache.safe_evictable_size_by_tier(allow_low=True, allow_high=True), 0
        )

    def test_high_ref_host_safe_evictable_size_counts_already_backuped_tokens(self):
        cache = self._make_cache(available_host_tokens=0)
        gpu_high = self._append_node(
            cache.root_node,
            [1, 2, 3, 4],
            evicted=False,
            backuped=True,
            high_ref=1,
        )
        cache.high_ref_evictable_leaves.add(gpu_high)
        cache.high_ref_evictable_size_ = 4

        self.assertEqual(cache.high_ref_host_safe_evictable_size(), 4)
        self.assertEqual(
            cache.safe_evictable_size_by_tier(allow_low=True, allow_high=True), 4
        )

    def test_write_backup_never_evicts_host_high_ref_nodes(self):
        cache = self._make_cache(available_host_tokens=0)
        host_high = self._append_node(
            cache.root_node,
            [1, 2, 3, 4],
            evicted=True,
            backuped=True,
            high_ref=1,
        )
        gpu_high = self._append_node(
            cache.root_node,
            [5, 6, 7, 8],
            evicted=False,
            backuped=False,
            high_ref=1,
        )
        cache.evictable_host_leaves.add(host_high)

        written = cache.write_backup(gpu_high, write_back=True)

        self.assertEqual(written, 0)
        self.assertEqual(cache.cache_controller.evicted_host_lengths, [])


class TestUpdateRefPropagatesPriority(unittest.TestCase):
    def test_update_ref_writes_back_priority_to_running_and_waiting_reqs(self):
        from types import SimpleNamespace
        from sglang.srt.managers.io_struct import UpdateRefReqInput
        from sglang.srt.managers.scheduler import Scheduler

        # Build a minimal scheduler stub that exposes only the fields
        # handle_update_ref reads.
        sched = Scheduler.__new__(Scheduler)
        sched.enable_ref_aware_kv_buffer = True

        class _FakeCache:
            def __init__(self):
                self.calls = []
            def update_ref(self, rid, new_priority):
                self.calls.append((rid, new_priority))
                return True, "ok"

        from sglang.srt.mem_cache.ref_aware_hiradix_cache import RefAwareHiRadixCache
        sched.tree_cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        # Patch the bound method so isinstance(cache, RefAwareHiRadixCache) holds.
        cache = _FakeCache()
        sched.tree_cache.update_ref = cache.update_ref  # type: ignore

        running = SimpleNamespace(rid="r1", priority=0)
        waiting = SimpleNamespace(rid="r1", priority=0)
        chunked = SimpleNamespace(rid="r2", priority=0)
        sched.running_batch = SimpleNamespace(reqs=[running])
        sched.waiting_queue = [waiting]
        sched.chunked_req = chunked

        out = sched.handle_update_ref(UpdateRefReqInput(rid="r1", new_priority=5))
        self.assertTrue(out.success)
        self.assertEqual(running.priority, 5)
        self.assertEqual(waiting.priority, 5)
        # rid r2 unchanged
        self.assertEqual(chunked.priority, 0)
        self.assertEqual(cache.calls, [("r1", 5)])


class TestReleaseRefIdempotent(unittest.TestCase):
    def test_release_unknown_rid_returns_success(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.rid_to_ref_info = {}
        ok, msg = cache.release_ref("never-registered")
        self.assertTrue(ok)
        self.assertIn("not tracked", msg)


class TestRefAwareEndToEndAccounting(unittest.TestCase):
    """register → update → release should leave tier sizes at zero."""

    def _make_cache(self):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache.unused_evictable_leaves = set()
        cache.low_ref_evictable_leaves = set()
        cache.high_ref_evictable_leaves = set()
        cache.unused_evictable_size_ = 0
        cache.low_ref_evictable_size_ = 0
        cache.high_ref_evictable_size_ = 0
        cache.rid_to_ref_info = {}
        return cache

    def _append(self, parent, ids):
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(ids)
        node.value = torch.tensor(ids, dtype=torch.int64)
        node.children = {}
        parent.children[ids[0]] = node
        return node

    def test_register_update_release_cycle_zeroes_accounting(self):
        from types import SimpleNamespace
        cache = self._make_cache()
        a = self._append(cache.root_node, [1, 2, 3, 4])
        b = self._append(a, [5, 6, 7, 8])
        for n in (a, b):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        # Register as LP first.
        req = SimpleNamespace(rid="r1", priority=0, last_node=b)
        cache.register_ref(req)
        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 8)
        self.assertEqual(cache.high_ref_evictable_size_, 0)

        # Promote to HP.
        ok, _ = cache.update_ref("r1", 5)
        self.assertTrue(ok)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 8)

        # Release.
        ok, _ = cache.release_ref("r1")
        self.assertTrue(ok)
        self.assertEqual(cache.unused_evictable_size_, 8)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertEqual(cache.rid_to_ref_info, {})
        self.assertEqual(a.tracked_rids, set())
        self.assertEqual(b.tracked_rids, set())


if __name__ == "__main__":
    unittest.main()

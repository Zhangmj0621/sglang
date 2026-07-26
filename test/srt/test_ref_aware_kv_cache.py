"""Unit tests for ref-aware tiered KV cache eviction.

Group A: mixin-level tests over plain RadixCache via the _MixinCache harness.
Group B: HiRadix-level tests (skipped when the hiradix import chain is
unavailable).
"""

import unittest
import unittest.mock
from collections import OrderedDict
from types import SimpleNamespace

import torch

# Must precede any sglang import: on machines without a working triton the
# torch dynamo module fails to initialize lazily inside sglang's import chain.
try:
    import torch._inductor.runtime.triton_heuristics  # noqa: F401
except Exception:
    pass

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, InsertResult
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    _key_match_page_size1,
    get_child_key,
)

try:
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.ref_aware_hiradix_cache import RefAwareHiRadixCache

    HAS_HIRADIX = True
except Exception:  # heavy import chain unavailable on some dev machines
    HiRadixCache = None
    RefAwareHiRadixCache = None
    HAS_HIRADIX = False


class TestBaseExtensionPoints(unittest.TestCase):
    """The hooks RefAwareCacheMixin plugs into."""

    def test_insert_result_last_node_defaults_to_none(self):
        self.assertIsNone(InsertResult(prefix_len=0).last_node)

    def test_tree_node_ref_fields_default_to_zero(self):
        node = TreeNode()
        self.assertEqual(node.high_ref, 0)
        self.assertEqual(node.low_ref, 0)
        self.assertEqual(node.tracked_rids, set())

    def test_tree_node_tracked_rids_are_not_shared(self):
        a, b = TreeNode(), TreeNode()
        a.tracked_rids.add("r1")
        self.assertEqual(b.tracked_rids, set())

    def test_account_new_evictable_node_is_a_noop_hook(self):
        cache = RadixCache.__new__(RadixCache)
        self.assertIsNone(cache._account_new_evictable_node(TreeNode()))


from sglang.srt.mem_cache.ref_aware_cache_mixin import RefInfo  # noqa: F401
from sglang.srt.mem_cache.ref_aware_cache_mixin import (  # noqa: E402
    TIER_HIGH_REF,
    TIER_LOW_REF,
    TIER_UNUSED,
    RefAwareCacheMixin,
    _classify_node_tier,
)


class _MixinCache(RefAwareCacheMixin, RadixCache):
    """Light harness: mixin over plain RadixCache, constructed via __new__."""


def _make_mixin_cache():
    cache = _MixinCache.__new__(_MixinCache)
    cache.root_node = TreeNode()
    cache.root_node.key = RadixKey([])
    cache.root_node.value = torch.tensor([], dtype=torch.int64)
    cache.root_node.lock_ref = 1
    cache.high_priority_threshold = 1
    cache._enable_priority_scheduling = True
    cache.unused_evictable_leaves = set()
    cache.low_ref_evictable_leaves = set()
    cache.high_ref_evictable_leaves = set()
    cache.unused_evictable_size_ = 0
    cache.low_ref_evictable_size_ = 0
    cache.high_ref_evictable_size_ = 0
    cache.rid_to_ref_info = {}
    cache._evict_scope_stack = []
    cache._last_inserted_node = None
    # RadixCache-level state needed by inc/dec_lock_ref and _update_leaf_status
    cache.evictable_leaves = set()
    cache.disable = False
    cache.evictable_size_ = 0
    cache.protected_size_ = 0
    cache.page_size = 1
    cache.get_child_key_fn = get_child_key
    cache.key_match_fn = _key_match_page_size1
    return cache


def _append_plain_node(parent, token_ids):
    node = TreeNode()
    node.parent = parent
    node.key = RadixKey(token_ids)
    node.value = torch.tensor(token_ids, dtype=torch.int64)
    node.children = {}
    parent.children[token_ids[0] if token_ids else 0] = node
    return node


class _StubInsertBase:
    """Stand-in for the RadixCache/HiRadixCache half of the MRO."""

    def __init__(self):
        self.reported_last_node = None
        self.dec_lock_ref_arg = "unset"

    def insert(self, params):
        return InsertResult(prefix_len=0, last_node=self.reported_last_node)

    def cache_finished_req(self, req, is_insert=True):
        # Mirrors RadixCache.cache_finished_req: inserts, then releases the
        # lock held on the PRE-insert last_node.
        self.insert(InsertParams(key=RadixKey([1, 2])))
        self.dec_lock_ref_arg = req.last_node


class _InsertProbe(RefAwareCacheMixin, _StubInsertBase):
    pass


class TestMixinLastNodeWriteBack(unittest.TestCase):
    """cache_finished_req must refresh req.last_node from InsertResult so the
    next register_ref sees the freshly inserted suffix -- but only after the
    base class has released the lock on the pre-insert node."""

    def _make_probe(self, reported_last_node):
        probe = _InsertProbe.__new__(_InsertProbe)
        _StubInsertBase.__init__(probe)
        probe._last_inserted_node = None
        probe.reported_last_node = reported_last_node
        return probe

    def test_insert_captures_last_node_from_result(self):
        node = TreeNode()
        probe = self._make_probe(node)

        probe.insert(InsertParams(key=RadixKey([1, 2])))

        self.assertIs(probe._last_inserted_node, node)

    def test_cache_finished_req_writes_back_last_node(self):
        old_node, new_node = TreeNode(), TreeNode()
        probe = self._make_probe(new_node)
        req = SimpleNamespace(rid="s1", priority=1, last_node=old_node)

        probe.cache_finished_req(req)

        self.assertIs(req.last_node, new_node)
        # The base class must still have seen the PRE-insert node.
        self.assertIs(probe.dec_lock_ref_arg, old_node)

    def test_last_node_untouched_when_cache_does_not_report_one(self):
        """RadixCache leaves last_node=None; req.last_node must not be clobbered."""
        old_node = TreeNode()
        probe = self._make_probe(None)
        req = SimpleNamespace(rid="s1", priority=1, last_node=old_node)

        probe.cache_finished_req(req)

        self.assertIs(req.last_node, old_node)


class TestRefAwareTierAccounting(unittest.TestCase):
    """Test _account_new_evictable_node, _inc_priority_ref_single,
    _dec_priority_ref_single, and _move_node_tier."""

    def _make_cache(self):
        return _make_mixin_cache()

    def _append_node(self, parent, token_ids):
        return _append_plain_node(parent, token_ids)

    def test_new_evictable_node_starts_in_unused_tier(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])

        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        self.assertEqual(cache.unused_evictable_size_, 4)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertIn(node, cache.unused_evictable_leaves)

    def test_evictable_size_by_tier_unused_only(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        # allow_low=False, allow_high=False → only unused
        self.assertEqual(
            cache.evictable_size_by_tier(allow_low=False, allow_high=False), 4
        )
        # allow_low=True → still 4 since no low-ref nodes
        self.assertEqual(
            cache.evictable_size_by_tier(allow_low=True, allow_high=False), 4
        )

    def test_inc_priority_ref_low_moves_unused_to_low_ref(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        cache._inc_priority_ref_single(node, is_high=False)

        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 4)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertNotIn(node, cache.unused_evictable_leaves)
        self.assertIn(node, cache.low_ref_evictable_leaves)

    def test_inc_priority_ref_high_from_unused_moves_to_high_ref(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        cache._inc_priority_ref_single(node, is_high=True)

        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 4)
        self.assertNotIn(node, cache.unused_evictable_leaves)
        self.assertIn(node, cache.high_ref_evictable_leaves)

    def test_inc_priority_ref_high_from_low_ref_moves_to_high_ref(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)
        cache._inc_priority_ref_single(node, is_high=False)

        cache._inc_priority_ref_single(node, is_high=True)

        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 4)
        self.assertIn(node, cache.high_ref_evictable_leaves)

    def test_ref_tier_move_preserves_total_evictable_tokens(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        # unused → low_ref
        cache._inc_priority_ref_single(node, is_high=False)
        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 4)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertEqual(
            cache.evictable_size_by_tier(allow_low=True, allow_high=False), 4
        )

        # low_ref → high_ref
        cache._inc_priority_ref_single(node, is_high=True)
        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 4)
        self.assertEqual(
            cache.evictable_size_by_tier(allow_low=True, allow_high=True), 4
        )

    def test_dec_priority_ref_single_moves_back_to_unused(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)
        cache._inc_priority_ref_single(node, is_high=False)

        cache._dec_priority_ref_single(node, is_high=False)

        self.assertEqual(cache.unused_evictable_size_, 4)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertIn(node, cache.unused_evictable_leaves)
        self.assertNotIn(node, cache.low_ref_evictable_leaves)

    def test_dec_priority_ref_single_high_moves_back(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)
        cache._inc_priority_ref_single(node, is_high=True)

        cache._dec_priority_ref_single(node, is_high=True)

        self.assertEqual(cache.unused_evictable_size_, 4)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertIn(node, cache.unused_evictable_leaves)
        self.assertNotIn(node, cache.high_ref_evictable_leaves)

    def test_dec_priority_ref_does_not_go_below_zero(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        # Decrement without prior increment — should not crash or go negative
        cache._dec_priority_ref_single(node, is_high=False)
        self.assertEqual(node.low_ref, 0)

        cache._dec_priority_ref_single(node, is_high=True)
        self.assertEqual(node.high_ref, 0)

    def test_move_node_tier_updates_sets_and_sizes(self):
        cache = self._make_cache()
        node = self._append_node(cache.root_node, [1, 2, 3, 4])
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)

        # Manually put node in unused tier set to test _move_node_tier directly
        cache._move_node_tier(node, TIER_UNUSED, TIER_LOW_REF)

        self.assertNotIn(node, cache.unused_evictable_leaves)
        self.assertIn(node, cache.low_ref_evictable_leaves)
        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 4)


class TestRefAwareRegisterRef(unittest.TestCase):
    """Test register_ref only adds new suffix from last_node."""

    def _make_cache(self):
        return _make_mixin_cache()

    def _append_node(self, parent, token_ids):
        return _append_plain_node(parent, token_ids)

    def test_register_ref_high_priority_sets_high_ref(self):
        """High priority (priority >= threshold) increments high_ref."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req = SimpleNamespace(rid="s1", priority=1, last_node=a)
        cache.register_ref(req)

        self.assertEqual(a.high_ref, 1)
        self.assertEqual(a.low_ref, 0)
        self.assertIn("s1", a.tracked_rids)

    def test_register_ref_low_priority_sets_low_ref(self):
        """Low priority (priority < threshold) increments low_ref."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req = SimpleNamespace(rid="s1", priority=0, last_node=a)
        cache.register_ref(req)

        self.assertEqual(a.low_ref, 1)
        self.assertEqual(a.high_ref, 0)

    def test_register_ref_priority_class_mismatch_raises(self):
        """A follow-up request that changes priority class without /update_ref
        must fail fast instead of corrupting high_ref/low_ref accounting."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req = SimpleNamespace(rid="s1", priority=0, last_node=a)
        cache.register_ref(req)

        b = self._append_node(a, [5, 6, 7, 8])
        req2 = SimpleNamespace(rid="s1", priority=1, last_node=b)
        with self.assertRaisesRegex(ValueError, "Priority class mismatch"):
            cache.register_ref(req2)

        # Accounting must be untouched by the rejected request
        self.assertEqual(a.low_ref, 1)
        self.assertEqual(a.high_ref, 0)
        self.assertEqual(b.low_ref, 0)
        self.assertEqual(b.high_ref, 0)

    def test_register_ref_same_class_different_priority_ok(self):
        """Priority changes within the same class (e.g. 1 -> 2, both high)
        are harmless and must not raise."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req = SimpleNamespace(rid="s1", priority=1, last_node=a)
        cache.register_ref(req)

        b = self._append_node(a, [5, 6, 7, 8])
        req2 = SimpleNamespace(rid="s1", priority=2, last_node=b)
        cache.register_ref(req2)

        self.assertEqual(a.high_ref, 1)
        self.assertEqual(b.high_ref, 1)
        self.assertEqual(cache.rid_to_ref_info["s1"].priority, 2)

    def test_register_ref_only_adds_new_suffix_from_last_node(self):
        """Second register_ref only adds nodes not previously tracked."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])
        c = self._append_node(b, [9, 10, 11, 12])

        req = SimpleNamespace(rid="s1", priority=1, last_node=c)
        cache.register_ref(req)

        self.assertEqual(a.high_ref, 1)
        self.assertEqual(b.high_ref, 1)
        self.assertEqual(c.high_ref, 1)
        self.assertEqual(len(cache.rid_to_ref_info["s1"].nodes), 3)

        # Extend the chain by one node and call register_ref again
        d = self._append_node(c, [13, 14, 15, 16])
        req.last_node = d
        cache.register_ref(req)

        # Old nodes should NOT have their ref doubled
        self.assertEqual(a.high_ref, 1)
        self.assertEqual(b.high_ref, 1)
        self.assertEqual(c.high_ref, 1)
        # New node should now be tracked
        self.assertEqual(d.high_ref, 1)
        self.assertEqual(len(cache.rid_to_ref_info["s1"].nodes), 4)

    def test_register_ref_tracks_rids_on_nodes(self):
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req = SimpleNamespace(rid="s1", priority=1, last_node=a)
        cache.register_ref(req)

        self.assertIn("s1", a.tracked_rids)
        self.assertIn("s1", cache.rid_to_ref_info)

    def test_register_ref_multiple_rids_on_shared_node(self):
        """Two different rids that share a node both track it."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        req1 = SimpleNamespace(rid="s1", priority=1, last_node=a)
        req2 = SimpleNamespace(rid="s2", priority=1, last_node=a)
        cache.register_ref(req1)
        cache.register_ref(req2)

        self.assertEqual(a.high_ref, 2)
        self.assertIn("s1", a.tracked_rids)
        self.assertIn("s2", a.tracked_rids)


class TestReleaseRefIdempotent(unittest.TestCase):
    """Release unknown rid returns success."""

    def test_release_unknown_rid_returns_success(self):
        cache = _MixinCache.__new__(_MixinCache)
        cache.rid_to_ref_info = {}
        ok, msg = cache.release_ref("never-registered")
        self.assertTrue(ok)
        self.assertIn("not tracked", msg)

    def test_release_idempotent_after_first_release(self):
        """Releasing the same rid twice should succeed both times."""
        cache = _make_mixin_cache()
        node = _append_plain_node(cache.root_node, [1, 2, 3, 4])

        req = SimpleNamespace(rid="s1", priority=1, last_node=node)
        cache.register_ref(req)

        ok1, _ = cache.release_ref("s1")
        self.assertTrue(ok1)

        # Second release of same rid should also return success (idempotent)
        ok2, msg2 = cache.release_ref("s1")
        self.assertTrue(ok2)
        self.assertIn("not tracked", msg2)


class TestUpdateRef(unittest.TestCase):
    """Test priority change moves nodes between tiers."""

    def _make_cache(self):
        return _make_mixin_cache()

    def _append_node(self, parent, token_ids):
        return _append_plain_node(parent, token_ids)

    def test_update_ref_unknown_rid_returns_false(self):
        cache = self._make_cache()
        ok, msg = cache.update_ref("unknown-rid", 5)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_update_ref_low_to_high_priority_moves_nodes(self):
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])

        for n in (a, b):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        # Register as low priority
        req = SimpleNamespace(rid="s1", priority=0, last_node=b)
        cache.register_ref(req)
        self.assertEqual(cache.low_ref_evictable_size_, 8)
        self.assertEqual(cache.high_ref_evictable_size_, 0)

        # Promote to high priority
        ok, _ = cache.update_ref("s1", 5)
        self.assertTrue(ok)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 8)
        self.assertEqual(a.high_ref, 1)
        self.assertEqual(a.low_ref, 0)
        self.assertEqual(b.high_ref, 1)
        self.assertEqual(b.low_ref, 0)

    def test_update_ref_high_to_low_priority_moves_nodes(self):
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])

        for n in (a, b):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        # Register as high priority
        req = SimpleNamespace(rid="s1", priority=5, last_node=b)
        cache.register_ref(req)
        self.assertEqual(cache.high_ref_evictable_size_, 8)

        # Demote to low priority
        ok, _ = cache.update_ref("s1", 0)
        self.assertTrue(ok)
        self.assertEqual(cache.low_ref_evictable_size_, 8)
        self.assertEqual(cache.high_ref_evictable_size_, 0)

    def test_update_ref_same_class_is_noop(self):
        """If priority class doesn't change, update_ref is a no-op."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        for n in (a,):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        req = SimpleNamespace(rid="s1", priority=5, last_node=a)
        cache.register_ref(req)
        self.assertEqual(cache.high_ref_evictable_size_, 4)

        # Update with another high-priority value (still above threshold)
        ok, msg = cache.update_ref("s1", 10)
        self.assertTrue(ok)
        self.assertIn("unchanged", msg)
        # Size should not have changed
        self.assertEqual(cache.high_ref_evictable_size_, 4)


class TestScopedEvict(unittest.TestCase):
    """Verify context manager controls eviction scope."""

    def _make_cache(self):
        return _make_mixin_cache()

    def test_scoped_evict_empty_stack_by_default(self):
        cache = self._make_cache()
        self.assertEqual(len(cache._evict_scope_stack), 0)

    def test_scoped_evict_pushes_and_pops_stack(self):
        cache = self._make_cache()
        with cache.scoped_evict(allow_low=True, allow_high=False):
            self.assertEqual(len(cache._evict_scope_stack), 1)
            self.assertEqual(cache._evict_scope_stack[-1], (True, False))
        self.assertEqual(len(cache._evict_scope_stack), 0)

    def test_scoped_evict_nested_stacks(self):
        cache = self._make_cache()
        with cache.scoped_evict(allow_low=True, allow_high=False):
            with cache.scoped_evict(allow_low=True, allow_high=True):
                self.assertEqual(len(cache._evict_scope_stack), 2)
                self.assertEqual(cache._evict_scope_stack[-1], (True, True))
            self.assertEqual(len(cache._evict_scope_stack), 1)
            self.assertEqual(cache._evict_scope_stack[-1], (True, False))
        self.assertEqual(len(cache._evict_scope_stack), 0)

    def test_scoped_evict_cleans_up_on_exception(self):
        """Context manager should clean up even when exception is raised."""
        cache = self._make_cache()
        try:
            with cache.scoped_evict(allow_low=True, allow_high=True):
                self.assertEqual(len(cache._evict_scope_stack), 1)
                raise ValueError("test exception")
        except ValueError:
            pass
        # Stack should be clean after exception
        self.assertEqual(len(cache._evict_scope_stack), 0)

    def test_scoped_evict_high_only_scope(self):
        cache = self._make_cache()
        with cache.scoped_evict(allow_low=False, allow_high=True):
            self.assertEqual(cache._evict_scope_stack[-1], (False, True))

    def test_evict_uses_scope_stack_when_not_empty(self):
        """evict() should read allow_low/allow_high from the scope stack."""
        cache = self._make_cache()

        # Verify scope stack is read: push a scope, check it's visible
        with cache.scoped_evict(allow_low=False, allow_high=True):
            self.assertTrue(len(cache._evict_scope_stack) > 0)
            allow_low, allow_high = cache._evict_scope_stack[-1]
            self.assertFalse(allow_low)
            self.assertTrue(allow_high)


class TestEndToEndAccounting(unittest.TestCase):
    """register → update → release cycle zeroes all counters."""

    def _make_cache(self):
        return _make_mixin_cache()

    def _append_node(self, parent, token_ids):
        return _append_plain_node(parent, token_ids)

    def test_register_update_release_cycle_zeroes_accounting(self):
        """Full lifecycle: register (LP) → update (HP) → release → counters at zero."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])

        for n in (a, b):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        # Register as low priority
        req = SimpleNamespace(rid="s1", priority=0, last_node=b)
        cache.register_ref(req)
        self.assertEqual(cache.unused_evictable_size_, 0)
        self.assertEqual(cache.low_ref_evictable_size_, 8)
        self.assertEqual(cache.high_ref_evictable_size_, 0)

        # Promote to high priority
        ok, _ = cache.update_ref("s1", 5)
        self.assertTrue(ok)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 8)

        # Release
        ok, _ = cache.release_ref("s1")
        self.assertTrue(ok)
        self.assertEqual(cache.unused_evictable_size_, 8)
        self.assertEqual(cache.low_ref_evictable_size_, 0)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertNotIn("s1", cache.rid_to_ref_info)
        self.assertEqual(a.tracked_rids, set())
        self.assertEqual(b.tracked_rids, set())

    def test_register_release_cycle_with_two_rids(self):
        """Two rids on the same nodes both release cleanly."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])

        for n in (a,):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        req1 = SimpleNamespace(rid="s1", priority=0, last_node=a)
        req2 = SimpleNamespace(rid="s2", priority=0, last_node=a)
        cache.register_ref(req1)
        cache.register_ref(req2)

        self.assertEqual(a.low_ref, 2)
        self.assertEqual(cache.low_ref_evictable_size_, 4)

        cache.release_ref("s1")
        self.assertEqual(a.low_ref, 1)
        # Still in low_ref tier since s2 still holds it
        self.assertEqual(cache.low_ref_evictable_size_, 4)

        cache.release_ref("s2")
        self.assertEqual(a.low_ref, 0)
        # Back to unused
        self.assertEqual(cache.unused_evictable_size_, 4)
        self.assertEqual(cache.low_ref_evictable_size_, 0)

    def test_register_high_release_moves_to_unused(self):
        """High-priority register then release returns nodes to unused tier."""
        cache = self._make_cache()
        a = self._append_node(cache.root_node, [1, 2, 3, 4])
        b = self._append_node(a, [5, 6, 7, 8])

        for n in (a, b):
            cache._account_new_evictable_node(n)
            cache._update_ref_aware_leaf_status(n)

        req = SimpleNamespace(rid="s1", priority=5, last_node=b)
        cache.register_ref(req)

        self.assertEqual(cache.high_ref_evictable_size_, 8)

        cache.release_ref("s1")

        self.assertEqual(cache.unused_evictable_size_, 8)
        self.assertEqual(cache.high_ref_evictable_size_, 0)
        self.assertNotIn("s1", cache.rid_to_ref_info)
        self.assertEqual(a.tracked_rids, set())
        self.assertEqual(b.tracked_rids, set())


class TestClassifyNodeTier(unittest.TestCase):
    def test_both_zero_is_unused(self):
        node = TreeNode()
        self.assertEqual(_classify_node_tier(node), TIER_UNUSED)

    def test_low_ref_only(self):
        node = TreeNode()
        node.low_ref = 3
        self.assertEqual(_classify_node_tier(node), TIER_LOW_REF)

    def test_high_ref_wins_over_low_ref(self):
        node = TreeNode()
        node.high_ref = 1
        node.low_ref = 5
        self.assertEqual(_classify_node_tier(node), TIER_HIGH_REF)


class _FakeAllocator:
    def __init__(self):
        self.freed = []
        self.device = torch.device("cpu")

    def free(self, indices):
        self.freed.append(len(indices))


def _make_ref_aware_radix_cache():
    from sglang.srt.mem_cache.ref_aware_radix_cache import RefAwareRadixCache

    cache = RefAwareRadixCache.__new__(RefAwareRadixCache)
    cache.root_node = TreeNode()
    cache.root_node.key = RadixKey([])
    cache.root_node.value = torch.tensor([], dtype=torch.int64)
    cache.root_node.lock_ref = 1
    cache.high_priority_threshold = 1
    cache._enable_priority_scheduling = True
    cache.unused_evictable_leaves = set()
    cache.low_ref_evictable_leaves = set()
    cache.high_ref_evictable_leaves = set()
    cache.unused_evictable_size_ = 0
    cache.low_ref_evictable_size_ = 0
    cache.high_ref_evictable_size_ = 0
    cache.rid_to_ref_info = {}
    cache._evict_scope_stack = []
    cache._last_inserted_node = None
    cache.evictable_leaves = set()
    cache.disable = False
    cache.evictable_size_ = 0
    cache.protected_size_ = 0
    cache.page_size = 1
    cache.get_child_key_fn = get_child_key
    cache.key_match_fn = _key_match_page_size1
    cache.enable_kv_cache_events = False
    cache.metrics_collector = None
    cache.token_to_kv_pool_allocator = _FakeAllocator()
    cache.eviction_strategy = SimpleNamespace(
        get_priority=lambda node: node.last_access_time
    )
    return cache


class TestRefAwareRadixCacheEviction(unittest.TestCase):
    def _seed(self, cache, token_ids, *, high_ref=0, low_ref=0):
        node = _append_plain_node(cache.root_node, token_ids)
        node.high_ref = high_ref
        node.low_ref = low_ref
        cache.evictable_size_ += len(token_ids)
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)
        return node

    def test_evict_prefers_unused_over_low_ref(self):
        cache = _make_ref_aware_radix_cache()
        unused = self._seed(cache, [1, 2, 3, 4])
        low = self._seed(cache, [5, 6, 7, 8], low_ref=1)

        evicted = cache._evict_tiered(4, allow_low=True, allow_high=False)

        self.assertEqual(evicted, 4)
        self.assertNotIn(unused, cache.root_node.children.values())
        self.assertIn(low, cache.root_node.children.values())

    def test_high_ref_survives_when_allow_high_is_false(self):
        cache = _make_ref_aware_radix_cache()
        high = self._seed(cache, [1, 2, 3, 4], high_ref=1)

        evicted = cache._evict_tiered(4, allow_low=True, allow_high=False)

        self.assertEqual(evicted, 0)
        self.assertIn(high, cache.root_node.children.values())

    def test_high_ref_evicted_when_allow_high_is_true(self):
        cache = _make_ref_aware_radix_cache()
        high = self._seed(cache, [1, 2, 3, 4], high_ref=1)

        evicted = cache._evict_tiered(4, allow_low=True, allow_high=True)

        self.assertEqual(evicted, 4)
        self.assertNotIn(high, cache.root_node.children.values())

    def test_high_tier_uses_the_configured_strategy_not_mru(self):
        """Within a tier, ties on ref count fall through to the configured
        eviction strategy (LRU here) for every tier alike — the high tier must
        not invert it and tear down the hottest prefix first."""
        cache = _make_ref_aware_radix_cache()
        stale = self._seed(cache, [1, 2, 3, 4], high_ref=1)
        hot = self._seed(cache, [5, 6, 7, 8], high_ref=1)
        stale.last_access_time = 100.0
        hot.last_access_time = 200.0

        evicted = cache._evict_tiered(4, allow_low=True, allow_high=True)

        self.assertEqual(evicted, 4)
        self.assertNotIn(stale, cache.root_node.children.values())
        self.assertIn(hot, cache.root_node.children.values())

    def test_more_refs_evict_later_within_a_tier(self):
        """Ref count outranks the strategy: a node held by more sessions
        survives even when it is the least recently used."""
        cache = _make_ref_aware_radix_cache()
        one_holder = self._seed(cache, [1, 2, 3, 4], high_ref=1)
        many_holders = self._seed(cache, [5, 6, 7, 8], high_ref=3)
        one_holder.last_access_time = 200.0
        many_holders.last_access_time = 100.0

        evicted = cache._evict_tiered(4, allow_low=True, allow_high=True)

        self.assertEqual(evicted, 4)
        self.assertNotIn(one_holder, cache.root_node.children.values())
        self.assertIn(many_holders, cache.root_node.children.values())

    def test_high_ref_outranks_low_ref_within_the_high_tier(self):
        """low_ref only breaks ties in high_ref — it never outweighs it, even
        when the low_ref count is much larger."""
        cache = _make_ref_aware_radix_cache()
        fewer_high = self._seed(cache, [1, 2, 3, 4], high_ref=1, low_ref=10)
        more_high = self._seed(cache, [5, 6, 7, 8], high_ref=2, low_ref=0)
        fewer_high.last_access_time = 100.0
        more_high.last_access_time = 100.0

        evicted = cache._evict_tiered(4, allow_low=True, allow_high=True)

        self.assertEqual(evicted, 4)
        self.assertNotIn(fewer_high, cache.root_node.children.values())
        self.assertIn(more_high, cache.root_node.children.values())

    def test_low_ref_breaks_ties_in_high_ref(self):
        """With high_ref equal, the node held by more low-priority sessions
        survives."""
        cache = _make_ref_aware_radix_cache()
        fewer_low = self._seed(cache, [1, 2, 3, 4], high_ref=1, low_ref=1)
        more_low = self._seed(cache, [5, 6, 7, 8], high_ref=1, low_ref=4)
        fewer_low.last_access_time = 100.0
        more_low.last_access_time = 100.0

        evicted = cache._evict_tiered(4, allow_low=True, allow_high=True)

        self.assertEqual(evicted, 4)
        self.assertNotIn(fewer_low, cache.root_node.children.values())
        self.assertIn(more_low, cache.root_node.children.values())


class TestUpdateRefPropagatesPriority(unittest.TestCase):
    def test_update_ref_writes_back_priority_to_running_and_waiting_reqs(self):
        from sglang.srt.managers.io_struct import UpdateRefReqInput
        from sglang.srt.managers.scheduler import Scheduler

        # Minimal scheduler stub exposing only what handle_update_ref reads.
        sched = Scheduler.__new__(Scheduler)
        sched.enable_ref_aware_kv_buffer = True

        running = SimpleNamespace(rid="r1", priority=0)
        waiting = SimpleNamespace(rid="r1", priority=0)
        chunked = SimpleNamespace(rid="r2", priority=0)

        class _FakeCache:
            """Spies on the observed reqs' priority AT CALL TIME. This pins the
            write-before-call ordering requirement: if handle_update_ref ever
            called update_ref before writing back priorities, observed_at_call
            would still show the pre-update priority (0), not the new one."""

            def __init__(self, observed_reqs):
                self.calls = []
                self.observed_reqs = observed_reqs
                self.observed_at_call = None

            def update_ref(self, rid, new_priority):
                self.calls.append((rid, new_priority))
                self.observed_at_call = [r.priority for r in self.observed_reqs]
                return True, "ok"

        sched.tree_cache = _MixinCache.__new__(_MixinCache)
        # Patch the bound method so isinstance(cache, RefAwareCacheMixin) holds.
        cache = _FakeCache(observed_reqs=[running, waiting])
        sched.tree_cache.update_ref = cache.update_ref  # type: ignore

        sched.running_batch = SimpleNamespace(reqs=[running])
        sched.waiting_queue = [waiting]
        sched.chunked_req = chunked

        out = sched.handle_update_ref(UpdateRefReqInput(rid="r1", new_priority=5))
        self.assertTrue(out.success)
        self.assertEqual(running.priority, 5)
        self.assertEqual(waiting.priority, 5)
        self.assertEqual(chunked.priority, 0)
        self.assertEqual(cache.calls, [("r1", 5)])
        # If update_ref were invoked before the priority write-back loops,
        # this would observe [0, 0] instead of [5, 5].
        self.assertEqual(cache.observed_at_call, [5, 5])

    def test_update_ref_writes_back_priority_to_matching_chunked_req(self):
        """A chunked_req whose rid matches must also get its priority updated.

        (The sibling test above only exercises the chunked_req *non*-match
        path; this proves the match path also fires.)
        """
        from sglang.srt.managers.io_struct import UpdateRefReqInput
        from sglang.srt.managers.scheduler import Scheduler

        sched = Scheduler.__new__(Scheduler)
        sched.enable_ref_aware_kv_buffer = True

        chunked = SimpleNamespace(rid="r1", priority=0)

        class _FakeCache:
            def update_ref(self, rid, new_priority):
                return True, "ok"

        sched.tree_cache = _MixinCache.__new__(_MixinCache)
        sched.tree_cache.update_ref = _FakeCache().update_ref  # type: ignore

        sched.running_batch = SimpleNamespace(reqs=[])
        sched.waiting_queue = []
        sched.chunked_req = chunked

        out = sched.handle_update_ref(UpdateRefReqInput(rid="r1", new_priority=7))
        self.assertTrue(out.success)
        self.assertEqual(chunked.priority, 7)


class TestReleaseRefWhenDisabled(unittest.TestCase):
    def test_release_ref_reports_disabled(self):
        from sglang.srt.managers.io_struct import ReleaseRefReqInput
        from sglang.srt.managers.scheduler import Scheduler

        sched = Scheduler.__new__(Scheduler)
        sched.enable_ref_aware_kv_buffer = False
        out = sched.handle_release_ref(ReleaseRefReqInput(rid="r1"))
        self.assertFalse(out.success)
        self.assertIn("not enabled", out.message)


class TestPrefillAdderRefAwareBudget(unittest.TestCase):
    def _make_adder(self, *, available, unused, low, high):
        from sglang.srt.managers.schedule_policy import PrefillAdder

        adder = PrefillAdder.__new__(PrefillAdder)
        cache = _make_mixin_cache()
        cache.unused_evictable_size_ = unused
        cache.low_ref_evictable_size_ = low
        cache.high_ref_evictable_size_ = high
        adder.tree_cache = cache
        adder.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=lambda: available
        )
        adder.rem_total_token_offset = 0
        adder.enable_ref_aware_kv_buffer = True
        adder.high_priority_threshold = 1
        return adder

    def test_low_priority_budget_excludes_high_ref_tier(self):
        adder = self._make_adder(available=10, unused=5, low=7, high=100)
        self.assertEqual(adder._rem_total_tokens_ref_aware(is_high=False), 22)

    def test_high_priority_budget_includes_high_ref_tier(self):
        adder = self._make_adder(available=10, unused=5, low=7, high=100)
        self.assertEqual(adder._rem_total_tokens_ref_aware(is_high=True), 122)

    def test_budget_subtracts_reserved_offset(self):
        adder = self._make_adder(available=10, unused=5, low=7, high=0)
        adder.rem_total_token_offset = 12
        self.assertEqual(adder._rem_total_tokens_ref_aware(is_high=False), 10)


class TestEvictForDecodeTierOrder(unittest.TestCase):
    """Phase 1 evicts low tiers only; phase 2 evicts high tier capped at the
    high-priority requests' own token need."""

    def _make_batch(self, *, available, reqs):
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        batch = ScheduleBatch.__new__(ScheduleBatch)
        cache = _make_mixin_cache()
        calls = []

        def fake_evict_tiered(num_tokens, allow_low, allow_high):
            calls.append((num_tokens, allow_low, allow_high))
            return 0

        cache._evict_tiered = fake_evict_tiered
        cache.is_chunk_cache = lambda: False
        cache.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=lambda: available
        )
        batch.tree_cache = cache
        batch.reqs = reqs
        batch.new_tokens_required_next_decode = lambda idx=None: (
            len(reqs if idx is None else idx) * 8
        )
        return batch, calls

    def test_low_tier_first_then_capped_high_tier(self):
        hp = SimpleNamespace(priority=1)
        lp = SimpleNamespace(priority=0)
        batch, calls = self._make_batch(available=0, reqs=[hp, lp])

        with unittest.mock.patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=SimpleNamespace(enable_ref_aware_kv_buffer=True),
        ):
            batch._evict_for_decode(16, None)

        self.assertEqual(calls[0], (16, True, False))
        # shortfall == 16, but only one HP req -> capped at its own 8 tokens
        self.assertEqual(calls[1], (8, False, True))

    def test_no_high_tier_pass_without_high_priority_reqs(self):
        lp = SimpleNamespace(priority=0)
        batch, calls = self._make_batch(available=0, reqs=[lp])

        with unittest.mock.patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=SimpleNamespace(enable_ref_aware_kv_buffer=True),
        ):
            batch._evict_for_decode(8, None)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], (8, True, False))


@unittest.skipUnless(HAS_HIRADIX, "hiradix import chain unavailable")
class TestHiRadixHostLeafStatus(unittest.TestCase):
    """_update_host_leaf_status must key off `backuped`, not `evicted`."""

    def _make_cache(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.evictable_host_leaves = set()
        return cache

    def _node(self, *, evicted, backuped, lock_ref=0):
        node = TreeNode()
        node.value = None if evicted else torch.tensor([1], dtype=torch.int64)
        node.host_value = torch.tensor([1], dtype=torch.int64) if backuped else None
        node.lock_ref = lock_ref
        node.children = {}
        return node

    def test_node_with_backuped_child_is_not_a_host_leaf(self):
        cache = self._make_cache()
        parent = self._node(evicted=True, backuped=True)
        # Child is live on device AND backed up on host -> parent must not be
        # host-evictable, otherwise freeing it breaks the host prefix chain.
        child = self._node(evicted=False, backuped=True)
        parent.children = {1: child}

        cache._update_host_leaf_status(parent)

        self.assertNotIn(parent, cache.evictable_host_leaves)

    def test_node_with_device_only_child_is_a_host_leaf(self):
        cache = self._make_cache()
        parent = self._node(evicted=True, backuped=True)
        child = self._node(evicted=False, backuped=False)
        parent.children = {1: child}

        cache._update_host_leaf_status(parent)

        self.assertIn(parent, cache.evictable_host_leaves)


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


class _HiRadixHarness(unittest.TestCase):
    """Shared harness for hiradix-level tests (defines no tests itself)."""

    def _make_cache(self, available_host_tokens: int = 0):
        cache = RefAwareHiRadixCache.__new__(RefAwareHiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache.root_node.value = torch.tensor([], dtype=torch.int64)
        cache.root_node.lock_ref = 1
        cache.high_priority_threshold = 1
        cache._enable_priority_scheduling = True
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
        cache._adaptively_demoted_rids = OrderedDict()
        cache._idle_hp_heap = []
        cache._idle_hp_rids = set()
        cache._evict_scope_stack = []
        cache.evictable_leaves = set()
        # Real key functions: _evict_host_from_tier resolves children via
        # get_child_key_fn (page_size=1 → first token, matching _append_node).
        cache.page_size = 1
        cache.get_child_key_fn = get_child_key
        cache.key_match_fn = _key_match_page_size1
        # _evict_host_from_tier emits remove events; keep them disabled.
        cache.enable_kv_cache_events = False
        # mixin state + bookkeeping base that `_detach_backuped` decrements from
        cache._last_inserted_node = None
        cache.metrics_collector = None
        cache.evictable_size_ = 0
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


@unittest.skipUnless(HAS_HIRADIX, "hiradix import chain unavailable")
class TestRefAwareHostSafety(_HiRadixHarness):
    def test_safe_evictable_size_counts_full_high_tier(self):
        cache = self._make_cache(available_host_tokens=0)
        gpu_high = self._append_node(
            cache.root_node, [1, 2, 3, 4], evicted=False, backuped=False, high_ref=1
        )
        cache.high_ref_evictable_leaves.add(gpu_high)
        cache.high_ref_evictable_size_ = 4

        self.assertEqual(
            cache.safe_evictable_size_by_tier(allow_low=True, allow_high=True), 4
        )
        self.assertEqual(
            cache.safe_evictable_size_by_tier(allow_low=True, allow_high=False), 0
        )

    def test_write_backup_on_host_pressure_evicts_high_host_when_no_demotable(self):
        cache = self._make_cache(available_host_tokens=0)
        host_high = self._append_node(
            cache.root_node, [1, 2, 3, 4], evicted=True, backuped=True, high_ref=1
        )
        gpu_node = self._append_node(
            cache.root_node, [5, 6, 7, 8], evicted=False, backuped=False, high_ref=1
        )
        cache.evictable_host_leaves.add(host_high)

        written = cache.write_backup(gpu_node, write_back=True)

        # write returns None twice (dummy), so backup fails, but the host-full
        # fallback must have tried allow_high eviction and dropped host_high.
        self.assertEqual(written, 0)
        self.assertEqual(cache.cache_controller.evicted_host_lengths, [4])


@unittest.skipUnless(HAS_HIRADIX, "hiradix import chain unavailable")
class TestAdaptiveDemotion(_HiRadixHarness):
    def _register_hp(self, cache, rid, node):
        # Account the node into the tier structures first so demote/restore's
        # _move_node_tier set/size accounting stays consistent.
        cache._account_new_evictable_node(node)
        cache._update_ref_aware_leaf_status(node)
        req = SimpleNamespace(rid=rid, priority=1, last_node=node)
        cache.register_ref(req)

    def test_evict_host_demotes_shortest_idle_hp_first(self):
        cache = self._make_cache(available_host_tokens=0)
        short = self._append_node(cache.root_node, [1, 2], high_ref=0)
        long = self._append_node(cache.root_node, [3, 4, 5, 6], high_ref=0)
        self._register_hp(cache, "r_short", short)
        self._register_hp(cache, "r_long", long)

        victim = cache._select_shortest_hp_rid()
        self.assertEqual(victim, "r_short")

        moved = cache._adaptive_demote("r_short")
        self.assertEqual(moved, 2)
        self.assertEqual(short.high_ref, 0)
        self.assertEqual(short.low_ref, 1)
        self.assertIn("r_short", cache._adaptively_demoted_rids)

    def test_release_hp_restores_oldest_demoted(self):
        cache = self._make_cache(available_host_tokens=0)
        a = self._append_node(cache.root_node, [1, 2], high_ref=0)
        b = self._append_node(cache.root_node, [3, 4], high_ref=0)
        self._register_hp(cache, "ra", a)
        self._register_hp(cache, "rb", b)
        cache._adaptive_demote("ra")

        cache.release_ref("rb")

        self.assertNotIn("ra", cache._adaptively_demoted_rids)
        self.assertEqual(a.high_ref, 1)
        self.assertEqual(a.low_ref, 0)

    def test_register_ref_restores_demoted_rid_before_mismatch_check(self):
        cache = self._make_cache(available_host_tokens=0)
        a = self._append_node(cache.root_node, [1, 2], high_ref=0)
        self._register_hp(cache, "ra", a)
        cache._adaptive_demote("ra")

        # re-register as HP must restore instead of raising ValueError
        self._register_hp(cache, "ra", a)
        self.assertNotIn("ra", cache._adaptively_demoted_rids)
        self.assertTrue(cache.rid_to_ref_info["ra"].is_high)

    def test_update_ref_on_demoted_rid_restores(self):
        cache = self._make_cache(available_host_tokens=0)
        a = self._append_node(cache.root_node, [1, 2], high_ref=0)
        self._register_hp(cache, "ra", a)
        cache._adaptive_demote("ra")

        ok, msg = cache.update_ref("ra", 5)
        self.assertTrue(ok)
        self.assertIn("restored", msg)
        self.assertEqual(a.high_ref, 1)

    def test_register_ref_as_low_clears_demotion_record(self):
        """A demoted rid whose next turn is genuinely LP must leave the demoted
        set, otherwise release_ref of an unrelated HP rid force-restores it to
        HP while its requests are LP."""
        cache = self._make_cache(available_host_tokens=0)
        a = self._append_node(cache.root_node, [1, 2], high_ref=0)
        b = self._append_node(cache.root_node, [3, 4], high_ref=0)
        self._register_hp(cache, "ra", a)
        self._register_hp(cache, "rb", b)
        cache._adaptive_demote("ra")

        # "ra" comes back as genuinely low priority.
        cache.register_ref(SimpleNamespace(rid="ra", priority=0, last_node=a))
        self.assertNotIn("ra", cache._adaptively_demoted_rids)
        self.assertFalse(cache.rid_to_ref_info["ra"].is_high)

        # Releasing the unrelated HP rid must not force "ra" back to HP.
        cache.release_ref("rb")
        self.assertFalse(cache.rid_to_ref_info["ra"].is_high)
        self.assertEqual(a.high_ref, 0)
        self.assertEqual(a.low_ref, 1)

    def test_idle_hp_heap_holds_at_most_one_entry_per_rid(self):
        """register_ref runs once per finished turn; without dedupe the heap
        grows one tuple per turn for the process lifetime."""
        cache = self._make_cache(available_host_tokens=0)
        a = self._append_node(cache.root_node, [1, 2], high_ref=0)
        for _ in range(5):
            self._register_hp(cache, "ra", a)

        self.assertEqual(len(cache._idle_hp_heap), 1)
        self.assertEqual(cache._idle_hp_rids, {"ra"})


@unittest.skipUnless(HAS_HIRADIX, "hiradix import chain unavailable")
class TestDetachBackupedKeepsHost(_HiRadixHarness):
    def test_detach_backuped_does_not_free_device_indices(self):
        cache = self._make_cache(available_host_tokens=0)
        node = self._append_node(
            cache.root_node, [1, 2, 3, 4], evicted=False, backuped=True
        )
        cache.evictable_size_ = 4
        cache.unused_evictable_leaves.add(node)
        cache.unused_evictable_size_ = 4

        freed = cache._detach_backuped(node)

        self.assertEqual(freed, 4)
        self.assertIsNone(node.value)
        self.assertIsNotNone(node.host_value)
        # tier bookkeeping released along with the device copy
        self.assertNotIn(node, cache.unused_evictable_leaves)
        self.assertEqual(cache.unused_evictable_size_, 0)


if __name__ == "__main__":
    unittest.main()

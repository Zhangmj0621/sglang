"""Unit tests for ref-aware tiered KV cache eviction.

Group A: mixin-level tests runnable on any machine (RefAwareCacheMixin over
plain RadixCache via the _MixinCache harness).
Group B: HiRadix-level tests that require the full hiradix import chain
(skipped automatically when unavailable, e.g. on local dev machines).
"""

import sys
import types
import unittest
from collections import OrderedDict
from types import SimpleNamespace

import torch

# --- Local-dev stubs: only take effect when triton is unavailable (mac). ---
try:
    import triton  # noqa: F401
except ImportError:
    _triton = types.ModuleType("triton")
    _tl = types.ModuleType("triton.language")
    _triton.language = _tl
    _triton.jit = lambda fn=None, **kw: fn if callable(fn) else (lambda f: f)
    _triton.cdiv = lambda a, b: (a + b - 1) // b

    class _Constexpr:
        def __getitem__(self, item):
            return self

    _tl.constexpr = _Constexpr()
    sys.modules["triton"] = _triton
    sys.modules["triton.language"] = _tl
if not hasattr(torch.mps, "Stream"):
    torch.mps.Stream = torch.Stream

from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    _key_match_page_size1,
    get_child_key,
)
from sglang.srt.mem_cache.ref_aware_cache_mixin import RefInfo  # noqa: F401
from sglang.srt.mem_cache.ref_aware_cache_mixin import (
    TIER_HIGH_REF,
    TIER_LOW_REF,
    TIER_UNUSED,
    RefAwareCacheMixin,
    _classify_node_tier,
)

try:
    from sglang.srt.mem_cache.ref_aware_hiradix_cache import RefAwareHiRadixCache

    HAS_HIRADIX = True
except Exception:  # heavy import chain unavailable on local dev machines
    RefAwareHiRadixCache = None
    HAS_HIRADIX = False


class _MixinCache(RefAwareCacheMixin, RadixCache):
    """Light harness: mixin over plain RadixCache, constructed via __new__."""


# ---------------------------------------------------------------------------
# Group A: mixin-level tests (no hiradix import chain required)
# ---------------------------------------------------------------------------


class TestClassifyNodeTier(unittest.TestCase):
    """Test _classify_node_tier with different ref combinations."""

    def test_unused_both_zero(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 0
        self.assertEqual(_classify_node_tier(node), TIER_UNUSED)

    def test_low_ref_only(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 1
        self.assertEqual(_classify_node_tier(node), TIER_LOW_REF)

    def test_low_ref_large_value(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 100
        self.assertEqual(_classify_node_tier(node), TIER_LOW_REF)

    def test_high_ref_only(self):
        node = TreeNode()
        node.high_ref = 1
        node.low_ref = 0
        self.assertEqual(_classify_node_tier(node), TIER_HIGH_REF)

    def test_high_ref_overrides_low_ref(self):
        """When both high_ref and low_ref > 0, high_ref wins."""
        node = TreeNode()
        node.high_ref = 1
        node.low_ref = 5
        self.assertEqual(_classify_node_tier(node), TIER_HIGH_REF)

    def test_high_ref_large_value_overrides_low(self):
        node = TreeNode()
        node.high_ref = 10
        node.low_ref = 20
        self.assertEqual(_classify_node_tier(node), TIER_HIGH_REF)


class TestTreeNodeRefFields(unittest.TestCase):
    """Verify default values are 0/empty."""

    def test_default_high_ref_is_zero(self):
        node = TreeNode()
        self.assertEqual(node.high_ref, 0)

    def test_default_low_ref_is_zero(self):
        node = TreeNode()
        self.assertEqual(node.low_ref, 0)

    def test_default_tracked_rids_is_empty(self):
        node = TreeNode()
        self.assertEqual(node.tracked_rids, set())

    def test_new_node_classifies_as_unused(self):
        node = TreeNode()
        self.assertEqual(_classify_node_tier(node), TIER_UNUSED)


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
    # RadixCache-level state needed by inc/dec_lock_ref and _update_leaf_status
    cache.evictable_leaves = set()
    cache.disable = False
    cache.evictable_size_ = 0
    cache.protected_size_ = 0
    return cache


def _append_plain_node(parent, token_ids):
    node = TreeNode()
    node.parent = parent
    node.key = RadixKey(token_ids)
    node.value = torch.tensor(token_ids, dtype=torch.int64)
    node.children = {}
    parent.children[token_ids[0] if token_ids else 0] = node
    return node


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


class TestUpdateRefPropagatesPriority(unittest.TestCase):
    def test_update_ref_writes_back_priority_to_running_and_waiting_reqs(self):
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

        sched.tree_cache = _MixinCache.__new__(_MixinCache)
        # Patch the bound method so isinstance(cache, RefAwareCacheMixin) holds.
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


# ---------------------------------------------------------------------------
# Group B: HiRadix-level tests (require the full hiradix import chain)
# ---------------------------------------------------------------------------


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
        cache._evict_scope_stack = []
        cache.evictable_leaves = set()
        # Real key functions: _evict_host_from_tier resolves children via
        # get_child_key_fn (page_size=1 → first token, matching _append_node).
        cache.page_size = 1
        cache.get_child_key_fn = get_child_key
        cache.key_match_fn = _key_match_page_size1
        # _evict_host_from_tier emits remove events; keep them disabled.
        cache.enable_kv_cache_events = False
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


if __name__ == "__main__":
    unittest.main()

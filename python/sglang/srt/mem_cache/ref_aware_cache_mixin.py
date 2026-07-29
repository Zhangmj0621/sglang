"""
Mixin providing ref-aware tiered eviction logic.
"""

from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.ref_aware_cache_core import (  # noqa: F401  (re-export)
    TIER_HIGH_REF,
    TIER_LOW_REF,
    TIER_UNUSED,
    RefAwareCacheCore,
    RefInfo,
    _classify_node_tier,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


class RefAwareCacheMixin(RefAwareCacheCore):
    """
    Mixin that adds ref-aware tiered eviction to any radix-style cache.
    """

    def _init_ref_aware_state(self, server_args: ServerArgs):
        """
        Initialize all ref-aware tier tracking state.
        """
        self._init_ref_aware_core_state(server_args)
        self.unused_evictable_leaves: set = set()
        self.low_ref_evictable_leaves: set = set()
        self.high_ref_evictable_leaves: set = set()
        self.unused_evictable_size_: int = 0
        self.low_ref_evictable_size_: int = 0
        self.high_ref_evictable_size_: int = 0

    def _reset_ref_aware_state(self):
        """
        Clear all ref-aware tier tracking state.
        """
        self._reset_ref_aware_core_state()
        self.unused_evictable_leaves.clear()
        self.low_ref_evictable_leaves.clear()
        self.high_ref_evictable_leaves.clear()
        self.unused_evictable_size_ = 0
        self.low_ref_evictable_size_ = 0
        self.high_ref_evictable_size_ = 0

    def _on_node_tier_changed(self, node, old_tier: int, new_tier: int):
        # Guard mirrors the pre-refactor condition inside
        # _inc/_dec_priority_ref_single.
        if not node.evicted and node.lock_ref == 0:
            self._move_node_tier(node, old_tier, new_tier)

    def _move_node_tier(self, node: TreeNode, old_tier: int, new_tier: int):
        assert (
            not node.evicted and node.lock_ref == 0
        ), "_move_node_tier called for evicted or lock-held node"
        node_size = len(node.key)
        old_set = self._tier_leaf_set(old_tier)
        new_set = self._tier_leaf_set(new_tier)
        if node in old_set:
            old_set.discard(node)
            # Only re-add if node is still a valid evictable leaf
            is_leaf = all(c.evicted for c in node.children.values())
            if is_leaf:
                new_set.add(node)
        self._add_tier_size(old_tier, -node_size)
        self._add_tier_size(new_tier, node_size)

    def _tier_leaf_set(self, tier: int) -> set:
        if tier == TIER_UNUSED:
            return self.unused_evictable_leaves
        elif tier == TIER_LOW_REF:
            return self.low_ref_evictable_leaves
        else:
            return self.high_ref_evictable_leaves

    def _add_tier_size(self, tier: int, delta: int):
        if tier == TIER_UNUSED:
            self.unused_evictable_size_ += delta
        elif tier == TIER_LOW_REF:
            self.low_ref_evictable_size_ += delta
        else:
            self.high_ref_evictable_size_ += delta

    def _account_new_evictable_node(self, node: TreeNode):
        # Overrides RadixCache's no-op hook.
        if node in (None, self.root_node) or node.evicted or node.lock_ref > 0:
            return
        self._add_tier_size(_classify_node_tier(node), len(node.key))

    def _update_leaf_status(self, node: TreeNode):
        super()._update_leaf_status(node)
        self._update_ref_aware_leaf_status(node)

    def _update_ref_aware_leaf_status(self, node: TreeNode):
        self.unused_evictable_leaves.discard(node)
        self.low_ref_evictable_leaves.discard(node)
        self.high_ref_evictable_leaves.discard(node)

        if node.evicted or node.lock_ref > 0:
            return

        for child in node.children.values():
            if not child.evicted:
                return

        tier = _classify_node_tier(node)
        self._tier_leaf_set(tier).add(node)

    def _on_lock_ref_node(self, node: TreeNode):
        pass

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
                if not node.evicted:
                    tier = _classify_node_tier(node)
                    tier_set = self._tier_leaf_set(tier)
                    if node in tier_set:
                        tier_set.discard(node)
                    self._add_tier_size(tier, -len(node.key))
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._on_lock_ref_node(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
                if not node.evicted:
                    tier = _classify_node_tier(node)
                    self._add_tier_size(tier, len(node.key))
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._on_lock_ref_node(node)
            if node.parent is None:
                assert node is self.root_node
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _delete_leaf(self, node):
        tier = _classify_node_tier(node)
        self._tier_leaf_set(tier).discard(node)
        self._add_tier_size(tier, -len(node.key))
        self._untrack_node_rids(node)
        super()._delete_leaf(node)

    def evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        total = self.unused_evictable_size_
        if allow_low:
            total += self.low_ref_evictable_size_
        if allow_high:
            total += self.high_ref_evictable_size_
        return total

    def safe_evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        """Return safely evictable size by tier.

        Default implementation returns the same as evictable_size_by_tier.
        Override in HiCache variants where host-backed nodes change the
        calculation.
        """
        return self.evictable_size_by_tier(allow_low=allow_low, allow_high=allow_high)

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        protected_size = self.protected_size()
        pool_size = getattr(self.token_to_kv_pool_allocator, "size", None)
        tier_sum = (
            self.unused_evictable_size_
            + self.low_ref_evictable_size_
            + self.high_ref_evictable_size_
        )
        leaked = (
            pool_size - (available_size + evictable_size + protected_size)
            if pool_size is not None
            else None
        )
        return (
            f"Available tokens: {available_size + evictable_size} "
            f"({available_size=} + {evictable_size=}, "
            f"unused_evictable_size={self.unused_evictable_size_}, "
            f"low_ref_evictable_size={self.low_ref_evictable_size_}, "
            f"high_ref_evictable_size={self.high_ref_evictable_size_}, "
            f"{protected_size=}, {pool_size=}, {tier_sum=}, {leaked=})\n"
        )

    def _get_tier_priority(self, node: TreeNode, target_tier: int):
        """Compute eviction priority for a node. The heap pops the minimum.

        1. tier (unused < low_ref < high_ref), so the cheapest tier goes first
           even if a heap ever mixes tiers.
        2. high_ref (more -> evict later).
        3. low_ref (more -> evict later), only breaking ties in high_ref.
        4. the configured eviction strategy, identically for all three tiers.
        """
        _ = target_tier  # tier is derived from the node itself
        return (
            _classify_node_tier(node),
            node.high_ref,
            node.low_ref,
            self.eviction_strategy.get_priority(node),
        )

    def _evict_from_tier_heap(
        self,
        num_tokens: int,
        leaf_set: set,
        target_tier: int,
        evict_one_fn,
    ) -> int:
        """
        Shared heap-based eviction framework.
        """
        leaves = list(leaf_set)
        eviction_heap = [
            (self._get_tier_priority(node, target_tier), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and eviction_heap:
            _priority, x = heapq.heappop(eviction_heap)
            if x.lock_ref > 0:
                continue
            if _classify_node_tier(x) != target_tier:
                continue

            num_evicted += evict_one_fn(x)

            for child in x.parent.children.values():
                if not child.evicted:
                    break
            else:
                if x.parent.lock_ref == 0 and x.parent != self.root_node:
                    if _classify_node_tier(x.parent) == target_tier:
                        new_priority = self._get_tier_priority(x.parent, target_tier)
                        heapq.heappush(eviction_heap, (new_priority, x.parent))

        return num_evicted

    def _split_node(self, key, child, split_len):
        new_node = super()._split_node(key, child, split_len)
        new_node.high_ref = child.high_ref
        new_node.low_ref = child.low_ref
        new_node.tracked_rids = set(child.tracked_rids)
        # Update rid_to_ref_info: add new_node to each tracking rid's node set
        for rid in new_node.tracked_rids:
            ref_info = self.rid_to_ref_info.get(rid)
            if ref_info is not None:
                ref_info.nodes.add(new_node)
        self._update_ref_aware_leaf_status(new_node)
        self._update_ref_aware_leaf_status(child)
        return new_node

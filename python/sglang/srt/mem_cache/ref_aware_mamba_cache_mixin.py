"""Mixin providing ref-aware tiered eviction for MambaRadixCache."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.ref_aware_cache_core import (
    TIER_HIGH_REF,
    TIER_LOW_REF,
    TIER_UNUSED,
    RefAwareCacheCore,
    _classify_node_tier,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.mamba_radix_cache import TreeNode
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class RefAwareMambaCacheMixin(RefAwareCacheCore):
    """Ref-aware tier accounting and eviction over MambaRadixCache's dual
    resources (full KV tokens + mamba states). The LRU lists and tombstone
    semantics of the base class are untouched; tiering is implemented as
    per-tier size counters plus tier-filtered traversal of the existing
    LRU lists at eviction time."""

    def _init_ref_aware_state(self, server_args: ServerArgs):
        self._init_ref_aware_core_state(server_args)
        self._zero_tier_counters()

    def _reset_ref_aware_state(self):
        self._reset_ref_aware_core_state()
        self._zero_tier_counters()

    def _zero_tier_counters(self):
        self.full_unused_evictable_size_ = 0
        self.full_low_ref_evictable_size_ = 0
        self.full_high_ref_evictable_size_ = 0
        self.mamba_unused_evictable_size_ = 0
        self.mamba_low_ref_evictable_size_ = 0
        self.mamba_high_ref_evictable_size_ = 0

    # ---- tier counter primitives ----

    def _add_full_tier_size(self, tier: int, delta: int):
        if tier == TIER_UNUSED:
            self.full_unused_evictable_size_ += delta
        elif tier == TIER_LOW_REF:
            self.full_low_ref_evictable_size_ += delta
        else:
            self.full_high_ref_evictable_size_ += delta

    def _add_mamba_tier_size(self, tier: int, delta: int):
        if tier == TIER_UNUSED:
            self.mamba_unused_evictable_size_ += delta
        elif tier == TIER_LOW_REF:
            self.mamba_low_ref_evictable_size_ += delta
        else:
            self.mamba_high_ref_evictable_size_ += delta

    # ---- core hook ----

    def _on_node_tier_changed(self, node: TreeNode, old_tier: int, new_tier: int):
        if node.value is not None and node.full_lock_ref == 0:
            self._add_full_tier_size(old_tier, -len(node.value))
            self._add_full_tier_size(new_tier, len(node.value))
        if node.mamba_value is not None and node.mamba_lock_ref == 0:
            self._add_mamba_tier_size(old_tier, -len(node.mamba_value))
            self._add_mamba_tier_size(new_tier, len(node.mamba_value))

    # ---- base-class accounting hooks ----

    def _account_new_node_evictable(self, node: TreeNode):
        tier = _classify_node_tier(node)
        self._add_full_tier_size(tier, len(node.value))
        self._add_mamba_tier_size(tier, len(node.mamba_value))

    def _account_mamba_refill_evictable(self, node: TreeNode):
        self._add_mamba_tier_size(_classify_node_tier(node), len(node.mamba_value))

    # ---- structural overrides ----

    def _delete_leaf(self, node: TreeNode):
        tier = _classify_node_tier(node)
        self._add_full_tier_size(tier, -len(node.key))
        self._add_mamba_tier_size(tier, -len(node.mamba_value))
        self._untrack_node_rids(node)
        super()._delete_leaf(node)

    def _delete_tombstone_leaf(self, node: TreeNode):
        self._add_full_tier_size(_classify_node_tier(node), -len(node.key))
        self._untrack_node_rids(node)
        super()._delete_tombstone_leaf(node)

    def _tombstone_internal_node(self, node: TreeNode):
        self._add_mamba_tier_size(_classify_node_tier(node), -len(node.mamba_value))
        super()._tombstone_internal_node(node)

    def _split_node(self, key, child, split_len: int):
        new_node = super()._split_node(key, child, split_len)
        new_node.high_ref = child.high_ref
        new_node.low_ref = child.low_ref
        new_node.tracked_rids = set(child.tracked_rids)
        for rid in new_node.tracked_rids:
            ref_info = self.rid_to_ref_info.get(rid)
            if ref_info is not None:
                ref_info.nodes.add(new_node)
        # Full-tier counters need no adjustment: new_node and child share the
        # same tier and lock state, and their value lengths sum to the
        # original child's. The mamba value stays on child.
        return new_node

    # ---- lock overrides (copies of the base methods + tier lines) ----

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()

        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
                self._add_mamba_tier_size(
                    _classify_node_tier(node), -len(node.mamba_value)
                )
            node.mamba_lock_ref += 1

        while node != self.root_node:
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
                self._add_full_tier_size(_classify_node_tier(node), -len(node.value))
            node.full_lock_ref += 1
            node = node.parent
        return IncLockRefResult()

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()

        if node.mamba_value is not None:
            assert (
                node.mamba_lock_ref > 0
            ), f"dec_lock_ref on node with {node.mamba_lock_ref=}, {node.id=}"
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
                self._add_mamba_tier_size(
                    _classify_node_tier(node), len(node.mamba_value)
                )
            node.mamba_lock_ref -= 1

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
                self._add_full_tier_size(_classify_node_tier(node), len(node.value))
            node.full_lock_ref -= 1
            node = node.parent
        return DecLockRefResult()

    # ---- budget interfaces (full-token side, scheduler-facing) ----

    def evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        total = self.full_unused_evictable_size_
        if allow_low:
            total += self.full_low_ref_evictable_size_
        if allow_high:
            total += self.full_high_ref_evictable_size_
        return total

    def safe_evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        return self.evictable_size_by_tier(allow_low=allow_low, allow_high=allow_high)

    def mamba_evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        total = self.mamba_unused_evictable_size_
        if allow_low:
            total += self.mamba_low_ref_evictable_size_
        if allow_high:
            total += self.mamba_high_ref_evictable_size_
        return total

    # ---- observability ----

    def sanity_check(self):
        super().sanity_check()
        if not self.disable:
            self._sanity_check_tier_counters()

    def _sanity_check_tier_counters(self):
        full_sum = (
            self.full_unused_evictable_size_
            + self.full_low_ref_evictable_size_
            + self.full_high_ref_evictable_size_
        )
        mamba_sum = (
            self.mamba_unused_evictable_size_
            + self.mamba_low_ref_evictable_size_
            + self.mamba_high_ref_evictable_size_
        )
        assert full_sum == self.full_evictable_size_, (
            f"full tier counters leaked: {full_sum=} != "
            f"{self.full_evictable_size_=}"
        )
        assert mamba_sum == self.mamba_evictable_size_, (
            f"mamba tier counters leaked: {mamba_sum=} != "
            f"{self.mamba_evictable_size_=}"
        )
        # Full recompute from the tree, compared against incremental counters.
        expect = {("full", t): 0 for t in (TIER_UNUSED, TIER_LOW_REF, TIER_HIGH_REF)}
        expect.update(
            {("mamba", t): 0 for t in (TIER_UNUSED, TIER_LOW_REF, TIER_HIGH_REF)}
        )
        stack = list(self.root_node.children.values())
        while stack:
            n = stack.pop()
            tier = _classify_node_tier(n)
            if n.value is not None and n.full_lock_ref == 0:
                expect[("full", tier)] += len(n.value)
            if n.mamba_value is not None and n.mamba_lock_ref == 0:
                expect[("mamba", tier)] += len(n.mamba_value)
            stack.extend(n.children.values())
        actual = {
            ("full", TIER_UNUSED): self.full_unused_evictable_size_,
            ("full", TIER_LOW_REF): self.full_low_ref_evictable_size_,
            ("full", TIER_HIGH_REF): self.full_high_ref_evictable_size_,
            ("mamba", TIER_UNUSED): self.mamba_unused_evictable_size_,
            ("mamba", TIER_LOW_REF): self.mamba_low_ref_evictable_size_,
            ("mamba", TIER_HIGH_REF): self.mamba_high_ref_evictable_size_,
        }
        assert expect == actual, f"tier counter drift: {expect=} vs {actual=}"

    def available_and_evictable_str(self) -> str:
        return super().available_and_evictable_str() + (
            f"full tiers: unused={self.full_unused_evictable_size_}, "
            f"low={self.full_low_ref_evictable_size_}, "
            f"high={self.full_high_ref_evictable_size_}; "
            f"mamba tiers: unused={self.mamba_unused_evictable_size_}, "
            f"low={self.mamba_low_ref_evictable_size_}, "
            f"high={self.mamba_high_ref_evictable_size_}\n"
        )

    # ---- tiered eviction over the existing LRU lists ----
    #
    # Multi-pass by design: unused tier is drained (globally, across the
    # whole LRU list) before low-ref, which is drained before high-ref. This
    # is what makes tiering worth having -- a hot-but-low-priority KV chain
    # must never be evicted ahead of a cold-but-unreferenced one just because
    # it happens to sit closer to the LRU end within its own tier. Recency
    # only breaks ties *within* a tier.

    @staticmethod
    def _allowed_tiers(allow_low: bool, allow_high: bool):
        tiers = [TIER_UNUSED]
        if allow_low:
            tiers.append(TIER_LOW_REF)
        if allow_high:
            tiers.append(TIER_HIGH_REF)
        return tiers

    def _get_prev_no_lock_tier(self, lru_list, node, tier: int, leaf_only: bool):
        """Tier-filtered variant of LRUList.get_prev(_leaf)_no_lock: walk from
        `node` toward the MRU end, skipping locked nodes, non-leaves (when
        leaf_only), and nodes outside `tier`. `node` may be lru_list.tail."""
        prv = lru_list.prv
        lock = lru_list.lock_ref
        x = getattr(node, prv)
        while x is not lru_list.head and (
            getattr(x, lock) > 0
            or (leaf_only and len(x.children) > 0)
            or _classify_node_tier(x) != tier
        ):
            x = getattr(x, prv)
        return None if x is lru_list.head else x

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        allow_low, allow_high = self._current_evict_scope()
        full_n = (
            self.evict_full(params.num_tokens, allow_low, allow_high)
            if params.num_tokens > 0
            else 0
        )
        mamba_n = (
            self.evict_mamba(params.mamba_num, allow_low, allow_high)
            if params.mamba_num > 0
            else 0
        )
        return EvictResult(num_tokens_evicted=full_n, mamba_num_evicted=mamba_n)

    def _evict_tiered(self, num_tokens: int, allow_low: bool, allow_high: bool) -> int:
        # Full-token side only; signature aligned with
        # ScheduleBatch._evict_for_decode.
        return self.evict_full(num_tokens, allow_low, allow_high)

    def evict_full(
        self,
        full_num_tokens: int,
        allow_low: Optional[bool] = None,
        allow_high: Optional[bool] = None,
    ) -> int:
        if self.disable or full_num_tokens <= 0:
            return 0
        if allow_low is None or allow_high is None:
            allow_low, allow_high = self._current_evict_scope()
        evicted = 0
        for tier in self._allowed_tiers(allow_low, allow_high):
            if evicted >= full_num_tokens:
                break
            evicted += self._evict_full_one_tier(full_num_tokens - evicted, tier)
        return evicted

    def _evict_full_one_tier(self, num_tokens: int, tier: int) -> int:
        evicted = 0
        lru = self.full_lru_list
        x = self._get_prev_no_lock_tier(lru, lru.tail, tier, leaf_only=True)
        while evicted < num_tokens and x is not None and lru.in_list(x):
            # Compute the successor BEFORE eviction mutates the list. The
            # tombstone cascade only removes tombstone nodes, which are never
            # returned by the tier-filtered leaf walk, so x_next stays valid.
            x_next = self._get_prev_no_lock_tier(lru, x, tier, leaf_only=True)
            delta, _, deepest, _ = self._evict_leaf_node(x, False)
            evicted += delta
            if len(deepest.parent.children) == 0:
                # Parent became a leaf at an arbitrary LRU position; restart
                # the walk from the tail (mirrors base evict_full).
                x_next = self._get_prev_no_lock_tier(
                    lru, lru.tail, tier, leaf_only=True
                )
            x = x_next
        return evicted

    def evict_mamba(
        self,
        mamba_num: int,
        allow_low: Optional[bool] = None,
        allow_high: Optional[bool] = None,
    ) -> int:
        if self.disable or mamba_num <= 0:
            return 0
        if allow_low is None or allow_high is None:
            allow_low, allow_high = self._current_evict_scope()
        evicted = 0
        for tier in self._allowed_tiers(allow_low, allow_high):
            if evicted >= mamba_num:
                break
            evicted += self._evict_mamba_one_tier(mamba_num - evicted, tier)
        return evicted

    # ---- OOM escalation (last resort before crashing) ----

    def _escalate_and_evict_mamba(self, mamba_num: int):
        logger.warning(
            "Ref-aware mamba cache: scoped eviction exhausted; escalating to "
            "high-ref tier to avoid OOM (mamba_num=%d).",
            mamba_num,
        )
        with self.scoped_evict(allow_low=True, allow_high=True):
            self.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))

    def _alloc_mamba_slot_with_evict(self, last_node):
        pool = self.req_to_token_pool.mamba_pool
        dst_index = pool.alloc(1)
        if dst_index is None:
            self.inc_lock_ref(last_node)
            try:
                self.evict(EvictParams(num_tokens=0, mamba_num=1))
                dst_index = pool.alloc(1)
                if dst_index is None:
                    self._escalate_and_evict_mamba(1)
                    dst_index = pool.alloc(1)
            finally:
                self.dec_lock_ref(last_node)
            assert dst_index is not None, "Can not alloc mamba cache"
        return dst_index

    def _fork_mamba_with_evict(self, mamba_value):
        pool = self.req_to_token_pool.mamba_pool
        forked = pool.fork_from(mamba_value)
        if forked is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            forked = pool.fork_from(mamba_value)
        if forked is None:
            self._escalate_and_evict_mamba(1)
            forked = pool.fork_from(mamba_value)
        assert forked is not None, "Can not alloc mamba cache"
        return forked

    def _evict_mamba_one_tier(self, mamba_num: int, tier: int) -> int:
        evicted = 0
        lru = self.mamba_lru_list
        x = self._get_prev_no_lock_tier(lru, lru.tail, tier, leaf_only=False)
        while evicted < mamba_num and x is not None and lru.in_list(x):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            x_next = self._get_prev_no_lock_tier(lru, x, tier, leaf_only=False)
            if len(x.children) > 0:
                # Internal node: free mamba state and tombstone (mirrors base
                # evict_mamba; the tombstone cascade never touches the mamba
                # LRU list, so x_next stays valid).
                self.req_to_token_pool.mamba_pool.free(x.mamba_value)
                evicted += len(x.mamba_value)
                lru.remove_node(x)
                self._tombstone_internal_node(x)
            else:
                _, delta, _, _ = self._evict_leaf_node(x, True)
                evicted += delta
            x = x_next
        return evicted

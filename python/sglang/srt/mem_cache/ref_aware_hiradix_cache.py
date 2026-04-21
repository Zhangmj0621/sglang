from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


@dataclass
class RefInfo:
    is_high: bool
    nodes: Set[TreeNode] = field(default_factory=set)

logger = logging.getLogger(__name__)

# Eviction tier constants
TIER_UNUSED = 0    # high_ref == 0, low_ref == 0
TIER_LOW_REF = 1   # high_ref == 0, low_ref > 0
TIER_HIGH_REF = 2  # high_ref > 0


def _classify_node_tier(node: TreeNode) -> int:
    if node.high_ref > 0:
        return TIER_HIGH_REF
    if node.low_ref > 0:
        return TIER_LOW_REF
    return TIER_UNUSED


class RefAwareHiRadixCache(HiRadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self.high_priority_threshold = server_args.high_priority_threshold
        self.unused_evictable_leaves: set = set()
        self.low_ref_evictable_leaves: set = set()
        self.high_ref_evictable_leaves: set = set()
        self.unused_evictable_size_: int = 0
        self.low_ref_evictable_size_: int = 0
        self.high_ref_evictable_size_: int = 0
        self.rid_to_ref_info: Dict[str, RefInfo] = {}
        super().__init__(params=params, server_args=server_args)

    def reset(self):
        self.unused_evictable_leaves.clear()
        self.low_ref_evictable_leaves.clear()
        self.high_ref_evictable_leaves.clear()
        self.unused_evictable_size_ = 0
        self.low_ref_evictable_size_ = 0
        self.high_ref_evictable_size_ = 0
        self.rid_to_ref_info.clear()
        super().reset()

    def is_high_priority(self, priority: int) -> bool:
        return priority >= self.high_priority_threshold

    # --- Priority ref management ---

    def inc_priority_ref(self, node: TreeNode, is_high: bool):
        while node != self.root_node:
            old_tier = _classify_node_tier(node)
            if is_high:
                node.high_ref += 1
            else:
                node.low_ref += 1
            new_tier = _classify_node_tier(node)
            if node.lock_ref == 0 and old_tier != new_tier:
                self._move_node_tier(node, old_tier, new_tier)
            node = node.parent

    def dec_priority_ref(self, node: TreeNode, is_high: bool):
        while node != self.root_node:
            old_tier = _classify_node_tier(node)
            if is_high:
                node.high_ref = max(0, node.high_ref - 1)
            else:
                node.low_ref = max(0, node.low_ref - 1)
            new_tier = _classify_node_tier(node)
            if node.lock_ref == 0 and old_tier != new_tier:
                self._move_node_tier(node, old_tier, new_tier)
            node = node.parent

    def _move_node_tier(self, node: TreeNode, old_tier: int, new_tier: int):
        node_size = len(node.key)
        old_set = self._tier_leaf_set(old_tier)
        new_set = self._tier_leaf_set(new_tier)
        if node in old_set:
            old_set.discard(node)
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

    # --- Override leaf status tracking ---

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

    # --- Override inc_lock_ref / dec_lock_ref ---

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
                tier = _classify_node_tier(node)
                tier_set = self._tier_leaf_set(tier)
                if node in tier_set:
                    tier_set.discard(node)
                self._add_tier_size(tier, -len(node.key))
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
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
                tier = _classify_node_tier(node)
                self._add_tier_size(tier, len(node.key))
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert node is self.root_node
            node = node.parent
        return DecLockRefResult(delta=delta)

    # --- Override _delete_leaf ---

    def _delete_leaf(self, node):
        tier = _classify_node_tier(node)
        self._tier_leaf_set(tier).discard(node)
        self._add_tier_size(tier, -len(node.key))
        for rid in node.tracked_rids:
            ref_info = self.rid_to_ref_info.get(rid)
            if ref_info is not None:
                ref_info.nodes.discard(node)
        node.tracked_rids.clear()
        super()._delete_leaf(node)

    # --- Tiered eviction ---

    def evictable_size_by_tier(self, allow_low: bool = True, allow_high: bool = False) -> int:
        total = self.unused_evictable_size_
        if allow_low:
            total += self.low_ref_evictable_size_
        if allow_high:
            total += self.high_ref_evictable_size_
        return total

    def evict(self, params: EvictParams) -> EvictResult:
        allow_low = getattr(params, "allow_low", True)
        allow_high = getattr(params, "allow_high", False)
        return self._evict_tiered(params.num_tokens, allow_low, allow_high)

    def _evict_tiered(
        self, num_tokens: int, allow_low: bool = True, allow_high: bool = False
    ) -> EvictResult:
        start_time = time.perf_counter()
        num_evicted = 0

        num_evicted += self._evict_from_tier(
            num_tokens - num_evicted, self.unused_evictable_leaves, TIER_UNUSED
        )

        if allow_low and num_evicted < num_tokens:
            num_evicted += self._evict_from_tier(
                num_tokens - num_evicted, self.low_ref_evictable_leaves, TIER_LOW_REF
            )

        if allow_high and num_evicted < num_tokens:
            num_evicted += self._evict_from_tier(
                num_tokens - num_evicted, self.high_ref_evictable_leaves, TIER_HIGH_REF
            )

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_from_tier(self, num_tokens: int, leaf_set: set, target_tier: int) -> int:
        leaves = list(leaf_set)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []

        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if _classify_node_tier(x) != target_tier:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    written = self.write_backup(x, write_back=True)
                    num_evicted += written
                    if written > 0:
                        write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                if x.parent.lock_ref == 0 and x.parent != self.root_node:
                    if _classify_node_tier(x.parent) == target_tier:
                        new_priority = self.eviction_strategy.get_priority(x.parent)
                        heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        return num_evicted

    # --- Tiered host eviction ---

    def evict_host(self, num_tokens: int, allow_high: bool = True):
        num_evicted = 0
        num_evicted += self._evict_host_from_tier(
            num_tokens - num_evicted, TIER_UNUSED
        )
        if num_evicted < num_tokens:
            num_evicted += self._evict_host_from_tier(
                num_tokens - num_evicted, TIER_LOW_REF
            )
        if allow_high and num_evicted < num_tokens:
            num_evicted += self._evict_host_from_tier(
                num_tokens - num_evicted, TIER_HIGH_REF
            )
        return num_evicted

    def _evict_host_from_tier(self, num_tokens: int, target_tier: int) -> int:
        leaves = [
            n for n in self.evictable_host_leaves
            if n.evicted and n.host_ref_counter == 0 and _classify_node_tier(n) == target_tier
        ]
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            if not x.evicted or x.host_ref_counter > 0:
                continue
            if _classify_node_tier(x) != target_tier:
                continue

            from sglang.srt.disaggregation.kv_events import StorageMedium
            self._record_remove_event(x, medium=StorageMedium.CPU)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            key = self.get_child_key_fn(x.key)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            for rid in x.tracked_rids:
                ref_info = self.rid_to_ref_info.get(rid)
                if ref_info is not None:
                    ref_info.nodes.discard(x)
            x.tracked_rids.clear()
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                if _classify_node_tier(x.parent) == target_tier:
                    new_priority = self.eviction_strategy.get_priority(x.parent)
                    heapq.heappush(eviction_heap, (new_priority, x.parent))

        return num_evicted

    def write_backup(self, node: TreeNode, write_back=False) -> int:
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0

        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )
        if host_indices is None:
            # Determine eviction permission based on the node's ref tier
            allow_high = (node.high_ref > 0)
            self.evict_host(len(node.value), allow_high=allow_high)
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                **self._get_extra_pools(),
            )
        if host_indices is not None:
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0
            self.ongoing_write_through[node.id] = node
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    # --- Explicit ref management for RL multi-turn ---

    def register_ref(self, req: Req):
        rid = req.rid
        is_high = self.is_high_priority(getattr(req, "priority", 0) or 0)
        is_first_turn = getattr(req, "is_first_turn", False)

        if rid not in self.rid_to_ref_info:
            self.rid_to_ref_info[rid] = RefInfo(is_high=is_high)

        ref_info = self.rid_to_ref_info[rid]

        token_ids = (req.origin_input_ids + req.output_ids)[: req.kv_committed_len]
        if not token_ids:
            return

        radix_key = RadixKey(
            list(token_ids), getattr(req, "extra_key", None)
        ).page_aligned(self.page_size)
        if len(radix_key) == 0:
            return

        nodes_on_path = self._collect_nodes_on_path(radix_key)

        # For non-first turn: set difference gives only NEW nodes (extend part).
        # If previous cache was fully evicted (ref_info.nodes all gone),
        # set difference = all nodes on path, which correctly re-adds ref to all tokens.
        new_nodes = set(nodes_on_path) - ref_info.nodes

        for node in new_nodes:
            self._inc_priority_ref_single(node, is_high)
            ref_info.nodes.add(node)
            node.tracked_rids.add(rid)

    def _collect_nodes_on_path(self, key: RadixKey):
        node = self.root_node
        nodes = []
        child_key_fn = self.get_child_key_fn

        while len(key) > 0:
            ck = child_key_fn(key)
            if ck not in node.children:
                break
            child = node.children[ck]
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len <= 0:
                break
            nodes.append(child)
            if prefix_len < len(child.key):
                break
            key = key[prefix_len:]
        return nodes

    def _inc_priority_ref_single(self, node: TreeNode, is_high: bool):
        old_tier = _classify_node_tier(node)
        if is_high:
            node.high_ref += 1
        else:
            node.low_ref += 1
        new_tier = _classify_node_tier(node)
        if node.lock_ref == 0 and old_tier != new_tier:
            self._move_node_tier(node, old_tier, new_tier)

    def _dec_priority_ref_single(self, node: TreeNode, is_high: bool):
        old_tier = _classify_node_tier(node)
        if is_high:
            node.high_ref = max(0, node.high_ref - 1)
        else:
            node.low_ref = max(0, node.low_ref - 1)
        new_tier = _classify_node_tier(node)
        if node.lock_ref == 0 and old_tier != new_tier:
            self._move_node_tier(node, old_tier, new_tier)

    def release_ref(self, rid: str) -> Tuple[bool, str]:
        ref_info = self.rid_to_ref_info.pop(rid, None)
        if ref_info is None:
            return False, f"rid {rid} not found in ref tracking"

        for node in ref_info.nodes:
            self._dec_priority_ref_single(node, ref_info.is_high)
            node.tracked_rids.discard(rid)
        return True, f"released {len(ref_info.nodes)} nodes for rid {rid}"

    def update_ref(self, rid: str, new_priority: int) -> Tuple[bool, str]:
        ref_info = self.rid_to_ref_info.get(rid)
        if ref_info is None:
            return False, f"rid {rid} not found in ref tracking"

        new_is_high = self.is_high_priority(new_priority)
        if new_is_high == ref_info.is_high:
            return True, "priority class unchanged"

        for node in ref_info.nodes:
            self._dec_priority_ref_single(node, ref_info.is_high)
            self._inc_priority_ref_single(node, new_is_high)
        ref_info.is_high = new_is_high
        return True, f"updated {len(ref_info.nodes)} nodes for rid {rid}"

    # --- Split node override to propagate high_ref/low_ref ---

    def _split_node(self, key, child, split_len):
        new_node = super()._split_node(key, child, split_len)
        new_node.high_ref = child.high_ref
        new_node.low_ref = child.low_ref
        # new_node inherits tracked_rids from child (already done in RadixCache._split_node)
        # Update rid_to_ref_info: add new_node to each tracking rid's node set
        for rid in new_node.tracked_rids:
            ref_info = self.rid_to_ref_info.get(rid)
            if ref_info is not None:
                ref_info.nodes.add(new_node)
        self._update_ref_aware_leaf_status(new_node)
        self._update_ref_aware_leaf_status(child)
        return new_node

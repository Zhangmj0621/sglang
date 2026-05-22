from __future__ import annotations

from contextlib import contextmanager
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
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode, compute_node_hash_values

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
        self._evict_scope_stack: list[tuple[bool, bool]] = []
        super().__init__(params=params, server_args=server_args)

    def reset(self):
        self.unused_evictable_leaves.clear()
        self.low_ref_evictable_leaves.clear()
        self.high_ref_evictable_leaves.clear()
        self.unused_evictable_size_ = 0
        self.low_ref_evictable_size_ = 0
        self.high_ref_evictable_size_ = 0
        self.rid_to_ref_info.clear()
        self._evict_scope_stack.clear()
        super().reset()

    def is_high_priority(self, priority: int) -> bool:
        return priority >= self.high_priority_threshold

    def _move_node_tier(self, node: TreeNode, old_tier: int, new_tier: int):
        node_size = len(node.key)
        old_set = self._tier_leaf_set(old_tier)
        new_set = self._tier_leaf_set(new_tier)
        if node in old_set:
            old_set.discard(node)
            new_set.add(node)
        if not node.evicted and node.lock_ref == 0:
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
        if node in (None, self.root_node) or node.evicted or node.lock_ref > 0:
            return
        self._add_tier_size(_classify_node_tier(node), len(node.key))

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
                if not node.evicted:
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
                if not node.evicted:
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

    def _tier_rank(self, tier: int, allow_low: bool = True, allow_high: bool = False):
        if tier == TIER_UNUSED:
            return 0
        if tier == TIER_LOW_REF:
            return 1 if allow_low else None
        if tier == TIER_HIGH_REF:
            return 2 if allow_high else None
        return None

    def host_evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        removed_nodes = set()
        eviction_heap = []

        for node in self.evictable_host_leaves:
            tier = _classify_node_tier(node)
            tier_rank = self._tier_rank(tier, allow_low=allow_low, allow_high=allow_high)
            if (
                tier_rank is None
                or not node.evicted
                or node.host_ref_counter > 0
                or node.host_value is None
            ):
                continue
            heapq.heappush(
                eviction_heap,
                ((tier_rank, self._get_tier_priority(node, tier)), node),
            )

        num_evicted = 0
        while eviction_heap:
            (_priority, x) = heapq.heappop(eviction_heap)

            if (
                x in removed_nodes
                or x == self.root_node
                or not x.evicted
                or x.host_ref_counter > 0
                or x.host_value is None
            ):
                continue

            tier = _classify_node_tier(x)
            tier_rank = self._tier_rank(tier, allow_low=allow_low, allow_high=allow_high)
            if tier_rank is None:
                continue

            num_evicted += len(x.host_value)
            removed_nodes.add(x)

            parent = x.parent
            if parent in (None, self.root_node):
                continue

            if (
                parent.evicted
                and parent.host_ref_counter == 0
                and parent.host_value is not None
                and all(child in removed_nodes for child in parent.children.values())
            ):
                parent_tier = _classify_node_tier(parent)
                parent_rank = self._tier_rank(
                    parent_tier, allow_low=allow_low, allow_high=allow_high
                )
                if parent_rank is not None:
                    heapq.heappush(
                        eviction_heap,
                        (
                            (parent_rank, self._get_tier_priority(parent, parent_tier)),
                            parent,
                        ),
                    )

        return num_evicted

    def high_ref_host_safe_evictable_size(self) -> int:
        if self.cache_controller.write_policy != "write_back":
            return self.high_ref_evictable_size_

        host_budget = self.cache_controller.mem_pool_host.available_size()
        host_budget += self.host_evictable_size_by_tier(
            allow_low=True,
            allow_high=False,
        )

        num_evicted = 0
        virtually_evicted = set()
        eviction_heap = [
            ((0, self._get_tier_priority(node, TIER_HIGH_REF)), node)
            for node in self.high_ref_evictable_leaves
        ]
        heapq.heapify(eviction_heap)

        while eviction_heap:
            (_priority, x) = heapq.heappop(eviction_heap)
            if (
                x in virtually_evicted
                or x.lock_ref > 0
                or x.evicted
                or _classify_node_tier(x) != TIER_HIGH_REF
            ):
                continue

            spill_cost = 0 if x.backuped else len(x.value)
            if spill_cost > host_budget:
                continue

            host_budget -= spill_cost
            num_evicted += len(x.key)
            virtually_evicted.add(x)

            parent = x.parent
            if (
                parent not in (None, self.root_node)
                and parent.lock_ref == 0
                and _classify_node_tier(parent) == TIER_HIGH_REF
                and all(
                    child.evicted or child in virtually_evicted
                    for child in parent.children.values()
                )
            ):
                heapq.heappush(
                    eviction_heap,
                    ((0, self._get_tier_priority(parent, TIER_HIGH_REF)), parent),
                )

        return num_evicted

    def safe_evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        total = self.unused_evictable_size_
        if allow_low:
            total += self.low_ref_evictable_size_
        if allow_high:
            total += self.high_ref_host_safe_evictable_size()
        return total

    @contextmanager
    def scoped_evict(self, allow_low: bool = True, allow_high: bool = False):
        self._evict_scope_stack.append((allow_low, allow_high))
        try:
            yield
        finally:
            self._evict_scope_stack.pop()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return (
            f"Available tokens: {available_size + evictable_size} "
            f"({available_size=} + {evictable_size=}, "
            f"unused_evictable_size={self.unused_evictable_size_}, "
            f"low_ref_evictable_size={self.low_ref_evictable_size_}, "
            f"high_ref_evictable_size={self.high_ref_evictable_size_})\n"
        )

    def evict(self, params: EvictParams) -> EvictResult:
        if self._evict_scope_stack:
            allow_low, allow_high = self._evict_scope_stack[-1]
        else:
            allow_low = True
            allow_high = False
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

    def _get_tier_priority(self, node: TreeNode, target_tier: int):
        # For multi-turn interactive workloads, a high_ref node that became idle
        # earlier is often more likely to be reused sooner. Evict the most recently
        # idle high_ref nodes first instead of applying the global LRU policy there.
        if target_tier == TIER_HIGH_REF:
            return -node.last_access_time
        return self.eviction_strategy.get_priority(node)

    def _evict_from_tier(self, num_tokens: int, leaf_set: set, target_tier: int) -> int:
        leaves = list(leaf_set)
        eviction_heap = [
            (self._get_tier_priority(node, target_tier), node) for node in leaves
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
                        new_priority = self._get_tier_priority(x.parent, target_tier)
                        heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        return num_evicted

    # --- Tiered host eviction ---

    def evict_host(self, num_tokens: int, allow_high: bool = False):
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
            (self._get_tier_priority(node, target_tier), node) for node in leaves
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
                    new_priority = self._get_tier_priority(x.parent, target_tier)
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
            # Never evict host-side high_ref nodes to make room for a spill.
            # Otherwise a newly admitted high-priority request can cause another
            # referenced high-priority prefix to be dropped from DRAM entirely.
            self.evict_host(len(node.value), allow_high=False)
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

        if rid not in self.rid_to_ref_info:
            self.rid_to_ref_info[rid] = RefInfo(is_high=is_high)

        ref_info = self.rid_to_ref_info[rid]

        last_node = getattr(req, "last_node", None)
        if last_node not in (None, self.root_node):
            new_nodes = self._collect_untracked_nodes_from_last_node(
                last_node, ref_info.nodes
            )
        else:
            token_ids = (req.origin_input_ids + req.output_ids)[: req.kv_committed_len]
            if not token_ids:
                return

            radix_key = RadixKey(
                list(token_ids), getattr(req, "extra_key", None)
            ).page_aligned(self.page_size)
            if len(radix_key) == 0:
                return

            nodes_on_path = self._collect_nodes_on_path(radix_key)
            new_nodes = [node for node in nodes_on_path if node not in ref_info.nodes]

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

    def _collect_untracked_nodes_from_last_node(
        self, node: Optional[TreeNode], tracked_nodes: Set[TreeNode]
    ) -> list[TreeNode]:
        nodes = []
        while node not in (None, self.root_node):
            if node in tracked_nodes:
                break
            nodes.append(node)
            node = node.parent
        return nodes

    def _inc_priority_ref_single(self, node: TreeNode, is_high: bool):
        old_tier = _classify_node_tier(node)
        if is_high:
            node.high_ref += 1
        else:
            node.low_ref += 1
        new_tier = _classify_node_tier(node)
        if not node.evicted and node.lock_ref == 0 and old_tier != new_tier:
            self._move_node_tier(node, old_tier, new_tier)

    def _dec_priority_ref_single(self, node: TreeNode, is_high: bool):
        old_tier = _classify_node_tier(node)
        if is_high:
            node.high_ref = max(0, node.high_ref - 1)
        else:
            node.low_ref = max(0, node.low_ref - 1)
        new_tier = _classify_node_tier(node)
        if not node.evicted and node.lock_ref == 0 and old_tier != new_tier:
            self._move_node_tier(node, old_tier, new_tier)

    def _evict_backuped(self, node: TreeNode):
        tier = _classify_node_tier(node)
        self._tier_leaf_set(tier).discard(node)
        self._add_tier_size(tier, -len(node.key))
        return super()._evict_backuped(node)

    def release_ref(self, rid: str) -> Tuple[bool, str]:
        ref_info = self.rid_to_ref_info.pop(rid, None)
        if ref_info is None:
            return True, f"rid {rid} not tracked"

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

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None, req: Optional["Req"] = None
    ) -> Optional[torch.Tensor]:
        from sglang.srt.disaggregation.kv_events import StorageMedium

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            **self._get_extra_pools(),
        )
        if device_indices is None:
            allow_high = (
                req is not None
                and self.is_high_priority(getattr(req, "priority", 0) or 0)
            )
            self._evict_tiered(
                len(host_indices),
                allow_low=True,
                allow_high=allow_high,
            )
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
            self._account_new_evictable_node(node)
            # Block promoted from host to GPU -- emit store(GPU) so downstream
            # indexers see it as device-local again.
            self._record_store_event(node, medium=StorageMedium.GPU)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        req = params.req
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota, req=req)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def _insert_with_last_node(
        self, params: InsertParams
    ) -> tuple[InsertResult, Optional[TreeNode]]:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]

        if len(key) == 0:
            return InsertResult(prefix_len=0), self.root_node

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)
                    self._account_new_evictable_node(node)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._account_new_evictable_node(new_node)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        last_node = node
        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._account_new_evictable_node(new_node)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage or kv events are enabled
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(
                    new_node, self.page_size
                )

            # Emit BlockStored so the router indexes this block.
            self._record_store_event(new_node)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
            last_node = new_node

        return InsertResult(prefix_len=total_prefix_length), last_node

    def insert(self, params: InsertParams) -> InsertResult:
        result, _ = self._insert_with_last_node(params)
        return result

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        # In deterministic mode, disable finished request insertion to radix cache.
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        old_last_node = req.last_node
        new_last_node = old_last_node

        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result, new_last_node = self._insert_with_last_node(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            new_prefix_len = result.prefix_len
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
            req.last_node = new_last_node
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : key_len]
            )

        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])
        self.dec_lock_ref(old_last_node)

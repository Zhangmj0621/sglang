from __future__ import annotations

import heapq
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InitLoadBackParams,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.ref_aware_cache_mixin import (
    TIER_HIGH_REF,
    TIER_LOW_REF,
    TIER_UNUSED,
    RefAwareCacheMixin,
    _classify_node_tier,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


class RefAwareHiRadixCache(RefAwareCacheMixin, HiRadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._init_ref_aware_state(server_args)
        self._adaptively_demoted_rids: OrderedDict[str, int] = OrderedDict()
        self._idle_hp_heap: list[tuple[int, str]] = []
        # Mirrors exactly the set of rids present in _idle_hp_heap, so a rid is
        # queued at most once. Without it the heap grows one tuple per finished
        # HP turn forever on a server that never runs a demotion scan.
        self._idle_hp_rids: set[str] = set()
        super().__init__(params=params, server_args=server_args)

    def reset(self):
        self._reset_ref_aware_state()
        self._adaptively_demoted_rids.clear()
        self._idle_hp_heap.clear()
        self._idle_hp_rids.clear()
        super().reset()

    def _push_idle_hp(self, rid: str, cached_tokens: int) -> None:
        """Queue an idle HP rid as a demotion candidate, at most one entry each."""
        if rid in self._idle_hp_rids:
            return
        self._idle_hp_rids.add(rid)
        heapq.heappush(self._idle_hp_heap, (cached_tokens, rid))

    def _on_lock_ref_node(self, node: TreeNode):
        self._update_host_leaf_status(node)

    def safe_evictable_size_by_tier(
        self, allow_low: bool = True, allow_high: bool = False
    ) -> int:
        # A high-priority eviction scope (allow_high) can free every high-ref
        # device node -- backuped ones via `_evict_backuped` (host copy kept),
        # the rest via `_evict_regular`. So the admission budget equals the full
        # high-ref evictable size, keeping admission consistent with what
        # eviction can actually reclaim. O(1) counter read.
        total = self.unused_evictable_size_
        if allow_low:
            total += self.low_ref_evictable_size_
        if allow_high:
            total += self.high_ref_evictable_size_
        return total

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

    def _evict_from_tier(self, num_tokens: int, leaf_set: set, target_tier: int) -> int:
        if self.cache_controller.write_policy == "write_back":
            return self._evict_from_tier_write_back(num_tokens, leaf_set, target_tier)
        else:
            return self._evict_from_tier_write_through(
                num_tokens, leaf_set, target_tier
            )

    def _make_tier_eviction_heap(self, leaf_set: set, target_tier: int):
        heap = [(self._get_tier_priority(node, target_tier), node) for node in leaf_set]
        heapq.heapify(heap)
        return heap

    def _promote_tier_parent(self, node: TreeNode, heap, target_tier: int):
        p = node.parent
        if (
            p is not self.root_node
            and _classify_node_tier(p) == target_tier
            and all(c.evicted for c in p.children.values())
        ):
            heapq.heappush(heap, (self._get_tier_priority(p, target_tier), p))

    def _evict_from_tier_write_through(
        self, num_tokens: int, leaf_set: set, target_tier: int
    ) -> int:
        heap = self._make_tier_eviction_heap(leaf_set, target_tier)
        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0:
                continue
            if _classify_node_tier(x) != target_tier:
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            else:
                num_evicted += self._evict_regular(x)
            self._promote_tier_parent(x, heap, target_tier)
        return num_evicted

    def _evict_from_tier_write_back(
        self, num_tokens: int, leaf_set: set, target_tier: int
    ) -> int:
        """eviction for write_back mode: demote already-backuped leaves, stage
        non-backuped ones to host if possible, otherwise drop them.
        note this path will be deprecated in the future.
        """
        heap = self._make_tier_eviction_heap(leaf_set, target_tier)
        num_evicted = 0
        staged: List[Tuple[TreeNode, torch.Tensor]] = []

        def flush_staged() -> None:
            if not staged:
                return
            self.writing_check(write_back=True)
            for node, device_indices in staged:
                self.cache_controller.evict_device(device_indices)
                node.release_host()
            staged.clear()

        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0:
                continue
            if _classify_node_tier(x) != target_tier:
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            elif self.write_backup(x, write_back=True) > 0:
                x.protect_host()
                staged.append((x, x.value))
                num_evicted += self._detach_backuped(x)
            else:
                flush_staged()
                num_evicted += self._drop_subtree_no_host(x)
            self._promote_tier_parent(x, heap, target_tier)
        flush_staged()
        return num_evicted

    def _detach_backuped(self, node: TreeNode) -> int:
        tier = _classify_node_tier(node)
        self._tier_leaf_set(tier).discard(node)
        self._add_tier_size(tier, -len(node.key))
        return super()._detach_backuped(node)

    def _drop_subtree_no_host(self, root: TreeNode) -> int:
        nodes = []
        stack = [root]
        while stack:
            n = stack.pop()
            nodes.append(n)
            stack.extend(n.children.values())

        if any(n.host_ref_counter > 0 for n in nodes):
            return 0

        logger.warning(
            "write_back: KV cache on device are dropped without backup "
            "due to host memory pressure, subtree root %d, num_nodes %d",
            root.id,
            len(nodes),
        )

        freed_device = 0
        for n in nodes:
            tier = _classify_node_tier(n)
            self._tier_leaf_set(tier).discard(n)
            if n.host_value is not None:
                self.cache_controller.evict_host(n.host_value)
                n.host_value = None
            if n.value is not None:
                self.cache_controller.mem_pool_device_allocator.free(n.value)
                freed_device += len(n.value)
                self.evictable_size_ -= len(n.value)
                self._add_tier_size(tier, -len(n.key))
                n.value = None
            self.ongoing_write_through.pop(n.id, None)
            self.evictable_leaves.discard(n)
            self.evictable_host_leaves.discard(n)
            for rid in n.tracked_rids:
                ref_info = self.rid_to_ref_info.get(rid)
                if ref_info is not None:
                    ref_info.nodes.discard(n)
            n.tracked_rids.clear()
            # node removed from the tree entirely
            self._record_remove_event(n)

        key = self.get_child_key_fn(root.key)
        root.parent.children.pop(key, None)
        self._update_leaf_status(root.parent)
        self._update_host_leaf_status(root.parent)
        return freed_device

    # --- Adaptive HP demotion on host pressure ---

    def _select_shortest_hp_rid(self) -> Optional[str]:
        """O(1) amortized: pop from min-heap, skip stale entries."""
        while self._idle_hp_heap:
            _tokens, rid = self._idle_hp_heap[0]
            ref_info = self.rid_to_ref_info.get(rid)
            if (
                ref_info is None
                or not ref_info.is_high
                or ref_info.is_generating
                or rid in self._adaptively_demoted_rids
            ):
                heapq.heappop(self._idle_hp_heap)
                self._idle_hp_rids.discard(rid)
                continue
            return rid
        return None

    def _adaptive_demote(self, rid: str) -> int:
        """Demote rid from HP to LP at cache tier level only. Returns tokens moved."""
        ref_info = self.rid_to_ref_info.get(rid)
        if ref_info is None or not ref_info.is_high:
            return 0

        tokens_moved = 0
        for node in ref_info.nodes:
            self._dec_priority_ref_single(node, True)
            self._inc_priority_ref_single(node, False)
            tokens_moved += len(node.key)

        ref_info.is_high = False
        self._adaptively_demoted_rids[rid] = self.high_priority_threshold
        logger.info(
            "[adaptive-demote] rid=%s tokens=%d demoted_count=%d",
            rid,
            tokens_moved,
            len(self._adaptively_demoted_rids),
        )
        return tokens_moved

    def _adaptive_restore(self, rid: str) -> bool:
        """Restore a previously demoted rid back to HP. Returns True if restored."""
        original_priority = self._adaptively_demoted_rids.pop(rid, None)
        if original_priority is None:
            return False

        ref_info = self.rid_to_ref_info.get(rid)
        if ref_info is None:
            return False

        tokens_restored = 0
        for node in ref_info.nodes:
            self._dec_priority_ref_single(node, False)
            self._inc_priority_ref_single(node, True)
            tokens_restored += len(node.key)
        ref_info.is_high = True
        if not ref_info.is_generating:
            self._push_idle_hp(rid, ref_info.cached_tokens)
        logger.info(
            "[adaptive-restore] rid=%s tokens=%d demoted_remaining=%d",
            rid,
            tokens_restored,
            len(self._adaptively_demoted_rids),
        )
        return True

    def evict_host(self, num_tokens: int, allow_high: bool = False):
        num_evicted = 0
        num_evicted += self._evict_host_from_tier(num_tokens - num_evicted, TIER_UNUSED)
        if num_evicted < num_tokens:
            num_evicted += self._evict_host_from_tier(
                num_tokens - num_evicted, TIER_LOW_REF
            )

        if allow_high and num_evicted < num_tokens:
            # Adaptive demotion: demote shortest HP request(s) to LP so their
            # host nodes become LOW_REF-evictable, avoiding permanent loss of
            # TIER_HIGH_REF entries which would require full recomputation.
            while num_evicted < num_tokens:
                victim_rid = self._select_shortest_hp_rid()
                if victim_rid is None:
                    break
                self._adaptive_demote(victim_rid)
                num_evicted += self._evict_host_from_tier(
                    num_tokens - num_evicted, TIER_LOW_REF
                )

            if num_evicted < num_tokens:
                num_evicted += self._evict_host_from_tier(
                    num_tokens - num_evicted, TIER_HIGH_REF
                )

        return num_evicted

    def _evict_host_from_tier(self, num_tokens: int, target_tier: int) -> int:
        leaves = [
            n
            for n in self.evictable_host_leaves
            if n.evicted
            and n.host_ref_counter == 0
            and _classify_node_tier(n) == target_tier
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

            self._record_remove_event(x)
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
        )
        if host_indices is None:
            self.evict_host(len(node.value), allow_high=True)
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
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

    def mark_rid_generating(self, rid: str):
        """Called by scheduler when a request with this rid enters the batch.
        O(1) flag flip — prevents this rid from being selected for demotion."""
        ref_info = self.rid_to_ref_info.get(rid)
        if ref_info is not None:
            ref_info.is_generating = True

    # --- Explicit ref management for RL multi-turn (adaptive-demotion aware) ---

    def register_ref(self, req: Req):
        rid = req.rid
        is_high = self.is_high_priority(getattr(req, "priority", 0) or 0)

        # If this rid was adaptively demoted and now re-enters as HP, restore it
        # first so the mixin's priority-class check sees consistent state.
        if rid in self._adaptively_demoted_rids:
            if is_high:
                self._adaptive_restore(rid)
            else:
                # Re-registered as genuinely low priority. Drop the demotion
                # record: otherwise release_ref() of some other HP rid could
                # force-restore this one back to HP while its requests are LP,
                # and the next turn would hit the priority-class mismatch.
                self._adaptively_demoted_rids.pop(rid, None)

        super().register_ref(req)

        ref_info = self.rid_to_ref_info.get(rid)
        if ref_info is None:
            return
        ref_info.is_generating = False
        if ref_info.is_high and rid not in self._adaptively_demoted_rids:
            self._push_idle_hp(rid, ref_info.cached_tokens)

    def release_ref(self, rid: str) -> Tuple[bool, str]:
        self._adaptively_demoted_rids.pop(rid, None)
        ref_info = self.rid_to_ref_info.get(rid)
        was_high = ref_info.is_high if ref_info is not None else False

        ok, msg = super().release_ref(rid)

        # A released HP rollout frees host budget: restore the earliest
        # adaptively-demoted rid back to HP.
        if was_high and self._adaptively_demoted_rids:
            self._adaptive_restore(next(iter(self._adaptively_demoted_rids)))
        return ok, msg

    def update_ref(self, rid: str, new_priority: int) -> Tuple[bool, str]:
        ref_info = self.rid_to_ref_info.get(rid)
        if ref_info is None:
            return False, f"rid {rid} not found in ref tracking"

        if rid in self._adaptively_demoted_rids:
            ref_info.priority = new_priority
            if self.is_high_priority(new_priority):
                self._adaptive_restore(rid)
                return (
                    True,
                    f"restored adaptively-demoted rid {rid} via external update_ref",
                )
            self._adaptively_demoted_rids.pop(rid, None)
            return True, "priority class unchanged (already demoted)"

        return super().update_ref(rid, new_priority)

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None, req: Optional[Req] = None
    ) -> Optional[torch.Tensor]:
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

        # Protect the nodes being loaded from host eviction.
        for n in nodes_to_load:
            n.protect_host()

        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
        )
        if device_indices is None:
            allow_high = req is not None and self.is_high_priority(
                getattr(req, "priority", 0) or 0
            )
            self._evict_tiered(
                len(host_indices),
                allow_low=True,
                allow_high=allow_high,
            )
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            for n in nodes_to_load:
                n.release_host()
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        for n in nodes_to_load:
            n.release_host()

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
            self._account_new_evictable_node(node)
            # Block promoted from host to GPU -- emit store so downstream
            # indexers see it as device-local again.
            self._record_store_event(node)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(self, params: InitLoadBackParams):
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

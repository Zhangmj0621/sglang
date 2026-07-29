"""
Resource-agnostic rid/ref tracking core shared by ref-aware caches.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, InsertResult
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    convert_to_bigram_key,
    page_align_keys,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import TreeNode


logger = logging.getLogger(__name__)


@dataclass
class RefInfo:
    is_high: bool
    priority: int = 0
    nodes: Set["TreeNode"] = field(default_factory=set)
    cached_tokens: int = 0
    is_generating: bool = False


TIER_UNUSED = 0  # high_ref == 0, low_ref == 0
TIER_LOW_REF = 1  # high_ref == 0, low_ref > 0
TIER_HIGH_REF = 2  # high_ref > 0


def _classify_node_tier(node: "TreeNode") -> int:
    if node.high_ref > 0:
        return TIER_HIGH_REF
    if node.low_ref > 0:
        return TIER_LOW_REF
    return TIER_UNUSED


class RefAwareCacheCore:
    """Resource-agnostic rid/ref tracking shared by RadixCache- and
    MambaRadixCache-based ref-aware caches. Subclasses own the per-tier
    size accounting and eviction; they react to tier moves via
    _on_node_tier_changed."""

    def _init_ref_aware_core_state(self, server_args):
        self.high_priority_threshold = getattr(
            server_args, "high_priority_threshold", 1
        )
        self._enable_priority_scheduling = getattr(
            server_args, "enable_priority_scheduling", False
        )
        self.rid_to_ref_info: Dict[str, RefInfo] = {}
        self._evict_scope_stack: list[tuple[bool, bool]] = []
        self._last_inserted_node = None

    def _reset_ref_aware_core_state(self):
        self.rid_to_ref_info.clear()
        self._evict_scope_stack.clear()
        self._last_inserted_node = None

    def _current_evict_scope(self) -> tuple[bool, bool]:
        if self._evict_scope_stack:
            return self._evict_scope_stack[-1]
        return True, False

    def _on_node_tier_changed(self, node, old_tier: int, new_tier: int):
        """Hook: node.high_ref/low_ref change moved it across tiers.
        Subclasses update their tier accounting; guards (evicted/locked)
        live in the subclass because resource semantics differ."""
        pass

    def _inc_priority_ref_single(self, node, is_high: bool):
        old_tier = _classify_node_tier(node)
        if is_high:
            node.high_ref += 1
        else:
            node.low_ref += 1
        new_tier = _classify_node_tier(node)
        if old_tier != new_tier:
            self._on_node_tier_changed(node, old_tier, new_tier)

    def _dec_priority_ref_single(self, node, is_high: bool):
        old_tier = _classify_node_tier(node)
        if is_high:
            node.high_ref = max(0, node.high_ref - 1)
        else:
            node.low_ref = max(0, node.low_ref - 1)
        new_tier = _classify_node_tier(node)
        if old_tier != new_tier:
            self._on_node_tier_changed(node, old_tier, new_tier)

    def _untrack_node_rids(self, node):
        for rid in node.tracked_rids:
            ref_info = self.rid_to_ref_info.get(rid)
            if ref_info is not None:
                ref_info.nodes.discard(node)
        node.tracked_rids.clear()

    def is_high_priority(self, priority: int) -> bool:
        if not self._enable_priority_scheduling:
            return True
        return priority >= self.high_priority_threshold

    @contextmanager
    def scoped_evict(self, allow_low: bool = True, allow_high: bool = False):
        self._evict_scope_stack.append((allow_low, allow_high))
        try:
            yield
        finally:
            self._evict_scope_stack.pop()

    def register_ref(self, req: Req):
        rid = req.rid
        priority = getattr(req, "priority", 0) or 0
        is_high = self.is_high_priority(priority)

        if rid not in self.rid_to_ref_info:
            self.rid_to_ref_info[rid] = RefInfo(is_high=is_high, priority=priority)

        ref_info = self.rid_to_ref_info[rid]
        if is_high != ref_info.is_high:
            msg = "Priority class mismatch for ref-aware KV buffer."
            logger.error(msg)
            raise ValueError(msg)
        ref_info.priority = priority

        last_node = getattr(req, "last_node", None)
        if last_node not in (None, self.root_node):
            new_nodes = self._collect_untracked_nodes_from_last_node(
                last_node, ref_info.nodes
            )
        else:
            token_ids = (req.origin_input_ids + req.output_ids)[: req.kv_committed_len]
            if not token_ids:
                return

            # Mirror cache_finished_req's key construction: EAGLE stores bigram
            # keys in the tree, so a raw-token key would never match.
            is_eagle = getattr(self, "is_eagle", False)
            keys = convert_to_bigram_key(token_ids) if is_eagle else token_ids
            keys = page_align_keys(keys, self.page_size)
            if not keys:
                return
            radix_key = RadixKey(
                keys, getattr(req, "extra_key", None), is_bigram=is_eagle
            )

            nodes_on_path = self._collect_nodes_on_path(radix_key)
            new_nodes = [node for node in nodes_on_path if node not in ref_info.nodes]

        for node in new_nodes:
            self._inc_priority_ref_single(node, is_high)
            ref_info.nodes.add(node)
            node.tracked_rids.add(rid)

        ref_info.cached_tokens = sum(len(n.key) for n in ref_info.nodes)

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
            node = child
            key = key[prefix_len:]
        return nodes

    def _collect_untracked_nodes_from_last_node(
        self, node: Optional["TreeNode"], tracked_nodes: Set["TreeNode"]
    ) -> list["TreeNode"]:
        nodes = []
        while node not in (None, self.root_node):
            if node in tracked_nodes:
                break
            nodes.append(node)
            node = node.parent
        return nodes

    def release_ref(self, rid: str) -> Tuple[bool, str]:
        if rid is None:
            return False, "rid is None"
        ref_info = self.rid_to_ref_info.pop(rid, None)
        if ref_info is None:
            return True, f"rid {rid} not tracked"

        for node in ref_info.nodes:
            self._dec_priority_ref_single(node, ref_info.is_high)
            node.tracked_rids.discard(rid)

        return True, f"released {len(ref_info.nodes)} nodes for rid {rid}"

    def update_ref(self, rid: str, new_priority: int) -> Tuple[bool, str]:
        if rid is None:
            return False, "rid is None"
        ref_info = self.rid_to_ref_info.get(rid)
        if ref_info is None:
            return False, f"rid {rid} not found in ref tracking"

        new_is_high = self.is_high_priority(new_priority)
        ref_info.priority = new_priority

        if new_is_high == ref_info.is_high:
            return True, "priority class unchanged"

        for node in ref_info.nodes:
            self._dec_priority_ref_single(node, ref_info.is_high)
            self._inc_priority_ref_single(node, new_is_high)
        ref_info.is_high = new_is_high
        return True, f"updated {len(ref_info.nodes)} nodes for rid {rid}"

    def insert(self, params: InsertParams) -> InsertResult:
        result = super().insert(params)
        self._last_inserted_node = result.last_node
        return result

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        self._last_inserted_node = None
        super().cache_finished_req(req, is_insert=is_insert)
        # Refresh req.last_node to the freshly inserted deepest node so the
        # following register_ref picks up this turn's new suffix. Done after
        # super(), which still needs the pre-insert node for dec_lock_ref.
        if self._last_inserted_node is not None:
            req.last_node = self._last_inserted_node

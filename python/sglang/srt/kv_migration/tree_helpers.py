"""Tree-walking helpers for KV migration."""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode


def collect_path_with_pages(
    tree_cache: "HiRadixCache",
    key: "RadixKey",
    page_size: int,
    force_backup: bool = False,
) -> Tuple[List[int], List["TreeNode"]]:
    """Walk root → matched leaf along `key`. Return:
      - `pages`: host pool page indices for every host-resident matched token
      - `path_nodes`: TreeNode visited (in root→leaf order, root excluded)

    Stops at the first child that is either (a) absent / divergent, or
    (b) not host-backuped. A node may be device-resident but not backuped
    when its initial `write_backup` was never triggered or failed (host
    pool OOM).

    With `force_backup=True` (used by the KV-migration source path), an
    unbacked child gets `tree_cache.write_backup(child)` invoked: this
    schedules a device→host DMA and sets `child.host_value` immediately
    (so subsequent path nodes pass their parent-must-be-backuped check).
    The caller is responsible for synchronizing the resulting CUDA
    `finish_event` before reading host bytes — see
    `KVMigrationManager._snapshot_inflight_events`. If `write_backup`
    returns 0 (host pool OOM survives the internal evict_host retry),
    the walk stops there and the caller surfaces a page shortfall.
    """
    pages: List[int] = []
    path_nodes: List["TreeNode"] = []
    node = tree_cache.root_node
    remaining = key
    if len(remaining) == 0:
        return pages, path_nodes
    child_key = tree_cache.get_child_key_fn(remaining)
    while len(remaining) > 0 and child_key in node.children:
        child = node.children[child_key]
        prefix_len = tree_cache.key_match_fn(child.key, remaining)
        if prefix_len == 0:
            break
        if not child.backuped:
            if not force_backup:
                break
            # write_backup sets child.host_value (so child.backuped becomes
            # True) and registers a finish_event in cache_controller.ack_write_queue.
            n_written = tree_cache.write_backup(child)
            if n_written == 0 or not child.backuped:
                # host pool OOM even after internal retry; cannot migrate further
                break
        host_idx = child.host_value[:prefix_len].tolist()
        if page_size == 1:
            pages.extend(int(p) for p in host_idx)
        else:
            pages.extend(int(p) // page_size for p in host_idx[::page_size])
        path_nodes.append(child)
        if prefix_len < len(child.key):
            break
        node = child
        remaining = remaining[prefix_len:]
        if len(remaining) > 0:
            child_key = tree_cache.get_child_key_fn(remaining)
    return pages, path_nodes


def collect_host_pages(
    tree_cache: "HiRadixCache",
    key: "RadixKey",
    page_size: int,
) -> List[int]:
    """Backward-compat wrapper around `collect_path_with_pages`."""
    pages, _ = collect_path_with_pages(tree_cache, key, page_size)
    return pages


def inc_host_refs_along_path(
    leaf_node: "TreeNode",
    root_node: "TreeNode",
) -> List["TreeNode"]:
    """Walk leaf → root, incrementing `host_ref_counter` on each non-root
    node. Returns the list of touched nodes (for later `dec_host_refs`).
    """
    touched: List["TreeNode"] = []
    node = leaf_node
    while node is not None and node is not root_node:
        node.host_ref_counter += 1
        touched.append(node)
        node = node.parent
    return touched


def dec_host_refs(nodes: List["TreeNode"]) -> None:
    """Decrement `host_ref_counter` on each node. Mirror of
    `inc_host_refs_along_path`."""
    for node in nodes:
        node.host_ref_counter -= 1

"""Tree-walking helpers for KV migration."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode


def collect_host_pages(
    tree_cache: "HiRadixCache",
    key: "RadixKey",
    page_size: int,
) -> List[int]:
    """Walk root → matched leaf along `key`, returning the host pool page
    indices of every matched token. Asserts each matched node is backuped
    (write_through invariant).

    Stops early if a child diverges mid-key. Returns the page indices
    collected up to that point.
    """
    pages: List[int] = []
    node = tree_cache.root_node
    remaining = key
    while len(remaining) > 0:
        first = remaining.token_ids[0]
        if first not in node.children:
            break
        child = node.children[first]
        prefix_len = tree_cache.key_match_fn(child.key, remaining)
        if prefix_len == 0:
            break
        assert child.backuped, (
            f"source node not backuped (write_through invariant violated); "
            f"node.key.token_ids[:8]={list(child.key.token_ids[:8])}"
        )
        host_idx = child.host_value[:prefix_len].tolist()
        # token-level → page-level: take every page_size-th token, divide
        if page_size == 1:
            pages.extend(int(p) for p in host_idx)
        else:
            pages.extend(int(p) // page_size for p in host_idx[::page_size])
        if prefix_len < len(child.key):
            break
        node = child
        remaining = remaining[prefix_len:]
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

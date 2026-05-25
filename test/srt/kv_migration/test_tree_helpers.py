"""Unit tests for kv_migration.tree_helpers."""

from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.kv_migration.tree_helpers import (
    collect_host_pages,
    dec_host_refs,
    inc_host_refs_along_path,
)
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode


def _make_node(token_ids, host_indices, value=None, parent=None):
    n = TreeNode()
    n.key = RadixKey(token_ids)
    n.host_value = torch.tensor(host_indices, dtype=torch.int64)
    n.value = value if value is None else torch.tensor(value, dtype=torch.int64)
    n.parent = parent
    return n


class FakeTreeCache:
    """Minimal stand-in for HiRadixCache: only what tree_helpers needs."""

    def __init__(self, root):
        self.root_node = root
        self.page_size = 1

    @staticmethod
    def key_match_fn(node_key, query_key):
        n = min(len(node_key), len(query_key))
        for i in range(n):
            if node_key.token_ids[i] != query_key.token_ids[i]:
                return i
        return n


def test_collect_host_pages_single_node_backuped():
    root = TreeNode()
    root.key = RadixKey([])
    child = _make_node([10, 11, 12, 13], host_indices=[100, 101, 102, 103], parent=root)
    root.children = {10: child}
    cache = FakeTreeCache(root)
    pages = collect_host_pages(cache, RadixKey([10, 11, 12, 13]), page_size=1)
    assert pages == [100, 101, 102, 103]


def test_collect_host_pages_partial_match_stops():
    root = TreeNode()
    root.key = RadixKey([])
    child = _make_node([10, 11, 99], host_indices=[100, 101, 102], parent=root)
    root.children = {10: child}
    cache = FakeTreeCache(root)
    pages = collect_host_pages(cache, RadixKey([10, 11, 88]), page_size=1)
    assert pages == [100, 101]


def test_collect_host_pages_not_backuped_raises():
    root = TreeNode()
    root.key = RadixKey([])
    child = TreeNode()
    child.key = RadixKey([10, 11])
    child.host_value = None  # not backuped
    child.value = torch.tensor([200, 201], dtype=torch.int64)
    child.parent = root
    root.children = {10: child}
    cache = FakeTreeCache(root)
    with pytest.raises(AssertionError, match="backuped"):
        collect_host_pages(cache, RadixKey([10, 11]), page_size=1)


def test_inc_dec_host_refs_along_path():
    root = TreeNode()
    root.key = RadixKey([])
    n1 = _make_node([10], host_indices=[100], parent=root)
    n2 = _make_node([10, 11], host_indices=[101], parent=n1)
    touched = inc_host_refs_along_path(n2, root)
    assert n1.host_ref_counter == 1
    assert n2.host_ref_counter == 1
    assert root.host_ref_counter == 0  # root excluded
    dec_host_refs(touched)
    assert n1.host_ref_counter == 0
    assert n2.host_ref_counter == 0

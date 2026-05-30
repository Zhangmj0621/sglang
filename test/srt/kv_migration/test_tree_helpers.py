"""Unit tests for kv_migration.tree_helpers."""

from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.kv_migration.tree_helpers import (
    collect_host_pages,
    collect_path_with_pages,
    dec_host_refs,
    inc_host_refs_along_path,
)
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode, get_child_key


def _make_node(token_ids, host_indices, value=None, parent=None):
    n = TreeNode()
    n.key = RadixKey(token_ids)
    n.host_value = torch.tensor(host_indices, dtype=torch.int64)
    n.value = value if value is None else torch.tensor(value, dtype=torch.int64)
    n.parent = parent
    return n


class FakeTreeCache:
    """Minimal stand-in for HiRadixCache: only what tree_helpers needs."""

    def __init__(self, root, page_size: int = 1):
        self.root_node = root
        self.page_size = page_size

    def get_child_key_fn(self, key):
        # Mirrors HiRadixCache.__init__: page_size=1 -> token_ids[0],
        # page_size>1 -> tuple(token_ids[:page_size]).
        return get_child_key(key, page_size=self.page_size)

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


def test_collect_host_pages_page_size_gt_1():
    # Test the page_size > 1 branch: token-level indices are downsampled
    # and divided by page_size to get page indices.
    # 8 token-level indices with page_size=2 means 4 pages.
    root = TreeNode()
    root.key = RadixKey([])
    child = _make_node(
        [10, 11, 12, 13, 14, 15, 16, 17],
        host_indices=[200, 201, 202, 203, 204, 205, 206, 207],
        parent=root,
    )
    # With page_size=2, the children-dict key is `tuple(token_ids[:2])`,
    # not the leading int. Match the real HiRadixCache layout.
    root.children = {(10, 11): child}
    cache = FakeTreeCache(root, page_size=2)
    pages = collect_host_pages(
        cache, RadixKey([10, 11, 12, 13, 14, 15, 16, 17]), page_size=2
    )
    # host_idx[::2] = [200, 202, 204, 206]
    # divided by page_size=2 = [100, 101, 102, 103]
    assert pages == [100, 101, 102, 103]


def test_collect_host_pages_stops_at_not_backuped_child():
    """A child can be device-resident (`value` set) but not yet host-backuped
    (`host_value is None`) -- e.g. write_backup hasn't run yet, or it ran and
    failed due to host pool OOM. The walk must stop there and return only
    the host-resident prefix; the caller then surfaces a page shortfall."""
    root = TreeNode()
    root.key = RadixKey([])
    child = TreeNode()
    child.key = RadixKey([10, 11])
    child.host_value = None  # not backuped
    child.value = torch.tensor([200, 201], dtype=torch.int64)
    child.parent = root
    root.children = {10: child}
    cache = FakeTreeCache(root)
    pages = collect_host_pages(cache, RadixKey([10, 11]), page_size=1)
    assert pages == []


def test_collect_path_with_pages_stops_at_not_backuped_grandchild():
    """First child is backuped, grandchild is device-only -- return pages for
    the first child only, and path_nodes contains just the first child."""
    root = TreeNode()
    root.key = RadixKey([])
    child = _make_node([10, 11], host_indices=[100, 101], parent=root)
    grand = TreeNode()
    grand.key = RadixKey([12, 13])
    grand.host_value = None  # not backuped
    grand.value = torch.tensor([202, 203], dtype=torch.int64)
    grand.parent = child
    child.children = {12: grand}
    root.children = {10: child}
    cache = FakeTreeCache(root)
    pages, path_nodes = collect_path_with_pages(
        cache, RadixKey([10, 11, 12, 13]), page_size=1
    )
    assert pages == [100, 101]
    assert path_nodes == [child]


def test_collect_path_with_pages_force_backup_invokes_write_backup():
    """With force_backup=True, an unbacked child is pushed to host via
    `tree_cache.write_backup(...)`; the walk continues using the freshly
    populated host_value."""
    root = TreeNode()
    root.key = RadixKey([])
    child = TreeNode()
    child.key = RadixKey([10, 11])
    child.host_value = None
    child.value = torch.tensor([200, 201], dtype=torch.int64)
    child.parent = root
    root.children = {10: child}

    cache = FakeTreeCache(root)
    write_backup_calls = []

    def fake_write_backup(node):
        write_backup_calls.append(node)
        node.host_value = torch.tensor([500, 501], dtype=torch.int64)
        return len(node.host_value)

    cache.write_backup = fake_write_backup

    pages, path_nodes = collect_path_with_pages(
        cache, RadixKey([10, 11]), page_size=1, force_backup=True
    )
    assert write_backup_calls == [child]
    assert pages == [500, 501]
    assert path_nodes == [child]


def test_collect_path_with_pages_force_backup_oom_stops_walk():
    """If write_backup returns 0 (host pool OOM survives internal retry),
    the walk stops at the unbacked node; previous backuped pages are kept."""
    root = TreeNode()
    root.key = RadixKey([])
    first = _make_node([10, 11], host_indices=[100, 101], parent=root)
    second = TreeNode()
    second.key = RadixKey([12, 13])
    second.host_value = None
    second.value = torch.tensor([300, 301], dtype=torch.int64)
    second.parent = first
    first.children = {12: second}
    root.children = {10: first}

    cache = FakeTreeCache(root)
    cache.write_backup = lambda node: 0  # always OOM

    pages, path_nodes = collect_path_with_pages(
        cache, RadixKey([10, 11, 12, 13]), page_size=1, force_backup=True
    )
    assert pages == [100, 101]
    assert path_nodes == [first]


def test_collect_path_with_pages_force_backup_skips_already_backuped():
    """force_backup=True must not re-call write_backup on already-backuped
    nodes; otherwise we'd overwrite valid host_value with fresh allocations."""
    root = TreeNode()
    root.key = RadixKey([])
    child = _make_node([10, 11], host_indices=[100, 101], parent=root)
    root.children = {10: child}

    cache = FakeTreeCache(root)
    cache.write_backup = MagicMock()

    pages, path_nodes = collect_path_with_pages(
        cache, RadixKey([10, 11]), page_size=1, force_backup=True
    )
    assert pages == [100, 101]
    cache.write_backup.assert_not_called()


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

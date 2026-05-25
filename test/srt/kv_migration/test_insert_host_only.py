"""Unit tests for HiRadixCache._insert_host_only (KV migration commit path)."""

import pytest
import torch

from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode


def _build_minimal_hiradix():
    """Build the smallest HiRadixCache stub usable for _insert_host_only tests.
    Bypasses heavy init (no real KV pool, no controller).
    """
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    cache = HiRadixCache.__new__(HiRadixCache)
    cache.root_node = TreeNode()
    cache.root_node.key = RadixKey([])
    cache.page_size = 1
    cache.evictable_size_ = 0
    cache.protected_size_ = 0
    cache.evictable_leaves = set()
    cache.evictable_host_leaves = set()

    # get_child_key_fn closes over page_size; for page_size=1 just take the first token id
    cache.get_child_key_fn = lambda key: key.token_ids[0] if len(key) > 0 else None
    return cache


def test_insert_host_only_creates_evicted_backuped_node():
    cache = _build_minimal_hiradix()
    parent = cache.root_node
    suffix_key = RadixKey([10, 11, 12, 13])
    host_value = torch.tensor([100, 101, 102, 103], dtype=torch.int64)

    new_last = cache._insert_host_only(parent, suffix_key, host_value)

    assert new_last is not None
    assert new_last.parent is parent
    assert new_last.value is None  # device empty
    assert new_last.host_value is not None
    assert new_last.host_value.tolist() == [100, 101, 102, 103]
    assert new_last.evicted is True
    assert new_last.backuped is True
    # Leaf should be in the host-evictable set (lock_ref==0, no non-evicted children)
    assert new_last in cache.evictable_host_leaves
    # parent gained an evicted child, so parent should NOT be host-evictable anymore
    # (since _update_host_leaf_status removes a node when any child is evicted)
    assert parent not in cache.evictable_host_leaves


def test_insert_host_only_under_existing_subtree():
    """parent is an interior node; _insert_host_only attaches a new child."""
    cache = _build_minimal_hiradix()
    interior = TreeNode()
    interior.key = RadixKey([1, 2])
    interior.host_value = torch.tensor([50, 51], dtype=torch.int64)
    interior.value = None  # already evicted
    interior.parent = cache.root_node
    cache.root_node.children = {1: interior}

    suffix_key = RadixKey([3, 4])
    host_value = torch.tensor([200, 201], dtype=torch.int64)
    new_last = cache._insert_host_only(interior, suffix_key, host_value)

    assert new_last.parent is interior
    assert interior.children[3] is new_last
    assert new_last.host_value.tolist() == [200, 201]
    assert new_last in cache.evictable_host_leaves


def test_insert_host_only_clones_host_value():
    """Cloning ensures the caller can free their tensor without affecting the tree."""
    cache = _build_minimal_hiradix()
    suffix_key = RadixKey([10, 11])
    host_value = torch.tensor([100, 101], dtype=torch.int64)
    new_last = cache._insert_host_only(cache.root_node, suffix_key, host_value)
    # Mutating the original should not affect the tree's copy
    host_value[0] = 999
    assert new_last.host_value[0].item() == 100


def test_insert_host_only_rejects_empty_suffix():
    cache = _build_minimal_hiradix()
    with pytest.raises(AssertionError):
        cache._insert_host_only(
            cache.root_node, RadixKey([]), torch.tensor([], dtype=torch.int64)
        )


def test_insert_host_only_rejects_length_mismatch():
    cache = _build_minimal_hiradix()
    with pytest.raises(AssertionError):
        cache._insert_host_only(
            cache.root_node,
            RadixKey([10, 11, 12]),
            torch.tensor([100, 101], dtype=torch.int64),
        )

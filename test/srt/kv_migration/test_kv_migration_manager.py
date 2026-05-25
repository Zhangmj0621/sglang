"""Unit tests for kv_migration.transfer_worker and KVMigrationManager."""

from unittest.mock import MagicMock

import pytest

from sglang.srt.kv_migration.io_types import TransferTarget
from sglang.srt.kv_migration.transfer_worker import (
    do_host_to_host_rdma,
    group_concurrent_contiguous,
)


def test_group_concurrent_contiguous_merges_runs():
    src = [10, 11, 12, 20, 21, 30]
    dst = [50, 51, 52, 60, 61, 70]
    groups = list(group_concurrent_contiguous(src, dst))
    assert groups == [
        ([10, 11, 12], [50, 51, 52]),
        ([20, 21], [60, 61]),
        ([30], [70]),
    ]


def test_group_concurrent_contiguous_empty():
    assert list(group_concurrent_contiguous([], [])) == []


def test_group_concurrent_contiguous_no_merges_when_dst_disjoint():
    src = [10, 11, 12]
    dst = [50, 60, 70]  # not contiguous in dst
    groups = list(group_concurrent_contiguous(src, dst))
    assert groups == [([10], [50]), ([11], [60]), ([12], [70])]


def test_do_host_to_host_rdma_calls_engine_with_blocks():
    engine = MagicMock()
    engine.batch_transfer_sync.return_value = 0

    target = TransferTarget(
        tp=0, pp=0, session_id="peer:1234",
        host_kv_data_ptrs=[1000, 2000],   # 2 layers (e.g., K layer 0 + V layer 0)
        host_kv_item_lens=[64, 64],
        kv_indices=[5, 6, 7],             # destination page indices
    )

    src_kv_data_ptrs = [10000, 20000]
    src_kv_item_lens = [64, 64]
    src_send_pages = [1, 2, 3]            # source page indices (contiguous run)

    ret = do_host_to_host_rdma(
        engine=engine,
        target=target,
        src_kv_data_ptrs=src_kv_data_ptrs,
        src_kv_item_lens=src_kv_item_lens,
        src_send_pages=src_send_pages,
    )
    assert ret == 0
    engine.batch_transfer_sync.assert_called_once()
    args, _ = engine.batch_transfer_sync.call_args
    session_id, src_addrs, dst_addrs, lens = args
    assert session_id == "peer:1234"
    # Two layers × one contiguous block each = 2 blocks
    assert len(src_addrs) == 2
    assert len(dst_addrs) == 2
    assert len(lens) == 2
    # Layer 0: src starts at 10000 + 1*64 = 10064; dst at 1000 + 5*64 = 1320; len = 3 pages * 64
    assert src_addrs[0] == 10000 + 1 * 64
    assert dst_addrs[0] == 1000 + 5 * 64
    assert lens[0] == 3 * 64
    # Layer 1: src 20000 + 1*64; dst 2000 + 5*64; len 3*64
    assert src_addrs[1] == 20000 + 1 * 64
    assert dst_addrs[1] == 2000 + 5 * 64
    assert lens[1] == 3 * 64


def test_do_host_to_host_rdma_empty_pages_returns_zero():
    engine = MagicMock()
    target = TransferTarget(
        tp=0, pp=0, session_id="peer:1234",
        host_kv_data_ptrs=[1000, 2000],
        host_kv_item_lens=[64, 64],
        kv_indices=[],
    )
    ret = do_host_to_host_rdma(
        engine=engine, target=target,
        src_kv_data_ptrs=[10000, 20000],
        src_kv_item_lens=[64, 64],
        src_send_pages=[],
    )
    assert ret == 0
    engine.batch_transfer_sync.assert_not_called()


def test_do_host_to_host_rdma_propagates_engine_failure():
    engine = MagicMock()
    engine.batch_transfer_sync.return_value = 7  # non-zero failure code
    target = TransferTarget(
        tp=0, pp=0, session_id="peer:1234",
        host_kv_data_ptrs=[1000],
        host_kv_item_lens=[64],
        kv_indices=[5],
    )
    ret = do_host_to_host_rdma(
        engine=engine, target=target,
        src_kv_data_ptrs=[10000],
        src_kv_item_lens=[64],
        src_send_pages=[1],
    )
    assert ret == 7


# ---------- Manager-level tests (skeleton + match_extra) ----------

import torch  # ensure torch imported at top if not already

from sglang.srt.managers.io_struct import (
    GetTransferSessionInfoReqInput,
)


def _build_manager_with_fake_tree(matched_aligned: int, page_size: int = 64,
                                  device_indices_len: int = None,
                                  host_hit_length: int = None):
    """Build a KVMigrationManager with __new__ + a fake tree_cache returning a
    canned match_prefix result."""
    from sglang.srt.kv_migration.manager import KVMigrationManager

    if device_indices_len is None:
        # Default split: half device, half host
        device_indices_len = matched_aligned // 2
    if host_hit_length is None:
        host_hit_length = matched_aligned - device_indices_len

    fake_match = MagicMock()
    # `len(match.device_indices)` should equal device_indices_len
    fake_match.device_indices = MagicMock()
    fake_match.device_indices.__len__ = lambda self: device_indices_len
    fake_match.host_hit_length = host_hit_length
    fake_match.last_device_node = MagicMock()
    fake_match.last_host_node = MagicMock()

    fake_tree = MagicMock()
    fake_tree.match_prefix.return_value = fake_match
    fake_tree.token_to_kv_pool_host = MagicMock()

    mgr = KVMigrationManager.__new__(KVMigrationManager)
    mgr.tree_cache = fake_tree
    mgr.host_pool = fake_tree.token_to_kv_pool_host
    mgr.page_size = page_size
    mgr.tp_rank = 0
    mgr.pp_rank = 0
    mgr.session_id = "fake:1234"
    mgr.host_kv_data_ptrs = [1000, 2000]
    mgr.host_kv_item_lens = [128, 128]
    return mgr


def test_get_session_info_returns_per_rank_metadata():
    mgr = _build_manager_with_fake_tree(matched_aligned=0, page_size=64)
    out = mgr.get_session_info(GetTransferSessionInfoReqInput())
    assert out.success is True
    assert out.tp_rank == 0
    assert out.pp_rank == 0
    assert out.session_id == "fake:1234"
    assert out.host_kv_data_ptrs == [1000, 2000]
    assert out.host_kv_item_lens == [128, 128]
    assert out.page_size == 64


def test_match_extra_page_aligned():
    mgr = _build_manager_with_fake_tree(matched_aligned=64, page_size=64)
    out = mgr.match_extra(input_ids=list(range(200)), extra_key=None)
    # total_aligned = 200 // 64 * 64 = 192; matched=64 → extra=128
    assert out.success is True
    assert out.total_token_size == 192
    assert out.matched_token_size == 64
    assert out.extra_token_size == 128


def test_match_extra_short_input():
    """Input shorter than page_size returns zero across the board."""
    mgr = _build_manager_with_fake_tree(matched_aligned=0, page_size=64)
    out = mgr.match_extra(input_ids=list(range(10)), extra_key=None)
    assert out.success is True
    assert out.total_token_size == 0
    assert out.matched_token_size == 0
    assert out.extra_token_size == 0


def test_match_extra_full_match():
    """All matched (extra=0)."""
    mgr = _build_manager_with_fake_tree(matched_aligned=192, page_size=64)
    out = mgr.match_extra(input_ids=list(range(200)), extra_key=None)
    assert out.total_token_size == 192
    assert out.matched_token_size == 192
    assert out.extra_token_size == 0

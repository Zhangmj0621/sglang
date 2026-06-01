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
        tp=0,
        pp=0,
        session_id="peer:1234",
        host_kv_data_ptrs=[1000, 2000],  # 2 layers (e.g., K layer 0 + V layer 0)
        host_kv_item_lens=[64, 64],
        kv_indices=[5, 6, 7],  # destination page indices
    )

    src_kv_data_ptrs = [10000, 20000]
    src_kv_item_lens = [64, 64]
    src_send_pages = [1, 2, 3]  # source page indices (contiguous run)

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
    # Two layers x one contiguous block each = 2 blocks
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
        tp=0,
        pp=0,
        session_id="peer:1234",
        host_kv_data_ptrs=[1000, 2000],
        host_kv_item_lens=[64, 64],
        kv_indices=[],
    )
    ret = do_host_to_host_rdma(
        engine=engine,
        target=target,
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
        tp=0,
        pp=0,
        session_id="peer:1234",
        host_kv_data_ptrs=[1000],
        host_kv_item_lens=[64],
        kv_indices=[5],
    )
    ret = do_host_to_host_rdma(
        engine=engine,
        target=target,
        src_kv_data_ptrs=[10000],
        src_kv_item_lens=[64],
        src_send_pages=[1],
    )
    assert ret == 7


# ---------- Manager-level tests (skeleton + match_extra) ----------

import torch  # ensure torch imported at top if not already

from sglang.srt.managers.io_struct import GetTransferSessionInfoReqInput


def _build_manager_with_fake_tree(
    matched_aligned: int,
    page_size: int = 64,
    device_indices_len: int = None,
    host_hit_length: int = None,
):
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
    # total_aligned = 200 // 64 * 64 = 192; matched=64 -> extra=128
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


# ---------- allocate / commit tests ----------

import time as _time

from sglang.srt.kv_migration.io_types import PendingMigration
from sglang.srt.managers.io_struct import (
    AllocateTokenForTransferReqInput,
    CommitTransferRequestKVCacheReqInput,
)
from sglang.srt.mem_cache.radix_cache import RadixKey


def test_allocate_returns_migration_id_and_indices():
    mgr = _build_manager_with_fake_tree(
        matched_aligned=64,
        page_size=64,
        device_indices_len=32,
        host_hit_length=32,
    )
    # Page-aligned contiguous token run: 128 tokens == 2 pages (page_size 64),
    # starting at page-aligned token index 2048 (== page 32).
    mgr.host_pool.alloc.return_value = torch.tensor(
        list(range(2048, 2176)), dtype=torch.int64
    )
    mgr.tree_cache.inc_lock_ref = MagicMock()
    mgr.pending = {}

    out = mgr.allocate(
        AllocateTokenForTransferReqInput(
            input_ids=list(range(192)),
            extra_key=None,
            extra_token_size=128,
            migration_id="mig-from-http",
        )
    )
    assert out.success is True
    assert out.migration_id == "mig-from-http"
    # kv_indices are PAGE indices (token_idx // page_size), one per page.
    assert len(out.kv_indices) == 2
    assert out.kv_indices == [32, 33]
    assert out.migration_id in mgr.pending
    # The pending entry keeps the token-level indices for commit/free.
    assert len(mgr.pending["mig-from-http"].host_tail_indices) == 128
    mgr.tree_cache.inc_lock_ref.assert_called_once()


def test_allocate_rejects_missing_migration_id():
    """migration_id must be minted by the HTTP layer. If a request reaches
    the manager without one, all ranks would mint divergent UUIDs and the
    pending dict would be keyed inconsistently -- reject loudly instead."""
    mgr = _build_manager_with_fake_tree(
        matched_aligned=64,
        page_size=64,
        device_indices_len=32,
        host_hit_length=32,
    )
    mgr.tree_cache.inc_lock_ref = MagicMock()
    mgr.pending = {}

    out = mgr.allocate(
        AllocateTokenForTransferReqInput(
            input_ids=list(range(192)),
            extra_key=None,
            extra_token_size=128,
            # migration_id intentionally omitted
        )
    )
    assert out.success is False
    assert "migration_id" in out.message
    # Must short-circuit before any allocation or lock acquisition.
    assert mgr.pending == {}
    mgr.tree_cache.inc_lock_ref.assert_not_called()
    mgr.host_pool.alloc.assert_not_called()


def test_allocate_uses_explicit_migration_id_when_provided_v2():
    """HTTP layer mints a shared id across ranks; manager should reuse it."""
    mgr = _build_manager_with_fake_tree(
        matched_aligned=64,
        page_size=64,
        device_indices_len=32,
        host_hit_length=32,
    )
    mgr.host_pool.alloc.return_value = torch.tensor(
        list(range(64)), dtype=torch.int64
    )
    mgr.tree_cache.inc_lock_ref = MagicMock()
    mgr.pending = {}

    explicit_id = "shared-mig-id-from-http"
    out = mgr.allocate(
        AllocateTokenForTransferReqInput(
            input_ids=list(range(128)),  # total_aligned = 128
            extra_key=None,
            extra_token_size=64,  # matched=64, so 64+64=128
            migration_id=explicit_id,
        )
    )
    assert out.success is True
    assert out.migration_id == explicit_id
    assert explicit_id in mgr.pending


def test_allocate_oom_returns_failure_and_rolls_back_locks():
    mgr = _build_manager_with_fake_tree(
        matched_aligned=64,
        page_size=64,
        device_indices_len=32,
        host_hit_length=32,
    )
    mgr.host_pool.alloc.return_value = None  # OOM
    mgr.tree_cache.inc_lock_ref = MagicMock()
    mgr.tree_cache.dec_lock_ref = MagicMock()
    mgr.pending = {}

    out = mgr.allocate(
        AllocateTokenForTransferReqInput(
            input_ids=list(range(128)),
            extra_key=None,
            extra_token_size=64,
            migration_id="mig-oom",
        )
    )
    assert out.success is False
    assert "alloc" in out.message.lower() or "oom" in out.message.lower()
    assert mgr.pending == {}
    mgr.tree_cache.dec_lock_ref.assert_called_once()


def test_allocate_size_mismatch_returns_failure():
    """matched_aligned + extra_token_size must equal total_aligned."""
    mgr = _build_manager_with_fake_tree(
        matched_aligned=64,
        page_size=64,
        device_indices_len=32,
        host_hit_length=32,
    )
    mgr.tree_cache.inc_lock_ref = MagicMock()
    mgr.tree_cache.dec_lock_ref = MagicMock()
    mgr.pending = {}

    # total_aligned for len=128 is 128; matched=64; client claims extra=999
    out = mgr.allocate(
        AllocateTokenForTransferReqInput(
            input_ids=list(range(128)),
            extra_key=None,
            extra_token_size=999,  # doesn't sum to total_aligned
            migration_id="mig-mismatch",
        )
    )
    assert out.success is False
    assert "matched_aligned" in out.message or "extra_token_size" in out.message


def test_commit_inserts_and_releases():
    mgr = _build_manager_with_fake_tree(
        matched_aligned=64,
        page_size=64,
        device_indices_len=32,
        host_hit_length=32,
    )
    new_last = MagicMock()
    mgr.tree_cache._insert_host_only = MagicMock(return_value=new_last)
    mgr.tree_cache.dec_lock_ref = MagicMock()

    mig_id = "test-mig"
    locked_nodes = [MagicMock(host_ref_counter=1) for _ in range(2)]
    mgr.pending[mig_id] = PendingMigration(
        input_ids=list(range(192)),
        extra_key=None,
        full_key=RadixKey(list(range(128))),  # 128 tokens, matched_aligned=64, suffix=64
        matched_aligned=64,
        matched_node=MagicMock(),
        host_locked_nodes=locked_nodes,
        host_tail_indices=torch.tensor(list(range(2000, 2064)), dtype=torch.int64),
        created_at=_time.monotonic(),
    )

    out = mgr.commit(CommitTransferRequestKVCacheReqInput(migration_id=mig_id))
    assert out.success is True
    assert mig_id not in mgr.pending
    mgr.tree_cache._insert_host_only.assert_called_once()
    mgr.tree_cache.dec_lock_ref.assert_called_once()
    # host_ref_counter should have been decremented on each locked node
    for n in locked_nodes:
        assert n.host_ref_counter == 0


def test_commit_unknown_migration_id_fails():
    mgr = _build_manager_with_fake_tree(
        matched_aligned=64,
        page_size=64,
    )
    mgr.pending = {}
    out = mgr.commit(CommitTransferRequestKVCacheReqInput(migration_id="ghost"))
    assert out.success is False
    assert "ghost" in out.message or "not found" in out.message


def test_commit_race_ahead_frees_tail_and_returns_success():
    """Another path inserted the same prefix during the migration window:
    attached_offset > matched_aligned -> free our tail, return success."""
    mgr = _build_manager_with_fake_tree(
        matched_aligned=128,
        page_size=64,  # AT COMMIT: deeper match
        device_indices_len=64,
        host_hit_length=64,
    )
    mgr.tree_cache.dec_lock_ref = MagicMock()
    mgr.tree_cache._insert_host_only = MagicMock()

    mig_id = "raced"
    tail = torch.tensor(list(range(2000, 2064)), dtype=torch.int64)
    mgr.pending[mig_id] = PendingMigration(
        input_ids=list(range(128)),
        extra_key=None,
        full_key=RadixKey(list(range(128))),
        matched_aligned=64,  # AT ALLOCATE: was 64
        matched_node=MagicMock(),
        host_locked_nodes=[MagicMock(host_ref_counter=1)],
        host_tail_indices=tail,
        created_at=_time.monotonic(),
    )

    out = mgr.commit(CommitTransferRequestKVCacheReqInput(migration_id=mig_id))
    assert out.success is True
    assert "raced" in out.message.lower() or "ahead" in out.message.lower()
    mgr.host_pool.free.assert_called_once()
    # _insert_host_only must NOT have been called in this path
    mgr.tree_cache._insert_host_only.assert_not_called()


# ---------- write_through host-readiness race: wait_events plumbing ----------

from types import SimpleNamespace


def test_worker_synchronizes_wait_events_before_rdma():
    """do_host_to_host_rdma must call .synchronize() on every wait_event
    BEFORE submitting batch_transfer_sync -- otherwise we'd RDMA pre-DMA bytes."""
    call_order = []

    ev1 = MagicMock()
    ev1.synchronize.side_effect = lambda: call_order.append("sync1")
    ev2 = MagicMock()
    ev2.synchronize.side_effect = lambda: call_order.append("sync2")

    engine = MagicMock()
    engine.batch_transfer_sync.side_effect = (
        lambda *a, **k: call_order.append("rdma") or 0
    )

    target = TransferTarget(
        tp=0,
        pp=0,
        session_id="peer:1234",
        host_kv_data_ptrs=[1000],
        host_kv_item_lens=[64],
        kv_indices=[5],
    )
    ret = do_host_to_host_rdma(
        engine=engine,
        target=target,
        src_kv_data_ptrs=[10000],
        src_kv_item_lens=[64],
        src_send_pages=[1],
        wait_events=[ev1, ev2],
    )
    assert ret == 0
    # Both events must be synchronized; RDMA must come last.
    assert call_order == ["sync1", "sync2", "rdma"]


def test_worker_no_wait_events_default_unchanged():
    """Default wait_events=() preserves existing call shape."""
    engine = MagicMock()
    engine.batch_transfer_sync.return_value = 0
    target = TransferTarget(
        tp=0,
        pp=0,
        session_id="peer:1234",
        host_kv_data_ptrs=[1000],
        host_kv_item_lens=[64],
        kv_indices=[5],
    )
    ret = do_host_to_host_rdma(
        engine=engine,
        target=target,
        src_kv_data_ptrs=[10000],
        src_kv_item_lens=[64],
        src_send_pages=[1],
    )
    assert ret == 0


def _build_manager_with_writethrough_state(ongoing_ids, ack_queue):
    """Manager stub exposing only what _snapshot_inflight_events needs."""
    from sglang.srt.kv_migration.manager import KVMigrationManager

    mgr = KVMigrationManager.__new__(KVMigrationManager)
    fake_tree = MagicMock()
    fake_tree.ongoing_write_through = {nid: MagicMock() for nid in ongoing_ids}
    fake_tree.cache_controller = MagicMock()
    fake_tree.cache_controller.ack_write_queue = ack_queue
    mgr.tree_cache = fake_tree
    return mgr


def _ack(node_ids, finish_event=None):
    """Build a HiCacheAck-shaped namespace (start_event, finish_event, node_ids)."""
    if finish_event is None:
        finish_event = MagicMock()
    return SimpleNamespace(
        start_event=MagicMock(),
        finish_event=finish_event,
        node_ids=list(node_ids),
    )


def test_snapshot_inflight_events_returns_events_for_send_window():
    ev_a = MagicMock()
    ev_b = MagicMock()
    mgr = _build_manager_with_writethrough_state(
        ongoing_ids={1, 2, 3, 4},
        ack_queue=[
            _ack(node_ids=[1, 2], finish_event=ev_a),
            _ack(node_ids=[3], finish_event=ev_b),
            _ack(node_ids=[4], finish_event=MagicMock()),  # not in send window
        ],
    )
    # send window covers nodes 1, 2, 3 -- node 4 is on a different path
    wait_events, missing = mgr._snapshot_inflight_events({1, 2, 3})
    assert missing == set()
    assert ev_a in wait_events
    assert ev_b in wait_events
    assert len(wait_events) == 2  # ev_a appears once even though it covers 1+2


def test_snapshot_inflight_events_skips_already_drained_nodes():
    """Nodes whose write_through already completed (not in ongoing_write_through)
    must not contribute to wait_events even if their old ack entry lingers."""
    mgr = _build_manager_with_writethrough_state(
        ongoing_ids=set(),  # everything drained
        ack_queue=[_ack(node_ids=[1, 2, 3])],  # stale entries
    )
    wait_events, missing = mgr._snapshot_inflight_events({1, 2, 3})
    assert wait_events == []
    assert missing == set()


def test_snapshot_inflight_events_reports_missing_when_ack_entry_absent():
    """Invariant violation: node is in ongoing_write_through but its
    finish_event is nowhere in ack_write_queue."""
    mgr = _build_manager_with_writethrough_state(
        ongoing_ids={1, 2},
        ack_queue=[_ack(node_ids=[1])],  # node 2 has no ack entry
    )
    wait_events, missing = mgr._snapshot_inflight_events({1, 2})
    assert missing == {2}
    assert len(wait_events) == 1


def test_snapshot_inflight_events_empty_send_window_short_circuits():
    mgr = _build_manager_with_writethrough_state(
        ongoing_ids={1, 2, 3},
        ack_queue=[_ack(node_ids=[1, 2, 3])],
    )
    wait_events, missing = mgr._snapshot_inflight_events(set())
    assert wait_events == []
    assert missing == set()

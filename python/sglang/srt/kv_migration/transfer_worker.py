"""RDMA execution helpers for KV migration.

Designed to be called from a ThreadPoolExecutor; performs no scheduler-state
mutations. The functions here are pure data plumbing -- given source and target
host pool layouts plus a list of source/destination page indices, build
(src_addr, dst_addr, length) blocks per layer and submit a single
Mooncake batch_transfer_sync.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        MooncakeTransferEngine,
    )
    from sglang.srt.kv_migration.io_types import TransferTarget


def group_concurrent_contiguous(
    src: Sequence[int], dst: Sequence[int]
) -> Iterator[Tuple[List[int], List[int]]]:
    """Yield runs (src_run, dst_run) where both `src` and `dst` increment by 1
    simultaneously. Used to merge contiguous page indices into single RDMA blocks.

    Note: a separate implementation from `disaggregation/common/utils.py` lives
    here because the kv_migration package is intentionally independent of the
    disaggregation pipeline.
    """
    if len(src) == 0:
        return
    assert len(src) == len(dst), "src/dst length mismatch"
    cur_s = [src[0]]
    cur_d = [dst[0]]
    for i in range(1, len(src)):
        if src[i] == cur_s[-1] + 1 and dst[i] == cur_d[-1] + 1:
            cur_s.append(src[i])
            cur_d.append(dst[i])
        else:
            yield cur_s, cur_d
            cur_s = [src[i]]
            cur_d = [dst[i]]
    yield cur_s, cur_d


def do_host_to_host_rdma(
    engine: "MooncakeTransferEngine",
    target: "TransferTarget",
    src_kv_data_ptrs: Sequence[int],
    src_kv_item_lens: Sequence[int],
    src_send_pages: Sequence[int],
    wait_events: Sequence = (),
) -> int:
    """Build (src_addr, dst_addr, length) blocks for every layer and submit one
    `batch_transfer_sync`. Returns 0 on success, non-zero on engine failure.

    Layer count is `len(src_kv_data_ptrs)` (= 2 * num_attn_layers, K layers
    followed by V layers -- same scheme as MHATokenToKVPool.kv_data_ptrs).

    Caller must ensure `len(src_send_pages) == len(target.kv_indices)`.

    `wait_events` is a list of CUDA events guarding any in-flight device->host
    write_through DMA on the source side; each is synchronized before the
    RDMA submission to guarantee host pages contain final data, not pre-DMA
    bytes. Synchronizing already-completed events is a cheap no-op.
    """
    assert len(src_send_pages) == len(target.kv_indices), (
        f"src/dst page count mismatch: "
        f"src={len(src_send_pages)}, dst={len(target.kv_indices)}"
    )
    assert len(src_kv_data_ptrs) == len(
        target.host_kv_data_ptrs
    ), "layer count mismatch between src and target"
    assert list(src_kv_item_lens) == list(
        target.host_kv_item_lens
    ), "per-layer page byte size must match between src and target host pools"

    for ev in wait_events:
        ev.synchronize()

    if len(src_send_pages) == 0:
        return 0

    src_addrs: List[int] = []
    dst_addrs: List[int] = []
    lens: List[int] = []

    for layer_i in range(len(src_kv_data_ptrs)):
        src_base = src_kv_data_ptrs[layer_i]
        dst_base = target.host_kv_data_ptrs[layer_i]
        item_len = src_kv_item_lens[layer_i]
        for src_run, dst_run in group_concurrent_contiguous(
            src_send_pages, target.kv_indices
        ):
            src_addrs.append(src_base + src_run[0] * item_len)
            dst_addrs.append(dst_base + dst_run[0] * item_len)
            lens.append(item_len * len(src_run))

    return engine.batch_transfer_sync(target.session_id, src_addrs, dst_addrs, lens)

"""KVMigrationManager: per-scheduler-rank coordinator for HTTP-driven
KV cache migration.

Each scheduler rank owns one instance. It owns:
- the rank's MooncakeTransferEngine session
- a 1-worker ThreadPoolExecutor for batch_transfer_sync
- a pending dict for in-flight target-side migrations (allocated tail
  awaiting commit)

This module is intentionally independent of disaggregation/ -- KV migration
is a peer-to-peer, host->host RDMA over a shared HiRadixCache, distinct from
the prefill/decode pipeline state machine.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple

from sglang.srt.kv_migration.io_types import PendingMigration
from sglang.srt.managers.io_struct import (
    AllocateTokenForTransferReqInput,
    AllocateTokenForTransferReqOutput,
    CommitTransferRequestKVCacheReqInput,
    CommitTransferRequestKVCacheReqOutput,
    GetRequestExtraTokenSizeReqInput,
    GetRequestExtraTokenSizeReqOutput,
    GetTransferSessionInfoReqInput,
    GetTransferSessionInfoReqOutput,
    TransferRequestKVCacheReqInput,
    TransferRequestKVCacheReqOutput,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        MooncakeTransferEngine,
    )
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

logger = logging.getLogger(__name__)


class KVMigrationManager:
    """One instance per scheduler rank."""

    def __init__(self, scheduler: "Scheduler"):
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            init_mooncake_transfer_engine,
        )

        self.scheduler = scheduler
        self.tree_cache: "HiRadixCache" = scheduler.tree_cache
        self.host_pool = self.tree_cache.token_to_kv_pool_host
        assert (
            self.host_pool is not None
        ), "KV migration requires HiRadixCache with host pool"

        # Engine + memory registration (one-time at startup).
        # Pass the configured IB device(s) through; without it Mooncake falls
        # back to auto-discovery and may select unrelated NICs (e.g. a storage
        # RoCE bond like mlx5_bond_0), causing cross-NIC QP handshake timeouts.
        # `--mooncake-ib-device` accepts a single/CSV device list or a per-GPU
        # JSON map (see get_ib_devices_for_gpu).
        self.engine: "MooncakeTransferEngine" = init_mooncake_transfer_engine(
            hostname=scheduler.server_args.host,
            gpu_id=scheduler.gpu_id,
            ib_device=scheduler.server_args.mooncake_ib_device,
        )
        h_ptrs, h_lens, h_item_lens = self.host_pool.get_contiguous_buf_infos()
        self.host_kv_data_ptrs: List[int] = list(h_ptrs)
        self.host_kv_item_lens: List[int] = list(h_item_lens)
        self.engine.batch_register(self.host_kv_data_ptrs, list(h_lens))
        self.session_id: str = self.engine.get_session_id()
        self.page_size: int = self.host_pool.page_size

        self.tp_rank: int = scheduler.tp_rank
        self.pp_rank: int = scheduler.pp_rank

        # 1 worker is sufficient: FanOutCommunicator serializes /transfer
        # at the tokenizer_manager layer.
        self.transfer_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kv-migration"
        )

        self.pending: Dict[str, PendingMigration] = {}
        self.watchdog_timeout_s: float = (
            scheduler.server_args.kv_migration_watchdog_timeout
        )

        logger.info(
            "KVMigrationManager initialized: tp=%d pp=%d session=%s page_size=%d",
            self.tp_rank,
            self.pp_rank,
            self.session_id,
            self.page_size,
        )

    # -- dispatcher entries (sync, called from scheduler main loop) --

    def get_session_info(
        self, recv_req: GetTransferSessionInfoReqInput
    ) -> GetTransferSessionInfoReqOutput:
        return GetTransferSessionInfoReqOutput(
            success=True,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            session_id=self.session_id,
            host_kv_data_ptrs=list(self.host_kv_data_ptrs),
            host_kv_item_lens=list(self.host_kv_item_lens),
            page_size=self.page_size,
        )

    def match_extra(
        self,
        input_ids: List[int],
        extra_key: Optional[str],
    ) -> GetRequestExtraTokenSizeReqOutput:
        page = self.page_size
        total_aligned = (len(input_ids) // page) * page
        if total_aligned == 0:
            return GetRequestExtraTokenSizeReqOutput(
                success=True,
                extra_token_size=0,
                matched_token_size=0,
                total_token_size=0,
            )
        key = RadixKey(input_ids[:total_aligned], extra_key).page_aligned(page)
        match = self.tree_cache.match_prefix(MatchPrefixParams(key=key))
        matched_aligned = len(match.device_indices) + match.host_hit_length
        extra = total_aligned - matched_aligned
        return GetRequestExtraTokenSizeReqOutput(
            success=True,
            extra_token_size=extra,
            matched_token_size=matched_aligned,
            total_token_size=total_aligned,
        )

    def get_extra_token_size_wrapped(
        self, recv_req: GetRequestExtraTokenSizeReqInput
    ) -> GetRequestExtraTokenSizeReqOutput:
        return self.match_extra(recv_req.input_ids, recv_req.extra_key)

    # -- allocate (target side) --

    def allocate(
        self, recv_req: AllocateTokenForTransferReqInput
    ) -> AllocateTokenForTransferReqOutput:
        from sglang.srt.kv_migration.tree_helpers import (
            dec_host_refs,
            inc_host_refs_along_path,
        )

        # migration_id MUST be minted at the HTTP layer (uuid.uuid4().hex) and
        # broadcast via FanOutCommunicator so every rank uses the same id --
        # otherwise per-rank `pending` dicts would key on different ids and
        # `commit` would race-fail on some ranks while succeeding on others.
        if not recv_req.migration_id:
            return AllocateTokenForTransferReqOutput(
                success=False,
                tp_rank=self.tp_rank,
                pp_rank=self.pp_rank,
                message=(
                    "AllocateTokenForTransferReqInput.migration_id is required; "
                    "the HTTP layer must mint it once and pass the same value "
                    "to every rank"
                ),
            )

        page = self.page_size
        total_aligned = (len(recv_req.input_ids) // page) * page

        full_key = RadixKey(
            recv_req.input_ids[:total_aligned], recv_req.extra_key
        ).page_aligned(page)
        match = self.tree_cache.match_prefix(MatchPrefixParams(key=full_key))
        matched_aligned = len(match.device_indices) + match.host_hit_length

        if matched_aligned + recv_req.extra_token_size != total_aligned:
            return AllocateTokenForTransferReqOutput(
                success=False,
                tp_rank=self.tp_rank,
                pp_rank=self.pp_rank,
                message=(
                    f"matched_aligned ({matched_aligned}) + extra_token_size "
                    f"({recv_req.extra_token_size}) != total_aligned ({total_aligned})"
                ),
            )

        # Lock matched device portion
        self.tree_cache.inc_lock_ref(match.last_device_node)
        # Lock matched host portion (returns the touched nodes for later release)
        host_locked = inc_host_refs_along_path(
            match.last_host_node, self.tree_cache.root_node
        )

        host_tail = self.host_pool.alloc(recv_req.extra_token_size)
        if host_tail is None:
            self.tree_cache.dec_lock_ref(match.last_device_node)
            dec_host_refs(host_locked)
            return AllocateTokenForTransferReqOutput(
                success=False,
                tp_rank=self.tp_rank,
                pp_rank=self.pp_rank,
                message=f"host pool alloc OOM: requested {recv_req.extra_token_size} tokens",
            )

        migration_id = recv_req.migration_id
        self.pending[migration_id] = PendingMigration(
            input_ids=list(recv_req.input_ids),
            extra_key=recv_req.extra_key,
            full_key=full_key,
            matched_aligned=matched_aligned,
            matched_node=match.last_device_node,
            host_locked_nodes=host_locked,
            host_tail_indices=host_tail,
            created_at=time.monotonic(),
        )

        # `kv_indices` returned to the controller are PAGE indices, not token
        # indices: the RDMA in do_host_to_host_rdma addresses whole pages
        # (item_len == one page's bytes), and the source side likewise emits
        # page indices via collect_path_with_pages. The host pool allocates
        # page-aligned contiguous token runs, so page_idx = token_idx // page.
        # (The token-level `host_tail` stays in `pending` for commit/free, which
        # operate on the per-token radix host_value.)
        host_tail_tokens = host_tail.cpu().tolist()
        if page == 1:
            page_indices = [int(t) for t in host_tail_tokens]
        else:
            page_indices = [int(t) // page for t in host_tail_tokens[::page]]

        return AllocateTokenForTransferReqOutput(
            success=True,
            migration_id=migration_id,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            kv_indices=page_indices,
        )

    # -- commit (target side) --

    def commit(
        self, recv_req: CommitTransferRequestKVCacheReqInput
    ) -> CommitTransferRequestKVCacheReqOutput:
        from sglang.srt.kv_migration.tree_helpers import dec_host_refs

        p = self.pending.pop(recv_req.migration_id, None)
        if p is None:
            return CommitTransferRequestKVCacheReqOutput(
                success=False,
                message=f"migration_id {recv_req.migration_id} not found (timed out?)",
            )

        # Re-match to find the current attach point (tolerates concurrent tree changes).
        match = self.tree_cache.match_prefix(MatchPrefixParams(key=p.full_key))
        attached_offset = len(match.device_indices) + match.host_hit_length

        if attached_offset > p.matched_aligned:
            # Tree raced ahead: another path already inserted (some of) this prefix.
            self.host_pool.free(p.host_tail_indices)
            self.tree_cache.dec_lock_ref(p.matched_node)
            dec_host_refs(p.host_locked_nodes)
            return CommitTransferRequestKVCacheReqOutput(
                success=True,
                matched_after_commit=attached_offset,
                message="tree raced ahead during migration window; tail freed",
            )
        if attached_offset < p.matched_aligned:
            # Matched portion shrank despite our locks -- should not happen.
            self.host_pool.free(p.host_tail_indices)
            self.tree_cache.dec_lock_ref(p.matched_node)
            dec_host_refs(p.host_locked_nodes)
            return CommitTransferRequestKVCacheReqOutput(
                success=False,
                message=(
                    f"matched portion shrank from {p.matched_aligned} "
                    f"to {attached_offset} during migration window"
                ),
            )

        suffix_key = p.full_key[attached_offset:]
        try:
            self.tree_cache._insert_host_only(
                match.last_host_node, suffix_key, p.host_tail_indices
            )
        except Exception as e:
            # Cannot safely free host_tail here: _insert_host_only attaches
            # `new_node` to the tree (parent.children[k] = new_node) BEFORE
            # the leaf-status updates that may raise. If we re-parked the
            # pending and the watchdog freed `host_tail_indices`, those
            # indices would simultaneously live in the tree (via
            # new_node.host_value, a clone with identical values) and on
            # `host_pool.free_slots` -- a future alloc would hand out the
            # same slots, aliasing the tree's host bytes.
            #
            # The conservative choice is to release locks (the migration is
            # over) and intentionally leak the host tail. This is a real
            # leak only if `_insert_host_only` actually raises, which it
            # does not under normal conditions (the leaf-status updates are
            # pure dict/set manipulations); we log loudly so an incident
            # would be visible.
            self.tree_cache.dec_lock_ref(p.matched_node)
            dec_host_refs(p.host_locked_nodes)
            logger.error(
                "kv-migration: _insert_host_only raised for %s; leaking "
                "%d host tokens because the tree may already reference "
                "them. exception=%r",
                recv_req.migration_id,
                len(p.host_tail_indices),
                e,
            )
            return CommitTransferRequestKVCacheReqOutput(
                success=False,
                message=f"_insert_host_only failed: {e!r}",
            )

        self.tree_cache.dec_lock_ref(p.matched_node)
        dec_host_refs(p.host_locked_nodes)

        return CommitTransferRequestKVCacheReqOutput(
            success=True,
            matched_after_commit=attached_offset + len(suffix_key),
        )

    # -- transfer (source side) --

    def transfer(
        self, recv_req: TransferRequestKVCacheReqInput
    ) -> TransferRequestKVCacheReqOutput:
        """Run the RDMA transfer for this rank's shard and return its result.

        This is synchronous: the RDMA runs on `transfer_executor` (so the CUDA
        finish_event waits happen off the GIL-heavy path) but we block on the
        result before returning. The scheduler must call this on every rank and
        gather the per-rank outputs to the tokenizer head -- a TP/PP-sharded KV
        cache is only fully migrated once every rank's shard has transferred.

        Synchronous failures (no target for this rank, source missing pages,
        missing finish_event coverage) return a failure response immediately.

        Before submission we also snapshot the CUDA `finish_event` of any
        device->host write_through DMA still in flight on the source side that
        covers our send window. The worker thread `synchronize()`s these
        before issuing the RDMA, so host pages are guaranteed to contain
        post-DMA bytes (not pre-DMA garbage).
        """
        from sglang.srt.kv_migration.io_types import TransferTarget
        from sglang.srt.kv_migration.transfer_worker import do_host_to_host_rdma
        from sglang.srt.kv_migration.tree_helpers import (
            collect_path_with_pages,
            dec_host_refs,
            inc_host_refs_along_path,
        )

        # Find the per-rank target entry
        my_target = None
        for t in recv_req.target_per_rank:
            if t.tp == self.tp_rank and t.pp == self.pp_rank:
                my_target = TransferTarget(
                    tp=t.tp,
                    pp=t.pp,
                    session_id=t.session_id,
                    host_kv_data_ptrs=list(t.host_kv_data_ptrs),
                    host_kv_item_lens=list(t.host_kv_item_lens),
                    kv_indices=list(t.kv_indices),
                )
                break
        if my_target is None:
            return TransferRequestKVCacheReqOutput(
                success=False,
                message=f"no target entry for rank tp={self.tp_rank} pp={self.pp_rank}",
            )

        page = self.page_size
        total_aligned = recv_req.matched_token_size + recv_req.extra_token_size
        full_key = RadixKey(
            recv_req.input_ids[:total_aligned], recv_req.extra_key
        ).page_aligned(page)

        # Drain anything already-done so the in-flight set is minimal.
        self.tree_cache.flush_write_through_acks()

        # `force_backup=True` actively pushes any device-only matched node to
        # host (write_backup schedules an async DMA, finish_event registered
        # in ack_write_queue). The wait_events snapshot below picks those up
        # and the worker thread synchronizes before issuing the RDMA.
        src_host_pages, path_nodes = collect_path_with_pages(
            self.tree_cache, full_key, page, force_backup=True
        )

        required_pages = total_aligned // page
        if len(src_host_pages) < required_pages:
            return TransferRequestKVCacheReqOutput(
                success=False,
                message=(
                    f"source missing pages: have {len(src_host_pages)}, "
                    f"need {required_pages} (possibly evicted)"
                ),
            )

        send_start_page = recv_req.matched_token_size // page
        send_end_page = total_aligned // page
        src_send_pages = src_host_pages[send_start_page:send_end_page]

        # Lock source side for the duration of the RDMA write
        match = self.tree_cache.match_prefix(MatchPrefixParams(key=full_key))
        self.tree_cache.inc_lock_ref(match.last_device_node)
        host_locked = inc_host_refs_along_path(
            match.last_host_node, self.tree_cache.root_node
        )

        # Identify path nodes whose token range overlaps the send window.
        send_start_token = recv_req.matched_token_size
        send_end_token = total_aligned
        send_window_node_ids: set = set()
        cumulative = 0
        for n in path_nodes:
            n_tokens = len(n.host_value)
            n_start = cumulative
            n_end = cumulative + n_tokens
            if n_end > send_start_token and n_start < send_end_token:
                send_window_node_ids.add(n.id)
            cumulative = n_end

        wait_events, missing = self._snapshot_inflight_events(send_window_node_ids)
        if missing:
            self.tree_cache.dec_lock_ref(match.last_device_node)
            dec_host_refs(host_locked)
            return TransferRequestKVCacheReqOutput(
                success=False,
                message=(
                    f"finish_event missing for {len(missing)} in-flight "
                    f"node(s); node_ids={sorted(missing)[:5]}"
                ),
            )

        future = self.transfer_executor.submit(
            do_host_to_host_rdma,
            self.engine,
            my_target,
            self.host_kv_data_ptrs,
            self.host_kv_item_lens,
            src_send_pages,
            wait_events,
        )
        try:
            ret = future.result()
            output = TransferRequestKVCacheReqOutput(
                success=(ret == 0),
                message=("" if ret == 0 else f"batch_transfer_sync ret={ret}"),
            )
        except Exception as e:
            output = TransferRequestKVCacheReqOutput(
                success=False, message=f"transfer raised: {e!r}"
            )
        finally:
            # Release the source-side locks held for the duration of the write.
            self.tree_cache.dec_lock_ref(match.last_device_node)
            dec_host_refs(host_locked)
        return output

    def _snapshot_inflight_events(
        self, send_window_node_ids: set
    ) -> Tuple[list, set]:
        """Return `(wait_events, missing_ids)` for nodes in `send_window_node_ids`
        that are still in `tree_cache.ongoing_write_through`.

        `wait_events` is a list of CUDA finish_events (deduplicated, in
        ack_write_queue order). `missing_ids` is the subset of in-flight
        send-window node ids whose finish_event was not found in
        `ack_write_queue` -- should be empty in steady state; non-empty
        signals a controller invariant violation.
        """
        inflight = self.tree_cache.ongoing_write_through
        send_inflight: set = send_window_node_ids & set(inflight.keys())
        if not send_inflight:
            return [], set()

        wait_events: list = []
        covered: set = set()
        seen_event_ids: set = set()
        for ack in self.tree_cache.cache_controller.ack_write_queue:
            ack_node_ids = set(ack.node_ids)
            overlap = send_inflight & ack_node_ids
            if not overlap:
                continue
            ev = ack.finish_event
            ev_id = id(ev)
            if ev_id not in seen_event_ids:
                wait_events.append(ev)
                seen_event_ids.add(ev_id)
            covered |= overlap

        missing = send_inflight - covered
        return wait_events, missing

    # -- main-loop tick helpers --

    def watchdog_tick(self) -> None:
        """Roll back any pending allocate that has been waiting past
        `watchdog_timeout_s` for a commit."""
        from sglang.srt.kv_migration.tree_helpers import dec_host_refs

        now = time.monotonic()
        timed_out = [
            mig_id
            for mig_id, p in self.pending.items()
            if now - p.created_at > self.watchdog_timeout_s
        ]
        for mig_id in timed_out:
            p = self.pending.pop(mig_id)
            try:
                self.host_pool.free(p.host_tail_indices)
            except Exception as e:
                logger.warning(
                    "kv-migration watchdog: host_pool.free failed for %s: %r",
                    mig_id,
                    e,
                )
            try:
                self.tree_cache.dec_lock_ref(p.matched_node)
            except Exception as e:
                logger.warning(
                    "kv-migration watchdog: dec_lock_ref failed for %s: %r",
                    mig_id,
                    e,
                )
            dec_host_refs(p.host_locked_nodes)
            logger.warning(
                "kv-migration watchdog rolled back %s after %.1fs",
                mig_id,
                now - p.created_at,
            )

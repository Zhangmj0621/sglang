"""KVMigrationManager: per-scheduler-rank coordinator for HTTP-driven
KV cache migration.

Each scheduler rank owns one instance. It owns:
- the rank's MooncakeTransferEngine session
- a 1-worker ThreadPoolExecutor for batch_transfer_sync
- a pending dict for in-flight target-side migrations (allocated tail
  awaiting commit)

This module is intentionally independent of disaggregation/ — KV migration
is a peer-to-peer, host→host RDMA over a shared HiRadixCache, distinct from
the prefill/decode pipeline state machine.
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

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
        assert self.host_pool is not None, (
            "KV migration requires HiRadixCache with host pool"
        )

        # Engine + memory registration (one-time at startup)
        self.engine: "MooncakeTransferEngine" = init_mooncake_transfer_engine(
            hostname=scheduler.server_args.host,
            gpu_id=scheduler.gpu_id,
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
        # at the tokenizer_manager layer (see spec §3.6).
        self.transfer_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kv-migration"
        )

        self.pending: Dict[str, PendingMigration] = {}
        # (future, meta-dict). Polled per scheduler tick (Task 11).
        self.pending_futures: List[Tuple[Future, dict]] = []
        self.watchdog_timeout_s: float = (
            scheduler.server_args.kv_migration_watchdog_timeout
        )

        logger.info(
            "KVMigrationManager initialized: tp=%d pp=%d session=%s page_size=%d",
            self.tp_rank, self.pp_rank, self.session_id, self.page_size,
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

        migration_id = recv_req.migration_id or uuid.uuid4().hex
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

        return AllocateTokenForTransferReqOutput(
            success=True,
            migration_id=migration_id,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            kv_indices=host_tail.cpu().tolist(),
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
            # Matched portion shrank despite our locks — should not happen.
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
            # Re-park pending so the watchdog (Task 10) can reclaim it.
            self.pending[recv_req.migration_id] = p
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
    ) -> Optional[TransferRequestKVCacheReqOutput]:
        """Submit the RDMA transfer to `transfer_executor` and return None.

        The scheduler tick polls `pending_futures` and emits the response when
        the future completes. Synchronous failures (no target for this rank,
        source missing pages, write_through invariant violation) return a
        ready `TransferRequestKVCacheReqOutput` immediately.
        """
        from sglang.srt.kv_migration.io_types import TransferTarget
        from sglang.srt.kv_migration.transfer_worker import do_host_to_host_rdma
        from sglang.srt.kv_migration.tree_helpers import (
            collect_host_pages,
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

        try:
            src_host_pages = collect_host_pages(self.tree_cache, full_key, page)
        except AssertionError as e:
            return TransferRequestKVCacheReqOutput(success=False, message=str(e))

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

        future = self.transfer_executor.submit(
            do_host_to_host_rdma,
            self.engine,
            my_target,
            self.host_kv_data_ptrs,
            self.host_kv_item_lens,
            src_send_pages,
        )
        self.pending_futures.append(
            (
                future,
                {
                    "matched_node": match.last_device_node,
                    "host_locked": host_locked,
                },
            )
        )
        # None signals the caller that the response will be emitted via
        # `poll_pending_futures` once the future completes.
        return None

    # -- main-loop tick helpers --

    def poll_pending_futures(self) -> List[TransferRequestKVCacheReqOutput]:
        """Called once per scheduler tick. Returns ZMQ responses for completed
        transfer futures and releases their source-side locks. The scheduler is
        responsible for sending each returned output back to the tokenizer.
        """
        from sglang.srt.kv_migration.tree_helpers import dec_host_refs

        ready: List[TransferRequestKVCacheReqOutput] = []
        still_pending: List[Tuple[Future, dict]] = []
        for future, meta in self.pending_futures:
            if not future.done():
                still_pending.append((future, meta))
                continue
            self.tree_cache.dec_lock_ref(meta["matched_node"])
            dec_host_refs(meta["host_locked"])
            try:
                ret = future.result()
                ready.append(
                    TransferRequestKVCacheReqOutput(
                        success=(ret == 0),
                        message=("" if ret == 0 else f"batch_transfer_sync ret={ret}"),
                    )
                )
            except Exception as e:
                ready.append(
                    TransferRequestKVCacheReqOutput(
                        success=False, message=f"transfer raised: {e!r}"
                    )
                )
        self.pending_futures = still_pending
        return ready

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
                    mig_id, e,
                )
            try:
                self.tree_cache.dec_lock_ref(p.matched_node)
            except Exception as e:
                logger.warning(
                    "kv-migration watchdog: dec_lock_ref failed for %s: %r",
                    mig_id, e,
                )
            dec_host_refs(p.host_locked_nodes)
            logger.warning(
                "kv-migration watchdog rolled back %s after %.1fs",
                mig_id, now - p.created_at,
            )

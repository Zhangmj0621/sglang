"""CPU-only tests for ref-aware Mamba scheduler/COW integration."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

# Must precede sglang imports on machines without a working Triton runtime.
try:
    import torch._inductor.runtime.triton_heuristics  # noqa: F401
except Exception:
    pass

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import ChunkedReqStatus
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.common import (
    alloc_for_extend,
    evict_from_tree_cache,
    evict_mamba_from_tree_cache,
)
from sglang.srt.mem_cache.mamba_radix_cache import (
    MambaChunkStashResult,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    get_child_key,
)
from sglang.srt.mem_cache.ref_aware_mamba_radix_cache import RefAwareMambaRadixCache
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)


class _FakeAllocator:
    def __init__(self, size=64):
        self.size = size
        self._next = 0
        self.device = torch.device("cpu")
        self.page_size = 1

    def alloc(self, n):
        indices = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return indices

    def free(self, indices):
        self._next -= len(indices)

    def available_size(self):
        return self.size - self._next


class _FakeMambaPool:
    def __init__(self, size=8):
        self.size = size
        self._free = list(range(size))
        self.copy_calls = []

    def alloc(self, n):
        if len(self._free) < n:
            return None
        return torch.tensor([self._free.pop() for _ in range(n)], dtype=torch.int64)

    def free(self, indices):
        self._free.extend(indices.tolist())

    def available_size(self):
        return len(self._free)

    def copy_from(self, src, dst):
        self.copy_calls.append((src.clone(), dst.clone()))


def _make_cache(mamba_slots=8):
    cache = RefAwareMambaRadixCache.__new__(RefAwareMambaRadixCache)
    cache._init_ref_aware_state(
        SimpleNamespace(high_priority_threshold=1, enable_priority_scheduling=True)
    )
    cache.req_to_token_pool = SimpleNamespace(
        mamba_pool=_FakeMambaPool(size=mamba_slots)
    )
    cache.token_to_kv_pool_allocator = _FakeAllocator()
    cache.page_size = 1
    cache.disable = False
    cache.enable_mamba_extra_buffer = False
    cache.device = torch.device("cpu")
    cache.key_match_fn = _key_match_page_size1
    cache.get_child_key_fn = get_child_key
    cache.is_eagle = False
    cache.metrics_collector = None
    cache.reset()
    return cache


def _insert(cache, token_ids):
    return cache.insert(
        InsertParams(
            key=RadixKey(token_ids),
            value=cache.token_to_kv_pool_allocator.alloc(len(token_ids)),
            mamba_value=cache.req_to_token_pool.mamba_pool.alloc(1),
        )
    )


def _set_high_ref(cache, node):
    cache._inc_priority_ref_single(node, True)


class TestWaitingMatch(unittest.TestCase):
    def test_ref_aware_pp_request_count_does_not_reapply_req_slot_capacity(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pp_size = 2
        scheduler.enable_ref_aware_kv_buffer = True
        scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 0)

        with patch(
            "sglang.srt.managers.scheduler.get_global_server_args",
            return_value=SimpleNamespace(pp_max_micro_batch_size=8),
        ):
            self.assertEqual(scheduler.get_num_allocatable_reqs(running_bs=3), 5)

    def test_ref_aware_pp_logical_slot_need_tracks_running_reclaim_and_batch_growth(
        self,
    ):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pp_size = 2
        scheduler.enable_ref_aware_kv_buffer = True
        scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 99)
        scheduler.running_batch = SimpleNamespace(reqs=[object(), object()])
        adder = SimpleNamespace(can_run_list=[])

        with patch(
            "sglang.srt.managers.scheduler.get_global_server_args",
            return_value=SimpleNamespace(pp_max_micro_batch_size=2),
        ):
            self.assertEqual(scheduler._ref_aware_running_slot_reclaim_need(adder), 1)
            scheduler.running_batch.reqs.pop()
            adder.can_run_list.append(object())
            self.assertEqual(scheduler._ref_aware_running_slot_reclaim_need(adder), 1)
            scheduler.running_batch.reqs.clear()
            self.assertEqual(scheduler._ref_aware_running_slot_reclaim_need(adder), 0)

    def test_ref_aware_pd_prefill_gate_defers_req_slots_to_exact_ledger(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 0)

        scheduler.enable_ref_aware_kv_buffer = True
        self.assertFalse(scheduler._prefill_req_pool_gate_reached(num_can_run=1))

        scheduler.enable_ref_aware_kv_buffer = False
        self.assertTrue(scheduler._prefill_req_pool_gate_reached(num_can_run=1))

    def test_only_hp_scans_past_full_batch_for_locked_ref_aware_planning(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = SimpleNamespace(
            is_high_priority=lambda priority: priority >= 1
        )

        self.assertTrue(
            scheduler._should_scan_ref_aware_full_batch_req(SimpleNamespace(priority=1))
        )
        self.assertFalse(
            scheduler._should_scan_ref_aware_full_batch_req(SimpleNamespace(priority=0))
        )

    def test_ref_aware_waiting_queue_stably_partitions_hp_before_lp(self):
        scheduler = Scheduler.__new__(Scheduler)
        hp1 = SimpleNamespace(priority=2)
        hp2 = SimpleNamespace(priority=1)
        lp1 = SimpleNamespace(priority=0)
        lp2 = SimpleNamespace(priority=-1)
        scheduler.waiting_queue = [lp1, hp1, lp2, hp2]
        scheduler.tree_cache = SimpleNamespace(
            is_high_priority=lambda priority: priority >= 1
        )

        scheduler._stable_partition_ref_aware_waiting_queue()

        self.assertEqual(scheduler.waiting_queue, [hp1, hp2, lp1, lp2])

    def test_ref_aware_mamba_waiting_match_explicitly_disables_cow(self):
        cache = SimpleNamespace(supports_mamba=lambda: True)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_ref_aware_kv_buffer = True
        scheduler.tree_cache = cache
        req = Mock()

        scheduler._init_waiting_req_for_admission(req)

        req.init_next_round_input.assert_called_once_with(cache, cow_mamba=False)

    def test_non_ref_aware_waiting_match_preserves_default_cow_behavior(self):
        cache = SimpleNamespace(supports_mamba=lambda: True)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_ref_aware_kv_buffer = False
        scheduler.tree_cache = cache
        req = Mock()

        scheduler._init_waiting_req_for_admission(req)

        req.init_next_round_input.assert_called_once_with(cache)

    def test_failed_ref_aware_match_preserves_preexisting_pool_ownership(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_ref_aware_kv_buffer = True
        free = Mock()
        scheduler.tree_cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_pool=SimpleNamespace(free=free))
        )
        main = object()
        ping_pong = object()
        req = SimpleNamespace(
            req_pool_idx=19,
            mamba_pool_idx=main,
            mamba_ping_pong_track_buffer=ping_pong,
        )
        ownership = scheduler._capture_waiting_req_ownership(req)

        scheduler._cleanup_failed_waiting_match(req, ownership)

        self.assertEqual(req.req_pool_idx, 19)
        self.assertIs(req.mamba_pool_idx, main)
        self.assertIs(req.mamba_ping_pong_track_buffer, ping_pong)
        free.assert_not_called()

    def test_side_effect_free_match_preserves_high_ref_node_and_metadata(self):
        cache = _make_cache(mamba_slots=1)
        inserted = _insert(cache, [1, 2, 3])
        node = inserted.last_node
        _set_high_ref(cache, node)
        mamba_value = node.mamba_value
        high_before = cache.mamba_high_ref_evictable_size_
        req = SimpleNamespace(mamba_pool_idx=None)

        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3]), req=req, cow_mamba=False)
        )

        self.assertEqual(result.device_indices.tolist(), [0, 1, 2])
        self.assertIs(result.last_device_node, node)
        self.assertIs(result.last_host_node, node)
        self.assertIsNone(result.mamba_branching_seqlen)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIs(node.mamba_value, mamba_value)
        self.assertEqual(cache.mamba_high_ref_evictable_size_, high_before)
        self.assertEqual(cache.req_to_token_pool.mamba_pool.copy_calls, [])


class _CopyObserved(Exception):
    pass


class TestDeferredCow(unittest.TestCase):
    def test_empty_deferred_cow_is_a_noop_for_non_mamba_pool(self):
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.req_to_token_pool = SimpleNamespace()

        batch._materialize_deferred_mamba_cow([])

    def test_prepare_for_extend_copies_only_after_allocation_while_node_is_locked(self):
        cache = _make_cache()
        node = _insert(cache, [1]).last_node
        cache.inc_lock_ref(node)  # persistent admission lock
        req = SimpleNamespace(
            fill_ids=[1, 2],
            prefix_indices=torch.tensor([0], dtype=torch.int64),
            origin_input_ids=[1, 2],
            extend_input_len=1,
            dimensions=None,
            token_type_ids=None,
            priority=0,
            last_node=node,
            mamba_pool_idx=None,
        )
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.reqs = [req]
        batch.dllm_config = None
        batch.model_config = SimpleNamespace(is_matryoshka=False)
        batch.device = "cpu"
        batch.tree_cache = cache
        batch.req_to_token_pool = cache.req_to_token_pool

        def fake_alloc_for_extend(_batch):
            self.assertIsNone(req.mamba_pool_idx)
            req.mamba_pool_idx = cache.req_to_token_pool.mamba_pool.alloc(1)[0]
            return torch.tensor([1]), torch.tensor([0]), [0]

        def observe_copy(src, dst):
            self.assertGreater(node.mamba_lock_ref, 0)
            self.assertIs(src, node.mamba_value)
            self.assertTrue(torch.equal(dst, req.mamba_pool_idx.unsqueeze(0)))
            raise _CopyObserved

        cache.req_to_token_pool.mamba_pool.copy_from = observe_copy
        with patch(
            "sglang.srt.managers.schedule_batch.alloc_for_extend",
            side_effect=fake_alloc_for_extend,
        ):
            with self.assertRaises(_CopyObserved):
                batch.prepare_for_extend()

    def test_existing_main_state_is_not_selected_for_deferred_cow(self):
        cache = _make_cache()
        node = _insert(cache, [1]).last_node
        existing_main = cache.req_to_token_pool.mamba_pool.alloc(1)[0]
        req = SimpleNamespace(last_node=node, mamba_pool_idx=existing_main)
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.tree_cache = cache
        batch.reqs = [req]

        deferred = batch._collect_deferred_mamba_cow()

        self.assertEqual(deferred, [])

    def test_failed_allocation_never_materializes_deferred_cow(self):
        cache = _make_cache()
        node = _insert(cache, [1]).last_node
        cache.inc_lock_ref(node)
        req = SimpleNamespace(
            fill_ids=[1, 2],
            prefix_indices=torch.tensor([0], dtype=torch.int64),
            origin_input_ids=[1, 2],
            extend_input_len=1,
            dimensions=None,
            token_type_ids=None,
            priority=0,
            last_node=node,
            mamba_pool_idx=None,
        )
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.reqs = [req]
        batch.dllm_config = None
        batch.model_config = SimpleNamespace(is_matryoshka=False)
        batch.device = "cpu"
        batch.tree_cache = cache
        batch.req_to_token_pool = cache.req_to_token_pool

        with patch(
            "sglang.srt.managers.schedule_batch.alloc_for_extend",
            side_effect=RuntimeError("allocation rejected"),
        ):
            with self.assertRaisesRegex(RuntimeError, "allocation rejected"):
                batch.prepare_for_extend()

        self.assertEqual(cache.req_to_token_pool.mamba_pool.copy_calls, [])
        self.assertIsNone(req.mamba_pool_idx)

    def test_ref_aware_cow_allocator_does_not_escalate_to_high_ref(self):
        cache = _make_cache(mamba_slots=2)
        target = _insert(cache, [1]).last_node
        other = _insert(cache, [2]).last_node
        _set_high_ref(cache, target)
        _set_high_ref(cache, other)
        target_value = target.mamba_value
        other_value = other.mamba_value

        with self.assertRaisesRegex(AssertionError, "Can not alloc mamba cache"):
            cache._alloc_mamba_slot_with_evict(target)

        self.assertIs(target.mamba_value, target_value)
        self.assertIs(other.mamba_value, other_value)
        self.assertEqual(cache.mamba_high_ref_evictable_size_, 2)


class TestAuthorizedShortfallEviction(unittest.TestCase):
    def test_standard_allocator_evicts_only_true_shortfall(self):
        allocator = SimpleNamespace(available_size=lambda: 7)
        cache = SimpleNamespace(
            is_chunk_cache=lambda: False,
            token_to_kv_pool_allocator=allocator,
            evict=Mock(),
        )

        evict_from_tree_cache(cache, 10)

        self.assertEqual(cache.evict.call_args.args[0].num_tokens, 3)

    def test_ref_aware_full_high_eviction_is_bounded_by_matching_authorization(self):
        cache = _make_cache()
        cache.token_to_kv_pool_allocator.size = 1
        high = _insert(cache, [1]).last_node
        _set_high_ref(cache, high)

        with self.assertRaisesRegex(RuntimeError, "authorization"):
            evict_from_tree_cache(cache, 1, high_authorization=0)

        self.assertIsNotNone(high.value)

        high_evicted = evict_from_tree_cache(cache, 1, high_authorization=1)

        self.assertGreaterEqual(high_evicted, 1)
        self.assertNotIn(high.id, cache.full_lru_list.cache)

    def test_best_effort_ref_aware_eviction_without_authorization_stops_at_safe_tier(
        self,
    ):
        cache = _make_cache()
        cache.token_to_kv_pool_allocator.size = 1
        high = _insert(cache, [1]).last_node
        _set_high_ref(cache, high)

        self.assertEqual(evict_from_tree_cache(cache, 1), 0)

        self.assertIn(high.id, cache.full_lru_list.cache)

    def test_mamba_high_eviction_requires_mamba_authorization(self):
        cache = _make_cache(mamba_slots=1)
        high = _insert(cache, [1]).last_node
        _set_high_ref(cache, high)

        with self.assertRaisesRegex(RuntimeError, "Mamba.*authorization"):
            evict_mamba_from_tree_cache(cache, 1, high_authorization=0)

        self.assertIsNotNone(high.mamba_value)
        high_evicted = evict_mamba_from_tree_cache(cache, 1, high_authorization=1)
        self.assertGreaterEqual(high_evicted, 1)
        self.assertNotIn(high.id, cache.mamba_lru_list.cache)

    def test_schedule_batch_carries_original_and_remaining_authorization(self):
        req = SimpleNamespace(
            rid="batch-owner",
            return_logprob=False,
            stream=False,
            grammar=None,
            return_hidden_states=False,
            return_routed_experts=False,
            is_prefill_only=False,
        )
        req_pool = SimpleNamespace(device="cpu")

        batch = ScheduleBatch.init_new(
            [req],
            req_pool,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            False,
            SimpleNamespace(),
            authorized_high_full_shortfall=5,
            authorized_high_mamba_shortfall=2,
            admission_reserved_full_current=11,
            admission_reserved_full_future=7,
            admission_reserved_req_slots=1,
            admission_reserved_mamba_states=3,
            diagnostic_active_chunked_req=req,
            diagnostic_new_chunked_req=req,
        )

        self.assertEqual(batch.authorized_high_full_shortfall, 5)
        self.assertEqual(batch.remaining_high_full_authorization, 5)
        self.assertEqual(batch.authorized_high_mamba_shortfall, 2)
        self.assertEqual(batch.remaining_high_mamba_authorization, 2)
        self.assertEqual(batch.admission_reserved_full_current, 11)
        self.assertEqual(batch.admission_reserved_full_future, 7)
        self.assertEqual(batch.admission_reserved_req_slots, 1)
        self.assertEqual(batch.admission_reserved_mamba_states, 3)
        self.assertIs(batch.diagnostic_active_chunked_req, req)
        self.assertIs(batch.diagnostic_new_chunked_req, req)

    def test_zero_authorization_preflight_drift_reports_complete_ledger(self):
        cache = _make_cache(mamba_slots=1)
        cache.token_to_kv_pool_allocator.size = 1
        high = _insert(cache, [1]).last_node
        _set_high_ref(cache, high)
        high_value = high.value
        high_mamba_value = high.mamba_value

        cache.req_to_token_pool.available_size = lambda: 2
        cache.req_to_token_pool.req_slots_need = lambda reqs: sum(
            req.req_pool_idx is None for req in reqs
        )
        cache.req_to_token_pool.mamba_states_need = lambda reqs: sum(
            int(req.mamba_pool_idx is None)
            + 2 * int(req.mamba_ping_pong_track_buffer is None)
            for req in reqs
        )
        req = SimpleNamespace(
            rid="candidate",
            priority=0,
            prefix_indices=torch.tensor([], dtype=torch.int64),
            req_pool_idx=None,
            mamba_pool_idx=None,
            mamba_ping_pong_track_buffer=None,
        )
        active = SimpleNamespace(rid="active")
        new = SimpleNamespace(rid="new")
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.reqs = [req]
        batch.tree_cache = cache
        batch.token_to_kv_pool_allocator = cache.token_to_kv_pool_allocator
        batch.req_to_token_pool = cache.req_to_token_pool
        batch.prefix_lens = [0]
        batch.extend_lens = [1]
        batch.extend_num_tokens = 1
        batch.device = "cpu"
        batch.maybe_evict_swa = lambda: None
        batch.admission_reserved_full_current = 1
        batch.admission_reserved_full_future = 2
        batch.admission_reserved_req_slots = 1
        batch.admission_reserved_mamba_states = 3
        batch.authorized_high_full_shortfall = 0
        batch.authorized_high_mamba_shortfall = 0
        batch.remaining_high_full_authorization = 0
        batch.remaining_high_mamba_authorization = 0
        batch.actual_high_full_evicted = 0
        batch.actual_high_mamba_evicted = 0
        batch.diagnostic_active_chunked_req = active
        batch.diagnostic_deferred_chunked_req = None
        batch.diagnostic_new_chunked_req = new
        batch.chunked_req = req

        with self.assertRaises(RuntimeError) as raised:
            alloc_for_extend(batch)

        message = str(raised.exception)
        for expected in (
            "Ref-aware allocation ledger drift after successful admission",
            "stage=full_kv_preflight",
            "full={required=1,reserved_current=1,reserved_future=2,available=0",
            "unused_evictable=0,low_evictable=0,high_evictable=1",
            "req_slots={required=1,reserved=1,available=2",
            "mamba={required=3,reserved=3,available=0",
            "rid='candidate',priority=0,req_slot_reuse=False",
            "mamba_main_reuse=False,mamba_ping_pong_reuse=False",
            "active='active',deferred=None,new='new',batch='candidate'",
            "full={original=0,remaining=0}",
            "mamba={original=0,remaining=0}",
        ):
            self.assertIn(expected, message)

        # A fail-loud invariant report must not turn zero authorization into a
        # retry that widens the eviction scope.
        self.assertIs(high.value, high_value)
        self.assertIs(high.mamba_value, high_mamba_value)


class _HashableReq:
    pass


class TestBestEffortChunkStashCallers(unittest.TestCase):
    def test_standard_scheduler_returns_cache_stash_result(self):
        req = _HashableReq()
        cache = Mock()
        cache.cache_unfinished_req.return_value = (
            MambaChunkStashResult.LIVE_PREFIX_FALLBACK
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache

        result = scheduler.stash_chunked_request(req)

        self.assertIs(result, MambaChunkStashResult.LIVE_PREFIX_FALLBACK)
        cache.cache_unfinished_req.assert_called_once_with(req, chunked=True)

    def test_disaggregated_fallback_returns_same_result_and_sends_live_range(self):
        req = _HashableReq()
        req.req_pool_idx = 0
        req.fill_ids = list(range(7))
        req.origin_input_ids = list(range(10))
        req.start_send_idx = 2
        req.rid = "pd-chunk"
        req.bootstrap_room = 123
        req.disagg_kv_sender = Mock()

        cache = Mock()
        cache.cache_unfinished_req.return_value = (
            MambaChunkStashResult.LIVE_PREFIX_FALLBACK
        )
        scheduler = SimpleNamespace(
            chunked_req=req,
            tree_cache=cache,
            enable_overlap=False,
            running_batch=SimpleNamespace(batch_is_full=True),
            last_batch=None,
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=2, get_kvcache=lambda: object()
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(32, dtype=torch.int64).reshape(1, 32)
            ),
        )
        scheduler.stash_chunked_request = lambda item: Scheduler.stash_chunked_request(
            scheduler, item
        )
        scheduler.send_kv_chunk = lambda item: (
            SchedulerDisaggregationPrefillMixin.send_kv_chunk(scheduler, item)
        )

        result = SchedulerDisaggregationPrefillMixin.process_prefill_chunk(scheduler)

        self.assertIs(result, MambaChunkStashResult.LIVE_PREFIX_FALLBACK)
        cache.cache_unfinished_req.assert_called_once_with(req, chunked=True)
        self.assertEqual(req.start_send_idx, 6)
        sent_pages, sent_state = req.disagg_kv_sender.send.call_args.args
        self.assertEqual(sent_pages.tolist(), [1, 2])
        self.assertIsNone(sent_state)
        self.assertFalse(scheduler.running_batch.batch_is_full)


class _ChunkReq:
    def __init__(self, rid):
        self.rid = rid
        self.is_chunked = 0
        self.is_retracted = False
        self.priority = 0
        self.reset_calls = 0
        self.time_stats = Mock()

    def reset_for_retract(self):
        self.reset_calls += 1
        self.is_retracted = True
        self.is_chunked = 0


class _DeferredChunkAdder:
    def __init__(self, req, status):
        self.deferred_chunked_req = req
        self.new_chunked_req = None
        self.can_run_list = []
        self.requeue_after_scan = []
        self.status = status

    def add_chunked_req(self, req, running_slot_reclaim_need=0):
        self.running_slot_reclaim_need = running_slot_reclaim_need
        if self.status is not ChunkedReqStatus.NOT_ADMITTED:
            self.can_run_list.append(req)
        return self.status

    def resolve_deferred_chunked_req(self, req):
        assert self.deferred_chunked_req is req
        self.deferred_chunked_req = None

    def append_requeue_after_scan(self, req):
        if req not in self.requeue_after_scan:
            self.requeue_after_scan.append(req)


class _ConflictAdder(_DeferredChunkAdder):
    """Records whether the conflict check itself destroyed the old owner."""

    def __init__(self, req, would_chunk):
        super().__init__(req, ChunkedReqStatus.UNFINISHED)
        self._would_chunk = would_chunk

    def would_become_chunk(self, req, truncation_align_size):
        return self._would_chunk


class TestDelayedChunkSingleOwner(unittest.TestCase):
    def _scheduler(self, old):
        try:
            server_args = get_global_server_args()
        except ValueError:
            server_args = ServerArgs(model_path="dummy")
            set_global_server_args_for_scheduler(server_args)
        server_args.pp_max_micro_batch_size = 8
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.chunked_req = old
        scheduler.enable_ref_aware_kv_buffer = True
        scheduler.tree_cache = Mock()
        scheduler.running_batch = SimpleNamespace(reqs=[])
        return scheduler

    def _takeover_scheduler(self, old, *, old_priority=0):
        old.priority = old_priority
        scheduler = self._scheduler(old)
        scheduler.truncation_align_size = None
        scheduler.enable_ref_aware_kv_buffer = True
        scheduler.tree_cache.is_high_priority = lambda priority: priority >= 1
        return scheduler

    def test_deferred_old_chunk_status_table_converges_to_one_owner(self):
        cases = (
            (ChunkedReqStatus.UNFINISHED, True, 1, False),
            (ChunkedReqStatus.COMPLETED, False, 0, False),
            (ChunkedReqStatus.NOT_ADMITTED, False, 0, True),
        )
        for status, owns_batch, expected_count, requeued in cases:
            with self.subTest(status=status):
                old = _ChunkReq(f"old-{status.name}")
                scheduler = self._scheduler(old)
                adder = _DeferredChunkAdder(old, status)

                with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
                    actual = scheduler._try_add_deferred_chunk(adder, old)

                batch_owner = scheduler._commit_batch_chunk_owner(adder)
                self.assertIs(actual, status)
                self.assertIsNone(adder.deferred_chunked_req)
                self.assertIs(batch_owner, old if owns_batch else None)
                self.assertIs(scheduler.chunked_req, old if owns_batch else None)
                self.assertEqual(old.is_chunked, expected_count)
                self.assertEqual(adder.requeue_after_scan == [old], requeued)
                self.assertEqual(old.reset_calls, int(requeued))
                self.assertEqual(release.call_count, int(requeued))

    def test_deferred_lp_after_waiting_hp_is_retracted_when_not_admitted(self):
        old = _ChunkReq("deferred-lp")
        waiting_hp = _ChunkReq("waiting-hp")
        waiting_hp.priority = 1
        running = _ChunkReq("running")
        scheduler = self._scheduler(old)
        scheduler.pp_size = 2
        scheduler.running_batch = SimpleNamespace(reqs=[running])
        adder = _DeferredChunkAdder(old, ChunkedReqStatus.NOT_ADMITTED)
        adder.can_run_list.append(waiting_hp)

        with (
            patch(
                "sglang.srt.managers.scheduler.get_global_server_args",
                return_value=SimpleNamespace(pp_max_micro_batch_size=2),
            ),
            patch("sglang.srt.managers.scheduler.release_kv_cache"),
        ):
            status = scheduler._try_add_deferred_chunk(adder, old)

        self.assertIs(status, ChunkedReqStatus.NOT_ADMITTED)
        self.assertEqual(adder.running_slot_reclaim_need, 0)
        self.assertEqual(adder.can_run_list, [waiting_hp])
        self.assertEqual(len(scheduler.running_batch.reqs) + len(adder.can_run_list), 2)
        self.assertIsNone(scheduler.chunked_req)
        self.assertEqual(adder.requeue_after_scan, [old])

    def test_ref_aware_active_chunk_is_not_charged_a_pp_slot_need(self):
        """A live owner reuses its req slot, so a full microbatch must not
        charge it a logical slot even when pp_max_micro_batch_size is
        already saturated by the running batch."""
        active = _ChunkReq("active-hp")
        active.priority = 1
        scheduler = self._scheduler(active)
        scheduler.pp_size = 2
        scheduler.running_batch = SimpleNamespace(reqs=[_ChunkReq("running")])
        adder = Mock()
        adder.can_run_list = []

        def complete(req, **_kwargs):
            adder.can_run_list.append(req)
            return ChunkedReqStatus.COMPLETED

        adder.add_chunked_req.side_effect = complete

        with patch(
            "sglang.srt.managers.scheduler.get_global_server_args",
            return_value=SimpleNamespace(pp_max_micro_batch_size=1),
        ):
            status = scheduler._try_add_active_chunk(adder, active)

        self.assertIs(status, ChunkedReqStatus.COMPLETED)
        adder.add_chunked_req.assert_called_once_with(active)
        self.assertIsNone(scheduler.chunked_req)

    def test_deferred_retraction_records_metrics_exactly_once(self):
        old = _ChunkReq("deferred-lp")
        old.origin_input_ids = [1, 2]
        old.output_ids = [3]
        scheduler = self._scheduler(old)
        scheduler.pp_size = 2
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.num_retracted_reqs = 0
        scheduler.enable_metrics = True
        scheduler.metrics_collector = Mock()
        adder = _DeferredChunkAdder(old, ChunkedReqStatus.NOT_ADMITTED)

        with (
            patch(
                "sglang.srt.managers.scheduler.get_global_server_args",
                return_value=SimpleNamespace(pp_max_micro_batch_size=2),
            ),
            patch("sglang.srt.managers.scheduler.release_kv_cache"),
        ):
            scheduler._try_add_deferred_chunk(adder, old)

        self.assertEqual(scheduler.num_retracted_reqs, 1)
        self.assertEqual(old.reset_calls, 1)
        scheduler.metrics_collector.increment_retracted_reqs.assert_called_once_with(
            num_retracted_reqs=1,
            num_retracted_input_tokens=2,
            num_retracted_output_tokens=1,
        )

    def test_old_final_chunk_can_yield_ownership_to_later_new_chunk(self):
        old = _ChunkReq("old")
        new = _ChunkReq("new")
        scheduler = self._scheduler(old)
        adder = _DeferredChunkAdder(old, ChunkedReqStatus.COMPLETED)

        scheduler._try_add_deferred_chunk(adder, old)
        adder.can_run_list.append(new)
        adder.new_chunked_req = new

        batch_owner = scheduler._commit_batch_chunk_owner(adder)

        self.assertIs(batch_owner, new)
        self.assertIs(scheduler.chunked_req, new)
        self.assertEqual(old.is_chunked, 0)
        self.assertEqual(new.is_chunked, 1)

    def test_candidate_conflict_table_never_retracts_the_old_owner(self):
        """The conflict check must never retract the deferred owner itself.

        is_high_priority is a real lambda keyed off the candidate's
        ``priority`` field (not a Mock return_value), so this also exercises
        the actual priority comparison across the (high, would_chunk) grid
        in one place -- catching a future edit that flips one branch.
        _resolve_candidate_chunk_conflict is side-effect free: the deferred
        owner survives until _plan_high_priority_admission decides the HP
        candidate is actually admissible.
        """
        cases = (
            # high, would_chunk, expected proceed
            (True, False, True),
            (True, True, True),
            (False, True, False),
        )
        for high, would_chunk, expected in cases:
            with self.subTest(high=high, would_chunk=would_chunk):
                old = _ChunkReq("old")
                candidate = _ChunkReq("candidate")
                candidate.priority = int(high)
                scheduler = self._scheduler(old)
                scheduler.truncation_align_size = None
                scheduler.tree_cache.is_high_priority = lambda priority: priority >= 1
                adder = Mock()
                adder.deferred_chunked_req = old
                adder.new_chunked_req = None
                adder.would_become_chunk.return_value = would_chunk

                proceed = scheduler._resolve_candidate_chunk_conflict(
                    adder, candidate, old
                )

                self.assertEqual(proceed, expected)
                self.assertIs(scheduler.chunked_req, old)
                self.assertIs(adder.deferred_chunked_req, old)

    def test_non_ref_active_owner_conflict_does_not_require_priority_api(self):
        old = _ChunkReq("old")
        candidate = _ChunkReq("candidate")
        scheduler = self._scheduler(old)
        scheduler.enable_ref_aware_kv_buffer = False
        scheduler.tree_cache = SimpleNamespace()  # Radix/ChunkCache priority-less API
        scheduler.truncation_align_size = None
        adder = Mock()
        adder.would_become_chunk.return_value = True
        adder.deferred_chunked_req = None
        adder.new_chunked_req = None

        proceed = scheduler._resolve_candidate_chunk_conflict(
            adder, candidate, deferred_chunked_req=None
        )

        self.assertFalse(proceed)

    def test_merely_deferred_owner_is_never_a_batch_owner(self):
        old = _ChunkReq("old")
        scheduler = self._scheduler(old)
        adder = _DeferredChunkAdder(old, ChunkedReqStatus.UNFINISHED)

        batch_owner = scheduler._commit_batch_chunk_owner(adder)

        self.assertIsNone(batch_owner)
        self.assertIs(scheduler.chunked_req, old)
        self.assertEqual(old.is_chunked, 0)

    def test_after_scan_requeue_is_direct_deduplicated_and_preserves_pd_state(self):
        victim = _ChunkReq("victim")
        victim.disagg_kv_sender = object()
        victim.metadata_buffer_index = 7
        victim.start_send_idx = 11
        scheduler = self._scheduler(None)
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.waiting_queue = []
        scheduler.enable_ref_aware_kv_buffer = False
        scheduler._add_request_to_queue = Mock(
            side_effect=AssertionError("internal requeue must bypass bootstrap")
        )
        adder = SimpleNamespace(
            preempt_list=[victim], requeue_after_scan=[victim, victim]
        )

        scheduler._flush_requeue_after_scan(adder)

        self.assertEqual(scheduler.waiting_queue, [victim])
        self.assertEqual(victim.metadata_buffer_index, 7)
        self.assertEqual(victim.start_send_idx, 11)
        self.assertIsNotNone(victim.disagg_kv_sender)
        scheduler._add_request_to_queue.assert_not_called()

    def test_internal_null_requeue_bypasses_queue_limit_revalidation(self):
        victim = _ChunkReq("victim")
        scheduler = self._scheduler(None)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.waiting_queue = []
        scheduler.enable_ref_aware_kv_buffer = False
        scheduler.max_queued_requests = 0
        scheduler._add_request_to_queue = Mock(
            side_effect=AssertionError("must not reject an internal victim")
        )
        adder = SimpleNamespace(preempt_list=[], requeue_after_scan=[victim])

        scheduler._flush_requeue_after_scan(adder)

        self.assertEqual(scheduler.waiting_queue, [victim])
        scheduler._add_request_to_queue.assert_not_called()

    def test_pd_overlap_late_chunk_result_ignores_internally_retracted_req(self):
        req = _ChunkReq("pd-overlap-victim")
        req.is_chunked = 1  # one launched chunk result is still outstanding
        req.req_pool_idx = 17
        req.output_ids = []
        req.return_logprob = False
        req.grammar = None
        req.disagg_kv_sender = object()
        req.metadata_buffer_index = 9
        req.start_send_idx = 13
        req.init_next_round_input = Mock()

        scheduler = self._scheduler(req)
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.waiting_queue = []
        scheduler.enable_ref_aware_kv_buffer = False
        scheduler.enable_metrics = False
        scheduler.enable_overlap = True
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.report_prefill_stats = Mock()
        scheduler.send_kv_chunk = Mock(
            side_effect=AssertionError("late result must not send released KV")
        )

        sender = req.disagg_kv_sender
        metadata_buffer_index = req.metadata_buffer_index
        start_send_idx = req.start_send_idx

        def release_mapping(victim, _tree_cache, is_insert=True):
            self.assertIs(victim, req)
            self.assertFalse(is_insert)
            victim.req_pool_idx = None

        with patch(
            "sglang.srt.managers.scheduler.release_kv_cache",
            side_effect=release_mapping,
        ):
            scheduler._retract_chunked_req(req)
        scheduler._restore_internal_requeued_req(req)

        self.assertTrue(req.is_retracted)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(scheduler.waiting_queue, [req])

        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_info=None,
            prefill_stats=SimpleNamespace(),
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            logits_output=None,
            next_token_ids=torch.tensor([123], dtype=torch.int64),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=None,
            can_run_cuda_graph=False,
        )

        scheduler.process_batch_result_disagg_prefill(batch, result)

        self.assertEqual(req.output_ids, [])
        self.assertEqual(req.is_chunked, 0)
        self.assertIs(req.disagg_kv_sender, sender)
        self.assertEqual(req.metadata_buffer_index, metadata_buffer_index)
        self.assertEqual(req.start_send_idx, start_send_idx)
        scheduler.tree_cache.cache_unfinished_req.assert_not_called()
        scheduler.send_kv_chunk.assert_not_called()
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        req.time_stats.set_prefill_finished_time.assert_not_called()
        req.time_stats.set_last_chunked_prefill_finish_time.assert_not_called()

        scheduler._init_waiting_req_for_admission(req)
        req.init_next_round_input.assert_called_once_with(scheduler.tree_cache)

    def test_pd_non_retracted_final_result_keeps_normal_transfer_path(self):
        req = _ChunkReq("pd-normal-final")
        req.output_ids = []
        req.return_logprob = False
        req.grammar = None

        scheduler = self._scheduler(None)
        scheduler.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.report_prefill_stats = Mock()
        scheduler.send_kv_chunk = Mock()

        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_info=None,
            prefill_stats=SimpleNamespace(),
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            logits_output=None,
            next_token_ids=torch.tensor([321], dtype=torch.int64),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=None,
            can_run_cuda_graph=False,
        )

        scheduler.process_batch_result_disagg_prefill(batch, result)

        self.assertEqual(req.output_ids, [321])
        scheduler.tree_cache.cache_unfinished_req.assert_called_once_with(req)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
        scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=True)
        req.time_stats.set_prefill_finished_time.assert_called_once_with()
        req.time_stats.set_prefill_transfer_queue_entry_time.assert_called_once_with()

    def test_pp_full_active_chunk_reaches_structured_not_admitted_requeue(self):
        req = _ChunkReq("pp-full-owner")
        req.init_next_round_input = Mock()

        scheduler = self._scheduler(req)
        scheduler.grammar_manager = Mock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_priority_preemption = False
        scheduler.running_batch = SimpleNamespace(batch_is_full=False, reqs=[])
        scheduler.waiting_queue = []
        scheduler.get_num_allocatable_reqs = Mock(return_value=0)
        scheduler.policy = Mock()
        scheduler.chunked_prefill_size = 4
        scheduler.enable_dynamic_chunking = False
        scheduler.page_size = 1
        scheduler.token_to_kv_pool_allocator = Mock()
        scheduler.new_token_ratio = 1.0
        scheduler.max_prefill_tokens = 4
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 1
        scheduler.max_running_requests = 1
        scheduler.server_args = SimpleNamespace(prefill_max_requests=None)
        scheduler.dllm_config = None
        scheduler.high_priority_threshold = 1
        scheduler.enable_lora = False
        scheduler.enable_hicache_storage = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_metrics = False

        adder = SimpleNamespace(
            can_run_list=[],
            preempt_list=[],
            requeue_after_scan=[],
            deferred_chunked_req=None,
            new_chunked_req=None,
            set_internal_retraction_recorder=Mock(),
        )
        adder.add_chunked_req = Mock(return_value=ChunkedReqStatus.NOT_ADMITTED)

        def append_requeue(victim):
            adder.requeue_after_scan.append(victim)

        adder.append_requeue_after_scan = append_requeue

        with (
            patch("sglang.srt.managers.scheduler.TEST_RETRACT", False),
            patch(
                "sglang.srt.managers.scheduler.PrefillAdder",
                return_value=adder,
            ),
            patch("sglang.srt.managers.scheduler.release_kv_cache") as release,
        ):
            batch = scheduler._get_new_batch_prefill_raw(None)

        self.assertIsNone(batch)
        adder.add_chunked_req.assert_called_once_with(req)
        adder.set_internal_retraction_recorder.assert_called_once()
        release.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        self.assertIsNone(scheduler.chunked_req)
        self.assertTrue(req.is_retracted)
        self.assertEqual(scheduler.waiting_queue, [req])

    def test_active_owner_is_not_charged_a_microbatch_slot(self):
        """A live owner already holds its req slot, so req_slot_need == 0.

        Charging it a logical slot routed it through LP reclaim; with
        ref-aware on and priority scheduling off is_high_priority() is always
        True, so no LP victims exist and the in-flight chunk was retracted.
        """
        old = _ChunkReq("active-owner")
        scheduler = self._scheduler(old)
        scheduler.pp_size = 2
        scheduler.running_batch = SimpleNamespace(
            reqs=[_ChunkReq(f"running-{i}") for i in range(8)]
        )
        adder = _DeferredChunkAdder(old, ChunkedReqStatus.UNFINISHED)
        adder.deferred_chunked_req = None

        scheduler._try_add_active_chunk(adder, old)

        self.assertEqual(adder.running_slot_reclaim_need, 0)
        self.assertIs(scheduler.chunked_req, old)
        self.assertEqual(adder.requeue_after_scan, [])

    def test_deferred_owner_is_not_charged_a_microbatch_slot(self):
        old = _ChunkReq("deferred-owner")
        scheduler = self._scheduler(old)
        scheduler.pp_size = 2
        scheduler.running_batch = SimpleNamespace(
            reqs=[_ChunkReq(f"running-{i}") for i in range(8)]
        )
        adder = _DeferredChunkAdder(old, ChunkedReqStatus.UNFINISHED)

        scheduler._try_add_deferred_chunk(adder, old)

        self.assertEqual(adder.running_slot_reclaim_need, 0)
        self.assertIs(scheduler.chunked_req, old)
        self.assertEqual(adder.requeue_after_scan, [])

    def test_waiting_candidate_still_gets_a_nonzero_reclaim_need(self):
        """Slot accounting must stay correct for genuinely new requests."""
        scheduler = self._scheduler(None)
        scheduler.pp_size = 2
        scheduler.running_batch = SimpleNamespace(
            reqs=[_ChunkReq(f"running-{i}") for i in range(8)]
        )
        adder = _DeferredChunkAdder(None, ChunkedReqStatus.UNFINISHED)
        adder.deferred_chunked_req = None

        self.assertGreater(scheduler._ref_aware_running_slot_reclaim_need(adder), 0)

    def test_conflict_check_does_not_retract_the_deferred_owner(self):
        """The check runs before capacity is known, so it must not destroy.

        would_become_chunk() is a prediction: init_load_back can change the
        post-lock shape, and the HP candidate may still be rejected for
        capacity.  Retracting here cost the LP chunk its progress while the
        HP did not run either.
        """
        old = _ChunkReq("deferred-lp")
        scheduler = self._scheduler(old)
        scheduler.truncation_align_size = None
        scheduler.tree_cache.is_high_priority = lambda priority: priority >= 1
        adder = _ConflictAdder(old, would_chunk=True)
        candidate = _ChunkReq("hp-candidate")
        candidate.priority = 1

        proceed = scheduler._resolve_candidate_chunk_conflict(adder, candidate, old)

        self.assertTrue(proceed)
        self.assertIs(adder.deferred_chunked_req, old)
        self.assertIs(scheduler.chunked_req, old)
        self.assertEqual(old.reset_calls, 0)
        self.assertEqual(adder.requeue_after_scan, [])

    def test_conflict_check_still_blocks_a_second_owner(self):
        """A non-HP candidate that would chunk must not be let through."""
        old = _ChunkReq("deferred-lp")
        scheduler = self._scheduler(old)
        scheduler.truncation_align_size = None
        scheduler.tree_cache.is_high_priority = Mock(return_value=False)
        adder = _ConflictAdder(old, would_chunk=True)
        candidate = _ChunkReq("lp-candidate")

        proceed = scheduler._resolve_candidate_chunk_conflict(adder, candidate, old)

        self.assertFalse(proceed)
        self.assertIs(adder.deferred_chunked_req, old)

    def test_non_chunking_candidate_always_proceeds(self):
        old = _ChunkReq("deferred-lp")
        scheduler = self._scheduler(old)
        scheduler.truncation_align_size = None
        adder = _ConflictAdder(old, would_chunk=False)
        candidate = _ChunkReq("small-candidate")

        self.assertTrue(
            scheduler._resolve_candidate_chunk_conflict(adder, candidate, old)
        )
        self.assertIs(adder.deferred_chunked_req, old)
        self.assertIs(scheduler.chunked_req, old)
        self.assertEqual(old.reset_calls, 0)

    def test_hp_candidate_may_displace_a_deferred_lp_owner(self):
        old = _ChunkReq("deferred-lp")
        scheduler = self._takeover_scheduler(old)
        adder = _ConflictAdder(old, would_chunk=True)
        candidate = _ChunkReq("hp-candidate")
        candidate.priority = 1

        self.assertTrue(
            scheduler._resolve_candidate_chunk_conflict(adder, candidate, old)
        )
        # The slot is displaceable, so add_one_req must NOT be told the slot
        # is taken -- that gate is what previously made takeover unreachable.
        self.assertFalse(scheduler._chunk_slot_is_taken(adder, candidate, old))
        # Still side-effect free: the planner owns the retract.
        self.assertIs(scheduler.chunked_req, old)
        self.assertIs(adder.deferred_chunked_req, old)
        self.assertEqual(old.reset_calls, 0)

    def test_lp_candidate_may_not_displace_a_deferred_lp_owner(self):
        old = _ChunkReq("deferred-lp")
        scheduler = self._takeover_scheduler(old)
        adder = _ConflictAdder(old, would_chunk=True)
        candidate = _ChunkReq("lp-candidate")
        candidate.priority = 0

        self.assertFalse(
            scheduler._resolve_candidate_chunk_conflict(adder, candidate, old)
        )
        self.assertTrue(scheduler._chunk_slot_is_taken(adder, candidate, old))
        self.assertIs(adder.deferred_chunked_req, old)

    def test_hp_candidate_may_not_displace_a_committed_new_owner(self):
        old = _ChunkReq("deferred-lp")
        scheduler = self._takeover_scheduler(old)
        adder = _ConflictAdder(old, would_chunk=True)
        adder.new_chunked_req = _ChunkReq("already-new-owner")
        candidate = _ChunkReq("hp-candidate")
        candidate.priority = 1

        self.assertTrue(scheduler._chunk_slot_is_taken(adder, candidate, old))
        self.assertFalse(
            scheduler._resolve_candidate_chunk_conflict(adder, candidate, old)
        )

    def test_hp_candidate_may_not_displace_a_deferred_hp_owner(self):
        old = _ChunkReq("deferred-hp")
        scheduler = self._takeover_scheduler(old, old_priority=1)
        adder = _ConflictAdder(old, would_chunk=True)
        candidate = _ChunkReq("hp-candidate")
        candidate.priority = 1

        self.assertTrue(scheduler._chunk_slot_is_taken(adder, candidate, old))

    def test_non_ref_aware_never_displaces_an_owner(self):
        old = _ChunkReq("owner")
        scheduler = self._takeover_scheduler(old)
        scheduler.enable_ref_aware_kv_buffer = False
        adder = _ConflictAdder(old, would_chunk=True)
        candidate = _ChunkReq("candidate")
        candidate.priority = 1

        self.assertTrue(scheduler._chunk_slot_is_taken(adder, candidate, old))
        self.assertFalse(
            scheduler._resolve_candidate_chunk_conflict(adder, candidate, old)
        )

    def test_non_chunking_candidate_ignores_the_slot_entirely(self):
        old = _ChunkReq("deferred-lp")
        scheduler = self._takeover_scheduler(old)
        adder = _ConflictAdder(old, would_chunk=False)
        candidate = _ChunkReq("small-lp")
        candidate.priority = 0

        self.assertTrue(
            scheduler._resolve_candidate_chunk_conflict(adder, candidate, old)
        )


if __name__ == "__main__":
    unittest.main()

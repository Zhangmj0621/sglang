"""CPU scheduler contracts backed by the actual Unified cache and allocator."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import UpdateSessionPriorityReqInput
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def make_cache(priority=True):
    pool = MHATokenToKVPool(
        size=64,
        page_size=1,
        dtype=torch.float16,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=64,
        dtype=torch.float16,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            enable_session_radix_cache=True,
            enable_priority_scheduling=priority,
            tree_components=(ComponentType.FULL,),
        )
    )


def cache_session(cache, tokens, session_id, priority):
    generation = cache.ensure_session_generation(session_id)
    value = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    node_id = cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=value.to(torch.int64),
        )
    ).last_device_node
    req = SimpleNamespace(
        session_id=session_id,
        session=None,
        session_generation=generation,
        priority=priority,
        last_node=node_id,
    )
    assert cache.session_refs.begin_request(req)[0]
    cache.session_refs.register_session_ref(req)
    cache.session_refs.end_request(req)
    return node_id


def make_req(cache, priority=0, tokens=(101, 102, 103, 104), session_id=None):
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(max_new_tokens=1),
        priority=priority,
        session_id=session_id,
    )
    if session_id is not None:
        req.session_generation = cache.ensure_session_generation(session_id)
    req.init_next_round_input(cache)
    return req


def allocate_chunk(cache, req, num_tokens=4):
    cache.req_to_token_pool.alloc([req])
    indices = cache.token_to_kv_pool_allocator.alloc(num_tokens)
    cache.req_to_token_pool.req_to_token[req.req_pool_idx, :num_tokens] = indices
    req.kv = ReqKvInfo(kv_allocated_len=num_tokens, swa_evicted_seqlen=0)
    req.kv_committed_len = num_tokens
    cache.inc_lock_ref(req.last_node)


def make_adder(cache, chunk=None):
    return PrefillAdder(
        page_size=1,
        tree_cache=cache,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        running_batch=ScheduleBatch(reqs=[]),
        new_token_ratio=1,
        rem_input_tokens=64,
        rem_chunk_tokens=chunk,
    )


def make_decode(cache, priorities, page_size=1, committed=None):
    cache.token_to_kv_pool_allocator.page_size = page_size
    if committed is None:
        committed = [page_size] * len(priorities)
    return ScheduleBatch(
        reqs=[
            SimpleNamespace(priority=p, kv_committed_len=n)
            for p, n in zip(priorities, committed)
        ],
        tree_cache=cache,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )


class SessionPrioritySchedulerTests(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.cache = make_cache()

    def test_lp_admission_excludes_hp_cache(self):
        cache_session(self.cache, list(range(20)), "hp", 1)
        self.cache.token_to_kv_pool_allocator.alloc(41)
        req = make_req(self.cache, priority=0)
        adder = make_adder(self.cache)
        result = adder.add_one_req(req, False, None)
        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(self.cache.session_evictable_size(allow_high=True), 20)

    def test_hp_admission_can_budget_hp_cache(self):
        cache_session(self.cache, list(range(20)), "hp", 1)
        self.cache.token_to_kv_pool_allocator.alloc(41)
        req = make_req(self.cache, priority=1)
        adder = make_adder(self.cache)
        self.assertEqual(adder.add_one_req(req, False, None), AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [req])
        self.assertFalse(self.cache.tree_core.session_allow_high)

    def test_lp_budget_is_rechecked_after_prefix_lock(self):
        node = cache_session(self.cache, [1, 2, 3, 4], "lp", 0)
        self.cache.token_to_kv_pool_allocator.alloc(59)
        req = make_req(self.cache, priority=0, tokens=(1, 2, 3, 4, 5))
        adder = make_adder(self.cache)
        self.assertEqual(adder.add_one_req(req, False, None), AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(self.cache.session_evictable_size(), 4)
        self.assertEqual(
            self.cache.tree_core.node_by_id(node)
            .component_data[ComponentType.FULL]
            .lock_ref,
            0,
        )

    def test_lp_chunk_yields_when_only_hp_cache_remains(self):
        cache_session(self.cache, list(range(20)), "hp", 1)
        self.cache.token_to_kv_pool_allocator.alloc(44)
        req = make_req(self.cache, priority=0)
        adder = make_adder(self.cache, chunk=4)
        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual(adder.can_run_list, [])

    def test_hp_chunk_uses_original_fallback(self):
        self.cache.token_to_kv_pool_allocator.alloc(64)
        req = make_req(self.cache, priority=1, tokens=range(100, 108))
        adder = make_adder(self.cache, chunk=4)
        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(req.extend_range.length, 4)

    def test_request_scope_restores_after_exception(self):
        adder = make_adder(self.cache)
        req = make_req(self.cache, priority=1)
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with adder._session_request_scope(req):
                self.assertTrue(self.cache.tree_core.session_allow_high)
                raise RuntimeError("failure")
        self.assertFalse(self.cache.tree_core.session_allow_high)

    def test_lp_decode_does_not_evict_hp(self):
        cache_session(self.cache, [1], "hp", 1)
        self.cache.token_to_kv_pool_allocator.alloc(63)
        batch = make_decode(self.cache, [0])
        self.assertFalse(batch.check_decode_mem())
        self.assertEqual(self.cache.session_evictable_size(allow_high=True), 1)

    def test_mixed_decode_caps_high_eviction_to_hp_demand(self):
        cache_session(self.cache, [1], "hp1", 1)
        cache_session(self.cache, [2], "hp2", 1)
        self.cache.token_to_kv_pool_allocator.alloc(62)
        batch = make_decode(self.cache, [1, 0])
        with patch.object(
            self.cache, "evict_for_session", wraps=self.cache.evict_for_session
        ) as evict:
            self.assertFalse(batch.check_decode_mem())
        self.assertEqual(len(evict.call_args_list), 2)
        self.assertEqual(evict.call_args_list[0].args[0].num_tokens, 2)
        self.assertEqual(evict.call_args_list[1].args[0].num_tokens, 1)
        self.assertEqual(self.cache.token_to_kv_pool_allocator.available_size(), 1)

    def test_decode_selected_subset_cannot_borrow_excluded_hp_permission(self):
        cache_session(self.cache, [1], "hp", 1)
        self.cache.token_to_kv_pool_allocator.alloc(63)
        batch = make_decode(self.cache, [1, 0])
        self.assertFalse(batch.check_decode_mem(selected_indices=[1]))
        self.assertEqual(self.cache.token_to_kv_pool_allocator.available_size(), 0)

    def test_hp_that_needs_no_new_page_cannot_authorize_lp_decode(self):
        cache_session(self.cache, list(range(8)), "hp", 1)
        self.cache.token_to_kv_pool_allocator.alloc(56)
        batch = make_decode(self.cache, [1, 0], page_size=4, committed=[1, 4])
        with patch.object(
            self.cache, "evict_for_session", wraps=self.cache.evict_for_session
        ) as evict:
            self.assertFalse(batch.check_decode_mem())
        self.assertEqual(len(evict.call_args_list), 1)
        self.assertEqual(evict.call_args.args[0].num_tokens, 4)
        self.assertFalse(evict.call_args.kwargs["allow_high"])

    def test_low_tier_decode_preserves_total_demand_headroom(self):
        cache_session(self.cache, [1], "lp1", 0)
        cache_session(self.cache, [2], "lp2", 0)
        self.cache.token_to_kv_pool_allocator.alloc(61)
        batch = make_decode(self.cache, [0, 0])
        self.assertTrue(batch.check_decode_mem())
        self.assertEqual(self.cache.token_to_kv_pool_allocator.available_size(), 3)

    def test_speculative_hp_with_existing_reserve_does_not_authorize_lp(self):
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = "EAGLE"
        args.speculative_num_steps = 3
        args.speculative_eagle_topk = 1
        args.speculative_num_draft_tokens = 4
        set_global_server_args_for_scheduler(args)
        cache_session(self.cache, list(range(8)), "hp", 1)
        self.cache.token_to_kv_pool_allocator.alloc(56)
        batch = make_decode(self.cache, [1, 0], page_size=4, committed=[4, 4])
        batch.spec_algorithm = SimpleNamespace(is_none=lambda: False)
        batch.reqs[0].kv = ReqKvInfo(kv_allocated_len=12, swa_evicted_seqlen=0)
        batch.reqs[1].kv = ReqKvInfo(kv_allocated_len=4, swa_evicted_seqlen=0)
        self.assertEqual(batch.new_tokens_required_next_decode([0]), 0)
        self.assertEqual(batch.new_tokens_required_next_decode([1]), 8)
        with patch.object(
            self.cache, "evict_for_session", wraps=self.cache.evict_for_session
        ) as evict:
            self.assertFalse(batch.check_decode_mem())
        self.assertEqual(len(evict.call_args_list), 1)
        self.assertFalse(evict.call_args.kwargs["allow_high"])

    def test_hp_memory_preemption_releases_lp_and_preserves_activity(self):
        args = ServerArgs(model_path="dummy", enable_priority_scheduling=True)
        set_global_server_args_for_scheduler(args)
        low = make_req(self.cache, priority=0, session_id="lp")
        low.sampling_params.max_new_tokens = 100
        self.cache.session_refs.begin_request(low)
        allocate_chunk(self.cache, low)
        running = ScheduleBatch(
            reqs=[low],
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.cache.token_to_kv_pool_allocator,
            req_to_token_pool=self.cache.req_to_token_pool,
        )
        adder = make_adder(self.cache)
        adder.running_batch = running
        adder.rem_total_token_offset = 100
        high = make_req(self.cache, priority=1)
        self.assertEqual(adder.add_one_req(high, False, None), AddReqResult.CONTINUE)
        self.assertEqual(adder.preempt_list, [low])
        self.assertEqual(running.reqs, [])
        self.assertTrue(low.is_retracted)
        self.assertEqual(
            self.cache.session_refs._session_states["lp"].active_requests, 1
        )

    def test_later_admission_rejection_retains_preempted_request_for_requeue(self):
        args = ServerArgs(model_path="dummy", enable_priority_scheduling=True)
        set_global_server_args_for_scheduler(args)
        low = make_req(self.cache, priority=0, session_id="lp")
        low.sampling_params.max_new_tokens = 100
        self.assertTrue(self.cache.session_refs.begin_request(low)[0])
        allocate_chunk(self.cache, low)
        running = ScheduleBatch(
            reqs=[low],
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.cache.token_to_kv_pool_allocator,
            req_to_token_pool=self.cache.req_to_token_pool,
        )
        adder = make_adder(self.cache)
        adder.running_batch = running
        adder.rem_total_token_offset = 100
        # The first admission gate preempts LP; reject later at the delayer gate.
        adder.prefill_delayer_single_pass = SimpleNamespace(
            negotiate_should_allow_prefill=lambda **_kwargs: False
        )
        adder.max_prefill_bs = 1
        adder.max_running_requests = 8
        adder.waiting_queue_len = 1

        high = make_req(self.cache, priority=1)
        self.assertEqual(adder.add_one_req(high, False, None), AddReqResult.OTHER)
        self.assertEqual(running.reqs, [])
        self.assertTrue(low.is_retracted)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.preempt_list, [low])

    def test_empty_prefill_pass_requeues_requests_preempted_during_admission(self):
        scheduler = self.make_scheduler()
        high = make_req(self.cache, priority=1)
        low = make_req(self.cache, priority=0)
        scheduler.waiting_queue = [high]
        scheduler.grammar_manager.has_waiting_grammars = lambda: False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_priority_preemption = True
        scheduler.is_hybrid_swa = False
        scheduler.min_free_slots_delayer = None
        scheduler.get_num_allocatable_reqs = lambda _running_bs: 8
        scheduler.policy = SimpleNamespace(calc_priority=lambda *_args: None)
        scheduler.enable_dynamic_chunking = False
        scheduler.chunked_prefill_size = 4
        scheduler.page_size = 1
        scheduler.token_to_kv_pool_allocator = self.cache.token_to_kv_pool_allocator
        mamba_allocator = MagicMock()
        scheduler.req_to_token_pool = SimpleNamespace(
            available_size=self.cache.req_to_token_pool.available_size,
            mamba_allocator=mamba_allocator,
        )
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1)
        scheduler.max_prefill_tokens = 64
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 10
        scheduler.truncation_align_size = None
        scheduler.max_prefill_bs = 1
        scheduler.max_running_requests = 8
        scheduler.dllm_config = None
        scheduler.enable_lora = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hicache_storage = False
        scheduler._add_request_to_queue = MagicMock()

        class RejectingAdder:
            def __init__(self, *_args, **_kwargs):
                self.can_run_list = []
                self.preempt_list = [low]
                self.new_chunked_req = None
                # Shared Full/Mamba pools must not reserve a group before the
                # request-scoped Full budget has admitted the batch.
                self._mamba_slot_cost = 1

            def add_one_req(self, *_args, **_kwargs):
                return AddReqResult.OTHER

        with (
            patch("sglang.srt.managers.scheduler.PrefillAdder", RejectingAdder),
            patch(
                "sglang.srt.managers.scheduler.get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_schedule",
                return_value=SimpleNamespace(prefill_max_requests=None),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(pp_max_micro_batch_size=8),
            ),
        ):
            batch, returned_running = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=scheduler.running_batch,
            )

        self.assertIsNone(batch)
        self.assertIs(returned_running, scheduler.running_batch)
        scheduler._add_request_to_queue.assert_called_once_with(low)
        mamba_allocator.alloc_group_begin.assert_not_called()
        mamba_allocator.alloc_group_end.assert_not_called()

    def test_session_memory_preemption_honors_disable_flag(self):
        args = ServerArgs(
            model_path="dummy",
            enable_priority_scheduling=True,
            disable_priority_preemption=True,
        )
        set_global_server_args_for_scheduler(args)
        low = make_req(self.cache, priority=0)
        low.sampling_params.max_new_tokens = 100
        allocate_chunk(self.cache, low)
        adder = make_adder(self.cache)
        adder.running_batch = ScheduleBatch(reqs=[low])
        adder.rem_total_token_offset = 100
        self.assertEqual(
            adder.add_one_req(make_req(self.cache, 1), False, None),
            AddReqResult.NO_TOKEN,
        )
        self.assertEqual(adder.preempt_list, [])
        self.assertIsNotNone(low.req_pool_idx)

    def test_priority_disabled_allows_default_request_to_use_high_cache(self):
        cache = make_cache(priority=False)
        cache_session(cache, [1], "session", None)
        cache.token_to_kv_pool_allocator.alloc(63)
        self.assertTrue(make_decode(cache, [None]).check_decode_mem())

    def test_retraction_orders_lp_before_hp_without_changing_length_tiebreak(self):
        reqs = [
            SimpleNamespace(priority=1, output_ids=[], origin_input_ids=[1]),
            SimpleNamespace(priority=0, output_ids=[1, 2], origin_input_ids=[1]),
            SimpleNamespace(priority=0, output_ids=[1], origin_input_ids=[1]),
        ]
        args = SimpleNamespace(retraction_policy="length")
        self.assertEqual(
            ScheduleBatch._get_decode_retraction_order(reqs, args, self.cache),
            [0, 1, 2],
        )

    def make_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = self.cache
        scheduler.waiting_queue = []
        scheduler.running_batch = ScheduleBatch(reqs=[])
        scheduler.last_batch = None
        scheduler.chunked_req = None
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
        )
        return scheduler

    def test_update_priority_updates_live_requests_but_not_old_generation(self):
        scheduler = self.make_scheduler()
        waiting = make_req(self.cache, priority=0, session_id="s")
        running = make_req(self.cache, priority=0, session_id="s")
        dllm_waiting = make_req(self.cache, priority=0, session_id="s")
        dllm_staging = make_req(self.cache, priority=0, session_id="s")
        hisparse_staging = make_req(self.cache, priority=0, session_id="s")
        stale = make_req(self.cache, priority=0, session_id="s")
        for req in (
            waiting,
            running,
            dllm_waiting,
            dllm_staging,
            hisparse_staging,
        ):
            self.assertTrue(scheduler._begin_session_request(req))
        stale._session_ref_activity = ("s", waiting.session_generation - 1)
        scheduler.waiting_queue = [waiting, stale]
        scheduler.running_batch.reqs = [running]
        scheduler.chunked_req = running
        scheduler.dllm_manager = SimpleNamespace(
            waiting_queue=[dllm_waiting], staging_queue=[dllm_staging]
        )
        scheduler.hisparse_coordinator = SimpleNamespace(
            ack_staging_queue=[SimpleNamespace(req=hisparse_staging)]
        )
        result = scheduler.update_session_priority(
            UpdateSessionPriorityReqInput(session_id="s", priority=5)
        )
        self.assertTrue(result.success and result.found)
        self.assertEqual(
            [
                waiting.priority,
                running.priority,
                dllm_waiting.priority,
                dllm_staging.priority,
                hisparse_staging.priority,
                stale.priority,
            ],
            [5, 5, 5, 5, 5, 0],
        )
        self.assertEqual(
            self.cache.session_refs._session_states["s"].active_requests, 5
        )

    def test_dllm_terminal_paths_release_session_activity(self):
        for fdfo in (False, True):
            with self.subTest(first_done_first_out_mode=fdfo):
                cache = make_cache()
                scheduler = self.make_scheduler()
                scheduler.tree_cache = cache
                scheduler.token_to_kv_pool_allocator = cache.token_to_kv_pool_allocator
                scheduler.dllm_config = SimpleNamespace(
                    first_done_first_out_mode=fdfo,
                    block_size=1,
                )
                scheduler.metrics_reporter = SimpleNamespace(
                    num_generated_tokens=0,
                    report_prefill_stats=MagicMock(),
                )
                scheduler.output_streamer = SimpleNamespace(stream_output=MagicMock())

                req = make_req(
                    cache,
                    priority=1,
                    tokens=(101 + int(fdfo), 102, 103, 104),
                    session_id=f"dllm-{int(fdfo)}",
                )
                req.full_untruncated_fill_ids.append(0)
                req.set_extend_range(0, len(req.full_untruncated_fill_ids))
                allocate_chunk(cache, req, num_tokens=5)
                self.assertTrue(scheduler._begin_session_request(req))
                state = cache.session_refs._session_states[req.session_id]
                self.assertEqual(state.active_requests, 1)

                batch = ScheduleBatch(reqs=[req])
                result = SimpleNamespace(
                    copy_done=None,
                    next_token_ids=[torch.tensor([200])],
                    accept_length_per_req_cpu=[1] if fdfo else None,
                    dllm_algo_state=None,
                    can_run_cuda_graph=False,
                )
                scheduler.process_batch_result_dllm(batch, result)

                self.assertTrue(req.finished())
                self.assertEqual(state.active_requests, 0)
                self.assertIsNone(req.req_pool_idx)
                self.assertIsNone(req.kv)

    def test_unknown_rank_reports_not_found_without_failing_other_dp_ranks(self):
        result = self.make_scheduler().update_session_priority(
            UpdateSessionPriorityReqInput(session_id="missing", priority=1)
        )
        self.assertTrue(result.success)
        self.assertFalse(result.found)

    def test_requeue_is_idempotent_and_rejection_does_not_leak_activity(self):
        scheduler = self.make_scheduler()
        first = make_req(self.cache, priority=1, session_id="s")
        rejected = make_req(self.cache, priority=0, session_id="s")
        self.assertTrue(scheduler._begin_session_request(first))
        self.assertTrue(scheduler._begin_session_request(first))
        self.assertFalse(scheduler._begin_session_request(rejected))
        self.assertEqual(
            self.cache.session_refs._session_states["s"].active_requests, 1
        )
        scheduler._end_session_request(first)
        scheduler._end_session_request(first)
        self.assertEqual(
            self.cache.session_refs._session_states["s"].active_requests, 0
        )
        self.assertEqual(
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_count, 1
        )

    def test_deferred_chunk_is_retracted_even_with_no_admitted_requests(self):
        scheduler = self.make_scheduler()
        req = make_req(self.cache, priority=0)
        scheduler.chunked_req = req
        adder = make_adder(self.cache, chunk=4)
        allocate_chunk(self.cache, req)
        self.cache.token_to_kv_pool_allocator.alloc(60)
        result = scheduler._try_add_session_chunk(adder, req)
        self.assertIs(result, req)
        self.assertIsNone(scheduler.chunked_req)
        self.assertTrue(req.is_retracted)
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.kv)
        self.assertEqual(self.cache.token_to_kv_pool_allocator.available_size(), 4)

    def test_new_hp_chunk_keeps_single_owner(self):
        scheduler = self.make_scheduler()
        low = make_req(self.cache, priority=0)
        high = make_req(self.cache, priority=1)
        scheduler.chunked_req = low
        adder = make_adder(self.cache, chunk=4)
        adder.new_chunked_req = high
        adder.can_run_list.append(high)
        allocate_chunk(self.cache, low)
        self.assertIs(scheduler._try_add_session_chunk(adder, low), low)
        self.assertIsNone(low.req_pool_idx)
        self.assertEqual(self.cache.token_to_kv_pool_allocator.available_size(), 64)
        self.assertIsNone(scheduler.chunked_req)
        self.assertIs(adder.new_chunked_req, high)
        self.assertEqual(adder.can_run_list, [high])


if __name__ == "__main__":
    unittest.main()

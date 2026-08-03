import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    ChunkedReqStatus,
    PrefillAdder,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.ref_aware_cache_core import RefAwareCacheCore
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


class _LedgerReqPool:
    def __init__(self, *, req_slots: int, mamba_states: int, ping_pong_size: int = 2):
        self._req_slots = req_slots
        self._mamba_states = mamba_states
        self._ping_pong_size = ping_pong_size
        # Mirror HybridReqToTokenPool: mamba_state_need charges the ping-pong
        # buffers, so release-gain accounting must be able to see them too.
        self.enable_mamba_extra_buffer = True
        self.mamba_ping_pong_track_buffer_size = ping_pong_size
        self.mamba_pool = SimpleNamespace(available_size=lambda: self._mamba_states)

    def available_size(self):
        return self._req_slots

    def req_slot_need(self, req):
        return int(req.req_pool_idx is None)

    def mamba_state_need(self, req):
        return int(req.mamba_pool_idx is None) + (
            self._ping_pong_size if req.mamba_ping_pong_track_buffer is None else 0
        )


class _LedgerRefAwareCache(RefAwareCacheCore):
    """Small ref-aware cache seam for scheduler-ledger unit tests."""

    def __init__(
        self,
        *,
        req_slots: int,
        mamba_states: int,
        full_safe_evictable: int = 0,
        mamba_safe_evictable: int = 0,
        full_lock_cost: int = 0,
        mamba_lock_cost: int = 0,
        full_high_evictable: int = 0,
        mamba_high_evictable: int = 0,
    ):
        self.req_to_token_pool = _LedgerReqPool(
            req_slots=req_slots, mamba_states=mamba_states
        )
        self.full_safe_evictable = full_safe_evictable
        self.mamba_safe_evictable = mamba_safe_evictable
        self.full_lock_cost = full_lock_cost
        self.mamba_lock_cost = mamba_lock_cost
        self.full_high_evictable = full_high_evictable
        self.mamba_high_evictable = mamba_high_evictable
        self.lock_depth = 0
        self.inc_lock_calls = 0
        self.dec_lock_calls = 0
        self.disable = False

    def supports_mamba(self):
        return True

    def supports_swa(self):
        return False

    def is_tree_cache(self):
        return True

    def is_high_priority(self, priority):
        return priority >= 1

    def safe_evictable_size_by_tier(self, allow_low=True, allow_high=False):
        assert allow_low
        safe = max(
            0,
            self.full_safe_evictable - (self.full_lock_cost if self.lock_depth else 0),
        )
        return safe + (self.full_high_evictable if allow_high else 0)

    def mamba_evictable_size_by_tier(self, allow_low=True, allow_high=False):
        assert allow_low
        safe = max(
            0,
            self.mamba_safe_evictable
            - (self.mamba_lock_cost if self.lock_depth else 0),
        )
        return safe + (self.mamba_high_evictable if allow_high else 0)

    def inc_lock_ref(self, _node):
        self.inc_lock_calls += 1
        self.lock_depth += 1
        return IncLockRefResult()

    def dec_lock_ref(self, _node, _params=None):
        self.dec_lock_calls += 1
        self.lock_depth -= 1
        return DecLockRefResult()


class TestPrefillAdder(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.mock_tree_cache = self.create_tree_cache()
        self.mock_token_allocator = self.create_token_allocator()

    def create_tree_cache(
        self,
        *,
        full_evictable_size: int = 0,
        swa_evictable_size: int = 0,
        evictable_size: int = 0,
    ) -> MagicMock:
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = full_evictable_size
        tree_cache.swa_evictable_size.return_value = swa_evictable_size
        tree_cache.evictable_size.return_value = evictable_size
        tree_cache.disable = False
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        return tree_cache

    def create_token_allocator(
        self,
        *,
        full_available_size: int = 0,
        swa_available_size: int = 0,
        available_size: int = 0,
    ) -> MagicMock:
        allocator = MagicMock()
        allocator.full_available_size.return_value = full_available_size
        allocator.swa_available_size.return_value = swa_available_size
        allocator.available_size.return_value = available_size
        return allocator

    def create_running_batch(self, reqs=None) -> MagicMock:
        batch = MagicMock()
        batch.reqs = list(reqs or [])
        batch.release_req.return_value = None
        batch.filter_batch.return_value = None
        return batch

    def create_server_args(
        self, *, schedule_low_priority_values_first: bool
    ) -> MagicMock:
        server_args = MagicMock()
        server_args.schedule_low_priority_values_first = (
            schedule_low_priority_values_first
        )
        return server_args

    def create_mock_req(self, rid, priority, max_new_tokens, output_len=0, wait_time=0):
        req = MagicMock(spec=Req)
        req.rid = str(rid)
        req.priority = priority
        req.extend_input_len = 0
        req.extend_logprob_start_len = 0
        req.output_ids = [0] * output_len
        req.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
        req.time_stats = SimpleNamespace(wait_queue_entry_time=wait_time)
        req.finished.return_value = False
        return req

    def create_adder(self, running_batch, **kwargs):
        defaults = dict(
            page_size=1,
            tree_cache=self.mock_tree_cache,
            token_to_kv_pool_allocator=self.mock_token_allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=None,
            mixed_with_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )
        defaults.update(kwargs)
        return PrefillAdder(**defaults)

    def test_preempt_success_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req, mock_server_args)

        self.assertTrue(success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)  # 50 + 75 + 100 - 50 = 175
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req, mock_server_args)

        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 125)  # 50 + 75 + 100 - 100 = 125
        running_batch.release_req.assert_called_once()

    def test_preempt_fail_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=2, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_fail_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=0, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=-1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_skip_already_preempted_request(self):
        params = [
            ("req_prio_0", 0, 50),
            ("req_prio_1", 1, 75),
            ("req_prio_2", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = 225
        self.mock_token_allocator.available_size.return_value = 225

        # New request preempts req_prio_0
        first_req = self.create_mock_req(
            "new_req_prio_1", priority=1, max_new_tokens=49
        )
        first_success = adder.preempt_to_schedule(first_req, mock_server_args)
        self.assertTrue(first_success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)
        running_batch.release_req.assert_called_once()

        # Second call needs more tokens than currently free, so it would need to
        # preempt req_prio_0 again if already-preempted requests were not filtered out.
        second_req = self.create_mock_req(
            "second_new_req_prio_1", priority=1, max_new_tokens=76
        )
        second_success = adder.preempt_to_schedule(second_req, mock_server_args)

        self.assertFalse(second_success)
        self.assertEqual(adder.rem_total_token_offset, 175)
        self.assertEqual(adder.preempt_list.count(running_reqs[0]), 1)
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_once(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=75)

        success = adder.preempt_to_schedule(new_req, mock_server_args)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 375
        )  # 50 + 75 + 100 + 125 + 125 - 100 = 375
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_twice(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=200)

        success = adder.preempt_to_schedule(new_req, mock_server_args)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertIn(running_reqs[3], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 250
        )  # 50 + 75 + 100 + 125 + 125 - 100 - 125 = 250
        self.assertEqual(running_batch.release_req.call_count, 2)

    def test_mixed_chunk_prefill_budgets(self):
        self.mock_token_allocator.available_size.return_value = 1000

        decode_reqs = [
            self.create_mock_req(f"decode_{i}", priority=0, max_new_tokens=50)
            for i in range(8)
        ]
        running_batch = self.create_running_batch(decode_reqs)

        adder = self.create_adder(
            running_batch,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            mixed_with_decode_tokens=len(decode_reqs),
        )

        self.assertEqual(adder.rem_input_tokens, 192)  # 200 - 8
        self.assertEqual(adder.rem_chunk_tokens, 56)  # 64 - 8
        self.assertEqual(adder.rem_total_token_offset, 408)  # 8 + 8 * 50
        self.assertEqual(adder.cur_rem_token_offset, 8)
        self.assertEqual(adder.budget_state(), AddReqResult.CONTINUE)

        # Add a prefill that exactly consumes the chunk budget
        req1 = self.create_mock_req("req1", priority=0, max_new_tokens=64)
        req1.extend_input_len = 56
        req1.host_hit_length = 0
        req1.prefix_indices = []
        req1.fill_ids = list(range(56))
        req1.last_node = MagicMock()
        req1.sampling_params.ignore_eos = False

        result1 = adder.add_one_req(
            req1, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder.can_run_list), 1)
        self.assertEqual(adder.rem_chunk_tokens, 0)  # 56 - 56
        self.assertEqual(adder.rem_input_tokens, 136)  # 192 - 56
        self.assertEqual(result1, AddReqResult.OTHER)

        # 3 decode requests finished
        remaining_decode_reqs = decode_reqs[3:]
        running_batch2 = self.create_running_batch(remaining_decode_reqs)

        adder2 = self.create_adder(
            running_batch2,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            mixed_with_decode_tokens=len(remaining_decode_reqs),
        )

        self.assertEqual(adder2.rem_input_tokens, 195)  # 200 - 5
        self.assertEqual(adder2.rem_chunk_tokens, 59)  # 64 - 5
        self.assertEqual(adder2.rem_total_token_offset, 255)  # 5 + 5 * 50
        self.assertEqual(adder2.budget_state(), AddReqResult.CONTINUE)

        # Same prefill no longer exhausts the chunk budget
        req2 = self.create_mock_req("req2", priority=0, max_new_tokens=64)
        req2.extend_input_len = 56
        req2.host_hit_length = 0
        req2.prefix_indices = []
        req2.fill_ids = list(range(56))
        req2.last_node = MagicMock()
        req2.sampling_params.ignore_eos = False

        result2 = adder2.add_one_req(
            req2, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 1)
        self.assertEqual(adder2.rem_chunk_tokens, 3)  # 59 - 56 = 3 remaining
        self.assertEqual(result2, AddReqResult.CONTINUE)

        # Fit last small prefill request
        req3 = self.create_mock_req("req3", priority=0, max_new_tokens=16)
        req3.extend_input_len = 3
        req3.host_hit_length = 0
        req3.prefix_indices = []
        req3.fill_ids = list(range(3))
        req3.last_node = MagicMock()
        req3.sampling_params.ignore_eos = False

        result3 = adder2.add_one_req(
            req3, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 2)
        self.assertEqual(adder2.rem_chunk_tokens, 0)  # 3 - 3 = 0
        self.assertEqual(result3, AddReqResult.OTHER)

    def test_ignore_eos_initial_request_is_tracked_once(self):
        self.mock_tree_cache.disable = True
        self.mock_token_allocator.available_size.return_value = 1000
        adder = self.create_adder(self.create_running_batch())
        req = self.create_mock_req("ignore-eos", priority=0, max_new_tokens=10)
        req.sampling_params.ignore_eos = True
        req.extend_input_len = 1
        req.origin_input_ids = [1]
        req.host_hit_length = 0
        req.prefix_indices = []
        req.fill_ids = [1]
        req.last_node = MagicMock()

        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(adder.req_states, [(10.0, 1)])

    def test_non_ref_aware_chunk_is_admitted_when_mixed_chunk_exhausts_budget(self):
        """rem_chunk_tokens = chunked_prefill_size - running_bs can go <= 0
        under --enable-mixed-chunk.  HEAD always admitted the owner in that
        case ('otherwise it will cause a memory leak'); returning
        NOT_ADMITTED makes the scheduler retract and requeue the chunk every
        round, which livelocks the request."""
        tree_cache = self.create_tree_cache()
        allocator = self.create_token_allocator()
        running_batch = self.create_running_batch()
        adder = PrefillAdder(
            page_size=1,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=4096,
            rem_chunk_tokens=512,
            mixed_with_decode_tokens=512,  # running_bs == chunked_prefill_size
        )
        self.assertLessEqual(adder.rem_chunk_tokens, 0)

        owner = self.create_mock_req("owner", priority=0, max_new_tokens=8)
        owner.extend_input_len = 256
        owner.prefix_indices = []
        owner.fill_ids = list(range(256))

        def set_extend_input_len(value):
            owner.extend_input_len = value

        owner.set_extend_input_len.side_effect = set_extend_input_len

        status = adder.add_chunked_req(owner)

        self.assertIsNot(status, ChunkedReqStatus.NOT_ADMITTED)
        self.assertIn(owner, adder.can_run_list)


class TestPrefillAdderResourceLedger(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _make_adder(
        self,
        *,
        full_available=100,
        full_safe_evictable=0,
        req_slots=100,
        mamba_states=100,
        mamba_safe_evictable=0,
        full_lock_cost=0,
        mamba_lock_cost=0,
        full_high_evictable=0,
        mamba_high_evictable=0,
        rem_chunk_tokens=None,
        page_size=1,
        mixed_with_decode_tokens=0,
        running_reqs=None,
    ):
        cache = _LedgerRefAwareCache(
            req_slots=req_slots,
            mamba_states=mamba_states,
            full_safe_evictable=full_safe_evictable,
            mamba_safe_evictable=mamba_safe_evictable,
            full_lock_cost=full_lock_cost,
            mamba_lock_cost=mamba_lock_cost,
            full_high_evictable=full_high_evictable,
            mamba_high_evictable=mamba_high_evictable,
        )
        allocator = SimpleNamespace(available_size=lambda: full_available)
        running_batch = MagicMock()
        running_batch.reqs = list(running_reqs or [])
        adder = PrefillAdder(
            page_size=page_size,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=100,
            rem_chunk_tokens=rem_chunk_tokens,
            mixed_with_decode_tokens=mixed_with_decode_tokens,
            enable_ref_aware_kv_buffer=True,
        )
        return adder, cache

    def _make_req(
        self,
        rid,
        *,
        extend=1,
        max_new_tokens=0,
        priority=0,
        reuse_req_slot=False,
        reuse_main=False,
        reuse_ping_pong=False,
    ):
        req = MagicMock(spec=Req)
        req.rid = rid
        req.priority = priority
        req.extend_input_len = extend
        req.host_hit_length = 0
        req.prefix_indices = []
        req.fill_ids = list(range(extend))
        req.last_node = object()
        req.req_pool_idx = 7 if reuse_req_slot else None
        req.mamba_pool_idx = 8 if reuse_main else None
        req.mamba_ping_pong_track_buffer = object() if reuse_ping_pong else None
        req.output_ids = []
        req.sampling_params = SimpleNamespace(
            max_new_tokens=max_new_tokens,
            ignore_eos=False,
        )
        req.finished.return_value = False

        def set_extend_input_len(value):
            req.extend_input_len = value

        req.set_extend_input_len.side_effect = set_extend_input_len
        return req

    def _add(self, adder, req):
        return adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

    def test_mamba_reservations_accumulate_and_failed_candidate_does_not_commit(self):
        adder, cache = self._make_adder(mamba_states=5)
        first = self._make_req("first")
        second = self._make_req("second")

        self._add(adder, first)
        result = self._add(adder, second)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [first])
        self.assertEqual(adder.reserved_mamba_states, 3)
        self.assertEqual(adder.reserved_req_slots, 1)
        self.assertEqual(adder.reserved_full_current, 1)
        self.assertEqual(adder.reserved_full_future, 0)
        self.assertEqual(cache.lock_depth, 1)  # only first's persistent lock remains
        # The cumulative miss is visible before locking the second request.
        self.assertEqual(cache.inc_lock_calls, 2)
        self.assertEqual(cache.dec_lock_calls, 1)

    def test_mixed_mamba_demand_and_reused_resources_are_charged_exactly(self):
        adder, _ = self._make_adder(req_slots=2, mamba_states=5)
        needs_three = self._make_req("three")
        needs_two = self._make_req("two", reuse_main=True)
        needs_zero = self._make_req(
            "zero", reuse_req_slot=True, reuse_main=True, reuse_ping_pong=True
        )

        self._add(adder, needs_three)
        self._add(adder, needs_two)
        result = self._add(adder, needs_zero)

        self.assertNotEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [needs_three, needs_two, needs_zero])
        self.assertEqual(adder.reserved_mamba_states, 5)
        self.assertEqual(adder.reserved_req_slots, 2)

    def test_all_resources_allow_exact_fit(self):
        adder, _ = self._make_adder(
            full_available=3,
            req_slots=1,
            mamba_states=3,
        )
        req = self._make_req("exact", extend=1, max_new_tokens=2)

        result = self._add(adder, req)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(adder.reserved_full_current, 1)
        self.assertEqual(adder.reserved_full_future, 2)
        self.assertEqual(adder.reserved_req_slots, 1)
        self.assertEqual(adder.reserved_mamba_states, 3)

    def test_committed_paged_reservation_is_not_charged_a_second_overhead(self):
        adder, _ = self._make_adder(
            full_available=4,
            req_slots=1,
            mamba_states=3,
            page_size=4,
        )
        req = self._make_req("paged", extend=1)

        self._add(adder, req)

        self.assertEqual(adder.reserved_full_current, 4)
        self.assertEqual(adder.rem_total_token_offset, 4)
        self.assertEqual(adder.cur_rem_token_offset, 4)

    def test_each_resource_deficit_rejects_without_partial_reservation(self):
        cases = [
            dict(full_available=0, req_slots=1, mamba_states=3),
            dict(full_available=1, req_slots=0, mamba_states=3),
            dict(full_available=1, req_slots=1, mamba_states=2),
        ]
        for i, capacities in enumerate(cases):
            with self.subTest(capacities=capacities):
                adder, cache = self._make_adder(**capacities)
                req = self._make_req(f"reject-{i}")

                result = self._add(adder, req)

                self.assertEqual(result, AddReqResult.NO_TOKEN)
                self.assertEqual(adder.can_run_list, [])
                self.assertEqual(adder.reserved_req_slots, 0)
                self.assertEqual(adder.reserved_mamba_states, 0)
                self.assertEqual(adder.reserved_full_current, 0)
                self.assertEqual(adder.reserved_full_future, 0)
                self.assertEqual(cache.lock_depth, 0)

    def test_locked_capacity_is_rechecked_and_temporary_lock_is_released(self):
        adder, cache = self._make_adder(
            full_available=0,
            full_safe_evictable=1,
            full_lock_cost=1,
        )
        req = self._make_req("locked")

        result = self._add(adder, req)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.reserved_full_current, 0)
        self.assertEqual(cache.inc_lock_calls, 1)
        self.assertEqual(cache.dec_lock_calls, 1)
        self.assertEqual(cache.lock_depth, 0)

    def test_hp_reservation_reduces_safe_capacity_for_later_lp(self):
        adder, _ = self._make_adder(full_available=2, mamba_states=6)
        hp = self._make_req("hp", priority=1)
        lp = self._make_req("lp", priority=0, max_new_tokens=1)

        self._add(adder, hp)
        result = self._add(adder, lp)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [hp])
        self.assertEqual(adder.reserved_full_current, 1)
        self.assertEqual(adder.reserved_full_future, 0)

    def test_zero_increment_lp_reuse_is_rejected_under_hp_authorization(self):
        """Strict semantics: LP admission ignores authorized_high_*.

        Deficits are cumulative, so once an HP commits a reservation beyond
        safe capacity every later LP is rejected -- including a chunk
        continuation whose marginal demand is zero.  This throughput cost is
        accepted deliberately (see spec 4.4.1); do not "fix" it by letting
        LP read the authorization again.
        """
        adder, _ = self._make_adder(
            full_available=0,
            req_slots=1,
            mamba_states=0,
            full_high_evictable=1,
            mamba_high_evictable=3,
        )
        hp = self._make_req("hp", priority=1)
        reused_lp = self._make_req(
            "reused-lp",
            extend=0,
            priority=0,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )

        self._add(adder, hp)
        full_authorization = adder.authorized_high_full_shortfall
        mamba_authorization = adder.authorized_high_mamba_shortfall
        result = self._add(adder, reused_lp)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [hp])
        self.assertEqual(adder.authorized_high_full_shortfall, full_authorization)
        self.assertEqual(adder.authorized_high_mamba_shortfall, mamba_authorization)

    def test_lp_within_safe_capacity_is_still_admitted(self):
        adder, _ = self._make_adder(full_available=100, req_slots=10, mamba_states=10)
        lp = self._make_req("lp", priority=0)

        result = self._add(adder, lp)

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [lp])
        self.assertEqual(adder.authorized_high_full_shortfall, 0)
        self.assertEqual(adder.authorized_high_mamba_shortfall, 0)

    def test_lp_cannot_expand_existing_hp_authorization(self):
        adder, _ = self._make_adder(
            full_available=0,
            req_slots=1,
            mamba_states=0,
            full_high_evictable=2,
            mamba_high_evictable=3,
        )
        hp = self._make_req("hp", priority=1)
        growing_lp = self._make_req(
            "growing-lp",
            extend=1,
            priority=0,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )

        self._add(adder, hp)
        self.assertEqual(adder.authorized_high_full_shortfall, 1)
        result = self._add(adder, growing_lp)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [hp])
        self.assertEqual(adder.authorized_high_full_shortfall, 1)
        self.assertEqual(adder.authorized_high_mamba_shortfall, 3)

    def _make_running_lp(self, rid, *, tokens=4, main=True, ping_pong=True):
        req = self._make_req(
            rid,
            extend=tokens,
            priority=0,
            reuse_req_slot=True,
            reuse_main=main,
            reuse_ping_pong=ping_pong,
        )
        req.origin_input_ids = list(range(tokens))
        req.output_ids = []
        return req

    def test_infeasible_hp_destroys_no_victim(self):
        # One LP victim can never cover a mamba deficit this large, and no
        # high-ref capacity exists.  The planner must bail out before any
        # destructive release.
        victim = self._make_running_lp("lp-victim", main=False, ping_pong=False)
        adder, _ = self._make_adder(
            mamba_states=0,
            mamba_safe_evictable=0,
            mamba_high_evictable=0,
            running_reqs=[victim],
        )
        hp = self._make_req("hp", priority=1)

        result = self._add(adder, hp)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.requeue_after_scan, [])

    def test_insufficient_logical_slots_destroys_no_victim(self):
        # running_slot_reclaim_need exceeds the number of LP victims.
        victim = self._make_running_lp("lp-only")
        adder, _ = self._make_adder(running_reqs=[victim])
        hp = self._make_req("hp", priority=1)
        demand = adder._make_ref_aware_demand(hp, truncation_align_size=None)

        admitted = adder._plan_high_priority_admission(
            demand, running_slot_reclaim_need=3
        )

        self.assertFalse(admitted)
        self.assertEqual(adder.requeue_after_scan, [])

    def test_pre_check_accepts_a_deficit_the_victims_can_cover(self):
        # The victim holds a main state; releasing it covers this HP's
        # one-state demand, so the pre-check must let the real release run.
        victim = self._make_running_lp("lp-victim")
        adder, _ = self._make_adder(
            mamba_states=0,
            mamba_safe_evictable=0,
            running_reqs=[victim],
        )
        hp = self._make_req("hp", priority=1)
        demand = adder._make_ref_aware_demand(hp, truncation_align_size=None)

        self.assertTrue(adder._reclaim_could_satisfy(demand, [victim]))

    def test_pre_check_rejects_a_deficit_no_victim_can_cover(self):
        # A victim that holds no mamba state contributes no mamba gain.
        victim = self._make_running_lp("lp-stateless", main=False, ping_pong=False)
        adder, _ = self._make_adder(
            mamba_states=0,
            mamba_safe_evictable=0,
            mamba_high_evictable=0,
            running_reqs=[victim],
        )
        hp = self._make_req("hp", priority=1)
        demand = adder._make_ref_aware_demand(hp, truncation_align_size=None)

        self.assertFalse(adder._reclaim_could_satisfy(demand, [victim]))

    def test_authorization_is_never_lowered_by_a_later_smaller_hp(self):
        adder, _ = self._make_adder(
            full_available=0,
            full_high_evictable=5,
            mamba_high_evictable=5,
        )
        adder.authorized_high_full_shortfall = 5
        adder.authorized_high_mamba_shortfall = 5
        small = self._make_req("hp-small", priority=1)
        demand = adder._make_ref_aware_demand(small, truncation_align_size=None)

        adder._plan_high_priority_admission(demand)

        self.assertEqual(adder.authorized_high_full_shortfall, 5)
        self.assertEqual(adder.authorized_high_mamba_shortfall, 5)

    def test_intermediate_chunk_has_no_future_reservation(self):
        adder, _ = self._make_adder(
            full_available=20,
            mamba_states=3,
            rem_chunk_tokens=2,
        )
        req = self._make_req("chunk", extend=4, max_new_tokens=5)

        self._add(adder, req)

        self.assertIs(adder.new_chunked_req, req)
        self.assertEqual(req.extend_input_len, 2)
        self.assertEqual(adder.reserved_full_current, 2)
        self.assertEqual(adder.reserved_full_future, 0)

        final_adder, _ = self._make_adder(
            full_available=20,
            mamba_states=3,
            rem_chunk_tokens=4,
        )
        final = self._make_req("final", extend=4, max_new_tokens=5)
        self._add(final_adder, final)
        self.assertEqual(final_adder.reserved_full_current, 4)
        self.assertEqual(final_adder.reserved_full_future, 5)

    def test_would_become_chunk_and_admission_share_the_exact_shape(self):
        cases = (
            # extend, budget, alignment, expected chunk, expected admitted len
            (4, 2, None, True, 2),
            (4, 4, None, False, 4),
            (9, 6, 4, True, 4),
        )
        for extend, budget, alignment, expected_chunk, expected_len in cases:
            with self.subTest(extend=extend, budget=budget, alignment=alignment):
                adder, _ = self._make_adder(
                    full_available=32,
                    mamba_states=3,
                    rem_chunk_tokens=budget,
                )
                req = self._make_req("shape", extend=extend)

                self.assertEqual(
                    adder.would_become_chunk(req, alignment), expected_chunk
                )
                adder.add_one_req(
                    req,
                    has_chunked_req=False,
                    truncation_align_size=alignment,
                )

                self.assertIs(adder.new_chunked_req is req, expected_chunk)
                self.assertEqual(req.extend_input_len, expected_len)
                self.assertIn(req, adder.can_run_list)

    def test_existing_owner_blocks_candidate_before_second_owner_is_created(self):
        adder, cache = self._make_adder(
            full_available=20,
            mamba_states=3,
            rem_chunk_tokens=2,
        )
        req = self._make_req("second-owner", extend=4)

        result = adder.add_one_req(
            req, has_chunked_req=True, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(adder.new_chunked_req)
        self.assertEqual(cache.inc_lock_calls, 0)

    def test_existing_owner_does_not_block_a_non_chunk_candidate(self):
        adder, _ = self._make_adder(
            full_available=20,
            mamba_states=3,
            rem_chunk_tokens=4,
        )
        req = self._make_req("non-owner", extend=2)

        adder.add_one_req(req, has_chunked_req=True, truncation_align_size=None)

        self.assertEqual(adder.can_run_list, [req])
        self.assertIsNone(adder.new_chunked_req)

    def test_chunk_continuation_reports_all_structured_outcomes(self):
        cases = (
            (4, 2, 20, ChunkedReqStatus.UNFINISHED, True),
            (2, 2, 20, ChunkedReqStatus.COMPLETED, True),
            (4, 2, 0, ChunkedReqStatus.NOT_ADMITTED, False),
        )
        for extend, budget, available, expected, admitted in cases:
            with self.subTest(expected=expected):
                adder, _ = self._make_adder(
                    full_available=available,
                    mamba_states=3,
                    rem_chunk_tokens=budget,
                )
                req = self._make_req(
                    "continuation",
                    extend=extend,
                    reuse_req_slot=True,
                    reuse_main=True,
                    reuse_ping_pong=True,
                )

                result = adder.add_chunked_req(req)

                self.assertIs(result, expected)
                self.assertEqual(req in adder.can_run_list, admitted)
                self.assertEqual(adder.reserved_full_current, 2 if admitted else 0)

    def test_hp_chunk_reclaims_one_running_lp_for_logical_slot(self):
        lp = self._make_req(
            "running-lp",
            priority=0,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        lp.origin_input_ids = [1]
        lp.retraction_count = 0
        adder, _ = self._make_adder(
            full_available=1,
            req_slots=1,
            mamba_states=3,
            running_reqs=[lp],
        )
        recorder = Mock()
        adder.set_internal_retraction_recorder(recorder)

        def release_req(*_args):
            lp.retraction_count += 1
            lp.reset_for_retract()

        adder.running_batch.release_req.side_effect = release_req

        def filter_batch(*, keep_indices):
            old_reqs = adder.running_batch.reqs
            adder.running_batch.reqs = [old_reqs[i] for i in keep_indices]

        adder.running_batch.filter_batch.side_effect = filter_batch
        hp = self._make_req(
            "active-hp",
            priority=1,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )

        status = adder.add_chunked_req(hp, running_slot_reclaim_need=1)

        self.assertIs(status, ChunkedReqStatus.COMPLETED)
        self.assertEqual(adder.can_run_list, [hp])
        self.assertEqual(adder.running_batch.reqs, [])
        self.assertEqual(adder.requeue_after_scan, [lp])
        self.assertEqual(lp.retraction_count, 1)
        lp.reset_for_retract.assert_called_once_with()
        recorder.assert_called_once_with(lp)

    def test_chunk_logical_slot_rejects_lp_and_hp_without_lp_victim(self):
        cases = (
            (0, 0, "active-lp"),
            (1, 1, "active-hp"),
        )
        for priority, running_priority, rid in cases:
            with self.subTest(rid=rid):
                running = self._make_req(
                    "running",
                    priority=running_priority,
                    reuse_req_slot=True,
                    reuse_main=True,
                    reuse_ping_pong=True,
                )
                running.origin_input_ids = [1]
                adder, _ = self._make_adder(
                    full_available=1,
                    req_slots=1,
                    mamba_states=3,
                    running_reqs=[running],
                )
                active = self._make_req(
                    rid,
                    priority=priority,
                    reuse_req_slot=True,
                    reuse_main=True,
                    reuse_ping_pong=True,
                )

                status = adder.add_chunked_req(active, running_slot_reclaim_need=1)

                self.assertIs(status, ChunkedReqStatus.NOT_ADMITTED)
                self.assertEqual(adder.can_run_list, [])
                self.assertEqual(adder.running_batch.reqs, [running])
                adder.running_batch.release_req.assert_not_called()

    def test_impossible_hp_records_each_released_lp_once(self):
        lp = self._make_req(
            "running-lp",
            priority=0,
            max_new_tokens=1,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        lp.origin_input_ids = [1, 2]
        lp.output_ids = [3]
        lp.retraction_count = 0
        # The HP must survive the feasibility pre-check (releasing this LP
        # could plausibly cover its mamba demand) and only then be rejected by
        # the post-release final check -- that is the path this test records.
        adder, _ = self._make_adder(
            full_available=10,
            req_slots=1,
            mamba_states=0,
            running_reqs=[lp],
        )
        scheduler_metrics = Scheduler.__new__(Scheduler)
        scheduler_metrics.num_retracted_reqs = 0
        scheduler_metrics.enable_metrics = True
        scheduler_metrics.metrics_collector = Mock()
        adder.set_internal_retraction_recorder(
            scheduler_metrics._record_internal_retraction
        )

        def release_req(*_args):
            lp.retraction_count += 1
            lp.reset_for_retract()

        adder.running_batch.release_req.side_effect = release_req
        hp = self._make_req(
            "impossible-hp",
            priority=1,
            reuse_req_slot=True,
        )

        status = adder.add_chunked_req(hp, running_slot_reclaim_need=1)

        self.assertIs(status, ChunkedReqStatus.NOT_ADMITTED)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.requeue_after_scan, [lp])
        self.assertEqual(scheduler_metrics.num_retracted_reqs, 1)
        self.assertEqual(lp.retraction_count, 1)
        lp.reset_for_retract.assert_called_once_with()
        increment_retracted = (
            scheduler_metrics.metrics_collector.increment_retracted_reqs
        )
        increment_retracted.assert_called_once_with(
            num_retracted_reqs=1,
            num_retracted_input_tokens=2,
            num_retracted_output_tokens=1,
        )

    def test_hp_shortfall_is_authorized_only_after_locked_final_check(self):
        adder, cache = self._make_adder(
            full_available=0,
            mamba_states=0,
            full_high_evictable=1,
            mamba_high_evictable=3,
        )
        hp = self._make_req("hp", priority=1)

        result = self._add(adder, hp)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [hp])
        self.assertEqual(adder.authorized_high_full_shortfall, 1)
        self.assertEqual(adder.authorized_high_mamba_shortfall, 3)
        self.assertEqual(cache.inc_lock_calls, 2)
        self.assertEqual(cache.dec_lock_calls, 1)

    def test_hp_batch_authorization_is_recomputed_from_total_committed_demand(self):
        adder, _ = self._make_adder(
            full_available=0,
            mamba_states=0,
            full_high_evictable=2,
            mamba_high_evictable=6,
        )
        first = self._make_req("hp-1", priority=1)
        second = self._make_req("hp-2", priority=1)

        self._add(adder, first)
        self._add(adder, second)

        self.assertEqual(adder.can_run_list, [first, second])
        self.assertEqual(adder.authorized_high_full_shortfall, 2)
        self.assertEqual(adder.authorized_high_mamba_shortfall, 6)

    def test_hp_req_slot_deficit_cannot_be_high_authorized(self):
        adder, cache = self._make_adder(
            full_available=0,
            req_slots=0,
            mamba_states=0,
            full_high_evictable=1,
            mamba_high_evictable=3,
        )
        hp = self._make_req("hp", priority=1)

        result = self._add(adder, hp)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.authorized_high_full_shortfall, 0)
        self.assertEqual(adder.authorized_high_mamba_shortfall, 0)
        self.assertEqual(cache.lock_depth, 0)

    def test_hp_mamba_deficit_reclaims_running_lp_inside_prefix_lock(self):
        adder, cache = self._make_adder(
            full_available=1,
            req_slots=1,
            mamba_states=0,
        )
        lp = self._make_req(
            "running-lp",
            priority=0,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        lp.origin_input_ids = [1, 2, 3]
        lp.output_ids = []
        adder.running_batch.reqs = [lp]

        def release_req(_idx, _remaining, _server_args):
            self.assertGreater(cache.lock_depth, 0)
            cache.req_to_token_pool._mamba_states += 3

        adder.running_batch.release_req.side_effect = release_req
        hp = self._make_req("hp", priority=1)

        self._add(adder, hp)

        self.assertEqual(adder.can_run_list, [hp])
        self.assertEqual(adder.authorized_high_mamba_shortfall, 0)
        self.assertEqual(adder.requeue_after_scan, [lp])
        adder.running_batch.release_req.assert_called_once()

    def test_hp_logical_slots_reclaim_one_running_lp_per_candidate(self):
        first_lp = self._make_req(
            "running-lp-long",
            priority=0,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        first_lp.origin_input_ids = [1, 2]
        first_lp.output_ids = []
        second_lp = self._make_req(
            "running-lp-short",
            priority=0,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        second_lp.origin_input_ids = [1]
        second_lp.output_ids = []
        adder, cache = self._make_adder(
            full_available=2,
            req_slots=2,
            mamba_states=6,
            running_reqs=[second_lp, first_lp],
        )

        def filter_batch(*, keep_indices):
            old_reqs = adder.running_batch.reqs
            adder.running_batch.reqs = [old_reqs[i] for i in keep_indices]

        adder.running_batch.filter_batch.side_effect = filter_batch
        adder.running_batch.release_req.side_effect = lambda *_args: self.assertGreater(
            cache.lock_depth, 0
        )
        first_hp = self._make_req("hp-1", priority=1)
        second_hp = self._make_req("hp-2", priority=1)

        first_result = adder.add_one_req(
            first_hp,
            has_chunked_req=False,
            truncation_align_size=None,
            running_slot_reclaim_need=1,
        )
        self.assertGreater(cache.lock_depth, 0)
        self.assertLessEqual(len(adder.running_batch.reqs) + len(adder.can_run_list), 2)
        second_result = adder.add_one_req(
            second_hp,
            has_chunked_req=False,
            truncation_align_size=None,
            running_slot_reclaim_need=1,
        )

        self.assertIn(first_result, (AddReqResult.CONTINUE, AddReqResult.NO_TOKEN))
        self.assertIn(second_result, (AddReqResult.CONTINUE, AddReqResult.NO_TOKEN))
        self.assertEqual(adder.can_run_list, [first_hp, second_hp])
        self.assertEqual(adder.running_batch.reqs, [])
        self.assertEqual(adder.requeue_after_scan, [first_lp, second_lp])
        self.assertEqual(adder.running_batch.release_req.call_count, 2)
        self.assertLessEqual(len(adder.running_batch.reqs) + len(adder.can_run_list), 2)

    def test_hp_logical_slot_rejects_when_only_running_hp_exists(self):
        running_hp = self._make_req(
            "running-hp",
            priority=1,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        running_hp.origin_input_ids = [1]
        running_hp.output_ids = []
        adder, cache = self._make_adder(
            full_available=1,
            req_slots=1,
            mamba_states=3,
            running_reqs=[running_hp],
        )

        result = adder.add_one_req(
            self._make_req("waiting-hp", priority=1),
            has_chunked_req=False,
            truncation_align_size=None,
            running_slot_reclaim_need=1,
        )

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.running_batch.reqs, [running_hp])
        adder.running_batch.release_req.assert_not_called()
        self.assertEqual(cache.lock_depth, 0)

    def test_hp_reclaims_deferred_lp_before_running_lp(self):
        adder, cache = self._make_adder(
            full_available=1,
            req_slots=1,
            mamba_states=0,
        )
        running_lp = self._make_req(
            "running-lp",
            priority=0,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        running_lp.origin_input_ids = [1]
        running_lp.output_ids = []
        adder.running_batch.reqs = [running_lp]
        deferred_lp = self._make_req(
            "deferred-lp",
            priority=0,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        events = []

        def reclaim_deferred(req):
            self.assertGreater(cache.lock_depth, 0)
            events.append(req.rid)
            cache.req_to_token_pool._mamba_states += 3

        adder.set_deferred_chunked_req(deferred_lp, reclaim_deferred)
        adder.running_batch.release_req.side_effect = lambda *_args: events.append(
            "running-lp"
        )

        self._add(adder, self._make_req("hp", priority=1))

        self.assertEqual(events, ["deferred-lp"])
        self.assertIsNone(adder.deferred_chunked_req)
        self.assertEqual(adder.requeue_after_scan, [deferred_lp])

    def test_hp_new_chunk_reclaims_deferred_lp_even_without_resource_deficit(self):
        adder, _ = self._make_adder(
            full_available=20,
            req_slots=2,
            mamba_states=6,
            rem_chunk_tokens=1,
        )
        deferred_lp = self._make_req(
            "deferred-lp",
            priority=0,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        reclaimed = []
        adder.set_deferred_chunked_req(deferred_lp, reclaimed.append)
        hp = self._make_req("hp", priority=1, extend=2)

        self._add(adder, hp)

        self.assertEqual(reclaimed, [deferred_lp])
        self.assertIsNone(adder.deferred_chunked_req)
        self.assertIs(adder.new_chunked_req, hp)

    def test_running_reclaim_updates_round_start_future_and_mixed_offsets(self):
        lp = self._make_req(
            "running-lp",
            priority=0,
            max_new_tokens=2,
            reuse_req_slot=True,
            reuse_main=True,
            reuse_ping_pong=True,
        )
        lp.origin_input_ids = [1]
        lp.output_ids = []
        adder, cache = self._make_adder(
            full_available=4,
            req_slots=1,
            mamba_states=0,
            mixed_with_decode_tokens=1,
            running_reqs=[lp],
        )
        adder.running_batch.release_req.side_effect = lambda *_args: setattr(
            cache.req_to_token_pool,
            "_mamba_states",
            cache.req_to_token_pool._mamba_states + 3,
        )

        self._add(adder, self._make_req("hp", priority=1))

        self.assertEqual(adder._round_start_total_token_offset, 0)
        self.assertEqual(adder._round_start_cur_token_offset, 0)
        self.assertEqual(adder.rem_total_token_offset, 1)
        self.assertEqual(adder.cur_rem_token_offset, 1)


if __name__ == "__main__":
    unittest.main()

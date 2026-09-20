"""CPU tests for session-priority admission on the PD decode side."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from session_priority_test_utils import FULL, MAMBA, SWA, insert, make_cache, register

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_queue(cache):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.tree_cache = cache
    queue.token_to_kv_pool_allocator = cache.token_to_kv_pool_allocator
    queue.token_to_kv_pool = cache.token_to_kv_pool_allocator.get_kvcache()
    queue.req_to_token_pool = cache.req_to_token_pool
    queue.num_reserved_decode_tokens = 0
    queue.retracted_queue = []
    queue.transfer_queue = SimpleNamespace(queue=[])
    queue.scheduler = SimpleNamespace(
        enable_hisparse=False,
        enable_priority_scheduling=True,
        schedule_low_priority_values_first=False,
        running_batch=SimpleNamespace(reqs=[]),
        waiting_queue=[],
        last_batch=None,
        sliding_window_size=4,
        server_args=SimpleNamespace(
            disaggregation_decode_enable_radix_cache=True,
        ),
    )
    return queue


def req(priority, rid="req"):
    return SimpleNamespace(
        rid=rid,
        priority=priority,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
    )


def fill_full_pool(cache):
    allocator = cache.token_to_kv_pool_allocator
    remaining = allocator.available_size()
    assert allocator.alloc(remaining) is not None
    assert allocator.available_size() == 0


def test_pd_full_budget_and_recovery_follow_request_tier():
    cache = make_cache((FULL,), size=4)
    insert(cache, [1, 2])
    register(cache, [1, 2], "high", priority=1)
    fill_full_pool(cache)
    queue = make_queue(cache)
    low, high = req(0, "low"), req(1, "high")

    assert (
        queue._allocatable_token_budgets(
            count_retracted=False,
            req=low,
        )
        == 0
    )
    assert (
        queue._allocatable_token_budgets(
            count_retracted=False,
            req=high,
        )
        == 2
    )

    queue._ensure_session_kv_capacity(low, full_required=1)
    assert cache.token_to_kv_pool_allocator.available_size() == 0
    queue._ensure_session_kv_capacity(high, full_required=1)
    assert cache.token_to_kv_pool_allocator.available_size() >= 1


def test_pd_full_and_swa_budgets_and_recovery_are_independent():
    cache = make_cache((FULL, SWA), size=4)
    insert(cache, [1])
    register(cache, [1], "high", priority=1)
    allocator = cache.token_to_kv_pool_allocator
    assert allocator.full_attn_allocator.alloc(3) is not None
    assert allocator.swa_attn_allocator.alloc(3) is not None
    assert allocator.full_available_size() == 0
    assert allocator.swa_available_size() == 0
    queue = make_queue(cache)
    low, high = req(0, "low"), req(1, "high")

    assert queue._swa_aware_allocatable_token_budgets(
        count_retracted=False,
        req=low,
    ) == (0, 0)
    assert queue._swa_aware_allocatable_token_budgets(
        count_retracted=False,
        req=high,
    ) == (1, 1)

    with patch.object(
        cache,
        "evict_for_session",
        wraps=cache.evict_for_session,
    ) as evict:
        queue._ensure_session_kv_capacity(
            high,
            full_required=1,
            swa_required=1,
        )
    params = evict.call_args.args[0]
    assert params.num_tokens == 1
    assert params.swa_num_tokens == 1
    assert evict.call_args.kwargs == {"allow_low": True, "allow_high": True}
    assert allocator.full_available_size() == 1
    assert allocator.swa_available_size() == 1


def test_pd_mamba_admission_and_recovery_follow_request_tier():
    cache = make_cache((FULL, MAMBA), size=4)
    insert(cache, [1])
    register(cache, [1], "high", priority=1)
    mamba_allocator = cache.req_to_token_pool.mamba_allocator
    assert mamba_allocator.alloc(3) is not None
    assert mamba_allocator.schedulable_available_size() == 0
    queue = make_queue(cache)
    low, high = req(0, "low"), req(1, "high")

    assert queue._session_mamba_slots_needed(low) == 1
    assert queue._session_mamba_slots_available(low) == 0
    assert queue._session_mamba_slots_available(high) == 1
    assert not queue._ensure_session_mamba_capacity(low)
    assert queue._ensure_session_mamba_capacity(high)
    assert mamba_allocator.schedulable_available_size() == 1


def test_pd_mamba_budget_counts_only_missing_main_and_ping_pong_slots():
    cache = make_cache((FULL, MAMBA), size=8)
    queue = make_queue(cache)
    pool = cache.req_to_token_pool
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = False
    pool.mamba_ping_pong_track_buffer_size = 2
    request = req(1)

    assert queue._session_mamba_slots_needed(request) == 3
    request.mamba_pool_idx = torch.tensor(1)
    assert queue._session_mamba_slots_needed(request) == 2
    request.mamba_ping_pong_track_buffer = torch.tensor([2, 3])
    assert queue._session_mamba_slots_needed(request) == 0

    request.mamba_pool_idx = None
    request.mamba_ping_pong_track_buffer = None
    queue.token_to_kv_pool_allocator = SimpleNamespace(
        mamba_slot_full_token_cost=lambda: 5
    )
    assert queue._mamba_gap_budget_for_req(request) == 15


def test_pd_prealloc_and_prefix_match_use_request_scoped_eviction():
    cache = make_cache((FULL,), size=4)
    queue = make_queue(cache)
    high = req(1, "high")
    high.origin_input_ids = []
    queue._pre_alloc_impl = MagicMock(return_value=torch.tensor([1]))

    with patch.object(cache, "scoped_evict", wraps=cache.scoped_evict) as scoped:
        result = queue._pre_alloc(high)
    assert torch.equal(result, torch.tensor([1]))
    assert scoped.call_args.kwargs == {"allow_low": True, "allow_high": True}

    match_result = SimpleNamespace(last_device_node=cache.root_node.id)
    queue._build_decode_prefix_match = MagicMock(return_value="matched")
    with (
        patch.object(cache, "scoped_evict", wraps=cache.scoped_evict) as scoped,
        patch(
            "sglang.srt.disaggregation.decode.match_prefix_for_req",
            return_value=match_result,
        ),
    ):
        assert queue._match_prefix_and_lock(high) == "matched"
    assert scoped.call_args.kwargs == {"allow_low": True, "allow_high": True}
    cache.dec_lock_ref(cache.root_node.id)


def test_pd_resume_retracted_recomputes_request_scoped_budget():
    cache = make_cache((FULL,), size=4)
    queue = make_queue(cache)
    high = req(1, "high")
    high.is_retracted = True
    high.load_kv_cache = MagicMock()
    queue.retracted_queue = [high]
    queue.req_to_token_pool = SimpleNamespace(available_size=MagicMock(return_value=1))
    queue._allocatable_token_budgets = MagicMock(return_value=1)
    queue._prealloc_required_tokens = MagicMock(return_value=(1, 1))
    queue._pre_alloc = MagicMock()

    assert queue.resume_retracted_reqs() == [high]
    queue._allocatable_token_budgets.assert_called_once_with(
        count_retracted=False,
        req=high,
    )
    queue._pre_alloc.assert_called_once_with(high)
    high.load_kv_cache.assert_called_once()
    assert not high.is_retracted


def test_pd_resume_retracted_prioritizes_hp_that_can_reclaim_hp_cache():
    cache = make_cache((FULL,), size=4)
    queue = make_queue(cache)
    low, high = req(0, "low"), req(1, "high")
    for request in (low, high):
        request.is_retracted = True
        request.load_kv_cache = MagicMock()
    queue.retracted_queue = [low, high]
    queue.req_to_token_pool = SimpleNamespace(available_size=MagicMock(return_value=2))
    queue._allocatable_token_budgets = MagicMock(
        side_effect=lambda *, req, **_: 1 if req is high else 0
    )
    queue._prealloc_required_tokens = MagicMock(return_value=(1, 1))
    queue._pre_alloc = MagicMock()

    assert queue.resume_retracted_reqs() == [high]
    assert queue.retracted_queue == [low]
    queue._pre_alloc.assert_called_once_with(high)
    high.load_kv_cache.assert_called_once()
    low.load_kv_cache.assert_not_called()

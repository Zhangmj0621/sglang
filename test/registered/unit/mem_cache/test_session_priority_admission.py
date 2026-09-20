"""CPU admission tests for Unified cache component-specific priority budgets."""

from array import array
from types import SimpleNamespace
from unittest.mock import patch

from session_priority_test_utils import (
    FULL,
    MAMBA,
    SWA,
    attach_host_cache,
    backup,
    insert,
    make_cache,
    register,
)

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def request(cache, priority, token=100):
    req = Req(
        rid=str(token),
        origin_input_text="",
        origin_input_ids=array("q", [token]),
        sampling_params=SamplingParams(max_new_tokens=1),
        priority=priority,
    )
    req.init_next_round_input(cache)
    return req


def adder(cache):
    return PrefillAdder(
        page_size=1,
        tree_cache=cache,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        running_batch=ScheduleBatch(reqs=[]),
        new_token_ratio=1,
        rem_input_tokens=128,
        rem_chunk_tokens=None,
    )


def cache_with_high_mamba():
    cache = make_cache((FULL, MAMBA), size=8)
    insert(cache, [1])
    register(cache, [1], "hp", priority=1)
    cache.req_to_token_pool.mamba_allocator.alloc(7)
    return cache


def test_lp_cannot_use_high_mamba_slots_despite_free_full_kv():
    cache = cache_with_high_mamba()
    pending = adder(cache)
    req = request(cache, 0)
    assert cache.token_to_kv_pool_allocator.available_size() == 7
    assert pending.add_one_req(req, False, None) == AddReqResult.NO_TOKEN
    assert not pending.can_run_list
    assert cache.session_evictable_size(MAMBA, allow_high=True) == 1


def test_hp_can_budget_high_mamba_slots():
    cache = cache_with_high_mamba()
    pending = adder(cache)
    req = request(cache, 1)
    assert pending.add_one_req(req, False, None) == AddReqResult.CONTINUE
    assert pending.can_run_list == [req]


def test_multiple_lp_requests_reserve_mamba_slots_independently():
    cache = make_cache((FULL, MAMBA), size=16)
    cache.req_to_token_pool.mamba_allocator.alloc(15)
    pending = adder(cache)
    first, second = request(cache, 0, 101), request(cache, 0, 102)
    assert pending.add_one_req(first, False, None) == AddReqResult.CONTINUE
    assert pending.add_one_req(second, False, None) == AddReqResult.NO_TOKEN
    assert pending.can_run_list == [first]
    assert pending._session_mamba_reserved == 1


def test_mamba_group_prefetch_remains_available_to_session_admission():
    cache = make_cache((FULL, MAMBA), size=8)
    pending = adder(cache)
    req = request(cache, 0)
    mamba = cache.req_to_token_pool.mamba_allocator
    mamba.alloc_group_begin(8)
    try:
        assert mamba.schedulable_available_size() == 0
        assert mamba.group_available_size() == 8
        assert pending.add_one_req(req, False, None) == AddReqResult.CONTINUE
        assert pending.can_run_list == [req]
    finally:
        mamba.alloc_group_end()


def test_mamba_host_loadback_is_not_also_reserved_as_a_future_slot():
    cache = make_cache((FULL, MAMBA), size=16)
    attach_host_cache(cache, size=16)
    target = insert(cache, [1, 2, 3, 4])
    assert backup(cache, target) == 4
    evicted = cache.evict_for_session(
        EvictParams(num_tokens=4, mamba_num=1), allow_high=True
    )
    assert evicted.num_tokens_evicted == 4
    assert evicted.mamba_num_evicted == 1

    req = Req(
        rid="host-loadback",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3, 4, 100]),
        sampling_params=SamplingParams(max_new_tokens=1),
        priority=0,
    )
    req.init_next_round_input(cache)
    assert req.mamba_pool_idx is None
    assert req.mamba_host_hit_length > 0
    before = cache.req_to_token_pool.mamba_allocator.schedulable_available_size()

    pending = adder(cache)
    assert pending.add_one_req(req, False, None) == AddReqResult.CONTINUE
    assert req.mamba_pool_idx is not None
    # Load-back materializes both the tree checkpoint and its per-request CoW.
    assert cache.req_to_token_pool.mamba_allocator.schedulable_available_size() < before
    # prepare_load_back allocated the main state already. Only slots still
    # missing at batch allocation time belong in the pending reservation.
    assert pending._session_mamba_reserved == 0
    assert pending._session_mamba_slots_available() == (
        cache.req_to_token_pool.mamba_allocator.schedulable_available_size()
        + cache.session_evictable_size(MAMBA, allow_high=False)
    )


def test_mamba_ping_pong_slot_demand_matches_pool_allocation_policy():
    cache = make_cache((FULL, MAMBA), size=8)
    req = request(cache, 0)
    pool = cache.req_to_token_pool
    # Exercise the pool's configuration values with real allocator capacity;
    # no KV allocation or component eviction is replaced.
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = False
    pending = adder(cache)
    assert (
        pending._session_mamba_slots_needed(req)
        == 1 + pool.mamba_ping_pong_track_buffer_size
    )
    pool.enable_mamba_extra_buffer_lazy = True
    assert pending._session_mamba_slots_needed(req) == 2
    cache.req_to_token_pool.mamba_allocator.alloc(7)
    assert pending.add_one_req(req, False, None) == AddReqResult.NO_TOKEN


def test_swa_decode_caps_each_pool_to_actual_hp_demand():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    cache = make_cache((FULL, SWA), size=8)
    for token in [1, 2]:
        insert(cache, [token])
        register(cache, [token], f"hp{token}", priority=1)
    allocator = cache.token_to_kv_pool_allocator
    allocator.full_attn_allocator.alloc(5)  # 1 full slot available
    allocator.swa_attn_allocator.alloc(6)  # 0 SWA slots available
    batch = ScheduleBatch(
        reqs=[SimpleNamespace(priority=p, kv_committed_len=1) for p in [1, 0]],
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )
    with patch.object(
        cache, "evict_for_session", wraps=cache.evict_for_session
    ) as evict:
        assert not batch.check_decode_mem()
    assert len(evict.call_args_list) == 2
    high_call = evict.call_args_list[1]
    assert high_call.kwargs == {"allow_low": False, "allow_high": True}
    assert high_call.args[0].num_tokens == 1
    assert high_call.args[0].swa_num_tokens == 1
    assert allocator.full_available_size() == 2
    assert allocator.swa_available_size() == 1

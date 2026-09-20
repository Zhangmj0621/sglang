"""Real CPU regression coverage for priority-constrained allocation pressure."""

from array import array
from unittest.mock import patch

import pytest
import torch
from session_priority_test_utils import (
    FULL,
    MAMBA,
    SWA,
    attach_host_cache,
    backup,
    insert,
    make_cache,
    register,
    request,
)
from test_session_priority_admission import adder
from test_session_priority_pd import make_queue

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InitLoadBackParams
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def cow_case(priority=0, enabled=True):
    cache = make_cache((FULL, MAMBA), size=8, enable_session=enabled)
    nodes = [insert(cache, [token]) for token in (1, 3)]
    for token in (1, 3):
        register(cache, [token], f"hp-{token}", priority=1)
    assert cache.req_to_token_pool.mamba_allocator.alloc(6) is not None
    req = Req(
        rid="cow-pressure",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=SamplingParams(max_new_tokens=1),
        priority=priority,
    )
    return cache, req, nodes


def test_lp_cow_pressure_becomes_miss_and_admission_waits():
    cache, req, nodes = cow_case()
    with cache.scoped_evict(allow_high=False):
        req.init_next_round_input(cache)
    assert len(req.prefix_indices) == 0
    assert req.mamba_pool_idx is None
    assert req.mamba_cow_src_index is None
    assert req.mamba_branching_seqlen is None
    pending = adder(cache)
    assert pending.add_one_req(req, False, None) == AddReqResult.NO_TOKEN
    assert not pending.can_run_list
    for node in nodes:
        assert node.id in cache.tree_core._node_arena
        assert node.component_data[FULL].lock_ref == 0
        assert node.component_data[MAMBA].lock_ref == 0
    cache.sanity_check()


@pytest.mark.parametrize("enabled", (False, True))
def test_authorized_cow_can_reclaim_another_checkpoint(enabled):
    cache, req, nodes = cow_case(priority=1, enabled=enabled)
    with cache.scoped_evict(allow_high=True):
        req.init_next_round_input(cache)
    assert len(req.prefix_indices) == 1
    assert req.mamba_pool_idx is not None
    assert req.mamba_cow_src_index is not None
    assert nodes[0].id in cache.tree_core._node_arena
    assert nodes[1].id not in cache.tree_core._node_arena
    cache.sanity_check()


def test_pd_cow_pressure_is_a_miss_with_no_writable_slot_budget():
    cache, req, nodes = cow_case()
    queue = make_queue(cache)
    queue.scheduler.enable_decode_hicache = False
    matched = queue._match_prefix_and_lock(req)
    assert matched.l1_prefix_len == 0
    assert req.mamba_pool_idx is None
    assert queue._session_mamba_slots_available(req) == 0
    assert all(n.id in cache.tree_core._node_arena for n in nodes)
    cache.sanity_check()


def test_optional_unfinished_checkpoint_keeps_active_state_when_lp_cannot_allocate():
    cache = make_cache((FULL, MAMBA), size=8)
    protected = insert(cache, [1])
    register(cache, [1], "hp", priority=1)
    req = Req(
        rid="unfinished",
        origin_input_text="",
        origin_input_ids=array("q", [101]),
        sampling_params=SamplingParams(max_new_tokens=2),
        priority=0,
    )
    req.init_next_round_input(cache)
    mamba = cache.req_to_token_pool.mamba_allocator
    req.mamba_pool_idx = mamba.alloc(1)[0]
    original = req.mamba_pool_idx.clone()
    assert mamba.alloc(6) is not None
    cache.req_to_token_pool.alloc([req])
    indices = cache.token_to_kv_pool_allocator.alloc(1)
    cache.req_to_token_pool.write((req.req_pool_idx, slice(0, 1)), indices)
    req.kv_committed_len = 1
    req.set_extend_range(0, 1)
    cache.cache_unfinished_req(req)
    assert torch.equal(req.mamba_pool_idx, original)
    assert mamba.available_size() == 0
    assert protected.id in cache.tree_core._node_arena
    assert protected.component_data[MAMBA].session_high_ref == 1
    assert len(req.prefix_indices) == 1
    cache.sanity_check()


def test_load_back_prepare_pressure_returns_without_leaking_locks():
    cache = make_cache((FULL, MAMBA), size=8)
    attach_host_cache(cache, size=16)
    target = insert(cache, [1, 2])
    assert backup(cache, target) == 2
    assert (
        cache.evict_for_session(
            EvictParams(num_tokens=2), allow_high=True
        ).num_tokens_evicted
        == 2
    )
    protected = insert(cache, [3])
    register(cache, [3], "hp", priority=1)
    assert cache.req_to_token_pool.mamba_allocator.alloc(7) is not None
    req = request(cache, [1, 2], "lp", priority=0)
    indices, _ = cache.init_load_back(
        InitLoadBackParams(best_match_node=target.id, host_hit_length=2, req=req)
    )
    assert len(indices) == 0
    assert req.mamba_pool_idx is None
    assert not cache.ongoing_load_back
    assert protected.id in cache.tree_core._node_arena
    for ct in (FULL, MAMBA):
        assert target.component_data[ct].lock_ref == 0
        assert target.component_data[ct].host_lock_ref == 0
    cache.sanity_check()


@pytest.mark.parametrize("aux", (SWA, MAMBA))
@pytest.mark.parametrize("count", (16, 128))
def test_aux_host_small_eviction_visits_only_needed_candidates(aux, count):
    cache = make_cache((FULL, aux), size=count + 16)
    attach_host_cache(cache, size=count + 16)
    for token in range(count):
        node = insert(cache, [1000 + token])
        assert backup(cache, node) == 1
    cache.evict(EvictParams(num_tokens=count))
    lru = cache.tree_core.host_lru_lists[aux]
    comp = cache.components[aux]
    with patch.object(lru, "cursor_next", wraps=lru.cursor_next) as visit, patch.object(
        cache.session_refs,
        "idle_hp_candidates",
        side_effect=AssertionError("unreferenced eviction scanned sessions"),
    ), patch.object(comp, "session_tier", wraps=comp.session_tier) as tier:
        assert cache.evict_host(1, component_type=aux) == 1
        assert visit.call_count == 1
        assert tier.call_count < 10
    assert all(cursor.lru_prev[lru._pt] is None for cursor in lru.cursors)
    cache.sanity_check()


def test_finished_int8_checkpoint_pressure_frees_request_without_partial_cache():
    from unittest.mock import MagicMock

    from sglang.srt.managers.schedule_batch import FINISH_LENGTH

    cache = make_cache((FULL, MAMBA), size=8)
    pool = cache.req_to_token_pool
    req = Req(
        rid="int8-pressure",
        origin_input_text="",
        origin_input_ids=array("q", [101]),
        sampling_params=SamplingParams(max_new_tokens=1),
        priority=0,
    )
    req.init_next_round_input(cache)
    pool.alloc([req])
    indices = cache.token_to_kv_pool_allocator.alloc(1)
    pool.write((req.req_pool_idx, slice(0, 1)), indices)
    req.kv_committed_len = 1
    req.set_extend_range(0, 1)
    req.finished_reason = FINISH_LENGTH(1)
    checkpoint_pool = MagicMock()
    checkpoint_pool.alloc.return_value = None
    checkpoint_pool.store_from_active.side_effect = AssertionError(
        "no checkpoint was allocated"
    )
    pool.mamba_ckpt_pool = checkpoint_pool
    cache.cache_finished_req(req, kv_len_to_handle=1)
    checkpoint_pool.store_from_active.assert_not_called()
    assert req.mamba_pool_idx is None
    assert pool.mamba_allocator.available_size() == 8
    assert cache.total_size()[0] == 0
    cache.sanity_check()

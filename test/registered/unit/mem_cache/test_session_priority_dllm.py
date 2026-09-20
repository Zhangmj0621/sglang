"""CPU admission coverage for session-priority dLLM staging."""

from array import array

import torch
from session_priority_test_utils import FULL, SWA, insert, make_cache, register

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _dllm_config():
    return DllmConfig(
        "LowConfidence",
        {},
        block_size=2,
        mask_id=999,
        max_running_requests=4,
    )


def _protect_high_prefix(cache, length):
    token_ids = list(range(length))
    insert(cache, token_ids)
    register(cache, token_ids, "hp", priority=1)


def _make_staging_req(
    cache,
    config,
    rid,
    *,
    priority=0,
    retain_incomplete_block=False,
):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", [101, 102, 103, 104]),
        sampling_params=SamplingParams(max_new_tokens=2),
        priority=priority,
        dllm_config=config,
    )
    req.init_next_round_input(cache)

    retained_len = config.block_size if retain_incomplete_block else 0
    allocated = cache.token_to_kv_pool_allocator.alloc(config.block_size + retained_len)
    assert allocated is not None
    assert cache.req_to_token_pool.alloc([req]) is not None
    cache.req_to_token_pool.req_to_token[req.req_pool_idx, : len(allocated)] = allocated
    req.prefix_indices = allocated[: config.block_size]
    req.kv = ReqKvInfo(
        kv_allocated_len=len(allocated),
        swa_evicted_seqlen=0,
    )
    req.kv_committed_len = config.block_size

    if retain_incomplete_block:
        req.dllm_incomplete_ids = array("q", [701, 702])
    req.init_next_round_input()

    expected_phase = (
        DllmReqPhase.STAGING_DECODE
        if retain_incomplete_block
        else DllmReqPhase.STAGING_PREFILL
    )
    assert req.dllm_phase == expected_phase
    retained = allocated[config.block_size :]
    return req, retained


def _adder(cache, config):
    return PrefillAdder(
        page_size=1,
        tree_cache=cache,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        running_batch=ScheduleBatch(reqs=[]),
        new_token_ratio=1,
        rem_input_tokens=128,
        rem_chunk_tokens=None,
        dllm_config=config,
    )


def _allocate_admitted(cache, config, reqs):
    prefix_lens = [len(req.prefix_indices) for req in reqs]
    extend_lens = [req.extend_range.length for req in reqs]
    seq_lens_cpu = torch.tensor(
        [req.extend_range.end for req in reqs], dtype=torch.int64
    )
    batch = ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=cache.req_to_token_pool,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        tree_cache=cache,
        device="cpu",
        dllm_config=config,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        seq_lens=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=sum(extend_lens),
    )
    allow_high = any(cache.is_high_priority(req.priority) for req in reqs)
    with cache.scoped_evict(allow_low=True, allow_high=allow_high):
        return alloc_for_extend(batch)


def test_lp_fresh_staging_cannot_spend_high_priority_cache():
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", attention_backend="torch_native")
    )
    config = _dllm_config()
    cache = make_cache((FULL,), size=8)
    _protect_high_prefix(cache, 6)
    req, _ = _make_staging_req(cache, config, "lp-fresh")
    pending = _adder(cache, config)

    with pending._session_request_scope(req):
        assert pending.rem_total_tokens == 0
    assert pending.add_dllm_staging_req(req) == AddReqResult.NO_TOKEN
    assert pending.can_run_list == []
    assert cache.token_to_kv_pool_allocator.available_size() == 0


def test_hp_fresh_staging_can_reclaim_high_priority_cache():
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", attention_backend="torch_native")
    )
    config = _dllm_config()
    cache = make_cache((FULL,), size=8)
    _protect_high_prefix(cache, 6)
    req, _ = _make_staging_req(cache, config, "hp-fresh", priority=1)
    pending = _adder(cache, config)

    assert pending.add_dllm_staging_req(req) == AddReqResult.CONTINUE
    assert pending.can_run_list == [req]
    out_cache_loc, _, _ = _allocate_admitted(cache, config, [req])
    assert len(out_cache_loc) == config.block_size
    assert req.kv.kv_allocated_len == 2 * config.block_size


def test_lp_fresh_staging_needs_low_tier_capacity_in_each_swa_pool():
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", attention_backend="torch_native")
    )
    config = _dllm_config()
    cache = make_cache((FULL, SWA), size=8)
    _protect_high_prefix(cache, 4)
    req, _ = _make_staging_req(cache, config, "lp-swa")
    allocator = cache.token_to_kv_pool_allocator
    held_swa = allocator.swa_attn_allocator.alloc(2)
    assert held_swa is not None
    pending = _adder(cache, config)

    with pending._session_request_scope(req):
        assert pending.rem_total_tokens == 2
        assert pending.rem_swa_tokens == 0
    assert pending.add_dllm_staging_req(req) == AddReqResult.NO_TOKEN
    assert pending.can_run_list == []


def test_multiple_incomplete_blocks_reuse_owned_kv_at_zero_budget():
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", attention_backend="torch_native")
    )
    config = _dllm_config()
    cache = make_cache((FULL,), size=12)
    _protect_high_prefix(cache, 4)
    first, first_retained = _make_staging_req(
        cache, config, "reuse-1", retain_incomplete_block=True
    )
    second, second_retained = _make_staging_req(
        cache, config, "reuse-2", retain_incomplete_block=True
    )
    pending = _adder(cache, config)

    with pending._session_request_scope(first):
        assert pending.rem_total_tokens == 0
    assert pending.add_dllm_staging_req(first) == AddReqResult.CONTINUE
    assert pending.add_dllm_staging_req(second) == AddReqResult.CONTINUE
    assert pending.can_run_list == [first, second]
    # The retained blocks consume compute and future-output budgets, but no
    # extend allocation or per-allocation page overhead.
    assert pending.cur_rem_token_offset == 0
    assert pending.rem_total_token_offset == 4

    available_before = cache.token_to_kv_pool_allocator.available_size()
    out_cache_loc, _, _ = _allocate_admitted(cache, config, [first, second])
    assert (
        out_cache_loc.tolist() == torch.cat([first_retained, second_retained]).tolist()
    )
    assert cache.token_to_kv_pool_allocator.available_size() == available_before == 0


def test_reused_block_does_not_grant_a_fresh_request_high_tier_capacity():
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", attention_backend="torch_native")
    )
    config = _dllm_config()
    cache = make_cache((FULL,), size=12)
    _protect_high_prefix(cache, 6)
    reused, _ = _make_staging_req(cache, config, "reuse", retain_incomplete_block=True)
    fresh, _ = _make_staging_req(cache, config, "fresh")
    pending = _adder(cache, config)

    assert pending.add_dllm_staging_req(reused) == AddReqResult.CONTINUE
    assert pending.add_dllm_staging_req(fresh) == AddReqResult.NO_TOKEN
    assert pending.can_run_list == [reused]


def test_incomplete_metadata_without_retained_kv_is_not_zero_cost():
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", attention_backend="torch_native")
    )
    config = _dllm_config()
    cache = make_cache((FULL,), size=8)
    _protect_high_prefix(cache, 6)
    req, _ = _make_staging_req(cache, config, "missing-retained")
    req.dllm_incomplete_ids = array("q", [701, 702])
    req.init_next_round_input()
    pending = _adder(cache, config)

    assert req.kv.kv_allocated_len == len(req.prefix_indices)
    assert pending.add_dllm_staging_req(req) == AddReqResult.NO_TOKEN
    assert pending.can_run_list == []


def test_session_feature_off_keeps_legacy_dllm_fallback():
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", attention_backend="torch_native")
    )
    config = _dllm_config()
    cache = make_cache((FULL,), size=8, enable_session=False)
    req, _ = _make_staging_req(cache, config, "legacy")
    held = cache.token_to_kv_pool_allocator.alloc(6)
    assert held is not None
    pending = _adder(cache, config)

    assert pending.rem_total_tokens == 0
    assert pending._get_dllm_remain_tokens() == pending.rem_dllm_tokens
    assert pending.add_dllm_staging_req(req) == AddReqResult.CONTINUE
    assert pending.can_run_list == [req]
    assert req.extend_range.length == 4
    assert pending.rem_total_token_offset == 7

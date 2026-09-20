"""Small real CPU pools for Unified session cache behavior tests."""

from array import array
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=0, suite="base-a-test-cpu", disabled="helper module, no tests")

FULL = ComponentType.FULL
SWA = ComponentType.SWA
MAMBA = ComponentType.MAMBA
COMPONENT_CASES = ((FULL,), (FULL, SWA), (FULL, MAMBA))


def make_cache(
    components=(FULL,),
    *,
    size=128,
    enable_session=True,
    enable_priority_scheduling=True,
    **kwargs,
):
    dtype = torch.float32
    has_mamba = MAMBA in components
    has_swa = SWA in components
    if has_mamba:
        args = ServerArgs(model_path="dummy", page_size=1)
        args._mamba_cache_chunk_size = 64
        set_global_server_args_for_scheduler(args)
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=16,
            n_groups=1,
            num_heads=2,
            head_dim=8,
            state_size=4,
            conv_kernel=4,
        )
        req_pool = HybridReqToTokenPool(
            size=8,
            mamba_size=size,
            mamba_spec_state_size=8,
            max_context_len=256,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=[1]),
            mamba_layer_ids=[1],
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=None,
        )
    else:
        req_pool = ReqToTokenPool(
            size=8,
            max_context_len=256,
            device="cpu",
            enable_memory_saver=False,
        )
    if has_swa:
        kv_pool = SWAKVPool(
            size=size,
            size_swa=size,
            page_size=1,
            dtype=dtype,
            head_num=2,
            head_dim=8,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            device="cpu",
        )
        allocator = SWATokenToKVPoolAllocator(
            size=size,
            size_swa=size,
            page_size=1,
            dtype=dtype,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
    else:
        if has_mamba:
            kv_pool = HybridLinearKVPool(
                size=size,
                dtype=dtype,
                page_size=1,
                head_num=2,
                head_dim=8,
                full_attention_layer_ids=[0],
                device="cpu",
                enable_memory_saver=False,
                mamba_pool=req_pool.mamba_pool,
            )
        else:
            kv_pool = MHATokenToKVPool(
                size=size,
                page_size=1,
                dtype=dtype,
                head_num=2,
                head_dim=8,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
            )
        allocator = TokenToKVPoolAllocator(
            size=size,
            dtype=dtype,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            eviction_policy="lru",
            enable_session_radix_cache=enable_session,
            enable_priority_scheduling=enable_priority_scheduling,
            tree_components=components,
            sliding_window_size=4 if has_swa else None,
            **kwargs,
        )
    )


def insert(cache, token_ids, *, priority=0):
    allocator = cache.token_to_kv_pool_allocator
    if SWA in cache.components:
        values = allocator.full_attn_allocator.alloc(len(token_ids))
        swa_values = allocator.swa_attn_allocator.alloc(len(token_ids))
        assert values is not None and swa_values is not None
        allocator.full_to_swa_index_mapping[values] = swa_values
    else:
        values = allocator.alloc(len(token_ids))
    assert values is not None
    params = InsertParams(
        key=RadixKey(array("q", token_ids)),
        value=values.to(torch.int64),
        priority=priority,
    )
    if MAMBA in cache.components:
        params.mamba_value = cache.req_to_token_pool.mamba_allocator.alloc(1)
        assert params.mamba_value is not None
    result = cache.insert(params)
    return cache.tree_core.node_by_id(result.last_device_node)


def request(cache, token_ids, session_id, *, priority=0, generation=None):
    if generation is None:
        generation = cache.ensure_session_generation(session_id)
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", token_ids))))
    return SimpleNamespace(
        session_id=session_id,
        session_generation=generation,
        session=None,
        priority=priority,
        last_node=match.last_device_node,
        origin_input_ids=array("q", token_ids),
        output_ids=array("q"),
        kv_committed_len=len(token_ids),
        extra_key=None,
        mamba_pool_idx=None,
        swa_host_hit_length=match.swa_host_hit_length,
        mamba_host_hit_length=match.mamba_host_hit_length,
    )


def register(cache, token_ids, session_id, *, priority=0, generation=None):
    req = request(
        cache, token_ids, session_id, priority=priority, generation=generation
    )
    accepted, message = cache.session_refs.begin_request(req)
    assert accepted, message
    cache.session_refs.register_session_ref(req)
    cache.session_refs.end_request(req)
    return req


def match_len(cache, token_ids):
    return len(
        cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", token_ids)),
            )
        ).device_indices
    )


class CPUTransferController:
    """Transfer boundary fake; real cache code owns eviction and commit/ACKs."""

    def __init__(self, cache, size):
        self.cache = cache
        self.pools = {}
        names = {FULL: PoolName.KV, SWA: PoolName.SWA, MAMBA: PoolName.MAMBA}
        attrs = {
            FULL: "_full_kv_pool_host",
            SWA: "_swa_kv_pool_host",
            MAMBA: "_mamba_pool_host",
        }
        for ct, component in cache.components.items():
            pool = TokenToKVPoolAllocator(
                size=size,
                dtype=torch.float32,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
            self.pools[names[ct]] = pool
            setattr(component, attrs[ct], pool)
        self.mem_pool_host = self.pools[PoolName.KV]
        self.write_policy = "write_through"
        self.writes = []
        self.ack_load_queue = []
        self.fail_load = False

    def write(self, device_indices, node_id, extra_pools=None):
        allocated = []
        for name, count in [(PoolName.KV, len(device_indices))] + [
            (xfer.name, len(xfer.device_indices)) for xfer in extra_pools or ()
        ]:
            indices = self.pools[name].alloc(count)
            if indices is None:
                for old_name, old_indices in allocated:
                    self.pools[old_name].free(old_indices)
                return None
            allocated.append((name, indices))
        for xfer, (_, indices) in zip(extra_pools or (), allocated[1:]):
            xfer.host_indices = indices
        self.writes.append(node_id)
        return allocated[0][1]

    def append_host_mem_release(self, *, extra_pools):
        for xfer in extra_pools:
            self.pools[xfer.name].free(xfer.host_indices)

    def load(self, host_indices, node_id, extra_pools=None):
        if self.fail_load:
            return None
        allocator = self.cache.token_to_kv_pool_allocator
        full_allocator = (
            allocator.full_attn_allocator if SWA in self.cache.components else allocator
        )
        device_indices = full_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        allocated = [(full_allocator, device_indices)]
        for xfer in extra_pools or ():
            if xfer.device_indices is not None:
                continue
            pool = (
                allocator.swa_attn_allocator
                if xfer.name == PoolName.SWA
                else self.cache.req_to_token_pool.mamba_allocator
            )
            values = pool.alloc(len(xfer.host_indices))
            if values is None:
                for allocated_pool, indices in allocated:
                    allocated_pool.free(indices)
                return None
            xfer.device_indices = values
            allocated.append((pool, values))
        # CPU copies finish immediately; this event only models transfer ACKs.
        self.ack_load_queue.append(
            SimpleNamespace(
                finish_event=SimpleNamespace(
                    query=lambda: True, synchronize=lambda: None
                ),
                node_ids=[node_id],
                num_tokens=len(host_indices),
                timing_enabled=False,
            )
        )
        return device_indices


def attach_host_cache(cache, *, size=128):
    controller = CPUTransferController(cache, size)
    cache.cache_controller = controller
    cache.tree_core.set_hicache_enabled()
    cache.tree_core.has_swa_host_pool = SWA in cache.components
    cache.write_through_threshold = 1 << 30
    cache.load_back_threshold = 0
    return controller


def backup(cache, node, *, write_back=False, finish=True):
    action = cache.tree_core._build_backup_kv_action(node, write_back)
    written = cache._execute_and_commit_kv_backup(action, write_back)
    if finish:
        for ack_id in tuple(cache.ongoing_write_through):
            cache._finish_write_through_ack(ack_id)
    return written

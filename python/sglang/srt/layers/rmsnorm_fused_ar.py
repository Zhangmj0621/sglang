"""Single-kernel fused all-reduce + residual-add + RMSNorm via mega_ops.

The mega_ops.rmsnorm_fused_ar kernel performs reduce-scatter (multimem
ld_reduce), residual add, RMSNorm, and all-gather (multimem st) in one
launch. Each rank computes only its own token shard; the normalized output
is broadcast to every rank by the kernel, but the RESIDUAL is only fresh in
this rank's shard. The shard state is tracked with a tensor attribute and
completed (all-gathered) at every non-fused consumer (see
ensure_full_residual).

Enabled by --enable-rmsnorm-fused-ar. The flag is a hard commitment: any
ineligible input or missing resource raises instead of falling back.

sglang-internal imports are kept lazy so this module stays importable in
lightweight/unit-test contexts.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

# Tensor attribute marking a residual whose data is fresh only in
# [start, end) (this rank's token shard). Same tagging idiom as
# `_sglang_needs_allreduce_fusion` in communicator.py.
_SHARD_ATTR = "_mega_residual_shard"

# The kernel's barrier supports at most this many blocks (kMaxBarrierBlocks).
_MAX_BARRIER_BLOCKS = 256

_RESOURCES: dict[str, _FusedArResources] = {}


@functools.lru_cache(maxsize=1)
def _max_ctas() -> int:
    """Grid size for the fused kernel, from SGLANG_RMSNORM_FUSED_AR_MAX_CTAS.

    Read once (init-static: env vars are frozen for the process lifetime);
    capped further by the shard's token count at call time. Values outside
    [1, kMaxBarrierBlocks] raise per the flag's crash-on-ineligible contract.
    """
    from sglang.srt.environ import envs

    max_ctas = envs.SGLANG_RMSNORM_FUSED_AR_MAX_CTAS.get()
    if not 1 <= max_ctas <= _MAX_BARRIER_BLOCKS:
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: SGLANG_RMSNORM_FUSED_AR_MAX_CTAS must "
            f"be in [1, {_MAX_BARRIER_BLOCKS}] (kMaxBarrierBlocks), got "
            f"{max_ctas}."
        )
    return max_ctas


@functools.lru_cache(maxsize=1)
def _needs_outbound_copy() -> bool:
    """Whether the fused output must be copied out of the symm buffer.

    The buffer is overwritten by the next layer's producer GEMM, so a
    consumer that holds the normalized output past the immediately
    following GEMM needs a private copy. Today that is hidden-state
    capture: both the boolean flag and any non-None mode capture hidden
    states (see get_server_return_hidden_states_mode), and `last` keeps a
    reference to the final layer's output. Read once — the fields are
    server-level config, frozen for the process lifetime.
    """
    from sglang.srt.runtime_context import get_exec

    features = get_exec().features
    return bool(features.enable_return_hidden_states) or (
        features.return_hidden_states_mode is not None
    )


class _FusedArResources(msgspec.Struct):
    buffer: torch.Tensor  # the group's TorchSymmMemCommunicator buffer
    multicast_ptr: int
    flags_ptrs_dev: int
    state_ptr: int
    rank: int
    world_size: int
    max_size: int  # symm buffer capacity in bytes
    refs: tuple  # keeps flags/state/handles alive


def rmsnorm_fused_ar_enabled() -> bool:
    from sglang.srt.runtime_context import get_exec

    return get_exec().comm.enable_rmsnorm_fused_ar


def _token_shard(num_tokens: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Split num_tokens as evenly as possible; the remainder goes to the
    lowest ranks. Pure function of shapes -> CUDA-graph replayable."""
    base, rem = divmod(num_tokens, world_size)
    start = rank * base + min(rank, rem)
    return start, start + base + (1 if rank < rem else 0)


def _select_group(use_attn_tp_group: bool) -> GroupCoordinator:
    from sglang.srt.distributed import (
        get_attn_tp_group,
        get_moe_ep_group,
        get_moe_tp_group,
    )
    from sglang.srt.runtime_context import get_parallel

    if use_attn_tp_group:
        return get_attn_tp_group()
    if get_parallel().moe_ep_size > 1:
        return get_moe_ep_group()
    return get_moe_tp_group()


def _get_resources(group: GroupCoordinator) -> _FusedArResources:
    comm = group.torch_symm_mem_comm
    if comm is None or comm.disabled:
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: group has no usable torch symm-mem "
            "communicator (need --enable-torch-symm-mem and a supported "
            "device/world-size; for MoE domains the moe group must also "
            "construct one)."
        )
    key = comm.group.group_name
    res = _RESOURCES.get(key)
    if res is not None:
        return res
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: resources for group "
            f"'{key}' were not built before CUDA-graph capture. An eager "
            "warmup forward must run first (do not skip server warmup)."
        )

    import mega_ops
    import torch.distributed._symmetric_memory as torch_symm_mem

    if not mega_ops.is_available():
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: mega_ops unavailable at runtime."
        )
    device = comm.buffer.device
    hdl = torch_symm_mem.rendezvous(comm.buffer, key)  # idempotent
    if hdl.multicast_ptr == 0:
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: multicast is not supported on this "
            "topology (multicast_ptr == 0)."
        )
    flags = torch_symm_mem.empty(
        mega_ops.flags_numel(comm.world_size), device=device, dtype=torch.uint32
    )
    flags.zero_()
    hflags = torch_symm_mem.rendezvous(flags, key)
    # Device-side barrier so no peer's first fused kernel can write an epoch
    # into our flags while our zero_ is still pending (same rationale as the
    # inkling AR resource build).
    hflags.barrier()
    state = torch.zeros(mega_ops.STATE_SIZE, device=device, dtype=torch.uint32)
    res = _FusedArResources(
        buffer=comm.buffer,
        multicast_ptr=hdl.multicast_ptr,
        flags_ptrs_dev=hflags.buffer_ptrs_dev,
        state_ptr=state.data_ptr(),
        rank=hdl.rank,
        world_size=comm.world_size,
        max_size=comm.max_size,
        refs=(flags, state, hdl, hflags),
    )
    _RESOURCES[key] = res
    logger.info(
        "rmsnorm-fused-ar resources ready for group '%s' (world=%d)",
        key,
        comm.world_size,
    )
    return res


def _check_eligible(x: torch.Tensor, res: _FusedArResources) -> None:
    if x.dim() != 2:
        raise RuntimeError(
            f"--enable-rmsnorm-fused-ar: expected 2D input, got {x.dim()}D."
        )
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"--enable-rmsnorm-fused-ar: bf16 only, got {x.dtype}.")
    if x.shape[-1] % 8 != 0:
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: hidden_size must be a multiple of 8, "
            f"got {x.shape[-1]}."
        )
    if not x.is_contiguous():
        raise RuntimeError("--enable-rmsnorm-fused-ar: input must be contiguous.")
    payload = x.numel() * x.element_size()
    if payload > res.max_size:
        raise RuntimeError(
            f"--enable-rmsnorm-fused-ar: payload {payload} bytes exceeds the "
            f"symm buffer capacity {res.max_size}. Enlarge the torch symm-mem "
            "buffer for the maximum prefill payload."
        )


def _prepare_residual(
    residual: torch.Tensor, res: _FusedArResources, group: GroupCoordinator
) -> torch.Tensor:
    """Return a residual whose shard state matches this call's group.

    Same group ⇒ same (num_tokens, rank, world) ⇒ same window, pass through.
    Different group ⇒ complete on the MARKED group first (group identity is
    identical on every rank of the group, so this decision is rank-uniform;
    comparing shard windows would NOT be — windows from different world
    sizes diverge per rank and would desync the collective)."""
    marker = getattr(residual, _SHARD_ATTR, None)
    if marker is None:
        return residual
    _start, _end, marked_group = marker
    if marked_group is group:
        return residual
    return ensure_full_residual(residual)


def is_fused_ar_buffer_view(tensor: Optional[torch.Tensor]) -> bool:
    """True when `tensor` is a zero-copy view of a fused-AR symm buffer.

    Callers that park a fused-AR output across a boundary the buffer's
    lifetime does not cover (e.g. split-prefill stashing hidden_states on
    the ForwardBatch between calls) must clone it first: the next call's
    producer GEMM direct-writes the same buffer.
    """
    if tensor is None or not _RESOURCES:
        return False
    ptr = tensor.data_ptr()
    return any(res.buffer.data_ptr() == ptr for res in _RESOURCES.values())


def get_fused_ar_staging_view(
    *, num_tokens: int, hidden: int, use_attn_tp_group: bool = True
) -> Optional[torch.Tensor]:
    """Return a [num_tokens, hidden] view of the fused-AR symm buffer for a
    producer GEMM to write into, or None when the fused path is off/not ready.

    None (instead of raise) because the caller decides between direct-write
    and the plain path BEFORE the GEMM; returning None keeps that call site
    branch-free-safe during warmup (resources not built yet) and under
    capture of the very first forward."""
    if not rmsnorm_fused_ar_enabled():
        return None
    group = _select_group(use_attn_tp_group)
    comm = group.torch_symm_mem_comm
    if comm is None or comm.disabled:
        return None
    key = comm.group.group_name
    res = _RESOURCES.get(key)
    if res is None:
        return None  # first eager call will build; that call still does copy_
    n = num_tokens * hidden
    if n * 2 > res.max_size:
        return None
    return res.buffer[:n].view(num_tokens, hidden)


def rmsnorm_fused_ar_forward(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    use_attn_tp_group: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused AR+add+RMSNorm. Returns (normalized_full, residual_sharded).

    Collective: every rank of the group must call this for the same layer
    (guaranteed by SPMD model execution)."""
    import mega_ops

    group = _select_group(use_attn_tp_group)
    res = _get_resources(group)
    _check_eligible(x, res)
    if not residual.is_contiguous():
        raise RuntimeError("--enable-rmsnorm-fused-ar: residual must be contiguous.")
    if residual.shape != x.shape:
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: residual shape "
            f"{tuple(residual.shape)} != input shape {tuple(x.shape)}."
        )

    num_tokens, hidden = x.shape
    if num_tokens == 0:
        # Uniform across the group (same batch), so every rank skips the
        # kernel together — no barrier desync. A marker cannot exist on an
        # empty residual's producing chain, but clear defensively.
        if getattr(residual, _SHARD_ATTR, None) is not None:
            delattr(residual, _SHARD_ATTR)
        return x, residual

    residual = _prepare_residual(residual, res, group)
    start, end = _token_shard(num_tokens, res.rank, res.world_size)

    buf = res.buffer[: x.numel()].view(num_tokens, hidden)
    if x.data_ptr() == buf.data_ptr():
        # Producer GEMM already wrote into the staging buffer (see
        # get_fused_ar_staging_view); pointer equality suffices because the
        # direct-write path passes back the very same view.
        pass
    else:
        buf.copy_(x)  # staging copy: this rank's partial sums, all tokens
    mega_ops.rmsnorm_fused_ar(
        input=buf[start:end],
        residual=residual[start:end],
        weight=weight,
        mcptr=res.multicast_ptr + start * hidden * x.element_size(),
        flags_ptrs=res.flags_ptrs_dev,
        state_ptr=res.state_ptr,
        rank=res.rank,
        world_size=res.world_size,
        max_ctas=min(_max_ctas(), max(end - start, 1)),
        eps=eps,
    )
    if x.data_ptr() == buf.data_ptr() and not _needs_outbound_copy():
        # Zero-copy: the normalized result the kernel just wrote stays
        # resident in the symm buffer. Safe because the only consumer (the
        # next GEMM) reads it before the next producer direct-write
        # overwrites it — same-stream program order.
        out = buf
    else:
        out = torch.empty_like(x)
        out.copy_(buf)  # kernel's multimem.st already all-gathered the result
    setattr(residual, _SHARD_ATTR, (start, end, group))
    return out, residual


def ensure_full_residual(
    residual: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """All-gather a sharded residual back to full and clear the marker.

    No-op (one attribute check) for unmarked tensors. Capture-safe: static
    shapes, dense copies, and one all_gather_into_tensor."""
    if residual is None:
        return None
    marker = getattr(residual, _SHARD_ATTR, None)
    if marker is None:
        return residual

    import torch.distributed as dist

    start, end, group = marker
    num_tokens, hidden = residual.shape
    world = group.world_size
    per = (num_tokens + world - 1) // world  # padded per-rank slot

    padded_in = residual.new_zeros(per, hidden)
    if end > start:
        padded_in[: end - start].copy_(residual[start:end])
    padded_out = residual.new_empty(world * per, hidden)
    dist.all_gather_into_tensor(padded_out, padded_in, group=group.device_group)
    for r in range(world):
        r_start, r_end = _token_shard(num_tokens, r, world)
        if r_end > r_start:
            residual[r_start:r_end].copy_(
                padded_out[r * per : r * per + (r_end - r_start)]
            )
    delattr(residual, _SHARD_ATTR)
    return residual

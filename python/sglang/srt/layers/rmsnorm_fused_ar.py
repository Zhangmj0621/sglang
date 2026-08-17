"""Single-kernel fused all-reduce + residual-add + RMSNorm via mega_ops."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch

from sglang.srt.distributed import get_tp_group

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

# The kernel's barrier supports at most this many blocks (kMaxBarrierBlocks).
_MAX_BARRIER_BLOCKS = 256

# Per-group kernel workspace (symm staging buffer + multicast/flag/state
# pointers). Built lazily on the first eager forward.
_rmsnorm_fused_ar_workspaces: dict[str, _RmsnormFusedArWorkspace] = {}
_rmsnorm_fused_ar_unavailable: Optional[bool] = None


def is_rmsnorm_fused_ar_unavailable() -> bool:
    global _rmsnorm_fused_ar_unavailable
    if _rmsnorm_fused_ar_unavailable is None:
        try:
            import mega_ops

            usable = mega_ops.is_available()
        except ImportError:
            usable = False
        if usable:
            comm = get_tp_group().torch_symm_mem_comm
            usable = comm is not None and not comm.disabled
        _rmsnorm_fused_ar_unavailable = not usable
    return _rmsnorm_fused_ar_unavailable


@functools.lru_cache(maxsize=1)
def _max_ctas() -> int:
    """Grid size for the fused kernel, from SGLANG_RMSNORM_FUSED_AR_MAX_CTAS."""
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
    # if enable return hidden states, copy is necessity for gemm output to symm mem.
    from sglang.srt.runtime_context import get_exec

    features = get_exec().features
    return bool(features.enable_return_hidden_states) or (
        features.return_hidden_states_mode is not None
    )


class _RmsnormFusedArWorkspace(msgspec.Struct):
    buffer: torch.Tensor  # symm staging buffer; producer GEMMs write here
    multicast_ptr: int  # buffer's multicast address (the kernel's ld/st target)
    flags_ptrs_dev: int  # device array of every rank's barrier-flag pointer
    state_ptr: int  # kernel-private state (barrier epoch counters)
    rank: int
    world_size: int
    max_size: int  # buffer capacity in bytes
    refs: tuple  # owns flags/state tensors + handles so they outlive this struct


def rmsnorm_fused_ar_enabled() -> bool:
    from sglang.srt.runtime_context import get_exec

    return get_exec().comm.enable_rmsnorm_fused_ar


def rmsnorm_fused_ar_ready() -> bool:
    if is_rmsnorm_fused_ar_unavailable():
        return False
    if torch.cuda.is_current_stream_capturing():
        key = get_tp_group().torch_symm_mem_comm.group.group_name
        return key in _rmsnorm_fused_ar_workspaces
    return True


def _token_shard(num_tokens: int, rank: int, world_size: int) -> Tuple[int, int]:
    base, rem = divmod(num_tokens, world_size)
    start = rank * base + min(rank, rem)
    return start, start + base + (1 if rank < rem else 0)


def _get_workspace(group: GroupCoordinator) -> _RmsnormFusedArWorkspace:
    comm = group.torch_symm_mem_comm
    if comm is None or comm.disabled:
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: group has no usable torch symm-mem "
            "communicator (need --enable-torch-symm-mem and a supported "
            "device/world-size; for MoE domains the moe group must also "
            "construct one)."
        )
    key = comm.group.group_name
    workspace = _rmsnorm_fused_ar_workspaces.get(key)
    if workspace is not None:
        return workspace
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: the workspace for group "
            f"'{key}' was not built before CUDA-graph capture. An eager "
            "warmup forward must run first (do not skip server warmup)."
        )

    import mega_ops
    import torch.distributed._symmetric_memory as torch_symm_mem

    if not mega_ops.is_available():
        raise RuntimeError(
            "--enable-rmsnorm-fused-ar: mega_ops unavailable at runtime."
        )
    from sglang.srt.layers.communicator import RMSNORM_FUSED_AR_MAX_BATCH_SIZE
    from sglang.srt.runtime_context import process_model_config

    device = comm.device
    # Create dedicated staging buffer
    staging = torch_symm_mem.empty(
        RMSNORM_FUSED_AR_MAX_BATCH_SIZE * process_model_config().hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    hdl = torch_symm_mem.rendezvous(staging, key)
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
    hflags.barrier()
    state = torch.zeros(mega_ops.STATE_SIZE, device=device, dtype=torch.uint32)
    workspace = _RmsnormFusedArWorkspace(
        buffer=staging,
        multicast_ptr=hdl.multicast_ptr,
        flags_ptrs_dev=hflags.buffer_ptrs_dev,
        state_ptr=state.data_ptr(),
        rank=hdl.rank,
        world_size=comm.world_size,
        max_size=staging.numel() * staging.element_size(),
        refs=(flags, state, hdl, hflags),
    )
    _rmsnorm_fused_ar_workspaces[key] = workspace
    logger.info(
        "rmsnorm-fused-ar workspace ready for group '%s' (world=%d)",
        key,
        comm.world_size,
    )
    return workspace


def _is_eligible(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    workspace: _RmsnormFusedArWorkspace,
) -> bool:
    payload = x.numel() * x.element_size()
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and residual.dtype == x.dtype
        and weight.dtype == x.dtype
        and x.shape[-1] > 0
        and x.shape[-1] % 8 == 0
        and residual.shape == x.shape
        and weight.numel() == x.shape[-1]
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and payload <= workspace.max_size
    )


def is_fused_ar_buffer_view(tensor: Optional[torch.Tensor]) -> bool:
    if tensor is None or not _rmsnorm_fused_ar_workspaces:
        return False
    ptr = tensor.data_ptr()
    return any(
        w.buffer.data_ptr() == ptr for w in _rmsnorm_fused_ar_workspaces.values()
    )


def get_fused_ar_staging_view(
    *,
    num_tokens: int,
    hidden: int,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    if (
        not rmsnorm_fused_ar_enabled()
        or (dtype is not None and dtype != torch.bfloat16)
        or hidden <= 0
        or hidden % 8 != 0
        or is_rmsnorm_fused_ar_unavailable()
    ):
        return None
    workspace = _rmsnorm_fused_ar_workspaces.get(
        get_tp_group().torch_symm_mem_comm.group.group_name
    )
    if workspace is None:
        return None
    n = num_tokens * hidden
    if n * 2 > workspace.max_size:
        return None
    return workspace.buffer[:n].view(num_tokens, hidden)


def rmsnorm_fused_ar_forward(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused AR+add+RMSNorm. Returns (normalized_full, residual_sharded)."""
    if is_rmsnorm_fused_ar_unavailable():
        return None

    import mega_ops

    group = get_tp_group()
    try:
        workspace = _get_workspace(group)
    except RuntimeError as exc:
        logger.debug("rmsnorm-fused-ar workspace unavailable: %s", exc)
        return None
    if not _is_eligible(x, residual, weight, workspace):
        return None
    if post_residual_addition is not None and (
        post_residual_addition.shape != residual.shape
        or post_residual_addition.dtype != residual.dtype
        or not post_residual_addition.is_contiguous()
    ):
        return None

    num_tokens, hidden = x.shape
    if num_tokens == 0:
        if getattr(residual, "_mega_residual_shard", None) is not None:
            delattr(residual, "_mega_residual_shard")
        return x, residual

    start, end = _token_shard(num_tokens, workspace.rank, workspace.world_size)
    if post_residual_addition is not None and end > start:
        residual[start:end].add_(post_residual_addition[start:end])

    buf = workspace.buffer[: x.numel()].view(num_tokens, hidden)
    if x.data_ptr() == buf.data_ptr():
        pass
    else:
        buf.copy_(x)
    mega_ops.rmsnorm_fused_ar(
        input=buf[start:end],
        residual=residual[start:end],
        weight=weight,
        mcptr=workspace.multicast_ptr + start * hidden * x.element_size(),
        flags_ptrs=workspace.flags_ptrs_dev,
        state_ptr=workspace.state_ptr,
        rank=workspace.rank,
        world_size=workspace.world_size,
        max_ctas=min(_max_ctas(), max(end - start, 1)),
        eps=eps,
    )
    if x.data_ptr() == buf.data_ptr() and not _needs_outbound_copy():
        out = buf
    else:
        out = torch.empty_like(x)
        out.copy_(buf)
    # Mark the residual as fresh only in [start, end) -- this rank's shard.
    # Need to be gathered before the last layer attention
    setattr(residual, "_mega_residual_shard", (start, end, group))
    return out, residual


def ensure_full_residual(
    residual: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if residual is None:
        return None
    marker = getattr(residual, "_mega_residual_shard", None)
    if marker is None:
        return residual

    import torch.distributed as dist

    start, end, group = marker
    num_tokens, hidden = residual.shape
    world = group.world_size
    per = (num_tokens + world - 1) // world

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
    delattr(residual, "_mega_residual_shard")
    return residual

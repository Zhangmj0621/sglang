"""Symmetric-memory workspace shared by the two mega_ops fused-AR features.

``--enable-rmsnorm-fused-ar`` (standalone fused AR + residual + RMSNorm) and
``--enable-gemm-ar-rmsnorm-fused`` (the same collective fused INTO the GEMM
epilogue) both need a multicast-bound symmetric buffer and one set of
``sglang::inkling_ar`` barrier resources. The two flags are independent -- either
can be enabled alone -- so neither feature module can own the allocation: this
module does, and creates it on whichever feature asks first.

The barrier layout is identical for both kernels (``comm_barrier.cuh``:
``kLeaderStateWords=8``, ``kMaxBarrierBlocks=256``), so one set is shared.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

# Per-group workspace, keyed by symm-mem group name.
_workspaces: dict[str, MegaSymmWorkspace] = {}


class MegaSymmWorkspace(msgspec.Struct):
    buffer: torch.Tensor  # symm data buffer; GEMMs write partials here
    multicast_ptr: int  # buffer's multicast address (the kernels' ld/st target)
    flags_ptrs_dev: int  # device array of every rank's barrier-flag pointer
    state_ptr: int  # barrier epoch counters (device-local)
    rank: int
    world_size: int
    max_size: int  # buffer capacity in bytes
    gemm_op: Optional[Any]  # mega_ops.GemmRSNormAG, or None if that flag is off
    refs: tuple  # owns flags/state/handles so they outlive this struct


def workspace_data_numel() -> int:
    """Element count of the shared data buffer.

    The max of what either feature can ask for, so one allocation serves both:
    rmsnorm-fused-ar admits up to RMSNORM_FUSED_AR_MAX_BATCH_SIZE tokens, and
    the GEMM-fused path is sized for GEMM_AR_RMSNORM_FUSED_MAX_M rows. Both are
    token counts over the same hidden size.
    """
    from sglang.srt.layers.communicator import (
        GEMM_AR_RMSNORM_FUSED_MAX_M,
        RMSNORM_FUSED_AR_MAX_BATCH_SIZE,
    )
    from sglang.srt.runtime_context import process_model_config

    max_tokens = max(RMSNORM_FUSED_AR_MAX_BATCH_SIZE, GEMM_AR_RMSNORM_FUSED_MAX_M)
    return max_tokens * process_model_config().hidden_size


def get_workspace(*, group: GroupCoordinator) -> MegaSymmWorkspace:
    """The group's workspace, created on first call.

    Raises RuntimeError rather than returning None: every caller has already
    decided it wants the fused path, so a missing prerequisite is a
    configuration error worth reporting, not a silent fallback.
    """
    comm = group.torch_symm_mem_comm
    if comm is None or comm.disabled:
        raise RuntimeError(
            "mega fused-AR: group has no usable torch symm-mem communicator "
            "(need --enable-torch-symm-mem and a supported device/world-size)."
        )
    key = comm.group.group_name
    workspace = _workspaces.get(key)
    if workspace is not None:
        return workspace
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"mega fused-AR: the workspace for group '{key}' was not built "
            "before CUDA-graph capture. An eager warmup forward must run first "
            "(do not skip server warmup)."
        )

    import mega_ops
    import torch.distributed._symmetric_memory as torch_symm_mem

    if not mega_ops.is_available():
        raise RuntimeError("mega fused-AR: mega_ops unavailable at runtime.")

    device = comm.device
    buffer = torch_symm_mem.empty(
        workspace_data_numel(), device=device, dtype=torch.bfloat16
    )
    hdl = torch_symm_mem.rendezvous(buffer, key)
    if hdl.multicast_ptr == 0:
        raise RuntimeError(
            "mega fused-AR: multicast is not supported on this topology "
            "(multicast_ptr == 0)."
        )
    flags = torch_symm_mem.empty(
        mega_ops.flags_numel(comm.world_size), device=device, dtype=torch.uint32
    )
    flags.zero_()
    hflags = torch_symm_mem.rendezvous(flags, key)
    hflags.barrier()
    state = torch.zeros(mega_ops.STATE_SIZE, device=device, dtype=torch.uint32)

    gemm_op = _build_gemm_op(
        group_name=key,
        buffer=buffer,
        buffer_handle=hdl,
        flags=flags,
        flags_handle=hflags,
        state=state,
        rank=hdl.rank,
        world_size=comm.world_size,
        device=device,
    )

    workspace = MegaSymmWorkspace(
        buffer=buffer,
        multicast_ptr=hdl.multicast_ptr,
        flags_ptrs_dev=hflags.buffer_ptrs_dev,
        state_ptr=state.data_ptr(),
        rank=hdl.rank,
        world_size=comm.world_size,
        max_size=buffer.numel() * buffer.element_size(),
        gemm_op=gemm_op,
        refs=(flags, state, hdl, hflags),
    )
    _workspaces[key] = workspace
    logger.info(
        "mega fused-AR workspace ready for group '%s' (world=%d, %.1f MiB, "
        "gemm_op=%s)",
        key,
        comm.world_size,
        workspace.max_size / (1 << 20),
        gemm_op is not None,
    )
    return workspace


def _build_gemm_op(
    *,
    group_name: str,
    buffer: torch.Tensor,
    buffer_handle,
    flags: torch.Tensor,
    flags_handle,
    state: torch.Tensor,
    rank: int,
    world_size: int,
    device,
) -> Optional[Any]:
    """The GemmRSNormAG instance, or None when that flag is off.

    One instance serves BOTH o_proj and down_proj: their N is the same
    (hidden_size) and K does not participate in instance sizing (the launcher
    only checks K % kTileK), so the weight is a per-forward argument. The
    instance's own buffers (tile_flags / sq_partial / counters) are sized from
    (max_m, n) and stay private to it -- only the data buffer and the barrier
    are shared with the standalone fused-AR path.
    """
    from sglang.srt.layers.communicator import GEMM_AR_RMSNORM_FUSED_MAX_M
    from sglang.srt.runtime_context import get_exec, process_model_config

    if not get_exec().comm.enable_gemm_ar_rmsnorm_fused:
        return None

    import mega_ops

    model_config = process_model_config()
    hidden = model_config.hidden_size

    # Ask mega_ops which (tile_m, tile_n, cluster_m) it wants for this shape
    # rather than assuming one: select_config applies the tuning rules that live
    # with the kernel (tile_n 256 when it divides N and still fills a wave of
    # SMs, else 128; cluster_m per flux's production rule on row count), and its
    # answer is what every downstream gate then reads off the instance.
    #
    # Both failure modes end in "disable the fused path", never in an exception
    # reaching the model forward:
    #   ValueError     -- select_config refuses the shape outright (N not a
    #                     multiple of 128, so N cannot divide into whole
    #                     n-tiles).
    #   AssertionError -- select_config returned a triple that is NOT in
    #                     SUPPORTED_CONFIGS, so no compiled kernel exists for
    #                     it. Today select_config hard-codes tile_m=128 and all
    #                     four compiled configs use it, so this cannot fire;
    #                     it becomes reachable the moment the heuristic learns
    #                     to pick a tile_m (e.g. 64, once a Pingpong variant
    #                     exists) that the C++ side has not instantiated. That
    #                     is exactly the divergence this must survive as a
    #                     fallback rather than a crashed request.
    try:
        tile_m, tile_n, cluster_m = mega_ops.select_config(
            hidden, world_size, GEMM_AR_RMSNORM_FUSED_MAX_M, device
        )
    except ValueError as exc:
        logger.warning(
            "gemm-ar-rmsnorm-fused disabled: mega_ops.select_config rejected "
            "this model's shape (hidden=%d, world_size=%d): %s. Falling back to "
            "the ordinary all-reduce path for every layer.",
            hidden,
            world_size,
            exc,
        )
        return None

    # GemmRSNormAG asserts max_m % (tile_m * world_size) == 0. The admission cap
    # is a round token count, not necessarily a multiple of that quantum --
    # 16384 is not, at world_size 6 -- so floor it here rather than letting the
    # assert kill the server at first forward. The quantum comes from the
    # selected tile_m, not a literal, so it tracks whatever config was chosen.
    # The shared buffer is sized independently by workspace_data_numel(), so
    # flooring costs no memory; it only means the fused path admits slightly
    # fewer tokens at such world sizes.
    quantum = tile_m * world_size
    usable_max_m = GEMM_AR_RMSNORM_FUSED_MAX_M // quantum * quantum
    if usable_max_m == 0:
        logger.warning(
            "gemm-ar-rmsnorm-fused disabled: admission cap %d is smaller than "
            "the kernel's M quantum %d (tile_m %d * world_size %d). Falling "
            "back to the ordinary all-reduce path for every layer.",
            GEMM_AR_RMSNORM_FUSED_MAX_M,
            quantum,
            tile_m,
            world_size,
        )
        return None

    # Reject a triple with no compiled kernel HERE, before construction, rather
    # than letting GemmRSNormAG's own assert escape: an AssertionError out of
    # this function propagates through get_workspace and try_forward (which only
    # catches RuntimeError) all the way into the model forward, killing the
    # request. A configuration mismatch must degrade to the ordinary all-reduce
    # path, not to a crash.
    if (tile_m, tile_n, cluster_m) not in mega_ops.SUPPORTED_CONFIGS:
        logger.warning(
            "gemm-ar-rmsnorm-fused disabled: mega_ops.select_config chose "
            "(tile_m=%d, tile_n=%d, cluster_m=%d) for hidden=%d world_size=%d, "
            "but no kernel is compiled for that triple (compiled: %s). Falling "
            "back to the ordinary all-reduce path for every layer.",
            tile_m,
            tile_n,
            cluster_m,
            hidden,
            world_size,
            sorted(mega_ops.SUPPORTED_CONFIGS),
        )
        return None

    # The kernel takes eps as an instance field, so every layer routed through
    # this instance must share one value. For Qwen3 dense every RMSNorm is built
    # with config.rms_norm_eps (qwen3.py:354, :378), so they do. try_forward
    # re-checks per call and falls back on mismatch -- so getting this wrong is
    # a silent permanent fallback, not a wrong result. Log it to make that
    # visible.
    eps = model_config.hf_config.rms_norm_eps
    logger.info(
        "gemm-ar-rmsnorm-fused instance: tile=(%d, %d) cluster_m=%d max_m=%d "
        "(cap %d floored to the %d quantum) hidden=%d eps=%g",
        tile_m,
        tile_n,
        cluster_m,
        usable_max_m,
        GEMM_AR_RMSNORM_FUSED_MAX_M,
        quantum,
        hidden,
        eps,
    )
    # Construction still validates things this function cannot pre-check (world
    # size, dtype support, the injected buffers' capacity floors, the multicast
    # handle). Any of those failing means the fused path is unusable, which is a
    # fallback condition -- so convert it to one here rather than letting an
    # AssertionError or ValueError reach the model forward.
    try:
        return mega_ops.GemmRSNormAG(
            group_name,
            rank,
            world_size,
            usable_max_m,
            hidden,
            device,
            eps=eps,
            tile_shape=(tile_m, tile_n),
            cluster_m=cluster_m,
            data=buffer,
            data_handle=buffer_handle,
            bflags=flags,
            bflags_handle=flags_handle,
            state=state,
        )
    except (AssertionError, ValueError) as exc:
        logger.warning(
            "gemm-ar-rmsnorm-fused disabled: GemmRSNormAG construction failed "
            "for tile=(%d, %d) cluster_m=%d max_m=%d hidden=%d world_size=%d: "
            "%s: %s. Falling back to the ordinary all-reduce path for every "
            "layer.",
            tile_m,
            tile_n,
            cluster_m,
            usable_max_m,
            hidden,
            world_size,
            type(exc).__name__,
            exc,
        )
        return None


def peek_workspace(*, group_name: str) -> Optional[MegaSymmWorkspace]:
    """The cached workspace, or None. Never creates -- for callers on a path
    that must not allocate (CUDA-graph capture readiness checks, buffer-view
    identity tests)."""
    return _workspaces.get(group_name)


def is_workspace_buffer(tensor: Optional[torch.Tensor]) -> bool:
    """True when ``tensor`` is pointer-identical to a workspace's data buffer
    at offset 0. This is an exact identity check, NOT a containment test: a
    view into the same storage at a nonzero offset has a different
    ``data_ptr()`` and answers False here even though the memory is shared.

    It is sufficient today because every producer views the buffer from index
    0 (``workspace.buffer[:n]`` or the buffer itself), which leaves
    ``data_ptr()`` unchanged. A future sub-view starting at a nonzero offset
    would stop being covered by this guard -- silently, not by raising -- so
    treat this as offset-0-only unless it is extended into a real range check
    against the buffer's extent.

    Callers use this to decide whether a hidden-states tensor IS a workspace's
    data buffer, and must therefore be cloned to survive across calls: the
    NEXT fused call will overwrite that buffer in place (split-prefill keeps
    hidden states on the ForwardBatch).
    """
    if tensor is None or not _workspaces:
        return False
    ptr = tensor.data_ptr()
    return any(w.buffer.data_ptr() == ptr for w in _workspaces.values())

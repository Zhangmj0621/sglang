"""GEMM-fused all-reduce + residual-add + RMSNorm via mega_ops.GemmRSNormAG."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.mega_symm_workspace import get_workspace, peek_workspace

logger = logging.getLogger(__name__)

# Ensure each type of warning is logged only once
_warned_fallbacks: set = set()


def warn_fallback(reason: str, detail: str) -> None:
    key = (reason, detail)
    if key in _warned_fallbacks:
        return
    _warned_fallbacks.add(key)
    logger.warning(
        "--enable-gemm-ar-rmsnorm-fused: falling back to the ordinary "
        "all-reduce path for this shape (%s: %s). The fused kernel is not "
        "used here; correctness is unaffected.",
        reason,
        detail,
    )


def _kernel_tile_k(op) -> int:
    """The instance's K-tile, derived the way the kernel derives it."""
    return 128 // torch.tensor([], dtype=op.dtype).element_size()


_unavailable: Optional[bool] = None


def _is_unavailable() -> bool:
    global _unavailable
    if _unavailable is None:
        try:
            import mega_ops

            usable = mega_ops.is_available()
        except ImportError:
            usable = False
        if usable:
            comm = get_tp_group().torch_symm_mem_comm
            usable = comm is not None and not comm.disabled
        _unavailable = not usable
    return _unavailable


def gemm_ar_rmsnorm_fused_enabled() -> bool:
    from sglang.srt.runtime_context import get_exec

    return get_exec().comm.enable_gemm_ar_rmsnorm_fused


def gemm_ar_rmsnorm_fused_ready() -> bool:
    if _is_unavailable():
        return False
    if torch.cuda.is_current_stream_capturing():
        # Skip since this path is prefill-only.
        return False
    return True


def is_gemm_ar_eligible(*, m: int, n: int, k: int, world_size: int) -> bool:
    from sglang.srt.layers.communicator import GEMM_AR_RMSNORM_FUSED_MAX_M

    if m <= 0:
        return False
    if m > GEMM_AR_RMSNORM_FUSED_MAX_M:
        warn_fallback(
            "token count exceeds the admission ceiling",
            f"m={m} > GEMM_AR_RMSNORM_FUSED_MAX_M={GEMM_AR_RMSNORM_FUSED_MAX_M}",
        )
        return False
    op = _peek_op(create_if_missing=True)
    if op is None:
        return False

    if op.config_for(m) is None:
        warn_fallback(
            "no compiled tile configuration fits this token count",
            f"m={m}, n={op.n}, world_size={world_size} "
            f"(sized_tile_m_cnt={op._sized_tile_m_cnt}, "
            f"row stride={op._sized_n_tiles})",
        )
        return False
    tile_k = _kernel_tile_k(op)
    if k % tile_k:
        warn_fallback(
            "reduction dim is not a multiple of the K-tile",
            f"k={k}, kTileK={tile_k} (dtype {op.dtype})",
        )
        return False
    if m > op.max_m:
        warn_fallback(
            "token count exceeds this instance's max_m",
            f"m={m} > op.max_m={op.max_m}",
        )
        return False
    if n != op.n:
        warn_fallback(
            "output width does not match the instance",
            f"n={n} != op.n={op.n}",
        )
        return False
    return True


def _peek_op(*, create_if_missing: bool = False):
    """The GemmRSNormAG for this TP group, or None."""
    comm = get_tp_group().torch_symm_mem_comm
    if comm is None or comm.disabled:
        return None
    workspace = peek_workspace(group_name=comm.group.group_name)
    if workspace is None:
        if not create_if_missing:
            return None
        try:
            workspace = get_workspace(group=get_tp_group())
        except RuntimeError as exc:
            logger.debug("gemm-ar-rmsnorm-fused workspace unavailable: %s", exc)
            return None
    return workspace.gemm_op


def mark_normed(tensor: torch.Tensor) -> None:
    setattr(tensor, "_mega_gemm_ar_normed", True)


def is_normed(tensor: Optional[torch.Tensor]) -> bool:
    if tensor is None:
        return False
    return getattr(tensor, "_mega_gemm_ar_normed", False) is True


def try_forward(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the fused GEMM+RS+add+norm+AG, or return None to fall back."""
    from sglang.srt.layers.rmsnorm_fused_ar import _token_shard

    if _is_unavailable():
        return None
    try:
        workspace = get_workspace(group=get_tp_group())
    except RuntimeError as exc:
        logger.debug("gemm-ar-rmsnorm-fused workspace unavailable: %s", exc)
        return None
    op = workspace.gemm_op
    if op is None:
        return None
    if x.dim() != 2:
        warn_fallback(
            "activation is not 2-D",
            f"x.shape={tuple(x.shape)} (the kernel takes a [M, K] matrix)",
        )
        return None
    if weight.dim() != 2:
        warn_fallback(
            "weight is not 2-D",
            f"weight.shape={tuple(weight.shape)} (the kernel takes a [N, K] matrix)",
        )
        return None

    m, k = x.shape
    n = weight.shape[0]
    if not is_gemm_ar_eligible(m=m, n=n, k=k, world_size=workspace.world_size):
        return None
    if not _tensors_eligible(
        x=x,
        weight=weight,
        norm_weight=norm_weight,
        residual=residual,
        m=m,
        n=n,
        dtype=op.dtype,
        eps=eps,
        op_eps=op.eps,
    ):
        return None

    start, end = _token_shard(m, workspace.rank, workspace.world_size)
    call_tile_m, _call_tile_n, _call_cluster_m = op.config_for(m)
    kernel_start = (
        workspace.rank * (m // (call_tile_m * workspace.world_size)) * call_tile_m
    )
    assert (start, end) == (kernel_start, kernel_start + m // workspace.world_size), (
        f"shard window ({start}, {end}) disagrees with the kernel's "
        f"({kernel_start}, {kernel_start + m // workspace.world_size}) for "
        f"M={m}, world_size={workspace.world_size}"
    )
    residual_shard = residual[start:end]
    # A row-slice of a contiguous [M, N] tensor is contiguous by construction --
    # assert it rather than assume it, because the kernel takes a bare pointer
    # with an implied row stride, and a non-contiguous shard would corrupt
    # silently instead of raising.
    assert residual_shard.is_contiguous(), (
        f"residual[{start}:{end}] is not contiguous for residual.shape="
        f"{tuple(residual.shape)}"
    )

    normed = op.forward(x, weight, norm_weight, residual_shard)
    mark_normed(normed)
    setattr(residual, "_mega_residual_shard", (start, end, get_tp_group()))
    return normed, residual


def _tensors_eligible(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    residual: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype,
    eps: float,
    op_eps: float,
) -> bool:
    if eps != op_eps:
        warn_fallback(
            "layer eps differs from the instance's eps",
            f"layer eps={eps!r} != instance eps={op_eps!r} (the kernel takes "
            "eps as an instance field, so one instance cannot serve both)",
        )
        return False
    for name, t in (
        ("x", x),
        ("weight", weight),
        ("norm_weight", norm_weight),
        ("residual", residual),
    ):
        if t.dtype != dtype:
            warn_fallback(
                "tensor dtype does not match the instance",
                f"{name}.dtype={t.dtype} != instance dtype={dtype}",
            )
            return False
        if not t.is_contiguous():
            warn_fallback(
                "tensor is not contiguous",
                f"{name} with shape {tuple(t.shape)}",
            )
            return False
    if not (
        x.dim() == 2
        and weight.dim() == 2
        and weight.shape[1] == x.shape[1]
        and norm_weight.numel() == n
        and residual.shape == (m, n)
    ):
        warn_fallback(
            "tensor shapes are inconsistent with (m, n)",
            f"x={tuple(x.shape)} weight={tuple(weight.shape)} "
            f"norm_weight={norm_weight.numel()} residual={tuple(residual.shape)} "
            f"for m={m}, n={n}",
        )
        return False
    return True

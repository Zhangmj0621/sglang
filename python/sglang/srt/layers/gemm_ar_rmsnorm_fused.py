"""GEMM-fused all-reduce + residual-add + RMSNorm via mega_ops.GemmRSNormAG.

Where ``rmsnorm_fused_ar`` runs the collective as its OWN kernel after the GEMM
has landed in a staging buffer, this path fuses it INTO the GEMM epilogue: one
kernel does GEMM(partial) -> multimem ReduceScatter -> residual add -> RMSNorm
-> multimem AllGather. That saves a full [M, hidden] store/load round trip and
one kernel launch, at the cost of two producer warps taken from the GEMM.

The kernel's tiling makes ``M % (tile_m * world_size) == 0`` a hard requirement
(swizzle_m_tile is a bijection only over that range), so this is a prefill-only
path: decode batch sizes never reach it. ``tile_m``, ``tile_n`` and
``cluster_m`` are whatever ``mega_ops.select_config`` chose for this model's
shape -- every gate here reads them off the built instance rather than assuming
a value, so the gates track the compiled configuration. Everything that does not
qualify falls back through the caller's normal route, with one warning per
distinct reason (see ``warn_fallback``).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.mega_symm_workspace import get_workspace, peek_workspace

logger = logging.getLogger(__name__)

# A shape that the kernel cannot service falls back to the ordinary all-reduce
# path. That is a correctness-preserving outcome but a silent performance
# cliff, so each distinct reason is logged once per (reason, shape) rather than
# per call -- a prefill runs this on every layer, and an unconditional warning
# would emit 64 identical lines per forward.
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
    """The instance's K-tile, derived the way the kernel derives it.

    csrc/sm90_gemm_rs_norm_ag_launch.inl: ``kTileK = 128 / sizeof(ElementAB)``
    -- 128 bytes of K per tile, so it is a function of the element type, not a
    constant. bf16 and fp16 both give 64; reading it from the dtype keeps this
    correct if a wider or narrower element type is ever instantiated.
    """
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
        # This path is prefill-only by design (see the module docstring): the M
        # alignment gate can never pass at decode batch sizes, so rather than
        # carry capture-time workspace plumbing for a path that never fires, it
        # is simply off under capture.
        return False
    return True


def is_gemm_ar_eligible(*, m: int, n: int, k: int, world_size: int) -> bool:
    """True when the kernel can service this (M, N, K) at this world size.

    Every condition is a hard requirement of the INSTANCE's compiled
    configuration, read off the instance rather than assumed. The instance was
    built from mega_ops.select_config's chosen (tile_m, tile_n, cluster_m), so
    these gates track whatever configuration that heuristic picked:

    * ``m % (op.tile_m * world_size)``: swizzle_m_tile is a bijection only over
      ``[0, world_size * tile_m_cnt_per_rank)``; a surplus m-tile makes the
      reduce warp's ``owner`` index run past the tile-flag table -- a wild
      REMOTE store, not a local fault. tile_m is 128 for every currently
      instantiated config, but a tile_m=64 Pingpong variant would halve this
      quantum, so it must come from the instance.
    * ``n % op.tile_n`` and ``k % kTileK``: the tile-flag row stride and the
      K-tile, the latter derived from the element size (see _kernel_tile_k).
    * ``m <= op.max_m``: the instance's own buffers (tile_flags, sq, counters)
      are sized for its actual max_m, not for the GEMM_AR_RMSNORM_FUSED_MAX_M
      ceiling -- the workspace can build the instance with a smaller max_m
      (floored to a multiple of tile_m * world_size), so e.g. at world_size=6
      the constant is 16384 but op.max_m is 16128.

    Every rejection logs once per (reason, shape) -- see warn_fallback.
    """
    from sglang.srt.layers.communicator import GEMM_AR_RMSNORM_FUSED_MAX_M

    # Cheap early-out on the module ceiling, before paying for _peek_op()'s
    # group/workspace lookup on an obviously-oversized m. Not authoritative by
    # itself -- the real bound is op.max_m, checked below.
    if m <= 0:
        return False
    if m > GEMM_AR_RMSNORM_FUSED_MAX_M:
        warn_fallback(
            "token count exceeds the admission ceiling",
            f"m={m} > GEMM_AR_RMSNORM_FUSED_MAX_M={GEMM_AR_RMSNORM_FUSED_MAX_M}",
        )
        return False
    op = _peek_op()
    if op is None:
        # No workspace yet (first forward builds it lazily) -- not a shape
        # rejection, so it is not worth a warning.
        return False

    quantum = op.tile_m * world_size
    if m % quantum:
        warn_fallback(
            "token count is not a multiple of tile_m * world_size",
            f"m={m}, tile_m={op.tile_m}, world_size={world_size} "
            f"(needs a multiple of {quantum})",
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
    if n % op.tile_n:
        warn_fallback(
            "output width is not a multiple of tile_n",
            f"n={n}, tile_n={op.tile_n}",
        )
        return False
    return True


def _peek_op():
    """The cached GemmRSNormAG, or None when no workspace exists yet."""
    comm = get_tp_group().torch_symm_mem_comm
    if comm is None or comm.disabled:
        return None
    workspace = peek_workspace(group_name=comm.group.group_name)
    return workspace.gemm_op if workspace is not None else None


# Attribute marking a hidden-states tensor as already all-reduced AND
# normalized by the fused kernel, so the LayerCommunicator stage that would
# otherwise do those two things must be skipped (a second norm would be wrong,
# not merely wasteful). Compare presence only -- the value carries nothing.
_NORMED_ATTR = "_mega_gemm_ar_normed"


def mark_normed(tensor: torch.Tensor) -> None:
    setattr(tensor, _NORMED_ATTR, True)


def is_normed(tensor: Optional[torch.Tensor]) -> bool:
    if tensor is None:
        return False
    return getattr(tensor, _NORMED_ATTR, False) is True


def try_forward(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the fused GEMM+RS+add+norm+AG, or return None to fall back.

    ``x`` [M, K_local] is this rank's activation shard; ``weight`` [N, K_local]
    the row-parallel weight (torch Linear layout, which is exactly the kernel's
    column-major B). ``residual`` [M, N] is the FULL residual -- the caller only
    ever holds the full tensor, never a pre-sharded one. This rank's
    ``residual[start:end]`` shard is updated IN PLACE to ``partial_sum +
    residual``, and the FULL tensor is marked with that shard window.

    Returns ``(normed_full, residual)``. ``normed_full`` [M, N] is a view into
    the workspace data buffer, valid only until the next fused call on this
    instance -- the same contract rmsnorm_fused_ar's output already has.
    """
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
        # Guard the unpack below explicitly: _tensors_eligible also checks
        # x.dim() == 2, but that check can only run -- and fall back cleanly --
        # once we get there. `m, k = x.shape` on a non-2D x raises before any
        # eligibility check would see it.
        warn_fallback(
            "activation is not 2-D",
            f"x.shape={tuple(x.shape)} (the kernel takes a [M, K] matrix)",
        )
        return None
    if weight.dim() != 2:
        # Symmetric guard: `n = weight.shape[0]` below reads index 0
        # unconditionally, which raises on a 0-d weight and would silently take
        # the wrong axis as N on anything else non-2D -- before
        # _tensors_eligible's weight.dim() == 2 check ever runs.
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
    # The kernel writes the shard at rank * tile_m_cnt_per_rank * kTileM
    # (gemm_rs_comm.cuh: normag_process_shard's global_row), with
    # tile_m_cnt_per_rank = m // tile_m // world_size for THIS call's m. Compute
    # that literal expression -- not a different one that merely happens to
    # coincide under the alignment gate above -- because the residual marker
    # below tells every later consumer which rows are fresh: if the tile-
    # alignment gate were ever weakened or bypassed, this is the check that
    # must be able to catch the divergence instead of silently mis-marking rows.
    kernel_start = (
        workspace.rank * (m // (op.tile_m * workspace.world_size)) * op.tile_m
    )
    assert (start, end) == (kernel_start, kernel_start + m // workspace.world_size), (
        f"shard window ({start}, {end}) disagrees with the kernel's "
        f"({kernel_start}, {kernel_start + m // workspace.world_size}) for "
        f"M={m}, world_size={workspace.world_size}"
    )
    # try_forward owns the sharding: the caller (LayerCommunicator) only ever
    # holds the FULL [M, N] residual and passes it straight through -- but
    # mega_ops.GemmRSNormAG.forward requires this rank's [M/world_size, N]
    # shard (see its own residual.shape assert). Slicing here, instead of
    # pushing a second pre-sharded-residual convention onto the caller, means
    # the marker below is set on the FULL tensor with GLOBAL (start, end) --
    # byte-identical to what rmsnorm_fused_ar_forward already sets on its own
    # FULL residual -- so there is exactly one _mega_residual_shard convention
    # in the codebase, and the shared ensure_full_residual reads either
    # producer's marker the same way without needing to know which one fired.
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
    """Per-tensor preconditions the kernel's host side would otherwise raise on.

    Checked here so an unsupported shape falls back to the ordinary all-reduce
    path instead of killing the request. ``residual`` is checked at its FULL
    ``[M, N]`` shape -- the caller only ever holds the full tensor;
    ``try_forward`` slices out this rank's ``[M/world_size, N]`` shard for the
    kernel call only after this check passes. ``eps`` is an INSTANCE field of
    the kernel, so a layer whose eps differs from the one the instance was built
    with must not be routed through it (Qwen3 dense builds every RMSNorm with
    config.rms_norm_eps, so this only fires on a model that mixes them).

    Each rejection warns once per (reason, detail) -- an eps or dtype mismatch
    disables the fused path permanently for that layer, which is exactly the
    kind of silent performance cliff worth a log line.
    """
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

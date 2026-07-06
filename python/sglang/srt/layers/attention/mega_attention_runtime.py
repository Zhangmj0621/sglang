"""Per-rank runtime holder for the megaAttention fused prefill kernel.

megaAttention(`3rdparty/megaAttention`)的 Hopper SM90 fused persistent kernel
`causal-varlen-FA → O_proj → tensor-parallel NVLS AllReduce` 通过 `FusedFaOprojArWorkspace`
接进 SGLang,在 Qwen3 MoE 的 **prefill EXTEND** 路径替换 `self.attn + self.o_proj + 延迟 TP
AllReduce`(设计见仓库根 `开发文档.md`)。

本模块只做 SGLang 侧的资源持有与 metadata 装配:
  - 懒初始化并持有 per-rank `FusedFaOprojArWorkspace`(首个命中 gate 的 EXTEND forward 触发,
    SGLang SPMD 锁步下 symm-mem rendezvous 安全);
  - `should_use_mega(forward_batch, layer)`:fast-path gate,**同一 forward 内对所有层结果一致**;
  - `get_or_build_prefill_meta(forward_batch)`:每 forward 建一次 page_table / cache_seqlens /
    seqlens_q,并调一次 `ws.build_prefill_meta`,结果缓存在 forward_batch 上,48 层复用;
  - `run_layer(...)`:每层借调 q / w_o / 本层 paged KV view,`ws.launch_layer_prefill`,返回
    C_sym 的零拷贝 `[tot_q, hidden]` view(post-o_proj、已 all-reduced)。

总开关 `SGLANG_USE_MEGA_ATTENTION`(默认关)。关闭或不满足 gate 时返回 None / False,调用方
fallback 到现有 `self.attn + self.o_proj + SGLang AllReduce`,字节级保持原状。
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

logger = logging.getLogger(__name__)

# 缓存在 forward_batch 上的属性名(每 forward 一个新对象,天然按 forward 失效)。
_FB_META_ATTR = "_mega_prepared_meta"

_RUNTIME: Optional[MegaAttentionRuntime] = None


class _WsConfig:
    """从首个命中层 + 全局 pool 推导出的、跨层不变的 workspace 配置。"""

    __slots__ = ("H_local", "H_kv_local", "D", "hidden", "q_per_kv")

    def __init__(self, H_local, H_kv_local, D, hidden, q_per_kv):
        self.H_local = H_local
        self.H_kv_local = H_kv_local
        self.D = D
        self.hidden = hidden
        self.q_per_kv = q_per_kv


class _PreparedMeta:
    """每 forward 一份:mega 调度 meta + borrowed page_table / cache_seqlens。"""

    __slots__ = ("mega_meta", "page_table", "cache_seqlens", "num_row_tiles")

    def __init__(self, mega_meta, page_table, cache_seqlens, num_row_tiles):
        self.mega_meta = mega_meta
        self.page_table = page_table
        self.cache_seqlens = cache_seqlens
        self.num_row_tiles = num_row_tiles


class MegaAttentionRuntime:
    def __init__(self):
        self.ws = None  # FusedFaOprojArWorkspace(lazy)
        self.cfg: Optional[_WsConfig] = None
        self.page_size = 128
        self.mega_launch_count = 0  # debug: 用于验证 mega 真被触发(防静默 fallback)
        self._disabled = False  # 懒初始化失败后熔断,后续直接 fallback

    # ---------------------------------------------------------------- gate
    def should_use_mega(self, forward_batch: ForwardBatch, layer) -> bool:
        """fast-path gate。任何一项不满足都返回 False(调用方 fallback)。

        判定只依赖 forward 级与静态配置量,**同一 forward 对所有层一致**;num_row_tiles 用实际
        extend_seq_lens_cpu 计算并与 workspace 容量比对,超容则 fallback(不污染 workspace)。
        """
        if self._disabled or not envs.SGLANG_USE_MEGA_ATTENTION.get():
            return False
        # 仅纯 prefill EXTEND(不含 MIXED / spec / split-prefill 等;mega 不吃混入的 decode 行)。
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return False
        if getattr(forward_batch, "tbo_split_seq_index", None) is not None:
            return False  # 两-batch overlap 与单块 C_sym 复用冲突
        if forward_batch.extend_seq_lens_cpu is None:
            return False

        sa = self._server_args()
        # 纯 TP:无 dp-attention / CP / EP / 量化通信。属性名用 getattr 兜底,缺省视为关闭。
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        if is_dp_attention_enabled():
            return False
        if getattr(sa, "enable_quant_communications", False):
            return False
        if int(getattr(sa, "context_parallel_size", 1) or 1) > 1:
            return False
        if int(getattr(sa, "page_size", 128) or 128) != 128:
            return False

        # 层级静态条件(KV 必须在 RoPE 阶段经 fused-set-kv-buffer 写入)。
        if not getattr(layer, "compatible_with_fused_kv_buffer", False):
            return False
        try:
            from sglang.srt.models.utils import enable_fused_set_kv_buffer

            if not enable_fused_set_kv_buffer(forward_batch):
                return False
        except Exception:
            return False

        # 懒初始化 workspace(SPMD 锁步,collective 安全);失败则熔断 fallback。
        if self.ws is None:
            try:
                self._build_workspace(layer, forward_batch)
            except Exception as e:  # noqa: BLE001
                logger.warning("megaAttention workspace init failed, fallback: %r", e)
                self._disabled = True
                return False

        # 实际 num_row_tiles 容量校验(超则本 batch fallback)。
        num_row_tiles = sum(
            (int(s) + 127) // 128 for s in forward_batch.extend_seq_lens_cpu
        )
        if (
            num_row_tiles > self.ws.max_num_row_tiles
            or len(forward_batch.extend_seq_lens_cpu) > self.ws.max_num_batch
        ):
            return False
        return True

    # ------------------------------------------------------ workspace init
    def _server_args(self):
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args()

    def _build_workspace(self, layer, forward_batch):
        from mega_attention.runtime import FusedFaOprojArWorkspace

        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        # ---- per-rank 形状(从命中层 + config 推) ----
        H_local = layer.num_heads  # local q heads
        H_kv_local = layer.num_kv_heads  # local kv heads
        D = layer.head_dim
        q_per_kv = H_local // H_kv_local
        hidden = layer.hidden_size  # = config.hidden_size
        self.cfg = _WsConfig(H_local, H_kv_local, D, hidden, q_per_kv)

        pool = get_token_to_kv_pool()
        req_to_token = get_req_to_token_pool().req_to_token
        kbuf0 = pool.get_key_buffer(0)  # [size+page_size, H_kv_local, D]
        max_num_pages = int(kbuf0.shape[0]) // 128
        max_num_pages_per_seq = math.ceil(int(req_to_token.shape[1]) / 128)

        # ---- row-tile 容量上界(用已解析配置;runtime guard 仍兜底,见 should_use_mega) ----
        sa = self._server_args()
        T = 0
        try:
            T = int(sa.max_prefill_buffer_tokens())
        except Exception:
            T = int(getattr(sa, "chunked_prefill_size", 0) or 0)
        if T <= 0:
            raise RuntimeError(
                "chunked_prefill_size 未启用,mega 需要 chunked prefill 容量上界"
            )
        N = int(getattr(sa, "prefill_max_requests", None) or 0) or int(
            getattr(sa, "max_running_requests", None) or 0
        )
        if N <= 0:
            N = min(int(req_to_token.shape[0]), 256)
        N = min(N, T)
        max_num_row_tiles = math.ceil(T / 128) + N
        max_num_batch = N

        tp_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        group_name = get_tp_group().device_group.group_name

        # ---- launch config: 用首个 EXTEND batch 的 shape 经 choose_launch_config 调优
        #      (megaAttention benchmark 的 --auto 同款)。compile once 服务所有 shape,故按首个真实
        #      prefill shape 选一次;失败回退默认 (w_fa=4,w_oproj=1,w_ar=1,sg=4)。----
        w_fa, w_oproj, w_ar, sg = 4, 1, 1, 4
        try:
            from mega_attention.metadata.launch_heuristic import choose_launch_config
            from mega_attention.metadata.row_desc import build_row_desc

            _cfg = choose_launch_config(
                build_row_desc(forward_batch.extend_seq_lens_cpu), hidden, tp_size
            )
            w_fa, w_oproj, w_ar, sg = _cfg.w_fa, _cfg.w_oproj, _cfg.w_ar, _cfg.sg
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "megaAttention choose_launch_config failed, using defaults: %r", e
            )

        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        ws = FusedFaOprojArWorkspace.create(
            group_name,
            max_num_row_tiles=max_num_row_tiles,
            hidden=hidden,
            H_local=H_local,
            D=D,
            tp_size=tp_size,
            rank=rank,
            q_per_kv=q_per_kv,
            super_group_n_tiles=sg,
            max_num_batch=max_num_batch,
            max_tot_k=max_num_pages * 128,
            dtype=torch.bfloat16,
            device=dev,
            paged=True,
            page_size=128,
            max_num_pages=max_num_pages,
            max_num_pages_per_seq=max_num_pages_per_seq,
        )
        ws.compile(
            w_fa=w_fa, w_oproj=w_oproj, w_ar=w_ar, softmax_scale=float(D) ** -0.5
        )
        self.ws = ws
        logger.info(
            "megaAttention workspace ready: tp=%d rank=%d H_local=%d H_kv=%d D=%d hidden=%d "
            "q_per_kv=%d max_row_tiles=%d max_batch=%d max_pages=%d pages_per_seq=%d "
            "cfg(w_fa=%d,w_oproj=%d,w_ar=%d,sg=%d)",
            tp_size,
            rank,
            H_local,
            H_kv_local,
            D,
            hidden,
            q_per_kv,
            max_num_row_tiles,
            max_num_batch,
            max_num_pages,
            max_num_pages_per_seq,
            w_fa,
            w_oproj,
            w_ar,
            sg,
        )

    # ------------------------------------------------------------- per fwd
    def get_or_build_prefill_meta(self, forward_batch: ForwardBatch) -> _PreparedMeta:
        cached = getattr(forward_batch, _FB_META_ATTR, None)
        if cached is not None:
            return cached

        from sglang.srt.model_executor.forward_context import get_req_to_token_pool

        seq_lens = forward_batch.seq_lens  # [B] 总 KV 长度(prefix+extend)
        max_pages = int((int(seq_lens.max().item()) + 127) // 128)
        req_to_token = get_req_to_token_pool().req_to_token
        # page_table[b, p] = 物理 page index;复用 FA3 算法(req_to_token[:, ::128] // 128)。
        page_table = (
            req_to_token[forward_batch.req_pool_indices][:, : max_pages * 128 : 128]
            // 128
        ).to(torch.int32)
        cache_seqlens = seq_lens.to(torch.int32)
        seqlens_q = (
            forward_batch.extend_seq_lens_cpu
        )  # host List[int],build_row_desc 用

        mega_meta = self.ws.build_prefill_meta(seqlens_q)
        num_row_tiles = mega_meta.num_row_tiles
        prepared = _PreparedMeta(mega_meta, page_table, cache_seqlens, num_row_tiles)
        setattr(forward_batch, _FB_META_ATTR, prepared)
        return prepared

    def paged_kv_view(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        pool = get_token_to_kv_pool()
        H_kv, D = self.cfg.H_kv_local, self.cfg.D
        k = pool.get_key_buffer(layer_id)
        v = pool.get_value_buffer(layer_id)
        npages = int(k.shape[0]) // 128
        kc = k[: npages * 128].view(npages, 128, H_kv, D)
        vc = v[: npages * 128].view(npages, 128, H_kv, D)
        return kc, vc

    def run_layer(
        self,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        w_o: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """单层 mega 入口:返回 C_sym 零拷贝 `[tot_q, hidden]`(post-o_proj、已 all-reduced)。"""
        meta = self.get_or_build_prefill_meta(forward_batch)
        tot_q = q.shape[0]
        H_local, D, hidden = self.cfg.H_local, self.cfg.D, self.cfg.hidden
        q3 = q.view(tot_q, H_local, D).contiguous()
        kc, vc = self.paged_kv_view(layer_id)
        self.ws.launch_layer_prefill(
            meta=meta.mega_meta,
            q=q3,
            w_o=w_o,
            k_cache=kc,
            v_cache=vc,
            page_table=meta.page_table,
            cache_seqlens=meta.cache_seqlens,
        )
        self.mega_launch_count += 1
        if self.mega_launch_count == 1:
            logger.info("megaAttention prefill ENGAGED (first launch).")
        num_out = self.ws.num_out
        # C_sym compact token-row 布局,num_out*128 == hidden(Qwen3 hidden 整除 128)→ 纯 view。
        return self.ws.csym.reshape(-1, num_out * 128)[:tot_q, :hidden]


def get_mega_runtime() -> Optional[MegaAttentionRuntime]:
    """进程内(per-rank)单例;总开关关时返回 None。"""
    global _RUNTIME
    if not envs.SGLANG_USE_MEGA_ATTENTION.get():
        return None
    if _RUNTIME is None:
        _RUNTIME = MegaAttentionRuntime()
    return _RUNTIME

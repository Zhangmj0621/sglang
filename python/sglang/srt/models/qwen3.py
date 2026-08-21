# Adapted from qwen2.py
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
)
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    apply_gemm_ar_rmsnorm_fused,
    apply_rmsnorm_fused_ar,
)
from sglang.srt.layers.gemm_ar_rmsnorm_fused import is_normed, mark_normed
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rmsnorm_fused_ar import (
    get_fused_ar_staging_view,
    is_fused_ar_buffer_view,
)
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_exec, get_forward, get_parallel, get_stream
from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_hip, is_npu

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

_has_fused_qk_norm_mrope = False
if _use_aiter:
    try:
        from aiter import fused_qk_norm_mrope_3d_cache_pts_quant_shuffle

        _has_fused_qk_norm_mrope = True
        logger.info("aiter fused_qk_norm_mrope_3d kernel available")
    except ImportError:
        pass

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

    from sglang.srt.hardware_backend.npu.cmo import get_cmo_stream, wait_cmo_stream


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        start_layer: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.start_layer = start_layer
        self.tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_parallel().tp_rank

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_exec().deterministic.rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream

        self.use_fused_qk_norm_mrope = (
            _has_fused_qk_norm_mrope
            and isinstance(self.rotary_emb, MRotaryEmbedding)
            and getattr(self.rotary_emb, "mrope_section", None) is not None
        )
        if self.use_fused_qk_norm_mrope:
            # Scale tensors MUST stay on CPU: the C++ kernel uses .item<float>()
            # which triggers hipMemcpy D2H + sync on CUDA tensors, breaking graph capture.
            # Explicit device='cpu' is required because SGLang constructs models inside
            # a `with torch.device('cuda'):` context that changes the default device.
            self._fused_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
            self._fused_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == self.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward_prepare_aiter_fused_mrope(
        self, positions, hidden_states, forward_batch
    ):
        """Fused QK-norm + 3D mRoPE + KV cache write for decode (ROCm/aiter).

        The fused HIP kernel replaces split → QK norm → mRoPE → cache write,
        so KV is already in the paged cache when this returns.
        Returns (q, None, None); caller must pass save_kv_cache=False to attn.
        """
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = qkv.shape[0]

        qkv_3d = qkv.view(num_tokens, -1, self.head_dim)

        token_to_kv_pool = get_token_to_kv_pool()
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(self.attn.layer_id)
        slot_mapping = forward_batch.out_cache_loc

        cos_sin = self.rotary_emb.cos_sin_cache
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(dtype=qkv.dtype)

        q_out = torch.empty(
            num_tokens,
            self.num_heads,
            self.head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        )

        fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
            qkv_3d,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin,
            positions,
            num_tokens,
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_emb.is_neox_style,
            self.rotary_emb.mrope_section,
            self.rotary_emb.mrope_interleaved,
            self.q_norm.variance_epsilon,
            q_out,
            k_cache,
            v_cache,
            slot_mapping,
            self._fused_k_scale,
            self._fused_v_scale,
            None,
            None,
            False,
            False,
            0,
            0,
        )

        q = q_out.reshape(num_tokens, -1)
        return q, None, None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        output_tensor: Optional[torch.Tensor] = None,
        *,
        fused_norm=None,
    ) -> torch.Tensor:
        if get_exec().deterministic.rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        save_kv_cache = True
        use_aiter_fused = (
            self.use_fused_qk_norm_mrope
            and forward_batch.forward_mode.is_decode()
            and get_exec().deterministic.rl_on_policy_target is None
        )

        if use_aiter_fused:
            q, k, v = self.forward_prepare_aiter_fused_mrope(
                positions, hidden_states, forward_batch
            )
            save_kv_cache = False
        elif not _is_npu:
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if get_exec().deterministic.rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
        output, _ = self.o_proj(
            attn_output, output_tensor=output_tensor, fused_norm=fused_norm
        )
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        if (
            hasattr(config, "rope_parameters")
            and config.rope_parameters
            and "rope_theta" in config.rope_parameters
        ):
            rope_theta = config.rope_parameters["rope_theta"]
            rope_scaling = config.rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", 1000000)
            rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_exec().deterministic.rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )
        # Filled in by Qwen3Model.__init__ once every layer exists, and only
        # when the fusion flag is on (see there): the MLP-side fusion needs
        # the NEXT layer's input_layernorm, since that is the norm that would
        # otherwise run on this layer's down_proj output. Stays None on the
        # last layer (and at a PP segment boundary), which is exactly where
        # the MLP side must not fuse.
        #
        # Stored as a 1-tuple, NOT a bare Module: `nn.Module.__setattr__`
        # intercepts any value that IS an nn.Module and registers it as a
        # submodule. That would alias the next layer's `input_layernorm.weight`
        # under this layer's `next_input_layernorm.weight` name too, and
        # `named_parameters()` dedups by tensor identity, returning only the
        # FIRST name it reaches in `_modules` insertion order -- this layer's
        # alias, not the next layer's own `input_layernorm.weight`. The next
        # layer's real parameter name would then be silently absent from
        # `load_weights`'s `dict(named_parameters())`, so it would keep its
        # random init forever while looking like a successful load. Wrapping
        # in a tuple keeps it out of `_modules` entirely. Do not "simplify"
        # this back into a plain Module attribute -- see the
        # `next_input_layernorm` property below for the read side.
        self._next_input_layernorm = None

    @property
    def next_input_layernorm(self):
        """Unwraps `_next_input_layernorm`'s 1-tuple, or None if unset.

        Returns a plain RMSNorm module (or None), exactly what a normal
        attribute would give a reader -- specifically `_maybe_fused_norm`'s
        `norm_module is None` check below. See `_next_input_layernorm` in
        `__init__` above for why the tuple wrapping exists.
        """
        return self._next_input_layernorm[0] if self._next_input_layernorm else None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            post_residual_addition=post_residual_addition,
        )
        if hidden_states.shape[0] != 0:
            attn_fused_norm = self._maybe_fused_norm(
                num_tokens=hidden_states.shape[0],
                k_local=self.self_attn.o_proj.input_size_per_partition,
                norm_module=self.post_attention_layernorm,
                residual=residual,
            )
            attn_staging = (
                get_fused_ar_staging_view(
                    num_tokens=hidden_states.shape[0],
                    hidden=self.hidden_size,
                    dtype=hidden_states.dtype,
                )
                if attn_fused_norm is None
                and apply_rmsnorm_fused_ar(hidden_states.shape[0])
                else None
            )
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                output_tensor=attn_staging,
                fused_norm=attn_fused_norm,
            )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            cache=(
                [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
                if _is_npu
                and check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                and (
                    hasattr(self.mlp.gate_up_proj, "weight")
                    and hasattr(self.mlp.down_proj, "weight")
                )
                else None
            ),
        )

        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            mlp_fused_norm = (
                self._maybe_fused_norm(
                    num_tokens=hidden_states.shape[0],
                    k_local=self.mlp.down_proj.input_size_per_partition,
                    norm_module=self.next_input_layernorm,
                    residual=residual,
                )
                if fuse_mlp_allreduce
                else None
            )
            mlp_staging = (
                get_fused_ar_staging_view(
                    num_tokens=hidden_states.shape[0],
                    hidden=self.hidden_size,
                    dtype=hidden_states.dtype,
                )
                if mlp_fused_norm is None
                and fuse_mlp_allreduce
                and apply_rmsnorm_fused_ar(hidden_states.shape[0])
                and hidden_states.shape[0] != 0
                else None
            )
            hidden_states = self.mlp(
                hidden_states,
                forward_batch=forward_batch,
                output_tensor=mlp_staging,
                fused_norm=mlp_fused_norm,
            )
        if _is_npu and get_cmo_stream():
            wait_cmo_stream()

        if is_normed(hidden_states):
            # down_proj fused the next layer's input_layernorm: hidden_states is
            # already all-reduced and normed, residual already marked. Neither
            # the fusion handoff flag nor postprocess_layer applies.
            pass
        elif fuse_mlp_allreduce:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
        return hidden_states, residual

    def _maybe_fused_norm(self, *, num_tokens, k_local, norm_module, residual):
        """A FusedNormSpec when the GEMM-fused path can take this call, else None.

        norm_module is None at the last layer's MLP side (no next layer to fuse
        into), and residual is None on the very first prepare_attn, before any
        residual exists -- both mean no fusion.

        k_local is the linear's own input_size_per_partition, i.e. the width of
        the activation the GEMM will actually see: for o_proj that is
        total_num_heads * head_dim / attn_tp_size, which is exactly
        num_heads * head_dim, the width of attn_output.
        """
        from sglang.srt.layers.gemm_ar_rmsnorm_fused import is_gemm_ar_eligible
        from sglang.srt.layers.linear import FusedNormSpec

        if norm_module is None or residual is None:
            return None
        if not apply_gemm_ar_rmsnorm_fused(num_tokens):
            return None
        # o_proj is built with attn_tp_size while down_proj uses tp_size. The
        # kernel's collective spans the symm-mem group, i.e. the full TP group,
        # so the two must coincide -- which they do because this flag disables
        # dp-attention at startup. Assert rather than assume: a divergence would
        # make the M-alignment gate check the wrong world size.
        parallel = get_parallel()
        assert parallel.attn_tp_size == parallel.tp_size, (
            f"--enable-gemm-ar-rmsnorm-fused: attn_tp_size "
            f"{parallel.attn_tp_size} != tp_size {parallel.tp_size}"
        )
        if not is_gemm_ar_eligible(
            m=num_tokens,
            n=self.hidden_size,
            k=k_local,
            world_size=parallel.tp_size,
        ):
            return None
        return FusedNormSpec(
            norm_weight=norm_module.weight,
            eps=norm_module.variance_epsilon,
            residual=residual,
        )


class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = get_stream("alt") if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )
        self._fuse_ar_norm = False
        # self.layers is a full-length ModuleList whose out-of-range slots are
        # PPMissingLayer (utils/common.py: make_layers), so walk only the range
        # this rank actually owns -- and stop one short, since the last layer
        # has no next norm to fuse. Gated on the flag itself, not left
        # unconditional: this loop pokes a private attribute on another
        # layer's module from the outside, and a disabled flag should not
        # touch the module graph at all.
        if get_exec().comm.enable_gemm_ar_rmsnorm_fused:
            for i in range(self.start_layer, self.end_layer - 1):
                self.layers[i]._next_input_layernorm = (
                    self.layers[i + 1].input_layernorm,
                )


class Qwen3ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            if is_fused_ar_buffer_view(forward_batch.hidden_states):
                # The fused GEMM+AR+norm path may have marked this tensor
                # `is_normed` (mark_normed / is_normed in
                # layers/gemm_ar_rmsnorm_fused.py): "already all-reduced and
                # normed" is a property of the VALUE, not of the storage, but
                # `clone()` only copies storage -- it does not carry Python
                # attributes -- so the marker would otherwise be silently
                # dropped here, and the next split's prepare_attn would
                # re-normalize already-normalized data (a double norm). Read
                # the marker before the reassignment below replaces
                # `forward_batch.hidden_states`, then re-apply it to the clone
                # only if the source actually carried it: marking
                # unconditionally would cause a missed norm on inputs that
                # were never fused in the first place. (`forward_batch.residual`
                # is never cloned, so its own `_mega_residual_shard` marker
                # needs no equivalent fix here.)
                was_normed = is_normed(forward_batch.hidden_states)
                forward_batch.hidden_states = forward_batch.hidden_states.clone()
                if was_normed:
                    mark_normed(forward_batch.hidden_states)
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if name == "model.embed_tokens.weight":
                if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        if hasattr(self.model.embed_tokens, "weight"):
            del self.model.embed_tokens.weight
        if hasattr(self.lm_head, "weight"):
            del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        # SGLang captures "before layer i". To capture the hidden state after target
        # layer `k` (HF-style), we capture before layer `k + 1`.
        self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Qwen3ForCausalLM

"""Install the CPU ops into vLLM's Kimi K3 (nvidia) implementation."""

import os

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from . import ops

logger = init_logger(__name__)
_DEBUG_KDA = bool(os.environ.get("K3CPU_DEBUG_KDA"))


def _patch_attn_res() -> None:
    import vllm.models.kimi_k3.nvidia.model as model

    model.attn_res = ops.attn_res

    # These projections are read via .weight directly; CPU packing would drop it.
    cls = model.KimiLinearModel
    orig_init = cls.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for name, mod in self.named_modules():
            if name.endswith(("res_proj", "routed_expert_up_proj")):
                mod.skip_weight_relayout = True

    cls.__init__ = init


def _patch_mla() -> None:
    import vllm.models.kimi_k3.nvidia.mla as mla

    mla.fused_q_kv_rmsnorm = ops.fused_q_kv_rmsnorm
    mla.fused_mla_key_concat_kv_cache_insert = ops.fused_mla_key_concat_kv_cache_insert
    mla.fused_mla_decode_q_concat_kv_cache_insert = (
        ops.fused_mla_decode_q_concat_kv_cache_insert
    )

    # process_weights_after_loading reads kv_b_proj.weight; CPU packing drops it.
    cls = mla.MultiHeadLatentAttention
    orig_init = cls.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.kv_b_proj.skip_weight_relayout = True
        # The side-stream gate path slices the (packed-away) raw weight.
        self._gate_events = None
        self.aux_stream = None

    cls.__init__ = init

    if os.environ.get("K3CPU_MLA_DECODE") == "exact":
        orig_attention = cls._attention

        def attention(self, positions, q, kv_c_normed, k_pe, attn_out):
            md = get_forward_context().attn_metadata
            m = md[self.layer_name] if md is not None else None
            if m is None or m.num_decode_tokens == 0:
                return orig_attention(self, positions, q, kv_c_normed, k_pe, attn_out)
            if m.num_decode_tokens != m.num_actual_tokens:
                raise NotImplementedError("K3CPU_MLA_DECODE=exact: decode-only batches")
            _mla_decode_exact(self, q, kv_c_normed, k_pe, attn_out, m)

        cls._attention = attention


def _mla_decode_exact(self, q, kv_c_normed, k_pe, attn_out, m) -> None:
    """Non-absorbed fp32 MLA decode: rebuild K/V from the latent cache through
    kv_b_proj. Reference for the absorbed bf16 path."""
    fc = get_forward_context()
    n = m.num_actual_tokens
    ops._write_latent(kv_c_normed[:n], k_pe[:n], self.kv_cache, fc.slot_mapping[self.layer_name][:n])
    flat = self.kv_cache.view(-1, self.kv_cache.shape[-1])
    H, P, V, L = self.num_local_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank
    bs = self.kv_cache.shape[1]
    wkv = self.kv_b_proj.weight.float()
    for r in range(n):
        slen = int(m.decode.seq_lens[r])
        pos = torch.arange(slen)
        rows = m.decode.block_table[r][pos // bs].long() * bs + pos % bs
        lat = flat[rows].float()
        kvn = torch.nn.functional.linear(lat[:, :L], wkv).view(slen, H, P + V)
        k = torch.cat([kvn[..., :P], lat[:, None, L:].expand(-1, H, -1)], dim=-1)
        sc = torch.einsum("hd,shd->hs", q[r].float(), k) * self.scale
        o = torch.einsum("hs,shd->hd", torch.softmax(sc, -1), kvn[..., P:])
        attn_out[r] = o.reshape(H * V).to(attn_out.dtype)


def _kda_forward_cpu(self, mixed_qkv, g1, g2, beta, core_attn_out) -> None:
    """KimiK3DeltaAttention._forward for CPU: conv + gate + recurrence per sequence
    (no speculative decode)."""
    from vllm.forward_context import get_forward_context
    from vllm.models.kimi_k3.nvidia import kda as kda_mod

    md = get_forward_context().attn_metadata
    if md is None:
        return
    m = md.get(self.prefix)
    if m is None:
        return
    if m.num_spec_decodes > 0:
        raise NotImplementedError("k3cpu: speculative decode is not supported")

    n = m.num_actual_tokens
    conv_state, recurrent_state, *_ = self.kv_cache
    if not kda_mod.is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    weight = self.conv1d.weight.view(self.conv1d.weight.size(0), -1).float()
    H, D, P = self.local_num_heads, self.head_dim, self.local_projection_size
    indices = m.non_spec_state_indices_tensor
    if m.num_prefills > 0:
        qsl = m.non_spec_query_start_loc.tolist()
        init = m.has_initial_state.tolist()
    else:
        qsl = list(range(n + 1))
        init = [True] * n
    scale = D**-0.5
    for i in range(len(qsl) - 1):
        s, e = qsl[i], qsl[i + 1]
        if e <= s:
            continue
        slot = int(indices[i])
        st = conv_state[slot].float() if init[i] else None
        y, st_new = ops.kda_conv(mixed_qkv[s:e].float(), st, weight)
        conv_state[slot].copy_(st_new)
        q, k, v = (t.reshape(e - s, H, D) for t in y.split(P, dim=-1))
        q = ops.l2norm(q) * scale
        k = ops.l2norm(k)
        gate = ops.kda_gate(g1[0, s:e].float(), self.A_log, self.dt_bias, self.gate_lower_bound)
        b = torch.sigmoid(beta[0, s:e].float())
        S = recurrent_state[slot].float() if init[i] else q.new_zeros(H, D, D)
        if _DEBUG_KDA:
            print(f"DBG kda {self.prefix} pre={m.num_prefills} seq={i} tok={s}:{e} slot={slot} "
                  f"init={init[i]} |S|={float(S.norm()):.4g} |conv|={float(conv_state[slot].float().norm()):.4g} "
                  f"conv={tuple(conv_state.shape)}/{conv_state.dtype}/{conv_state.stride()} "
                  f"rec={tuple(recurrent_state.shape)}/{recurrent_state.dtype}", flush=True)
        o = ops.kda_recurrence(q, k, v, gate, b, S)
        recurrent_state[slot].copy_(S)
        core_attn_out[0, s:e] = o.to(core_attn_out.dtype)
    core_attn_out.copy_(self.o_norm(core_attn_out, g2))


def _mamba_get_block_table_tensor(block_table, seq_lens, kv_cache_spec, mamba_cache_mode):
    """kda_metadata._get_aligned_state_indices_kernel: pick the state slots that
    hold each request's last token."""
    if mamba_cache_mode in ("all", "none"):
        return block_table
    n_slots = 1 + kv_cache_spec.num_speculative_blocks
    first = ((seq_lens.long() - 1) // kv_cache_spec.block_size).clamp(min=0)
    cols = first[:, None] + torch.arange(n_slots, device=first.device)
    cols = cols.clamp(max=block_table.shape[1] - 1)
    return torch.gather(block_table, 1, cols).to(block_table.dtype)


def _zero_new_kda_state(self, scheduler_output) -> None:
    """The scheduler only lists new attention blocks for zeroing, never Mamba
    state blocks, and a 1-token prompt runs as a decode that continues from the
    slot's state. Zero the KDA state of requests that start from scratch."""
    from vllm.v1.kv_cache_interface import MambaSpec

    groups = [
        (g, grp.layer_names)
        for g, grp in enumerate(self.kv_cache_config.kv_cache_groups)
        if isinstance(grp.kv_cache_spec, MambaSpec)
    ]
    ctx = self.compilation_config.static_forward_context
    for req in scheduler_output.scheduled_new_reqs:
        if req.num_computed_tokens != 0:
            continue
        for g, names in groups:
            idx = torch.tensor(req.block_ids[g], dtype=torch.long)
            if idx.numel() == 0:
                continue
            for name in names:
                for t in ctx[name].kv_cache:
                    t.index_fill_(0, idx, 0)


def _patch_kda() -> None:
    import vllm._custom_ops as cops
    import vllm.models.kimi_k3.nvidia.kda_metadata as kda_md
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    KimiK3DeltaAttention._forward = _kda_forward_cpu
    kda_md._mamba_get_block_table_tensor = _mamba_get_block_table_tensor
    orig_update = CPUModelRunner._update_states

    def update_states(self, scheduler_output):
        _zero_new_kda_state(self, scheduler_output)
        return orig_update(self, scheduler_output)

    CPUModelRunner._update_states = update_states

    # K3 keeps fp32 conv weights; the CPU prepack (for GDN) is bf16-only.
    orig_pack = cops.causal_conv1d_weight_pack

    def conv_pack(w: torch.Tensor) -> torch.Tensor:
        return w if w.dtype == torch.float32 else orig_pack(w)

    cops.causal_conv1d_weight_pack = conv_pack


def _patch_moe() -> None:
    import vllm.model_executor.layers.quantization.mxfp4 as mxfp4
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    from .experts import K3CpuExpertsMxfp4Situ

    orig_select = mxfp4.select_deepseek_v4_mxfp4_moe_backend
    orig_convert = mxfp4.convert_weight_to_mxfp4_moe_kernel_format

    def select(moe):
        if moe.activation == MoEActivation.SITU:
            return Mxfp4MoeBackend.CPU, K3CpuExpertsMxfp4Situ
        return orig_select(moe)

    def convert(*, mxfp4_backend, activation=None, **kw):
        if mxfp4_backend == Mxfp4MoeBackend.CPU and activation == MoEActivation.SITU:
            return K3CpuExpertsMxfp4Situ.convert_weights(**kw)
        return orig_convert(mxfp4_backend=mxfp4_backend, activation=activation, **kw)

    mxfp4.select_deepseek_v4_mxfp4_moe_backend = select
    mxfp4.convert_weight_to_mxfp4_moe_kernel_format = convert
    # The CPU experts compute in fp32, so an fp32 model (reference runs) is fine.
    mxfp4.Mxfp4Config.get_supported_act_dtypes = classmethod(
        lambda cls: [torch.bfloat16, torch.float32]
    )


def apply() -> None:
    import os

    _patch_attn_res()
    _patch_mla()
    _patch_kda()
    _patch_moe()
    if os.environ.get("K3CPU_DEBUG_MLA"):
        from . import debug_mla

        debug_mla.install()
    logger.info("k3cpu: Kimi K3 CPU ops installed")

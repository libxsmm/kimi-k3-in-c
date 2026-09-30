"""Torch reference implementations of the Kimi K3 GPU-only ops.

Each function mirrors the semantics of the Triton/CUDA kernel it replaces
(see the matching file under vllm/models/kimi_k3/nvidia/).
"""

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------- AttnRes ----


def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    """ops/attn_res.py::_attn_res_kernel: prefix += delta (in place, rounded to
    the prefix dtype), optional block write, softmax mix over the stored blocks
    plus the prefix, optional output RMSNorm."""
    p = prefix.float()
    if delta is not None:
        prefix.copy_(p + delta.float())
        p = prefix.float()
    if block_write_idx >= 0:
        blocks[:, block_write_idx].copy_(p)
    if num_blocks == 0:
        mixed = p
    else:
        w = norm_weight.float() * qk_weight.float()
        vals = torch.cat([blocks[:, :num_blocks].float(), p[:, None]], dim=1)
        rstd = torch.rsqrt(vals.pow(2).mean(-1) + eps)
        scores = torch.softmax((vals * w).sum(-1) * rstd, dim=-1)
        mixed = (scores[..., None] * vals).sum(1)
    if output_norm_weight is not None:
        mixed = mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + output_norm_eps)
        mixed = mixed * output_norm_weight.float()
    return mixed.to(prefix.dtype)


# --------------------------------------------------------------------- KDA ----


def kda_gate(
    raw_g: torch.Tensor,  # [L, H, K] float32
    A_log: torch.Tensor,  # [H]
    dt_bias: torch.Tensor,  # [H*K]
    lower_bound: float | None,
) -> torch.Tensor:
    L, H, K = raw_g.shape
    g = raw_g + dt_bias.float().view(H, K)
    a = torch.exp(A_log.float()).view(H, 1)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * g)
    return -a * F.softplus(g, threshold=20.0)


def kda_conv(
    x: torch.Tensor,  # [L, C] float32
    state: torch.Tensor | None,  # [C, W-1] float32 or None (zeros)
    weight: torch.Tensor,  # [C, W] float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv + SiLU; returns (y [L, C], new state [C, W-1])."""
    C, W = weight.shape
    if state is None:
        state = x.new_zeros(C, W - 1)
    xx = torch.cat([state.t(), x], dim=0)  # [W-1+L, C]
    win = xx.unfold(0, W, 1)  # [L, C, W]
    y = F.silu((win * weight).sum(-1))
    return y, xx[-(W - 1) :].t().contiguous()


def kda_recurrence(
    q: torch.Tensor,  # [L, H, K] float32, l2-normalised and scaled
    k: torch.Tensor,  # [L, H, K] float32, l2-normalised
    v: torch.Tensor,  # [L, H, V] float32
    gate: torch.Tensor,  # [L, H, K] log-decay
    beta: torch.Tensor,  # [L, H] after sigmoid
    S: torch.Tensor,  # [H, V, K] float32, updated in place
) -> torch.Tensor:
    """fused_recurrent_kda_packed_decode_kernel, one token at a time."""
    out = torch.empty_like(v)
    decay = torch.exp(gate)
    for t in range(q.shape[0]):
        S.mul_(decay[t][:, None, :])
        u = (v[t] - torch.einsum("hvk,hk->hv", S, k[t])) * beta[t][:, None]
        S.add_(u[:, :, None] * k[t][:, None, :])
        out[t] = torch.einsum("hvk,hk->hv", S, q[t])
    return out


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


# ------------------------------------------------------------------- MXFP4 ----

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def mxfp4_dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[R, C/2] uint8 (low nibble = even element) + [R, C/32] E8M0 -> [R, C] fp32.
    Scale 255 (NaN by spec) decodes to 0, as in the C engine."""
    R = packed.shape[0]
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(R, -1).long()
    vals = _E2M1[codes].view(R, scale.shape[1], -1)
    e = scale.to(torch.int32)
    s = torch.where(e == 255, 0.0, torch.exp2((e - 127).float()))
    return (vals * s[..., None]).view(R, -1)


def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    d = x.shape[-1] // 2
    g, u = x[..., :d], x[..., d:]
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return a * u


def moe_mxfp4_situ(
    x: torch.Tensor,  # [M, K]
    w13: torch.Tensor,  # [E, 2I, K/2] uint8
    w13_s: torch.Tensor,  # [E, 2I, K/32] uint8
    w2: torch.Tensor,  # [E, K, I/2] uint8
    w2_s: torch.Tensor,  # [E, K, I/32] uint8
    topk_w: torch.Tensor,  # [M, topk]
    topk_ids: torch.Tensor,  # [M, topk]
    beta: float,
    linear_beta: float | None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    xf = x.float()
    out = torch.zeros(x.shape, dtype=torch.float32)
    ids = topk_ids.long()
    if expert_map is not None:
        ids = expert_map[ids].long()
    for e in torch.unique(ids).tolist():
        if e < 0:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = xf[tok] @ mxfp4_dequant(w13[e], w13_s[e]).t()
        h = situ_and_mul(h, beta, linear_beta)
        y = h @ mxfp4_dequant(w2[e], w2_s[e]).t()
        out.index_add_(0, tok, y * topk_w[tok, slot].float()[:, None])
    return out.to(x.dtype)


# --------------------------------------------------------------------- MLA ----


def fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
    def rms(x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * w.float()).to(x.dtype)

    return rms(qr, q_weight), rms(kv, kv_weight)


def _write_latent(kv_c_normed, k_pe, kv_cache, slot_mapping):
    flat = kv_cache.view(-1, kv_cache.shape[-1])
    L = kv_c_normed.shape[-1]
    slots = slot_mapping.flatten().long()
    ok = slots >= 0
    slots, kv_c_normed, k_pe = slots[ok], kv_c_normed[ok], k_pe.reshape(k_pe.shape[0], -1)[ok]
    flat[slots, :L] = kv_c_normed.to(flat.dtype)
    flat[slots, L:] = k_pe.to(flat.dtype)


def fused_mla_key_concat_kv_cache_insert(
    q, k_nope, k_pe, kv_c_normed, kv_cache, slot_mapping, positions=None, cos_sin_cache=None
):
    assert cos_sin_cache is None, "Kimi K3 MLA is NoPE-only"
    _write_latent(kv_c_normed, k_pe, kv_cache, slot_mapping)
    pe = k_pe.reshape(k_pe.shape[0], 1, -1).expand(-1, k_nope.shape[1], -1)
    return torch.cat([k_nope, pe.to(k_nope.dtype)], dim=-1)


def fused_mla_decode_q_concat_kv_cache_insert(
    ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping, positions=None, cos_sin_cache=None, **kw
):
    assert cos_sin_cache is None, "Kimi K3 MLA is NoPE-only"
    assert not kw.get("ds_mla") and kw.get("q_scale_inv") is None, "CPU: bf16 KV cache only"
    _write_latent(kv_c_normed, k_pe, kv_cache, slot_mapping)
    return torch.cat([ql_nope, q_pe.to(ql_nope.dtype)], dim=-1)

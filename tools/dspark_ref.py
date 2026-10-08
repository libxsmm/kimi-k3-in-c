#!/usr/bin/env python3
"""PyTorch reference for the DSpark draft block (src/model/k3_dspark.c).

Recomputes the first drafted block from the engine's K3_DSPARK_DUMP files: the target
taps of the prompt, the anchor/position, and checks the draft's final hidden state, and
the Markov-chained tokens given the engine's base logits. MLA is computed in the plain
(non-absorbed) form, YaRN rope as vLLM's DeepseekScalingRotaryEmbedding.

  python tools/dspark_ref.py /path/to/Kimi-K3-DSpark <dump prefix>
"""
import json
import math
import sys

import numpy as np
import torch
from safetensors import safe_open


def yarn_inv_freq(dim, base, factor, orig, bfast, bslow):
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    extra = 1.0 / pos_freqs
    inter = 1.0 / (factor * pos_freqs)

    def corr_dim(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(bfast)), 0)
    high = min(math.ceil(corr_dim(bslow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return inter * (1 - mask) + extra * mask


def rope(x, pos, inv_freq, mag):
    """interleaved (gptj) rotation of x[..., rope] at positions pos (broadcast on dim 0)"""
    f = pos[:, None].float() * inv_freq[None, :]
    c, s = torch.cos(f) * mag, torch.sin(f) * mag
    while c.dim() < x.dim():
        c, s = c.unsqueeze(1), s.unsqueeze(1)
    a, b = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = b * c + a * s
    return out


def rms(x, w, eps):
    return x * torch.rsqrt((x.double() ** 2).mean(-1, keepdim=True).float() + eps) * w


def main():
    mdir, pre = sys.argv[1], sys.argv[2]
    cfg = json.load(open(f"{mdir}/config.json"))
    W = {}
    with safe_open(f"{mdir}/model.safetensors", "pt") as f:
        for k in f.keys():
            W[k] = f.get_tensor(k).float()
    H, NH = cfg["hidden_size"], cfg["num_attention_heads"]
    NOPE, ROPE, VH = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    KL, eps = cfg["kv_lora_rank"], cfg["rms_norm_eps"]
    rp = cfg["rope_parameters"]
    factor = rp.get("factor", 1.0)
    inv_freq = yarn_inv_freq(ROPE, rp["rope_theta"], factor, rp["original_max_position_embeddings"],
                             rp.get("beta_fast", 32), rp.get("beta_slow", 1))

    def ym(s, m):
        return 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0

    mag = ym(factor, rp.get("mscale", 1)) / ym(factor, rp.get("mscale_all_dim", 0))
    scale = (NOPE + ROPE) ** -0.5 * ym(factor, rp.get("mscale_all_dim", 0)) ** 2
    ntgt = len(cfg["target_layer_ids"])

    anchor, pos, nq = np.fromfile(pre + ".meta", dtype=np.int32)
    taps = torch.from_numpy(np.fromfile(pre + ".taps", dtype=np.float32)).view(-1, ntgt * H)
    n = taps.shape[0]
    assert n == pos, f"taps cover {n} positions, anchor at {pos}"
    hs_c = torch.from_numpy(np.fromfile(pre + ".hs", dtype=np.float32)).view(nq, H)
    base = torch.from_numpy(np.fromfile(pre + ".base", dtype=np.float32)).view(nq, -1)
    tok_c = np.fromfile(pre + ".tok", dtype=np.int32)

    c = rms(taps @ W["context_proj.weight"].T, W["context_norm.weight"], eps)
    cpos = torch.arange(n)
    qpos = torch.arange(pos, pos + nq)
    x = W["embed_tokens.weight"][[anchor] + [cfg["mask_token_id"]] * (nq - 1)]
    resid, m = x, None
    for l in range(cfg["num_hidden_layers"]):
        p = f"layers.{l}."
        if m is not None:
            resid = resid + m
        hs = rms(resid, W[p + "input_layernorm.weight"], eps)
        a = p + "self_attn."
        q = rms(hs @ W[a + "q_a_proj.weight"].T, W[a + "q_a_layernorm.weight"], eps) @ W[a + "q_b_proj.weight"].T
        q = q.view(nq, NH, NOPE + ROPE)
        q_nope, q_pe = q[..., :NOPE], rope(q[..., NOPE:], qpos, inv_freq, mag)
        kv_ctx = c @ W[a + "kv_a_proj_with_mqa.weight"].T
        kv_blk = hs @ W[a + "kv_a_proj_with_mqa.weight"].T
        lat = rms(torch.cat([kv_ctx[:, :KL], kv_blk[:, :KL]]), W[a + "kv_a_layernorm.weight"], eps)
        kpe = torch.cat([rope(kv_ctx[:, KL:], cpos, inv_freq, mag), rope(kv_blk[:, KL:], qpos, inv_freq, mag)])
        kvb = (lat @ W[a + "kv_b_proj.weight"].T).view(-1, NH, NOPE + VH)
        k_nope, v = kvb[..., :NOPE], kvb[..., NOPE:]
        sc = torch.einsum("qhd,shd->hqs", q_nope, k_nope) + torch.einsum("qhd,sd->hqs", q_pe, kpe)
        pr = torch.softmax(sc * scale, dim=-1)
        ao = torch.einsum("hqs,shd->qhd", pr, v).reshape(nq, NH * VH)
        resid = resid + ao @ W[a + "o_proj.weight"].T
        hs = rms(resid, W[p + "post_attention_layernorm.weight"], eps)
        g = hs @ W[p + "mlp.gate_proj.weight"].T
        u = hs @ W[p + "mlp.up_proj.weight"].T
        m = (torch.nn.functional.silu(g) * u) @ W[p + "mlp.down_proj.weight"].T
    hs_r = rms(resid + m, W["final_norm.weight"], eps)

    cos = torch.nn.functional.cosine_similarity(hs_r, hs_c, dim=-1)
    rel = (hs_r - hs_c).abs().max() / hs_r.abs().max()
    print("final hidden per position: cos", [f"{v:.6f}" for v in cos.tolist()], f"max rel diff {rel:.3e}")

    prev, toks = int(anchor), []
    for i in range(nq):
        bias = W["markov_head.markov_w2.weight"] @ W["markov_head.markov_w1.weight"][prev]
        prev = int(torch.argmax(base[i] + bias))
        toks.append(prev)
    print("tokens engine   ", tok_c.tolist())
    print("tokens reference", toks, "(Markov chain on the engine's base logits)")
    ok = cos.min() > 0.999 and toks == tok_c.tolist()
    print("DSPARK REFERENCE", "MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

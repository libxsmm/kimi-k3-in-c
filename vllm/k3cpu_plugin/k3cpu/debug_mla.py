"""Debug: print MLA decode addressing (slots written vs addresses read)."""

import vllm.v1.attention.backends.mla.amx_mla as amx

import k3cpu.ops as ops

_orig_insert = ops.fused_mla_decode_q_concat_kv_cache_insert
_orig_pinsert = ops.fused_mla_key_concat_kv_cache_insert
_orig_mqa = amx.AMXMLAImpl.forward_mqa


def pinsert(q, k_nope, k_pe, kv_c_normed, kv_cache, slot_mapping, *a, **kw):
    print("DBG prefill insert slots", slot_mapping.tolist(), "q", tuple(q.shape), flush=True)
    return _orig_pinsert(q, k_nope, k_pe, kv_c_normed, kv_cache, slot_mapping, *a, **kw)


def insert(ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping, **kw):
    print("DBG insert cache", tuple(kv_cache.shape), "slots", slot_mapping.tolist(), flush=True)
    return _orig_insert(ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping, **kw)


def mqa(self, q, cache, md, layer):
    import torch

    d = md.decode
    n = int(d.seq_lens[0])
    print("DBG mqa cache", tuple(cache.shape), "block_table", d.block_table[0, :4].tolist(),
          "seq_len", n, "req_to_token", d.req_to_token[0, :n].tolist(), flush=True)
    o, lse = _orig_mqa(self, q, cache, md, layer)
    kv = cache.view(-1, cache.shape[-1])[d.req_to_token[0, :n].long()].float()
    qq = q[0].float()
    p = torch.softmax(qq @ kv.t() * self.scale, dim=-1)
    ref = p @ kv[:, : self.kv_lora_rank]
    print("DBG mqa maxdiff", float((o[0].float() - ref).abs().max()),
          "refmax", float(ref.abs().max()), flush=True)
    return o, lse


def install():
    import vllm.models.kimi_k3.nvidia.mla as mla

    mla.fused_mla_decode_q_concat_kv_cache_insert = insert
    mla.fused_mla_key_concat_kv_cache_insert = pinsert
    amx.AMXMLAImpl.forward_mqa = mqa

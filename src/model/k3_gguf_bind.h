/* SPDX-License-Identifier: Apache-2.0 */
/* k3_gguf_bind.h - bind a GGUF Kimi K3 (llama.cpp naming) into the engine's weight structs.
 *
 * The same K3LayerBind / K3ModelBind the safetensors path fills, so the decoder is shared.
 * What differs from the released checkpoint, and how it is undone at load:
 *   <x>_res_norm * <x>_res_proj  ship fused as <x>_res_score: bound as the norm, with an
 *                                all-ones proj, so the engine's fold reproduces it exactly
 *   A_log                        ships as ssm_a = -exp(A_log[:H]); A_log = log(-ssm_a)
 *   kv_b_proj                    ships split, attn_k_b per head TRANSPOSED [kv_lora][nope]
 *                                plus attn_v_b [v_head][kv_lora]; rebuilt as fp32 kv_b
 *   b_proj (ssm_beta)            ships F32 while the rest of the layer is Q8_0, which is
 *                                why K3KdaW carries b_wdt
 *   routed experts               one [n_experts][rows][in] tensor per matrix, IQ2_XS or
 *                                IQ3_XXS, held resident (this rank's rows only under TP)
 */
#ifndef K3_GGUF_BIND_H
#define K3_GGUF_BIND_H

#include "k3_bind.h"
#include "k3_gguf.h"

int k3_gguf_bind_layer(const K3Gguf *g, const K3Cfg *c, int L, K3LayerBind *b);
int k3_gguf_bind_model(const K3Gguf *g, const K3Cfg *c, K3ModelBind *m);

/* Bytes the layer's trunk tensors take as bound (this rank's share). -1 if unbindable. */
int64_t k3_gguf_layer_bytes(const K3Gguf *g, const K3Cfg *c, int L);

typedef struct {
    K3ExpertSrc     src;          /* first member: the engine sees a K3ExpertSrc */
    int             n_layers, n_experts;
    unsigned char **base;         /* [layer] -> [expert][w1 rows | w3 rows | w2 rows] */
    size_t         *ebytes;       /* [layer] bytes per expert */
    size_t         *o3, *o2;      /* [layer] offsets of w3 and w2 inside an expert */
    size_t         *so;           /* [layer][3] MXFP4 scale offsets inside an expert */
    unsigned char  *qt;           /* [layer][3]: K3_EQ_* of w1, w3, w2 */
    int             ilv;          /* K3_MX_* layout of MXFP4 experts */
    double          bytes;
} K3GgufExperts;

/* Load every routed expert of layers [0, nl) resident. 0 = ok. */
int  k3_gguf_experts_init(K3GgufExperts *x, const K3Gguf *g, const K3Cfg *c, int nl);
void k3_gguf_experts_free(K3GgufExperts *x);
/* Bytes k3_gguf_experts_init will hold for this rank, without loading. */
double k3_gguf_experts_bytes(const K3Gguf *g, const K3Cfg *c, int nl);

#endif /* K3_GGUF_BIND_H */

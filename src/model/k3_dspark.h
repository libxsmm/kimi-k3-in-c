/* SPDX-License-Identifier: Apache-2.0 */
/* k3_dspark.h - the DSpark draft model (Inferact/Kimi-K3-DSpark) for speculative decode.
 *
 * DSpark drafts a block of N tokens in ONE forward pass: a 5-layer dense MLA stack whose
 * keys/values for already-verified positions come from the target's hidden states (the
 * residual stream after target layers target_layer_ids), and whose queries are the
 * pending (anchor) token followed by N-1 mask tokens, attending non-causally within the
 * block. A low-rank Markov head then adds a dependency on the previously drafted token
 * while the block is sampled left to right (greedy here).
 *
 * Reference: vLLM models/kimi_k3/nvidia/dspark_mla.py, v1/worker/gpu/spec_decode/dspark.
 * The draft only proposes; the target verifies every token, so the draft needs no
 * bit-exactness: it computes in fp32 from bf16 weights. Under tensor parallelism every
 * rank runs the draft redundantly (identical results, no communication) except for the
 * target's vocab-sliced lm_head, whose logits are gathered. */
#ifndef K3_DSPARK_H
#define K3_DSPARK_H

#include <stddef.h>
#include <stdint.h>

#define K3_DSPARK_MAXL 8
#define K3_DSPARK_MAXT 8         /* tokens per multi-token GEMV pass */

typedef struct {
    const uint16_t *in_norm, *post_norm, *q_a, *q_a_norm, *q_b, *kv_a, *kv_a_norm, *kv_b,
                   *o, *gate, *up, *down;
} K3DSparkLayer;

typedef struct {
    /* geometry */
    int H, nl, ntgt, tgt[K3_DSPARK_MAXL], nh, q_lora, kv_lora, nope, rope, vh, inter;
    int vocab, mrank, mask_id;
    float eps, scale, rope_mag;
    float inv_freq[64];          /* rope, rope/2 entries */
    /* weights, bf16, pointing into blob */
    const uint16_t *ctx_proj, *ctx_norm, *final_norm, *embed, *mw1, *mw2;
    K3DSparkLayer L[K3_DSPARK_MAXL];
    void  *blob;
    size_t blob_bytes;
    /* context KV per layer: latent [cap][kv_lora] (normed), k_pe [cap][rope] (roped) */
    float *lat, *kpe;
    int    cap;
    /* work buffers */
    float *wk;
    size_t wk_floats;
    /* stats */
    long   steps, drafted;
    double t_ctx, t_draft;
    /* K3_DSPARK_DUMP=<prefix>: the first block's final hidden, base logits and tokens */
    const char *dump;
} K3DSpark;

/* Load dir/config.json and dir/model.safetensors. vocab/hidden must match the target. */
int  k3_dspark_open(K3DSpark *d, const char *dir, int vocab, int hidden, int cap);
void k3_dspark_close(K3DSpark *d);

/* Feed n verified positions pos0..pos0+n-1: taps is [n][ntgt][H], the target's residual
 * stream after layers d->tgt[] at those positions (in that order). */
int  k3_dspark_context(K3DSpark *d, const float *taps, int n, int pos0);

/* Draft nq tokens following `anchor`, which sits at position pos (all positions < pos
 * have been fed through k3_dspark_context). lm_head is the target's (wdt K3_WQ8_0,
 * K3_WBF16 or K3_WF32; under k3_tp.local only this rank's vocab rows). out[i] is the
 * proposal for position pos + 1 + i. Returns nq, or -1 on error. */
int  k3_dspark_propose(K3DSpark *d, int anchor, int pos, int nq, const void *lm_head,
                       int lm_wdt, int *out);

#endif /* K3_DSPARK_H */

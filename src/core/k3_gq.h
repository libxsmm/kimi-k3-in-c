/* SPDX-License-Identifier: Apache-2.0 */
/* k3_gq.h - GEMV kernels for the GGUF (ggml) weight formats of the Unsloth Kimi K3 quants.
 *
 *   Q8_0     the trunk         34 B / 32 weights:  fp16 d, int8 q[32];  w = q*d
 *   IQ2_XS   routed experts    74 B / 256 weights: 9-bit index into iq2xs_grid + 7 sign bits
 *                                                   per 8, 4-bit scale per 16
 *   IQ3_XXS  13 expert tensors 98 B / 256 weights: 8-bit index into iq3xxs_grid per 4,
 *                                                   7 sign bits per 8 and a 4-bit scale per 32
 *
 * The *_rows kernels form every weight exactly as ggml's dequantize_row_* does and use the
 * engine's fp32 GEMV scheme, so a fused row equals dequantise-then-k3_matmul to the bit.
 * The *_rows_q8 kernels take int8 activations (k3_gq_quant_x, per 256 like ggml's q8_K)
 * and use integer dot products: faster, deterministic, NOT bit-identical to fp32. */
#ifndef K3_GQ_H
#define K3_GQ_H

#include <stddef.h>
#include <stdint.h>

/* ggml type ids, as stored in GGUF tensor infos */
enum { K3_GG_F32 = 0, K3_GG_F16 = 1, K3_GG_Q8_0 = 8, K3_GG_IQ2_XS = 17,
       K3_GG_IQ3_XXS = 18, K3_GG_BF16 = 30, K3_GG_MXFP4 = 39 };

#define K3_GQ_QK 256

/* Bytes of one row of `in` weights in ggml type `t`, or 0 if `in` does not fit its blocks
 * or the type is not one of the above. */
static inline size_t k3_gq_row_bytes(int t, int64_t in)
{
    switch (t) {
    case K3_GG_F32:     return (size_t)in * 4;
    case K3_GG_F16:
    case K3_GG_BF16:    return (size_t)in * 2;
    case K3_GG_Q8_0:    return in % 32 ? 0 : (size_t)in / 32 * 34;
    case K3_GG_IQ2_XS:  return in % K3_GQ_QK ? 0 : (size_t)in / K3_GQ_QK * 74;
    case K3_GG_IQ3_XXS: return in % K3_GQ_QK ? 0 : (size_t)in / K3_GQ_QK * 98;
    case K3_GG_MXFP4:   return in % 32 ? 0 : (size_t)in / 32 * 17;
    default:            return 0;
    }
}

/* y[o] = W[o] . x for o in [o0, o1); W points at row 0. in must be a block multiple. */
void k3_q80_rows(float *y, const float *x, const void *W, int in, int o0, int o1);
/* The same rows for T tokens, each row read once: Y[t * ldy + o] = W[o] . X[t * ldx],
 * per token bit-identical to k3_q80_rows. */
void k3_q80_rows_T(float *Y, int ldy, const float *X, int ldx, int T, const void *W, int in,
                   int o0, int o1);
/* K3_ACT_Q8=1 (and AMX-INT8 present): both of the above quantize activations to int8 per
 * 32 and run k3_amx_q80_rows. k3_act_q8 < 0 until first use; set it to force a mode.
 * K3_ACT_Q8=2 restricts it to multi-token batches (speculative verify, short prefill
 * chunks); one-token decode then stays on the fp32-activation path. */
extern int k3_act_q8;
int  k3_act_q8_on(void);
int  k3_act_q8_T(int T);   /* whether a T-token Q8_0 matmul uses int8 activations */
void k3_iq2xs_rows(float *y, const float *x, const void *W, int in, int o0, int o1);
void k3_iq3xxs_rows(float *y, const float *x, const void *W, int in, int o0, int o1);

/* ggml MXFP4 rows -> the engine's MXFP4 layout: packed [rows][in/2] (element 2k in the low
 * nibble) followed by E8M0 scales [rows][in/32]. Same bytes in total; codes unchanged. */
void k3_gq_mxfp4_to_k3(unsigned char *dst, const unsigned char *src, int rows, int64_t in);

/* Dequantise n weights (a block multiple) of type t to fp32, exactly as ggml does. */
void k3_gq_dequant(int t, const void *src, float *dst, int64_t n);

/* Dequantise n weights (a block multiple, n % 32 == 0) to bf16 (round to nearest even),
 * 32 at a time: weights [32i, 32i + 32) go to dst + i * stride. */
void k3_gq_to_bf16(int t, const void *src, int64_t n, uint16_t *dst, size_t stride);

/* int8 activations: xq[n] and one scale per 256 (dx[n/256]); n % 256 == 0. */
void k3_gq_quant_x(int8_t *xq, float *dx, const float *x, int n);
/* 1 when this build has the integer kernels (AVX-512 VNNI). */
int  k3_gq_have_q8(void);
void k3_iq2xs_rows_q8(float *y, const int8_t *xq, const float *dx, const void *W, int in,
                      int o0, int o1);
/* T tokens, token g's xq/scales at XQ[g]/DX[g] and output at Y[g][o], each row decoded
 * once; per token bit-identical to k3_iq2xs_rows_q8. */
void k3_iq2xs_rows_q8_P(float *const *Y, const int8_t *const *XQ, const float *const *DX, int T,
                        const void *W, int in, int o0, int o1);
void k3_iq3xxs_rows_q8(float *y, const int8_t *xq, const float *dx, const void *W, int in,
                       int o0, int o1);

#endif /* K3_GQ_H */

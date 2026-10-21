/* SPDX-License-Identifier: Apache-2.0 */
/* k3_gq.c - see k3_gq.h. */
#include "k3_gq.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ggml_iq_grids.h"
#include "k3_amx.h"

int k3_act_q8 = -1;

int k3_act_q8_on(void)
{
    if (k3_act_q8 < 0) {
        const char *e = getenv("K3_ACT_Q8");
        k3_act_q8 = e && atoi(e) > 0 && k3_amx_q8_ok();
    }
    return k3_act_q8;
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define K3_GQ_AVX512 1
#endif
#if defined(K3_GQ_AVX512) || defined(__F16C__)
#include <immintrin.h>
#endif
#if defined(K3_GQ_AVX512) && defined(__AVX512VNNI__)
#define K3_GQ_VNNI 1
#endif

typedef struct { uint16_t d; int8_t qs[32]; } __attribute__((packed)) BQ80;
typedef struct { uint16_t d; uint16_t qs[K3_GQ_QK / 8]; uint8_t sc[K3_GQ_QK / 32]; } __attribute__((packed)) BIQ2XS;
typedef struct { uint16_t d; uint8_t qs[3 * K3_GQ_QK / 8]; } __attribute__((packed)) BIQ3XXS;
_Static_assert(sizeof(BQ80) == 34, "q8_0 block");
_Static_assert(sizeof(BIQ2XS) == 74, "iq2_xs block");
_Static_assert(sizeof(BIQ3XXS) == 98, "iq3_xxs block");

#ifndef K3_GQ_PF
#define K3_GQ_PF 512
#endif
/* The expert kernels read a few hundred KB per (expert, thread) at random places, so they
 * need the prefetch much further ahead: measured on GNR 140 -> 201 GB/s (512 -> 4096). */
#ifndef K3_GQ_PF_IQ
#define K3_GQ_PF_IQ 4096
#endif

static inline float h2f(uint16_t h)
{
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    union { uint32_t u; float f; } v;
    if (e == 0) {
        v.f = ldexpf((float)m, -24);
        v.u |= s;
        return v.f;
    }
    v.u = s | (e == 31 ? 0x7f800000u | (m << 13) : ((e + 112) << 23) | (m << 13));
    return v.f;
#endif
}

/* ------------------------------------------------------------- reference dequant */
static void deq_q80(const BQ80 *b, float *y)
{
    const float d = h2f(b->d);
    for (int j = 0; j < 32; j++) y[j] = b->qs[j] * d;
}

static void deq_iq2xs(const BIQ2XS *b, float *y)
{
    const float d = h2f(b->d);
    float db[2];
    for (int ib32 = 0; ib32 < K3_GQ_QK / 32; ++ib32) {
        db[0] = d * (0.5f + (b->sc[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (b->sc[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint16_t q = b->qs[4 * ib32 + l];
            const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q & 511));
            const uint8_t signs = ksigns_iq2xs[q >> 9];
            for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
    }
}

static void deq_iq3xxs(const BIQ3XXS *b, float *y)
{
    const float d = h2f(b->d);
    const uint8_t *qs = b->qs, *sas = b->qs + K3_GQ_QK / 4;
    for (int ib32 = 0; ib32 < K3_GQ_QK / 32; ++ib32) {
        uint32_t aux;
        memcpy(&aux, sas + 4 * ib32, 4);
        const float db = d * (0.5f + (aux >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux >> 7 * l) & 127];
            const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 0]);
            const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * g1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                y[j + 4] = db * g2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
    }
}

void k3_gq_dequant(int t, const void *src, float *dst, int64_t n)
{
    const unsigned char *p = (const unsigned char *)src;
    if (t == K3_GG_Q8_0)
        for (int64_t i = 0; i < n / 32; i++) deq_q80((const BQ80 *)p + i, dst + i * 32);
    else if (t == K3_GG_IQ2_XS)
        for (int64_t i = 0; i < n / K3_GQ_QK; i++) deq_iq2xs((const BIQ2XS *)p + i, dst + i * K3_GQ_QK);
    else if (t == K3_GG_IQ3_XXS)
        for (int64_t i = 0; i < n / K3_GQ_QK; i++) deq_iq3xxs((const BIQ3XXS *)p + i, dst + i * K3_GQ_QK);
    else if (t == K3_GG_MXFP4)
        for (int64_t i = 0; i < n / 32; i++) {
            static const int8_t kv[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
            const unsigned char *b = p + i * 17;
            const float d = ldexpf(1.0f, (int)b[0] - 128);   /* ggml's E8M0 half: kv is 2*E2M1 */
            for (int j = 0; j < 16; j++) {
                dst[i * 32 + j]      = kv[b[1 + j] & 15] * d;
                dst[i * 32 + j + 16] = kv[b[1 + j] >> 4] * d;
            }
        }
    else if (t == K3_GG_F32)
        memcpy(dst, src, (size_t)n * 4);
    else if (t == K3_GG_F16)
        for (int64_t i = 0; i < n; i++) dst[i] = h2f(((const uint16_t *)src)[i]);
    else if (t == K3_GG_BF16)
        for (int64_t i = 0; i < n; i++) {
            union { uint32_t u; float f; } v;
            v.u = (uint32_t)((const uint16_t *)src)[i] << 16;
            dst[i] = v.f;
        }
}

/* ggml MXFP4 blocks [e8m0][16 bytes: element j low nibble, j + 16 high] -> the engine's
 * checkpoint layout: packed [rows][in/2] (element 2k low, 2k + 1 high), then E8M0 scales
 * [rows][in/32]. The codes and exponents are copied unchanged. */
void k3_gq_mxfp4_to_k3(unsigned char *dst, const unsigned char *src, int rows, int64_t in)
{
    const int64_t nb = in / 32;
    unsigned char *sc = dst + (size_t)rows * (size_t)(in / 2);
    for (int r = 0; r < rows; r++)
        for (int64_t b = 0; b < nb; b++) {
            const unsigned char *blk = src + ((size_t)r * nb + b) * 17;
            unsigned char *p = dst + (size_t)r * (in / 2) + b * 16;
            sc[(size_t)r * nb + b] = blk[0];
            for (int k = 0; k < 16; k++) {
                const int e0 = 2 * k, e1 = 2 * k + 1;
                const int c0 = e0 < 16 ? blk[1 + e0] & 15 : blk[1 + e0 - 16] >> 4;
                const int c1 = e1 < 16 ? blk[1 + e1] & 15 : blk[1 + e1 - 16] >> 4;
                p[k] = (unsigned char)(c0 | c1 << 4);
            }
        }
}

/* ----------------------------------------------- portable path: the scalar scheme */
/* Dequantise one block, then the 16-lane double accumulation of k3_ops.c's scalar
 * matmul_f32_rows, so the portable build is also dequantise-then-k3_matmul exact. */
static void rows_scalar(int t, float *y, const float *x, const void *W, int in, int o0, int o1)
{
    const int blk = t == K3_GG_Q8_0 ? 32 : K3_GQ_QK;
    const size_t rb = k3_gq_row_bytes(t, in), bb = rb / (size_t)(in / blk);
    float w[K3_GQ_QK];
    for (int o = o0; o < o1; o++) {
        const unsigned char *row = (const unsigned char *)W + (size_t)o * rb;
        double a[16] = { 0 };
        for (int i = 0; i < in; i += blk) {
            k3_gq_dequant(t, row + (size_t)(i / blk) * bb, w, blk);
            for (int j = 0; j < blk; j++) a[j & 15] = fma((double)w[j], (double)x[i + j], a[j & 15]);
        }
        const double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
        const double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
        const double b2 = (a[2] + a[6]) + (a[10] + a[14]);
        const double b3 = (a[3] + a[7]) + (a[11] + a[15]);
        y[o] = (float)((b0 + b1) + (b2 + b3));
    }
}

#if defined(K3_GQ_AVX512)
/* The shared K3_AVX512 row reduction of k3_ops.c. */
static inline float v512_sum(const __m512 a[4])
{
    const __m512 s = _mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3]));
    const __m256 h = _mm256_add_ps(_mm512_castps512_ps256(s),
        _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(s), 1)));
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1));
    q = _mm_add_ps(q, _mm_movehl_ps(q, q));
    q = _mm_add_ss(q, _mm_shuffle_ps(q, q, 1));
    return _mm_cvtss_f32(q);
}

#define I8X16(w, k) _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32((w), (k))))

/* 64 int8 weights from 8 iq2_xs words: grid magnitudes, negated where the sign bit
 * (7 stored, the 8th their parity) is set. */
static inline __m512i iq2xs_mag(const uint16_t *q)
{
    const __m128i qw = _mm_loadu_si128((const __m128i *)q);
    return _mm512_i32gather_epi64(_mm256_cvtepu16_epi32(_mm_and_si128(qw, _mm_set1_epi16(511))),
                                  (const void *)iq2xs_grid, 8);
}

#if defined(__GFNI__)
/* gf2p8affine: output bit i is the parity of (row i & byte), row i being matrix byte 7-i.
 * Both matrices put a byte's 7 stored sign bits in bits 0-6 and their parity in bit 7,
 * which is exactly ksigns_iq2xs[]. */
#define K3_GF_ROW(i, r) ((uint64_t)(r) << (8 * (7 - (i))))
/* from the high byte of an iq2_xs word: index bit 8 | signs << 1 */
static const uint64_t GF_IQ2 = K3_GF_ROW(0, 0x02) | K3_GF_ROW(1, 0x04) | K3_GF_ROW(2, 0x08) |
    K3_GF_ROW(3, 0x10) | K3_GF_ROW(4, 0x20) | K3_GF_ROW(5, 0x40) | K3_GF_ROW(6, 0x80) | K3_GF_ROW(7, 0xFE);
/* from a byte holding the 7 signs in bits 0-6 */
static const uint64_t GF_SGN7 = K3_GF_ROW(0, 0x01) | K3_GF_ROW(1, 0x02) | K3_GF_ROW(2, 0x04) |
    K3_GF_ROW(3, 0x08) | K3_GF_ROW(4, 0x10) | K3_GF_ROW(5, 0x20) | K3_GF_ROW(6, 0x40) | K3_GF_ROW(7, 0x7F);
#endif

static inline __mmask64 iq2xs_neg(const uint16_t *q)
{
#if defined(__GFNI__)
    const __m128i hi = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)q),
                                        _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1));
    return _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(
        _mm_gf2p8affine_epi64_epi8(hi, _mm_set1_epi64x((long long)GF_IQ2), 0)));
#else
    const __m128i s7 = _mm_srli_epi16(_mm_loadu_si128((const __m128i *)q), 9);
    __m128i p = _mm_xor_si128(s7, _mm_srli_epi16(s7, 4));
    p = _mm_xor_si128(p, _mm_srli_epi16(p, 2));
    p = _mm_xor_si128(p, _mm_srli_epi16(p, 1));
    const __m128i sb = _mm_or_si128(s7, _mm_slli_epi16(_mm_and_si128(p, _mm_set1_epi16(1)), 7));
    return _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(_mm_cvtepi16_epi8(sb)));
#endif
}

/* 64 magnitudes and the sign mask of one 64-weight step of an iq3_xxs block */
static inline __m512i iq3xxs_mag(const uint8_t *qs)
{
    return _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)qs)),
                                  (const void *)iq3xxs_grid, 4);
}

static inline __mmask64 iq3xxs_neg(uint32_t a0, uint32_t a1)
{
#if defined(__GFNI__) && defined(__BMI2__)
    const uint64_t s = _pdep_u64(a0 & 0x0FFFFFFFu, 0x7F7F7F7Full) |
                       (_pdep_u64(a1 & 0x0FFFFFFFu, 0x7F7F7F7Full) << 32);
    return _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(
        _mm_gf2p8affine_epi64_epi8(_mm_cvtsi64_si128((long long)s), _mm_set1_epi64x((long long)GF_SGN7), 0)));
#else
    const __m256i s7 = _mm256_and_si256(
        _mm256_srlv_epi32(_mm256_setr_epi32(a0, a0, a0, a0, a1, a1, a1, a1),
                          _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21)),
        _mm256_set1_epi32(127));
    __m256i p = _mm256_xor_si256(s7, _mm256_srli_epi32(s7, 4));
    p = _mm256_xor_si256(p, _mm256_srli_epi32(p, 2));
    p = _mm256_xor_si256(p, _mm256_srli_epi32(p, 1));
    const __m256i sb = _mm256_or_si256(s7, _mm256_slli_epi32(_mm256_and_si256(p, _mm256_set1_epi32(1)), 7));
    return _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(_mm256_cvtepi32_epi8(sb)));
#endif
}
#endif

/* --------------------------------------------------------------- fp32 kernels */
#if defined(K3_GQ_AVX512)
#ifndef K3_GQ_PFH
#define K3_GQ_PFH _MM_HINT_T0
#endif
/* R consecutive rows at once: each x chunk is loaded once for all R. Every row keeps its
 * own four accumulators in the single-row order, so the bits do not depend on R. */
static inline __attribute__((always_inline)) void q80_rows_r(float *y, const float *x,
    const unsigned char *W, size_t rb, int in, int o, const int R)
{
    __m512 a[4][4];
    for (int r = 0; r < R; r++)
        for (int k = 0; k < 4; k++) a[r][k] = _mm512_setzero_ps();
    const unsigned char *w0 = W + (size_t)o * rb;
    for (int i = 0, j = 0; i < in; i += 64, j += 2) {
        __m512 xv[4];
        for (int k = 0; k < 4; k++) xv[k] = _mm512_loadu_ps(x + i + 16 * k);
        for (int r = 0; r < R; r++) {
            const BQ80 *b = (const BQ80 *)(w0 + (size_t)r * rb) + j;
            _mm_prefetch((const char *)b + K3_GQ_PF, K3_GQ_PFH);
            for (int k = 0; k < 4; k++) {
                const BQ80 *bk = b + (k >> 1);
                const __m512 w = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                    _mm_loadu_si128((const __m128i *)(bk->qs + 16 * (k & 1)))));
                a[r][k] = _mm512_fmadd_ps(_mm512_mul_ps(w, _mm512_set1_ps(h2f(bk->d))), xv[k], a[r][k]);
            }
        }
    }
    for (int r = 0; r < R; r++) y[o + r] = v512_sum(a[r]);
}

/* One row against G tokens: the weights are decoded once per 16 and fed to every token's
 * own four accumulators, each in the single-token order. */
static inline __attribute__((always_inline)) void q80_row_g(float *Y, int ldy, const float *X,
    int ldx, const unsigned char *w0, int in, int o, const int G)
{
    __m512 a[4][4];
    for (int g = 0; g < G; g++)
        for (int k = 0; k < 4; k++) a[g][k] = _mm512_setzero_ps();
    for (int i = 0, j = 0; i < in; i += 64, j += 2) {
        const BQ80 *b = (const BQ80 *)w0 + j;
        _mm_prefetch((const char *)b + K3_GQ_PF, K3_GQ_PFH);
        for (int k = 0; k < 4; k++) {
            const BQ80 *bk = b + (k >> 1);
            const __m512 w = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128((const __m128i *)(bk->qs + 16 * (k & 1))))),
                _mm512_set1_ps(h2f(bk->d)));
            for (int g = 0; g < G; g++)
                a[g][k] = _mm512_fmadd_ps(w, _mm512_loadu_ps(X + (size_t)g * ldx + i + 16 * k), a[g][k]);
        }
    }
    for (int g = 0; g < G; g++) Y[(size_t)g * ldy + o] = v512_sum(a[g]);
}
#endif

void k3_q80_rows_T(float *Y, int ldy, const float *X, int ldx, int T, const void *W, int in,
                   int o0, int o1)
{
    if (k3_act_q8_on() && k3_amx_q80_rows(Y, ldy, X, ldx, T, W, in, o0, o1) == 0) return;
#if defined(K3_GQ_AVX512)
    if (in % 64 == 0) {
        const size_t rb = k3_gq_row_bytes(K3_GG_Q8_0, in);
        for (int o = o0; o < o1; o++) {
            const unsigned char *w0 = (const unsigned char *)W + (size_t)o * rb;
            for (int t = 0; t < T; t += 4) {
                const int G = T - t < 4 ? T - t : 4;
                float *Yt = Y + (size_t)t * ldy;
                const float *Xt = X + (size_t)t * ldx;
                if (G == 4)      q80_row_g(Yt, ldy, Xt, ldx, w0, in, o, 4);
                else if (G == 3) q80_row_g(Yt, ldy, Xt, ldx, w0, in, o, 3);
                else if (G == 2) q80_row_g(Yt, ldy, Xt, ldx, w0, in, o, 2);
                else             q80_row_g(Yt, ldy, Xt, ldx, w0, in, o, 1);
            }
        }
        return;
    }
#endif
    for (int t = 0; t < T; t++)
        k3_q80_rows(Y + (size_t)t * ldy, X + (size_t)t * ldx, W, in, o0, o1);
}

/* Rows per pass of the Q8_0 kernel (1, 2 or 4); env K3_Q80_ROWS overrides. */
#ifndef K3_GQ_Q80R
#define K3_GQ_Q80R 4
#endif
static int q80_nr(void)
{
    static int nr = 0;
    if (!nr) {
        const char *e = getenv("K3_Q80_ROWS");
        const int v = e ? atoi(e) : K3_GQ_Q80R;
        nr = v >= 4 ? 4 : v >= 2 ? 2 : 1;
    }
    return nr;
}

void k3_q80_rows(float *y, const float *x, const void *W, int in, int o0, int o1)
{
    if (k3_act_q8_on() && k3_amx_q80_rows(y, 0, x, 0, 1, W, in, o0, o1) == 0) return;
#if defined(K3_GQ_AVX512)
    if (in % 64 == 0) {
        const size_t rb = k3_gq_row_bytes(K3_GG_Q8_0, in);
        const unsigned char *Wb = (const unsigned char *)W;
        const int nr = q80_nr();
        int o = o0;
        if (nr == 4)
            for (; o + 4 <= o1; o += 4) q80_rows_r(y, x, Wb, rb, in, o, 4);
        if (nr >= 2)
            for (; o + 2 <= o1; o += 2) q80_rows_r(y, x, Wb, rb, in, o, 2);
        for (; o < o1; o++) q80_rows_r(y, x, Wb, rb, in, o, 1);
        return;
    }
#endif
    rows_scalar(K3_GG_Q8_0, y, x, W, in, o0, o1);
}

void k3_iq2xs_rows(float *y, const float *x, const void *W, int in, int o0, int o1)
{
#if defined(K3_GQ_AVX512)
    const size_t rb = k3_gq_row_bytes(K3_GG_IQ2_XS, in);
    for (int o = o0; o < o1; o++) {
        const BIQ2XS *b = (const BIQ2XS *)((const unsigned char *)W + (size_t)o * rb);
        __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
        for (int i = 0; i < in; i += K3_GQ_QK, b++) {
            _mm_prefetch((const char *)b + K3_GQ_PF_IQ, _MM_HINT_T0);
            _mm_prefetch((const char *)b + K3_GQ_PF_IQ + 64, _MM_HINT_T0);
            const float d = h2f(b->d);
            for (int j = 0; j < 4; j++) {
                const __m512i mag = iq2xs_mag(b->qs + 8 * j);
                const __m512i w = _mm512_mask_sub_epi8(mag, iq2xs_neg(b->qs + 8 * j), _mm512_setzero_si512(), mag);
                const uint8_t s0 = b->sc[2 * j], s1 = b->sc[2 * j + 1];
                const float db0 = d * (0.5f + (s0 & 0xf)) * 0.25f, db1 = d * (0.5f + (s0 >> 4)) * 0.25f;
                const float db2 = d * (0.5f + (s1 & 0xf)) * 0.25f, db3 = d * (0.5f + (s1 >> 4)) * 0.25f;
                const float *xj = x + i + 64 * j;
                a[0] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 0), _mm512_set1_ps(db0)), _mm512_loadu_ps(xj +  0), a[0]);
                a[1] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 1), _mm512_set1_ps(db1)), _mm512_loadu_ps(xj + 16), a[1]);
                a[2] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 2), _mm512_set1_ps(db2)), _mm512_loadu_ps(xj + 32), a[2]);
                a[3] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 3), _mm512_set1_ps(db3)), _mm512_loadu_ps(xj + 48), a[3]);
            }
        }
        y[o] = v512_sum(a);
    }
#else
    rows_scalar(K3_GG_IQ2_XS, y, x, W, in, o0, o1);
#endif
}

void k3_iq3xxs_rows(float *y, const float *x, const void *W, int in, int o0, int o1)
{
#if defined(K3_GQ_AVX512)
    const size_t rb = k3_gq_row_bytes(K3_GG_IQ3_XXS, in);
    for (int o = o0; o < o1; o++) {
        const BIQ3XXS *b = (const BIQ3XXS *)((const unsigned char *)W + (size_t)o * rb);
        __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
        for (int i = 0; i < in; i += K3_GQ_QK, b++) {
            _mm_prefetch((const char *)b + K3_GQ_PF_IQ, _MM_HINT_T0);
            _mm_prefetch((const char *)b + K3_GQ_PF_IQ + 64, _MM_HINT_T0);
            const float d = h2f(b->d);
            const uint8_t *sas = b->qs + K3_GQ_QK / 4;
            for (int j = 0; j < 4; j++) {
                uint32_t a0, a1;
                memcpy(&a0, sas + 8 * j, 4);
                memcpy(&a1, sas + 8 * j + 4, 4);
                const __m512i mag = iq3xxs_mag(b->qs + 16 * j);
                const __m512i w = _mm512_mask_sub_epi8(mag, iq3xxs_neg(a0, a1), _mm512_setzero_si512(), mag);
                const float db0 = d * (0.5f + (a0 >> 28)) * 0.5f, db1 = d * (0.5f + (a1 >> 28)) * 0.5f;
                const float *xj = x + i + 64 * j;
                a[0] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 0), _mm512_set1_ps(db0)), _mm512_loadu_ps(xj +  0), a[0]);
                a[1] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 1), _mm512_set1_ps(db0)), _mm512_loadu_ps(xj + 16), a[1]);
                a[2] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 2), _mm512_set1_ps(db1)), _mm512_loadu_ps(xj + 32), a[2]);
                a[3] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 3), _mm512_set1_ps(db1)), _mm512_loadu_ps(xj + 48), a[3]);
            }
        }
        y[o] = v512_sum(a);
    }
#else
    rows_scalar(K3_GG_IQ3_XXS, y, x, W, in, o0, o1);
#endif
}

/* ------------------------------------------------------------- bf16 panels */
static inline uint16_t f2bf_rne(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

#if defined(K3_GQ_AVX512) && defined(__AVX512BF16__)
static inline void put_bf16x32(uint16_t *dst, __m512 lo, __m512 hi)
{
    _mm512_storeu_si512((void *)dst, (__m512i)_mm512_cvtne2ps_pbh(hi, lo));
}

#if defined(__AVX512VBMI__) && defined(__GFNI__)
#define K3_IQ2XS_TAB 1
/* Every iq2xs_grid byte is 8, 25 or 43: entry e as eight 2-bit codes (0, 1, 2), weight k
 * in bits 2k..2k+1. */
static const uint16_t iq2xs_code[512] __attribute__((aligned(64))) = {
    0x0000, 0x0002, 0x0005, 0x0008, 0x000a, 0x0011, 0x0014, 0x0016, 0x0019, 0x0020, 0x0022, 0x0025,
    0x0028, 0x0041, 0x0044, 0x0046, 0x0049, 0x0050, 0x0052, 0x0055, 0x0058, 0x0061, 0x0064, 0x0080,
    0x0082, 0x0085, 0x0088, 0x0091, 0x0094, 0x0099, 0x00a0, 0x0101, 0x0104, 0x0106, 0x0109, 0x0110,
    0x0112, 0x0115, 0x0118, 0x011a, 0x0121, 0x0124, 0x0140, 0x0142, 0x0145, 0x0148, 0x0151, 0x0154,
    0x0160, 0x0168, 0x0181, 0x0184, 0x0190, 0x0200, 0x0202, 0x0205, 0x0208, 0x0211, 0x0214, 0x0220,
    0x0241, 0x0244, 0x0250, 0x0255, 0x0280, 0x028a, 0x0401, 0x0404, 0x0406, 0x0409, 0x0410, 0x0412,
    0x0415, 0x0418, 0x0421, 0x0424, 0x0440, 0x0442, 0x0445, 0x0448, 0x0451, 0x0454, 0x0456, 0x0460,
    0x0481, 0x0484, 0x0490, 0x0500, 0x0502, 0x0505, 0x0508, 0x0511, 0x0514, 0x0520, 0x0541, 0x0544,
    0x0550, 0x0561, 0x0580, 0x0601, 0x0604, 0x0610, 0x0626, 0x0640, 0x0642, 0x0684, 0x0800, 0x0802,
    0x0805, 0x0808, 0x080a, 0x0811, 0x0814, 0x0820, 0x0825, 0x0841, 0x0844, 0x0850, 0x0858, 0x0880,
    0x08a0, 0x08aa, 0x0901, 0x0904, 0x0910, 0x0940, 0x0981, 0x0989, 0x0a00, 0x0a20, 0x0a28, 0x0a96,
    0x0aa0, 0x1001, 0x1004, 0x1006, 0x1009, 0x1010, 0x1012, 0x1015, 0x1018, 0x1021, 0x1024, 0x1040,
    0x1042, 0x1045, 0x1048, 0x1051, 0x1054, 0x1060, 0x106a, 0x1081, 0x1084, 0x1090, 0x1100, 0x1102,
    0x1105, 0x1108, 0x1111, 0x1114, 0x1120, 0x1141, 0x1144, 0x1150, 0x1180, 0x1194, 0x1196, 0x1201,
    0x1204, 0x1206, 0x1210, 0x1240, 0x1260, 0x1400, 0x1402, 0x1405, 0x1408, 0x1411, 0x1414, 0x1420,
    0x1441, 0x1444, 0x1449, 0x1450, 0x1464, 0x1480, 0x1501, 0x1504, 0x1510, 0x1540, 0x1600, 0x1614,
    0x1649, 0x1801, 0x1804, 0x1810, 0x1812, 0x1840, 0x1854, 0x1886, 0x1900, 0x1905, 0x1966, 0x1a51,
    0x1aa9, 0x2000, 0x2002, 0x2005, 0x2008, 0x200a, 0x2011, 0x2014, 0x2020, 0x2041, 0x2044, 0x2050,
    0x2080, 0x20a0, 0x2101, 0x2104, 0x2110, 0x2140, 0x2148, 0x2165, 0x2200, 0x2222, 0x2280, 0x22a8,
    0x2401, 0x2404, 0x2410, 0x2429, 0x2440, 0x2500, 0x2541, 0x2552, 0x2599, 0x2601, 0x261a, 0x26a6,
    0x2800, 0x2808, 0x280a, 0x2820, 0x2855, 0x2888, 0x28a2, 0x2968, 0x2990, 0x2a08, 0x2a20, 0x2a82,
    0x2a88, 0x2a8a, 0x4001, 0x4004, 0x4006, 0x4009, 0x4010, 0x4012, 0x4015, 0x4018, 0x4021, 0x4024,
    0x4040, 0x4042, 0x4045, 0x4048, 0x404a, 0x4051, 0x4054, 0x4060, 0x4065, 0x4081, 0x4084, 0x4090,
    0x4100, 0x4102, 0x4105, 0x4108, 0x4111, 0x4114, 0x4120, 0x4141, 0x4144, 0x4150, 0x4180, 0x4185,
    0x41a2, 0x4201, 0x4204, 0x4210, 0x4212, 0x4229, 0x4240, 0x4400, 0x4402, 0x4405, 0x4408, 0x4411,
    0x4414, 0x4419, 0x4420, 0x4441, 0x4444, 0x4450, 0x4480, 0x4494, 0x4501, 0x4504, 0x4510, 0x4524,
    0x4540, 0x459a, 0x4600, 0x460a, 0x4644, 0x4650, 0x4801, 0x4804, 0x4810, 0x4840, 0x4845, 0x4854,
    0x4862, 0x4900, 0x4911, 0x4944, 0x4950, 0x4969, 0x4a04, 0x5000, 0x5002, 0x5005, 0x5008, 0x5011,
    0x5014, 0x5020, 0x5028, 0x5041, 0x5044, 0x5050, 0x5080, 0x5101, 0x5104, 0x5110, 0x5115, 0x5140,
    0x5142, 0x5200, 0x5244, 0x52aa, 0x5401, 0x5404, 0x5410, 0x5421, 0x5440, 0x5460, 0x5481, 0x54a1,
    0x5500, 0x5508, 0x5580, 0x5588, 0x5621, 0x5668, 0x56a1, 0x5800, 0x5814, 0x5841, 0x5850, 0x5899,
    0x591a, 0x5940, 0x5942, 0x5a85, 0x6001, 0x6004, 0x6010, 0x6040, 0x6054, 0x6062, 0x6086, 0x60a9,
    0x6100, 0x6224, 0x624a, 0x6292, 0x6400, 0x6416, 0x6510, 0x6540, 0x6545, 0x65a4, 0x6801, 0x686a,
    0x6925, 0x6a06, 0x6a54, 0x6a62, 0x8000, 0x8002, 0x8005, 0x8008, 0x8011, 0x8014, 0x8020, 0x802a,
    0x8041, 0x8044, 0x8050, 0x8080, 0x8082, 0x80a8, 0x80aa, 0x8101, 0x8104, 0x8106, 0x8110, 0x8140,
    0x8151, 0x8159, 0x8200, 0x8220, 0x8280, 0x8282, 0x82a0, 0x82a8, 0x8401, 0x8404, 0x8410, 0x8412,
    0x8415, 0x8440, 0x8460, 0x8489, 0x8500, 0x8544, 0x85a5, 0x8618, 0x866a, 0x8800, 0x8808, 0x8825,
    0x885a, 0x8880, 0x8882, 0x88a8, 0x8906, 0x8a22, 0x8a80, 0x8a88, 0x8a96, 0x8aa8, 0x9001, 0x9004,
    0x9010, 0x9040, 0x9056, 0x9084, 0x9100, 0x9122, 0x9164, 0x9256, 0x9289, 0x9400, 0x9405, 0x9444,
    0x9450, 0x9458, 0x9529, 0x9590, 0x9592, 0x9641, 0x9851, 0x98a6, 0x9949, 0x9a15, 0x9a60, 0xa000,
    0xa002, 0xa008, 0xa00a, 0xa020, 0xa02a, 0xa0a0, 0xa151, 0xa159, 0xa1a6, 0xa200, 0xa202, 0xa208,
    0xa22a, 0xa280, 0xa2a0, 0xa440, 0xa495, 0xa665, 0xa698, 0xa80a, 0xa820, 0xa822, 0xa828, 0xa8a0,
    0xa8a8, 0xa904, 0xa984, 0xa986, 0xaa28, 0xaa2a, 0xaa91, 0xaaaa,
};

/* One iq2_xs block to eight 32-wide bf16 rows. The block's 48 values {8,25,43} x 16 scales
 * are rounded once into two tables (the same fp32 products and rounding as the per-weight
 * path); each weight then indexes them by code << 4 | scale, and a set sign bit flips the
 * bf16 sign, which is exact since rounding to nearest even is symmetric. */
static inline void iq2xs_bf16_block(const BIQ2XS *b, uint16_t *o, size_t stride)
{
    static const uint8_t msctl[64] __attribute__((aligned(64))) = {
        60, 60, 62, 62, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18,
        20, 20, 22, 22, 24, 24, 26, 26, 28, 28, 30, 30, 32, 32, 34, 34, 36, 36, 38, 38, 40, 40,
        42, 42, 44, 44, 46, 46, 48, 48, 50, 50, 52, 52, 54, 54, 56, 56, 58, 58 };
    const __m512 d = _mm512_set1_ps(h2f(b->d));
    const __m128i sc = _mm_loadl_epi64((const __m128i *)b->sc), m4 = _mm_set1_epi8(15);
    const __m512 S = _mm512_mul_ps(_mm512_mul_ps(d, _mm512_add_ps(_mm512_set1_ps(0.5f),
        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(_mm_and_si128(sc, m4),
                                                                  _mm_and_si128(_mm_srli_epi16(sc, 4), m4)))))),
        _mm512_set1_ps(0.25f));
    const __m512i X = (__m512i)_mm512_cvtne2ps_pbh(_mm512_mul_ps(_mm512_set1_ps(25.f), S),
                                                   _mm512_mul_ps(_mm512_set1_ps(8.f), S));
    const __m512i Y = _mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(_mm512_set1_ps(43.f), S)));

    const __m512i q = _mm512_loadu_si512((const void *)b->qs);
    const __m512i idx = _mm512_and_si512(q, _mm512_set1_epi16(511));
    __m512i r[8];
    for (int p = 0; p < 8; p++)
        r[p] = _mm512_permutex2var_epi16(_mm512_load_si512((const void *)(iq2xs_code + 64 * p)), idx,
                                         _mm512_load_si512((const void *)(iq2xs_code + 64 * p + 32)));
    const __mmask32 k6 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(64));
    const __mmask32 k7 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(128));
    const __mmask32 k8 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(256));
    const __m512i G = _mm512_mask_blend_epi16(k8,
        _mm512_mask_blend_epi16(k7, _mm512_mask_blend_epi16(k6, r[0], r[1]), _mm512_mask_blend_epi16(k6, r[2], r[3])),
        _mm512_mask_blend_epi16(k7, _mm512_mask_blend_epi16(k6, r[4], r[5]), _mm512_mask_blend_epi16(k6, r[6], r[7])));
    const __m256i sb = _mm256_gf2p8affine_epi64_epi8(_mm512_cvtepi16_epi8(_mm512_srli_epi16(q, 8)),
                                                     _mm256_set1_epi64x((long long)GF_IQ2), 0);
    uint64_t gq[8];
    uint32_t sg[8];
    _mm512_storeu_si512((void *)gq, G);
    _mm256_storeu_si256((__m256i *)sg, sb);

    const __m512i ctl = _mm512_load_si512((const void *)msctl), c30 = _mm512_set1_epi16(0x30);
    const __m512i sgn = _mm512_set1_epi16((short)0x8000);
    __m512i half = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int g = 0; g < 8; g++, half = _mm512_add_epi16(half, _mm512_set1_epi16(2))) {
        const __m512i code = _mm512_multishift_epi64_epi8(ctl, _mm512_set1_epi64((long long)gq[g]));
        const __m512i v = _mm512_permutex2var_epi16(X, _mm512_ternarylogic_epi32(code, c30, half, 0xEA), Y);
        _mm512_storeu_si512((void *)(o + (size_t)g * stride), _mm512_mask_add_epi16(v, _cvtu32_mask32(sg[g]), v, sgn));
    }
}
#endif
#endif

void k3_gq_to_bf16(int t, const void *src, int64_t n, uint16_t *dst, size_t stride)
{
    const unsigned char *p = (const unsigned char *)src;
#if defined(K3_GQ_AVX512) && defined(__AVX512BF16__)
    if (t == K3_GG_Q8_0) {
        for (int64_t i = 0; i < n / 32; i++) {
            const BQ80 *b = (const BQ80 *)p + i;
            const __m512 d = _mm512_set1_ps(h2f(b->d));
            put_bf16x32(dst + (size_t)i * stride,
                _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)b->qs))), d),
                _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b->qs + 16)))), d));
        }
        return;
    }
    if (t == K3_GG_IQ2_XS) {
#if defined(K3_IQ2XS_TAB)
        for (int64_t i = 0; i < n / K3_GQ_QK; i++)
            iq2xs_bf16_block((const BIQ2XS *)p + i, dst + (size_t)i * 8 * stride, stride);
        return;
#endif
        for (int64_t i = 0; i < n / K3_GQ_QK; i++) {
            const BIQ2XS *b = (const BIQ2XS *)p + i;
            const float d = h2f(b->d);
            for (int j = 0; j < 4; j++) {
                const __m512i mag = iq2xs_mag(b->qs + 8 * j);
                const __m512i w = _mm512_mask_sub_epi8(mag, iq2xs_neg(b->qs + 8 * j), _mm512_setzero_si512(), mag);
                const uint8_t s0 = b->sc[2 * j], s1 = b->sc[2 * j + 1];
                uint16_t *o = dst + (size_t)(i * 8 + 2 * j) * stride;
                put_bf16x32(o, _mm512_mul_ps(I8X16(w, 0), _mm512_set1_ps(d * (0.5f + (s0 & 0xf)) * 0.25f)),
                               _mm512_mul_ps(I8X16(w, 1), _mm512_set1_ps(d * (0.5f + (s0 >> 4)) * 0.25f)));
                put_bf16x32(o + stride, _mm512_mul_ps(I8X16(w, 2), _mm512_set1_ps(d * (0.5f + (s1 & 0xf)) * 0.25f)),
                                        _mm512_mul_ps(I8X16(w, 3), _mm512_set1_ps(d * (0.5f + (s1 >> 4)) * 0.25f)));
            }
        }
        return;
    }
    if (t == K3_GG_F32) {
        const float *f = (const float *)src;
        for (int64_t i = 0; i < n / 32; i++)
            put_bf16x32(dst + (size_t)i * stride, _mm512_loadu_ps(f + i * 32), _mm512_loadu_ps(f + i * 32 + 16));
        return;
    }
    if (t == K3_GG_IQ3_XXS) {
        for (int64_t i = 0; i < n / K3_GQ_QK; i++) {
            const BIQ3XXS *b = (const BIQ3XXS *)p + i;
            const float d = h2f(b->d);
            const uint8_t *sas = b->qs + K3_GQ_QK / 4;
            for (int j = 0; j < 4; j++) {
                uint32_t a0, a1;
                memcpy(&a0, sas + 8 * j, 4);
                memcpy(&a1, sas + 8 * j + 4, 4);
                const __m512i mag = iq3xxs_mag(b->qs + 16 * j);
                const __m512i w = _mm512_mask_sub_epi8(mag, iq3xxs_neg(a0, a1), _mm512_setzero_si512(), mag);
                const __m512 d0 = _mm512_set1_ps(d * (0.5f + (a0 >> 28)) * 0.5f);
                const __m512 d1 = _mm512_set1_ps(d * (0.5f + (a1 >> 28)) * 0.5f);
                uint16_t *o = dst + (size_t)(i * 8 + 2 * j) * stride;
                put_bf16x32(o, _mm512_mul_ps(I8X16(w, 0), d0), _mm512_mul_ps(I8X16(w, 1), d0));
                put_bf16x32(o + stride, _mm512_mul_ps(I8X16(w, 2), d1), _mm512_mul_ps(I8X16(w, 3), d1));
            }
        }
        return;
    }
#endif
    float tmp[K3_GQ_QK];
    const int blk = t == K3_GG_IQ2_XS || t == K3_GG_IQ3_XXS ? K3_GQ_QK : 32;
    const size_t bb = k3_gq_row_bytes(t, blk);
    for (int64_t i = 0; i < n / blk; i++) {
        k3_gq_dequant(t, p + (size_t)i * bb, tmp, blk);
        for (int k = 0; k < blk; k++)
            dst[(size_t)((i * blk + k) / 32) * stride + (k & 31)] = f2bf_rne(tmp[k]);
    }
}

/* ------------------------------------------------------------ int8 activations */
void k3_gq_quant_x(int8_t *xq, float *dx, const float *x, int n)
{
    for (int b = 0; b < n / K3_GQ_QK; b++) {
        const float *xb = x + (size_t)b * K3_GQ_QK;
#if defined(K3_GQ_AVX512)
        /* max is order-free, and vcvtps2dq rounds to nearest even as lrintf does */
        __m512 m = _mm512_setzero_ps();
        for (int j = 0; j < K3_GQ_QK; j += 16)
            m = _mm512_max_ps(m, _mm512_abs_ps(_mm512_loadu_ps(xb + j)));
        const float amax = _mm512_reduce_max_ps(m);
#else
        float amax = 0.0f;
        for (int j = 0; j < K3_GQ_QK; j++) amax = fmaxf(amax, fabsf(xb[j]));
#endif
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        dx[b] = d;
#if defined(K3_GQ_AVX512)
        const __m512 vi = _mm512_set1_ps(id);
        for (int j = 0; j < K3_GQ_QK; j += 16) {
            const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(xb + j), vi);
            _mm_storeu_si128((__m128i *)(xq + (size_t)b * K3_GQ_QK + j),
                             _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v)));
        }
#else
        for (int j = 0; j < K3_GQ_QK; j++) xq[(size_t)b * K3_GQ_QK + j] = (int8_t)lrintf(xb[j] * id);
#endif
    }
}

int k3_gq_have_q8(void)
{
#if defined(K3_GQ_VNNI)
    return 1;
#else
    return 0;
#endif
}

#if defined(K3_GQ_VNNI)
void k3_iq2xs_rows_q8(float *y, const int8_t *xq, const float *dx, const void *W, int in,
                      int o0, int o1)
{
    const size_t rb = k3_gq_row_bytes(K3_GG_IQ2_XS, in);
    __m512i idx4[4];
    for (int j = 0; j < 4; j++)
        idx4[j] = _mm512_setr_epi32(4*j, 4*j, 4*j, 4*j, 4*j+1, 4*j+1, 4*j+1, 4*j+1,
                                    4*j+2, 4*j+2, 4*j+2, 4*j+2, 4*j+3, 4*j+3, 4*j+3, 4*j+3);
    const __m128i m4 = _mm_set1_epi8(0xf);
    for (int o = o0; o < o1; o++) {
        const BIQ2XS *b = (const BIQ2XS *)((const unsigned char *)W + (size_t)o * rb);
        __m512 facc = _mm512_setzero_ps();
        for (int i = 0; i < in; i += K3_GQ_QK, b++) {
            _mm_prefetch((const char *)b + K3_GQ_PF_IQ, _MM_HINT_T0);
            _mm_prefetch((const char *)b + K3_GQ_PF_IQ + 64, _MM_HINT_T0);
            /* lane c: 2*s+1 of 16-weight chunk c */
            const __m128i s = _mm_loadl_epi64((const __m128i *)b->sc);
            const __m128i nib = _mm_unpacklo_epi8(_mm_and_si128(s, m4), _mm_and_si128(_mm_srli_epi16(s, 4), m4));
            const __m512i sc = _mm512_cvtepu8_epi32(_mm_add_epi8(_mm_add_epi8(nib, nib), _mm_set1_epi8(1)));
            __m512i iacc = _mm512_setzero_si512();
            for (int j = 0; j < 4; j++) {
                const uint16_t *q = b->qs + 8 * j;
                const __m512i xv = _mm512_loadu_si512((const void *)(xq + i + 64 * j));
                const __m512i xs = _mm512_mask_sub_epi8(xv, iq2xs_neg(q), _mm512_setzero_si512(), xv);
                const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), iq2xs_mag(q), xs);
                /* |dp| <= 4*43*127 fits int16, so vpdpwssd with (scale, 0) is dp*scale */
                iacc = _mm512_dpwssd_epi32(iacc, dp, _mm512_permutexvar_epi32(idx4[j], sc));
            }
            facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc),
                                   _mm512_set1_ps(h2f(b->d) * dx[i / K3_GQ_QK] * 0.125f), facc);
        }
        y[o] = _mm512_reduce_add_ps(facc);
    }
}

/* G tokens against one row, the block decoded once; per token the single-token order */
static inline __attribute__((always_inline)) void iq2xs_row_q8_g(float *const *Y,
    const int8_t *const *XQ, const float *const *DX, const BIQ2XS *b, int in, int o,
    const int G, const __m512i *idx4)
{
    const __m128i m4 = _mm_set1_epi8(0xf);
    __m512 facc[4];
    for (int g = 0; g < G; g++) facc[g] = _mm512_setzero_ps();
    for (int i = 0; i < in; i += K3_GQ_QK, b++) {
        _mm_prefetch((const char *)b + K3_GQ_PF_IQ, _MM_HINT_T0);
        _mm_prefetch((const char *)b + K3_GQ_PF_IQ + 64, _MM_HINT_T0);
        const __m128i s = _mm_loadl_epi64((const __m128i *)b->sc);
        const __m128i nib = _mm_unpacklo_epi8(_mm_and_si128(s, m4), _mm_and_si128(_mm_srli_epi16(s, 4), m4));
        const __m512i sc = _mm512_cvtepu8_epi32(_mm_add_epi8(_mm_add_epi8(nib, nib), _mm_set1_epi8(1)));
        __m512i iacc[4];
        for (int g = 0; g < G; g++) iacc[g] = _mm512_setzero_si512();
        for (int j = 0; j < 4; j++) {
            const uint16_t *q = b->qs + 8 * j;
            const __mmask64 neg = iq2xs_neg(q);
            const __m512i mag = iq2xs_mag(q), scj = _mm512_permutexvar_epi32(idx4[j], sc);
            for (int g = 0; g < G; g++) {
                const __m512i xv = _mm512_loadu_si512((const void *)(XQ[g] + i + 64 * j));
                const __m512i xs = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
                iacc[g] = _mm512_dpwssd_epi32(iacc[g], _mm512_dpbusd_epi32(_mm512_setzero_si512(), mag, xs), scj);
            }
        }
        const float d = h2f(b->d);
        for (int g = 0; g < G; g++)
            facc[g] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc[g]),
                                      _mm512_set1_ps(d * DX[g][i / K3_GQ_QK] * 0.125f), facc[g]);
    }
    for (int g = 0; g < G; g++) Y[g][o] = _mm512_reduce_add_ps(facc[g]);
}

void k3_iq2xs_rows_q8_P(float *const *Y, const int8_t *const *XQ, const float *const *DX, int T,
                        const void *W, int in, int o0, int o1)
{
    const size_t rb = k3_gq_row_bytes(K3_GG_IQ2_XS, in);
    __m512i idx4[4];
    for (int j = 0; j < 4; j++)
        idx4[j] = _mm512_setr_epi32(4*j, 4*j, 4*j, 4*j, 4*j+1, 4*j+1, 4*j+1, 4*j+1,
                                    4*j+2, 4*j+2, 4*j+2, 4*j+2, 4*j+3, 4*j+3, 4*j+3, 4*j+3);
    for (int o = o0; o < o1; o++) {
        const BIQ2XS *b = (const BIQ2XS *)((const unsigned char *)W + (size_t)o * rb);
        for (int t = 0; t < T; t += 4) {
            const int G = T - t < 4 ? T - t : 4;
            if (G == 4)      iq2xs_row_q8_g(Y + t, XQ + t, DX + t, b, in, o, 4, idx4);
            else if (G == 3) iq2xs_row_q8_g(Y + t, XQ + t, DX + t, b, in, o, 3, idx4);
            else if (G == 2) iq2xs_row_q8_g(Y + t, XQ + t, DX + t, b, in, o, 2, idx4);
            else             iq2xs_row_q8_g(Y + t, XQ + t, DX + t, b, in, o, 1, idx4);
        }
    }
}

void k3_iq3xxs_rows_q8(float *y, const int8_t *xq, const float *dx, const void *W, int in,
                       int o0, int o1)
{
    const size_t rb = k3_gq_row_bytes(K3_GG_IQ3_XXS, in);
    for (int o = o0; o < o1; o++) {
        const BIQ3XXS *b = (const BIQ3XXS *)((const unsigned char *)W + (size_t)o * rb);
        __m512 facc = _mm512_setzero_ps();
        for (int i = 0; i < in; i += K3_GQ_QK, b++) {
            _mm_prefetch((const char *)b + K3_GQ_PF_IQ, _MM_HINT_T0);
            _mm_prefetch((const char *)b + K3_GQ_PF_IQ + 64, _MM_HINT_T0);
            const uint8_t *sas = b->qs + K3_GQ_QK / 4;
            __m512i iacc = _mm512_setzero_si512();
            for (int j = 0; j < 4; j++) {
                uint32_t a0, a1;
                memcpy(&a0, sas + 8 * j, 4);
                memcpy(&a1, sas + 8 * j + 4, 4);
                const __m512i xv = _mm512_loadu_si512((const void *)(xq + i + 64 * j));
                const __m512i xs = _mm512_mask_sub_epi8(xv, iq3xxs_neg(a0, a1), _mm512_setzero_si512(), xv);
                const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), iq3xxs_mag(b->qs + 16 * j), xs);
                const int s0 = 2 * (int)(a0 >> 28) + 1, s1 = 2 * (int)(a1 >> 28) + 1;
                iacc = _mm512_add_epi32(iacc, _mm512_mullo_epi32(dp, _mm512_setr_epi32(
                    s0, s0, s0, s0, s0, s0, s0, s0, s1, s1, s1, s1, s1, s1, s1, s1)));
            }
            facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc),
                                   _mm512_set1_ps(h2f(b->d) * dx[i / K3_GQ_QK] * 0.25f), facc);
        }
        y[o] = _mm512_reduce_add_ps(facc);
    }
}
#else
void k3_iq2xs_rows_q8_P(float *const *Y, const int8_t *const *XQ, const float *const *DX, int T,
                        const void *W, int in, int o0, int o1)
{
    (void)Y; (void)XQ; (void)DX; (void)T; (void)W; (void)in; (void)o0; (void)o1;
}
void k3_iq2xs_rows_q8(float *y, const int8_t *xq, const float *dx, const void *W, int in,
                      int o0, int o1)
{
    (void)y; (void)xq; (void)dx; (void)W; (void)in; (void)o0; (void)o1;
}
void k3_iq3xxs_rows_q8(float *y, const int8_t *xq, const float *dx, const void *W, int in,
                       int o0, int o1)
{
    (void)y; (void)xq; (void)dx; (void)W; (void)in; (void)o0; (void)o1;
}
#endif

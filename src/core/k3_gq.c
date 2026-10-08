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
        const int m = e ? atoi(e) : 0;
        k3_act_q8 = m > 0 && k3_amx_q8_ok() ? (m == 2 ? 2 : 1) : 0;
    }
    return k3_act_q8;
}

int k3_act_q8_T(int T)
{
    const int m = k3_act_q8_on();
    return m == 1 || (m == 2 && T > 1);
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
    if (k3_act_q8_T(T) && k3_amx_q80_rows(Y, ldy, X, ldx, T, W, in, o0, o1) == 0) return;
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
    if (k3_act_q8_T(1) && k3_amx_q80_rows(y, 0, x, 0, 1, W, in, o0, o1) == 0) return;
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
        float amax = 0.0f;
        for (int j = 0; j < K3_GQ_QK; j++) amax = fmaxf(amax, fabsf(xb[j]));
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        dx[b] = d;
        for (int j = 0; j < K3_GQ_QK; j++) xq[(size_t)b * K3_GQ_QK + j] = (int8_t)lrintf(xb[j] * id);
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

/* kbench.c - prototype AVX-512 GEMV kernels for the GGUF types of Kimi-K3 UD-Q2_K_XL.
 *
 *   Q8_0    trunk     (34 B / 32 w)
 *   IQ2_XS  experts   (74 B / 256 w)
 *   IQ3_XXS experts   (98 B / 256 w, 13 tensors)
 *
 * Every fused kernel uses the engine's fp32 GEMV scheme (element i -> lane i%16 of
 * accumulator (i/16)%4, v512_sum tree) and forms each weight exactly as ggml's
 * dequantize_row_* does, so fused == dequantise + k3_matmul to the bit ("check").
 *
 *   ./kbench check
 *   ./kbench l2   [kernel]           1 thread, L2-resident: compute throughput
 *   ./kbench dram [kernel] [GB]      all OpenMP threads, DRAM-resident: GB/s of weights
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gg_tables.h"

#define QK_K 256
typedef struct { uint16_t d; int8_t qs[32]; } __attribute__((packed)) bq8_0;
typedef struct { uint16_t d; uint16_t qs[QK_K / 8]; uint8_t scales[QK_K / 32]; } __attribute__((packed)) biq2xs;
typedef struct { uint16_t d; uint8_t qs[3 * QK_K / 8]; } __attribute__((packed)) biq3xxs;
_Static_assert(sizeof(bq8_0) == 34, "q8_0");
_Static_assert(sizeof(biq2xs) == 74, "iq2_xs");
_Static_assert(sizeof(biq3xxs) == 98, "iq3_xxs");

enum { KQ8_0, KIQ2XS, KIQ2XS_S, KIQ3XXS, KN };

static size_t row_bytes(int k, int in)
{
    if (k == KQ8_0) return (size_t)in / 32 * sizeof(bq8_0);
    if (k == KIQ3XXS) return (size_t)in / QK_K * sizeof(biq3xxs);
    return (size_t)in / QK_K * sizeof(biq2xs);
}

static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------- ggml reference dequant */
static void deq_q8_0(const bq8_0 *x, float *y, int k)
{
    for (int i = 0; i < k / 32; i++) {
        const float d = h2f(x[i].d);
        for (int j = 0; j < 32; j++) y[i * 32 + j] = x[i].qs[j] * d;
    }
}

static void deq_iq2xs(const biq2xs *x, float *y, int k)
{
    float db[2];
    for (int i = 0; i < k / QK_K; i++) {
        const float d = h2f(x[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4 * ib32 + l] & 511));
                const uint8_t signs = ksigns_iq2xs[x[i].qs[4 * ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

static void deq_iq3xxs(const biq3xxs *x, float *y, int k)
{
    uint32_t aux32;
    for (int i = 0; i < k / QK_K; i++) {
        const float d = h2f(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *sas = qs + QK_K / 4;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            memcpy(&aux32, sas + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
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
}

/* --------------------------------------------------- engine fp32 scheme (k3_ops.c) */
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

static void ref_rows(float *y, const float *x, const float *W, int in, int r0, int r1)
{
    for (int o = r0; o < r1; o++) {
        const float *row = W + (size_t)o * in;
        __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
        for (int i = 0; i < in; i += 64)
            for (int k = 0; k < 4; k++)
                a[k] = _mm512_fmadd_ps(_mm512_loadu_ps(row + i + 16 * k), _mm512_loadu_ps(x + i + 16 * k), a[k]);
        y[o] = v512_sum(a);
    }
}

/* ------------------------------------------------------------------ fused kernels */
#ifndef PF
#define PF 512
#endif

static void q8_0_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{
    const size_t rb = row_bytes(KQ8_0, in);
    for (int r = r0; r < r1; r++) {
        const bq8_0 *b = (const bq8_0 *)((const char *)W + (size_t)r * rb);
        __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
        for (int i = 0; i < in; i += 64, b += 2) {
            _mm_prefetch((const char *)b + PF, _MM_HINT_T0);
            for (int k = 0; k < 4; k++) {
                const bq8_0 *bk = b + (k >> 1);
                const __m512 d = _mm512_set1_ps(h2f(bk->d));
                const __m512 w = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                    _mm_loadu_si128((const __m128i *)(bk->qs + 16 * (k & 1)))));
                a[k] = _mm512_fmadd_ps(_mm512_mul_ps(w, d), _mm512_loadu_ps(x + i + 16 * k), a[k]);
            }
        }
        y[r] = v512_sum(a);
    }
}

/* 64 signed int8 weights from 8 iq2_xs words: grid magnitudes, signs as one k-mask */
static inline __m512i iq2xs_w64(const uint16_t *q, int scalar_lut)
{
    const __m128i qw = _mm_loadu_si128((const __m128i *)q);
    __m512i mag;
    if (scalar_lut)
        mag = _mm512_set_epi64(iq2xs_grid[q[7] & 511], iq2xs_grid[q[6] & 511], iq2xs_grid[q[5] & 511],
                               iq2xs_grid[q[4] & 511], iq2xs_grid[q[3] & 511], iq2xs_grid[q[2] & 511],
                               iq2xs_grid[q[1] & 511], iq2xs_grid[q[0] & 511]);
    else
        mag = _mm512_i32gather_epi64(_mm256_cvtepu16_epi32(_mm_and_si128(qw, _mm_set1_epi16(511))),
                                     (const void *)iq2xs_grid, 8);
    const __m128i s7 = _mm_srli_epi16(qw, 9);
    const __m128i par = _mm_slli_epi16(_mm_and_si128(_mm_popcnt_epi16(s7), _mm_set1_epi16(1)), 7);
    const __mmask64 neg = _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(_mm_cvtepi16_epi8(_mm_or_si128(s7, par))));
    return _mm512_mask_sub_epi8(mag, neg, _mm512_setzero_si512(), mag);
}

#define I8X16(w, k) _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32((w), (k))))

static void iq2xs_rows_impl(float *y, const float *x, const void *W, int in, int r0, int r1, int slut)
{
    const size_t rb = row_bytes(KIQ2XS, in);
    for (int r = r0; r < r1; r++) {
        const biq2xs *b = (const biq2xs *)((const char *)W + (size_t)r * rb);
        __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
        for (int i = 0; i < in; i += QK_K, b++) {
            _mm_prefetch((const char *)b + PF, _MM_HINT_T0);
            _mm_prefetch((const char *)b + PF + 64, _MM_HINT_T0);
            const float d = h2f(b->d);
            for (int j = 0; j < 4; j++) {
                const __m512i w = iq2xs_w64(b->qs + 8 * j, slut);
                const uint8_t s0 = b->scales[2 * j], s1 = b->scales[2 * j + 1];
                const float db[4] = { d * (0.5f + (s0 & 0xf)) * 0.25f, d * (0.5f + (s0 >> 4)) * 0.25f,
                                      d * (0.5f + (s1 & 0xf)) * 0.25f, d * (0.5f + (s1 >> 4)) * 0.25f };
                const float *xj = x + i + 64 * j;
                a[0] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 0), _mm512_set1_ps(db[0])), _mm512_loadu_ps(xj +  0), a[0]);
                a[1] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 1), _mm512_set1_ps(db[1])), _mm512_loadu_ps(xj + 16), a[1]);
                a[2] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 2), _mm512_set1_ps(db[2])), _mm512_loadu_ps(xj + 32), a[2]);
                a[3] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 3), _mm512_set1_ps(db[3])), _mm512_loadu_ps(xj + 48), a[3]);
            }
        }
        y[r] = v512_sum(a);
    }
}
static void iq2xs_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{ iq2xs_rows_impl(y, x, W, in, r0, r1, 0); }
static void iq2xs_s_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{ iq2xs_rows_impl(y, x, W, in, r0, r1, 1); }

static void iq3xxs_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{
    const size_t rb = row_bytes(KIQ3XXS, in);
    const __m256i sh = _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21);
    for (int r = r0; r < r1; r++) {
        const biq3xxs *b = (const biq3xxs *)((const char *)W + (size_t)r * rb);
        __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
        for (int i = 0; i < in; i += QK_K, b++) {
            _mm_prefetch((const char *)b + PF, _MM_HINT_T0);
            _mm_prefetch((const char *)b + PF + 64, _MM_HINT_T0);
            const float d = h2f(b->d);
            const uint8_t *qs = b->qs;
            const uint32_t *sas = (const uint32_t *)(b->qs + QK_K / 4);
            for (int j = 0; j < 4; j++) {
                uint32_t a0, a1;
                memcpy(&a0, sas + 2 * j, 4); memcpy(&a1, sas + 2 * j + 1, 4);
                const __m512i mag = _mm512_i32gather_epi32(
                    _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(qs + 16 * j))),
                    (const void *)iq3xxs_grid, 4);
                const __m256i s7 = _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_setr_epi32(a0, a0, a0, a0, a1, a1, a1, a1), sh),
                    _mm256_set1_epi32(127));
                const __m256i par = _mm256_slli_epi32(_mm256_and_si256(_mm256_popcnt_epi32(s7), _mm256_set1_epi32(1)), 7);
                const __mmask64 neg = _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(
                    _mm256_cvtepi32_epi8(_mm256_or_si256(s7, par))));
                const __m512i w = _mm512_mask_sub_epi8(mag, neg, _mm512_setzero_si512(), mag);
                const float db0 = d * (0.5f + (a0 >> 28)) * 0.5f, db1 = d * (0.5f + (a1 >> 28)) * 0.5f;
                const float *xj = x + i + 64 * j;
                a[0] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 0), _mm512_set1_ps(db0)), _mm512_loadu_ps(xj +  0), a[0]);
                a[1] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 1), _mm512_set1_ps(db0)), _mm512_loadu_ps(xj + 16), a[1]);
                a[2] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 2), _mm512_set1_ps(db1)), _mm512_loadu_ps(xj + 32), a[2]);
                a[3] = _mm512_fmadd_ps(_mm512_mul_ps(I8X16(w, 3), _mm512_set1_ps(db1)), _mm512_loadu_ps(xj + 48), a[3]);
            }
        }
        y[r] = v512_sum(a);
    }
}

typedef void (*rows_fn)(float *, const float *, const void *, int, int, int);

/* ---- int8-activation variants (VNNI). Not bit-exact vs fp32: x is quantised per 256
 * like ggml's q8_K, then all products are exact int32. Opt-in mode in the engine. */
static int8_t *g_xq;      /* [in] */
static float  *g_dx;      /* [in/256] */

static void perm_x(int in);

static void quant_x(const float *x, int in)
{
    for (int b = 0; b < in / QK_K; b++) {
        float amax = 0;
        for (int j = 0; j < QK_K; j++) amax = fmaxf(amax, fabsf(x[b * QK_K + j]));
        const float dx = amax / 127.f, id = dx ? 1.f / dx : 0.f;
        g_dx[b] = dx;
        for (int j = 0; j < QK_K; j++) g_xq[b * QK_K + j] = (int8_t)lrintf(x[b * QK_K + j] * id);
    }
    perm_x(in);
}

/* chunk c (16 weights) scale 2*nib+1 in lane c; idx4[j] spreads chunks 4j..4j+3 over 4 lanes each */
static inline __m512i iq2xs_scales16(const uint8_t *sc)
{
    const __m128i s = _mm_loadl_epi64((const __m128i *)sc);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i nib = _mm_unpacklo_epi8(_mm_and_si128(s, m4), _mm_and_si128(_mm_srli_epi16(s, 4), m4));
    return _mm512_cvtepu8_epi32(_mm_add_epi8(_mm_add_epi8(nib, nib), _mm_set1_epi8(1)));
}

static inline __m512i iq2xs_mag64(const uint16_t *q, int nogather)
{
    if (nogather) return _mm512_loadu_si512((const void *)(iq2xs_grid + (q[0] & 255)));
    const __m128i qw = _mm_loadu_si128((const __m128i *)q);
    return _mm512_i32gather_epi64(_mm256_cvtepu16_epi32(_mm_and_si128(qw, _mm_set1_epi16(511))),
                                  (const void *)iq2xs_grid, 8);
}

static inline __mmask64 iq2xs_neg64(const uint16_t *q)
{
    const __m128i s7 = _mm_srli_epi16(_mm_loadu_si128((const __m128i *)q), 9);
    const __m128i par = _mm_slli_epi16(_mm_and_si128(_mm_popcnt_epi16(s7), _mm_set1_epi16(1)), 7);
    return _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(_mm_cvtepi16_epi8(_mm_or_si128(s7, par))));
}

/* GFNI: affine row i (byte 7-i of the matrix) selects the input bits that XOR into output
 * bit i. High byte of a word = idx bit 8 | s7 << 1: bits 0-6 <- input bits 1-7, bit 7 <-
 * their parity, i.e. exactly ksigns_iq2xs[s7]. */
#define K3_GF_ROW(i, r) ((uint64_t)(r) << (8 * (7 - (i))))
static const uint64_t GF_IQ2 = K3_GF_ROW(0, 0x02) | K3_GF_ROW(1, 0x04) | K3_GF_ROW(2, 0x08) |
    K3_GF_ROW(3, 0x10) | K3_GF_ROW(4, 0x20) | K3_GF_ROW(5, 0x40) | K3_GF_ROW(6, 0x80) | K3_GF_ROW(7, 0xFE);
/* 7 sign bits already in bits 0-6: keep them, bit 7 <- their parity */
static const uint64_t GF_SGN7 = K3_GF_ROW(0, 0x01) | K3_GF_ROW(1, 0x02) | K3_GF_ROW(2, 0x04) |
    K3_GF_ROW(3, 0x08) | K3_GF_ROW(4, 0x10) | K3_GF_ROW(5, 0x20) | K3_GF_ROW(6, 0x40) | K3_GF_ROW(7, 0x7F);

static inline __mmask64 iq2xs_neg64_gfni(const uint16_t *q)
{
    const __m128i hi = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)q),
                                        _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1));
    const __m128i sb = _mm_gf2p8affine_epi64_epi8(hi, _mm_set1_epi64x((long long)GF_IQ2), 0);
    return _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(sb));
}

/* iq3_xxs: two scale-and-sign words -> 8 bytes of 7 sign bits (BMI2 pdep) -> parity bit 7 */
static inline __mmask64 iq3xxs_neg64_gfni(uint32_t a0, uint32_t a1)
{
    const uint64_t s = _pdep_u64(a0 & 0x0FFFFFFFu, 0x7F7F7F7Full) |
                       (_pdep_u64(a1 & 0x0FFFFFFFu, 0x7F7F7F7Full) << 32);
    const __m128i sb = _mm_gf2p8affine_epi64_epi8(_mm_cvtsi64_si128((long long)s),
                                                  _mm_set1_epi64x((long long)GF_SGN7), 0);
    return _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(sb));
}

static void iq2xs_vnni_impl(float *y, const void *W, int in, int r0, int r1, int nogather, int gfni)
{
    const size_t rb = row_bytes(KIQ2XS, in);
    __m512i idx4[4];
    for (int j = 0; j < 4; j++)
        idx4[j] = _mm512_setr_epi32(4*j, 4*j, 4*j, 4*j, 4*j+1, 4*j+1, 4*j+1, 4*j+1,
                                    4*j+2, 4*j+2, 4*j+2, 4*j+2, 4*j+3, 4*j+3, 4*j+3, 4*j+3);
    for (int r = r0; r < r1; r++) {
        const biq2xs *b = (const biq2xs *)((const char *)W + (size_t)r * rb);
        __m512 facc = _mm512_setzero_ps();
        for (int i = 0; i < in; i += QK_K, b++) {
            _mm_prefetch((const char *)b + PF, _MM_HINT_T0);
            _mm_prefetch((const char *)b + PF + 64, _MM_HINT_T0);
            const __m512i sc = iq2xs_scales16(b->scales);
            __m512i iacc = _mm512_setzero_si512();
            for (int j = 0; j < 4; j++) {
                const uint16_t *q = b->qs + 8 * j;
                const __m512i mag = iq2xs_mag64(q, nogather);
                const __m512i xq = _mm512_loadu_si512((const void *)(g_xq + i + 64 * j));
                const __m512i xs = _mm512_mask_sub_epi8(xq, gfni ? iq2xs_neg64_gfni(q) : iq2xs_neg64(q),
                                                        _mm512_setzero_si512(), xq);
                const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), mag, xs);
                /* |dp| <= 4*43*127 fits int16, so vpdpwssd on (lo16, 0) is dp*scale */
                iacc = _mm512_dpwssd_epi32(iacc, dp, _mm512_permutexvar_epi32(idx4[j], sc));
            }
            facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc),
                                   _mm512_set1_ps(h2f(b->d) * g_dx[i / QK_K] * 0.125f), facc);
        }
        y[r] = _mm512_reduce_add_ps(facc);
    }
}
static void iq2xs_vnni_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{ (void)x; iq2xs_vnni_impl(y, W, in, r0, r1, 0, 0); }
static void iq2xs_nogather_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{ (void)x; iq2xs_vnni_impl(y, W, in, r0, r1, 1, 0); }
static void iq2xs_gfni_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{ (void)x; iq2xs_vnni_impl(y, W, in, r0, r1, 0, 1); }

#if defined(__AVX10_2__)
/* VPDPBSSD (s8 x s8): the sign goes on the WEIGHTS, so the activations are untouched and
 * a decoded weight vector can be reused across several tokens. */
static void iq2xs_bssd_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{
    (void)x;
    const size_t rb = row_bytes(KIQ2XS, in);
    __m512i idx4[4];
    for (int j = 0; j < 4; j++)
        idx4[j] = _mm512_setr_epi32(4*j, 4*j, 4*j, 4*j, 4*j+1, 4*j+1, 4*j+1, 4*j+1,
                                    4*j+2, 4*j+2, 4*j+2, 4*j+2, 4*j+3, 4*j+3, 4*j+3, 4*j+3);
    for (int r = r0; r < r1; r++) {
        const biq2xs *b = (const biq2xs *)((const char *)W + (size_t)r * rb);
        __m512 facc = _mm512_setzero_ps();
        for (int i = 0; i < in; i += QK_K, b++) {
            const __m512i sc = iq2xs_scales16(b->scales);
            __m512i iacc = _mm512_setzero_si512();
            for (int j = 0; j < 4; j++) {
                const uint16_t *q = b->qs + 8 * j;
                const __m512i mag = iq2xs_mag64(q, 0);
                const __m512i w = _mm512_mask_sub_epi8(mag, iq2xs_neg64(q), _mm512_setzero_si512(), mag);
                const __m512i dp = _mm512_dpbssd_epi32(_mm512_setzero_si512(), w,
                                       _mm512_loadu_si512((const void *)(g_xq + i + 64 * j)));
                iacc = _mm512_dpwssd_epi32(iacc, dp, _mm512_permutexvar_epi32(idx4[j], sc));
            }
            facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc),
                                   _mm512_set1_ps(h2f(b->d) * g_dx[i / QK_K] * 0.125f), facc);
        }
        y[r] = _mm512_reduce_add_ps(facc);
    }
}
#endif

static void iq3xxs_vnni_impl(float *y, const void *W, int in, int r0, int r1, int gfni)
{
    const size_t rb = row_bytes(KIQ3XXS, in);
    const __m256i sh = _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21);
    for (int r = r0; r < r1; r++) {
        const biq3xxs *b = (const biq3xxs *)((const char *)W + (size_t)r * rb);
        __m512 facc = _mm512_setzero_ps();
        for (int i = 0; i < in; i += QK_K, b++) {
            _mm_prefetch((const char *)b + PF, _MM_HINT_T0);
            _mm_prefetch((const char *)b + PF + 64, _MM_HINT_T0);
            const uint8_t *qs = b->qs;
            const uint32_t *sas = (const uint32_t *)(b->qs + QK_K / 4);
            __m512i iacc = _mm512_setzero_si512();
            for (int j = 0; j < 4; j++) {
                uint32_t a0, a1;
                memcpy(&a0, sas + 2 * j, 4); memcpy(&a1, sas + 2 * j + 1, 4);
                const __m512i mag = _mm512_i32gather_epi32(
                    _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(qs + 16 * j))),
                    (const void *)iq3xxs_grid, 4);
                __mmask64 neg;
                if (gfni) {
                    neg = iq3xxs_neg64_gfni(a0, a1);
                } else {
                    const __m256i s7 = _mm256_and_si256(
                        _mm256_srlv_epi32(_mm256_setr_epi32(a0, a0, a0, a0, a1, a1, a1, a1), sh),
                        _mm256_set1_epi32(127));
                    const __m256i par = _mm256_slli_epi32(_mm256_and_si256(_mm256_popcnt_epi32(s7), _mm256_set1_epi32(1)), 7);
                    neg = _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(
                        _mm256_cvtepi32_epi8(_mm256_or_si256(s7, par))));
                }
                const __m512i xq = _mm512_loadu_si512((const void *)(g_xq + i + 64 * j));
                const __m512i xs = _mm512_mask_sub_epi8(xq, neg, _mm512_setzero_si512(), xq);
                const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), mag, xs);
                const int s0 = 2 * (a0 >> 28) + 1, s1 = 2 * (a1 >> 28) + 1;
                iacc = _mm512_add_epi32(iacc, _mm512_mullo_epi32(dp,
                        _mm512_setr_epi32(s0, s0, s0, s0, s0, s0, s0, s0, s1, s1, s1, s1, s1, s1, s1, s1)));
            }
            facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc),
                                   _mm512_set1_ps(h2f(b->d) * g_dx[i / QK_K] * 0.25f), facc);
        }
        y[r] = _mm512_reduce_add_ps(facc);
    }
}

#if defined(__AVX10_2__)
#define NK 12
#else
#define NK 11
#endif
static void iq3xxs_vnni_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{ (void)x; iq3xxs_vnni_impl(y, W, in, r0, r1, 0); }
static void iq3xxs_gfni_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{ (void)x; iq3xxs_vnni_impl(y, W, in, r0, r1, 1); }

/* ---- IQ2_XS repacked at load time ("_r"): the same 74 bytes per block, but the 32 words are
 * reordered so gather g (0..3), slot s holds entry r_entry(g, s): chunk s for g < 2, chunk
 * 8 + s for g >= 2. Every dword lane then meets ONE 16-weight chunk per gather pair, so the
 * scale is applied once per 128 weights. x is permuted the same way, once per token. */
static inline int r_entry(int g, int s) { return g < 2 ? 2 * s + g : 16 + 2 * s + (g - 2); }

static void repack_iq2xs_r(void *W, size_t nblk)
{
    biq2xs *b = (biq2xs *)W;
    for (size_t i = 0; i < nblk; i++) {
        uint16_t t[32], q[32];
        memcpy(q, b[i].qs, 64);
        for (int g = 0; g < 4; g++)
            for (int s = 0; s < 8; s++) t[8 * g + s] = q[r_entry(g, s)];
        memcpy(b[i].qs, t, 64);
    }
}

#define R_BIAS 64
static int8_t   *g_xp;     /* [in] permuted xq */
static int32_t  *g_corr;   /* [in/256][2][16]: R_BIAS * the x a lane meets over gathers {0,1} / {2,3} */
static uint64_t *g_tab;    /* [65536]: magnitude*sign + R_BIAS of the 8 weights of an iq2_xs word */

static void perm_x(int in)
{
    for (int sb = 0; sb < in / QK_K; sb++) {
        const int8_t *src = g_xq + sb * QK_K;
        int8_t *dst = g_xp + sb * QK_K;
        for (int g = 0; g < 4; g++)
            for (int s = 0; s < 8; s++) memcpy(dst + 64 * g + 8 * s, src + 8 * r_entry(g, s), 8);
        int32_t *c = g_corr + sb * 32;
        for (int h = 0; h < 2; h++)
            for (int L = 0; L < 16; L++) {
                int sum = 0;
                for (int gg = 0; gg < 2; gg++)
                    for (int t = 0; t < 4; t++) sum += dst[64 * (2 * h + gg) + 4 * L + t];
                c[16 * h + L] = R_BIAS * sum;
            }
    }
}

static void build_tab(void)
{
    g_tab = aligned_alloc(64, 65536 * 8);
    for (int w = 0; w < 65536; w++) {
        const uint8_t *g = (const uint8_t *)(iq2xs_grid + (w & 511));
        const uint8_t s = ksigns_iq2xs[w >> 9];
        uint8_t v[8];
        for (int j = 0; j < 8; j++) v[j] = (uint8_t)(((s >> j) & 1 ? -(int)g[j] : (int)g[j]) + R_BIAS);
        memcpy(&g_tab[w], v, 8);
    }
}

/* per 64 weights: widen 8 words, ONE gather from the 512 KB signed table, one vpdpbusd */
static void iq2xs_r_tab_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{
    (void)x;
    const size_t rb = row_bytes(KIQ2XS, in);
    const __m512i pa = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const __m512i pb = _mm512_add_epi32(pa, _mm512_set1_epi32(8));
    for (int r = r0; r < r1; r++) {
        const biq2xs *b = (const biq2xs *)((const char *)W + (size_t)r * rb);
        __m512 facc = _mm512_setzero_ps();
        for (int i = 0; i < in; i += QK_K, b++) {
            _mm_prefetch((const char *)b + PF, _MM_HINT_T0);
            _mm_prefetch((const char *)b + PF + 64, _MM_HINT_T0);
            const int8_t *xp = g_xp + i;
            const int32_t *cr = g_corr + (i / QK_K) * 32;
            __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
            for (int g = 0; g < 2; g++) {
                const __m512i m = _mm512_i32gather_epi64(
                    _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(b->qs + 8 * g))), (const void *)g_tab, 8);
                acc0 = _mm512_dpbusd_epi32(acc0, m, _mm512_loadu_si512((const void *)(xp + 64 * g)));
            }
            for (int g = 2; g < 4; g++) {
                const __m512i m = _mm512_i32gather_epi64(
                    _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(b->qs + 8 * g))), (const void *)g_tab, 8);
                acc1 = _mm512_dpbusd_epi32(acc1, m, _mm512_loadu_si512((const void *)(xp + 64 * g)));
            }
            const __m512i sc = iq2xs_scales16(b->scales);
            const __m512i ia = _mm512_add_epi32(
                _mm512_mullo_epi32(_mm512_sub_epi32(acc0, _mm512_loadu_si512((const void *)cr)),
                                   _mm512_permutexvar_epi32(pa, sc)),
                _mm512_mullo_epi32(_mm512_sub_epi32(acc1, _mm512_loadu_si512((const void *)(cr + 16))),
                                   _mm512_permutexvar_epi32(pb, sc)));
            facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia),
                                   _mm512_set1_ps(h2f(b->d) * g_dx[i / QK_K] * 0.125f), facc);
        }
        y[r] = _mm512_reduce_add_ps(facc);
    }
}

/* the 4 KB grid and GFNI signs, but chunk-consistent lanes: no per-64 scale permute */
static void iq2xs_r_gfni_rows(float *y, const float *x, const void *W, int in, int r0, int r1)
{
    (void)x;
    const size_t rb = row_bytes(KIQ2XS, in);
    const __m512i pa = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const __m512i pb = _mm512_add_epi32(pa, _mm512_set1_epi32(8));
    for (int r = r0; r < r1; r++) {
        const biq2xs *b = (const biq2xs *)((const char *)W + (size_t)r * rb);
        __m512 facc = _mm512_setzero_ps();
        for (int i = 0; i < in; i += QK_K, b++) {
            _mm_prefetch((const char *)b + PF, _MM_HINT_T0);
            _mm_prefetch((const char *)b + PF + 64, _MM_HINT_T0);
            const int8_t *xp = g_xp + i;
            const __m512i sc = iq2xs_scales16(b->scales);
            const __m512i sab[2] = { _mm512_permutexvar_epi32(pa, sc), _mm512_permutexvar_epi32(pb, sc) };
            __m512i iacc = _mm512_setzero_si512();
            for (int g = 0; g < 4; g++) {
                const uint16_t *q = b->qs + 8 * g;
                const __m512i xv = _mm512_loadu_si512((const void *)(xp + 64 * g));
                const __m512i xs = _mm512_mask_sub_epi8(xv, iq2xs_neg64_gfni(q), _mm512_setzero_si512(), xv);
                const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), iq2xs_mag64(q, 0), xs);
                iacc = _mm512_dpwssd_epi32(iacc, dp, sab[g >> 1]);
            }
            facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc),
                                   _mm512_set1_ps(h2f(b->d) * g_dx[i / QK_K] * 0.125f), facc);
        }
        y[r] = _mm512_reduce_add_ps(facc);
    }
}
static const rows_fn kfn[NK] = { q8_0_rows, iq2xs_rows, iq2xs_s_rows, iq3xxs_rows,
                                 iq2xs_vnni_rows, iq2xs_nogather_rows, iq3xxs_vnni_rows,
                                 iq2xs_gfni_rows, iq3xxs_gfni_rows, iq2xs_r_tab_rows, iq2xs_r_gfni_rows,
#if defined(__AVX10_2__)
                                 iq2xs_bssd_rows,
#endif
};
static const char *kname2[] = { "q8_0", "iq2_xs", "iq2_xs_scalarlut", "iq3_xxs",
                                "iq2_xs_vnni", "iq2_xs_nogather*", "iq3_xxs_vnni",
                                "iq2_xs_gfni", "iq3_xxs_gfni", "iq2_xs_r_tab", "iq2_xs_r_gfni", "iq2_xs_bssd" };
static int kfmt(int k) { return k == 4 || k == 5 || k == 7 || k >= 9 ? KIQ2XS : k == 6 || k == 8 ? KIQ3XXS : k; }
/* integer variants that must equal another kernel to the bit */
static int ktwin(int k) { return k == 7 || k == 11 ? 4 : k == 8 ? 6 : -1; }
/* repacked layouts, checked against iq2_xs_vnni to rounding (the lanes sum differently) */
static int is_r(int k) { return k == 9 || k == 10; }

/* ------------------------------------------------------------------ data generation */
static __thread uint64_t rng = 88172645463325252ull;
static inline uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static inline uint16_t rand_d(void) { return _cvtss_sh(1e-3f * (1 + (xr() % 1000) / 100.f), 0); }

static void fill_rows(int k, void *W, int in, size_t r0, size_t r1)
{
    const size_t rb = row_bytes(k, in);
    for (size_t r = r0; r < r1; r++) {
        unsigned char *p = (unsigned char *)W + r * rb;
        for (size_t t = 0; t < rb; t++) p[t] = (unsigned char)xr();
        if (k == KQ8_0) for (int b = 0; b < in / 32; b++) ((bq8_0 *)p)[b].d = rand_d();
        else if (k == KIQ3XXS) for (int b = 0; b < in / QK_K; b++) ((biq3xxs *)p)[b].d = rand_d();
        else for (int b = 0; b < in / QK_K; b++) ((biq2xs *)p)[b].d = rand_d();
    }
}

static void deq_row(int k, const void *row, float *y, int in)
{
    if (k == KQ8_0) deq_q8_0(row, y, in);
    else if (k == KIQ3XXS) deq_iq3xxs(row, y, in);
    else deq_iq2xs(row, y, in);
}

static int check(void)
{
    const int shapes[][2] = { { 3584, 64 }, { 3072, 64 }, { 7168, 32 }, { 12288, 16 } };
    int bad = 0;
    for (int k = 0; k < NK; k++)
        for (unsigned s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
            const int in = shapes[s][0], rows = shapes[s][1], f = kfmt(k);
            if (k == 5 || (f == KQ8_0 && k)) continue;
            void *W = aligned_alloc(64, row_bytes(f, in) * rows + 4096);
            float *Wd = aligned_alloc(64, sizeof(float) * in * rows), *x = aligned_alloc(64, sizeof(float) * in);
            float *y0 = calloc(rows, 4), *y1 = calloc(rows, 4);
            fill_rows(f, W, in, 0, rows);
            for (int i = 0; i < in; i++) x[i] = (float)((int)(xr() % 2001) - 1000) / 997.f;
            for (int r = 0; r < rows; r++) deq_row(f, (char *)W + r * row_bytes(f, in), Wd + (size_t)r * in, in);
            ref_rows(y0, x, Wd, in, 0, rows);
            quant_x(x, in);
            void *Wk = W;
            if (is_r(k)) {
                Wk = aligned_alloc(64, row_bytes(f, in) * rows + 4096);
                memcpy(Wk, W, row_bytes(f, in) * rows);
                repack_iq2xs_r(Wk, (size_t)rows * in / QK_K);
            }
            kfn[k](y1, x, Wk, in, 0, rows);
            if (Wk != W) free(Wk);
            double md = 0, ny = 0;
            for (int r = 0; r < rows; r++) { md = fmax(md, fabs((double)y0[r] - y1[r])); ny = fmax(ny, fabs(y0[r])); }
            const int same = !memcmp(y0, y1, sizeof(float) * rows), exact = k < 4;
            printf("check %-18s in=%5d rows=%3d  %s  maxdiff=%.3g (rel to max|y| %.2e)\n", kname2[k], in, rows,
                   same ? "BIT-IDENTICAL" : "DIFFERENT", md, md / ny);
            bad += exact ? !same : md / ny > 2e-2;
            if (ktwin(k) >= 0) {
                kfn[ktwin(k)](y0, x, W, in, 0, rows);
                const int eq = !memcmp(y0, y1, sizeof(float) * rows);
                printf("check %s == %s: %s\n", kname2[k], kname2[ktwin(k)], eq ? "BIT-IDENTICAL" : "DIFFERENT");
                bad += !eq;
            }
            if (is_r(k)) {
                kfn[4](y0, x, W, in, 0, rows);
                double d = 0, n = 0;
                for (int r = 0; r < rows; r++) { d = fmax(d, fabs((double)y0[r] - y1[r])); n = fmax(n, fabs(y0[r])); }
                printf("check %s vs iq2_xs_vnni: max rel diff %.2e\n", kname2[k], d / n);
                bad += d / n > 1e-5;
            }
            free(W); free(Wd); free(x); free(y0); free(y1);
        }
    return bad;
}

static void l2(int k, int in)
{
    const int rows = 64, f = kfmt(k);
    void *W = aligned_alloc(64, row_bytes(f, in) * rows + 4096);
    float *x = aligned_alloc(64, 4 * in), *y = calloc(rows, 4);
    fill_rows(f, W, in, 0, rows);
    if (is_r(k)) repack_iq2xs_r(W, (size_t)rows * in / QK_K);
    for (int i = 0; i < in; i++) x[i] = 0.001f * i;
    quant_x(x, in);
    kfn[k](y, x, W, in, 0, rows);
    int it = 0; double t0 = now_s(), t;
    do { for (int j = 0; j < 50; j++) kfn[k](y, x, W, in, 0, rows); it += 50; } while ((t = now_s() - t0) < 1.0);
    const double w = (double)it * rows * in;
    printf("l2   %-18s in=%5d  %.2f Gweights/s/core  %.1f GB/s/core  (%.2f MB)\n", kname2[k], in,
           w / t * 1e-9, (double)it * rows * row_bytes(f, in) / t * 1e-9, rows * row_bytes(f, in) / 1e6);
    free(W); free(x); free(y);
}

static void dram(int k, int in, double gb)
{
    const int f = k < 0 ? KIQ2XS : kfmt(k);
    const size_t rb = row_bytes(f, in), rows = (size_t)(gb * 1e9 / rb);
    void *W = aligned_alloc(2 << 20, rb * rows + 4096);
    float *x = aligned_alloc(64, 4 * in), *y = aligned_alloc(64, 4 * rows);
    const int nt = omp_get_max_threads();
    #pragma omp parallel
    {
        const int t = omp_get_thread_num();
        const size_t r0 = rows * t / nt, r1 = rows * (t + 1) / nt;
        rng = 88172645463325252ull ^ (uint64_t)(t + 1) * 0x9E3779B97F4A7C15ull;
        fill_rows(f, W, in, r0, r1);
        if (k >= 0 && is_r(k)) repack_iq2xs_r((char *)W + r0 * rb, (r1 - r0) * in / QK_K);
        memset(y + r0, 0, 4 * (r1 - r0));
    }
    for (int i = 0; i < in; i++) x[i] = 0.001f * i;
    quant_x(x, in);
    double best = 1e30, sum = 0;
    for (int rep = 0; rep < 6; rep++) {
        const double t0 = now_s();
        if (k < 0) {
            #pragma omp parallel reduction(+:sum)
            {
                const int t = omp_get_thread_num();
                const size_t n = rb * rows / 64, c0 = n * t / nt, c1 = n * (t + 1) / nt;
                const __m512i *p = (const __m512i *)W;
                __m512i s[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
                size_t c = c0;
                for (; c + 3 < c1; c += 4)
                    for (int u = 0; u < 4; u++) s[u] = _mm512_add_epi64(s[u], _mm512_load_si512(p + c + u));
                sum += (double)_mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(s[0], s[1]), _mm512_add_epi64(s[2], s[3])));
            }
        } else {
            #pragma omp parallel
            {
                const int t = omp_get_thread_num();
                kfn[k](y, x, W, in, (int)(rows * t / nt), (int)(rows * (t + 1) / nt));
            }
        }
        const double dt = now_s() - t0;
        if (rep && dt < best) best = dt;
    }
    printf("dram %-18s in=%5d threads=%3d  %.1f GB/s  %.1f Gweights/s  (%.2f GB, chk %g)\n",
           k < 0 ? "read-roof" : kname2[k], in, nt, rb * rows / best * 1e-9,
           k < 0 ? 0 : rows * (double)in / best * 1e-9, rb * rows * 1e-9, sum + y[rows / 2]);
    free(W); free(x); free(y);
}

static int kind(const char *s)
{
    if (!strcmp(s, "roof")) return -1;
    if (!strcmp(s, "all")) return -2;
    for (int k = 0; k < NK; k++) if (!strcmp(s, kname2[k])) return k;
    fprintf(stderr, "unknown kernel %s\n", s); exit(2);
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "check";
    g_xq = aligned_alloc(64, 1 << 16);
    g_dx = aligned_alloc(64, 1 << 12);
    g_xp = aligned_alloc(64, 1 << 16);
    g_corr = aligned_alloc(64, (1 << 16) / QK_K * 32 * 4);
    build_tab();
    if (!strcmp(mode, "check")) return check() ? 1 : 0;
    const int k0 = argc > 2 ? kind(argv[2]) : -2;
    if (!strcmp(mode, "l2")) {
        for (int k = 0; k < NK; k++)
            if (k0 == -2 || k0 == k) { l2(k, k == KQ8_0 ? 7168 : 3584); if (k != KQ8_0) l2(k, 3072); }
        return 0;
    }
    if (!strcmp(mode, "dram")) {
        const double gb = argc > 3 ? atof(argv[3]) : 4.0;
        for (int k = -1; k < NK; k++)
            if (k0 == -2 || k0 == k) dram(k, k == KQ8_0 || k < 0 ? 7168 : 3584, gb);
        return 0;
    }
    fprintf(stderr, "usage: %s check | l2 [kernel] | dram [kernel|roof] [GB]\n", argv[0]);
    return 2;
}

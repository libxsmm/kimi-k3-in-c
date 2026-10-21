/* SPDX-License-Identifier: Apache-2.0 */
/* k3_simd.h - AVX-512 helpers shared by the attention kernels (k3_ops.c, k3_dspark.c). */
#ifndef K3_SIMD_H
#define K3_SIMD_H

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#include <immintrin.h>
#include <stdint.h>

static inline __m512 bf16x16(const uint16_t *p)
{
    return _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}

/* e^x to ~2 ulp (Cephes polynomial on x - n ln2, then 2^n by scalef); x >= -87.3 */
static inline __m512 exp512(__m512 x)
{
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.3f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500e-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), r);
    return _mm512_scalef_ps(_mm512_add_ps(p, _mm512_set1_ps(1.0f)), n);
}

/* lane j = the horizontal sum of a[j], by a fixed tree */
static inline __m512 v512_sum16(const __m512 a[16])
{
    __m512 s[8], v[4];
    for (int i = 0; i < 8; i++)
        s[i] = _mm512_add_ps(_mm512_unpacklo_ps(a[2 * i], a[2 * i + 1]),
                             _mm512_unpackhi_ps(a[2 * i], a[2 * i + 1]));
    for (int k = 0; k < 4; k++)
        v[k] = _mm512_add_ps(_mm512_shuffle_ps(s[2 * k], s[2 * k + 1], 0x44),
                             _mm512_shuffle_ps(s[2 * k], s[2 * k + 1], 0xEE));
    const __m512 w0 = _mm512_add_ps(_mm512_shuffle_f32x4(v[0], v[1], 0x88), _mm512_shuffle_f32x4(v[0], v[1], 0xDD));
    const __m512 w1 = _mm512_add_ps(_mm512_shuffle_f32x4(v[2], v[3], 0x88), _mm512_shuffle_f32x4(v[2], v[3], 0xDD));
    return _mm512_add_ps(_mm512_shuffle_f32x4(w0, w1, 0x88), _mm512_shuffle_f32x4(w0, w1, 0xDD));
}

/* r[i][j] -> r[j][i] for a 16x16 block */
static inline void v512_tr16(__m512 r[16])
{
    __m512 t[16];
    for (int i = 0; i < 8; i++) {
        t[2 * i]     = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
    }
    for (int i = 0; i < 4; i++) {                  /* r[4i+c] lane L: rows 4i..4i+3 of column 4L+c */
        r[4 * i]     = _mm512_shuffle_ps(t[4 * i], t[4 * i + 2], 0x44);
        r[4 * i + 1] = _mm512_shuffle_ps(t[4 * i], t[4 * i + 2], 0xEE);
        r[4 * i + 2] = _mm512_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0x44);
        r[4 * i + 3] = _mm512_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0xEE);
    }
    for (int c = 0; c < 4; c++) {
        const __m512 u0 = _mm512_shuffle_f32x4(r[c], r[4 + c], 0x88), u1 = _mm512_shuffle_f32x4(r[c], r[4 + c], 0xDD);
        const __m512 u2 = _mm512_shuffle_f32x4(r[8 + c], r[12 + c], 0x88), u3 = _mm512_shuffle_f32x4(r[8 + c], r[12 + c], 0xDD);
        t[c]      = _mm512_shuffle_f32x4(u0, u2, 0x88);
        t[4 + c]  = _mm512_shuffle_f32x4(u1, u3, 0x88);
        t[8 + c]  = _mm512_shuffle_f32x4(u0, u2, 0xDD);
        t[12 + c] = _mm512_shuffle_f32x4(u1, u3, 0xDD);
    }
    for (int i = 0; i < 16; i++) r[i] = t[i];
}
#endif
#endif

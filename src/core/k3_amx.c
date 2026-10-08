/* SPDX-License-Identifier: Apache-2.0 */
/* k3_amx.c - see k3_amx.h.
 *
 * Work is split over (row block, token block) tasks so every thread gets one; within a
 * task the K dimension is walked in chunks so that the decoded weight panel (row block x
 * K chunk) and the token block's activations (token block x K chunk) both stay in L2.
 * Per task: for each K chunk, decode the panel once, then every token tile pair streams
 * through every weight tile pair (2x2 tiles of fp32 accumulators, kept in a per-thread
 * buffer between chunks). */
#define _GNU_SOURCE
#include "k3_amx.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"
#include "k3_gq.h"

#if defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AVX512BF16__) && defined(_OPENMP)
#define K3_AMX 1
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

size_t k3_amx_xv_elems(int T, int in)
{
    const size_t mt = (size_t)((T + 31) / 32) * 2;   /* token tiles, padded to pairs */
    return mt * (size_t)(in / 32) * 512;
}

#if !defined(K3_AMX)
int k3_amx_ok(void) { return 0; }
void k3_amx_pack_x(uint16_t *Xv, const float *X, int ldx, int T, int in)
{ (void)Xv; (void)X; (void)ldx; (void)T; (void)in; }
void k3_amx_gemm(float *Y, int ldy, const uint16_t *Xv, int T, const void *W, int wt,
                 int in, int wbase, int r0, int r1)
{ (void)Y; (void)ldy; (void)Xv; (void)T; (void)W; (void)wt; (void)in; (void)wbase; (void)r0; (void)r1; }
void k3_amx_gemm_groups(const K3AmxGroup *g, int ng, const uint16_t *Xv, int in, int r0, int r1)
{ (void)g; (void)ng; (void)Xv; (void)in; (void)r0; (void)r1; }
void k3_amx_pack_rows(uint16_t *Xv, const float *X, int ldx, const int *map, int N, int in)
{ (void)Xv; (void)X; (void)ldx; (void)map; (void)N; (void)in; }
void k3_amx_pack_cols(uint16_t *Xv, const float *X, int ldx, const int *map, int N, int in)
{ (void)Xv; (void)X; (void)ldx; (void)map; (void)N; (void)in; }
#else

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

int k3_amx_ok(void)
{
    static int ok = -1;
    if (ok < 0) {
        unsigned a, b, c, d;
        __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(7), "c"(0));
        const int cpu = (d >> 22 & 1) && (d >> 24 & 1);   /* AMX-BF16, AMX-TILE */
        const char *e = getenv("K3_AMX");
        ok = cpu && !(e && !strcmp(e, "0")) &&
             syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
    }
    return ok;
}

typedef struct {
    uint8_t  palette, start_row, rsv[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
} __attribute__((packed)) TileCfg;

static __thread int tl_cfg;
static __thread uint16_t *tl_panel;
static __thread size_t tl_panel_n;
static __thread float *tl_c;
static __thread size_t tl_c_n;

static void tiles_on(void)
{
    if (tl_cfg) return;
    TileCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette = 1;
    for (int i = 0; i < 8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
    _tile_loadconfig(&cfg);
    tl_cfg = 1;
}

static void *grow(void *p, size_t *have, size_t need)
{
    if (*have >= need) return p;
    free(p);
    *have = need;
    return aligned_alloc(64, (need + 63) & ~(size_t)63);
}

/* Tile (mt, kt) holds tokens mt*16 + c and k = kt*32 + 2j + e at [j][c][e]. */
void k3_amx_pack_x(uint16_t *Xv, const float *X, int ldx, int T, int in)
{
    const int KT = in / 32, MT = (T + 31) / 32 * 2;
    int lo, hi;
    k3_split(MT * KT, &lo, &hi);
    const __m512i pos = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                           _mm512_set1_epi32(16));   /* dword j*16: row j of the tile */
    for (int i = lo; i < hi; i++) {
        const int mt = i / KT, kt = i % KT;
        uint16_t *tile = Xv + (size_t)i * 512;
        for (int c = 0; c < 16; c++) {
            const int t = mt * 16 + c;
            __m512i v = _mm512_setzero_si512();
            if (t < T) {
                const float *x = X + (size_t)t * ldx + (size_t)kt * 32;
                v = (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + 16), _mm512_loadu_ps(x));
            }
            /* dword j (k = 2j, 2j + 1) of token c goes to row j, column c */
            _mm512_i32scatter_epi32((void *)(tile + c * 2), pos, v, 4);
        }
    }
}

/* Rows [n0, n0 + nr) x k tiles [kc0, kc0 + kcn) of W into ntp (even) A tiles per kt:
 * panel[(nt * kcn + kt) * 512 + row * 32 + k]; rows past nr are zero. */
static void upconvert(uint16_t *panel, int ntp, const void *W, int wt, int in, size_t rb,
                      int wbase, int n0, int nr, int kc0, int kcn)
{
    const size_t koff = k3_gq_row_bytes(wt, (int64_t)kc0 * 32);
    for (int r = 0; r < ntp * 16; r++) {
        uint16_t *dst = panel + (size_t)(r / 16) * kcn * 512 + (size_t)(r % 16) * 32;
        if (r >= nr) {
            for (int kt = 0; kt < kcn; kt++) memset(dst + (size_t)kt * 512, 0, 64);
            continue;
        }
        const unsigned char *src = (const unsigned char *)W + (size_t)(n0 + r - wbase) * rb + koff;
        k3_gq_to_bf16(wt, src, (int64_t)kcn * 32, dst, 512);
    }
}

static int env_int(const char *name, int def)
{
    const char *e = getenv(name);
    return e && atoi(e) > 0 ? atoi(e) : def;
}

static int kc_max, rb_max, tb_max;
static pthread_once_t blk_once = PTHREAD_ONCE_INIT;
static void blk_init(void)
{
    kc_max = env_int("K3_AMX_KC", 1024) / 256 * 8;
    rb_max = env_int("K3_AMX_RB", 256) / 32 * 32;
    tb_max = env_int("K3_AMX_TB", 128) / 32 * 32;
    if (kc_max < 8) kc_max = 8;
    if (rb_max < 32) rb_max = 32;
    if (tb_max < 32) tb_max = 32;
}

void k3_amx_gemm(float *Y, int ldy, const uint16_t *Xv, int T, const void *W, int wt,
                 int in, int wbase, int r0, int r1)
{
    const int KT = in / 32, MT = (T + 31) / 32 * 2, rows = r1 - r0;
    if (rows <= 0 || T <= 0) return;
    const size_t rb = k3_gq_row_bytes(wt, in);

    /* blocks: K chunk of KC k-tiles (a multiple of 8, i.e. of 256 weights), row blocks
     * of RB and token blocks of TB (multiples of 32), halved until every thread has a task */
    pthread_once(&blk_once, blk_init);
    const int KC = KT < kc_max ? KT : kc_max;
    int RB = rb_max, TB = tb_max;
    const int Tp = MT * 16, nth = k3_nth();
    if (TB > Tp) TB = Tp;
    for (;;) {
        const long tasks = (long)((rows + RB - 1) / RB) * ((Tp + TB - 1) / TB);
        if (tasks >= nth || (RB == 32 && TB == 32)) break;
        if (TB >= RB && TB > 32) TB /= 2;
        else if (RB > 32) RB /= 2;
        else TB /= 2;
        TB = TB < 32 ? 32 : TB / 32 * 32;
    }
    const int nrb = (rows + RB - 1) / RB, ntb = (Tp + TB - 1) / TB;
    int lo, hi;
    k3_split(nrb * ntb, &lo, &hi);
    if (hi <= lo) return;
    tiles_on();
    tl_panel = (uint16_t *)grow(tl_panel, &tl_panel_n, (size_t)RB / 16 * KC * 1024 + 2048);
    tl_c = (float *)grow(tl_c, &tl_c_n, (size_t)(RB / 16 + 1) * (TB / 16) * 1024);

    for (int task = lo; task < hi; task++) {
        const int n0 = r0 + task / ntb * RB, nr = r1 - n0 < RB ? r1 - n0 : RB;
        const int ntp = ((nr + 15) / 16 + 1) & ~1;
        const int mt0 = task % ntb * TB / 16;
        const int nmt = (Tp - mt0 * 16 < TB ? Tp - mt0 * 16 : TB) / 16;
        for (int kc0 = 0; kc0 < KT; kc0 += KC) {
            const int kcn = KT - kc0 < KC ? KT - kc0 : KC;
            upconvert(tl_panel, ntp, W, wt, in, rb, wbase, n0, nr, kc0, kcn);
            for (int mt = 0; mt < nmt; mt += 2) {
                const uint16_t *b0 = Xv + ((size_t)(mt0 + mt) * KT + kc0) * 512;
                const uint16_t *b1 = b0 + (size_t)KT * 512;
                for (int nt = 0; nt < ntp; nt += 2) {
                    const uint16_t *a0 = tl_panel + (size_t)nt * kcn * 512, *a1 = a0 + (size_t)kcn * 512;
                    float *c0 = tl_c + ((size_t)nt * nmt + mt) * 256, *c1 = c0 + 256;
                    float *c2 = c0 + (size_t)nmt * 256, *c3 = c2 + 256;
                    if (kc0 == 0) { _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); }
                    else {
                        _tile_loadd(0, c0, 64); _tile_loadd(1, c1, 64);
                        _tile_loadd(2, c2, 64); _tile_loadd(3, c3, 64);
                    }
                    for (int kt = 0; kt < kcn; kt++) {
                        _tile_loadd(4, a0 + (size_t)kt * 512, 64);
                        _tile_loadd(6, b0 + (size_t)kt * 512, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, b1 + (size_t)kt * 512, 64);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(5, a1 + (size_t)kt * 512, 64);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_dpbf16ps(3, 5, 7);
                    }
                    _tile_stored(0, c0, 64); _tile_stored(1, c1, 64);
                    _tile_stored(2, c2, 64); _tile_stored(3, c3, 64);
                }
            }
        }
        /* C tiles [16 rows][16 tokens] -> Y[t][n], one token's 16 rows at a time */
        for (int nt = 0; nt * 16 < nr; nt++) {
            const int nv = nr - nt * 16 < 16 ? nr - nt * 16 : 16;
            const __mmask16 m = (__mmask16)((1u << nv) - 1);
            for (int mt = 0; mt < nmt; mt++) {
                const float *c = tl_c + ((size_t)nt * nmt + mt) * 256;
                for (int j = 0; j < 16 && (mt0 + mt) * 16 + j < T; j++) {
                    const __m512 col = _mm512_i32gather_ps(_mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                        128, 144, 160, 176, 192, 208, 224, 240), c + j, 4);
                    _mm512_mask_storeu_ps(Y + (size_t)((mt0 + mt) * 16 + j) * ldy + n0 + nt * 16, m, col);
                }
            }
        }
    }
}

/* ------------------------------------------------------------- grouped (MoE) */
/* One task: rows [n0, n0 + nr) of W against token tiles [mt0, mt0 + nmt) of Xv (any
 * count; an odd last tile runs a 2x1 tile step), into Y[t * ldy + n - ycol0] for t < tend. */
static void run_block(float *Y, int ldy, int ycol0, const uint16_t *Xv, int KT, int mt0, int nmt,
                      int tend, const void *W, int wt, int in, size_t rb, int wbase, int n0, int nr)
{
    pthread_once(&blk_once, blk_init);
    const int KC = KT < kc_max ? KT : kc_max;
    const int ntp = ((nr + 15) / 16 + 1) & ~1;
    tl_panel = (uint16_t *)grow(tl_panel, &tl_panel_n, (size_t)ntp * KC * 1024 + 2048);
    tl_c = (float *)grow(tl_c, &tl_c_n, (size_t)ntp * (nmt + 1) * 1024);
    for (int kc0 = 0; kc0 < KT; kc0 += KC) {
        const int kcn = KT - kc0 < KC ? KT - kc0 : KC;
        upconvert(tl_panel, ntp, W, wt, in, rb, wbase, n0, nr, kc0, kcn);
        for (int mt = 0; mt < nmt; mt += 2) {
            const int two = mt + 1 < nmt;
            const uint16_t *b0 = Xv + ((size_t)(mt0 + mt) * KT + kc0) * 512;
            const uint16_t *b1 = b0 + (size_t)KT * 512;
            for (int nt = 0; nt < ntp; nt += 2) {
                const uint16_t *a0 = tl_panel + (size_t)nt * kcn * 512, *a1 = a0 + (size_t)kcn * 512;
                float *c0 = tl_c + ((size_t)nt * nmt + mt) * 256, *c1 = c0 + 256;
                float *c2 = c0 + (size_t)nmt * 256, *c3 = c2 + 256;
                if (two) {
                    if (kc0 == 0) { _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); }
                    else {
                        _tile_loadd(0, c0, 64); _tile_loadd(1, c1, 64);
                        _tile_loadd(2, c2, 64); _tile_loadd(3, c3, 64);
                    }
                    for (int kt = 0; kt < kcn; kt++) {
                        _tile_loadd(4, a0 + (size_t)kt * 512, 64);
                        _tile_loadd(6, b0 + (size_t)kt * 512, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, b1 + (size_t)kt * 512, 64);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(5, a1 + (size_t)kt * 512, 64);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_dpbf16ps(3, 5, 7);
                    }
                    _tile_stored(0, c0, 64); _tile_stored(1, c1, 64);
                    _tile_stored(2, c2, 64); _tile_stored(3, c3, 64);
                } else {
                    if (kc0 == 0) { _tile_zero(0); _tile_zero(2); }
                    else { _tile_loadd(0, c0, 64); _tile_loadd(2, c2, 64); }
                    for (int kt = 0; kt < kcn; kt++) {
                        _tile_loadd(6, b0 + (size_t)kt * 512, 64);
                        _tile_loadd(4, a0 + (size_t)kt * 512, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(5, a1 + (size_t)kt * 512, 64);
                        _tile_dpbf16ps(2, 5, 6);
                    }
                    _tile_stored(0, c0, 64); _tile_stored(2, c2, 64);
                }
            }
        }
    }
    const __m512i gi = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                         128, 144, 160, 176, 192, 208, 224, 240);
    for (int nt = 0; nt * 16 < nr; nt++) {
        const int nv = nr - nt * 16 < 16 ? nr - nt * 16 : 16;
        const __mmask16 m = (__mmask16)((1u << nv) - 1);
        for (int mt = 0; mt < nmt; mt++) {
            const float *c = tl_c + ((size_t)nt * nmt + mt) * 256;
            for (int j = 0; j < 16 && (mt0 + mt) * 16 + j < tend; j++)
                _mm512_mask_storeu_ps(Y + (size_t)((mt0 + mt) * 16 + j) * ldy + n0 + nt * 16 - ycol0, m,
                                      _mm512_i32gather_ps(gi, c + j, 4));
        }
    }
}

static int grp_next;

void k3_amx_gemm_groups(const K3AmxGroup *g, int ng, const uint16_t *Xv, int in, int r0, int r1)
{
    const int KT = in / 32, rows = r1 - r0;
    if (rows <= 0 || ng <= 0) return;
    pthread_once(&blk_once, blk_init);
    const int RB = rows < 64 ? (rows + 15) / 16 * 16 : 64;
    const int nrb = (rows + RB - 1) / RB, ntask = ng * nrb;
    k3_sync();                               /* nobody still takes tasks of a previous call */
    if (k3_tid() == 0) __atomic_store_n(&grp_next, 0, __ATOMIC_RELAXED);
    k3_sync();
    tiles_on();
    for (;;) {
        const int task = __atomic_fetch_add(&grp_next, 1, __ATOMIC_RELAXED);
        if (task >= ntask) break;
        const K3AmxGroup *q = &g[task / nrb];
        const int n0 = r0 + task % nrb * RB, nr = r1 - n0 < RB ? r1 - n0 : RB;
        if (q->n <= 0 || !q->W) continue;
        run_block(q->Y, q->ldy, q->ycol0, Xv, KT, q->t0 / 16, (q->n + 15) / 16, q->t0 + q->n,
                  q->W, q->wt, in, k3_gq_row_bytes(q->wt, in), q->wbase, n0, nr);
    }
}

/* tile (mt, kt) row j = k pair (2j, 2j + 1) of 16 tokens */
void k3_amx_pack_rows(uint16_t *Xv, const float *X, int ldx, const int *map, int N, int in)
{
    const int KT = in / 32, MT = N / 16;
    const __m512i pos = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                           _mm512_set1_epi32(16));
    int lo, hi;
    k3_split(MT * KT, &lo, &hi);
    for (int i = lo; i < hi; i++) {
        const int mt = i / KT, kt = i % KT;
        uint16_t *tile = Xv + (size_t)i * 512;
        for (int c = 0; c < 16; c++) {
            const int r = map[mt * 16 + c];
            __m512i v = _mm512_setzero_si512();
            if (r >= 0) {
                const float *x = X + (size_t)r * ldx + (size_t)kt * 32;
                v = (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + 16), _mm512_loadu_ps(x));
            }
            _mm512_i32scatter_epi32((void *)(tile + c * 2), pos, v, 4);
        }
    }
}

void k3_amx_pack_cols(uint16_t *Xv, const float *X, int ldx, const int *map, int N, int in)
{
    const int KT = in / 32, MT = N / 16;
    /* [a0..a15 | b0..b15] -> a0 b0 a1 b1 ... */
    static const uint16_t ilv16[32] __attribute__((aligned(64))) = {
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31 };
    const __m512i ilv = _mm512_load_si512((const void *)ilv16);
    int lo, hi;
    k3_split(MT * KT, &lo, &hi);
    for (int i = lo; i < hi; i++) {
        const int mt = i / KT, kt = i % KT;
        uint16_t *tile = Xv + (size_t)i * 512;
        int nv = 0;
        while (nv < 16 && map[mt * 16 + nv] >= 0) nv++;
        const __mmask16 m = (__mmask16)((1u << nv) - 1);
        const float *x = nv ? X + map[mt * 16] : X;
        for (int j = 0; j < 16; j++) {
            const __m512 a = _mm512_maskz_loadu_ps(m, x + (size_t)(kt * 32 + 2 * j) * ldx);
            const __m512 b = _mm512_maskz_loadu_ps(m, x + (size_t)(kt * 32 + 2 * j + 1) * ldx);
            _mm512_storeu_si512((void *)(tile + j * 32),
                                _mm512_permutexvar_epi16(ilv, (__m512i)_mm512_cvtne2ps_pbh(b, a)));
        }
    }
}
#endif

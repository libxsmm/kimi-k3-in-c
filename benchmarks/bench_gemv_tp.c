/* bench_gemv_tp.c - every decode GEMV of the GGUF (UD-Q2_K_XL) model at its per-rank
 * tensor-parallel shape, streamed from DRAM.
 *
 * Each call reads the next copy of its matrix from a pool much larger than the LLC, so
 * every byte comes from memory, as in decode. The team splits rows as the engine does
 * (k3_split, or expert_spans for the routed experts) and meets at k3_sync after every
 * call. Reports GB/s against a read roof measured the same way, and the projected
 * ms/token per engine profile phase (compare with K3_PROF=1 output).
 *
 *   OMP_NUM_THREADS=42 OMP_PLACES=cores OMP_PROC_BIND=close numactl -N 0 -m 0 \
 *       bin/bench_gemv_tp [-P ranks] [-g pool_GB] [-t rows|t0] [-f filter]
 *
 * -t rows: each page is first touched by the thread that computes it (default);
 * -t t0:   thread 0 touches everything (what a single loader thread would do). */
#define _GNU_SOURCE
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "k3.h"
#include "k3_gq.h"

/* Kimi K3 */
enum { H = 7168, KP = 12288, NH = 96, QL = 1536, KVW = 576, KVD = 256, VH = 128,
       NE = 896, LAT = 3584, INTER = 3072, SI = 6144, DI = 33792, VOCAB = 163840, TOPK = 16 };

typedef struct {
    const char *name, *phase;
    int type, rows, in, split, count;   /* rows of the full model; count = calls per token */
} Case;

static const Case cases[] = {
    { "kda q",        "kda proj",  K3_GG_Q8_0, KP,        H,     1, 69 },
    { "kda k",        "kda proj",  K3_GG_Q8_0, KP,        H,     1, 69 },
    { "kda v",        "kda proj",  K3_GG_Q8_0, KP,        H,     1, 69 },
    { "kda f_a",      "kda proj",  K3_GG_Q8_0, 128,       H,     0, 69 },
    { "kda f_b",      "kda proj",  K3_GG_Q8_0, KP,        128,   1, 69 },
    { "kda g",        "kda out",   K3_GG_Q8_0, KP,        H,     1, 69 },
    { "kda o",        "kda out",   K3_GG_Q8_0, H,         KP,    1, 69 },
    { "mla q_a",      "mla proj",  K3_GG_Q8_0, QL,        H,     1, 24 },
    { "mla q_b",      "mla proj",  K3_GG_Q8_0, NH * 192,  QL,    1, 24 },
    { "mla kv_a",     "mla proj",  K3_GG_Q8_0, KVW,       H,     1, 24 },
    { "mla kv_b f32", "mla proj",  K3_GG_F32,  NH * KVD,  512,   1, 24 },
    { "mla g",        "mla out",   K3_GG_Q8_0, NH * VH,   H,     1, 24 },
    { "mla o",        "mla out",   K3_GG_Q8_0, H,         NH * VH, 1, 24 },
    { "router f32",   "router",    K3_GG_F32,  NE,        H,     1, 92 },
    { "latent down",  "moe down",  K3_GG_Q8_0, LAT,       H,     1, 92 },
    { "latent up",    "moe up",    K3_GG_Q8_0, H,         LAT,   1, 92 },
    { "shared gate",  "shared",    K3_GG_Q8_0, SI,        H,     1, 92 },
    { "shared up",    "shared",    K3_GG_Q8_0, SI,        H,     1, 92 },
    { "shared down",  "shared",    K3_GG_Q8_0, H,         SI,    1, 92 },
    { "dense gate",   "dense mlp", K3_GG_Q8_0, DI,        H,     1, 1 },
    { "dense up",     "dense mlp", K3_GG_Q8_0, DI,        H,     1, 1 },
    { "dense down",   "dense mlp", K3_GG_Q8_0, H,         DI,    1, 1 },
    { "lm_head",      "head",      K3_GG_Q8_0, VOCAB,     H,     1, 1 },
};

/* routed experts: type, calls per token of gate|up (layers) and of down */
static const struct { const char *name; int type, gu_count, dn_count; } xcases[] = {
    { "experts iq2_xs",  K3_GG_IQ2_XS,  91, 81 },
    { "experts iq3_xxs", K3_GG_IQ3_XXS, 1,  11 },
};

static size_t pool_bytes = (size_t)2 << 30;
static int touch_t0 = 0;
static double roof_gbs = 0;

#define NPH 12
static const char *ph_name[NPH];
static double ph_ms[NPH], ph_gb[NPH];
static void phase_add(const char *ph, double ms, double gb)
{
    int i = 0;
    while (i < NPH && ph_name[i] && strcmp(ph_name[i], ph)) i++;
    if (i == NPH) return;
    ph_name[i] = ph; ph_ms[i] += ms; ph_gb[i] += gb;
}

static void *huge_alloc(size_t n)
{
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, n, MADV_HUGEPAGE);
    return p;
}

/* Valid-looking bytes for [lo, hi) of a buffer of rows of type t: random codes, sane
 * fp16 block scales (the leading 2 bytes of every block) and finite fp32. */
static void fill_rows(unsigned char *W, int t, int in, size_t r0, size_t r1)
{
    const size_t rb = k3_gq_row_bytes(t, in);
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (r0 * 0xBF58476D1CE4E5B9ull);
    if (t == K3_GG_F32) {
        float *f = (float *)(W + r0 * rb);
        for (size_t i = 0; i < (r1 - r0) * (size_t)in; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            f[i] = (float)((int)(s >> 40) - (1 << 23)) * (1.0f / (1 << 28));
        }
        return;
    }
    unsigned char *p = W + r0 * rb, *e = W + r1 * rb;
    for (; p + 8 <= e; p += 8) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; memcpy(p, &s, 8); }
    for (; p < e; p++) *p = (unsigned char)s;
    const size_t bb = t == K3_GG_Q8_0 ? 34 : t == K3_GG_IQ2_XS ? 74 : 98;
    const uint16_t d = 0x2000;   /* 2^-7 */
    for (size_t r = r0; r < r1; r++)
        for (size_t b = 0; b < rb / bb; b++) memcpy(W + r * rb + b * bb, &d, 2);
}

/* ------------------------------------------------------------------ read roof */
static double read_roof(void)
{
    const size_t n = pool_bytes;
    unsigned char *buf = huge_alloc(n);
    double t = 0, sink = 0;
#pragma omp parallel reduction(+ : sink)
    {
        const size_t nt = omp_get_num_threads(), me = omp_get_thread_num();
        const size_t lo = n / 64 * me / nt * 64, hi = n / 64 * (me + 1) / nt * 64;
        memset(buf + lo, 1, hi - lo);
        k3_sync();
        for (int it = 0; it < 6; it++) {
            k3_sync();
            const double t0 = omp_get_wtime();
#if defined(__AVX512F__)
            __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
            for (size_t i = lo; i + 256 <= hi; i += 256) {
                a0 = _mm512_xor_si512(a0, _mm512_load_si512(buf + i));
                a1 = _mm512_xor_si512(a1, _mm512_load_si512(buf + i + 64));
                a2 = _mm512_xor_si512(a2, _mm512_load_si512(buf + i + 128));
                a3 = _mm512_xor_si512(a3, _mm512_load_si512(buf + i + 192));
            }
            sink += (double)_mm512_reduce_add_epi32(
                _mm512_xor_si512(_mm512_xor_si512(a0, a1), _mm512_xor_si512(a2, a3)));
#else
            uint64_t a = 0;
            for (size_t i = lo; i + 8 <= hi; i += 8) a ^= *(const uint64_t *)(buf + i);
            sink += (double)a;
#endif
            k3_sync();
            if (it > 0 && me == 0) t += omp_get_wtime() - t0;
        }
    }
    munmap(buf, n);
    if (sink == 12345.678) printf(" ");
    return (double)n * 5 / t / 1e9;
}

/* ------------------------------------------------------------------ trunk GEMVs */
static void run_case(const Case *c, int P)
{
    const int rows = c->split ? (c->rows + P - 1) / P : c->rows;
    const size_t rb = k3_gq_row_bytes(c->type, c->in);
    const size_t mat = (size_t)rows * rb, stride = (mat + 4095) & ~(size_t)4095;
    size_t nc = pool_bytes / stride;
    if (nc < 2) nc = 2;
    unsigned char *pool = huge_alloc(nc * stride);
    float *x = aligned_alloc(64, ((size_t)c->in * 4 + 127) & ~(size_t)63);
    float *y = aligned_alloc(64, ((size_t)rows * 4 + 127) & ~(size_t)63);
    for (int i = 0; i < c->in; i++) x[i] = (float)((i * 37) % 101 - 50) * 0.01f;
    const long iters = nc * 4 > 64 ? (long)nc * 4 : 64;
    double t = 0;
#pragma omp parallel
    {
        int lo, hi;
        k3_split(rows, &lo, &hi);
        if (!touch_t0)
            for (size_t k = 0; k < nc; k++) fill_rows(pool + k * stride, c->type, c->in, lo, hi);
        else if (omp_get_thread_num() == 0)
            for (size_t k = 0; k < nc; k++) fill_rows(pool + k * stride, c->type, c->in, 0, rows);
        k3_sync();
        double t0 = 0;
        for (long it = -(long)nc; it < iters; it++) {           /* one warm pass */
            if (it == 0) { k3_sync(); t0 = omp_get_wtime(); }
            const unsigned char *W = pool + (size_t)((it + nc) % nc) * stride;
            if (c->type == K3_GG_F32) k3_matmul(y, x, (const float *)W, c->in, rows);
            else                      k3_q80_rows(y, x, W, c->in, lo, hi);
            k3_sync();
        }
        if (omp_get_thread_num() == 0) t = omp_get_wtime() - t0;
    }
    const double us = t / iters * 1e6, gbs = (double)mat / (t / iters) / 1e9;
    const double ms_tok = us * c->count / 1e3;
    printf("%-16s %-8s %6d x %-6d %8.2f MB %4d %9.2f us %7.1f GB/s %5.1f%% %8.3f ms/tok  y0=%g\n",
           c->name, c->type == K3_GG_F32 ? "f32" : "q8_0", rows, c->in, mat / 1e6, c->count,
           us, gbs, 100 * gbs / roof_gbs, ms_tok, y[0]);
    phase_add(c->phase, ms_tok, (double)mat * c->count / 1e9);
    munmap(pool, nc * stride);
    free(x); free(y);
}

/* ------------------------------------------------------------------ routed experts */
/* This thread's share of nq x n rows as (expert slot, [r0, r1)) spans: k3_ops.c's split. */
static int spans(int nq, int n, int *sj, int *s0, int *s1)
{
    int lo, hi, ns = 0;
    k3_split(nq * n, &lo, &hi);
    for (int s = lo; s < hi; ) {
        const int j = s / n, e = (j + 1) * n < hi ? (j + 1) * n : hi;
        sj[ns] = j; s0[ns] = s - j * n; s1[ns] = e - j * n; ns++;
        s = e;
    }
    return ns;
}

static void run_experts(int xi, int P, int q8)
{
    const int t = xcases[xi].type;
    const int ngu = (INTER + P - 1) / P, ndn = (LAT + P - 1) / P;
    const size_t rbg = k3_gq_row_bytes(t, LAT), rbd = k3_gq_row_bytes(t, INTER);
    const size_t gsz = (size_t)ngu * rbg, dsz = (size_t)ndn * rbd;
    const size_t rec = (2 * gsz + dsz + 4095) & ~(size_t)4095;
    size_t ne = pool_bytes / rec;
    if (ne < 4 * TOPK) ne = 4 * TOPK;
    unsigned char *pool = huge_alloc(ne * rec);
    float *z = aligned_alloc(64, LAT * 4), *act = aligned_alloc(64, (size_t)TOPK * INTER * 4);
    float *gu = aligned_alloc(64, (size_t)TOPK * 2 * INTER * 4), *dn = aligned_alloc(64, (size_t)TOPK * LAT * 4);
    int8_t *zq = aligned_alloc(64, LAT), *aq = aligned_alloc(64, (size_t)TOPK * INTER);
    float *zdx = malloc(LAT / 256 * 4), *adx = malloc((size_t)TOPK * INTER / 256 * 4);
    for (int i = 0; i < LAT; i++) z[i] = (float)((i * 37) % 101 - 50) * 0.01f;
    for (int i = 0; i < TOPK * INTER; i++) act[i] = (float)((i * 53) % 97 - 48) * 0.01f;
    k3_gq_quant_x(zq, zdx, z, LAT);
    for (int j = 0; j < TOPK; j++) k3_gq_quant_x(aq + j * INTER, adx + j * INTER / 256, act + j * INTER, INTER);
    const long iters = (long)ne / TOPK * 4 > 64 ? (long)ne / TOPK * 4 : 64;
    double tgu = 0, tdn = 0;
#pragma omp parallel
    {
        const int me = omp_get_thread_num(), nth = omp_get_num_threads();
        for (size_t e = 0; e < ne; e++) {
            unsigned char *r = pool + e * rec;
            if (touch_t0) {
                if (me) continue;
                fill_rows(r, t, LAT, 0, ngu); fill_rows(r + gsz, t, LAT, 0, ngu);
                fill_rows(r + 2 * gsz, t, INTER, 0, ndn);
            } else {
                fill_rows(r, t, LAT, (size_t)ngu * me / nth, (size_t)ngu * (me + 1) / nth);
                fill_rows(r + gsz, t, LAT, (size_t)ngu * me / nth, (size_t)ngu * (me + 1) / nth);
                fill_rows(r + 2 * gsz, t, INTER, (size_t)ndn * me / nth, (size_t)ndn * (me + 1) / nth);
            }
        }
        k3_sync();
        int sj[2 * TOPK + 2], s0[2 * TOPK + 2], s1[2 * TOPK + 2];
        double a = 0, b = 0;
        uint64_t rng = 88172645463325252ull;
        for (long it = -(long)(ne / TOPK); it < iters; it++) {
            const unsigned char *sel[TOPK];
            for (int j = 0; j < TOPK; j++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                sel[j] = pool + (rng % ne) * rec;
            }
            k3_sync();
            const double t0 = omp_get_wtime();
            int ns = spans(TOPK, ngu, sj, s0, s1);
            for (int s = 0; s < ns; s++) {
                const unsigned char *W = sel[sj[s]];
                float *g = gu + (size_t)sj[s] * 2 * INTER;
                if (q8 && t == K3_GG_IQ2_XS) {
                    k3_iq2xs_rows_q8(g, zq, zdx, W, LAT, s0[s], s1[s]);
                    k3_iq2xs_rows_q8(g + INTER, zq, zdx, W + gsz, LAT, s0[s], s1[s]);
                } else if (q8) {
                    k3_iq3xxs_rows_q8(g, zq, zdx, W, LAT, s0[s], s1[s]);
                    k3_iq3xxs_rows_q8(g + INTER, zq, zdx, W + gsz, LAT, s0[s], s1[s]);
                } else if (t == K3_GG_IQ2_XS) {
                    k3_iq2xs_rows(g, z, W, LAT, s0[s], s1[s]);
                    k3_iq2xs_rows(g + INTER, z, W + gsz, LAT, s0[s], s1[s]);
                } else {
                    k3_iq3xxs_rows(g, z, W, LAT, s0[s], s1[s]);
                    k3_iq3xxs_rows(g + INTER, z, W + gsz, LAT, s0[s], s1[s]);
                }
            }
            k3_sync();
            const double t1 = omp_get_wtime();
            ns = spans(TOPK, ndn, sj, s0, s1);
            for (int s = 0; s < ns; s++) {
                const unsigned char *W = sel[sj[s]] + 2 * gsz;
                float *d = dn + (size_t)sj[s] * LAT;
                const int j = sj[s];
                if (q8 && t == K3_GG_IQ2_XS)
                    k3_iq2xs_rows_q8(d, aq + j * INTER, adx + j * INTER / 256, W, INTER, s0[s], s1[s]);
                else if (q8)
                    k3_iq3xxs_rows_q8(d, aq + j * INTER, adx + j * INTER / 256, W, INTER, s0[s], s1[s]);
                else if (t == K3_GG_IQ2_XS)
                    k3_iq2xs_rows(d, act + j * INTER, W, INTER, s0[s], s1[s]);
                else
                    k3_iq3xxs_rows(d, act + j * INTER, W, INTER, s0[s], s1[s]);
            }
            k3_sync();
            if (it >= 0) { a += t1 - t0; b += omp_get_wtime() - t1; }
        }
        if (me == 0) { tgu = a; tdn = b; }
    }
    const double gub = 2.0 * gsz * TOPK, dnb = (double)dsz * TOPK;
    const double gus = tgu / iters, dns = tdn / iters;
    const char *ty = q8 ? "int8" : "fp32";
    const int cg = xcases[xi].gu_count, cd = xcases[xi].dn_count;
    printf("%-16s %-8s %6d x %-6d %8.2f MB %4d %9.2f us %7.1f GB/s %5.1f%% %8.3f ms/tok  (16 x gate+up)\n",
           xcases[xi].name, ty, ngu, LAT, gub / 1e6, cg, gus * 1e6, gub / gus / 1e9,
           100 * gub / gus / 1e9 / roof_gbs, gus * 1e3 * cg);
    printf("%-16s %-8s %6d x %-6d %8.2f MB %4d %9.2f us %7.1f GB/s %5.1f%% %8.3f ms/tok  (16 x down) y0=%g\n",
           "", ty, ndn, INTER, dnb / 1e6, cd, dns * 1e6, dnb / dns / 1e9,
           100 * dnb / dns / 1e9 / roof_gbs, dns * 1e3 * cd, gu[0]);
    phase_add(q8 ? "experts" : "experts fp32", gus * 1e3 * cg + dns * 1e3 * cd,
              (gub * cg + dnb * cd) / 1e9);
    munmap(pool, ne * rec);
    free(z); free(act); free(gu); free(dn); free(zq); free(aq); free(zdx); free(adx);
}

int main(int argc, char **argv)
{
    int P = 6;
    const char *filter = NULL;
    Case custom = { "custom", "custom", K3_GG_Q8_0, 0, 0, 0, 1 };
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-P")) P = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-g")) pool_bytes = (size_t)(atof(argv[i + 1]) * (1 << 30));
        else if (!strcmp(argv[i], "-t")) touch_t0 = !strcmp(argv[i + 1], "t0");
        else if (!strcmp(argv[i], "-f")) filter = argv[i + 1];
        else if (!strcmp(argv[i], "-s")) {   /* rows,in[,f32]: one shape, rows per rank */
            char ty[8] = "";
            sscanf(argv[i + 1], "%d,%d,%7s", &custom.rows, &custom.in, ty);
            if (!strcmp(ty, "f32")) custom.type = K3_GG_F32;
        }
        else { fprintf(stderr, "usage: %s [-P ranks] [-g pool_GB] [-t rows|t0] [-f name] [-s rows,in[,f32]]\n", argv[0]); return 2; }
    }
    if (P < 1) P = 1;
    printf("per-rank decode GEMVs at TP=%d, %d threads, pool %.1f GB per case, first touch %s, int8 kernels %s\n",
           P, omp_get_max_threads(), pool_bytes / 1e9, touch_t0 ? "thread 0" : "by compute rows",
           k3_gq_have_q8() ? "yes" : "no");
    roof_gbs = read_roof();
    printf("read roof (same pool, all threads): %.1f GB/s\n\n", roof_gbs);
    printf("%-16s %-8s %15s %11s %4s %12s %12s %6s %15s\n", "gemv", "type", "rows x in",
           "bytes", "n/tk", "per call", "bandwidth", "roof", "per token");
    if (custom.rows > 0) { run_case(&custom, 1); return 0; }
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
        if (!filter || strstr(cases[i].name, filter)) run_case(&cases[i], P);
    for (int xi = 0; xi < 2; xi++)
        if (!filter || strstr(xcases[xi].name, filter))
            for (int q8 = 1; q8 >= 0; q8--)
                if (!q8 || k3_gq_have_q8()) run_experts(xi, P, q8);
    printf("\nper engine profile phase (int8 experts):\n");
    double tot = 0, gb = 0;
    for (int i = 0; i < NPH && ph_name[i]; i++) {
        printf("  %-14s %8.3f ms/token  %6.2f GB  %7.1f GB/s\n", ph_name[i], ph_ms[i], ph_gb[i],
               ph_gb[i] / ph_ms[i] * 1e3);
        if (strcmp(ph_name[i], "experts fp32")) { tot += ph_ms[i]; gb += ph_gb[i]; }
    }
    printf("  %-14s %8.3f ms/token  %6.2f GB  %7.1f GB/s  (roof time %.3f ms)\n", "TOTAL", tot, gb,
           gb / tot * 1e3, gb / roof_gbs * 1e3);
    return 0;
}

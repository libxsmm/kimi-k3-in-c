/* bench_expert_stream.c - the routed-expert access pattern of one decode step on one
 * tensor-parallel rank, without loading a model.
 *
 * Per call: top-k experts drawn at random from a pool far larger than the last-level
 * cache; this rank's gate|up rows split over the team as moe_routed splits them
 * (expert_spans), a barrier, then the down rows and a barrier. Reports us per call, the
 * wall GB/s, and the kernels' GB/s measured on the slowest thread. Run one copy per rank
 * (pinned) to reproduce the contention of a TP decode.
 *
 *   make bin/bench_expert_stream
 *   OMP_NUM_THREADS=<threads per rank> OMP_PROC_BIND=close OMP_PLACES=cores \
 *       bin/bench_expert_stream [tp=1] [pool_layers=8] [calls=2000]
 */
#include "k3_gq.h"

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { NE = 896, TOPK = 16, LAT = 3584, INTER = 3072 };

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void split(int n, int t, int p, int *lo, int *hi)
{
    *lo = (int)((long)n * t / p);
    *hi = (int)((long)n * (t + 1) / p);
}

static uint64_t xs(uint64_t *r)
{
    *r ^= *r << 13; *r ^= *r >> 7; *r ^= *r << 17;
    return *r;
}

int main(int argc, char **argv)
{
    const int tp = argc > 1 ? atoi(argv[1]) : 1;
    const int pool = argc > 2 ? atoi(argv[2]) : 8;
    const int calls = argc > 3 ? atoi(argv[3]) : 2000;
    if (tp < 1 || tp > INTER || pool < 1 || calls < 1) {
        fprintf(stderr, "usage: %s [tp] [pool_layers] [calls]\n", argv[0]);
        return 2;
    }
    if (!k3_gq_have_q8()) {
        fprintf(stderr, "needs the int8 expert kernels\n");
        return 2;
    }
    const int ni = INTER / tp, nl = LAT / tp;       /* rank 0's part of each split */
    const size_t rbg = k3_gq_row_bytes(K3_GG_IQ2_XS, LAT);
    const size_t rbd = k3_gq_row_bytes(K3_GG_IQ2_XS, INTER);
    const size_t ng = (size_t)ni * rbg, nd = (size_t)nl * rbd, eb = 2 * ng + nd;
    const size_t total = eb * NE * (size_t)pool;
    unsigned char *W = NULL;
    int8_t *xq = NULL;
    float *dx = NULL, *y = NULL;
    if (posix_memalign((void **)&W, (size_t)2 << 20, total) ||
        posix_memalign((void **)&xq, 64, INTER > LAT ? INTER : LAT) ||
        posix_memalign((void **)&dx, 64, 64 * sizeof(float)) ||
        posix_memalign((void **)&y, 64, (size_t)TOPK * 2 * INTER * sizeof(float))) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    int *sel = malloc(sizeof(int) * (size_t)calls * TOPK);
    if (!sel) { fprintf(stderr, "out of memory\n"); return 1; }
    const int nx = INTER > LAT ? INTER : LAT;
    for (int i = 0; i < nx; i++) xq[i] = (int8_t)(i * 37 % 255 - 127);
    for (int i = 0; i < 64; i++) dx[i] = 0.01f;

    /* random IQ2_XS blocks with d = 1.0, first-touched by the threads that stream them */
    const size_t nblk = total / 74;
#pragma omp parallel
    {
        const size_t id = (size_t)omp_get_thread_num(), n = (size_t)omp_get_num_threads();
        uint64_t r = 0x9E3779B97F4A7C15ull * (id + 1);
        for (size_t b = nblk * id / n; b < nblk * (id + 1) / n; b++) {
            unsigned char *p = W + b * 74;
            for (int k = 0; k < 74; k += 8) {
                const uint64_t v = xs(&r);
                memcpy(p + k, &v, k + 8 <= 74 ? 8 : 74 - k);
            }
            p[0] = 0x00; p[1] = 0x3C;
        }
    }
    uint64_t r = 88172645463325252ull;
    for (int i = 0; i < calls * TOPK; i++) sel[i] = (int)(xs(&r) % ((uint64_t)NE * pool));

    const int nt = omp_get_max_threads();
    double *own_gu = calloc((size_t)nt, sizeof(double));
    double *own_dn = calloc((size_t)nt, sizeof(double));
    if (!own_gu || !own_dn) { fprintf(stderr, "out of memory\n"); return 1; }
    double wall = 0;
    for (int pass = 0; pass < 2; pass++) {          /* pass 0 warms up */
        memset(own_gu, 0, (size_t)nt * sizeof(double));
        memset(own_dn, 0, (size_t)nt * sizeof(double));
        const double t0 = now();
#pragma omp parallel
        {
            const int t = omp_get_thread_num();
            for (int c = 0; c < calls; c++) {
                const int *s = sel + (size_t)c * TOPK;
                int lo, hi;
                double a = now();
                split(TOPK * ni, t, nt, &lo, &hi);
                for (int q = lo; q < hi; ) {
                    const int j = q / ni, e = (j + 1) * ni < hi ? (j + 1) * ni : hi;
                    const int r0 = q - j * ni, r1 = e - j * ni;
                    const unsigned char *ex = W + (size_t)s[j] * eb;
                    float *yj = y + (size_t)j * 2 * INTER;
                    k3_iq2xs_rows_q8(yj, xq, dx, ex, LAT, r0, r1);
                    k3_iq2xs_rows_q8(yj + INTER, xq, dx, ex + ng, LAT, r0, r1);
                    q = e;
                }
                own_gu[t] += now() - a;
#pragma omp barrier
                a = now();
                split(TOPK * nl, t, nt, &lo, &hi);
                for (int q = lo; q < hi; ) {
                    const int j = q / nl, e = (j + 1) * nl < hi ? (j + 1) * nl : hi;
                    const unsigned char *ex = W + (size_t)s[j] * eb + 2 * ng;
                    k3_iq2xs_rows_q8(y + (size_t)j * LAT, xq, dx, ex, INTER,
                                     q - j * nl, e - j * nl);
                    q = e;
                }
                own_dn[t] += now() - a;
#pragma omp barrier
            }
        }
        wall = now() - t0;
    }
    double mg = 0, md = 0;
    for (int t = 0; t < nt; t++) {
        if (own_gu[t] > mg) mg = own_gu[t];
        if (own_dn[t] > md) md = own_dn[t];
    }
    const double gu_b = (double)TOPK * 2 * ng * calls, dn_b = (double)TOPK * nd * calls;
    printf("tp=%d threads=%d pool=%.1f GB: %.1f us/call, wall %.1f GB/s | "
           "gate|up %.1f GB/s, down %.1f GB/s (slowest thread)\n",
           tp, nt, total / 1e9, wall / calls * 1e6, (gu_b + dn_b) / wall / 1e9,
           gu_b / mg / 1e9, dn_b / md / 1e9);
    free(W); free(xq); free(dx); free(y); free(sel); free(own_gu); free(own_dn);
    return 0;
}

/* bench_amx.c - prefill GEMM (k3_amx_gemm, AMX-BF16 with weight panels decoded once per
 * call) against the per-token GEMV path the prefill uses today, at per-rank shapes.
 *
 *   OMP_NUM_THREADS=42 OMP_PLACES=cores OMP_PROC_BIND=close numactl -N 0 -m 0 \
 *       bin/bench_amx [-P ranks] [-T 16,64,256,1024] [-s rows,in[,iq2]]
 *
 * Reports ms per call, TFLOP/s, the speedup over T GEMVs, and the error against the
 * exact GEMV (max |diff| / max |ref| over two tokens). */
#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"
#include "k3_amx.h"
#include "k3_gq.h"

typedef struct { const char *name; int type, rows, in; } Shape;
static const Shape shapes[] = {
    { "kda q/k/v/g",  K3_GG_Q8_0,   12288,  7168 },
    { "kda/mla o",    K3_GG_Q8_0,   7168,   12288 },
    { "mla q_b",      K3_GG_Q8_0,   18432,  1536 },
    { "shared gate",  K3_GG_Q8_0,   6144,   7168 },
    { "shared down",  K3_GG_Q8_0,   7168,   6144 },
    { "latent down",  K3_GG_Q8_0,   3584,   7168 },
    { "dense down",   K3_GG_Q8_0,   7168,   33792 },
    { "router f32",   K3_GG_F32,    896,    7168 },
    { "expert gate",  K3_GG_IQ2_XS, 3072,   3584 },
    { "expert down",  K3_GG_IQ2_XS, 3584,   3072 },
};

static void fill(unsigned char *W, int t, int in, size_t rows)
{
    const size_t rb = k3_gq_row_bytes(t, in);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    if (t == K3_GG_F32) {
        float *f = (float *)W;
        for (size_t i = 0; i < rows * in; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            f[i] = (float)((int)(s >> 40) - (1 << 23)) * (1.0f / (1 << 28));
        }
        return;
    }
    for (size_t i = 0; i + 8 <= rows * rb; i += 8) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; memcpy(W + i, &s, 8); }
    const size_t bb = t == K3_GG_Q8_0 ? 34 : t == K3_GG_IQ2_XS ? 74 : 98;
    const uint16_t d = 0x2000;
    for (size_t r = 0; r < rows; r++)
        for (size_t b = 0; b < rb / bb; b++) memcpy(W + r * rb + b * bb, &d, 2);
}

static void gemv_rows(float *y, const float *x, const unsigned char *W, int t, int in, int lo, int hi)
{
    if (t == K3_GG_Q8_0)        k3_q80_rows(y, x, W, in, lo, hi);
    else if (t == K3_GG_IQ2_XS) k3_iq2xs_rows(y, x, W, in, lo, hi);
    else for (int o = lo; o < hi; o++) {
        const float *w = (const float *)W + (size_t)o * in;
        double a = 0;
        for (int k = 0; k < in; k++) a += (double)w[k] * x[k];
        y[o] = (float)a;
    }
}

static void run(const Shape *sh, int P, const int *Ts, int nT)
{
    const int rows = (sh->rows + P - 1) / P, in = sh->in;
    const size_t rb = k3_gq_row_bytes(sh->type, in);
    unsigned char *W = aligned_alloc(4096, (rows * rb + 4095) & ~(size_t)4095);
    fill(W, sh->type, in, rows);
    const int Tmax = Ts[nT - 1];
    float *X = aligned_alloc(64, (size_t)Tmax * in * 4);
    for (size_t i = 0; i < (size_t)Tmax * in; i++) X[i] = (float)((int)((i * 2654435761u) >> 20 & 1023) - 512) / 512.0f;
    float *Y = aligned_alloc(64, (size_t)Tmax * rows * 4 + 64), *R = aligned_alloc(64, (size_t)rows * 4 + 64);
    uint16_t *Xv = aligned_alloc(64, k3_amx_xv_elems(Tmax, in) * 2);

    /* one GEMV over all rows, the team splitting rows: today's cost per prompt token */
    double tg = 0;
    const int greps = 20;
#pragma omp parallel
    {
        int lo, hi;
        k3_split(rows, &lo, &hi);
        for (int r = -2; r < greps; r++) {
            k3_sync();
            const double t0 = omp_get_wtime();
            gemv_rows(R, X + (size_t)(r & 7) * in, W, sh->type, in, lo, hi);
            k3_sync();
            if (r >= 0 && omp_get_thread_num() == 0) tg += omp_get_wtime() - t0;
        }
    }
    tg /= greps;

    for (int i = 0; i < nT; i++) {
        const int T = Ts[i];
        double ta = 0;
        const int reps = T >= 512 ? 3 : 6;
#pragma omp parallel
        for (int r = -1; r < reps; r++) {
            k3_sync();
            const double t0 = omp_get_wtime();
            k3_amx_pack_x(Xv, X, in, T, in);
            k3_sync();
            k3_amx_gemm(Y, rows, Xv, T, W, sh->type, in, 0, 0, rows);
            k3_sync();
            if (r >= 0 && omp_get_thread_num() == 0) ta += omp_get_wtime() - t0;
        }
        ta /= reps;
        double emax = 0, rmax = 0;
        for (int t = 0; t < T; t += T - 1 > 0 ? T - 1 : 1) {
            gemv_rows(R, X + (size_t)t * in, W, sh->type, in, 0, rows);
            for (int o = 0; o < rows; o++) {
                emax = fmax(emax, fabs(Y[(size_t)t * rows + o] - R[o]));
                rmax = fmax(rmax, fabs(R[o]));
            }
            if (T == 1) break;
        }
        const double fl = 2.0 * T * rows * (double)in;
        printf("%-12s %-7s %6d x %-6d T=%-5d %9.3f ms %7.2f TFLOP/s  (T GEMVs %9.3f ms: %6.1fx)  err %.2e\n",
               sh->name, sh->type == K3_GG_Q8_0 ? "q8_0" : sh->type == K3_GG_F32 ? "f32" : "iq2_xs",
               rows, in, T, ta * 1e3, fl / ta / 1e12, tg * T * 1e3, tg * T / ta, emax / rmax);
    }
    free(W); free(X); free(Y); free(R); free(Xv);
}

int main(int argc, char **argv)
{
    int P = 6, Ts[16] = { 16, 64, 256, 1024 }, nT = 4;
    Shape custom = { "custom", K3_GG_Q8_0, 0, 0 };
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-P")) P = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-T")) {
            nT = 0;
            for (char *p = argv[i + 1]; *p && nT < 16; ) { Ts[nT++] = (int)strtol(p, &p, 10); if (*p == ',') p++; }
        } else if (!strcmp(argv[i], "-s")) {
            char ty[8] = "";
            sscanf(argv[i + 1], "%d,%d,%7s", &custom.rows, &custom.in, ty);
            if (!strcmp(ty, "iq2")) custom.type = K3_GG_IQ2_XS;
            if (!strcmp(ty, "f32")) custom.type = K3_GG_F32;
        } else { fprintf(stderr, "usage: %s [-P ranks] [-T list] [-s rows,in[,iq2|f32]]\n", argv[0]); return 2; }
    }
    if (!k3_amx_ok()) { printf("AMX-BF16 not available\n"); return 1; }
    printf("prefill GEMM, AMX-BF16 panels vs per-token GEMV: TP=%d shapes, %d threads\n", P, omp_get_max_threads());
    if (custom.rows > 0) { run(&custom, 1, Ts, nT); return 0; }
    for (size_t i = 0; i < sizeof shapes / sizeof *shapes; i++) run(&shapes[i], P, Ts, nT);
    return 0;
}

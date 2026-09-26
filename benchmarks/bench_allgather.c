/* bench_allgather.c - latency of the collectives tensor parallelism issues per token.
 *
 * The engine's decode step issues ~650 small in-place allgathers (k3_tp_gather). This
 * times the same shapes in isolation, so the transport's own latency can be separated
 * from rank skew inside the engine. Sizes are TOTAL floats of the gathered vector.
 *
 * usage: mpirun ... bench_allgather [iters]
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"
#include "k3_mpi.h"

/* the engine's own team gather, one thread: one-sided under K3_TP_ONESIDED=1 */
static double time_engine(float *buf, int n, int iters, int P)
{
    (void)P;
    const K3Seg sg = { buf, n, 1, 0 };
    for (int i = 0; i < 20; i++) k3_tp_gather(&sg, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++) k3_tp_gather(&sg, 1);
    return (MPI_Wtime() - t0) / iters;
}

static double tbegin, tend;

/* begin, `work` seconds of computation without MPI calls, end; returns the time beyond
 * the work, i.e. what overlap failed to hide */
static double time_overlap(float *buf, int n, int iters, int P, double work)
{
    (void)P;
    const K3Seg sg = { buf, n, 1, 0 };
    tbegin = tend = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++) {
        const double b0 = MPI_Wtime();
        k3_tp_gather_begin(&sg, 1);
        const double w0 = MPI_Wtime();
        tbegin += w0 - b0;
        while (MPI_Wtime() - w0 < work) { }
        const double e0 = MPI_Wtime();
        k3_tp_gather_end(&sg, 1);
        tend += MPI_Wtime() - e0;
    }
    tbegin /= iters; tend /= iters;
    return (MPI_Wtime() - t0) / iters - work;
}

static double time_allgatherv(float *buf, int n, int iters, int P, int rank)
{
    int *cnt = malloc(sizeof(int) * P), *dsp = malloc(sizeof(int) * P);
    for (int r = 0; r < P; r++) {
        const int lo = (int)((long)n * r / P), hi = (int)((long)n * (r + 1) / P);
        dsp[r] = lo; cnt[r] = hi - lo;
    }
    (void)rank;
    for (int i = 0; i < 20; i++)
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, cnt, dsp, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++)
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, cnt, dsp, MPI_FLOAT, MPI_COMM_WORLD);
    const double dt = (MPI_Wtime() - t0) / iters;
    free(cnt); free(dsp);
    return dt;
}

static double time_allgather(float *buf, int n, int iters, int P)
{
    const int per = (n + P - 1) / P;
    for (int i = 0; i < 20; i++)
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, per, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++)
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, per, MPI_FLOAT, MPI_COMM_WORLD);
    return (MPI_Wtime() - t0) / iters;
}

int main(int argc, char **argv)
{
    if (k3_mpi_init(&argc, &argv) != 0) return 2;
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    const int iters = argc > 1 ? atoi(argv[1]) : 2000;
    k3_act_bf16 = getenv("K3_BF16_ACT") && k3_act_bf16_supported() ? K3_BF16_TP : 0;
    /* hidden, KDA/MLA head output, router+latent, experts' act + shared act */
    const int sizes[] = { 1, 896, 7168, 12288, 896 + 3584, 16 * 3072 + 6144, 163840 };
    float *buf = calloc(163840 + 64, sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++) MPI_Barrier(MPI_COMM_WORLD);
    const double tb = (MPI_Wtime() - t0) / iters;
    if (rank == 0) printf("ranks %d  barrier %.2f us\n", P, tb * 1e6);
    if (rank == 0) printf("%10s %14s %14s %14s %14s %8s %8s\n", "floats", "allgatherv us", "allgather us",
                          "engine us", "+50us work us", "begin", "end");
    for (size_t s = 0; s < sizeof sizes / sizeof *sizes; s++) {
        const int n = sizes[s] < P ? P : sizes[s];
        const double a = time_allgatherv(buf, n, iters, P, rank);
        const double b = time_allgather(buf, n, iters, P);
        const double c = time_engine(buf, n, iters, P);
        const double d = time_overlap(buf, n, iters, P, 50e-6);
        if (rank == 0) printf("%10d %14.2f %14.2f %14.2f %14.2f %8.2f %8.2f\n", n, a * 1e6, b * 1e6,
                              c * 1e6, d * 1e6, tbegin * 1e6, tend * 1e6);
    }
    free(buf);
    k3_mpi_finalize();
    return 0;
}

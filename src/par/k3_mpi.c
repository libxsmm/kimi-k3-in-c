/* k3_mpi.c - see k3_mpi.h. Built only when K3_MPI is defined (make MPI=1). */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_mpi.h"

static void mpi_allgatherv(float *buf, const int *counts, const int *displs, void *ctx)
{
    (void)ctx;
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   buf, counts, displs, MPI_FLOAT, MPI_COMM_WORLD);
}

/* ONE-SIDED TRANSPORT for the team gather in k3_ops.c (default; K3_TP_ONESIDED=0 off).
 *
 * Measured at 16 ranks on 8 nodes: 41 -> 21 us per engine gather, 24.2 -> 22.9 ms/step
 * at 24 layers, 91.7 -> 85.9 ms/token for the full model. The win comes from the team:
 * every thread receives its share straight into the vectors while thread 0 sends. */
/*
 * Every rank MPI_Puts its block straight into every peer's window, and the receiving
 * team polls its own memory. Each 8-byte word carries 4 bytes of payload and the call's
 * 32-bit sequence number, so a word is complete exactly when its sequence number
 * matches: no flag message, no matching, no rendezvous, no round trip.
 *
 * Two window halves alternate by call parity. A rank can only reach call s+2, and so
 * reuse half s&1 at a peer, after its whole team has received that peer's block of call
 * s+1, which the peer sends only after its team finished reading call s. So a half is
 * never overwritten while its reader still needs it. That needs every rank to send
 * something in every call: a rank with an empty block puts one tagged word into its own
 * flag slot past the data area instead. The payload bits are moved verbatim, so results
 * are bit-identical to MPI_Allgatherv.
 *
 * A send buffer may only be rewritten once its puts completed locally, but with osc/ucx
 * any flush waits a round trip to every peer, even with nothing outstanding (7 -> 18 us
 * for a small gather at 16 ranks). So sends rotate through a ring of LL_NSEND buffers
 * and one flush per lap covers them all. Every float travels as 8 bytes (a bf16 pair
 * with k3_act_bf16), so gathers above LL_MAXW words use MPI_Allgatherv; 2^16 keeps the
 * MoE act gather (16 x 3072 + 6144 floats) one-sided. UCX endpoints are wired up at
 * init: a peer spinning on its window never progresses the lazy handshake (deadlock at
 * 8+ ranks). */
#define LL_MAXW  (1 << 16)                /* data words per half */
#define LL_NSEND 32                       /* send buffers in the ring */
static MPI_Win   ll_win = MPI_WIN_NULL;

static void ll_put(const uint64_t *src, int nw, long disp)
{
    const int P = k3_tp.size, me = k3_tp.rank;
    for (int k = 1; k < P; k++)
        MPI_Put(src, nw, MPI_UINT64_T, (me + k) % P, (MPI_Aint)disp, nw, MPI_UINT64_T, ll_win);
}

static void ll_flush(void) { MPI_Win_flush_local_all(ll_win); }

static void ll_init(void)
{
    const int P = k3_tp.size;
    const size_t hw = (size_t)LL_MAXW + (size_t)P;
    uint64_t *recv = NULL;
    const MPI_Aint bytes = (MPI_Aint)(2 * hw * sizeof(uint64_t));
    if (MPI_Win_allocate(bytes, sizeof(uint64_t), MPI_INFO_NULL, MPI_COMM_WORLD,
                         &recv, &ll_win) != MPI_SUCCESS) {
        ll_win = MPI_WIN_NULL;
        return;
    }
    memset(recv, 0, (size_t)bytes);
    uint64_t *send = (uint64_t *)calloc((size_t)LL_NSEND * hw, sizeof(uint64_t));
    if (!send) { MPI_Win_free(&ll_win); ll_win = MPI_WIN_NULL; return; }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, ll_win);
    /* word 0 of half 0 never matches a sequence number, so writing zero there is harmless */
    ll_put(send, 1, 0);
    MPI_Win_flush_all(ll_win);
    MPI_Barrier(MPI_COMM_WORLD);
    k3_tp.ll_recv = recv;
    k3_tp.ll_send = send;
    k3_tp.ll_hw = (int)hw;
    k3_tp.ll_maxw = LL_MAXW;
    k3_tp.ll_nsend = LL_NSEND;
    k3_tp.ll_flush = ll_flush;
    k3_tp.ll_put = ll_put;
}

static void mpi_barrier(void *ctx)
{
    (void)ctx;
    MPI_Barrier(MPI_COMM_WORLD);
}

int k3_mpi_init(int *argc, char ***argv)
{
    int provided = 0;
    if (MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        fprintf(stderr, "k3_mpi: MPI_Init_thread failed\n");
        return -1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "k3_mpi: MPI provides thread level %d, FUNNELED needed\n", provided);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &k3_tp.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &k3_tp.size);
    k3_tp.allgatherv = mpi_allgatherv;
    k3_tp.barrier = mpi_barrier;
    k3_tp.skew = getenv("K3_TP_SKEW") != NULL;
    k3_tp.ctx = NULL;
    /* K3_TP_ONESIDED=0 keeps plain MPI_Allgatherv (see above); one-sided is the default. */
    const char *os = getenv("K3_TP_ONESIDED");
    if (k3_tp.size > 1 && !(os && !strcmp(os, "0"))) ll_init();
    return 0;
}

void k3_mpi_finalize(void)
{
    if (ll_win != MPI_WIN_NULL) {
        MPI_Win_unlock_all(ll_win);
        MPI_Win_free(&ll_win);
        free(k3_tp.ll_send);
        k3_tp.ll_put = NULL;
    }
    MPI_Finalize();
}

int k3_mpi_max_int(int v)
{
    int out = v;
    MPI_Allreduce(&v, &out, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return out;
}

void k3_mpi_abort(int rc)
{
    MPI_Abort(MPI_COMM_WORLD, rc);
}

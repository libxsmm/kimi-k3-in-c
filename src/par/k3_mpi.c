/* k3_mpi.c - see k3_mpi.h. Built only when K3_MPI is defined (make MPI=1). */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_mpi.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef K3_UCX
#include <ucp/api/ucp.h>
#endif

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

static void ll_put(int lane, const uint64_t *src, int nw, long disp)
{
    const int P = k3_tp.size, me = k3_tp.rank;
    (void)lane;
    for (int k = 1; k < P; k++)
        MPI_Put(src, nw, MPI_UINT64_T, (me + k) % P, (MPI_Aint)disp, nw, MPI_UINT64_T, ll_win);
}

static void ll_flush(int lane) { (void)lane; MPI_Win_flush_local_all(ll_win); }

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
    ll_put(0, send, 1, 0);
    MPI_Win_flush_all(ll_win);
    MPI_Barrier(MPI_COMM_WORLD);
    k3_tp.ll_recv = recv;
    k3_tp.ll_send = send;
    k3_tp.ll_hw = (int)hw;
    k3_tp.ll_maxw = LL_MAXW;
    k3_tp.ll_nsend = LL_NSEND;
    k3_tp.ll_nlanes = 1;
    k3_tp.ll_flush = ll_flush;
    k3_tp.ll_put = ll_put;
}

static void mpi_barrier(void *ctx)
{
    (void)ctx;
    MPI_Barrier(MPI_COMM_WORLD);
}

/* SHARED-MEMORY TRANSPORT (default when every rank is on one node; K3_TP_SHM=0 off): the
 * same window layout and tagged words, but a put is plain 8-byte stores straight into the
 * peer's window (an MPI-3 shared window, each rank's part first-touched by its owner), so
 * it is complete on return and needs no flush. One lane per peer: P-1 team threads write
 * in parallel. Payload bits are moved verbatim, so results are bit-identical. */
static MPI_Win    sh_win = MPI_WIN_NULL;
static MPI_Comm   sh_comm = MPI_COMM_NULL;
static uint64_t **sh_peer;

/* Whole aligned lines go as non-temporal stores, so the writer never has to take
 * ownership of a line its reader is spinning on; every 8-byte word is still written
 * whole, which is all the tagged protocol needs. */
static void sh_copy(uint64_t *d, const uint64_t *s, int n)
{
    volatile uint64_t *v = d;
    int i = 0;
#if defined(__AVX512F__)
    for (; i < n && ((uintptr_t)(d + i) & 63); i++) v[i] = s[i];
    for (; i + 8 <= n; i += 8) _mm512_stream_si512((void *)(d + i), _mm512_loadu_si512(s + i));
#endif
    for (; i < n; i++) v[i] = s[i];
#if defined(__AVX512F__)
    _mm_sfence();
#endif
}

static void sh_put(int lane, const uint64_t *src, int nw, long disp)
{
    const int P = k3_tp.size, me = k3_tp.rank, L = k3_tp.ll_nlanes;
    const int k0 = 1 + (int)((long)(P - 1) * lane / L), k1 = 1 + (int)((long)(P - 1) * (lane + 1) / L);
    for (int k = k0; k < k1; k++) sh_copy(sh_peer[(me + k) % P] + disp, src, nw);
}

static void sh_flush(int lane) { (void)lane; }

static int sh_init(void)
{
    const int P = k3_tp.size;
    MPI_Comm node;
    int ns, one, all;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_size(node, &ns);
    one = ns == P;
    MPI_Allreduce(&one, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all) { MPI_Comm_free(&node); return -1; }
    /* key 0: node ranks keep the world order, so node rank r is world rank r */
    const size_t hw = (size_t)LL_MAXW + (size_t)P;
    const MPI_Aint bytes = (MPI_Aint)(2 * hw * sizeof(uint64_t));
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "alloc_shared_noncontig", "true");
    uint64_t *recv = NULL;
    const int rc = MPI_Win_allocate_shared(bytes, sizeof(uint64_t), info, node, &recv, &sh_win);
    MPI_Info_free(&info);
    const char *le = getenv("K3_TP_SHM_LANES");
    int L = le ? atoi(le) : P - 1;
    if (L < 1) L = 1;
    if (L > P - 1) L = P - 1;
    const int nsend = 2;
    uint64_t *send = rc == MPI_SUCCESS ? (uint64_t *)calloc((size_t)L * nsend * hw, sizeof(uint64_t)) : NULL;
    sh_peer = (uint64_t **)calloc((size_t)P, sizeof *sh_peer);
    one = rc == MPI_SUCCESS && send && sh_peer;
    MPI_Allreduce(&one, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all) {
        if (rc == MPI_SUCCESS) MPI_Win_free(&sh_win);
        sh_win = MPI_WIN_NULL;
        free(send); free(sh_peer); sh_peer = NULL;
        MPI_Comm_free(&node);
        return -1;
    }
    memset(recv, 0, (size_t)bytes);
    for (int r = 0; r < P; r++) {
        MPI_Aint sz;
        int du;
        void *base;
        MPI_Win_shared_query(sh_win, r, &sz, &du, &base);
        sh_peer[r] = (uint64_t *)base;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    sh_comm = node;
    k3_tp.ll_recv = recv;
    k3_tp.ll_send = send;
    k3_tp.ll_hw = (int)hw;
    k3_tp.ll_maxw = LL_MAXW;
    k3_tp.ll_nsend = nsend;
    k3_tp.ll_nlanes = L;
    k3_tp.ll_flush = sh_flush;
    k3_tp.ll_put = sh_put;
    if (k3_tp.rank == 0)
        fprintf(stderr, "k3_mpi: TP gather over node shared memory, %d ranks, %d writer lanes\n", P, L);
    return 0;
}

#ifdef K3_UCX
/* The same one-sided protocol straight on UCP (default in UCX builds; K3_TP_UCX=0 uses
 * osc/ucx): same window layout, same
 * tagged words, so results are bit-identical. What it removes is osc/ucx's per-MPI_Put
 * software cost and its flush, which waits a round trip to every peer; ucp_put_nbx
 * completes locally, so a flush here only reaps this rank's own requests. The puts of a
 * gather are split over K3_TP_UCX_LANES lanes (default (P-1)/4 rounded up, at most 8),
 * each its own UCP worker with endpoints to a contiguous share of the peers, driven by
 * one team thread at a time. */
#define UX_MAXREQ 4096
#define UX_SHORT  1024                    /* at most this many bytes a put goes inline */
#define UX_MAXL   16
typedef struct {
    ucp_worker_h w;
    ucp_ep_h    *ep;                      /* [P], NULL outside this lane's peers */
    ucp_rkey_h  *rkey;
    int          k0, k1;                  /* peers (me + k) % P for k in [k0, k1) */
    int          nreq;
    long         npend;                   /* puts that did not complete at once */
    void        *req[UX_MAXREQ];
} UxLane;
static ucp_context_h ux_ctx;
static UxLane       *ux_l;
static int           ux_nl;
static uint64_t     *ux_raddr;
static ucp_mem_h     ux_rmem, ux_smem;

static void ux_die(const char *what, ucs_status_t st)
{
    fprintf(stderr, "k3_mpi: UCX %s failed: %s\n", what, ucs_status_string(st));
    MPI_Abort(MPI_COMM_WORLD, 3);
}

static void ux_wait(ucp_worker_h w, void *req, const char *what)
{
    ucs_status_t st;
    while ((st = ucp_request_check_status(req)) == UCS_INPROGRESS) ucp_worker_progress(w);
    if (st != UCS_OK) ux_die(what, st);
    ucp_request_free(req);
}

static void ux_flush(int lane)
{
    UxLane *l = &ux_l[lane];
    for (int i = 0; i < l->nreq; i++) ux_wait(l->w, l->req[i], "put");
    l->nreq = 0;
    ucp_worker_progress(l->w);
}

static void ux_poll(int lane) { ucp_worker_progress(ux_l[lane].w); }

static void ux_put(int lane, const uint64_t *src, int nw, long disp)
{
    const int P = k3_tp.size, me = k3_tp.rank;
    UxLane *l = &ux_l[lane];
    ucp_request_param_t rp;
    memset(&rp, 0, sizeof rp);
    rp.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
    rp.memh = ux_smem;
    for (int k = l->k0; k < l->k1; k++) {
        const int r = (me + k) % P;
        ucs_status_ptr_t s = ucp_put_nbx(l->ep[r], src, (size_t)nw * sizeof(uint64_t),
                                         ux_raddr[r] + (uint64_t)disp * sizeof(uint64_t),
                                         l->rkey[r], &rp);
        if (UCS_PTR_IS_ERR(s)) ux_die("put", UCS_PTR_STATUS(s));
        /* measured: without this, a put that returned UCS_OK may sit unsent until the
         * next progress call (32 ms stalls with polling off) */
        ucp_worker_progress(l->w);
        if (!s) continue;
        l->npend++;
        /* A short put is complete once posted. Left queued (no TX room), it would go out
         * only at our next progress call, which may come after a long compute phase or
         * never, if we next block in MPI: peers would spin on it. */
        if ((size_t)nw * sizeof(uint64_t) <= UX_SHORT) { ux_wait(l->w, s, "put"); continue; }
        if (l->nreq == UX_MAXREQ) ux_flush(lane);
        l->req[l->nreq++] = s;
    }
    ucp_worker_progress(l->w);
}

/* Every rank's variable-length blob, concatenated in rank order; *off gets the offsets. */
static char *ux_allgather_blob(const void *mine, int len, int **off)
{
    const int P = k3_tp.size;
    int *cnt = (int *)malloc((size_t)2 * P * sizeof(int));
    if (!cnt) MPI_Abort(MPI_COMM_WORLD, 3);
    MPI_Allgather(&len, 1, MPI_INT, cnt, 1, MPI_INT, MPI_COMM_WORLD);
    int *dsp = cnt + P, tot = 0;
    for (int r = 0; r < P; r++) { dsp[r] = tot; tot += cnt[r]; }
    char *all = (char *)malloc((size_t)tot);
    if (!all) MPI_Abort(MPI_COMM_WORLD, 3);
    MPI_Allgatherv(mine, len, MPI_BYTE, all, cnt, dsp, MPI_BYTE, MPI_COMM_WORLD);
    *off = cnt;
    return all;
}

/* Barrier that keeps progressing UCX: a peer finishing endpoint wireup needs our replies. */
static void ux_barrier(void)
{
    MPI_Request rq;
    int done = 0;
    MPI_Ibarrier(MPI_COMM_WORLD, &rq);
    while (!done) {
        for (int l = 0; l < ux_nl; l++) ucp_worker_progress(ux_l[l].w);
        MPI_Test(&rq, &done, MPI_STATUS_IGNORE);
    }
}

static int ux_init(void)
{
    const int P = k3_tp.size, me = k3_tp.rank;
    const char *le = getenv("K3_TP_UCX_LANES");
    /* measured full model: TP=32 1/4/8 lanes 44.7/42.9/42.5 ms, TP=16 1/4 lanes 55.1/55.3 */
    int L = le ? atoi(le) : (P + 2) / 4;
    if (!le && L > 8) L = 8;
    if (L > P - 1) L = P - 1;
    if (L > UX_MAXL) L = UX_MAXL;
    if (L < 1) L = 1;
    const size_t hw = (size_t)LL_MAXW + (size_t)P;
    const size_t rbytes = 2 * hw * sizeof(uint64_t);
    const size_t sbytes = (size_t)L * LL_NSEND * hw * sizeof(uint64_t);
    ucs_status_t st;

    ucp_config_t *cfg;
    /* The default moderation (a signalled completion every 64 sends) let a QP fill with
     * a few hundred-byte inline puts before any completion came back, so puts queued
     * (2-18% of them); 8 left none. Our context only: Open MPI's was created already. */
    setenv("UCX_RC_MLX5_TX_CQ_MODERATION", "8", 0);
    if ((st = ucp_config_read(NULL, NULL, &cfg)) != UCS_OK) ux_die("config", st);
    ucp_params_t pp;
    memset(&pp, 0, sizeof pp);
    pp.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    pp.features = UCP_FEATURE_RMA;
    pp.estimated_num_eps = (size_t)P;
    st = ucp_init(&pp, cfg, &ux_ctx);
    ucp_config_release(cfg);
    if (st != UCS_OK) ux_die("init", st);

    ux_nl = L;
    ux_l = (UxLane *)calloc((size_t)L, sizeof *ux_l);
    ux_raddr = (uint64_t *)calloc((size_t)P, sizeof *ux_raddr);
    if (!ux_l || !ux_raddr) MPI_Abort(MPI_COMM_WORLD, 3);
    ucp_worker_params_t wp;
    memset(&wp, 0, sizeof wp);
    wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    /* One lane is only ever driven by the main thread. Several lanes are driven by team
     * threads, or all by the main thread outside a team, so they need SERIALIZED, which
     * costs ~3 us per gather at TP=32 (SINGLE asserts on the owner thread). */
    wp.thread_mode = L == 1 ? UCS_THREAD_MODE_SINGLE : UCS_THREAD_MODE_SERIALIZED;
    for (int l = 0; l < L; l++) {
        if ((st = ucp_worker_create(ux_ctx, &wp, &ux_l[l].w)) != UCS_OK) ux_die("worker", st);
        ux_l[l].k0 = 1 + (int)((long)(P - 1) * l / L);
        ux_l[l].k1 = 1 + (int)((long)(P - 1) * (l + 1) / L);
        ux_l[l].ep = (ucp_ep_h *)calloc((size_t)P, sizeof(ucp_ep_h));
        ux_l[l].rkey = (ucp_rkey_h *)calloc((size_t)P, sizeof(ucp_rkey_h));
        if (!ux_l[l].ep || !ux_l[l].rkey) MPI_Abort(MPI_COMM_WORLD, 3);
    }

    uint64_t *recv = NULL, *send = NULL;
    if (posix_memalign((void **)&recv, 4096, rbytes) || posix_memalign((void **)&send, 4096, sbytes))
        MPI_Abort(MPI_COMM_WORLD, 3);
    memset(recv, 0, rbytes);
    memset(send, 0, sbytes);
    ucp_mem_map_params_t mp;
    memset(&mp, 0, sizeof mp);
    mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mp.address = recv; mp.length = rbytes;
    if ((st = ucp_mem_map(ux_ctx, &mp, &ux_rmem)) != UCS_OK) ux_die("mem_map", st);
    mp.address = send; mp.length = sbytes;
    if ((st = ucp_mem_map(ux_ctx, &mp, &ux_smem)) != UCS_OK) ux_die("mem_map", st);

    void *rkb;
    size_t rkl;
    if ((st = ucp_rkey_pack(ux_ctx, ux_rmem, &rkb, &rkl)) != UCS_OK) ux_die("rkey_pack", st);
    int *koff;
    char *rkeys = ux_allgather_blob(rkb, (int)rkl, &koff);
    ucp_rkey_buffer_release(rkb);
    const uint64_t mine = (uint64_t)(uintptr_t)recv;
    MPI_Allgather(&mine, 1, MPI_UINT64_T, ux_raddr, 1, MPI_UINT64_T, MPI_COMM_WORLD);

    /* lane l of this rank connects to lane l of each of its peers */
    for (int l = 0; l < L; l++) {
        ucp_address_t *addr;
        size_t alen;
        if ((st = ucp_worker_get_address(ux_l[l].w, &addr, &alen)) != UCS_OK) ux_die("address", st);
        int *aoff;
        char *addrs = ux_allgather_blob(addr, (int)alen, &aoff);
        ucp_worker_release_address(ux_l[l].w, addr);
        for (int k = ux_l[l].k0; k < ux_l[l].k1; k++) {
            const int r = (me + k) % P;
            ucp_ep_params_t ep;
            memset(&ep, 0, sizeof ep);
            ep.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
            ep.address = (const ucp_address_t *)(addrs + aoff[P + r]);
            if ((st = ucp_ep_create(ux_l[l].w, &ep, &ux_l[l].ep[r])) != UCS_OK) ux_die("ep_create", st);
            if ((st = ucp_ep_rkey_unpack(ux_l[l].ep[r], rkeys + koff[P + r], &ux_l[l].rkey[r])) != UCS_OK)
                ux_die("rkey_unpack", st);
        }
        free(addrs); free(aoff);
    }
    free(rkeys); free(koff);

    /* wire every endpoint up now, as for osc/ucx: word 0 of half 0 never matches a
     * sequence number, so writing zero there is harmless */
    ucp_request_param_t fp;
    memset(&fp, 0, sizeof fp);
    for (int l = 0; l < L; l++) {
        ux_put(l, send, 1, 0);
        ux_flush(l);
        void *fr = ucp_worker_flush_nbx(ux_l[l].w, &fp);
        if (UCS_PTR_IS_ERR(fr)) ux_die("flush", UCS_PTR_STATUS(fr));
        if (fr) ux_wait(ux_l[l].w, fr, "flush");
    }
    ux_barrier();

    k3_tp.ll_recv = recv;
    k3_tp.ll_send = send;
    k3_tp.ll_hw = (int)hw;
    k3_tp.ll_maxw = LL_MAXW;
    k3_tp.ll_nsend = LL_NSEND;
    k3_tp.ll_nlanes = L;
    k3_tp.ll_flush = ux_flush;
    k3_tp.ll_poll = ux_poll;
    k3_tp.ll_done = ux_flush;
    k3_tp.ll_put = ux_put;
    return 0;
}

static void ux_fini(void)
{
    const int P = k3_tp.size, me = k3_tp.rank;
    ucp_request_param_t fp;
    memset(&fp, 0, sizeof fp);
    for (int l = 0; l < ux_nl; l++) {
        if (getenv("K3_TP_UCX_STATS"))
            fprintf(stderr, "k3_mpi: rank %d lane %d UCX puts not completed at once: %ld\n",
                    me, l, ux_l[l].npend);
        ux_flush(l);
        void *fr = ucp_worker_flush_nbx(ux_l[l].w, &fp);
        if (fr && !UCS_PTR_IS_ERR(fr)) ux_wait(ux_l[l].w, fr, "flush");
    }
    ux_barrier();
    for (int l = 0; l < ux_nl; l++)
        for (int r = 0; r < P; r++) {
            if (!ux_l[l].ep[r]) continue;
            ucp_rkey_destroy(ux_l[l].rkey[r]);
            void *cr = ucp_ep_close_nbx(ux_l[l].ep[r], &fp);
            if (cr && !UCS_PTR_IS_ERR(cr)) {
                while (ucp_request_check_status(cr) == UCS_INPROGRESS) ucp_worker_progress(ux_l[l].w);
                ucp_request_free(cr);
            }
        }
    ux_barrier();
    ucp_mem_unmap(ux_ctx, ux_rmem);
    ucp_mem_unmap(ux_ctx, ux_smem);
    for (int l = 0; l < ux_nl; l++) {
        ucp_worker_destroy(ux_l[l].w);
        free(ux_l[l].ep); free(ux_l[l].rkey);
    }
    ucp_cleanup(ux_ctx);
    free(k3_tp.ll_recv); free(k3_tp.ll_send);
    free(ux_l); free(ux_raddr);
    k3_tp.ll_put = NULL;
}

static int use_ucx;
#endif


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
    const char *ux = getenv("K3_TP_UCX");
    const char *sh = getenv("K3_TP_SHM");
    if (k3_tp.size > 1 && !(os && !strcmp(os, "0"))) {
        if (!(sh && !strcmp(sh, "0")) && sh_init() == 0) return 0;
#ifdef K3_UCX
        use_ucx = !(ux && !strcmp(ux, "0"));
        if (use_ucx) { ux_init(); return 0; }
#else
        if (ux && strcmp(ux, "0") && k3_tp.rank == 0)
            fprintf(stderr, "k3_mpi: K3_TP_UCX set but this build has no UCX (make MPI=1 UCX=<prefix>)\n");
#endif
        ll_init();
    }
    return 0;
}

void k3_mpi_finalize(void)
{
    if (sh_win != MPI_WIN_NULL) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_free(&sh_win);
        MPI_Comm_free(&sh_comm);
        free(k3_tp.ll_send); free(sh_peer);
        k3_tp.ll_put = NULL;
        MPI_Finalize();
        return;
    }
#ifdef K3_UCX
    if (use_ucx) { ux_fini(); MPI_Finalize(); return; }
#endif
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

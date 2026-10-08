/* k3_ops.c - the numeric core of the Kimi K3 engine.
 *
 * Every routine here is gated on a JSON fixture under tests/fixtures/ops/, generated
 * by tools/emit_fixtures.py from the pure-torch reference. Per-op fixtures exist
 * alongside the full-model oracle because the oracle is pass-or-fail: it proves the
 * stack is wrong without indicating which of ~40 kernels is responsible.
 *
 * FLOATING-POINT CONTRACT. This file requires -ffp-contract=off and no -ffast-math;
 * both build systems set them. The kernels are written so that three implementations
 * of the same operation agree exactly:
 *
 *   - the scalar C99 path, which is the reference,
 *   - the OpenMP path (guarded by _OPENMP), which splits independent output rows across
 *     threads and forms every norm statistic in the fixed chunk order of the canonical
 *     reductions below, so no result depends on the thread or rank count,
 *   - the AVX2 path (guarded by __AVX2__), which reproduces the scalar code's
 *     four-accumulator partition and reduction tree exactly rather than choosing a
 *     more natural one,
 *   - the NEON path (guarded by __ARM_NEON on aarch64), which maps the same scalar
 *     accumulators onto 2-lane double vectors, element i in the same accumulator and
 *     the same reduction tree, so it is bound by the identical bit-exactness contract.
 *
 * That last point is the reason several loops here look hand-unrolled for no visible
 * gain: the unrolling fixes a summation ORDER that the vector path must match. Reduce
 * them to the obvious form and the paths diverge in the last bits, which shows up as
 * a fixture failure on one machine and a pass on another.
 *
 * Accumulators are double in the scalar, AVX2 and NEON paths. The AVX-512 GEMVs
 * accumulate in fp32 instead (see K3_AVX512): they are bandwidth bound, and fp32 holds
 * the model's precision with room to spare.
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime under -std=c99 */

#include "k3.h"
#include "k3_gq.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define K3_AVX512 1
#include <immintrin.h>

/* The fp32 GEMV row scheme shared by the fp32, bf16 and MXFP4 kernels: element i goes
 * to lane i%16 of accumulator (i/16)%4, the tail is a masked chunk that leaves the
 * other lanes alone, and a row reduces as (a0+a1)+(a2+a3) and then this fixed lane
 * tree. So the same weight values give the same bits whatever their storage, which
 * keeps bf16 == fp32 and fused MXFP4 == dequantise-then-matmul exact. */
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

static inline __mmask16 v512_tail(int n) { return n >= 16 ? 0xFFFF : (__mmask16)((1u << n) - 1); }
#endif

int    k3_prof_on = 0;
double k3_prof_s[K3P_N];
const char *const k3_prof_name[K3P_N] = {
    "embed", "kda proj", "kda core", "kda out",
    "mla proj", "mla core", "mla out", "attnres+norms",
    "router", "moe down", "experts", "moe up", "shared",
    "dense mlp", "head", "tp comm", "tp wait"
};

double k3_prof_now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

K3Tp k3_tp = { .rank = 0, .size = 1 };

#if defined(K3_AVX512) && defined(__AVX512BF16__)
#define K3_DPBF16 1
#endif

int k3_act_bf16 = 0;

int k3_act_bf16_supported(void)
{
#if defined(K3_DPBF16)
    return 1;
#else
    return 0;
#endif
}

/* fp32 -> bf16, round to nearest even: the one rounding every bf16 path uses */
static inline uint16_t k3_f2bf(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static inline int seg_lp(const K3Seg *s) { return (k3_act_bf16 & K3_BF16_TP) && !s->exact; }


#ifdef _OPENMP
/* Dissemination barrier: in round r thread i raises the flag of thread (i + 2^r) mod n
 * and waits for its own, so log2(n) one-line handoffs replace a central counter.
 * 0.9 us at 64 threads against 3.8 us for GOMP's barrier. Epochs only grow, so a flag
 * left from an earlier barrier never releases a later one. That needs every full-size
 * team to call it the same number of times per thread, which SPMD code does; a team of
 * any other size uses the OpenMP barrier instead. */
#define K3_BAR_MAXT   512
#define K3_BAR_ROUNDS 9
typedef struct { unsigned v; char pad[60]; } k3_bar_line;
static k3_bar_line k3_bar_flag[K3_BAR_MAXT][K3_BAR_ROUNDS];
static k3_bar_line k3_bar_epoch[K3_BAR_MAXT];

void k3_team_barrier(void)
{
    const int n = omp_get_num_threads();
    if (n <= 1) return;
    if (n > K3_BAR_MAXT || n != omp_get_max_threads()) {
#pragma omp barrier
        return;
    }
    const int me = omp_get_thread_num();
    const unsigned e = ++k3_bar_epoch[me].v;
    for (int r = 0, d = 1; d < n; r++, d <<= 1) {
        __atomic_store_n(&k3_bar_flag[(me + d) % n][r].v, e, __ATOMIC_RELEASE);
        while ((int)(__atomic_load_n(&k3_bar_flag[me][r].v, __ATOMIC_ACQUIRE) - e) < 0) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    }
}
#endif

static void k3_fatal_oom(const char *what, size_t bytes);

/* Packed exchange layout, rank-major: rank r's parts of every segment, in segment order.
 * The pack buffer, counts and send state belong to thread 0. */
static float *tp_pack;
static size_t tp_pack_cap;
static int   *tp_cnt;
#define TP_MAXSEG (K3_MAX_TOPK + 2)
static int      tp_open;                   /* a gather is outstanding */
static uint32_t tp_seq;                    /* one-sided sequence number, set before a barrier */
#define TP_MAXLANES 64
static struct { int v; char pad[60]; } tp_slot[TP_MAXLANES];   /* per lane: position in its send ring */

/* Threads that issue one-sided puts; thread t drives lanes t, t + S, ... Skew mode
 * barriers on thread 0 before sending, so it keeps one sender. */
static inline int tp_senders(void)
{
    const int L = k3_tp.ll_nlanes > 0 ? k3_tp.ll_nlanes : 1, n = k3_nth();
    if (L == 1 || k3_tp.skew) return 1;
    return L < n ? L : n;
}

static void tp_send(const K3Seg *seg, int nseg, int total, int ll);
static void tp_put_lanes(const K3Seg *seg, int nseg);
static void tp_recv(const K3Seg *seg, int nseg, int total, int ll);

static int tp_total(const K3Seg *seg, int nseg)
{
    long n = 0;
    for (int s = 0; s < nseg; s++) n += seg[s].n;
    return (int)n;
}

/* Rank r's block of segment s in transport units: floats, or one-sided words holding a
 * bf16 pair each. *nf gets its length in floats. */
static inline int tp_blk(const K3Seg *s, int r, int ll, int *nf)
{
    const int nu = s->n / s->unit, P = k3_tp.size;
    const int len = ((int)((long)nu * (r + 1) / P) - (int)((long)nu * r / P)) * s->unit;
    if (nf) *nf = len;
    return ll && seg_lp(s) ? (len + 1) >> 1 : len;
}

static int tp_is_ll(int total) { return k3_tp.ll_put && total <= k3_tp.ll_maxw; }

/* Transport units of a gather, and whether it goes one-sided. */
static int tp_units(const K3Seg *seg, int nseg, int *ll)
{
    long w = 0;
    for (int r = 0; r < k3_tp.size; r++)
        for (int s = 0; s < nseg; s++) w += tp_blk(&seg[s], r, 1, NULL);
    *ll = tp_is_ll((int)w);
    return *ll ? (int)w : tp_total(seg, nseg);
}

/* One rank: round this thread's share of every bf16 segment, as a gather would. */
static void tp_round_local(const K3Seg *seg, int nseg)
{
    for (int s = 0; s < nseg; s++) {
        if (!seg_lp(&seg[s])) continue;
        int lo, hi;
        k3_split(seg[s].n, &lo, &hi);
        for (int i = lo; i < hi; i++) seg[s].p[i] = k3_bf16f(k3_f2bf(seg[s].p[i]));
    }
}

/* This thread's share of this rank's own block of every bf16 segment, rounded. */
static void tp_round_own(const K3Seg *seg, int nseg)
{
    for (int s = 0; s < nseg; s++) {
        if (!seg_lp(&seg[s])) continue;
        int u0, u1, lo, hi;
        k3_tp_part(seg[s].n / seg[s].unit, &u0, &u1);
        float *p = seg[s].p + (size_t)u0 * seg[s].unit;
        k3_split((u1 - u0) * seg[s].unit, &lo, &hi);
        for (int i = lo; i < hi; i++) p[i] = k3_bf16f(k3_f2bf(p[i]));
    }
}

/* Team collectives. begin: a barrier makes every thread's rows visible, then thread 0
 * sends (MPI_THREAD_FUNNELED) while the others go on with independent work. end: every
 * thread receives its share straight into the segments, then a barrier publishes them.
 * One-sided words carry their own readiness, so receiving needs no barrier with thread 0;
 * allgatherv has to wait for it to come back from MPI. With one rank a gather is still
 * the team barrier callers rely on between producing and consuming a vector. */
void k3_tp_gather_begin(const K3Seg *seg, int nseg)
{
    if (k3_tp.size <= 1) {
        k3_sync();
        if (k3_act_bf16 & K3_BF16_TP) tp_round_local(seg, nseg);
        return;
    }
    int ll;
    const int total = tp_units(seg, nseg, &ll);
    if (ll && k3_tid() == 0 && ++tp_seq == 0) tp_seq = 1;   /* 0 is what a fresh window holds */
    k3_sync();
    if (k3_tid() == 0) tp_send(seg, nseg, total, ll);
    if (ll) tp_put_lanes(seg, nseg);
}

void k3_tp_gather_end(const K3Seg *seg, int nseg)
{
    if (k3_tp.size <= 1) { if (k3_act_bf16 & K3_BF16_TP) k3_sync(); return; }
    int ll;
    const int total = tp_units(seg, nseg, &ll);
    if (!ll) k3_sync();
    const double t0 = k3_prof_on && k3_tid() == 0 ? k3_prof_now() : 0.0;
    tp_recv(seg, nseg, total, ll);
    /* the owner keeps exactly what its peers received; nothing reads its own block
     * between begin and end, so rounding it here equals rounding it while packing */
    if (ll && (k3_act_bf16 & K3_BF16_TP)) tp_round_own(seg, nseg);
    if (ll && k3_tp.ll_done) {
        const int S = tp_senders();
        for (int l = k3_tid(); k3_tid() < S && l < k3_tp.ll_nlanes; l += S) k3_tp.ll_done(l);
    }
    if (k3_prof_on && k3_tid() == 0) k3_prof_s[K3P_COMM] += k3_prof_now() - t0;
    k3_sync();
    if (k3_tid() == 0) tp_open = 0;
}

void k3_tp_gather(const K3Seg *seg, int nseg)
{
    k3_tp_gather_begin(seg, nseg);
    k3_tp_gather_end(seg, nseg);
}

/* Spin until each of n one-sided words carries this call's sequence number; store the
 * payloads to d unless d is NULL (a flag word). lp: each word is a bf16 pair, stored as
 * two floats, of which nf are wanted. */
static void ll_wait(float *d, const volatile uint64_t *w, int n, uint32_t seq, int lp, int nf)
{
    for (int i = 0; i < n; i++) {
        uint64_t v;
        long spins = 0;
        while (((v = w[i]) >> 32) != seq) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
            /* sending threads progress their own lanes (MPI: one lane, thread 0);
             * a flush may cost a round trip, so rarely */
            if (k3_tid() < k3_tp.ll_nlanes && k3_tid() < tp_senders()) {
                const int S = tp_senders();
                ++spins;
                for (int l = k3_tid(); l < k3_tp.ll_nlanes; l += S) {
                    if (spins % (1L << 20) == 0) k3_tp.ll_flush(l);
                    else if (k3_tp.ll_poll && (spins & 63) == 0) k3_tp.ll_poll(l);
                }
            }
        }
        if (!d) continue;
        const uint32_t u = (uint32_t)v;
        if (lp) {
            d[2 * i] = k3_bf16f((uint16_t)u);
            if (2 * i + 1 < nf) d[2 * i + 1] = k3_bf16f((uint16_t)(u >> 16));
        } else {
            memcpy(d + i, &u, 4);
        }
    }
}

static void tp_send(const K3Seg *seg, int nseg, int total, int ll)
{
    if (tp_open || nseg > TP_MAXSEG) {
        fprintf(stderr, "k3_tp_gather_begin: %s\n",
                tp_open ? "a gather is already outstanding" : "too many segments");
        abort();
    }
    tp_open = 1;
    if (k3_tp.skew && k3_tp.barrier) {
        const double tw = k3_prof_on ? k3_prof_now() : 0.0;
        k3_tp.barrier(k3_tp.ctx);
        if (k3_prof_on) k3_prof_s[K3P_WAIT] += k3_prof_now() - tw;
    }
    const double t0 = k3_prof_on ? k3_prof_now() : 0.0;
    const int P = k3_tp.size, me = k3_tp.rank;
    if (!tp_cnt) {
        tp_cnt = (int *)malloc((size_t)2 * P * sizeof(int));
        if (!tp_cnt) k3_fatal_oom("TP gather counts", (size_t)2 * P * sizeof(int));
    }
    int *cnt = tp_cnt, *dsp = tp_cnt + P;
    int off = 0;
    for (int r = 0; r < P; r++) {
        dsp[r] = off;
        for (int s = 0; s < nseg; s++) off += tp_blk(&seg[s], r, ll, NULL);
        cnt[r] = off - dsp[r];
    }
    if (ll) {
        /* the puts are issued by tp_put_lanes, on every sending thread */
    } else {
        if ((size_t)total > tp_pack_cap) {
            free(tp_pack);
            tp_pack = (float *)malloc((size_t)total * sizeof(float));
            if (!tp_pack) k3_fatal_oom("TP gather buffer", (size_t)total * sizeof(float));
            tp_pack_cap = (size_t)total;
        }
        float *mine = tp_pack + dsp[me];
        for (int s = 0; s < nseg; s++) {
            int lo, hi; k3_tp_part(seg[s].n / seg[s].unit, &lo, &hi);
            const int n = (hi - lo) * seg[s].unit;
            float *p = seg[s].p + (size_t)lo * seg[s].unit;
            if (seg_lp(&seg[s]))
                for (int i = 0; i < n; i++) p[i] = k3_bf16f(k3_f2bf(p[i]));
            memcpy(mine, p, (size_t)n * sizeof(float));
            mine += n;
        }
        k3_tp.allgatherv(tp_pack, cnt, dsp, k3_tp.ctx);
    }
    k3_tp.calls++;
    k3_tp.floats += (double)tp_total(seg, nseg);
    if (k3_prof_on) k3_prof_s[K3P_COMM] += k3_prof_now() - t0;
}

/* This rank's block as tagged words; bf16 segments go as rounded pairs. */
static void tp_pack_words(uint64_t *o, const K3Seg *seg, int nseg, uint64_t tag)
{
    for (int s = 0; s < nseg; s++) {
        int lo, hi; k3_tp_part(seg[s].n / seg[s].unit, &lo, &hi);
        const float *p = seg[s].p + (size_t)lo * seg[s].unit;
        const int n = (hi - lo) * seg[s].unit;
        if (seg_lp(&seg[s])) {
            for (int i = 0; i < n; i += 2) {
                const uint16_t b0 = k3_f2bf(p[i]), b1 = i + 1 < n ? k3_f2bf(p[i + 1]) : 0;
                *o++ = tag | (uint32_t)b1 << 16 | b0;
            }
        } else {
            for (int i = 0; i < n; i++) {
                uint32_t u;
                memcpy(&u, p + i, 4);
                *o++ = tag | u;
            }
        }
    }
}

/* One-sided send: each sending thread packs this rank's block into its lanes' own send
 * rings (see k3_mpi.c: one flush per lap covers a ring) and puts it to their peers. */
static void tp_put_lanes(const K3Seg *seg, int nseg)
{
    const int S = tp_senders(), t = k3_tid();
    if (t >= S) return;
    const double t0 = k3_prof_on && t == 0 ? k3_prof_now() : 0.0;
    const int me = k3_tp.rank;
    int dsp = 0, cnt = 0;
    for (int r = 0; r <= me; r++)
        for (int s = 0; s < nseg; s++) {
            const int b = tp_blk(&seg[s], r, 1, NULL);
            if (r < me) dsp += b; else cnt += b;
        }
    const uint64_t tag = (uint64_t)tp_seq << 32;
    const long half = (long)(tp_seq & 1) * k3_tp.ll_hw;
    for (int l = t; l < k3_tp.ll_nlanes; l += S) {
        if (++tp_slot[l].v == k3_tp.ll_nsend) { tp_slot[l].v = 0; k3_tp.ll_flush(l); }
        uint64_t *snd = k3_tp.ll_send + ((size_t)l * k3_tp.ll_nsend + tp_slot[l].v) * k3_tp.ll_hw;
        /* an empty block sends its flag word, so every call orders every pair of ranks */
        if (cnt > 0) {
            tp_pack_words(snd, seg, nseg, tag);
            k3_tp.ll_put(l, snd, cnt, half + dsp);
        } else {
            snd[0] = tag;
            k3_tp.ll_put(l, snd, 1, half + k3_tp.ll_maxw + me);
        }
    }
    if (k3_prof_on && t == 0) k3_prof_s[K3P_COMM] += k3_prof_now() - t0;
}

/* This thread's share of the packed words, other ranks' blocks only, into the segments. */
static void tp_recv(const K3Seg *seg, int nseg, int total, int ll)
{
    const int P = k3_tp.size, me = k3_tp.rank;
    const uint32_t seq = tp_seq;
    const volatile uint64_t *rcv =
        ll ? k3_tp.ll_recv + (size_t)(seq & 1) * k3_tp.ll_hw : NULL;
    int a, b;
    k3_split(total, &a, &b);
    int off = 0;
    for (int r = 0; r < P && off < b; r++)
        for (int s = 0; s < nseg; s++) {
            int nf;
            const int len = tp_blk(&seg[s], r, ll, &nf), lp = ll && seg_lp(&seg[s]);
            const int nu = seg[s].n / seg[s].unit;
            const int lo = (int)((long)nu * r / P);
            const int x0 = off > a ? off : a, x1 = off + len < b ? off + len : b;
            if (r != me && x1 > x0) {
                float *d = seg[s].p + (size_t)lo * seg[s].unit + (size_t)(x0 - off) * (lp ? 2 : 1);
                if (ll) ll_wait(d, rcv + x0, x1 - x0, seq, lp, nf - 2 * (x0 - off));
                else    memcpy(d, tp_pack + x0, (size_t)(x1 - x0) * sizeof(float));
            }
            off += len;
        }
    if (ll && k3_tid() == 0)
        for (int r = 0; r < P; r++) {
            int len = 0;
            for (int s = 0; s < nseg; s++) len += tp_blk(&seg[s], r, 1, NULL);
            if (r != me && len == 0) ll_wait(NULL, rcv + k3_tp.ll_maxw + r, 1, seq, 0, 0);
        }
}

/* --------------------------------------------------------- fatal errors ---- */
/* Several kernels here need a small temporary that cannot be hoisted into caller-owned
 * scratch without changing a published signature. They are hundreds of bytes to a few
 * kilobytes, on a path where the engine has already reserved tens of gigabytes, so a
 * failure means the process is finished either way.
 *
 * What matters is HOW it finishes. These kernels return void, so a failed allocation
 * could only be handled by returning early, which leaves the output buffer holding
 * whatever was in it before, and the caller consumes it as a result. The run then
 * completes and prints a plausible token computed from uninitialised memory. That is the
 * one failure mode this engine treats as unacceptable, so allocation failure aborts
 * loudly instead.
 *
 * This is deliberately NOT how streamed experts are handled: an expert that fails to
 * load is recoverable in principle, so it is counted in k3_expert_drops and the decision
 * is left to the caller. Memory exhaustion inside a kernel is not recoverable. */
static void k3_fatal_oom(const char *what, size_t bytes)
{
    fprintf(stderr,
            "k3: FATAL, could not allocate %zu bytes for %s.\n"
            "    Aborting rather than continuing with an uninitialised buffer, which\n"
            "    would produce plausible-looking but meaningless output.\n",
            bytes, what);
    abort();
}

/* The same rule, for a bound that is exceeded rather than an allocation that fails.
 * k3_mla_cached writes its output only after the capacity check, so returning early
 * would hand the caller back whatever the scratch buffer held from the previous layer,
 * and k3_decoder_layer_inc folds that straight into the residual. The run finishes and
 * prints a token derived from the wrong layer's activations. Abort instead. */
static void k3_fatal_bound(const char *what, long value, long limit)
{
    fprintf(stderr,
            "k3: FATAL, %s is %ld, which exceeds the limit of %ld.\n"
            "    Aborting rather than returning without writing the output buffer,\n"
            "    which would fold the previous layer's values into the residual and\n"
            "    produce plausible-looking but meaningless output.\n"
            "    Shorten the prompt, lower --gen, or drop --incremental.\n",
            what, value, limit);
    abort();
}

/* -------------------------------------------------------- thread scratch ---- */
/* Persistent per-thread temporaries, so the forward pass does not allocate once the
 * buffers have grown to their working size. Slot s of team thread t grows (doubling,
 * 64-byte aligned) and is never freed. The main thread is tid 0 both inside and outside
 * a team, so a slot must not be live across a call that uses the same slot. */
enum { K3S_MLA_SC, K3S_MOE_AN, K3S_ROUTER_CHOICE, K3S_ROUTER_SCORE, K3S_MOE_SCORE,
       K3S_MXFP4_XD, K3S_PREFILL, K3S_SEGS, K3S_KDA_STATE, K3S_XBF, K3S_QKV, K3S_XQ8, K3S_N };
#define K3S_MAXT 1024
static void  *k3s_buf[K3S_MAXT][K3S_N];
static size_t k3s_cap[K3S_MAXT][K3S_N];

static void *k3_scratch(int slot, size_t bytes, const char *what)
{
    const int t = k3_tid();
    if (t >= K3S_MAXT) k3_fatal_bound("threads per team", t + 1, K3S_MAXT);
    if (bytes > k3s_cap[t][slot]) {
        size_t cap = k3s_cap[t][slot] * 2;
        if (cap < bytes) cap = bytes;
        cap = (cap + 63) & ~(size_t)63;
        free(k3s_buf[t][slot]);
        void *p = NULL;
        if (posix_memalign(&p, 64, cap) != 0) k3_fatal_oom(what, cap);
        k3s_buf[t][slot] = p;
        k3s_cap[t][slot] = cap;
    }
    return k3s_buf[t][slot];
}

/* --------------------------------------------------- canonical reductions ---- */
/* Every sum of squares and every AttnRes dot product is formed over fixed K3_RC-element
 * chunks, sequentially in double within a chunk, and the chunk sums are then added in
 * chunk order. The order depends on n alone, never on the thread or rank count, so the
 * results are bit-identical at any team or TP size while the chunks run in parallel or
 * come out of the epilogue of the op writing the vector. Norms are split in three: chunk
 * sums (parallel, or fused into the producer), statistics (combined by every thread,
 * which needs no broadcast), and the scale (fused into writing the consumer's input). */
#define K3_RC 128
#define K3_RC_MAXPART 32768
static inline int rc_n(int n) { return (n + K3_RC - 1) / K3_RC; }

static double rc_sq(const float *x, int c, int n)       /* chunk c: sum of x^2 */
{
    const int i0 = c * K3_RC, i1 = i0 + K3_RC < n ? i0 + K3_RC : n;
    double s = 0.0;
    for (int i = i0; i < i1; i++) s += (double)x[i] * (double)x[i];
    return s;
}

static double rc_total(const double *part, int nc)
{
    double s = 0.0;
    for (int c = 0; c < nc; c++) s += part[c];
    return s;
}

static double rc_sumsq(const float *x, int n)            /* one thread, same order */
{
    double s = 0.0;
    for (int c = 0; c < rc_n(n); c++) s += rc_sq(x, c, n);
    return s;
}

/* Team-shared chunk sums for one reduction: written, barrier, read. A ring of four with
 * every thread on the same turn, so a buffer is rewritten only after two later barriers.
 * A one-thread team has its own ring, which keeps the team threads' turns in step. */
static double   rc_ring[4][K3_RC_MAXPART];
static double   rc_solo[4][K3_RC_MAXPART];
static struct { unsigned v; char pad[60]; } rc_turn[K3S_MAXT];
static unsigned rc_solo_turn;

static double *rc_parts(int need)
{
    if (need > K3_RC_MAXPART) k3_fatal_bound("reduction chunks", need, K3_RC_MAXPART);
    if (k3_nth() == 1) return rc_solo[rc_solo_turn++ & 3];
    return rc_ring[rc_turn[k3_tid()].v++ & 3];
}

/* this thread's chunks of the sum of squares of x */
static void rc_sq_team(double *part, const float *x, int n)
{
    int lo, hi;
    k3_split(rc_n(n), &lo, &hi);
    for (int c = lo; c < hi; c++) part[c] = rc_sq(x, c, n);
}

static float rms_inv(const double *part, int n, float eps)
{
    return (float)(1.0 / sqrt(rc_total(part, rc_n(n)) / (double)n + (double)eps));
}

/* this thread's slice of y = w * x * inv */
static void rms_scale(float *y, const float *x, const float *w, int n, float inv)
{
    int lo, hi;
    k3_split(n, &lo, &hi);
    for (int i = lo; i < hi; i++) y[i] = w[i] * x[i] * inv;
}

/* ------------------------------------------------------------- layer map ---- */
/* The released config lists full_attn_layers ONE-BASED, and
 * configuration_kimi_k3.py:152-156 tests (layer_idx + 1) in kda_layers. Getting this
 * off by one silently swaps KDA and MLA layers throughout the stack. */
int k3_is_mla(const K3Cfg *c, int layer)
{
    for (int i = 0; i < c->n_full_attn; i++)
        if (c->full_attn[i] == layer + 1) return 1;
    return 0;
}
int k3_is_kda(const K3Cfg *c, int layer)   { return !k3_is_mla(c, layer); }
int k3_is_dense(const K3Cfg *c, int layer) { return layer < c->first_dense; }

/* --------------------------------------------------------------- rmsnorm ---- */
static void rmsnorm_serial(float *y, const float *x, const float *w, int n, float eps)
{
    /* double accumulator: 7168 squared terms in float32 loses real precision, and
     * every downstream comparison against the reference depends on this. */
    const float inv = (float)(1.0 / sqrt(rc_sumsq(x, n) / (double)n + (double)eps));
    for (int i = 0; i < n; i++) y[i] = w[i] * x[i] * inv;
}

/* Team form, in the three parts: chunk sums split over the team, one barrier, then every
 * thread forms the statistic itself and scales its own slice. In place is safe because
 * every read of x for the sums precedes the barrier. */
void k3_rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    double *part = rc_parts(rc_n(n));
    rc_sq_team(part, x, n);
    k3_sync();
    rms_scale(y, x, w, n, rms_inv(part, n, eps));
}

/* -------------------------------------------------------------- SiTU-GLU ---- */
static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

static void situ_range(float *y, const float *gate, const float *up, int n,
                       float b1, float b2);
static void mxfp4_check(int in, int group);
static void mxfp4_rows(float *y, const float *x, const double *xd,
                       const unsigned char *packed, const unsigned char *scales,
                       int in, int r0, int r1, int rbase, int group, int ilv);
static void matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                         const unsigned char *scales, int in, int rows, int group, int ilv);

/* See k3.h. Incremented whenever a streamed expert cannot be fetched. */
long k3_expert_drops = 0;
int  k3_expert_q8 = 0;

/* Rows [r0, r1) of one expert matrix in its K3_EQ_* format, indexed as mxfp4_rows is.
 * xq/dx select int8 activations for the GGUF formats; NULL keeps the exact fp32 path. */
static void eq_rows(float *y, const float *x, const double *xd, const int8_t *xq,
                    const float *dx, const unsigned char *p, const unsigned char *s, int qt,
                    int in, int r0, int r1, int rbase, int group, int ilv)
{
    if (qt == K3_EQ_MXFP4) { mxfp4_rows(y, x, xd, p, s, in, r0, r1, rbase, group, ilv); return; }
    if (r1 <= r0) return;
    const int t = qt == K3_EQ_IQ2XS ? K3_GG_IQ2_XS : K3_GG_IQ3_XXS;
    const unsigned char *W = p + (size_t)(r0 - rbase) * k3_gq_row_bytes(t, in);
    if (xq) {
        if (qt == K3_EQ_IQ2XS) k3_iq2xs_rows_q8(y + r0, xq, dx, W, in, 0, r1 - r0);
        else                   k3_iq3xxs_rows_q8(y + r0, xq, dx, W, in, 0, r1 - r0);
    } else {
        if (qt == K3_EQ_IQ2XS) k3_iq2xs_rows(y + r0, x, W, in, 0, r1 - r0);
        else                   k3_iq3xxs_rows(y + r0, x, W, in, 0, r1 - r0);
    }
}

static void eq_matmul(float *y, const float *x, const unsigned char *p, const unsigned char *s,
                      int qt, int in, int rows, int ilv)
{
    if (qt == K3_EQ_MXFP4) { matmul_mxfp4(y, x, p, s, in, rows, K3_MXFP4_GROUP, ilv); return; }
    K3_TEAM_IF(rows > 64, int lo, hi; k3_split(rows, &lo, &hi);
               eq_rows(y, x, NULL, NULL, NULL, p, s, qt, in, lo, hi, 0, 0, 0));
}

/* Rows [r0, r1) of y = W x. W holds every row, or under k3_tp.local only this rank's
 * rows, in which case r0 is by construction where the slice starts. Per row this IS
 * k3_mmw. */
static inline void mmw_rows(float *y, const float *x, const void *W, int wdt, int in,
                            int r0, int r1)
{
    if (r1 > r0)
        k3_mmw(y + r0, x, (const unsigned char *)W
                              + (k3_tp.local ? 0 : (size_t)r0 * k3_row_bytes(wdt, in)),
               wdt, in, r1 - r0);
}

static void matmul_f32_rows(float *y, const float *x, const float *W, int in, int o0, int o1);
static void matmul_bf16_rows(float *y, const float *x, const uint16_t *W, int in,
                             int o0, int o1);
static void matmul_q8_rows(float *y, const float *x, const void *W, int in, int o0, int o1);

/* Rows [r0, r1) of y = W x by the CALLING thread alone, no team split; `base` is where
 * this rank's slice starts when W holds only that slice (k3_tp.local). Lets a thread
 * produce and then consume its own rows with no barrier in between. */
static void mmw_part(float *y, const float *x, const void *W, int wdt, int in,
                     int base, int r0, int r1)
{
    if (r1 <= r0) return;
    const unsigned char *Wr = (const unsigned char *)W
        + (size_t)(k3_tp.local ? r0 - base : r0) * k3_row_bytes(wdt, in);
    if (wdt == K3_WBF16)    matmul_bf16_rows(y + r0, x, (const uint16_t *)Wr, in, 0, r1 - r0);
    else if (wdt == K3_WQ8_0) k3_q80_rows(y + r0, x, Wr, in, 0, r1 - r0);
    else if (wdt == K3_WI8) matmul_q8_rows(y + r0, x, Wr, in, 0, r1 - r0);
    else                    matmul_f32_rows(y + r0, x, (const float *)Wr, in, 0, r1 - r0);
}

/* This thread's k3_split share of rows [r0, r1) of q, k and v, the split mmw_rows uses,
 * read from the row-interleaved bf16 qkv matrix as one contiguous run. */
static void qkv_rows(float *q, float *k, float *v, const float *x, const void *W, int in,
                     int r0, int r1)
{
    int lo, hi;
    k3_split(r1 - r0, &lo, &hi);
    if (hi <= lo) return;
    const int n = 3 * (hi - lo);
    float *y = (float *)k3_scratch(K3S_QKV, (size_t)n * sizeof(float), "qkv rows");
    const size_t row0 = (size_t)3 * (k3_tp.local ? lo : r0 + lo);
    matmul_bf16_rows(y, x, (const uint16_t *)W + row0 * in, in, 0, n);
    for (int i = 0; i < hi - lo; i++) {
        q[r0 + lo + i] = y[3 * i]; k[r0 + lo + i] = y[3 * i + 1]; v[r0 + lo + i] = y[3 * i + 2];
    }
}

void k3_mmw_tp(float *y, const float *x, const void *W, int wdt, int in, int out)
{
    int r0, r1;
    k3_tp_part(out, &r0, &r1);
    mmw_rows(y, x, W, wdt, in, r0, r1);
    const K3Seg s = { y, out, 1, 0 };
    k3_tp_gather(&s, 1);
}

/* Gather `rows` consecutive vectors of n floats, each partitioned in blocks of unit. */
static void tp_gather_rows(float *p, int rows, int n, int unit)
{
    if ((k3_tp.size <= 1 && !(k3_act_bf16 & K3_BF16_TP)) || rows <= 0) { k3_sync(); return; }
    K3Seg *s = (K3Seg *)k3_scratch(K3S_SEGS, (size_t)rows * sizeof(K3Seg), "TP gather segments");
    for (int r = 0; r < rows; r++) {
        s[r].p = p + (size_t)r * n; s[r].n = n; s[r].unit = unit; s[r].exact = 0;
    }
    k3_tp_gather(s, rows);
}

void k3_situ_glu(float *y, const float *x, int n, float b1, float b2)
{
    situ_range(y, x, x + n, n, b1, b2);
}

static void situ_range(float *y, const float *gate, const float *up, int n,
                       float b1, float b2)
{
    for (int i = 0; i < n; i++) {
        const float g = gate[i];
        /* The sigmoid takes the UNCAPPED gate. Feeding it the capped value instead
         * still yields a bounded, plausible function and is WRONG.
         * modeling_kimi_linear.py:79 */
        const float a = b1 * tanhf(g / b1) * sigmoidf_(g);
        const float u = b2 * tanhf(up[i] / b2);
        y[i] = a * u;
    }
}

/* Widest column block k3_kda_step works on at once; wider heads are walked in blocks. */
#define K3_KDA_STEP_DV 256
#define K3_CONV_HIST_MAX 16

/* ------------------------------------------------------------- ShortConv ---- */
/* Causal depthwise conv, SiLU fused, exactly as ShortConvolution(activation='silu').
 * state[c*(k-1) + j] holds the previous inputs for channel c, oldest first.
 * Updated in place so a decode loop can carry it forward. Channels are independent, so
 * they are split across threads with no change to any channel's arithmetic.
 * Channels [c0, c1) only; x and y advance by `stride` floats per time step. */
static void shortconv_part(float *y, const float *x, const float *w, float *state,
                           int c0, int c1, int stride, int k, int T)
{
    const int hist = k - 1;
    {
        /* hist can be 0 when k == 1; every use of buf below is guarded on hist. */
        float lbuf[K3_CONV_HIST_MAX];
        float *buf = lbuf;
        if (hist > K3_CONV_HIST_MAX) {
            buf = (float *)malloc((size_t)hist * sizeof(float));
            if (!buf) k3_fatal_oom("ShortConv history", (size_t)hist * sizeof(float));
        }
        int lo, hi;
        k3_split(c1 - c0, &lo, &hi);
        for (int c = c0 + lo; c < c0 + hi; c++) {
            if (hist) {   /* memcpy/memset with a NULL pointer is UB even at length 0 */
                if (state) memcpy(buf, state + (size_t)c * hist, (size_t)hist * sizeof(float));
                else       memset(buf, 0, (size_t)hist * sizeof(float));
            }

            for (int t = 0; t < T; t++) {
                const float cur = x[(size_t)t * stride + c];
                /* taps are ordered oldest..newest, matching conv1d over a left-padded
                 * sequence: w[k-1] multiplies the CURRENT input. */
                float acc = w[(size_t)c * k + hist] * cur;
                for (int j = 0; j < hist; j++)
                    acc += w[(size_t)c * k + j] * buf[j];

                for (int j = 0; j + 1 < hist; j++) buf[j] = buf[j + 1];
                if (hist > 0) buf[hist - 1] = cur;

                y[(size_t)t * stride + c] = acc * sigmoidf_(acc);     /* SiLU, fused */
            }
            if (state && hist) memcpy(state + (size_t)c * hist, buf, (size_t)hist * sizeof(float));
        }
        if (buf != lbuf) free(buf);
    }
}

/* Channels [c0, c1); in a team, this thread's share of them, with no barrier. */
static void shortconv_range(float *y, const float *x, const float *w, float *state,
                            int c0, int c1, int stride, int k, int T)
{
    K3_TEAM_IF((long)(c1 - c0) * T > 4096,
               shortconv_part(y, x, w, state, c0, c1, stride, k, T));
}

void k3_shortconv(float *y, const float *x, const float *w, float *state,
                  int channels, int k, int T)
{
    shortconv_range(y, x, w, state, 0, channels, channels, k, T);
}

/* ------------------------------------------------------------ KDA decay ----- */
static void kda_decay_part(float *g, float *alpha, const float *z, const float *A_log,
                           const float *dt_bias, int H, int D, float lb)
{
    int lo, hi;
    k3_split(H * D, &lo, &hi);
    for (int i = lo; i < hi; i++) {
        /* PER HEAD. The checkpoint stores head_dim floats but only the first H are
         * nonzero. Indexing this per channel is a silent, fatal error. */
        const float a = expf(A_log[i / D]);
        const float u  = a * (z[i] + dt_bias[i]);
        const float gi = lb * sigmoidf_(u);   /* in (lb, 0] */
        g[i] = gi;
        alpha[i] = expf(gi);                  /* in (e^lb, 1] */
    }
}

void k3_kda_decay(float *g, float *alpha, const float *z, const float *A_log,
                  const float *dt_bias, int H, int D, float lb)
{
    K3_TEAM_IF((long)H * D > 4096, kda_decay_part(g, alpha, z, A_log, dt_bias, H, D, lb));
}

/* -------------------------------------------------------- KDA recurrence ---- */
/* Columns [j0, j1) of one step; j1 - j0 <= K3_KDA_STEP_DV. Every step below touches
 * column j only through S[.][j], u[j], v[j] and o[j], so column blocks are independent
 * and a caller may run them on different threads with bit-identical results. */
static void kda_step_cols(float *S, float *o, const float *q, const float *k,
                          const float *v, const float *alpha, float beta,
                          int dk, int dv, int j0, int j1)
{
    /* 1. channel-wise decay: scale ROW i of S by alpha[i]. The gate is per key
     *    channel, not a scalar, which is what "channel-wise forget gate" means. */
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float a = alpha[i];
        for (int j = j0; j < j1; j++) row[j] *= a;
    }

    /* 2. read the state along k:  u = S^T k. Automatic storage: this runs once per head
     *    per token per KDA layer from inside an OpenMP loop. */
    float u[K3_KDA_STEP_DV];
    for (int j = j0; j < j1; j++) u[j - j0] = 0.0f;
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        if (ki == 0.0f) continue;
        const float *row = S + (size_t)i * dv;
        for (int j = j0; j < j1; j++) u[j - j0] += ki * row[j];
    }

    /* 3. rank-one delta write. (v - u) is the prediction error: this is what makes
     *    it a DELTA rule rather than plain accumulation. */
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        if (ki == 0.0f) continue;
        float *row = S + (size_t)i * dv;
        for (int j = j0; j < j1; j++) row[j] += ki * beta * (v[j] - u[j - j0]);
    }

    /* 4. output from the ALREADY UPDATED state: o = S^T q */
    for (int j = j0; j < j1; j++) o[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
        const float qi = q[i];
        if (qi == 0.0f) continue;
        const float *row = S + (size_t)i * dv;
        for (int j = j0; j < j1; j++) o[j] += qi * row[j];
    }
}

void k3_kda_step(float *S, float *o, const float *q, const float *k,
                 const float *v, const float *alpha, float beta, int dk, int dv)
{
    for (int j0 = 0; j0 < dv; j0 += K3_KDA_STEP_DV) {
        const int j1 = (dv - j0 < K3_KDA_STEP_DV) ? dv : j0 + K3_KDA_STEP_DV;
        kda_step_cols(S, o, q, k, v, alpha, beta, dk, dv, j0, j1);
    }
}

/* ---------------------------------------------------------------- matmul ---- */
/* The dominant cost in the engine: essentially all compute time is spent here or in
 * k3_matmul_mxfp4. Two properties of this kernel are load bearing.
 *
 *   1. OUTPUT ROWS ARE INDEPENDENT. Parallelising the outer loop introduces no
 *      reduction and no race, and changes no arithmetic at all, each row is summed
 *      by exactly one thread in exactly the order below. Results are therefore
 *      identical at any thread count, which the fixtures rely on.
 *
 *   2. FOUR ACCUMULATORS, PARTITIONED BY i%4, REDUCED AS (a0+a1)+(a2+a3). The split
 *      keeps the FMA pipeline full, a single accumulator serialises on the latency
 *      chain, but it is written out explicitly rather than left to the compiler
 *      because it fixes a summation ORDER. k3_matmul_bf16 and both AVX2 paths
 *      reproduce this exact partition and this exact tree, which is what makes the
 *      three implementations agree bit for bit.
 *
 * Floating-point addition is not associative, so the four-way split is a real change
 * to the arithmetic relative to a sequential sum. Keeping the accumulators in double
 * bounds the difference far below fp32 output precision; making them float would not.
 */
static void matmul_f32_rows(float *y, const float *x, const float *W, int in, int o0, int o1)
{
#if defined(K3_AVX512)
    for (int o = o0; o < o1; o++) {
        const float *row = W + (size_t)o * in;
        __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps(), _mm512_setzero_ps() };
        int i = 0;
        for (; i + 63 < in; i += 64)
            for (int k = 0; k < 4; k++)
                a[k] = _mm512_fmadd_ps(_mm512_loadu_ps(row + i + 16 * k),
                                       _mm512_loadu_ps(x + i + 16 * k), a[k]);
        for (int k = 0; i < in; i += 16, k++) {
            const __mmask16 m = v512_tail(in - i);
            a[k] = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(m, row + i),
                                         _mm512_maskz_loadu_ps(m, x + i), a[k], m);
        }
        y[o] = v512_sum(a);
    }
    return;
#endif
    for (int o = o0; o < o1; o++) {
        const float *row = W + (size_t)o * in;
        /* Sixteen accumulators, EXPLICITLY fused products. fma() in double is the
         * same IEEE operation as _mm256_fmadd_pd per lane, so the scalar and vector
         * paths stay bit-identical while the dependent-add latency chain that made
         * one accumulator ~10x slower than the machine's floor disappears. The
         * reduction pairs lanes exactly the way the vector path's (v0+v1)+(v2+v3)
         * then cross-lane tree does; change one and you must change the other. */
        double a[16] = {0};
        int i = 0;
        for (; i + 15 < in; i += 16)
            for (int l = 0; l < 16; l++)
                a[l] = fma((double)row[i + l], (double)x[i + l], a[l]);
        double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
        double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
        double b2 = (a[2] + a[6]) + (a[10] + a[14]);
        double b3 = (a[3] + a[7]) + (a[11] + a[15]);
        double acc = (b0 + b1) + (b2 + b3);
        for (; i < in; i++) acc = fma((double)row[i], (double)x[i], acc);
        y[o] = (float)acc;
    }
}

void k3_matmul(float *y, const float *x, const float *W, int in, int out)
{
    K3_TEAM_IF(out > 64, int lo, hi; k3_split(out, &lo, &hi);
               matmul_f32_rows(y, x, W, in, lo, hi));
}

/* ------------------------------------------------------------- Gated MLA ---- */
/* MLA with an optional KV cache, which is what makes incremental decode possible.
 *
 * WHY MLA IS THE ONLY PIECE THAT NEEDS ONE
 *   KDA already carries everything it needs: k3_kda_layer updates its recurrent state
 *   and its ShortConv history in place, so a decode step just declines to clear them.
 *   The attn-res block stack is per token with no cross-token dependency. MLA is the
 *   exception, because softmax attention must see every previous key and value, and
 *   this function recomputes them all from x on every call. That is what makes decode
 *   O(T^2).
 *
 * WHAT IS CACHED, AND WHY NOT THE COMPRESSED LATENT
 *   The obvious saving is to cache the 576-float compressed latent and re-expand it
 *   through kv_b each step, which is 42x smaller. It is also far slower: kv_b is
 *   24576x512, so re-expanding every cached position costs an O(T) sweep of 12.6M-MAC
 *   matmuls per layer per token. The EXPANDED per-head keys and values are cached
 *   instead: 96 heads x 256 floats = 98,304 B per position per layer, and 2.37 MB per
 *   position across the 24 MLA layers. A 64-token generation is therefore 151 MB of
 *   KV cache, small enough not to affect the memory budget at any preset.
 *
 *   The rope slot is cached separately because it is SHARED across heads: 64 values per
 *   position, not per head. Folding it into the per-head block would waste 96x the space
 *   and, worse, invites treating it as per-head somewhere.
 *
 * kvc == NULL selects the self-contained path, which recomputes all keys and values
 * from x and caches nothing. Both paths must produce identical output; the op fixtures
 * gate the uncached path and tests/unit/k3_model.c gates them against each other.
 */
static void mla_team(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,
                     int T, float *scratch,
                     float *kvc, float *ropec, int cached, int cap)
{
    const int E  = c->hidden, H = c->n_heads;
    const int qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr;                       /* 192: the FULL head width      */
    const int kvw = c->kv_lora + qr;              /* 576: latent + shared rope slot */
    const int kvd = qn + vh;                      /* 256: cached width per head    */
    const float scale = 1.0f / sqrtf((float)qh);  /* :359, over qh not qn           */
    if (!kvc) cached = 0;
    const int last = cached + T - 1;              /* highest absolute position      */
    if (kvc && last >= cap)
        k3_fatal_bound("MLA KV cache position", (long)last, (long)cap - 1);

    /* Scratch layout. Every region below is DISJOINT and must stay so. Overlapping
     * any two of them can appear to work, aliasing the gate buffer onto q, say, is
     * safe only while H*vh < H*qh holds, but that is an accident of the released
     * dimensions, not an invariant, and it breaks silently the moment v_head grows.
     * Size the buffer with k3_mla_scratch_cached(); do not compute it by hand. */
    float *q    = scratch;                          /* [T][H][qh]     */
    float *ct   = q    + (size_t)T * H * qh;        /* [kvw] transient, one token */
    float *ql   = ct   + (size_t)kvw;               /* [q_lora]       */
    float *acc  = ql   + (size_t)c->q_lora;         /* [H][vh]        */
    float *gbuf = acc  + (size_t)H * vh;            /* [H][vh] gate   */
    float *sc   = gbuf + (size_t)H * vh;            /* [last+1] scores */
    /* Without a cache the keys/values live in scratch and cover only this call. */
    float *kvs  = sc   + (size_t)(last + 1);        /* [T][H][kvd]    */
    float *rps  = kvs  + (kvc ? 0 : (size_t)T * H * kvd);   /* [T][qr] */

    #define K3_KV_AT(p)   (kvc   ? kvc   + (size_t)(p) * H * kvd : kvs + (size_t)(p) * H * kvd)
    #define K3_ROPE_AT(p) (ropec ? ropec + (size_t)(p) * qr      : rps + (size_t)(p) * qr)

    /* Tensor parallel: rows of q_a/kv_a are split and gathered (their outputs are
     * normalised as a whole); heads [h0, h1) own q_b, kv_b, their KV slots, attention and
     * the gate; rows [e0, e1) of o_proj. */
    int h0, h1, a0, a1, b0, b1, e0, e1;
    k3_tp_part(H, &h0, &h1);
    k3_tp_part(c->q_lora, &a0, &a1);
    k3_tp_part(kvw, &b0, &b1);
    k3_tp_part(E, &e0, &e1);

    /* ---- per-token projections ---- */
    double tp = k3_prof_t0();
    for (int t = 0; t < T; t++) {
        const int p = cached + t;
        const float *xt = x + (size_t)t * E;
        mmw_rows(ql, xt, w->q_a, w->wdt, E, a0, a1);
        /* ONE projection emits the compressed latent AND the shared rope slot */
        mmw_rows(ct, xt, w->kv_a, w->wdt, E, b0, b1);
        const K3Seg sg[2] = { { ql, c->q_lora, 1, 0 }, { ct, kvw, 1, 0 } };
        k3_tp_gather_begin(sg, 2);
        /* decode: the gate reads only x, so it runs while the latents are in flight */
        if (T == 1 && w->g) mmw_rows(gbuf, xt, w->g, w->wdt, E, h0 * vh, h1 * vh);
        k3_tp_gather_end(sg, 2);

        /* both norms in three parts under one barrier; the norm covers the latent
         * only, never the rope slot */
        double *qp = rc_parts(rc_n(c->q_lora)), *kp = rc_parts(rc_n(c->kv_lora));
        rc_sq_team(qp, ql, c->q_lora);
        rc_sq_team(kp, ct, c->kv_lora);
        k3_sync();
        rms_scale(ql, ql, w->q_a_norm, c->q_lora, rms_inv(qp, c->q_lora, c->rms_eps));
        rms_scale(ct, ct, w->kv_a_norm, c->kv_lora, rms_inv(kp, c->kv_lora, c->rms_eps));
        if (k3_tid() == 0) memcpy(K3_ROPE_AT(p), ct + c->kv_lora, (size_t)qr * sizeof(float));
        k3_sync();
        mmw_rows(q + (size_t)t * H * qh, ql, w->q_b, w->wdt, c->q_lora, h0 * qh, h1 * qh);
        mmw_rows(K3_KV_AT(p), ct, w->kv_b, w->kv_b_wdt, c->kv_lora, h0 * kvd, h1 * kvd);
        k3_sync();                          /* ql/ct are rewritten by the next token */
    }
    k3_prof_add(K3P_MLA_PROJ, tp);

    /* ---- attention, per head, causal; heads split over the team ---- */
    for (int t = 0; t < T; t++) {
        const int p = cached + t;
        tp = k3_prof_t0();
        int hl, hh;
        k3_split(h1 - h0, &hl, &hh);
        float *scp = sc;
        if (hh > hl && k3_nth() > 1)
            scp = (float *)k3_scratch(K3S_MLA_SC, (size_t)(p + 1) * sizeof(float), "MLA scores");
        for (int h = h0 + hl; h < h0 + hh; h++) {
            const float *qt = q + ((size_t)t * H + h) * qh;
            float m = -INFINITY;
            for (int s = 0; s <= p; s++) {                 /* causal: s <= p */
                const float *ks = K3_KV_AT(s) + (size_t)h * kvd;
                const float *kr = K3_ROPE_AT(s);           /* shared slot */
                double d = 0.0;
                for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)ks[i];
                /* the rope slot is UNROTATED but still scored, and the SAME 64
                 * values serve every head. Dropping this term is the silent bug. */
                for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
                scp[s] = (float)d * scale;
                if (scp[s] > m) m = scp[s];
            }
            double z = 0.0;
            for (int s = 0; s <= p; s++) { scp[s] = expf(scp[s] - m); z += scp[s]; }

            float *o = acc + (size_t)h * vh;
            for (int j = 0; j < vh; j++) o[j] = 0.0f;
            for (int s = 0; s <= p; s++) {
                const float pr = (float)(scp[s] / z);
                const float *vs = K3_KV_AT(s) + (size_t)h * kvd + qn;
                for (int j = 0; j < vh; j++) o[j] += pr * vs[j];
            }
        }
        k3_prof_add(K3P_MLA_CORE, tp);

        /* ---- output gate then projection. Gate BEFORE o_proj, and no norm on it,
         * unlike KDA which norms first. :470-473 ---- */
        tp = k3_prof_t0();
        if (w->g) {
            if (T != 1) mmw_rows(gbuf, x + (size_t)t * E, w->g, w->wdt, E, h0 * vh, h1 * vh);
            k3_sync();
            int lo, hi;
            k3_split((h1 - h0) * vh, &lo, &hi);
            for (int i = h0 * vh + lo; i < h0 * vh + hi; i++)
                acc[i] *= 1.0f / (1.0f + expf(-gbuf[i]));
        }
        const K3Seg sa = { acc, H * vh, vh, 0 };
        k3_tp_gather(&sa, 1);
        mmw_rows(out + (size_t)t * E, acc, w->o, w->wdt, H * vh, e0, e1);
        const K3Seg so = { out + (size_t)t * E, E, 1, 0 };
        k3_tp_gather(&so, 1);
        k3_prof_add(K3P_MLA_OUT, tp);
    }
    #undef K3_KV_AT
    #undef K3_ROPE_AT
}

void k3_mla_cached(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,
                   int T, float *scratch,
                   float *kvc, float *ropec, int cached, int cap)
{
    K3_TEAM_IF(1, mla_team(out, x, w, c, T, scratch, kvc, ropec, cached, cap));
}

void k3_mla(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,
            int T, float *scratch)
{
    k3_mla_cached(out, x, w, c, T, scratch, NULL, NULL, 0, 0);
}

/* ---------------------------------------------------------------- router ---- */
/* Split in two so tensor parallelism can score its own experts, gather the scores, and
 * then select on every rank. Together they are exactly k3_router. */
static void router_scores_part(float *score, const float *x, const float *W, int hidden,
                               int e0, int r0, int r1);

static void router_scores(float *score, const float *x, const float *W, int hidden,
                          int e0, int e1)
{
    /* logits in float32 with no bias, then an independent sigmoid per expert. The
     * reference upcasts both operands explicitly; a double accumulator here matches
     * it and costs nothing at this width. */
    /* PARALLEL over experts, and bit-identical because of it.
     *
     * This is 896 dot products of length 7168 = 6.4M multiply-adds, per token, per MoE
     * layer, so 590M across the 92 of them -- and it ran on ONE core while every other
     * matmul in the engine was already threaded. It is pure arithmetic with no I/O to
     * hide behind, so it sat squarely on the critical path.
     *
     * Each iteration writes only its own score[e], and the ACCUMULATION ORDER INSIDE an
     * expert is untouched: thread t still sums i = 0..hidden-1 in sequence into its own
     * double. Splitting the outer loop therefore cannot change a single bit, which is
     * why this needs no tolerance and no re-gating. */
    K3_TEAM_IF(e1 - e0 > 64, int lo, hi; k3_split(e1 - e0, &lo, &hi);
               router_scores_part(score, x, W, hidden, e0, e0 + lo, e0 + hi));
}

/* this thread's experts [r0, r1) of the rank's [e0, e1) */
static void router_scores_part(float *score, const float *x, const float *W, int hidden,
                               int e0, int r0, int r1)
{
#if defined(K3_AVX512)
    /* the fp32 GEMV scheme, as a serial double chain here was latency bound */
    if (r1 > r0)
        matmul_f32_rows(score + r0, x, W + (size_t)(r0 - (k3_tp.local ? e0 : 0)) * hidden,
                        hidden, 0, r1 - r0);
    for (int e = r0; e < r1; e++) score[e] = 1.0f / (1.0f + expf(-score[e]));
    return;
#endif
    for (int e = r0; e < r1; e++) {
        const float *row = W + (size_t)(e - (k3_tp.local ? e0 : 0)) * hidden;
        double acc = 0.0;
        for (int i = 0; i < hidden; i++) acc += (double)row[i] * (double)x[i];
        score[e] = 1.0f / (1.0f + expf(-(float)acc));
    }
}

static void router_select(int *idx, float *w, const float *score, const float *bias,
                          int n_experts, int topk, int renorm, float routed_scale)
{
    /* Returning early here would leave idx[] and w[] untouched, and k3_moe forms
     * `w->w1 + idx[j]*I*L` from them one line later -- an arbitrary pointer built from
     * uninitialised stack. */
    float *choice = (float *)k3_scratch(K3S_ROUTER_CHOICE, (size_t)n_experts * sizeof(float),
                                        "router scores");
    for (int e = 0; e < n_experts; e++)
        choice[e] = score[e] + (bias ? bias[e] : 0.0f);   /* selection score only */

    /* top-k by repeated max. n_experts is 896 and topk is 16, so this is 14k
     * comparisons per token per layer: cheap next to an 18 MB expert read, and it
     * avoids a sort. Marking taken entries with -INFINITY keeps ties deterministic
     * in first-index order, matching a stable selection. */
    for (int j = 0; j < topk; j++) {
        int best = -1; float bv = -INFINITY;
        for (int e = 0; e < n_experts; e++)
            if (choice[e] > bv) { bv = choice[e]; best = e; }
        if (best < 0) { idx[j] = 0; w[j] = 0.0f; continue; }
        idx[j] = best;
        w[j]   = score[best];              /* UNBIASED score, not choice[best] */
        choice[best] = -INFINITY;
    }

    if (renorm && topk > 1) {
        double s = 0.0;
        for (int j = 0; j < topk; j++) s += (double)w[j];
        const float inv = (float)(1.0 / (s + 1e-20));
        for (int j = 0; j < topk; j++) w[j] *= inv;
    }
    for (int j = 0; j < topk; j++) w[j] *= routed_scale;
}

void k3_router(int *idx, float *w, const float *x, const float *W,
               const float *bias, int hidden, int n_experts, int topk,
               int renorm, float routed_scale)
{
    float *score = (float *)k3_scratch(K3S_ROUTER_SCORE, (size_t)n_experts * sizeof(float),
                                       "router scores");
    router_scores(score, x, W, hidden, 0, n_experts);
    router_select(idx, w, score, bias, n_experts, topk, renorm, routed_scale);
}

/* --------------------------------------------------------------- AttnRes ---- */
#define K3_AR_MAXSRC 64

/* Team form over source pointers v[0..nsrc). The score of source s is its fold dot
 * product scaled by its RMS inverse, (float)(inv_s * sum v*fold), both sums in the
 * canonical chunk order and all (source, chunk) pairs in one parallel pass. The mix is
 * split by chunk with the sources summed in order; with opart != NULL it also leaves the
 * chunk sums of squares of out there, part one of the norm that follows. No barrier at
 * the end: out and opart are complete only after the caller's next k3_sync. */
static void attn_res_v(float *out, const float *const *v, const float *fold,
                       int nsrc, int n, float eps, double *opart)
{
    if (nsrc > K3_AR_MAXSRC) k3_fatal_bound("AttnRes sources", nsrc, K3_AR_MAXSRC);
    const int nc = rc_n(n);
    double *part = rc_parts(2 * nsrc * nc);          /* [s][c] squares, then [s][c] dots */
    {
        int lo, hi;
        k3_split(nsrc * nc, &lo, &hi);
        for (int task = lo; task < hi; task++) {
            const int s = task / nc, c = task % nc;
            const int i0 = c * K3_RC, i1 = i0 + K3_RC < n ? i0 + K3_RC : n;
            const float *vs = v[s];
            double q = 0.0, d = 0.0;
            for (int i = i0; i < i1; i++) {
                q += (double)vs[i] * (double)vs[i];
                d += (double)vs[i] * (double)fold[i];
            }
            part[task] = q;
            part[nsrc * nc + task] = d;
        }
    }
    k3_sync();

    float p[K3_AR_MAXSRC];
    for (int s = 0; s < nsrc; s++) {
        const float inv = (float)(1.0 / sqrt(rc_total(part + s * nc, nc) / (double)n
                                             + (double)eps));
        /* key is the NORMALISED source; fold already carries norm.weight*proj.weight */
        p[s] = (float)((double)inv * rc_total(part + (nsrc + s) * nc, nc));
    }
    float m = p[0];
    for (int s = 1; s < nsrc; s++) if (p[s] > m) m = p[s];
    double z = 0.0;
    for (int s = 0; s < nsrc; s++) { p[s] = expf(p[s] - m); z += p[s]; }
    for (int s = 0; s < nsrc; s++) p[s] = (float)(p[s] / z);

    int lo, hi;
    k3_split(nc, &lo, &hi);
    for (int c = lo; c < hi; c++) {
        const int i0 = c * K3_RC, i1 = i0 + K3_RC < n ? i0 + K3_RC : n;
        for (int i = i0; i < i1; i++) {
            float a = 0.0f;
            for (int s = 0; s < nsrc; s++) a += p[s] * v[s][i];   /* the RAW source */
            out[i] = a;
        }
        if (opart) opart[c] = rc_sq(out, c, n);
    }
}

void k3_attn_res(float *out, const float *src, const float *fold,
                 int nsrc, int n, float eps)
{
    const float *v[K3_AR_MAXSRC];
    if (nsrc > K3_AR_MAXSRC) k3_fatal_bound("AttnRes sources", nsrc, K3_AR_MAXSRC);
    for (int s = 0; s < nsrc; s++) v[s] = src + (size_t)s * n;
    attn_res_v(out, v, fold, nsrc, n, eps, NULL);
    k3_sync();
}

/* Exact scratch requirement for k3_mla. Callers should use this rather than
 * duplicating the arithmetic; getting it wrong overruns silently. */
/* Scratch for the self-contained path: keys and values live here, so they scale with T.
 * `cap` is the highest position that will be attended over plus one; without a cache
 * that is just T. */
size_t k3_mla_scratch_cached(const K3Cfg *c, int T, int cap, int cached_mode)
{
    const int H = c->n_heads, qh = c->qk_nope + c->qk_rope, vh = c->v_head;
    const size_t kvd = (size_t)(c->qk_nope + vh);
    size_t n = (size_t)T * H * qh                      /* q            */
             + (size_t)(c->kv_lora + c->qk_rope)       /* ct transient */
             + (size_t)c->q_lora
             + (size_t)2 * H * vh                      /* acc, gbuf    */
             + (size_t)(cap > T ? cap : T);            /* scores       */
    if (!cached_mode) n += (size_t)T * H * kvd + (size_t)T * c->qk_rope;
    return n;
}

size_t k3_mla_scratch(const K3Cfg *c, int T)
{
    return k3_mla_scratch_cached(c, T, T, 0);
}

/* ------------------------------------------------------- Stable LatentMoE ---- */
/* Verified against modeling_kimi_linear.py:815-838. The ORDER is load bearing:
 *   1. route on the FULL hidden width, before any projection      :818
 *   2. down-project to the latent width                            :822
 *   3. run the selected experts IN LATENT SPACE and sum, weighted  :825
 *   4. RMSNorm the AGGREGATE, never per expert                     :831
 *   5. up-project back to hidden                                   :832
 *   6. add the shared expert computed on the ORIGINAL input,
 *      with NO routing weight and NO scaling                       :837
 *
 * Routed experts live at the latent width (latent -> moe_inter -> latent); the
 * shared expert is one wider MLP at full width with intermediate moe_inter*n_shared.
 *
 * Every scratch region below is DISJOINT and must stay so. Reusing the gate/up buffer
 * for the down-projection output is safe only while 2*moe_inter >= latent. That holds
 * for the released config and for the tiny one, but it is an accident of the
 * numbers rather than an invariant, and it is the same class of hazard documented at
 * the scratch layout in k3_mla_cached. Size with k3_moe_scratch().
 */
/* Scratch floats for moe_routed: gate|up, act and down outputs for every selected
 * expert, then (double-aligned) z and each act widened to double. */
static size_t moe_batch_floats(const K3Cfg *c)
{
    const size_t K = (size_t)c->topk, L = (size_t)c->latent, I = (size_t)c->moe_inter;
    return K * (3 * I + L) + 2 + 2 * (L + K * I);
}

/* The selected experts of ONE token, all of them in one parallel region per phase.
 * Calling k3_matmul_mxfp4 three times per expert opened 48 small parallel regions per
 * layer, each over 3072-3584 rows, and ran the experts one after another: measured
 * 58 GB/s against a 305 GB/s socket. Here the work is split over (expert, matrix, row
 * block) instead. q != NULL selects streamed MXFP4 experts; otherwise the resident fp32
 * bank w->w1/w3/w2 indexed by eidx.
 *
 * Tensor parallel: this rank computes rows [i0, i1) of every gate/up and rows [l0, l1)
 * of every down projection; act is gathered between the two (together with `extra`, the
 * shared expert's act, to save a collective) and accL after.
 *
 * BIT-IDENTICAL to the per-expert loop: every output row goes through the same row
 * kernel with the same input, SiTU is the same elementwise function, and accL is summed
 * per element in the original top-k order j = 0..nq-1. */
static void moe_routed(float *accL, const float *z, const K3MoeW *w, const K3ExpertQ *q,
                       const int *eidx, const float *wq, int nq, const K3Cfg *c,
                       float *buf, K3Seg extra)
{
    const int L = c->latent, I = c->moe_inter, G = K3_MXFP4_GROUP;
    const int sliced = w->src && w->src->sliced;
    int i0, i1, l0, l1;
    k3_tp_part(I, &i0, &i1);
    k3_tp_part(L, &l0, &l1);
    float  *gu  = buf;                                  /* [nq][2I] */
    float  *act = gu  + (size_t)nq * 2 * I;             /* [nq][I]  */
    float  *edn = act + (size_t)nq * I;                 /* [nq][L]  */
    const size_t used = (size_t)nq * (3 * (size_t)I + L);
#if defined(K3_AVX512)
    double *zd = NULL, *ad = NULL;                      /* the fp32 path reads z, act */
    (void)used;
#else
    double *zd  = (double *)(buf + used + (used & 1));  /* [L]      */
    double *ad  = zd + L;                               /* [nq][I]  */
#endif

    const int ni = i1 - i0, nl = l1 - l0;
    if (q) {
        mxfp4_check(L, G);
        mxfp4_check(I, G);
#if !defined(K3_AVX512)
        int lo, hi;
        k3_split(L, &lo, &hi);
        for (int i = lo; i < hi; i++) zd[i] = (double)z[i];
        k3_sync();
#endif
    }

    /* int8 activations for GGUF experts (K3_EXPERT_Q8): z once here, act after its gather */
    static int8_t *zq, *aq;
    static float  *zdx, *adx;
    const int nzb = L / K3_GQ_QK, nab = I / K3_GQ_QK;
    const int q8 = q && nq > 0 && k3_expert_q8 && q[0].qt1 != K3_EQ_MXFP4 &&
                   k3_gq_have_q8() && L % K3_GQ_QK == 0 && I % K3_GQ_QK == 0;
    if (q8) {
        if (k3_tid() == 0) {
            const size_t nb8 = (size_t)L + (size_t)nq * I;
            unsigned char *pool = (unsigned char *)k3_scratch(K3S_XQ8,
                nb8 + 64 + sizeof(float) * ((size_t)nzb + (size_t)nq * nab), "int8 activations");
            zq = (int8_t *)pool; aq = zq + L;
            zdx = (float *)(pool + ((nb8 + 63) & ~(size_t)63)); adx = zdx + nzb;
        }
        k3_sync();
        int lo, hi;
        k3_split(nzb, &lo, &hi);
        for (int b = lo; b < hi; b++)
            k3_gq_quant_x(zq + (size_t)b * K3_GQ_QK, zdx + b, z + (size_t)b * K3_GQ_QK, K3_GQ_QK);
        k3_sync();
    }

    /* gate|up split over (expert, row block) tasks, about one per thread, so each thread
     * streams two long contiguous runs rather than a few rows of every expert, and SiTU
     * on the same rows by the same thread: no barrier between them */
    {
        const int nb = ni > 0 ? (k3_nth() + nq - 1) / (nq > 0 ? nq : 1) : 0;
        int t0, t1;
        k3_split(nq * nb, &t0, &t1);
        for (int task = t0; task < t1; task++) {
            const int j = task / nb, b = task % nb;
            const int r0 = i0 + (int)((long)ni * b / nb), r1 = i0 + (int)((long)ni * (b + 1) / nb);
            if (r1 <= r0) continue;
            float *g = gu + (size_t)j * 2 * I;
            if (q) {
                eq_rows(g,     z, zd, q8 ? zq : NULL, zdx, q[j].p1, q[j].s1, q[j].qt1, L,
                        r0, r1, sliced ? i0 : 0, G, q[j].ilv);
                eq_rows(g + I, z, zd, q8 ? zq : NULL, zdx, q[j].p3, q[j].s3, q[j].qt3, L,
                        r0, r1, sliced ? i0 : 0, G, q[j].ilv);
            } else {
                matmul_f32_rows(g,     z, w->w1 + (size_t)eidx[j] * I * L, L, r0, r1);
                matmul_f32_rows(g + I, z, w->w3 + (size_t)eidx[j] * I * L, L, r0, r1);
            }
            situ_range(act + (size_t)j * I + r0, g + r0, g + I + r0, r1 - r0,
                       c->situ_b1, c->situ_b2);
        }
    }

    {
        K3Seg sg[K3_MAX_TOPK + 1];
        int ng = 0;
        for (int j = 0; j < nq; j++) {
            sg[ng].p = act + (size_t)j * I; sg[ng].n = I; sg[ng].unit = 1; sg[ng].exact = 0; ng++;
        }
        if (extra.p) sg[ng++] = extra;
        k3_tp_gather(sg, ng);
    }
    if (q8) {
        int lo, hi;
        k3_split(nq * nab, &lo, &hi);
        for (int b = lo; b < hi; b++) {
            const size_t off = (size_t)(b / nab) * I + (size_t)(b % nab) * K3_GQ_QK;
            k3_gq_quant_x(aq + off, adx + b, act + off, K3_GQ_QK);
        }
        k3_sync();
    }

#if !defined(K3_AVX512)
    if (q) {
        int lo, hi;
        k3_split(nq * I, &lo, &hi);
        for (int i = lo; i < hi; i++) ad[i] = (double)act[i];
        k3_sync();
    }
#endif

    /* down split over (expert, row block) tasks for long contiguous runs, as for
     * gate|up; the weighted sum then crosses threads, so it follows a barrier, per
     * element in top-k order */
    {
        const int nb = nl > 0 ? (k3_nth() + nq - 1) / (nq > 0 ? nq : 1) : 0;
        int t0, t1;
        k3_split(nq * nb, &t0, &t1);
        for (int task = t0; task < t1; task++) {
            const int j = task / nb, b = task % nb;
            const int r0 = l0 + (int)((long)nl * b / nb), r1 = l0 + (int)((long)nl * (b + 1) / nb);
            if (r1 <= r0) continue;
            if (q)
                eq_rows(edn + (size_t)j * L, act + (size_t)j * I, ad ? ad + (size_t)j * I : NULL,
                        q8 ? aq + (size_t)j * I : NULL, q8 ? adx + (size_t)j * nab : NULL,
                        q[j].p2, q[j].s2, q[j].qt2, I, r0, r1, sliced ? l0 : 0, G, q[j].ilv);
            else
                matmul_f32_rows(edn + (size_t)j * L, act + (size_t)j * I,
                                w->w2 + (size_t)eidx[j] * L * I, I, r0, r1);
        }
        k3_sync();
        int lo, hi;
        k3_split(nl, &lo, &hi);
        for (int i = l0 + lo; i < l0 + hi; i++) {
            float a = accL[i];
            for (int j = 0; j < nq; j++) a += wq[j] * edn[(size_t)j * L + i];
            accL[i] = a;
        }
    }
    const K3Seg sa = { accL, L, 1, 0 };
    k3_tp_gather_begin(&sa, 1);                /* the caller ends it, after independent work */
}

/* The token's selection, published by thread 0 to the team. */
static K3ExpertQ moe_q[K3_MAX_TOPK];
static float     moe_wq[K3_MAX_TOPK];
static int       moe_idx[K3_MAX_TOPK];
static int       moe_nq;
static float    *moe_score;                /* router scores when they do not fit scratch */

static void moe_team(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
                     int T, int *idx, float *wt, float *scratch)
{
    const int E = c->hidden, L = c->latent, I = c->moe_inter;
    const int SI = I * c->n_shared, NE = c->n_experts;

    float *z    = scratch;              /* [L]    latent input              */
    float *accL = z    + L;             /* [L]    weighted expert aggregate */
    float *rsc  = accL + L;             /* [3I+L] router scores when NE fits */
    float *sgu  = rsc  + 3 * I + L;     /* [2*SI] shared gate|up            */
    float *sact = sgu  + 2 * SI;        /* [SI]   shared after SiTU         */
    float *sdn  = sact + SI;            /* [E]    shared down-projection    */
    float *xbuf = sdn  + E;             /* moe_batch_floats                 */
    if ((size_t)(xbuf - scratch) & 1) xbuf++;
    float *score = rsc;
    if (NE > 3 * I + L) {
        if (k3_tid() == 0)
            moe_score = (float *)k3_scratch(K3S_MOE_SCORE, (size_t)NE * sizeof(float),
                                            "router scores");
        k3_sync();
        score = moe_score;
    }

    /* Tensor parallel: experts scored, latent rows of down, shared-expert intermediate
     * rows, and output rows of up and of the shared down projection. */
    int x0, x1, l0, l1, s0, s1, o0, o1;
    k3_tp_part(NE, &x0, &x1);
    k3_tp_part(L, &l0, &l1);
    k3_tp_part(SI, &s0, &s1);
    k3_tp_part(E, &o0, &o1);

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = out + (size_t)t * E;

        /* 1. route on the FULL width, before the down-projection, and 2. down-project
         * into the latent space. Both read xt only, so they share one gather, and it is
         * in flight while 6a runs: the shared expert's gate|up reads only xt as well. */
        double tp = k3_prof_t0();
        router_scores(score, xt, w->gate, E, x0, x1);
        k3_prof_add(K3P_ROUTER, tp);
        tp = k3_prof_t0();
        mmw_rows(z, xt, w->down, w->wdt, E, l0, l1);
        const K3Seg sg[2] = { { score, NE, 1, 1 }, { z, L, 1, 0 } };   /* scores pick experts: fp32 */
        k3_tp_gather_begin(sg, 2);
        k3_prof_add(K3P_MOE_DOWN, tp);

        /* 6a. shared expert gate|up on the ORIGINAL full-width input, and SiTU on the
         * same rows by the same thread. Its act is gathered with the routed experts'. */
        tp = k3_prof_t0();
        {
            int lo, hi;
            k3_split(s1 - s0, &lo, &hi);
            const int r0 = s0 + lo, r1 = s0 + hi;
            mmw_part(sgu,      xt, w->sh1, w->wdt, E, s0, r0, r1);
            mmw_part(sgu + SI, xt, w->sh3, w->wdt, E, s0, r0, r1);
            situ_range(sact + r0, sgu + r0, sgu + SI + r0, r1 - r0, c->situ_b1, c->situ_b2);
        }
        k3_prof_add(K3P_SHARED, tp);
        k3_tp_gather_end(sg, 2);

        /* Selection, and resolving experts through the source (which may do I/O and is
         * not thread safe), on thread 0; the team picks the result up after the barrier. */
        tp = k3_prof_t0();
        if (k3_tid() == 0) {
            router_select(idx, wt, score, w->bias, NE, c->topk, c->moe_renorm, c->routed_scale);
            int nk = c->topk;
            /* Draft cache-only routing: keep only the top-k experts already resident, and
             * renormalise their weights so the mixture still sums as intended. This makes
             * a draft token read ZERO new expert bytes. It is an approximation, which is
             * exactly what a draft is; the exact model verifies every proposed token. */
            if (w->cache_only && w->src && w->src->resident) {
                int m = 0; float wsum = 0.0f;
                for (int j = 0; j < c->topk; j++) {
                    if (w->src->resident(w->src, w->layer, idx[j], NULL)) {
                        idx[m] = idx[j]; wt[m] = wt[j]; wsum += wt[j]; m++;
                    }
                }
                nk = m;
                if (wsum > 0.0f) for (int j = 0; j < nk; j++) wt[j] /= wsum;
            }
            for (int i = 0; i < L; i++) accL[i] = 0.0f;
            /* Hand the WHOLE top-k to the source first, so its reads can overlap. Without
             * this the loop below misses, blocks on a 17.55 MB read, computes, misses
             * again: a queue depth of one against a drive that needs depth to reach its
             * rated bandwidth. getmany is optional and may be NULL, in which case nothing
             * changes and the loop reads them one at a time exactly as before. */
            if (!w->cache_only && w->src && w->src->getmany)
                w->src->getmany(w->src, w->layer, idx, nk);
            if (w->src) {
                /* Streamed: the experts stay MXFP4 and the matmuls read nibbles. Resolve
                 * the whole top-k first; the source keeps every pointer valid for the
                 * token. In cache-only mode every idx[j] is known resident, so resident()
                 * serves it with no disk read; otherwise get() may read it. */
                int nq = 0;
                for (int j = 0; j < nk; j++) {
                    int miss = w->cache_only
                        ? !w->src->resident(w->src, w->layer, idx[j], &moe_q[nq])
                        : (w->src->get(w->src, w->layer, idx[j], &moe_q[nq]) != 0);
                    if (miss) {
                        /* A cache-only draft filtered to resident experts already, so a
                         * miss here is a benign race at worst; skip it, since the draft is
                         * approximate by construction and the exact model verifies. On
                         * the exact path a miss is the unacceptable silent-corruption
                         * case: count it in k3_expert_drops so the caller fails the run
                         * (see docs/API.md). */
                        if (w->cache_only) continue;
                        k3_expert_drops++;
                        fprintf(stderr, "EXPERT DROP: layer %d expert %d failed to load; "
                                        "this token is CORRUPT\n", w->layer, idx[j]);
                        continue;
                    }
                    moe_wq[nq++] = wt[j];
                }
                moe_nq = nq;
            } else {
                for (int j = 0; j < nk; j++) { moe_idx[j] = idx[j]; moe_wq[j] = wt[j]; }
                moe_nq = nk;
            }
        }
        k3_prof_add(K3P_ROUTER, tp);
        k3_sync();

        /* 3. the selected experts, in latent space, weighted and summed */
        tp = k3_prof_t0();
        const K3Seg shared_act = { sact, SI, 1, 0 };
        moe_routed(accL, z, w, w->src ? moe_q : NULL, moe_idx, moe_wq, moe_nq, c, xbuf,
                   shared_act);
        k3_prof_add(K3P_EXPERTS, tp);

        /* 6b. the shared expert down-projection needs only the gathered act, so it runs
         * while the aggregate is in flight; rows split as for 5, so the add below is
         * thread-local. */
        tp = k3_prof_t0();
        int lo, hi;
        k3_split(o1 - o0, &lo, &hi);
        const int r0 = o0 + lo, r1 = o0 + hi;
        mmw_part(sdn, sact, w->sh2, w->wdt, SI, o0, r0, r1);
        k3_prof_add(K3P_SHARED, tp);
        const K3Seg sa = { accL, L, 1, 0 };
        k3_tp_gather_end(&sa, 1);

        /* 4. RMSNorm the AGGREGATE (not per expert), 5. up-project, and add 6b UNWEIGHTED.
         * The norm in three parts: chunk sums over the team, a barrier, then each thread
         * scales its own copy of the aggregate as the input of its up-projection rows. */
        tp = k3_prof_t0();
        const float *ain = accL;
        if (c->latent_norm) {
            double *ap = rc_parts(rc_n(L));
            rc_sq_team(ap, accL, L);
            k3_sync();
            if (r1 > r0) {
                const float inv = rms_inv(ap, L, c->rms_eps);
                float *an = (float *)k3_scratch(K3S_MOE_AN, (size_t)L * sizeof(float),
                                                "MoE aggregate");
                for (int i = 0; i < L; i++) an[i] = w->latent_norm[i] * accL[i] * inv;
                ain = an;
            }
        }
        if (r1 > r0) {
            mmw_part(ot, ain, w->up, w->wdt, L, o0, r0, r1);
            for (int i = r0; i < r1; i++) ot[i] += sdn[i];
        }
        const K3Seg so = { ot, E, 1, 0 };
        k3_tp_gather(&so, 1);
        k3_prof_add(K3P_MOE_UP, tp);
    }
}

void k3_moe(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
            int T, int *idx, float *wt, float *scratch)
{
    K3_TEAM_IF(1, moe_team(out, x, w, c, T, idx, wt, scratch));
}

size_t k3_moe_scratch(const K3Cfg *c)
{
    const int SI = c->moe_inter * c->n_shared;
    return (size_t)2 * c->latent          /* z, accL            */
         + (size_t)3 * c->moe_inter       /* gu (2*I) + act (I) */
         + (size_t)c->latent              /* edn                */
         + (size_t)3 * SI                 /* sgu (2*SI) + sact  */
         + (size_t)c->hidden              /* sdn                */
         + 1 + moe_batch_floats(c);       /* streamed expert batch, double-aligned */
}

/* Batched MoE for PREFILL over a chunk of T tokens, streamed experts only.
 *
 * k3_moe walks the top-k for each token independently, so across a T-token chunk it
 * fetches an expert once per token that routes to it. Under near-uniform routing that is
 * mostly waste: measured on the released trace, a 32-token chunk touches only ~2.7x fewer
 * unique experts than 16*32 draws, so reading each unique expert ONCE and reusing it for
 * every token in the chunk cuts prefill expert bytes ~3-4x. Prefill is where that matters,
 * because decode feeds one token at a time and has nothing to batch.
 *
 * Exactness is preserved to the last bit. Per token the arithmetic is identical to
 * k3_moe: the routed latent contributions are accumulated in the ORIGINAL top-k order
 * (j = 0..k-1) from a per-(token, slot) buffer, then normalised, up-projected and given
 * the shared expert exactly as before. Only the ORDER in which experts are fetched from
 * disk changes, and that touches no floating-point result.
 *
 * out/x are [T][E], idx/wt scratch are topk-wide (reused per token), scratch is one
 * k3_moe_scratch. This path requires w->src (streamed); the resident path stays on
 * k3_moe, which is what the oracle gates exercise. */
static void moe_prefill_chunk(float *out, const float *x, const K3MoeW *w,
                              const K3Cfg *c, int T, float *scratch);

/* Whether k3_moe_prefill takes the batched chunk path, which forks per kernel and so
 * must run outside a team. */
static int moe_batched(const K3MoeW *w, int T)
{
    /* K3_NO_BATCH_PREFILL forces the per-token path, so one binary can produce both the
     * batched and the reference token streams for a bit-identity A/B. */
    static int no_batch = -1;
    if (no_batch < 0) no_batch = getenv("K3_NO_BATCH_PREFILL") ? 1 : 0;
    /* cache_only renormalises per token over the resident subset, which the per-token
     * path already does; the draft's prompt prefill is one-time, so defer rather than
     * duplicate the renorm in the batch. */
    return !(!w->src || T <= 1 || no_batch || w->cache_only || k3_tp.size > 1);
}

void k3_moe_prefill(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
                    int T, int *idx, float *wt, float *scratch)
{
    if (!moe_batched(w, T)) {
        k3_moe(out, x, w, c, T, idx, wt, scratch);
        return;
    }
    /* Fixed sub-chunks bound the contribution buffer (14.7 MB at 64 tokens) no matter
     * how long the prompt is; a 32k prefill would otherwise want 7.3 GB of it. Most of
     * the dedup is already captured at this width: the unique-expert count grows far
     * slower than the request count under near-uniform routing. */
    const int CHUNK = 64;
    for (int t0 = 0; t0 < T; t0 += CHUNK) {
        const int n = (T - t0) < CHUNK ? (T - t0) : CHUNK;
        if (n == 1) { k3_moe(out + (size_t)t0 * c->hidden, x + (size_t)t0 * c->hidden,
                             w, c, 1, idx, wt, scratch); continue; }
        moe_prefill_chunk(out + (size_t)t0 * c->hidden, x + (size_t)t0 * c->hidden,
                          w, c, n, scratch);
    }
}

static void moe_prefill_chunk(float *out, const float *x, const K3MoeW *w,
                              const K3Cfg *c, int T, float *scratch)
{
    const int E = c->hidden, Ll = c->latent, I = c->moe_inter;
    const int SI = I * c->n_shared, K = c->topk;

    /* Per-token routing decisions and latent inputs, plus a contribution buffer holding
     * every routed expert's latent output for every token: [T][K][Ll]. At T=32, K=16,
     * Ll=3584 that is ~7.3 MB, trivial beside the tens of GB already reserved. All of it
     * is carved from one persistent slot. */
    #define K3_R64(b) (((b) + 63) & ~(size_t)63)
    const size_t b_con = K3_R64((size_t)T * K * Ll * sizeof(float));
    const size_t b_zz  = K3_R64((size_t)T * Ll * sizeof(float));
    const size_t b_tk  = K3_R64((size_t)T * K * sizeof(float));
    unsigned char *pool = (unsigned char *)k3_scratch(
        K3S_PREFILL, b_con + b_zz + 3 * b_tk + K3_R64((size_t)c->n_experts), "MoE prefill batch");
    float *contrib = (float *)pool;
    float *zz   = (float *)(pool + b_con);
    float *rwt  = (float *)(pool + b_con + b_zz);
    int   *ridx = (int *)  (pool + b_con + b_zz + b_tk);
    int   *uniq = (int *)  (pool + b_con + b_zz + 2 * b_tk);
    char  *seen = (char *) (pool + b_con + b_zz + 3 * b_tk);
    #undef K3_R64
    memset(seen, 0, (size_t)c->n_experts);

    /* 1. route every token and down-project it, and collect the batch's unique experts. */
    int nu = 0;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        int   *it = ridx + (size_t)t * K;
        float *wtt = rwt + (size_t)t * K;
        k3_router(it, wtt, xt, w->gate, w->bias, E, c->n_experts, K,
                  c->moe_renorm, c->routed_scale);
        k3_mmw(zz + (size_t)t * Ll, xt, w->down, w->wdt, E, Ll);
        for (int j = 0; j < K; j++) {
            const int e = it[j];
            if (e >= 0 && e < c->n_experts && !seen[e]) { seen[e] = 1; uniq[nu++] = e; }
        }
    }

    /* 2. expert-major: fetch each unique expert ONCE, apply it to every (token, slot)
     * that selected it. gu/act/edn are reused per (expert, token). */
    float *gu  = scratch;                 /* [2*I] */
    float *act = gu + 2 * I;              /* [I]   */
    float *edn = act + I;                 /* [Ll]  */
    if (w->src->getmany) w->src->getmany(w->src, w->layer, uniq, nu);
    for (int u = 0; u < nu; u++) {
        const int e = uniq[u];
        K3ExpertQ q;
        memset(&q, 0, sizeof q);
        if (w->src->get(w->src, w->layer, e, &q) != 0) {
            k3_expert_drops++;
            fprintf(stderr, "EXPERT DROP: layer %d expert %d failed to load; "
                            "this chunk is CORRUPT\n", w->layer, e);
            /* contrib is malloc'd, not calloc'd. The per-token path (k3_moe,
             * above) starts its accumulator at zero and simply never adds a
             * dropped expert's contribution, which is a correct zero. This
             * batched path instead fills contrib per (token, slot) as each
             * expert is processed, so skipping expert e here would leave
             * every slot it was going to fill holding whatever malloc handed
             * back, and step 3 below sums all of contrib unconditionally.
             * Zero those slots so a drop here contributes zero exactly like
             * the per-token path, instead of reading uninitialized memory
             * into the model's output. */
            for (int t = 0; t < T; t++) {
                const int *it = ridx + (size_t)t * K;
                for (int j = 0; j < K; j++)
                    if (it[j] == e)
                        memset(contrib + ((size_t)t * K + j) * Ll, 0,
                               (size_t)Ll * sizeof(float));
            }
            continue;
        }
        for (int t = 0; t < T; t++) {
            const int   *it = ridx + (size_t)t * K;
            const float *zt = zz  + (size_t)t * Ll;
            for (int j = 0; j < K; j++) {
                if (it[j] != e) continue;
                eq_matmul(gu,     zt, q.p1, q.s1, q.qt1, Ll, I, q.ilv);
                eq_matmul(gu + I, zt, q.p3, q.s3, q.qt3, Ll, I, q.ilv);
                k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                eq_matmul(edn, act, q.p2, q.s2, q.qt2, I, Ll, q.ilv);
                memcpy(contrib + ((size_t)t * K + j) * Ll, edn, (size_t)Ll * sizeof(float));
            }
        }
    }

    /* 3. per token, sum contributions in the ORIGINAL top-k order, then the tail of the
     * MoE exactly as k3_moe does it, so every float matches the per-token path. */
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = out + (size_t)t * E;
        const float *wtt = rwt + (size_t)t * K;
        /* Reuse this token's now-dead down-projection slot as the aggregate. */
        float *acc = zz + (size_t)t * Ll;
        for (int i = 0; i < Ll; i++) acc[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            const float wj = wtt[j];
            const float *cb = contrib + ((size_t)t * K + j) * Ll;
            for (int i = 0; i < Ll; i++) acc[i] += wj * cb[i];
        }
        if (c->latent_norm) k3_rmsnorm(acc, acc, w->latent_norm, Ll, c->rms_eps);
        k3_mmw(ot, acc, w->up, w->wdt, Ll, E);

        float *sgu  = gu;                 /* [2*SI] */
        float *sact = sgu + 2 * SI;       /* [SI]   */
        float *sdn  = sact + SI;          /* [E]    */
        k3_mmw(sgu,      xt, w->sh1, w->wdt, E, SI);
        k3_mmw(sgu + SI, xt, w->sh3, w->wdt, E, SI);
        k3_situ_glu(sact, sgu, SI, c->situ_b1, c->situ_b2);
        k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);
        for (int i = 0; i < E; i++) ot[i] += sdn[i];
    }
}

/* --------------------------------------------------------- KDA full layer ---- */
/* L2 normalisation over the last dimension. The reference uses the SUM of squares
 * with eps inside the rsqrt, NOT the mean: k3_ref.py l2norm(). Using the mean here
 * scales every q and k by sqrt(d_k) and quietly changes the attention temperature. */
static void l2norm_(float *v, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)v[i] * (double)v[i];
    const float inv = (float)(1.0 / sqrt(ss + (double)eps));
    for (int i = 0; i < n; i++) v[i] *= inv;
}

size_t k3_kda_scratch(const K3Cfg *c, int T)
{
    const size_t P = (size_t)c->kda_heads * c->kda_head_dim;
    return 3 * (size_t)T * P        /* q, k, v after conv            */
         + 2 * (size_t)T * P        /* z then alpha                  */
         + (size_t)T * c->kda_heads /* beta                          */
         + (size_t)T * P            /* recurrence output             */
         + 2 * P                    /* gate buffer and one work row  */
         + (size_t)c->kda_head_dim; /* f_a output                    */
}

static float *kda_fresh;                   /* zeroed state for the stateless form */

static void kda_layer_team(float *out, const float *x, const K3KdaW *w, const K3Cfg *c,
                           int T, float *state, float *scratch)
{
    const int E = c->hidden, H = c->kda_heads, D = c->kda_head_dim;
    const int P = H * D, K = c->conv_k, hist = K - 1;

    /* Tensor parallel: this rank owns heads [h0, h1), i.e. channels [c0, c1), and rows
     * [e0, e1) of the output projection. Every other head's slots are left untouched. */
    int h0, h1, e0, e1;
    k3_tp_part(H, &h0, &h1);
    k3_tp_part(E, &e0, &e1);
    const int c0 = h0 * D, c1 = h1 * D;

    float *q  = scratch;                 float *k  = q + (size_t)T * P;
    float *v  = k + (size_t)T * P;       float *z  = v + (size_t)T * P;
    float *al = z + (size_t)T * P;       float *bt = al + (size_t)T * P;
    float *o  = bt + (size_t)T * H;      float *gb = o + (size_t)T * P;
    float *wr = gb + P;                  float *fa = wr + P;

    /* 1. projections */
    double tp = k3_prof_t0();
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        if (w->qkv) {
            qkv_rows(q + (size_t)t * P, k + (size_t)t * P, v + (size_t)t * P, xt, w->qkv, E, c0, c1);
        } else {
            mmw_rows(q + (size_t)t * P, xt, w->q, w->wdt, E, c0, c1);
            mmw_rows(k + (size_t)t * P, xt, w->k, w->wdt, E, c0, c1);
            mmw_rows(v + (size_t)t * P, xt, w->v, w->wdt, E, c0, c1);
        }
        mmw_rows(bt + (size_t)t * H, xt, w->b, w->b_wdt, E, h0, h1);
        /* ONE shared low-rank pair feeds every head: [E->D] then [D->H*D]. The D-row f_a
         * is recomputed on every rank rather than gathered. */
        k3_mmw(fa, xt, w->f_a, w->wdt, E, D);
        k3_sync();
        mmw_rows(z + (size_t)t * P, fa, w->f_b, w->wdt, D, c0, c1);
        if (t + 1 < T) k3_sync();           /* fa is rewritten for the next token */
    }
    k3_prof_add(K3P_KDA_PROJ, tp);
    tp = k3_prof_t0();

    /* 2. ShortConv with fused SiLU, carrying state across calls, and 4/5 the decay chain:
     * both channel-local over the same team split as the q/k/v/f_b rows above, so every
     * thread finishes the rows it wrote itself, with no barrier (a GEMV epilogue) */
    float *cs = state ? state + (size_t)H * D * D : NULL;
    shortconv_range(q, q, w->q_conv, cs, c0, c1, P, K, T);
    shortconv_range(k, k, w->k_conv, cs ? cs + (size_t)P * hist : NULL, c0, c1, P, K, T);
    shortconv_range(v, v, w->v_conv, cs ? cs + (size_t)2 * P * hist : NULL, c0, c1, P, K, T);
    const int nh = h1 - h0;
    for (int t = 0; t < T; t++) {
        if (k3_tid() == 0)
            for (int h = h0; h < h1; h++) bt[(size_t)t * H + h] = sigmoidf_(bt[(size_t)t * H + h]);
        k3_kda_decay(z + (size_t)t * P + c0, al + (size_t)t * P + c0, z + (size_t)t * P + c0,
                     w->A_log + h0, w->dt_bias + c0, nh, D, c->gate_lb);
    }
    k3_sync();

    /* 3. L2Norm on q and k ONLY, per head (v is deliberately left alone), then q is
     * pre-scaled by d_k^-0.5 for 6; q is dead after the recurrence, so in place. */
    const float qscale = 1.0f / sqrtf((float)D);
    {
        int lo, hi;
        k3_split(T * nh, &lo, &hi);
        for (int th = lo; th < hi; th++) {
            const size_t off = (size_t)(th / nh) * P + (size_t)(h0 + th % nh) * D;
            l2norm_(q + off, D, 1e-6f);
            l2norm_(k + off, D, 1e-6f);
            for (int i = 0; i < D; i++) q[off + i] *= qscale;
        }
    }
    k3_sync();

    /* 6. recurrence, per head */
    float *S = state;
    if (!S) {
        /* Dereferenced at a computed offset immediately below; an unchecked NULL here
         * is a wild write, not a missing result. */
        if (k3_tid() == 0) {
            const size_t nb = (size_t)H * D * D * sizeof(float);
            kda_fresh = (float *)k3_scratch(K3S_KDA_STATE, nb, "KDA recurrent state");
            memset(kda_fresh, 0, nb);
        }
        k3_sync();
        S = kda_fresh;
    }
    /* Heads are independent, and within a head so are the value columns (see
     * kda_step_cols), so the work is split over (head, column block). Each task walks its
     * own t in order; per-element arithmetic is untouched, so the results are
     * bit-identical to the serial form at any thread count. 96 heads alone left a
     * 64-thread socket with two uneven waves. */
    const int CB = D <= K3_KDA_STEP_DV ? (D >= 64 ? 32 : D) : K3_KDA_STEP_DV;
    const int nbc = (D + CB - 1) / CB;
    K3_TEAM_IF(1, int lo, hi; k3_split(nh * nbc, &lo, &hi);
    for (int task = lo; task < hi; task++) {
        const int h = h0 + task / nbc;
        const int j0 = (task % nbc) * CB;
        const int j1 = (D - j0 < CB) ? D : j0 + CB;
        for (int t = 0; t < T; t++) {
            const size_t off = (size_t)t * P + (size_t)h * D;
            kda_step_cols(S + (size_t)h * D * D, o + off, q + off, k + off, v + off,
                          al + off, bt[(size_t)t * H + h], D, D, j0, j1);
        }
    });
    k3_sync();

    k3_prof_add(K3P_KDA_CORE, tp);
    tp = k3_prof_t0();

    /* 7/8/9. head-wise RMSNorm, THEN the gate, THEN the output projection */
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = o + (size_t)t * P;
        int lo, hi;
        k3_split(nh, &lo, &hi);
        for (int h = h0 + lo; h < h0 + hi; h++)
            rmsnorm_serial(ot + (size_t)h * D, ot + (size_t)h * D, w->o_norm, D, c->rms_eps);
        mmw_rows(gb, xt, w->g, w->wdt, E, c0, c1);
        k3_sync();
        k3_split(c1 - c0, &lo, &hi);
        for (int i = c0 + lo; i < c0 + hi; i++) ot[i] *= sigmoidf_(gb[i]);
        if (t + 1 < T) k3_sync();           /* gb is rewritten for the next token */
    }
    tp_gather_rows(o, T, P, D);
    for (int t = 0; t < T; t++)
        mmw_rows(out + (size_t)t * E, o + (size_t)t * P, w->o, w->wdt, P, e0, e1);
    tp_gather_rows(out, T, E, 1);
    k3_prof_add(K3P_KDA_OUT, tp);
}

void k3_kda_layer(float *out, const float *x, const K3KdaW *w, const K3Cfg *c,
                  int T, float *state, float *scratch)
{
    K3_TEAM_IF(1, kda_layer_team(out, x, w, c, T, state, scratch));
}

/* ----------------------------------------------------------- decoder layer ---- */
size_t k3_layer_scratch(const K3Cfg *c, int T)
{
    size_t a = k3_mla_scratch(c, T);
    size_t b = k3_kda_scratch(c, T);
    size_t m = k3_moe_scratch(c);
    size_t sub = a > b ? a : b;
    if (m > sub) sub = m;
    /* prefix_sum, tmp, fold vectors, one attn_res source stack, plus the sub-block */
    return (size_t)3 * T * c->hidden
         + (size_t)2 * c->hidden
         + (size_t)(c->n_layers / c->attn_res_block + 2) * c->hidden
         + (size_t)2 * c->dense_inter
         + sub;
}

/* The incremental form. Everything except MLA already carries its own state:
 *   - k3_kda_layer updates the recurrent matrix and the ShortConv history in place, so
 *     a decode step simply does not clear them.
 *   - The attn-res block stack is built per token from that token's own hidden states,
 *     with no dependency on earlier tokens, so T=1 needs nothing carried.
 * Only softmax attention has to see every earlier position, which is why the KV cache
 * exists and why it is threaded through here rather than hidden inside k3_mla.
 *
 * kvc == NULL gives exactly the behaviour k3_decoder_layer has always had. */
enum { K3L_PRE = 1, K3L_MLP = 2, K3L_POST = 4, K3L_ALL = 7 };

/* The layer as a team body: PRE is the aggregation, attention and the norms up to the
 * MLP input, MLP the MoE or dense block, POST the final residual. */
static void layer_team(float *h, float *block_residual, int *n_blocks,
                       const K3LayerW *w, const K3Cfg *c, int layer_idx,
                       int T, float *state, float *scratch,
                       float *kvc, float *ropec, int cached, int cap, int part)
{
    const int E = c->hidden;
    const int maxb = c->n_layers / c->attn_res_block + 2;

    float *pref   = scratch;                    /* [T][E] the running residual   */
    float *tmp    = pref + (size_t)T * E;       /* [T][E] module output          */
    float *hin    = tmp  + (size_t)T * E;       /* [T][E] normalised layer input */
    float *foldA  = hin  + (size_t)T * E;       /* [E] attention aggregator      */
    float *foldM  = foldA + E;                  /* [E] mlp aggregator            */
    /* [maxb][E] after foldM is the old source stack, still reserved by k3_layer_scratch */
    float *dgu    = foldM + E + (size_t)maxb * E; /* [2*dense_inter]             */
    float *sub    = dgu + (size_t)2 * c->dense_inter;
    int lo, hi;
    k3_split(E, &lo, &hi);                      /* this thread's elements of a vector */

    if (part & K3L_PRE) {
        int nb = *n_blocks;                     /* read by all before thread 0 bumps it */
        /* The norm gain and the scoring projection collapse to ONE vector. Folding them
         * here costs 2*hidden multiplies per layer; a real engine folds at load time. */
        for (int i = lo; i < hi; i++) {
            foldA[i] = w->attn_res_norm[i] * w->attn_res_proj[i];
            foldM[i] = w->mlp_res_norm[i]  * w->mlp_res_proj[i];
        }

        double tp = k3_prof_t0();
        for (int t = 0; t < T; t++)
            memcpy(pref + (size_t)t * E + lo, h + (size_t)t * E + lo,
                   (size_t)(hi - lo) * sizeof(float));
        int have_prefix = 1;                    /* mirrors "prefix_sum is not None" */
        k3_sync();

        const float *v[K3_AR_MAXSRC];
        if (nb + 1 > K3_AR_MAXSRC) k3_fatal_bound("AttnRes sources", nb + 1, K3_AR_MAXSRC);
        const int boundary = layer_idx % c->attn_res_block == 0;
        for (int t = 0; t < T; t++) {
            float *ht = h + (size_t)t * E;
            double *hp = rc_parts(rc_n(E));
            /* aggregation before attention, only when snapshots already exist; its mix
             * leaves the chunk sums the input norm needs */
            if (nb > 0) {
                for (int b = 0; b < nb; b++) v[b] = block_residual + ((size_t)t * maxb + b) * E;
                v[nb] = pref + (size_t)t * E;
                attn_res_v(ht, v, foldA, nb + 1, E, c->rms_eps, hp);
            } else {
                rc_sq_team(hp, ht, E);
            }
            /* block boundary: snapshot the running residual, then CLEAR it */
            if (boundary)
                memcpy(block_residual + ((size_t)t * maxb + nb) * E + lo,
                       pref + (size_t)t * E + lo, (size_t)(hi - lo) * sizeof(float));
            k3_sync();
            rms_scale(hin + (size_t)t * E, ht, w->in_norm, E, rms_inv(hp, E, c->rms_eps));
        }
        if (boundary) {
            nb++;
            if (k3_tid() == 0) *n_blocks = nb;
            have_prefix = 0;
        }
        k3_prof_add(K3P_ATTNRES, tp);
        k3_sync();

        /* attention */
        if (w->kda) kda_layer_team(tmp, hin, w->kda, c, T, state, sub);
        else        mla_team(tmp, hin, w->mla, c, T, sub, kvc, ropec, cached, cap);

        tp = k3_prof_t0();
        for (int t = 0; t < T; t++) {
            float *pt = pref + (size_t)t * E;
            const float *mt = tmp + (size_t)t * E;
            if (have_prefix) for (int i = lo; i < hi; i++) pt[i] += mt[i];
            else             memcpy(pt + lo, mt + lo, (size_t)(hi - lo) * sizeof(float));
        }
        k3_sync();

        /* aggregation before the MLP, whose mix leaves the chunk sums for the post norm.
         * NO emptiness guard in the reference. */
        for (int t = 0; t < T; t++) {
            float *ht = h + (size_t)t * E;
            double *hp = rc_parts(rc_n(E));
            for (int b = 0; b < nb; b++) v[b] = block_residual + ((size_t)t * maxb + b) * E;
            v[nb] = pref + (size_t)t * E;
            attn_res_v(ht, v, foldM, nb + 1, E, c->rms_eps, hp);
            k3_sync();
            rms_scale(hin + (size_t)t * E, ht, w->post_norm, E, rms_inv(hp, E, c->rms_eps));
        }
        k3_prof_add(K3P_ATTNRES, tp);
        k3_sync();
    }

    if (part & K3L_MLP) {
        if (w->moe) {
            int   idx[K3_MAX_TOPK]; float wt[K3_MAX_TOPK];
            /* Prefill batches (T > 1, streamed source) fetch each unique expert once for
             * the whole chunk; decode (T == 1) and the resident path fall straight
             * through to k3_moe inside, byte-identical. */
            k3_moe_prefill(tmp, hin, w->moe, c, T, idx, wt, sub);
        } else {
            double tp = k3_prof_t0();
            const int DI = c->dense_inter;
            int d0, d1, e0, e1;
            k3_tp_part(DI, &d0, &d1);
            k3_tp_part(E, &e0, &e1);
            for (int t = 0; t < T; t++) {
                int dl, dh;
                k3_split(d1 - d0, &dl, &dh);
                const int r0 = d0 + dl, r1 = d0 + dh;
                mmw_part(dgu, hin + (size_t)t * E, w->dense_gate, w->wdt, E, d0, r0, r1);
                mmw_part(dgu + DI, hin + (size_t)t * E, w->dense_up, w->wdt, E, d0, r0, r1);
                situ_range(sub + r0, dgu + r0, dgu + DI + r0, r1 - r0, c->situ_b1, c->situ_b2);
                const K3Seg sa = { sub, DI, 1, 0 };
                k3_tp_gather(&sa, 1);
                mmw_rows(tmp + (size_t)t * E, sub, w->dense_down, w->wdt, DI, e0, e1);
                if (t + 1 < T) k3_sync();       /* sub is rewritten for the next token */
            }
            tp_gather_rows(tmp, T, E, 1);
            k3_prof_add(K3P_DENSE, tp);
        }
    }

    if (part & K3L_POST) {
        double tp = k3_prof_t0();
        for (int t = 0; t < T; t++) {
            float *pt = pref + (size_t)t * E, *ht = h + (size_t)t * E;
            const float *mt = tmp + (size_t)t * E;
            for (int i = lo; i < hi; i++) { pt[i] += mt[i]; ht[i] = pt[i]; }
        }
        k3_prof_add(K3P_ATTNRES, tp);
    }
}

/* One parallel region per layer rather than one per kernel: a fork costs ~12 us on a
 * 64-core socket and a decode step used to open ~17 per layer; a barrier costs ~4. The
 * batched prefill MoE forks per kernel itself, so that case runs between two regions. */
void k3_decoder_layer_inc(float *h, float *block_residual, int *n_blocks,
                          const K3LayerW *w, const K3Cfg *c, int layer_idx,
                          int T, float *state, float *scratch,
                          float *kvc, float *ropec, int cached, int cap)
{
    if (k3_in_team() || !(w->moe && moe_batched(w->moe, T))) {
        K3_TEAM_IF(1, layer_team(h, block_residual, n_blocks, w, c, layer_idx, T, state,
                                 scratch, kvc, ropec, cached, cap, K3L_ALL));
        return;
    }
    K3_TEAM_IF(1, layer_team(h, block_residual, n_blocks, w, c, layer_idx, T, state,
                             scratch, kvc, ropec, cached, cap, K3L_PRE));
    layer_team(h, block_residual, n_blocks, w, c, layer_idx, T, state, scratch,
               kvc, ropec, cached, cap, K3L_MLP);
    K3_TEAM_IF(1, layer_team(h, block_residual, n_blocks, w, c, layer_idx, T, state,
                             scratch, kvc, ropec, cached, cap, K3L_POST));
}

void k3_decoder_layer(float *h, float *block_residual, int *n_blocks,
                      const K3LayerW *w, const K3Cfg *c, int layer_idx,
                      int T, float *state, float *scratch)
{
    k3_decoder_layer_inc(h, block_residual, n_blocks, w, c, layer_idx, T, state,
                         scratch, NULL, NULL, 0, 0);
}

/* ---------------------------------------------------------------- MXFP4 ---- */
/* OCP MX E2M1: index by the 4-bit code; bit 3 is the sign. */
static const float K3_E2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

/* y[out] = W[out][in] . x[in], with W stored as bf16 and widened on read.
 *
 * WHY THIS EXISTS
 *   The checkpoint ships the dense trunk as bf16. Holding it as fp32 would double the
 *   resident set and double the weight traffic per token. These kernels are bandwidth
 *   bound, so halving the bytes read makes them faster, not slower, the same reason
 *   the MXFP4 path beats dequantise-then-multiply.
 *
 * WHY IT LOSES NOTHING
 *   bf16 is what the checkpoint already contains. Widening bf16 to fp32 is a pure left
 *   shift by 16 bits with no rounding, so multiplying from bf16 storage computes with
 *   exactly the values an fp32 copy would have supplied. The only difference from
 *   k3_matmul is WHERE the widening happens, not what is widened.
 *
 * The accumulator layout mirrors k3_matmul deliberately, four partial sums in double,
 * reduced in the same order, so the two kernels agree to the bit on identical input.
 *
 * THE AVX2 PATH IS BIT-IDENTICAL TO THE SCALAR PATH, not merely close. A __m256d holds
 * exactly four doubles, and loading four consecutive elements per iteration places
 * element i in lane i%4: the same partition as the scalar accumulators, with the same
 * sequential order within each lane. Reducing with (a0+a1)+(a2+a3) then reproduces the
 * scalar result exactly. Two details carry that guarantee:
 *
 *   - MUL THEN ADD, never _mm256_fmadd_pd. The build sets -ffp-contract=off, so the
 *     scalar code rounds the product and the sum separately while an FMA rounds once.
 *     Here the product happens to be exact (a bf16 widens exactly, and float x float
 *     needs 48 mantissa bits, which fits double's 53), so the two would agree anyway
 *     but that is a proof about the inputs. Mul-then-add is a proof about the code.
 *   - The scalar tail loop is reused verbatim for the in % 4 remainder.
 */
#if defined(K3_DPBF16)
/* bf16 activations: x rounded once per call into this thread's scratch, then
 * vdpbf16ps (a bf16 pair per fp32 lane) with 32-element chunk c in accumulator c % 4
 * and the v512_sum tree, so every row is summed in one fixed order. */
static inline __m256i v512_f2bf(__m512 v)
{
    const __m512i u = _mm512_castps_si512(v);
    const __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF),
                          _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1))));
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16));
}

static const uint16_t *xbf16(const float *x, int n)
{
    uint16_t *b = (uint16_t *)k3_scratch(K3S_XBF, (size_t)n * 2 + 64, "bf16 activations");
    int i = 0;
    for (; i + 15 < n; i += 16)
        _mm256_storeu_si256((__m256i *)(b + i), v512_f2bf(_mm512_loadu_ps(x + i)));
    for (; i < n; i++) b[i] = k3_f2bf(x[i]);
    return b;
}

static inline __mmask32 v512_tail32(int n) { return n >= 32 ? 0xFFFFFFFFu : (1u << n) - 1; }

#define K3_DP(a, w, x) _mm512_dpbf16_ps((a), (__m512bh)(w), (__m512bh)(x))

static void dot_bf16_rows(float *y, const uint16_t *xb, const uint16_t *W, int in,
                          int o0, int o1)
{
    for (int o = o0; o < o1; o++) {
        const uint16_t *row = W + (size_t)o * in;
        __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps(), _mm512_setzero_ps() };
        int i = 0;
        for (; i + 127 < in; i += 128)
            for (int k = 0; k < 4; k++)
                a[k] = K3_DP(a[k], _mm512_loadu_si512(row + i + 32 * k),
                             _mm512_loadu_si512(xb + i + 32 * k));
        for (int k = 0; i < in; i += 32, k++) {
            const __mmask32 m = v512_tail32(in - i);
            a[k] = K3_DP(a[k], _mm512_maskz_loadu_epi16(m, row + i),
                         _mm512_maskz_loadu_epi16(m, xb + i));
        }
        y[o] = v512_sum(a);
    }
}
#endif

static void matmul_bf16_rows(float *y, const float *x, const uint16_t *W, int in,
                             int o0, int o1)
{
#if defined(K3_DPBF16)
    if (k3_act_bf16 & K3_BF16_MM) {
        if (o1 > o0) dot_bf16_rows(y, xbf16(x, in), W, in, o0, o1);
        return;
    }
#endif
    for (int o = o0; o < o1; o++) {
        const uint16_t *row = W + (size_t)o * in;
        int i = 0;
        double acc;
#if defined(K3_AVX512)
        {
            /* fp32 in the shared K3_AVX512 scheme, so it equals k3_matmul on the
             * widened values to the bit; bf16 widens to fp32 exactly by a 16-bit shift */
            #define K3_BF16X16(p, m) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32( \
                                         _mm256_maskz_loadu_epi16((m), (p))), 16))
            __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                            _mm512_setzero_ps(), _mm512_setzero_ps() };
            for (; i + 63 < in; i += 64)
                for (int k = 0; k < 4; k++)
                    a[k] = _mm512_fmadd_ps(K3_BF16X16(row + i + 16 * k, 0xFFFF),
                                           _mm512_loadu_ps(x + i + 16 * k), a[k]);
            for (int k = 0; i < in; i += 16, k++) {
                const __mmask16 m = v512_tail(in - i);
                a[k] = _mm512_mask3_fmadd_ps(K3_BF16X16(row + i, m),
                                             _mm512_maskz_loadu_ps(m, x + i), a[k], m);
            }
            #undef K3_BF16X16
            y[o] = v512_sum(a);
            continue;
        }
#elif defined(__AVX2__)
        {
            /* Four vector accumulators, fused. _mm256_fmadd_pd per lane is the same
             * IEEE operation as scalar fma() in double, and the reduction below is
             * lane-for-lane the tree k3_matmul's sixteen scalar accumulators use, so
             * the two kernels remain BITWISE identical (test_ops asserts it). The old
             * one-accumulator mul+add form serialized on add latency at 4 elements
             * per ~4 cycles; this runs the memory-bound side of the roof instead. */
            __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
            __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
            for (; i + 15 < in; i += 16) {
                const __m128i h0 = _mm_loadl_epi64((const __m128i *)(row + i));
                const __m128i h1 = _mm_loadl_epi64((const __m128i *)(row + i + 4));
                const __m128i h2 = _mm_loadl_epi64((const __m128i *)(row + i + 8));
                const __m128i h3 = _mm_loadl_epi64((const __m128i *)(row + i + 12));
                v0 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h0), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i)), v0);
                v1 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h1), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4)), v1);
                v2 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h2), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i + 8)), v2);
                v3 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h3), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i + 12)), v3);
            }
            /* (v0+v1)+(v2+v3) lanewise, then the same cross-lane pairing as scalar */
            const __m256d vt = _mm256_add_pd(_mm256_add_pd(v0, v1),
                                             _mm256_add_pd(v2, v3));
            double a[4];
            _mm256_storeu_pd(a, vt);
            acc = (a[0] + a[1]) + (a[2] + a[3]);
        }
#elif defined(__ARM_NEON) && defined(__aarch64__)
        {
            /* Eight 2-lane double accumulators: wk holds the scalar path's
             * {a[2k], a[2k+1]}, so element i lands in accumulator i%16 exactly as in
             * the scalar and AVX2 forms, and vfmaq_f64 per lane is the same IEEE fma()
             * in double. bf16 -> f32 is the usual 16-bit left shift; vshll_n_u16
             * widens and shifts in one instruction. */
            float64x2_t w0 = vdupq_n_f64(0.0), w1 = vdupq_n_f64(0.0);
            float64x2_t w2 = vdupq_n_f64(0.0), w3 = vdupq_n_f64(0.0);
            float64x2_t w4 = vdupq_n_f64(0.0), w5 = vdupq_n_f64(0.0);
            float64x2_t w6 = vdupq_n_f64(0.0), w7 = vdupq_n_f64(0.0);
            for (; i + 15 < in; i += 16) {
                const uint16x8_t h0 = vld1q_u16(row + i);
                const uint16x8_t h1 = vld1q_u16(row + i + 8);
                const float32x4_t f0 =
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16));
                const float32x4_t f1 =
                    vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h0), 16));
                const float32x4_t f2 =
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
                const float32x4_t f3 =
                    vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
                const float32x4_t x0 = vld1q_f32(x + i);
                const float32x4_t x1 = vld1q_f32(x + i + 4);
                const float32x4_t x2 = vld1q_f32(x + i + 8);
                const float32x4_t x3 = vld1q_f32(x + i + 12);
                w0 = vfmaq_f64(w0, vcvt_f64_f32(vget_low_f32(f0)),
                                   vcvt_f64_f32(vget_low_f32(x0)));
                w1 = vfmaq_f64(w1, vcvt_high_f64_f32(f0), vcvt_high_f64_f32(x0));
                w2 = vfmaq_f64(w2, vcvt_f64_f32(vget_low_f32(f1)),
                                   vcvt_f64_f32(vget_low_f32(x1)));
                w3 = vfmaq_f64(w3, vcvt_high_f64_f32(f1), vcvt_high_f64_f32(x1));
                w4 = vfmaq_f64(w4, vcvt_f64_f32(vget_low_f32(f2)),
                                   vcvt_f64_f32(vget_low_f32(x2)));
                w5 = vfmaq_f64(w5, vcvt_high_f64_f32(f2), vcvt_high_f64_f32(x2));
                w6 = vfmaq_f64(w6, vcvt_f64_f32(vget_low_f32(f3)),
                                   vcvt_f64_f32(vget_low_f32(x3)));
                w7 = vfmaq_f64(w7, vcvt_high_f64_f32(f3), vcvt_high_f64_f32(x3));
            }
            /* (a[l]+a[4+l])+(a[8+l]+a[12+l]) lanewise -- t0 = {b0,b1}, t1 = {b2,b3} --
             * then (b0+b1)+(b2+b3): the scalar reduction tree exactly. */
            const float64x2_t t0 = vaddq_f64(vaddq_f64(w0, w2), vaddq_f64(w4, w6));
            const float64x2_t t1 = vaddq_f64(vaddq_f64(w1, w3), vaddq_f64(w5, w7));
            acc = vaddvq_f64(t0) + vaddvq_f64(t1);
        }
#else
        {
            double a[16] = {0};
            for (; i + 15 < in; i += 16)
                for (int l = 0; l < 16; l++)
                    a[l] = fma((double)k3_bf16f(row[i + l]), (double)x[i + l], a[l]);
            double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
            double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
            double b2 = (a[2] + a[6]) + (a[10] + a[14]);
            double b3 = (a[3] + a[7]) + (a[11] + a[15]);
            acc = (b0 + b1) + (b2 + b3);
        }
#endif
        for (; i < in; i++) acc = fma((double)k3_bf16f(row[i]), (double)x[i], acc);
        y[o] = (float)acc;
    }
}

void k3_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out)
{
    K3_TEAM_IF(out > 64, int lo, hi; k3_split(out, &lo, &hi);
               matmul_bf16_rows(y, x, W, in, lo, hi));
}

void k3_matmul_q80(float *y, const float *x, const void *W, int in, int out)
{
    K3_TEAM_IF(out > 64, int lo, hi; k3_split(out, &lo, &hi);
               k3_q80_rows(y, x, W, in, lo, hi));
}

/* Per-row int8 matmul for the draft model: each row is [f32 scale][int8 * in]. The int8
 * weights are widened to float, dotted with the fp32 activation, and the row's scale is
 * applied once at the end. Unlike the trunk kernels this carries NO cross-path
 * determinism contract (K3_WI8 is draft-only, and the exact model decides every emitted
 * token), so it accumulates in float with fused products and the natural AVX2 reduction,
 * which is what makes it fast. */
static void matmul_q8_rows(float *y, const float *x, const void *W, int in, int o0, int o1)
{
    const unsigned char *base = (const unsigned char *)W;
    const size_t rowb = (size_t)4 + (size_t)in;
    for (int o = o0; o < o1; o++) {
        const unsigned char *row = base + (size_t)o * rowb;
        float scale;
        memcpy(&scale, row, 4);
        const int8_t *w = (const int8_t *)(row + 4);
        int i = 0;
        float acc;
#if defined(__AVX2__)
        {
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            for (; i + 15 < in; i += 16) {
                const __m128i b0 = _mm_loadl_epi64((const __m128i *)(w + i));
                const __m128i b1 = _mm_loadl_epi64((const __m128i *)(w + i + 8));
                v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)),
                                     _mm256_loadu_ps(x + i), v0);
                v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)),
                                     _mm256_loadu_ps(x + i + 8), v1);
            }
            __m256 vs = _mm256_add_ps(v0, v1);
            __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs),
                                   _mm256_extractf128_ps(vs, 1));
            lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
            lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
            acc = _mm_cvtss_f32(lo);
        }
#elif defined(__ARM_NEON) && defined(__aarch64__)
        {
            /* Draft-only kernel, no determinism contract: fused float accumulation
             * and the natural NEON reduction, same as the AVX2 form's spirit. */
            float32x4_t v0 = vdupq_n_f32(0.0f), v1 = vdupq_n_f32(0.0f);
            float32x4_t v2 = vdupq_n_f32(0.0f), v3 = vdupq_n_f32(0.0f);
            for (; i + 15 < in; i += 16) {
                const int8x16_t b = vld1q_s8(w + i);
                const int16x8_t s0 = vmovl_s8(vget_low_s8(b));
                const int16x8_t s1 = vmovl_s8(vget_high_s8(b));
                v0 = vfmaq_f32(v0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))),
                               vld1q_f32(x + i));
                v1 = vfmaq_f32(v1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))),
                               vld1q_f32(x + i + 4));
                v2 = vfmaq_f32(v2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))),
                               vld1q_f32(x + i + 8));
                v3 = vfmaq_f32(v3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))),
                               vld1q_f32(x + i + 12));
            }
            acc = vaddvq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
        }
#else
        {
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (; i + 3 < in; i += 4) {
                a0 += (float)w[i]     * x[i];
                a1 += (float)w[i + 1] * x[i + 1];
                a2 += (float)w[i + 2] * x[i + 2];
                a3 += (float)w[i + 3] * x[i + 3];
            }
            acc = (a0 + a1) + (a2 + a3);
        }
#endif
        for (; i < in; i++) acc += (float)w[i] * x[i];
        y[o] = acc * scale;
    }
}

void k3_matmul_q8(float *y, const float *x, const void *W, int in, int out)
{
    K3_TEAM_IF(out > 64, int lo, hi; k3_split(out, &lo, &hi);
               matmul_q8_rows(y, x, W, in, lo, hi));
}

/* A whole BYTE to its two E2M1 values, so the inner loop does one 8-byte load instead
 * of masking, shifting and two separate lookups. 2 KB, built once, shared by all
 * threads after initialisation.
 *
 * SCALAR PATH ONLY. The AVX2 path decodes nibbles with permutevar8x32 in register and
 * never touches this table, so building it there would be 2 KB of cold cache and an
 * unused-symbol warning. See k3_matmul_mxfp4. */
#if !defined(__AVX2__)
static float K3_E2M1_PAIR[256][2];
static int   k3_pair_ready = 0;

static void k3_pair_init(void)
{
    for (int b = 0; b < 256; b++) {
        K3_E2M1_PAIR[b][0] = K3_E2M1[b & 0x0F];   /* low nibble  = EVEN element */
        K3_E2M1_PAIR[b][1] = K3_E2M1[b >> 4];     /* high nibble = ODD element  */
    }
    k3_pair_ready = 1;
}
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
/* Every E2M1 value's f32 bit pattern has zero low 16 bits, so a 4-bit code expands to
 * (B3[code] << 24) | (B2[code] << 16). Two 16-entry byte tables therefore cover the
 * whole lookup, and vqtbl1q_u8 resolves 16 codes per instruction -- this is what lets
 * the NEON path expand a group in registers instead of through the wf[] buffer. The
 * expanded bit patterns are identical to K3_E2M1_PAIR's floats, so using them changes
 * no arithmetic. */
static const uint8_t K3_E2M1_B2[16] = {
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0
};
static const uint8_t K3_E2M1_B3[16] = {
    0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
    0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0
};
#endif

/* E8M0 byte to its power of two. 255 is NaN by spec and maps to zero. Precomputed
 * because ldexpf in the group loop is a function call the compiler will not inline
 * into a vectorised body. */
static float K3_E8M0[256];
static int   k3_e8m0_ready = 0;
#if defined(K3_AVX512)
/* Every E2M1 code times every E8M0 scale, formed as k3_mxfp4_dequant forms it:
 * [scale][code], 16 KB. A NaN scale gives signed zeros, as dequantised. */
static float K3_MXTAB[256][16];
#define K3_MXPF   512       /* packed bytes prefetched ahead; measured 0..4096 */
#define K3_MXPF_S 8         /* scale rows prefetched ahead */
#endif
#if defined(K3_DPBF16)
/* The same products as bf16, exact (3 mantissa bits), each code twice so vpermw may
 * see the next nibble's low bit as index bit 4: [scale][32], 16 KB. */
static uint16_t K3_MXTAB16[256][32];
#endif

static void k3_e8m0_init(void)
{
    for (int b = 0; b < 256; b++) K3_E8M0[b] = (b == 255) ? 0.0f : ldexpf(1.0f, b - 127);
#if defined(K3_AVX512)
    for (int b = 0; b < 256; b++)
        for (int code = 0; code < 16; code++)
            K3_MXTAB[b][code] = K3_E2M1[code] * K3_E8M0[b];
#endif
#if defined(K3_DPBF16)
    for (int b = 0; b < 256; b++)
        for (int j = 0; j < 32; j++) K3_MXTAB16[b][j] = k3_f2bf(K3_E2M1[j & 15] * K3_E8M0[b]);
#endif
    k3_e8m0_ready = 1;
}

/* y[rows] = W[rows][in] . x[in], with W read straight out of packed MXFP4 and never
 * materialised as floats. This is not an optimisation; it is what makes streaming
 * experts possible at all.
 *
 * One routed expert is 33,030,144 parameters. Dequantised to fp32 that is 132 MB, so
 * the 1,472 experts a single token touches would be 194 GB of materialised weights. As
 * packed nibbles the same expert is 17.55 MB. A matrix-vector product is memory bound,
 * so reading 7.5x fewer bytes makes this kernel FASTER than dequantising first.
 *
 * The loop is structured around the 32-element group because the scale is constant
 * within one: it factors out of the inner sum and is applied once per group instead of
 * once per element. At group 32 each group is exactly 16 packed bytes.
 *
 * PRECONDITIONS:
 *   - group <= 64. The expanded group goes into a fixed wf[64] stack buffer.
 *     Checked below; violating it aborts with a FATAL message rather than
 *     overflowing the buffer.
 *   - `in` is even. The packed row stride is in/2 bytes, two elements per byte.
 *     Checked below for the same reason.
 *   - `packed` is rows x (in/2) bytes; `scales` is rows x ceil(in/group) bytes.
 *     Not checked; the caller owns these buffer sizes.
 *   - A scale byte of 255 is NaN by the OCP MX spec and zeroes its whole group.
 *
 * ACCURACY CONTRACT. This kernel is deliberately NOT bit-identical to
 * dequantise-then-k3_matmul, and no caller should assume it is. On AVX2 with a
 * group that is a multiple of 16 (K3 uses 32) it runs the FLAT ROW PATH below: the
 * E8M0 power-of-two scale is folded into each E2M1 weight exactly, and the whole
 * row is summed by sixteen independent double accumulator lanes (four __m256d).
 * Otherwise it sums each
 * group of 32 and applies that group's scale before accumulating. Both orderings
 * differ from dequantise-then-matmul's single accumulator set.
 *
 * The difference is bounded and tiny. Every individual product is EXACT in double, an
 * E2M1 value carries 3 mantissa bits and x carries 24, so the product needs 27 of the
 * 53 available, so only the additions round, and reassociating exact terms moves the
 * result by roughly 1 ULP of double, order 1e-16 relative. The required agreement is
 * 1e-6 against dequantise-then-matmul on real checkpoint weights, gated by
 * tests/unit/test_expert.c. The margin is nine orders of magnitude.
 */
static void mxfp4_check(int in, int group)
{
    if (in & 1) {
        fprintf(stderr,
                "k3: FATAL, k3_matmul_mxfp4 called with in=%d, which is odd.\n"
                "    Packed rows are in/2 bytes, two elements per byte; an odd `in`\n"
                "    truncates that stride below what the trailing group's odd\n"
                "    remainder reads, a heap read past the caller's buffer instead\n"
                "    of failing loudly.\n",
                in);
        abort();
    }
    if (group > 64) {
        fprintf(stderr,
                "k3: FATAL, k3_matmul_mxfp4 called with group=%d, which exceeds 64.\n"
                "    Each group is expanded into a fixed wf[64] stack buffer before\n"
                "    the dot product; a larger group overflows it instead of failing\n"
                "    loudly.\n",
                group);
        abort();
    }
#if !defined(__AVX2__)
    if (!k3_pair_ready)  k3_pair_init();
#endif
    if (!k3_e8m0_ready)  k3_e8m0_init();
}

/* Rows [r0, r1) of y = W x. xd is x widened to double, or NULL. packed/scales start at
 * row rbase (0 for a whole matrix, the slice start for a tensor-parallel slice); y is
 * indexed by absolute row. Every row is computed exactly the same way whoever calls
 * this and however the rows are split. */
static void mxfp4_rows(float *y, const float *x, const double *xd,
                       const unsigned char *packed, const unsigned char *scales,
                       int in, int r0, int r1, int rbase, int group, int ilv);

static void matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                         const unsigned char *scales, int in, int rows, int group, int ilv)
{
    mxfp4_check(in, group);

    /* WIDEN x ONCE, NOT ONCE PER ROW. The accumulators are double, so every row used to
     * re-run the same `in` float-to-double conversions -- 3072 rows x 3584 elements is
     * 11 M conversions per call to produce 3584 distinct values. x does not depend on r,
     * so it is hoisted here and the row loop reads doubles directly.
     *
     * BIT-IDENTICAL: float to double is exact (24 mantissa bits into 53), so the widened
     * copy holds precisely what _mm256_cvtps_pd produced in place.
     *
     * Read-only, one persistent copy per thread: at the K3 shapes it is 28 KB, which
     * stays in L2 while the packed weights stream past it. The fp32 AVX-512 path reads
     * x itself. */
#if defined(K3_AVX512)
    K3_TEAM_IF(rows > 64, int lo, hi; k3_split(rows, &lo, &hi);
               mxfp4_rows(y, x, NULL, packed, scales, in, lo, hi, 0, group, ilv));
#else
    K3_TEAM_IF(rows > 64,
        double *const xd = (double *)k3_scratch(K3S_MXFP4_XD, (size_t)in * sizeof(double),
                                                "MXFP4 widened input");
        for (int i = 0; i < in; i++) xd[i] = (double)x[i];
        int lo, hi;
        k3_split(rows, &lo, &hi);
        mxfp4_rows(y, x, xd, packed, scales, in, lo, hi, 0, group, ilv));
#endif
}

void k3_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                     const unsigned char *scales, int in, int rows, int group)
{
    matmul_mxfp4(y, x, packed, scales, in, rows, group, K3_MX_CKPT);
}

void k3_matmul_mxfp4_ilv(float *y, const float *x, const unsigned char *packed,
                         const unsigned char *scales, int in, int rows, int group,
                         int layout)
{
    matmul_mxfp4(y, x, packed, scales, in, rows, group, layout);
}

int k3_mxfp4_interleave(unsigned char *packed, int rows, int in, int group, int layout)
{
    if ((in & 1) || (group & (group - 1))) return 0;
#if defined(K3_AVX512)
    if (layout == K3_MX_F32 && (group & 15) == 0) {
        const size_t pcols = (size_t)in / 2;
        for (int r = 0; r < rows; r++)
            for (int b = 0; b + 127 < in; b += 128) {
                unsigned char *p = packed + r * pcols + b / 2, t[64] = { 0 };
                for (int e = 0; e < 128; e++) {
                    const int code = (p[e >> 1] >> ((e & 1) * 4)) & 15, s = e >> 4, l = e & 15;
                    t[4 * l + (s >> 1)] |= (unsigned char)(code << ((s & 1) * 4));
                }
                memcpy(p, t, 64);
            }
        return 1;
    }
#endif
#if defined(K3_DPBF16)
    if (layout == K3_MX_BF16 && (group & 31) == 0) {
        const size_t pcols = (size_t)in / 2;
        for (int r = 0; r < rows; r++)
            for (int b = 0; b + 127 < in; b += 128) {
                unsigned char *p = packed + r * pcols + b / 2, t[64] = { 0 };
                for (int e = 0; e < 128; e++) {
                    const int code = (p[e >> 1] >> ((e & 1) * 4)) & 15, s = e >> 5, m = e & 31;
                    t[2 * m + (s >> 1)] |= (unsigned char)(code << ((s & 1) * 4));
                }
                memcpy(p, t, 64);
            }
        return 1;
    }
#endif
    (void)packed; (void)rows; (void)layout;
    return 0;
}

#if defined(K3_DPBF16)
/* 32 codes from checkpoint order starting at element i (n <= 32 of them) as bf16 */
static inline __m512i mx_chunk_ckpt(const unsigned char *pr, const unsigned char *sr,
                                    int i, int n, int group)
{
    uint16_t t[32] = { 0 };
    for (int e = 0; e < n; e++) {
        const int j = i + e;
        t[e] = K3_MXTAB16[sr[j / group]][(pr[j >> 1] >> ((j & 1) * 4)) & 15];
    }
    return _mm512_loadu_si512(t);
}

/* MXFP4 x bf16 activations. K3_MX_BF16 blocks: one 64-byte load is 128 codes, then per
 * 32 one shift, one vpermw into the group's table and one vdpbf16ps. */
static void mxfp4_rows_bf16(float *y, const uint16_t *xb, const unsigned char *packed,
                            const unsigned char *scales, int in, int r0, int r1,
                            int rbase, int group, int ilv)
{
    const int pcols = in / 2, ngrp = (in + group - 1) / group;
    const int gsh = __builtin_ctz((unsigned)group);
    for (int r = r0; r < r1; r++) {
        const unsigned char *pr = packed + (size_t)(r - rbase) * pcols;
        const unsigned char *sr = scales + (size_t)(r - rbase) * ngrp;
        __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps(), _mm512_setzero_ps() };
        int i = 0;
        if (ilv == K3_MX_BF16) {
            _mm_prefetch((const char *)(sr + K3_MXPF_S * ngrp), _MM_HINT_T0);
            _mm_prefetch((const char *)(sr + K3_MXPF_S * ngrp + ngrp - 1), _MM_HINT_T0);
            for (; i + 127 < in; i += 128) {
                const __m512i w = _mm512_loadu_si512((const void *)(pr + (i >> 1)));
                _mm_prefetch((const char *)(pr + (i >> 1)) + K3_MXPF, _MM_HINT_T0);
                for (int s = 0; s < 4; s++) {
                    const int j = i + 32 * s;
                    const __m512i v = _mm512_permutexvar_epi16(_mm512_srli_epi16(w, 4 * s),
                                          _mm512_loadu_si512(K3_MXTAB16[sr[j >> gsh]]));
                    a[s] = K3_DP(a[s], v, _mm512_loadu_si512(xb + j));
                }
            }
        }
        for (; i < in; i += 32) {
            const int n = in - i < 32 ? in - i : 32;
            a[(i >> 5) & 3] = K3_DP(a[(i >> 5) & 3], mx_chunk_ckpt(pr, sr, i, n, group),
                                    _mm512_maskz_loadu_epi16(v512_tail32(n), xb + i));
        }
        y[r] = v512_sum(a);
    }
}
#endif

static void mxfp4_rows(float *y, const float *x, const double *xd,
                       const unsigned char *packed, const unsigned char *scales,
                       int in, int r0, int r1, int rbase, int group, int ilv)
{
    const int bfm = (k3_act_bf16 & K3_BF16_MX) != 0;
    if (ilv != K3_MX_CKPT && ilv != (bfm ? K3_MX_BF16 : K3_MX_F32)) {
        fprintf(stderr, "k3: FATAL, MXFP4 rows in layout %d, which the %s mode does not "
                        "decode\n", ilv, bfm ? "bf16" : "fp32");
        abort();
    }
#if defined(K3_DPBF16)
    if (bfm) {
        if (r1 > r0)
            mxfp4_rows_bf16(y, xbf16(x, in), packed, scales, in, r0, r1, rbase, group, ilv);
        return;
    }
#endif
    const int pcols = in / 2;                     /* two elements per byte */
    const int ngrp  = (in + group - 1) / group;
    const int gbyte = group / 2;

#if defined(K3_AVX512)
    /* fp32 in the shared K3_AVX512 scheme, so it equals dequantise-then-k3_matmul to
     * the bit. 8 packed bytes become 16 dword codes in element order: each byte is
     * doubled, widened, and the odd lanes shifted down a nibble (vpermps reads only the
     * low 4 bits), then vpermps looks them up in the group scale's table. */
    if ((group & 15) == 0 && (group & (group - 1)) == 0) {
        const __m512i nsh = _mm512_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4);
        const int gsh = __builtin_ctz((unsigned)group);
        #define K3_MX16(b, s) _mm512_permutexvar_ps(_mm512_srlv_epi32( \
                _mm512_cvtepu8_epi32(_mm_unpacklo_epi8((b), (b))), nsh), \
                _mm512_loadu_ps(K3_MXTAB[(s)]))
        for (int r = r0; r < r1; r++) {
            const unsigned char *pr = packed + (size_t)(r - rbase) * pcols;
            const unsigned char *sr = scales + (size_t)(r - rbase) * ngrp;
            __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                            _mm512_setzero_ps(), _mm512_setzero_ps() };
            int i = 0;
            /* interleaved: 128 codes per 64-byte load, one shift + one vpermps per 16
             * (vpermps ignores the upper index bits), no widening; chunk s of a block
             * still lands in a[s & 3], so the sums are the checkpoint-order ones */
            if (ilv == K3_MX_F32) {
                /* expert streams are short, so do not wait for the hardware prefetcher */
                _mm_prefetch((const char *)(sr + K3_MXPF_S * ngrp), _MM_HINT_T0);
                _mm_prefetch((const char *)(sr + K3_MXPF_S * ngrp + ngrp - 1), _MM_HINT_T0);
                for (; i + 127 < in; i += 128) {
                    const __m512i w = _mm512_loadu_si512((const void *)(pr + (i >> 1)));
                    _mm_prefetch((const char *)(pr + (i >> 1)) + K3_MXPF, _MM_HINT_T0);
                    for (int s = 0; s < 8; s++) {
                        const int j = i + 16 * s;
                        a[s & 3] = _mm512_fmadd_ps(
                            _mm512_permutexvar_ps(_mm512_srli_epi32(w, 4 * s),
                                                  _mm512_loadu_ps(K3_MXTAB[sr[j >> gsh]])),
                            _mm512_loadu_ps(x + j), a[s & 3]);
                    }
                }
            }
            for (; i + 63 < in; i += 64)
                for (int k = 0; k < 4; k++) {
                    const int j = i + 16 * k;
                    const __m128i b = _mm_loadl_epi64((const __m128i *)(pr + (j >> 1)));
                    a[k] = _mm512_fmadd_ps(K3_MX16(b, sr[j >> gsh]), _mm512_loadu_ps(x + j), a[k]);
                }
            for (int k = 0; i < in; i += 16, k++) {
                const __mmask16 m = v512_tail(in - i);
                const __m128i b = _mm_maskz_loadu_epi8(v512_tail((in - i + 1) >> 1), pr + (i >> 1));
                a[k] = _mm512_mask3_fmadd_ps(K3_MX16(b, sr[i >> gsh]),
                                             _mm512_maskz_loadu_ps(m, x + i), a[k], m);
            }
            y[r] = v512_sum(a);
        }
        #undef K3_MX16
        return;
    }
#endif
    if (ilv) {
        fprintf(stderr, "k3: FATAL, interleaved MXFP4 rows without their kernel\n");
        abort();
    }

    for (int r = r0; r < r1; r++) {
        const unsigned char *pr = packed + (size_t)(r - rbase) * pcols;
        const unsigned char *sr = scales + (size_t)(r - rbase) * ngrp;
        double acc = 0.0;

#if defined(__AVX2__)
        if (xd && (group & 15) == 0) {
            /* FLAT ROW PATH. Every E8M0 scale is a power of two (K3_E8M0[sb] =
             * 2^(sb-127), 0 for the NaN byte 255), so folding it into the E2M1 weight
             * before the FMA is EXACT in fp32: a 3-bit mantissa shifted by an exponent
             * does not round, and the only overflow case (sb >= 251 times weight 6)
             * produces inf exactly as the dequantised reference does. That folds the
             * per-group scale step out of the accumulation, so the whole row is one
             * flat vectorised dot product: four independent double accumulator chains
             * v0..v3, a single horizontal reduction at the end, and none of the
             * per-group scalar reduction, the a[4] store-forwarding, or the serial
             * `acc` chain that the grouped path below pays once per group.
             *
             * The accumulator count is a deliberate trade-off. An earlier version
             * ran eight chains over 32-element blocks, but eight accumulators plus
             * the decode temporaries exceed the 16 ymm registers and the compiler
             * spills one accumulator to the stack every block - reintroducing the
             * store-forwarding this path exists to avoid. Four chains over 16-element
             * chunks fit without a spill, and the loop is decode-bound on the shuffle
             * port (permutevar8x32, cvtepu8_epi32 and cvtps_pd all issue there), not
             * FMA-latency-bound, so the shorter chain depth is not what limits it.
             *
             * Lane k of v0..v3 holds elements == k (mod 16), and the final tree is
             * ((v0+v2)+(v1+v3)) lane-wise and then (a0+a1)+(a2+a3), the same shape as
             * the scalar reduction, so every lane holds a sum of the same element
             * classes in the same order as the dequantised reference and the error
             * stays a few ulps of double, far inside the 1e-6 gate.
             *
             * Requires `group` to be a multiple of 16 so no 16-element chunk straddles
             * a scale boundary (K3 uses group 32). Anything else takes the grouped path
             * below, which handles arbitrary group <= 64. */
            const __m256  LUT  = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f,
                                                2.0f, 3.0f, 4.0f, 6.0f);
            const __m128i m0f  = _mm_set1_epi8(0x0F);
            const __m256i m07  = _mm256_set1_epi32(7);
            const __m256i m08  = _mm256_set1_epi32(8);
            __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
            __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
            int i = 0, g = 0, c = 0;
            const int cpg = group >> 4;           /* 16-element chunks per group */

            for (; i + 15 < in; i += 16) {
                const unsigned char sb = sr[g];
                if (sb == 255) goto flat_skip;
                {
                    const __m256 LUTs = _mm256_mul_ps(
                        LUT, _mm256_set1_ps(K3_E8M0[sb]));
                    const __m128i b = _mm_loadl_epi64(
                        (const __m128i *)(pr + (i >> 1)));
                    const __m128i u = _mm_unpacklo_epi8(
                        _mm_and_si128(b, m0f),
                        _mm_and_si128(_mm_srli_epi16(b, 4), m0f));
                    const __m256i c0 = _mm256_cvtepu8_epi32(u);
                    const __m256i c1 = _mm256_cvtepu8_epi32(_mm_srli_si128(u, 8));
                    const __m256  w0 = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUTs, _mm256_and_si256(c0, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c0, m08), 28)));
                    const __m256  w1 = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUTs, _mm256_and_si256(c1, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c1, m08), 28)));
                    v0 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_castps256_ps128(w0)),
                        _mm256_loadu_pd(xd + i), v0);
                    v1 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_extractf128_ps(w0, 1)),
                        _mm256_loadu_pd(xd + i + 4), v1);
                    v2 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_castps256_ps128(w1)),
                        _mm256_loadu_pd(xd + i + 8), v2);
                    v3 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_extractf128_ps(w1, 1)),
                        _mm256_loadu_pd(xd + i + 12), v3);
                }
            flat_skip:
                if (++c == cpg) { c = 0; g++; }
            }
            {
                /* Horizontal reduction without touching memory, same tree as the
                 * grouped path: (a0+a1)+(a2+a3). */
                const __m256d q = _mm256_add_pd(
                    _mm256_add_pd(v0, v2), _mm256_add_pd(v1, v3));
                const __m128d t = _mm_add_pd(
                    _mm256_castpd256_pd128(q), _mm256_extractf128_pd(q, 1));
                acc = _mm_cvtsd_f64(_mm_add_sd(t, _mm_unpackhi_pd(t, t)));
            }
            /* Scalar tail, at most 15 elements, all inside one group (i is 16-aligned
             * and group is a multiple of 16, so a tail this short cannot cross a scale
             * boundary). Scale folded the same way as the vector path. */
            for (; i < in; i++) {
                const unsigned char by = pr[i >> 1];
                const unsigned char nib = (i & 1) ? (by >> 4) : (by & 0x0F);
                acc = fma((double)(K3_E2M1[nib] * K3_E8M0[sr[i / group]]), xd[i], acc);
            }
            y[r] = (float)acc;
            continue;
        }
#endif
        for (int g = 0; g < ngrp; g++) {
            const unsigned char sb = sr[g];
            if (sb == 255) continue;              /* NaN scale: contribute nothing */
            const unsigned char *pb = pr + (size_t)g * gbyte;
            const float *xg = x + (size_t)g * group;

            int n = in - g * group;
            if (n > group) n = group;

            /* One pointer for both paths, so the hot loop carries no test. The fallback
             * branch is per group, not per element, and only runs when the hoist above
             * could not allocate. */
            double xlocal[64];                    /* group <= 64, same bound as wf */
            const double *xdg;
            if (xd) {
                xdg = xd + (size_t)g * group;
            } else {
                for (int j = 0; j < n; j++) xlocal[j] = (double)xg[j];
                xdg = xlocal;
            }

            double sub;
#if defined(__ARM_NEON) && defined(__aarch64__)
            if (n == 32) {
                /* Full-group fast path: expand all 32 nibbles in registers and feed
                 * the dot product directly, no wf[] round-trip. vzip1/vzip2 on the
                 * (low nibble, high nibble) pair restores element order -- low nibble
                 * is the EVEN element -- then vqtbl1q_u8 resolves 16 codes at a time
                 * against the two byte tables and vzip+vshll assembles the f32 bit
                 * patterns. The values, the element order, and the u0..u3 accumulator
                 * partition are exactly the wf path's, so this stays bit-identical. */
                const uint8x16_t pkv = vld1q_u8(pb);
                const uint8x16_t nlo = vandq_u8(pkv, vdupq_n_u8(0x0F));
                const uint8x16_t nhi = vshrq_n_u8(pkv, 4);
                const uint8x16_t t2  = vld1q_u8(K3_E2M1_B2);
                const uint8x16_t t3  = vld1q_u8(K3_E2M1_B3);
                float64x2_t u0 = vdupq_n_f64(0.0), u1 = vdupq_n_f64(0.0);
                float64x2_t u2 = vdupq_n_f64(0.0), u3 = vdupq_n_f64(0.0);
                for (int k = 0; k < 2; k++) {
                    const uint8x16_t c = k ? vzip2q_u8(nlo, nhi)  /* codes 16..31 */
                                           : vzip1q_u8(nlo, nhi); /* codes  0..15 */
                    const uint8x16_t b2 = vqtbl1q_u8(t2, c);
                    const uint8x16_t b3 = vqtbl1q_u8(t3, c);
                    /* byte pairs (b2,b3) as u16 = b3<<8 | b2; shift 16 more for f32 */
                    const uint16x8_t h0 = vreinterpretq_u16_u8(vzip1q_u8(b2, b3));
                    const uint16x8_t h1 = vreinterpretq_u16_u8(vzip2q_u8(b2, b3));
                    const float32x4_t w0v =
                        vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16));
                    const float32x4_t w1v =
                        vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h0), 16));
                    const float32x4_t w2v =
                        vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
                    const float32x4_t w3v =
                        vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
                    const float32x4_t x0v = vld1q_f32(xg + 16 * k);
                    const float32x4_t x1v = vld1q_f32(xg + 16 * k + 4);
                    const float32x4_t x2v = vld1q_f32(xg + 16 * k + 8);
                    const float32x4_t x3v = vld1q_f32(xg + 16 * k + 12);
                    u0 = vfmaq_f64(u0, vcvt_f64_f32(vget_low_f32(w0v)),
                                       vcvt_f64_f32(vget_low_f32(x0v)));
                    u1 = vfmaq_f64(u1, vcvt_high_f64_f32(w0v), vcvt_high_f64_f32(x0v));
                    u2 = vfmaq_f64(u2, vcvt_f64_f32(vget_low_f32(w1v)),
                                       vcvt_f64_f32(vget_low_f32(x1v)));
                    u3 = vfmaq_f64(u3, vcvt_high_f64_f32(w1v), vcvt_high_f64_f32(x1v));
                    u0 = vfmaq_f64(u0, vcvt_f64_f32(vget_low_f32(w2v)),
                                       vcvt_f64_f32(vget_low_f32(x2v)));
                    u1 = vfmaq_f64(u1, vcvt_high_f64_f32(w2v), vcvt_high_f64_f32(x2v));
                    u2 = vfmaq_f64(u2, vcvt_f64_f32(vget_low_f32(w3v)),
                                       vcvt_f64_f32(vget_low_f32(x3v)));
                    u3 = vfmaq_f64(u3, vcvt_high_f64_f32(w3v), vcvt_high_f64_f32(x3v));
                }
                const float64x2_t t0 = vaddq_f64(u0, u2);
                const float64x2_t t1 = vaddq_f64(u1, u3);
                sub = vaddvq_f64(t0) + vaddvq_f64(t1);
                acc += sub * (double)K3_E8M0[sb];
                continue;
            }
#endif

            /* Four double lanes partitioned by i%4, reduced as (s0+s1)+(s2+s3), in
             * the scalar path, so on machines without AVX2 the reduction is the
             * same on every compiler. The split is written out rather than left to
             * the compiler because a sequential floating-point reduction may not be
             * reassociated without -ffast-math, which this build does not set:
             * expressed as one serial accumulator, the hottest loop in the engine
             * compiles to scalar adds no matter what the surrounding code looks
             * like.
             *
             * The lane split changes the summation order. See the accuracy contract
             * on the function above for why that is bounded at ~1e-16 relative. */
            /* The AVX2 grouped path below uses four __m256d accumulators over each
             * 16-element chunk, so its intra-lane summation order differs from the
             * scalar path; see the NIBBLE DECODE comment below for the accuracy
             * contract. The group is short, so four accumulators suffice to break
             * the add-latency chain. On aarch64 without AVX2, the NEON path above
             * already took the fast n==32 case, so the wf-based path below is the
             * general-group fallback with its own two-accumulator NEON loop. */
            int i = 0;
#if defined(__AVX2__)
            {
                /* NIBBLE DECODE IN REGISTER. The expand-to-wf[64]-then-reload form this
                 * replaces cost more than the arithmetic it fed: per 32-element group it
                 * ran 16 scalar table lookups and 32 four-byte stores, then reloaded all
                 * 32 floats one vector at a time, and the reload of a just-written stack
                 * slot is a store-forwarding stall on every group.
                 *
                 * E2M1 is small enough to decode with a shuffle instead of a table. The
                 * eight magnitudes {0,.5,1,1.5,2,3,4,6} are indexed by the low three bits
                 * of the code, which is exactly _mm256_permutevar8x32_ps of a register
                 * constant, and bit 3 is the sign, which is that bit moved to 31 and
                 * XORed in. Code 8 gives 0.0f ^ 0x80000000 = -0.0f, which is what
                 * K3_E2M1[8] holds, so the negative zero survives.
                 *
                 * ACCURACY CONTRACT (test_expert.c:219) is maxrel < 1e-6 against
                 * dequant-then-matmul, NOT bit-identity. This grouped path is the
                 * FALLBACK for group not a multiple of 16, or a failed xd hoist; on
                 * the normal K3 shape the flat row path above is taken instead. It
                 * keeps four independent accumulators (v0..v3) to break the FMA
                 * latency chain, so its intra-lane accumulation order differs from
                 * the scalar path below; the difference is a few ulps of double,
                 * orders of magnitude inside the 1e-6 gate. The bench FNV1a of the
                 * mxfp4 output differs from the pre-optimisation value -- that hash
                 * is a determinism check, not a correctness oracle. */
                const __m256  LUT = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f,
                                                   2.0f, 3.0f, 4.0f, 6.0f);
                const __m128i m0f = _mm_set1_epi8(0x0F);
                const __m256i m07 = _mm256_set1_epi32(7);
                const __m256i m08 = _mm256_set1_epi32(8);
                /* 16-element iteration: ONE 8-byte load yields all 16 nibbles, decoded
                 * into c0/c1 by unpacking the low/high nibble masks. Each block feeds its
                 * own accumulator (v0..v3), so per iteration every accumulator chain is a
                 * single FMA -- four independent depth-1 chains hide both the decode
                 * latency and the FMA latency. Group 32 collapses to two iterations. The
                 * scalar tail handles 8-15 and 0-7 remainders. */
                __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
                __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
                for (; i + 15 < n; i += 16) {
                    const __m128i b  = _mm_loadl_epi64((const __m128i *)(pb + (i >> 1)));
                    const __m128i lo = _mm_and_si128(b, m0f);
                    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
                    /* unpacklo interleaves the low 8 bytes of lo and hi, which together
                     * cover all 16 elements in order: [e0,e1,e2,...,e15]. Split that 16
                     * bytes into the low 8 (elems 0-7) and high 8 (elems 8-15). */
                    const __m128i u16 = _mm_unpacklo_epi8(lo, hi);
                    const __m256i c0 = _mm256_cvtepu8_epi32(u16);
                    const __m256i c1 = _mm256_cvtepu8_epi32(_mm_srli_si128(u16, 8));
                    const __m256  w0 = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUT, _mm256_and_si256(c0, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c0, m08), 28)));
                    const __m256  w1 = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUT, _mm256_and_si256(c1, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c1, m08), 28)));
                    v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(w0)),
                                         _mm256_loadu_pd(xdg + i), v0);
                    v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(w0, 1)),
                                         _mm256_loadu_pd(xdg + i + 4), v1);
                    v2 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(w1)),
                                         _mm256_loadu_pd(xdg + i + 8), v2);
                    v3 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(w1, 1)),
                                         _mm256_loadu_pd(xdg + i + 12), v3);
                }
                /* 8-element remainder: same AVX2 decode + FMA as before. Runs at most
                 * once, for the 8-15 remainder after the 16-element loop. */
                for (; i + 7 < n; i += 8) {
                    int32_t four;
                    memcpy(&four, pb + (i >> 1), 4);
                    const __m128i b  = _mm_cvtsi32_si128(four);
                    const __m128i lo = _mm_and_si128(b, m0f);
                    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
                    const __m256i c  = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi));
                    const __m256  wv = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUT, _mm256_and_si256(c, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c, m08), 28)));
                    v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(wv)),
                                         _mm256_loadu_pd(xdg + i), v0);
                    v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(wv, 1)),
                                         _mm256_loadu_pd(xdg + i + 4), v1);
                }
                double a[4];
                _mm256_storeu_pd(a, _mm256_add_pd(_mm256_add_pd(v0, v2),
                                                  _mm256_add_pd(v1, v3)));
                sub = (a[0] + a[1]) + (a[2] + a[3]);
            }
            /* Sub-8 remainder, decoded one nibble at a time. K3 never reaches it (group
             * 32 divides evenly) but a short final group must still be correct. */
            for (; i < n; i++) {
                const unsigned char by = pb[i >> 1];
                sub = fma((double)K3_E2M1[(i & 1) ? (by >> 4) : (by & 0x0F)],
                          xdg[i], sub);
            }
#else
            {
                /* Expand the group to floats first, then take a plain dot product. The
                 * split exists so the second loop can vectorise, which it cannot do
                 * while a table lookup sits in the middle of the accumulation. */
                float wf[64];                     /* group is 32 for K3; 64 is headroom */
                const int half = n >> 1;
                for (int j = 0; j < half; j++) {
                    const float *pv = K3_E2M1_PAIR[pb[j]];
                    wf[2 * j]     = pv[0];
                    wf[2 * j + 1] = pv[1];
                }
                if (n & 1) wf[n - 1] = K3_E2M1_PAIR[pb[half]][0];

#if defined(__ARM_NEON) && defined(__aarch64__)
                {
                    /* uk holds the scalar path's {s[2k], s[2k+1]}: element i in
                     * accumulator i%8, vfmaq_f64 the same IEEE fma() per lane. u0+u2 is
                     * lanewise {s0+s4, s1+s5} = {b0,b1} and u1+u3 is {b2,b3}, so the
                     * reduction below is the scalar (b0+b1)+(b2+b3) tree exactly. */
                    float64x2_t u0 = vdupq_n_f64(0.0), u1 = vdupq_n_f64(0.0);
                    float64x2_t u2 = vdupq_n_f64(0.0), u3 = vdupq_n_f64(0.0);
                    for (; i + 7 < n; i += 8) {
                        const float32x4_t wv0 = vld1q_f32(wf + i);
                        const float32x4_t wv1 = vld1q_f32(wf + i + 4);
                        const float32x4_t xv0 = vld1q_f32(xg + i);
                        const float32x4_t xv1 = vld1q_f32(xg + i + 4);
                        u0 = vfmaq_f64(u0, vcvt_f64_f32(vget_low_f32(wv0)),
                                           vcvt_f64_f32(vget_low_f32(xv0)));
                        u1 = vfmaq_f64(u1, vcvt_high_f64_f32(wv0), vcvt_high_f64_f32(xv0));
                        u2 = vfmaq_f64(u2, vcvt_f64_f32(vget_low_f32(wv1)),
                                           vcvt_f64_f32(vget_low_f32(xv1)));
                        u3 = vfmaq_f64(u3, vcvt_high_f64_f32(wv1), vcvt_high_f64_f32(xv1));
                    }
                    const float64x2_t t0 = vaddq_f64(u0, u2);
                    const float64x2_t t1 = vaddq_f64(u1, u3);
                    sub = vaddvq_f64(t0) + vaddvq_f64(t1);
                }
#else
                {
                    double s[8] = {0};
                    for (; i + 7 < n; i += 8)
                        for (int l = 0; l < 8; l++)
                            s[l] = fma((double)wf[i + l], xdg[i + l], s[l]);
                    double b0 = s[0] + s[4], b1 = s[1] + s[5];
                    double b2 = s[2] + s[6], b3 = s[3] + s[7];
                    sub = (b0 + b1) + (b2 + b3);
                }
#endif
                for (; i < n; i++) sub = fma((double)wf[i], xdg[i], sub);
            }
#endif
            acc += sub * (double)K3_E8M0[sb];
        }
        y[r] = (float)acc;
    }
}

void k3_mxfp4_dequant(float *out, const unsigned char *packed,
                      const unsigned char *scales, int rows, int pcols, int group)
{
    const int width = pcols * 2;                  /* logical elements per row */
    const int ngrp  = (width + group - 1) / group;

    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = packed + (size_t)r * pcols;
        const unsigned char *sr = scales + (size_t)r * ngrp;
        float *orow = out + (size_t)r * width;

        for (int g = 0; g < ngrp; g++) {
            /* E8M0: a bare biased exponent. 255 is NaN by spec; map it to zero so one
             * bad byte cannot poison the row. ldexpf is exact for powers of two. */
            const unsigned char sb = sr[g];
            const float mult = (sb == 255) ? 0.0f : ldexpf(1.0f, (int)sb - 127);

            const int lo = g * group;
            int hi = lo + group;
            if (hi > width) hi = width;

            for (int i = lo; i < hi; i++) {
                const unsigned char byte = pr[i >> 1];
                /* low nibble = EVEN element. Reversing this gives right values in
                 * wrong places, which every statistical check would pass. */
                const unsigned char nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
                orow[i] = K3_E2M1[nib] * mult;
            }
        }
    }
}

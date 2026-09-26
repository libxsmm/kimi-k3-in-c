/* k3_resident.c - see k3_resident.h. */
#define _POSIX_C_SOURCE 200809L

#include "k3_portable_io.h"   /* first: sets _DARWIN_C_SOURCE; posix_memalign on Windows */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <sys/mman.h>
#endif

#include "k3_resident.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int res_get(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3Resident *r = (K3Resident *)self;
    if (layer < r->layer_lo || layer >= r->layer_hi || expert < 0 ||
        expert >= r->n_experts || !r->base[layer]) {
        fprintf(stderr, "k3_resident: L%d expert %d is not resident\n", layer, expert);
        return -1;
    }
    r->requests++;
    const K3ExpertRef *e = &r->ref[(size_t)layer * r->n_experts + expert];
    const unsigned char *b = r->base[layer] + e->off;
    out->p1 = b + e->m[0].p_off; out->s1 = b + e->m[0].s_off;
    out->p2 = b + e->m[1].p_off; out->s2 = b + e->m[1].s_off;
    out->p3 = b + e->m[2].p_off; out->s3 = b + e->m[2].s_off;
    out->ilv = r->ilv;
    return 0;
}

static int res_resident(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3Resident *r = (K3Resident *)self;
    if (layer < r->layer_lo || layer >= r->layer_hi || expert < 0 ||
        expert >= r->n_experts || !r->base[layer])
        return 0;
    if (out) return res_get(self, layer, expert, out) == 0;
    return 1;
}

/* Resolve every expert of layer L, rebase offsets onto the span start, and return the
 * span. All experts must be contiguous runs in one shard. */
static int layer_span(const K3St *st, int L, int n_experts, K3ExpertRef *refs,
                      int *shard, int64_t *lo, int64_t *hi, int64_t *own)
{
    *lo = INT64_MAX; *hi = 0; *own = 0; *shard = -1;
    for (int e = 0; e < n_experts; e++) {
        K3ExpertRef *x = &refs[e];
        if (k3_expert_ref(st, L, e, x) != 0) return -1;
        if (!x->contiguous) {
            fprintf(stderr, "k3_resident: L%d expert %d is not one contiguous run\n", L, e);
            return -1;
        }
        if (*shard < 0) *shard = x->shard;
        if (x->shard != *shard) {
            fprintf(stderr, "k3_resident: L%d experts span shards %d and %d\n",
                    L, *shard, x->shard);
            return -1;
        }
        if (x->off < *lo) *lo = x->off;
        if (x->off + x->nbytes > *hi) *hi = x->off + x->nbytes;
        *own += x->nbytes;
    }
    /* A span far larger than its experts would read unrelated tensors; refuse it. */
    if (*hi - *lo > *own + *own / 8) {
        fprintf(stderr, "k3_resident: L%d expert span %lld B holds only %lld B of experts\n",
                L, (long long)(*hi - *lo), (long long)*own);
        return -1;
    }
    for (int e = 0; e < n_experts; e++) refs[e].off -= *lo;
    return 0;
}

int64_t k3_resident_bytes(const K3St *st, const K3Cfg *cfg, int layer_lo, int layer_hi)
{
    K3ExpertRef *refs = (K3ExpertRef *)malloc((size_t)cfg->n_experts * sizeof *refs);
    if (!refs) return -1;
    int64_t total = 0;
    for (int L = layer_lo; L < layer_hi; L++) {
        if (k3_is_dense(cfg, L)) continue;
        int shard; int64_t lo, hi, own;
        if (layer_span(st, L, cfg->n_experts, refs, &shard, &lo, &hi, &own) != 0) {
            free(refs); return -1;
        }
        total += hi - lo;
    }
    free(refs);
    return total;
}

static int alloc_span(unsigned char **out, size_t bytes)
{
    const int huge = !getenv("K3_NOHUGE");
    const size_t al = huge ? (2u << 20) : 4096u;
    const size_t len = (bytes + al - 1) & ~(al - 1);
    if (posix_memalign((void **)out, al, len) != 0) return -1;
#if defined(MADV_HUGEPAGE)
    if (huge) madvise(*out, len, MADV_HUGEPAGE);
#endif
    return 0;
}

int k3_resident_init(K3Resident *r, const K3St *st, const K3Cfg *cfg,
                     int layer_lo, int layer_hi);

/* Tensor parallel: keep only this rank's rows of every expert of layer L, w1/w3 rows
 * k3_tp_part(moe_inter) and w2 rows k3_tp_part(latent), each read straight from the
 * shard. A slot holds [p1 s1 p3 s3 p2 s2] and refs are rewritten to point into it. */
static int load_layer_sliced(K3Resident *r, const K3St *st, const K3Cfg *cfg, int L,
                             K3ExpertRef *refs, int shard, int64_t lo)
{
    const int NE = r->n_experts;
    int i0, i1, l0, l1;
    k3_tp_part(cfg->moe_inter, &i0, &i1);
    k3_tp_part(cfg->latent, &l0, &l1);
    const K3QMat *m1 = &refs[0].m[0], *m2 = &refs[0].m[1];
    if (m1->rows != cfg->moe_inter || m2->rows != cfg->latent) {
        fprintf(stderr, "k3_resident: L%d expert shapes %dx%d / %dx%d do not match the "
                        "config\n", L, m1->rows, m1->pcols * 2, m2->rows, m2->pcols * 2);
        return -1;
    }
    /* piece k: matrix index into m[], first row, row count, packed or scale */
    const int mat[6] = { 0, 0, 2, 2, 1, 1 };
    int64_t len[6], at[6], slot = 0, maxp = 0;
    for (int k = 0; k < 6; k++) {
        const K3QMat *m = &refs[0].m[mat[k]];
        const int nrow = mat[k] == 1 ? l1 - l0 : i1 - i0;
        len[k] = (int64_t)nrow * ((k & 1) ? m->scols : m->pcols);
        at[k] = slot;
        slot += (len[k] + 63) & ~(int64_t)63;
        if (len[k] > maxp) maxp = len[k];
    }
    if (alloc_span(&r->alloc[L], (size_t)slot * NE) != 0) {
        fprintf(stderr, "k3_resident: cannot allocate %.2f GB for layer %d\n",
                (double)slot * NE / 1e9, L);
        return -1;
    }

    const double t0 = now_s();
    volatile int failed = 0;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        const int64_t cap = maxp + 2 * K3_ST_ALIGN;
        unsigned char *bounce = NULL;
        if (posix_memalign((void **)&bounce, K3_ST_ALIGN, (size_t)cap) != 0) failed = 1;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 16)
#endif
        for (int task = 0; task < NE * 6; task++) {
            if (failed) continue;
            const int e = task / 6, k = task % 6;
            const K3QMat *m = &refs[e].m[mat[k]];
            const int r0 = mat[k] == 1 ? l0 : i0;
            const int64_t off = lo + refs[e].off + ((k & 1) ? m->s_off : m->p_off)
                              + (int64_t)r0 * ((k & 1) ? m->scols : m->pcols);
            int64_t pad = 0;
            if (k3_st_read_aligned(st, shard, off, len[k], bounce, cap, &pad) != len[k]) {
                failed = 1;
                continue;
            }
            memcpy(r->alloc[L] + (size_t)e * slot + at[k], bounce + pad, (size_t)len[k]);
            if (r->ilv && !(k & 1))
                k3_mxfp4_interleave(r->alloc[L] + (size_t)e * slot + at[k],
                                    (int)(len[k] / m->pcols), m->pcols * 2, K3_MXFP4_GROUP,
                                    r->ilv);
        }
        k3_aligned_free(bounce);
    }
    const double dt = now_s() - t0;
    if (failed) {
        fprintf(stderr, "k3_resident: short read on layer %d expert slices\n", L);
        return -1;
    }
    for (int e = 0; e < NE; e++) {
        refs[e].off = (int64_t)e * slot;
        refs[e].m[0].p_off = at[0]; refs[e].m[0].s_off = at[1];
        refs[e].m[2].p_off = at[2]; refs[e].m[2].s_off = at[3];
        refs[e].m[1].p_off = at[4]; refs[e].m[1].s_off = at[5];
    }
    r->base[L] = r->alloc[L];
    r->bytes_resident += (uint64_t)slot * NE;
    r->load_seconds += dt;
    printf("  experts L%-3d %.2f GB (rank %d of %d rows) in %.1f s (%.2f GB/s)\n",
           L, (double)slot * NE / 1e9, k3_tp.rank, k3_tp.size, dt,
           (double)slot * NE / 1e9 / dt);
    fflush(stdout);
    return 0;
}

int k3_resident_init(K3Resident *r, const K3St *st, const K3Cfg *cfg,
                     int layer_lo, int layer_hi)
{
    memset(r, 0, sizeof *r);
    r->src.get = res_get;
    r->src.resident = res_resident;
    r->src.getmany = NULL;          /* nothing to prefetch */
    r->src.ctx = r;
    r->src.sliced = k3_tp.size > 1;
    {
        const char *ev = getenv("K3_MXFP4_ILV");
        const int lay = (k3_act_bf16 & K3_BF16_MX) ? K3_MX_BF16 : K3_MX_F32;
        r->ilv = !(ev && ev[0] == '0') && k3_mxfp4_interleave(NULL, 0, 0, K3_MXFP4_GROUP, lay)
               ? lay : K3_MX_CKPT;
    }
    r->st = st;
    r->n_layers = cfg->n_layers;
    r->n_experts = cfg->n_experts;
    r->layer_lo = layer_lo < 0 ? 0 : layer_lo;
    r->layer_hi = layer_hi > cfg->n_layers ? cfg->n_layers : layer_hi;

    r->alloc = (unsigned char **)calloc((size_t)r->n_layers, sizeof *r->alloc);
    r->base  = (unsigned char **)calloc((size_t)r->n_layers, sizeof *r->base);
    r->ref   = (K3ExpertRef *)calloc((size_t)r->n_layers * r->n_experts, sizeof *r->ref);
    if (!r->alloc || !r->base || !r->ref) { k3_resident_free(r); return -1; }

    const double t_all = now_s();
    for (int L = r->layer_lo; L < r->layer_hi; L++) {
        if (k3_is_dense(cfg, L)) continue;
        K3ExpertRef *refs = r->ref + (size_t)L * r->n_experts;
        int shard; int64_t lo, hi, own;
        if (layer_span(st, L, r->n_experts, refs, &shard, &lo, &hi, &own) != 0) {
            k3_resident_free(r); return -1;
        }
        if (r->src.sliced) {
            if (load_layer_sliced(r, st, cfg, L, refs, shard, lo) != 0) {
                k3_resident_free(r); return -1;
            }
            continue;
        }
        const int64_t lo_al = lo & ~(int64_t)(K3_ST_ALIGN - 1);
        const int64_t hi_al = (hi + K3_ST_ALIGN - 1) & ~(int64_t)(K3_ST_ALIGN - 1);
        const int64_t len = hi_al - lo_al;
        if (alloc_span(&r->alloc[L], (size_t)len) != 0) {
            fprintf(stderr, "k3_resident: cannot allocate %.2f GB for layer %d\n",
                    (double)len / 1e9, L);
            k3_resident_free(r); return -1;
        }

        const double t0 = now_s();
        int64_t pad = 0;
        const int64_t got = k3_st_read_par(st, shard, lo, hi - lo, r->alloc[L], len, &pad);
        const double dt = now_s() - t0;
        if (got != hi - lo) {
            fprintf(stderr, "k3_resident: short read on layer %d experts\n", L);
            k3_resident_free(r); return -1;
        }
        r->base[L] = r->alloc[L] + pad;
        if (r->ilv) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
            for (int task = 0; task < r->n_experts * 3; task++) {
                const K3QMat *m = &refs[task / 3].m[task % 3];
                k3_mxfp4_interleave(r->base[L] + refs[task / 3].off + m->p_off,
                                    m->rows, m->pcols * 2, K3_MXFP4_GROUP, r->ilv);
            }
        }
        r->bytes_resident += (uint64_t)(hi - lo);
        r->load_seconds += dt;
        printf("  experts L%-3d %.2f GB in %.1f s (%.2f GB/s)\n",
               L, (double)(hi - lo) / 1e9, dt, (double)(hi - lo) / 1e9 / dt);
        fflush(stdout);
    }
    printf("resident experts: %.2f GB for layers [%d, %d) in %.1f s\n",
           (double)r->bytes_resident / 1e9, r->layer_lo, r->layer_hi, now_s() - t_all);
    return 0;
}

void k3_resident_free(K3Resident *r)
{
    if (r->alloc)
        for (int L = 0; L < r->n_layers; L++) k3_aligned_free(r->alloc[L]);
    free(r->alloc); free(r->base); free(r->ref);
    memset(r, 0, sizeof *r);
}

void k3_resident_report(const K3Resident *r, const char *label)
{
    printf("resident experts [%s]\n", label ? label : "");
    printf("  %.2f GB resident for layers [%d, %d), loaded in %.1f s\n",
           (double)r->bytes_resident / 1e9, r->layer_lo, r->layer_hi, r->load_seconds);
    printf("  requests %llu, all served from RAM\n", (unsigned long long)r->requests);
}

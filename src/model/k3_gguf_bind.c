/* SPDX-License-Identifier: Apache-2.0 */
/* k3_gguf_bind.c - see k3_gguf_bind.h. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE        /* madvise */

#include "k3_portable_io.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "k3_gguf_bind.h"
#include "k3_gq.h"

#define MAXR 48
#define HUGE_ALIGN ((size_t)2 << 20)

enum { GR_MAT, GR_TAG, GR_F32, GR_ONES, GR_ALOG, GR_KVB };

/* One bound tensor. GR_MAT matrices share the struct's wdt; GR_TAG carries its own (KDA b);
 * GR_F32 is widened to fp32; the rest are synthesised from other tensors. */
typedef struct {
    char             name[128];
    int              kind;
    const void     **dest;
    int             *tag;          /* GR_TAG: receives the weight tag */
    int64_t          rows, in;     /* the full tensor's logical shape */
    int              r0, r1;       /* rows kept */
    const K3GTensor *t, *t2;
    size_t           off, bytes;
} GReq;

typedef struct { GReq r[MAXR]; int n, bad, wdt; } GPlan;

static size_t al64(size_t x) { return (x + 63) & ~(size_t)63; }

static GReq *gp_add(GPlan *p, int kind, const void **dest, int64_t rows, int64_t in,
                    int r0, int r1, const char *fmt, ...)
{
    if (p->n >= MAXR) { fprintf(stderr, "k3_gguf_bind: too many tensors\n"); p->bad++; return NULL; }
    GReq *q = &p->r[p->n++];
    memset(q, 0, sizeof *q);
    q->kind = kind; q->dest = dest; q->rows = rows; q->in = in;
    if (k3_tp.local) { q->r0 = r0; q->r1 = r1; } else { q->r0 = 0; q->r1 = (int)rows; }
    if (fmt) { va_list ap; va_start(ap, fmt); vsnprintf(q->name, sizeof q->name, fmt, ap); va_end(ap); }
    return q;
}

/* A whole tensor, never sliced. */
static GReq *gp_whole(GPlan *p, int kind, const void **dest, int64_t rows, int64_t in,
                      const char *fmt, int L)
{
    GReq *q = gp_add(p, kind, dest, rows, in, 0, (int)rows, fmt, L);
    if (q) { q->r0 = 0; q->r1 = (int)rows; }
    return q;
}

static int wtag(int type)
{
    switch (type) {
    case K3_GG_Q8_0: return K3_WQ8_0;
    case K3_GG_BF16: return K3_WBF16;
    case K3_GG_F32:  return K3_WF32;
    default:         return -1;
    }
}

static int shape_ok(const GReq *q, const K3GTensor *t)
{
    int64_t numel = 1;
    for (int d = 0; d < t->ndim; d++) numel *= t->ne[d];
    if (t->ne[0] != q->in || numel != q->rows * q->in) {
        fprintf(stderr, "k3_gguf_bind: %s has ne [%lld,%lld,%lld], engine expects %lld rows of %lld\n",
                q->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                (long long)q->rows, (long long)q->in);
        return 0;
    }
    if (t->nbytes < 0) {
        fprintf(stderr, "k3_gguf_bind: %s has ggml type %d, which this engine does not read\n",
                q->name, t->type);
        return 0;
    }
    return 1;
}

static int64_t gp_resolve(GPlan *p, const K3Gguf *g, const K3Cfg *c)
{
    size_t off = 0;
    p->wdt = -1;
    for (int i = 0; i < p->n; i++) {
        GReq *q = &p->r[i];
        const int64_t keep = q->r1 - q->r0;
        switch (q->kind) {
        case GR_ONES:
            q->bytes = (size_t)q->in * 4;
            break;
        case GR_KVB: {
            char kb[128];
            snprintf(kb, sizeof kb, "%s", q->name);
            char *dot = strstr(kb, "attn_k_b");
            q->t = k3_gguf_find(g, kb);
            if (dot) memcpy(dot, "attn_v_b", 8);
            q->t2 = k3_gguf_find(g, kb);
            if (!q->t || !q->t2) { fprintf(stderr, "k3_gguf_bind: missing %s or its attn_v_b\n", q->name); p->bad++; continue; }
            const K3GTensor *k = q->t, *v = q->t2;
            if (k->ne[0] != c->qk_nope || k->ne[1] != c->kv_lora || k->ne[2] != c->n_heads ||
                v->ne[0] != c->kv_lora || v->ne[1] != c->v_head || v->ne[2] != c->n_heads ||
                k->nbytes < 0 || v->nbytes < 0) {
                fprintf(stderr, "k3_gguf_bind: %s / attn_v_b do not have the MLA shapes\n", q->name);
                p->bad++; continue;
            }
            q->bytes = (size_t)keep * (size_t)q->in * 4;
            break;
        }
        default:
            q->t = k3_gguf_find(g, q->name);
            if (!q->t) { fprintf(stderr, "k3_gguf_bind: missing tensor %s\n", q->name); p->bad++; continue; }
            if (q->kind == GR_ALOG) {
                if (q->t->ne[0] < q->in || q->t->type != K3_GG_F32) {
                    fprintf(stderr, "k3_gguf_bind: %s is not %lld F32 values\n", q->name, (long long)q->in);
                    p->bad++; continue;
                }
                q->bytes = (size_t)q->in * 4;
                break;
            }
            if (!shape_ok(q, q->t)) { p->bad++; continue; }
            if (q->kind == GR_F32) {
                q->bytes = (size_t)keep * (size_t)q->in * 4;
            } else {
                const int tg = wtag(q->t->type);
                if (tg < 0 || (tg == K3_WQ8_0 && q->in % 32)) {
                    fprintf(stderr, "k3_gguf_bind: %s: ggml type %d cannot be a trunk matrix\n",
                            q->name, q->t->type);
                    p->bad++; continue;
                }
                if (q->kind == GR_MAT) {
                    if (p->wdt >= 0 && p->wdt != tg) {
                        fprintf(stderr, "k3_gguf_bind: %s is ggml type %d, unlike the rest of its layer\n",
                                q->name, q->t->type);
                        p->bad++; continue;
                    }
                    p->wdt = tg;
                }
                q->bytes = (size_t)keep * k3_gq_row_bytes(q->t->type, q->in);
            }
        }
        q->off = off;
        off += al64(q->bytes);
    }
    if (p->wdt < 0) p->wdt = K3_WF32;
    return p->bad ? -1 : (int64_t)off;
}

static int read_rows(const K3Gguf *g, const K3GTensor *t, int64_t in, int r0, int r1, void *dst)
{
    const size_t rb = k3_gq_row_bytes(t->type, in);
    return k3_gguf_read(g, t->shard, t->off + (int64_t)r0 * (int64_t)rb, (int64_t)(r1 - r0) * (int64_t)rb, dst);
}

static int gp_load(GPlan *p, const K3Gguf *g, const K3Cfg *c, unsigned char *blob)
{
    for (int i = 0; i < p->n; i++) {
        GReq *q = &p->r[i];
        unsigned char *dst = blob + q->off;
        const int keep = q->r1 - q->r0;
        int rc = 0;
        if (q->kind == GR_ONES) {
            float *o = (float *)dst;
            for (int64_t k = 0; k < q->in; k++) o[k] = 1.0f;
        } else if (q->kind == GR_ALOG) {
            float *a = (float *)malloc((size_t)q->t->ne[0] * 4);
            rc = a ? k3_gguf_read(g, q->t->shard, q->t->off, q->t->ne[0] * 4, a) : -1;
            float *o = (float *)dst;
            for (int64_t h = 0; rc == 0 && h < q->in; h++) {
                if (!(a[h] < 0.0f)) {
                    fprintf(stderr, "k3_gguf_bind: %s[%lld] = %g, -exp(A_log) must be negative\n",
                            q->name, (long long)h, a[h]);
                    rc = -1;
                }
                o[h] = logf(-a[h]);
            }
            free(a);
        } else if (q->kind == GR_KVB) {
            /* rows h*kvd + n (n < nope) = k_b[h][:, n], rows h*kvd + nope + d = v_b[h][d][:] */
            const int nope = c->qk_nope, vh = c->v_head, R = c->kv_lora, kvd = nope + vh;
            const int h0 = q->r0 / kvd, h1 = q->r1 / kvd, nh = h1 - h0;
            float *kb = (float *)malloc(sizeof(float) * (size_t)nh * R * nope);
            float *vb = (float *)malloc(sizeof(float) * (size_t)nh * vh * R);
            void *raw = malloc((size_t)nh * R * k3_gq_row_bytes(q->t->type, nope) +
                               (size_t)nh * vh * k3_gq_row_bytes(q->t2->type, R));
            rc = (kb && vb && raw) ? 0 : -1;
            if (rc == 0) rc = read_rows(g, q->t, nope, h0 * R, h1 * R, raw);
            if (rc == 0) k3_gq_dequant(q->t->type, raw, kb, (int64_t)nh * R * nope);
            if (rc == 0) rc = read_rows(g, q->t2, R, h0 * vh, h1 * vh, raw);
            if (rc == 0) k3_gq_dequant(q->t2->type, raw, vb, (int64_t)nh * vh * R);
            float *o = (float *)dst;
            for (int h = 0; rc == 0 && h < nh; h++) {
                for (int n = 0; n < nope; n++)
                    for (int r = 0; r < R; r++)
                        o[((size_t)h * kvd + n) * R + r] = kb[((size_t)h * R + r) * nope + n];
                memcpy(o + ((size_t)h * kvd + nope) * R, vb + (size_t)h * vh * R, sizeof(float) * (size_t)vh * R);
            }
            free(kb); free(vb); free(raw);
        } else if (q->kind == GR_F32 && q->t->type != K3_GG_F32) {
            void *raw = malloc((size_t)keep * k3_gq_row_bytes(q->t->type, q->in));
            rc = raw ? read_rows(g, q->t, q->in, q->r0, q->r1, raw) : -1;
            if (rc == 0) k3_gq_dequant(q->t->type, raw, (float *)dst, (int64_t)keep * q->in);
            free(raw);
        } else {
            rc = read_rows(g, q->t, q->in, q->r0, q->r1, dst);
        }
        if (rc != 0) { fprintf(stderr, "k3_gguf_bind: could not load %s\n", q->name); return -1; }
        *q->dest = dst;
        if (q->tag) *q->tag = wtag(q->t->type);
    }
    return 0;
}

static void *blob_alloc(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, HUGE_ALIGN, n ? n : 64) != 0) return NULL;
#ifdef MADV_HUGEPAGE
    madvise(p, n, MADV_HUGEPAGE);
#endif
    return p;
}

/* ------------------------------------------------------------------ one layer */
static void plan_layer(GPlan *p, const K3Cfg *c, int L, K3LayerBind *b, int is_mla, int is_dense)
{
    const int64_t H = c->hidden, P = (int64_t)c->kda_heads * c->kda_head_dim;
    int e0, e1;
    k3_tp_part(c->hidden, &e0, &e1);

    gp_whole(p, GR_F32, (const void **)&b->lay.in_norm,   1, H, "blk.%d.attn_norm.weight", L);
    gp_whole(p, GR_F32, (const void **)&b->lay.post_norm, 1, H, "blk.%d.ffn_norm.weight", L);
    gp_whole(p, GR_F32, (const void **)&b->lay.attn_res_norm, 1, H, "blk.%d.attn_res_score.weight", L);
    gp_whole(p, GR_F32, (const void **)&b->lay.mlp_res_norm,  1, H, "blk.%d.ffn_res_score.weight", L);
    gp_whole(p, GR_ONES, (const void **)&b->lay.attn_res_proj, 1, H, NULL, L);
    gp_whole(p, GR_ONES, (const void **)&b->lay.mlp_res_proj,  1, H, NULL, L);

    if (is_mla) {
        const int64_t qh = (int64_t)c->qk_nope + c->qk_rope, kvd = (int64_t)c->qk_nope + c->v_head;
        const int64_t kvw = c->kv_lora + c->qk_rope, NH = c->n_heads, vh = c->v_head;
        int h0, h1, a0, a1, b0, b1;
        k3_tp_part(c->n_heads, &h0, &h1);
        k3_tp_part(c->q_lora, &a0, &a1);
        k3_tp_part((int)kvw, &b0, &b1);
        gp_add(p, GR_MAT, &b->mla.q_a, c->q_lora, H, a0, a1, "blk.%d.attn_q_a.weight", L);
        gp_whole(p, GR_F32, (const void **)&b->mla.q_a_norm, 1, c->q_lora, "blk.%d.attn_q_a_norm.weight", L);
        gp_add(p, GR_MAT, &b->mla.q_b, NH * qh, c->q_lora, (int)(h0 * qh), (int)(h1 * qh), "blk.%d.attn_q_b.weight", L);
        gp_add(p, GR_MAT, &b->mla.kv_a, kvw, H, b0, b1, "blk.%d.attn_kv_a_mqa.weight", L);
        gp_whole(p, GR_F32, (const void **)&b->mla.kv_a_norm, 1, c->kv_lora, "blk.%d.attn_kv_a_norm.weight", L);
        gp_add(p, GR_KVB, &b->mla.kv_b, NH * kvd, c->kv_lora, (int)(h0 * kvd), (int)(h1 * kvd), "blk.%d.attn_k_b.weight", L);
        gp_add(p, GR_MAT, &b->mla.o, H, NH * vh, e0, e1, "blk.%d.attn_output.weight", L);
        if (c->mla_out_gate)
            gp_add(p, GR_MAT, &b->mla.g, NH * vh, H, (int)(h0 * vh), (int)(h1 * vh), "blk.%d.attn_gate.weight", L);
    } else {
        const int D = c->kda_head_dim;
        int h0, h1;
        k3_tp_part(c->kda_heads, &h0, &h1);
        const int c0 = h0 * D, c1 = h1 * D;
        gp_add(p, GR_MAT, &b->kda.q, P, H, c0, c1, "blk.%d.attn_q.weight", L);
        gp_add(p, GR_MAT, &b->kda.k, P, H, c0, c1, "blk.%d.attn_k.weight", L);
        gp_add(p, GR_MAT, &b->kda.v, P, H, c0, c1, "blk.%d.attn_v.weight", L);
        gp_add(p, GR_MAT, &b->kda.g, P, H, c0, c1, "blk.%d.ssm_g.weight", L);
        gp_add(p, GR_MAT, &b->kda.o, H, P, e0, e1, "blk.%d.attn_output.weight", L);
        gp_whole(p, GR_F32, (const void **)&b->kda.q_conv, P, c->conv_k, "blk.%d.ssm_conv1d_q.weight", L);
        gp_whole(p, GR_F32, (const void **)&b->kda.k_conv, P, c->conv_k, "blk.%d.ssm_conv1d_k.weight", L);
        gp_whole(p, GR_F32, (const void **)&b->kda.v_conv, P, c->conv_k, "blk.%d.ssm_conv1d_v.weight", L);
        gp_whole(p, GR_MAT, &b->kda.f_a, D, H, "blk.%d.ssm_f_a.weight", L);
        gp_add(p, GR_MAT, &b->kda.f_b, P, D, c0, c1, "blk.%d.ssm_f_b.weight", L);
        GReq *bq = gp_add(p, GR_TAG, &b->kda.b, c->kda_heads, H, h0, h1, "blk.%d.ssm_beta.weight", L);
        if (bq) bq->tag = &b->kda.b_wdt;
        gp_whole(p, GR_ALOG, (const void **)&b->kda.A_log, 1, c->kda_heads, "blk.%d.ssm_a", L);
        gp_whole(p, GR_F32, (const void **)&b->kda.dt_bias, 1, P, "blk.%d.ssm_dt.bias", L);
        gp_whole(p, GR_F32, (const void **)&b->kda.o_norm, 1, D, "blk.%d.ssm_norm.weight", L);
    }

    if (is_dense) {
        const int64_t DI = c->dense_inter;
        int d0, d1;
        k3_tp_part(c->dense_inter, &d0, &d1);
        gp_add(p, GR_MAT, &b->lay.dense_gate, DI, H, d0, d1, "blk.%d.ffn_gate.weight", L);
        gp_add(p, GR_MAT, &b->lay.dense_up,   DI, H, d0, d1, "blk.%d.ffn_up.weight", L);
        gp_add(p, GR_MAT, &b->lay.dense_down, H, DI, e0, e1, "blk.%d.ffn_down.weight", L);
    } else {
        const int64_t SI = (int64_t)c->moe_inter * c->n_shared;
        int x0, x1, l0, l1, s0, s1;
        k3_tp_part(c->n_experts, &x0, &x1);
        k3_tp_part(c->latent, &l0, &l1);
        k3_tp_part((int)SI, &s0, &s1);
        gp_add(p, GR_F32, (const void **)&b->moe.gate, c->n_experts, H, x0, x1, "blk.%d.ffn_gate_inp.weight", L);
        gp_whole(p, GR_F32, (const void **)&b->moe.bias, 1, c->n_experts, "blk.%d.exp_probs_b.bias", L);
        gp_add(p, GR_MAT, &b->moe.down, c->latent, H, l0, l1, "blk.%d.ffn_routed_down.weight", L);
        gp_add(p, GR_MAT, &b->moe.up, H, c->latent, e0, e1, "blk.%d.ffn_routed_up.weight", L);
        if (c->latent_norm)
            gp_whole(p, GR_F32, (const void **)&b->moe.latent_norm, 1, c->latent, "blk.%d.ffn_routed_norm.weight", L);
        gp_add(p, GR_MAT, &b->moe.sh1, SI, H, s0, s1, "blk.%d.ffn_gate_shexp.weight", L);
        gp_add(p, GR_MAT, &b->moe.sh3, SI, H, s0, s1, "blk.%d.ffn_up_shexp.weight", L);
        gp_add(p, GR_MAT, &b->moe.sh2, H, SI, e0, e1, "blk.%d.ffn_down_shexp.weight", L);
    }
}

int64_t k3_gguf_layer_bytes(const K3Gguf *g, const K3Cfg *c, int L)
{
    K3LayerBind tmp; memset(&tmp, 0, sizeof tmp);
    GPlan p; memset(&p, 0, sizeof p);
    plan_layer(&p, c, L, &tmp, k3_is_mla(c, L), k3_is_dense(c, L));
    return gp_resolve(&p, g, c);
}

int k3_gguf_bind_layer(const K3Gguf *g, const K3Cfg *c, int L, K3LayerBind *b)
{
    memset(b, 0, sizeof *b);
    b->layer = L;
    const int is_mla = k3_is_mla(c, L), is_dense = k3_is_dense(c, L);
    GPlan p; memset(&p, 0, sizeof p);
    plan_layer(&p, c, L, b, is_mla, is_dense);
    const int64_t need = gp_resolve(&p, g, c);
    if (need < 0) return -1;
    b->blob = blob_alloc((size_t)need);
    if (!b->blob) {
        fprintf(stderr, "k3_gguf_bind: cannot allocate %.2f GB for layer %d\n", need / 1e9, L);
        return -1;
    }
    b->nbytes = (size_t)need;
    if (gp_load(&p, g, c, (unsigned char *)b->blob) != 0) {
        k3_aligned_free(b->blob); b->blob = NULL; return -1;
    }
    b->kda.wdt = b->mla.wdt = b->moe.wdt = b->lay.wdt = p.wdt;
    b->mla.kv_b_wdt = K3_WF32;
    b->lay.kda = is_mla ? NULL : &b->kda;
    b->lay.mla = is_mla ? &b->mla : NULL;
    b->lay.moe = is_dense ? NULL : &b->moe;
    return 0;
}

/* ------------------------------------------------------------------ model level */
int k3_gguf_bind_model(const K3Gguf *g, const K3Cfg *c, K3ModelBind *m)
{
    memset(m, 0, sizeof *m);
    const int64_t H = c->hidden;
    int v0, v1;
    k3_tp_part(c->vocab, &v0, &v1);
    GPlan p; memset(&p, 0, sizeof p);
    gp_whole(&p, GR_MAT, &m->embed, c->vocab, H, "token_embd.weight", 0);
    gp_add(&p, GR_MAT, &m->lm_head, c->vocab, H, v0, v1, "output.weight");
    gp_whole(&p, GR_F32, (const void **)&m->norm, 1, H, "output_norm.weight", 0);
    gp_whole(&p, GR_F32, (const void **)&m->out_res_norm, 1, H, "output_res_score.weight", 0);
    gp_whole(&p, GR_ONES, (const void **)&m->out_res_proj, 1, H, NULL, 0);
    const int64_t need = gp_resolve(&p, g, c);
    if (need < 0) return -1;
    m->blob = blob_alloc((size_t)need);
    if (!m->blob) return -1;
    m->nbytes = (size_t)need;
    if (gp_load(&p, g, c, (unsigned char *)m->blob) != 0) {
        k3_aligned_free(m->blob); m->blob = NULL; return -1;
    }
    m->wdt = p.wdt;
    return 0;
}

/* ------------------------------------------------------------------ experts */
static int eq_of(int type)
{
    return type == K3_GG_IQ2_XS ? K3_EQ_IQ2XS : type == K3_GG_IQ3_XXS ? K3_EQ_IQ3XXS :
           type == K3_GG_MXFP4 ? K3_EQ_MXFP4 : -1;
}

typedef struct {
    const K3GTensor *t[3];   /* w1 gate, w3 up, w2 down */
    int64_t rows[3], in[3];
    int r0[3], r1[3];
    size_t n[3], rb[3], o3, o2, eb;
} ELay;

static int expert_layout(const K3Gguf *g, const K3Cfg *c, int L, ELay *e)
{
    static const char *nm[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    int i0 = 0, i1 = c->moe_inter, l0 = 0, l1 = c->latent;
    if (k3_tp.size > 1) { k3_tp_part(c->moe_inter, &i0, &i1); k3_tp_part(c->latent, &l0, &l1); }
    for (int m = 0; m < 3; m++) {
        char name[96];
        snprintf(name, sizeof name, "blk.%d.%s.weight", L, nm[m]);
        const K3GTensor *t = k3_gguf_find(g, name);
        e->rows[m] = m < 2 ? c->moe_inter : c->latent;
        e->in[m]   = m < 2 ? c->latent : c->moe_inter;
        e->r0[m] = m < 2 ? i0 : l0;
        e->r1[m] = m < 2 ? i1 : l1;
        if (!t) { fprintf(stderr, "k3_gguf_bind: missing %s\n", name); return -1; }
        if (t->ne[0] != e->in[m] || t->ne[1] != e->rows[m] || t->ne[2] != c->n_experts ||
            eq_of(t->type) < 0 || t->nbytes < 0) {
            fprintf(stderr, "k3_gguf_bind: %s: ggml type %d ne [%lld,%lld,%lld] is not a supported "
                            "[%d][%lld][%lld] expert bank (IQ2_XS, IQ3_XXS or MXFP4)\n", name, t->type,
                    (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                    c->n_experts, (long long)e->rows[m], (long long)e->in[m]);
            return -1;
        }
        e->t[m] = t;
        e->rb[m] = k3_gq_row_bytes(t->type, e->in[m]);
        e->n[m] = (size_t)(e->r1[m] - e->r0[m]) * e->rb[m];
    }
    e->o3 = al64(e->n[0]);
    e->o2 = e->o3 + al64(e->n[1]);
    e->eb = e->o2 + al64(e->n[2]);
    return 0;
}

double k3_gguf_experts_bytes(const K3Gguf *g, const K3Cfg *c, int nl)
{
    double s = 0;
    for (int L = 0; L < nl; L++) {
        if (k3_is_dense(c, L)) continue;
        ELay e;
        if (expert_layout(g, c, L, &e) != 0) return -1;
        s += (double)e.eb * c->n_experts;
    }
    return s;
}

static int gx_get(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    const K3GgufExperts *x = (const K3GgufExperts *)self;
    if (layer < 0 || layer >= x->n_layers || expert < 0 || expert >= x->n_experts || !x->base[layer]) {
        fprintf(stderr, "k3_gguf: L%d expert %d is not resident\n", layer, expert);
        return -1;
    }
    const unsigned char *b = x->base[layer] + (size_t)expert * x->ebytes[layer];
    memset(out, 0, sizeof *out);
    out->p1 = b;
    out->p3 = b + x->o3[layer];
    out->p2 = b + x->o2[layer];
    out->qt1 = x->qt[3 * layer + 0];
    out->qt3 = x->qt[3 * layer + 1];
    out->qt2 = x->qt[3 * layer + 2];
    out->s1 = b + x->so[3 * layer + 0];
    out->s3 = b + x->so[3 * layer + 1];
    out->s2 = b + x->so[3 * layer + 2];
    out->ilv = x->ilv;
    return 0;
}

static int gx_resident(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3ExpertQ tmp;
    return gx_get(self, layer, expert, out ? out : &tmp) == 0;
}

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int k3_gguf_experts_init(K3GgufExperts *x, const K3Gguf *g, const K3Cfg *c, int nl)
{
    memset(x, 0, sizeof *x);
    x->n_layers = nl;
    x->n_experts = c->n_experts;
    x->base = (unsigned char **)calloc((size_t)nl, sizeof *x->base);
    x->ebytes = (size_t *)calloc((size_t)nl, sizeof(size_t));
    x->o3 = (size_t *)calloc((size_t)nl, sizeof(size_t));
    x->o2 = (size_t *)calloc((size_t)nl, sizeof(size_t));
    x->qt = (unsigned char *)calloc((size_t)nl * 3, 1);
    x->so = (size_t *)calloc((size_t)nl * 3, sizeof(size_t));
    if (!x->base || !x->ebytes || !x->o3 || !x->o2 || !x->qt || !x->so) return -1;
    x->src.get = gx_get;
    x->src.resident = gx_resident;
    x->src.sliced = k3_tp.size > 1;
    {
        const char *ev = getenv("K3_MXFP4_ILV");
        const int lay = (k3_act_bf16 & K3_BF16_MX) ? K3_MX_BF16 : K3_MX_F32;
        x->ilv = !(ev && ev[0] == '0') && k3_mxfp4_interleave(NULL, 0, 0, K3_MXFP4_GROUP, lay)
               ? lay : K3_MX_CKPT;
    }

    const double t0 = now_s();
    for (int L = 0; L < nl; L++) {
        if (k3_is_dense(c, L)) continue;
        ELay e;
        if (expert_layout(g, c, L, &e) != 0) return -1;
        const size_t total = e.eb * (size_t)c->n_experts;
        x->base[L] = (unsigned char *)blob_alloc(total);
        if (!x->base[L]) {
            fprintf(stderr, "k3_gguf: cannot allocate %.2f GB of experts for layer %d\n", total / 1e9, L);
            return -1;
        }
        x->ebytes[L] = e.eb; x->o3[L] = e.o3; x->o2[L] = e.o2;
        for (int m = 0; m < 3; m++) x->qt[3 * L + m] = (unsigned char)eq_of(e.t[m]->type);
        const size_t om[3] = { 0, e.o3, e.o2 };
        for (int m = 0; m < 3; m++)
            x->so[3 * L + m] = om[m] + (size_t)(e.r1[m] - e.r0[m]) * (size_t)(e.in[m] / 2);
        size_t nmax = e.n[0] > e.n[1] ? e.n[0] : e.n[1];
        if (e.n[2] > nmax) nmax = e.n[2];
        const int64_t cap = (int64_t)nmax + 2 * K3_ST_ALIGN;
        volatile int failed = 0;
        /* one expert's three slices per task: queue depth on the store, and first touch of
         * the destination by the reading thread */
#ifdef _OPENMP
#       pragma omp parallel
#endif
        {
            unsigned char *bounce = NULL;
            if (posix_memalign((void **)&bounce, K3_ST_ALIGN, (size_t)cap) != 0) failed = 1;
#ifdef _OPENMP
#           pragma omp for schedule(dynamic, 4)
#endif
            for (int ex = 0; ex < c->n_experts; ex++) {
                if (failed) continue;
                for (int m = 0; m < 3; m++) {
                    const K3GTensor *t = e.t[m];
                    const int64_t off = t->off + ((int64_t)ex * e.rows[m] + e.r0[m]) * (int64_t)e.rb[m];
                    int64_t pad = 0;
                    if (k3_st_read_aligned(&g->st, t->shard, off, (int64_t)e.n[m], bounce, cap, &pad)
                            != (int64_t)e.n[m]) { failed = 1; break; }
                    unsigned char *dst = x->base[L] + (size_t)ex * e.eb + om[m];
                    if (t->type == K3_GG_MXFP4) {
                        const int rows = e.r1[m] - e.r0[m];
                        k3_gq_mxfp4_to_k3(dst, bounce + pad, rows, e.in[m]);
                        if (x->ilv) k3_mxfp4_interleave(dst, rows, (int)e.in[m], K3_MXFP4_GROUP, x->ilv);
                    } else {
                        memcpy(dst, bounce + pad, e.n[m]);
                    }
                }
            }
            k3_aligned_free(bounce);
        }
        if (failed) { fprintf(stderr, "k3_gguf: reading the experts of layer %d failed\n", L); return -1; }
        x->bytes += (double)total;
        if ((L + 1) % 8 == 0 || L + 1 == nl) {
            const double dt = now_s() - t0;
            printf("  experts: %d/%d layers, %.1f GB in %.0f s (%.2f GB/s)\n",
                   L + 1, nl, x->bytes / 1e9, dt, x->bytes / 1e9 / dt);
            fflush(stdout);
        }
    }
    return 0;
}

void k3_gguf_experts_free(K3GgufExperts *x)
{
    if (x->base) for (int L = 0; L < x->n_layers; L++) k3_aligned_free(x->base[L]);
    free(x->base); free(x->ebytes); free(x->o3); free(x->o2); free(x->qt); free(x->so);
    memset(x, 0, sizeof *x);
}

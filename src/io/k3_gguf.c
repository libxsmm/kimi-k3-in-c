/* SPDX-License-Identifier: Apache-2.0 */
/* k3_gguf.c - see k3_gguf.h. */
#define _GNU_SOURCE            /* O_DIRECT */

#include "k3_portable_io.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "k3_gguf.h"
#include "k3_gq.h"

#define MAXSHARD    1024
#define MAXSTR      ((int64_t)64 << 20)   /* longest string value accepted (chat templates) */
#define MAXKEY      4096
#define MAXTENSORS  (1 << 22)
#define MAXKV       (1 << 20)

/* ------------------------------------------------------------- buffered reader */
typedef struct {
    int            fd;
    int64_t        size, pos;          /* file size and logical read position */
    unsigned char *buf;
    int64_t        b0, blen;           /* file offset of buf[0] and bytes held */
    int            bad;
} Rd;

#define RDCAP ((int64_t)4 << 20)

static int rd_need(Rd *r, int64_t n)
{
    if (r->bad) return 0;
    if (n > RDCAP || r->pos + n > r->size) { r->bad = 1; return 0; }
    if (r->pos >= r->b0 && r->pos + n <= r->b0 + r->blen) return 1;
    r->b0 = r->pos;
    r->blen = 0;
    const int64_t want = r->size - r->pos < RDCAP ? r->size - r->pos : RDCAP;
    while (r->blen < want) {
        const ssize_t got = pread(r->fd, r->buf + r->blen, (size_t)(want - r->blen),
                                  (off_t)(r->b0 + r->blen));
        if (got <= 0) break;
        r->blen += got;
    }
    if (r->blen < n) { r->bad = 1; return 0; }
    return 1;
}

static void rd_bytes(Rd *r, void *dst, int64_t n)
{
    if (!rd_need(r, n)) { memset(dst, 0, (size_t)n); return; }
    memcpy(dst, r->buf + (r->pos - r->b0), (size_t)n);
    r->pos += n;
}

static uint64_t rd_u64(Rd *r) { uint64_t v; rd_bytes(r, &v, 8); return v; }
static uint32_t rd_u32(Rd *r) { uint32_t v; rd_bytes(r, &v, 4); return v; }

/* A string: kept (malloc, NUL-terminated) when keep, else skipped. */
static char *rd_str(Rd *r, int keep, int64_t maxlen)
{
    const uint64_t n = rd_u64(r);
    if (r->bad || n > (uint64_t)maxlen || (int64_t)n > r->size - r->pos) { r->bad = 1; return NULL; }
    if (!keep) { r->pos += (int64_t)n; return NULL; }
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) { r->bad = 1; return NULL; }
    int64_t done = 0;
    while (done < (int64_t)n) {
        const int64_t c = (int64_t)n - done < RDCAP ? (int64_t)n - done : RDCAP;
        rd_bytes(r, s + done, c);
        done += c;
    }
    s[n] = 0;
    return s;
}

static int vt_size(int vt)
{
    switch (vt) {
    case K3_GV_U8: case K3_GV_I8: case K3_GV_BOOL: return 1;
    case K3_GV_U16: case K3_GV_I16: return 2;
    case K3_GV_U32: case K3_GV_I32: case K3_GV_F32: return 4;
    case K3_GV_U64: case K3_GV_I64: case K3_GV_F64: return 8;
    default: return 0;
    }
}

/* One numeric scalar as (integer, double). */
static void rd_num(Rd *r, int vt, int64_t *iv, double *fv)
{
    unsigned char b[8] = { 0 };
    rd_bytes(r, b, vt_size(vt));
    int64_t i = 0; double f = 0;
    switch (vt) {
    case K3_GV_U8:  case K3_GV_BOOL: i = b[0]; break;
    case K3_GV_I8:  i = (int8_t)b[0]; break;
    case K3_GV_U16: { uint16_t v; memcpy(&v, b, 2); i = v; } break;
    case K3_GV_I16: { int16_t v;  memcpy(&v, b, 2); i = v; } break;
    case K3_GV_U32: { uint32_t v; memcpy(&v, b, 4); i = v; } break;
    case K3_GV_I32: { int32_t v;  memcpy(&v, b, 4); i = v; } break;
    case K3_GV_U64: { uint64_t v; memcpy(&v, b, 8); i = (int64_t)v; } break;
    case K3_GV_I64: { int64_t v;  memcpy(&v, b, 8); i = v; } break;
    case K3_GV_F32: { float v;    memcpy(&v, b, 4); f = v; i = (int64_t)v; } break;
    case K3_GV_F64: { double v;   memcpy(&v, b, 8); f = v; i = (int64_t)v; } break;
    }
    if (vt != K3_GV_F32 && vt != K3_GV_F64) f = (double)i;
    *iv = i; *fv = f;
}

static int rd_kv(Rd *r, K3GKv *kv, int keep)
{
    memset(kv, 0, sizeof *kv);
    kv->key = rd_str(r, keep, MAXKEY);
    kv->vt = (int)rd_u32(r);
    if (r->bad) return -1;
    if (kv->vt == K3_GV_STR) {
        kv->s = rd_str(r, keep, MAXSTR);
    } else if (kv->vt == K3_GV_ARR) {
        kv->at = (int)rd_u32(r);
        const uint64_t n = rd_u64(r);
        if (r->bad || n > (uint64_t)1 << 40) return -1;
        kv->na = (int64_t)n;
        if (kv->at == K3_GV_STR) {
            for (uint64_t k = 0; k < n && !r->bad; k++) rd_str(r, 0, MAXSTR);
        } else {
            const int es = vt_size(kv->at);
            if (!es || (int64_t)n > (r->size - r->pos) / es) return -1;
            if (keep && n <= K3_GGUF_MAXARR) {
                kv->a = (int64_t *)malloc(sizeof(int64_t) * (n ? n : 1));
                if (!kv->a) return -1;
                for (uint64_t k = 0; k < n; k++) { double f; rd_num(r, kv->at, &kv->a[k], &f); }
            } else {
                r->pos += (int64_t)n * es;
            }
        }
    } else if (vt_size(kv->vt)) {
        rd_num(r, kv->vt, &kv->i, &kv->f);
    } else {
        return -1;
    }
    return r->bad ? -1 : 0;
}

/* ------------------------------------------------------------------------ index */
static uint64_t fnv1a(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; }
    return h;
}

static int build_index(K3Gguf *g)
{
    int nb = 1;
    while (nb < 2 * g->nt + 2) nb <<= 1;
    g->bucket = (int32_t *)malloc(sizeof(int32_t) * (size_t)nb);
    if (!g->bucket) return -1;
    for (int i = 0; i < nb; i++) g->bucket[i] = -1;
    g->nbucket = nb;
    for (int i = 0; i < g->nt; i++) {
        int j = (int)(fnv1a(g->t[i].name) & (uint64_t)(nb - 1));
        while (g->bucket[j] >= 0) {
            if (!strcmp(g->t[g->bucket[j]].name, g->t[i].name)) {
                fprintf(stderr, "k3_gguf: duplicate tensor %s\n", g->t[i].name);
                return -1;
            }
            j = (j + 1) & (nb - 1);
        }
        g->bucket[j] = i;
    }
    return 0;
}

const K3GTensor *k3_gguf_find(const K3Gguf *g, const char *name)
{
    if (!g->bucket) return NULL;
    int j = (int)(fnv1a(name) & (uint64_t)(g->nbucket - 1));
    while (g->bucket[j] >= 0) {
        const K3GTensor *t = &g->t[g->bucket[j]];
        if (!strcmp(t->name, name)) return t;
        j = (j + 1) & (g->nbucket - 1);
    }
    return NULL;
}

const K3GKv *k3_gguf_kv(const K3Gguf *g, const char *key)
{
    for (int i = 0; i < g->nkv; i++)
        if (g->kv[i].key && !strcmp(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

/* ------------------------------------------------------------------------ open */
static int parse_shard(K3Gguf *g, int shard, int keep_kv, int *cap)
{
    const char *path = g->st.path[shard];
    Rd r; memset(&r, 0, sizeof r);
    r.fd = g->st.fd[shard];
    struct stat sb;
    if (fstat(r.fd, &sb) != 0) { perror(path); return -1; }
    r.size = (int64_t)sb.st_size;
    r.buf = (unsigned char *)malloc((size_t)RDCAP);
    if (!r.buf) return -1;

    char magic[4];
    rd_bytes(&r, magic, 4);
    const uint32_t ver = rd_u32(&r);
    const uint64_t nt = rd_u64(&r), nkv = rd_u64(&r);
    if (r.bad || memcmp(magic, "GGUF", 4) || ver < 2 || ver > 3 ||
        nt > MAXTENSORS || nkv > MAXKV) {
        fprintf(stderr, "k3_gguf: %s is not a GGUF v2/v3 file this reader accepts\n", path);
        free(r.buf); return -1;
    }
    int64_t align = 32;
    K3GKv *kv = keep_kv ? (K3GKv *)calloc(nkv ? nkv : 1, sizeof(K3GKv)) : NULL;
    if (keep_kv && !kv) { free(r.buf); return -1; }
    if (keep_kv) { g->kv = kv; g->nkv = (int)nkv; }
    for (uint64_t k = 0; k < nkv; k++) {
        K3GKv tmp, *dst = keep_kv ? &kv[k] : &tmp;
        /* keys are needed even when values are not, for general.alignment */
        if (rd_kv(&r, dst, 1) != 0) {
            fprintf(stderr, "k3_gguf: %s: malformed metadata entry %llu\n", path, (unsigned long long)k);
            free(r.buf); return -1;
        }
        if (dst->key && !strcmp(dst->key, "general.alignment") && dst->i > 0 && dst->i <= 65536 &&
            !(dst->i & (dst->i - 1)))
            align = dst->i;
        if (!keep_kv) { free(tmp.key); free(tmp.s); free(tmp.a); }
    }

    const int first = g->nt;
    if (g->nt + (int64_t)nt > *cap) {
        int nc = *cap ? *cap : 1024;
        while (nc < g->nt + (int64_t)nt) nc *= 2;
        K3GTensor *t = (K3GTensor *)realloc(g->t, sizeof(K3GTensor) * (size_t)nc);
        if (!t) { free(r.buf); return -1; }
        g->t = t; *cap = nc;
    }
    for (uint64_t k = 0; k < nt; k++) {
        K3GTensor *t = &g->t[g->nt];
        memset(t, 0, sizeof *t);
        t->name = rd_str(&r, 1, 1024);
        t->ndim = (int)rd_u32(&r);
        if (r.bad || !t->name || t->ndim < 1 || t->ndim > 4) {
            fprintf(stderr, "k3_gguf: %s: malformed tensor info %llu\n", path, (unsigned long long)k);
            free(t->name); free(r.buf); return -1;
        }
        g->nt++;
        int64_t numel = 1;
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (int d = 0; d < t->ndim; d++) {
            const uint64_t ne = rd_u64(&r);
            if (ne == 0 || ne > ((uint64_t)1 << 40) || numel > ((int64_t)1 << 42) / (int64_t)ne) {
                fprintf(stderr, "k3_gguf: %s: tensor %s has an implausible shape\n", path, t->name);
                free(r.buf); return -1;
            }
            t->ne[d] = (int64_t)ne;
            numel *= (int64_t)ne;
        }
        t->type = (int)rd_u32(&r);
        t->off = (int64_t)rd_u64(&r);
        t->shard = shard;
        const size_t rb = k3_gq_row_bytes(t->type, t->ne[0]);
        t->nbytes = rb ? (int64_t)rb * (numel / t->ne[0]) : -1;
    }
    if (r.bad) { fprintf(stderr, "k3_gguf: %s: truncated header\n", path); free(r.buf); return -1; }
    const int64_t data = (r.pos + align - 1) / align * align;
    for (int i = first; i < g->nt; i++) {
        K3GTensor *t = &g->t[i];
        if (t->off < 0 || t->off > r.size || (t->nbytes >= 0 && t->nbytes > r.size - data - t->off)) {
            fprintf(stderr, "k3_gguf: %s: tensor %s lies outside the file\n", path, t->name);
            free(r.buf); return -1;
        }
        t->off += data;
    }
    free(r.buf);
    return 0;
}

/* NAME-00001-of-00013.gguf -> prefix "NAME-", count 13; else count 1. */
static int shard_pattern(const char *path, size_t *plen, int *count)
{
    const size_t n = strlen(path);
    const char *tail = "-00000-of-00000.gguf";
    const size_t tl = strlen(tail);
    *count = 1;
    if (n < tl) return 0;
    const char *s = path + n - tl;
    for (size_t i = 0; i < tl; i++) {
        const char want = tail[i];
        if (want == '0' ? (s[i] < '0' || s[i] > '9') : s[i] != want) return 0;
    }
    *plen = (size_t)(s - path) + 1;
    *count = atoi(s + 10);
    return *count >= 1 && *count <= MAXSHARD;
}

int k3_gguf_open(K3Gguf *g, const char *path)
{
    memset(g, 0, sizeof *g);
    size_t plen = 0;
    int ns = 1;
    const int split = shard_pattern(path, &plen, &ns);
    g->st.nshard = ns;
    g->st.fd = (int *)malloc(sizeof(int) * (size_t)ns);
    g->st.dfd = (int *)malloc(sizeof(int) * (size_t)ns);
    g->st.path = (char **)calloc((size_t)ns, sizeof(char *));
    if (!g->st.fd || !g->st.dfd || !g->st.path) { k3_gguf_close(g); return -1; }
    for (int i = 0; i < ns; i++) g->st.fd[i] = g->st.dfd[i] = -1;
    int cap = 0;
    for (int i = 0; i < ns; i++) {
        const size_t len = (split ? plen : strlen(path)) + 32;
        g->st.path[i] = (char *)malloc(len);
        if (!g->st.path[i]) { k3_gguf_close(g); return -1; }
        if (split) snprintf(g->st.path[i], len, "%.*s%05d-of-%05d.gguf", (int)plen, path, i + 1, ns);
        else snprintf(g->st.path[i], len, "%s", path);
        g->st.fd[i] = open(g->st.path[i], O_RDONLY);
        if (g->st.fd[i] < 0) { perror(g->st.path[i]); k3_gguf_close(g); return -1; }
        g->st.dfd[i] = open(g->st.path[i], O_RDONLY | O_DIRECT);
        if (parse_shard(g, i, i == 0, &cap) != 0) { k3_gguf_close(g); return -1; }
    }
    if (build_index(g) != 0) { k3_gguf_close(g); return -1; }
    const K3GKv *a = k3_gguf_kv(g, "general.architecture");
    if (a && a->s) snprintf(g->arch, sizeof g->arch, "%s", a->s);
    return 0;
}

void k3_gguf_close(K3Gguf *g)
{
    for (int i = 0; i < g->nt; i++) free(g->t[i].name);
    free(g->t);
    for (int i = 0; i < g->nkv; i++) { free(g->kv[i].key); free(g->kv[i].s); free(g->kv[i].a); }
    free(g->kv);
    free(g->bucket);
    k3_st_close(&g->st);
    memset(g, 0, sizeof *g);
}

int k3_gguf_read(const K3Gguf *g, int shard, int64_t off, int64_t n, void *dst)
{
    if (n <= 0) return 0;
    const int64_t cap = n + 2 * K3_ST_ALIGN;
    unsigned char *b = NULL;
    if (posix_memalign((void **)&b, K3_ST_ALIGN, (size_t)cap) != 0) return -1;
    int64_t pad = 0;
    const int ok = k3_st_read_par(&g->st, shard, off, n, b, cap, &pad) == n;
    if (ok) memcpy(dst, b + pad, (size_t)n);
    k3_aligned_free(b);
    return ok ? 0 : -1;
}

/* ------------------------------------------------------------------------ config */
typedef struct { const K3Gguf *g; const char *missing[40]; int nmissing; } CfgSrc;

static const K3GKv *cfg_kv(CfgSrc *s, const char *key)
{
    char full[160];
    snprintf(full, sizeof full, "%s.%s", s->g->arch, key);
    const K3GKv *v = k3_gguf_kv(s->g, full);
    if (!v || v->vt == K3_GV_STR || v->vt == K3_GV_ARR) {
        if (s->nmissing < 40) s->missing[s->nmissing] = key;
        s->nmissing++;
        return NULL;
    }
    return v;
}
static int   cfg_i(CfgSrc *s, const char *k) { const K3GKv *v = cfg_kv(s, k); return v ? (int)v->i : 0; }
static float cfg_f(CfgSrc *s, const char *k) { const K3GKv *v = cfg_kv(s, k); return v ? (float)v->f : 0.0f; }

int k3_gguf_cfg(const K3Gguf *g, K3Cfg *c, int *fa, int fa_max)
{
    memset(c, 0, sizeof *c);
    if (strcmp(g->arch, "kimi-k3")) {
        fprintf(stderr, "k3_gguf: architecture is '%s', this engine runs 'kimi-k3'\n", g->arch);
        return 0;
    }
    CfgSrc s = { g, { 0 }, 0 };
    c->hidden        = cfg_i(&s, "embedding_length");
    c->n_layers      = cfg_i(&s, "block_count");
    c->vocab         = cfg_i(&s, "vocab_size");
    c->rms_eps       = cfg_f(&s, "attention.layer_norm_rms_epsilon");
    c->n_heads       = cfg_i(&s, "attention.head_count");
    c->kda_heads     = c->n_heads;
    c->kda_head_dim  = cfg_i(&s, "kda.head_dim");
    c->conv_k        = cfg_i(&s, "ssm.conv_kernel");
    c->gate_lb       = cfg_f(&s, "kda.gate_lower_bound");
    c->q_lora        = cfg_i(&s, "attention.q_lora_rank");
    c->kv_lora       = cfg_i(&s, "attention.kv_lora_rank");
    c->qk_rope       = cfg_i(&s, "rope.dimension_count");
    c->qk_nope       = cfg_i(&s, "attention.key_length_mla") - c->qk_rope;
    c->v_head        = cfg_i(&s, "attention.value_length_mla");
    c->n_experts     = cfg_i(&s, "expert_count");
    c->topk          = cfg_i(&s, "expert_used_count");
    c->n_shared      = cfg_i(&s, "expert_shared_count");
    c->latent        = cfg_i(&s, "expert_latent_length");
    c->moe_inter     = cfg_i(&s, "expert_feed_forward_length");
    c->routed_scale  = cfg_f(&s, "expert_weights_scale");
    c->moe_renorm    = cfg_i(&s, "expert_weights_norm");
    c->first_dense   = cfg_i(&s, "leading_dense_block_count");
    c->dense_inter   = cfg_i(&s, "feed_forward_length");
    c->attn_res_block = cfg_i(&s, "attn_res.block_size");
    c->situ_b1       = cfg_f(&s, "activation.situ_beta");
    c->situ_b2       = cfg_f(&s, "activation.situ_linear_beta");
    const int gating = cfg_i(&s, "expert_gating_func");
    if (s.nmissing) {
        fprintf(stderr, "k3_gguf: missing %d required metadata key(s):\n", s.nmissing);
        for (int i = 0; i < s.nmissing && i < 40; i++) fprintf(stderr, "    %s.%s\n", g->arch, s.missing[i]);
        return 0;
    }
    if (gating != 2) {
        fprintf(stderr, "k3_gguf: expert_gating_func %d, the engine implements sigmoid (2)\n", gating);
        return 0;
    }

    /* head_count_kv per layer: 0 marks a KDA layer, MLA layers are served as one KV head */
    char key[160];
    snprintf(key, sizeof key, "%s.attention.head_count_kv", g->arch);
    const K3GKv *hk = k3_gguf_kv(g, key);
    if (!hk || hk->vt != K3_GV_ARR || !hk->a || hk->na != c->n_layers) {
        fprintf(stderr, "k3_gguf: %s must be a per-layer array of %d values\n", key, c->n_layers);
        return 0;
    }
    int nfa = 0;
    for (int L = 0; L < c->n_layers; L++)
        if (hk->a[L]) {
            if (nfa >= fa_max) { fprintf(stderr, "k3_gguf: too many MLA layers\n"); return 0; }
            fa[nfa++] = L + 1;
        }
    c->n_full_attn = nfa;
    c->full_attn = fa;

    /* Optional pieces of the architecture are recorded by their tensors' presence. */
    c->mla_out_gate = 0;
    c->latent_norm = 0;
    for (int L = 0; L < c->n_layers; L++) {
        snprintf(key, sizeof key, "blk.%d.attn_gate.weight", L);
        if (k3_gguf_find(g, key)) c->mla_out_gate = 1;
        snprintf(key, sizeof key, "blk.%d.ffn_routed_norm.weight", L);
        if (k3_gguf_find(g, key)) c->latent_norm = 1;
    }

    if (c->n_layers <= 0 || c->hidden <= 0 || c->vocab <= 0 || nfa == 0 || nfa >= c->n_layers ||
        c->topk <= 0 || c->topk > K3_MAX_TOPK || c->topk > c->n_experts || c->attn_res_block <= 0 ||
        c->conv_k < 1 || c->qk_nope <= 0 || c->kda_head_dim <= 0) {
        fprintf(stderr, "k3_gguf: metadata does not describe a usable kimi-k3 model\n");
        return 0;
    }
    printf("config: gguf (%s) | hidden=%d layers=%d vocab=%d | %d MLA + %d KDA | "
           "experts %d top%d shared%d | latent=%d\n", g->arch, c->hidden, c->n_layers, c->vocab,
           c->n_full_attn, c->n_layers - c->n_full_attn, c->n_experts, c->topk, c->n_shared, c->latent);
    return 1;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* k3_dspark.c - see k3_dspark.h. */
#define _GNU_SOURCE
#include "k3_dspark.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
#define K3D_AVX512 1
#include <immintrin.h>
#endif

#include "k3.h"
#include "k3_gq.h"
#include "k3_st.h"

static double k3d_now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static inline float bf(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

static float h2f(uint16_t h)
{
    const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    union { uint32_t u; float f; } v;
    if (e == 0) { v.f = ldexpf((float)m, -24); v.u |= s; return v.f; }
    v.u = s | (e == 31 ? 0x7f800000u | (m << 13) : ((e + 112) << 23) | (m << 13));
    return v.f;
}

/* ------------------------------------------------------------- config.json */
/* The value after "key": in a flat JSON object (first match). */
static const char *js_find(const char *js, const char *key)
{
    char pat[128];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) return NULL;
    p = strchr(p + strlen(pat), ':');
    return p ? p + 1 : NULL;
}

static double js_num(const char *js, const char *key, double dflt)
{
    const char *p = js_find(js, key);
    return p ? strtod(p, NULL) : dflt;
}

static int js_ints(const char *js, const char *key, int *out, int max)
{
    const char *p = js_find(js, key);
    if (!p || !(p = strchr(p, '['))) return 0;
    int n = 0;
    for (p++; *p && *p != ']' && n < max; ) {
        char *e;
        long v = strtol(p, &e, 10);
        if (e == p) { p++; continue; }
        out[n++] = (int)v;
        p = e;
    }
    return n;
}

static char *read_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = (char *)malloc((size_t)n + 1);
    if (s && fread(s, 1, (size_t)n, f) != (size_t)n) { free(s); s = NULL; }
    if (s) s[n] = 0;
    fclose(f);
    return s;
}

/* ------------------------------------------------------------------ kernels */
/* Y[t][o] = W[o] . X[t] for o in [0, out), t < T; W bf16 [out][in], X fp32 [T][in]. */
#if defined(K3D_AVX512)
static inline __m512 bf16x16(const uint16_t *p)
{
    return _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}

#define K3D_GEMM_BF16_T(TT)                                                              \
static void gemm_bf16_##TT(float *Y, int ldy, const float *X, int ldx,                   \
                           const uint16_t *W, int in, int o0, int o1)                    \
{                                                                                         \
    for (int o = o0; o < o1; o++) {                                                       \
        const uint16_t *w = W + (size_t)o * in;                                           \
        __m512 a[TT];                                                                     \
        for (int t = 0; t < TT; t++) a[t] = _mm512_setzero_ps();                          \
        int i = 0;                                                                        \
        for (; i + 16 <= in; i += 16) {                                                   \
            _mm_prefetch((const char *)(w + i) + 1024, _MM_HINT_T0);                     \
            const __m512 wv = bf16x16(w + i);                                             \
            for (int t = 0; t < TT; t++)                                                  \
                a[t] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(X + (size_t)t * ldx + i), a[t]); \
        }                                                                                 \
        for (int t = 0; t < TT; t++) {                                                    \
            float s = _mm512_reduce_add_ps(a[t]);                                         \
            for (int j = i; j < in; j++) s += bf(w[j]) * X[(size_t)t * ldx + j];          \
            Y[(size_t)t * ldy + o] = s;                                                   \
        }                                                                                 \
    }                                                                                     \
}
K3D_GEMM_BF16_T(1) K3D_GEMM_BF16_T(2) K3D_GEMM_BF16_T(3) K3D_GEMM_BF16_T(4)
K3D_GEMM_BF16_T(5) K3D_GEMM_BF16_T(6) K3D_GEMM_BF16_T(7) K3D_GEMM_BF16_T(8)

/* the same for Q8_0 rows (34 B per 32 weights: fp16 scale, 32 x int8) */
#define K3D_GEMM_Q80_T(TT)                                                               \
static void gemm_q80_##TT(float *Y, int ldy, const float *X, int ldx,                    \
                          const unsigned char *W, int in, int o0, int o1, int obase)     \
{                                                                                         \
    const size_t rb = (size_t)in / 32 * 34;                                               \
    for (int o = o0; o < o1; o++) {                                                       \
        const unsigned char *b = W + (size_t)(o - obase) * rb;                            \
        __m512 a[TT];                                                                     \
        for (int t = 0; t < TT; t++) a[t] = _mm512_setzero_ps();                          \
        for (int i = 0; i < in; i += 32, b += 34) {                                       \
            _mm_prefetch((const char *)b + 512, _MM_HINT_T0);                            \
            uint16_t dh; memcpy(&dh, b, 2);                                               \
            const __m512 d = _mm512_set1_ps(_cvtsh_ss(dh));                               \
            const __m512 w0 = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(   \
                _mm_loadu_si128((const __m128i *)(b + 2)))));                             \
            const __m512 w1 = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(   \
                _mm_loadu_si128((const __m128i *)(b + 18)))));                            \
            for (int t = 0; t < TT; t++) {                                                \
                a[t] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(X + (size_t)t * ldx + i), a[t]); \
                a[t] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(X + (size_t)t * ldx + i + 16), a[t]); \
            }                                                                             \
        }                                                                                 \
        for (int t = 0; t < TT; t++) Y[(size_t)t * ldy + o] = _mm512_reduce_add_ps(a[t]); \
    }                                                                                     \
}
K3D_GEMM_Q80_T(1) K3D_GEMM_Q80_T(2) K3D_GEMM_Q80_T(3) K3D_GEMM_Q80_T(4)
K3D_GEMM_Q80_T(5) K3D_GEMM_Q80_T(6) K3D_GEMM_Q80_T(7) K3D_GEMM_Q80_T(8)
#endif

static void gemm_bf16_rows(float *Y, int ldy, const float *X, int ldx, int T,
                           const uint16_t *W, int in, int o0, int o1)
{
#if defined(K3D_AVX512)
    if (in % 16 == 0) {
        switch (T) {
        case 1: gemm_bf16_1(Y, ldy, X, ldx, W, in, o0, o1); return;
        case 2: gemm_bf16_2(Y, ldy, X, ldx, W, in, o0, o1); return;
        case 3: gemm_bf16_3(Y, ldy, X, ldx, W, in, o0, o1); return;
        case 4: gemm_bf16_4(Y, ldy, X, ldx, W, in, o0, o1); return;
        case 5: gemm_bf16_5(Y, ldy, X, ldx, W, in, o0, o1); return;
        case 6: gemm_bf16_6(Y, ldy, X, ldx, W, in, o0, o1); return;
        case 7: gemm_bf16_7(Y, ldy, X, ldx, W, in, o0, o1); return;
        case 8: gemm_bf16_8(Y, ldy, X, ldx, W, in, o0, o1); return;
        default: break;
        }
    }
#endif
    for (int o = o0; o < o1; o++)
        for (int t = 0; t < T; t++) {
            const uint16_t *w = W + (size_t)o * in;
            const float *x = X + (size_t)t * ldx;
            float s = 0.0f;
            for (int i = 0; i < in; i++) s += bf(w[i]) * x[i];
            Y[(size_t)t * ldy + o] = s;
        }
}

static void gemm_q80_rows(float *Y, int ldy, const float *X, int ldx, int T,
                          const unsigned char *W, int in, int o0, int o1, int obase)
{
#if defined(K3D_AVX512)
    switch (T) {
    case 1: gemm_q80_1(Y, ldy, X, ldx, W, in, o0, o1, obase); return;
    case 2: gemm_q80_2(Y, ldy, X, ldx, W, in, o0, o1, obase); return;
    case 3: gemm_q80_3(Y, ldy, X, ldx, W, in, o0, o1, obase); return;
    case 4: gemm_q80_4(Y, ldy, X, ldx, W, in, o0, o1, obase); return;
    case 5: gemm_q80_5(Y, ldy, X, ldx, W, in, o0, o1, obase); return;
    case 6: gemm_q80_6(Y, ldy, X, ldx, W, in, o0, o1, obase); return;
    case 7: gemm_q80_7(Y, ldy, X, ldx, W, in, o0, o1, obase); return;
    case 8: gemm_q80_8(Y, ldy, X, ldx, W, in, o0, o1, obase); return;
    default: break;
    }
#endif
    const size_t rb = (size_t)in / 32 * 34;
    for (int o = o0; o < o1; o++) {
        const unsigned char *row = W + (size_t)(o - obase) * rb;
        for (int t = 0; t < T; t++) {
            const float *x = X + (size_t)t * ldx;
            float s = 0.0f;
            for (int i = 0; i < in; i += 32) {
                const unsigned char *b = row + (size_t)i / 32 * 34;
                uint16_t dh; memcpy(&dh, b, 2);
                const float d = h2f(dh);
                for (int j = 0; j < 32; j++) s += d * (float)(int8_t)b[2 + j] * x[i + j];
            }
            Y[(size_t)t * ldy + o] = s;
        }
    }
}

/* Y[T][out] = X[T][in] W^T, T any: passes of up to K3_DSPARK_MAXT tokens, rows split
 * over the threads and, under TP, over the ranks (then gathered, so every rank holds Y). */
static void gemm_bf16(float *Y, const float *X, int T, const uint16_t *W, int in, int out)
{
    int r0 = 0, r1 = out;
    if (k3_tp.size > 1 && out >= 64 * k3_tp.size && !getenv("K3_DSPARK_REDUNDANT"))
        k3_tp_part(out, &r0, &r1);
    for (int t0 = 0; t0 < T; t0 += K3_DSPARK_MAXT) {
        const int nt = T - t0 < K3_DSPARK_MAXT ? T - t0 : K3_DSPARK_MAXT;
        #pragma omp parallel
        {
            int lo, hi;
            k3_split(r1 - r0, &lo, &hi);
            gemm_bf16_rows(Y + (size_t)t0 * out, out, X + (size_t)t0 * in, in, nt, W, in,
                           r0 + lo, r0 + hi);
        }
        if (r1 - r0 < out) {
            K3Seg s[K3_DSPARK_MAXT];
            for (int t = 0; t < nt; t++) {
                s[t].p = Y + (size_t)(t0 + t) * out; s[t].n = out; s[t].unit = 1; s[t].exact = 1;
            }
            k3_tp_gather(s, nt);
        }
    }
}

static void rmsnorm_bf(float *y, const float *x, const uint16_t *w, int n, float eps)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    const float inv = (float)(1.0 / sqrt(s / n + eps));
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * bf(w[i]);
}

/* interleaved (non-neox) rotary embedding of x[rope] at position pos */
static void rope_apply(const K3DSpark *d, float *x, int pos)
{
    for (int i = 0; i < d->rope / 2; i++) {
        const float f = (float)pos * d->inv_freq[i];
        const float c = cosf(f) * d->rope_mag, s = sinf(f) * d->rope_mag;
        const float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i]     = a * c - b * s;
        x[2 * i + 1] = b * c + a * s;
    }
}

/* DeepSeek YaRN inverse frequencies (vLLM DeepseekScalingRotaryEmbedding); the cos/sin
 * magnitude scale is yarn_get_mscale(f, mscale) / yarn_get_mscale(f, mscale_all_dim). */
static void yarn_init(K3DSpark *d, double base, double factor, double orig, double bfast,
                      double bslow)
{
    const int dim = d->rope;
    const double lowd = dim * log(orig / (bfast * 2 * M_PI)) / (2 * log(base));
    const double highd = dim * log(orig / (bslow * 2 * M_PI)) / (2 * log(base));
    double low = floor(lowd), high = ceil(highd);
    if (low < 0) low = 0;
    if (high > dim - 1) high = dim - 1;
    if (low == high) high += 0.001;
    for (int i = 0; i < dim / 2; i++) {
        const float pf = (float)pow(base, (2.0 * i) / dim);
        const float extra = 1.0f / pf;
        const float inter = factor > 1.0 ? 1.0f / ((float)factor * pf) : extra;
        double ramp = (i - low) / (high - low);
        if (ramp < 0) ramp = 0;
        if (ramp > 1) ramp = 1;
        const float mask = factor > 1.0 ? (float)(1.0 - ramp) : 1.0f;
        d->inv_freq[i] = inter * (1.0f - mask) + extra * mask;
    }
}

/* ------------------------------------------------------------------ loading */
static const uint16_t *tensor(const K3DSpark *d, const K3St *st, int64_t base, const char *name,
                              int64_t r, int64_t c)
{
    const K3Tensor *t = k3_st_find(st, name);
    if (!t) { fprintf(stderr, "dspark: tensor %s missing\n", name); return NULL; }
    const int64_t n = k3_st_numel(t);
    if (t->dtype != K3_DT_BF16 || n != r * c) {
        fprintf(stderr, "dspark: tensor %s has unexpected type or shape\n", name);
        return NULL;
    }
    return (const uint16_t *)((const unsigned char *)d->blob + (t->off - base));
}

int k3_dspark_open(K3DSpark *d, const char *dir, int vocab, int hidden, int cap)
{
    memset(d, 0, sizeof *d);
    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", dir);
    char *js = read_text(path);
    if (!js) { fprintf(stderr, "dspark: cannot read %s\n", path); return -1; }
    d->H = (int)js_num(js, "hidden_size", 0);
    d->nl = (int)js_num(js, "num_hidden_layers", 0);
    d->nh = (int)js_num(js, "num_attention_heads", 0);
    d->q_lora = (int)js_num(js, "q_lora_rank", 0);
    d->kv_lora = (int)js_num(js, "kv_lora_rank", 0);
    d->nope = (int)js_num(js, "qk_nope_head_dim", 0);
    d->rope = (int)js_num(js, "qk_rope_head_dim", 0);
    d->vh = (int)js_num(js, "v_head_dim", 0);
    d->inter = (int)js_num(js, "intermediate_size", 0);
    d->vocab = (int)js_num(js, "vocab_size", 0);
    d->mrank = (int)js_num(js, "markov_rank", 0);
    d->mask_id = (int)js_num(js, "mask_token_id", -1);
    d->eps = (float)js_num(js, "rms_norm_eps", 1e-5);
    d->ntgt = js_ints(js, "target_layer_ids", d->tgt, K3_DSPARK_MAXL);
    const double theta = js_num(js, "rope_theta", 10000.0);
    const char *rp = js_find(js, "rope_parameters");
    double factor = 1.0, orig = 4096, bfast = 32, bslow = 1, msc = 1.0, msc_all = 0.0;
    if (rp) {
        factor = js_num(rp, "factor", 1.0);
        orig = js_num(rp, "original_max_position_embeddings", 4096);
        bfast = js_num(rp, "beta_fast", 32);
        bslow = js_num(rp, "beta_slow", 1);
        msc = js_num(rp, "mscale", 1.0);
        msc_all = js_num(rp, "mscale_all_dim", 0.0);
    }
    free(js);
    if (d->nl < 1 || d->nl > K3_DSPARK_MAXL || d->ntgt < 1 || d->H != hidden ||
        d->vocab != vocab || d->rope > 128 || d->rope % 2 || d->mask_id < 0 ||
        d->kv_lora > 512 || d->mrank > 512 || d->mrank % 16 ||
        (d->kv_lora + d->rope) % 16 || d->H % 16) {
        fprintf(stderr, "dspark: config does not match the target (hidden %d/%d, vocab %d/%d)\n",
                d->H, hidden, d->vocab, vocab);
        return -1;
    }
    yarn_init(d, theta, factor, orig, bfast, bslow);
    d->scale = 1.0f / sqrtf((float)(d->nope + d->rope));
    d->rope_mag = 1.0f;
    if (factor > 1.0) {
        const float m = (float)(0.1 * msc_all * log(factor) + 1.0);
        d->scale *= m * m;
        d->rope_mag = (float)((0.1 * msc * log(factor) + 1.0) / (msc_all > 0 ? m : 1.0));
    }

    /* the whole data region in one parallel read, then pointers into it */
    K3St st;
    if (k3_st_open(&st, dir) != 0) { fprintf(stderr, "dspark: cannot open %s\n", dir); return -1; }
    int64_t lo = INT64_MAX, hi = 0;
    for (int i = 0; i < st.nt; i++) {
        if (st.t[i].shard != 0) { fprintf(stderr, "dspark: expected one safetensors file\n"); k3_st_close(&st); return -1; }
        if (st.t[i].off < lo) lo = st.t[i].off;
        if (st.t[i].off + st.t[i].nbytes > hi) hi = st.t[i].off + st.t[i].nbytes;
    }
    const int64_t a0 = lo / K3_ST_ALIGN * K3_ST_ALIGN;
    const int64_t cap_b = (hi - a0 + 2 * K3_ST_ALIGN + 4095) / 4096 * 4096;
    void *buf = mmap(NULL, (size_t)cap_b, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) { k3_st_close(&st); return -1; }
    madvise(buf, (size_t)cap_b, MADV_HUGEPAGE);
    int64_t poff = 0;
    const double t0 = k3d_now();
    if (k3_st_read_par(&st, 0, lo, hi - lo, buf, cap_b, &poff) != hi - lo) {
        fprintf(stderr, "dspark: weight read failed\n");
        munmap(buf, (size_t)cap_b); k3_st_close(&st); return -1;
    }
    d->blob = buf;
    d->blob_bytes = (size_t)cap_b;
    const int64_t base = lo - poff;       /* file offset that maps to blob[0] */
    const int H = d->H, NH = d->nh, KW = d->kv_lora + d->rope;
    int bad = 0;
#define T_(var, nm, r, c) do { if (!(var = tensor(d, &st, base, nm, r, c))) bad = 1; } while (0)
    T_(d->ctx_proj, "context_proj.weight", H, (int64_t)H * d->ntgt);
    T_(d->ctx_norm, "context_norm.weight", 1, H);
    T_(d->final_norm, "final_norm.weight", 1, H);
    T_(d->embed, "embed_tokens.weight", vocab, H);
    T_(d->mw1, "markov_head.markov_w1.weight", vocab, d->mrank);
    T_(d->mw2, "markov_head.markov_w2.weight", vocab, d->mrank);
    for (int l = 0; l < d->nl; l++) {
        char nm[256];
        K3DSparkLayer *L = &d->L[l];
#define TL_(var, suf, r, c) do { snprintf(nm, sizeof nm, "layers.%d.%s", l, suf); T_(var, nm, r, c); } while (0)
        TL_(L->in_norm, "input_layernorm.weight", 1, H);
        TL_(L->post_norm, "post_attention_layernorm.weight", 1, H);
        TL_(L->q_a, "self_attn.q_a_proj.weight", d->q_lora, H);
        TL_(L->q_a_norm, "self_attn.q_a_layernorm.weight", 1, d->q_lora);
        TL_(L->q_b, "self_attn.q_b_proj.weight", (int64_t)NH * (d->nope + d->rope), d->q_lora);
        TL_(L->kv_a, "self_attn.kv_a_proj_with_mqa.weight", KW, H);
        TL_(L->kv_a_norm, "self_attn.kv_a_layernorm.weight", 1, d->kv_lora);
        TL_(L->kv_b, "self_attn.kv_b_proj.weight", (int64_t)NH * (d->nope + d->vh), d->kv_lora);
        TL_(L->o, "self_attn.o_proj.weight", H, (int64_t)NH * d->vh);
        TL_(L->gate, "mlp.gate_proj.weight", d->inter, H);
        TL_(L->up, "mlp.up_proj.weight", d->inter, H);
        TL_(L->down, "mlp.down_proj.weight", H, d->inter);
#undef TL_
    }
#undef T_
    k3_st_close(&st);
    if (bad) { k3_dspark_close(d); return -1; }

    d->cap = cap;
    d->dump = getenv("K3_DSPARK_DUMP");
    d->lat = (float *)calloc((size_t)d->nl * cap * d->kv_lora, sizeof(float));
    d->kpe = (float *)calloc((size_t)d->nl * cap * d->rope, sizeof(float));
    const size_t T = K3_DSPARK_MAXT, MQ = (size_t)NH * (d->nope + d->rope);
    d->wk_floats = T * ((size_t)5 * H + (size_t)H * d->ntgt + d->q_lora + MQ + KW +
                        (size_t)NH * d->vh + 2 * (size_t)d->inter + (size_t)vocab) +
                   (size_t)vocab + 64;
    d->wk = (float *)malloc(d->wk_floats * sizeof(float));
    if (!d->lat || !d->kpe || !d->wk) { k3_dspark_close(d); return -1; }
    if (k3_tp.rank == 0)
        printf("dspark: %d layers, %d heads, target layers", d->nl, d->nh);
    for (int j = 0; j < d->ntgt && k3_tp.rank == 0; j++) printf(" %d", d->tgt[j]);
    if (k3_tp.rank == 0)
        printf(", %.2f GB loaded in %.1f s, softmax scale %.5f\n",
               (double)(hi - lo) / 1e9, k3d_now() - t0, d->scale);
    return 0;
}

void k3_dspark_close(K3DSpark *d)
{
    if (d->blob) munmap(d->blob, d->blob_bytes);
    free(d->lat); free(d->kpe); free(d->wk);
    memset(d, 0, sizeof *d);
}

/* ------------------------------------------------------------------ forward */
/* latent (normed) and k_pe (roped) of rows kv[n][kv_lora + rope] into layer l's cache */
static void kv_store(K3DSpark *d, int l, const float *kv, int n, int pos0)
{
    const int KW = d->kv_lora + d->rope;
    for (int t = 0; t < n; t++) {
        const int p = pos0 + t;
        float *lat = d->lat + ((size_t)l * d->cap + p) * d->kv_lora;
        float *kpe = d->kpe + ((size_t)l * d->cap + p) * d->rope;
        rmsnorm_bf(lat, kv + (size_t)t * KW, d->L[l].kv_a_norm, d->kv_lora, d->eps);
        memcpy(kpe, kv + (size_t)t * KW + d->kv_lora, (size_t)d->rope * sizeof(float));
        rope_apply(d, kpe, p);
    }
}

int k3_dspark_context(K3DSpark *d, const float *taps, int n, int pos0)
{
    if (pos0 < 0 || pos0 + n > d->cap) return -1;
    const double t0 = k3d_now();
    const int H = d->H, KW = d->kv_lora + d->rope, IN = H * d->ntgt;
    float *c = d->wk, *kv = c + (size_t)K3_DSPARK_MAXT * H;
    for (int t0c = 0; t0c < n; t0c += K3_DSPARK_MAXT) {
        const int nt = n - t0c < K3_DSPARK_MAXT ? n - t0c : K3_DSPARK_MAXT;
        gemm_bf16(c, taps + (size_t)t0c * IN, nt, d->ctx_proj, IN, H);
        for (int t = 0; t < nt; t++) rmsnorm_bf(c + (size_t)t * H, c + (size_t)t * H, d->ctx_norm, H, d->eps);
        for (int l = 0; l < d->nl; l++) {
            gemm_bf16(kv, c, nt, d->L[l].kv_a, H, KW);
            kv_store(d, l, kv, nt, pos0 + t0c);
        }
    }
    d->t_ctx += k3d_now() - t0;
    return 0;
}

/* One head of one query token: absorbed MLA over the cached context [0, pos) plus the
 * block's own nq rows, non-causal within the block. */
static void attend(const K3DSpark *d, int l, const uint16_t *kvb, const float *q,
                   const float *blat, const float *bkpe, int nq, int pos, float *out,
                   float *sc, float *ql, float *ol)
{
    const int KL = d->kv_lora, R = d->rope, NP = d->nope;
    for (int k = 0; k < KL; k++) ql[k] = 0.0f;
    for (int i = 0; i < NP; i++) {
        const float qi = q[i];
        const uint16_t *row = kvb + (size_t)i * KL;
        for (int k = 0; k < KL; k++) ql[k] += qi * bf(row[k]);
    }
    const float *qpe = q + NP;
    const float *lat = d->lat + (size_t)l * d->cap * KL, *kpe = d->kpe + (size_t)l * d->cap * R;
    const int ns = pos + nq;
    float m = -INFINITY;
    for (int s = 0; s < ns; s++) {
        const float *ls = s < pos ? lat + (size_t)s * KL : blat + (size_t)(s - pos) * KL;
        const float *ks = s < pos ? kpe + (size_t)s * R : bkpe + (size_t)(s - pos) * R;
        float a = 0.0f, b = 0.0f;
        for (int k = 0; k < KL; k++) a += ql[k] * ls[k];
        for (int k = 0; k < R; k++) b += qpe[k] * ks[k];
        sc[s] = (a + b) * d->scale;
        if (sc[s] > m) m = sc[s];
    }
    double z = 0.0;
    for (int s = 0; s < ns; s++) { sc[s] = expf(sc[s] - m); z += sc[s]; }
    const float iz = (float)(1.0 / z);
    for (int k = 0; k < KL; k++) ol[k] = 0.0f;
    for (int s = 0; s < ns; s++) {
        const float p = sc[s] * iz;
        const float *ls = s < pos ? lat + (size_t)s * KL : blat + (size_t)(s - pos) * KL;
        for (int k = 0; k < KL; k++) ol[k] += p * ls[k];
    }
    for (int j = 0; j < d->vh; j++) {
        const uint16_t *row = kvb + (size_t)(NP + j) * KL;
        float a = 0.0f;
        for (int k = 0; k < KL; k++) a += bf(row[k]) * ol[k];
        out[j] = a;
    }
}

static int argmax_f(const float *v, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

static void dump_f(const char *prefix, const char *suf, const void *p, size_t bytes)
{
    char path[4096];
    snprintf(path, sizeof path, "%s.%s", prefix, suf);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, 1, bytes, f);
    fclose(f);
}

int k3_dspark_propose(K3DSpark *d, int anchor, int pos, int nq, const void *lm_head,
                      int lm_wdt, int *out)
{
    if (nq < 1 || nq > K3_DSPARK_MAXT || pos < 0 || pos + nq > d->cap) return -1;
    const double t0 = k3d_now();
    const int H = d->H, NH = d->nh, KW = d->kv_lora + d->rope, QH = d->nope + d->rope;
    const int MQ = NH * QH, AO = NH * d->vh, I = d->inter, V = d->vocab;
    float *R  = d->wk;                       /* [nq][H] residual stream */
    float *hs = R  + (size_t)K3_DSPARK_MAXT * H;
    float *a  = hs + (size_t)K3_DSPARK_MAXT * H;
    float *qa = a  + (size_t)K3_DSPARK_MAXT * H;
    float *q  = qa + (size_t)K3_DSPARK_MAXT * d->q_lora;
    float *kv = q  + (size_t)K3_DSPARK_MAXT * MQ;
    float *ao = kv + (size_t)K3_DSPARK_MAXT * KW;
    float *g  = ao + (size_t)K3_DSPARK_MAXT * AO;
    float *u  = g  + (size_t)K3_DSPARK_MAXT * I;
    float *lg = u  + (size_t)K3_DSPARK_MAXT * I;   /* [nq][V] */
    float *bias = lg + (size_t)K3_DSPARK_MAXT * V; /* [V] */
    float blat[K3_DSPARK_MAXT * 512], bkpe[K3_DSPARK_MAXT * 128];
    if (d->kv_lora > 512) return -1;

    for (int t = 0; t < nq; t++) {
        const uint16_t *e = d->embed + (size_t)(t == 0 ? anchor : d->mask_id) * H;
        for (int i = 0; i < H; i++) R[(size_t)t * H + i] = bf(e[i]);
    }
    for (int l = 0; l < d->nl; l++) {
        const K3DSparkLayer *L = &d->L[l];
        if (l > 0)
            for (size_t i = 0; i < (size_t)nq * H; i++) R[i] += a[i];   /* previous MLP out */
        for (int t = 0; t < nq; t++) rmsnorm_bf(hs + (size_t)t * H, R + (size_t)t * H, L->in_norm, H, d->eps);
        gemm_bf16(qa, hs, nq, L->q_a, H, d->q_lora);
        for (int t = 0; t < nq; t++)
            rmsnorm_bf(qa + (size_t)t * d->q_lora, qa + (size_t)t * d->q_lora, L->q_a_norm, d->q_lora, d->eps);
        gemm_bf16(q, qa, nq, L->q_b, d->q_lora, MQ);
        gemm_bf16(kv, hs, nq, L->kv_a, H, KW);
        for (int t = 0; t < nq; t++) {
            rmsnorm_bf(blat + (size_t)t * d->kv_lora, kv + (size_t)t * KW, L->kv_a_norm, d->kv_lora, d->eps);
            memcpy(bkpe + (size_t)t * d->rope, kv + (size_t)t * KW + d->kv_lora, (size_t)d->rope * sizeof(float));
            rope_apply(d, bkpe + (size_t)t * d->rope, pos + t);
            for (int h = 0; h < NH; h++) rope_apply(d, q + (size_t)t * MQ + (size_t)h * QH + d->nope, pos + t);
        }
        #pragma omp parallel
        {
            float *sc = (float *)malloc((size_t)(pos + nq) * sizeof(float));
            float ql[512], ol[512];
            #pragma omp for schedule(dynamic, 1)
            for (int th = 0; th < nq * NH; th++) {
                const int t = th / NH, h = th % NH;
                attend(d, l, L->kv_b + (size_t)h * (d->nope + d->vh) * d->kv_lora,
                       q + (size_t)t * MQ + (size_t)h * QH, blat, bkpe, nq, pos,
                       ao + (size_t)t * AO + (size_t)h * d->vh, sc, ql, ol);
            }
            free(sc);
        }
        gemm_bf16(a, ao, nq, L->o, AO, H);
        for (size_t i = 0; i < (size_t)nq * H; i++) R[i] += a[i];
        for (int t = 0; t < nq; t++) rmsnorm_bf(hs + (size_t)t * H, R + (size_t)t * H, L->post_norm, H, d->eps);
        gemm_bf16(g, hs, nq, L->gate, H, I);
        gemm_bf16(u, hs, nq, L->up, H, I);
        for (size_t i = 0; i < (size_t)nq * I; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
        gemm_bf16(a, g, nq, L->down, I, H);
    }
    for (size_t i = 0; i < (size_t)nq * H; i++) R[i] += a[i];
    for (int t = 0; t < nq; t++) rmsnorm_bf(hs + (size_t)t * H, R + (size_t)t * H, d->final_norm, H, d->eps);

    /* base logits through the target's lm_head: this rank's vocab rows, then gathered */
    int v0 = 0, v1 = V;
    if (k3_tp.size > 1) k3_tp_part(V, &v0, &v1);
    const int obase = k3_tp.local ? v0 : 0;
    if (lm_wdt == K3_WQ8_0 && H % 32 == 0) {
        const size_t rb = k3_row_bytes(K3_WQ8_0, H);
        #pragma omp parallel
        {
            int lo, hi;
            k3_split(v1 - v0, &lo, &hi);
            if (k3_act_q8_on())
                k3_q80_rows_T(lg + v0 + lo, V, hs, H, nq,
                              (const unsigned char *)lm_head + (size_t)(v0 + lo - obase) * rb, H, 0, hi - lo);
            else
                gemm_q80_rows(lg, V, hs, H, nq, (const unsigned char *)lm_head, H, v0 + lo, v0 + hi, obase);
        }
    } else if (lm_wdt == K3_WBF16) {
        #pragma omp parallel
        {
            int lo, hi;
            k3_split(v1 - v0, &lo, &hi);
            gemm_bf16_rows(lg, V, hs, H, nq, (const uint16_t *)lm_head - (size_t)obase * H, H,
                           v0 + lo, v0 + hi);
        }
    } else {
        for (int t = 0; t < nq; t++)
            k3_mmw(lg + (size_t)t * V, hs + (size_t)t * H, lm_head, lm_wdt, H, V);
        v0 = 0; v1 = V;
    }
    const int sliced = k3_tp.size > 1 && (v0 != 0 || v1 != V);
    const int dump = d->dump && d->steps == 0 && k3_tp.rank == 0;
    if (sliced && d->dump) {
        K3Seg seg[K3_DSPARK_MAXT];
        for (int t = 0; t < nq; t++) { seg[t].p = lg + (size_t)t * V; seg[t].n = V; seg[t].unit = 1; seg[t].exact = 1; }
        k3_tp_gather(seg, nq);
    }

    /* sequential Markov stage: bias from the previously chosen token, greedy */
    if (dump) {
        dump_f(d->dump, "hs", hs, (size_t)nq * H * sizeof(float));
        dump_f(d->dump, "base", lg, (size_t)nq * V * sizeof(float));
        /* the multi-token lm_head against the engine's own GEMV, position 0 */
        k3_mmw_tp(bias, hs, lm_head, lm_wdt, H, V);
        double md = 0, mx = 0;
        for (int v = 0; v < V; v++) {
            md = fmax(md, fabs((double)bias[v] - lg[v]));
            mx = fmax(mx, fabs((double)bias[v]));
        }
        printf("dspark check: lm_head max|diff| %.3g (max|logit| %.3g)\n", md, mx);
    } else if (d->dump && d->steps == 0) {
        k3_mmw_tp(bias, hs, lm_head, lm_wdt, H, V);   /* the gather is collective */
    }
    int prev = anchor;
    /* with the logits still vocab-sliced, each rank takes its slice's argmax and the
     * (value, index) pairs are gathered: the first maximum overall, as argmax_f */
    const int m0 = sliced && !d->dump ? v0 : 0, m1 = sliced && !d->dump ? v1 : V;
    for (int i = 0; i < nq; i++) {
        float e[512];
        for (int r = 0; r < d->mrank; r++) e[r] = bf(d->mw1[(size_t)prev * d->mrank + r]);
        #pragma omp parallel
        {
            int lo, hi;
            k3_split(m1 - m0, &lo, &hi);
            gemm_bf16_rows(bias, V, e, d->mrank, 1, d->mw2, d->mrank, m0 + lo, m0 + hi);
        }
        float *li = lg + (size_t)i * V;
        for (int v = m0; v < m1; v++) li[v] += bias[v];
        const int best = m0 + argmax_f(li + m0, m1 - m0);
        if (m1 - m0 < V) {
            float pr[2 * 64];
            if (k3_tp.size > 64) return -1;
            pr[2 * k3_tp.rank] = li[best];
            pr[2 * k3_tp.rank + 1] = (float)best;
            const K3Seg s = { pr, 2 * k3_tp.size, 2, 1 };
            k3_tp_gather(&s, 1);
            int r = 0;
            for (int k = 1; k < k3_tp.size; k++) if (pr[2 * k] > pr[2 * r]) r = k;
            out[i] = prev = (int)pr[2 * r + 1];
        } else {
            out[i] = prev = best;
        }
    }
    if (dump) {
        dump_f(d->dump, "tok", out, (size_t)nq * sizeof(int));
        const int meta[3] = { anchor, pos, nq };
        dump_f(d->dump, "meta", meta, sizeof meta);
    }
    d->steps++;
    d->drafted += nq;
    d->t_draft += k3d_now() - t0;
    return nq;
}

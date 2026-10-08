/* test_gguf.c - GGUF reader and the GGUF weight-format kernels, with no model weights.
 *
 *   1. k3_q80/iq2xs/iq3xxs rows == dequantise + k3_matmul, to the bit
 *   2. the int8-activation kernels (when built) stay within int8 quantisation error
 *   3. a two-shard GGUF written here parses back: metadata, tensor index, offsets, bytes;
 *      and truncated or out-of-range headers are refused
 *
 *   test_gguf DIR      (DIR receives the scratch GGUF files)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "k3.h"
#include "k3_gguf.h"
#include "k3_gq.h"
#include "k3_amx.h"

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }

/* random block bytes with a finite, modest fp16 scale in each block's first two bytes */
static void fill(int t, unsigned char *p, int rows, int in)
{
    const size_t rb = k3_gq_row_bytes(t, in), bb = t == K3_GG_Q8_0 ? 34 : (t == K3_GG_IQ2_XS ? 74 : 98);
    for (size_t i = 0; i < rb * (size_t)rows; i++) p[i] = (unsigned char)rnd();
    for (size_t b = 0; b < rb * (size_t)rows / bb; b++) {
        const uint16_t h = (uint16_t)(0x1400 + (rnd() & 0x7ff));
        memcpy(p + b * bb, &h, 2);
    }
}

static void test_kernels(void)
{
    static const int types[3] = { K3_GG_Q8_0, K3_GG_IQ2_XS, K3_GG_IQ3_XXS };
    static const char *names[3] = { "q8_0", "iq2_xs", "iq3_xxs" };
    static const int ins[3] = { 3584, 3072, 7168 };
    k3_act_q8 = 0;
    for (int ti = 0; ti < 3; ti++)
        for (int s = 0; s < 3; s++) {
            const int t = types[ti], in = ins[s], rows = 96;
            unsigned char *W = (unsigned char *)malloc(k3_gq_row_bytes(t, in) * rows);
            float *Wd = (float *)malloc(sizeof(float) * (size_t)in * rows);
            float *x = (float *)malloc(sizeof(float) * in);
            float *y0 = (float *)calloc(rows, 4), *y1 = (float *)calloc(rows, 4), *y2 = (float *)calloc(rows, 4);
            int8_t *xq = (int8_t *)malloc(in);
            float *dx = (float *)malloc(sizeof(float) * (in / 256 + 1));
            fill(t, W, rows, in);
            for (int i = 0; i < in; i++) x[i] = (float)((int)(rnd() % 2001) - 1000) / 997.0f;
            k3_gq_dequant(t, W, Wd, (int64_t)in * rows);
            k3_matmul(y0, x, Wd, in, rows);
            if (t == K3_GG_Q8_0) { k3_q80_rows(y1, x, W, in, 0, rows); k3_matmul_q80(y2, x, W, in, rows); }
            else if (t == K3_GG_IQ2_XS) { k3_iq2xs_rows(y1, x, W, in, 0, rows); memcpy(y2, y1, 4 * rows); }
            else { k3_iq3xxs_rows(y1, x, W, in, 0, rows); memcpy(y2, y1, 4 * rows); }
            CHECK(!memcmp(y0, y1, 4 * (size_t)rows), "%s in=%d: fused rows differ from dequantise + k3_matmul", names[ti], in);
            CHECK(!memcmp(y0, y2, 4 * (size_t)rows), "%s in=%d: team matmul differs", names[ti], in);
            if (t == K3_GG_Q8_0)
                for (int T = 1; T <= 8; T++) {
                    float *X = (float *)malloc(sizeof(float) * (size_t)T * in);
                    float *Y = (float *)malloc(sizeof(float) * (size_t)T * rows);
                    for (int i = 0; i < T * in; i++) X[i] = (float)((int)(rnd() % 2001) - 1000) / 991.0f;
                    k3_q80_rows_T(Y, rows, X, in, T, W, in, 0, rows);
                    for (int u = 0; u < T; u++) {
                        k3_q80_rows(y1, X + (size_t)u * in, W, in, 0, rows);
                        CHECK(!memcmp(y1, Y + (size_t)u * rows, 4 * (size_t)rows),
                              "q8_0 in=%d T=%d token %d: multi-token rows differ", in, T, u);
                    }
                    free(X); free(Y);
                }
            if (t == K3_GG_Q8_0 && k3_amx_q8_ok()) {
                k3_act_q8 = 1;
                for (int T = 1; T <= 17; T += 2) {
                    float *X = (float *)malloc(sizeof(float) * (size_t)T * in);
                    float *Y = (float *)malloc(sizeof(float) * (size_t)T * rows);
                    for (int i = 0; i < T * in; i++) X[i] = (float)((int)(rnd() % 2001) - 1000) / 991.0f;
                    memcpy(X, x, sizeof(float) * in);
                    k3_q80_rows_T(Y, rows, X, in, T, W, in, 0, rows);
                    double md = 0, ny = 0;
                    for (int r = 0; r < rows; r++) { md = fmax(md, fabs((double)Y[r] - y0[r])); ny = fmax(ny, fabs(y0[r])); }
                    CHECK(md <= 2e-2 * ny, "q8_0 in=%d act-int8: max error %.3g of max |y| %.3g", in, md, ny);
                    for (int u = 0; u < T; u++) {
                        k3_q80_rows(y1, X + (size_t)u * in, W, in, 0, 37);
                        k3_q80_rows(y1, X + (size_t)u * in, W, in, 37, rows);
                        CHECK(!memcmp(y1, Y + (size_t)u * rows, 4 * (size_t)rows),
                              "q8_0 in=%d act-int8 T=%d token %d: batch differs from one token", in, T, u);
                    }
                    free(X); free(Y);
                }
                k3_act_q8 = 0;
            }
            if (t != K3_GG_Q8_0 && k3_gq_have_q8()) {
                k3_gq_quant_x(xq, dx, x, in);
                if (t == K3_GG_IQ2_XS) k3_iq2xs_rows_q8(y1, xq, dx, W, in, 0, rows);
                else k3_iq3xxs_rows_q8(y1, xq, dx, W, in, 0, rows);
                if (t == K3_GG_IQ2_XS)
                    for (int T = 1; T <= 6; T++) {
                        const int nb = in / 256;
                        int8_t *XQ = (int8_t *)malloc((size_t)T * in);
                        float *DX = (float *)malloc(sizeof(float) * (size_t)T * nb);
                        float *Y = (float *)malloc(sizeof(float) * (size_t)T * rows), *X = (float *)malloc(sizeof(float) * in);
                        for (int u = 0; u < T; u++) {
                            for (int i = 0; i < in; i++) X[i] = (float)((int)(rnd() % 2001) - 1000) / 993.0f;
                            k3_gq_quant_x(XQ + (size_t)u * in, DX + (size_t)u * nb, X, in);
                        }
                        float *yp[8]; const int8_t *xp[8]; const float *dp[8];
                        for (int u = 0; u < T; u++) { yp[u] = Y + (size_t)u * rows; xp[u] = XQ + (size_t)u * in; dp[u] = DX + (size_t)u * nb; }
                        k3_iq2xs_rows_q8_P(yp, xp, dp, T, W, in, 0, rows);
                        float *y3 = (float *)malloc(sizeof(float) * rows);
                        for (int u = 0; u < T; u++) {
                            k3_iq2xs_rows_q8(y3, XQ + (size_t)u * in, DX + (size_t)u * nb, W, in, 0, rows);
                            CHECK(!memcmp(y3, Y + (size_t)u * rows, 4 * (size_t)rows),
                                  "iq2_xs in=%d int8 T=%d token %d: multi-token rows differ", in, T, u);
                        }
                        free(XQ); free(DX); free(Y); free(X); free(y3);
                    }
                double md = 0, ny = 0;
                for (int r = 0; r < rows; r++) { md = fmax(md, fabs((double)y1[r] - y0[r])); ny = fmax(ny, fabs(y0[r])); }
                CHECK(md <= 2e-2 * ny, "%s in=%d int8: max error %.3g of max |y| %.3g", names[ti], in, md, ny);
            }
            free(W); free(Wd); free(x); free(y0); free(y1); free(y2); free(xq); free(dx);
        }
    printf("  kernels: q8_0, iq2_xs, iq3_xxs fused == dequantise + k3_matmul%s%s\n",
           k3_gq_have_q8() ? "; int8 paths within tolerance" : " (no int8 kernels in this build)",
           k3_amx_q8_ok() ? "; AMX-INT8 q8_0 batch == one token" : "");
}

/* ggml MXFP4 blocks repacked to the engine layout dequantise to the same values */
static void test_mxfp4(void)
{
    const int rows = 8, in = 3584, nb = in / 32;
    unsigned char *g = (unsigned char *)malloc((size_t)rows * nb * 17), *k = (unsigned char *)malloc((size_t)rows * nb * 17);
    float *a = (float *)malloc(sizeof(float) * rows * in), *b = (float *)malloc(sizeof(float) * rows * in);
    for (int i = 0; i < rows * nb; i++) {
        g[i * 17] = (unsigned char)(100 + rnd() % 40);
        for (int j = 1; j < 17; j++) g[i * 17 + j] = (unsigned char)rnd();
    }
    k3_gq_dequant(K3_GG_MXFP4, g, a, (int64_t)rows * in);
    k3_gq_mxfp4_to_k3(k, g, rows, in);
    k3_mxfp4_dequant(b, k, k + (size_t)rows * in / 2, rows, in / 2, 32);
    /* == not memcmp: code 8 is -0 in the engine's E2M1 table and +0 in ggml's */
    int bad = 0;
    for (int i = 0; i < rows * in; i++) bad += a[i] != b[i];
    CHECK(!bad, "mxfp4: %d repacked GGUF weights dequantise differently", bad);
    free(g); free(k); free(a); free(b);
    printf("  mxfp4: GGUF blocks repacked to the engine layout, every weight unchanged\n");
}

/* ------------------------------------------------------------- a tiny GGUF writer */
static void w_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void w_str(FILE *f, const char *s) { w_u64(f, strlen(s)); fwrite(s, 1, strlen(s), f); }

/* shard with one Q8_0 [64 x 3] matrix and one F32 [5] vector; metadata on shard 1 only */
static void write_shard(const char *path, int shard, int corrupt)
{
    FILE *f = fopen(path, "wb");
    fwrite("GGUF", 1, 4, f);
    w_u32(f, 3);
    const int nt = shard == 1 ? 0 : 2, nkv = shard == 1 ? 4 : 1;
    w_u64(f, nt); w_u64(f, nkv);
    if (shard == 1) {
        w_str(f, "general.architecture"); w_u32(f, K3_GV_STR); w_str(f, "kimi-k3");
        w_str(f, "general.alignment"); w_u32(f, K3_GV_U32); w_u32(f, 64);
        w_str(f, "kimi-k3.attention.head_count_kv"); w_u32(f, K3_GV_ARR); w_u32(f, K3_GV_I32); w_u64(f, 3);
        { int32_t a[3] = { 0, 1, 0 }; fwrite(a, 4, 3, f); }
        w_str(f, "tokenizer.ggml.tokens"); w_u32(f, K3_GV_ARR); w_u32(f, K3_GV_STR); w_u64(f, 2);
        w_str(f, "a"); w_str(f, "bc");
    } else {
        w_str(f, "general.alignment"); w_u32(f, K3_GV_U32); w_u32(f, 64);
        w_str(f, "blk.0.m.weight"); w_u32(f, 2); w_u64(f, 64); w_u64(f, 3); w_u32(f, K3_GG_Q8_0); w_u64(f, 0);
        w_str(f, "blk.0.v.weight"); w_u32(f, 1); w_u64(f, 5); w_u32(f, K3_GG_F32);
        w_u64(f, corrupt == 2 ? 1u << 20 : 256);
        long pos = ftell(f);
        while (pos % 64) { fputc(0, f); pos++; }
        for (int i = 0; i < 6 * 34; i++) fputc(i & 0xff, f);
        while ((ftell(f) - pos) < 256) fputc(0, f);
        float v[5] = { 1, 2, 3, 4, 5 };
        fwrite(v, 4, corrupt == 1 ? 2 : 5, f);
    }
    fclose(f);
}

static void test_reader(const char *dir)
{
    char p1[512], p2[512];
    snprintf(p1, sizeof p1, "%s/t-00001-of-00002.gguf", dir);
    snprintf(p2, sizeof p2, "%s/t-00002-of-00002.gguf", dir);
    write_shard(p1, 1, 0);
    write_shard(p2, 2, 0);
    K3Gguf g;
    CHECK(k3_gguf_open(&g, p2) == 0, "open a two-shard GGUF by its second shard");
    if (!fails) {
        CHECK(g.st.nshard == 2 && g.nt == 2, "shards %d tensors %d", g.st.nshard, g.nt);
        CHECK(!strcmp(g.arch, "kimi-k3"), "architecture '%s'", g.arch);
        const K3GKv *kv = k3_gguf_kv(&g, "kimi-k3.attention.head_count_kv");
        CHECK(kv && kv->vt == K3_GV_ARR && kv->na == 3 && kv->a && kv->a[1] == 1, "int array metadata");
        kv = k3_gguf_kv(&g, "tokenizer.ggml.tokens");
        CHECK(kv && kv->na == 2 && !kv->a, "string arrays are counted and skipped");
        const K3GTensor *m = k3_gguf_find(&g, "blk.0.m.weight"), *v = k3_gguf_find(&g, "blk.0.v.weight");
        CHECK(m && m->type == K3_GG_Q8_0 && m->ne[0] == 64 && m->ne[1] == 3 && m->nbytes == 6 * 34, "Q8_0 tensor info");
        CHECK(v && v->nbytes == 20 && m && v->off == m->off + 256, "F32 tensor info and offsets");
        CHECK(!k3_gguf_find(&g, "blk.0.missing"), "absent tensor");
        if (m && v) {
            unsigned char b[6 * 34]; float fv[5];
            CHECK(k3_gguf_read(&g, m->shard, m->off, m->nbytes, b) == 0 && b[0] == 0 && b[203] == 203,
                  "read tensor bytes");
            CHECK(k3_gguf_read(&g, v->shard, v->off, v->nbytes, fv) == 0 && fv[4] == 5.0f, "read F32 values");
        }
        k3_gguf_close(&g);
    }
    write_shard(p2, 2, 1);
    CHECK(k3_gguf_open(&g, p1) != 0, "a truncated data region must be refused");
    write_shard(p2, 2, 2);
    CHECK(k3_gguf_open(&g, p1) != 0, "a tensor offset past the end must be refused");
    FILE *f = fopen(p2, "wb"); fwrite("GGUF\3\0\0", 1, 7, f); fclose(f);
    CHECK(k3_gguf_open(&g, p1) != 0, "a truncated header must be refused");
    unlink(p1); unlink(p2);
    printf("  reader: two-shard metadata, index and reads; 3 malformed files refused\n");
}

int main(int argc, char **argv)
{
    test_kernels();
    test_mxfp4();
    test_reader(argc > 1 ? argv[1] : ".");
    if (fails) { printf("test_gguf: %d FAILURE(S)\n", fails); return 1; }
    printf("test_gguf: all passed\n");
    return 0;
}

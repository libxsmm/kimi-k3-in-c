/* SPDX-License-Identifier: Apache-2.0 */
/* k3_gguf.h - GGUF reader for the Unsloth Kimi K3 quants (arch "kimi-k3").
 *
 * A GGUF file is [magic "GGUF"][u32 version][u64 n_tensors][u64 n_kv][kv...][tensor
 * infos...][pad to general.alignment][tensor data]. Tensor offsets are relative to the
 * data start. The model ships as NAME-0000k-of-0000N.gguf shards, each with its own
 * header; shard 1 holds the metadata (tokenizer, template) and no tensors.
 *
 * Dimensions are ggml order: ne[0] is the innermost (contiguous) axis, so a matrix with
 * ne = [in, out] is `out` rows of `in` weights, which is the engine's row-major W[out][in].
 *
 * Every count, length and offset read from a file is bounded before use: a hostile or
 * truncated header fails the open instead of steering a read past the end of a buffer. */
#ifndef K3_GGUF_H
#define K3_GGUF_H

#include <stdint.h>

#include "k3.h"
#include "k3_st.h"

typedef struct {
    char   *name;
    int     shard, type, ndim;
    int64_t ne[4];
    int64_t off;            /* ABSOLUTE byte offset within its shard */
    int64_t nbytes;         /* -1 when the ggml type is not one this engine reads */
} K3GTensor;

/* GGUF value types */
enum { K3_GV_U8, K3_GV_I8, K3_GV_U16, K3_GV_I16, K3_GV_U32, K3_GV_I32, K3_GV_F32,
       K3_GV_BOOL, K3_GV_STR, K3_GV_ARR, K3_GV_U64, K3_GV_I64, K3_GV_F64 };

typedef struct {
    char    *key;
    int      vt;            /* K3_GV_*; arrays keep their element type in at       */
    int      at;
    int64_t  i;             /* integer and bool values                             */
    double   f;             /* every numeric value, also integers                  */
    char    *s;             /* strings, or NULL                                    */
    int64_t *a;             /* numeric arrays of at most K3_GGUF_MAXARR elements    */
    int64_t  na;            /* element count, also for arrays that were not kept   */
} K3GKv;

#define K3_GGUF_MAXARR 65536

typedef struct {
    K3St       st;          /* shard descriptors only: reuses the parallel O_DIRECT reads */
    K3GTensor *t;
    int        nt;
    int32_t   *bucket;
    int        nbucket;
    K3GKv     *kv;          /* metadata of the first shard */
    int        nkv;
    char       arch[64];
} K3Gguf;

/* path is any shard (or a single-file model); every sibling shard is opened. 0 = ok. */
int  k3_gguf_open(K3Gguf *g, const char *path);
void k3_gguf_close(K3Gguf *g);

const K3GTensor *k3_gguf_find(const K3Gguf *g, const char *name);
const K3GKv     *k3_gguf_kv(const K3Gguf *g, const char *key);

/* Bytes [off, off + n) of a shard into dst, as parallel O_DIRECT chunks. 0 = ok. */
int k3_gguf_read(const K3Gguf *g, int shard, int64_t off, int64_t n, void *dst);

/* K3Cfg from the kimi-k3.* metadata, refusing (0) on any missing key. fa receives the
 * ONE-based MLA layer list, as k3_cfg_load produces it. 1 = ok. */
int k3_gguf_cfg(const K3Gguf *g, K3Cfg *c, int *fa, int fa_max);

#endif /* K3_GGUF_H */

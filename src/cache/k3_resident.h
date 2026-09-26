/* k3_resident.h - every routed expert of a layer range held in DRAM, no eviction.
 *
 * The streaming cache (k3_cache.h) exists because 1.45 TB of experts cannot sit in one
 * machine's RAM. Once the model is spread over enough sockets it can, and then the cache
 * is pure overhead: LRU bookkeeping, a histogram and a trace per request, and a batch
 * prefetch that has nothing left to fetch. This source loads each layer's experts once,
 * up front, and get() is a pointer computation.
 *
 * Each layer's 896 experts form one contiguous span inside that layer's shard (measured
 * on the released checkpoint: 15,722,348,544 bytes, no gaps), so a layer is loaded as a
 * single span read in parallel aligned chunks. The bytes are the checkpoint's own, so the
 * kernels see exactly what the cache would have handed them and output is unchanged.
 *
 * Memory is placed by first touch from the loading threads; run under the NUMA binding
 * of the socket that will compute on it.
 */
#ifndef K3_RESIDENT_H
#define K3_RESIDENT_H

#include "k3.h"
#include "k3_load.h"
#include "k3_st.h"

typedef struct {
    K3ExpertSrc  src;             /* MUST be first: pass &res->src to K3MoeW */

    const K3St  *st;
    int          n_layers, n_experts;
    int          layer_lo, layer_hi;   /* resident MoE layers are [layer_lo, layer_hi) */

    unsigned char **alloc;        /* [n_layers] aligned allocation, or NULL      */
    unsigned char **base;         /* [n_layers] where the layer's span starts    */
    K3ExpertRef *ref;             /* [n_layers*n_experts], off relative to base  */

    uint64_t     bytes_resident, requests;
    double       load_seconds;
    int          ilv;             /* K3_MX_* layout the packed rows were given    */
} K3Resident;

/* Load every routed expert of the MoE layers in [layer_lo, layer_hi). Dense layers in the
 * range are skipped. Returns 0 on success; on failure nothing is left allocated. */
int  k3_resident_init(K3Resident *r, const K3St *st, const K3Cfg *cfg,
                      int layer_lo, int layer_hi);
void k3_resident_free(K3Resident *r);

/* Expert bytes the range would occupy, without loading anything. -1 if unresolvable. */
int64_t k3_resident_bytes(const K3St *st, const K3Cfg *cfg, int layer_lo, int layer_hi);

void k3_resident_report(const K3Resident *r, const char *label);

#endif /* K3_RESIDENT_H */

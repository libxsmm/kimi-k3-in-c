/* SPDX-License-Identifier: Apache-2.0 */
/* k3_amx.h - batched (prefill) GEMM on AMX-BF16 for the GGUF weight formats.
 *
 * Y[t][r] = sum_k X[t][k] W[r][k] for T tokens at once. Each thread dequantises a panel
 * of its weight rows (block scales folded in) to bf16 once, keeps it in L2 and streams
 * every token through it, so the weights are read and decoded once per call instead of
 * once per token. Weights are the A operand (row-major, no transpose); activations are
 * packed once per call into the B-tile (VNNI pair) layout and shared by the team.
 *
 * NOT bit-identical to the GEMV path: weights and activations are rounded to bf16 and the
 * tile accumulates in its own order. Callers keep the exact path behind a switch. */
#ifndef K3_AMX_H
#define K3_AMX_H

#include <stddef.h>
#include <stdint.h>

/* 1 when AMX-BF16 is compiled in, present, and the OS granted tile state. */
int k3_amx_ok(void);

/* uint16 elements of the packed activations for T tokens of `in` (in % 32 == 0). */
size_t k3_amx_xv_elems(int T, int in);

/* Team function: this thread's share of packing X [T][ldx] fp32 into Xv. The caller
 * syncs before k3_amx_gemm reads Xv. */
void k3_amx_pack_x(uint16_t *Xv, const float *X, int ldx, int T, int in);

/* Team function: rows [r0, r1) of W (ggml type wt, `in` weights per row, row r at
 * W + (r - wbase) * row_bytes) for all T tokens into Y[t * ldy + r]. The rows are split
 * over the team in 16-row tiles; no barrier at the end. */
void k3_amx_gemm(float *Y, int ldy, const uint16_t *Xv, int T, const void *W, int wt,
                 int in, int wbase, int r0, int r1);

/* GROUPED (MoE): token range [t0, t0 + n) of one packed Xv (t0 % 16 == 0, tokens past n
 * zero) against one matrix each; output Y[t * ldy + r - ycol0] for t in the range. */
typedef struct {
    const void *W;
    int wt, wbase;
    float *Y;
    int ldy, ycol0, t0, n;
} K3AmxGroup;

/* Team: rows [r0, r1) of every group's matrix; (group, 64-row block) tasks handed out
 * dynamically. Starts with a barrier, none at the end. */
void k3_amx_gemm_groups(const K3AmxGroup *g, int ng, const uint16_t *Xv, int in, int r0, int r1);

/* Team: pack N tokens (N % 16 == 0). rows: token i is row map[i] of X [.][ldx];
 * cols: token i is column map[i] of X [in][ldx], the valid tokens of every 16-token tile
 * a prefix of consecutive columns. map[i] < 0 is a zero token. The caller syncs after. */
void k3_amx_pack_rows(uint16_t *Xv, const float *X, int ldx, const int *map, int N, int in);
void k3_amx_pack_cols(uint16_t *Xv, const float *X, int ldx, const int *map, int N, int in);

#endif /* K3_AMX_H */

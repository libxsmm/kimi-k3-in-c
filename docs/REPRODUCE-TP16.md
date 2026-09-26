# Reproducing 56.4 ms/token: full Kimi K3, 16-way tensor parallel

This is the recipe for the fastest measured decode of the **full model** (93 layers,
all 896 routed experts per layer) with the exact fp32 engine. The engine is
tensor parallel over MPI, with one rank per CPU socket, and runs at batch 1.

| | |
|---|---|
| Result | **56.4 ms/token** (17.7 tokens/s). Best single run 56.38 ms; repeated runs gave 56.4 to 56.8 ms. With one NUMA-local HCA per rank (section 5): 56.0 ms |
| Output | the same 16 tokens as the single-process reference; the oracle matches exactly at 1, 3 and 16 ranks |
| Scale | 8 nodes × 2 sockets = 16 ranks, 64 threads per rank |

## 1. Hardware and software used

| | |
|---|---|
| Nodes | 8 |
| CPU per node | 2 × Intel Xeon Platinum 8592+ (Emerald Rapids): 64 cores and 320 MB L3 per socket, SMT on but unused |
| Memory per node | 1 TB DDR5-5600, 8 channels per socket (358 GB/s theoretical per socket); 2 NUMA nodes |
| Interconnect | InfiniBand NDR 400 Gb/s: `mlx5_0` on NUMA 0 and `mlx5_1` on NUMA 1, one per socket. The nodes also have an unused Omni-Path `hfi1_0` |
| OS | Rocky Linux 10.2, kernel 7.2; transparent huge pages set to `always` |
| Compiler | GCC 14.3.1 |
| MPI | Open MPI 5.0.11 with UCX 1.22.0 (`pml ucx`, `osc ucx`) |
| Checkpoint | `moonshotai/Kimi-K3`, 1.56 TB in 96 safetensors shards, on a parallel file system that every node can read |

The fp32 AVX-512 kernels need AVX-512 F, BW and VL. The optional bf16 mode (section 6)
also needs AVX512-BF16. Without these ISA extensions the engine still runs and gives
correct results, but it uses slower kernels.

Memory: each rank holds its slice of the trunk plus a slice of every routed expert,
90.4 GB of experts per rank. Peak RSS was about 101 GB per rank, so about 203 GB per
node.

## 2. Get the checkpoint

```bash
scripts/download-model.sh /path/to/k3model      # verifies shard count and byte total
```

Put it on storage that every node mounts at the same path. Each rank reads only its
own rows, so no copy to local disk is needed.

## 3. Build (on a compute node)

The default `ARCH` is `-march=native`, so build on a node with the target CPU, or pass
an explicit `ARCH`.

```bash
# inside an allocation, on one of the compute nodes
make -j32                                          # serial build + weightless tests' binaries
make test                                          # expect: ALL WEIGHTLESS TESTS PASSED
make MPI=1 -j32 bin-mpi/k3 bin-mpi/k3_model        # name the targets: see note below
```

Note: `make MPI=1` on its own does not relink `bin-mpi/k3_model`, so always name the
binaries you need. Cross-compiling works too, for example
`make MPI=1 ARCH="-march=emeraldrapids" bin-mpi/k3 bin-mpi/k3_model`.

## 4. Check correctness first (no weights needed)

```bash
mpirun -np 16 --map-by ppr:1:package --bind-to package \
       -x OMP_NUM_THREADS=64 -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
       ./bin-mpi/k3_model tests/fixtures
# expect: VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY
```

## 5. The timed run

Allocate 8 whole nodes (for example `salloc -N 8 --exclusive`), then run:

```bash
mpirun -np 16 --map-by ppr:1:package --bind-to package \
       -x OMP_NUM_THREADS=64 -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
       -x K3_PROF=1 \
       ./bin-mpi/k3 /path/to/k3model \
       --incremental --experts-resident --ids 19180 --gen 16 --out run.json
```

What each part does:
- `--map-by ppr:1:package --bind-to package`: one rank per socket, bound to it, so each
  rank's weights are in its own socket's memory (first touch by the loading threads).
- `OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores`: one thread per physical
  core.
- `--experts-resident`: loads this rank's rows of every routed expert into RAM before
  decoding starts. Loading takes about 3 minutes and is not timed.
- `--incremental`: decodes with the KV cache and the recurrent state (the normal decode
  path).
- `--ids 19180 --gen 16`: a one-token prompt, then 16 greedy tokens.
- `K3_PROF=1`: prints the per-phase profile used for the number below.

### Optional: one HCA per rank

By default UCX runs dual rail: short and eager messages use one HCA per rank, but
rendezvous transfers (about 23 KB and up, inter-node) are split 50/50 over `mlx5_0`
and `mlx5_1`. Half of those bytes then cross the socket link. Pinning each rank to
its socket's HCA is a small but measurable win: 56.21 → 55.98 ms/token, with TP
communication going from 5.43 to 5.01 ms (9.3 → 8.6 µs per gather). Tokens are
unchanged.

```bash
cat > ucxpin.sh <<'EOF'
#!/bin/bash
case "$OMPI_COMM_WORLD_LOCAL_RANK" in       # ppr:1:package: local rank = socket
    0) export UCX_NET_DEVICES=mlx5_0:1 ;;
    1) export UCX_NET_DEVICES=mlx5_1:1 ;;
esac
exec "$@"
EOF
chmod +x ucxpin.sh
mpirun -np 16 --map-by ppr:1:package --bind-to package \
       -x OMP_NUM_THREADS=64 -x OMP_PROC_BIND=close -x OMP_PLACES=cores -x K3_PROF=1 \
       ./ucxpin.sh ./bin-mpi/k3 /path/to/k3model \
       --incremental --experts-resident --ids 19180 --gen 16 --out run.json
```

Check which devices UCX picked with `-x UCX_PROTO_INFO=y`: every rank should list only
its own `mlx5_N`. Check the device-to-socket mapping on your nodes first with
`cat /sys/class/infiniband/*/device/numa_node`.

The figure to read is this line from rank 0's report:

```
profile: 15 steps after step 0, 56.38 ms/step wall
```

It averages steps 1 to 15. The `s/token average` line printed above it also includes
step 0, which is the prompt, cold caches and first-touch effects, so it reads higher.

### Expected output

`run.json` → `generated_ids` must be:

```
[11, 374, 1491, 261, 220, 577, 1389, 3410, 18426, 13, 374, 714, 1479, 4747, 261, 3242]
```

A representative profile (ms/token, 16 ranks, fp32 engine):

| phase | ms | | phase | ms |
|---|---|---|---|---|
| routed experts (MXFP4) | 9.3 | | shared expert | 4.1 |
| KDA projections (q/k/v, gates) | 9.0 | | MoE up | 3.3 |
| KDA output (g, o_proj) | 7.6 | | KDA recurrence | 3.0 |
| TP communication | 5.4 | | AttnRes + norms | 2.9 |
| router + MoE down | 4.5 | | MLA proj + out | 3.8 |

TP communication is about 581 gathers per token, 9.6 µs each on average. Each rank
reads 8.8 GB of weights per token.

## 6. Settings that change the result

All of these are off or at their fastest by default. They exist for A/B testing.

| setting | effect |
|---|---|
| `K3_TP_ONESIDED=0` | use `MPI_Allgatherv` instead of the one-sided (MPI RMA) team gather: slower |
| `K3_MXFP4_ILV=0` | keep MXFP4 experts in checkpoint order instead of reordering them at load time for the AVX-512 kernel: slower, same bits |
| `K3_NO_QKV=1` | do not merge the q/k/v rows into one stream per thread: slightly slower, same bits |
| `K3_NOHUGE=1` | no huge-page hint for the resident experts |
| `--bf16-act` | inputs to matmuls rounded to bf16, bf16 dot products, bf16 gathers. Not the exact engine; measured 56.6 to 58.3 ms, no faster at this scale. `K3_BF16_PARTS=1\|2\|4` enables one part at a time (bf16 matmuls, MXFP4, gathers) |
| `K3_PROF` unset | no profile; decode speed itself is unchanged |

Results are bit-identical across rank and thread counts in the default fp32 mode, and
also in `--bf16-act` mode. So any number of ranks reproduces the tokens above, but the
timing only applies to 16 ranks on the hardware in section 1.

## 7. Variance and pitfalls

- Repeated runs on the same 8 nodes spread about ±0.4 ms. Report the median of
  several runs, not the best one.
- A rank that shares its socket with other load, or that is not bound to a package,
  costs far more than that. Allocate whole nodes.
- Do not time step 0. It includes the prompt and first-touch effects.
- The checkpoint file system only affects the load time, never the decode time, because
  experts are resident.

## 8. Future work: a native transport for the team gather

TP communication is 5.0 ms of the 56.0 ms per token: 581 gathers at 8.6 µs each. A
run with `K3_TP_SKEW=1` puts a barrier before every gather and splits that time:

| per token | ms | |
|---|---|---|
| transfer, ranks aligned | 4.2 | 7.2 µs per gather, latency-bound: a 16-float gather already costs 6.65 µs |
| waiting for slower ranks | 0.8 to 2.9 | load imbalance; no transport change helps |

The floor is one network one-way latency per gather, about 1.5 to 2 µs on NDR
(`MPI_Barrier`, 4 dissemination rounds, takes 7.3 µs). Estimated gain from bypassing
Open MPI's one-sided layer (osc/ucx) is **2 to 3 ms/token (4 to 5%), about 53 to
54 ms/token**:

1. **Issue cost, about 2.3 ms.** Today each gather makes 15 `MPI_Put` calls. With
   raw verbs, one doorbell posts 15 unsignaled RDMA writes with small payloads inline,
   so a gather could take about 2.5 to 3.5 µs.
2. **Wire format, about 0.3 to 0.5 ms.** Every float travels as an 8-byte
   `seq << 32 | bits` word so the receiver can detect its arrival. RDMA writes on one
   RC queue pair land in order, so raw 4-byte floats plus one flag word per block would
   halve the bytes. This matters mostly for the 55k-float MoE activation gather.
3. **Flushes, under 0.2 ms.** Poll completions instead of calling
   `MPI_Win_flush_local_all` every 32 sends.

Suggested order:
1. Call UCX directly (`ucp_put_nbi` + `ucp_worker_fence`/`flush`) inside
   `src/par/k3_mpi.c`. It is transport-only, stays portable across fabrics, and
   should show how much of the ~4 µs per gather is software overhead.
2. Then raw verbs (RC queue pairs, one per peer, on the rank's NUMA-local HCA) if
   UCX falls short.
3. Independently, fuse gathers: every gather removed saves its whole ~7 µs, more than
   any transport change can.

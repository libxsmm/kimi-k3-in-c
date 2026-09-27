# Reproducing distributed decode: full Kimi K3 at TP=16 and TP=32 on Xeon

This is a standalone recipe for the fastest measured decode of the **full model**:
93 layers, all 896 routed experts per layer, batch 1. It uses the exact fp32 engine,
tensor parallel over MPI, with one rank per CPU socket. Everything below is needed; no
other documentation is required.

| config | nodes × sockets | ms/token | tokens/s | peak RSS per rank | status |
|---|---|---|---|---|---|
| **TP=32**, 1 rank/socket, pinned HCA (16 nodes) | 16 × 2 | **47.6** | 21.0 | 52.5 GB | measured (47.62 and 47.53; same session TP=16: 57.0) |
| TP=16, 1 rank/socket, pinned HCA | 8 × 2 | 56.0 | 17.9 | 101 GB | measured (best 55.98; unpinned 56.2 to 56.8) |
| TP=32, 2 ranks/socket (8 nodes) | 8 × 2 | 58.9 | 17.0 | 52.5 GB | measured: same hardware, so no faster |

For every configuration the output is the same 16 tokens as the single-process
reference. Logits are byte-identical at any rank count, and the correctness oracle
matches exactly at 1, 3, 16 and 32 ranks.

## 1. Hardware and software used

| | |
|---|---|
| CPU per node | 2 × Intel Xeon Platinum 8592+ (Emerald Rapids), 64 cores and 320 MB L3 per socket; SMT is on but unused |
| Memory per node | 1 TB DDR5-5600, 8 channels per socket (358 GB/s theoretical per socket); 2 NUMA nodes |
| Interconnect | InfiniBand NDR 400 Gb/s, one HCA per socket: `mlx5_0` on NUMA 0, `mlx5_1` on NUMA 1 |
| OS | Rocky Linux 10.2, kernel 7.2; transparent huge pages set to `always` |
| Compiler | GCC 14.3.1 (any C99 compiler with OpenMP works) |
| MPI | Open MPI 5.0.11 with UCX 1.22.0 (`pml ucx`, `osc ucx`); needs MPI RMA (`MPI_Win_allocate`, `MPI_Put`) |
| Scheduler | Slurm; any launcher that can place one rank per socket works |
| Checkpoint | `moonshotai/Kimi-K3`, 1.56 TB in 96 safetensors shards, on a file system every node mounts |

ISA: the fast kernels need AVX-512 F, BW and VL. `--bf16-act` (section 8) also needs
AVX512-BF16. Without these extensions the engine still gives exact results, but runs
slower kernels.

Other CPUs: any number of sockets per node works. The rule is always one rank per
socket (or per NUMA domain, if the machine runs sub-NUMA clustering), with threads =
physical cores in that domain.

## 2. Get the code and the checkpoint

```bash
git clone https://github.com/alheinecke/kimi-k3-in-c.git
cd kimi-k3-in-c
git switch xeon_dist_magic

scripts/download-model.sh /path/to/k3model   # 1.56 TB; verifies shard count and byte total
```

Put the checkpoint on storage that every node mounts at the same path. Each rank reads
only its own rows at load time, so no local copy is needed. Decode never touches the
disk.

## 3. Build (on a compute node, inside an allocation)

`ARCH` defaults to `-march=native`, so build on a node with the target CPU.

```bash
salloc -N 8 --exclusive            # 16 for TP=32 at 1 rank/socket
srun -N1 -n1 bash -c '
  module load openmpi 2>/dev/null   # or: source <openmpi>/openmpi_vars.sh
  make -j32                                      # serial engine + test binaries
  make test                                      # expect: ALL WEIGHTLESS TESTS PASSED
  make MPI=1 -j32 bin-mpi/k3 bin-mpi/k3_model    # name the targets explicitly
'
```

`make MPI=1` on its own does not relink `bin-mpi/k3_model`, so always name the targets.
You can also cross-compile, for example
`make MPI=1 ARCH="-march=emeraldrapids" bin-mpi/k3 bin-mpi/k3_model`.

## 4. HCA pinning script

By default UCX runs **dual rail**. Short messages use one HCA per rank, but rendezvous
transfers (about 23 KB and up) are split 50/50 over both HCAs, so half of those bytes
cross the socket link. Pin every rank to its socket's HCA instead. At TP=16 this took
decode from 56.21 to 55.98 ms/token (communication 5.43 → 5.01 ms), with unchanged
tokens.

```bash
cat > ucxpin.sh <<'EOF'
#!/bin/bash
# ranks are packed per socket by --map-by ppr:N:package: local rank r is on socket r*S/LOCAL_SIZE
S=${K3_SOCKETS:-2}
s=$(( OMPI_COMM_WORLD_LOCAL_RANK * S / OMPI_COMM_WORLD_LOCAL_SIZE ))
export UCX_NET_DEVICES=mlx5_${s}:1
exec "$@"
EOF
chmod +x ucxpin.sh
```

Check the device-to-socket mapping first with
`cat /sys/class/infiniband/mlx5_*/device/numa_node`, and adapt the name pattern if your
HCAs are not `mlx5_<socket>`. To verify, add `-x UCX_PROTO_INFO=y` to a run: every rank
must list only its own `mlx5_N`.

## 5. Check correctness (no weights needed, seconds)

```bash
mpirun -np 16 --map-by ppr:1:package --bind-to package \
       -x OMP_NUM_THREADS=64 -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
       ./ucxpin.sh ./bin-mpi/k3_model tests/fixtures
# expect: VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY
```

## 6. The timed runs

All `-x` options belong to `mpirun`, so they must come **before** `./ucxpin.sh`.

### TP=16: 8 nodes, 1 rank per socket (the 56.0 ms result)

```bash
mpirun -np 16 --map-by ppr:1:package --bind-to package \
       -x OMP_NUM_THREADS=64 -x OMP_PROC_BIND=close -x OMP_PLACES=cores -x K3_PROF=1 \
       ./ucxpin.sh ./bin-mpi/k3 /path/to/k3model \
       --incremental --experts-resident --ids 19180 --gen 16 --out run16.json
```

### TP=32: 16 nodes, 1 rank per socket (the 47.6 ms result)

```bash
salloc -N 16 --exclusive
mpirun -np 32 --map-by ppr:1:package --bind-to package \
       -x OMP_NUM_THREADS=64 -x OMP_PROC_BIND=close -x OMP_PLACES=cores -x K3_PROF=1 \
       ./ucxpin.sh ./bin-mpi/k3 /path/to/k3model \
       --incremental --experts-resident --ids 19180 --gen 16 --out run32.json
```

### TP=32 on 8 nodes: 2 ranks per socket, 32 cores each (measured 58.9 ms)

```bash
mpirun -np 32 --map-by ppr:2:package:PE=32 --bind-to core \
       -x OMP_NUM_THREADS=32 -x OMP_PROC_BIND=close -x OMP_PLACES=cores -x K3_PROF=1 \
       ./ucxpin.sh ./bin-mpi/k3 /path/to/k3model \
       --incremental --experts-resident --ids 19180 --gen 16 --out run32.json
```

If Slurm hands out fewer slots than ranks, add `:OVERSUBSCRIBE` to `--map-by`. The
`PE=32` still gives each rank its own 32 cores.

What each part does:
- `--map-by ppr:1:package --bind-to package`: one rank per socket, bound to it, so each
  rank's weights live in its own socket's memory (first touch by the loading threads).
- `OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores`: one thread per physical
  core.
- `--experts-resident`: loads this rank's rows of every routed expert into RAM before
  decode starts. This takes 3 to 5 minutes and is not timed.
- `--incremental`: decodes with the KV cache and the recurrent state (the normal decode
  path).
- `--ids 19180 --gen 16`: a one-token prompt, then 16 greedy tokens.
- `K3_PROF=1`: prints the per-phase profile that contains the number to report.

### What to read, and what to expect

Rank 0 prints the number to report:

```
profile: 15 steps after step 0, 55.98 ms/step wall
```

It averages steps 1 to 15. The `s/token average` line above it also includes step 0
(the prompt, cold caches, first touch), so it reads higher.

`generated_ids` in the `--out` JSON must be, at every TP size:

```
[11, 374, 1491, 261, 220, 577, 1389, 3410, 18426, 13, 374, 714, 1479, 4747, 261, 3242]
```

To prove bit-identity across TP sizes, add `--layers 8 --dump-logits a.bin` to two runs
with different `-np` values and `cmp` the two files.

Representative TP=16 profile (ms/token):

| phase | ms | | phase | ms |
|---|---|---|---|---|
| routed experts (MXFP4) | 9.3 | | shared expert | 4.1 |
| KDA projections (q/k/v, gates) | 9.0 | | MoE up | 3.3 |
| KDA output (g, o_proj) | 7.6 | | KDA recurrence | 3.0 |
| TP communication | 5.0 | | AttnRes + norms | 2.9 |
| router + MoE down | 4.5 | | MLA proj + out | 3.8 |

Each rank reads 8.8 GB of weights per token. There are 581 gathers per token, 8.6 µs
each when pinned.

## 7. Scaling notes: what changes at TP=32

- **Memory halves:** 45.2 GB of experts per rank, 52.5 GB peak RSS, against 90.4 and
  101 GB at TP=16.
- **2 ranks per socket on 8 nodes is not faster** (58.9 vs 56.0 ms). Each socket
  still streams the same bytes, and two ranks now share one HCA. Communication roughly
  doubled, to 10.0 ms (17.3 µs per gather).
- **TP=32 on 16 nodes, measured: 47.6 ms/token (21.0 tokens/s), 1.20x over TP=16.**
  Each rank reads 4.50 GB per token instead of 8.8 GB. Both runs below are from the
  same session on the same 16 nodes (TP=16 used 8 of them), HCA-pinned, and give the
  reference tokens:

  | ms/token | TP=16 (8 nodes) | TP=32 (16 nodes) | speedup |
  |---|---|---|---|
  | **wall** | 57.04 | **47.62** (repeat 47.53) | 1.20x |
  | compute (wall − comm − untimed) | 49.8 | 35.3 | 1.41x |
  | TP communication | 5.06 (8.7 µs/gather) | 9.65 (16.6 µs/gather) | 0.52x |
  | KDA projections | 8.99 | 5.17 | 1.74x |
  | KDA output | 8.11 | 5.97 | 1.36x |
  | routed experts | 9.41 | 6.65 | 1.41x |
  | shared expert | 4.26 | 1.22 | 3.5x |
  | MoE up | 3.25 | 2.79 | 1.17x |
  | MLA proj + out | 3.92 | 2.98 | 1.32x |
  | KDA recurrence | 3.27 | 2.99 | 1.09x |
  | AttnRes + norms | 2.99 | 2.98 | 1.0x |
  | router + MoE down | 4.57 | 3.99 | 1.15x |
  | peak RSS per rank | 101.2 GB | 52.5 GB | |

  The weight-streaming phases scale well; AttnRes, norms, the router and the KDA
  recurrence are replicated or fixed per rank and do not shrink. Communication doubles:
  still 581 gathers per token, but each waits on 31 peers instead of 15, so it becomes
  the largest phase (20%). That makes the native transport (section 10) worth more at
  TP=32. The pre-measurement estimate was 35 to 45 ms.
- All sharded dimensions of Kimi K3 are divisible by 32, but they need not be: every
  split is a balanced contiguous partition, so any rank count works.

## 8. Settings

All of these are off, or at their fastest, by default. They exist for A/B testing.

| setting | effect |
|---|---|
| `K3_TP_ONESIDED=0` | `MPI_Allgatherv` instead of the one-sided (MPI RMA) team gather: slower |
| `K3_MXFP4_ILV=0` | keep MXFP4 experts in checkpoint order instead of the AVX-512 load-time layout: slower, same bits |
| `K3_NO_QKV=1` | do not merge the q/k/v rows into one stream per thread: slightly slower, same bits |
| `K3_NOHUGE=1` | no huge-page hint for the resident experts |
| `K3_TP_SKEW=1` | barrier before every gather, so that rank skew (`tp wait`) is split from transfer (`tp comm`) |
| `--bf16-act` | matmul inputs rounded to bf16, bf16 dot products, bf16 gathers. Not the exact engine; measured 56.6 to 58.3 ms at TP=16, so no faster here. `K3_BF16_PARTS=1\|2\|4` enables one part at a time: bf16 matmuls, MXFP4, gathers |

## 9. Variance and pitfalls

- Repeated runs on the same nodes spread about ±0.4 ms. Report the median of several
  runs.
- Allocate whole nodes. A rank sharing its socket with other load, or not bound to a
  package, costs far more than that spread.
- Do not time step 0.
- The checkpoint file system only affects load time, never decode time, because the
  experts are resident.
- Put `-x` options before the wrapper script; otherwise they become arguments of the
  wrapper's `exec` and the run fails.

## 10. Future work: a native transport for the team gather

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

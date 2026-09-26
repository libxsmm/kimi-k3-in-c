---
name: c-llm-inference-xeon
description: "Use when: optimizing a C (or C++) LLM inference engine for CPU decode speed; porting a model to multi-socket or multi-node Xeon; adding OpenMP, AVX-512, MXFP4/int4/bf16 GEMV kernels, NUMA placement, or MPI tensor parallelism; making results bit-identical across thread and rank counts; profiling memory-bandwidth-bound decode; deciding whether bf16 activations, LIBXSMM, or RDMA/verbs will pay off. Generalized from the Kimi K3 C engine (117 -> 56 ms/token on 16 Xeon sockets)."
---

# Fast, exact CPU decode for LLMs in C

A playbook that turned a correct but slow C engine for a 2.8T-parameter MoE model
into a 16-socket tensor-parallel engine running 2.1× faster, with bit-identical
output. The steps are ordered: each one relies on the measurements and guarantees of
the ones before it. Apply them to any decoder-only transformer, dense or MoE.

## 0. Ground rules (do these first, keep them forever)

- **An oracle before any optimization.** Build a small model with the real tensor
  graph and random weights, a reference implementation (PyTorch or numpy), and a gate
  that asserts **exact** equality, or a documented tolerance, for logits and tokens.
  Make it run in seconds with no real weights (`make test`).
- **Fix the reduction order** (step 3) so results do not depend on thread or rank
  count. Then every optimization can be validated with `cmp` on dumped logits instead
  of by eyeballing tolerances.
- **Compile with `-ffp-contract=off`, and never with `-ffast-math`.** Choose FMA and
  accumulation order explicitly, in the code.
- **Each step:** unit tests, then the oracle at 1 and N ranks, then logits `cmp`
  against the previous reference, then a timing run. Record the number, the command
  and the reference file.
- **Every optimization gets an env-var off switch** (`K3_NO_X=1`), so it can be A/B'd
  on the real model without a rebuild.
- **Pitfalls that cost hours:**
  - stale binaries: make targets that don't relink, NFS mtime skew, so name the
    targets and check `ls -l --time-style=full-iso`;
  - comparing a JSON key that does not exist, which makes the check vacuous;
  - wrapper scripts that swallow `mpirun -x` options placed after them.

## 1. Measure the roofline before touching code

- Compute **bytes read per token per socket**: weights touched (dense + active
  experts) ÷ sockets. Divided by measured socket bandwidth, that is the floor. Decode
  at batch 1 is almost always memory-bound.
- Write a tiny bandwidth benchmark that links the engine's kernel objects:
  - **read roof:** a plain sum with 4 zmm accumulators, all cores of one socket;
  - **each GEMV kernel** on production shapes, many distinct matrices, so it runs
    from DRAM;
  - **1-thread, L2-resident**, to get compute throughput (GFLOP/s).

  A kernel whose DRAM number is well below the read roof and does not depend on the
  shape is **compute-bound**: fix the kernel. If it reaches the roof but the engine
  is slower, fix the **access pattern** or the **synchronization**.
- A per-phase profiler, recorded by thread 0 only, cheap enough to leave on
  (`K3_PROF=1`). Report steady-state steps and exclude step 0.

## 2. Threading: one parallel region per layer (SPMD)

- Fork/join per kernel costs 3–12 µs, and a layer has dozens of kernels. Open **one
  OpenMP region per decoder layer**. Each kernel takes its rows by thread id
  (`split(n, &lo, &hi)`) and synchronizes only where data crosses threads.
- A kernel callable both inside and outside a team: `TEAM_IF(cond, body)`. Inside a
  team it runs the body directly, with no barrier; outside, it opens a region.
- **A custom barrier:** a dissemination barrier on padded per-thread flags with epoch
  counters, 0.9 µs at 64 threads against 3.8 µs for GOMP. Rule: every thread calls
  it the same number of times.
- **Owner-computes epilogues:** when the next op is elementwise on the rows a thread
  just produced (conv, activation, decay, gating), run it with the same split, in the
  same thread, with no barrier.
- **No allocation in the forward pass:** per-thread persistent scratch slots, grown
  (doubling) and 64-byte aligned. Count calls with an `LD_PRELOAD` malloc counter to
  prove zero.
- Serial work that must happen once (routing selection, I/O) goes on thread 0 into
  statics, followed by one barrier.

## 3. Deterministic reductions (rank- and thread-invariant results)

- Every sum that crosses threads (norm sum of squares, attention-residual dots, dot
  products) is formed over **fixed-size chunks** (e.g. 128 elements), sequentially
  within a chunk, then the chunk sums are added **in chunk order**. The order depends
  only on n.
- **Three-part norms:**
  1. chunk sums, split over threads or fused into the producer's epilogue;
  2. every thread computes the statistic itself (no broadcast needed);
  3. each thread scales its own slice, fused into writing the consumer's input.

  This removes barriers and makes results bit-identical at any team or rank size.
- A ring of 4 shared partial buffers, so a buffer is rewritten only two barriers
  later.

## 4. GEMV kernels for decode

- **One accumulation scheme for all weight types** (fp32, bf16, quantized): element i
  goes to lane i%16 of accumulator (i/16)%4, the tail is masked, and there is one
  fixed reduction tree. Then bf16 == fp32-widened and fused-dequant == dequant-then-
  matmul hold **to the bit**, and are testable.
- **fp32 accumulation is usually enough** for GEMV, and it doubles SIMD width compared
  with double. Verify on real logits (max diff, tokens), and ask the owner to accept
  the change.
- **Check port pressure:** on AVX-512 Xeons, 512-bit shuffles (`vpermps`, `vpmovzx*`,
  `vpunpck*`) only issue on port 5. Count p5 µops per 16 weights; that is often the
  real limit of quantized kernels.
- **Reorder weights at load time to match the SIMD decode.** This is free at run time,
  and bit-identical if the element-to-lane mapping is preserved. For 4-bit weights
  (MXFP4, int4): lay out each 128-element block so that nibble s of dword l =
  element 16s+l. Then one 64-byte load gives 128 codes, and each 16 weights cost one
  shift (p0) plus one `vpermps` into a per-scale 16-entry table (`[scale][code]`,
  built once). There is no widening. Measured: 38 → 52 GFLOP/s per core and +12% DRAM
  bandwidth.
- **Software prefetch** a few hundred bytes ahead on short per-thread weight streams,
  plus the scale rows. Sweep the distance (0–4 KB) on the real model.
- **Borrow ideas from LIBXSMM's generators** (LUT decode, VNNI packing, prefetch) even
  when the library itself doesn't fit. Check which ISA its JIT actually targets, and
  that the kernel is not its reference fallback, before linking it.

## 5. Access pattern: long contiguous streams per thread

- **Split work over (matrix, row block) tasks,** about one per thread, instead of
  splitting every small matrix over all threads. With MoE, the top-k experts
  × row blocks give each thread two long runs, not a few rows of every expert.
- **Merge matrices that read the same input** (q/k/v, gate/up) into one row-interleaved
  matrix at load time (row 3r+m). One stream per thread; same bits.
- **Huge pages** (`MADV_HUGEPAGE`, 2 MB-aligned allocations) for resident weights.
  **First touch** by the threads that will read the data, under the socket binding.

## 6. Distribution: tensor parallelism over MPI, one rank per socket

- **One rank per socket** (or per NUMA domain), threads = physical cores. This beats
  one rank per node: memory stays local and each socket streams its own slice.
- **Shard rows, gather activations:** every rank holds the same activations, computes
  a contiguous balanced block of each sharded matmul's output rows or heads, then all
  ranks allgather. Replicate tiny ops instead of communicating. At load, bind weights
  as **per-rank slices**, read straight from the checkpoint.
- **Gathers are team collectives:** a barrier, thread 0 sends (`MPI_THREAD_FUNNELED`),
  all threads receive their share. Merge small vectors into one gather, and **overlap**
  each gather with work that doesn't touch its segments (begin/end API).
- **One-sided transport** (MPI RMA over UCX), 2× lower latency than `MPI_Allgatherv`
  at small sizes:
  - tagged words `seq<<32 | payload`, so the receiver polls for arrival with no
    handshake;
  - a two-half window with alternating sequence numbers, and a send ring flushed
    once per lap;
  - a flag word for empty blocks;
  - warm-up puts at init (UCX lazy wire-up deadlocks at 8+ ranks otherwise);
  - window size ≥ the largest gather: an oversized gather that silently falls back
    to allgatherv cost 5 ms/token here.
- **NUMA-local NIC:** UCX runs dual rail by default. Pin each rank with
  `UCX_NET_DEVICES`, derived from its local rank, and verify with `UCX_PROTO_INFO=y`.
- **Split skew from transfer:** an optional barrier before each gather charges waiting
  to `tp wait` instead of `tp comm`. At 16 ranks gathers are latency-bound, about
  7 µs each; the floor is one network one-way latency. The remaining levers are
  **fewer gathers** (fusion) and a leaner transport (direct UCX or verbs).

## 7. Precision experiments: measure before adopting

- **bf16 activations** (round inputs to bf16, `vdpbf16ps`, bf16 gathers) change the
  numerics. They helped nothing here, because weight streaming dominates the bytes and
  the gathers are latency-bound. Keep such modes **opt-in**, rank-invariant (the owner
  rounds its own block like its peers), and keep selection-critical values (router
  scores) in fp32.
- **int8 activations with VNNI** for 4-bit weights: only worth it when a kernel is
  still compute-bound after step 4.

## 8. Checklist per model

1. [ ] Oracle, fixtures, exact gates in CI; a logits dump; the JSON key for tokens
   verified.
2. [ ] Roofline: bytes per token per socket; bandwidth benchmark; profiler.
3. [ ] SPMD region per layer, custom barrier, no forks or allocations in the hot loop.
4. [ ] Chunked, deterministic reductions; 3-part norms; `cmp` passes at any thread
   count.
5. [ ] Shared GEMV accumulation scheme; fp32 accumulation; port-pressure check.
6. [ ] Load-time weight reordering for quantized kernels; prefetch sweep.
7. [ ] Task splits for long streams; merged same-input matrices; huge pages; first
   touch.
8. [ ] TP: per-rank slices, one-sided team gather, overlap, window sizing, NIC pinning.
9. [ ] Skew/transfer split; gather count; decide on transport work.
10. [ ] A reproduction guide: hardware, exact commands, expected tokens, variance.

## Reference numbers (Kimi K3, 8 × 2 Xeon 8592+, TP=16, ms/token)

| step | ms/token |
|---|---|
| start | 117.2 |
| SPMD | 91.7 |
| one-sided gather | 85.9 |
| fused norms | 81.6 |
| AVX-512 + expert split | 70.0 |
| fp32 accumulation | 63.3 |
| MXFP4 load-time layout + prefetch | 61.3 |
| one-sided MoE gather + merged q/k/v | 56.4 |
| NUMA-local HCA | 56.0 |

For the full recipe, see `docs/REPRODUCE-TP.md`.

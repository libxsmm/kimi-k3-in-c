# Kimi K3 on vLLM (CPU, multi-node)

vLLM 0.30.0+cpu (torch 2.13) serving the full Kimi K3 checkpoint on Xeon nodes, used as
a baseline for the C engine. The model runs correctly (greedy tokens identical to the C
engine) but is about 100x slower per token.

| file | purpose |
|---|---|
| `k3cpu_plugin/` | vLLM general plugin (`uv pip install -e k3cpu_plugin`). vLLM ships Kimi K3 only for GPUs; this adds torch reference CPU ops for AttnRes, KDA (conv, gate, recurrence), MLA cache insert, MXFP4 SiTU experts, plus fixes for KDA state zeroing and CPU weight packing |
| `k3cpu_plugin/k3cpu/prof.py` | `K3CPU_PROF=1` region profiler (below) |
| `vprof_agg.py` | aggregates the per-rank profiles |
| `patches/0001-*.patch` | vLLM fix: multi-node CPU single-reader message queue |
| `run_node.sh` | per-node launcher (one Slurm task per node, mp backend, `--headless` workers) |
| `smoke_k3.py` | offline smoke test / greedy comparison against a C engine `--out` JSON |
| `ccl_*` | abandoned oneCCL experiment |

## Running TP=16 + EP on 8 nodes

```bash
salloc --partition=emr --nodes=8 --constraint=ddr5600 --exclusive --time=03:59:00 --no-shell
IFNAME=ibs1 HEAD_IP=<ibs1 address of the first node> \
srun --jobid=$JOB --overlap -N8 --ntasks-per-node=1 ./run_node.sh /path/to/k3model \
    --trust-remote-code --max-model-len 1024 --enforce-eager --enable-expert-parallel \
    --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 1
# once "Application startup complete" (weight loading takes 11 to 16 minutes from NFS):
curl --noproxy '*' http://<head>:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"/path/to/k3model","prompt":[19180],"max_tokens":24,"temperature":0,"ignore_eos":true}'
```

`--noproxy '*'` is needed on the login node; otherwise the site proxy answers. The
first request of a server spends about 200 s in its first step (warm-up); time the second.

## Profiling

`K3CPU_PROF=1` (exported before `srun`) wraps every module under the decoder layers, the
plugin ops, the CPU communicator collectives and the model runner with wall-clock
timers. Each region keeps inclusive and exclusive time; each collective is keyed by the
region that issued it (`comm.all_reduce@kda.o_proj`). Every rank writes
`$K3CPU_PROF_DIR/prof_r<rank>.{txt,json}` after each step, averaging the decode steps of
the current request after its first step. The overhead is not measurable (5.80 s/token
with and without).

```bash
python vprof_agg.py $K3CPU_PROF_DIR    # per-rank walls, categories (mean, min..max), top regions
```

## Findings (2026-09-30)

Full model, TP=16 + EP, 8 nodes x 2 sockets (sprh01-03, 05-09), bf16, Gloo over IPoIB
(ibs1), prompt `[19180]`, 24 greedy tokens identical to the C engine reference. Mean of
16 ranks over 23 decode steps; the C engine column is its best TP=16 run on 8 nodes
(docs/REPRODUCE-TP.md, section 7).

| ms/token | vLLM | C engine | ratio |
|---|---|---|---|
| **wall** | **5803** | **55.09** | **105x** |
| communication | 4980 (281 all-reduces x 17.7 ms) | 4.98 (581 gathers x 8.6 us) | 1000x |
| routed experts | 440 (MXFP4 dequant 354, matmul 49, MoE glue 37) | 9.07 | 48x |
| AttnRes + norms | 80 | 2.91 | 27x |
| MoE latent up | 81 | 3.10 | 26x |
| router + MoE latent down | 65 | 4.22 | 15x |
| KDA core (conv, gate, recurrence) | 35 | 3.12 | 11x |
| MLA | 30 | 3.75 | 8x |
| shared expert | 22 | 4.32 | 5x |
| KDA projections | 21 | 8.86 | 2.3x |
| KDA output | 9.4 | 7.54 | 1.2x |
| Python glue, runner, idle, dense layer, embedding, lm_head | 40 | 2.21 | 18x |
| **all but communication** | **823** | **50.1** | **16x** |

The categories do not map one to one between the engines; read the ratios as approximate.

1. **Communication is 86% of the step.** 281 all-reduces per token (184 in the MoE
   experts, 69 KDA o_proj, 24 MLA o_proj), 10 to 14 KB each, 17.7 ms each through Gloo
   over TCP/IPoIB. The time is identical on all ranks (4958 to 5020 ms), so it is
   per-call latency, not skew. The C engine does twice as many collectives, each about
   2000x cheaper (one-sided UCX over the NDR400 HCAs).
2. **Most of the remaining time is the plugin's torch reference ops, not vLLM.** The
   largest item is `mxfp4_dequant`: about 2 calls per MoE layer per rank (about one local
   expert per rank with EP), 1.9 ms each, because each call materialises the expert's
   weights in fp32 before the matmul. AttnRes, MoE latent up, the router and the KDA
   recurrence follow.
3. **vLLM's own kernels are close to the C engine**: the oneDNN-packed bf16 linears (KDA
   projections 2.3x, KDA output 1.2x). Runner overhead is small: 2.3 ms per step plus
   12.7 ms between steps.
4. Expert-parallel load is even (dequant 327 to 368 ms per rank).

Estimate: a transport with tens of microseconds per all-reduce (MPI or UCX over mlx5)
would bring vLLM from 5.8 s to about 0.8 s/token; a fused MXFP4 SiTU expert kernel would
then remove about another 0.4 s. After that, AttnRes and MoE latent up are the next targets.

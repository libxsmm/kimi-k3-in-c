# Kimi K3 behind llama.cpp: performance and reproduction

The engine runs as a compute backend (`k3 --serve SOCKET`) under a lightly patched
llama.cpp. llama.cpp keeps everything a client sees: the tokenizer, the chat template,
sampling, and the OpenAI-compatible HTTP API of `llama-server`. Every decode batch goes to
the engine over a UNIX socket.

On the engine side:

| Feature | What it does |
|---|---|
| Tensor parallelism | Over MPI, one rank per SNC NUMA node |
| AMX-BF16 prompt processing | `K3_PREFILL_AMX=1`, int8 activations with `K3_ACT_Q8=1` |
| State checkpoints | A chat's next turn reuses the cached prefix even though the recurrent KDA state cannot be truncated |
| DSpark speculative decoding | Done inside the engine, so it is invisible to the client |

DSpark details:

- The draft model is [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark).
- Every drafted token is verified by the full model, and the output is bit-identical to plain decode.
- The client still samples; on a mismatch the engine rewinds.

All numbers below were measured on branch `feauture_run_2bit_llamacpp_frontend_mtp`.

## Measured system

| | |
|---|---|
| node | 1 × 2-socket Intel Xeon 6980P (Granite Rapids, 128 cores per socket), SNC3: 6 NUMA nodes of 42/43 cores, 1.5 TB DDR5 |
| software | GCC 14.3.1, CMake 4.3.1, Open MPI 5.0.11 |
| model | [unsloth/Kimi-K3-GGUF](https://huggingface.co/unsloth/Kimi-K3-GGUF) `UD-Q2_K_XL` (19 shards) |
| draft model | [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) (bf16, 6.7 GB) |
| front end | [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) `iq1-narrow` at `ef45f21`, plus [integrations/llama.cpp](../integrations/llama.cpp) |
| layout | TP=6: `mpirun -np 6 --map-by ppr:1:numa --bind-to numa`, 42 OpenMP threads per rank, `K3_EXPERT_Q8=1` |

## Performance summary

### 1. Weight-streaming kernels

Decode is bound by memory bandwidth: every token streams the trunk plus the routed
experts. The measurements:

- Source: [benchmarks/kbench](../benchmarks/kbench) streams 8 GB of weights from DRAM through each
  decode kernel.
- Each figure is the average of 20 runs.
- The "read roof" is a plain read with no arithmetic.

| Scope | Read roof | Q8_0 GEMV (trunk) | IQ2_XS GEMV (experts) |
|---|---|---|---|
| 1 SNC, 42 cores (one TP rank) | 248 GB/s | 218 GB/s (**88%**) | 205 GB/s (**83%**) |
| 1 socket, 128 cores | 762 GB/s | 647 GB/s (85%) | 587 GB/s (77%) |
| node, 256 cores | 1564 GB/s | 1293 GB/s (83%) | 1146 GB/s (73%) |

### 2. Where a decode step goes

Plain decode, TP=6, int8 activations, 125-token prompt, 128 generated tokens, `K3_PROF=1`:
86.6 ms per token. Each rank streams 12.79 GB per token, which is 148 GB/s per rank, or
59% of the SNC roof.

| Phase | ms/token | Share |
|---|---|---|
| KDA projections | 15.6 | 18.0% |
| routed experts | 15.5 | 17.9% |
| KDA output | 12.2 | 14.1% |
| shared experts | 10.1 | 11.7% |
| MLA output | 5.6 | 6.5% |
| KDA core (recurrence) | 5.1 | 5.9% |
| MLA projections | 4.4 | 5.0% |
| MoE up | 3.8 | 4.4% |
| attention residuals and norms | 3.0 | 3.5% |
| MoE down | 2.7 | 3.1% |
| router | 2.1 | 2.4% |
| lm_head | 1.1 | 1.3% |
| dense MLP | 0.7 | 0.8% |
| TP communication | 0.6 | 0.7% |
| untimed | 4.1 | 4.8% |

About 15 ms per token goes to work that streams little or no weight:

- the KDA recurrence;
- attention residuals and norms;
- the router;
- TP communication;
- untimed work.

Over the weight-streaming phases alone, the rate is about 180 GB/s, roughly 73% of the roof.

### 3. End to end through `llama-server`

All requests use greedy sampling (`temperature 0`). Prompt speed is reported for the
123-token and 1296-token prompts.

| Engine configuration | Prompt tok/s (123 / 1296 tokens) | Chat and code, ms/token | 1296-token document, ms/token |
|---|---|---|---|
| exact prefill, fp32 activations | 24 / 22 | 81–85 | 100 |
| `K3_PREFILL_AMX=1 K3_ACT_Q8=1` | 41 / 78 | 85–89 | 110 |
| **+ `--dspark DIR --dspark-n 3`** | **41 / 78** | **50–57** | **77** |
| + `--dspark DIR --dspark-n 4` | 41 / 77 | 49–56 | 81 |

- **Multi-turn chat:** the second turn reuses all 100 cached tokens of the first and
  processes only the 45 new ones. The engine restores the state checkpoint taken at the
  end of the previous prompt.
- **DSpark acceptance:** 2.28 drafts per block with `-n 3`, 2.83 with `-n 4`.
  - The chat, code and document prompts run 1.5× faster than plain decode, at 17–20 tokens/s.
  - Prompt processing is unchanged.
- **Exactness:**
  - Every configuration gives byte-identical output when the same request is sent twice.
  - DSpark output is byte-identical to plain decode for every request.
  - AMX prompt processing is not bit-exact against the exact path, so the first two rows
    generate different, equally valid text.
- **Raw output:** `docs/data/serve-bench-384319.txt`.

**Recommended configuration:**
`K3_PREFILL_AMX=1 K3_ACT_Q8=1 k3 MODEL --serve SOCKET --dspark DIR --dspark-n 3`.
Use `-n 4` when outputs are short and predictable.

## Reproduce from an empty directory

The guide assumes:

- a Slurm cluster with a node like the one above;
- Open MPI 5, GCC ≥ 12, CMake ≥ 3.18, Python 3, git and curl on that node.

Replace `<partition>` with your partition and `<mpi-env.sh>` with the script that puts
Open MPI on your `PATH`; on the measured system that is
`/swtools/openmpi_x86_64/openmpi_5.0.11/openmpi_vars.sh`.

### 1. Workspace and sources

```bash
mkdir k3ws && cd k3ws
git clone --branch feauture_run_2bit_llamacpp_frontend_mtp https://github.com/libxsmm/kimi-k3-in-c
git clone --branch iq1-narrow https://github.com/unslothai/llama.cpp
git -C llama.cpp checkout ef45f21
git -C llama.cpp -c user.name=k3 -c user.email=k3@localhost am ../kimi-k3-in-c/integrations/llama.cpp/*.patch
```

### 2. Models

The GGUF model is about 800 GB and the draft model 6.7 GB.

**If they are already on disk, do not download them again.** Link them instead:

```bash
mkdir -p models
ln -s /path/to/Kimi-K3-GGUF  models/Kimi-K3-GGUF     # must contain UD-Q2_K_XL/*.gguf
ln -s /path/to/Kimi-K3-DSpark models/Kimi-K3-DSpark  # config.json, model.safetensors
```

Otherwise, fetch them once. The download is resumable, so re-run the same command if it stops:

```bash
python3 -m pip install -U "huggingface_hub[cli]" hf_xet
hf download unsloth/Kimi-K3-GGUF --include "UD-Q2_K_XL/*" --local-dir models/Kimi-K3-GGUF
hf download Inferact/Kimi-K3-DSpark --local-dir models/Kimi-K3-DSpark
ls models/Kimi-K3-GGUF/UD-Q2_K_XL/*.gguf | wc -l    # 19
```

### 3. Build the engine on the compute node

The engine compiles with `-march=native`, so build it on the node type it will run on:

```bash
srun -p <partition> -N1 --exclusive bash -c 'source <mpi-env.sh> &&
    make -C kimi-k3-in-c -j64 MPI=1 BUILD=build-mpi-gnr BIN=bin-mpi-gnr bin-mpi-gnr/k3'
```

`UCX=<ucx prefix>` is optional; it adds a UCX transport that is not used on a single node.

### 4. Build llama.cpp

```bash
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF \
      -DGGML_AVX512=ON -DLLAMA_CURL=OFF
cmake --build build -j 32 --target llama-server llama-simple
cd ..
```

llama.cpp only tokenizes and samples here; the model weights are never loaded by it.

On an NFS file system a parallel link can occasionally leave `libllama.so.*` missing. If
that happens, run `cmake --build build --target llama` once, then repeat the line above.

### 5. Smoke test

Run all of this inside one interactive allocation of the node:

```bash
salloc -p <partition> -N1 --ntasks-per-node=6 --exclusive
source <mpi-env.sh>
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY   # curl to localhost
MODEL=$PWD/models/Kimi-K3-GGUF/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf
SOCK=/tmp/k3-$USER.sock

# engine: about 3-4 minutes until "serving on" appears in engine.log
mpirun -np 6 --map-by ppr:1:numa --bind-to numa \
    -x OMP_NUM_THREADS=42 -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
    -x K3_EXPERT_Q8=1 -x K3_PREFILL_AMX=1 -x K3_ACT_Q8=1 \
    kimi-k3-in-c/bin-mpi-gnr/k3 "$MODEL" --serve "$SOCK" --serve-ctx 8192 \
    --dspark models/Kimi-K3-DSpark --dspark-n 3 > engine.log 2>&1 &
until grep -q "serving on" engine.log; do sleep 5; done

# front end
LLAMA_REMOTE_BACKEND=$SOCK llama.cpp/build/bin/llama-server -m "$MODEL" \
    --host 127.0.0.1 --port 18080 -c 8192 -np 1 --jinja \
    --no-warmup --cache-ram 0 --ctx-checkpoints 0 -t 4 > server.log 2>&1 &
until curl -s 127.0.0.1:18080/health | grep -q ok; do sleep 1; done

curl -s 127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."}],"max_tokens":128}' \
  | python3 -m json.tool

kill %2 %1; pkill -9 -f bin-mpi-gnr/k3     # stop both; see the note on leftover ranks below
```

The answer is "The capital of France is Paris.", preceded by the model's reasoning.

`llama-server` flags that matter:

| Flag | Why |
|---|---|
| `-np 1` | The engine holds one sequence. |
| `--cache-ram 0 --ctx-checkpoints 0` | The engine keeps its own state checkpoints. |
| `--no-warmup` | Avoids a pointless forward pass at start-up. |

### 6. The end-to-end table (section 3)

Run from the workspace root:

```bash
sbatch -p <partition> --export=ALL,MPI_ENV=<mpi-env.sh> kimi-k3-in-c/benchmarks/serve/serve-bench.sbatch
```

What the job does:

- Runs [serve-bench.sh](../benchmarks/serve/serve-bench.sh) for the configurations `base`,
  `amx`, `amx_ds3` and `amx_ds4`. Each starts the engine and `llama-server` and sends the
  same five requests.
- Prints one line per request: prompt and generation speed, plus the number of cached
  tokens (`cached 100` on turn 2).
- Ends with the exactness comparisons.
- Results land in `results/serve-bench-<jobid>/`.
- It takes about 40 minutes, most of it the four engine loads.

To run a single configuration, add `CONFIGS=amx_ds3` to `--export`.

### 7. The kernel table (section 1)

```bash
srun -p <partition> -N1 --exclusive kimi-k3-in-c/benchmarks/kbench/roofs.sh 20
```

This builds `kbench` and checks that its kernels match ggml's dequantisation bit for bit.
It then prints GB/s for the read roof, Q8_0 and IQ2_XS at SNC, socket and node scope,
once per repeat.

### 8. The decode profile (section 2)

```bash
srun -p <partition> -N1 --ntasks-per-node=6 --exclusive bash -c 'source <mpi-env.sh> &&
  mpirun -np 6 --map-by ppr:1:numa --bind-to numa -x OMP_NUM_THREADS=42 -x OMP_PROC_BIND=close \
    -x OMP_PLACES=cores -x K3_EXPERT_Q8=1 -x K3_ACT_Q8=1 -x K3_PROF=1 \
    kimi-k3-in-c/bin-mpi-gnr/k3 models/Kimi-K3-GGUF/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf \
    --ids "$(cat kimi-k3-in-c/benchmarks/serve/ids_chat.txt)" --gen 128 --incremental' | grep -A20 "^profile:"
```

## Pitfalls

- **Leftover ranks.** Killing `mpirun` can leave engine ranks running, each holding about
  150 GB. A second engine started on top of them runs much slower and can crash with a
  segfault at a small address while allocating.
  - Always `pkill -9 -f bin-mpi-gnr/k3` before starting a new engine.
  - `serve-bench.sh` does this between configurations.
- **Proxy variables.** With `http_proxy` set, curl sends requests for `127.0.0.1` to the proxy.
- **Startup output.** The engine prints `serving on ...` once all weights are resident. The
  first connection from `llama-server` only reads the model geometry, so
  `client gone after 1 requests` at startup is normal.

#!/bin/bash
#SBATCH -J k3-serve-bench
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --exclusive
#SBATCH -t 02:00:00
# Slurm wrapper for serve-bench.sh on one 2-socket GNR node in SNC3 mode (6 NUMA nodes).
# Submit from the workspace root (the directory holding kimi-k3-in-c/ and llama.cpp/):
#   sbatch -p <partition> kimi-k3-in-c/benchmarks/serve/serve-bench.sbatch
# Paths can be overridden with --export=ALL,MODEL=...,DSPARK=...,MPI_ENV=...
[ -n "${MPI_ENV:-}" ] && source "$MPI_ENV"   # before set -u: site env scripts read unset variables
set -uo pipefail
WS=${WS:-$SLURM_SUBMIT_DIR}
export K3=${K3:-$WS/kimi-k3-in-c/bin-mpi-gnr/k3}
export MODEL=${MODEL:-$WS/models/Kimi-K3-GGUF/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf}
export DSPARK=${DSPARK:-$WS/models/Kimi-K3-DSpark}
export LLAMA_BIN=${LLAMA_BIN:-$WS/llama.cpp/build/bin}
export OUT=${OUT:-$WS/results/serve-bench-$SLURM_JOB_ID}
exec "$WS/kimi-k3-in-c/benchmarks/serve/serve-bench.sh"

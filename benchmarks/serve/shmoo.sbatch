#!/bin/bash
#SBATCH -J k3-shmoo
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --exclusive
#SBATCH -t 08:00:00
# Slurm wrapper for shmoo.sh on one 2-socket GNR node in SNC3 mode (6 NUMA nodes, TP=6).
# Submit from the workspace root (the directory holding kimi-k3-in-c/ and llama.cpp/):
#   sbatch -p <partition> kimi-k3-in-c/benchmarks/serve/shmoo.sbatch
# Paths and shmoo settings can be overridden with --export=ALL,MODEL=...,LENS=...,CFGS=...
[ -n "${MPI_ENV:-}" ] && source "$MPI_ENV"   # before set -u: site env scripts read unset variables
set -uo pipefail
WS=${WS:-$SLURM_SUBMIT_DIR}
export K3=${K3:-$WS/kimi-k3-in-c/bin-mpi-gnr/k3}
export MODEL=${MODEL:-$WS/models/Kimi-K3-GGUF/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf}
export DSPARK=${DSPARK:-$WS/models/Kimi-K3-DSpark}
export LLAMA_BIN=${LLAMA_BIN:-$WS/llama.cpp/build/bin}
export OUT=${OUT:-$WS/results/shmoo-$SLURM_JOB_ID}
exec "$WS/kimi-k3-in-c/benchmarks/serve/shmoo.sh"

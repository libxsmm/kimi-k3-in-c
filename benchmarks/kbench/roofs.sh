#!/bin/bash
# Weight-streaming roofs of one GNR node: a plain read roof and the two decode kernels
# (Q8_0 trunk GEMV, IQ2_XS expert GEMV) at SNC, socket and node scope, 8 GB of weights.
# Usage (on the compute node): benchmarks/kbench/roofs.sh [repeats]
set -euo pipefail
cd "$(dirname "$0")"
gcc -O3 -std=gnu11 -march=native -fopenmp -ffp-contract=off kbench.c -o kbench -lm
./kbench check | tail -1
cores_of() { lscpu -p=CORE,NODE | grep -v '^#' | awk -F, -v n="$1" '$2==n' | sort -u | wc -l; }
nodes=$(lscpu -p=NODE | grep -v '^#' | sort -u | wc -l)
sockets=$(lscpu -p=SOCKET | grep -v '^#' | sort -u | wc -l)
per_socket=$((nodes / sockets))
s0_nodes=$(seq -s, 0 $((per_socket - 1)))
s0_cores=$(lscpu -p=CORE,SOCKET | grep -v '^#' | awk -F, '$2==0' | sort -u | wc -l)
all_cores=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)
run() {   # label threads numactl-args...
    local lab=$1 nt=$2; shift 2
    for k in roof q8_0 iq2_xs_gfni; do
        echo -n "$lab "
        OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close numactl "$@" ./kbench dram $k 8
    done
}
for r in $(seq 1 "${1:-3}"); do
    run "SNC0 $(cores_of 0)c" "$(cores_of 0)" -N 0 -m 0
    run "socket0 ${s0_cores}c" "$s0_cores" -N "$s0_nodes" --localalloc
    run "node ${all_cores}c" "$all_cores" --localalloc
done

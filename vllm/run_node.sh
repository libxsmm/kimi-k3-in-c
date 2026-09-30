#!/bin/bash
# Per-node vLLM CPU launcher for multi-node TP (one srun task per node).
#   srun --jobid=$JOB --overlap -N8 --ntasks-per-node=1 vllm/run_node.sh MODEL [vllm serve args]
# Env: HEAD_IP (node 0 address on IFNAME, required), IFNAME (gloo/zmq interface,
#      default enp50s0f0), TP (default 2*nnodes),
#      PORT (API, default 8000), MPORT (torch.distributed master port, default 29555),
#      DTYPE (default bfloat16)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/.venv/bin/activate"

MODEL=$1; shift
IFNAME=${IFNAME:-enp50s0f0}
NN=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES}}
RANK=${SLURM_NODEID}
TP=${TP:-$((2 * NN))}

ipof() { ip -4 -o addr show "$1" | awk '{print $4}' | cut -d/ -f1 | head -1; }
MY_IP=$(ipof "$IFNAME")
[[ -n "$MY_IP" ]] || { echo "no IPv4 on $IFNAME" >&2; exit 1; }
if [[ $RANK -eq 0 ]]; then HEAD_IP=$MY_IP; else
  HEAD_IP=${HEAD_IP:?set HEAD_IP to node 0 address on $IFNAME}
fi

export VLLM_HOST_IP=$MY_IP GLOO_SOCKET_IFNAME=$IFNAME
export VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-8}
export VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND:-auto}
export HF_HOME=${HF_HOME:-/scratch/aheineck/hf}
export LD_PRELOAD="$HERE/.venv/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$HERE/.venv/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"

ARGS=(--tensor-parallel-size "$TP" --nnodes "$NN" --node-rank "$RANK"
      --master-addr "$HEAD_IP" --master-port "${MPORT:-29555}"
      --distributed-executor-backend mp --dtype "${DTYPE:-bfloat16}")
echo "[$(hostname)] node_rank=$RANK/$NN tp=$TP ip=$MY_IP head=$HEAD_IP if=$IFNAME"
if [[ $RANK -eq 0 ]]; then
  exec vllm serve "$MODEL" "${ARGS[@]}" --host 0.0.0.0 --port "${PORT:-8000}" "$@"
else
  exec vllm serve "$MODEL" "${ARGS[@]}" --headless "$@"
fi

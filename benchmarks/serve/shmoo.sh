#!/bin/bash
# Prefill/decode shmoo: llama-server (with the remote-backend patches) in front of the
# engine's --serve mode, one engine configuration at a time on one node, driven by
# shmoo.py: exact-length uncached prompts over LENS, GEN generated tokens each.
#
# Configurations (CFGS): plain (no speculative decoding) and dsN (DSpark drafting N tokens),
# both with the recommended settings: AMX prefill, int8 activations, int8 expert
# activations, bf16 KV cache (KV_BF16=0 for fp32).
#
# Required environment:
#   K3        engine binary built with MPI=1 (bin-mpi-gnr/k3)
#   MODEL     first shard of Kimi-K3 UD-Q2_K_XL (…-00001-of-00019.gguf)
#   LLAMA_BIN llama.cpp build/bin directory (llama-server)
# Optional:
#   DSPARK    Kimi-K3-DSpark directory (needed for the ds* configurations)
#   CFGS="plain ds3"  LENS="1024 4096 8192 16384 32512"  REPEATS=2  GEN=256
#   WORKLOADS=code,chat  (also code_direct, copy, count)
#   NP=6 THREADS=42 MAP=ppr:1:numa BIND=numa KV_BF16=1 PORT=18080 OUT=./shmoo-<date>
#   CTX       default: the longest prompt plus GEN, rounded up to 1024
#
# System settings (root, not changed here; the run prints them): transparent huge pages
# "always" and automatic NUMA balancing off (kernel.numa_balancing=0).
#
# Ends with a summary: median prefill and decode tok/s per length and configuration.
set -uo pipefail
: "${K3:?set K3}" "${MODEL:?set MODEL}" "${LLAMA_BIN:?set LLAMA_BIN}"
NP=${NP:-6}; THREADS=${THREADS:-42}; MAP=${MAP:-ppr:1:numa}; BIND=${BIND:-numa}
CFGS=${CFGS:-plain ds3}; LENS=${LENS:-1024 4096 8192 16384 32512}; KV_BF16=${KV_BF16:-1}
export REPEATS=${REPEATS:-2} GEN=${GEN:-256} WORKLOADS=${WORKLOADS:-code,chat}
PORT=${PORT:-18080}
max=0; for n in $LENS; do (( n > max )) && max=$n; done
CTX=${CTX:-$(( (max + GEN + 1023) / 1024 * 1024 ))}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/shmoo-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
hostname; date
echo "ranks $NP x $THREADS threads, map $MAP, bind $BIND, context $CTX, lengths $LENS"
echo "workloads $WORKLOADS, $REPEATS repeats, $GEN generated tokens, bf16 KV cache: $KV_BF16"
echo "transparent huge pages: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)" \
     "| NUMA balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null)   (recommended: [always], 0)"

cleanup() {   # mpirun can leave ranks behind (~150 GB each); never start an engine on top of them
    pkill -9 -f "$K3 " 2>/dev/null; sleep 5
    echo "  engine processes left: $(pgrep -fc "$K3 "), $(free -g | awk '/Mem:/{print $7}') GB available"
}
run_cfg() {   # name extra-engine-args
    local name=$1 extra=$2 SOCK=/tmp/k3-shmoo-$$-$1.sock
    echo; echo "=== $name $extra ($(date +%T))"
    mpirun -np "$NP" --map-by "$MAP" --bind-to "$BIND" -x OMP_NUM_THREADS="$THREADS" -x OMP_PROC_BIND=close \
        -x OMP_PLACES=cores -x K3_EXPERT_Q8=1 -x K3_PREFILL_AMX=1 -x K3_ACT_Q8=1 -x K3_KV_BF16="$KV_BF16" \
        "$K3" "$MODEL" --serve "$SOCK" --serve-ctx "$CTX" $extra > "$OUT/engine-$name.log" 2>&1 &
    local EPID=$!
    for i in $(seq 1 900); do
        grep -q "serving on" "$OUT/engine-$name.log" 2>/dev/null && break
        kill -0 $EPID 2>/dev/null || { echo "  engine died"; tail -20 "$OUT/engine-$name.log"; cleanup; return; }
        sleep 2
    done
    LLAMA_REMOTE_BACKEND=$SOCK "$LLAMA_BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" \
        -np 1 --jinja --no-warmup --cache-ram 0 --ctx-checkpoints 0 -t 4 > "$OUT/server-$name.log" 2>&1 &
    local SPID=$!
    for i in $(seq 1 120); do curl -s 127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && break; sleep 1; done
    python3 "$HERE/shmoo.py" "$PORT" "$OUT/$name.tsv" $LENS
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    sleep 2; grep -E "client gone|signal" "$OUT/engine-$name.log" | tail -1
    kill $EPID 2>/dev/null; wait $EPID 2>/dev/null
    cleanup
}
cleanup
for c in $CFGS; do
    case $c in
        plain) run_cfg plain "" ;;
        ds*)   run_cfg "$c" "--dspark ${DSPARK:?set DSPARK} --dspark-n ${c#ds}" ;;
        *)     echo "unknown configuration $c" ;;
    esac
done

echo; echo "=== summary: median tok/s (prefill over all configurations; decode per configuration)"
python3 - "$OUT" $CFGS <<'EOF'
import csv, os, sys
out, cfgs = sys.argv[1], [c for c in sys.argv[2:] if os.path.exists(f"{sys.argv[1]}/{c}.tsv")]
med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
rows = {c: list(csv.DictReader(open(f"{out}/{c}.tsv"), delimiter="\t")) for c in cfgs}
lens = sorted({int(r["len"]) for c in cfgs for r in rows[c]})
print("%8s %10s" % ("length", "prefill") + "".join(" %10s" % ("decode " + c) for c in cfgs))
for n in lens:
    pf = [float(r["prefill_tok_s"]) for c in cfgs for r in rows[c] if int(r["len"]) == n]
    print("%8d %10.1f" % (n, med(pf)) + "".join(
        " %10.2f" % med([float(r["decode_tok_s"]) for r in rows[c] if int(r["len"]) == n]) for c in cfgs))
EOF
date

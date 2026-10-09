#!/bin/bash
# End-to-end benchmark: llama-server (with the remote-backend patches) in front of the
# engine's --serve mode, one engine configuration at a time on one node.
#
# Per configuration: a 2-turn chat (turn 2 must reuse the cached turn 1), a 123-token code
# prompt with 512 generated tokens, a 1296-token document prompt with 256 generated
# tokens, and the code prompt again (determinism). Ends with exactness comparisons.
#
# Required environment:
#   K3        engine binary built with MPI=1 (bin-mpi-gnr/k3)
#   MODEL     first shard of Kimi-K3 UD-Q2_K_XL (…-00001-of-00019.gguf)
#   LLAMA_BIN llama.cpp build/bin directory (llama-server)
# Optional:
#   DSPARK    Kimi-K3-DSpark directory (needed for the *_ds* configurations)
#   CONFIGS   default "base amx amx_ds3 amx_ds4"
#   NP=6 THREADS=42 MAP=ppr:1:numa BIND=numa CTX=8192 PORT=18080 OUT=./serve-bench-<date>
set -uo pipefail
: "${K3:?set K3}" "${MODEL:?set MODEL}" "${LLAMA_BIN:?set LLAMA_BIN}"
NP=${NP:-6}; THREADS=${THREADS:-42}; MAP=${MAP:-ppr:1:numa}; BIND=${BIND:-numa}
CTX=${CTX:-8192}; PORT=${PORT:-18080}; CONFIGS=${CONFIGS:-base amx amx_ds3 amx_ds4}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/serve-bench-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"; cd "$OUT"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
hostname; date
echo '[{"role":"user","content":"What is the capital of France? Answer in one sentence."}]' > m_t1.json

cleanup() {   # mpirun can leave ranks behind (~150 GB each); never start an engine on top of them
    pkill -9 -f "$K3 " 2>/dev/null; sleep 5
    echo "  engine processes left: $(pgrep -fc "$K3 "), $(free -g | awk '/Mem:/{print $7}') GB available"
}
req() {   # messages-file max_tokens out
    python3 -c "import json; print(json.dumps({'messages': json.load(open('$1')), 'max_tokens': $2, 'temperature': 0}))" > req.json
    curl -s -m 1800 127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @req.json > "$3"
}
show() {
    python3 - "$1" "$2" <<'EOF'
import json, sys
try:
    r = json.load(open(sys.argv[1])); t = r.get('timings', {}); m = r['choices'][0]['message']
    print('  %-6s prompt %4d tok (cached %4s) %6.1f tok/s | gen %3d tok %6.2f tok/s (%5.1f ms/tok) | %s' % (
        sys.argv[2], t.get('prompt_n', 0), t.get('cache_n', '?'), t.get('prompt_per_second', 0),
        t.get('predicted_n', 0), t.get('predicted_per_second', 0), t.get('predicted_per_token_ms', 0),
        repr((m.get('content') or '')[:50])))
except Exception as e:
    print('  %-6s FAILED %s' % (sys.argv[2], e))
EOF
}
srv() {   # name env-vars extra-args
    local name=$1 envs=$2 extra=$3 xs=""
    local SOCK=/tmp/k3-serve-$$-$name.sock
    mkdir -p "$OUT/$name"; cd "$OUT/$name"
    echo; echo "=== $name: $envs $extra"
    for e in $envs; do export "$e"; xs="$xs -x ${e%%=*}"; done
    mpirun -np "$NP" --map-by "$MAP" --bind-to "$BIND" -x OMP_NUM_THREADS="$THREADS" -x OMP_PROC_BIND=close \
        -x OMP_PLACES=cores -x K3_EXPERT_Q8=1 $xs "$K3" "$MODEL" --serve "$SOCK" --serve-ctx "$CTX" $extra \
        > engine.log 2>&1 &
    local EPID=$!
    for i in $(seq 1 900); do
        grep -q "serving on" engine.log 2>/dev/null && break
        kill -0 $EPID 2>/dev/null || { echo "  engine died"; tail -20 engine.log; cleanup; cd "$OUT"; return; }
        sleep 2
    done
    for e in $envs; do unset "${e%%=*}"; done
    LLAMA_REMOTE_BACKEND=$SOCK "$LLAMA_BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" \
        -np 1 --jinja --no-warmup --cache-ram 0 --ctx-checkpoints 0 -t 4 > server.log 2>&1 &
    local SPID=$!
    for i in $(seq 1 120); do curl -s 127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && break; sleep 1; done
    req "$OUT/m_t1.json" 192 t1.json; show t1.json turn1
    python3 - <<'EOF'
import json
a = json.load(open('t1.json'))['choices'][0]['message'].get('content') or ''
json.dump([{"role": "user", "content": "What is the capital of France? Answer in one sentence."},
           {"role": "assistant", "content": a}, {"role": "user", "content": "And of Germany?"}], open('m_t2.json', 'w'))
EOF
    req m_t2.json 192 t2.json; show t2.json turn2
    req "$HERE/m_code.json" 512 c1.json; show c1.json code
    req "$HERE/m_long.json" 256 l1.json; show l1.json long
    req "$HERE/m_code.json" 512 c2.json; show c2.json code2
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    sleep 2; grep -E "client gone|signal" engine.log | tail -1
    kill $EPID 2>/dev/null; wait $EPID 2>/dev/null
    cleanup
    cd "$OUT"
}
cleanup
for c in $CONFIGS; do
    case $c in
        base)    srv base    ""                             "" ;;
        amx)     srv amx     "K3_PREFILL_AMX=1 K3_ACT_Q8=1" "" ;;
        amx_ds*) srv "$c"    "K3_PREFILL_AMX=1 K3_ACT_Q8=1" "--dspark ${DSPARK:?set DSPARK} --dspark-n ${c#amx_ds}" ;;
        *)       echo "unknown configuration $c" ;;
    esac
done

echo; echo "=== exactness / determinism"
python3 - $CONFIGS <<'EOF'
import json, sys
cfgs = sys.argv[1:]
def t(p):
    try:
        m = json.load(open(p))['choices'][0]['message']
        return (m.get('reasoning_content') or '') + '|' + (m.get('content') or '')
    except Exception:
        return None
def cmp(a, b):
    x, y = t(a), t(b)
    if x is None or y is None: return 'n/a'
    n = min(len(x), len(y)); i = next((i for i in range(n) if x[i] != y[i]), None)
    return 'identical' if i is None and len(x) == len(y) else 'differ at char %s' % i
for c in cfgs:
    print('  %-8s code twice: %s' % (c, cmp(c + '/c1.json', c + '/c2.json')))
for c in cfgs:
    if c.startswith('amx_ds') and 'amx' in cfgs:
        print('  amx vs %-8s %s' % (c, ', '.join('%s %s' % (f, cmp('amx/%s.json' % f, '%s/%s.json' % (c, f)))
                                               for f in ('t1', 't2', 'c1', 'l1', 'c2'))))
EOF
date

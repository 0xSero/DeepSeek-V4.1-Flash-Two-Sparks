#!/bin/bash
# Serve DeepSeek-V4.1-Flash-Spark (EXL3 routed experts) with vLLM, tensor parallel 2 over two DGX Sparks.
#
# Run on the HEAD Spark (rank 0, serves the API). It starts rank 1 on the WORKER Spark over ssh, then rank 0 here.
#
#   WORKER=user@10.10.10.11 scripts/launch.sh          start both ranks, wait until the API answers
#   scripts/launch.sh stop                             stop both ranks and the memory guards
#   scripts/launch.sh status                           containers, health, memory on both nodes
#   scripts/launch.sh logs [r0|r1]                     follow a rank's container log (default r0)
#
# Required:  WORKER        ssh target of the second Spark, ideally its fabric (QSFP) address, e.g. user@10.10.10.11.
#                          Must work non-interactively (key auth) from the head.
# Optional:  MODEL_DIR     HF download dir on the head (default ~/models/DeepSeek-V4.1-Flash-Spark)
#            WORKER_MODEL_DIR   same dir on the worker (default: MODEL_DIR with the head's $HOME swapped for the worker's)
#            STATE_DIR     api key, plan copy, kernel caches, memguard (default ~/.dsv41-two-sparks, both nodes)
#            HEAD_IP / WORKER_IP          fabric IPv4 of each node (default: auto-detected from the active RoCE ports)
#            HEAD_IFNAME / WORKER_IFNAME  fabric netdev for NCCL/Gloo sockets (default: auto)
#            HEAD_HCAS / WORKER_HCAS      comma list of RDMA devices for NCCL (default: all ACTIVE RoCE devices)
#            PORT (8000)   MPORT (29655)  IMG  API_KEY_FILE  MEMGUARD_GIB (1)  SSH_OPTS  EXTRA (extra vllm args)
#            KV_BYTES MAX_LEN MAX_SEQS NSPEC BATCH GPU_UTIL   (defaults = the measured S016 config)
#
# Never caps outputs; never touches GPU power or clocks.
set -euo pipefail

IMG=${IMG:-ghcr.io/0xsero/deepseek-v4.1-flash-spark@sha256:5668e35e5ee021da4b9ce88a1964caf0e082e8ca01d81d1d64b6213a55c6add5}
PLAN_NAME=${PLAN_NAME:-J268-ho-v31-K5n.json}
MODEL_DIR=${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash-Spark}
STATE_DIR=${STATE_DIR:-$HOME/.dsv41-two-sparks}
API_KEY_FILE=${API_KEY_FILE:-$STATE_DIR/api_key}
PORT=${PORT:-8000}; MPORT=${MPORT:-29655}; MEMGUARD_GIB=${MEMGUARD_GIB:-1}
KV_BYTES=${KV_BYTES:-2700000000}       # 2.7e9 B of fp8 KV -> 2,023,717 tokens at batch 2048
MAX_LEN=${MAX_LEN:-262144}; MAX_SEQS=${MAX_SEQS:-2}; NSPEC=${NSPEC:-7}; BATCH=${BATCH:-2048}
GPU_UTIL=${GPU_UTIL:-0.88}; GRAPH_MODE=${GRAPH_MODE:-FULL_DECODE_ONLY}
R0=dsv41-r0; R1=dsv41-r1
HERE=$(cd "$(dirname "$0")" && pwd)

SSH_OPTS=${SSH_OPTS:-}
wssh() { timeout "${T:-60}" ssh -o BatchMode=yes -o ConnectTimeout=10 $SSH_OPTS "$WORKER" "$@"; }
need_worker() { [ -n "${WORKER:-}" ] || { echo "set WORKER=user@<second-spark-fabric-ip>" >&2; exit 2; }; }

# Prints "<ifname> <ipv4> <hca,hca>" for this node: RDMA devices whose port is ACTIVE, and the first of their
# netdevs that has an IPv4 address (the fabric address vLLM/NCCL must bind to, never WiFi or Tailscale).
DETECT='
hcas=""; ifn=""; ip4=""
for d in /sys/class/infiniband/*; do
  [ -e "$d" ] || continue
  grep -q ACTIVE "$d"/ports/1/state 2>/dev/null || continue
  hcas="${hcas:+$hcas,}$(basename "$d")"
  for n in "$d"/device/net/*; do
    n=$(basename "$n"); a=$(ip -4 -o addr show dev "$n" 2>/dev/null | awk "{print \$4}" | cut -d/ -f1 | head -1)
    [ -z "$ifn" ] && [ -n "$a" ] && { ifn=$n; ip4=$a; }
  done
done
echo "${ifn:-none} ${ip4:-none} ${hcas:-none}"'

case "${1:-start}" in
stop)
  need_worker
  timeout 60 docker rm -f $R0 >/dev/null 2>&1 || true
  WHOME=$(T=20 wssh 'echo $HOME' || echo "$HOME")
  wssh "docker rm -f $R1 >/dev/null 2>&1 || true; bash '${STATE_DIR/#$HOME/$WHOME}/memguard.sh' stop 2>/dev/null || true" || true
  bash "$STATE_DIR/memguard.sh" stop 2>/dev/null || true
  echo stopped; exit 0 ;;
status)
  docker ps -a --filter name=$R0 --format '{{.Names}} {{.Status}}'
  [ -n "${WORKER:-}" ] && T=20 wssh "docker ps -a --filter name=$R1 --format '{{.Names}} {{.Status}}'; awk '/MemAvailable/{printf \"worker MemAvailable %.1f GiB\n\", \$2/1048576}' /proc/meminfo" || true
  awk '/MemAvailable/{printf "head MemAvailable %.1f GiB\n", $2/1048576}' /proc/meminfo
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health || true); echo "health: $code"
  exit 0 ;;
logs)
  if [ "${2:-r0}" = r1 ]; then need_worker; T=86400 wssh "docker logs -f --tail 200 $R1"; else docker logs -f --tail 200 $R0; fi
  exit 0 ;;
start) ;;
*) echo "usage: $0 [start|stop|status|logs [r0|r1]]" >&2; exit 2 ;;
esac

need_worker
command -v python3 >/dev/null || { echo "python3 is required on the head" >&2; exit 1; }
timeout 10 docker info >/dev/null 2>&1 || { echo "docker is not answering on the head (is the daemon up, is $USER in the docker group?)" >&2; exit 1; }

# ---- nodes, paths, fabric -------------------------------------------------------------------------------------
WHOME=$(T=20 wssh 'echo $HOME') || { echo "cannot ssh to $WORKER non-interactively" >&2; exit 1; }
WORKER_MODEL_DIR=${WORKER_MODEL_DIR:-${MODEL_DIR/#$HOME/$WHOME}}
WORKER_STATE_DIR=${STATE_DIR/#$HOME/$WHOME}

read -r h_if h_ip h_hca <<<"$(bash -c "$DETECT")"
read -r w_if w_ip w_hca <<<"$(printf '%s\n' "$DETECT" | T=20 wssh 'bash -s')"
HEAD_IFNAME=${HEAD_IFNAME:-$h_if}; HEAD_IP=${HEAD_IP:-$h_ip}; HEAD_HCAS=${HEAD_HCAS:-$h_hca}
WORKER_IFNAME=${WORKER_IFNAME:-$w_if}; WORKER_IP=${WORKER_IP:-$w_ip}; WORKER_HCAS=${WORKER_HCAS:-$w_hca}
for v in HEAD_IFNAME HEAD_IP HEAD_HCAS WORKER_IFNAME WORKER_IP WORKER_HCAS; do
  [ "${!v}" != none ] || { echo "could not detect $v (no ACTIVE RoCE port with an IPv4 address?). Set it explicitly." >&2; exit 1; }
done
echo "head   $HEAD_IP on $HEAD_IFNAME, HCAs $HEAD_HCAS, model $MODEL_DIR"
echo "worker $WORKER_IP on $WORKER_IFNAME, HCAs $WORKER_HCAS, model $WORKER_MODEL_DIR"

# ---- preflight: weights reassembled on both nodes, image present ------------------------------------------------
CHECK='for f in config.json model.safetensors.index.json model-00005-of-00006.safetensors model-00006-of-00006.safetensors exl3/plan/'"$PLAN_NAME"'; do [ -e "$1/$f" ] || { echo "missing $1/$f"; exit 1; }; done'
bash -c "$CHECK" _ "$MODEL_DIR" || { echo "head: download + reassemble first (scripts/download.sh)" >&2; exit 1; }
printf '%s\n' "$CHECK" | T=20 wssh "bash -s -- '$WORKER_MODEL_DIR'" || { echo "worker: copy the weights first (scripts/copy-to-peer.sh)" >&2; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || timeout 3600 docker pull "$IMG"
T=3600 wssh "docker image inspect '$IMG' >/dev/null 2>&1 || docker pull '$IMG'"

# ---- state: api key, plan with absolute bank paths, caches, memguard ---------------------------------------------
mkdir -p "$STATE_DIR/plans" "$STATE_DIR/cache"
[ -s "$API_KEY_FILE" ] || { (umask 077; python3 -c "import secrets;print(secrets.token_urlsafe(32))" > "$API_KEY_FILE"); echo "new API key in $API_KEY_FILE"; }
# The HF plan names bank dirs relative to exl3/ ("qn:q31:q31s"); inside the container the repo is mounted at /model.
python3 - "$MODEL_DIR/exl3/plan/$PLAN_NAME" "$STATE_DIR/plans/$PLAN_NAME" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
p["qdir"] = ":".join("/model/exl3/" + d.split("/")[-1] for d in p["qdir"].split(":"))
json.dump(p, open(sys.argv[2], "w"))
PY
T=30 wssh "mkdir -p '$WORKER_STATE_DIR/plans' '$WORKER_STATE_DIR/cache'"
scp -q -o BatchMode=yes $SSH_OPTS "$STATE_DIR/plans/$PLAN_NAME" "$WORKER:$WORKER_STATE_DIR/plans/$PLAN_NAME"
cp "$HERE/memguard.sh" "$STATE_DIR/memguard.sh"
scp -q -o BatchMode=yes $SSH_OPTS "$HERE/memguard.sh" "$WORKER:$WORKER_STATE_DIR/memguard.sh"

# ---- vLLM configuration (S016) -----------------------------------------------------------------------------------
CAP=$((MAX_SEQS * (NSPEC + 1)))
CAPS=${CAPS:-$(python3 -c "d=$NSPEC+1;c=$CAP;print(','.join(map(str,sorted(set(list(range(1,min(d,c)+1))+list(range(d,c+1,4))+[c])))))")}
SPEC=$(printf '{"method":"dspark","num_speculative_tokens":%s,"draft_tensor_parallel_size":2,"attention_backend":"B12X","draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":true}' "$NSPEC")
ENGRAM='{"cpu_offload":false,"table_memory":"disk","disk_resident_scales":false,"disk_prefetch_max_tokens":0,"projection_tp":false}'
COMP=$(printf '{"cudagraph_mode":"%s","custom_ops":["all"],"cudagraph_capture_sizes":[%s]}' "$GRAPH_MODE" "$CAPS")

envs() {  # $1 ifname, $2 hcas, $3 host ip
  local a=(-e CUDA_VISIBLE_DEVICES=0 -e CUTE_DSL_ARCH=sm_121a -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0
    -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    -e OMP_NUM_THREADS=16 -e MALLOC_ARENA_MAX=2 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1
    -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none -e NCCL_IB_GID_INDEX=3
    -e NCCL_IB_HCA="$2" -e NCCL_IB_MERGE_NICS=1 -e NCCL_SOCKET_IFNAME="$1" -e GLOO_SOCKET_IFNAME="$1" -e NCCL_DEBUG=WARN
    -e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB
    -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce -e B12X_ROCE_TRAFFIC_CLASS=106 -e NCCL_IB_TC=106
    -e B12X_COMPILE_CACHE_DIR=/cache/b12x -e XDG_CACHE_HOME=/cache
    -e ST_EXL3_PLAN=/plans/$PLAN_NAME -e ST_EXL3_PREFILL=st -e ST_EXL3_PREFILL_MIN_ROWS=1
    -e VLLM_HOST_IP="$3")
  printf '%q ' "${a[@]}"
}
dock() {  # $1 model dir, $2 state dir
  printf '%q ' -d --gpus all --network host --ipc host --privileged --ulimit memlock=-1 --shm-size 32g \
    -v "$1:/model:ro" -v "$2/plans:/plans:ro" -v "$2/cache:/cache"
}
ARGS=(/model --served-model-name deepseek-v4.1-flash --dtype bfloat16 --tensor-parallel-size 2
  --nnodes 2 --master-addr "$HEAD_IP" --master-port "$MPORT"
  --kv-cache-dtype fp8 --block-size 256 --swa-block-size 128 --kv-cache-memory-bytes "$KV_BYTES" --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCH"
  --max-cudagraph-capture-size "$CAP" --compilation-config "$COMP"
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching --safetensors-load-strategy lazy
  --engram-config "$ENGRAM" --attention-backend B12X --linear-backend b12x --moe-backend b12x
  --speculative-config "$SPEC" --generation-config vllm --tokenizer-mode deepseek_v41
  --limit-mm-per-prompt '{"image":1}')
q() { printf '%q ' "$@"; }

# ---- memory guards (both nodes), then rank 1 on the worker, rank 0 here ------------------------------------------
G="nohup setsid bash $WORKER_STATE_DIR/memguard.sh $MEMGUARD_GIB >/dev/null 2>&1 < /dev/null &"
T=30 wssh "$G"
nohup setsid bash "$STATE_DIR/memguard.sh" "$MEMGUARD_GIB" >/dev/null 2>&1 < /dev/null &

T=120 wssh "docker rm -f $R1 >/dev/null 2>&1; docker run --name $R1 $(dock "$WORKER_MODEL_DIR" "$WORKER_STATE_DIR") \
  $(envs "$WORKER_IFNAME" "$WORKER_HCAS" "$WORKER_IP") $(q "$IMG") $(q "${ARGS[@]}") --node-rank 1 --headless" >/dev/null
timeout 60 docker rm -f $R0 >/dev/null 2>&1 || true
eval "docker run --name $R0 $(dock "$MODEL_DIR" "$STATE_DIR") $(envs "$HEAD_IFNAME" "$HEAD_HCAS" "$HEAD_IP") $(q "$IMG") $(q "${ARGS[@]}") \
  --node-rank 0 --host 0.0.0.0 --port $PORT --api-key \"\$(cat $(q "$API_KEY_FILE"))\" \
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --enable-prompt-tokens-details --enable-force-include-usage ${EXTRA:-}" >/dev/null
echo "started $R0 here and $R1 on $WORKER (KV $KV_BYTES B, max len $MAX_LEN, seqs $MAX_SEQS, DSpark k=$NSPEC)"

# ---- wait for the API (first boot ~12 min: weights, EXL3 banks, kernel compile, CUDA graphs) ---------------------
[ "${WAIT:-1}" = 1 ] || exit 0
KEY=$(cat "$API_KEY_FILE"); t0=$(date +%s)
while :; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $KEY" http://127.0.0.1:$PORT/v1/models || true)
  [ "$code" = 200 ] && { echo "ready after $(( $(date +%s) - t0 )) s: http://$HEAD_IP:$PORT/v1  model deepseek-v4.1-flash"; exit 0; }
  r0=$(docker inspect -f '{{.State.Running}}' $R0 2>/dev/null || echo false)
  r1=$(T=20 wssh "docker inspect -f '{{.State.Running}}' $R1 2>/dev/null" || echo unknown)
  if [ "$r0" != true ] || [ "$r1" = false ]; then
    echo "a rank exited (r0 running=$r0, r1 running=$r1). Last log lines:" >&2
    docker logs --tail 40 $R0 2>&1 | sed 's/^/r0| /' >&2 || true
    T=20 wssh "docker logs --tail 40 $R1 2>&1" | sed 's/^/r1| /' >&2 || true
    exit 1
  fi
  [ $(( $(date +%s) - t0 )) -gt 2700 ] && { echo "not ready after 45 min; check scripts/launch.sh logs" >&2; exit 1; }
  printf '.'; sleep 15
done

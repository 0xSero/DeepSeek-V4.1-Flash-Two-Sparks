#!/bin/bash
# Copy the downloaded, reassembled checkpoint to the second Spark over the QSFP fabric, then verify it there.
#   WORKER=user@10.10.10.11 scripts/copy-to-peer.sh [MODEL_DIR]
# WORKER should be the peer's fabric address (fast link), not its WiFi/Tailscale name.
# Env: WORKER_MODEL_DIR (default: MODEL_DIR with the local $HOME swapped for the peer's), SSH_OPTS.
# Resumable (rsync --partial). Alternative: run scripts/download.sh on the peer as well.
set -euo pipefail
: "${WORKER:?set WORKER=user@<second-spark-fabric-ip>}"
MODEL_DIR=${1:-${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash-Spark}}
SSH_OPTS=${SSH_OPTS:-}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 $SSH_OPTS"
[ -f "$MODEL_DIR/model-00006-of-00006.safetensors" ] || { echo "run scripts/download.sh first" >&2; exit 1; }
WHOME=$(timeout 20 $SSH "$WORKER" 'echo $HOME')
WORKER_MODEL_DIR=${WORKER_MODEL_DIR:-${MODEL_DIR/#$HOME/$WHOME}}
timeout 20 $SSH "$WORKER" "mkdir -p '$WORKER_MODEL_DIR' && command -v rsync >/dev/null" \
  || { echo "peer needs rsync (sudo apt install rsync)" >&2; exit 1; }

echo "rsync $MODEL_DIR/ -> $WORKER:$WORKER_MODEL_DIR/ (~462 GB)"
# Skip the Engram split parts (the peer gets the rebuilt shards) and the hf download cache.
rsync -a --partial --info=progress2 -e "$SSH" \
  --exclude '*.safetensors.part[0-9][0-9]' --exclude '.cache/' \
  "$MODEL_DIR/" "$WORKER:$WORKER_MODEL_DIR/"

echo "verifying on the peer"
timeout 3600 $SSH "$WORKER" "cd '$WORKER_MODEL_DIR' && grep -E 'model-0000[1-6]-of-00006\.safetensors\$' sha256-manifest.txt | sha256sum -c -"
l=$(find "$MODEL_DIR/exl3" -type f | wc -l); r=$(timeout 60 $SSH "$WORKER" "find '$WORKER_MODEL_DIR/exl3' -type f | wc -l")
[ "$l" = "$r" ] || { echo "exl3 file count differs: local $l, peer $r" >&2; exit 1; }
echo "peer ready: $WORKER:$WORKER_MODEL_DIR ($r exl3 files)"

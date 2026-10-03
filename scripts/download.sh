#!/bin/bash
# Download 0xSero/DeepSeek-V4.1-Flash-Spark (~462 GB), rebuild the two split Engram shards, verify sha256.
#   scripts/download.sh [MODEL_DIR]          default ~/models/DeepSeek-V4.1-Flash-Spark
# Env: REVISION (pin a commit), KEEP_PARTS=1 (keep the .partNN files after verification; they are ~197 GB),
#      HF_TOKEN (only if the repo is private for you).
# Safe to re-run: hf download resumes, reassemble skips shards that already exist.
set -euo pipefail
REPO=0xSero/DeepSeek-V4.1-Flash-Spark
MODEL_DIR=${1:-${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash-Spark}}
STATE_DIR=${STATE_DIR:-$HOME/.dsv41-two-sparks}
mkdir -p "$MODEL_DIR"

# Free space: 462 GB of files plus ~197 GB while the Engram parts and the rebuilt shards coexist.
free_gb=$(df -Pk "$MODEL_DIR" | awk 'NR==2{print int($4/1048576)}')
have_gb=$(du -sk "$MODEL_DIR" 2>/dev/null | awk '{print int($1/1048576)}')
if [ $((free_gb + have_gb)) -lt 680 ]; then
  echo "warning: ${free_gb} GB free at $MODEL_DIR (+${have_gb} GB already there); a full download and reassembly needs ~680 GB" >&2
fi

# hf CLI: use one on PATH, else a private venv (DGX OS blocks pip --user, PEP 668).
HF=$(command -v hf || true)
if [ -z "$HF" ]; then
  [ -x "$STATE_DIR/venv/bin/hf" ] || { python3 -m venv "$STATE_DIR/venv" && "$STATE_DIR/venv/bin/pip" install -q -U "huggingface_hub[hf_xet]"; }
  HF=$STATE_DIR/venv/bin/hf
fi

EXCL=()   # shards already rebuilt by an earlier run: do not fetch their ~197 GB of parts again
[ -f "$MODEL_DIR/model-00005-of-00006.safetensors" ] && [ -f "$MODEL_DIR/model-00006-of-00006.safetensors" ] \
  && EXCL=(--exclude '*.safetensors.part*')
echo "downloading $REPO -> $MODEL_DIR"
"$HF" download "$REPO" ${REVISION:+--revision "$REVISION"} --local-dir "$MODEL_DIR" ${EXCL[@]+"${EXCL[@]}"}

cd "$MODEL_DIR"
bash ./reassemble.sh                 # cat model-0000{5,6}-of-00006.safetensors.partNN -> shard, sha256 check of both
echo "verifying the other native shards against sha256-manifest.txt"
grep -E 'model-0000[1-4]-of-00006\.safetensors$' sha256-manifest.txt | sha256sum -c -
[ -d exl3/qn ] && [ -d exl3/q31 ] && [ -d exl3/q31s ] && [ -f exl3/plan/J268-ho-v31-K5n.json ] \
  || { echo "exl3/ banks or plan missing; re-run this script" >&2; exit 1; }
n=$(find exl3 -name 'L??.part.safetensors' | wc -l); echo "EXL3 bank files: $n"

if [ "${KEEP_PARTS:-0}" != 1 ]; then
  rm -f model-0000[56]-of-00006.safetensors.part[0-9][0-9]
  echo "removed the Engram .partNN files (re-running hf download would fetch them again; KEEP_PARTS=1 keeps them)"
fi
du -sh "$MODEL_DIR"
echo "done. Next: WORKER=user@<peer-fabric-ip> scripts/copy-to-peer.sh $MODEL_DIR"

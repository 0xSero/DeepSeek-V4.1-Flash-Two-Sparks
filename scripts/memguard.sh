#!/bin/bash
# Host memory guard for DGX Spark (GB10 unified memory). The GPU and the OS share one 128 GB pool: if the model
# pushes MemAvailable to zero, the host does not OOM-kill cleanly, it wedges until the hardware watchdog reboots it.
# This loop kills the dsv41-r* containers first when MemAvailable drops below FLOOR GiB.
#   memguard.sh [FLOOR_GIB]   start (replaces a running instance)      memguard.sh stop
# launch.sh starts it on both nodes with FLOOR=1 (the S016 config runs at ~2-3 GiB available in steady state).
DIR=$(cd "$(dirname "$0")" && pwd); LOG=$DIR/memguard.log; PID=$DIR/memguard.pid
FLOOR=${1:-1}
if [ "$FLOOR" = stop ]; then [ -f "$PID" ] && kill "$(cat "$PID")" 2>/dev/null; rm -f "$PID"; exit 0; fi
[ -f "$PID" ] && kill "$(cat "$PID")" 2>/dev/null
echo $$ > "$PID"
echo "$(date -Is) memguard start floor ${FLOOR}G pid $$" >> "$LOG"
n=0
while sleep 1; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  n=$((n+1))
  if [ "$a" -lt 10 ] && [ $((n % 10)) = 0 ]; then   # low-memory trace: what holds host RSS
    echo "$(date -Is) avail ${a}G top-rss: $(ps -eo rss,comm --sort=-rss | sed -n 2,4p | awk '{printf "%s=%.1fG ", $2, $1/1048576}')" >> "$LOG"
  fi
  if [ "$a" -lt "$FLOOR" ]; then
    ids=$(docker ps -q --filter name=dsv41-r)
    [ -n "$ids" ] && { echo "$(date -Is) MemAvailable ${a}G < ${FLOOR}G: killing $ids" >> "$LOG"; docker kill $ids >> "$LOG" 2>&1; }
  fi
done

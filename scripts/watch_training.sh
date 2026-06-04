#!/bin/bash
# Training crash/stall watchdog. Polls every INTERVAL seconds and appends a one-line
# status to results/rl/training_watchdog.log. Flags three failure modes that the bursty
# 1%-GPU-util reading does NOT catch:
#   1. PROCESS_DEAD  — no rl_train_cube python process running
#   2. LOG_STALLED   — the train log hasn't been written to in > STALL_SEC seconds
#   3. NAN/TRACEBACK — "nan" or a Python traceback appeared in the tail of the log
# On any failure it writes a line starting with "CRASH" and (if FLAG_FILE set) touches it,
# so a supervising loop can detect the crash without re-reading the whole log.
#
# Usage (detached):
#   nohup setsid bash scripts/watch_training.sh > /dev/null 2>&1 < /dev/null & disown
# Env: LOG (default = newest *_train.log), INTERVAL=120, STALL_SEC=300, FLAG_FILE
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
INTERVAL=${INTERVAL:-120}
STALL_SEC=${STALL_SEC:-300}
LOG=${LOG:-$(ls -t "$ROOT"/*_train.log 2>/dev/null | head -1)}
FLAG_FILE=${FLAG_FILE:-}
OUT="$ROOT/results/rl/training_watchdog.log"
mkdir -p "$(dirname "$OUT")"

say(){ echo "$(date '+%F %T') $*" | tee -a "$OUT"; }
say "WATCH start log=$LOG interval=${INTERVAL}s stall=${STALL_SEC}s"

while true; do
  ts=$(date '+%F %T')
  procs=$(pgrep -fc "scripts/rl_train_cube.py" || true)
  # log freshness
  if [ -f "$LOG" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    last_iter=$(grep -aoE "Learning iteration [0-9]+/[0-9]+" "$LOG" | tail -1)
  else
    age=-1; last_iter="(no log)"
  fi
  nan=$(tail -n 200 "$LOG" 2>/dev/null | grep -ciE "\bnan\b|Traceback|CUDA error|out of memory" || true)
  done_marker=$(tail -n 40 "$LOG" 2>/dev/null | grep -c "rl_train_cube] DONE" || true)

  status="OK"
  if [ "${done_marker:-0}" -gt 0 ]; then status="DONE"
  elif [ "${procs:-0}" -eq 0 ]; then status="CRASH PROCESS_DEAD"
  elif [ "$age" -gt "$STALL_SEC" ]; then status="CRASH LOG_STALLED ${age}s"
  elif [ "${nan:-0}" -gt 0 ]; then status="CRASH NAN/TRACEBACK"
  fi

  say "$status procs=$procs log_age=${age}s | $last_iter"
  if [ "${status#CRASH}" != "$status" ]; then
    [ -n "$FLAG_FILE" ] && echo "$ts $status" > "$FLAG_FILE"
    say "WATCH stopping after $status — inspect $LOG"
    exit 1
  fi
  [ "$status" = "DONE" ] && { say "WATCH stopping: training DONE"; exit 0; }
  sleep "$INTERVAL"
done

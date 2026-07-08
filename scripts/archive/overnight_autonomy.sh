#!/bin/bash
# Overnight autonomy orchestrator.
#
# Survives SSH/laptop disconnect (run via setsid). Waits for the detached Stage-1
# smoothness sweep to finish, dumps a metrics summary, then triggers a bypass-
# permissions Claude Code run to assess Stage 1 + launch Stage 2 (quick mechanisms).
# Then waits for Stage 2 and triggers a second Claude run to finalize + document.
#
# Launch (detached) with:
#   cd /home/humanoid/Programs/hand
#   nohup setsid bash scripts/overnight_autonomy.sh > overnight_autonomy.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"
CLAUDE=/home/humanoid/.local/bin/claude
log(){ echo "[$(date '+%F %T')] $*"; }

train_running(){ pgrep -f rl_train_cube.py >/dev/null 2>&1 || pgrep -f queue_reorient_smooth_sweep >/dev/null 2>&1; }

dump_metrics(){  # $1 = output file
  local out="$1"; : > "$out"
  for L in 5x 10x; do
    local f="$ROOT/policyB_v2_$L.log"
    [ -f "$f" ] || continue
    {
      echo "########## $L  ($(basename "$f")) ##########"
      echo "iters logged: $(grep -c 'Iteration time' "$f")   NaN/traceback: $(grep -ciE 'nan|traceback' "$f")"
      echo "--- tip_lost trend (every ~50 iters) ---"
      grep -E 'Episode_Termination/tip_lost' "$f" | awk 'NR%50==1{print "  iter~"NR": "$2}'
      echo "--- alignment_success trend ---"
      grep -E 'Episode_Termination/alignment_success' "$f" | awk 'NR%50==1{print "  iter~"NR": "$2}'
      echo "--- FINAL metrics block ---"
      grep -E 'Mean reward|target_axis_alignment:|target_axis_progress:|object_ang_acc_l2:|action_rate_l2:|alignment_success_bonus:|alignment_speed_bonus:|reorient_time_cost:|Metrics/lift_height/object_height|Episode_Termination/(tip_lost|time_out|object_drop|alignment_success)|Curriculum/smoothness' "$f" | tail -16
      echo
    } >> "$out"
  done
  log "wrote $out"
}

run_claude(){  # $1 = prompt file  $2 = log file
  log "invoking Claude: prompt=$1 log=$2"
  "$CLAUDE" -p "$(cat "$1")" \
    --dangerously-skip-permissions \
    --model opus \
    --max-turns 300 \
    > "$2" 2>&1
  log "Claude run finished (prompt=$1, rc=$?)"
}

log "================ OVERNIGHT AUTONOMY START ================"

# ---- 1. Wait for Stage 1 to finish --------------------------------------
log "waiting for Stage 1 sweep to finish..."
sleep 60                                   # let it be clearly running first
while train_running; do sleep 60; done
log "Stage 1 finished."
sleep 10
dump_metrics "$ROOT/STAGE1_RESULTS.txt"

# ---- 2. Claude run #1: assess Stage 1 + launch Stage 2 ------------------
run_claude "$ROOT/scripts/overnight_prompt_1.md" "$ROOT/overnight_claude_1.log"

# ---- 3. Wait for Stage 2 to spawn, then finish --------------------------
log "waiting up to 10 min for Stage 2 to spawn..."
for _ in $(seq 1 60); do train_running && break; sleep 10; done
if train_running; then
  log "Stage 2 running; waiting for it to finish..."
  while train_running; do sleep 60; done
  log "Stage 2 finished."
else
  log "WARNING: no Stage-2 training detected after Claude run #1; proceeding to finalize anyway."
fi
sleep 10
dump_metrics "$ROOT/STAGE2_RESULTS.txt"

# ---- 4. Claude run #2: finalize + document ------------------------------
run_claude "$ROOT/scripts/overnight_prompt_2.md" "$ROOT/overnight_claude_2.log"

log "================ OVERNIGHT AUTONOMY DONE ================"

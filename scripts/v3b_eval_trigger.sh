#!/bin/bash
# Trigger: wait for the v3b normal-lift grace-window runs (repro + soft) to finish, evaluate
# them honestly (deterministic held-cos + continuous-handoff min-z), render the seamless A->B
# video for the best holder, write STATE_HANDOFF_RESULTS.txt, then spawn a FRESH bypass-perms
# Claude Code run (overnight_prompt_v3b_document.md) to assess + document + synthesize + STOP.
# Survives SSH/laptop sleep (setsid).
#
# Launch (detached):
#   cd /home/humanoid/Programs/hand
#   nohup setsid bash scripts/v3b_eval_trigger.sh > v3b_eval_trigger.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
CLAUDE=/home/humanoid/.local/bin/claude
RESULTS="$ROOT/STATE_HANDOFF_RESULTS.txt"
BASE="$ROOT/results/rl/20260602-1636-policyB_abl_signed:model_405.pt"
P2="$ROOT/results/rl/20260603-1746-policyB_p2_lateral_only:model_541.pt"
log(){ echo "[$(date '+%F %T')] $*"; }
ck(){ ls -t "$1"/tensorboard/model_*.pt 2>/dev/null | head -1; }

log "waiting for v3b (rl_train_cube) to finish..."
sleep 120
while pgrep -f rl_train_cube.py >/dev/null 2>&1; do sleep 60; done
log "training finished. evaluating."

R=$(ls -dt "$ROOT"/results/rl/*policyB_normallift_v3b_repro 2>/dev/null | head -1)
S=$(ls -dt "$ROOT"/results/rl/*policyB_normallift_v3b_soft 2>/dev/null | head -1)
RCK=$(ck "$R"); SCK=$(ck "$S")

{
  echo "# v3b grace-window handoff results ($(date '+%F %T'))"
  echo "## deterministic held-cos / jerk / min_z / drop (standalone, own normal-lift env)"
  args=("signed+critic=$BASE" "P2_lateral=$P2")
  [ -n "$RCK" ] && args+=("v3b_repro=$R:$(basename "$RCK")")
  [ -n "$SCK" ] && args+=("v3b_soft=$S:$(basename "$SCK")")
  WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python \
    "$ROOT/scripts/rl_eval_reorient_metrics.py" "${args[@]}" 2>&1 \
    | grep -E "policy|signed|P2_|v3b_|ERROR" | grep -vE "VIRTUAL|Recording|Saved|INFO|cuda"
  echo "## continuous A->B handoff hold (min-z>0.05 = B holds A's delivery; handoff-step 40, blend 8)"
  for nm in repro soft; do
    D=$(ls -dt "$ROOT"/results/rl/*policyB_normallift_v3b_$nm 2>/dev/null | head -1)
    C=$(ck "$D"); [ -z "$C" ] && { echo "### v3b_$nm: NO CHECKPOINT (NaN'd?)"; continue; }
    echo "### v3b_$nm  ($C)"
    WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python \
      "$ROOT/scripts/rl_demo_handoff_continuous.py" --policy-b "$C" \
      --output "$ROOT/docs/rl/videos/reorient/handoff_v3b_$nm.mp4" \
      --handoff-step 40 --blend-steps 8 --total-steps 240 2>&1 \
      | grep -E "object z at handoff|min object-center z"
  done
  echo "## NOTE: best holder's video = docs/rl/videos/reorient/handoff_v3b_{repro,soft}.mp4;"
  echo "## if min-z>0.05 that is the SEAMLESS HANDOFF (the goal). Judge held-cos vs P2 0.988."
} > "$RESULTS" 2>&1
log "wrote $RESULTS"

cd "$ROOT" && git add -A && git commit -q -m "results: v3b grace-window handoff eval (auto)
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>/dev/null || true

log "spawning v3b document Claude run (bypass perms)..."
"$CLAUDE" -p "$(cat "$ROOT/scripts/overnight_prompt_v3b_document.md")" \
  --dangerously-skip-permissions --model opus --max-turns 300 \
  > "$ROOT/v3b_document_claude.log" 2>&1
log "v3b document Claude run finished (rc=$?). ============ DONE ============"

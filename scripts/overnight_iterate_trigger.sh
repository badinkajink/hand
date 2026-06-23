#!/bin/bash
# Trigger: wait for the in-progress P1/P2/P3 reorient runs to finish, evaluate them
# (deterministic held-cos + handoff hold), write STATE_HANDOFF_RESULTS.txt, then spawn a
# FRESH bypass-permissions Claude Code run (overnight_prompt_iterate.md) to analyze,
# iterate, launch new runs, and document/synthesize. Survives SSH/laptop sleep (setsid).
#
# Launch (detached):
#   cd /home/humanoid/Programs/hand
#   nohup setsid bash scripts/overnight_iterate_trigger.sh > iterate_trigger.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
CLAUDE=/home/humanoid/.local/bin/claude
RESULTS="$ROOT/STATE_HANDOFF_RESULTS.txt"
log(){ echo "[$(date '+%F %T')] $*"; }

log "waiting for P1/P2/P3 (rl_train_cube) to finish..."
sleep 120
while pgrep -f rl_train_cube.py >/dev/null 2>&1; do sleep 60; done
log "training finished. evaluating."

P1=$(ls -dt "$ROOT"/results/rl/*policyB_p1_handoff_dronly 2>/dev/null | head -1)
P2=$(ls -dt "$ROOT"/results/rl/*policyB_p2_lateral_only 2>/dev/null | head -1)
P3=$(ls -dt "$ROOT"/results/rl/*policyB_p3_statebank 2>/dev/null | head -1)
ck(){ ls -t "$1"/tensorboard/model_*.pt 2>/dev/null | head -1; }

{
  echo "# P1/P2/P3 results ($(date '+%F %T'))"
  echo "## deterministic held-cos / jerk / min_z / drop"
  WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_eval_reorient_metrics.py" \
    "signed+critic=$ROOT/results/rl/b03_20260602-1636-policyB_abl_signed:model_405.pt" \
    "P1_handoffDR=$P1:$(basename "$(ck "$P1")")" \
    "P2_lateral=$P2:$(basename "$(ck "$P2")")" \
    "P3_statebank=$P3:$(basename "$(ck "$P3")")" 2>&1 | grep -E "policy|signed|P[123]_|ERROR" | grep -vE "VIRTUAL|Recording|Saved|INFO|cuda"
  echo "## handoff-hold (min-z>0.05 = B holds A's delivery)"
  for tag in policyB_p1_handoff_dronly policyB_p3_statebank; do
    D=$(ls -dt "$ROOT"/results/rl/*$tag 2>/dev/null | head -1)
    echo "### $tag"
    WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_demo_handoff_continuous.py" \
      --policy-b "$(ck "$D")" --output "$ROOT/docs/rl/videos/reorient/handoff_${tag}.mp4" \
      --handoff-step 45 --total-steps 240 2>&1 | grep -E "object z at handoff|min object-center z"
  done
  echo "## de-centering note: compare P2 xy-drift to signed+critic 5.1cm (use /tmp/diag2.py pattern)."
} > "$RESULTS" 2>&1
log "wrote $RESULTS"

# preserve work before spawning the agent
cd "$ROOT" && git add -A && git commit -q -m "results: P1/P2/P3 reorient handoff/de-centering eval (auto)
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>/dev/null || true

log "spawning iterate Claude run (bypass perms)..."
"$CLAUDE" -p "$(cat "$ROOT/scripts/overnight_prompt_iterate.md")" \
  --dangerously-skip-permissions --model opus --max-turns 400 \
  > "$ROOT/iterate_claude.log" 2>&1
log "iterate Claude run finished (rc=$?). ============ DONE ============"

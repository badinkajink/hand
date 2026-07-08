#!/bin/bash
# Trigger: wait for B12 (smoothness finetune of B10) + B13 (soft-committing) to finish, eval
# them honestly (held-cos/jerk standalone + continuous A->B handoff min-z), render comparison
# videos (incl. deploy-time blend + critic-gate on B10), write STATE_HANDOFF_RESULTS.txt, then
# spawn a FRESH bypass-perms Claude conversation to assess + document into rl.typ / RESEARCH_STATE.
#
# Launch (detached):
#   nohup setsid bash scripts/handoff_iter2_eval_trigger.sh > handoff_iter2_eval_trigger.log 2>&1 < /dev/null & disown
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
CLAUDE=/home/humanoid/.local/bin/claude
RESULTS="$ROOT/STATE_HANDOFF_RESULTS.txt"
VID="$ROOT/docs/rl/videos/reorient"
ASSETS="$ROOT/webpaper/src/assets"
B4="$ROOT/results/rl/b04_20260603-1746-policyB_p2_lateral_only"
B10="$ROOT/results/rl/b10_20260604-1642-policyB_holdonlyws_repro"
log(){ echo "[$(date '+%F %T')] $*"; }
ck(){ ls -t "$1"/tensorboard/model_*.pt 2>/dev/null | head -1; }
render(){ # $1 ckpt  $2 out  $3... extra flags ; prints min-z line
  timeout 1800 env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python \
    "$ROOT/scripts/rl_demo_handoff_continuous.py" --policy-b "$1" --output "$2" \
    --total-steps 240 "${@:3}" 2>&1 | grep -E "object z at handoff|min object-center z|critic-gated switch"
}

log "waiting for B12 + B13 (+ the B12 retry wrapper) to finish..."
sleep 120
while pgrep -f rl_train_cube.py >/dev/null 2>&1 || pgrep -f _relaunch_B12.sh >/dev/null 2>&1; do sleep 60; done
log "training finished. evaluating."

B12=$(ls -dt "$ROOT"/results/rl/*policyB_handoff_B12_smooth 2>/dev/null | head -1)
B13=$(ls -dt "$ROOT"/results/rl/*policyB_handoff_B13_softcommit 2>/dev/null | head -1)
B12C=$(ck "$B12"); B13C=$(ck "$B13"); B10C=$(ck "$B10")

{
  echo "# Handoff iter2 results (B12 smooth / B13 soft-commit) ($(date '+%F %T'))"
  echo "## standalone held-cos / peak / obj_jerk / min_z / drop"
  echo "## (NOTE: standalone env is SKIP-LIFT = OOD for normal-lift Bs; drop here is an artifact."
  echo "##  Judge survival on the continuous-handoff min-z below, quality on held-cos, violence on obj_jerk.)"
  echo "## reference: B4_lateral held-cos 0.990 jerk 27 ; B10_repro held-cos 0.977 jerk 108 (violent)"
  args=("B4_lateral=$B4:model_541.pt" "B10_repro=$B10:model_541.pt")
  [ -n "$B12C" ] && args+=("B12_smooth=$B12:$(basename "$B12C")")
  [ -n "$B13C" ] && args+=("B13_softcommit=$B13:$(basename "$B13C")")
  timeout 1200 env WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python \
    "$ROOT/scripts/rl_eval_reorient_metrics.py" "${args[@]}" 2>&1 \
    | grep -E "policy|held_cos|B4_|B10_|B12_|B13_|ERROR" | grep -vE "Module|compiled|INFO|cuda"

  echo "## continuous A->B handoff (normal-lift, no reset). min-z>0.05 = B SURVIVES the seam."
  echo "### B10_repro  hard switch @40"
  render "$B10C" "$VID/handoff_B10_hard.mp4" --handoff-step 40
  echo "### B10_repro  blend-12 (deploy-time action ramp — does blending tame the violence?)"
  render "$B10C" "$VID/handoff_B10_blend12.mp4" --handoff-step 40 --blend-steps 12
  echo "### B10_repro  critic-gated switch (B10 critic is IN-dist now — does it pick a clean seam?)"
  render "$B10C" "$VID/handoff_B10_criticgate.mp4" --switch-on-critic --blend-steps 8
  if [ -n "$B12C" ]; then echo "### B12_smooth  blend-8"; render "$B12C" "$VID/handoff_B12_smooth.mp4" --handoff-step 40 --blend-steps 8; fi
  if [ -n "$B13C" ]; then echo "### B13_softcommit  blend-8"; render "$B13C" "$VID/handoff_B13_softcommit.mp4" --handoff-step 40 --blend-steps 8; fi
  echo "## videos in $VID/handoff_B1*.mp4 ; copy the keepers into webpaper/src/assets/ when documenting."
} > "$RESULTS" 2>&1
log "wrote $RESULTS"

# Make rendered videos available to the web docs.
cp -u "$VID"/handoff_B10_hard.mp4 "$VID"/handoff_B10_blend12.mp4 "$VID"/handoff_B12_smooth.mp4 "$VID"/handoff_B13_softcommit.mp4 "$ASSETS/" 2>/dev/null || true

cd "$ROOT" && git add -A ':!external/mujoco_warp' 2>/dev/null && git commit -q -m "results: handoff iter2 (B12/B13) eval + videos (auto)
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>/dev/null || true

log "spawning fresh Claude conversation to document..."
"$CLAUDE" -p "$(cat "$ROOT/scripts/handoff_iter2_document_prompt.md")" \
  --dangerously-skip-permissions --model opus --max-turns 200 \
  > "$ROOT/handoff_iter2_document_claude.log" 2>&1
log "document run finished (rc=$?). ============ DONE ============"

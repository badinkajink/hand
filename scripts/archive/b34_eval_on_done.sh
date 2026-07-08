#!/bin/bash
# Wait for the b34 force-floor sweep (3 trainers) to finish, then run the continuous-
# handoff eval on each at MATCHED PARITY (scale 0.2 / linear / contact-gate OFF —
# gotcha #13; defaults 0.5/ease_out_quad/gate-ON would be an artifact for this lineage).
set -u
ROOT=/home/humanoid/Programs/hand; cd "$ROOT"
VID="$ROOT/docs/rl/videos/reorient"; OUT="$ROOT/b34_EVAL_RESULTS.txt"
: > "$OUT"
echo "[b34-eval] waiting for trainers to finish..." | tee -a "$OUT"
while [ "$(pgrep -fc 'rl_train_cube.py.*forcefloor' 2>/dev/null || echo 0)" -gt 0 ]; do sleep 60; done
echo "[b34-eval] all trainers exited $(date)" | tee -a "$OUT"

eval_one () { # $1=tag  $2=thresh
  local tag="$1" th="$2"
  local dir ckpt
  dir=$(ls -dt "$ROOT"/results/rl/*forcefloor_"$tag" 2>/dev/null | head -1)
  if [ -z "$dir" ]; then echo "=== $tag (thresh $th): NO RUN DIR ===" | tee -a "$OUT"; return; fi
  ckpt=$(ls -t "$dir"/tensorboard/model_*.pt 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then echo "=== $tag (thresh $th): NO CKPT in $dir ===" | tee -a "$OUT"; return; fi
  local collapsed=""
  ls "$ROOT"/b34_"$tag".trainer.log.COLLAPSED >/dev/null 2>&1 && collapsed=" [WATCHDOG-COLLAPSED]"
  echo "=== $tag (thresh $th)$collapsed  ckpt=$(basename "$ckpt") ===" | tee -a "$OUT"
  local log="$ROOT/b34_${tag}.eval.log"
  WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu \
    python "$ROOT/scripts/rl_demo_handoff_continuous.py" \
    --policy-b "$ckpt" \
    --finger-residual-scale 0.2 --finger-close-easing linear --no-contact-gate \
    --output "$VID/b34_${tag}_cont.mp4" > "$log" 2>&1
  grep -iE "POST-handoff min|min object-center z|held-vertical|lin.*jerk|ang.*jerk|wander|horizontal path|lateral drift|grip.*force|contact.*force|VERDICT|sink" "$log" \
    | sed 's/^/  /' | tee -a "$OUT" || echo "  (eval produced no summary; see $log)" | tee -a "$OUT"
  echo "" | tee -a "$OUT"
}

# sequential (clean output; eval is light) — gentlest threshold first
eval_one t30 3.0
eval_one t25 2.5
eval_one t20 2.0
echo "[b34-eval] DONE — results in $OUT" | tee -a "$OUT"

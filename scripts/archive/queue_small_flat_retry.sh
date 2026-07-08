#!/bin/bash
# Retry small_flat (the run that NaN'd) with two parallel seeds + bump num_envs.
# At 2048 envs/run × 2 runs = ~8 GB VRAM, should sit at ~50% GPU util.
# We pick the higher-quality of the two for the final eval.

set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"
QUEUE_LOG="$ROOT/small_flat_retry.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

log "================ SMALL_FLAT RETRY START ================"

COMMON=(
  --morphology-run "$ROOT/results/phase1/run18_multi_object_adapt/foundational/screwdriver_small_flat/run_20260521_150831"
  --object-body-name screwdriver_small
  --num-envs 2048
  --total-timesteps 50000000
  --cube-spawn-x-jitter 0.002 --cube-spawn-y-jitter 0.002
  --cube-spawn-yaw-jitter 0.17
  --dr-anneal-iters 400
  --tracking-anneal-iters 400
  --tracking-final-scale 0.0
  --finger-residual-scale 0.5
  --finger-close-easing ease_out_quad
  --contact-gate-stability-rewards
  --enable-lift-terminations
  --object-xy-drift-weight=-30
  --object-orientation-drift-weight=-20
  --finger-drift-weight=-10
  --no-wandb
)

log "[1/2] launching small_flat seed=43"
MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
  "${COMMON[@]}" --seed 43 \
  --tag screwdriver_small_flat_short_proximal_stable_v1_s43 \
  > "$ROOT/sd_small_flat_s43.log" 2>&1 &
PID43=$!
log "  pid=$PID43"

sleep 30  # stagger kernel-cache loading

log "[2/2] launching small_flat seed=44"
MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
  "${COMMON[@]}" --seed 44 \
  --tag screwdriver_small_flat_short_proximal_stable_v1_s44 \
  > "$ROOT/sd_small_flat_s44.log" 2>&1 &
PID44=$!
log "  pid=$PID44"

log "waiting for both seeds to finish..."
wait $PID43; RC43=$?
wait $PID44; RC44=$?
log "seed 43 rc=$RC43  seed 44 rc=$RC44"

# Pick best post-curriculum ckpt of each surviving run
pick_best_ckpt_safe() {
  local run_dir="$1"
  uv run --extra rl python - <<EOF 2>/dev/null
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re
ea = EventAccumulator('$run_dir/tensorboard', size_guidance={'scalars': 0})
ea.Reload()
s = ea.Scalars('Metrics/lift_height/object_height')
post = [(e.step, e.value) for e in s if e.step >= 400]
if not post: post = [(e.step, e.value) for e in s]
best_step, best_val = max(post, key=lambda x: x[1])
ckpts = sorted(int(re.search(r'model_(\d+)\.pt', p.name).group(1))
               for p in Path('$run_dir/tensorboard').glob('model_*.pt'))
nearest = min(ckpts, key=lambda c: abs(c - best_step)) if ckpts else best_step
print(f"{nearest} {best_val:.4f}")
EOF
}

S43_DIR=$(ls -dt "$ROOT/results/rl/"*-screwdriver_small_flat_short_proximal_stable_v1_s43 2>/dev/null | head -1)
S44_DIR=$(ls -dt "$ROOT/results/rl/"*-screwdriver_small_flat_short_proximal_stable_v1_s44 2>/dev/null | head -1)
log "s43 dir: $S43_DIR"
log "s44 dir: $S44_DIR"

S43_INFO=$(pick_best_ckpt_safe "$S43_DIR" 2>/dev/null)
S44_INFO=$(pick_best_ckpt_safe "$S44_DIR" 2>/dev/null)
log "s43 best: $S43_INFO"
log "s44 best: $S44_INFO"

# Eval each surviving run
for entry in "$S43_DIR:$S43_INFO" "$S44_DIR:$S44_INFO"; do
  D="${entry%:*}"
  INFO="${entry##*:}"
  ITER="${INFO%% *}"
  if [ -z "$ITER" ] || [ ! -d "$D" ] || [ ! -f "$D/tensorboard/model_${ITER}.pt" ]; then
    log "skipping $D (missing ckpt or info)"
    continue
  fi
  log "evaluating $D iter=$ITER"
  MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_eval_object.py" \
    --checkpoint "$D/tensorboard/model_${ITER}.pt" \
    --foundational-run "$ROOT/results/phase1/run18_multi_object_adapt/foundational/screwdriver_small_flat/run_20260521_150831" \
    --object-body-name screwdriver_small \
    --x-jitter 0.002 --y-jitter 0.002 --yaw-jitter 0.17 \
    --finger-residual-scale 0.5 --num-envs 64 --pose-grid 3x3x1 \
    >> "$QUEUE_LOG" 2>&1 || log "eval failed for $D"
done

log "================ SMALL_FLAT RETRY DONE ================"

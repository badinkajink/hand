#!/bin/bash
# Overnight queue: wait for prism_stable_v1 to complete, then launch
# screwdriver_vertical_stable_v1 with the same recipe.
#
# Polls prism_stable_v1.log every 30s for "DONE" or a crash signature.
# Logs everything to overnight_queue.log.

set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$ROOT/overnight_queue.log"; }

log "queue starting; waiting on prism_stable_v1.log to finish"

# --- 1. Wait for prism_stable_v1 ---
until tail -50 "$ROOT/prism_stable_v1.log" 2>/dev/null | grep -qE "DONE|Traceback|Killed|OOM|CUDA out of memory"; do
  sleep 30
done

if tail -50 "$ROOT/prism_stable_v1.log" | grep -qE "Traceback|Killed|OOM|CUDA out of memory"; then
  log "prism crashed — aborting queue"
  exit 1
fi
log "prism_stable_v1 DONE"

# --- 2. Launch screwdriver_vertical_stable_v1 ---
log "launching screwdriver_vertical_stable_v1"
MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
  --morphology-run "$ROOT/results/phase1/run18_final/foundational/screwdriver_medium_vertical/run_20260521_162624" \
  --object-body-name screwdriver_medium \
  --tag screwdriver_vertical_stable_v1 \
  --num-envs 1024 \
  --total-timesteps 50000000 \
  --cube-spawn-x-jitter 0.002 --cube-spawn-y-jitter 0.002 \
  --cube-spawn-yaw-jitter 0.17 \
  --dr-anneal-iters 400 \
  --tracking-anneal-iters 400 \
  --tracking-final-scale 0.0 \
  --finger-residual-scale 0.5 \
  --finger-close-easing ease_out_quad \
  --contact-gate-stability-rewards \
  --enable-lift-terminations \
  --object-xy-drift-weight=-30 \
  --object-orientation-drift-weight=-20 \
  --finger-drift-weight=-10 \
  --no-wandb \
  > "$ROOT/screwdriver_vertical_stable_v1.log" 2>&1

# --- 3. Eval both fresh stable models ---
log "screwdriver_vertical_stable_v1 finished; running deterministic evals on best ckpts"

# Helper: pick the best checkpoint post-curriculum from a TB log
pick_best_ckpt() {
  local run_dir="$1"
  uv run --extra rl python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
ea = EventAccumulator('$run_dir/tensorboard', size_guidance={'scalars': 0})
ea.Reload()
s = ea.Scalars('Metrics/lift_height/object_height')
post = [(e.step, e.value) for e in s if e.step >= 400]
best = max(post, key=lambda x: x[1])
print(f'{best[0]}')
" 2>/dev/null
}

# prism eval
PRISM_DIR=$(ls -dt "$ROOT/results/rl/"*-prism_stable_v1 2>/dev/null | head -1)
PRISM_ITER=$(pick_best_ckpt "$PRISM_DIR")
if [ -n "$PRISM_ITER" ]; then
  log "prism best iter post-curriculum: $PRISM_ITER"
  MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_eval_object.py" \
    --checkpoint "$PRISM_DIR/tensorboard/model_${PRISM_ITER}.pt" \
    --foundational-run "$ROOT/results/phase1/run18_final/foundational/prism/run_20260521_161958" \
    --object-body-name prism \
    --x-jitter 0.006 --y-jitter 0.010 \
    --x-center 0.006 --y-center -0.003 \
    --yaw-jitter 0.52 \
    --finger-residual-scale 0.5 \
    --num-envs 64 \
    --pose-grid 3x3x1 \
    >> "$ROOT/overnight_queue.log" 2>&1
fi

# screwdriver eval
SCR_DIR=$(ls -dt "$ROOT/results/rl/"*-screwdriver_vertical_stable_v1 2>/dev/null | head -1)
SCR_ITER=$(pick_best_ckpt "$SCR_DIR")
if [ -n "$SCR_ITER" ]; then
  log "screwdriver best iter post-curriculum: $SCR_ITER"
  MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_eval_object.py" \
    --checkpoint "$SCR_DIR/tensorboard/model_${SCR_ITER}.pt" \
    --foundational-run "$ROOT/results/phase1/run18_final/foundational/screwdriver_medium_vertical/run_20260521_162624" \
    --object-body-name screwdriver_medium \
    --x-jitter 0.002 --y-jitter 0.002 \
    --yaw-jitter 0.17 \
    --finger-residual-scale 0.5 \
    --num-envs 64 \
    --pose-grid 3x3x1 \
    >> "$ROOT/overnight_queue.log" 2>&1
fi

# --- 4. Render fresh plots overlaying all three stable runs ---
log "rendering plots"
uv run --extra rl python "$ROOT/scripts/rl_plot_training.py" \
  --run "$ROOT/results/rl/20260528-2109-cube_stable_v1" \
  --run "$PRISM_DIR" \
  --run "$SCR_DIR" \
  --out "$ROOT/results/rl/plots_stable_v1_all" >> "$ROOT/overnight_queue.log" 2>&1

log "OVERNIGHT QUEUE DONE"

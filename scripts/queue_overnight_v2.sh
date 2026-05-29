#!/bin/bash
# Overnight v2 queue (2026-05-29, parallel):
#   1. Train medium_flat + small_flat IN PARALLEL (saves ~50% wall-time)
#   2. After both finish: pick best ckpts + eval each
#   3. Plot overlay across all 5 stable_v1 runs
#   4. Multimorphology Path-A
#
# Includes the env_cfg.py fix: object spawn quat now correctly read from
# the keyframe (was identity before, causing flat cylinders to stand up).
#
# Each training at 1024 envs ≈ 3.3 GB VRAM. Two trainings ≈ 6.6 GB (well
# under the 16 GB on the 4070 Ti). GPU contention will roughly double per-iter
# time, but total wall-time is still halved vs sequential.

set -u
ROOT=/home/humanoid/Programs/hand
cd "$ROOT"

QUEUE_LOG="$ROOT/overnight_v2.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

CUBE_RUN="$ROOT/results/rl/20260528-2109-cube_stable_v1"
PRISM_RUN="$ROOT/results/rl/20260528-2259-prism_stable_v1"
SCR_VERT_RUN="$ROOT/results/rl/20260529-0009-screwdriver_vertical_stable_v1"
CUBE_BASE_CKPT="$CUBE_RUN/tensorboard/model_1400.pt"
for f in "$CUBE_BASE_CKPT"; do
  if [ ! -f "$f" ]; then log "FATAL: missing $f"; exit 1; fi
done

log "================ QUEUE V2 (PARALLEL) START ================"

# -------------- common trainer args (matches last night's stable_v1) --------------
# num_envs bumped 1024 -> 2048: at 1024 we saw 4 GB VRAM total for two parallel
# trainings (22% GPU util). 2048 each ~= 8 GB total, well under the 16 GB limit,
# with more compute headroom for the parallel pair.
COMMON_ARGS=(
  --num-envs 2048
  --total-timesteps 50000000
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

# ----------------------- Stage 1+2: launch both in parallel -----------------------
log "[stage 1] launching medium_flat training (background)"
MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
  --morphology-run "$ROOT/results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259" \
  --object-body-name screwdriver_medium \
  --tag screwdriver_medium_flat_short_proximal_stable_v1 \
  --cube-spawn-x-jitter 0.003 --cube-spawn-y-jitter 0.003 --cube-spawn-yaw-jitter 0.26 \
  "${COMMON_ARGS[@]}" \
  > "$ROOT/sd_medium_flat_stable_v1.log" 2>&1 &
PID_MED=$!
log "[stage 1] medium_flat pid=$PID_MED"

sleep 30   # stagger to avoid two simultaneous mjwarp kernel-cache loads

log "[stage 2] launching small_flat training (background)"
MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_train_cube.py" \
  --morphology-run "$ROOT/results/phase1/run18_multi_object_adapt/foundational/screwdriver_small_flat/run_20260521_150831" \
  --object-body-name screwdriver_small \
  --tag screwdriver_small_flat_short_proximal_stable_v1 \
  --cube-spawn-x-jitter 0.002 --cube-spawn-y-jitter 0.002 --cube-spawn-yaw-jitter 0.17 \
  "${COMMON_ARGS[@]}" \
  > "$ROOT/sd_small_flat_stable_v1.log" 2>&1 &
PID_SMALL=$!
log "[stage 2] small_flat pid=$PID_SMALL"

log "[stage 1+2] waiting for both trainings to finish..."
wait $PID_MED
RC_MED=$?
wait $PID_SMALL
RC_SMALL=$?
log "[stage 1+2] medium_flat rc=$RC_MED  small_flat rc=$RC_SMALL"
if [ $RC_MED -ne 0 ] && [ $RC_SMALL -ne 0 ]; then
  log "FATAL: both trainings failed"; exit 1
fi

# ----------------------- Stage 3+4: pick + eval -----------------------
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
best_step, _ = max(post, key=lambda x: x[1])
ckpts = sorted(
    int(re.search(r'model_(\d+)\.pt', p.name).group(1))
    for p in Path('$run_dir/tensorboard').glob('model_*.pt')
)
print(min(ckpts, key=lambda c: abs(c - best_step)) if ckpts else best_step)
EOF
}

MED_DIR=$(ls -dt "$ROOT/results/rl/"*-screwdriver_medium_flat_short_proximal_stable_v1 | head -1)
SMALL_DIR=$(ls -dt "$ROOT/results/rl/"*-screwdriver_small_flat_short_proximal_stable_v1 | head -1)
MED_ITER=$(pick_best_ckpt_safe "$MED_DIR")
SMALL_ITER=$(pick_best_ckpt_safe "$SMALL_DIR")
log "[stage 3] medium_flat best ckpt iter=$MED_ITER  (dir=$MED_DIR)"
log "[stage 3] small_flat  best ckpt iter=$SMALL_ITER (dir=$SMALL_DIR)"

if [ -n "$MED_ITER" ] && [ -f "$MED_DIR/tensorboard/model_${MED_ITER}.pt" ]; then
  log "[stage 4] eval medium_flat"
  MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_eval_object.py" \
    --checkpoint "$MED_DIR/tensorboard/model_${MED_ITER}.pt" \
    --foundational-run "$ROOT/results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259" \
    --object-body-name screwdriver_medium \
    --x-jitter 0.003 --y-jitter 0.003 --yaw-jitter 0.26 \
    --finger-residual-scale 0.5 --num-envs 64 --pose-grid 3x3x1 \
    >> "$QUEUE_LOG" 2>&1 || log "[stage 4] medium_flat eval FAILED (continuing)"
fi

if [ -n "$SMALL_ITER" ] && [ -f "$SMALL_DIR/tensorboard/model_${SMALL_ITER}.pt" ]; then
  log "[stage 4] eval small_flat"
  MUJOCO_GL=egl uv run --extra rl --extra gpu python "$ROOT/scripts/rl_eval_object.py" \
    --checkpoint "$SMALL_DIR/tensorboard/model_${SMALL_ITER}.pt" \
    --foundational-run "$ROOT/results/phase1/run18_multi_object_adapt/foundational/screwdriver_small_flat/run_20260521_150831" \
    --object-body-name screwdriver_small \
    --x-jitter 0.002 --y-jitter 0.002 --yaw-jitter 0.17 \
    --finger-residual-scale 0.5 --num-envs 64 --pose-grid 3x3x1 \
    >> "$QUEUE_LOG" 2>&1 || log "[stage 4] small_flat eval FAILED (continuing)"
fi

# ----------------------- Stage 5: plot overlay -----------------------
log "[stage 5] training-curve overlay across all 5 stable_v1 runs"
uv run --extra rl python "$ROOT/scripts/rl_plot_training.py" \
  --run "$CUBE_RUN" \
  --run "$PRISM_RUN" \
  --run "$SCR_VERT_RUN" \
  --run "$MED_DIR" \
  --run "$SMALL_DIR" \
  --out "$ROOT/results/rl/plots_stable_v1_all5" \
  >> "$QUEUE_LOG" 2>&1 || log "[stage 5] plot FAILED (continuing)"

# ----------------------- Stage 6: multimorphology -----------------------
log "[stage 6] multimorphology Path-A pipeline"

MULTIMORPH_DIR="$ROOT/results/phase1/multimorph_cube_v1"
mkdir -p "$MULTIMORPH_DIR"

log "[stage 6.1] picking 4 cube candidates by 9-dim distance from id=0"
uv run --extra rl python "$ROOT/scripts/multimorph_pick_candidates.py" \
  --candidates-csv "$ROOT/results/phase1/run18_final/cube/all_candidates.csv" \
  --generated-mjcf-dir "$ROOT/results/phase1/run18_final/generated_mjcf" \
  --object-prefix cube \
  > "$MULTIMORPH_DIR/picks.json" 2>>"$QUEUE_LOG"
log "[stage 6.1] picks:"
cat "$MULTIMORPH_DIR/picks.json" | tee -a "$QUEUE_LOG" | grep -E '"candidate_id"|"rank_label"|"distance"'

log "[stage 6.2] CEM + eval + conditional finetune for each candidate"
uv run --extra rl --extra gpu python "$ROOT/scripts/multimorph_run_pipeline.py" \
  --picks-json "$MULTIMORPH_DIR/picks.json" \
  --base-policy-ckpt "$CUBE_BASE_CKPT" \
  --existing-foundational-id0 "$ROOT/results/phase1/run18_final/foundational/cube/run_20260521_161817" \
  --out-dir "$MULTIMORPH_DIR" \
  --log-path "$ROOT/multimorph.log" \
  --finetune-threshold 0.80 \
  --finetune-iters 250 \
  --eval-x-jitter 0.02 --eval-y-jitter 0.005 --eval-yaw-jitter 0.52 \
  >> "$QUEUE_LOG" 2>&1 || log "[stage 6] multimorph pipeline FAILED (continuing)"

log "================ QUEUE V2 DONE ================"

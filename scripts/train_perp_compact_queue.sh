#!/usr/bin/env bash
# r5: train perp_single on the TOP-N compact morphologies from the mechanism sweep.
#
# Iteration count comes from the recipe/PPO timestep budget (r4 ran 339); there is no
# --max-iterations flag — max_iterations is derived via ppo_cfg.iters_for_timesteps().
#
# Sequential on purpose — one 16 GB GPU. Resumable: each design writes a .DONE sentinel, so
# re-running the script skips finished designs and picks up where it stopped. Every trainer
# process gets its OWN Warp kernel cache (a shared cache races and NaNs), and we wait for GPU
# memory to fall back before launching the next one.
#
# Designs come from docs/experiments/perp_compact_sweep.json (ranked, gate-valid, physics-HELD).
# Per design: bake the rigid scene -> IK-retarget `open_ik` onto the reference grasp -> train.
# The retarget is NOT optional: a moved mount holding the shipped joint angles puts its tip
# somewhere else entirely and the grip is simply wrong (CLAUDE.md gotcha #5).
#
#   bash scripts/train_perp_compact_queue.sh          # run/resume the queue
#   TOP=3 bash scripts/train_perp_compact_queue.sh    # only the first 3
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
TOP="${TOP:-5}"
# LIFT HEIGHT IS NOT IN THE RECIPE — it is a run knob, and the trainer's default is 0.05.
# The first r5 queue omitted it and every design therefore trained at 0.05 while the r2/r4
# reference trained at 0.14, which is not a small difference on a 100 mm shaft: hanging
# vertical puts the shaft's lower end 0.085 m below the pinch, so at a 0.05 lift it is in the
# floor, the floor-proximity gate false-fires, and the reorient cannot complete. The one design
# that ran to 339 iterations under that bug (t0.00_x0.25) evaluates at align_rate 0.0% / hold
# 0.0% with every env ending flat on the floor -- a design ranking built on those numbers would
# have measured the bug, not the morphology. See CLAUDE.md gotcha: diff a new launcher's flags
# against the last WORKING run's config.yaml (results/rl/<r4 run>/config.yaml has 0.14).
LIFT_DELTA="${LIFT_DELTA:-0.14}"
# ...and lift was only ONE of four. Relaunching this queue with just the lift fix still produced
# num_envs 1024 (vs 3072), total_timesteps 200M (vs 25M -> a 6 h run instead of 40 min) and
# open_finger_from_keyframe FALSE (vs True -> the wrong open pose, gotcha #5). All three are
# silent. Every value below is read off REF_RUN's config.yaml, and the pre-flight re-checks that
# claim against the config the trainer actually writes, because a comment has now failed to
# prevent this four times.
REF_RUN="${REF_RUN:-$ROOT/results/rl/20260731-1300-perp_single_r4}"
NUM_ENVS="${NUM_ENVS:-3072}"
TOTAL_STEPS="${TOTAL_STEPS:-25000000}"
SWEEP_JSON="$ROOT/docs/experiments/perp_compact_sweep.json"
BASE_SCENE="$ROOT/assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml"
MORPH_RUN="$ROOT/results/phase1/perp/perp_v1"
QUEUE_DIR="$ROOT/results/rl/perp_compact_queue"
LOG_DIR="$ROOT/logs"
mkdir -p "$QUEUE_DIR" "$LOG_DIR"

STAMP="$(date +%Y%m%d-%H%M)"
QLOG="$LOG_DIR/perp_compact_queue_${STAMP}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

[[ -f "$SWEEP_JSON" ]] || { say "FATAL: $SWEEP_JSON missing — run scripts/sweep_perp_compact.py first"; exit 1; }

mapfile -t LABELS < <(python3 -c "
import json,sys
top=json.load(open('$SWEEP_JSON'))['top'][:$TOP]
print('\n'.join(r['label'] for r in top))")

say "queue: ${#LABELS[@]} designs (TOP=$TOP) -> $QLOG"

wait_for_gpu() {
  # After a Warp run exits, memory takes a moment to actually free. Launching into the tail of
  # the previous run's allocation is how these jobs OOM.
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [[ "$used" -lt 2500 ]] && return 0
    sleep 10
  done
  say "WARN: GPU still at ${used} MiB after 10 min; launching anyway"
}

for LABEL in "${LABELS[@]}"; do
  DEST="$QUEUE_DIR/$LABEL"
  if [[ -f "$DEST/.DONE" ]]; then say "SKIP $LABEL (.DONE)"; continue; fi
  mkdir -p "$DEST"

  say "=== $LABEL: bake + retarget ==="
  SCENE="$DEST/frozen_scene.xml"
  MUJOCO_GL=egl uv run python - "$LABEL" "$SCENE" <<'PY' >>"$QLOG" 2>&1
import json, sys
sys.path.insert(0, "src")
from pathlib import Path
from morphohand.tools.morphology_xml import MorphologyValues, create_rigid_morphology_xml
label, out = sys.argv[1], Path(sys.argv[2])
rec = next(r for r in json.load(open("docs/experiments/perp_compact_sweep.json"))["all"]
           if r["label"] == label)
create_rigid_morphology_xml(
    base_xml_path=Path("assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml"),
    morphology=MorphologyValues(**rec["morph"]), output_xml_path=out)
print("baked", out)
PY
  [[ -f "$SCENE" ]] || { say "FAIL $LABEL: bake produced no scene"; continue; }

  MUJOCO_GL=egl uv run python scripts/retarget_keyframe_ik.py \
      --base-scene "$BASE_SCENE" --keyframe open \
      --target-scene "$SCENE" --out-keyframe open_ik \
      --write-keyframe --validate >>"$QLOG" 2>&1 \
    || { say "FAIL $LABEL: keyframe retarget failed"; continue; }

  RUN_NAME="perp_single_r5_${LABEL}"
  # NaN retry ladder. The recipe pins init_noise_std 0.05 because the perp scene NaNs above
  # ~0.1, but that was tuned on ONE morphology. A design that NaNs at iteration 0 is not
  # necessarily a bad design — it may just need quieter exploration — and silently skipping it
  # biases the whole ranking toward designs that happen to survive the pinned value.
  #
  # This ladder is NOT a substitute for finding a real cause. The first time every design in
  # the queue NaN'd, the cause was a scene change (a non-colliding palm plate), not the noise,
  # and no amount of retrying would have helped. If the WHOLE queue fails, stop and diff the
  # scene against a run that trained — do not just lower the noise.
  RC=1
  for NOISE in "" 0.02 0.01; do
    if [[ -n "$NOISE" ]]; then
      say "=== $LABEL: RETRY at init_noise_std=$NOISE ==="
      NOISE_ARG=(--init-noise-std "$NOISE")
    else
      say "=== $LABEL: train (perp_single, recipe noise) ==="
      NOISE_ARG=()
    fi
    export WARP_CACHE_PATH="$(mktemp -d)"
    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/rl_train_cube.py \
        --recipe perp_single \
        --morphology-run "$MORPH_RUN" \
        --frozen-scene-xml "$SCENE" \
        --tag "$RUN_NAME" \
        --lift-target-z-above-init "$LIFT_DELTA" --lift-delta-z "$LIFT_DELTA" \
        --open-finger-from-keyframe \
        --num-envs "$NUM_ENVS" \
        --total-timesteps "$TOTAL_STEPS" \
        "${NOISE_ARG[@]}" \
        >>"$DEST/train.log" 2>&1 &
    TRAIN_PID=$!

    # Pre-flight against r4 while the run is still free to kill. A design's SCENE differs by
    # construction (that is the experiment) and assert_config_parity ignores scene/dir keys, so
    # only the knobs are compared.
    sleep 20
    NEW_RUN="$(ls -dt "$ROOT/results/rl/"*"$RUN_NAME" 2>/dev/null | head -1)"
    if [[ -n "$NEW_RUN" ]]; then
      if ! uv run python scripts/assert_config_parity.py \
             --run "$NEW_RUN" --reference "$REF_RUN" \
             --allow ppo.init_noise_std 2>&1 | tee -a "$QLOG" | grep -q "^\[parity\] OK"; then
        say "ABORT $LABEL: config parity check failed — killing $TRAIN_PID"
        kill "$TRAIN_PID" 2>/dev/null; wait "$TRAIN_PID" 2>/dev/null
        rm -rf "$WARP_CACHE_PATH"; RC=99; wait_for_gpu; break
      fi
    fi

    wait "$TRAIN_PID"
    RC=$?
    rm -rf "$WARP_CACHE_PATH"
    [[ $RC -eq 0 ]] && break
    grep -q "contains NaN values" "$DEST/train.log" || { say "FAIL $LABEL (rc=$RC, not a NaN) — no retry"; break; }
    wait_for_gpu
  done

  if [[ $RC -eq 0 ]]; then
    date -Is > "$DEST/.DONE"; say "DONE $LABEL"
  else
    say "FAIL $LABEL (rc=$RC) — see $DEST/train.log; tail:"
    tail -5 "$DEST/train.log" | tee -a "$QLOG"
  fi
  wait_for_gpu
done

say "queue finished: $(ls -d "$QUEUE_DIR"/*/.DONE 2>/dev/null | wc -l)/${#LABELS[@]} designs done"

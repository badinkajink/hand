#!/usr/bin/env bash
# r7: can the opposed-pair hand be made to brace with THREE fingers?
#
# The problem. r4 reorients 100% of rollouts to cos 0.995 and loses every one of them, because
# the shaft turns BY sliding through a two-point pinch -- rotation and retention are the same
# degree of freedom (r6 proved that: penalise the slide and the rotation stops). The thumb reads
# 0 N throughout and has never contributed anything on this topology. The inline arrangement, by
# contrast, keeps the object on three loaded fingers. This asks whether the opposed pair can be
# given the same property.
#
# Why it has never been tested, despite two prior thumb studies:
#
#   * The 2026-07-30 thumb sweep swept `thumb_len` and `index/middle x` on SEPARATE axes and
#     concluded "reach is not support -- the objectives fight over one parameter". That
#     conclusion came from the im_x rows, which were run at thumb_len = 0, where reach could
#     only be bought by moving the pinch (and moving the pinch is what kills the swing).
#   * Its own thumb_len rows already showed the answer sitting there: at thumb_len +0.035 the
#     hanging shaft is 150.6 mm from the thumb mount against a reach shell of [121.7, 155.0] --
#     INSIDE -- with the swing untouched at cos +0.986. The two axes were never crossed.
#   * Re-run crossed (2026-08-18, docs/experiments/20260818-perp_thumb_len_x_pinch.md): the
#     thumb does reach, and the scripted press then DROPS the object. That looks like a refutation
#     and is not one. Measured on the same probe, the open-loop grip has bled to 0.4 N per pad by
#     press time and the object's escape force is 0.4-1.2 N in EVERY direction -- it is barely
#     held at all, so any press ejects it regardless of geometry. The scripted probe cannot
#     represent the regime a trained policy actually holds (10-27 N). That is what this run is for.
#
# The design. thumb_len +0.035, mounts unchanged, so the pinch is untouched at 34.9 mm and the
# swing is preserved. Gate: physically real, retargeted. The thumb CANNOT be a grasp finger here
# -- lengthening it moves its MINIMUM reach out to 121.7 mm while the grasp sits nearer, so it is
# parked clear in every keyframe (yaw 0.6, not the 1.1 the gate also passes: the policy's thumb
# residual is +-0.5 rad, so a thumb at the 1.1 joint limit could never be driven to point forward)
# and deployed only by the policy's own residual, only once the shaft is up.
#
#   bash scripts/train_perp_thumbbrace.sh
#   WEIGHTS="5" bash scripts/train_perp_thumbbrace.sh
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

SCENE="${SCENE:-$ROOT/results/rl/perp_longthumb/frozen_scene.xml}"
MORPH_RUN="$ROOT/results/phase1/perp/perp_v1"
WEIGHTS="${WEIGHTS:-5 15}"
BRACE_THRESH="${BRACE_THRESH:-0.7}"     # same gate the grip catch uses -- protects the swing
BRACE_MAXN="${BRACE_MAXN:-4.0}"         # the thumb force the 2026-07-30 sweep did achieve
# Read off REF_RUN's config.yaml; the pre-flight re-checks it against what the trainer wrote.
REF_RUN="${REF_RUN:-$ROOT/results/rl/20260731-1300-perp_single_r4}"
LIFT_DELTA="${LIFT_DELTA:-0.14}"
NUM_ENVS="${NUM_ENVS:-3072}"
TOTAL_STEPS="${TOTAL_STEPS:-25000000}"

LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/perp_thumbbrace_$(date +%Y%m%d-%H%M).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

wait_for_gpu() {
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [[ "$used" -lt 2500 ]] && return 0
    sleep 10
  done
  say "WARN: GPU still at ${used} MiB after 10 min; launching anyway"
}

[[ -f "$SCENE" ]] || { say "FATAL: scene missing: $SCENE"; exit 1; }
say "r7 thumb-brace: weights [$WEIGHTS] gate cos>=$BRACE_THRESH sat ${BRACE_MAXN}N -> $QLOG"

for W in $WEIGHTS; do
  TAG="perp_single_r7_brace${W}"
  DEST="$ROOT/results/rl/thumbbrace_${W}"; mkdir -p "$DEST"
  [[ -f "$DEST/.DONE" ]] && { say "SKIP $W (.DONE)"; continue; }

  RC=1
  for NOISE in "" 0.02 0.01; do
    if [[ -n "$NOISE" ]]; then
      say "=== $TAG: RETRY at init_noise_std=$NOISE ==="; NOISE_ARG=(--init-noise-std "$NOISE")
    else
      say "=== train $TAG (recipe noise) ==="; NOISE_ARG=()
    fi
    export WARP_CACHE_PATH="$(mktemp -d)"
    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/rl_train_cube.py \
        --recipe perp_single \
        --morphology-run "$MORPH_RUN" \
        --frozen-scene-xml "$SCENE" \
        --tag "$TAG" \
        --lift-target-z-above-init "$LIFT_DELTA" --lift-delta-z "$LIFT_DELTA" \
        --open-finger-from-keyframe \
        --num-envs "$NUM_ENVS" --total-timesteps "$TOTAL_STEPS" \
        --thumb-brace-weight "$W" \
        --thumb-brace-align-thresh "$BRACE_THRESH" \
        --thumb-brace-max-force "$BRACE_MAXN" \
        "${NOISE_ARG[@]}" \
        >>"$DEST/train.log" 2>&1 &
    TRAIN_PID=$!

    sleep 20
    NEW_RUN="$(ls -dt "$ROOT/results/rl/"*"$TAG" 2>/dev/null | head -1)"
    if [[ -n "$NEW_RUN" ]]; then
      if ! uv run python scripts/assert_config_parity.py \
             --run "$NEW_RUN" --reference "$REF_RUN" \
             --allow env.thumb_brace_weight --allow env.thumb_brace_align_thresh \
             --allow env.thumb_brace_max_force --allow ppo.init_noise_std \
             2>&1 | tee -a "$QLOG" | grep -q "^\[parity\] OK"; then
        say "ABORT $W: config parity failed — killing $TRAIN_PID"
        kill "$TRAIN_PID" 2>/dev/null; wait "$TRAIN_PID" 2>/dev/null
        rm -rf "$WARP_CACHE_PATH"; RC=99; wait_for_gpu; break
      fi
    fi

    wait "$TRAIN_PID"; RC=$?
    rm -rf "$WARP_CACHE_PATH"
    [[ $RC -eq 0 ]] && break
    grep -q "contains NaN values" "$DEST/train.log" || { say "FAIL $W (rc=$RC, not a NaN)"; break; }
    wait_for_gpu
  done

  if [[ $RC -eq 0 ]]; then date -Is > "$DEST/.DONE"; say "DONE $W"
  else say "FAIL $W (rc=$RC)"; tail -5 "$DEST/train.log" | tee -a "$QLOG"; fi
  wait_for_gpu
done
say "r7 sweep finished"

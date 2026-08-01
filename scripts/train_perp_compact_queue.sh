#!/usr/bin/env bash
# r5: train perp_single on the TOP-N compact morphologies from the mechanism sweep.
#
# ⚠ THE BUDGET MUST BE PASSED AS FLAGS. There is no --max-iterations; max_iterations is derived
# as total_timesteps // (num_envs * num_steps_per_env). The recipe does NOT carry a timestep
# budget, so a launcher that omits these inherits the PPOConfig DEFAULTS — 1024 envs and 200M
# timesteps = 8138 iterations, i.e. 5.7 h/design instead of 42 min. The r5 queue did exactly
# that and burned 13.4 h for 0/5 designs. Every working perp run (r2/r3/r4) used 3072 envs and
# 25M timesteps = 339 iterations, passed on the command line, same as train_A_on_morph.sh.
#
# 339 is deliberately a SCREENING budget, not convergence — r4 was still climbing steeply when
# it ended (reward 1207->1375 over its last 38 iters). It is enough to see a clear positive or
# negative signal on pick-up + reorientation, which is what ranking designs needs; the
# converged-cost question is deferred until a design is worth converging.
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
# The working perp settings, as FLAGS (see header). Overridable, but do not drop them.
NUM_ENVS="${NUM_ENVS:-3072}"
TOTAL_TS="${TOTAL_TS:-25000000}"        # -> 339 iters at 3072x24, ~42 min/design
# Health gate. It answers exactly one question — "did the object ever leave the floor?" — and
# must NOT be used to judge lift QUALITY. The object spawns at z=0.0125, so 0.02 means "still
# on the floor" with 1.6x margin, and nothing that is genuinely lifting comes near it.
#
# ⚠ RECALIBRATED 2026-08-01 after a FALSE KILL. The first value (0.04) was fitted to r5/r6
# curves in which every dead run sat at exactly 0.0123 — the signature of the
# open_finger_from_keyframe bug, i.e. fitted to an artifact. With that bug fixed the failure
# distribution changed, and 0.04 promptly killed t0.00_x0.25_y0.00 at iteration 60 while it
# was RECOVERING: obj_z 0.0268 (iter 50) -> 0.0349 -> 0.0390, with mean reward climbing
# monotonically 42.6 -> 76.7 the whole time. A struggling-but-improving design is a result to
# be measured over the full 339, not a run to abort.
#
# The lesson is about what a screening gate is FOR. "Never lifted" is a binary the gate can
# read in 7 minutes; "lifts poorly" is the measurement itself and costs the full 42 minutes,
# which is the price of an honest number. Do not raise this to reclaim GPU time — an early
# kill that correlates with design quality silently biases the whole ranking.
WATCH_Z="${WATCH_Z:-0.02}"
WATCH_FROM="${WATCH_FROM:-60}"
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
  # ⚠ THE NaN IS INTRINSIC TO THE PERP SCENE, not to any morphology or palm setting. Measured
  # 2026-08-01 on CPU MuJoCo from r4's own XML: under randn*0.05 ctrl noise, seed 1 drives
  # |qvel| to 5e6 by step 22, and the DOF that blows up is the SCREWDRIVER's free joint. The
  # same divergence occurs with and without the palm<->finger excludes, so neither the excludes
  # nor the earlier non-colliding plate introduced it — both attributions were wrong.
  #
  # What it means for scheduling: it is a per-iteration hazard, ~1 NaN per 1250 iterations
  # across the 15 r5 attempts. P(finish 339) ~ 0.76, P(finish 8138) ~ 0.002 — which is why the
  # over-long budget turned a survivable rate into 0/5. Retrying is legitimate here; it is
  # re-rolling the dice, not papering over a scene bug. But if the WHOLE queue fails, still
  # stop and diff against a run that trained.
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
    rm -f "$DEST/.COLLAPSED"          # stale sentinel from a previous attempt would misreport
    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/rl_train_cube.py \
        --recipe perp_single \
        --morphology-run "$MORPH_RUN" \
        --frozen-scene-xml "$SCENE" \
        --tag "$RUN_NAME" \
        --num-envs "$NUM_ENVS" --total-timesteps "$TOTAL_TS" \
        --watchdog-collapse-z "$WATCH_Z" --watchdog-from-iter "$WATCH_FROM" \
        --watchdog-sentinel "$DEST/.COLLAPSED" \
        "${NOISE_ARG[@]}" \
        >>"$DEST/train.log" 2>&1
    RC=$?
    rm -rf "$WARP_CACHE_PATH"
    [[ $RC -eq 0 ]] && break
    # A watchdog abort is a VERDICT on the design (it never lifted), not a noise problem.
    # Retrying it quieter would bias the ranking toward designs that tolerate low exploration
    # — the same bias the NaN ladder exists to avoid. Only a NaN earns a retry.
    if [[ -f "$DEST/.COLLAPSED" ]]; then
      say "COLLAPSED $LABEL — never lifted: $(cat "$DEST/.COLLAPSED"); no retry, that IS the result"
      break
    fi
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

#!/usr/bin/env bash
# Evaluate every finished sp25 opposed-hand run against the scripted chuck, on the same terms.
#
# The comparison that matters is not peak alignment -- the two-finger pinch reaches vertical too,
# on its way to dropping the shaft. It is `three_finger` and the per-finger forces over the
# aligned-and-held steps, which is what the scripted probe's [hold] line reports, so the numbers
# line up directly against:
#
#   scripted chuck (frozen sp25):  100.0% held vertical, 100.0% three fingers, 20.4/11.6/15.5 N
#
# Each run is evaluated at ITS OWN --finger-residual-scale. r9 deliberately breaks train/deploy
# parity (gotcha #13) to give the policy enough authority to reach the hold pose at all, and
# evaluating it at the trainer default would be a 3x residual mismatch -- the exact artifact
# gotcha #13 exists to warn about, which has already impersonated a seam failure once.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

MORPH_RUN="$ROOT/results/phase1/perp_thumb_engage/sp25_manual"
OUT="$ROOT/docs/experiments/$(date +%Y%m%d)-perp_sp25_training"
mkdir -p "$OUT"
N="${N:-64}"
STEPS="${STEPS:-700}"

for RUN in "$@"; do
  [[ -d "$RUN" ]] || { echo "skip (missing): $RUN"; continue; }
  NAME="$(basename "$RUN")"
  CKPT="$(ls -t "$RUN"/tensorboard/model_*.pt 2>/dev/null | sort -V | tail -1)"
  [[ -n "$CKPT" ]] || { echo "skip (no checkpoint): $NAME"; continue; }
  FRS="$(python - "$RUN/config.yaml" <<'PY'
import sys, yaml
v = yaml.safe_load(open(sys.argv[1]))["env"].get("finger_residual_scale", 0.5)
print(",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v)
PY
)"
  echo "=== $NAME   $(basename "$CKPT")   residual_scale=$FRS"
  MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/policy_eval_suite.py \
      --policy "$CKPT" --morphology-run "$MORPH_RUN" \
      --n "$N" --steps "$STEPS" --lift-delta 0.14 \
      --finger-residual-scale "$FRS" \
      --open-finger-from-keyframe --closed-ctrl-from-keyframe closed_manual \
      --label "$NAME" --json-out "$OUT/$NAME.json" --plot "$OUT/$NAME.png" 2>&1 \
    | grep -E "^[│┌└]|Traceback|Error"
done
echo "[eval] artifacts -> $OUT"

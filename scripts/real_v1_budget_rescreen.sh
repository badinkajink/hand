#!/usr/bin/env bash
# Re-screen the Sobol-4096 population across the RESIDUAL CLIP, not at one value of it.
#
# Every morphology screen this program has run -- the 108-hand search, the 128-hand pilot, the
# 4096-hand retention screen -- planned its turn with `budget=0.5` rad, hardcoded.  That number
# is Policy B's residual ACTION budget; it arrived here by inheritance and has no business
# constraining an open-loop trajectory.  The 2026-08-30 band scan showed each design holds only
# inside a contiguous range of clips and drops on both sides, with the best alignment at the
# lower edge -- so a screen at one clip scores every hand at a point that may be nowhere near
# its own band, and rejects hands for being clipped rather than for being bad.
#
# Stage 1 is that screen redone as a 2-D sweep: 5,696 fitted grasp cells x 5 clips, one trial
# each.  It also carries the new servo-command gate (`servo_short_deg`), so a hand that cannot
# be TOLD to do its own trajectory is rejected here instead of four stages later at export.
#
# Resumable: every stage checkpoints into its --out and skips what is already there.
set -euo pipefail
cd "$(dirname "$0")/.."

POP=docs/experiments/20260830-real_v1-sobol4096
OUT=docs/experiments/20260830-real_v1-budget-rescreen
GEN=assets/mjcf/experimental/20260830-real_v1-sobol4096
BUDGETS=${BUDGETS:-0.5,0.7,0.9,1.1,1.3}
WORKERS=${WORKERS:-18}
mkdir -p "$OUT"

# the retention gate, verbatim from render_real_v1_retention.py -- a 60 mm proof lift the object
# has to follow, then five free seconds with at most 10 mm of slip, under the SCS0009 torque caps
RETENTION=(--hold-steps 2500 --selfcollision
           --load-target-units 250 --load-gain 0.0024 --capture-steps 400
           --proof-lift-mm 60 --proof-lift-steps 700 --proof-max-slip-mm 10
           --turn-torque-limit-nm 0.18044236 --hold-torque-limit-nm 0.06864655)
SCENE=(--object medium --bench-height 0.100 --post-y -35
       --flat-pads --pad-len-mm 14.8 --pad-width-mm 21.1)
CELL=(--cell-k 0.15,0.15,1 --cell-angles -80 --squeeze-mm 10
      --turn-steps 550 --hold-squeeze-mm 0)

for c in s32_t10 s32_t20 s40_t10 s40_t20; do
  echo "=== stage 1: $c at budgets $BUDGETS ==="
  python3 scripts/real_v1_deploy_envelope.py --mode cell \
    --designs-file "$POP/retention/retention_designs_$c.txt" \
    --design-table "$POP/retention/retention_table_$c.json" \
    --design-manifest "$POP/hardware_manifest.json" \
    --generated-dir "$GEN" \
    "${SCENE[@]}" "${CELL[@]}" "${RETENTION[@]}" \
    --cell-nom 1 --cell-ens 0 --budget "$BUDGETS" --workers "$WORKERS" \
    --out "$OUT/band_$c.json"
done
echo "=== stage 1 complete ==="

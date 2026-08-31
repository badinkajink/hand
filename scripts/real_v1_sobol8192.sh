#!/usr/bin/env bash
# Extend the Sobol population from 4,096 hands to 8,192, and screen the new half.
#
# `sobol_designs` is PREFIX-STABLE: asking the same seed for 8,192 points returns the first
# 4,096 unchanged, with the same tags and the same vectors.  So this is a strict superset --
# the existing scenes and grasp rows stay valid, `scene_for` finds them instead of regenerating
# them, and the new 4,096 (indices 4096..8191) are the only ones that cost anything.
#
# The grasp stage is sharded 16 ways because it is the expensive one (~1 h for 4,096 hands).
# The retention screen that follows is the BUDGET SWEEP, not the single-clip screen the first
# 4,096 got: see scripts/real_v1_budget_rescreen.sh for why one clip is not a screen.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-docs/experiments/20260831-real_v1-sobol8192}
GEN=${GEN:-assets/mjcf/experimental/20260830-real_v1-sobol4096}   # superset: reuse the scenes
COUNT=${COUNT:-8192}
SEED=${SEED:-20260830}
SHARDS=${SHARDS:-16}
BUDGETS=${BUDGETS:-0.5,0.7,0.9,1.1,1.3}
WORKERS=${WORKERS:-18}
mkdir -p "$OUT" "$OUT/retention"

echo "=== stage A: manifest (generate the new scenes, filter to the measured rails) ==="
python3 scripts/real_v1_design_search.py --stage manifest --set sobol \
  --sobol-count "$COUNT" --sobol-seed "$SEED" --generated-dir "$GEN" \
  --out "$OUT/grasp_screen.json"

echo "=== stage A2: keep only the hands the rails reach ==="
python3 scripts/real_v1_filter_reachable.py \
  --manifest "$OUT/grasp_screen_manifest.json" --out "$OUT/hardware_manifest.json"
python3 -c "
import json, sys
m = json.load(open(sys.argv[1]))
open(sys.argv[2], 'w').write('\n'.join(r['design'] for r in m['designs']) + '\n')
print(len(m['designs']), 'reachable ->', sys.argv[2])
" "$OUT/hardware_manifest.json" "$OUT/reachable.txt"

# --manifest is a WRITE path.  Pointing it at hardware_manifest.json here overwrites the filter's
# output with the unfiltered population -- the population is restricted with --only-file instead.
echo "=== stage B: grasp screen, $SHARDS shards ==="
for i in $(seq 0 $((SHARDS - 1))); do
  python3 scripts/real_v1_design_search.py --stage grasp --set sobol \
    --sobol-count "$COUNT" --sobol-seed "$SEED" --generated-dir "$GEN" \
    --only-file "$OUT/reachable.txt" \
    --straddle 0.032,0.040 --thumb-axial 0.01,0.02 --depth auto --squeeze 0.004 \
    --axis-k 0.15,0.25,0.35,0.5 --angle-deg -90 --modes ik \
    --shard "$i" --shards "$SHARDS" --out "$OUT/grasp_shard_$i.json" \
    > "$OUT/grasp_shard_$i.log" 2>&1 &
done
wait

echo "=== stage C: one table per grasp cell ==="
python3 scripts/real_v1_prepare_retention.py \
  --shards "$OUT/grasp_shard_*.json" --manifest "$OUT/hardware_manifest.json" \
  --out-dir "$OUT/retention"

echo "=== stage D: the retention screen, swept across the clip ==="
RETENTION=(--hold-steps 2500 --selfcollision
           --load-target-units 250 --load-gain 0.0024 --capture-steps 400
           --proof-lift-mm 60 --proof-lift-steps 700 --proof-max-slip-mm 10
           --turn-torque-limit-nm 0.18044236 --hold-torque-limit-nm 0.06864655)
SCENE=(--object medium --bench-height 0.100 --post-y -35
       --flat-pads --pad-len-mm 14.8 --pad-width-mm 21.1)
CELL=(--cell-k 0.15,0.15,1 --cell-angles -80 --squeeze-mm 10
      --turn-steps 550 --hold-squeeze-mm 0)
for c in s32_t10 s32_t20 s40_t10 s40_t20; do
  python3 scripts/real_v1_deploy_envelope.py --mode cell \
    --designs-file "$OUT/retention/retention_designs_$c.txt" \
    --design-table "$OUT/retention/retention_table_$c.json" \
    --design-manifest "$OUT/hardware_manifest.json" --generated-dir "$GEN" \
    "${SCENE[@]}" "${CELL[@]}" "${RETENTION[@]}" \
    --cell-nom 1 --cell-ens 0 --budget "$BUDGETS" --workers "$WORKERS" \
    --out "$OUT/band_$c.json"
done

echo "=== stage E: one operating point per hand ==="
python3 scripts/real_v1_select_budget.py --band-dir "$OUT" \
  --table-dir "$OUT/retention" --out-dir "$OUT/selected"
echo "=== done ==="

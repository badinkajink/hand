#!/usr/bin/env bash
# Stages 2-4 of the clip re-screen: choose one operating point per hand, then earn it.
#
# Stage 1 (real_v1_budget_rescreen.sh) is a SINGLE trial per (cell, clip).  On a contact solve
# that varies with the settle, one trial is not a measurement -- it is a filter, and its job is
# only to be cheap enough to run 28,000 times.  What follows spends the saved compute on the
# survivors: five nominal repeats and a small wrong-hand ensemble, then a wide one for the few
# that survive that.
set -euo pipefail
cd "$(dirname "$0")/.."

POP=docs/experiments/20260830-real_v1-sobol4096
OUT=${OUT:-docs/experiments/20260830-real_v1-budget-rescreen}
GEN=${GEN:-assets/mjcf/experimental/20260830-real_v1-sobol4096}
WORKERS=${WORKERS:-18}
SEL="$OUT/selected"

RETENTION=(--hold-steps 2500 --selfcollision
           --load-target-units 250 --load-gain 0.0024 --capture-steps 400
           --proof-lift-mm 60 --proof-lift-steps 700 --proof-max-slip-mm 10
           --turn-torque-limit-nm 0.18044236 --hold-torque-limit-nm 0.06864655)
SCENE=(--object medium --bench-height 0.100 --post-y -35
       --flat-pads --pad-len-mm 14.8 --pad-width-mm 21.1)
CELL=(--cell-k 0.15,0.15,1 --cell-angles -80 --squeeze-mm 10
      --turn-steps 550 --hold-squeeze-mm 0)

echo "=== stage 2: one (cell, clip) per hand ==="
python3 scripts/real_v1_select_budget.py --band-dir "$OUT" \
  --table-dir "$POP/retention" --out-dir "$SEL"

echo "=== stage 3: five nominal + four half-error, at each hand's own clip ==="
for f in "$SEL"/designs_b*.txt; do
  b=$(basename "$f" .txt); b=${b#designs_b}
  budget=$(python3 -c "print(int('$b') / 100)")
  echo "--- clip $budget ($(tr ',' '\n' < "$f" | grep -c .) designs)"
  python3 scripts/real_v1_deploy_envelope.py --mode cell \
    --designs-file "$f" --design-table "$SEL/selected_table.json" \
    --design-manifest "$POP/hardware_manifest.json" --generated-dir "$GEN" \
    "${SCENE[@]}" "${CELL[@]}" "${RETENTION[@]}" \
    --cell-nom 5 --cell-ens 4 --cell-level 0.5 --budget "$budget" \
    --workers "$WORKERS" --out "$OUT/confirm_b$b.json"
done

echo "=== stage 4: twenty full-error draws for whatever confirmed ==="
python3 scripts/real_v1_select_budget.py --band-dir "$OUT" --glob "confirm_b*.json" \
  --table "$SEL/selected_table.json" --min-kept-frac 0.6 --out-dir "$SEL/confirmed"
for f in "$SEL"/confirmed/designs_b*.txt; do
  b=$(basename "$f" .txt); b=${b#designs_b}
  budget=$(python3 -c "print(int('$b') / 100)")
  echo "--- clip $budget ($(tr ',' '\n' < "$f" | grep -c .) designs)"
  python3 scripts/real_v1_deploy_envelope.py --mode cell \
    --designs-file "$f" --design-table "$SEL/confirmed/selected_table.json" \
    --design-manifest "$POP/hardware_manifest.json" --generated-dir "$GEN" \
    "${SCENE[@]}" "${CELL[@]}" "${RETENTION[@]}" \
    --cell-nom 5 --cell-ens 20 --cell-level 1.0 --budget "$budget" \
    --workers "$WORKERS" --out "$OUT/full_error_b$b.json"
done
echo "=== done ==="

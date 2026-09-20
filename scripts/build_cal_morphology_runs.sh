#!/usr/bin/env bash
# Build the calibrated-plant morphology runs for the deployed hands, choosing the squeeze by
# the Phase-1 evaluator's lift (first squeeze in the order given that ends held on 3 pads with
# contact persistence >= 0.95 and no more than 10 mm dropped from the peak).
#   scripts/build_cal_morphology_runs.sh 20260919 "D1 D2 D3 D4 D5 D7 D8" "10 8 6"
set -u
DATE=${1:-$(date +%Y%m%d)}; HANDS=${2:-"D1 D2 D3 D4 D5 D7 D8"}; SQS=${3:-"10 8 6"}
cd "$(dirname "$0")/.."
declare -A TAG=( [D1]=sv1_w6689_b060 [D2]=sv1_w2360_b075 [D3]=sv1_u1364_b080 [D4]=g12_b095 [D5]=sv1_u0060_b75 [D6]=sv1_u0308_b050 [D7]=rv05_manual_b85 [D8]=sv1_w0099_b100 )
for h in $HANDS; do
  tag=${TAG[$h]}
  for sq in $SQS; do
    out=results/phase1/real_v1/${DATE}-${tag}_sq${sq}_cal
    if [ ! -f $out/summary.json ]; then
      env -u PYTHONPATH uv run --extra rl python scripts/make_cal_morphology_run.py --hand $tag --squeeze $sq --out $out > logs/${DATE}-morph_${tag}_sq${sq}.log 2>&1
    fi
    verdict=$(python3 - "$out/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d["best_metrics"]
ok = m["cube_tip_contacts"] >= 3 and m["contact_persistence"] >= 0.95 and m["cube_z_drop_from_peak"] <= 0.010 and m["cube_lift"] >= 0.08
print(("PASS" if ok else "fail") + f" contacts {m['cube_tip_contacts']:.0f} persist {m['contact_persistence']:.2f} lift {m['cube_lift']*1000:.0f}mm drop {m['cube_z_drop_from_peak']*1000:.1f}mm")
PY
)
    echo "$h $tag sq$sq: $verdict"
    case "$verdict" in PASS*) echo "$h $tag $sq $out" >> results/phase1/real_v1/${DATE}-cal_hands.txt; break;; esac
  done
done

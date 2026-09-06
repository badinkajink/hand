#!/usr/bin/env bash
# The held floor-free turn, on the hand that PHYSICALLY EXISTS (rv05_manual_stored).
# rv03_narrowy_sp40 has no deploy plan; every number from it is sim-only.
# Two budgets so the axis height and the residual budget are separable.
set -u
cd /home/humanoid/Programs/hand
OUT=docs/experiments/20260906-rv05_band
export WARP_CACHE_PATH=$(mktemp -d)
export MUJOCO_GL=egl
printf '%-22s %5s %6s %6s %6s %7s %8s %5s %8s %8s\n' case budget axisK axisH peak final z con N ikRes
for RUN in results/phase1/real_v1/rv05_manual_stored results/phase1/real_v1/rv04_mid_sp40; do
for B in 0.5 0.85; do
for K in 0.05 0.10 0.15 0.20 0.25 0.30; do
  J=$OUT/carry_$(basename $RUN)_b${B}_k${K}.json
  uv run --extra rl --extra gpu python scripts/probe_real_v1_carry.py \
    --morph-run "$RUN" --straddle 0.040 --lift 0.10 \
    --turn-steps 250 --angle-deg -90 --budget "$B" --axis-k "$K" \
    --linear-anchor --repeats 6 --out "$J" 2>>logs/20260906-rv05_band.err >/dev/null
  python3 - "$J" "$(basename $RUN)" "$B" "$K" <<'PY'
import json,sys,statistics as st
j,run,b,k=sys.argv[1:5]
try: rows=json.load(open(j))
except Exception as e:
    print('%-22s %5s %6s   FAILED %s'%(run,b,k,e)); raise SystemExit
rows=rows if isinstance(rows,list) else [rows]
g=lambda r,f: float(r.get(f) or 0.0)
fc=[g(r,'final_cos') for r in rows]
print('%-22s %5s %6s %6.1f %6.3f %6.3f %7.4f %5.1f %8.2f %8.2f  n=%d mean %+.3f sd %.3f'%(
  run,b,k,g(rows[0],'axis_height_mm'),
  st.mean([g(r,'peak_cos') for r in rows]), st.mean(fc),
  st.mean([g(r,'final_z') for r in rows]), st.mean([g(r,'contacts') for r in rows]),
  st.mean([g(r,'force_N') for r in rows]), max(g(r,'max_ik_residual_mm') for r in rows),
  len(rows), st.mean(fc), (st.stdev(fc) if len(fc)>1 else 0.0)))
PY
done; done; done
echo DONE

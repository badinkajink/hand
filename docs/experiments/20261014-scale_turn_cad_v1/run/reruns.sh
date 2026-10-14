#!/bin/bash
# Runs added after queue_cpu.sh and queue_gpu.sh (2026-10-14, 09:30):
#  1. GPU iCEM again with 256 contact matches per sensor (hand_object_scale_mpc_gpu.py); the first icem_gpu rows of
#     queue_gpu.sh overflowed MuJoCo-Warp's default 64 and stay in planners/turn.jsonl without a variant; the page uses mm256.
#  2. Pinned iCEM at placement 0 on every cell with a feasible grasp (39), two six-thread shards, about 1.4 h.
cd /home/humanoid/Programs/hand || exit 1
export SCALE_HAND=cad_v1
D=docs/experiments/20261014-scale_turn_cad_v1
L=logs/20261014-scale_turn_cad_v1
WARP_CACHE_PATH=$(mktemp -d) nohup setsid ~/.claude/bin/resguard.sh run --mem 4G --cpu 200 -- .venv/bin/python \
  scripts/hand_object_scale_planners.py run --configs icem_gpu \
  --cells cylinder:25:0.125 cylinder:16:0.625 cylinder:10:0.75 cylinder:16:-0.375 --seeds 0 1 --nthread 1 --variant mm256 \
  > $L/planners_gpu_mm256.log 2>&1 < /dev/null &
CELLS=$(.venv/bin/python -c "
import json
ks=sorted({(r['d_mm'], r['t']) for r in map(json.loads, open('$D/hom_postoff_servo/turn.jsonl')) if r.get('status')=='ok'})
print(' '.join(f'cylinder:{d:g}:{t:g}' for d,t in ks))")
for k in 0 1; do
  nohup setsid ~/.claude/bin/resguard.sh run --mem 3G --cpu 700 -- .venv/bin/python scripts/hand_object_scale_planners.py run \
    --configs icem_pin --cells $CELLS --seeds 0 --nthread 6 --out $D/planners/landscape_icem_pin.jsonl --shard $k/2 \
    > $L/landscape_icem_pin_$k.log 2>&1 < /dev/null &
done

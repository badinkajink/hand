#!/bin/bash
# HOM grids of the CAD-hand scale-turn study (2026-10-14, 01:2x): post present and post removed, servo plant, three
# placements per grasp, 10/16/25 mm cylinders on the diagonal layouts; three one-core shards per grid. queue_cpu.sh and
# queue_gpu.sh wait for these to end.
cd /home/humanoid/Programs/hand || exit 1
export SCALE_HAND=cad_v1
D=docs/experiments/20261014-scale_turn_cad_v1
L=logs/20261014-scale_turn_cad_v1
SERVO=kp0.72_kv0_fr0.18_fl0_dp0.0144
mkdir -p $L
for k in 0 1 2; do
  nohup setsid ~/.claude/bin/resguard.sh run --mem 1G --cpu 100 -- .venv/bin/python scripts/hand_object_scale_sim.py run \
    --cells diag --what turn --seeds 0 1 2 --plant $SERVO --set kp_model=0.72 post_off=1 --out-dir $D/hom_postoff_servo \
    --only cylinder:10 cylinder:16 cylinder:25 --shard $k/3 > $L/hom_postoff_servo_$k.log 2>&1 < /dev/null &
  nohup setsid ~/.claude/bin/resguard.sh run --mem 1G --cpu 100 -- .venv/bin/python scripts/hand_object_scale_sim.py run \
    --cells diag --what turn --seeds 0 1 2 --plant $SERVO --set kp_model=0.72 --out-dir $D/hom_post_servo \
    --only cylinder:10 cylinder:16 cylinder:25 --shard $k/3 > $L/hom_post_servo_$k.log 2>&1 < /dev/null &
done

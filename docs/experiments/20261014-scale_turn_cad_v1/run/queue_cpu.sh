#!/bin/bash
# CAD-hand rerun of the scale-turn study (owner 2026-10-14: "rerun this work, but with the newest CAD-faithful hand ...
# replicate the page but with the updated CAD results"). Serial stages, each in two resguard-wrapped shards of six
# threads; a stage starts when the previous one has ended. Stop it by killing this script's process group.
cd /home/humanoid/Programs/hand || exit 1
export SCALE_HAND=cad_v1
D=docs/experiments/20261014-scale_turn_cad_v1
L=logs/20261014-scale_turn_cad_v1
SERVO=kp0.72_kv0_fr0.18_fl0_dp0.0144
RG="$HOME/.claude/bin/resguard.sh run"
CELLS4="cylinder:25:0.125 cylinder:16:0.625 cylinder:10:0.75 cylinder:16:-0.375"
stamp() { echo "$(date '+%F %T') $*" >> $L/queue_cpu.log; }

# the grasp of every MPC run is the best HOM grasp without the post: wait for those rows
while pgrep -f "[h]and_object_scale_sim.py run.*20261014-scale_turn_cad_v1" > /dev/null; do sleep 30; done
stamp "HOM grids done"

stamp "planners start"
for k in 0 1; do
  $RG --mem 3G --cpu 700 -- .venv/bin/python scripts/hand_object_scale_planners.py run \
    --configs hom ps ps_pin mppi_pin cem_pin icem_pin anneal_pin --cells $CELLS4 --seeds 0 1 --nthread 6 \
    --shard $k/2 >> $L/planners_$k.log 2>&1 &
done
wait
stamp "planners done"

stamp "planner films start"
$RG --mem 3G --cpu 700 -- .venv/bin/python scripts/hand_object_scale_planners.py film --configs hom ps ps_pin icem_pin \
  --cells cylinder:25:0.125 cylinder:16:0.625 --seed 0 --nthread 6 >> $L/films_pl_0.log 2>&1 &
$RG --mem 3G --cpu 700 -- .venv/bin/python scripts/hand_object_scale_planners.py film --configs hom ps ps_pin icem_pin \
  --cells cylinder:10:0.75 cylinder:16:-0.375 --seed 0 --nthread 6 >> $L/films_pl_1.log 2>&1 &
wait
stamp "planner films done"

stamp "three-pad MPC grid start"
for k in 0 1; do
  $RG --mem 3G --cpu 700 -- .venv/bin/python scripts/hand_object_scale_mpc.py run --plant $SERVO --cells diag \
    --shapes cylinder --only cylinder:10 cylinder:16 cylinder:25 --seeds 0 --hom-rows $D/hom_postoff_servo/turn.jsonl \
    --out-dir $D/mpc3_servo --traces --nthread 6 --set w_contact=1 w_rate=1 nsample=64 horizon=0.5 \
    --shard $k/2 >> $L/mpc3_servo_$k.log 2>&1 &
done
wait
stamp "three-pad MPC grid done"

stamp "MPC grid start"
for k in 0 1; do
  $RG --mem 3G --cpu 700 -- .venv/bin/python scripts/hand_object_scale_mpc.py run --plant $SERVO --cells diag \
    --shapes cylinder --only cylinder:10 cylinder:16 cylinder:25 --seeds 0 1 2 \
    --hom-rows $D/hom_postoff_servo/turn.jsonl --out-dir $D/mpc_servo --traces --nthread 6 \
    --shard $k/2 >> $L/mpc_servo_$k.log 2>&1 &
done
wait
stamp "MPC grid done"

stamp "three-panel films start"
$RG --mem 4G --cpu 1300 -- .venv/bin/python scripts/hand_object_scale_mpc.py film --plant $SERVO \
  --hom-rows $D/hom_postoff_servo/turn.jsonl --cells $CELLS4 --seed 0 --nthread 12 --set3 nsample=64 horizon=0.5 \
  --size 640x480 --crf 22 >> $L/films_3panel.log 2>&1
stamp "three-panel films done"

stamp "timing and post probe start"
$RG --mem 3G --cpu 1300 -- .venv/bin/python scripts/hand_object_scale_mpc_timing.py --plant $SERVO >> $L/timing.log 2>&1
$RG --mem 2G --cpu 100 -- .venv/bin/python scripts/hand_object_scale_post_probe.py --plant $SERVO --cells $CELLS4 \
  --seeds 0 --rows $D/hom_post_servo/turn.jsonl >> $L/post_probe.log 2>&1
# the GPU check times 12 CPU threads against the GPU: only once the GPU planner runs have ended
while pgrep -f "[h]and_object_scale_planners.py run --configs (anneal|icem)_gpu" > /dev/null; do sleep 60; done
for nw in 256 1024; do
  WARP_CACHE_PATH=$(mktemp -d) $RG --mem 4G --cpu 1300 -- .venv/bin/python scripts/hand_object_scale_mpc_gpu.py check \
    --nworld $nw --pad-filter 1 >> $L/gpu_check.log 2>&1
done
WARP_CACHE_PATH=$(mktemp -d) $RG --mem 4G --cpu 1300 -- .venv/bin/python scripts/hand_object_scale_mpc_gpu.py check \
  --nworld 64 --pad-filter 0 >> $L/gpu_check.log 2>&1
stamp "queue done"

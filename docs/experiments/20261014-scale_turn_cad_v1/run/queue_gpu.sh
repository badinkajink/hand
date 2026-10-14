#!/bin/bash
# GPU part of the CAD-hand rerun: annealed sampling and iCEM on MuJoCo-Warp (1,024 rollouts per replanning) on the four
# planner cells, the GPU-against-CPU rollout check is the last stage of queue_cpu.sh. Runs beside queue_cpu.sh (one CPU thread each).
cd /home/humanoid/Programs/hand || exit 1
export SCALE_HAND=cad_v1
L=logs/20261014-scale_turn_cad_v1
RG="$HOME/.claude/bin/resguard.sh run"
CELLS4="cylinder:25:0.125 cylinder:16:0.625 cylinder:10:0.75 cylinder:16:-0.375"
stamp() { echo "$(date '+%F %T') $*" >> $L/queue_gpu.log; }
while pgrep -f "[h]and_object_scale_sim.py run.*20261014-scale_turn_cad_v1" > /dev/null; do sleep 30; done
for cfg in anneal_gpu icem_gpu; do
  stamp "$cfg start"
  WARP_CACHE_PATH=$(mktemp -d) $RG --mem 4G --cpu 200 -- .venv/bin/python scripts/hand_object_scale_planners.py run \
    --configs $cfg --cells $CELLS4 --seeds 0 1 --nthread 1 >> $L/planners_gpu.log 2>&1
  stamp "$cfg done"
done
stamp "queue done"

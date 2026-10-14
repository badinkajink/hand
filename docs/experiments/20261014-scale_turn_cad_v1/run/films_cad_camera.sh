#!/bin/bash
# Films of the CAD-hand page again from 10 deg below the grasp (hand_object_scale_mpc.py film_pair under SCALE_HAND=cad_v1):
# at real_v1's camera the printed links hide the cylinder (2026-10-14 15:40). Same commands as queue_cpu.sh's film stages;
# the new rows append to the films.jsonl files and the page uses the latest film of each cell.
cd /home/humanoid/Programs/hand || exit 1
export SCALE_HAND=cad_v1
D=docs/experiments/20261014-scale_turn_cad_v1
L=logs/20261014-scale_turn_cad_v1
SERVO=kp0.72_kv0_fr0.18_fl0_dp0.0144
RG="$HOME/.claude/bin/resguard.sh run"
CELLS4="cylinder:25:0.125 cylinder:16:0.625 cylinder:10:0.75 cylinder:16:-0.375"
stamp() { echo "$(date '+%F %T') $*" >> $L/films_cad_camera.log; }
stamp "planner films start"
$RG --mem 3G --cpu 700 -- .venv/bin/python scripts/hand_object_scale_planners.py film --configs hom ps ps_pin icem_pin \
  --cells cylinder:25:0.125 cylinder:16:0.625 --seed 0 --nthread 6 >> $L/films_pl_cam_0.log 2>&1 &
$RG --mem 3G --cpu 700 -- .venv/bin/python scripts/hand_object_scale_planners.py film --configs hom ps ps_pin icem_pin \
  --cells cylinder:10:0.75 cylinder:16:-0.375 --seed 0 --nthread 6 >> $L/films_pl_cam_1.log 2>&1 &
wait
stamp "planner films done"
stamp "three-panel films start"
$RG --mem 4G --cpu 1300 -- .venv/bin/python scripts/hand_object_scale_mpc.py film --plant $SERVO \
  --hom-rows $D/hom_postoff_servo/turn.jsonl --cells $CELLS4 --seed 0 --nthread 12 --set3 nsample=64 horizon=0.5 \
  --size 640x480 --crf 22 >> $L/films_3panel_cam.log 2>&1
stamp "three-panel films done"
stamp "queue done"

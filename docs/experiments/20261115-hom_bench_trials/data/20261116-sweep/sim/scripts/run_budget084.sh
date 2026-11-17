#!/bin/bash
# hand-4a, 2026-11-16 20:0x: the blind predictions again with HOM's targets under hom_bench.torque_budget 0.84 N m
# (owner: "i think we can try again with the capped targets"), the settings of 20261116-predictions.json otherwise.
cd /home/humanoid/Programs/hand/logs/20261116-sim_real_reversal
export HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
G="grid2dgr:329:4 grid2dgr:329:12 grid2dgr:329:8 grid2dgr:329:13"
../../.venv/bin/python rev_predict.py predict_budget084_kp035.json --grasps $G --kp-yaw-im 0.35 --torque-budget 0.84 --workers 8
../../.venv/bin/python rev_predict.py predict_budget084_all055.json --grasps $G --torque-budget 0.84 --workers 8
echo "DONE $(date +%H:%M)"

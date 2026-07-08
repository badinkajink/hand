#!/bin/bash
# Run 12: Trajectory Control with Contact-Focused Objectives
#
# Key Improvements over Run 11:
# 1. Increased contact_persistence from 0.8 to 6.0 (penalize lost contacts)
# 2. Increased min_finger_persistence from 2.0 to 10.0 (require each finger to stay)
# 3. Increased imbalance penalty from 1.0 to 0.5 (penalize losing any finger)
# 4. Same CEM parameters (120 iter, 160 pop) for consistency
#
# Hypothesis: These weight changes prioritize maintaining all three fingertips
# throughout the manipulation. The script defaults will keep lift_weight at 35.0,
# but our contact-focused weights should help overcome lift-biased optimization.

PYTHON_BIN=.venv/bin/python
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN=python
fi

"$PYTHON_BIN" scripts/phase1_optimize_grasp.py \
    --scene-xml assets/mjcf/scene_power_drill_short_proximal.xml \
    --keyframe open_flat \
    --optimizer cem \
    --iterations 120 \
    --population 160 \
    --elite-fraction 0.25 \
    --sigma-init 0.10 \
    --seed 12 \
    --output-dir results/phase1 \
    --tag run12_power_drill_short_proximal_contact_focused_traj \
    --lift-delta-z 0.060 \
    --lift-ramp-steps 100 \
    --pivot-steps 180 \
    --pivot-ramp-steps 120 \
    --pivot-delta-rz 1.5708 \
    --objective-weight-xy-drift-penalty 6.0 \
    --objective-weight-drop-penalty 12.0 \
    --objective-weight-contact-persistence 6.0 \
    --objective-weight-min-finger-persistence 10.0 \
    --objective-weight-finger-persistence-imbalance-penalty 0.5 \
    --objective-weight-finger-yaw-drift-penalty 0.8 \
    --objective-weight-finger-flex-drift-penalty 0.4 \
    --objective-weight-cube-yaw-drift-penalty 3.0 \
    --objective-weight-cube-axis-tilt-penalty 2.0 \
    --objective-weight-cube-ang-drift-penalty 1.0 \
    --traj-phases 4 \
    2>&1 | tee results/phase1/run12_contact_focused_traj_launch.log

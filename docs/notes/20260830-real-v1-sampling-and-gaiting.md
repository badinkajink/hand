# Catalogued sidequests: real_v1 sampling and post-turn gaiting

Status: **catalogue only; no experiments launched from this note.** Sampling is the higher
priority sidequest. Continued gaiting is lower priority still.

## 1. Sample substantially more hardware-valid hands

The current paper statement, "searched 108 morphologies," is accurate but easy to
overread. The set is 5 known anchors, one-axis sweeps, a 5x5 compact-family plane, and only
48 uniform random draws in the six movable XY coordinates. It is not dense coverage of the
six-dimensional `real_v1` gantry workspace. Of the 108, 80 found a grasp, 58 exhibited the
pinch-roll contact style, and 49 cleared the reorientation threshold; 97 were within gantry
travel, but buildability alone did not include trajectory self-collision. Only g12 currently
has a collision-safe exported deployment plan.

This matters to the paper framing. The existing result supports: under the present one-shot,
open-loop carry objective, successful morphologies occupy a structured subset and mostly
discover a simple pinch-roll role split. It does **not** yet support: the useful morphology
space itself is intrinsically tiny. Sparse sampling, grasp fitting, the fixed carry family,
and the task objective are all selection operators.

The present heuristics are good enough for a much larger staged search without paying RL cost
per hand:

1. Generate only from `assets/mjcf/real_v1/real_hand.xml`, within `REAL_V1_WORKSPACE`, and
   reject self-collision throughout the proposed trajectory—not merely at reset.
2. Use a space-filling sequence (Sobol or Latin hypercube) over the full six XY coordinates,
   retaining explicit asymmetric designs. Report generated, workspace-valid, collision-valid,
   graspable, pinch-roll, and successful counts separately.
3. Run the cheap pose/IK and grasp screens first. CEM only the survivors, then sweep the existing
   pivot-height and turn-angle carry cells.
4. Use current grasp/torque/style scores as filters and analysis features, not as trusted
   reorientation surrogates: only thumb torque had useful discrimination (AUC 0.821), while the
   fixed-contact ceiling and mount separations were weak.
5. Spend repeated robustness trials on a diverse Pareto set, not only the top nominal score.
   Nominal ranking already failed to predict the careful-bench ranking (`rv04_mid` versus g12).
6. Promote to RL and hardware only after export-plan and clearance gates. Keep at least one
   morphology-diverse control per behavior style so the search does not collapse to many near
   duplicates of the same middle-finger pinch-roll mechanism.

Before broadening the manipulation objective, this sampling pass should determine whether the
apparent thin slice persists under denser, unbiased coverage. If it does, that is a much stronger
co-design result. If it does not, the current 108-hand conclusion becomes a sampling artifact
that should be corrected before paper submission.

## 2. Continued axial rotation / in-hand gaiting (lower priority)

The current task ends after standing the cylinder up. A screwdriver-like task would continue
rotating about the shaft axis, eventually exhausting a finger's range and requiring release,
reposition, and re-contact. That introduces a qualitatively different morphology test: contact
role switching, periodic gaiting, grip recovery, and cumulative rotation while preserving axial
load.

This may elicit hands and behaviors that the one-shot pinch-roll objective cannot distinguish,
and it is a principled way to require more than the current middle-finger-dominated trajectory.
But it should follow the denser one-shot sampling study. Otherwise a new task, controller,
reward, and morphology distribution all change simultaneously, making it impossible to tell
whether new hands arise from genuine task complexity or from another sampling/configuration
artifact.


# DiffMJX evaluation plan

The [2026-10-07 real_v1 contact-gradient result](../experiments/20261007-diffmjx/20261007-contact_gradients_real_v1.html)
tests the authors' released MJX fork on the D1 sphere-packed fingertip and
actual screwdriver cylinder. The [runbook](../experiments/20261007-diffmjx/README.md)
pins the source revisions, scenes and commands. The April `diffmjx-mvp` cube
strategy was an MJX distance-reward experiment, not a run of the authors' code.

The released fork and native MJX both completed one step on the exact 2,388-pad
surface. At the D1 held grasp, the plain fork's 20 ms three-yaw gradient chose
an action with the same CPU MuJoCo orientation gain as six CPU finite
differences. Its warm gradient took 35.7 s; the six CPU evaluations took
23.7 ms. The fork's contact-from-distance and scan-loop settings changed the
action direction and reduced the gain from 0.000206 to 0.000200, with 56.7 s
warm gradient time. All action scores are from CPU MuJoCo 3.3.1.

## Next gate

1. Replay the D1 plan to the turn onset and compare 100 ms CPU MuJoCo and
   released MJX trajectories with the 1 mm packed tip and cylinder. Record
   tool pose, fingertip normal force and contact torque every 1 ms. Use
   `scripts/real_v1_diffmjx_official_gate.py` as the starting implementation.
2. Test first touch from 0–3 mm tool gaps with the same packed tip. Compare
   plain MJX, contact-from-distance with the scan-loop solver, CPU finite
   differences and the current CEM grasp search. Score retained grasp rate,
   final contact force and wall time over D1–D8.
3. If longer-horizon forward agreement holds, compare CPU finite differences,
   a fixed-budget sampling search and released MJX gradient steps on a 100 ms
   turn-and-hold score. Report vertical turn, object retention and wall time
   per control update over D1–D8 and five seeds per hand. The 20 ms gradient
   measurements set the CPU cost baseline.

The current PPO trainer uses MuJoCo-Warp through mjlab. A MuJoCo Playground
port would test a simulator and training framework change. Simulator
derivatives enter only when the training or control algorithm differentiates
through the physics rollout.

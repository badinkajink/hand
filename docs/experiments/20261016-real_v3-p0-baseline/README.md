# P0 baseline — D1's open-loop plan on the real_v3 hand, 20–40 mm × 20–150 g

2026-10-16. The reference the residual RL policy has to beat.

**What ran.** D1's (`sv1_w6689`) shipped bench trajectory `sv1_w6689_b050_traj.csv`, replayed
unchanged at 50 Hz. The sequence is: close (0.5 s), settle (0.8 s), turn −80° about the pinch axis
(1.1 s), then hold to 4.0 s.

The hand is real_v3 at gantry D1, using the MorphIt collision model: 408 link spheres and a 1 mm sphere
grid on every fingertip. The servos are SCS0009-like: 0.23 N·m and about 11 rad/s.

The cylinder is staged where the plan expects it. Its axis is 50.5 mm below the yaw axes, 100 mm up
on an 8 mm post at y −35 mm. It is held still until the grip closes. Every size sits at that same axis
position, so the plan is not adapted per object. Each trial adds ±2 mm and ±2° of staging error.

The grid is 9 diameters × 7 masses × 16 trials, 1,008 rollouts in total, run in 25 s on the M4 Pro.

**Scores.** Each one is measured over the reorientation phase, turn plus hold, from t ≥ 1.3 s.
- **contact retention**: the fraction of steps with at least 2 fingers on the cylinder. Any finger part
  counts; the table and post do not.
- **all three**: the fraction of steps with all 3 fingers on.
- **held**: at least 2 fingers on at the final step, and the tool never sank more than 30 mm below
  its height at the grip. A tool that slides out through the fingers fails, even while the fingers
  are still touching it.

**Result.**
- **Overall:** retention 91.5%, held 82.8% (`p0_retention.png`).
- **20–30 mm:** solid at every mass. Every cell is above 96% retention and every trial holds, except
  30 mm × 150 g at 12/16.
- **32.5–35 mm:** fine until the mass is high, then it collapses. At 32.5 mm × 150 g 2/16 hold; at
  35 mm × 125 g 4/16; at 35 mm × 150 g 0/16.
- **37.5 mm:** holds only when light. At 80 g and above, at most 1/16 holds.
- **40 mm:** weak at every mass, 0–12/16 held.
- **Grip quality:** the all-three-fingers fraction falls off well before retention does, so the
  grip loosens before it fails.

The drops match the bench's own failure shape (`20260901-real_v1-transfer-firstpass`): the shaft
turns first and is lost afterward. `p0_drop_37.5mm_100g.mp4` reaches |cos| 0.81 at 1.9 s and then
falls to the table.

**Files.**
- `p0_trials.json`: every trial, with a per-step trace of contact count, |cos| and height.
- `p0_retention.png`: the heatmaps.
- `p0_success_25mm_60g.mp4` (seed 1): a held turn ending at |cos| 0.72.
- `p0_drop_37.5mm_100g.mp4` (seed 5): turns about 35°, then drops.

**Reproduce.**
- The grid: `.venv-cpu/bin/python scripts/cpu_grasp_baseline_p0.py`
- The heatmaps: `scripts/cpu_grasp_baseline_plot.py <p0_trials.json>`
- One trial: `--record diameter,mass,seed --out x.mp4`

**Caveats.**
- The real v3 joint ranges are not calibrated; the v1 ranges are used.
- The staging hold stands in for the operator.
- The final turn is about 48° (|cos| ~0.71). Hardware reached ~0.83–0.88 on the v1-era fingers, so
  v3's longer middle link turns somewhat less with the same plan.

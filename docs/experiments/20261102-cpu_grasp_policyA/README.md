# 2026-11-02: policy A (residual PPO with the UR5 palm-height action) on the E1 trial set

**Result: success 0.975 vs 0.209 for the plan alone, on the same 1008 trials** (9 diameters 20–40 mm ×
7 masses 20–150 g × 16; same placements ±2 mm / ±2° and pad friction 1.5–2.5 as E1 v2, seed 0).
Lifted 0.997 vs 0.799.

- Run: `runs/cpu_grasp/A_palm_s0_full` (`scripts/cpu_grasp_train_ppo.py --cond A --iters 200 --envs 512`),
  about 61 min on the Mac CPU. Best evaluation-grid checkpoint is iteration 120 (`policy_best.pt`, 0.964 on
  the 4-trial evaluation grid); the final iteration-200 checkpoint scores 0.960.
- Policy: PPO (proximal policy optimization), 10 actions = 9 finger-target residuals (≤ DQ_MAX) + UR5 palm
  height residual (≤ 5 mm), low-pass filtered (BETA 0.5); actor sees joint history, tracking error, previous
  action, phase, palm residual (48 values); critic additionally sees object state, contacts, size, mass,
  friction. Default MuJoCo position actuator (no BAM actuator model yet: that is policy B).
- Scored with `scripts/cpu_grasp_eval_policy.py` (deterministic policy). `plan_alone/` is the same script with
  zero action and reproduces the E1 v2 baseline (0.209).

| | plan alone | plan + policy A |
|---|---|---|
| success (held through the D1 bench turn, tilt ≥ 45°) | 0.209 | **0.975** |
| lifted 150 mm | 0.799 | 0.997 |
| trials with any finger–table contact | 0.889 | 0.334 |
| fraction of grasp steps with finger–table contact | 0.257 | 0.004 |
| mean \|finger residual\| | 0 | 2.7° |
| mean peak \|palm residual\| | 0 | 4.9 mm (at the 5 mm limit) |

Weakest cells: 25 mm × 100 g and 20 mm × 125 g (0.62 each), 20 mm × 150 g (0.81).

## What the policy does

- **Palm:** raises the palm up to the 5 mm limit while the fingers close (steps ~5–30), then lowers it back
  to about 0 by the lift (and below the plan for heavy objects, −3.5 mm on 25 mm × 100 g). That lifts the
  fingertips off the table during the close, which was the plan's main failure. The limit is saturated, so a
  larger DZ_MAX may help.
- **Remaining table contact** is only at control steps 0–1 on the ≤ 25 mm cylinders: the plan's starting
  pre-shape already touches the table before any residual has acted. That is a plan / reset issue (raise the
  pre-shape or start the palm higher), not a policy failure.
- **Tilt:** lower than the plan's on the easy small-light case (25.8 mm × 28.5 g: 56° vs 86°, both successes):
  the policy holds the cylinder more axially in exchange for robustness.

## Files

- `compare.png`: plan alone vs policy, success per cell. `heatmap.png`: policy lifted / success / table contact.
- `learning_curve.png`: training and evaluation success per iteration, return and action std.
- `trials.csv`, `grid.csv` (and the same under `plan_alone/`).
- Videos: `manta/sim_media/20261102-policyA/` (`scripts/cpu_grasp_render.py`); top row plan alone, bottom row
  policy, side view (RealSense-like) and isometric, friction 2.0, centered placement:
  25.8 mm × 28.5 g (the real foam-taped test cylinder; both succeed), 20 × 60 (plan drops it in the lift),
  25 × 125, 35 × 80, 40 × 150 (plan loses them in the bench turn; policy holds all).

## Caveats before the zero-shot rollout

- Trained and scored on the same randomization (placement, friction). No joint-zero offsets (±3° from the
  2026-10-30 zero check), no BAM actuator, no link-mass or yaw-friction randomization yet: policies B / C.
- The judge's bench turn is applied open-loop after the grasp (the policy does not act during it), so the real
  test is the same: policy for the grasp and lift, then the D1 deploy turn.

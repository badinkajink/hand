# In-Hand Reorientation

A chronological log of the in-hand reorientation training journey:
flat-laying screwdriver_medium → vertical, finger-only control.

Style follows [phases.md](phases.md) — each section captures **goal**,
what **changed**, **result**, **takeaway**. New attempts append here; do
not edit prior entries except for factual corrections (this is the
historical record).

For the related grasp work see [phases.md](phases.md). For architecture
details on `LerpFinger` / `ScriptedPalm` see
[architecture.md](architecture.md). The earlier design memo lives at
[inhand_reorient.md](inhand_reorient.md) (kept for reference; this doc
supersedes it as the source of truth).

---

## Task definition

**Scene:** `scene_screwdriver_medium_flat_short_proximal.xml`. Cylinder
(R = 12.5 mm, L = 80 mm) lying flat on the floor with body quat ≈
(0.707, 0.707, 0, 0) — a 90° X-rotation that puts the cylinder's long
axis along world Y. Mass 16 g, friction (2.4, 0.2, 0.02).

**Goal:** the cylinder's body-local +Z (long axis) aligned with world +Z
(vertical). Cosine alignment goes from ~0 (flat) to ~1 (vertical) without
dropping the cylinder or letting it touch the floor.

**Constraints (per user):**
1. **Finger-only manipulation.** The palm's 6 joints are scripted; only
   the 9 finger DOFs are policy-controlled.
2. **Maintain contact throughout.** Tip-lost = drop, terminate.
3. **No floor assistance.** The reorientation must be in-hand; floor
   contact during the rotation phase = cheating. (This constraint was
   added after v4 — see Phase v4 below.)
4. **Sim-to-real safety.** Motion should be smooth (no high-frequency
   finger jitter that wouldn't survive motor delays / sensor noise).
   (Added after Policy B v1 — see Phase Policy B below.)

---

## Representation

### Observation (full mode)

Standard manipulation obs plus reorient-specific additions:

- Per-joint robot state (palm + fingers): `joint_pos`, `joint_vel`
- Object pose in palm frame: `object_pose_actual` (3 pos + 4 quat)
- Object linear + angular velocity
- Per-fingertip contact: `fingertip_cube_contact.found` (binary per tip)
- Per-step fingertip-to-object distance
- Last policy action (for residual modeling)
- **`target_axis_misalignment`**: scalar acos(cos(object_axis,
  target_axis_world)) — added for the reorient task so the policy sees
  the misalignment angle directly instead of having to infer it from
  the raw quat.

### Action

9-dim finger residual on top of `LerpFinger` setpoint, scale 0.5 rad.

`LerpFinger` interpolates from `open_finger_qpos` toward
`finger_default_ctrl` (CEM grip) over `finger_close_sim_steps`. Policy
adds ±0.5 rad on top of this time-varying setpoint.

**Palm action is scripted** (`ScriptedPalmAction`): hold at CEM keyframe
during settle (`settle_steps=240` sim steps), then ramp palm_pz by
`lift_delta_z` over `lift_ramp_steps=80` sim steps. Palm rotations
(rx/ry/rz) are held at the keyframe value.

A 12-dim action variant (9 finger + 3 palm rotation residuals) was
prototyped (v1) and rejected — see Phase v1.

### Warmstart

All runs warmstart the actor from
`medium_flat_stable_v1/model_500.pt` (the lift-only policy on the same
scene). When the obs / action dim differs, partial state_dict load:
overlapping shapes copied, new entries **zero-init** (not random — see
"Cross-phase lessons" at the bottom).

---

## Reward shaping

The reward landscape is the central design problem. Below is the final
working set, in order of weight magnitude.

| Term | Final weight | Notes |
|---|---|---|
| `target_axis_alignment` | +100 | `exp(-α·(1-cos)²)` of object-axis-to-target. α annealed 0.5 → 4.0 over 300 iters. |
| `target_axis_progress` | +300 | Δ(alignment cos) per step, clamped non-negative. Dense gradient toward rotating in the right direction. |
| `lift_height` (or skip-lift) | +80 | Clamped (object_z − settle_z) up to `lift_target`. Set to 0 weight in skip-lift mode. |
| `contact_min` | +15 | Min over the 3 fingertip contacts. Reduced from 30 (lift recipe) to allow regrip during rotation. |
| `contact_mean` | +10 | Mean of the 3 fingertip contacts. Encourages overall grip stability. |
| `fingertip_to_object` | −3 | Penalty per metre of distance. Keeps fingers near cylinder. |
| `joint_pos_limits` | −2 | Standard safety. |
| `finger_drift_from_grip` | −0.3 | Reduced from −2 (lift recipe) so fingers can articulate enough to roll the cylinder. |
| `object_xy_drift` | −3 | Standard stability. |
| `object_orientation_drift` | **0** | **Must be 0**: any non-zero penalty actively trains against rotation. Found by v2 diagnosis. |
| `action_rate_l2` | −0.005 → −0.1 | Bumped 20× for Policy B to suppress sim-only finger jitter. |
| `object_ang_acc_l2` | (new) −0.05 | New: L2 of cylinder angular-velocity Δ per step. Penalizes high-frequency vibration of the rotation. |

### Curricula

- **`target_axis_alpha_curriculum_iters=300`** anneals the alignment
  reward's `α` from `target_axis_alpha_start=0.5` (soft, wide basin —
  gradient at large tilts) to `target_axis_alpha=4.0` (sharp, focused
  on near-target). Soft start gives gradient when cos ≈ 0 (flat);
  sharp end focuses on the home stretch.
- **`tracking_anneal_iters=0`** (disabled). Tracking-from-CEM rewards
  are off — they were tuned for the lift-only behavior and conflict
  with rotation.

### Terminations

| Term | Role |
|---|---|
| `tip_lost` (10-step consecutive grace) | Real drop detection. Kept across all runs. |
| `object_drop` (z < spawn_z − 2 cm) | Belt-and-braces drop check. |
| `object_floor_proximity` (z < `object_min_z`) | **New v5+**: forbid floor-bracing — terminate if object center falls below 0.05 m world z during reorient phase. |
| `finger_slip` (drift > 0.3 rad from CEM grip) | **Disabled** for reorient (`100.0` rad). Fingers need to articulate. |
| `object_orientation_slip` (drift > 0.5 rad ≈ 28°) | **Disabled** for reorient (10.0 rad). Goal IS 90° rotation. |
| `object_slip_xy` (xy drift > 1.5 cm) | **Disabled** for reorient (0.5 m). Cylinder translates during regrip. |

---

## Phases

### Phase v1 (palm rotation, abandoned)

**Goal:** test whether adding 3 palm rotation residuals (rx/ry/rz) on
top of the scripted palm motion makes the task tractable.

**Setup:** 12-dim action (9 finger + 3 palm rot), `target_axis_alignment`
reward weight 50, `palm_rotation_residual_scale=0.3` (~17°),
`palm_rotation_active_from_sim_step=240`.

**Result:** NaN at training iter ~30 with 1024+ envs. Diagnosis: random
wrist rotations during a stable grip lever the cylinder into degenerate
physics. Stable values found: `scale=0.1`, activate@sim_step 500. Smoke
test passed but the user rejected the approach: *"the palm should not
move beyond the vertical scripted motion. all the reorientation should
be done with the 9 finger control dofs."*

**Takeaway:** finger-only is the constraint. All v2+ runs are 9-dim.

---

### Phase v2 (finger-only baseline, plateaued silently)

**Goal:** with palm fixed, find a reorientation strategy using finger
residuals alone.

**Setup:**
- 9-dim action, residual scale 0.5
- `target_axis_alignment` weight 100, `α=4`
- `contact_min` weight **30** (lift recipe value, kept high to enforce grip)
- `object_orientation_drift_weight=−20` (also a lift recipe holdover)
- `episode_length_s=2.6`, `reorient_start_step=50`
- Warmstart from medium_flat.

**Result:** policy plateaued. `target_axis_progress` ≈ 0 throughout.
The cylinder lifted and held but never rotated. Visually: a tight
tripod grip, frozen.

**Takeaway:** `object_orientation_drift_weight=−20` was actively
penalizing rotation — the very thing we wanted. **Any "stability"
reward designed for the lift task fights against the reorient task.**
Disabled in v3+.

---

### Phase v3 (rewards added, episodes died at step 40)

**Goal:** harder push for rotation: dense progress reward, alpha
curriculum, plus a `low_tilt_velocity` termination to kill stationary
local optima.

**Setup (delta from v2):**
- `object_orientation_drift_weight=0` (fixed v2's bug)
- `contact_min_weight=15` (relaxed from 30)
- `target_axis_progress_weight=300` (new dense gradient)
- `target_axis_alpha_curriculum_iters=300`, `alpha_start=0.5` (soft → sharp)
- `terminate_low_tilt_velocity=True` with `window=20`, `min_progress=0.05`
- `episode_length_s=4.0`, `reorient_start_step=50`

**Result:** target_axis reward stayed at **0.0 for all 2034 iters**.
Mean episode length = 40 (vs the 160-step `episode_length_s=4.0` cap).
Diagnosis from the termination breakdown:

| Termination | Per-iter count |
|---|---|
| `finger_slip` (drift > 0.3 rad) | **51** ← dominant |
| `object_orientation_slip` (drift > 0.5 rad ≈ 28°) | 3.2 |
| `object_slip_xy` (drift > 1.5 cm) | 1.7 |
| `tip_lost` | 0 |
| `low_tilt_velocity` | 0 ← never got a chance |

The lift-task stability terminations were killing every episode by
step 40 — before `reorient_start_step=50` could even fire the reward.
The policy literally never saw the reorient gradient.

**Takeaway:** **lift-task terminations are fundamentally hostile to
the reorient task.** Three constraints have to be lifted:
- `finger_slip` (fingers must articulate to roll cylinder)
- `object_orientation_slip` (we *want* 90° rotation)
- `object_slip_xy` (cylinder translates during regrip)

The reorient reward gate must fire *before* mean episode death, not
after.

---

### Phase v4 (terminations disabled, floor-bracing emerges)

**Goal:** unblock the v3 mechanism failure. Allow articulation +
rotation + translation.

**Setup (delta from v3):**
- `term_finger_slip=100.0` (effectively disabled)
- `term_object_slip_yaw=10.0` (disabled)
- `term_object_slip_xy=0.5` (disabled)
- `reorient_start_step=35` (pulled forward; fires post-lift, pre-death)
- `episode_length_s=8.0` (longer; 200 policy steps)
- `finger_drift_weight=−0.3` (relaxed from −2.0)
- `init_noise_std=0.15` (recover from v3's std-collapse to 0.05)
- Drop `low_tilt_velocity` (was never firing in v3)

**Result:** mechanism unblocked. Episode length 40 → **150**.
`target_axis_progress` flipped from 0 to **+0.125 at iter 941 peak**
(first positive in any run). Plateaued thereafter.

**Final iter (2033/2034) metrics:**

| Metric | Value |
|---|---|
| Mean reward | 318 |
| Mean episode length | 149 |
| `target_axis_alignment` (Σ episode) | 27.1 (peak was 32.0 at iter 941) |
| `target_axis_progress` (Σ episode) | +0.029 (peak +0.125) |
| `tip_lost` | 12.8/iter |
| `episode_success` | **0** ← never reached "lifted + vertical" |
| Mean action std | 0.09 (collapsed from 0.15 → exploration limited) |

**The headline finding — floor-bracing.**

Visual inspection of [v4_peak_floorbracing.mp4](videos/reorient/v4_peak_floorbracing.mp4)
and [v4_final_floorbracing.mp4](videos/reorient/v4_final_floorbracing.mp4)
revealed that the policy was **using the floor as an external pivot to
help roll the cylinder.** The cylinder is barely lifted
(object_height ≈ 0.045 m) and the cylinder's long axis (8 cm) means its
end can easily reach the floor. As the hand tilts the cylinder, the
distal end braces against the ground and uses ground reaction force to
assist the rotation.

This is a creative-but-not-in-hand solution that wasn't blocked by the
reward function: the reorient_start_step gate let target_axis rewards
fire as soon as the lift completed, with no requirement that the
cylinder actually stay clear of the floor.

**Takeaway:** RL exploits any available contact. To force *true* in-hand
reorient, we need: (a) higher lift target to put the cylinder clearly
above floor reach, and (b) a no-floor-contact termination during the
reorient phase.

---

### Phase v5 (forced lift to clearance + floor-proximity termination)

**Goal:** kill the floor-bracing strategy. Force the policy to find a
finger-only solution.

**Setup (delta from v4):**
- `lift_target_z_above_init=0.10` (up from 0.05)
- `lift_delta_z=0.10` (scripted palm raises higher — *both* knobs must
  match or the policy gets an unreachable goal)
- New termination: `terminate_object_floor_proximity` — terminate during
  reorient phase if object center z < 0.05 m. For an 8 cm cylinder at
  any orientation, 0.05 m world z gives ~1 cm worst-case clearance.
- `floor_proximity_phase_start_step=reorient_start_step=35`

**Smoke test** (10M timesteps): mechanism worked.
`floor_proximity` terminations dropped 13.4 → 4.0/iter as the policy
learned the higher lift. `object_height` reached 0.116 m (above target).

**Full run** (100M timesteps, 2:00 wall): **the rotation strategy
collapsed.**

| Metric | v4 iter 941 (peak) | v5 iter 935 (same point) |
|---|---|---|
| object_height | 0.042 m | 0.101 m ✅ (constraint satisfied) |
| floor_proximity terms | n/a | 3.6 ✅ |
| **target_axis_progress** | **+0.125** | **−0.112** ❌ flipped negative |
| target_axis_alignment | 32.0 | 9.6 |
| Mean reward | 318 | 133 |
| Action std | 0.10 | 0.09 (converging to "hold lift, don't rotate") |

The v4 floor-bracing wasn't a bug to fix — it was the policy correctly
identifying ground contact as the only viable rotation aid given the
combination of (smooth fingertips, smooth cylinder, finger-only,
warmstart bias toward "hold steady"). With that affordance removed,
the policy reverts to "do the lift, don't risk rotation."

**Takeaway:** the issue isn't morphology (9 DOF is enough). It's
*control / representation*:
1. **Warmstart prior:** the medium_flat policy was trained to *hold
   the cylinder still* after lift (`track_object_quat` weight 8). The
   middle MLP layers encode "no rotation" as the prior; even with
   target_axis rewards, fighting that prior takes ~thousands of iters.
2. **Residual scale clamps coordination:** `finger_residual_scale=0.5`
   keeps fingers within ±0.5 rad of the CEM tripod grip. Coordinated
   rolling needs *bigger* finger excursions — one finger opens while
   another closes.
3. **Action std collapse:** without positive reward signal, PPO
   shrinks std → less exploration → can't discover the pattern.
4. **Credit assignment over 24-step rollouts:** sustained
   coordination needs a reward signal that links across 50+ steps.

The cleaner architectural answer: **two policies.**

---

### Phase Policy B (two-policy split, the breakthrough)

**Goal:** train a focused phase-2 policy on a state distribution where
the cylinder is *already* lifted+gripped. Remove the
warmstart-fights-rotation problem at the root.

**Architecture:** keep `medium_flat_stable_v1` as **Policy A** (lift to
horizontal stable — already trained). Train **Policy B** in a new env
mode (`--skip-lift-phase`) where:

- Cylinder spawns at the post-lift pose (z = spawn_z + lift_delta_z +
  5 mm drop offset)
- Fingers spawn at the CEM grip pose (`finger_default_ctrl`)
- Palm spawns at the lifted z
- `settle_steps=0`, `lift_ramp_steps=1` — the lift is a no-op
- The 5 mm drop offset is critical: with fingers spawned exactly at
  the position-controller setpoint, equilibrium force = 0 → no grip.
  A 5 mm drop pushes the cylinder slightly into the grip, generating
  the contact force CEM had at lift-end.

**Reward changes:**
- `lift_target_z_above_init=0` (lift task already "done" at spawn)
- All target_axis rewards fire from step 0 (the lift is already there)
- `lift_phase_start_step=10` (10-step grace for grip to settle before
  terminations engage)
- `init_noise_std=0.05` (low — the warmstart's grip behavior dominates
  the first iters)
- `action_rate_weight=−0.1` (20× normal — suppress jitter)
- `object_ang_acc_weight=−0.05` (new — L2 of cylinder Δω/step)
- Episode length: 4 s (no lift phase needed, so shorter is fine)

**Smoke results** (5M timesteps, 9 min):

| Metric | Smoke iter 0 | Smoke iter 100 |
|---|---|---|
| `target_axis_progress` | +0.027 | **+0.463** (3.7× v4 peak) |
| `target_axis_alignment` | 1.07 | **27.7** |
| object_height | 0.030 (initial drop) | 0.114 (stable) |
| Mean episode length | 21 | 73 |
| contact_min | 0.11 | 4.06 |

The skip-lift design fixed the credit-assignment problem in 9 min of
training. But the rotation was visibly *jittery* — small high-frequency
finger adjustments doing the work, which would not survive sim-to-real
transfer. Smoothness penalties (action_rate, object_ang_acc) added
before the full run.

**Full Policy B v1 results** (100M timesteps, 2:59 wall):

| Metric | Policy B v1 final | v4 full (best, no skip-lift) |
|---|---|---|
| **Train time** | **2h59m27s** | 2h02m32s |
| Mean reward | **402.74** | 318 |
| Mean episode length | **182.45** | 170 |
| `target_axis_alignment` | **87.15** (2.7× v4) | 32.0 (peak) |
| `target_axis_progress` | **+0.253** (2× v4) | +0.125 (peak) |
| `contact_min` | 11.03 | 4.14 |
| `tip_lost`/iter | **1.17** (90% drop from smoke) | 10.5 |
| `floor_proximity`/iter | 0.04 | n/a |
| `object_ang_acc_l2` (Σ) | −2.53 | n/a |
| `action_rate_l2` (Σ) | −0.134 | −0.001 |
| **`time_out` is the dominant terminator** | 8.96/iter | — |

**Takeaway:**
- Two-policy split solves the warmstart-bias problem at the
  architectural level (Policy B never has to learn "hold still"
  while learning "rotate"). 2× v4's peak rotation rate, sustained.
- Grip is now *rock solid*: tip_lost dropped from v4's 10.5 → 1.17.
  Episodes run to time_out, not termination.
- **Sim-to-real concern still present.** Despite the smoothness
  penalties, the raw `object_ang_acc_l2` (~0.28 |Δω|² per step,
  ~15 rad/s² per axis) and `action_rate_l2` (~2× v4's per-step rate)
  show the policy is still using high-frequency finger motion. The
  smoothness weights were outvoted by the +100 alignment reward —
  policy traded jitter for rotation. To get *both* smooth motion and
  good rotation, the smoothness weights need to be 5–10× higher and
  retrained (at the cost of slower rotation rate).

### Phase Policy B v2 — Stage 1 (smoothness ramp + signed progress)

The v1 takeaway above predicted the experiment: dial the smoothness
weights 5–10× higher and retrain, expecting a smoothness↔rotation
tradeoff. Stage 1 finetunes Policy B v1 (warmstart from `model_2033.pt`)
with **only** two changes — no "quick" mechanisms yet:

1. **Smoothness ramp** (`action_rate_l2` + `object_ang_acc_l2`), held at
   the v1 base for the first 200 iters, then ramped over 400 iters to a
   swept final target: **5×** (`-0.5 / -0.25`) or **10×** (`-1.0 / -0.5`).
2. **Signed `target_axis_progress`** (no negative clamp) — slipping back
   toward flat is now penalized symmetrically, to fix v1's slip-back.

Both runs: 30M ts, 1024 envs, 1220 iters. Metrics below are the converged
FINAL block (well past the 600-iter ramp end). Smoothness reward terms are
`weight × raw_jitter`, so the comparable quantity is **raw jitter** =
`reward / |weight|` (shown in parentheses):

| Metric | Policy B v1 | **v2 5× (winner)** | v2 10× |
|---|---|---|---|
| Mean reward | 402.7 | 369.2 | 229.2 |
| `target_axis_alignment` | 87.2 | **80.0** | 43.3 |
| `target_axis_progress` | +0.253 | **+0.499** | +0.135 |
| `tip_lost`/iter | 1.17 | **0.96–1.6** | 4.4–5.3 |
| `object_drop`/iter | — | 0.17 | 0.17–0.42 |
| `action_rate_l2` (Σ) | −0.134 (raw ≈1.3) | −0.616 (**raw ≈1.23**) | −0.783 (raw ≈0.78) |
| `object_ang_acc_l2` (Σ) | −2.53 (raw ≈50) | −6.12 (**raw ≈24.5**) | −2.45 (raw ≈4.9) |
| min object-center z (eval) | — | **0.111 m** | 0.114 m |

Verification videos (single deterministic 4 s rollout from `model_1219.pt`,
rendered via `scripts/rl_render_reorient.py` — the phase-B half of the
handoff demo, driven from each run's own `config.yaml`):

- [policyB_v2_smooth5x.mp4](videos/reorient/policyB_v2_smooth5x.mp4) — **chosen**
- [policyB_v2_smooth10x.mp4](videos/reorient/policyB_v2_smooth10x.mp4)

Both keep the object aloft the whole rollout (min center-z 0.111 / 0.114 m,
well above the 0.05 m floor-proximity threshold) — genuine in-hand
reorientation, no floor-bracing.

**Takeaway — 5× is the clear winner:**
- **5× halves the object angular jerk** (raw `object_ang_acc_l2` 50 → 24.5)
  while *holding* rotation (alignment 80 vs v1's 87), *improving* progress
  (+0.50 vs +0.25 — the signed penalty killing the slip-back), and
  *preserving* the grip (tip_lost ~1.0, ≈ v1's 1.17). This is the
  smooth-AND-rotating policy v1 said would require 5–10× weights.
- **10× over-penalized: the policy stopped rotating to dodge the penalty.**
  Its raw jerk is lowest (4.9) but only because alignment collapsed to 43,
  progress to +0.14, and the grip degraded (tip_lost 4.4 and *rising* at
  the end). "Smooth because it gave up" is not the goal.
- Lower mean reward for 5× (369 vs 402) is expected and not a regression:
  the larger smoothness penalties simply subtract more from the same
  behavior; the task metrics (alignment / progress / tip_lost) are as-good
  or better.
- **→ Stage 2 warmstarts from 5× `model_1219.pt`.**

### Phase Policy B v2 — Stage 2 (warmstart + "quick" mechanisms)

Stage 2 warmstarts the Stage-1 **5× winner** (`20260601-2310-policyB_v2_smooth5x/model_1219.pt`)
and layers on the three "quick / shorter-trajectory" mechanisms, to test whether
reorientation can be made *faster* without losing the Stage-1 smoothness or grip:

1. **`alignment_success` termination + one-shot bonus** — once axis alignment
   crosses the threshold the episode ends and pays a bonus (success-bonus 30).
2. **Per-step `reorient_time_cost`** (−0.02/step) — a standing pressure to finish.
3. **`alignment_speed_bonus`** (15, thresh 0.9) — rewards crossing the line *early*.

To protect the warmstart's hard-won grip, the smoothness ramp **base** was anchored
at the warmstart level (`−0.5 / −0.25`) rather than v1's base, and the quick rewards
were softened from their first (tip_lost-spiking) values. Two runs swept the final
smoothness target: **5×** (flat `−0.5 / −0.25`) and **10×** (ramp to `−1.0 / −0.5`).
Both: 30M ts, 1024 envs, 1220 iters, warmstart from the same 5× ckpt. (`policyB_v2_5x.log`,
`policyB_v2_10x.log` were reused for these Stage-2 runs; the Stage-1 numbers below are
preserved from `STAGE1_RESULTS.txt`.)

Comparable raw jitter = `reward / |weight|` (in parens). Final converged block:

| Metric | v1 | v2 s1-5× (Stage-1 best) | v2 s2-5×-quick | **v2 s2-10×-quick (recommended)** |
|---|---|---|---|---|
| Mean reward | 402.7 | 369.2 | 300.9 | 360.6 |
| Mean episode length (/200) | ~188 | ~173 | **155** | 188 |
| `target_axis_alignment` | 87.2 | 80.0 | 64.4 | **75.2** |
| `target_axis_progress` | +0.253 | **+0.499** | +0.009 | +0.331 |
| `tip_lost`/iter | 1.17 | 0.96–1.6 | 2.58 | **0.63** (best) |
| `object_drop` term/iter | — | 0.17 | 0.04 | 0.21 |
| `action_rate_l2` (Σ → raw) | −0.134 (1.3) | −0.616 (1.23) | −0.778 (**1.56**) | −1.147 (**1.15**, smoothest) |
| `object_ang_acc_l2` (Σ → raw) | −2.53 (50) | −6.12 (24.5) | −3.37 (13.5*) | −4.52 (**9.0**, smoothest) |
| `alignment_success` term/iter | n/a | n/a | **1.00** | 0.17 |
| quick bonuses (succ / speed) | — | — | 0.022 / 0.024 | 0.004 / 0.005 |
| min object-center z (eval) | — | 0.111 | 0.115 | **0.120** (highest) |

\*5×-quick's object jerk looks low only because the object barely rotates (progress ≈ 0).

Verification videos (single deterministic 4 s rollout from `model_1219.pt`, via
`scripts/rl_render_reorient.py`):

- [policyB_v2_smooth10x_quick.mp4](videos/reorient/policyB_v2_smooth10x_quick.mp4) — **recommended final Policy B v2**
- [policyB_v2_smooth5x_quick.mp4](videos/reorient/policyB_v2_smooth5x_quick.mp4) — the "quick" variant (regressed; see below)

Both stay aloft the whole rollout (min center-z 0.120 / 0.115 m ≫ the 0.05 m
floor-proximity threshold) — genuine in-hand reorientation, no floor-bracing.

**Takeaway — the "quick" mechanisms did NOT deliver a clean win; the best policy
came out of Stage 2 almost incidentally:**

- **The quick mechanisms only fire at 5× — and there they backfire.** s2-5×-quick *is*
  ~10% shorter (ep-len 173 → 155, driven by `alignment_success` terminations at 1.0/iter),
  but it bought that speed badly: hand actions got **jerkier** under the *same* smoothness
  weights (raw `action_rate` 1.23 → 1.56), grip **degraded** (tip_lost 1.6 → 2.58), and
  `target_axis_progress` **collapsed to ≈ 0** while `alignment_success` terminations
  spiked. That signature — barely-positive progress + many success terminations — is
  **threshold-gaming**: the policy learned to snap just past the 0.9 alignment line to
  collect the bonus and end the episode, not to reorient robustly. Not shippable.
- **At 10× the quick terms are near-inert** (success/speed bonuses 0.004/0.005, episodes
  run to ~188 → time_out): the stronger smoothness penalty suppresses the fast snapping.
  The real win at 10× is that **warmstarting the good 5× ckpt + gently ramping to 10×
  rescued the 10× regime that collapsed in Stage 1** (Stage-1 10× from scratch: align 43,
  tip_lost 4.4). s2-10×-quick is the **smoothest** policy on both axes (raw `action_rate`
  1.15, raw `object_ang_acc` 9.0), has the **best grip** (tip_lost 0.63), **holds vertical
  highest** (min-z 0.120), and **still rotates well** (align 75, progress +0.33).
- **Recommended final Policy B v2 = s2-10×-quick**
  (`results/rl/20260602-0024-policyB_v2_smooth10x_quick/tensorboard/model_1219.pt`).
  It is the best smooth-AND-grippy-AND-holds-vertical policy in the whole sweep. Caveat:
  it is **not** actually "quicker" than v1, and its quality edge does **not** come from the
  quick mechanisms — those should be removed or redesigned (a no-terminate success bonus,
  to avoid threshold-gaming). If sustained **rotation quality** matters more than grip,
  Stage-1 5× (align 80, progress +0.50) remains a strong alternative.
- **"Quick" remains an open problem.** Episode-termination on success conflates "finished
  fast" with "failed fast" (s1-10× had the shortest episodes purely because it dropped the
  object), and the success-bonus invites threshold-gaming. Next attempt should decouple
  speed shaping from termination.

### Phase Policy B v2 — observations & open issues (→ motivate Phase 3)

Two behaviors surfaced on review of the v2 videos that the metrics didn't flag, plus the
engineering notes from this pass:

1. **De-centering / large lateral translation (undesirable, new in v2).** All v2 runs
   reorient by *sliding the cylinder sideways* as much as rotating it: the index/middle
   MCP+PIP flex strongly *inward* while the thumb MCP+PIP push *outward*, so the cylinder
   both tilts up **and** translates a long way to one side. Earlier policies (v1) rotated
   with minimal lateral shift. The existing `object_xy_drift` penalty (−3.0) is clearly
   being out-voted by the alignment/progress rewards under the larger smoothness penalties.
   **Fix candidates:** raise `object_xy_drift` weight and/or switch it to a tighter
   (e.g. quadratic past a deadband) penalty; consider penalizing object-center translation
   *in the palm frame* specifically; possibly penalize asymmetric finger excursion directly.
2. **Handoff "teleport" (A→B discontinuity).** In the handoff demo the cylinder appears to
   jump straight to a half-reoriented pose at the A→B switch — because the demo **resets the
   env** to load Policy B (skip-lift spawn), rather than continuing the same physics state.
   We never see B start from a *stable horizontal* grip. **Fix:** load BOTH policies into one
   env and gate which one's actions are applied by phase (A until lift+settle done, then B),
   with **no reset** — a single continuous rollout. This also stress-tests that B's spawn
   distribution actually matches A's terminal state (any residual jump = real distribution
   mismatch to close).
3. **Engineering (this pass).** The reorient pipeline had been lost from git (uncommitted)
   and was recovered from dangling blobs; **commit experiment code promptly.** Parallel
   training on one GPU works only with a per-process Warp kernel-cache
   (`WARP_CACHE_PATH=$(mktemp -d)`) — a shared cache races and NaNs. See
   `scripts/queue_reorient_smooth_sweep.sh` and `scripts/overnight_autonomy.sh`.

### Phase Policy B v2.1 — de-centering fix (workstream A) ✅

Resolves observation 1. Added `object_lateral_drift`: the object's **palm-frame** xy
displacement from spawn, penalised past a 1 cm deadband, **quadratic** beyond,
contact-gated (the deadband leaves the small regrip translations rotation needs free).
Finetuned the recommended s2-10×-quick policy (smoothness held flat at the converged
10× level, no alpha re-anneal) and swept the penalty weight −15 vs −40 (20 M ts each).

| Metric | warmstart (s2-10×-quick) | −15 | **−40 (winner)** |
|---|---|---|---|
| `object_xy_drift` (→ drift) | −0.174 (≈0.058 m) | −0.099 | **−0.112 (≈0.037 m, −36%)** |
| `target_axis_alignment` | 75.2 | 45.9 | **76.6** (preserved) |
| `target_axis_progress` | +0.331 | +1.03 | +0.348 |
| `tip_lost`/iter | 0.63 | 7.5 | **0.25** (best grip) |

**Takeaway — the *stronger* penalty (−40) is the clear winner**, and counterintuitively
gives *better* grip and alignment than −15: the strong centering pressure steers the
policy into a better-centered, more stable optimum, whereas −15 got stuck (tip_lost 7.5,
alignment 46). −40 cuts lateral drift ~36% while *preserving* rotation (align 76 ≈ 75) and
*improving* grip (tip_lost 0.25 vs 0.63). Winner ckpt:
`results/rl/20260602-1323-policyB_v21_decenter_w40.0/tensorboard/model_812.pt`. This is the
new recommended reorient policy and the warmstart for Phase 3.

### Continuous A→B handoff (workstream B) — mechanism ✅, distribution gap ⚠

Resolves observation 2's *mechanism*. `scripts/rl_demo_handoff_continuous.py` runs ONE env
with **no reset**: it builds the env in B-mode (66-dim) and feeds Policy A `obs[:, :65]`
(A is 65-dim; B's extra `target_axis_misalign` obs is appended last) through a real lift to
0.10 m, then switches to Policy B's actions on the same physics state — no teleport.

**Finding:** the handoff is seamless but **Policy B drops the cylinder shortly after the
switch** (z = 0.097 at handoff, object gone by ~step 75). This is a *real* train/deploy
distribution gap: B was trained spawning at a precise skip-lift pose (with a 5 mm
drop-into-grip establishing contact force), so A's genuinely-lifted state is off-distribution.
**Fix (future):** train B's successor in the normal-lift env (reorient rewards gated after
the real lift) so it experiences A's actual terminal state.

### Phase 3 (in progress): bracing.

Promote bracing the cylinder's lower end flat against the palm (power/pinch grip, consistent
with real screwdriver use), co-trained with reorientation in the skip-lift env (warmstart the
v2.1 de-centering winner). Sensing added: a `palm_cube_contact` sensor (palm_pose body) and
fingertip **force** readout. Rewards (gated on alignment ≥ 0.7, so reorient-then-brace):
`grip_force` (pinch-to-power), `palm_brace_force` (palm normal force), and — critically —
`palm_brace_distance`, a **dense** exp(−gap/scale) shaping term.

**Why dense shaping is required (diagnostic):** in the de-centering winner's rollout the
cylinder reorients to cos 0.90 but its nearer end **never gets within 7.7 cm of the palm
plate** (palm contact found = 0). The gripped cylinder simply sits ~8 cm below the palm, so
the sparse force reward can never fire — bracing is undiscoverable without a gradient pulling
the end toward the palm.

**Result — NEGATIVE: bracing is not reachable with fingers-only control + this grip.** A
sweep (dense-distance weight 8 vs 20, warmstart the v2.1 winner) was run to ~330 iters and
killed. The dense shaping barely moved the end (min end→palm distance 7.7 → ~7.1 cm), palm
contact force stayed **0** the whole time, and the brace objective *degraded* reorientation
(alignment 76 → 18–33). Root cause is geometric, not reward-tuning: the CEM grasp holds the
cylinder ~7–8 cm below the palm plate, and ±0.5 rad finger residuals cannot translate it that
far up to the palm. The sensing + reward scaffolding (`palm_cube_contact`, `grip_force`,
`palm_brace_force`, `palm_brace_distance`) is in place and correct — the task as posed is just
out of the morphology's reach.

**To actually enable bracing, the geometry must change** (one of): (a) re-grip the cylinder
**higher**, near the palm, via a different CEM grasp / keyframe, so bracing is a small
adjustment; (b) allow limited palm motion during a brace phase (relaxes the fingers-only
constraint); (c) brace against the finger structure instead of the top palm plate. This needs
a grasp-design decision upstream of RL; deferred pending that choice.

---

## Results

### Cross-run comparison plots

Two figures (the historical journey got too crowded once the v2 sweep landed,
so it's split):

**The journey — v2 → v3 → v4 → v5 → Policy B v1:**

![comparison](img/reorient_comparison.png)

**The v2 finetune zoom — Policy B v1 → Stage-1 (5×/10× smooth) → Stage-2 (quick):**

![comparison v2](img/reorient_comparison_v2.png)

Each is a 12-panel comparison (mean reward, episode length, alignment / progress,
object_height, contact_min, action_rate + angular-acceleration smoothness, a
termination panel — floor_proximity for the journey, alignment_success for v2 —
action std, value loss). Regenerate with `uv run python scripts/rl_plot_reorient.py`
(`--only v2` to refresh just the zoom; append new v2+ runs to `V2_RUNS`, leave the
historical figure frozen).

### Headline videos

- [handoff_demo.mp4](videos/reorient/handoff_demo.mp4) — **The
  end-to-end handoff demo.** Policy A picks up the flat-laying
  cylinder and holds it stable (3 s); Policy B then reorients it
  toward vertical (4 s). Both policies running live in simulation,
  concatenated in one rollout. Generated by
  [scripts/rl_demo_handoff.py](../../scripts/rl_demo_handoff.py).
- [reorient_comparison_grid.mp4](videos/reorient/reorient_comparison_grid.mp4)
  — **2×2 side-by-side comparison.** Top-left: v3 (reward never fires).
  Top-right: v4 floor-bracing. Bottom-left: v5 (no floor = no rotation).
  Bottom-right: Policy A → Policy B concatenated (the two-policy
  solution: pickup, then in-hand rotate). One-screen overview of the
  whole journey.
- [v4_peak_floorbracing.mp4](videos/reorient/v4_peak_floorbracing.mp4)
  — v4 at iter ~950 (peak target_axis_progress). The cylinder is rolled
  while its distal end braces against the floor. **This is the
  "floor-bracing" behavior** — RL found that the ground reaction force
  helps with rotation when fingers alone can't.
- [v4_final_floorbracing.mp4](videos/reorient/v4_final_floorbracing.mp4)
  — v4 at iter ~2000 (final). Same strategy, slightly regressed from
  peak.
- [v5_final.mp4](videos/reorient/v5_final.mp4) — v5 final. Floor
  contact forbidden; policy collapses to "hold the high lift, don't
  rotate." Showcases the no-floor-no-rotation result.
- [policyA_lift.mp4](videos/reorient/policyA_lift.mp4) — `medium_flat_stable_v1`
  rollout: cylinder lying flat on floor → lifted and held horizontal.
  The "Policy A" half of the two-policy chain.
- [policyB_final.mp4](videos/reorient/policyB_final.mp4) — Policy B v1
  final. True in-hand reorientation from a pre-lifted spawn. Visibly
  jittery (sim-only exploit) but the cylinder genuinely rotates
  without floor contact.
- [v3_final.mp4](videos/reorient/v3_final.mp4) — v3 final. Episodes die
  at step ~40 (before reorient reward gate at step 50); cylinder barely
  moves. Stuck-in-lift-phase artifact of hostile terminations.

### Scoreboard

| Run | Wall | target_axis_align Σ | target_axis_progress Σ | object_height | tip_lost/iter | Headline |
|---|---|---|---|---|---|---|
| v2 | n/a | ~0 | 0 | lifted, no rotation | 0 | `orientation_drift=−20` penalized rotation |
| v3 | ~2h | 0 (all 2034 iters) | 0 | n/a | 0 | episodes died at step 40, reward never fired |
| **v4** | 2h02 | 27.1 (peak 32) | +0.029 (peak +0.125) | 0.045 | 12.8 | **floor-bracing emerges** |
| v5 | ~2h | 9.6 (peak ~22) | −0.112 | 0.101 | 15 | no floor → no rotation |
| **Policy B v1** | **2h59** | **87.15** (3.2× v4) | **+0.253** (2× v4) | 0.114 | **1.17** | **two-policy split unlocks it** |
| v2 s1-5× | ~1h (ft) | 80.0 | **+0.499** | 0.111 | ~1.0 | **smoothness ramp halves jerk, holds rotation** |
| v2 s1-10× | ~1h (ft) | 43.3 | +0.135 | 0.114 | 4.4 | over-penalized → stopped rotating |
| v2 s2-5×-quick | ~1h (ft) | 64.4 | +0.009 | 0.115 | 2.58 | quick mechanisms → threshold-gaming |
| **v2 s2-10×-quick** | ~1h (ft) | 75.2 | +0.331 | **0.120** | **0.63** | **recommended: smoothest + grippiest + holds vertical** |

(ft = finetune from a warmstart; ~1h each at 1024 envs / 30M ts.)

---

## Cross-phase lessons (the load-bearing ones)

In rough order of how much they would cost to relearn:

1. **Lift-task terminations are hostile to reorient.** `finger_slip`
   (drift > 0.3 rad), `object_orientation_slip` (drift > 0.5 rad ≈ 28°),
   and `object_slip_xy` (drift > 1.5 cm) all need to be disabled for a
   rotation task. The remaining safety check is `tip_lost` (10-step
   consecutive grace).
2. **The reorient reward gate must fire BEFORE mean episode death.**
   v3 had `reorient_start_step=50` but episodes died at step 40 →
   reward was 0.0 for 2034 iters straight. Set the gate to just after
   lift completes (~35 with default `lift_phase_start_step=40`).
3. **`object_orientation_drift_weight` must be 0** for any reorient
   task. Any non-zero value penalizes the very behavior we want.
4. **RL exploits floor contact when available.** A "lifted" cylinder
   that's only 4.5 cm off the floor can still rotate using ground
   pivot. To force true in-hand: higher lift target (10 cm) + explicit
   `object_floor_proximity` termination during rotation phase.
5. **Warmstart bias compounds over training.** The medium_flat policy
   was trained for "hold still after lift"; even with zeroed
   orientation_drift_weight, the middle MLP layers encode that prior
   and PPO can't easily override it. The fix is architectural: train
   a **separate** policy on a focused state distribution.
6. **Two-policy split fixes credit assignment.** Phase A (lift, already
   trained) → Phase B (rotate from lifted state). Each policy has
   one objective and a clean reward landscape.
7. **Partial warmstart must zero-init new dims** (not random). Random
   init of new input cols lets random hidden activations propagate
   through trained layers → NaN within 1–2 iters. Implemented in
   `rl_train_cube.py` partial state_dict load.
8. **Action std collapse signals lost gradient.** v3 collapsed to
   0.05 because the only reward it saw was the lift baseline. The
   policy pinned to the warmstart's local optimum with no exploration
   to escape. Mitigations: higher `init_noise_std` (v4 used 0.15),
   add proper reward signal so PPO learns to keep std up.
9. **Skip-lift mode needs a small drop offset.** Spawning the cylinder
   exactly at the position-controller setpoint gives zero equilibrium
   force → grip immediately fails. A 5 mm above-target drop offset
   makes the cylinder fall into the grip and establish contact force.
10. **Sim-only rotation exploit: high-frequency finger jitter.** Even
    with `action_rate_l2 = −0.1` (20× normal), Policy B's policy uses
    significant action jitter to manipulate the cylinder. For sim-to-real
    transfer, smoothness weights need to be 5–10× higher; expect a
    rotation-rate tradeoff.

---

## Open questions / next steps

In priority order, if revisiting:

1. **Policy B + stronger smoothness penalties.** Bump
   `--action-rate-weight=−0.5`, `--object-ang-acc-weight=−0.2`,
   retrain Policy B 100M. Test whether smooth rotation is achievable
   at all on this morphology.
2. **Runtime policy handoff.** Currently Policy A and Policy B are
   trained separately but no runtime switching wrapper exists. Need
   to write a thin controller that detects "lift complete + stable
   horizontal" and swaps in Policy B. The handoff state's distribution
   mismatch (Policy A's terminal state vs Policy B's spawn distribution)
   may require some jitter on B's spawn.
3. **Larger network from scratch.** 512→512→256 actor, no warmstart,
   200M steps with the skip-lift env. Tests whether network capacity
   was a secondary constraint.
4. **Target-axis curriculum (anneal goal, not basin).** Start with a
   30° tilt target, ramp to 90°. Different from the current
   `target_axis_alpha_curriculum` (which shapes basin *width*, not
   *goal*). New knob would anneal `target_axis_world`.
5. **Commit-bonus reward.** Sharp threshold reward at `cos_theta > 0.5`
   to escape any "partial rotation local optimum" — would push the
   policy past the half-way point toward full vertical.
6. **Floor-bracing as a feature, not a bug.** If we relax the in-hand
   constraint, v4's policy is interesting — a learned ground-assisted
   manipulation strategy. Could be a starting point for table-edge
   tricks or place-then-pick research.

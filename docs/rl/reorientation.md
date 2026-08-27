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

Visual inspection of [v4_peak_floorbracing.mp4](videos/20260601_reorient/1502_v4_peak_floorbracing.mp4)
and [v4_final_floorbracing.mp4](videos/20260601_reorient/1502_v4_final_floorbracing.mp4)
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

> **⚠️ CORRECTION (read first):** every v2 result below (Stage 1, Stage 2,
> v2.1 de-centering) was produced by a finetune that **warmstarted only the actor
> and discarded the checkpoint's critic** — a bug that silently wrecked
> verticality. The reward-SUM metrics in these sections (e.g. "alignment 76 ≈ v1")
> are misleading; on the honest deterministic metric (held-vertical cos) all v2
> runs were 0.83–0.87 vs v1's 0.97. See **"Phase Policy B v2 — CRITICAL CORRECTION"**
> below. Treat the per-section "winners" as provisional pending re-runs with the fix.

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

- [policyB_v2_smooth5x.mp4](videos/20260602_reorient/0021_policyB_v2_smooth5x.mp4) — **chosen**
- [policyB_v2_smooth10x.mp4](videos/20260602_reorient/0021_policyB_v2_smooth10x.mp4)

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
preserved from `docs/experiments/STAGE1_RESULTS.txt`.)

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

- [policyB_v2_smooth10x_quick.mp4](videos/20260602_reorient/0133_policyB_v2_smooth10x_quick.mp4) — **recommended final Policy B v2**
- [policyB_v2_smooth5x_quick.mp4](videos/20260602_reorient/0133_policyB_v2_smooth5x_quick.mp4) — the "quick" variant (regressed; see below)

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
  (`results/rl/b02_20260602-0024-policyB_v2_smooth10x_quick/tensorboard/model_1219.pt`).
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
`results/rl/bx_20260602-1323-policyB_v21_decenter_w40.0/tensorboard/model_812.pt`. This is the
new recommended reorient policy and the warmstart for Phase 3.

### Continuous A→B handoff (workstream B) — mechanism ✅, distribution gap ⚠

Resolves observation 2's *mechanism*. `scripts/rl_demo_handoff_continuous.py` runs ONE env
with **no reset**: it builds the env in B-mode (66-dim) and feeds Policy A `obs[:, :65]`
(A is 65-dim; B's extra `target_axis_misalign` obs is appended last) through a real lift to
0.10 m, then switches to Policy B's actions on the same physics state — no teleport.

**Finding:** the handoff is seamless mechanically but **Policy B's fingers "freak out"
(erratic high-frequency motion) right after the switch and fail to articulate** — it then
drops the cylinder (z = 0.097 at handoff, gone by ~step 75). Root cause: B was trained on
its *exact* skip-lift spawn (object pre-placed at the lifted pose, 5 mm drop-into-grip, B
acting from step 0); A's *real* lift delivers a different contact/pose state and B starts at
step ~45, so B is **out-of-distribution** at the seam and emits garbage. (B is fine when run
alone — it spawns into exactly its training state.)

**Failed fix attempt — retrain a single normal-lift B (catch-22).** Finetuning B in the
normal-lift env (real lift, reorient gated after) hit a dead end:
- residual ON during the lift → the reorient-trained policy is OOD on the flat-object grasp
  and blows up the physics (**NaN at iter 0**);
- residual OFF during the lift (`--finger-residual-active-from-step`, added in this attempt)
  → the *scripted* grip alone is too weak, the object slips to only 0.06 m (not 0.10), and B
  then faces a too-low OOD object → **alignment stuck ~1.7, no reorientation**.
A lifts cleanly only *because* its residuals actively hold the grip through the lift, so one
policy can't both lift-grip and reorient without becoming the monolithic policy the
two-policy split was created to avoid.

**Correct fix (next, keeps the two-policy split + no-reset script):** make B robust to A's
terminal state — either (a) **train-the-handoff**: run A's lift inside B's training env so B
learns reorient from A's *actual* terminal state; or (b) **domain-randomise B's skip-lift
spawn** (jitter lifted height / orientation / contact) so it tolerates the handoff variation.
(b) is the simpler first try. The `residual_active_from_sim_step` knob and the continuous
handoff script remain useful building blocks.

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

### Phase Policy B v2 — CRITICAL CORRECTION: the critic-warmstart bug

After a user flagged that the v2 policies "looked worse" than v1, a deterministic
behavioral comparison (held-vertical cos = body +Z · world +Z averaged over the last
50 of 200 steps, `/tmp/cmp.py`) exposed that **the v2 reward-SUM metrics were
misleading**: v1 holds cos **0.97**, every v2 finetune only **0.83–0.87**. The world-
and palm-frame root-xy drift is ~0 for *all* policies including v1, so the earlier
"de-centering −36%" claim was also not real (the lateral motion seen on video is the
long cylinder's far **end** swinging during rotation, not a root translation).

**Root cause (found by ablation):** finetuning v1 with its *own* config and **no
changes** — the control — itself collapsed verticality to **0.66**. The bug:
`rl_train_cube.py`'s warmstart loaded only `actor_state_dict` and **discarded the
checkpoint's `critic_state_dict`**, so every finetune started with a *fresh random
value function*. The garbage early advantages knock the converged actor off its
optimum, and ~800 finetune iters can't recover. v1 only looked good because it trained
2000+ iters from scratch. **This single procedural bug — not any reward — caused the
entire v2 verticality regression.**

**Fix (commit `8830b27`):** also warmstart the critic (and optional optimizer), same
grow/zero-init partial-load as the actor (`--warmstart-critic`, default on). Validation
(held cos, 10M-ts finetunes from v1):

| finetune | without critic (bug) | **with critic (fix)** |
|---|---|---|
| control (v1-faithful, no change) | 0.66 | **0.96** |
| signed progress | 0.84 | **0.978** |

The control fully recovers to ≈ v1, confirming the diagnosis. And **signed progress,
once the critic isn't sabotaging it, holds vertical *best* (0.978 > v1)** — penalizing
slip-back is genuinely good, the opposite of what the buggy runs implied.

**Remaining real tradeoff (smoothness ↔ verticality).** With the fix, verticality is
restored (0.94–0.98) but the 5× smoothness ramp barely dented object jerk (≈42, ≈ v1).
The OLD v2-10×-quick was 5× *smoother* (jerk 8.3) — but largely by **under-rotating**
(it stopped at 0.83, ~34° short; the last ~17° to true vertical is where the jerky
finger corrections live). So smooth-vs-vertical is a genuine frontier to navigate, now
explorable cleanly with the critic fix. A proper 5×/10× smoothness sweep (signed +
critic, from v1) is running to map it.

**Implication:** all per-section "winners" above (Stage-1 5×, Stage-2 10×-quick, v2.1
de-center w40) were chosen under the bug and on misleading reward sums — they need
re-running with the critic fix and re-judging on held-cos + jerk. The genuinely durable
results from this work are: the **critic-warmstart fix**, that **signed progress helps**,
and the **honest held-cos/jerk evaluation harness**.

#### Definitive re-runs WITH the critic fix (held-cos / object-jerk)

All warmstart v1, critic ON, alpha-curriculum OFF, evaluated deterministically:

| policy | held cos | obj jerk | verdict |
|---|---|---|---|
| v1 (baseline) | 0.96 | 41 | jittery but vertical |
| **signed + critic** | **0.978** | 52 | **best — vertical + slip-back fixed** |
| + 5× smoothness | 0.69 | 91 | ✗ destabilises the hold (slips + wobbles) |
| + 10× smoothness | 0.92 | 70 | ✗ worse than base on both |
| + quick mechanisms | 0.78 | 63 | ✗ degrades |
| + de-centering (−40) | 0.78 | 61 | ✗ degrades |

**Every reward elaboration on top of `signed + critic` makes it worse** — the precise
vertical hold is a fragile optimum and any competing objective pulls the policy off it.
The buggy-critic runs had *masked* this (they couldn't reach the vertical optimum in the
first place, so adding terms looked harmless/helpful).

**RECOMMENDED reorientation policy: `signed + critic`** =
`results/rl/b03_20260602-1636-policyB_abl_signed/tensorboard/model_405.pt`
(v1 recipe + signed `target_axis_progress` + critic warmstart; no smoothness / quick /
de-centering / lateral terms). Holds cos 0.978 (beats v1's 0.96), fixes slip-back, stays
aloft (min-z 0.11 m). Video: [policyB_signed_critic.mp4](videos/20260603_reorient/1250_policyB_signed_critic.mp4).
A longer (30M) run from v1 would likely polish it further but isn't required.

**Smoothness / de-centering / bracing — where they stand:**
- *Smoothness:* not achievable via jerk penalties (the corrective jerk **is** the
  stabilisation). For sim-to-real, use a non-reward lever — action low-pass filter at
  deployment, or motor-delay/observation-noise domain randomisation during training.
- *De-centering:* **REAL** (corrected — an earlier "≈0 drift" claim was a measurement bug).
  Object-center lateral excursion, deterministic: v1 **2.9 cm**, signed+critic **5.1 cm** —
  signed+critic de-centers *more* than v1. Worth addressing (it inflates the end→palm 3D gap
  below), but the `object_lateral_drift` penalty as tried degraded the policy; needs a
  gentler formulation or to be applied during a longer-from-scratch run.
- *Bracing:* **closer than first reported** (an intermediate "0 cm" reading was also buggy).
  Ground-truthed: signed+critic holds the cylinder at **cos 0.99** with its top end only
  **~3 cm below the palm vertically** (≈8 cm in 3D — the extra is the lateral de-centering
  offset), **no palm contact yet**. So it's "almost bracing" — the *vertical* shortfall is
  ~3 cm, not the ~7–8 cm implied earlier. Closing it likely needs the de-centering fixed
  first (so the end is under the palm) plus a gentle upward nudge; reward-only attempts so
  far degraded the reorient. Not a hard geometric wall, but unsolved.

---

## Phase: de-centering + seamless A→B handoff (2026-06-03 → 06-04)

Two coupled workstreams: (1) curb the real de-centering, and (2) make Policy B hold the
object through a **continuous, no-reset** handoff from Policy A's lift. All judged on the
honest deterministic metrics (`scripts/rl_eval_reorient_metrics.py`: held-cos / obj_jerk /
min_z / drop) + the continuous-handoff hold test (`scripts/rl_demo_handoff_continuous.py`:
post-handoff `min-z`; **hold ⇔ min-z > 0.05**), never reward sums.

### P1 / P2 / P3 — three single-variable runs (40M ts, 3072 envs, warmstart signed+critic)

| policy | held_cos | peak | obj_jerk | min_z | drop | world Δlat |
|---|---|---|---|---|---|---|
| baseline signed+critic (405) | 0.979 | 0.989 | 52.3 | 0.109 | 0 | 3.8/5.2 cm |
| P1 handoff-DR alone (541) | 0.959 | 0.995 | 57.8 | 0.115 | 0 | — |
| **P2 lateral-only −8 (541)** | **0.988** | **0.997** | **26.9** | 0.117 | 0 | 5.6/4.8 cm |
| P3 statebank (541) | 0.933 | 0.946 | **8.1** | 0.114 | 0 | **3.0/3.1 cm** |

- **P2 (`--lateral-drift-weight=-8` alone) = best reorienter:** held-cos **0.988** (> baseline
  0.979) and obj_jerk **HALVED** (26.9 vs 52). Surprise: the lateral penalty did **not** reduce
  de-centering (it drifted ~1 cm *more*) — it acts as a **smoothing regulariser**. So a
  *position* penalty smooths where velocity/accel jerk penalties (which destabilise the hold)
  failed. New recommended **standalone** reorienter:
  `results/rl/b04_20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt`.
- **P3 (handoff state-bank) = best de-centerer (3.0/3.1 cm) + smoothest (jerk 8.1) but weakest
  reorienter (0.933).** Training on A's real centered grips keeps it centered; trades verticality.
- **P1 (handoff-DR alone) = worse, discard** — DR alone destabilises the grip (gotcha #4).

### Handoff seam drop — DIAGNOSED: an observation-discontinuity OOD shock, not grip weakness

Instrumented the continuous A→B rollout (object-z every step). The drop is **instantaneous at
the seam**, identical for P2 / P3 / baseline (all skip-lift-trained): z 0.094 → 0.073 → 0.022 →
floor within **3–5 steps** of the switch. B collapses the grip the moment it takes over → it is
**out-of-distribution** at the seam: a skip-lift B never saw the normal-lift env's lift-command
phase / `ref_object_pose` schedule that A delivers. That's why **neither DR (P1) nor the
state-bank (P3) helped** — both still trained B in the *skip-lift* env. The handoff demo's
`--blend-steps` (linear A→B action ramp) extends the hold ~6 → 18 steps and the object even
rises, but every skip-lift B still drops once in full control. **The fix must be in training:
train B in the normal-lift env so it sees the seam.**

### Normal-lift B — v2 collapse → v3 grace window (works, NaN'd) → v3b relaunch

- **v2 (`policyB_normallift_v2_fromP2`, warmstart P2, normal-lift): COLLAPSED.** Standalone
  held-cos **0.029**, **100 % drop**, training reward fell 12 → 3 by iter ~5. Cause: at step 35
  **everything fired at once** — B's residual activates, lift + floor terminations engage, *and*
  the full reorient reward (100 align + 300 progress). The skip-lift-prior warmstart is OOD on
  the post-scripted-lift state, fumbles, and the terminations kill OOD episodes short → reward
  collapse → never learns. Continuous-handoff min-z **0.005 m** (dropped).
- **v3 grace window (`policyB_normallift_v3_grace`): the candidate fix — and it WORKED.** Give B
  a **grace window**: it takes over (residual) at step 35 but only has to **HOLD** until step 50;
  terminations + reorient pressure engage at 50. Reward stayed **healthy and flat (~10) for 60
  iters** — the grace window prevented the v2 collapse. It then **NaN-crashed at iter ~60/750**
  (a transient warp env blowup; rsl_rl's `check_nan` raises on *any* env NaN and kills the whole
  run, no retry) → only `model_50` saved (undertrained → still drops, min-z 0.003). So the
  approach is sound; it died to bad luck at 60/750.
- **v3 hold-only control (`policyB_normallift_v3_holdonly`, grace window, reorient DISABLED):
  completed 40M and PROVES B can survive the scripted-lift → takeover.** `tip_lost` rose to a
  mid-training hump (~44) then **recovered to ~1–4** by the end — B learns to hold A's delivery
  in the normal-lift env. (Not deployable: 65-dim / no reorient, so it can't run in the 66-dim
  handoff demo; it's a pure isolation control.)

**→ v3b (in training): relaunch the grace window TO COMPLETION, NaN-resilient.**
`scripts/train_normallift_B_v3b_gracewindow.sh` runs two parallel variants (so a stochastic
single-env NaN doesn't waste the 70-min run): **R** = byte-identical repro (finger-residual
0.5, hard reorient onset at 50); **S** = soft onset (finger-residual 0.4 + basin-width
curriculum α 0.5→4.0 over 150 iters) to lower the step-50 OOD/physics shock. Both warmstart P2,
40M / 3072, staggered, per-process Warp cache. Early health verified: critic fully warmstarted
(actor 9 + critic 8 tensors copied), no NaN, reward ~9.5 (R) / ~15.6 (S, wider basin). **Want:
post-handoff min-z > 0.05 (holds) at held-cos near P2's 0.988.** A follow-on trigger
(`scripts/v3b_eval_trigger.sh`) waits for both → evals → renders the seamless video → writes
`docs/experiments/STATE_HANDOFF_RESULTS.txt`.

#### v3b OUTCOME (2026-06-04): ran to completion, but BOTH variants COLLAPSED — the grace window did not solve the handoff.

NaN-resilience succeeded — **both R and S ran the full 40M / 542 iters with no crash** (the
v3 NaN was bad luck, not a systemic bug). But running to completion *revealed* that the
"healthy flat ~10 reward" that looked promising at v3's 60 iters was **not learning — it was a
degenerate plateau.** Reward stayed flat at **~9.3 (R) / ~9.0 (S) for all 542 iters** and never
climbed. Authoritative deterministic eval (`rl_eval_reorient_metrics.py`, own normal-lift env):

| policy | held_cos | peak | obj_jerk | min_z | drop | handoff min-z (step 40, blend 8) |
|---|---|---|---|---|---|---|
| P2_lateral (ref) | 0.987 | 0.997 | 27.6 | 0.117 | 0.00 | (A's deliverer) |
| signed+critic (ref) | 0.979 | 0.988 | 51.7 | 0.109 | 0.00 | — |
| **v3b_repro (541)** | **−0.035** | 0.121 | 103.1 | 0.004 | **1.00** | **0.0037** |
| **v3b_soft (541)** | **−0.078** | 0.121 | 41.9 | 0.005 | **1.00** | **0.0069** |

Both held-cos are **negative** (object flat/random, not vertical) with **100 % drop**; both
handoff min-z (**0.004 / 0.007 m**) are far below the 0.05 threshold → **NO seamless handoff.**
Videos `handoff_v3b_{repro,soft}.mp4` show the object dropping at the seam, same as before.

**Mechanism (why the grace window failed):** the window successfully prevented the *v2-style
training collapse* (reward never crashed to 3) — but it did so by letting B settle into a
**hold-during-grace, drop-at-the-reorient-onset** local optimum. Training termination stats at
convergence: `object_floor_proximity` **95.9 (R) / 58.9 (S)**, `object_height` **0.012 / 0.015 m**
(on the floor), `episode_success` **0**. The reorient phase (step ≥50) never learned: episodes
accumulate ~50 steps of grace-window hold reward (→ the flat ~9.3) and then the object hits the
floor the moment reorient pressure + floor termination engage. The soft onset (S) bought nothing
material (slightly less floor-proximity, but still −0.078 / drop 1.0). **Confirmed conclusion: the
skip-lift-prior P2 warmstart is not made robust to the seam by a grace window in 40M ts** — the
grace window keeps training *alive* but also keeps it *stuck*, so it can't cross from "hold" into
a working post-seam reorient.

**Candidate next experiments (NOT run — autonomy budget spent; documented for the next session):**
1. **Warmstart the hold-only control, not P2.** The v3 hold-only run *proved* a normal-lift B can
   survive the takeover (tip_lost recovered to ~1–4). Finetuning *that* checkpoint toward reorient
   (instead of the OOD skip-lift P2) starts from a policy that already holds A's delivery, so the
   grace→reorient transition is in-distribution. **Most promising.**
2. **Add the P3 handoff state-bank to the normal-lift env.** Train B on A's *real* recorded
   post-delivery states (the seam obs) directly, so the seam is no longer OOD. (P3's state-bank was
   the best de-centering lever; here it would target the seam distribution.)
3. **Longer training** (≥80M) — least likely to help: reward was *flat*, not slowly climbing, so
   this is a local optimum, not undertraining. Deprioritize.

---

## Phase: closing the seam — A-side, B-side, and co-adaptation (2026-06-05 → 06-09)

This phase systematically swept *which policy to move* to close the seam. The blow-by-blow
(commands, infra gotchas, exact run dirs) lives in `RESEARCH_STATE.md`; this is the condensed
arc and its one durable conclusion. **Success metric throughout: continuous-handoff min-z
(`scripts/rl_demo_handoff_continuous.py`, handoff@40); hold ⇔ min-z > 0.05.** Everything below
still *drops* (min-z ≪ 0.05) — the seam is **not closed** — but the trend finally points somewhere.

### Branch B — un-freeze Policy A, migrate its grip onto B's (06-05)

**Goal:** leave B frozen; fine-tune A to *deliver* the grip B reorients from (the measured seam
gap was the finger config, ~0.16 rad/joint off B10's hold). **Changed:** seam-gated dense
`handoff_target_proximity` reward pulling A's finger qpos onto B10's recorded grip; v1 collapsed
(stripped A's drop terminations → floor became an attractor — see lesson 7), v2 restored every
guardrail and relaxed *only* the slip terms + widened the proximity basin. **Result:** trains
clean, A migrates its grip *partway* (proximity plateaus), survival rises **monotonically** with
migration (0.0029 → 0.0049 → 0.0073) — but plateaus; A resists fully adopting B's grip and pushing
the weight harder re-invites collapse. **Takeaway:** grip-match is **real but insufficient** alone.

### Adapt B → A (skip-lift state bank) and the obs-schedule diagnosis (06-08)

**Goal:** the symmetric move — leave A frozen, make B robust to A's *real* delivered state via a
recorded state bank. **Result:** trains clean, still drops (min-z 0.0028). **Takeaway (sharp):**
the bank only fires in the **skip-lift** env, so B still trained under the skip-lift *observation
schedule* — which differs from the normal-lift deploy even when the physical state matches. **The
binding constraint is the obs schedule, not the state.**

### Onset-grip injection — state AND obs in-distribution (06-09)

**Goal:** the one combination adapt-B/branch-B couldn't reach — train B in the *normal-lift* env
(obs schedule == deploy) **and** inject A's real delivered state at the seam onset (state ==
deploy), via a new step-mode event `inject_handoff_bank_at_onset`. **Result:** min-z **0.0081** —
a new best, but still a drop. **Takeaway:** matching state + obs schedule helps, but the *teleport*
remains: the injected snapshot was static (the bank had zero velocities — a recorder bug).

### Complete-state injection — make the teleport Markov-complete (06-09 eve)

**Goal (from the Opus-4.8 analysis):** remove the last teleport artifacts. **Changed:** fixed the
recorder's silent velocity bug (`root_link_velocity_w`, a nonexistent attribute → zeros; correct is
`root_link_vel_w`), added `robot_qvel`, captured A's last action; the inject now writes A's REAL
obj_vel + finger/palm qvel AND overrides the seam `last_action` obs (the only history-dependent obs;
there are no differenced/stacked-history obs, and position actuators carry no `act` state, so this
makes the injected seam *Markov-equivalent* to organic arrival). Measured: A's delivery velocity is
tiny (1.6 cm/s, settled) but `a_last` is substantial (**0.23 rad**) — the `last_action` mismatch was
the larger unaddressed OOD. **Result: min-z 0.0027 — WORSE than the static 0.0081** (run NaN-crashed
@iter221 after holding healthily; model_200 a fair late ckpt). **Takeaway (load-bearing):** making
the teleport more faithful did **not** help → the seam is **not a missing-state-info problem**, and
the entire *inject-A's-state-into-B* family is **saturated** (0.0028 / 0.0081 / 0.0027, all ≪ 0.05).

### Co-adaptation — move BOTH policies (06-09 eve, the new lever)

**Goal:** with each one-sided move saturated, test *moving both*. **Changed:** nothing new trained —
a *free* eval pairing the independently-migrated A (`Atol20`, A→B10 grip) with the independently-
adapted B (`Badapt`, B→frozen-A delivery). **Result:**

| pairing | min-z (handoff@40) |
|---|---|
| baseline frozen-A × B10 | ~0.0029 |
| A-moved × B10 (A alone) | −0.0001 |
| frozen-A × Badapt (B alone) | 0.0075 |
| **Atol20 × Badapt (BOTH moved)** | **0.0114** ← new best |

**Takeaway:** **co-adaptation — both policies migrating toward each other — beats either side
alone** (Lee 2021 / Röstel 2025, confirmed empirically). *Honest caveat:* 0.0114 is still a **drop**,
not a hold; and `Badapt` has no stable post-seam holding grip (drops by ~step 48), so the **weak link
is B catching**, not A delivering. An overnight **co-adaptation batch** (`scripts/overnight_batch.sh`
→ `docs/experiments/BATCH_RESULTS.md`) was launched to specialize B to the migrated A's delivery and push A-migration
further.

### Where this leaves the seam (→ next: the live-A reset)

Every adaptation so far trains B on a **teleport** into the seam (bank / inject); the deploy seam is
**organic** (A runs live). Even Markov-complete injection didn't close it — the remaining suspect is
the contact-solver warmstart / one-step contact-force ramp that no instantaneous teleport reproduces.
**The untried mechanism: run frozen Policy A LIVE during B's training reset** (steps 0..40, real
physics, zero teleport), then B's PPO rollout begins at the seam. This needs rsl_rl integration
(apply A's action pre-onset per-env and *mask those steps from the PPO update*), so it is a build,
not a flag — deferred as tomorrow's priority. Plus the co-adaptation loop (alternate A/B migration).

---

## Phase: the live-A reset CLOSES the seam (2026-06-10) 🎉

The untried mechanism worked. **The handoff seam is solved in principle** — the first
policy to *both* survive the continuous A→B handoff *and* reorient.

**Mechanism.** Frozen Policy A drives B's training env LIVE for steps 0..40 of every
episode (real physics, real contacts, real `last_action` — zero teleport); then B's PPO
rollout begins at the *organic* seam. The A-driven pre-onset steps are **masked from the
PPO update** (advantages zeroed + renormalized, returns kept — masking advantages, not
log-probs, avoids the log-prob trap). Code: `src/morphohand/rl/live_a_runner.py`
(`LiveAOnPolicyRunner`), `scripts/rl_train_cube.py --live-a-checkpoint/--live-a-onset`,
`scripts/train_handoff_liveA_reset.sh`.

**Result** (`20260610-1046-policyB_liveAreset_fromB10`, model_270, 20M/3072, warmstart
B10): continuous handoff@40, **post-handoff min-z 0.110 m (HELD ≫ 0.05)**, held-vertical
**cos 0.751**. B holds A's organic delivery at full height for the entire post-seam rollout
and reorients. Every prior teleport approach dropped within 3–5 steps (min-z 0.003–0.011).
Training signature confirmed it: masked-frac fell 0.95→0.20 (episodes lengthened ~5×),
align 0.45→58.9, tip_lost 51→8, episodes ran to time_out. Video:
[handoff_liveAreset_scale02.mp4](videos/20260610_reorient/1201_handoff_liveAreset_scale02.mp4).

> **⚠️ CONFIG-PARITY GOTCHA (#13).** That run trained at `finger_residual_scale=0.2` (the
> `rl_train_cube` default) while B10 (warmstart) AND the deploy demo use **0.5**. B relearned
> to hold at 0.2; the 0.5 eval applied its residuals 2.5× too large → instant seam collapse,
> an **artifact, not a failure**. The train env must match the deploy env on
> scale/easing/contact-gate. Also: **measure POST-HANDOFF min-z** — whole-rollout min-z is
> dominated by the pre-lift floor phase (z~0.012) so the 0.05 bar is unreachable by it.

### REORIENTATION QUALITY is now the open problem (the hold is solved)

User feedback on the live-A result: *"still jittery + doesn't reorient well."* Both flaws
trace to the **same root — the B10 warmstart** ("the violent survivor": held-cos 0.977 but
obj_jerk 108, 4× B4's 27). Three tests pinned it down:

| attempt | result | takeaway |
|---|---|---|
| +40M more training (from B10-live-A) | held-cos 0.742 (peak 0.817), no climb | the 0.74 reorient ceiling is the **warmstart, not undertraining** |
| warmstart **B4** (smooth, 0.988) instead | collapses at the seam (align 0, ep-len stuck 43) | B4's reorient-from-step-0 actions shock A's grip → drops before the reorient gate fires → zero gradient |
| deploy action low-pass (`--action-lowpass 0.5`) | drops the object (min-z 0.0027) | B10's corrective jerk **is** the stabilization — can't be filtered out (the smoothness gotcha, now at deploy) |

**Both complaints share one fix: get B4's smooth, full-vertical reorientation to survive the
seam.** The blocker is the **B4 catch-22** — a policy must *survive* the seam before it can
be *taught* to reorient there, but B4 reorients from step 0 and drops first. **The next
experiment** (untried; needs runner surgery, not a flag): a **training-time seam
action-ramp-in** — in `live_a_runner.py`, for the first ~8–12 steps after onset step the env
with `α·B + (1−α)·A` ramping α 0→1 (the training analog of the demo's `--blend-steps`), and
mask those blend steps from PPO too. This eases B4 into A's grip instead of shocking it, so
B4 survives long enough to get reorient gradient while keeping its smooth full
reorientation. Warmstart B4.

**OUTCOME (2026-06-11): the B4 path is DEAD; the win is lifting B10's quality
instead.** Built the seam action-ramp-in (`--live-a-blend-steps`, plus
`--term-tip-lost-steps` and a `LIFT_TERM_START` knob in
`scripts/train_handoff_liveA_reset.sh`). A layered smoke diagnosis showed the ramp
is *necessary but not sufficient*: it keeps the object aloft during the ramp, and
delaying terminations to B-takeover does yield trainable steps — but the moment B4
gets authority it drops A's flat delivery to the floor (object_height 0.012, align
≈0) and never recovers, because **B4 has no hold prior** (exactly why B10/hold-only
was the warmstart that originally worked). All three B4-warmstart full runs failed
(floor-drop + transient-NaN). The productive direction was the **inverse**: keep the
already-surviving B10-live-A policy and add a **non-terminating commit + speed bonus**
(`--success-bonus-weight`/`--speed-bonus-weight`, NO terminating success → no
threshold-gaming) plus a sharper near-vertical basin re-anneal.

| run | recipe | held-cos | peak | post-handoff min-z |
|---|---|---|---|---|
| b24 | B10-live-A baseline | 0.751 | 0.816 | 0.110 |
| b27 | +40M continuation | 0.742 | 0.817 | 0.105 |
| **b28** | + commit-bonus 30 | 0.759 | **0.891** | 0.110 |
| **b29** | + commit 60 + α-re-anneal (1→6) | **0.784** | 0.866 | 0.108 |

The commit bonus + sharper basin **moved the quality ceiling 0.75 → 0.78** (real but
modest, ~5° closer to vertical; the hold stays solid ~0.11) — but did **not** reach
B4's standalone 0.988. **The B10-warmstart basin is a stubborn quality ceiling that
reward-shaping only nudges.** New best handoff policy =
`b29_…B10qual_commit60/tensorboard/model_405.pt`
([handoff_B10qual_commit60.mp4](videos/20260611_reorient/1231_handoff_B10qual_commit60.mp4)).
Breaking past ~0.8 likely needs a *new mechanism* (distill B4's reorientation onto the
held post-seam states b29 visits — behavior cloning / teacher-student), not more
reward tuning. Iterate from b29, not b24.

### Known visual artifact: thumb penetrates the screwdriver (do NOT fix yet)

In the live-A `cont40M` rollout (`20260610-1355-policyB_liveAreset_cont40M`) and many
prior runs, **the thumb visibly phases/penetrates into the screwdriver geometry** during
the grip. This is believed to be a consequence of the deliberately *relaxed/soft contact
solver* used for this task — the RL env sets `impratio=10`, `cone="elliptic"`
([env_cfg.py:1399-1405](../../src/morphohand/rl/env_cfg.py#L1399-L1405)) and the screwdriver
scenes use a soft `solref="0.006 1" solimp="0.97 0.995 0.0005"` geom default
([scene_screwdriver_medium_flat_short_proximal.xml:11](../../assets/mjcf/baseline/scenes/scene_screwdriver_medium_flat_short_proximal.xml#L11)),
which allow some interpenetration in exchange for a stable, non-explosive grip.

It is **somewhat bad** (cosmetically and for sim-to-real fidelity), but **we are NOT changing
the contact parameters until we have a better working policy.** Retuning
`impratio`/`solimp`/`solref` now would perturb every policy's grip force and invalidate the
A/B lineage and all the seam comparisons above. Revisit the contact stiffness as a
sim-to-real hardening pass once reorientation quality is solved.

---

## Phase: firm+smooth handoff → the force chase → the correction (2026-06-12 → 06-13)

Iterating from b29 (held-cos 0.78), this arc first improved the handoff (b30→b32), then spent four
runs trying to make the grip "gentle," and finally **discovered the whole gentleness premise was a
phantom.** It is the cautionary tale of the project: a plausible visual story ("our grip looks like a
death-grip; B3 looks gentle") sent us optimising a non-problem for two days. The durable outputs are
the **honest eval diagnostics**, the **grip-force probe**, and the **measured force floor**.

### b30 → b32 — firmest + smoothest handoff yet; the diagnostics that exposed jitter

**Goal:** push b29's reorientation quality up while keeping the hold. **Changed (b32, warmstart
b30/model_405, live-A reset @ scale 0.2):** `grip_force` reward +6, smoothness `action_rate −0.05` +
`object_ang_acc −0.05` (gated step 45), schedule w4, `target_axis_alpha 4`. **Result:** post-handoff
min-z **0.1085** (firmest hold of any run), held-cos **0.891**, 4–5× smoother than the rejected b31
candidate (ang-jerk 113 vs 449). **Registered as the new best handoff reorienter.**

The **rejected b31 candidate** (`brace_d12f4_schedw8`) is the key lesson: it read held-cos 0.95 /
min-z 0.078 (looked like a win) but the **new eval diagnostics** exposed it — ang-jerk **449**, 99 cm
horizontal path while netting 0.2 cm (violent vibration in place). The user confirmed visually:
"slips/jitters the entire time." **Takeaway:** cos+min-z are blind to jitter; a frantically-shaking
object still averages near-vertical. The eval (`rl_demo_handoff_continuous.py`) now prints a per-20-step
heartbeat + end summary (lateral drift, horizontal path/wander, z sink-rate, **lin/ang jerk**, contact
force, auto VERDICT flagging SLIP/SINKING/JITTER). **Judge on this, never cos/min-z alone.**

### b33 / b34 — force-regularise the grip; find the floor

**Goal:** b32 held with an ~11 N fingertip clamp (believed to be the source of both the
thumb-into-screwdriver penetration and the residual jitter). Is 11 N **necessary, or learned
laziness**? **Changed:** added `grip_force_excess` — a quadratic penalty on fingertip force above a
threshold (`mjlab_terms.grip_force_excess`, CLI `--grip-force-penalty-*`). b33 = b32 + penalty (thresh
4 N, w −6). b34 = a threshold sweep (4 → 3 → 2.5 → 2 N) continuing from b33.

**Result (b33):** the 11 N was *partly* learned laziness — it compressed for free. Force 11→**7.5 N**,
ang-jerk 113→**49**, lin-jerk 3.6→**1.17**, wander 23→**8.9 cm** — all while still holding (min-z
0.111). But it cost a little verticality (cos 0.891→0.845) and **7.5 N still over-clamps + penetrates**.
A partial, diminishing-returns move, not a B3-level transformation.

**Result (b34 sweep) — the grip-penalty lever is DEAD:**

| thresh | fingertip force | held-cos | post-min-z | ang-jerk | palm force |
|---|---|---|---|---|---|
| 4.0 (b33) | 7.5 N | 0.845 | 0.111 | 49 | **0.0 N** |
| 3.0 (t30) | 7.5 N | 0.747 | 0.110 | 85 | **0.0 N** |
| 2.5 (t25) | 6.7 N | 0.771 | 0.112 | 69 | **0.0 N** |
| 2.0 (t20) | 6.6 N | 0.782 | 0.112 | 76 | **0.0 N** |

Halving the penalty knee 4→2 N moved fingertip force **< 1 N**. All HOLD (~0.11), all over-clamp
(~7 N), all jitter (69–85), and **palm force is 0.0 N in every run** — the object is held at a
fingertip pinch ~8 cm below the palm and **never seats**. **Takeaway (at the time):** a fingertip-only
hold of this rod has a physical minimum force ≈ 7 N (fingers resist gravity+torque by friction) — you
cannot reward your way below it; gentleness looked like a *seating* problem. **`b34_t20` is the
gentlest live-A handoff** (6.6 N) and the warmstart for the later gentle-B run.

### ⚠️ The correction: the grip-quality premise was a PHANTOM

Before committing to a big swing (seat / change-A / morphology), we **verified the premise the whole
arc rested on** — "B3 is gentle because it holds a ~3 N seated grip." It is **FALSE**. Direct
measurement (`scripts/probe_grip_force.py` — fingertip+palm contact force in each policy's own
standalone held+reorient rollout, steady-state):

| policy | regime | held-cos | fingertip force | palm force |
|---|---|---|---|---|
| **B3** (b03 signed) | standalone | 0.978 | **7.04 N** | 0.00 N |
| **B4** (b04 lateral) | standalone | 0.988 | **8.77 N** | 0.00 N |
| **b34_t20** (ours) | live-A seam | 0.782 | **6.6 N** | 0.00 N |

**Four facts that demolish the framing:** (1) B3/B4 are **NOT gentle** — they grip 7–9 N, as hard or
harder than our handoff (6.6 N). (2) **Nobody seats** — palm force 0.00 N in all policies, incl. the
two "good" ones; seating was never the differentiator (the b34 "seating problem" verdict was wrong).
(3) **Force does NOT cause jitter** — B4 is the smoothest policy (obj-jerk ~26) at the *highest* force
(8.77 N). (4) **Penetration is universal** — same ~7 N on the same soft contact solver ⇒ all penetrate
the same; any "B3 looks cleaner" is a contact-angle effect of the deliberately-soft `solimp` we froze,
orthogonal to the policy. **Root of the error:** the "B3 ≈ 3 N" benchmark was a **misread of
`grip_force_max=3.0`** (the grip-force REWARD's saturation cap), not B3's actual force. So b32 / b33 /
b34 all optimised toward a phantom. The sweep was still informative — it proved force is **decoupled**
from both hold (6.6–8.8 N all hold) and smoothness — it just wasn't the problem we thought.

### 3-stage composition (A→b32→B4) is DEAD; b32 reframed as a good reorienter

We tested the cheap "use the survivor to manufacture B4's start" idea: A lifts → b32 catches +
stabilises to a clean cos-0.90 pose → hand off a 2nd time to B4 to finish. **Result:** b32 delivered
B4 a rock-stable, near-vertical (cos 0.90, lateral < 1 cm) start — exactly the clean start the idea
assumed B4 needs — and **B4 still dropped it within 10–30 steps** (hard or blended). **Takeaway:** the
B4 catch-22 is **not about the object pose; it's about the GRIP configuration.** B4's competence is
inseparable from its own finger placement; you cannot drop it into another policy's grip. **Corollary:
naive action-distillation (DAgger with B4 as the action teacher) will hit the same wall** — B4's
actions are wrong for any other grip. **The reframe:** b32 is *already* a good handoff reorienter (holds
+ cos 0.895); the true remaining gaps vs B4 are JITTER (~112 vs ~26) and a modest verticality gap, both
tracing to the **B10-warmstart basin** + the marginal fingertip grip of a long rod on 3 smooth tips.

---

## Phase: pivot to SMOOTH / LOW-FORCE (2026-06-22)

**User directive** (the reprioritisation that opens this phase): *"i still don't love our best grasp and
am still very frustrated that we can't train a seamless handoff that doesn't use excess force. at this
point i don't care about a super close-to-brace pullup or a close-to-vertical reorientation, i just want
a SMOOTH, LOW-FORCE grasp and reorient. … then we can start thinking about morphology optimization."*

**The honest framing** (accepted by the user): genuinely *low* force — below the ~6.6 N fingertip floor
— needs the object to **seat into the palm** so the palm bears load and the fingers relax, which **this
morphology cannot do** (object sits 7–8 cm below the palm; `palm_brace_force` has fired in 0 of all
runs). **That is the morphology step, queued next.** But one lever is untried *within* the current
morphology: **every run so far maximised verticality** (alignment +100), which is exactly what forces
the tense corrective clamp + jitter. A **gentle partial reorient** (verticality relaxed) may let the
grip relax and smooth out for free.

**Two runs launched (2026-06-22, detached, watchdog'd — see `RESEARCH_STATE.md` for full recipes):**

1. **Gentle low-force REORIENT (Policy B)** — `scripts/train_gentle_lowforce_B.sh`. Warmstart b34_t20;
   live-A reset @ scale 0.2. Single change-set: **relax verticality** (align 100→40, alpha 4.0→1.5 wide
   basin, progress 300→120), **lower force** (grip reward 6→2, keep over-grip penalty thresh 2.5),
   **smoother** (lateral −8→−12). **WIN = holds at materially lower force AND lower ang-jerk than
   b34_t20, even at a lower held-cos.**
2. **Lower-force GRASP (re-open Policy A)** — `scripts/train_lowforce_A.sh`. The user chose to re-open A
   (branch-B). Objective = can A lift+deliver with less fingertip force? Warmstart a01; A's full lift
   recipe + the over-grip penalty (thresh 6.0, milder); **all grasp guardrails kept** (lesson #7), only
   precision-slip relaxed; collapse watchdog on. **Eval** with `scripts/probe_grip_force.py` vs the a01
   baseline.

**Then:** morphology optimisation (the real low-force lever — re-grip higher / seat into palm /
fingertip geometry), which deliberately breaks the A/B lineage and so comes after this round.

### Results: the grip defect is STRUCTURAL → pivot to morphology

**gentleB worked** (the smooth/low-force trade): vs b34_t20, ang-jerk **74→57**, fingertip force
**6.8→5.3 N**, at lower held-cos (0.77→0.64). Relaxing verticality does let the grip relax.

**But per-finger instrumentation** (`probe_grip_balance.py` + per-finger output added to
`rl_demo_handoff_continuous.py`) exposed the real defect: the grip is a **degenerate pinch** — the
**thumb is idle (~1.6 N)** while index+middle clamp ~8 N each (all three touch). B4 (the best
reorienter) is a **balanced** tripod (7/10/10 N), and our *total* force (~20 N) is already below
B4's (~27 N) — so "excessive force" is really a **lopsided** grip.

**The spread penalty could not fix it.** A new `grip_force_spread` reward term (penalise per-finger
max−min) + switching the over-grip penalty to `reduce=max`, warmstart gentleB → thumb 1.6→**1.8 N**
(still idle), index 8.0→7.1. **The policy cannot recruit the thumb** into a load-bearing opposition —
its *placement* can't oppose the other two against this object. That is geometry, not reward.

**Contact-hardening confirmed the grip is marginal.** A per-run `--frozen-scene-xml` override + a
stiffened scene showed that hardening the contact even mildly **breaks frozen Policy A's grasp** (the
object never leaves the floor). The soft contact is functionally load-bearing; the visible
penetration is a symptom of the marginal grip — a hard-contact run needs retraining A+B from scratch.

**Conclusion → morphology.** Both structural defects (idle thumb, no seating) map onto the existing
**9-param design space** (per-finger x/y/length, `src/morphohand/sampling/morphology.py`). The plan,
including an honest read on why the "Shape Your Body"/VGDS value-gradient method is the wrong tool for
a task this brittle (evaluate designs by rollout, not value gradient), is in
**`docs/rl/morphology_optimization_plan.md`**. First experiment: reposition the thumb for true
opposition, retrain B, measure per-finger balance.

---

## Phase: morphology LANDSCAPE — does the hand geometry move the task? (2026-06-25)

The thumb-opposition Stage-1(a) hand-pick (`m01_thumbWinner`) gave a *perfectly balanced
grasp* — and then **failed to reorient** (held-cos −0.68). That single result reframed the
question away from "find the one fix" toward the original co-synthesis goal: **does task
quality actually vary across the morphology landscape, or are all designs roughly
equivalent?** If equivalent, the lever is control; if not, morphology matters and we get a
map of the good region.

### Method — decompose the task to avoid the handoff confound

`scripts/morph_landscape_sweep.py` evaluates **12 morphologies** (the baseline, the
thumb-opposition winner, and 10 Latin-hypercube samples over the 9-param x/y/length space).
Each design is scored on **both halves of A→B separately**, so a handoff-control failure can't
be mistaken for a morphology failure:

- **GRASP / lift half (A):** CEM grasp optimizer (`phase1_optimize_grasp.py`, ~28 s) →
  graspability (`cube_lift`) + per-finger contact persistence balance.
- **REORIENT half (B):** a **skip-lift reorienter retrained on that grasp** (warmstart B4,
  15 M ts, ~25 min each) → deterministic held-cos + per-finger force/contact.

So **we did train a policy per morphology** — eight skip-lift Policy-B reorienters under one
*identical* recipe, morphology the only variable. The sweep is resumable (per-design JSON
checkpoint), sequential, and try/except-per-design so one blowup never sinks the run. Raw
results live in `docs/experiments/MORPH_LANDSCAPE.{json,txt}`.

### Landscape verdict — morphology matters STRONGLY, and grasp does not predict reorient

![morphology landscape summary](img/morph_landscape_summary.png)

| | finding |
|---|---|
| **Spread** | reorient held-cos ranges **−0.68 … +0.93** across 12 designs — designs are emphatically **not** equivalent. |
| **Graspability** | ~3/12 are ungraspable outright (`cube_lift < 0.02`); the CEM screen prunes them in 28 s before wasting a 25-min B run. |
| **Grasp ⊥ reorient** | **grasp balance does NOT predict reorient quality** (middle panel): `m01` has a *perfect* grasp balance and reorients *worst* (−0.68); `m05` has a *poor* grasp persistence and reorients *best* (+0.93). The cheap grasp screen is a graspability filter, **not** a reorient proxy. |
| **In-hand vs floor-brace** | held-cos alone is fooled: `m03` reaches cos 0.90 with **~0 N on all three fingers** — it floor-braces (cf. Phase v4). The per-finger force panel is what separates a genuine grip from a cheat: **only `m05` loads all three fingers** (11.7 / 12.7 / 9.3 N, contact 0.95 / 0.95 / 0.83). |

### Training dynamics — the designs train *differently*, not just to different endpoints

![morphology training dynamics](img/morph_landscape_training.png)

Overlaying all eight per-design trainer logs (regenerate with
`scripts/morph_landscape_plots.py`) shows the morphology gates learning from the very first
iterations, under one identical recipe:

- **Only `m05` ever finds the gradient.** Its mean reward climbs ~10 → **200**, alignment
  0 → ~42, and `target_axis_progress` ramps positive — the classic signature of a policy that
  *discovers* in-hand rotation. Every other graspable design stays pinned near its warmstart
  (reward flat ≲ 30, alignment flat), i.e. the morphology offers **no learnable rotation
  affordance**, not an undertraining artifact.
- **The floor-proximity termination is the tell.** All designs start with `object_floor_proximity`
  pegged at the cap (object sitting low at the skip-lift spawn). On `m05`/`m03` it collapses to
  ~0 within ~25 iters (the policy learns to hold the object up); on the idle-finger designs it
  stays high — they never even get the object reliably clear of the floor.
- **`m01`'s grip is *too* stable to articulate.** Its `tip_lost` sits high (~40/iter) and its
  reward never moves — the thumb-opposition geometry that made the grasp balanced also clamps
  the object so symmetrically that the ±0.5 rad finger residuals can't roll it. Balanced-for-grasp
  and articulable-for-reorient are **different geometric objectives** — the central surprise of
  the sweep.

### m05 — verified genuine smooth in-hand reorient

Authoritative deterministic eval (`rl_eval_reorient_metrics.py`, m05's own scene) confirms `m05`
is real, not floor-braced like `m03`:

| metric | m05 | reference (B4) |
|---|---|---|
| held-cos | **0.933** (peak 0.983) | 0.988 |
| object jerk | **21.5** (smoother) | 27 |
| min center-z | **0.064 m** (> 0.05 floor) | 0.117 |
| drop | **0.00** | 0.00 |
| per-finger force | 11.7 / 12.7 / 9.3 N (all loaded) | balanced tripod |

Video: [morph_landscape_grid.mp4](videos/20260630_reorient/1100_morph_landscape_grid.mp4) — a 2×3 grid of
six designs' trained rollouts (m05 winner, m03 floor-brace, m07/m11 partial, m06 idle,
m01 thumb-grasp-fails), which makes the failure modes legible at a glance; and
[m05_landscape_winner.mp4](videos/20260625_reorient/1656_m05_landscape_winner.mp4) — m05's clean
deterministic reorient on its own.

**m05's design:** thumb (+.015, +.005, +.011), index (+.004, +.002, +.012),
middle (+.025, +.024, +.016) — it **lengthens all three fingers and moves the middle outward**.
Notably this is the *opposite* of the hand-picked thumb-only reposition that failed: the lever
turned out to be finger *length* + middle placement, not thumb opposition.

**Takeaways.**
1. **Morphology is a real, strong lever for reorient** (cos span 1.6). The structural-ceiling
   diagnosis from the pivot was correct — but the fix is not the intuited thumb reposition.
2. **Grasp quality is not a reorient surrogate.** Optimizing the cheap grasp screen would have
   selected `m01` (the worst reorienter). Any morphology search must score on a **reorient
   rollout**, exactly as the plan argued.
3. **Balanced-grasp and articulable-grip are distinct geometric goals** — a too-symmetric grip
   can't be rolled with bounded residuals.

**Caveats / next:** `m05` is a **single training seed** and a **firm** grip (~11 N, not the
low-force goal); the project's smooth+low-force+balanced objective is only partly met (smooth ✓,
balanced ✓, low-force ✗). Next: re-seed `m05` to rule out a lucky run, then a local CEM refine
around it (Stage 2 of `morphology_optimization_plan.md`) scored on the reorient rollout, with
seating (palm contact) as the remaining low-force unlock.

---

## Phase: the FAIR per-morphology A→B pipeline — two artifacts corrected, first co-designed handoff (2026-07-01)

The landscape (above) had two confounds flagged on review: (1) it warmstarted the *baseline*
reorienter (B4) onto every design, and (2) it scored REORIENT on a **skip-lift teleport grip**
— Policy A never ran, so a design's "reorient quality" was never conditioned on whether a real
A could *achieve* its grip. Rebuilding the pipeline honestly on the winner (`m05`) surfaced two
**measurement artifacts** that overturn parts of the landscape read, and produced the first
genuine pickup→reorient handoff on a co-designed hand.

### Artifact 1 — graspability was gated by JOINT-SPACE keyframe transfer (not geometry)

Training a *native* Policy A on `m05` failed repeatedly: A never lifted (object pinned on the
floor, watchdog-killed). Root cause: the `open_short_manual` keyframe is **joint-space**, so on
`m05`'s repositioned/lengthened thumb the same joint angles put the fingertip at the wrong
**world** position — CEM seeded the thumb off-target and scored `m05` a "2-finger design"
(thumb contact persistence **0.0** across two CEM budgets).

**Fix — IK-retarget the keyframe in WORLD space** (`scripts/retarget_keyframe_ik.py`, plain
`mj_jacBody` damped-least-squares, no mink): read the baseline's 3 fingertip world XYZ, IK each
finger of the target morphology to those positions (<0.1 mm residual), inject an `open_ik`
keyframe, re-CEM from it. `m05`'s thumb persistence went **0.0 → 1.0** — a full persistent
tripod. Re-running this across the whole landscape (`scripts/ik_recem_landscape.py`):

![IK re-CEM graspability](img/ik_recem_graspability.png)

**10 of 12 designs form a full 3-finger tripod once the keyframe is retargeted — including all
three the landscape had declared "ungraspable" (m02/m04/m09).** The IK residual cleanly flags
the two genuine exceptions (m06/m08, fingertip physically can't reach the contact point). **The
landscape's graspability verdicts were overwhelmingly keyframe-transfer artifacts**; any
cross-morphology grasp comparison must IK-retarget per design first. See
[[feedback_ik_keyframe_retarget_across_morphologies]].

### The honest per-morphology pipeline (native A, two-phase B, live handoff)

With `m05`'s real 3-finger grip in hand, the fair pipeline is: **native A** (lift) → **live-A
reset B** (reorient from A's *organic* delivery, no teleport) → **continuous handoff eval**.

- **Native A_m05** (`scripts/train_A_on_morph.sh`, from **scratch** — the baseline `a01`
  warmstart is baseline-specific and *ejects* the tighter 3-finger grip; residual≈0 from scratch
  preserves the working open-loop lift). Result: stable lift to 0.062 m, tip_lost ~5, **zero
  drops**.
- **B_m05 — two phases** (the single-shot reorienter warmstart FAILS: it is a
  *reorient-from-step-0* policy, so on takeover it rotates immediately, disrupts A's grip, every
  episode dies at the seam via tip_lost → PPO advantage-normalisation over the near-empty kept
  set explodes the actor std → NaN crash. This is exactly the v3b lesson: **warmstart a
  hold-first policy, not the reorienter.**):
  1. **Phase 1 hold-only** — warmstart B **from A_m05 itself** (A *is* a holder, so B takes over
     seamlessly), reorient reward off. B learns to hold A's live delivery: tip_lost **3072 → ~22**,
     object held at 0.082, A-driven frac 1.0 → 0.36.
  2. **Phase 2 reorient** — finetune the hold-only ckpt with the target-axis reward on. B
     reorients *while holding*: alignment 0 → ~40, tip_lost → ~12, no collapse.
  (A robustness guard was added to `live_a_runner._mask_pre_onset_advantages` so a near-empty
  trainable set zeroes advantages that iter instead of NaN-crashing the run.)

### First genuine pickup→reorient handoff on a co-designed morphology

Deterministic continuous handoff (`rl_demo_handoff_continuous.py --lift-delta 0.05`, A drives
0..40 then B, no reset). **The seam is clean** — B takes over at cos 0.89 / z 0.066 with no
collapse (contrast: every prior attempt, incl. the baseline lineage, dropped *at* the seam,
min-z 0.003–0.008). B then **holds and reorients to cos 0.94 (peak 0.987) for ~120 steps.** The
remaining failure is a **late de-centering slip** (lateral drift creeps up, grip migrates off,
object drops ~step 190) — a qualitatively milder, different problem than the seam drop.
Video: [handoff_m05_continuous.mp4](videos/20260701_reorient/1411_handoff_m05_continuous.mp4).

### B→A co-refinement — moving A (by B's downstream reward) reduces the slip

The never-before-built lever: instead of a hand-crafted grip-match reward, train **A** by B's
**actual downstream reorient reward**. Implemented as an inverted live-A mode
(`live_a_runner` `drive_post`: A is the learner driving the lift 0..onset; a **frozen** B drives
the reorient; A's lift steps get discounted downstream credit for B's reorient reward via GAE;
low fixed LR = a gentle "slow" nudge). `scripts/train_corefine_BtoA.sh`. One 15M-step pass,
then re-eval the handoff with the co-refined A:

![m05 handoff co-refine](img/m05_handoff_corefine.png)

| post-handoff metric | frozen A | **co-refined A** |
|---|---|---|
| lateral drift @ step 160 | 3.3 cm | **1.2 cm** (~3× less) |
| object-z @ step 200 | 0.012 (dropped) | **0.047 (held)** |
| cos sustained to | ~step 160 | **~step 200** (peak 0.966) |
| held-cos (last-50 mean) | 0.022 | **0.235** |
| drop onset | ~step 190 | **~step 210** |

**Moving A — trained purely by B's downstream reorient signal — cut the de-centering drift ~3×
and extended the hold ~30 steps**, directly attacking the exact failure mode. It does not yet
eliminate the eventual drop; a longer / alternating co-adaptation (move A, refreeze, refinetune
B, repeat) is the natural next step. Video:
[handoff_m05_corefined.mp4](videos/20260701_reorient/1441_handoff_m05_corefined.mp4).

**Takeaways.**
1. **Cross-morphology grasp transfer must be world-frame (IK), not joint-space** — the single
   biggest correction; it flips the landscape's graspability map.
2. **Per-morphology A→B is buildable and the seam is closable on a new design** via native-A +
   two-phase (hold-first) live-reset B — the documented baseline playbook transfers.
3. **B→A gradient co-refinement works** (first implementation): A's lift is genuinely optimizable
   for downstream reorient quality, measurably reducing de-centering. This is the "slow gradient
   updates from B to A" lever, distinct from the earlier grip-match proxy.
4. Open: the **late de-centering slip** (both A variants eventually drop ~step 210) and the
   low-force goal (grip still firm) remain; alternating co-adaptation + a lateral-drift term are
   the next levers.

### ⚠️ CORRECTION (2026-07-02): the lift was a LATE-FINGER 2-finger grasp; "cos 0.94" was degenerate

Reviewing the handoff videos, the user caught what every aggregate metric (tip_lost, object-held,
mean reward, held-cos) had **masked**: the initial grasp was a **2-finger grip with a LATE third
finger**, unstable throughout, and the whole handoff was **jittery**. Root cause — *not* the IK
(the IK grip is a valid persistent tripod, verified open-loop) — but a **downstream env bug**: the
RL env reset the fingers to the hardcoded **baseline** `open_finger_qpos` (thumb mcp = 3.14, flung
out) for *both* the reset pose and the LerpFinger interpolation start, while CEM optimized the grip
resetting to the **`open_ik`** keyframe. So the LerpFinger closed from the wrong open pose and the
repositioned thumb/middle arrived late.

**Fix — `open_finger_from_keyframe`** (`env_cfg.py` → `rl_train_cube.py` → the handoff eval;
opt-in, baseline lineage byte-identical): start the fingers (reset AND LerpFinger start) from the
keyframe angles, matching CEM. Verified by a zero-policy scripted rollout: old open pose = **2
contacts at step 20 → drop**; fixed = **3 contacts from step 20, held**. Re-trained native A_m05
(`policyA_m05_ikopen`) is now a genuine holder: **all three fingers loaded (touch 1.00), 0 cm
drift, held 195 steps**.

### Trajectory-health monitoring — baked in, so this can't hide again

The deeper lesson (user): our metrics should have flagged this. New
`src/morphohand/rl/trajectory_health.py` turns a logged rollout into an explicit PASS/WARN/FAIL
scorecard for the degenerate patterns that reward hid: **late_finger** (per-finger first-contact
spread), **idle_finger** (2-finger / degenerate pinch), **drop**, **jitter** (object angular
jerk), **de_centering** (net drift + slide ratio), **over_clamp**. It is **baked into the handoff
eval by default** (writes a `.health.json`), available standalone as `scripts/policy_healthcheck.py`,
and **auto-runs at the end of `train_A_on_morph.sh`** as an acceptance gate.

Characterizing the m05 policies with it (![health](img/m05_health_characterization.png)):

| check | A_m05 DEFECTIVE | **A_m05 FIXED** | A→B "cos 0.94" (def lineage) |
|---|---|---|---|
| late_finger | borderline (spread 12) | ✓ **PASS (spread 1)** | ✗ FAIL (spread 14, mid@20) |
| idle_finger | ✗ FAIL (mid 2.4 N/0.35) | ✓ **PASS (all loaded)** | ▲ WARN (2.45 contacts) |
| drop | ✗ FAIL (0.049) | ✓ **PASS (0.055)** | ✓ PASS |
| jitter (ang-jerk) | ✗ FAIL (46) | ✓ **PASS (6.2)** | ✗ **FAIL (156!)** |
| verdict | **FAIL** | **WARN** (firm + micro-slide only) | **FAIL** |

**The previously-"recommended" cos-0.94 handoff is degenerate** (late finger + violent jitter 156
+ 2-finger) — its high held-cos hid all of it. Only the fixed A is genuinely clean. **All m05 B /
handoff / co-refinement numbers above (trained on the defective lift) are superseded**; the
pipeline is being re-run on the fixed A and every policy now passes through the health gate before
being called a result.

### The CLEAN, health-gated m05 pipeline (2026-07-02) — the real result

Re-running the full pipeline on the fixed A, plus one more fix found by *characterizing* the B
collapse (not reward-tuning): the earlier B's collapsed because reorienting the **8 cm** screwdriver
from a **0.05 m** lift swings its far end into the floor. **Delivering at 0.10 m** (the clearance
the baseline reorient lineage always used; `train_A_on_morph.sh` had hardcoded 0.05) gives the room:

- **A_m05 @ 0.10** (`policyA_m05_ik10`, from scratch + `open_finger_from_keyframe`): health gate
  PASS — instant 3-finger (thumb@1/index@0/mid@1), all loaded, held **0.123 m**, jitter 21.
- **B_m05 reorient** (`policyB_m05_reorient_ik10`, live-A reset, warmstart the hold-first A): held
  0.126 m, alignment ~60, tip_lost ~4 — no collapse (vs the 0.05-lift runs that dropped to floor).
- **Continuous A→B handoff** — the honest, health-gated result:

  | metric | "cos 0.94" (def lineage, SUPERSEDED) | **CLEAN m05 (this)** |
  |---|---|---|
  | post-handoff min-z | 0.008 (dropped ~step190) | **0.121 m (HELD whole rollout)** |
  | held-cos (last-50) | — | **0.898 (peak 0.911)** |
  | late_finger | ✗ FAIL (spread 14) | ✓ **PASS (spread 1)** |
  | idle_finger | ▲ WARN (2.45) | ✓ **PASS (9.7/5.7/6.4 N, all touch 1.00)** |
  | jitter (ang-jerk) | ✗ **FAIL (156)** | ✓ **PASS (9.6, 16× smoother)** |
  | drop | ✓ | ✓ **PASS (0.127)** |
  | **verdict** | **FAIL** | **WARN** (firm grip + micro-slide only) |

  Video: [handoff_m05_FIXED.mp4](videos/20260702_reorient/1431_handoff_m05_FIXED.mp4). This is the **first
  health-gated genuine pickup→reorient on a co-designed morphology**: instant balanced 3-finger
  grasp, held aloft the whole rollout, reoriented to cos ~0.90, smooth (jitter 9.6). The remaining
  WARNs are the known firm grip (~7 N, the low-force goal) and a benign micro-slide.

- **B→A co-refinement on the CLEAN pipeline** (`corefine_BtoA_m05_fixed`, `drive_post`, low-LR nudge
  of A by B's downstream reorient reward) — now a real validation (the earlier co-refine "gain" was
  on a degenerate policy). It **improved every axis while staying health-clean**: held-cos
  0.898→**0.974** (peak 0.981), post-handoff min-z 0.121→**0.139**, jitter 9.6→**5.9**, net drift
  0.8→**0.2 cm**, still 4/4 hard checks PASS. Video:
  [handoff_m05_FIXED_corefined.mp4](videos/20260702_reorient/1459_handoff_m05_FIXED_corefined.mp4). Confirms A's
  lift is genuinely optimizable for downstream reorient quality via gradient — the "slow gradient
  updates B→A" lever, validated on a healthy base.

**The three fixes that mattered — all found by characterizing failures, none by reward-tuning:**
(1) IK-retarget the keyframe (world-frame fingertips) → real 3-finger grip; (2)
`open_finger_from_keyframe` → close from the right open pose (no late finger); (3) deliver at 0.10
→ floor clearance for the long object to reorient. Each was invisible to aggregate reward and
visible on video / to the health scorecard — hence the monitoring is now baked in by default.

---

## Phase: co-design morphology sweep on the CLEAN pipeline (2026-07-03)

Goal (user): explore morphologies with the blessed policy (`handoff_m05_FIXED.mp4` = **a10** native
lift → **b33** live-A reset reorient) — a full per-design A→B pipeline, health-gated, no cheap
skip-lift proxy. Orchestrator `scripts/morph_pipeline_sweep.py` (resumable; per design: generate →
IK-retarget `open_ik` → CEM grasp gate → native A → live-A-reset B → continuous handoff +
trajectory-health scorecard). Analysis `scripts/morph_pipeline_plots.py`. Runbook +
resume/monitor commands: `morph_sweep_STATUS.md`.

### initial8 (from-scratch A, warmstart B from A) — the honest result is FRAGILITY

Eight interpretable coordinate moves around m05 (`s00` m05-anchor, `s01` m00-baseline ref, `s02–s07`
thumb-opposition / seating hypotheses; exact Δ in `docs/experiments/MORPH_PIPELINE_initial8_TABLE.md`). Summary +
training figures: `img/morph_pipeline_initial8_summary.png`, `..._training.png`.

| design | grasp (CEM lift/persist) | A→B outcome | verdict |
|---|---|---|---|
| **s05_shortgrasp** (index+mid len−4mm) | 0.054 / 1·1·1 | held cos 0.73, min-z 0.11, **jerk 47, force 12** | FAIL (jitter+clamp) |
| s02_thumbreach (thumb_x+6mm) | 0.054 / 1·1·1 | idle fingers, dropped (cos −0.1, drift 16cm) | FAIL |
| s00_m05anchor (= m05) | 0.054 / 1·1·1 | **A collapsed at iter 55** (0.127→0.026 in one iter) | A abort |
| s01_baseline | 0.051 / 1·1·1 | B held 0.147 to iter 204 then collapsed | B abort |
| s03/s04/s06/s07 | ~0.05 / 1·1·1 | B never lifted (A-driver can't lift perturbed grip) | B abort |

**Two load-bearing findings:**
1. **The 9-param designs are grasp-EQUIVALENT under the honest pipeline.** IK-retarget gives *every*
   design a persistent 3-finger tripod (CEM lift ~0.05, persistence 1.0/1.0/1.0). Graspability is
   **not** the differentiator here — contrary to how the 2026-06-25 skip-lift landscape read (which
   was confounded by the joint-space keyframe artifact).
2. **The bottleneck is RL-training ROBUSTNESS, not morphology.** 6/8 aborted in A or B training, and
   critically **the m05 anchor `s00` collapsed in A training** — the from-scratch A lifted cleanly to
   0.127 m through iter 54 then catastrophically dropped to 0.026 in a single iteration. This
   **confirms the standing caveat that a10/m05 was a lucky single seed**: from-scratch PPO on this
   contact-rich lift is seed-fragile, and a fresh per-design A frequently produces a policy that
   can't reliably lift (so the live-A driver drops the object and B never learns). The one design
   that held+reoriented (`s05`) fails only on **jitter/over-clamp — a policy-quality issue, not a
   design flaw**.

### Root-causing the failures (the first "fix" was a misdiagnosis — kept as a lesson)

Attempt 1 warmstarted **A from a10** ("adapt the blessed policy"). It **FAILED on the m05 anchor**:
A never lifted (object 0.0 from iter 0). This reproduces the *documented* reason A must train from
scratch — a warmstarted A loads a **grip-specific residual that EJECTS the re-CEM'd object**; only
from scratch is the residual ≈ 0, so the open-loop CEM grip + scripted lift does the lifting. So
**warmstart-A is wrong**; the real failures had two separate causes, found by reading configs/logs:

1. **B drops (`s03/s04/s06/s07`) = a genuine ORCHESTRATOR BUG, not a bad design.** Those designs'
   A's trained fine (lift 0.11, clean). But the B config showed **`open_finger_from_keyframe:
   false`** vs the good b33's `true` — the live-A-reset script never passes
   `--open-finger-from-keyframe`, so B reset to the **baseline flung-out-thumb open pose**, the
   LerpFinger closed from the wrong pose → wrong grip → the live-A driver dropped the object → B
   never learned (and NaN'd). Exactly the "late-finger/wrong-open-pose" bug the m05 work already
   fixed, silently reintroduced in the sweep's B wiring.
2. **A/B mid-training collapse (`s00` A @ iter 55, `s01` B @ iter 205) = from-scratch PPO
   instability**, not morphology. Both had a *healthy earlier* checkpoint (A model_50 @ 0.127; B
   model_200 @ 0.147) that the watchdog-abort discarded.

**The correct fix (in the orchestrator):** from-scratch A **+ use the FINAL A checkpoint on clean
completion** (salvage the best pre-collapse ckpt ONLY on a watchdog abort — an early model_50 lifts
marginally higher but has an under-refined grip → idle finger; `valfix2` FAILed exactly this way,
picking model_50 over model_609) **+ pass `--open-finger-from-keyframe` to B + salvage-eval the last
B checkpoint on abort**.

**Validation (m05 anchor, corrected pipeline):** `valfix3` (A model_609 + B open-finger) →
**WARN**, all three fingers loaded (16.8/10.5/9.6 N, idle-finger PASS), held aloft (min-z 0.111),
smooth (jitter 15) — the same health class as b33. Reorients less fully (cos 0.66) and grips
firmer (12 N) than a10→b33 (cos 0.90, 7 N) — **from-scratch seed variance**, the expected cost of
per-design retraining. Genuine, non-degenerate pickup→reorient → the pipeline is sound for ranking
designs. **Larger sweep launched:** 16 local designs around m05 (`--morph-set local --n 16 --center
m05 --seed 1`, `docs/experiments/MORPH_PIPELINE_large16.*`).

### large16 results — a clean map, but SEED VARIANCE dominates the ranking

All 16 ran (design 0 = m05 anchor). **14/16 held the object and reoriented — no drops**; 1
(`L01_03`) had a from-scratch A collapse (never lifted, auto-gated in 8 min, B skipped); 1
(`L01_05`) reoriented the wrong way. Every design is again grasp-equivalent (CEM 0.05 / 1·1·1).
Ranked by held-cos (figures `morph_pipeline_large16_summary.png` / `_training.png`, table
`docs/experiments/MORPH_PIPELINE_large16_TABLE.md`):

| design | Δm05 (biggest) | held-cos | force N | jerk | verdict | note |
|---|---|---|---|---|---|---|
| **L01_06** | ~m05 (all <5mm) | **0.899** (peak 0.984) | 11.9 | 23 | WARN | best cos — but ≈ m05 geometrically → likely seed |
| L01_00 (m05 anchor) | — | 0.779 | 10.9 | 12.5 | WARN | the in-sweep reference |
| **L01_13** | **thumb_x +9mm** | 0.761 | **7.4** | **6.0** | WARN | m05-level cos at LOWER force + HALF the jerk — best design lead |
| L01_10 | thumb_y −8mm | 0.587 | 9.3 | **5.0** | WARN | smoothest |
| L01_01 | index_x −6mm | 0.511 | **6.0** | 6.2 | WARN | low-force + smooth, under-reorients |
| L01_02 | — | 0.385 | **2.0** | 13 | **FAIL** | lowest force but by IDLING a finger (degenerate — scorecard caught it) |
| L01_05 / L01_09 | — | −0.45 / −0.40 | 5.5 / 13.5 | — | FAIL/WARN | reoriented the wrong way |

**Findings:**
1. **Seed variance dominates — no design is a CONFIRMED winner.** m05 alone spans held-cos **0.66
   (valfix3) → 0.78 (L01_00) → 0.90 (L01_06 ≈ m05)** across seeds — a ~0.24 spread from *seed
   alone*. So L01_06's 0.90 is within m05's own seed band, not a design win. This quantifies the
   plan's "re-seed m05 to rule out a lucky run" caveat: **single-seed per-design scoring cannot
   separate design effect from seed luck for held-cos.**
2. **The health monitor earns its keep:** `L01_02` posts the lowest force (2.0 N, toward the goal)
   but is **degenerate** — it FAILs `idle_finger` (low force by unloading a finger). Aggregate
   "low force" would have called it a win; the scorecard correctly rejects it. Genuine *balanced*
   low force bottoms out ~6–7 N (`L01_01`/`L01_13`), consistent with the standing ~6.6 N floor.
3. **Best design lead = `L01_13`** (`thumb_x +9 mm`, a real thumb reposition toward opposition):
   m05-level verticality (0.76) at **lower force (7.4 vs 10.9) and half the object jerk (6.0 vs
   12.5)**. If replicated across seeds, this is a smoother, lower-force design at equal verticality.
   Videos: [L01_13](videos/20260704_sweep/2015_L01_13_handoff.mp4),
   [L01_06](videos/20260704_sweep/0902_L01_06_handoff.mp4),
   [L01_00 (m05)](videos/20260704_sweep/0141_L01_00_center_handoff.mp4).

### Multi-seed confirmation — NEGATIVE: seed variance swamps any local design effect

Re-ran `L01_13` and `m05` ×3 fresh seeds each. Pooling with the earlier runs of each exact vector
(`docs/experiments/MORPH_PIPELINE_confirm.*`; figure `img/morph_confirm_seedbands.png`):

| design | held-cos (mean ± sd, range, n) | force N | jerk |
|---|---|---|---|
| **m05** | **+0.32 ± 0.38, [−0.29, 0.78], n=5** | 10.5 ± 4.0 | 17.6 ± 8.0 |
| **L01_13** | **+0.38 ± 0.44, [−0.36, 0.76], n=4** | 8.5 ± 3.5 | 14.6 ± 9.7 |

**The held-cos gap is 0.07 against a pooled seed sd of 0.41 → `L01_13` is statistically
INDISTINGUISHABLE from `m05`** on every axis (cos, force, jerk). The large16 "L01_13 lead" (0.76 /
7.4 N / jerk 6.0) was **one lucky draw** — its other seeds gave 0.63 / −0.36 / 0.51 and force 11.8 /
3.3 / 11.4. The same design reorients anywhere from *backwards* (−0.36) to *near-vertical* (0.76)
depending only on the training seed.

**Definitive conclusion (load-bearing).** Under the honest per-design from-scratch A→B pipeline,
**reorient quality has enormous run-to-run seed variance (held-cos sd ≈ 0.4, spanning negative to
0.8) that completely swamps any local morphology design effect.** No design in the m05 neighborhood
is distinguishable from m05. **The bottleneck is the RL training's seed-sensitivity, not the
9-param geometry** — consistent with the earlier finding that every design is grasp-equivalent.
Single-shot (even 3-shot) per-design scoring **cannot** resolve local design differences here.

**Implication for co-design.** Before any design search over this neighborhood can find a real
winner, the *evaluator's variance must be reduced* — e.g. (a) many seeds per design + report the
mean (expensive: the variance needs ≥5–10 seeds to pin a 0.1 cos difference); (b) a
variance-reduced / more-stable reorient trainer (fix the seed-fragility — warmstart a *shared*
reorient prior rather than each design's own noisy A); or (c) score on a cheaper, lower-variance
proxy than a full from-scratch A→B rollout. m05 remains the reference design; **the sweep's real
deliverable is this methodological finding + the health-gated pipeline + the honest variance
characterization**, not a new winning morphology. No design was promoted.

### Variance reduction — SOLVED: we can now delineate morphologies (2026-07-06)

The user asked to cut the evaluator variance so designs can be told apart. A controlled study
(`scripts/reorient_variance_study.py`): 2 designs (m05, L01_13) × 2 B-warm-start modes × 3 re-runs,
with a **fixed A per design** (isolating B's variance). Figure `img/variance_reduction_bands.png`.

| condition | held-cos | force (N) |
|---|---|---|
| m05, vary-A + self (the old pipeline) | 0.32 ± **0.38** | — |
| m05, **fix-A** + self | 0.78 ± **0.09** | 7.3 ± 0.8 |
| m05, **fix-A + shared** (warmstart b33) | 0.86 ± **0.04** | 8.5 ± 2.3 |
| L01_13, fix-A + self | 0.76 ± 0.03 | 10.2 ± 1.6 |
| L01_13, fix-A + shared | 0.72 ± 0.03 | 11.5 ± 0.6 |

**Two levers, both large. (1) Fixing A cut the cos sd ~4×** (0.38 → 0.09) — most of the "seed noise"
was Policy A's from-scratch collapse/variance, not B. **(2) Shared-warmstart-B (from the proven
reorienter b33) halved it again** (0.09 → 0.04) and raised the mean (every seed 0.81–0.91). Net
**~8× tighter** — small enough to resolve 0.1-scale design effects.

**The delineation, and a reversal.** With tight bands, m05 vs L01_13 is now testable:
- **Reorient cos (fair, *self* mode):** 0.78 vs 0.76 — gap 0.02 < pooled sd 0.058 → **equivalent**.
- **Force (both modes):** m05 ~7–8 N vs L01_13 ~10–12 N — gap ≈ 3 N ≫ noise → **separable, m05 is
  genuinely LOWER-force.**
- Under *shared* mode cos also separates (m05 0.86 vs 0.72), but that is **confounded**: b33 is
  m05's OWN reorienter (in-distribution for m05, OOD for L01_13). The fair comparison is *self*.

So L01_13's single-seed "lead" (lower force + smoother at 0.76) was **entirely a seed artifact** —
measured properly, m05 is equivalent-or-better and lower-force. This both **validates m05 as the
reference** and is the sharpest possible warning against 1-shot per-design scoring.

**Recommended evaluator for future design search:** **fixed-A + self-warmstart-B + ~3 seeds**
(±0.03–0.09 bands, no design-specific prior bias). Shared-warmstart is a stronger variance-reducer
but needs a *design-neutral* reorient prior (not one of the candidates) to be fair — which is
exactly the role a morphology-transferable **object-relative fingertip imitation** prior plays:

### Object-relative fingertip imitation (#3) — the best evaluator, and it transfers

Record the blessed a10→b33 reorient's **object-frame** fingertip trajectory once
(`--record-fingertip-traj` → (200,3,3)), then imitate it on any design via a `track_fingertip_obj`
reward (`src/morphohand/rl/imitation.py`) with a weight curriculum (learn the motion, then let the
task refine). Trained on m05 + L01_13 warm-starting each design's own holder-A (so the reorient
skill comes from the *demo*, not a policy), n=3 each:

| design | mode | held-cos | force N | jerk |
|---|---|---|---|---|
| m05 | self / shared / **imit** | 0.78±0.09 / 0.86±0.04 / **0.82±0.02** | 7.3 / 8.5 / **7.1** | 11 / 26 / **10** |
| L01_13 | self / shared / **imit** | 0.76±0.03 / 0.72±0.03 / **0.72±0.02** | 10.2 / 11.5 / **8.2** | 9 / 30 / **6** |

**Imitation is the best variance-reducer AND produces the best policies:** the **tightest bands
(±0.02 on both designs)**, and — unlike shared-b33 (jerky 26–30, and m05-biased since b33 is m05's
own) — it is **smooth (jerk 6–10) and lower-force** (one m05 run: cos 0.85 at 4.5 N / jerk 6.7, the
nicest policy in the whole study). Crucially the **object-relative trajectory TRANSFERS**: it lifts
L01_13's smoothness/force too, yet under this *fair, design-neutral* prior **m05 (0.82) still beats
L01_13 (0.72)** — gap 0.10 ≫ pooled sd 0.02, *separable*. So m05's geometry genuinely reorients
better; L01_13 never wins on any fair footing. Figure `img/variance_reduction_bands.png`.

**Bottom line for the whole co-design effort:** the seed variance that made designs
indistinguishable is *solved* (fix-A cuts it 4×; a design-neutral imitation prior cuts it to ±0.02
and yields smooth/low-force policies). Under proper measurement **m05 (a10→b33) is validated as the
reference design**, and the imitation prior is the recommended evaluator + warm-start for any future
design search. Study: `scripts/reorient_variance_study.py`, data `docs/experiments/REORIENT_VARIANCE.{json,txt}`.

---

## Phase: sim-to-real contact hardening (2026-07-07)

The task's contacts were deliberately soft (geom `solimp="0.97 0.995"`, env `impratio=10`/elliptic)
for a stable, non-explosive grip — at the cost of some fingertip interpenetration. With the policies
validated, the deferred sim-to-real pass: make the contact **slightly harder** (more realistic, less
penetration) and see what survives. Step: `solimp 0.97 0.995 → 0.985 0.999` (dmax 0.995→0.999 permits
~10× less penetration by construction), `solref` held at 0.006 (0.003 hits the 2·dt stability edge).
Infra: a hardened `frozen_scene.xml` in a copied morphology-run (`--frozen-scene-xml` in the trainer);
`assets/mjcf/experimental/sim2real/`.

**Finding — the grasp transfers, the reorient does NOT (contact-compliance-dependent).**

| condition | held (min-z) | reorient (held-cos) | force |
|---|---|---|---|
| soft baseline (a10→b33) | 0.130 (held) | 0.86 | 6.7 N |
| **hard, zero-shot** (soft policy) | **0.008 (DROPPED)** | −0.43 | — |
| **hard, RETRAINED** (A scratch + B imitation, n=2) | **0.117 (HELD, no drop)** | **−0.22 (peak 0.16)** | 9–10 N |

- **Grasp/lift/hold robustly transfers.** Zero-shot, the soft-tuned policy *drops* the object (it
  leaned on compliance); but a **from-scratch retrain under the harder contact recovers a clean,
  balanced, held grasp** (A health WARN, all 3 fingers loaded, no drop). The lift is not
  compliance-dependent.
- **In-hand reorient is CONTACT-SENSITIVE, with a bounded tolerance.** At the harder step
  (`solimp 0.985 0.999`) the retrained B **holds but barely rotates** — `target_axis_alignment`
  **13 vs 48** soft, held-cos ≈ 0 — even with the imitation prior guiding the finger motion. The
  reorient-by-*rolling* of a smooth cylinder relies on contact compliance (micro-sliding lets the
  fingers roll it); a stiffer, high-friction contact grips it too rigidly to roll. But at a
  **gentler step (`solimp 0.98 0.997`) the reorient re-learns** (train-time alignment **52.8 ≈ soft**),
  so the breaking point is between dmax **0.997 (learnable)** and **0.999 (breaks)** — there *is* a
  "slightly harder" contact the full task tolerates. (The first gentle probe's eval was unstable
  from a driver mismatch — hard-A driving a gentler env; a **clean matched A+B retrain at 0.98/0.997**
  confirms the end-to-end result.)
- **Next levers (for pushing reorient stiffer):** (a) a **contact-stiffness curriculum** (soft→hard
  anneal, or DR over `solimp`) so the reorient adapts gradually past dmax 0.997; (b) a
  **gaiting/regrasp** reorient (rotate by release-reposition-regrip, compliance-independent); (c)
  higher fingertip **friction / deformable pads** to enable rolling at higher stiffness. Videos:
  `docs/rl/videos/reorient/sim2real/`.

### Compliance-robustness sweep — trained policies are FRAGILE to contact stiffness

The single-stiffness retrains above were seed-noisy (a gentle-step B trained to alignment 52.8 in
one run, 14.8 in another). The clean way to ask "how robust is a policy to compliance" is
*eval-only*: fix a trained policy and sweep the contact stiffness. `scripts/compliance_robustness_sweep.py`
evaluates each policy across `solimp` dmax 0.995→0.9995 (soft→hard); figure
`img/compliance_robustness.png`, data `docs/experiments/COMPLIANCE_ROBUSTNESS.txt`.

| policy \ dmax | 0.995 (soft) | 0.997 | 0.998 | 0.999 | 0.9995 |
|---|---|---|---|---|---|
| **soft b33** (held-cos) | **0.94** | −0.19 ✗drop | **0.90** | −0.27 | −0.0 ✗drop |
| **soft imitB** | 0.84 | 0.30 | 0.61 | −0.0 ✗drop | −0.0 ✗drop |
| **hard-retrained** | −0.0 ✗drop | −0.24 | −0.02 | 0.29 | −0.0 ✗drop |

**Findings:**
- **No policy is robust across compliance.** `soft_b33` reorients at 0.995 *and* 0.998 but **drops at
  0.997 and fails at 0.999** — the response is **non-monotonic / contact-mode-sensitive**, not a
  smooth "harder = worse." Small stiffness changes flip success ↔ failure. This is a real sim-to-real
  robustness risk: a policy tuned to one contact model can fail at a nearby one.
- **Training at a single stiffness buys no broad robustness.** `hard_retrained` (trained at 0.999)
  doesn't reorient anywhere *and drops at soft* — it overfit its training contact.
- **Imitation degrades most gracefully** (`soft_imitB`: monotone-ish 0.84→0.30→0.61, holds to 0.998),
  a mild robustness edge — consistent with imitation instilling a more canonical motion.
- **The clear fix: train with COMPLIANCE DOMAIN RANDOMIZATION** (sample `solimp` per-env over a range)
  so the policy sees the whole band and becomes robust, instead of overfitting one stiffness. This
  is the recommended next step for sim-to-real, now motivated by a clean measurement.

*Caveat:* one deterministic rollout per (policy, stiffness); the non-monotonicity is a genuine
contact-mode effect, but a rigorous success-*rate* would average over spawn/noise per point.

---

## Phase: compliance DOMAIN RANDOMIZATION — per-env solimp DR built + retrain launched (2026-07-08)

**The fix from the sweep above, implemented as Approach A (true per-env DR) from
`docs/rl/compliance_dr_plan.md`.** mjwarp stores `geom_solimp` with a per-world leading dim, and
mjlab's DR framework (`mjlab/envs/mdp/dr/`) auto-expands any model field named by a
`@requires_model_fields` event term — so per-env physics DR needed no engine changes.

**Implementation** (`mjlab_terms.randomize_geom_solimp` + `env_cfg.compliance_dr*` +
`rl_train_cube --compliance-dr`):
- Reset-mode event; each resetting env draws ONE softness `u ~ U(0,1)` and lerps **both** solimp
  dmin ∈ [0.97, 0.985] and dmax ∈ [0.995, 0.999] from it. The **joint** draw keeps (dmin, dmax)
  correlated/ordered — independent draws could pair a soft dmin with a hard dmax, a regime the
  sweep never validated. Writes ALL geoms (matches the whole-scene sed the sweep evaluates with);
  solref stays 0.006 (0.003 sits on the 2·dt stability edge).
- **Verified by probe** (scratchpad `probe_compliance_dr.py`): `geom_solimp` expanded to
  (num_envs, ngeom, 5); per-env values distinct, in-range, rank-correlated; envs re-draw on
  mid-run resets; 20 steps finite. SMOKE A-train (1M ts) clean — `randomize_compliance`
  registered, object held ~0.111 m.

**Retrain launched** (`scripts/compliance_dr_pipeline.py`, detached, resumable, logs under
`logs/compliance_dr/`): (1) m05 A from scratch + DR (morph-pipeline recipe), (2) 2× imitation-B
seeds off that A via live-A reset + same DR (imitation = the most stiffness-graceful base),
(3) re-run the compliance-robustness sweep — `policies()` now auto-picks-up `*_m05_cdr*` runs as
`cdr_imitB_k{0,1}`. **Success bar: a FLAT, high held-cos curve across dmax 0.995→0.999** vs the
fragile single-stiffness curves above.

### RESULT (2026-07-09): hold is now robust; reorient quality is not — two separable failures

Pipeline completed clean (A `20260708-1509-policyA_m05_cdr` obj-height 0.115; both B seeds
model_270, no watchdog aborts; B's final `target_axis_alignment` 33–40 vs 49 for the known-good
non-DR imit run — the reorient signal is genuinely depressed under DR). Sweep rows
`cdr_imitB_k*` in `docs/experiments/COMPLIANCE_ROBUSTNESS.txt`:

- **WIN — the DR grasp/hold is stiffness-robust.** Wherever the grip lands, post-handoff min-z is
  0.12–0.137 across c997→c9995 — including **dmax 0.9995, where every single-stiffness policy
  drops**. Forces are low (0.1–1 N), de-centering ~1–2 cm.
- **FAIL 1 — DR-A whiffs the grasp at the SOFT EDGE (dmax 0.995) of its own training band:** zero
  fingertip contact ever, object slides 32.6 cm away (both seeds identically; it's the shared A).
  Cross-pairing confirms: **DR-A + b33 @ soft still drops** (min-z 0.0065), so A owns it. Reading:
  the per-env-average PPO objective sacrificed the band edge; eval-soft must be *interior* to the
  training band (classic DR practice), or the A gate needs a per-stiffness eval.
- **FAIL 2 — DR-B holds but cannot STABILIZE at vertical:** from a good delivery
  (**a10 + DR-B @ soft: held 0.127, peak cos 0.759 → settles −0.582**) it swings through/near
  vertical and falls past it; across the band held-cos ends ~0 to −0.7 with idle-finger verdicts
  (contact_count ~0.4 — a gingerly, minimal-contact hold). Reading: full-band DR from iter 0
  degrades the hard-exploration rolling gait into "hold gingerly, don't commit"; the annealed-to-0
  imitation prior wasn't enough to preserve the *stabilize-at-vertical* endgame.

**Next levers (in order):** (1) retrain A with the DR band widened so 0.995 is interior
(dmax ∈ [0.993, 0.999], dmin ∈ [0.965, 0.985]) — pure CLI change, launched as `policyA_m05_cdr2`;
(2) for B, keep a nonzero imitation floor (`--imitation-weight-final 12` ≈ 20%) instead of
annealing to 0, and/or (3) a stiffness *curriculum* (narrow-at-soft → widen to full band over
training) so the reorient skill is learned before it is robustified. Caveat as before: one
deterministic rollout per point.

---

## Phase: RATE-based sweep — two measurement artifacts corrected; the DR trade-off measured honestly (2026-07-10)

Round 2 executed the two levers: `policyA_m05_cdr2` (widened band) + `policyB_m05_cdr2_imitfloor_k0`
(recipe `b_liveA_imit` + `--imitation-weight-final 12`, same band). Training looked like a
breakthrough (final `target_axis_alignment` 66–72 vs 33–40 round-1, 49 non-DR). Deploy said
otherwise — and diagnosing the gap exposed that our **evaluator, not just the policies, was broken**:

1. **n=1 deterministic sweeps mis-score marginal policies.** cdr2-A still "failed" at soft — but a
   32-env identical-conditions probe showed the soft grasp is a **knife-edge decided by Warp solver
   noise** (13/32 grasp with the SAME mean actions; video shows the object squirted out during the
   close). And the baseline's famous "non-monotonic" stiffness response (works 0.995+0.998, fails
   0.997) resolves under n=32 into a **smooth fragility gradient** (hold 0.97 → 0.53 → 0.63 → 0.25
   → 0.0) — the non-monotonicity was coin-flip sampling of mid rates. **New standard:
   `scripts/compliance_rate_sweep.py`** (batched N=32 continuous handoffs per point → hold-rate /
   reorient-rate / median held-cos; `docs/experiments/COMPLIANCE_RATE.txt`).
2. **The honest DR verdict (rate curves):** DR **mirrors** the baseline instead of dominating it.
   Baseline a10+b33: hold 0.97 / reorient 0.94 at soft, decaying to 0 at 0.9995. All DR pairs:
   hold 0.22–0.25 at soft rising to 0.94–1.0 at c999/c9995 (where the baseline is dead), but
   **reorient-rate ≈ 0 everywhere at deploy**. Band-widening (cdr2) did NOT fix the soft end.
3. **The training/deploy reorient gap is a DETERMINISTIC-MEAN collapse, B-side edition.** At c999,
   imitfloor-B with mean actions peaks at cos 0.44; with `--stochastic-b` it reaches **peak 0.991**
   — the full rolling gait to vertical lives in the policy but is under-driven by the mean action,
   AND vertical is still never *stabilized* (drops right after the stochastic peak). The imitation
   floor did teach the gait under DR (real progress vs round-1: the skill now exists); the two
   remaining defects are (a) mean-action under-drive, (b) no hold-at-vertical endgame.

**Open next levers:** A-side: soft-biased `u` sampling in `randomize_geom_solimp` (band widening
is proven insufficient). B-side: anneal/pin the action std late in training so competence moves
from the noise into the mean; endgame: a hold-at-vertical bonus window or an imitation reference
that dwells at vertical. Evaluate ONLY via the rate sweep.

---

## Phase: POLICY-BOTTLENECK probes — is the morphology landscape gated by the optimizer? (2026-07-10, launched)

User directive on signing off: the compliance tangent is closed (DR mirrors, doesn't dominate);
back to the core problem — the joint performance×morphology landscape needs **>16 designs**, but
"many of the morphologies sampled never learned at all … even after fixing the initial keyframe
retargeting," so first **validate that the bottleneck is the policy optimizer, not the designs**.
This is the training-time twin of the rate-sweep lesson above (n=1 deterministic evals mis-score
marginal *policies*; here n=1 training draws mis-score *morphologies*).

The existing data already decomposes the noise: confirm gave joint A+B retrain sd 0.41 (m05 itself
spans −0.29..0.78, n=5), while the variance study's fixed-A B-only bands are 0.09 (self) / 0.02
(imit) — **Policy A's from-scratch draw is the dominant evaluator noise**, and indeed *every*
large16 failure had an A-side event (L01_03 iter-0 collapse; L01_05 late-collapse → undertrained
salvage; L01_02/07/09 delivery health-FAIL).

**Probes queued** (`scripts/probe_queue.sh`, detached; plan + decision tree in
`morph_sweep_STATUS.md` §POLICY-BOTTLENECK PROBES): **P1 rescue** — the 5 large16 failures under a
strong evaluator (A best-of-2 with all attempts recorded, then imit-B AND self-B *paired on the
same A*) → H1 failure-flip rate + H3 imitation-prior fairness off-m05; **P2 avar** — raw
(uncensored) A-draw cells av_m05×3 + av_L01_05×2 → H2 per-design collapse / health-FAIL rates and
cos spread. **P4 ready:** `--morph-set global` = Latin hypercube over the full 9-param box — the
honest-pipeline replacement for the 06-25 teleport-proxy global map — to fire once P1/P2 pick the
evaluator. Sweep upgrades: `--a-attempts` (best-of-N A with per-attempt records), `--b-recipe
plain|imit|both`, `--only`, stale-`.COLLAPSED`-sentinel clearing. Monitoring: waiter on the stage
sentinels + **claude-pulse deployed** (cron `*/15`, idle ≥75 min → autonomous session pointed at
the runbook decision tree).

### P1 interim — design 1/5 rs_L01_02: capability flips, quality doesn't; imit holds where self drops (2026-07-10 late)

rs_L01_02 failed large16 via **delivery health-FAIL**. Under the rescue evaluator it is a
**partial flip**:

- **Policy A: accepted on the FIRST fresh draw** (model_609, objheight 0.117, WARN; best-of-2
  short-circuited). The large16 A-side failure did not reproduce — direct evidence the earlier
  verdict was an A-draw event, not geometry.
- **imit-B: genuinely holds and reorients** — post-handoff min-z 0.1115, held-cos 0.561 (peak
  0.678), all three fingertips in contact the whole hold (9–10 N each). Scorecard verdict is
  still FAIL, but on **jitter alone**: ang-jerk 44.1 vs the FAIL bar 40 (b33 reference: 9.6,
  corefined: 5.9); plus WARNs for sliding (path 22 cm vs net drift 1.3 cm) and over-clamp 9.7 N.
- **self-B (same A, paired): DROPS** — post-handoff min-z 0.0062. Its peak-cos 0.999 is
  floor-bracing, and its held-cos tail 0.581 is object-on-floor, not in-hand. FAILs drop +
  jitter + de-centering.

Scoring: **H1 strict flip 0/1** (the flip bar requires verdict ≠ FAIL and jitter blocks it), but
the *capability* claim — "never learned to pick up and reorient" — flipped: held + cos ≥ 0.5 is
1/1. Worth tracking both counts across the remaining 4 designs; if the pattern holds, the
landscape story splits into a **trainability axis** (rescued by a stronger optimizer draw) and a
**quality axis** (jitter/smoothness, possibly genuinely morphology-dependent). **H3 1/1 for
imit**: Δcos(imit−self) = −0.02 is within the −0.05 tolerance, and the honest paired read is
stronger — on the same A, imit held while self dropped, so the m05 prior is not handicapping this
(near-m05) design; it is the difference between hold and drop.

### P1 interim — design 2/5 rs_L01_03: the iter-0 collapse rescues to lift+hold, but reorient never emerges under EITHER B recipe (2026-07-10 night)

rs_L01_03 was the large16 design whose Policy A **collapsed at iter 0** — the cleanest "never
learned" verdict in the set. Under the rescue evaluator (7775 s):

- **Policy A: rescued on the FIRST fresh draw** (model_609, objheight 0.1115, WARN, no collapse;
  best-of-2 short-circuited). That makes it **2/2 designs whose large16 A-side failure did not
  reproduce** on a fresh draw — the iter-0 collapse was draw luck, not geometry. Strong
  H1-direction evidence on the *A/trainability* axis.
- **imit-B: holds but does NOT reorient** — post-handoff min-z 0.1092, all three fingers engaged
  the entire hold (thumb 14.6 / index 17.6 / middle 8.5 N — no idle-thumb pinch), but held-cos
  **−0.047, peak-cos 0.109**: the hand clamps hard and static. Verdict WARN (jitter 31.6,
  over-clamp 13.6 N, sliding path 8 cm vs net 0.2 cm — micro-slip in place, no gait).
- **self-B (same A, paired): also holds, also no reorient** — min-z 0.1126, held-cos 0.174,
  peak 0.309, verdict FAIL on jitter (ang-jerk 114, worst in the sweep so far).

Scoring: **H1 strict flip 0/2** (cos −0.05 ≪ 0.5), and unlike rs_L01_02 the **capability count
also stays 1/2** — lift+hold is rescued but reorientation is absent. Crucially, BOTH recipes fail
to reorient on the same delivered A state, which leans *real design effect* (reorient-hostile
geometry: the object seats into a static full-force clamp that the m05-recorded gait can't roll)
rather than B-seed luck — though at n=1 per recipe it isn't conclusive. This is the first P1
data point *against* pure H1 on the reorient axis, and it sharpens the emerging split: the
**trainability axis flips readily** (A draws), while the **capability/quality axis may be
genuinely morphology-dependent** (L01_02: reorients-but-jitters; L01_03: holds-but-static).

**H3 1/2 for imit-fairness**: Δcos(imit−self) = −0.22 < −0.05, the first strike. Caveats: both
values sit in "no reorient" territory, self-B's edge comes with ang-jerk 114, and a single self-B
draw carries sd ≈ 0.09 — so the Δ is marginal, but per the decision tree, if this pattern
repeats across the remaining designs the P4 evaluator must switch to self-B ×2 seeds.

### P1 interim — design 3/5 rs_L01_05 (in flight): first design where trainability itself resists — A needs its retry AND imit-B collapses (2026-07-10 ~23:45)

rs_L01_05 was the large16 late-collapse design (its old A was salvaged as an undertrained
model_50). Interim state, self-B still training:

- **CEM fine** (lift 0.056, persist 1/1/1) — the grasp gate is not the problem.
- **Policy A: first design to consume the best-of-2 retry.** t0 draw COLLAPSED (watchdog:
  object-height 0.0123 < 0.045 at iter 98); t1 trained clean (kept model_609, objheight 0.1236,
  no abort). Cumulative A-draw record for this design: **2 collapses in 3 draws** (large16
  late-collapse, t0 collapse, t1 clean) vs 0 collapses in 2 first draws for L01_02/L01_03.
- **imit-B: watchdog-aborted** (object-height 0.0275 < 0.03 at iter 194) — the **first B-side
  collapse in the whole program**. Salvage eval of model_150: verdict FAIL on idle_finger —
  a degenerate ONE-finger pin (middle 7.4 N; thumb and index 0.0 N, touch-frac 0) that
  nonetheless keeps min-z at 0.1217; held-cos −0.469, peak-cos 0.001. No reorient, no real grip.
- **self-B (same A) training now**, and struggling at the start: 22 "trainable frac=0.000 →
  zeroing advantages" live-A guard hits in its first ~34 iters (envs terminating around onset),
  vs clean starts on the previous designs.

Interim scoring. **H1 strict: 0/3.** Capability flips: still **1/3** (only L01_02). But the
sharper story is that rs_L01_05 breaks the pattern of designs 1–2: there, A-failures flipped
instantly and only *reorient quality* stayed bad; here **collapse follows the design across
stages and draws** — A t0, imit-B, and (early signal) self-B all struggle to keep the object
off the floor. This is exactly the decision-tree step-5 clause ("L01_05's draws systematically
bad while m05's are fine ⇒ the design effect on TRAINABILITY is real — a first-class landscape
axis"). P2 `avar` (which pools these attempts and adds av_L01_05_k{0,1} raw draws) will decide
it. If it holds, P4 must report per-design collapse rate alongside cos, and rs_L01_05's
"rescue" verdict should be read as *trainability-hostile*, not merely reorient-hostile.

H3 for this design: pending self-B completion.

### P1 design 3/5 rs_L01_05 COMPLETE: trainability-hostile confirmed — self-B ALSO collapses; every leg of the pipeline fails on this geometry (2026-07-11 early)

The paired self-B run finished the picture (design total 7041 s):

- **self-B: watchdog-aborted too** (second B-side collapse, same design). Salvage eval of
  model_150: the policy learned *nothing* that survives deployment — **all three fingers idle**
  (0.0 N, touch-frac ≤ 0.02), the object slides off (post-handoff min-z **0.0441** < 0.05 =
  drop-FAIL), held-cos **−0.969**, peak-cos 0.004. The low ang-jerk (6.6) is vacuous — nothing
  is being manipulated.
- Final design ledger: A t0 collapse → A t1 clean (health-FAIL) → imit-B collapse (salvage:
  one-finger pin, cos −0.469) → self-B collapse (salvage: no-grip drop, cos −0.969). **Four
  training legs, three collapses, zero viable policies.** Combined with its large16 history
  (late-collapse), rs_L01_05's record is 3 collapses in 4 A/B-adjacent draws.
- Scoring: **H1 strict 0/3, capability 0-for-this-design** (nothing holds with a real grip).
  **H3: Δcos(imit−self) = +0.50** — imit *better*, the opposite sign of the L01_03 strike, but
  both values are salvage-garbage so this design is best treated as H3-uninformative. What it
  IS is the strongest step-5 evidence yet: **collapse propensity follows the design**, across
  stage (A vs B), recipe (imit vs self), and draw. P2 `avar` (av_L01_05_k{0,1} + these pooled
  attempts) will quantify it; P4 must carry per-design collapse rate as a first-class output.

Notably its 9-vector sits ≤ 8.2 mm from m05 on every param (max |Δ| on index-y) — if P2
confirms, trainability cliffs exist *within* a sub-centimeter ball around the best known design.

### P1 design 4/5 rs_L01_07: second "holds-but-static" design — clean training, healthy 3-finger grip, zero reorient under BOTH recipes (2026-07-11 ~02:15)

rs_L01_07 was a large16 delivery health-FAIL. Under the rescue evaluator (7769 s):

- **Policy A: rescued on the first fresh draw** (model_609, objheight 0.1153, WARN, no retry).
  A-side tally: 3/4 designs so far rescued A on the first draw (all but L01_05).
- **imit-B: trains clean, holds well, does not reorient.** Post-handoff min-z 0.1123, all three
  fingers engaged from step 1 (thumb 12.1 / index 9.9 / middle 9.3 N, touch-frac 0.94–1.0),
  verdict WARN (over-clamp 10.4 N, sliding path 10.1 cm vs net 0.6 cm). But held-cos **0.032**,
  peak-cos 0.051 — a static clamp, no rotation at all.
- **self-B (same A, paired): the same story** — min-z 0.1129, balanced 8–10 N grip, WARN, and
  held-cos **0.059** / peak 0.096. More micro-slip (path 23.1 cm, de-centering 63.5) but no gait.

Scoring: **H1 strict 0/4; capability flip YES** (lift+hold fully rescued → capability 2/4 as a
design count, 3/4 counting L01_03's hold). **H3 Δcos = −0.027 ≥ −0.05 ⇒ imit-fair** — the
L01_03 fairness strike did NOT repeat (H3 tally: fair 2, strike 1, uninformative 1). rs_L01_07
lands squarely in the L01_03 class: **holds-but-static**. Both recipes converge to a hard
symmetric clamp with saturated forces and near-zero angular progress — consistent with
reorient-hostile geometry (the grip is *too* stable to roll) rather than optimizer noise,
with the standing n=1-per-recipe caveat.

### P1 design 5/5 rs_L01_09 (in flight): second trainability-hostile design — both A draws abort, imit-B collapses; self-B running (2026-07-11 ~03:00)

- CEM fine (lift 0.051, persist 1/1/1). **Policy A consumed both attempts and BOTH aborted**;
  the kept checkpoint is a *salvaged* undertrained model_150 from t1 (objheight 0.128 at that
  checkpoint — the same known-risk salvage mode as large16's L01_05). First design where
  best-of-2 failed to produce a clean A.
- **imit-B: watchdog-collapsed at iter ~100** (third B-side collapse, second design). Salvage
  model_100: holds min-z 0.1211 but with an idle thumb (0.0 N; index 4.0 / middle 4.7 N —
  the degenerate 2-finger pinch), held-cos 0.249, peak 0.373, verdict FAIL (idle_finger).
- self-B on the same salvaged A is training now (last leg of P1).

Interim tallies with one leg outstanding: **H1 strict 0/4 complete (max possible 1/5)** — the
decision-tree step-3 branch "≤1/5 ⇒ failures are real geometry effects" is already decided on
the strict bar. The honest decomposition, though, is axis-dependent: **lift/hold failures were
mostly draw noise** (3/5 designs rescued to a healthy hold: L01_02/03/07), while **reorient
failures persisted in every case** (0/8 B runs reached cos 0.5 except L01_02's imit at 0.561),
and **two designs (L01_05, L01_09) are trainability-hostile** — collapse chases them across
stages, recipes, and draws. Landscape classes emerging within a ±8 mm ball around m05:
*reorients* (L01_02) / *holds-but-static* (L01_03, L01_07) / *trainability-hostile* (L01_05,
L01_09 pending its self-B).

### P1 rescue CLOSED (5/5, 2026-07-11 03:32, 9h44m wall): pick-up failures were optimizer noise (5/5 rescued), reorient failures are real geometry signal (0/5 rescued), and trainability itself is a design property

Design 5 finished the set. rs_L01_09's paired self-B — on the same *salvaged* A — **trained
clean** (its only clean B leg; softens the interim "trainability-hostile" call to *A-fragile*)
yet converged to the familiar static grip: held min-z 0.1249, forces 4.1/3.9/12.2 N
(middle-heavy, thumb touch-frac 0.45), held-cos **−0.014**, peak 0.048, FAIL.

**Final P1 scoreboard (5 designs, 10 paired B evals, every clean large16 failure re-run):**

- **H1 strict flip rate: 0/5** (bar: held + cos ≥ 0.5 + verdict ≠ FAIL; L01_02 came closest —
  0.561 with a jitter-only FAIL, 44.1 vs bar 40).
- **Pick-up/hold axis: 5/5 RESCUED.** Every design — including L01_03 ("A never lifted" in
  large16) and L01_09 (both fresh A draws aborted) — delivers and holds post-handoff min-z
  0.109–0.125 in BOTH paired evals. "Never learned to pick it up" was **pure optimizer draw
  noise**, fully repaired by best-of-2 + watchdog + salvage.
- **Reorient axis: 0/5 rescued, and the paired recipes AGREE per design** — two independent B
  draws converging on the same verdict makes these *measurements, not noise*: within ±8.2 mm of
  m05 (imit cos 0.82±0.02), reorientability under the same evaluator lands at 0.56 (L01_02),
  ~0.0 static (L01_03/07/09), −0.5 (L01_05). **The landscape has real, resolvable structure on
  the reorient axis once A-draw noise is removed.**
- **Trainability axis (unplanned, emergent):** collapse propensity clusters by design —
  L01_02/03/07: 0 collapses in 6 training legs; L01_05: 3 in 4; L01_09: 3 in 4. Pending P2's
  m05 control, per-design collapse rate becomes a first-class landscape output (tree step 5).
- **H3 fairness: KEEP imit-B as the evaluator.** Δcos(imit−self) = −0.020 / −0.221 / +0.500 /
  −0.027 / +0.263 — one strike (L01_03), three fair, one salvage-garbage; informative-mean ≈ 0
  ⇒ no systematic off-m05 handicap, and imit produced the set's only real reorient AND held
  where self dropped (L01_02).

**Verdict on the user's intuition: confirmed where it was aimed, sharpened where it wasn't.**
The evaluator WAS policy-bottlenecked — n=1 from-scratch verdicts were dominated by A-draw luck,
and every "couldn't even pick it up" dissolves under best-of-2. With that noise removed, the
residual failures do NOT dissolve: reorient capability varies by geometry at sub-centimeter
scale, and collapse-during-training follows the design. Caveat bounding the claim:
"reorient-hostile" means *under the b_liveA/b_liveA_imit recipe family at 20M ts* — a different
curriculum/architecture might still unlock these geometries; what we can now map honestly is
"capability under the blessed recipe," which is the actionable object for co-design.

Next per the tree: P2 `avar` auto-started 03:32 (m05 ×3 raw draws = the control deciding whether
L01_05-class fragility is design-dependent or endemic); on completion score H2 → fix P4's
evaluator shape → launch `global24`.

### P2 avar interim (2026-07-11 ~08:10) — m05 control COMPLETE (3/3 raw draws); H2's tightness clause is dead early

All three m05 raw A draws (`--a-attempts 1`, no retry — each row is one uncensored
CEM→A→imit-B pipeline draw on the *identical* 9-vector):

| draw | Policy A | A gate (objheight / verdict) | imit-B handoff | held-cos (peak) | jerk | notes |
|---|---|---|---|---|---|---|
| av_m05_k0 | clean, model_609 | 0.1176 / WARN | holds, min-z 0.1173 | **0.488** (0.612) | **9.8** | genuine moderate reorienter, b33-class smoothness |
| av_m05_k1 | clean, model_609 | 0.1175 / WARN | holds, min-z 0.1162, 3 fingers 7–8 N | **−0.158** (0.112) | 64.6 FAIL | holds-but-static + thrash (slide path 19 cm, net 0.2) |
| av_m05_k2 | **COLLAPSED** (salvage model_150) | 0.124 @150 / — | **drops**, min-z 0.0017 | −0.188 | 108 | salvage-A garbage; expected given no retry |

**Finding 1 — H2's tightness clause is already decided (dead), before L01_05's draws even land.**
The two *clean* m05 draws span held-cos 0.488 → −0.158 under the sd-0.02 imit-B evaluator
(range 0.65 ≫ the 0.10 bar). Pooling the historical a10→b33 reference (0.82, same recipe
family), m05's clean-A-draw distribution is {0.82, 0.49, −0.16} — sd ≈ 0.49. Step-5's middle
branch ("A quality varies continuously per design") is selected on the control alone.

**Finding 2 — the A health gate cannot see the variance that matters.** k0 and k1 are
gate-indistinguishable: same kept checkpoint index, objheight 0.1176 vs 0.1175, both WARN —
yet their deliveries send the same B recipe to cos 0.49 vs −0.16. Best-of-N *by the gate*
buys collapse insurance only; it cannot select for downstream reorientability. Averaging
multiple **full A→B draws** is the only variance reduction available under this evaluator.

**Finding 3 — this partially reopens P1's reorient-axis conclusion.** P1's "real sub-cm
geometry signal" rested on paired B recipes agreeing per design — which controls B-seed noise
but NOT A-draw noise (both recipes rode the *same* A). The m05 control now shows a single
clean A draw lands anywhere in {reorients ≈ 0.5–0.8, static ≈ −0.2..0} on the *best known
design* — a range that covers every P1 verdict except L01_05's salvage garbage. So
"L01_03/07/09 hold-but-static" is, per design, a single Bernoulli draw from a distribution
whose m05 version already produces "static" ~1/3 of the time. Weak counterpoint keeping the
geometry hypothesis alive: 3-of-4 completed P1 designs drawing static (and none drawing
≥0.6) is jointly unlikely (~4%) if they were all m05-equivalent — the signal is *weakened,
not overturned*, but per-design verdicts need ≥2–3 full draws to be measurements.

**Finding 4 — collapse tally: m05 = 1/3 raw draws.** Even the best design collapses without
retry, so collapse-retry stays mandatory in P4 and per-design collapse *rate* will be
statistically soft at feasible n (m05 1/3 vs L01_05 pooled 2/3 pre-avar doesn't separate);
keep it descriptive, not decisive.

**Trending P4 shape (formalize when the queue's L01_05 rows land):** `global` set at
`--n 12` with each LHS point emitted twice (`_r0/_r1` replica suffixes — ~3-line change in
`morph_set()`'s `"global"` branch, same trick the avar set uses) + `--a-attempts 2`, scoring
per-design **mean cos over 2 full draws** with collapse count as a side channel. Caveat to
carry into the writeup: with per-draw sd ≈ 0.45, a 2-draw mean still has sd ≈ 0.32 — the
landscape will resolve only coarse structure (reorients-sometimes vs never), which raises the
priority of the step-8a morphology-conditioned-policy spike as the fundamental fix to
evaluate-requires-optimize.

### P2 avar CLOSED (2026-07-11 09:15) → P4 global12×2 LAUNCHED (09:22): L01_05's clean draw scores 0.48 — the P1 class map was draw-dominated; probes conclude the policy draw is THE landscape bottleneck

The queue's last two rows sealed H2:

- **av_L01_05_k0: A clean first draw (objheight 0.1156, WARN) → imit-B WARN, holds min-z
  0.1155, held-cos 0.480, jerk 12.5** — statistically identical to m05's best raw draw
  (0.488/9.8). The design that went 0-viable-policies-in-4-legs during P1 produced a
  moderate reorienter on its very next uncensored draw.
- av_L01_05_k1: A iter-0 collapse (objheight 0.0) — the raw collapse tally lands at m05 1/3
  vs L01_05 2/4 pooled (avar + rescue attempts): direction design-dependent, n too small to
  separate; collapse rate stays a descriptive P4 output, with retry mandatory.

**Formal H2 verdict (decision-tree step 5, middle branch):** the per-design evaluator —
CEM → from-scratch A (health-gated) → imit-B — has per-draw held-cos sd ≈ 0.3–0.5 that is
**gate-invisible** (k0/k1 As identical on every recorded gate metric, 0.65 apart in B outcome)
and **design-overlapping** (L01_05's clean draw ≈ m05's). Single full-pipeline draws are not
measurements; retries help only against collapse. P4 therefore scores **mean cos over 2
independent full draws** per design.

**Probe-suite conclusion (P1+P2 together, revising P1's close-out):** the user's intuition is
confirmed at full depth. (1) "Never learned to pick up" — optimizer noise, 5/5 rescued, still
solid. (2) "Never learned to reorient" — *also unresolvable per-draw*: P1's static/hostile
classes dissolve under the m05 control's own draw spread and L01_05's 0.48 (the ~4%-joint-static
counterpoint stands as the only surviving hint of sub-cm geometry signal). (3) The honest object
to map is **P(reorient | design) under the blessed recipe**, estimated by replicated full draws —
exactly what P4 now does. The deeper fix (make evaluation cheap instead of replicated) remains
the morphology-conditioned policy spike (step 8a).

**P4 LAUNCHED 09:22** (detached, resumable, ETA ~44 h → ~07-13 morning):
`--morph-set global --n 12 --seed 2 --replicas 2 --tag global12x2 --b-recipe imit
--a-attempts 2` — 12-point LHS over the FULL 9-param box, replica-major (complete r0 pass ≈
n=1 map at ~20 h, then the r1 pass), 24 pipeline runs. Waiter fires at r0-complete / crash /
done. **Standing CPU task for idle ticks:** mine the ~20 accumulated (A scorecard, B outcome)
pairs from rescue/avar/vstudy for an A-side feature that predicts B fate (the current gate is
blind to it); a predictive gate would restore cheap single-draw evaluation →
`docs/notes/a_quality_predictor.md`.

### A-quality predictor: DONE, negative (2026-07-11 ~10:50 tick) — no A metric restores single-draw eval

Mined all 26 (kept-A scorecard, B outcome) pairs still on disk across confirm/large16/rescue/avar
(`scripts/a_quality_predictor.py`). Best predictor of B held-cos is A's mean tip force at
rho +0.44 (n=25, at the p≈0.05 boundary), with the whole "grip richness" family (per-finger force
min/max, touch fraction, contact count) trending +0.3–0.4 — but within m05's 5 same-geometry
draws the relation is non-monotone (7.0 N→0.49, 8.1 N→−0.16, 11.3 N→0.23), so the cross-design
correlation is the geometry landscape itself (firm-grip-affording designs also reorient), not a
draw-quality shortcut. Hold min-z, ang-jerk, drift, slide: rho ≈ 0. **P4's 2-replica design
stands; the structural fix remains the step-8a conditioned-policy spike.** Side-findings worth
adopting: (1) the only scored B-collapse pair (rs_L01_05) is the only A with an idle finger at
delivery (min force 0.0 N) → a free `min(force_mean) < 0.5 N` retry-trigger for `--a-attempts`;
(2) 4/26 A scorecards FAIL their drop check on a pre-lift-window min-z artifact (minz≈0.007,
the known floor-z trap) yet 3/4 fed good Bs — the A gate's verdicts are partly artifact, which
is why they can't rank draws. Details: `docs/notes/a_quality_predictor.md`.

**Re-run at n≈50 (2026-07-17, P4 `global12x2` complete — the note's queued revisit, now CLOSED):**
50 (A,B) pairs / 49 completed-B, double the n=25 above. The verdict **strengthens**: the n=25
front-runners were small-sample artifacts — mean tip force **+0.44→+0.18**, max finger force
**+0.41→+0.03** as n doubled; best rho is now fmin **+0.31**, nothing clears 0.31. So the earlier
"grip richness rho +0.44" is retired as a shortcut (it was a thin, geometry-confounded, tail-driven
signal that did not generalize to out-of-neighborhood designs). **Side-finding (1) is REFUTED:** at
n=50 `fmin<0.5 N` flags 5 As of which **3 gave good reorienters** (cos 0.528/0.445/0.681) — an
idle-finger delivery is B-recoverable, not a degenerate reject, so the min-force veto would discard
capability (withdrawn). Side-finding (2) confirms at scale (6/50 pre-lift drop-FAIL artifacts, 5/6
good B). **Net: no A-scorecard shortcut restores single-draw eval at 2× data ⇒ the conditioned
policy remains the fundamental fix** (§PROGRAM CLOSE-OUT point 4 stands, reinforced). Full update:
`docs/notes/a_quality_predictor.md`.

### P4 global12×2 interim — designs 1–2/24 (2026-07-11 ~12:45 tick): first full-box LHS point is a REORIENTER

First two r0 records landed (~94 min/design each — ahead of the ~110 min/design estimate;
r0/n=1 map ETA now ~07-12 early AM, full 24 ETA ~07-12 late night):

- **G02_00_r0 — cos 0.504, WARN**: A first-draw clean (model_609, objheight 0.1156); imit-B
  holds (post-handoff min-z 0.114, all 3 fingers 5.4–8.2 N) and **reorients** at m05's
  clean-draw level (held-cos 0.504 / peak 0.524 / jerk 9.3 — cf. m05 av_k0 0.488/9.8). WARN
  is sliding (path 5.1 cm vs net 0.2 cm) + over-clamp 7.1 N only; every capability check
  PASSes. Notably this design sits **well outside the ±8 mm local box** (delta-to-m05 up to
  3.9 cm on two placement params; thumb IK residual 4.4 mm, worst accepted so far) — the
  first honest-pipeline evidence that reorient capability isn't confined to m05's immediate
  neighborhood. Per H2 discipline this is one draw, not a measurement — its `_r1` replica
  decides whether it's a real hotspot or draw luck.
- **G02_01_r0 — cos −0.134, WARN**: same clean profile through A (first draw, objheight
  0.1142) and hold (min-z 0.1214, forces 7.9–11.6 N), but holds-but-static (peak 0.147),
  the familiar failure phenotype. Max delta-to-m05 4.5 cm.

Trainability tally so far: **2/2 A first-draw clean, 0 collapse legs** — consistent with
collapse being sparse across the full box rather than the norm (L01_05/09 remain the
concentrated cases). Both A health verdicts WARN; the gate again can't distinguish the
reorienter's A from the static design's A — the 2-replica pooling carries the inference
load, as designed. No conclusions until `_r1` pairs land; next analysis window at the
r0-complete waiter event (≥12 records).

### P4 global12×2 interim — design 3/24 (2026-07-11 ~15:00 tick): first A-defect row; flag it, don't read it as geometry

**G02_02_r0 — cos −0.388 (peak 0.001), WARN, A×2 (6072 s).** The first P4 design to burn
the A best-of-2, and the kept A is defective:

- **A t0 completed but health-FAILed on idle_finger**: index 0.0 N / touch-frac 0.05 — a
  thumb+middle 2-finger pinch (thumb 1.3 N, middle 10.7 N). **A t1 watchdog-collapsed**
  (first A-side collapse in P4), so the pipeline kept the FAIL-grade t0, exactly the
  "best-of-N-by-gate is collapse insurance only" regime established in P2.
- Plausible mechanism, not just draw luck: this design has the **worst accepted per-finger
  IK residual so far — index 6.76 mm** (G02_00's worst was thumb 4.4 mm, and that design
  reoriented). An `open_ik` index tip seeded ~7 mm off is a candidate cause for A never
  recruiting the index. One pair is anecdote; the (per-finger residual → same finger idle
  in A) correlation is now a standing thing to score once more rows land.
- **B inherited the defect**: it re-recruits the index to full touch (1.7 N, touch 1.0 —
  better than A delivered) but stays middle-dominant (11.2 N) and makes **zero reorient
  attempt** (peak-cos 0.001; held-cos −0.388). Verdict WARN (over-clamp only) because all
  capability checks pass — the row *looks* like the familiar holds-but-static phenotype,
  but the kept-A defect confounds it.
- **Scoring note for the pooled analysis:** treat G02_02_r0 as an **A-defect row**, not a
  geometry-static row. Its `_r1` replica is unusually informative: a clean-A draw there
  that reorients would pin the r0 verdict on the A defect; a repeat idle-index A would
  point at the geometry/IK-seed. This is also the second live datapoint for the
  a-quality-predictor note's **idle-finger veto** (min force_mean < 0.5 N at A-accept):
  the veto would have rejected t0, and with t1 collapsed the design would have been
  A-starved at 2 attempts — a veto needs a "spend a 3rd attempt" rule, not just rejection.

Live next-design note: **G02_03_r0's CEM passed the grasp gate with thumb persist 0.00**
(lift 0.047, persist 0.00/0.97/0.99) — the first thumb-dead grasp entering A training —
and its A t0 watchdog-collapsed; t1 in flight. P4 A-leg raw tally is now **2 collapses /
5 legs** after a 0-collapse start, and both collapse designs have a degraded input signal
(worst index residual; thumb-dead CEM grasp) — consistent with P1/P2's hint that
trainability-hostility tracks impoverished grasp/recruitment rather than being uniform
luck. Pace: ~95 min/design average (6072 s for the A×2 design); r0-map ETA still ~07-12
early AM.

### P4 global12×2 interim — designs 4–5/24 (2026-07-11 ~17:30 tick): best cos yet from the WORST grasp seed; first PASS verdict is static

**G02_03_r0 — cos 0.568 / peak 0.765, verdict FAIL (idle thumb), A×2 (5705 s). The best
reorient of P4 so far came from the most degraded input signal of P4 so far** — which
overturns the working hypothesis from the design-3 note ("trainability-hostility tracks
impoverished grasp/recruitment"):

- Input signal was the worst in the sweep on every axis: CEM grasp passed the gate
  **thumb-dead** (persist 0.00/0.97/0.99, tips 2.0, grasp-imbalance 0.99) and the IK
  seed has the **worst thumb residual accepted so far, 11.06 mm** (index 5.83). A t0
  watchdog-collapsed; the kept t1 health-FAILed on the predictable axis (idle thumb).
- imit-B inherited the 2-finger character (thumb 0.7 N / touch-frac 0.5, index 8.0 /
  middle 11.1 N) and **still reorients**: held-cos 0.568, peak 0.765, jerk 10.5, min-z
  0.117. The FAIL verdict is idle-finger only; every capability check passes. This is
  the L01_02 pattern again — **a capability flip hidden behind a non-capability FAIL** —
  and the strongest case yet for the softened flip bar (cos + held scored separately
  from health grades) in the pooled analysis.
- Two readings, r1 decides: (a) the geometry is a genuine reorienter that succeeds
  despite a crippled grip — under the softened bar it currently *beats* G02_00 (0.568
  vs 0.504); (b) a 2-finger roll may be a narrower skill than the 3-finger gait, and a
  clean-thumb draw could land anywhere. Either way "thumb-dead CEM grasp ⇒ hostile
  design" is dead: that design produced P4's best cos.

**G02_04_r0 — cos 0.148 / peak 0.205, verdict PASS (5620 s) — the first clean-sheet
PASS of the honest-pipeline sweeps, and it's static.** A first-draw clean (WARN,
objheight 0.1118); B holds with textbook health — all 6 checks PASS, 3-finger touch
1.0 each (1.3/2.8/9.6 N), jerk 6.5 (best of P4), drift 0.3 cm, force 4.5 N — and makes
no real reorient attempt. Health and capability are now demonstrated fully orthogonal
in BOTH directions within one afternoon: G02_03 FAILs health while reorienting; G02_04
PASSes everything while static. The scorecard is a *gate* (is this policy's hold
real?), never a *rank*; cos remains the only capability metric. (Per H2, single-draw
static ≠ static design — r1 decides.)

**Running r0 tally after 5/12:** cos = {0.504, −0.134, −0.388 (A-defect row), 0.568,
0.148}; 2/5 reorient ≥ 0.5 under the softened bar — the full-box LHS is finding
reorienters at a rate consistent with m05's own clean-draw rate (~1/2), at 3–4 cm from
m05. A-leg raw tally: 2 collapses / 7 legs, and the "degraded input ⇒ collapse" pattern
now reads as collapse-propensity-only (G02_03 collapsed once, then reoriented).
IK-residual ledger: index-residual→idle-index (G02_02) still stands as the one
suspicious pairing; thumb-residual 11 mm did NOT block reorienting (G02_03). Pace ~95
min/design holds; design 6 (G02_05_r0) A-leg training since ~17:25; r0 map (12) ETA
~07-12 ~04:30, full 24 ~07-12 ~23:30. Waiter armed (r0-complete ≥12 / crash / DONE).

### P4 global12×2 interim — design 6/24 (2026-07-11 ~19:30 tick): second A-defect row; the index-residual→idle-index pairing repeats

**G02_05_r0 — cos −0.499 / peak 0.25, verdict FAIL (idle thumb+index), A×2 (5877 s).
Score it as the second A-defect row (G02_02 pattern), not geometry; `_r1` is the
arbiter.** The headline number needs decoding: −0.499 is NOT an active anti-reorient —
the kept A *delivers* at cos −0.529 (flat-ish, awkward face), and B's tail −0.499 with
peak 0.25 means B held that delivery essentially unmoved. It's a holds-but-static row
whose static orientation happened to start negative.

- Input signal was clean on the CEM axis (lift 0.050, tips 3.0, persist 1/1/1,
  grasp-imbalance 0.0) but carries the **worst accepted index IK residual of the
  program: 12.84 mm** (thumb 2.64, middle 0.1). A t0 watchdog-collapsed at iter 40
  (objheight 0.0405 < 0.045); the kept t1 health-FAILs on **idle index** (0.9 N /
  touch-frac 0.25) while thumb+middle clamp 9.4/12.3 N — a two-finger A grip on the
  exact finger the IK seed placed worst.
- imit-B trained clean (2 trainable-frac guard hits, no collapse) and holds (min-z
  0.105) but the grip degrades to middle-dominated (5.6 N/1.00 vs thumb 4.1/0.38,
  index 1.8/0.32 → idle thumb+index FAIL), jerk WARN 35.3, and no reorient attempt.
- **The IK-residual→idle-finger pairing now has two same-finger data points:** G02_02
  (index resid 6.76 mm → idle-index A, static B) and G02_05 (index 12.84 mm →
  idle-index A, static B), while G02_03's thumb 11.06 mm did *not* block reorienting
  (2-finger index+middle roll). A speculative asymmetry worth one line: a thumb-dead
  grip still admits an index+middle gait, but an index-dead grip leaves thumb+middle —
  geometrically a pinch across the object, maybe no rolling pair. n=2, descriptive
  only; if the pooled analysis confirms it, the actionable fix is an IK-residual
  acceptance bar (or re-CEM on the retargeted keyframe) rather than any policy change.

**Running r0 tally after 6/12:** cos = {0.504, −0.134, −0.388 (A-defect), 0.568,
0.148, −0.499 (A-defect)}; softened-bar reorienters 2/6; A-defect rows 2/6 — the
defect rate is becoming its own column (both defect rows are the two worst index
residuals). A-leg raw tally: 3 collapses / 9 legs. Pace ~96 min/design; design 7
(G02_06_r0) CEM clean (lift 0.049, persist 1/1/1), A training since 19:03; r0 map ETA
~07-12 ~04:40, full 24 ~07-13 ~00:00.

### P4 global12×2 interim — design 7/24 (2026-07-11 ~21:00 tick): cleanest input row yet is still static; thumb-residual harmlessness repeats

**G02_06_r0 — cos 0.127 / peak 0.397, verdict WARN, A first-draw clean (5646 s).**
Third holds-but-static row (after G02_01, G02_04), and the most instructive of the
three because every input and health signal is clean — this is what "the pipeline
did everything right and the draw still didn't find the gait" looks like.

- **Inputs:** CEM grasp clean (lift 0.049, tips 3.0, persist 1/1/1, imbalance 0.0);
  index IK residual **0.09 mm** (near-perfect seed), middle 2.01 mm, thumb 9.66 mm.
  A accepted on its first draw (model_609, objheight 0.1119, WARN), no collapse.
- **Outcome:** B holds rock-solid — min-z 0.108, all three fingers engaged at
  touch-frac 1.00 (thumb 8.3 / index 10.3 / middle 6.7 N), jerk 10.0 PASS, net
  drift **0.0 cm**. Only WARNs are micro-slip (path 3.8 cm vs net 0.0) and
  over-clamp 8.4 N. Peak cos 0.397 vs tail 0.127: it *attempts* a partial
  reorient and settles back — not a frozen clamp like G02_03's peak 0.001.
- **Residual-asymmetry ledger grows, same direction.** Thumb residual 9.66 mm
  produced full thumb engagement here (and G02_03's thumb 11.06 mm produced P4's
  best reorienter, thumb-dead route); index residuals 6.76 / 12.84 mm both produced
  idle-index A-defect rows (G02_02, G02_05). Index-side seed error keeps looking
  like the harmful one; thumb-side keeps looking tolerable. Still n=2 vs n=2,
  descriptive.
- Geometry is far from m05 (max |Δ| 3.4 cm on middle-x; thumb-x −1.6 cm,
  index-x −1.3 cm) — another mid-box point that trains, lifts, and holds cleanly.

**Running r0 tally after 7/12:** cos = {0.504, −0.134, −0.388 (A-defect), 0.568,
0.148, −0.499 (A-defect), 0.127}; softened-bar reorienters 2/7; A-defect rows 2/7;
holds-but-static clean rows 3/7. A-leg raw tally: 3 collapses / 10 legs. Every
design so far holds ≥0.105 min-z — the pick-up/hold axis is a solved constant of
the full box, exactly as P1 predicted; all remaining variance is on the reorient
axis. Pace ~96 min/design; design 8 (G02_07_r0) CEM clean (lift 0.050, persist
1/1/1), A training since 20:37; r0 map ETA ~07-12 ~04:40, full 24 ~07-12 ~23:45.

**G02_07_r0 — cos 0.333 / peak 0.493, verdict WARN, A first-draw clean (5630 s).**
Third-best r0 cos, sitting just under the softened 0.5 bar — the first *partial*
reorienter (sustained, not transient: tail 0.333 is 68% of peak 0.493, unlike
G02_06's attempt-and-settle 0.127/0.397).

- **Inputs are the cleanest of the entire sweep:** CEM grasp clean (lift 0.050,
  tips 3.0, persist 1/1/1, imbalance 0.0) AND all three IK residuals ≤ 0.1 mm
  (thumb 0.10 / index 0.09 / middle 0.09) — every other r0 design has at least one
  residual ≥ 1 mm. A accepted on its first draw (model_609, objheight 0.1153,
  WARN), no collapse.
- **Geometry:** the closest r0 point to m05 (max |Δ| 2.18 cm on thumb-y; all nine
  params within ±2.2 cm) — the LHS point most "in m05's basin", and it produces a
  mid-grade reorient, not a standout.
- **Outcome:** full three-finger engagement from step 0 (touch-frac 1.00 each,
  thumb 8.1 / index 13.1 / middle 11.5 N), min-z 0.1104, jerk 11.7 PASS, net drift
  0.4 cm. WARNs: sliding (path 3.9 cm ≫ net 0.4 — consistent with rolling the
  object, which is how reorients look to this check) and over-clamp 10.9 N mean
  tip force — the highest of the eight r0 rows, mildly consistent with the
  A-predictor's grip-richness trend (tip-force rho +0.44 vs held-cos).
- **Residual-asymmetry ledger unchanged** (all residuals trivial here, no new
  index/thumb data point). The row instead sharpens the H2 point: near-m05
  geometry + perfect seeds + clean A still lands mid-distribution — exactly where
  m05's own draw spread {0.82, 0.49, −0.16} says a single draw can land.

**Running r0 tally after 8/12:** cos = {0.504, −0.134, −0.388 (A-defect), 0.568,
0.148, −0.499 (A-defect), 0.127, 0.333}; softened-bar reorienters 2/8; partial
1/8; A-defect rows 2/8; holds-but-static clean rows 3/8. A-leg raw tally: 3
collapses / 11 legs. Hold min-z ≥ 0.105 on 8/8 — pick-up/hold still a solved
constant of the box. Pace ~94 min/design; design 9 (G02_08_r0) CEM clean (lift
0.050, persist 1/1/1), A training since 22:11; r0 map ETA ~07-12 ~04:30, full 24
~07-12 ~23:30.

### P4 global12×2 interim — designs 9–10/24 (2026-07-12 ~02:00 tick): flattest clamp yet; third A-defect row is the first idle-THUMB one

**G02_08_r0 — cos −0.099 (peak 0.041), WARN (5668 s).** Fourth clean
holds-but-static row, and the flattest: peak cos 0.041 is the lowest of r0 — a
genuinely frozen clamp that never even attempts a reorient (contrast G02_06's
attempt-and-settle peak 0.397, G02_09's 0.342).

- **Inputs clean:** CEM lift 0.050, tips 3.0, persist 0.997³, imbalance 0.0;
  IK residuals thumb 2.99 / index 0.09 / middle 0.56 mm. A accepted on its
  first draw (model_609, objheight 0.1149, WARN) — no collapse.
- **Outcome:** textbook hold — min-z 0.1151 (hold-phase 0.1169), all three
  fingers touch-frac 1.00 from step 1 (thumb 12.2 / index 3.8 / middle 8.2 N),
  jerk 21.7 WARN, net drift 0.7 cm. All the health machinery is happy; the
  reorient axis is simply absent.
- **Geometry:** among the farther LHS points (max |Δ| from m05 = 4.6 cm) — but
  after G02_00 (reorienter at 3.9 cm) and G02_07 (mid-grade at 2.2 cm), r0
  distance-to-m05 still shows no monotone story.

**G02_09_r0 — cos −0.102 (peak 0.342), FAIL, A×2 (5880 s).** Third A-defect
row — and the first whose defect is an idle **thumb** rather than an idle
index. CEM clean (persist 1/1/1, imbalance 0.0) but thumb IK residual 6.77 mm
(index/middle 0.09). A t0 completed with health-FAIL; the t1 retry
watchdog-collapsed (objheight 0.0), so the pipeline kept the FAIL-grade t0 —
the same kept-FAIL signature as G02_02/G02_05. B trains clean and holds
(min-z 0.1158) on a thumb-idle two-finger clamp: thumb 1.3 N / touch-frac 0.24
vs index 10.8 / middle 16.3 N; jerk 112.4 = worst FAIL of r0; peak 0.342 →
tail −0.102 is attempt-and-settle on the two-finger grip.

- **Residual-asymmetry ledger gets its first split.** Index side stays 2/2
  (residual ≥ 6.8 mm → idle-index A-defect → static). Thumb side now splits:
  resid 9.66/11.06 mm tolerated (G02_06 full engagement; G02_03 thumb-dead yet
  P4's best cos 0.568 on an index+middle gait) vs 6.77 mm here → idle-thumb
  AND static. Note G02_03 and G02_09 had the *same* grip topology (thumb-idle,
  index+middle engaged) with opposite outcomes (0.568 vs −0.102) — per H2
  that's within single-draw spread, so the "thumb-dead leaves a rolling pair"
  reading survives only as a tendency, and the residual→idle-finger pairing
  stays descriptive (index-harmful 2/2, thumb 2 tolerated / 1 not).
- Magnitude ordering is now clearly NOT the axis (6.77 mm thumb hurt where
  11.06 mm didn't); if anything matters it's *which* finger the residual sits
  on plus draw luck.

**Running r0 tally after 10/12:** cos = {0.504, −0.134, −0.388 (A-defect),
0.568, 0.148, −0.499 (A-defect), 0.127, 0.333, −0.099, −0.102 (A-defect)};
softened-bar reorienters 2/10; partial 1/10; A-defect rows 3/10;
holds-but-static clean rows 4/10. A-leg raw tally: 4 collapses / 14 legs
(plus the in-flight G02_10 t0 collapse → 5/15). Hold min-z ≥ 0.105 on 10/10 —
pick-up/hold remains a solved constant of the full box; every point of
variance is still the reorient axis. Design 11 (G02_10_r0) CEM clean (lift
0.050, persist 1/1/1) but A t0 watchdog-collapsed; t1 training since ~01:30.
Pace ~96 min/design + the retry ⇒ r0 map ETA ~07-12 ~06:00, full 24
~07-13 early AM.

---

### P4 global12×2 — r0 close-out + r1 designs 13–16 (2026-07-12 ~07:00 tick): first replica-consistent reorienter; peak cos 0.999 that couldn't hold; r1 opens with an A-collapse cluster

**r0 close-out (backfill of the 03:12 STATUS entry).** G02_10_r0 cos 0.117
(jerk 36.3, WARN; A t0 watchdog-collapsed, t1 clean — fifth clean-static row).
G02_11_r0 = the program's first lift-level best-of-2 total miss: BOTH A draws
collapsed at objheight 0.0 (753 s) despite clean CEM (lift 0.055, persist
1/1/1). r0 finishes 11/12 evaluable, hold min-z ≥ 0.105 on all 11 — pick-up
solved across the full box whenever A trains at all.

**G02_00_r1 — cos 0.635 (peak 0.653), WARN, A×2 (5887 s). G02_00 is the
sweep's first replica-consistent reorienter.** r0 0.504 / r1 0.635 — both
replicas hold (min-z 0.114/0.112), reorient ≥ 0.5, with the cleanest jerk in
the sweep (9.3/9.9). The two replicas got different-grade A's (r0: first-draw
WARN; r1: kept-FAIL t0 after the t1 retry watchdog-collapsed) and the verdict
survived anyway — the first design whose reorient capability is robust to the
A draw. At 3.9 cm from m05 (thumb-x −0.029 the dominant axis) it is a genuine
far-from-m05 capability point. Under H2 rules n=2 is still not a ranking, but
G02_00 is now the top candidate for any post-sweep confirm replication.

**G02_01_r1, G02_02_r1 — double A collapse (724/754 s each), no B.** Both
designs LIFTED at r0 (cos −0.134 / −0.388). Best-of-2 total miss is therefore
not design fate for these two — it's draw luck; symmetrically, G02_11's r0
double-miss cannot be read as lift-hostile geometry until its r1 (last in the
queue) lands. Running cost of the evaluator: best-of-2 has now totally missed
3/16 design-legs, with the A-leg per-attempt abort rate at 14/26 = 54%. At
that rate ~21% of legs vanish entirely — an H2 cost that argues for
`--a-attempts 3` (or keeping per-design collapse rate as a first-class
landscape output, per the P2 clause) in any follow-on sweep.

**G02_03_r1 — tail cos 0.333 but PEAK COS 0.999, FAIL (dropped: hold-phase
min-z 0.0463, drift 4.2 cm, jerk 229.6; 2851 s).** The thumb-dead grasp seed
(persist 0.00/0.97/0.99, thumb resid 11.06 mm) reached essentially full
vertical — peak cos 0.999, the highest instantaneous reorient in the entire
sweep program — then lost retention and dropped. Confounder: its A was a
salvaged undertrained model_50 (t0 aborted at objheight 0.098, t1 collapsed) —
the known-bad valfix2-style salvage mode — so the drop is at least as
attributable to A quality as to the 2-finger topology. Replica pair: r0
0.568-held vs r1 0.333-dropped — same softened-bar family, same index+middle
gait, opposite retention (H2 spread) — but BOTH replicas attempt large
reorients (peaks 0.678 / 0.999) where most of the box never tries. G02_03 +
G02_00 are now the only designs with reorient-attempt signal in both replicas.

**r1 A-collapse cluster (flagged, then cleared).** r1 legs opened 7/8 A
attempts aborted (G02_00 t1; G02_01 ×2; G02_02 ×2; G02_03 ×2-with-salvage) —
~1% likely under r0's 41% per-attempt rate if draws were i.i.d. Checked live
at this tick: G02_04_r1 t0 (design 17) completed a full clean training run, so
it is not a wedged-GPU artifact; if the elevated rate persists through r1,
suspect a replica-seed-offset interaction rather than luck.

Pace: designs 13–16 took ~2.9 h total (collapsed legs are cheap), so full-24
ETA holds at ~07-13 early AM or slightly earlier.

### P4 global12×2 — r1 design 17 (2026-07-12 ~09:00 tick): G02_04 flips from clean-sheet PASS-static to reorienter — the sharpest replica inconsistency yet

**G02_04_r1 — cos 0.528 (peak 0.710), WARN, A×2 (8819 s, longest leg of the
sweep).** The replica pair is the sweep's cleanest head-to-head demonstration
of H2 (single draws are not measurements): identical geometry, identical clean
inputs (CEM lift 0.051 persist 1/1/1, IK residuals 3.7–4.9 mm), opposite
capability verdicts:

- **r0**: the sweep's only clean-sheet PASS (all 6 health checks) — and
  static (cos 0.148, peak 0.205), gentle grip (1.3/2.8/9.6 N, tip mean
  4.5 N).
- **r1**: WARN (jerk 26.1, sliding path 12.5 cm, over-clamp 9.2 N) — and a
  real sustained reorienter: cos 0.528, peak 0.710, tail = 74% of peak,
  min-z 0.1115, all three fingers engaged (6.6/11.0/10.1 N).

The r0 "clean-static" classification was a draw artifact. Health⊥capability
now demonstrated in both directions *within a single design*: the PASS draw
is static, the WARN draw (riding a kept-FAIL A) reorients. This is also the
second reorient-level result on a kept-FAIL A (after G02_00_r1) — the A
health gate keeps proving unable to predict B fate.

Two secondary reads. (1) **Grip richness**: the reorienting replica carries
~2× the fingertip force and a recruited thumb (6.6 N vs r0's near-idle
1.3 N) — same direction as the a_quality_predictor rho +0.44 mean-tip-force
trend, still descriptive. (2) **Evaluator cost, new mode**: both A attempts
completed but both health-FAILed (t0 objheight 0.1126 / t1 0.1089), so
best-of-2 ran its full budget and then chose between two FAILs by objheight —
essentially a coin flip. The gate can't rank draws it can't pass; this is the
most expensive leg type (~2.5 h) and it bought no draw-quality information.

The r1 abort-cluster flag from designs 13–16 is further cleared: G02_04_r1
had **zero** aborted attempts (r1 attempt tally 7/10 aborted, cumulative
A-leg 14/28 = 50%).

Tallies at 17/24 legs, r1 5/12: designs with reorient signal in ≥1 replica =
G02_00 (both), G02_03 (both attempt), **G02_04 (r1 only, pooled mean 0.338)**.
Hold min-z ≥ 0.105 on every evaluable leg (14/14). Design 18 (G02_05_r1,
whose r0 was the second A-defect row) on its A leg since 08:33.

### P4 global12×2 — r1 designs 18–19 (2026-07-12 ~12:00 tick): G02_05 posts the PROGRAM-BEST reorient (cos 0.887) on the design its r0 wrote off as an A-defect; G02_06 becomes the first replica-consistent static design

**G02_05_r1 — cos 0.887 (peak 0.927), WARN, A first-draw clean (5658 s).**
The best deterministic held-cos of the entire honest-pipeline program — above
m05's best clean draw (0.82, a10→b33) and every P1/P2/P4 row to date. The
rollout is also *clean*: min-z 0.1063, all three fingers engaged (thumb
9.2 N / index 10.8 N / middle 5.7 N, touch-frac 1.00/0.90/1.00), ang-jerk
7.8 **PASS** (below the b33 reference 9.6 — the first reorienter of the sweep
that doesn't buy rotation with jitter); WARNs are sliding (3.8 cm path vs
1.0 cm net — a rolling artifact) and over-clamp 8.6 N.

Three reads:

1. **The r0 "A-defect" classification is superseded, as designed.** r0 kept
   an idle-index FAIL A (after a t0 collapse) and produced cos −0.499; the
   protocol said "score it as A-defect, its `_r1` is the arbiter" — the
   arbiter arrived and says the geometry is the program's strongest
   reorienter so far. Replica span −0.499 → 0.887 (Δ = 1.386, widest of the
   sweep; G02_04's "sharpest inconsistency" was Δ 0.380) — but this pair is
   supersession, not contradiction: the r0 leg measured a defective A, not
   the design.
2. **The index-residual→idle-index pairing BREAKS.** IK residuals are
   deterministic per design — both replicas carry the sweep's worst accepted
   index residual, 12.84 mm. r0 drew an idle-index A (0.9 N); r1 drew an A
   whose index carries 10.8 N at touch-frac 0.90. Same residual, opposite
   index fates ⇒ the pairing (previously 2/2: G02_02 6.76 mm, G02_05
   12.84 mm) drops to 2/3 and the residual is exonerated as a *cause* —
   consistent with the design-9–10 conclusion that residual magnitude is not
   the axis; finger identity + draw luck are.
3. **Existence proof, not a ranking.** n=1 draw at cos 0.887 says capability
   ≥ 0.887 exists at this point (max |Δ| from m05 = 3.2 cm, param 8) — H2
   forbids reading it as "G02_05 > m05". Pooled mean over evaluable
   geometry-measuring legs is meaningless here because r0 measured a broken
   A; G02_05 joins the confirm-candidate list on the r1 leg alone and wants
   a dedicated replica set (or the morph-conditioned policy) before any
   promotion.

**G02_06_r1 — cos −0.018 (peak 0.043), FAIL (jerk 48.7), A×2 kept-FAIL
(5876 s).** A t0 completed but health-FAILed, t1 watchdog-collapsed →
kept-FAIL t0 by default (third kept-FAIL-A leg; the gate again spent its full
budget without ranking power). B holds (min-z 0.1036 — new sweep floor,
still ≫ the 0.05 drop bar) but never attempts a reorient (peak 0.043) and
churns (slide path 30.6 cm vs 1.4 net, jerk-FAIL 48.7).

Pooled G02_06: tails {0.127, −0.018}, peaks {0.397, 0.043} — the **first
design with both replicas evaluable and both static**, the null counterpart
of G02_00's both-replica reorienter. Caveat before calling it
reorient-hostile geometry: under m05-equivalence (clean-draw band {0.82,
0.49, −0.16} ⇒ ~1/3 static per draw) a both-static pair is ~11% likely, and
the r1 leg rode a kept-FAIL A — suggestive, not conclusive; it ranks as the
sweep's weakest reorient candidate rather than a proven null.

Tallies at 19/24 legs, r1 7/12: reorient-signal designs = G02_00 (both),
G02_03 (both attempt), G02_04 (r1), **G02_05 (r1, program-best 0.887)**;
replica-consistent static = G02_06. A-leg attempts 15/31 aborted (48%;
in-flight G02_07_r1 t0 also collapsed → trending 16/32 = 50%). Hold: every
evaluable leg except the G02_03_r1 drop held, min-z ≥ 0.1036 (15/16) —
pick-up remains a solved constant of the box. Design 20 (G02_07_r1 — r0 was
the sustained-partial 0.333 from the cleanest inputs) on its A t1 since
~11:50; full-24 ETA ~07-12 ~20:30–22:00.

### P4 global12×2 — r1 design 20 (2026-07-12 ~14:30 tick): G02_07 is the third replica-consistent design — a sustained-partial reorienter whose replicas agree to Δcos 0.033, the tightest pair of the sweep

**G02_07_r1 — cos 0.366 (peak 0.386), WARN, A×2 (t0 collapsed, t1 clean;
5939 s).** A's t0 watchdog-collapsed at objheight 0.0; t1 trained clean
(model_609, objheight 0.1151, WARN) — a textbook best-of-2 rescue, unlike
the recent kept-FAIL legs. imit-B holds (hold-phase min-z 0.1161) with all
three fingers engaged (thumb 3.8 N / index 7.4 N / middle 9.0 N, touch-frac
1.00 each) and posts **ang-jerk 6.6 PASS — the smoothest leg of the entire
sweep** (below G02_05_r1's 7.8 and the b33 reference 9.6). WARNs are the
usual pair: sliding (3.9 cm path vs 0.1 cm net) and over-clamp 6.7 N.

Three reads:

1. **Third replica-consistent design, and it fills in the middle of the
   outcome range.** Pooled G02_07: tails {0.333, 0.366}, Δ 0.033 — the
   tightest replica agreement of the sweep (G02_00's reorienter pair spans
   0.131, G02_06's static pair 0.145). The replica-consistent set now spans
   the whole verdict axis: G02_00 reorienter (0.504/0.635), **G02_07
   sustained-partial (0.333/0.366)**, G02_06 static (0.127/−0.018). At n=2
   the evaluator *can* resolve some designs cleanly — the H2 caveat is that
   G02_04 (0.148→0.528) and G02_05 (−0.499→0.887) prove it cannot resolve
   all of them; which class a design falls into is not knowable in advance.
2. **Verdict agreement despite different grips.** r0's B clamps hard
   (8.1/13.1/11.5 N, all fingers at contact from step 0, jerk 11.7); r1's B
   grips light (3.8/7.4/9.0 N, staggered contact, jerk 6.6). Two different
   policies, same capability tail (0.33 vs 0.37) — convergent capability
   rather than a repeated policy, which is stronger evidence the number
   belongs to the geometry. Notably r1's tail ≈ its peak (0.366/0.386):
   it *sustains* its partial angle rather than decaying from a higher one
   (r0 peaked 0.493 and gave some back).
3. **Cleanest-inputs design stays sub-bar.** G02_07 carries near-zero IK
   residuals (0.09–0.10 mm, the sweep's cleanest retarget) and 4.1 cm
   distance from m05, yet both replicas plateau at ~0.35 — input cleanliness
   neither predicts reorient capability (design 7 conclusion, now
   replica-confirmed) nor caps it (G02_05 reorients 0.887 through a
   12.84 mm residual).

Tallies at 20/24 legs, r1 8/12: reorient-signal designs = G02_00 (both),
G02_03 (both attempt), G02_04 (r1), G02_05 (r1, program-best 0.887), plus
G02_07 as consistent-partial; replica-consistent = G02_00 / G02_07 / G02_06.
A-leg attempts 16/33 aborted (48%). Hold: 16/17 evaluable legs min-z
≥ 0.1036 (G02_03_r1 the lone drop). Remaining: G02_08_r1 (A training since
~13:25; r0 was the flattest-clamp −0.099), G02_09_r1, G02_10_r1, and
G02_11_r1 — the lift-hostility arbiter. Full-24 ETA ~07-12 ~20:30–22:00.

### P4 global12×2 — r1 design 21 (2026-07-12 ~17:20 tick): G02_08 is the fourth replica-consistent design (second consistent static) — and the first design whose Policy A passed on the FIRST draw in BOTH replicas

**G02_08_r1 — cos 0.074 (peak 0.108), WARN, no A retry (5686 s).** A trained
clean on its first draw (model_609, objheight 0.1156, WARN) — and since r0's
A also passed first-draw, **G02_08 is the only design of the sweep so far with
zero A-leg aborts across both replicas** (against a program-wide attempt-abort
rate of ~48%): whatever drives collapse propensity, this geometry sits at the
benign end of it. imit-B holds (post-handoff min-z 0.1098, all three fingers
touch-frac 1.00 at 8.7/7.5/6.4 N) with **ang-jerk 13.8 PASS**; WARNs are the
usual sliding (13.4 cm path vs 1.2 cm net) + over-clamp 7.5 N. But it never
reorients: tail 0.074, peak 0.108.

Three reads:

1. **Fourth replica-consistent design; the static bin now has two members.**
   Pooled G02_08: tails {−0.099, 0.074}, Δ 0.173, both peaks ≤ 0.11 — deep in
   no-reorient territory both replicas, like G02_06 (0.127/−0.018). The
   consistent set is now G02_00 reorienter (0.504/0.635) / G02_07 partial
   (0.333/0.366) / G02_06 + G02_08 static — 4 of the 7 both-legs-evaluable
   designs resolve at n=2, vs the 2 proven irresolvable (G02_04 Δ0.380,
   G02_05 Δ1.386) and G02_03 (both attempt, r1 dropped).
2. **Convergent verdict from different grips, again.** r0's B is a
   thumb-dominant clamp (12.2/3.8/8.2 N, near-idle index, jerk 21.7 WARN);
   r1's is balanced and smoother (8.7/7.5/6.4 N, jerk 13.8 PASS). Same
   pattern as G02_07: grip style and smoothness are draw properties, the
   capability tail is a geometry property.
3. **Static despite the easiest trainability of the sweep.** G02_08 pairs
   the best A-trainability record (0 aborts / 2 draws) with a static
   capability verdict — one more axis-decoupling data point alongside
   health⊥capability (G02_04) and input-cleanliness⊥capability (G02_07).
   Its largest m05 offset is middle-finger y −4.6 cm (total distance
   ~6.5 cm, one of the more distant designs).

Tallies at 21/24 legs, r1 9/12: replica-consistent = G02_00 / G02_07 /
G02_06 / G02_08; irresolvable-at-n=2 = G02_04 / G02_05. A-leg attempts 16/34
aborted (47%). Hold: 17/18 evaluable legs min-z ≥ 0.1036 (G02_03_r1 the lone
drop). Context: machine rebooted 15:53 (see crash-recovery note); sweep
resumed clean, 21/24 records intact. In flight: G02_09_r1 (r0 = jitter-FAIL
112.4 static) — A t0 completed 16:54 but was gate-rejected (no collapse
sentinel ⇒ health/objheight reject), t1 training since ~16:56. Remaining:
G02_10_r1, G02_11_r1 (the lift-hostility arbiter). Full-24 ETA ~07-12
~21:30–23:00 (pushed by the reboot + G02_09 retry).

### P4 global12×2 — r1 design 22 (2026-07-12 ~19:00 tick): G02_09 is the fifth replica-consistent design (third consistent static) — and the only design whose every completed A draw is health-FAIL

**G02_09_r1 — cos 0.149 (peak 0.204), FAIL, A×2 (9076 s — longest leg of the
sweep, two full A trainings).** A t0 completed but health-FAIL (objheight
0.1084, gate-rejected); t1 also completed health-FAIL (0.1169); the pipeline
kept t1 by objheight — the second both-FAIL best-of-2 leg after G02_04_r0,
where the full retry budget is spent and the gate cannot rank the draws.
imit-B holds (post-handoff min-z 0.1141, hold-phase 0.1178) but the verdict
FAILs on **idle-finger of a new flavor**: not one dead finger but a *loose
intermittent* grip — touch-frac 0.42/0.47/0.56 (contact_count 1.45 of 3),
forces 5.5/2.6/4.3 N, mean tip force 4.1 N = the lowest-force evaluable leg
of the sweep. Jerk 10.2 PASSes; net drift 1.0 cm. Tail 0.149, peak 0.204 —
static.

Three reads:

1. **Fifth replica-consistent design; the static bin now has three members.**
   Pooled G02_09: tails {−0.102, 0.149} (Δ 0.251, the widest inside the
   static bin but both deep sub-bar), peaks 0.342/0.204 — no reorient either
   replica. Consistent set: G02_00 reorienter (0.504/0.635) / G02_07 partial
   (0.333/0.366) / G02_06 + G02_08 + G02_09 static — **5 of the 8
   both-legs-evaluable designs resolve at n=2** (G02_04 Δ0.380 and G02_05
   Δ1.386 the proven exceptions, G02_03 the both-attempt ambiguous case).
2. **Convergent verdict from maximally different grips — the fourth and
   starkest instance.** r0 is a thumb-idle two-finger HARD clamp
   (1.3/10.8/16.3 N, jerk 112.4 = worst FAIL of r0); r1 is a loose,
   intermittent, low-force three-finger juggle (4.1 N mean, jerk 10.2 PASS).
   Opposite ends of the force/smoothness spectrum, same static verdict —
   the capability tail keeps belonging to the geometry, not the policy draw.
   The residual→idle-finger pairing degrades further: the same deterministic
   6.77 mm thumb residual produced an idle thumb in r0 but a
   weakest-is-INDEX grip in r1 (2.6 N) — finger identity is draw luck, as
   G02_05 already showed on the index side.
3. **A-health hostility is its own pole.** Across both replicas G02_09 went
   FAIL, ABORT, FAIL, FAIL — the **only design in the sweep whose every
   completed A draw (3/3) is health-FAIL** (G02_02/G02_03/G02_11 are also
   zero-clean but abort-dominated). It is the exact opposite pole from
   G02_08 (clean first draw both replicas, zero aborts), splitting the
   trainability axis in two: *collapse propensity* (does A finish?) and
   *delivery health* (does a finished A deliver clean?) — G02_09 is
   abort-normal but health-hostile. Yet all three of its finished As
   delivered a holdable object (every leg min-z ≥ 0.114): one more point for
   hold-robustness ⊥ health, and a concrete P5 argument that health-FAIL
   rate — not just collapse rate — belongs in the per-design output vector.

Tallies at 22/24 legs, r1 10/12: replica-consistent = G02_00 / G02_07 /
G02_06 / G02_08 / G02_09; irresolvable-at-n=2 = G02_04 / G02_05. A-leg
attempts 16/36 aborted (44%). Hold: 18/19 evaluable legs min-z ≥ 0.1036
(G02_03_r1 the lone drop). In flight: G02_10_r1 (A t0 training since 18:30;
r0 = WARN 0.117 after an abort→WARN rescue). Then G02_11_r1 — the
lift-hostility arbiter (r0: both A attempts never lifted). Full-24 ETA
~07-12 ~20:15 (if G02_10_r1 accepts t0 and G02_11_r1 fast-aborts like r0)
to ~23:00 (if retries / G02_11 lifts and runs a full leg).

### P4 global12×2 — r1 design 23 (2026-07-12 ~20:45 tick): G02_10 becomes the THIRD irresolvable-at-n=2 design — and the first whose replica flip happened between SAME-health-grade A draws

**G02_10_r1 — cos 0.576 (peak 0.657), FAIL on jitter only, first-draw A
(5677 s).** A t0 accepted immediately (WARN, objheight 0.1166 — no retry
spent). imit-B holds cleanly (post-handoff min-z 0.1167, hold-phase 0.124,
all three fingers engaged 13.0/5.9/10.9 N, touch-frac 1.0 each) and posts a
genuine sustained reorient: tail 0.576, peak 0.657, net drift 0.7 cm. The
verdict FAILs on **ang-jerk 44.5** (bar 40; b33 ref 9.6), with WARN sliding
(path 10.1 cm vs net 0.7) and WARN over-clamp (9.9 N mean) — the
*reorients-but-jitters* signature, numerically almost a twin of P1's
rs_L01_02 (0.561 / jerk 44.1).

Three reads:

1. **Third proven irresolvable-at-n=2 design.** Pooled G02_10: tails
   {0.117, 0.576}, Δ 0.459 — r0 read static-with-a-hard-clamp
   (10.2/10.5/13.3 N, peak 0.178), r1 reorients past every static-bin
   member's *peak* on its tail alone. The both-legs-evaluable ledger at
   9 designs: **consistent 5** (G02_00 reorienter / G02_07 partial /
   G02_06+G02_08+G02_09 static), **irresolvable 3** (G02_04 Δ0.380,
   G02_05 Δ1.386, G02_10 Δ0.459), ambiguous 1 (G02_03, both attempt).
   A third of evaluable designs cannot be binned by a 2-draw evaluator,
   and the irresolvable class keeps producing the sweep's best reorients
   (0.887, 0.71-peak, now 0.576/0.657).
2. **The flip no longer needs an A-grade excuse.** G02_05's flip rode a
   broken-vs-clean A (FAIL→WARN) and G02_04's rode WARN→FAIL (inverted:
   the FAIL-graded A fed the better reorienter). G02_10 closes the loop:
   both kept As are health-WARN at near-identical objheight (0.1131 vs
   0.1166), yet the B outcomes differ by Δ0.459. With the avar result that
   imit-B on a *fixed* A has sd ≈ 0.02, the variance must live in
   delivery-state differences between same-grade A draws that the
   scorecard grade does not see — the health grade is a *gate*, not a
   sufficient statistic of the delivery. This is the sharpest form yet of
   health ⊥ capability, and a concrete P5 implication: ranking designs
   needs capability probes on the delivered state (or A-draw pooling),
   not better A grading.
3. **Reorient-signal census favors "capability is common, expression is
   draw-gated".** Designs showing a real reorient attempt in ≥1 replica:
   G02_00 (both), G02_03 (both), G02_04 (r1), G02_05 (r1), G02_07
   (partial, both), G02_10 (r1) — 6 of 10 designs with any evaluable leg.
   Under the n=1 evaluator the sweep would have called G02_04/G02_05/G02_10
   static/defective with a coin-flip's luck; the landscape's real shape is
   *fraction of draws that express*, which only shows up with draws ≥ 2.

Tallies at 23/24 legs, r1 11/12: A-leg attempts 16/37 aborted (43%);
hold 19/20 evaluable legs min-z ≥ 0.1036 (G02_03_r1 the lone drop);
jitter-FAIL now the only thing separating G02_10_r1 from a WARN pass —
the strict health verdict keeps conflating delivery quality with reorient
quality (P1's known axis-conflation, unresolved). In flight: G02_11_r1 —
the lift-hostility arbiter (r0: 2/2 A attempts never lifted, 753 s leg) —
A t0 training since 20:05 with the collapse watchdog armed; if it aborts
twice the design is a 4/4 never-lift and lift-hostility gets its first
replica-consistent member; if it lifts, total-miss-as-draw-luck gains its
strongest case. Sweep DONE waiter armed (DONE-only).

### P4 global12×2 COMPLETE (24/24, 2026-07-12 22:13, ~37 h GPU incl. crash-resume): no lift-hostile geometry exists in the box; reorient capability is common but draw-gated; two candidates advance to confirm

**G02_11_r1 arbitrated lift-hostility: it doesn't exist.** After r0's 2/2 never-lift, r1 (one
A retry) delivered and held (min-z 0.1197) and attempted a reorient (cos 0.445, peak > held,
jitter-FAIL 32.5). Every "never lifted" leg in the program now has a sibling draw that lifted
— **pick-up/hold is solved across the entire 9-param design box** (min-z ≥ 0.103 on all 20
policy-producing legs), the strongest and cleanest landscape result of the program.

**Pooled table:** `docs/experiments/MORPH_PIPELINE_global12x2_POOLED.md` (figs
`img/morph_pipeline_global12x2_{summary,training}.png`). The n=2 census: 5/12 designs bin
replica-consistently (G02_00 reorienter 0.504/0.635; G02_07 sustained-partial 0.333/0.366;
G02_06/G02_08/G02_09 static), 3/12 are irresolvable at n=2 with Δcos 0.38–1.39
(G02_04/G02_05/G02_10 — and this class contains the sweep's best draws), the rest are
luck-censored (a never-lift or defect leg). **6/10 both-evaluable designs attempt a reorient
in ≥1 replica ⇒ the landscape's honest observable is P(express | design), not a scalar cos.**

**Candidates advancing (confirm r2/r3 LAUNCHED 22:19, same tag/store, ETA ~04:40):**
- **G02_00** — the only replica-consistent reorienter; mean 0.570 over 2 draws vs m05's
  draw-band mean ≈ 0.38; 3.9 cm from m05.
- **G02_05** — program-best single draw **0.887 / jerk 7.8** (beats m05's best draw 0.82,
  smoother than b33) on the design r0 scored −0.499 as an A-defect; 3.2 cm from m05.
At n=4 each: consistent ≥0.5 means ⇒ genuine better-than-m05-band candidates (then promote
via a proper head-to-head vs m05 with matched draw counts); a reversion to the wide band ⇒
the irresolvable class verdict extends to the candidates and replication alone cannot rank
designs at feasible cost.

**Evaluator lessons carried to P5:** A health grade is a *gate*, not a sufficient statistic
of delivery (G02_10 flipped 0.117→0.576 between same-grade WARN As); collapse propensity ⊥
delivery health ⊥ capability (G02_08 zero-abort static, G02_09 all-FAIL-A static, G02_05
aborting program-best); best-of-2 can't rank two-FAIL draws → `--a-attempts 3` or
collapse-rate-as-output; report per-design health-FAIL rate alongside.

**Strategic read (for the user):** replication at n=2 leaves a third of designs unbinnable
and the interesting ones live there; the A-predictor came back negative, so there is no cheap
gate that restores single-draw ranking. The two escapes are (a) brute replication (n≥4 only
on candidates — what the confirm leg does), or (b) the **morphology-conditioned policy**
(spike verdict: feasible with zero mjwarp changes via mjlab `expand_model_fields`, ~2–4 days
plumbing) which amortizes training across designs and turns evaluation into rollouts. Given
capability-is-common + expression-is-draw-gated, (b) is now the principled next build.

### P4 confirm — leg 1/4 (2026-07-13 ~00:00 tick): G02_00_r2 breaks the design's replica-consistency — the "consistent reorienter" bin was itself draw luck

**Result (confirm leg 1/4, 5684 s):** `G02_00_r2` — A passed on its **first** draw (WARN,
objheight 0.110, no abort); imit-B trained clean and **holds** (post-handoff min-z 0.1108, all
3 fingers 100% touch-frac) but **never attempts a reorient**: held-cos tail **0.107**, peak
**0.159**, verdict WARN (de-centering 6.1 cm path / 0.1 cm net + over-clamp 12.1 N; jitter
17.4 PASS). The grip is the hardest of G02_00's three draws — thumb-dominant clamp
18.0/13.0/5.4 N — where both expressing draws had lighter grips: within this design the
static draw is also the over-clamp draw, consistent with the hard-clamp↔static pairing seen
in G02_06 but not universal (G02_09 went static at 4.1 N), so it stays an observation, not a
rule.

**The headline: G02_00's replica-consistency dissolves at n=3.** Draws now
{0.504, 0.635, **0.107**}. The sweep's only "replica-consistent reorienter" — the label that
made it confirm-candidate #1 — was a coincidence of two expressing draws. Three consequences:

1. **The n=2 census is optimistic by construction.** 5/12 designs binned "consistent" at
   n=2, but consistency-at-n=2 is itself a draw-luck observable: G02_00 shows a design can
   pass it and still be expression-gated. The static bins (G02_06/G02_08/G02_09, 6 static
   tails) are more robust than the single reorienter bin was, but none are proven.
2. **Second same-grade-A flip** (after G02_10): r0 and r2 both kept health-WARN first-draw
   As (objheight 0.116/0.110), Δcos 0.397. Reinforces the P5 lesson — A health grade is a
   gate, not a sufficient statistic of the delivered state.
3. **P(express | design) framing strengthens:** G02_00 expresses in 2/3 draws. Under the
   n=2 sweep it looked like a *reliable* reorienter; the honest read was always "expresses
   more often than most" — a fraction, not a scalar.

**Confirm-bar math:** G02_00 mean over 3 draws = **0.415**; it needs r3 ≥ 0.754 to reach the
≥0.5 promotion bar at n=4. Unless r3 lands near the design's best draws, G02_00 resolves as
"expression-gated like everything else" and the irresolvable-verdict branch (⇒
morphology-conditioned policy build as the default next step) gains its first confirm-side
evidence. G02_05's r2/r3 remain the open half — its A (r2 t0) is training now.

Hold streak intact: min-z 0.1108 ⇒ **21/21 policy-producing legs ≥ 0.103** — pick-up/hold
stays solved through the confirm draws. Video
`docs/rl/videos/20260712_sweep/2353_G02_00_r2_handoff.mp4` (+ `.health.json`).

### P4 confirm — leg 2/4 (2026-07-13 ~02:00 tick): G02_05_r2 goes fully static — the program-best design's promotion bar is now mathematically unreachable

**Result (confirm leg 2/4, 5678 s):** `G02_05_r2` — A passed on its **first** draw (WARN,
objheight 0.1064, no abort; G02_05's second consecutive clean first-draw A after r0's
double-abort FAIL). imit-B trained clean and **holds** (post-handoff min-z 0.107, all 3
fingers 100% touch-frac, jitter 13.9 PASS, net drift 0.1 cm) but **never attempts**:
held-cos tail **−0.079**, peak **0.046** — the flattest peak of any G02_05 draw and in the
range of the sweep's proven statics (G02_08 peaks ≤ 0.11). Verdict WARN on sliding
(3.5 cm path / 0.1 cm net) + over-clamp (8.2 N).

**The headline: the promotion bar is decided one leg early.** G02_05's draws are now
{−0.499, 0.887, **−0.079**}, mean **0.103** — reaching a ≥0.5 mean at n=4 would need
r3 ≥ **1.69**, above the cosine ceiling. The program-best design **cannot** be promoted
under the confirm bar regardless of its last draw. Even on the charitable reading that
excludes r0 (it measured a broken A), the clean-draw mean is 0.404 and r3 must clear 0.692
— higher than 2 of the 3 observed draws. Combined with leg 1 (G02_00 needs r3 ≥ 0.754,
above ALL three of its observed draws), **both confirm candidates now require a
better-than-any-observed draw to survive**: the head-to-head-vs-m05 branch is effectively
dead, and the irresolvable-verdict-extends branch (⇒ morphology-conditioned policy as the
default next move, user decision pending) is all but confirmed with two legs still to land.
The remaining value of r3×2 is band/expression estimation, not promotion.

**FOURTH same-grade-A flip — and the largest.** r1 and r2 both kept health-WARN
first-draw As (objheight 0.106 both), yet Δcos = **0.966** (0.887 → −0.079). After G02_10
(Δ0.459), G02_00 r0↔r2 (Δ0.397), and G02_04, the pattern is now beyond doubt: with imit-B
sd ≈ 0.02 on a fixed A, essentially the entire outcome spread lives in delivered-state
differences that the A health grade cannot see. Whatever P5 ranks on, it cannot be
per-draw cos through an A-grade gate.

**Within-design grip note — expression tracks index recruitment 3/3.** G02_05 carries the
sweep's largest IK residual (index 12.84 mm, identical across replicas). Its three draws:
r0 index **0.9 N** (idle) → static; r1 index **10.8 N** (recruited) → 0.887; r2 index
**3.9 N** (weakest finger; thumb 9.1 / middle 11.7) → static. Expression has now co-moved
with index engagement in every G02_05 draw. Still an observation, not a rule (n=3, one
design), but it sharpens the "idle-finger identity" thread: for THIS geometry the index is
the load-bearing finger for reorient, and whether a draw recruits it is the coin being
flipped.

**Census updates:** G02_05 expresses 1/3 (was "program-best" on the n=2 sweep read);
expression-fraction framing now covers both confirm candidates (G02_00 2/3, G02_05 1/3).
Hold streak: min-z 0.107 ⇒ **22/22 policy-producing legs ≥ 0.103**. In flight: G02_00_r3
(A t0 training since 01:32, CEM lift 0.052); G02_05_r3 last; batch ETA ~04:40. Video
`docs/rl/videos/20260713_sweep/0128_G02_05_r2_handoff.mp4` (+ `.health.json`).

### P4 confirm — leg 3/4 (2026-07-13 ~05:00 tick): G02_00_r3 reorients at its design-best (0.681) — and still misses the promotion bar by 0.018; the confirm question is formally CLOSED

**Result (confirm leg 3/4, 8904 s — longest confirm leg):** `G02_00_r3` — both A attempts
ran to completion and **both health-FAILed** (t0 objheight 0.1166 kept, t1 0.1126; third
both-FAIL best-of-2 leg after G02_04/G02_09 — full budget spent, gate picked by objheight
among draws it can't rank). imit-B on the kept t0 **holds** (post-handoff min-z 0.1156) and
**reorients at the design's best**: held-cos tail **0.681**, peak **0.734**, ang-jerk 9.8
PASS, net drift 1.1 cm. Verdict **FAIL on idle_finger alone** (thumb 3.8 N / 0.31 touch,
middle 2.4 N / 0.26 — the index carries the roll at 6.7 N / 1.00): the fourth
capability-behind-a-FAIL row (L01_02, G02_03_r0, G02_10_r1 pattern; the softened flip bar
keeps earning its place).

**The headline: the confirm is decided — both candidates miss.** G02_00's four draws are
{0.504, 0.635, 0.107, **0.681**}: mean **0.482**, sd 0.26. The bar needed r3 ≥ 0.754; the
design answered with its best-ever draw and still fell 0.018 short. Honest caveat recorded:
with SEM ≈ 0.13 a 0.482-vs-0.5 miss is not a statistical distinction — but the bar was
pre-registered exactly so a near-miss wouldn't get relitigated on draw luck, and the band
read doesn't change the story: G02_00 is **m05-class** (m05's 3-draw band {0.82, 0.49,
−0.16}, mean 0.383) at 3.9 cm from m05 — a second sample of the same expression-gated
capability, not a dominator. With G02_05's bar already unreachable (leg 2), **the
head-to-head-vs-m05 branch is dead on both legs ⇒ irresolvable-verdict-extends is now the
formal confirm outcome; the morphology-conditioned policy build is the default next move
(user decision pending).** G02_05_r3 (last leg) measures its band only.

**A-grade inversion inside one design (n=4).** G02_00's two WARN-grade As delivered cos
0.504 and 0.107 (mean 0.31); its two FAIL-grade As delivered **0.635 and 0.681** (mean
0.66). The grade doesn't just fail to order outcomes — on this design it anti-orders them.
Small n, but stacked on the four same-grade flips it closes the case: P5 must not gate or
rank per-draw results through the A health grade.

**Grip note — G02_00's coin is clamp intensity, not index identity.** The design's one
static draw (r2) is its hardest clamp (thumb 18.0 N, mean tip 12.1 N, the design's only
over-clamp WARN, jerk 17.4); its best draw (r3) is its lightest grip (mean tip 4.3 N,
index-led with thumb/middle only intermittently touching). G02_05's expression tracked
index recruitment 3/3; G02_00's tracks (inversely) clamp force. Common factor across both
candidates: **the draw picks a grip style, and the grip style decides expression** — which
finger/force axis matters is design-specific, so no single scalar (residual, force, grade)
will predict it program-wide.

**Census updates:** G02_00 expresses 3/4 — the strongest expression fraction in the
program (m05 2/3, G02_05 1/3). Hold streak: min-z 0.1156 ⇒ **23/23 policy-producing legs
≥ 0.103**. In flight: G02_05_r3 — CEM clean (lift 0.050, persist 1/1/1), A t0 completed
but health-FAILed → t1 training since ~04:56 (objheight 0.1106, healthy); batch ETA
~07:30 (both r3 legs spent the full best-of-2 budget). On completion: n=4/n=4 band
close-out + program synthesis; GPU goes free. Video
`docs/rl/videos/20260713_sweep/0356_G02_00_r3_handoff.mp4` (+ `.health.json`).

### PROGRAM CLOSE-OUT (2026-07-13 06:09): confirm 4/4 done, no promotion — the policy-bottleneck program's answer, in full

G02_05_r3 landed cos 0.532 / jerk 44.9 FAIL (an expressing draw on a thrashy grip; min-z
0.0903 — the program's first sub-0.103 hold, still ≫ the 0.05 held bar). Final bands:

| design | draws | mean | expresses | read |
|---|---|---|---|---|
| m05 (ref) | 0.82 / 0.49 / −0.16 | 0.383 | 2/3 | the reference's own draw reality |
| **G02_00** | 0.504 / 0.635 / 0.107 / 0.681 | **0.482** | **3/4** | m05-class at 3.9 cm; best expression fraction in the program; misses the pre-registered ≥0.5 bar by 0.018 (SEM ≈ 0.13 ⇒ inseparable from m05) |
| **G02_05** | −0.499 / 0.887 / −0.079 / 0.532 | 0.210 | 2/4 | wide-band; still owns the program-best single policy (0.887 / jerk 7.8) |

**What the 2026-07-10 directive bought, end to end** (P1 rescue → P2 avar → P4 global12×2 →
confirm; ~70 h GPU, 40 pipeline legs, 12 global designs + 5 rescue designs + 2 controls):

1. **The user's intuition was right, and we measured *how* right.** The evaluation bottleneck
   is the per-design policy draw: pick-up "failures" were 100% optimizer noise (rescued
   everywhere, ultimately including every double-collapse design), and reorient outcomes carry
   per-draw sd 0.3–0.5 that no gate we possess can see (A grade even *anti-orders* outcomes
   within G02_00), overlapping designs 4 cm apart in the box.
2. **The landscape's honest structure:** graspable ⇒ liftable ⇒ holdable everywhere in the
   9-param box (this part of morphology co-design is CLOSED — geometry doesn't gate pick-up);
   reorient *capability* is widespread (6/10 designs express somewhere) but *expression* is a
   per-draw coin whose bias — the real design property — needs either many replicas
   (P(express|design), sd ~0.25 at n=4) or a fundamentally cheaper evaluator.
3. **No promotion:** m05 (a10→b33) remains the reference. G02_00 is a validated second
   m05-class region (useful for hardware robustness arguments: the design optimum is a
   plateau, not a peak); G02_05_r1 is the best single reorient policy the program has produced
   (0.887/7.8, video `G02_05_r1_handoff.mp4`).
4. **The path forward is architectural, not statistical:** A-predictor negative, replication
   cost-capped, gates saturated. The **morphology-conditioned policy** (one policy conditioned
   on the 9-vector across per-env randomized geometry; spike-verified zero-mjwarp-changes via
   mjlab `expand_model_fields`, ~2–4 days plumbing) turns per-design evaluation into rollouts
   and amortizes the draw noise across the whole box — the principled resolution of the
   evaluate-requires-optimize loop this program was launched to characterize. **Build decision
   is with the user.** GPU free as of 06:09.

**Comparison videos (2026-07-13, `scripts/make_sweep_video_grids.py`):**
[global12_lift_grid.mp4](videos/20260713_reorient/1010_global12_lift_grid.mp4) (best draw per design,
lift phase 2×-slowed — all 12 geometries deliver),
[global12_reorient_grid.mp4](videos/20260713_reorient/1010_global12_reorient_grid.mp4) (same draws
post-handoff — the axis where everything varies),
[global12_highlights.mp4](videos/20260713_reorient/1010_global12_highlights.mp4) (m05 a10→b33 +0.90 vs
G02_00 r3 +0.68 vs G02_05 r1 +0.89). Also embedded in webpaper `rl.typ` §policy-bottleneck.

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

- [handoff_demo.mp4](videos/20260601_reorient/1637_handoff_demo.mp4) — **The
  end-to-end handoff demo.** Policy A picks up the flat-laying
  cylinder and holds it stable (3 s); Policy B then reorients it
  toward vertical (4 s). Both policies running live in simulation,
  concatenated in one rollout. Generated by
  [scripts/rl_demo_handoff.py](../../scripts/rl_demo_handoff.py).
- [reorient_comparison_grid.mp4](videos/20260601_reorient/1650_reorient_comparison_grid.mp4)
  — **2×2 side-by-side comparison.** Top-left: v3 (reward never fires).
  Top-right: v4 floor-bracing. Bottom-left: v5 (no floor = no rotation).
  Bottom-right: Policy A → Policy B concatenated (the two-policy
  solution: pickup, then in-hand rotate). One-screen overview of the
  whole journey.
- [v4_peak_floorbracing.mp4](videos/20260601_reorient/1502_v4_peak_floorbracing.mp4)
  — v4 at iter ~950 (peak target_axis_progress). The cylinder is rolled
  while its distal end braces against the floor. **This is the
  "floor-bracing" behavior** — RL found that the ground reaction force
  helps with rotation when fingers alone can't.
- [v4_final_floorbracing.mp4](videos/20260601_reorient/1502_v4_final_floorbracing.mp4)
  — v4 at iter ~2000 (final). Same strategy, slightly regressed from
  peak.
- [v5_final.mp4](videos/20260601_reorient/1502_v5_final.mp4) — v5 final. Floor
  contact forbidden; policy collapses to "hold the high lift, don't
  rotate." Showcases the no-floor-no-rotation result.
- [policyA_lift.mp4](videos/20260601_reorient/1630_policyA_lift.mp4) — `medium_flat_stable_v1`
  rollout: cylinder lying flat on floor → lifted and held horizontal.
  The "Policy A" half of the two-policy chain.
- [policyB_final.mp4](videos/20260601_reorient/1502_policyB_final.mp4) — Policy B v1
  final. True in-hand reorientation from a pre-lifted spawn. Visibly
  jittery (sim-only exploit) but the cylinder genuinely rotates
  without floor contact.
- [v3_final.mp4](videos/20260601_reorient/1610_v3_final.mp4) — v3 final. Episodes die
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

---

## 6-DIM XY-ONLY SWEEP + policy-learning quick-fix analysis (2026-07-20)

User (heading out overnight, ~until 09:00): run a global12x2-style sweep but **freeze the proximal
phalange lengths** → explore the **6 XY placement dims only**; and analyse past results for **quick
fixes to make policy learning more robust / reach its potential** (policy learning = the established
bottleneck). Full standalone note: `docs/notes/policy_bottleneck_quickfixes.md`; live runbook:
`morph_sweep_STATUS.md` §"6-DIM XY-ONLY SWEEP".

**Quick-fix analysis (zero GPU-training cost).** Decomposed the evaluator noise: B-side variance is
already SOLVED by the imit prior (sd ±0.02); the DOMINANT open term is the **Policy-A draw** (sd
0.3–0.5), which the A health gate cannot select on — gate-invisible and mildly *anti-ordering*
(health-FAIL As gave G02_00 its best draws, per §P2 avar). The highest-leverage candidate fix was a
**cheap downstream A-selector**: run the proven reorienter b33 **zero-shot** on each A's delivered
grip and keep the most reorientable (`scripts/probe_a_reorientability.py`, tested on all 31 on-disk
kept-A → known-trained-imit-B pairs from global12x2 + confirm).

**Result: NEGATIVE (an honest, useful one).** Within-design Spearman +0.345, best-A-hit 6/11, and it
**fails on exactly the standout draws** — G02_00_r3 (trained **0.681**, probe 0.14) and G02_05_r1
(trained **0.887**, probe **0.00**, b33 dropped it); cf_m05/cf_l13 mildly anti-correlate (rho −0.50).
Only a *coarse* signal survives (trained ≥0.5 designs avg probe +0.39 vs +0.07 for static, gap +0.31).
**Finding: zero-shot reorientability ≠ trainable reorientability** — the best trainable grips are
precisely the ones a *fixed* reorienter cannot roll cold, so the probe would discard the top designs.
This extends `a_quality_predictor.md` (no cheap A-quality predictor from scorecard metrics) to
downstream zero-shot probes too. **There is no cheap A-selector shortcut; A-draw variance is
intrinsic — which is exactly why the morphology-conditioned policy remains the real fix.** Data:
`docs/experiments/PROBE_A_REORIENTABILITY.{json,txt}`.

**Launched (2026-07-20 ~20:33 MDT, detached/resumable/waiter-armed):** the 6-dim sweep with the
blessed, comparable evaluator + the one bump the confirm close-out endorsed —
`morph_pipeline_sweep.py --morph-set global --freeze-len --n 12 --seed 6 --replicas 2
--tag global6xy --b-recipe imit --a-attempts 3` (designs H06_00…H06_11, lens frozen at m05). Honest
expectation: with the A-draw term unfixed, the XY-only landscape likely hits the same draw-noise wall
as 9-dim; the yield is whether freezing len lowers the A-collapse/hostile-geometry rate and whether an
XY-only map is any more resolvable. Analysis on completion per the STATUS decision tree.

**RESULT (2026-07-22, 24/24 done; pooled `MORPH_PIPELINE_global6xy_POOLED.md`).** Freezing the
proximal length **narrows the landscape UPWARD but does NOT touch the wall.** 6-dim vs 9-dim
design-means: floor **−0.105 vs −0.388**, median +0.224 vs +0.274, peak **+0.796 vs +0.482**,
#(mean ≥ 0.5) **1 vs 0**. Removing the length axis eliminated the worst designs (it contributed
mostly downside) and the XY box holds a peak above any 9-dim design — **H06_04 mean 0.796 (draws
0.852/0.741), the program's strongest reorienter** — while the *typical* design (median), the
per-draw expression wall (sd 0.33; 8/23 draws express ≥0.5), and A-training fragility (~49% leg-abort
≈ 9-dim's 47%, clustering by design: H06_06/H06_11 5/6 aborts vs H06_02/05/10 0/2) are all unchanged.
The A-collapse rate matching (49% ≈ 47%) directly refutes "the length axis was a special
noise/hostility source." **H06_06** is the program's clearest lift-hostile candidate (r0 all-A-collapse
to objheight 0.0, r1 barely lifts). **No promotion:** H06_04 is n=2 (9-dim G02_00 regressed 0.57→0.48
at n=4), so an **n=4 confirm** on H06_04 + H06_08 is running (same geometry, `--replicas 4 --only`).
Structural conclusion reinforced: the landscape is draw-gated, not geometry-flat; the
morphology-conditioned policy remains the real fix, and freezing len is a terrain improvement, not a
bottleneck fix.

### 6-dim interim — design 1/24 H06_00_r0 (2026-07-20 ~23:30 MDT tick): first XY-only leg holds cleanly but is STATIC — the draw-gate is unchanged by freezing len

First completed leg of the 6-dim sweep (`global6xy.txt` row 1; worker healthy, now on H06_01_r0's
Policy B). H06_00's XY placement is a moderate in/down move off m05 (Δm05 thumb −1.8 cm y, index
−2.7 cm y, middle −2.6 cm x / −3.8 cm y; the three proximal lens **confirmed frozen at m05**, Δ 0.0).
Pipeline ran clean end-to-end in **96 min**: CEM graspable (lift 0.050, persist 1/1/1), Policy A held
on the **FIRST draw** (`model_609`, objheight 0.117, WARN, no abort — no A-collapse), imit-B trained
to `model_270`.

**Handoff = a textbook static hold.** Post-handoff min-z **0.116** (≫0.05 floor — holds beautifully),
but held-cos **−0.044** with peak-cos **0.02**: the object never even transiently rotated toward
vertical. All three fingers touch (touch-frac 1/1/1, forces thumb 12.0 / index 7.4 / middle 16.7 N —
**no idle-finger, no degenerate pinch**), so the two WARNs are the familiar pair: over-clamp (mean tip
12.1 N ≫ 3 N sat) and de-centering (slide-ratio 57, path 7.9 cm vs net 0.1 cm — the object slides in
place rather than being carried). Net verdict **WARN**, driven entirely by clamp+slide, not by a drop
or a defect.

**Reading (n=1, one draw — not a design verdict).** This is the *low* end of the m05 draw-band
{0.82, 0.49, −0.16}: a static-hold draw, exactly the draw-gated-expression pattern the 9-dim program
closed on. One early, weakly-positive signal for the freeze-len question: **no A-collapse here** (A
held on draw 1 of an allowed 3), consistent with "no lift-hostile geometry in the box" carrying over
to the XY-only slice — but n=1 says nothing yet about whether len-freezing lowers the collapse rate or
sharpens resolvability. Note H06_01_r0's A already needed **2 attempts** (still held), so the A-draw
term is live here too. Video `docs/rl/videos/20260720_sweep/2209_H06_00_r0_handoff.mp4`; scorecard
`…H06_00_r0_handoff.health.json`. Continuing to accumulate legs per the STATUS tree (docs/commit only
while the worker runs).

### 6-dim interim — designs 2–3/24 H06_01–02_r0 (2026-07-21 ~03:00 MDT tick): two more XY-only legs — both HELD, both STATIC (draw-gate intact)

Rows 2–3 of `global6xy.txt` (worker healthy, now on H06_03_r0's Policy B). Both cleared the grasp
gate and held post-handoff, both static — the same pattern as design 1, reinforcing the 3/24
milestone the STATUS log already flagged. Committing the two handoff media pairs the last tick's
milestone commit left untracked (`H06_01_r0_handoff.{mp4,health.json}`, `H06_02_r0_handoff.*`).

- **H06_01_r0** — CEM lift 0.055 (persist 1/1/1); Policy A held on the **2nd draw** (`model_609`,
  objheight 0.115, no abort). Handoff **minZ 0.110 / cos −0.048 / peak 0.056** — static. All three
  fingers touch (touch-frac 0.89/1.0/1.0, forces 7.9/5.6/9.2 N — no idle, no pinch); WARNs are the
  usual over-clamp (mean tip 7.5 N) + de-centering (slide path 12.1 cm vs net 1.3 cm). Verdict WARN.
- **H06_02_r0** — CEM lift 0.051 (persist 1/1/1); Policy A held on the **1st draw** (objheight 0.118,
  no abort). Handoff **minZ 0.118 / cos −0.119 / peak 0.088** — static, and the **jerkiest** leg so
  far (ang-jerk 26.8, WARN) with the heaviest slide (path 21.4 cm vs net 0.3 cm, slide-ratio 69) and
  over-clamp 10.2 N. Verdict WARN. No drop, no degenerate pinch — clamp+slide again.

**Reading (still n=1 per design — not verdicts).** All three r0 held-cos land at the *low* end of the
m05 draw-band {0.82, 0.49, −0.16}: −0.044 / −0.048 / −0.119, i.e. three straight static-hold draws —
exactly the draw-gated-expression wall (m05 itself draws static ~1/3 the time; the `_r1` replicas are
what decide expression).

**Trainability watch — measured at the ATTEMPT level (the honest number).** The question freezing len
was meant to answer is "does the XY-only box have less lift-hostile geometry → fewer A-collapses?"
Checking the per-attempt `logs/sweep_A_H06_*_t*.trainer.log.COLLAPSED` watchdog sentinels: across the
first 4 designs, **3 of 7 A attempts collapsed (~43%)** — H06_00 0/1, H06_01 1/2 (t0 collapsed then
t1 held), H06_02 0/1, H06_03 2/3 (t0+t2 collapsed, t1 held). That is **comparable to the 9-dim
global12x2 ~47% abort rate, NOT lower.** What *does* differ is the design-level outcome: **0 of 4
designs aborted**, because this run adds `--a-attempts 3` (QF2, best-of-3) and one of the three shots
holds. So the early read is the opposite of the tempting one — **best-of-3 is what's carrying
trainability, not len-freezing**; freezing the proximal lengths shows no clear reduction in intrinsic
per-attempt A-collapse yet. Far too small to conclude (n=4 designs); the pooled r0/r1 table at DONE,
with attempt-collapse counted per-attempt not per-design, decides.

### 6-dim interim — design 4/24 H06_03_r0 (2026-07-21 ~03:30 MDT tick): 4th XY-only leg holds — first non-negative held-cos, still functionally STATIC; lightest-clamp leg
Row 4 of `global6xy.txt` (worker healthy, now on H06_04_r0's Policy A — its t0 already collapsed at
iter 54/objheight 0.015, on t1). H06_03_r0 completed 03:26, just after the 03:00 tick's snapshot;
committing its handoff media pair (`H06_03_r0_handoff.{mp4,health.json}`) that landed untracked.

- **H06_03_r0** — CEM lift 0.051 (persist 1/1/1); Policy A held only on the **3rd draw** (best
  `model_609`, objheight 0.118, no abort) — **2 of its 3 attempts collapsed** (t0 iter 220 @0.012,
  t2 iter 40 @0.030; t1 held), the worst A-leg of the four and the reason it logged `A×3`. Handoff
  **minZ 0.116 / cos +0.006 / peak 0.114** — the **first non-negative held-cos of the sweep**, but
  still ~0 (peak barely 0.11) = functionally static, not a reorient. Scorecard WARN: **lightest clamp
  of the four legs** (mean tip **5.6 N** vs 7.5/10.2/12.1), jitter PASS (17.3), de-centering WARN
  (slide path 13.0 cm vs net 1.0 cm), and an **idle-finger WARN** — mean tip-contacts 2.32<3 with the
  **index under-recruited** (index 2.9 N/touch 0.75 vs thumb 6.7 N/0.74, middle 7.2 N/0.83); not a
  true idle/pinch (all three touch) but the index carries least here. late-finger PASS, drop PASS.

**Reading (still n=1 per design).** Four straight static r0 draws now: held-cos −0.044 / −0.048 /
−0.119 / **+0.006**. H06_03's +0.006 is marginally the "best" of the four but functionally identical
to static — the draw-gated-expression wall is intact through 4/24 legs, exactly as H2 predicts (the
`_r1` replicas decide expression, not these singletons). The one *interpretable* XY signal so far is
that H06_03 is the **lightest-clamp** leg (5.6 N) — a small hint that XY placement moves the clamp
regime even when reorient stays unexpressed; worth confirming against the pooled clamp-force column at
DONE, not actionable now. Trainability unchanged: H06_03's 2/3-collapse (and H06_04's t0 collapse
already) keep the attempt-level rate near the ~43% flagged above — no len-freeze benefit visible.

### 6-dim interim — design 5/24 H06_04_r0 (2026-07-21 ~06:00 MDT tick): FIRST DYNAMIC REORIENT of the 6-dim sweep — held-cos 0.852, breaks the 4/4-static streak
Row 5 of `global6xy.txt` (worker healthy, now on H06_05_r0's Policy A, t0). H06_04_r0 completed
05:15, the first leg of the sweep to **express reorientation** — committing its untracked handoff
media pair (`H06_04_r0_handoff.{mp4,health.json}`).

- **H06_04_r0** — CEM lift 0.050 (persist 1/1/1); Policy A held on the **3rd draw** (best `model_609`,
  objheight 0.109, no abort) — **2 of its 3 attempts collapsed** (t0, t2; t1 held), logging `A×3` like
  H06_03. Handoff **minZ 0.103 / held-cos 0.852 / peak-cos 0.889** — a **genuinely dynamic reorient**,
  the object rotated toward vertical and stayed there. Scorecard WARN (2 soft checks): drop PASS
  (0.103≫0.05), jitter PASS (17.9), late-finger PASS, **idle-finger PASS** (all three recruited —
  thumb 9.8 N/1.00, index 13.5 N/1.00, middle 15.6 N/1.00; no pinch), de-centering WARN (slide path
  4.9 cm vs net 0.1 cm), over-clamp WARN (mean tip **13.0 N**, the heaviest-clamp leg). So this draw
  is a *hard-gripping* reorient — clamps hard, slides in place, but rotates the screwdriver up.

**Reading — the draw-gate breaks, exactly as H2 predicts.** Five r0 draws now: held-cos −0.044 /
−0.048 / −0.119 / +0.006 / **0.852**. The first four were static; H06_04 is the **first strong
positive**, landing **at/above the m05 draw-band high** ({0.82, 0.49, −0.16}). This is the expected
signature of **draw-gated expression**, not of a special design: freezing the proximal lengths did
**not** remove reorient capability from the 6-dim XY box, and a single lucky draw expresses it near
the program ceiling (cf. 9-dim G02_05_r1 0.887). It is emphatically **not** a promotion signal at
n=1 — H06_04's `_r1` replica (and the pooled r0/r1 mean at DONE) decide whether *this XY placement*
expresses more reliably than the static four, or whether it merely drew well once. Note the contrast
with the static legs: this dynamic draw is the **heaviest-clamp** leg (13.0 N) while the "best" static
leg H06_03 was the **lightest** (5.6 N) — i.e. clamp force is not (yet) predictive of reorient
expression across these singletons; that's a pooled-column question for DONE.

**Trainability watch (5 designs).** Per-attempt COLLAPSED sentinels now: H06_00 0/1, H06_01 1/2,
H06_02 0/1, H06_03 2/3, H06_04 2/3 → **5 of 10 A attempts collapsed (~50%)**, at/slightly-above the
9-dim global12x2 ~47% and up from the ~43% at 4 designs. **Design-abort stays 0/5** — the
`--a-attempts 3` best-of-3 rescue (QF2) is doing the work, not a len-freeze reduction in intrinsic
A-collapse propensity. The len-freeze "more-trainable" hypothesis has **no** support at 5 legs; the
pooled r0/r1 collapse column at DONE is the real test.

### 6-dim interim — designs 6–7/24 H06_05_r0 (partial reorient, scorecard-FAIL) + H06_06_r0 (FIRST DESIGN-ABORT of the 6-dim sweep: A never lifts) (2026-07-21 ~07:15 MDT tick)
Rows 6–7 of `global6xy.txt` (worker healthy, now on H06_07_r0's Policy A, t2). Committing H06_05's
handoff media pair (`H06_05_r0_handoff.{mp4,health.json}`) that the 6/24 milestone commit left
untracked (same orphaning as H06_01/02/03/04). H06_06 produced **no handoff media** — its A never lifted.

- **H06_05_r0** — CEM lift 0.052 (persist 1/1/1); Policy A held on the **1st draw** (objheight 0.118,
  no abort, `model_609`). Handoff HELD (**minZ 0.1155** ≫ 0.05) but verdict **FAIL** — the scorecard's
  jitter check FAILs at **ang-jerk 46.4 1/s²** (the thrashiest leg of the sweep) and de-centering WARNs
  at slide-ratio 18.2 (path 14.0 cm ≫ net 0.8 cm). Held-cos-tail **0.34**, **peak-cos 0.468** → the
  policy got the object *partway* to vertical then oscillated. This is exactly why we **judge on the
  scorecard, not reward or raw cos**: a bare +0.34 cos looks like "partial success," but the trajectory
  is a thrash, not a controlled reorient — the FAIL is correct. So the 6-dim r0 series to date is
  {−0.044, −0.048, −0.119, +0.006, **0.852**, +0.34(FAIL), [abort]}: **one clean strong reorient
  (H06_04), one thrashy partial (H06_05), four static** — the same draw-gated-expression shape as
  9-dim. Video `docs/rl/videos/20260721_sweep/0652_H06_05_r0_handoff.mp4`.
- **H06_06_r0 — FIRST DESIGN-LEVEL ABORT of the 6-dim sweep.** CEM found a **graspable** grip (lift
  0.055, persist 1/1/1) — the open-loop CEM grip + scripted lift holds — but **Policy A never lifted**:
  all **3 attempts collapsed** (`sweep_A_H06_06_r0_t{0,1,2}.trainer.log.COLLAPSED` all present, best
  objheight **0.0** < the 0.06 gate). Best-of-3 rescue (QF2) **failed outright** here → the design is
  skipped, no B, no reorient measurement. **Interpretation:** this is an **A-trainability-hostile** XY
  placement, *not* a graspability-hostile geometry — CEM's static grip is fine, but the learned
  lift-from-onset policy can't be trained on it in 3 seeds. It's the first concrete crack in the 9-dim
  program's "pick-up/hold solved everywhere, no lift-hostile geometry" verdict: freezing the lengths
  and moving only XY *can* land a placement where RL-A fails 3/3 even though a hand-tuned grasp holds.

**Trainability watch (7 designs) — len-freeze hope now NEGATIVE with a hard counterexample.** Per-attempt
COLLAPSED sentinels: H06_00 0/1, H06_01 1/2, H06_02 0/1, H06_03 2/3, H06_04 2/3, H06_05 0/1, **H06_06
3/3** → **8 of 14 A attempts collapsed (~57%)**, now **above** the 9-dim global12x2 ~47% and climbing
(43%→50%→57% at 4→5→7 designs). **Design-abort 1/7 (~14%)** — the first one, H06_06, where best-of-3
could not rescue. This is the clearest evidence yet that **freezing the proximal-phalange lengths does
NOT calm Policy-A training** (the original motivation for the 6-dim sweep); if anything the intrinsic
A-collapse propensity in the XY-only box is comparable-to-worse. The pooled r0/r1 collapse column at
DONE remains the definitive test, but the direction is set. Continuing docs/commit-only per the STATUS
tree (worker owns the GPU).

### 6-dim interim — design 8/24 H06_07_r0 (2026-07-21 ~09:45 MDT tick): SECOND clean reorienter (cos 0.547) — and the cleanest-QUALITY reorienting draw of the box
Row 8 of `global6xy.txt` (worker healthy, started H06_08_r0's CEM 09:44). Committing H06_07's handoff
media pair (`H06_07_r0_handoff.{mp4,health.json}`).

- **H06_07_r0** — CEM lift 0.052 (persist 1/1/1); Policy A held on the **3rd draw** (best `model_609`,
  objheight 0.115; the first 2 attempts collapsed — same 2/3 as H06_03/H06_04). Handoff **minZ 0.117 /
  held-cos 0.547 / peak-cos 0.629**, verdict **WARN**. What makes this draw notable is not the
  magnitude (0.547 < H06_04's 0.852) but the **quality**: it is the **smoothest** reorienting leg of
  the sweep (ang-jerk **10.4**, PASS — vs H06_04's 17.9 and H06_05's 46.4-thrash) at the **lowest
  clamp force** (5.0 N — vs H06_04's 13.0), with **all three fingers engaging at step 1** (late_finger
  PASS) and a **balanced grip**: thumb 4.8 / index 5.3 / middle 5.0 N. The **thumb is fully recruited
  here** (~4.8 N), in contrast to the historic m05 degenerate pinch where the thumb idled ~1.6 N while
  index+middle clamped ~8 N (see [[project_policyB_v2_overnight]]). Remaining WARNs are mild:
  contact-count 1.67<3, de-centering (path 3.6 >> net 0.7cm = sliding not translating), over-clamp
  5.0 N (reward saturates ~3 N). So this is a **genuine controlled partial reorient**, the opposite
  failure mode from H06_05's thrash — where H06_05 scored a similar raw cos (0.34) by jittering the
  object, H06_07 rotates it smoothly and gently to ~0.55.

- **What the box now shows (8 r0 legs).** r0 held-cos series
  {−0.044, −0.048, −0.119, +0.006, **0.852**, +0.34 (FAIL), [abort], **0.547**}; non-abort mean
  **+0.219**, sd **0.34** — statistically the **same draw-noise wall** as the 9-dim global12x2 result
  (per-draw sd 0.3–0.5). But the *content* of the reorienting tail is now richer: **two WARN
  reorienters** span the quality axis — H06_04 **strong-but-forceful** (0.852, 13 N, jerk 17.9) and
  H06_07 **gentle-and-smooth** (0.547, 5.0 N, jerk 10.4) — plus one **thrashy FAIL** partial (H06_05,
  0.34). ⇒ Reorient **capability and quality both clearly exist** in the XY-only box; as in 9-dim the
  binding constraint is **expression draw** (P(express | design)), not geometry. **8/8 held** (minZ
  0.103–0.117): pick-up/hold still solved at every non-abort placement. These remain n=1 per design —
  the `_r1` replicas (and the pooled r0/r1 mean at DONE) decide whether *these XY placements* are
  reorient-capable or merely lucky draws.

**Trainability watch (8 designs) — unchanged NEGATIVE.** Per-attempt A-collapses: H06_00 0/1, H06_01
1/2, H06_02 0/1, H06_03 2/3, H06_04 2/3, H06_05 0/1, H06_06 3/3, **H06_07 2/3** → **10 of 17 A attempts
collapsed (~59%)**, holding **above** the 9-dim ~47%. **Design-abort 1/8 (~13%).** Freezing the
proximal-phalange lengths continues to NOT calm Policy-A training. Continuing docs/commit-only per the
STATUS tree (worker owns the GPU).

### 6-dim interim — designs 9–10/24 (H06_08_r0 + H06_09_r0) (2026-07-21 ~13:05 MDT tick): two more draw-gated non-expressers close the r0 pass to 10/12
Rows 9–10 of `global6xy.txt` (worker healthy, advanced to H06_10_r0's CEM at 13:01). Committing the
H06_08 **and** H06_09 handoff media pairs (`H06_0{8,9}_r0_handoff.{mp4,health.json}`) — the 9/24
milestone commit narrated H06_08's numbers but left its media untracked (same orphaning as
H06_01..05/07); H06_09 is brand-new this tick. Both are non-expressing draws that **held cleanly**.

- **H06_08_r0** — CEM lift 0.052 (persist 1/1/1); Policy A held on the **1st draw** (best `model_609`,
  objheight 0.110, **1 attempt, no collapse**). Handoff **minZ 0.106 / held-cos 0.204 / peak-cos
  0.432**, verdict **WARN**. A **partial-but-thrashy** draw in the H06_05 mold: the object rotates
  part-way (peak 0.43) but the tail settles at only 0.20 with **ang-jerk 32.1** (WARN) and
  **de-centering by sliding** (path 8.5 ≫ net 0.4cm, slide-ratio 19.8) at **11.3 N** clamp — it jitters
  the object rather than controlling it. Not an expresser (0.204 < 0.5).

- **H06_09_r0** — CEM lift 0.054 (persist 1/1/1); Policy A held on the **2nd draw** (best `model_609`,
  objheight 0.114; 1st attempt collapsed). Handoff **minZ 0.111 / held-cos −0.282 / peak-cos 0.064**,
  verdict **WARN**. A **static/adverse** draw — the object never rotates toward vertical (peak 0.06) and
  the tail drifts slightly the wrong way; ang-jerk 14.7, clamp 10.2 N, drift 0.1cm. Held cleanly but no
  reorient — the **modal draw outcome** of both the 6-XY box and 9-dim.

- **What the box now shows (10 r0 legs, 9 evaluable).** r0 held-cos series
  {−0.044, −0.048, −0.119, +0.006, **0.852**, +0.34 (FAIL), [abort], **0.547**, +0.204, −0.282};
  non-abort mean **+0.16**, sd **0.34** — the mean eased down from +0.22 as the two new draws landed
  static/partial, but the **shape is unchanged**: same draw-noise wall as 9-dim (per-draw sd 0.3–0.5),
  **2/9 express ≥0.5** (H06_04 strong-forceful 0.852; H06_07 gentle-smooth 0.547), the rest
  static-to-partial. **9/10 held** (minZ 0.106–0.117; only the H06_06 lift-miss fails). ⇒ Reorient
  capability + quality exist in the XY-only box; the binding constraint is **expression draw**
  P(express | design), exactly the 9-dim story. Still n=1/design — the `_r1` replicas (and the pooled
  r0/r1 mean at DONE) decide whether these XY placements are reorient-capable or merely lucky draws.

**Trainability watch (10 designs) — unchanged NEGATIVE.** Per-attempt A-collapses: H06_00 0/1, H06_01
1/2, H06_02 0/1, H06_03 2/3, H06_04 2/3, H06_05 0/1, H06_06 3/3, H06_07 2/3, H06_08 0/1, **H06_09 1/2**
→ **11 of 20 A attempts collapsed (~55%)**, holding **above** the 9-dim ~47%. **Design-abort 1/10
(~10%).** Freezing the proximal-phalange lengths still does not calm Policy-A training. The r0 pass is
**10/12 done** (H06_10, H06_11 remain, then the r1 replica pass begins). Continuing docs/commit-only per
the STATUS tree (worker owns the GPU).

### 6-dim interim — designs 11–12/24 (H06_10_r0 + H06_11_r0): r0 PASS COMPLETE (12/12) (2026-07-21 ~16:46 MDT tick)
Rows 11–12 of `global6xy.txt` close the r0 map; the r1 replica pass has begun (worker on **H06_00_r1's
Policy A, 2nd attempt**, ETA ~30 min at 16:46 — GPU busy). Committing the two orphaned handoff media
pairs (`H06_1{0,1}_r0_handoff.{mp4,health.json}`) + this note — the 15:22 r0-complete commit shipped
json/txt/STATUS only, so both these media and the `reorientation.md` narrative were 2 rows behind.

- **H06_10_r0** — CEM lift 0.051 (persist 1/1/1); Policy A held on the **1st draw** (best `model_609`,
  objheight 0.116, no collapse). Handoff **minZ 0.111 / held-cos −0.274 / peak-cos 0.01**, verdict
  **FAIL**. A **thrashy static** draw: all three fingers grip firmly and balanced (8.6/9.4/7.7 N,
  touch_frac 0.83/0.98/0.97 — no idle/late finger) yet the object never rotates (peak 0.01) and the
  tail drifts slightly adverse (−0.274) while the hand **jitters hardest of the whole box** (ang-jerk
  **41.0**, FAIL) and slides it (path 20.2 ≫ net 0.7cm). Firm grip, zero productive rotation — a
  non-expresser in the H06_02 mold but jerkier.

- **H06_11_r0** — CEM lift 0.052 (persist 1/1/1), but the **weakest leg of the r0 pass**: Policy A
  **aborted** (best-of-3 exhausted, salvaged `model_150`, objheight 0.121, abort flag set) and Policy
  **B watchdog-aborted** (salvage-eval of last ckpt `model_50`). Handoff **minZ 0.132 / held-cos 0.054 /
  peak-cos 0.251**, verdict **FAIL** on **idle-grip** — all three fingertips read **0.0 N / touch_frac
  0.0**: the object is *cradled/balanced* (minZ 0.132, highest of the box) rather than gripped, and
  reorient is ~static (0.054). A degenerate salvaged leg, counted as a produced FAIL row (not a full
  skip like H06_06).

**r0 PASS COMPLETE — the XY-only landscape verdict (n=1 map).** 12/12 legs; 11 evaluable (H06_06 =
lift-miss skip). Held-cos series {−0.044, −0.048, −0.119, +0.006, **0.852**, +0.34 (FAIL), [abort],
**0.547**, +0.204, −0.282, −0.274, +0.054}: **mean +0.112, sd 0.333, max 0.852, 2/11 express ≥0.5**
(H06_04 strong-forceful, H06_07 gentle-smooth). **Held 11/12** (minZ 0.103–0.132; only H06_06 fails to
hold). **A-collapse final = 13/24 attempts (54%)**; **design-abort 1/12** (H06_06). ⇒ **On every axis
the 6-XY box is statistically indistinguishable from the 9-dim global12x2 landscape:** pick-up/hold
solved everywhere (11/12), reorient common-but-draw-gated (same sd ~0.33 wall, express ~2/11 ≈ 9-dim's
~1/4), and Policy A fragile at the **same** rate (54% vs ~47%). **Freezing the proximal-phalange length
neither calms A-training nor sharpens the reorient landscape — it is not a special axis.** The lone
lift-miss H06_06 mirrors 9-dim G02_11 (all A draws → objheight 0.0). Still n=1/design; the r1 replica
pass now underway (and the pooled r0/r1 mean at DONE) decides whether the two standouts (H06_04 0.852,
H06_07 0.547) replicate or were lucky draws. Continuing docs/commit-only per the STATUS tree (worker
owns the GPU).

### 6-dim interim — FIRST r0/r1 PAIRS (H06_00, H06_01): both static r0 draws FLIP to expressers (2026-07-21 ~20:15 MDT tick)
The r1 replica pass has produced its first two rows, giving the sweep's **first head-to-head r0/r1
pairs** — and both are a clean, within-design confirmation of H2 (draw-gated expression): a static r0
draw is **not** a measurement of design incapability, it is draw-luck that the replica flips.

- **H06_00** — r0 **−0.044** (static, WARN) → **r1 held-cos 0.611** (WARN, `H06_00_r1_handoff.mp4`).
  The r1 draw is a genuine reorienter: **all three fingers engage at step 0** (late_finger PASS),
  object held (minZ 0.115), peak-cos 0.659. WARN on jitter (ang-jerk 26.7 — thrashy), de-centering
  (slide path 8.6 ≫ net 0.4cm) and over-clamp (8.4 N); grip thumb 4.9 / index 9.3 / middle 11.1 N
  (thumb recruited but light, index/middle heavy). **Pair {−0.044, 0.611}: mean 0.284, range 0.655.**
- **H06_01** — r0 **−0.048** (static, WARN) → **r1 held-cos 0.51** (WARN, `H06_01_r1_handoff.mp4`).
  Smoother than H06_00_r1 (**jitter PASS, ang-jerk 14.7**) but forceful — over-clamp 12.8 N, grip
  thumb 13.2 / index 15.4 / middle 9.9 N (all firmly engaged, late_finger PASS index@1). Held minZ
  0.112, peak-cos 0.637; WARN on de-centering (slide 8.0 ≫ 0.3cm) + over-clamp. **Pair {−0.048, 0.51}:
  mean 0.231, range 0.558.**

**Reading.** The two designs that r0 happened to draw **static** (−0.04 each, indistinguishable from
the box's non-expressers) both draw a **clean ≥0.5 reorienter** on their replica. Within-design spread
(range 0.56–0.66) already **dwarfs** the entire between-design r0 signal (r0 mean +0.112, sd 0.333) —
i.e. the per-draw sd 0.3–0.5 wall is live in the pairs, and the r0 map's static verdicts on these two
designs were pure draw-luck. This is the strongest single-tick evidence in the 6-XY sweep that **design
capability is unresolvable at n≤2**: had the sweep stopped at r0 it would have mis-labelled H06_00 and
H06_01 as non-reorienters. Both r1 draws are WARN (never a clean PASS), consistent with every reorienter
in this box and the 9-dim landscape — no design yet lifts the *quality* ceiling, only expresses. Neither
edges the standouts (H06_04 0.852, H06_07 0.547); those two designs' replicas are still pending.

**Trainability watch (running).** Both r1 legs so far were A×2 (one collapse + kept model each):
r1 = 2 collapses / 4 attempts, so combined **15 of 28 A attempts collapsed (~54%)** — unchanged, still
matching/edging the 9-dim ~47% and refuting the "freezing len calms A-training" hope. Worker healthy on
**H06_02_r1's Policy A** (~37 min in). Continuing docs/commit-only per the STATUS tree (worker owns the
GPU); pooled r0/r1 means + memory update at DONE.

### 6-dim interim — THIRD pair H06_02: the FIRST design STATIC in BOTH draws (draw-gate is per-design) (2026-07-21 ~21:46 MDT tick)
The r1 pass added **H06_02_r1** — the sweep's third completed r0/r1 pair, and the first one that does
**not** flip. It sharpens the previous tick's headline: "both static-r0 designs flip" was premature (it
was only H06_00/01); H06_02 is the counterexample.

- **H06_02** — r0 **−0.119** (static, WARN, peak-cos 0.088) → **r1 −0.091** (WARN, static, peak-cos
  0.165, `H06_02_r1_handoff.mp4`). **Pair {−0.119, −0.091}: mean −0.105, range 0.028** — both draws
  land in nearly the *same* static place. r1's grip is lopsided (index 13.3 / thumb 9.7 / middle 3.0 N;
  over-clamp 8.7 N + de-centering slide 3.5≫0.4cm both WARN) but smoother than r0 (jitter **PASS**
  ang-jerk 12.7 vs r0's WARN 26.8, which slid 21.4 cm). Held both draws (minZ 0.117 / 0.120). Neither
  draw achieves rotation (peak-cos ≤ 0.17).

**Reading — the draw-gate is per-design, not uniform.** Three completed pairs now split **2 express /
1 non-express by design** (pooled max 0.611, 0.51, **−0.091**). Crucially the within-design draw spread
is itself **heterogeneous**: huge for the flippers (H06_00 range 0.655, H06_01 0.558) but near-zero for
H06_02 (0.028, both static). That is exactly the closed-program model — expression is *P(express|design)*
and that probability **varies by design**: H06_00/01 sit high (flip by n=2), H06_02 sits low (0/2, both
draws barely rotate). So a design that reads static at r0 is *usually* just draw-unlucky (H06_00/01), but
some designs are **reliably** low-expression (H06_02) — and n≤2 still cannot tell a genuinely-poor design
from a run of bad luck (m05 itself draws static ~1/3 of the time; H06_02 at 0/2 is only weak evidence).
This reproduces the 9-dim finding faithfully in the len-frozen box and reinforces that a **morphology-
conditioned policy** (which learns to *express reliably per design*) is the lever, not more replicas.

**Trainability watch (running).** H06_02_r1's A was a clean **1-attempt** leg (no collapse), so the
running count is **15 of 29 A attempts collapsed (~52%)** — flat vs last tick's 15/28, still matching the
9-dim ~47% and refuting the "freezing len calms A-training" hope. Standouts H06_04 (0.852) / H06_07
(0.547) remain un-replicated (their r1 legs are later in the pass). Worker healthy on **H06_03_r1's
Policy A**. Continuing docs/commit-only per the STATUS tree; pooled r0/r1 means + memory update at DONE.

### 6-dim interim — H06_03_r1 A landed A×3; no new B row yet; A-collapse holds ~53% (2026-07-21 ~23:15 MDT tick)
Docs/commit-only tick (worker RUNNING — never launch a GPU job). **No new completed B row** since the
last tick: H06_02_r1 is still the last row in `MORPH_PIPELINE_global6xy.txt`. The only progress is
**H06_03_r1's Policy A**, which finished at **23:05 after 3 attempts** (2 watchdog collapses, sentinel
`sweep_A_H06_03_r1_t2.trainer.log.COLLAPSED` @23:05; best `model_609.pt`, objheight 0.116, no design
abort). Its **Policy B is now training healthily** (`sweep_B_H06_03_r1.trainer.log`, ~10 min in / ETA
~24 min, no COLLAPSED sentinel, GPU 4.6 GB) → the H06_03_r1 row should land ~23:40 for the next tick.

**Trainability watch.** Adding H06_03_r1's A×3 bumps the running count to **17 of 32 A attempts collapsed
(~53%)** — flat vs last tick's 15/29 (~52%), still hugging the 9-dim global12x2 ~47% band. Five consecutive
ticks now show len-freeze does **not** calm A-training; the trainability hope that motivated the XY-only
box remains unsupported at 16/24 legs. Reorient standouts H06_04 (0.852) / H06_07 (0.547) are still
un-replicated (r1 legs later in the pass); the pooled r0/r1 verdict + memory refresh come at DONE.

### 6-dim interim — H06_03_r1 = 3rd static-r0→express-r1 FLIPPER (4th pair); A-collapse holds 53% (2026-07-22 ~01:01 MDT tick)
Docs/commit-only tick (worker RUNNING on H06_04_r1's Policy A — never launch a GPU job). The predicted
H06_03_r1 row landed **23:40**, so a 4th pair is now complete.

- **H06_03** — r0 **0.006** (static, WARN, jerk 17.3, peak-cos ~0) → **r1 0.556** (WARN, peak-cos
  **0.832**, `H06_03_r1_handoff.mp4`). **Pair {0.006, 0.556}: mean 0.281, range 0.550.** r1 is a genuine
  expresser: all three fingers recruited early (thumb@1 / index@0 / middle@1 → late_finger + idle_finger
  both **PASS**, held minZ 0.108), real rotation to peak-cos 0.83. It is *not* clean, though: **WARN** on
  jitter (ang-jerk 23.4), de-centering (slide path 8.0 cm ≫ net 0.9 cm) and over-clamp (11.0 N), with an
  index-dominant grip (index 15.6 / thumb 11.7 / middle 5.7 N). So the geometry can rotate but the drawn
  policy rolls/slides rather than pivoting cleanly — the usual WARN-tier expresser.

**Reading — the flip is now the rule, not the exception.** Four completed pairs split **3 express / 1
non-express by design**: pooled max {H06_00 0.611, H06_01 0.51, H06_02 −0.091, H06_03 0.556}, pooled mean
{0.284, 0.231, −0.105, 0.281}. Decisively, **every static-r0 design whose r1 has closed has FLIPPED to an
expresser** — H06_00/01/03 all read static at r0 (−0.044 / −0.048 / +0.006) and all cross ≥0.51 at r1.
The lone static-both design (H06_02, r1 −0.091) is the one design where P(express|design) is genuinely
low; its within-design draw spread (0.028) is near-zero while the three flippers span 0.55–0.66. This is
the draw-gated-expression model reproduced a 4th time in the len-frozen 6-XY box: the observable is
*P(express|design)*, it varies by design, and an r0 static verdict is (empirically 3/3 here) just
draw-luck rather than a design property. The lever remains a **morphology-conditioned policy** that
learns to express reliably per design, not more replicas per r0 static.

**Trainability watch (running).** No new completed A since the 23:15 tick (H06_04_r1's A is training), so
the count holds at **17 of 32 A attempts collapsed (~53%)**. Five+ ticks in, freezing the proximal lengths
still does **not** calm A-training — it hugs the 9-dim global12x2 ~47% band — so the original trainability
motivation for the XY-only box remains unsupported. Reorient standouts H06_04 (0.852) / H06_07 (0.547) are
still un-replicated; H06_04_r1's A is the current leg. Pooled r0/r1 verdict + memory refresh come at DONE.

### 6-dim interim — H06_04 STANDOUT REPLICATES: first express-both design, pair mean 0.797 (2026-07-22 ~02:30 MDT tick)
Docs/commit-only tick (worker RUNNING on H06_05_r1's Policy A — never launch a GPU job). The predicted
H06_04_r1 row landed **02:08**, closing the 5th pair — and it is the **most important row of the sweep so
far**: the program-standout design H06_04 (flagged "un-replicated, 0.852" for five consecutive ticks) has
**replicated its standout at n=2**.

- **H06_04** — r0 **0.852** (WARN, peak-cos 0.889, grip 9.8/13.5/15.6 N middle-dominant, slide 34.5 cm) →
  r1 **0.741** (WARN, peak-cos 0.792, grip 7.7/9.2/10.5 N much more balanced, slide 4.8 cm,
  `H06_04_r1_handoff.mp4`). **Pair {0.852, 0.741}: mean 0.797, range 0.111** — the tightest *high* pair
  in the entire program. Both draws are genuine 3-finger expressers (thumb@1 / index@0 / middle@1 →
  late_finger + idle_finger + drop + jitter all **PASS** in both draws, held minZ 0.103 / 0.115), and r1
  is actually the *cleaner* draw: its grip seats all three fingers more evenly (7.7/9.2/10.5 vs r0's
  lopsided 9.8/13.5/15.6) and it slides far less (4.8 vs 34.5 cm), trading only a little held-cos. Both
  still WARN on de-centering + over-clamp, i.e. WARN-tier expressers, but with real rotation.

**Reading — this is the express-BOTH design the draw-gate model predicts sits at the top of P(express).**
Until now the five completed pairs read as a spectrum of *P(express|design)*: H06_02 low (0/2, both
static, range 0.028); H06_00/01/03 high-but-magnitude-variable (**flippers**, one static + one express
draw, range 0.55–0.66); and now **H06_04 high-P AND high-magnitude** (2/2 express, both ≥0.74, range
0.111). Pooled means across the 5 pairs — {H06_00 0.284, H06_01 0.231, H06_02 −0.105, H06_03 0.281,
**H06_04 0.797**} — put H06_04 at **~3× the next-best design**, and its pooled max 0.852 trails only the
9-dim program's single-policy best G02_05_r1 (0.887). So H06_04 is the first design here that is not
merely *draw-luck away* from expression but **robustly expresses across independent policy draws** — the
durable per-design reorient signal the whole program has been hunting for. It is the design to flag for
the user's promotion / conditioned-policy decision; whether it beats m05 (a10→b33) on a fair multi-seed
comparison is the natural next probe, but **not** to be auto-launched (user's 09:00 call).

**Trainability watch (running).** H06_04_r1's A was **A×2** (1 collapse, 1 success), bumping the count to
**18 of 34 A attempts collapsed (~53%)** — flat vs the 17/32 last tick. Six ticks in, len-freeze still
does **not** calm A-training (9-dim ~47% band); the XY-only trainability motivation stays unsupported at
17/24 legs. Remaining reorient standout H06_07 (0.547) is still un-replicated (r1 later in the pass).
Worker healthy on H06_05_r1's Policy A. Pooled r0/r1 verdict + memory refresh come at DONE.

### 2026-07-22 ~07:30 MDT — global6xy interim: no new row; H06_07_r1's A is the pass's most retry-heavy (health-FAIL, not collapse) — the anti-ordering finding in one design

**Goal.** Idle pulse tick while the sweep worker runs (single GPU; analysis/commit only). No B row
has closed since the 05:30 tick — the worker has spent the whole window on **H06_07_r1's Policy A**,
which is the r1 pass's most retry-heavy A. Documenting *why*, because it is a clean per-design
illustration of the gate-invisible A-health finding rather than mere noise.

**Observation — H06_07_r1's A trains to a lift every attempt but keeps drawing health-FAIL.** The
sweep's best-of-3 A-acceptance (`train_A`, `morph_pipeline_sweep.py` L243) early-stops iff
`not-aborted ∧ ckpt ∧ lifted ∧ verdict ≠ FAIL`. Attempts t0 and t1 both:
- trained to a clean, **non-collapsing** lift (final eval objheight **0.118** / **0.111**; no
  `.COLLAPSED` sentinel for either — the collapse watchdog never fired), and
- were nonetheless **rejected**, so by the acceptance predicate their trajectory-health verdict was
  **FAIL** (the only unmet condition). Decisively *not* an objheight bar: t0's 0.118 exceeds
  H06_05_r1's **accepted** 0.112. The health gate is doing the rejecting.

So the worker is grinding the full best-of-3 (now on t2, ~16 min in / ETA ~37 min → H06_07_r1's B +
handoff land ~09:00, at the user's return).

**Why it matters — H06_07 is the sweep's most A-hostile design across BOTH replicas, yet its best
reorienter.** r0 was also **A×3** (there partly collapse-driven — `sweep_A_H06_07_r0_t0…COLLAPSED` is
the lone H06_07 collapse sentinel); r1 is heading to A×3 via repeated **health-FAILs**. Different
mechanism, same outcome: H06_07's Policy A is the hardest in the sweep to land at health-PASS. And
yet **H06_07_r0 reoriented to 0.547** — the program's 2nd-strongest 6xy design. This is exactly the
gate-invisible / mildly-anti-ordering A-health result (CLAUDE.md lesson #1; QF quick-fix note): *the
A draws that are hardest to certify healthy are among the ones that reorient best*. It is one more
concrete reason a health-based A-selector cannot pre-pick good-for-reorient draws, reinforcing the
morphology-conditioned policy as the real fix over any cheaper A-evaluator.

**Standings + trainability watch (unchanged).** Pooled per-design means hold — {H06_00 0.284, H06_01
0.231, H06_02 −0.105, H06_03 0.281, **H06_04 0.797**, H06_05 0.216, H06_06 0.055 (r1-only)} — H06_04
still ~2.8× the next-best and the only 2/2-express design. **A-collapse 20/38 (~53%)**, dead flat
across eight ticks; len-freeze continues to *not* calm A-training (9-dim ~47% band), leaving the
XY-only trainability motivation unsupported at 19/24 legs. **H06_07 (r0 0.547) is the last big-reorient
design still resolving** — its r1 closes just as the user returns. Pooled r0/r1 verdict + memory
refresh land at DONE; do **not** auto-launch the promotion / conditioned-policy program (user's
09:00 call).

### 2026-07-22 ~05:30 MDT — global6xy interim (H06_05_r1 + H06_06_r1): two clean-hold NON-EXPRESSERS break the "r1 flips to express" read

**Goal.** Idle-tick docs pass while the sweep worker runs (single GPU; analysis/commit only). Two rows
closed since the 02:30 tick — H06_05_r1's B landed 03:41, H06_06_r1's B landed 05:22 — with the worker
now on **H06_07_r1's Policy A** (ETA ~19 min). r1 pass is 7/12 (H06_00…H06_06).

**Result — both new draws hold cleanly but do not reorient.**
- **H06_05_r1** — held-cos **0.092** (WARN, peak-cos 0.169). All three fingers touch (contact_spread 1.0),
  holds minZ 0.110, jitter 12.7 — but B never found the rotation. Grip index/thumb-dominant
  (14.4/16.5/4.4 N, middle nearly idle at 4.4 N); WARN only on de-centering (slide 3.4 cm) + over-clamp
  (11.8 N). `H06_05_r1_handoff.mp4`.
- **H06_06_r1** — held-cos **0.055** (WARN, peak-cos 0.264). The *most balanced grip of the whole pass*
  (10.8/8.2/6.4 N), contact_spread 0.0 (all three land at step 1), holds minZ 0.109 — and still no
  rotation. WARN on de-centering (slide 3.0 cm) + over-clamp (8.5 N). `H06_06_r1_handoff.mp4`. A textbook
  demonstration that a *good grip is necessary but not sufficient* for reorient expression — this A→B
  draw simply landed in the non-express basin.

Both are textbook **static draws** (late_finger + idle_finger + drop + jitter all PASS; the only WARNs are
the ubiquitous de-centering + over-clamp that every draw in this sweep carries).

**Reading — the flip is draw-luck, not replica-order.** Three earlier pairs read as *static-r0 →
express-r1 flippers* (H06_00/01/03), which could have been mistaken for an r1-order effect (r1's A trains
later in the pass — maybe on a warmer machine, or benefiting from something systematic). The two new rows
**kill that reading**: H06_05 goes the *opposite* direction (r0 0.34-but-FAIL → r1 0.092 static; closed
pair mean 0.216, range 0.248 — **not** a flipper), and H06_06_r1 is flatly static (0.055). So r1 draws
are **not** systematically better than r0 — the flips were independent draws from *P(express|design)*, not
a training-order artifact. This is the same draw-gated-expression model, now with a clean counter-example
proving the gate is per-draw stochastic, not replica-indexed.

**Standout untouched.** Pooled means across completed pairs — {H06_00 0.284, H06_01 0.231, H06_02 −0.105,
H06_03 0.281, **H06_04 0.797**, H06_05 0.216, H06_06 0.055 (r1-only)} — leave **H06_04 at ~2.8× the
next-best design**, still the only 2/2-express design and the one to flag for the user's promotion /
conditioned-policy call. The last un-replicated reorient standout, **H06_07 (r0 0.547)**, is resolving
right now (its r1 A is the live job).

**Trainability watch.** H06_05_r1's A was A×1 (0 collapse), H06_06_r1's A×3 (2 collapse) → **20 of 38 A
attempts collapsed (~53%)** — dead flat across seven ticks. Len-freeze continues to **not** calm
A-training (9-dim ~47% band); the XY-only trainability motivation stays unsupported at 19/24 legs. Pooled
r0/r1 verdict + memory refresh land at DONE (do not auto-launch the promotion/conditioned-policy program —
user's 09:00 call).

### 2026-07-22 ~11:00 MDT — global6xy interim (H06_07_r1 + H06_08_r1): the 2nd-best design does NOT replicate, and a 1-finger phantom shows why held-cos alone lies

**Goal.** Docs pass while the sweep worker runs (single GPU → analysis/commit only; the worker is now on
**H06_09_r1's Policy B**, its A having just landed A×3). Two rows closed since the 07:30 tick —
H06_07_r1's B landed 08:44, H06_08_r1's B (salvaged) 09:25 — bringing the r1 pass to **9/12**
(H06_00…H06_08). Both are health-FAILs, and each is instructive in a different way.

**Result — one genuine drop, one phantom high-cos.**
- **H06_07_r1 — held-cos −0.268 (peak 0.515), a genuine DROP.** minZ 0.045 m sits *below* the 0.05 hold
  floor (drop **FAIL**), ang-jerk 149.8 (jitter **FAIL**), net drift 4.9 cm (de-centering **FAIL**),
  idle_finger WARN (2.06 mean contacts; thumb 4.7 / index 2.8 / middle 8.5 N). The object tumbled out.
  A was **A×3 health-FAIL-driven** — no r1 COLLAPSED sentinel exists (only `…H06_07_r0_t0…COLLAPSED`),
  matching the 07:30 read that r1's retries were certification failures, not collapses. `H06_07_r1_handoff.mp4`.
  **→ H06_07 does NOT replicate.** r0 was the sweep's **2nd-strongest 6xy design at 0.547**; r1 drops the
  object. Pooled mean collapses **0.547 → 0.140**. The "last big-reorient design still resolving" resolved
  as a non-replicator — the exact opposite of H06_04. **H06_04 is now the *only* design that reorients in
  both draws.**
- **H06_08_r1 — held-cos 0.78 (peak 0.994) that is a PHANTOM, not an express.** This leg was troubled
  end-to-end: Policy A collapsed on **all three** attempts (`…H06_08_r1_t0/t1/t2…COLLAPSED`), salvaging
  model_100 with `abort=True`; then Policy B **watchdog-aborted** at iter 90 (objheight 0.0126 < 0.03) and
  the sweep salvage-eval'd model_50. The scorecard exposes the "0.78": it is a **1-finger degenerate** —
  thumb 0.0 N *and* middle 0.0 N (both **idle**), only the index touches (1.0 N), mean tip-force 0.3 N —
  yet the object leans to near-vertical (cos_tail 0.78 / peak 0.994). idle_finger **FAIL**. This is a
  textbook CLAUDE.md lesson #1: **aggregate held-cos hides degeneracy.** Taken naively, H06_08's pooled
  mean (0.204, 0.78) = 0.492 would rank it 2nd in the program; the health scorecard correctly rejects it as
  a one-finger balance, not a three-finger reorient. **It does not count as an expresser.** `H06_08_r1_handoff.mp4`.

**Reading — two independent reinforcements of the program's core finding.**
1. *Capability does not transfer across the A→B draw, even for the 2nd-best design.* H06_07 (r0 0.547 →
   r1 drop) is the sharpest non-replication yet: only H06_04 expresses in both draws, while the design that
   looked like the clear runner-up fails outright on its replica. This is the seed/draw-domination result
   (CLAUDE.md #6) in its strongest form — a single per-design draw cannot resolve morphology, and the
   observable is P(express | design), not a fixed per-design cos.
2. *Held-cos alone lies; the scorecard is load-bearing.* H06_08_r1's phantom 0.78 is the concrete case the
   whole program is built to catch — a salvaged, watchdog-aborted B on a triple-collapsed A produces a
   high cos from a one-finger lean. Judging on the trajectory-health scorecard (not reward/cos sums) is
   what keeps it out of the standings.

**Standings.** Pooled per-design means — {H06_00 0.284, H06_01 0.231, H06_02 −0.105, H06_03 0.281,
**H06_04 0.797**, H06_05 0.216, H06_06 0.055 (r1-only), **H06_07 0.140** (was 0.547 r0-only; r1 drop),
H06_08 0.492\* (\*r1 = scorecard-rejected 1-finger phantom — not a genuine express)}. **H06_04 remains
~2.8× the next *valid* design and the only 2/2-express standout**, unchallenged; it is the design to flag
for the user's promotion / conditioned-policy call.

**Trainability watch.** New attempts: H06_07_r1 A×3 (0 collapse — all health-FAIL) + H06_08_r1 A×3 (3
collapse) → **23 of 44 A attempts collapsed (~52%)**, dead flat across eight ticks. Len-freeze still does
**not** calm A-training (9-dim ~47% band); the XY-only trainability motivation stays unsupported at 21/24
legs. Remaining: H06_09_r1 (B training now), then H06_10_r1 + H06_11_r1 close the sweep. Pooled r0/r1
verdict, `MORPH_PIPELINE_global6xy_POOLED.md`, and the XY-vs-9-dim comparison land at DONE — do **not**
auto-launch the promotion/conditioned-policy program (user's decision).

### 2026-07-22 ~12:35 MDT — global6xy interim (H06_09_r1): a THIRD distinct r1 FAIL mode — a genuine 3-finger reorient that got partway then jittered out

One new row since the 11:00 tick. The worker is healthy on H06_10_r1's Policy A; **r1 pass = 10/12**
(H06_00…H06_09). H06_09_r1 completes a striking triple: three consecutive r1 legs, all FAILs off the
identical `imit`-B recipe, each failing by a *different* mechanism.

- **H06_09_r1 — held-cos 0.356 (peak 0.545), verdict FAIL, but a GENUINE partial reorient, not a phantom.**
  Unlike H06_08_r1, the grip here is fully recruited: idle_finger **PASS** with all three fingers loaded
  (thumb 13.1 N / index 9.1 N / middle 8.5 N), drop **PASS** (min hold-z 0.109 m ≫ 0.05), late_finger PASS.
  The **only** failing check is **jitter** — ang-jerk 44.0 1/s² over the certification floor. de_centering
  is WARN (slide path 13.0 cm ≫ net drift 0.5 cm — the object is being worked in place, not walked off) and
  over_clamp is WARN (10.3 N). So this is a real three-finger attempt that rotated the screwdriver partway
  toward vertical (cos 0.356) and then shook rather than settling — qualitatively opposite to H06_08_r1's
  one-finger lean that scored 0.78 on a dead grip. Policy A took **A×3**: t0 and t1 both
  **watchdog-COLLAPSED** (`…H06_09_r1_t0/t1…COLLAPSED`), t2 trained clean (model_609, objheight 0.1145,
  abort False). `H06_09_r1_handoff.mp4`.

**Reading — the failure taxonomy is the finding.** Across the last three r1 legs the sweep produced a
drop (H06_07_r1, minZ 0.045), a one-finger phantom (H06_08_r1, idle thumb+middle), and now a jittery
partial reorient (H06_09_r1) — three separate ways to miss, all from the same recipe on three different
XY draws. This is exactly what draw-domination (CLAUDE.md #6) predicts: the B outcome is not a smooth
function of morphology but a *categorical* draw of how the A→B seam happened to land, and no single check
would have caught all three. The multi-check trajectory-health scorecard is what keeps each of these — a
drop, a degenerate, and an under-damped reorient — out of the standings; a held-cos or reward sum alone
would have admitted the phantom (0.78) and possibly the jittery partial (0.356).

**Standings.** Pooled per-design means unchanged at the top — {H06_00 0.284, H06_01 0.231, H06_02 −0.105,
H06_03 0.281, **H06_04 0.797**, H06_05 0.216, H06_06 0.055 (r1-only), H06_07 0.140, H06_08 0.492\*
(\*phantom), **H06_09 0.037** (r0 −0.282, r1 0.356-but-FAIL — non-expresser in both draws)}. **H06_04
stays ~2.8× the next valid design and the only 2/2-expresser**, unchallenged — the design to flag for the
user's promotion / conditioned-policy call.

**Trainability watch.** Adding H06_09_r1's A×3 (2 collapse + 1 clean): the A-hostility metric is flat.
Retry fraction (attempts needing a retry = collapse OR health-FAIL rejection) = **25 of 47 attempts (53%)**;
of those the **true watchdog-collapse sentinels = 22/47 (47%)**. (Prior ticks' "23/44" was the retry
fraction, not the sentinel count — same story either way.) Both sit dead on the 9-dim ~47% band, so
**freezing the proximal lengths does not calm A-training** — the XY-only trainability motivation is
unsupported at 22/24 legs. Remaining: H06_10_r1 (A training now) + H06_11_r1 close the sweep. Pooled r0/r1
verdict, `MORPH_PIPELINE_global6xy_POOLED.md`, and the XY-vs-9-dim comparison land at DONE — do **not**
auto-launch the promotion/conditioned-policy program (user's decision).

### 2026-07-22 ~16:45 MDT — n=4 confirm interim (H06_04_r2): H06_04's reorient property REPLICATES at n=3

The base 24-leg 6-dim sweep finished and was fully written up (the RESULT block above; pooled
`MORPH_PIPELINE_global6xy_POOLED.md`, committed f200287). Its own completion step — an **n=4 confirm**
on the two top designs H06_04 + H06_08 (`--replicas 4 --only H06_04_r2,H06_04_r3,H06_08_r2,H06_08_r3`,
replica-major, same geometry) — is now the live worker. This is the discipline the 9-dim workflow used
before any promotion: G02_00 looked like a 0.57 standout at n=2 but a static draw dragged it to 0.482
by n=4, so a 2-draw mean is not a promotable claim. **First confirm row is in.**

- **H06_04_r2 — held-cos 0.808 (peak 0.948), verdict FAIL, but a GENUINE near-complete reorient that
  fails on jitter ALONE.** The grip is fully recruited: idle_finger **PASS** with all three fingers
  loaded (thumb 11.2 N / index 9.3 N / middle 15.4 N), drop **PASS** (min hold-z 0.110 m ≫ 0.05),
  late_finger PASS. The **only** failing check is **jitter** — ang-jerk 159.3 1/s² (box-high, edging out
  H06_07_r1's 149.8). de_centering is WARN (slide path 18.9 cm ≫ net drift 0.9 cm — the screwdriver is
  worked in place, not walked off the palm) and over_clamp is WARN (12.0 N). So this is the H06_09_r1
  class again: a real three-finger drive that rotated the object nearly all the way to vertical
  (tail 0.808, peak 0.948) and then shook rather than damping to a hold. Policy A took **A×3**: t1 + t2
  **watchdog-COLLAPSED** (`…H06_04_r2_t1/t2…COLLAPSED`), t0 clean (model_609, objheight 0.108,
  abort False). `H06_04_r2_handoff.mp4`.

**Reading — this is a POSITIVE confirm signal, the first one the program has produced.** H06_04's held-cos
across three independent draws is now **0.852 (r0) / 0.741 (r1) / 0.808 (r2), running mean 0.800**, and
*every* draw drove the screwdriver near-vertical (all ≥0.74 tail, ≥0.94 peak). That is the exact opposite
of the 9-dim standout G02_00, whose apparent 0.57 collapsed to 0.48 once a static draw showed up at n=4.
Where G02_00's high mean was draw-luck, H06_04's high mean is **replicating** — the reorient-inducing
geometry is expressing on draw after draw, which is precisely what an n=4 confirm exists to test. The one
caveat the scorecard enforces: r2's *certified* verdict is FAIL (jitter), so H06_04's clean-express rate is
2/3 — it reliably reorients but not always to a settled, low-jerk hold. Reorientability is replicating;
hold-quality is still draw-variable. Neither the ≥0.5 promotion call nor any rename is made here — that
waits for H06_04_r3 + the pooled n=4 verdict, and remains the user's decision.

**Confirm progress + trainability.** Legs done 1/4: H06_04_r2 (14:54–16:39); worker now on H06_08_r2's
Policy A (started 16:43, GPU 3.5 GB). Remaining: H06_08_r2 → H06_04_r3 → H06_08_r3 (replica-major). A-side,
H06_04_r2's A×3 (2 watchdog collapses) is consistent with the base sweep's settled ~49% leg-abort rate —
freezing len does not calm A-training, unchanged in the confirm. Base-sweep handoff media that f200287 left
untracked (H06_10_r1 clean-but-static WARN 0.223; H06_11_r1 clean 3-finger EXPRESSER WARN 0.671, the row
that closed the base sweep) are committed alongside H06_04_r2 this tick. On confirm DONE (28 records): pool
H06_04/H06_08 at n=4 vs the ≥0.5 bar + m05 band, propose promotion to the user if H06_04 clears (do **not**
auto-promote), then RE-PAUSE pending the conditioned-policy decision.

### 2026-07-22 ~18:20 MDT — docs-sync tick (no new leg row): 6-dim result propagated to the publication docs

Pulse tick with the confirm worker still running (H06_08_r2's Policy A landed at 17:47 in 2 attempts,
clean; its B is training) and **no new completed leg row** since the 16:45 note — so nothing new to score.
Did the sanctioned idle work instead (STATUS decision-tree step 5, the CLAUDE.md three-doc rule): the
finished 24-leg 6-dim (`global6xy`) result was fully in this log + `MORPH_PIPELINE_global6xy_POOLED.md`
but **absent from both publication docs**. Propagated it: `webpaper/src/rl.typ` gets a new *"A follow-up:
does freezing the phalange lengths help?"* section (the 6-dim-vs-9-dim stats table + the "narrows the
landscape upward, wall unmoved, length is not a special noise source" reading; `typst compile --features
html` verified clean), and `paper/main.tex` gets an App.~bottleneck `XY-only follow-up` paragraph plus a
main-body pointer. No science added — a bookkeeping tick keeping the durable 6-dim finding in sync across
all three docs while the n=4 confirm finishes.

### 2026-07-22 ~21:10 MDT — 6-DIM SWEEP DONE (n=4 confirm complete, 28 records): H06_04 CONFIRMED, H06_08 FAILS — the program's first replicated design win

The n=4 confirm finished at 20:54 (sentinel `MORPH_PIPELINE_global6xy.DONE`, 28 records, no live worker;
STATUS decision-tree **step 4**). The remaining three confirm legs landed:

- **H06_04_r3 — held-cos 0.593 (WARN, passes the gate).** A clean partial-to-strong reorient: idle_finger
  PASS (all three loaded), drop PASS (minZ 0.116), jitter PASS (ang-jerk 18.2 — settled, not the r2 shake).
  Policy A trained clean on the **first** attempt (A×1, model_609, objheight 0.112). So H06_04's fourth
  independent draw again rotates the screwdriver substantially toward vertical, this time to a *settled*
  hold. `H06_04_r3_handoff.mp4`.
- **H06_08_r2 — held-cos −0.012 (WARN, static)** and **H06_08_r3 — held-cos 0.414 but a DROP (FAIL).**
  r3's 0.414 is not a real reorient: minZ 0.007, net drift 10.5 cm — the object left the palm and the
  cosine is measured on a falling object (A×3, the accepted A itself watchdog-aborted True, salvage-eval).
  r2 is a flat static. `H06_08_r2/r3_handoff.mp4`.

**POOLED n=4 verdict (`MORPH_PIPELINE_global6xy_POOLED.md`):**
- **H06_04 CONFIRMED — {0.852, 0.741, 0.808, 0.593}, mean 0.748, ALL FOUR draws ≥ 0.59 (three ≥ 0.74,
  peak ≥ 0.94), 4/4 express, 3/4 pass the full health gate.** This is the program's **first design to
  clear the ≥0.5 promotion bar AND replicate** — mean 0.748 is ≈2× the m05 band mean (0.38) and above
  m05's best single draw (0.82 ≈ H06_04's 0.852 peak). Where the 9-dim standout G02_00 *regressed*
  0.57→0.48 by n=4 (its high mean was draw-luck), H06_04's high mean is *replicating* — the exact
  opposite outcome, and precisely what an n=4 confirm exists to distinguish. The one caveat the
  scorecard enforces: r2's certified verdict is FAIL (jitter, ang-jerk 159), so H06_04 reliably
  **reorients** (4/4) but settles to a low-jerk hold on 3/4 — reorientability replicates, hold-quality
  is still draw-variable.
- **H06_08 FAILS the confirm — {0.204, 0.78, −0.012, 0.414}, mean 0.346, only 2/4 pass the gate.** The
  n=2 promise (0.492) rested entirely on the r1 0.78, which the confirm exposed as a phantom **1-finger**
  pinch (force 0.3 N, gate-rejected); the two new draws were a static and a drop. Exactly the "0.78 was
  never real" trap the health scorecard exists to catch. Not promotable.

**XY-only vs 9-dim landscape (apples-to-apples, both 12 designs / 28 records / same evaluator):** freezing
the proximal lengths **narrows the landscape UPWARD but leaves the per-draw variance wall untouched.**
Per-design mean floor rises (−0.39 → −0.11), leg-level median rises (0.15 → 0.22), never-lifted legs drop
(3 → 1), and designs-with-mean-≥0.5 goes **0 → 1 (H06_04)**. BUT leg-level sd is **unmoved (0.339 → 0.354)**
and A-collapse is **not lower (40% → 50%)** — so **length is not a special noise source**; freezing it buys
a better-behaved *mean surface*, not a quieter *evaluator*. H06_04 wins by having a higher **true mean**,
not by reducing draw noise — which is exactly the regime where replication *can* rank a design (and did).

**H06_04 geometry** (deltas from m05, lengths held): thumb +8.4 mm x / **−22.2 mm y** (large reposition
toward opposition), index +11.1/+12.0 mm (spread), middle −5.7/−6.7 mm (small inward). The thumb-toward-
opposition + index-spread pattern is the mechanistic story the whole morphology program set out to find
(project_policyB_v2_overnight: "reposition thumb for opposition"). **H06_06** remains the clearest
lift-hostile counter-design (thumb −27 mm x, middle −30.6 mm y; A never lifted on r0).

**Program status.** The 6-dim XY-only sweep is CLOSED. It did not break the draw-variance wall (the core
§PROGRAM CLOSE-OUT finding stands — a *broad* design search is still gate-limited, and the morphology-
conditioned policy remains the fundamental fix), but it produced the program's **first concrete design
win**: H06_04 is a real candidate to promote as a co-designed reference alongside/over m05 (a10→b33).
**Per the STATUS tree, I do NOT auto-launch promotion or the conditioned-policy build — both are the
user's 09:00 decision.** Deliverables refreshed this tick: `MORPH_PIPELINE_global6xy_POOLED.md` (n=4),
`morph_pipeline_plots.py --tag global6xy` figures + TABLE, this note, memory. Recommendation to surface
to the user: **(a)** promote H06_04 to a canonical co-designed design (rename its best A/B into the
registry, render a hero handoff) and/or **(b)** start the morphology-conditioned policy build — H06_04
gives it a concrete high-signal anchor design to condition on.

### 2026-07-23 ~02:05 MDT — pub-doc reconciliation (no new result): fixed stale mid-confirm numbers in webpaper + paper

Docs-only pulse tick (sweep still DONE, no live worker). The 18:20 publication-doc sync (above) ran
*during* the n=4 confirm and hard-coded interim figures that this log's 21:10 close-out then
superseded — so `webpaper/src/rl.typ` and `paper/main.tex` disagreed with the authoritative
`MORPH_PIPELINE_global6xy_POOLED.md`. Three fixes, all reconciled to POOLED: (1) H06_04 design-mean
**peak +0.80 → +0.75** (the +0.80 was the n=3 *running* mean; the final n=4 mean is 0.748); (2) the
head-to-head **A-abort "~49% vs ~47%, statistically identical" → 50% (global6xy) vs 40% (global12x2),
NOT lower / if anything higher** — the "~47%" had been the broader *program-wide* rate mis-borrowed as
the global12x2-specific figure. The conclusion ("length is not a special noise source") is unchanged
but its logic is corrected: it now rests on *freezing length did not lower the abort rate* rather than
on a false parity. (3) prose updated from *"confirm still running… no promotion claimed"* to the DONE
state. Also refreshed per-draw sd (0.35 vs 0.34) and the express count (10/27). `typst compile
--features html` verified clean (247 KB). NB `paper/main.tex` is gitignored (working copy) — its fixes
are on disk only; the committed changes are `webpaper/src/rl.typ` + the STATUS bullet + this note.
No GPU launch; promotion / conditioned-policy remain the user's decision.

### 2026-08-18 — INLINE ARRANGEMENT, sim2real: where it actually breaks, and what fingertip shape buys

Direction change (user): the opposed/perp arrangement is dropped — it needs a physical 90° remount the
±45° yaw servo cannot produce, and r5/r6/r7 all failed to make its pinch retain the tool. The **inline
pair + opposing thumb (m05, a10→b33)** is the arrangement the prototype natively supports and is now the
sim2real candidate. Two questions answered here: how robust is it, and what should the physical
fingertip look like.

New shared plumbing: `src/morphohand/studies/scene_mutate.py` — one place that perturbs a frozen scene
along a single physical axis (contact stiffness, friction, object mass/radius, fingertip shape),
inertia-pinned so a geometry edit never becomes a silent mass edit (`tests/test_scene_mutate.py`).

**1. Robustness battery** (`scripts/sim2real_robustness_sweep.py`, `SIM2REAL_ROBUSTNESS.txt`).
Frozen a10→b33, continuous handoff, n=32 × 300 steps per point, one axis moved at a time. Baseline
hold 1.00 / reorient 0.91 / cos|held 0.88. Ranked by how little it takes to break it:

| axis | breaks at | reorient rate |
|---|---|---|
| **pad friction ↓** | ×0.5 (μ 2.4→1.2) | 0.03, hold **0.09** — the worst cliff of all |
| **shaft diameter ↓** | ×0.85 (25→21 mm) | 0.00, hold 0.03, cos −0.65 |
| **contact stiffness ↑** | dmax 0.995→0.997 | 0.09 (0.998 → 0.28, 0.999 → 0.00) |
| **tool mass ↓** | ×0.6 | 0.28, hold 0.41 — a *lighter* tool breaks it |
| placement xy | ±5 mm | 0.28 (±2 mm → 0.81) |
| placement yaw | ±0.3 rad | 0.53 (±0.1 rad → 0.84) |
| tool mass ↑ | ×1.6, ×2.5 | 0.88, 0.78 — tolerant |
| pad friction ↑ | ×1.4 | 0.84 — tolerant |

The three worst are all the same physical statement: **anything that reduces purchase at the contact
kills it**, and it is one-sided — more friction, more mass and a fatter shaft are all fine. Friction is
the headline: the scene's μ=2.4 is an optimistic silicone-on-steel figure, and half of it is fatal. The
hardware tolerances that follow are ±2 mm placement, ±0.1 rad yaw, and a pad that must deliver μ ≳ 1.7.

**2. Fingertip shape.** Two findings before any sweep, both from looking rather than inferring:

- The shipped "capsule" tip meets the shaft on its **hemispherical end cap** — measured contact point
  5.01 mm from the cap centre, and a sphere substituted at matched reach reproduces its mechanics to
  three decimals. **The hand already makes spherical point contact.** The user's objection to spheres
  is an objection to what is shipping.
- Reach-normalisation is mandatory. In the first render six of eight candidate shapes were **buried
  inside the distal capsule** (r=7.5 mm, end cap on the tip origin) and never touched the object at
  all. Contact arrives off the tip's +x end at local (+10.4, 0, −2.4) mm, so every shape is now
  translated to present its surface at the shipped 11 mm. This is lesson #5 (IK-retarget across
  morphologies) in a new coordinate.

Zero-shot a10→b33 on each shape (`scripts/fingertip_policy_sweep.py`, n=32 × 300, `FINGERTIP_POLICY.txt`):

| tip | hold | reorient | peak cos |
|---|---|---|---|
| `cap_cross` r5 h6 (**shipped**) | 1.00 | 1.00 | 0.926 |
| `cap_cross` r6 h6 | 0.97 | 0.97 | **0.944** |
| `cap_cross` r5 h10 | 0.97 | 0.94 | 0.929 |
| `sphere` r5 | 1.00 | 0.84 | 0.914 |
| `ellipsoid` | 0.94 | 0.75 | 0.785 |
| `cap_cross` r4 h6 | 0.94 | 0.66 | 0.834 |
| `cap_cross` r8 h6 | 0.66 | 0.63 | 0.822 |
| `cylinder_line` | 0.72 | 0.38 | 0.584 |
| `cap_line` (line contact along the shaft) | 0.66 | 0.13 | 0.654 |
| `groove_cradle` / `groove_bite` | 0.53 / 0.63 | 0.13 / 0.09 | 0.37 / 0.51 |
| `pad_flat` | 0.28 | 0.06 | 0.460 |

**Compact convex point contact wins; every attempt to enlarge the contact patch loses.** Line contacts,
ridged/grooved pads and flat pads all degrade the reorient badly — the mechanism is a ROLL, and a patch
that resists rolling destroys it. Radius has an interior optimum around 5–6 mm (r4 and r8 both worse).

**Resolution caveat, measured not assumed:** the identical shipped configuration scored reorient 0.906
in the robustness sweep and 1.000 here — same policies, same scene, n=32 both times. So run-to-run
spread at n=32 is ~±0.1 and **differences below ~0.15 in these tables are not resolvable.** The top four
rows are one group; the bottom six are clearly worse.

**3. The turn/hold ratio is invariant to tip shape** (`scripts/probe_fingertip_mechanics.py`,
`FINGERTIP_MECHANICS.json`). Policy-free: hold at the best pad-loading closure, then ramp force along
the shaft (hold) and torque about it (turn), both normalised by pad load. Across all eight shapes the
axial-capacity-per-roll-resistance ratio sits at **2.04–2.32** — flat. Absolute levels move a lot
(axial/N from 0.21 for `groove_bite` to 1.45 for `pad_flat`) but the trade does not. This is the perp
programme's "rotation and retention are one variable" finding again, now in geometry rather than reward:
**fingertip shape sets how hard the contact grips, it does not let you hold better without turning
worse.** The probe also disagrees with the policy ranking (`pad_flat` has the best axial/N and the worst
policy score), so mechanics at a 5 N scripted grip does NOT predict behaviour at the 20 N operating
grip — the policy sweep is the one to trust.

**Pad compliance spec.** At the probed grip the shipped tip penetrates 0.82 mm at 4.84 N of pad load
(0.169 mm/N). That interpenetration is MuJoCo's stand-in for pad deflection, and the contact-stiffness
row above says the reorient dies as soon as it is reduced. So the compliance is a **hardware
requirement, not a solver setting**: the physical pad must deform on the order of 1–3 mm at operating
load. A rigid printed tip will not reproduce b33 regardless of its shape.

**Recommendation.** Keep the shipped tip geometry (optionally r 5→6 mm, the only variant that is
arguably better and never worse); do NOT go to a flat or grooved compliant pad, which is the intuitive
choice and the measurably wrong one; specify the pad for μ ≳ 1.7 and ~1–3 mm deflection at load. Open
and NOT started: hardening b33 by domain-randomising the three cliff axes (friction, shaft diameter,
contact stiffness) — multi-day GPU, and the earlier compliance-DR attempt reached a holding basin with
reorient 0.00, so it needs the friction/diameter axes added rather than a rerun. User's call.

### 2026-08-18 late — DR-hardening the inline reorienter: NEGATIVE, and it says fix the hardware instead

Ran `scripts/harden_b33_queue.py` overnight (user-authorized): three 40M-timestep finetunes of b33
with actor **and** critic warmstarted, under contact DR via the live-A reset, then all four policies
(baseline included) re-measured across both cliff axes. `docs/experiments/HARDEN_B33.txt`. No arm
collapsed; all three reached model_541 with healthy object height (0.095–0.125).

| policy | @trained | @0.997 | @0.998 | @0.999 | @μ×0.5 | @μ×0.7 |
|---|---|---|---|---|---|---|
| b33 baseline | **0.91** | 0.09 | 0.19 | 0.03 | 0.00 | 0.94 |
| + compliance DR | **0.00** | 0.16 | 0.22 | 0.00 | 0.03 | 0.03 |
| + friction DR | **0.94** | 0.13 | 0.06 | 0.00 | 0.03 | 0.94 |
| + both | **0.03** | 0.28 | 0.41 | 0.00 | 0.03 | 0.00 |

(reorient rate, n=32 each.) **Nothing was hardened.** Two arms destroyed the nominal policy and the
third gained nothing.

**Compliance DR does not broaden the policy, it RELOCATES it.** Both compliance arms lose the soft
end they used to work at (0.91 → 0.00 / 0.03) and peak in the *middle* of the DR band (`both` is best
at dmax 0.998: hold 0.906, reorient 0.41). The policy did not learn to span stiffnesses; it re-tuned
to the band's mean and gave up the end it had. This reproduces the earlier from-scratch compliance-DR
failure (`compliance_dr_pipeline.py` → reorient 0.00) under a completely different protocol, so the
**protocol was never the problem** — that hypothesis is now dead. Compliance DR is not a fix for this
skill.

**The friction arm is the internal control that makes this attributable.** It ran the identical 40M
finetune and *retained* nominal performance (0.938 vs baseline 0.906), which rules out "40M steps of
finetuning drifts the policy" as the explanation for the compliance arms' collapse. The cause is the
compliance DR specifically.

**The μ×0.5 cliff was not cracked, and is NOT proven impossible.** The friction arm trained roughly
half its envs in that regime and still scores 0.03 there. But a static-capacity probe at μ×0.5 finds
~5 N of axial capacity against a 0.24 N tool — holding is not the binding constraint, so this is a
dynamic/strategy limit this DR did not resolve, not a wall. Stated as unresolved rather than settled.

**What this converges with.** The fingertip study found the hold-per-turn ratio invariant across all
eight tip shapes, and this finds the reorient un-broadenable across contact stiffness. Both say the
same thing: **the rolling reorient is a narrow-band contact behaviour**, and neither geometry nor DR
widens the band.

**Recommendation — invert the sim2real strategy.** Stop trying to make the policy contact-agnostic;
make the hardware match the contact regime the policy already works in. That regime is now specified:
pad deflecting ~1–3 mm at operating load, μ ≳ 1.7, a compact convex (not flat, not grooved) tip at
r 5–6 mm, tool placement ±2 mm / ±0.1 rad. This is a narrower hardware spec than "build any pad and
train around it", but it is a *reachable* one, and three independent attempts now say the training-side
route is not.

### 2026-08-21 — the b33 seed band: b33 is an ORDINARY draw, and the warmstarted recipe is not seed-dominated

First study run on NSF ACCESS / DeltaAI (`docs/notes/20260820-deltaai_bulk_training_runbook.md`).
16 independent draws of b33's own lineage — a10 warmstart, live-A reset, 20M ts, m05, b33's own
timing (lift/reorient start 58, tip-loss grace 10) — each scored through the continuous handoff at
3 rollouts. 15 SU. `docs/experiments/B33_SEED_BAND.{txt,json}`.

| | held-cos |
|---|---|
| 15 "core" draws | 0.827 – 0.935, mean **0.873**, sd **0.032** |
| 1 tail draw (s10) | **0.452**, sd 0.439, held 2/3 |
| **b33 itself** | **0.891** — 11 of 16 draws at or below it |
| within-draw (rollout) sd | **0.080** |

**b33 is a typical draw, slightly above median.** Everything downstream of it — the three
robustness cliffs, the fingertip ranking, the hardware spec (μ ≳ 1.7, 1–3 mm pad deflection,
r 5–6 mm convex tip) — was measured on a representative policy, not a lucky one. That was the open
question and it resolves in the reassuring direction.

**The bigger result: between-draw sd is 0.032, SMALLER than the 0.080 rollout noise.** For this
recipe the draws are barely distinguishable from each other. The program-wide prior of 0.3–0.5 came
from *from-scratch* reorient training, which is a hard-exploration problem; **warmstarting from the
design's own A policy removes the exploration lottery.** Nobody had measured it because nobody had
ever taken a second draw.

That re-prices the whole design-search program. At sd 0.032 the seeds needed to resolve a
difference Δ in mean held-cos are `n ≈ 15.7σ²/Δ²`:

| Δ | seeds at the MEASURED sd | seeds at the assumed 0.35 |
|---|---|---|
| 0.20 | <1 | 48 |
| 0.10 | ~2 | 192 |
| 0.05 | ~6 | 769 |

The "n=48 per design" costing was built on a variance that does not apply to warmstarted runs.
**Caveat, and it is a real one:** this is one design (m05) in the warmstarted regime. A per-design
search cannot reuse a10 — warmstarting one hand's A onto another's geometry is gotcha #5, and it is
exactly what NaN'd the first attempt at this study (below). Each design needs its own from-scratch
A, and whether the low variance survives that is untested.

**The tail is real and is a DROP, not a bad reorient.** s10 was rendered rather than inferred
(`docs/experiments/20260821-b33_seed_band_filmstrips/`): it lifts cleanly, loses the shaft by
step ~102, and the rollout ends with the tool rolling on the floor and the hand empty. Its 0.452 is
a mixture of held rollouts and that drop. So ~1 draw in 16 is drop-prone, and a single-draw verdict
still carries a ~6% chance of calling a design bad when it is not — small, but the argument for
n≈3 rather than n=1, not for n=48.

**The failed first attempt, worth recording.** The initial 16-draw queue returned 1 completion and
15 NaN divergences. Cause: `--init-actor-checkpoint` pointed at **b10**, taken from
`train_handoff_liveA_reset.sh`'s DEFAULT `B_CKPT`, when b33 in fact warmstarts **a10** (REGISTRY:
b33 = "live-A reset, warmstart a10"; b24 is the b10 one). b10 was trained on the baseline hand, so
every task loaded an actor from one morphology onto m05's IK-retargeted geometry — gotcha #5 — and
mjwarp's contact solve diverged. Substitution on the three fastest failures: b10 → NaN at iters 21,
36, 22; a10 → clean 40/40/40. Two hypotheses died first and are recorded so they are not re-run:
the aarch64 platform (disproven — the same seeds NaN locally) and `init_noise_std` (disproven —
`docs/experiments/B33_NOISE_STABILITY.txt`, NaN at 0.05/0.1/0.15 alike; the perp cliff does not
transfer here).

**Load-bearing gotcha: `assert_config_parity.py` CANNOT see the warmstart checkpoint.** It is not
written into the dumped `config.yaml`. An in-flight parity check passed cleanly on runs loading the
wrong policy. Until the trainer records it, warmstarts must be checked by hand against
`results/rl/REGISTRY.md` — and the launcher DEFAULT is not evidence of what a published run used.

---

## Phase: object-frame reorientation primitive (2026-08-27)

Asked whether our best reorienters can be expressed hand-agnostically (object-centric /
fingertip-object-centric, in the spirit of UHAS and grasp-wrench representations) so a design
could be scored by re-targeting a known skill instead of re-learning one. Built
`scripts/reorient_primitive.py` and extracted the primitive from b33-on-m05 (through its a10
live-A seam) and r4-on-perp, 32 envs x 400 steps each.

Full writeup `docs/experiments/REORIENT_PRIMITIVE.txt`. Headlines:

- The representation (tilt-frame azimuth + axial + load, indexed by the shaft's alignment PHASE
  rather than by clock) reconstructs the recorded fingertip positions to **3.0 / 2.0 mm**. It is a
  faithful description.
- **The wrist contributes 0.0% of the turn on both policies**, despite both carrying
  `palm_rotation_residual_scale 0.3`. A fingertip/object representation is the right one.
- **The scripted grasp + lift reorients by cos 0.00** — necessary control, everything is policy.
- **69% of m05's alignment gain and 46% of perp's happens while the shaft still touches the
  floor.** On m05 that phase is over in 4 control steps and belongs to **Policy A**, not b33;
  b33 contributes the final 31%. Reorient credit in this program has been mis-assigned.
- **Opposite gravitational regimes.** perp grips ~45 mm from the COM and the shaft HANGS (hang
  angle 70->21 deg, monotone); m05 grips ~20 mm out with the COM ABOVE the contact and balances it
  inverted (121->162 deg). The signed grip offset from the COM selects the strategy and explains
  perp's axial slip and thumb-idle drops versus m05's 3-finger 21 N hold.
- Executed as a controller (IK on the schedule, residual on the grasp anchor), the primitive
  recovers **~60% of m05's turn and ~30% of perp's** while holding the object. Faithful
  description, incomplete generator — the missing part is exactly the floor and gravity share.
- **Kinematic feasibility against the schedule does NOT discriminate designs** (all of m05,
  H06_04, L01_13, G02_00, sp25, H06_08, H06_06 reach it to 1-2 mm on a 12.5 mm shaft, with m05
  mid-pack). Converges with the same day's link-length geometry gate: geometry stops ranking
  hands once they are modelled properly.

Traps recorded there, all of which produced convincing wrong answers first: replacing the grasp
anchor with the IK solution zeroes the position servo's squeeze and drops the shaft every time;
peak_cos without final_z and contact count reports a dropped shaft standing on the floor as a
perfect reorient; and a phase grid wider than the policy's own coverage puts NaN at the table ends
where np.interp poisons every query.

"""Config dataclass for the morphohand RL env.

`MorphoHandEnvCfg` is a plain dataclass so an env config can be inspected /
serialised without importing mjlab or torch. The translation into a live mjlab
`ManagerBasedRlEnvCfg` — spec factories, keyframe readers, and the per-manager
builders — lives in `env_build.py` (split per CODEBASE_AUDIT.md step 3);
`to_mjlab_cfg` and the factories are lazily re-exported here for back-compat.

Building blocks (`reward.py`, `observations.py`, `reference_trajectory.py`,
`scene_loader.py`) stay framework-agnostic and are unit-tested in isolation —
see `tests/test_rl_*.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Joint names on the hand (no `a_` prefix — those are actuator names).
FINGER_JOINT_NAMES: tuple[str, ...] = (
    "thumb_yaw", "thumb_mcp", "thumb_pip",
    "index_yaw", "index_mcp", "index_pip",
    "middle_yaw", "middle_mcp", "middle_pip",
)
PALM_JOINT_NAMES: tuple[str, ...] = (
    "palm_px", "palm_py", "palm_pz", "palm_rx", "palm_ry", "palm_rz",
)


# ----------------------------------------------------------------------
# Cfg
# ----------------------------------------------------------------------

@dataclass
class MorphoHandEnvCfg:
    """Top-level config bag for the morphohand RL env.

    Translates to `ManagerBasedRlEnvCfg` via `to_mjlab_cfg(cfg)`. Kept as
    a plain dataclass so the env can be inspected/serialised without
    importing mjlab/torch.

    `finger_default_ctrl` is the position-controller setpoint used as the
    finger action default_offset (action = residual on top of it). Defaults
    to the CEM `best_finger_ctrl` from the run18_final cube foundational
    run; overrides should pass the per-morphology CEM result.
    """
    frozen_scene_xml: Path
    keyframe_name: str = "open_short_manual"
    foundational_run_dir: Path | None = None
    finger_default_ctrl: tuple[float, ...] = (
        # Defaults from run18_final cube foundational best_finger_ctrl
        # (results/phase1/run18_final/foundational/cube/run_20260521_161817/summary.json).
        0.0835, 1.8962, -0.9321,
        -0.0861, 1.3707, 1.1002,
        0.0344, 1.3749, 1.1120,
    )
    num_envs: int = 1024
    env_spacing: float = 0.5
    sim_timestep: float = 0.002
    decimation: int = 10               # 50 Hz policy on 500 Hz sim
    episode_length_s: float = 1.4
    object_body_name: str = "cube"
    """Name of the object body in the frozen scene to extract and spawn.
    `cube` for the cube morphology run; `prism`, `screwdriver_medium`, etc.
    for those scenes."""
    object_size: float = 0.02
    object_mass: float = 0.016
    object_friction: tuple[float, float, float] = (2.4, 0.2, 0.02)
    lift_target_z_above_init: float = 0.05
    settle_steps: int = 240
    lift_ramp_steps: int = 80
    lift_delta_z: float = 0.05
    finger_residual_scale: float = 0.2
    """Scale applied to the policy's finger residual on top of the
    LerpFinger scripted setpoint. Larger = policy can deviate more from
    the open-loop schedule (useful under DR); smaller = policy is
    constrained to small corrections (useful when the schedule is the
    optimum)."""
    finger_close_sim_steps: int = 80
    """Sim steps over which the linear-interp finger setpoint sweeps from
    `open_finger_qpos` to `finger_default_ctrl`. Smaller = faster close;
    the rest of `settle_steps` becomes a hold-at-grip phase, giving the
    cube time to seat in the fingers before the palm ramps up. Must be
    <= `settle_steps`."""
    open_finger_qpos: tuple[float, ...] = (
        # Hand-open pose. The thumb's mcp axis is flipped relative to the
        # index/middle fingers — for the thumb, joint_pos = 3.14 means
        # fully extended away from the palm, while for index/middle the
        # mcp opens toward joint_pos = 0. Yaw and pip stay neutral.
        # Order matches FINGER_JOINT_NAMES.
        0.0, 3.14, 0.0,    # thumb_yaw, thumb_mcp, thumb_pip
        0.0, 0.0,  0.0,    # index_yaw, index_mcp, index_pip
        0.0, 0.0,  0.0,    # middle_yaw, middle_mcp, middle_pip
    )
    open_finger_from_keyframe: bool = False
    """If True, start the fingers (both the reset pose AND the LerpFinger
    interpolation START) from the KEYFRAME's finger angles instead of the
    hardcoded `open_finger_qpos`. REQUIRED for IK-retargeted morphologies: CEM
    resets to the keyframe (e.g. `open_ik`) and optimizes the grip from THERE, so
    the RL env must close from the same pose — otherwise the LerpFinger starts
    from the baseline open pose (thumb flung to mcp=3.14) and a finger arrives
    late, giving a 2-then-3-finger grasp that never stabilises. Default False
    keeps the baseline lineage (a01/B-registry) byte-identical."""
    # ---- domain randomization (cube spawn) -----------------------------
    # Separate x and y because the reachable region is asymmetric for
    # this morphology (see /tmp/sweep_reachable2.py): full x reach
    # ~±20mm but y limited to ~±5mm symmetric. Use `cube_spawn_xy_jitter`
    # for back-compat (sets both x and y to the same value).
    cube_spawn_xy_jitter: float = 0.0
    """Back-compat: when nonzero AND x/y jitter unset, sets BOTH x and y
    jitter to this value (symmetric)."""
    cube_spawn_x_jitter: float | None = None
    """Symmetric uniform x noise (m, ±). None = inherit from `cube_spawn_xy_jitter`."""
    cube_spawn_y_jitter: float | None = None
    """Symmetric uniform y noise (m, ±). None = inherit from `cube_spawn_xy_jitter`."""
    cube_spawn_x_center: float = 0.0
    """Mean offset (m) added to the cube spawn x. Use to recenter DR onto
    an offset reachable region (e.g., prism's good zone is at x=+0.006)."""
    cube_spawn_y_center: float = 0.0
    """Mean offset (m) added to the cube spawn y."""
    cube_spawn_yaw_jitter: float = 0.0
    """Symmetric uniform yaw noise (rad, ±) on cube spawn each reset."""
    # ---- contact-compliance DR (sim2real robustness) --------------------
    compliance_dr: bool = False
    """Per-env contact solimp DR: each reset samples one softness per env and
    lerps every geom's solimp (dmin, dmax) jointly between the bounds below,
    soft -> hard. Single-stiffness training overfits the contact model
    (fragile, non-monotonic robustness curve — docs/rl/compliance_dr_plan.md)."""
    compliance_dr_dmin: tuple[float, float] = (0.97, 0.985)
    """(soft, hard) bounds for solimp dmin."""
    compliance_dr_dmax: tuple[float, float] = (0.995, 0.999)
    """(soft, hard) bounds for solimp dmax — the range over which the
    compliance-robustness sweep found the task at least sometimes feasible."""
    # ---- curriculum on DR jitter ---------------------------------------
    dr_anneal_iters: int = 0
    """Number of PPO iters over which jitter linearly ramps from 0 to
    the configured `cube_spawn_*_jitter` values. 0 disables the
    curriculum (jitter is full from iter 0)."""
    tracking_anneal_iters: int = 0
    """Number of PPO iters over which tracking-from-CEM reward weights
    (track_finger_qpos / track_object_pos / track_object_quat /
    track_finger_ctrl_anchor) linearly scale from their initial values
    toward `tracking_final_scale` x the initial. 0 disables.

    Rationale: the CEM reference trajectory is recorded at a fixed cube
    position. Under DR the cube spawns elsewhere, so tracking the CEM
    object_pos is the WRONG signal late in training. Anneal the
    tracking weights down once the policy has the basin."""
    tracking_final_scale: float = 0.0
    """Multiplier applied to tracking reward weights at end of anneal.
    0.0 = tracking rewards fully off post-anneal; 0.25 = quarter strength."""
    cube_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    """Multiplicative range applied to cube friction (deferred)."""
    cube_mass_scale_range: tuple[float, float] = (1.0, 1.0)
    """Multiplicative range applied to cube mass (deferred)."""
    # ---- stability reward weights --------------------------------------
    object_xy_drift_weight: float = -3.0
    """Penalty per metre of cube xy drift from its spawn. Increase to
    enforce 'lift only, no translation' grasps."""
    object_orientation_drift_weight: float = -3.0
    """Penalty per radian of cube quat geodesic distance from spawn quat.
    Increase to enforce 'lift only, no rotation' grasps."""
    finger_drift_weight: float = -2.0
    """Penalty per L2 finger qpos distance from the grip ctrl. Encourages
    stable contact configuration."""
    contact_gate_stability_rewards: bool = False
    """If True, multiply the xy_drift / orientation_drift / finger_drift
    penalties by a contact gate (fires only when >= `contact_gate_min`
    fraction of tips are touching the cube). Concentrates the credit on
    'once you have the cube, hold it still' instead of penalising
    legitimate transients during approach + initial contact."""
    contact_gate_min: float = 0.5
    """Threshold on `contact_mean` (fraction of tips touching) above which
    the contact gate opens. 0.5 = at least 2 of 3 tips. 1.0 = all tips."""
    # ---- finger close timing -------------------------------------------
    finger_close_easing: str = "linear"
    """Easing for LerpFinger setpoint interpolation. 'linear' = uniform
    open->grip sweep. 'ease_out_quad' / 'ease_out_cubic' = fast early
    (clear-air approach) + slow late (careful contact formation)."""
    # ---- lift-phase early terminations ---------------------------------
    enable_lift_terminations: bool = False
    """Enable early terminations during the lift hold phase. Off by
    default for back-compat. When on, episodes that exhibit object slip,
    drop, sustained tip-loss, or finger-slip during the post-lift hold
    are terminated (GAE bootstrap cut), giving PPO a sharp negative
    signal for unstable grasps."""
    lift_phase_start_step: int = 40
    """Policy step from which lift-phase terminations engage. Must be
    after the scripted lift ramp completes (settle_steps + lift_ramp_steps
    in sim steps / decimation, + a few hold steps as grace).
    Default 40 = 32 (ramp done) + 8 (grace) at decimation=10."""
    term_object_slip_xy: float = 0.015
    """Terminate if cube xy drift > this (m) in lift phase."""
    term_object_slip_yaw: float = 0.5
    """Terminate if cube orientation geodesic drift > this (rad) in lift phase."""
    term_object_drop: float = 0.02
    """Terminate if cube z falls below spawn_z - this (m) in lift phase."""
    term_tip_lost_steps: int = 3
    """Terminate if any tip is off for >= this many consecutive policy
    steps in lift phase. Single-step tip-loss is allowed (physics jitter)."""
    term_finger_slip: float = 0.3
    """Terminate if finger qpos drift from grip > this (rad) in lift phase."""
    """Fraction of finger_default_ctrl (grip pose) used as the initial finger
    joint qpos. 0.0 = fingers fully extended at qpos=0 (cube can rest on
    floor without penetration); 1.0 = at grip ctrl (collapsed cage that
    CEM used). The position controller pulls fingers from init toward
    finger_default_ctrl over the settle phase, so 0.0 produces a real
    closing-grasp motion instead of a pre-cage."""
    reward_mode: str = "full"
    obs_mode: str = "full"
    viewer_distance: float = 0.6
    """Viewer camera distance; larger values zoom out."""
    # ---- in-hand reorient task -----------------------------------------
    enable_palm_rotation_residual: bool = False
    """If True, expand action space by 3 dims (palm rx/ry/rz residuals)
    on top of the scripted palm. Required for in-hand reorientation —
    the fingers alone can't apply enough torque on a smooth cylinder."""
    palm_rotation_residual_scale: float = 0.3
    """Scale on policy palm rotation residual (rad). 0.3 ~= 17deg."""
    palm_rotation_active_from_sim_step: int | None = None
    """If set, palm rotation residuals are zeroed before this sim step.
    Default: settle_steps (rotation begins after grasp/lift completes)."""
    enable_target_axis_reward: bool = False
    """If True, add a `target_axis_alignment` reward term for in-hand
    reorientation. Reward fires only after `reorient_start_step`."""
    target_axis_object_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Object body-local axis that should align with the world target.
    Default = body-local +Z (cylinder long axis in MuJoCo convention)."""
    target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """World-frame target axis. Default = vertical (cylinder stands up)."""
    target_axis_weight: float = 0.0
    """Reward weight for `target_axis_alignment`. Should be substantial
    once enabled (e.g. 50.0) since the lift reward is up to 80."""
    target_axis_alpha: float = 4.0
    """Sharpness of the alignment reward. exp(-alpha * (1 - cos)^2)."""
    reorient_start_step: int = 30
    """Policy step from which the target_axis reward fires. Should be
    after the scripted lift completes."""
    finger_residual_active_from_step: int = 0
    """Policy step from which the policy's finger residual is applied (zeroed
    before — the scripted LerpFinger grasp runs undisturbed). 0 = always on.
    Set to ~reorient_start_step in the normal-lift env so a reorient-trained
    warmstart doesn't destabilise the flat-object grasp (NaN)."""
    strict_tip_lost_termination: bool = False
    """If True, terminate immediately on any single-step tip loss during
    the lift phase (no consecutive-step grace). Required for in-hand
    reorientation where dropping the object is unrecoverable."""
    contact_min_weight: float = 30.0
    """Weight on fingertip_contact_min reward. Default 30 strongly
    incentivizes 3-finger grip; drop to 10-15 for reorient tasks where
    occasional regrip needs to be allowed."""
    target_axis_progress_weight: float = 0.0
    """Reward weight on positive Δ(alignment) per step. Provides dense
    gradient toward 'rotating in the right direction' even when state
    alignment reward is small. Typical: 200-500 (rewards small per-step
    changes ~0.01 to amount of ~2-5)."""
    target_axis_alpha_curriculum_iters: int = 0
    """If >0, anneal target_axis_alpha from `target_axis_alpha_start` to
    `target_axis_alpha` linearly over this many iters. Soft early shaping
    gives gradient at large tilts; sharp late shaping focuses on near-target."""
    target_axis_alpha_start: float = 0.5
    """Initial alpha for curriculum (soft / wide reward basin)."""

    # ---- object-relative fingertip IMITATION (morphology-transferable reorient prior) ----
    imitation_ref_npz: str = ""
    """Path to a recorded object-relative fingertip trajectory (T,3,3) from a good reorientation
    (scripts/rl_demo_handoff_continuous.py --record-fingertip-traj). Empty = disabled. Adds a
    `track_fingertip_obj` reward that pulls this design's fingertips (in the screwdriver frame)
    toward the reference motion — a DESIGN-NEUTRAL, morphology-transferable reorient prior (unlike
    joint-space imitation). See src/morphohand/rl/imitation.py."""
    imitation_weight: float = 0.0
    imitation_alpha: float = 300.0
    imitation_curriculum_iters: int = 0
    """If >0, anneal imitation_weight -> imitation_weight_final over this many iters (learn the
    motion from the demo first, then let the task reward refine it)."""
    imitation_weight_final: float = 0.0
    terminate_low_tilt_velocity: bool = False
    """If True, terminate episodes where the cylinder isn't actively
    reorienting during the reorient phase. Kills 'just hold the lift'
    local optima."""
    tilt_velocity_window: int = 20
    """Number of policy steps over which to measure tilt progress."""
    tilt_velocity_min_progress: float = 0.05
    """Minimum Δ(alignment) over the window. Below this triggers termination
    during reorient phase. 0.05 ~= ~3° change over the window."""
    enable_floor_proximity_termination: bool = False
    """If True, terminate during the reorient phase when the object center
    z falls below `object_min_z`. Forbids floor-bracing strategies that
    the v4 reorient policy discovered — the cylinder must stay clear of
    the ground for the rotation to count as in-hand."""
    object_min_z: float = 0.05
    """Minimum world-frame z (m) for the object center during the
    reorient phase. For an 8 cm cylinder, 0.05 m gives ~1 cm clearance
    to the floor in worst-case (vertical) orientation. Pair with
    `--lift-target-z-above-init 0.10` so the lift target is above this
    threshold by a comfortable margin."""
    floor_proximity_phase_start_step: int | None = None
    """Policy step from which the floor-proximity termination engages.
    If None, defaults to `reorient_start_step` so the check kicks in
    only after the lift completes."""
    skip_lift_phase: bool = False
    """If True, skip the settle/lift phases entirely: cylinder spawns
    already at lifted position (z = keyframe_z + lift_delta_z), fingers
    spawn at finger_default_ctrl (CEM grip pose), palm spawns at lifted
    height. Used for training a 'Policy B' that focuses exclusively on
    reorientation, assuming a prior 'Policy A' has already executed the
    lift. When True, settle_steps and lift_ramp_steps are forced to 0/1
    inside ScriptedPalm so the lift is a no-op."""
    action_rate_weight: float = -0.005
    """L2 penalty weight on action rate (Δa per step). Bump to -0.05 or
    lower for in-hand reorient runs that produce jittery finger motion;
    higher penalty enforces smooth control that's more sim-to-real safe."""
    object_ang_acc_weight: float = 0.0
    """L2 penalty weight on object angular-velocity CHANGE (Δω per step,
    proxy for angular jerk). Penalizes oscillatory rotation that's a
    sim-only exploit. Typical reorient value: -0.5 to -2.0 (cylinder
    moment of inertia is ~1e-5 kg·m², so Δω in rad/s is small per step
    and the penalty scales accordingly)."""
    object_ang_acc_phase_start_step: int = 0
    """Policy step from which the angular-acceleration penalty engages."""
    skip_lift_drop_offset: float = 0.005
    """In skip-lift mode, spawn the cylinder this many meters ABOVE the
    palm's lifted z so it falls into the pre-closed grip and establishes
    contact force. With fingers spawned exactly at finger_default_ctrl
    and cylinder right between them, position-controller equilibrium is
    zero-force — fingers can't grip. A small drop produces the contact
    pressure CEM had at lift-end. 5 mm is enough to settle in ~10 sim
    steps without bouncing."""
    skip_lift_spawn_tilt_jitter: float = 0.0
    """Skip-lift handoff-robustness DR: uniform roll/pitch jitter (rad, ±) on
    the lifted spawn each reset, so a reorient policy tolerates the varied
    tilted pose a real Policy-A lift hands off. Try 0.1-0.25."""
    skip_lift_spawn_z_jitter: float = 0.0
    """Skip-lift handoff-robustness DR: uniform z jitter (m, ±) on the lifted
    spawn height each reset (Policy A may lift to a different height than the
    nominal skip-lift spawn). Try 0.02-0.04."""
    handoff_dr_curriculum_iters: int = 0
    """If >0, ramp the skip-lift spawn tilt/z jitter from 0 to the configured
    max over this many PPO iters (gradual, so the warmstart grip adapts instead
    of being shocked — high fixed DR collapsed the grasp). 0 = fixed jitter."""
    handoff_state_bank: str | None = None
    """Path to a Policy-A terminal-state bank (npz from rl_record_handoff_states.py).
    When set (skip-lift), object+hand spawn from sampled bank states each reset
    (train-the-handoff); the LiftingCommand's object pose write is disabled."""
    handoff_onset_bank: str | None = None
    """Path to Policy A's delivery state bank (same npz format as handoff_state_bank).
    When set in the NORMAL-lift env (skip_lift_phase=False), a step-mode event
    (mjlab_terms.inject_handoff_bank_at_onset) overwrites the object pose+vel and robot
    qpos from a sampled bank state ONCE per episode at `handoff_onset_step` — so B trains
    on A's REAL delivered grip AND under the normal-lift observation schedule (the one
    combination that matches the continuous deploy). Distinct from handoff_state_bank,
    which only does a reset-time spawn in skip-lift; the normal-lift reset spawn (flat on
    floor) is left intact so the pre-onset lift obs schedule stays in-distribution."""
    handoff_onset_step: int | None = None
    """Episode step at which to inject the onset bank state. None -> lift_phase_start_step
    (and should match the deploy handoff step, e.g. 40 for the s40 bank)."""
    handoff_inject_velocity: bool = True
    """Onset-inject ablation: write A's REAL obj_vel + finger/palm qvel from the bank. False
    -> zero finger vel (the pre-2026-06-09 STATIC behavior). Markov-completeness lever."""
    handoff_inject_last_action: bool = True
    """Onset-inject ablation: override the seam `last_action` obs with A's delivered action
    (the only history-dependent obs). False -> leave B's gated value (the OOD static behavior)."""
    # ---- Branch B (un-freeze Policy A): terminal-state reg toward B10's set --
    handoff_target_bank: str | None = None
    """Path to Policy B10's INITIATION-set bank (npz from rl_record_initiation_bank.py).
    When set with a non-zero weight, Policy A's finetune gets a seam-gated dense
    reward for delivering B10's holding GRIP (finger qpos) — the only measured gap
    between A's delivery and B10's set (see mjlab_terms.handoff_target_proximity)."""
    handoff_target_weight: float = 0.0
    """Weight on the handoff_target_proximity reward (try ~4; comparable to a
    tracking term). 0 disables. Keep modest so A's grasp/lift reward stays in
    charge and the grip is protected."""
    handoff_target_seam_lo: int = 33
    handoff_target_seam_hi: int = 37
    """Policy-step window the grip-proximity reward is active over (the delivery /
    handoff window — default brackets the step-35 residual-onset handoff)."""
    handoff_target_qpos_tol: float = 0.05
    """Per-joint tolerance (rad) on the finger-qpos match."""
    handoff_target_scale_mult: float = 1.0
    """Multiplier on the per-joint tolerance (>1 = looser match)."""
    # ---- "smooth & quick" finetune curriculum (Policy B v2) -------------
    target_axis_progress_clamp_negative: bool = False
    """If True, only positive Δ(alignment) is rewarded (no penalty for
    slipping back down). If False (default), signed delta — rotating back
    toward flat is penalized symmetrically, which discourages the slip-back
    seen in Policy B v1."""
    action_rate_weight_final: float | None = None
    """Final action_rate_l2 weight for the smoothness curriculum. None =
    no ramp (weight stays at action_rate_weight)."""
    object_ang_acc_weight_final: float | None = None
    """Final object_ang_acc_l2 weight for the smoothness curriculum. None =
    no ramp."""
    smoothness_curriculum_start_iter: int = 0
    """PPO iter at which the smoothness-weight ramp begins (weights held at
    their base values before this). Use a small consolidation window when
    warmstarting so the base rotation settles before penalties dial up."""
    smoothness_curriculum_iters: int = 0
    """Number of PPO iters over which smoothness weights ramp base→final.
    0 disables the curriculum."""
    enable_alignment_success_termination: bool = False
    """Terminate (success) once alignment cos >= success_align_thresh is
    held for success_hold_steps consecutive steps. Earlier success → higher
    discounted return (rewards quickness) and locks in the result."""
    success_align_thresh: float = 0.9
    """Alignment cos threshold counted as 'aligned' for the success
    termination AND alignment_success_bonus reward."""
    success_hold_steps: int = 10
    """Consecutive aligned policy steps required to declare success."""
    success_bonus_weight: float = 0.0
    """Weight on the one-shot alignment_success_bonus reward (fires the step
    success is reached). 0 disables the bonus term."""
    time_cost_weight: float = 0.0
    """Weight on the per-step reorient_time_cost (constant during reorient
    phase). Small negative pressures shorter trajectories. 0 disables."""
    speed_bonus_weight: float = 0.0
    """Weight on the one-shot alignment_speed_bonus (∝ episode time remaining
    when alignment first crosses speed_bonus_align_thresh). 0 disables."""
    speed_bonus_align_thresh: float = 0.9
    """Alignment cos threshold whose first crossing triggers the speed bonus."""
    # ---- de-centering penalty (Policy B v2.1) ---------------------------
    lateral_drift_weight: float = 0.0
    """Penalty weight on the object's palm-frame lateral (xy) displacement
    from spawn, past a deadband (quadratic). Discourages the v2
    'slide-sideways' de-centering. 0 disables. Try -10 to -40."""
    lateral_drift_deadband: float = 0.01
    """Free lateral movement (m) before the penalty engages — leaves the
    small regrip translations rotation needs unpenalised."""
    lateral_drift_power: float = 2.0
    """Exponent on (drift − deadband): 2.0 = quadratic tail (bites the big
    slide much harder than a small one)."""
    # ---- Phase 3: bracing (palm normal force + grip strength) -----------
    brace_force_weight: float = 0.0
    """Reward weight for palm<->cylinder contact force (the cylinder's lower
    end pressed flat into the palm). GATED on alignment (brace_align_thresh)
    so it only fires once reoriented. 0 disables. Try +5 to +20."""
    brace_align_thresh: float = 0.7
    """Alignment cos at/above which the brace reward turns on (reorient first,
    then brace — mirrors the human 'reorient, then push into palm' motion)."""
    brace_max_force: float = 3.0
    """Palm contact force (N) that saturates the brace reward to 1.0."""
    grip_force_weight: float = 0.0
    """Reward weight for normalised fingertip grip force (pinch-to-power,
    consistent with screwdriver use). 0 disables. Try +2 to +10."""
    grip_force_max: float = 3.0
    """Fingertip contact force (N) that saturates the grip reward to 1.0."""
    grip_force_reduce: str = "mean"
    """'mean' (overall grip) or 'min' (worst finger) over the 3 fingertips."""
    grip_force_penalty_weight: float = 0.0
    """Penalty weight for fingertip force ABOVE grip_force_penalty_thresh
    (quadratic in normalised excess). NEGATIVE. Counters the learned death-grip
    (b32 over-clamps ~11 N) without touching the grip_force reward below thresh.
    0 disables. Try -3 to -10."""
    grip_force_penalty_thresh: float = 4.0
    """Fingertip force (N) above which the over-grip penalty engages."""
    grip_force_penalty_scale: float = 4.0
    """Normalisation (N) for the excess: penalty = ((force-thresh)/scale)**2."""
    grip_force_penalty_reduce: str = "mean"
    """'mean' (overall over-grip) or 'max' (worst finger) over the 3 fingertips."""
    grip_force_spread_weight: float = 0.0
    """Penalty weight for grip IMBALANCE: per-finger force spread (max-min)/scale over
    the 3 fingertips. NEGATIVE. Pushes toward a balanced tripod (B4-like, all fingers
    sharing load) so no single finger carries the grip — targets the user-observed
    lopsided grip (thumb idle, index/middle ~8 N). 0 disables. Try -2 to -8."""
    grip_force_spread_scale: float = 4.0
    """Normalisation (N) for the spread penalty: penalty = (maxF - minF) / scale."""
    brace_distance_weight: float = 0.0
    """Reward weight for DENSE brace shaping exp(-gap/scale): pulls the
    cylinder's nearer end toward the palm plate, gated on alignment. Needed
    because the gripped cylinder sits ~8cm from the palm, so the sparse
    brace-force reward never fires on its own. 0 disables. Try +5 to +20."""
    brace_distance_scale: float = 0.04
    """Length scale (m) of the dense brace-distance reward (smaller = sharper,
    must close more gap to earn reward)."""


# ----------------------------------------------------------------------
# Back-compat: the builders moved to env_build.py (lazy to keep this module
# importable without mjlab). `from morphohand.rl.env_cfg import to_mjlab_cfg`
# and the spec-factory imports in older scripts keep working via PEP 562.
# ----------------------------------------------------------------------

_ENV_BUILD_NAMES = (
    "to_mjlab_cfg",
    "make_hand_spec",
    "make_cube_spec",
    "make_object_spec_from_frozen",
    "_read_keyframe_object_pose",
    "_read_keyframe_state",
)


def __getattr__(name: str):
    if name in _ENV_BUILD_NAMES:
        from morphohand.rl import env_build
        return getattr(env_build, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

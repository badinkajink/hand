"""mjlab ManagerBasedRlEnvCfg assembly for the morphohand cube MVP.

This module is the mjlab-facing surface. Building blocks
(`reward.py`, `observations.py`, `reference_trajectory.py`,
`scene_loader.py`) stay framework-agnostic and are unit-tested in
isolation — see `tests/test_rl_*.py`.

# Entity decomposition

Our frozen scene XML contains BOTH the hand and the cube as a single MJCF.
mjlab's entity model wants them separate (one Entity per movable robot,
plus separate object Entities). We derive:

- `"robot"`: hand-only `MjSpec` produced by loading the frozen XML,
  deleting the cube body, dropping the floor (mjlab adds its own terrain),
  and dropping any existing keyframes (their qpos arrays still include
  cube columns that no longer exist). mjlab re-creates an "init_state"
  keyframe from `EntityCfg.InitialStateCfg(joint_pos=...)`.
- `"cube"`: a fresh `MjSpec` matching the cube's geometry (size, mass,
  friction). Built procedurally so mjlab can clone it per env.

# Actuator wiring

The scene XML already defines 15 `<position>` actuators (6 palm + 9
finger). We expose them to mjlab via `XmlPositionActuatorCfg` (wraps
pre-existing XML actuators) on the `EntityArticulationInfoCfg`. The
`JointPositionActionCfg` then references the *joint* names (not the
actuator names — see `Entity.find_joints_by_actuator_names`, which is
misleadingly named: it matches joint names among the actuated subset).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


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
# Spec factories
# ----------------------------------------------------------------------

def make_hand_spec(frozen_scene_xml: Path, object_body_name: str = "cube"):
    """Return an `mujoco.MjSpec` of the hand alone.

    Drops: the object body (`object_body_name`), the floor geom, and all
    keyframes (whose qpos arrays still reference the object freejoint
    columns). Adds a `palm_pose_site` to the palm_pose body so mjlab's
    site-based MDP terms have an anchor.
    """
    import mujoco
    spec = mujoco.MjSpec.from_file(str(frozen_scene_xml))

    for body in list(spec.worldbody.bodies):
        if body.name == object_body_name:
            spec.delete(body)
            break
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)
    # Keyframes reference cube qpos columns we just removed — drop them;
    # mjlab will create a new "init_state" keyframe from
    # EntityCfg.InitialStateCfg(joint_pos=...).
    for key in list(spec.keys):
        spec.delete(key)

    def _find_body_recursive(body, name):
        if body.name == name:
            return body
        for child in body.bodies:
            found = _find_body_recursive(child, name)
            if found is not None:
                return found
        return None
    for top in spec.worldbody.bodies:
        palm = _find_body_recursive(top, "palm_pose")
        if palm is not None:
            existing = [s for s in palm.sites if s.name == "palm_pose_site"]
            if not existing:
                palm.add_site(name="palm_pose_site", pos=(0.0, 0.0, 0.0))
            break
    return spec


def make_cube_spec(cube_size: float = 0.02, mass: float = 0.016,
                    friction: tuple[float, float, float] = (2.4, 0.2, 0.02)):
    """Back-compat: programmatic cube spec. Prefer `make_object_spec_from_frozen`
    which extracts whatever object body lives in the frozen scene."""
    import mujoco
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube", pos=(0.0, 0.0, 0.0))
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(cube_size,) * 3,
        mass=mass,
        rgba=(0.18, 0.5, 0.9, 1.0),
        friction=friction,
    )
    return spec


def make_object_spec_from_frozen(frozen_scene_xml, object_body_name: str,
                                   rename_to: str = "cube"):
    """Extract the named object body (with its geom + freejoint) from the
    frozen scene into a standalone MjSpec mjlab can attach as its own entity.

    The extracted body is RENAMED to `rename_to` ("cube" by default) so the
    rest of the env code (rewards, sensors, entity keys) can keep using the
    canonical name regardless of which underlying MJCF object we're training.

    Body pos is reset to (0, 0, 0) so the world placement is controlled
    entirely by `EntityCfg.InitialStateCfg.pos`.
    """
    import mujoco

    src = mujoco.MjSpec.from_file(str(frozen_scene_xml))
    def find_body(parent, name):
        for b in parent.bodies:
            if b.name == name:
                return b
            found = find_body(b, name)
            if found is not None:
                return found
        return None

    src_body = find_body(src.worldbody, object_body_name)
    if src_body is None:
        raise KeyError(f"object body '{object_body_name}' not found in {frozen_scene_xml}")

    spec = mujoco.MjSpec()
    new_body = spec.worldbody.add_body(name=rename_to, pos=(0.0, 0.0, 0.0))
    new_body.add_freejoint(name=f"{rename_to}_joint")
    for i, src_geom in enumerate(src_body.geoms):
        geom_name = src_geom.name or f"{rename_to}_geom_{i}"
        new_body.add_geom(
            name=geom_name,
            type=src_geom.type,
            size=tuple(src_geom.size),
            mass=float(src_geom.mass) if src_geom.mass else 0.0,
            density=float(src_geom.density) if src_geom.density else 0.0,
            friction=tuple(src_geom.friction) if hasattr(src_geom, "friction") else (1.0, 0.005, 0.0001),
            rgba=tuple(src_geom.rgba) if hasattr(src_geom, "rgba") else (0.18, 0.5, 0.9, 1.0),
            pos=tuple(src_geom.pos),
            quat=tuple(src_geom.quat),
        )
    return spec


def _read_keyframe_object_pose(
    frozen_scene_xml, keyframe_name: str, object_body_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Pull the object body's freejoint pose (xyz, wxyz quat) from the keyframe qpos.

    Used as the default `init_state.pos`/`init_state.rot` for the object
    entity, so each object spawns where CEM tuned it in the frozen scene.
    Quat matters for objects whose body has a non-identity rest quat
    (e.g. flat-laying cylinders use quat=0.707 0.707 0 0)."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(frozen_scene_xml))
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if kf_id < 0:
        raise KeyError(f"keyframe '{keyframe_name}' not in {frozen_scene_xml}")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
    if bid < 0:
        raise KeyError(f"body '{object_body_name}' not in {frozen_scene_xml}")
    jadr = int(model.body_jntadr[bid])
    qadr = int(model.jnt_qposadr[jadr])
    qpos = model.key_qpos[kf_id]
    pos = (float(qpos[qadr]), float(qpos[qadr + 1]), float(qpos[qadr + 2]))
    quat = (
        float(qpos[qadr + 3]), float(qpos[qadr + 4]),
        float(qpos[qadr + 5]), float(qpos[qadr + 6]),
    )
    return pos, quat


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
# Translation -> mjlab
# ----------------------------------------------------------------------

def _read_keyframe_state(frozen_scene_xml: Path, keyframe_name: str):
    """Pull qpos + key_ctrl for `keyframe_name` from the frozen scene.

    Returns a dict with palm_joint_pos (6,), finger_joint_pos (9,), and
    palm_default_ctrl (6,) — the values mjlab needs to initialise the env
    in the same pose CEM evaluated.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(frozen_scene_xml))
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if kf_id < 0:
        raise KeyError(f"keyframe '{keyframe_name}' not in {frozen_scene_xml}")

    def jadr(name: str) -> int:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint '{name}' not in {frozen_scene_xml}")
        return int(model.jnt_qposadr[jid])

    def aid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise KeyError(f"actuator '{name}' not in {frozen_scene_xml}")
        return int(i)

    qpos = model.key_qpos[kf_id]
    ctrl = model.key_ctrl[kf_id]
    return {
        "palm_joint_pos": tuple(float(qpos[jadr(n)]) for n in PALM_JOINT_NAMES),
        "finger_joint_pos": tuple(float(qpos[jadr(n)]) for n in FINGER_JOINT_NAMES),
        "palm_default_ctrl": tuple(float(ctrl[aid(f"a_{n}")]) for n in PALM_JOINT_NAMES),
    }


def to_mjlab_cfg(cfg: MorphoHandEnvCfg):
    """Build a live mjlab `ManagerBasedRlEnvCfg` from a `MorphoHandEnvCfg`.

    Imports mjlab lazily so this module is importable on machines without GPU.
    """
    from mjlab.actuator import XmlPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.tasks.manipulation import mdp as manipulation_mdp
    from mjlab.tasks.velocity import mdp as velocity_mdp
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.utils.noise import UniformNoiseCfg as Unoise
    from mjlab.viewer import ViewerConfig
    from morphohand.rl import mjlab_terms
    from morphohand.rl.actions import LerpFingerActionCfg, ScriptedPalmActionCfg
    from morphohand.rl.reward import DEFAULT_REWARD_WEIGHTS

    # ---- entities ------------------------------------------------------
    # Palm starts at the CEM keyframe pose (the scripted lift schedule
    # ramps palm_pz from there). Fingers start at `open_finger_qpos`
    # (thumb extended away from palm at mcp=3.14, index/middle extended
    # at mcp=0) so the cube can sit on the floor without finger
    # penetration. The Lerp finger action then drives the position
    # controller from this open pose toward `finger_default_ctrl` over
    # `finger_close_sim_steps`, producing a real closing-grasp motion.
    kf_state = _read_keyframe_state(cfg.frozen_scene_xml, cfg.keyframe_name)
    if len(cfg.open_finger_qpos) != len(FINGER_JOINT_NAMES):
        raise ValueError(
            f"open_finger_qpos length {len(cfg.open_finger_qpos)} != "
            f"{len(FINGER_JOINT_NAMES)} (must match FINGER_JOINT_NAMES order)"
        )
    palm_pz_idx = PALM_JOINT_NAMES.index("palm_pz")
    palm_joint_pos = list(kf_state["palm_joint_pos"])
    palm_default_ctrl = list(kf_state["palm_default_ctrl"])
    # IK-retargeted morphologies: start fingers from the keyframe (== the pose CEM
    # optimized the grip from), not the baseline hardcoded open pose. Used for BOTH
    # the reset pose (finger_init_pos) and the LerpFinger interpolation start below.
    open_finger_qpos = (tuple(kf_state["finger_joint_pos"])
                        if cfg.open_finger_from_keyframe
                        else tuple(cfg.open_finger_qpos))
    finger_init_pos = open_finger_qpos
    if cfg.skip_lift_phase:
        # Spawn palm + cylinder at the post-lift pose; fingers at the CEM grip.
        # Equilibrium grip force is small (actuators at setpoint), so the first
        # few sim steps are unstable — pair with --lift-phase-start-step >= 5
        # and a warmstart from the source lift policy, which has learned to
        # actively maintain the grip.
        palm_joint_pos[palm_pz_idx] = palm_joint_pos[palm_pz_idx] + float(cfg.lift_delta_z)
        palm_default_ctrl[palm_pz_idx] = palm_default_ctrl[palm_pz_idx] + float(cfg.lift_delta_z)
        finger_init_pos = cfg.finger_default_ctrl
    init_joint_pos = dict(zip(PALM_JOINT_NAMES, palm_joint_pos))
    init_joint_pos.update(zip(FINGER_JOINT_NAMES, (float(v) for v in finger_init_pos)))

    hand_entity = EntityCfg(
        spec_fn=lambda: make_hand_spec(cfg.frozen_scene_xml, cfg.object_body_name),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlPositionActuatorCfg(
                    target_names_expr=tuple(FINGER_JOINT_NAMES + PALM_JOINT_NAMES),
                ),
            ),
            soft_joint_pos_limit_factor=0.95,
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=init_joint_pos,
            joint_vel={".*": 0.0},
        ),
    )

    # Extract the object body from the frozen scene XML (works for any
    # object: cube, prism, screwdriver, ...). init_state.pos/rot default to
    # the keyframe qpos so the object spawns where CEM tuned it. Quat is
    # required for flat-laying cylinders whose source body has a non-identity
    # rest quat — without it the extracted body resets to identity and the
    # cylinder spawns standing up.
    # `reset_cube` event in the events dict below writes default_root_state
    # to sim each reset, ensuring the init pose actually applies.
    obj_init_xyz, obj_init_quat = _read_keyframe_object_pose(
        cfg.frozen_scene_xml, cfg.keyframe_name, cfg.object_body_name
    )
    if cfg.skip_lift_phase:
        obj_init_xyz = (obj_init_xyz[0], obj_init_xyz[1],
                        obj_init_xyz[2] + float(cfg.lift_delta_z)
                        + float(cfg.skip_lift_drop_offset))
    cube_entity = EntityCfg(
        spec_fn=lambda: make_object_spec_from_frozen(
            cfg.frozen_scene_xml, cfg.object_body_name
        ),
        init_state=EntityCfg.InitialStateCfg(pos=obj_init_xyz, rot=obj_init_quat),
    )

    # ---- actions: linear-interp finger closing + scripted palm lift ----
    # Policy outputs 9 dims as residuals (scale 0.2) on top of a
    # time-varying setpoint that lerps from `open_finger_qpos` toward
    # `finger_default_ctrl` over `finger_close_sim_steps`. action=0
    # produces a clean open->close motion. After the lerp finishes there
    # is a hold-at-grip phase (until `settle_steps`) so the cube has time
    # to seat, then the palm scripted action ramps palm_pz by
    # `lift_delta_z` over `lift_ramp_steps`.
    if cfg.skip_lift_phase:
        # Fingers start AT the grip pose — no closing motion. lerp(start, target, alpha)
        # is constant = finger_default_ctrl.
        finger_start = tuple(float(v) for v in cfg.finger_default_ctrl)
        # Palm holds at lifted height (palm_default_ctrl already offset above).
        scripted_settle = 0
        scripted_ramp = 1
        scripted_lift_delta = 0.0
    else:
        finger_start = tuple(float(v) for v in open_finger_qpos)
        scripted_settle = cfg.settle_steps
        scripted_ramp = cfg.lift_ramp_steps
        scripted_lift_delta = cfg.lift_delta_z
    actions = {
        "finger_ctrl": LerpFingerActionCfg(
            entity_name="robot",
            joint_names=tuple(FINGER_JOINT_NAMES),
            start_ctrl=finger_start,
            target_ctrl=tuple(float(v) for v in cfg.finger_default_ctrl),
            settle_sim_steps=cfg.finger_close_sim_steps,
            residual_scale=cfg.finger_residual_scale,
            easing=cfg.finger_close_easing,
            residual_active_from_sim_step=int(cfg.finger_residual_active_from_step) * int(cfg.decimation),
        ),
        "palm_ctrl": ScriptedPalmActionCfg(
            entity_name="robot",
            joint_names=tuple(PALM_JOINT_NAMES),
            palm_default_ctrl=tuple(palm_default_ctrl),
            palm_pz_index=palm_pz_idx,
            palm_rot_indices=(
                PALM_JOINT_NAMES.index("palm_rx"),
                PALM_JOINT_NAMES.index("palm_ry"),
                PALM_JOINT_NAMES.index("palm_rz"),
            ),
            settle_steps=scripted_settle,
            lift_ramp_steps=scripted_ramp,
            lift_delta_z=scripted_lift_delta,
            palm_rotation_residual_scale=(
                float(cfg.palm_rotation_residual_scale)
                if cfg.enable_palm_rotation_residual else 0.0
            ),
            rotation_active_from_sim_step=cfg.palm_rotation_active_from_sim_step,
        ),
    }

    # ---- observations --------------------------------------------------
    if cfg.obs_mode == "full":
        actor_terms = {
            "joint_pos": ObservationTermCfg(
                func=velocity_mdp.joint_pos_rel,
                noise=Unoise(n_min=-0.01, n_max=0.01),
            ),
            "joint_vel": ObservationTermCfg(
                func=velocity_mdp.joint_vel_rel,
                noise=Unoise(n_min=-0.5, n_max=0.5),
            ),
            "object_pos": ObservationTermCfg(
                func=manipulation_mdp.ee_to_object_distance,
                params={
                    "object_name": "cube",
                    "asset_cfg": SceneEntityCfg("robot", site_names=("palm_pose_site",)),
                },
                noise=Unoise(n_min=-0.005, n_max=0.005),
            ),
            # Actual cube pose in palm frame (7d) — critical under DR. Distinct
            # from `ref_object_pose` which is a fixed CEM trajectory.
            "object_pose_actual": ObservationTermCfg(
                func=mjlab_terms.object_pose_rel_palm,
                params={"object_name": "cube", "palm_body": "palm_pose"},
                noise=Unoise(n_min=-0.005, n_max=0.005),
            ),
            "ref_finger_qpos": ObservationTermCfg(
                func=mjlab_terms.ref_finger_qpos,
                params={
                    "run_dir": str(cfg.foundational_run_dir),
                    "frozen_scene_xml": str(cfg.frozen_scene_xml),
                },
            ),
            "ref_object_pose": ObservationTermCfg(
                func=mjlab_terms.ref_object_pose,
                params={
                    "run_dir": str(cfg.foundational_run_dir),
                    "frozen_scene_xml": str(cfg.frozen_scene_xml),
                },
            ),
            "actions": ObservationTermCfg(func=velocity_mdp.last_action),
        }
        if cfg.enable_target_axis_reward:
            # Give the policy a scalar "how off from vertical" so it can
            # learn the reorient direction. exp-shaped reward alone is too
            # sparse without an axis-error obs.
            actor_terms["target_axis_misalign"] = ObservationTermCfg(
                func=mjlab_terms.target_axis_misalignment,
                params={
                    "object_name": "cube",
                    "object_axis_local": cfg.target_axis_object_local,
                    "target_axis_world": cfg.target_axis_world,
                },
            )
    elif cfg.obs_mode == "ref_only":
        actor_terms = {
            "ref_finger_qpos": ObservationTermCfg(
                func=mjlab_terms.ref_finger_qpos,
                params={
                    "run_dir": str(cfg.foundational_run_dir),
                    "frozen_scene_xml": str(cfg.frozen_scene_xml),
                },
            ),
        }
    else:
        raise ValueError(f"Unknown obs_mode '{cfg.obs_mode}'")
    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    # ---- rewards: contact-shaped (replaces palm-based staged_position_reward,
    # ---- which is the wrong signal for our morphology — palm is static and
    # ---- fingers do the gripping).
    if cfg.foundational_run_dir is None:
        raise ValueError("foundational_run_dir is required for reference tracking rewards")

    task_scale = 1.0
    if cfg.reward_mode == "tracking_only":
        task_scale = 0.0
    elif cfg.reward_mode != "full":
        raise ValueError(f"Unknown reward_mode '{cfg.reward_mode}'")

    rewards = {
        "track_finger_qpos": RewardTermCfg(
            func=mjlab_terms.track_finger_qpos,
            weight=DEFAULT_REWARD_WEIGHTS["track_finger_qpos"][1],
            params={
                "run_dir": str(cfg.foundational_run_dir),
                "frozen_scene_xml": str(cfg.frozen_scene_xml),
                "alpha": DEFAULT_REWARD_WEIGHTS["track_finger_qpos"][2],
            },
        ),
        "track_object_pos": RewardTermCfg(
            func=mjlab_terms.track_object_pos,
            weight=DEFAULT_REWARD_WEIGHTS["track_object_pos"][1],
            params={
                "run_dir": str(cfg.foundational_run_dir),
                "frozen_scene_xml": str(cfg.frozen_scene_xml),
                "alpha": DEFAULT_REWARD_WEIGHTS["track_object_pos"][2],
            },
        ),
        "track_object_quat": RewardTermCfg(
            func=mjlab_terms.track_object_quat,
            weight=DEFAULT_REWARD_WEIGHTS["track_object_quat"][1],
            params={
                "run_dir": str(cfg.foundational_run_dir),
                "frozen_scene_xml": str(cfg.frozen_scene_xml),
                "alpha": DEFAULT_REWARD_WEIGHTS["track_object_quat"][2],
            },
        ),
        "track_finger_ctrl_anchor": RewardTermCfg(
            func=mjlab_terms.track_finger_ctrl_anchor,
            weight=DEFAULT_REWARD_WEIGHTS["track_finger_ctrl_anchor"][1],
            params={
                "run_dir": str(cfg.foundational_run_dir),
                "frozen_scene_xml": str(cfg.frozen_scene_xml),
                "alpha": DEFAULT_REWARD_WEIGHTS["track_finger_ctrl_anchor"][2],
            },
        ),
        # Fingertip contact rewards boosted: previous weights (2/3) were
        # drowned out by lift_height=80, so PPO would happily cage the cube
        # without ever touching it with fingertips. New weights make
        # prehensile contact a primary signal.
        "contact_mean": RewardTermCfg(
            func=mjlab_terms.fingertip_contact_mean, weight=10.0 * task_scale,
            params={"sensor_name": "fingertip_cube_contact"},
        ),
        "contact_min": RewardTermCfg(
            func=mjlab_terms.fingertip_contact_min,
            weight=float(cfg.contact_min_weight) * task_scale,
            params={"sensor_name": "fingertip_cube_contact"},
        ),
        "lift_height": RewardTermCfg(
            func=mjlab_terms.object_lift_height, weight=80.0 * task_scale,
            params={"object_name": "cube", "target_lift": cfg.lift_target_z_above_init},
        ),
        "object_drop": RewardTermCfg(
            func=mjlab_terms.object_drop_indicator, weight=-12.0 * task_scale,
            params={"object_name": "cube", "drop_threshold": 0.02},
        ),
        "object_xy_drift": RewardTermCfg(
            func=(mjlab_terms.object_xy_drift_gated if cfg.contact_gate_stability_rewards
                  else mjlab_terms.object_xy_drift),
            weight=cfg.object_xy_drift_weight * task_scale,
            params=(dict(object_name="cube", contact_gate_min=cfg.contact_gate_min,
                         sensor_name="fingertip_cube_contact")
                    if cfg.contact_gate_stability_rewards
                    else dict(object_name="cube")),
        ),
        "object_orientation_drift": RewardTermCfg(
            func=(mjlab_terms.object_orientation_drift_gated if cfg.contact_gate_stability_rewards
                  else mjlab_terms.object_orientation_drift),
            weight=cfg.object_orientation_drift_weight * task_scale,
            params=(dict(object_name="cube", contact_gate_min=cfg.contact_gate_min,
                         sensor_name="fingertip_cube_contact")
                    if cfg.contact_gate_stability_rewards
                    else dict(object_name="cube")),
        ),
        "finger_drift_from_grip": RewardTermCfg(
            func=(mjlab_terms.finger_drift_from_grip_gated if cfg.contact_gate_stability_rewards
                  else mjlab_terms.finger_drift_from_grip),
            weight=cfg.finger_drift_weight * task_scale,
            params=(dict(contact_gate_min=cfg.contact_gate_min,
                         sensor_name="fingertip_cube_contact")
                    if cfg.contact_gate_stability_rewards
                    else dict()),
        ),
        "fingertip_to_object": RewardTermCfg(
            func=mjlab_terms.fingertip_to_object_distance, weight=-3.0 * task_scale,
            params={"object_name": "cube"},
        ),
        "action_rate_l2": RewardTermCfg(
            func=velocity_mdp.action_rate_l2,
            weight=float(cfg.action_rate_weight) * task_scale,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=velocity_mdp.joint_pos_limits, weight=-2.0 * task_scale,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }
    if cfg.object_ang_acc_weight != 0.0 or cfg.object_ang_acc_weight_final is not None:
        rewards["object_ang_acc_l2"] = RewardTermCfg(
            func=mjlab_terms.object_ang_acc_l2,
            weight=float(cfg.object_ang_acc_weight) * task_scale,
            params={
                "object_name": "cube",
                "phase_start_step": int(cfg.object_ang_acc_phase_start_step),
            },
        )
    if cfg.lateral_drift_weight != 0.0:
        _lat_params = dict(object_name="cube",
                           deadband=float(cfg.lateral_drift_deadband),
                           power=float(cfg.lateral_drift_power))
        if cfg.contact_gate_stability_rewards:
            _lat_params.update(contact_gate_min=cfg.contact_gate_min,
                               sensor_name="fingertip_cube_contact")
        rewards["object_lateral_drift"] = RewardTermCfg(
            func=(mjlab_terms.object_lateral_drift_gated if cfg.contact_gate_stability_rewards
                  else mjlab_terms.object_lateral_drift),
            weight=float(cfg.lateral_drift_weight) * task_scale,
            params=_lat_params,
        )
    if cfg.grip_force_weight != 0.0:
        rewards["grip_force"] = RewardTermCfg(
            func=mjlab_terms.grip_force,
            weight=float(cfg.grip_force_weight) * task_scale,
            params=dict(sensor_name="fingertip_cube_contact",
                        max_force=float(cfg.grip_force_max),
                        reduce=str(cfg.grip_force_reduce)),
        )
    if cfg.grip_force_penalty_weight != 0.0:
        rewards["grip_force_excess"] = RewardTermCfg(
            func=mjlab_terms.grip_force_excess,
            weight=float(cfg.grip_force_penalty_weight) * task_scale,
            params=dict(sensor_name="fingertip_cube_contact",
                        thresh=float(cfg.grip_force_penalty_thresh),
                        scale=float(cfg.grip_force_penalty_scale),
                        reduce=str(cfg.grip_force_penalty_reduce)),
        )
    if cfg.grip_force_spread_weight != 0.0:
        rewards["grip_force_spread"] = RewardTermCfg(
            func=mjlab_terms.grip_force_spread,
            weight=float(cfg.grip_force_spread_weight) * task_scale,
            params=dict(sensor_name="fingertip_cube_contact",
                        scale=float(cfg.grip_force_spread_scale)),
        )
    if cfg.handoff_target_bank and cfg.handoff_target_weight != 0.0:
        rewards["handoff_target_proximity"] = RewardTermCfg(
            func=mjlab_terms.handoff_target_proximity,
            weight=float(cfg.handoff_target_weight) * task_scale,
            params=dict(bank_path=str(cfg.handoff_target_bank),
                        seam_lo=int(cfg.handoff_target_seam_lo),
                        seam_hi=int(cfg.handoff_target_seam_hi),
                        qpos_tol=float(cfg.handoff_target_qpos_tol),
                        scale_mult=float(cfg.handoff_target_scale_mult)),
        )
    if cfg.brace_distance_weight != 0.0 and cfg.enable_target_axis_reward:
        rewards["palm_brace_distance"] = RewardTermCfg(
            func=mjlab_terms.palm_brace_distance,
            weight=float(cfg.brace_distance_weight) * task_scale,
            params=dict(object_name="cube",
                        object_axis_local=cfg.target_axis_object_local,
                        target_axis_world=cfg.target_axis_world,
                        scale=float(cfg.brace_distance_scale),
                        align_thresh=float(cfg.brace_align_thresh),
                        reorient_start_step=int(cfg.reorient_start_step)),
        )
    if cfg.brace_force_weight != 0.0 and cfg.enable_target_axis_reward:
        rewards["palm_brace_force"] = RewardTermCfg(
            func=mjlab_terms.palm_brace_force,
            weight=float(cfg.brace_force_weight) * task_scale,
            params=dict(sensor_name="palm_cube_contact", object_name="cube",
                        object_axis_local=cfg.target_axis_object_local,
                        target_axis_world=cfg.target_axis_world,
                        align_thresh=float(cfg.brace_align_thresh),
                        reorient_start_step=int(cfg.reorient_start_step),
                        max_force=float(cfg.brace_max_force)),
        )
    if cfg.enable_target_axis_reward and cfg.target_axis_weight > 0:
        # If alpha curriculum is enabled, start at the soft alpha; curriculum
        # term will mutate the param each iter.
        init_alpha = (float(cfg.target_axis_alpha_start)
                      if cfg.target_axis_alpha_curriculum_iters > 0
                      else float(cfg.target_axis_alpha))
        rewards["target_axis_alignment"] = RewardTermCfg(
            func=mjlab_terms.target_axis_alignment,
            weight=float(cfg.target_axis_weight) * task_scale,
            params={
                "object_name": "cube",
                "object_axis_local": cfg.target_axis_object_local,
                "target_axis_world": cfg.target_axis_world,
                "alpha": init_alpha,
                "reorient_start_step": int(cfg.reorient_start_step),
            },
        )
    if cfg.enable_target_axis_reward and cfg.target_axis_progress_weight > 0:
        rewards["target_axis_progress"] = RewardTermCfg(
            func=mjlab_terms.target_axis_progress,
            weight=float(cfg.target_axis_progress_weight) * task_scale,
            params={
                "object_name": "cube",
                "object_axis_local": cfg.target_axis_object_local,
                "target_axis_world": cfg.target_axis_world,
                "reorient_start_step": int(cfg.reorient_start_step),
                "clamp_negative": bool(cfg.target_axis_progress_clamp_negative),
            },
        )
    # ---- "quick / shorter trajectory" incentives (Policy B v2) ---------
    if cfg.enable_target_axis_reward and cfg.success_bonus_weight != 0.0:
        rewards["alignment_success_bonus"] = RewardTermCfg(
            func=mjlab_terms.alignment_success_bonus,
            weight=float(cfg.success_bonus_weight) * task_scale,
            params={
                "object_name": "cube",
                "object_axis_local": cfg.target_axis_object_local,
                "target_axis_world": cfg.target_axis_world,
                "align_thresh": float(cfg.success_align_thresh),
                "hold_steps": int(cfg.success_hold_steps),
                "reorient_start_step": int(cfg.reorient_start_step),
            },
        )
    if cfg.time_cost_weight != 0.0:
        rewards["reorient_time_cost"] = RewardTermCfg(
            func=mjlab_terms.reorient_time_cost,
            weight=float(cfg.time_cost_weight) * task_scale,
            params={"reorient_start_step": int(cfg.reorient_start_step)},
        )
    if cfg.enable_target_axis_reward and cfg.speed_bonus_weight != 0.0:
        rewards["alignment_speed_bonus"] = RewardTermCfg(
            func=mjlab_terms.alignment_speed_bonus,
            weight=float(cfg.speed_bonus_weight) * task_scale,
            params={
                "object_name": "cube",
                "object_axis_local": cfg.target_axis_object_local,
                "target_axis_world": cfg.target_axis_world,
                "align_thresh": float(cfg.speed_bonus_align_thresh),
                "reorient_start_step": int(cfg.reorient_start_step),
            },
        )

    # ---- commands ------------------------------------------------------
    # `object_pose_range` is the cube SPAWN range — `LiftingCommand`
    # writes the cube to a value sampled from this range each reset.
    # Separate x and y because the hand's reachable envelope is
    # asymmetric (see /tmp/sweep_reachable2.py).
    x_j = float(cfg.cube_spawn_x_jitter if cfg.cube_spawn_x_jitter is not None
                else cfg.cube_spawn_xy_jitter)
    y_j = float(cfg.cube_spawn_y_jitter if cfg.cube_spawn_y_jitter is not None
                else cfg.cube_spawn_xy_jitter)
    x_c = float(cfg.cube_spawn_x_center)
    y_c = float(cfg.cube_spawn_y_center)
    yaw_j = float(cfg.cube_spawn_yaw_jitter)
    # If curriculum is enabled, start with center-only (no jitter) — the
    # curriculum term ramps the LiftingCommand's object_pose_range up to
    # (x_j, y_j, yaw_j) around (x_c, y_c).
    if cfg.dr_anneal_iters > 0:
        init_xj, init_yj, init_yawj = 0.0, 0.0, 0.0
    else:
        init_xj, init_yj, init_yawj = x_j, y_j, yaw_j
    # Handoff-DR curriculum ramps tilt/z from 0 — start them at 0 here.
    if cfg.handoff_dr_curriculum_iters > 0:
        init_tilt, init_zj = 0.0, 0.0
    else:
        init_tilt, init_zj = float(cfg.skip_lift_spawn_tilt_jitter), float(cfg.skip_lift_spawn_z_jitter)
    from morphohand.rl.lifting_command import LiftingCommandWithBaseQuatCfg
    commands = {
        "lift_height": LiftingCommandWithBaseQuatCfg(
            entity_name="cube",
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            difficulty="dynamic",
            object_pose_range=(None if cfg.handoff_state_bank else
                manipulation_mdp.LiftingCommandCfg.ObjectPoseRangeCfg(
                    x=(x_c - init_xj, x_c + init_xj),
                    y=(y_c - init_yj, y_c + init_yj),
                    z=(obj_init_xyz[2] - init_zj, obj_init_xyz[2] + init_zj),
                    yaw=(-init_yawj, init_yawj),
                )),
            base_quat=obj_init_quat,  # preserves flat orientation under yaw DR
            spawn_tilt_range=(-init_tilt, init_tilt),
        ),
    }

    # ---- events --------------------------------------------------------
    # `reset_base` (default asset="robot") writes the mocap pose for the
    # fixed-base robot — this is also what applies env_origin offsets so
    # every parallel env's hand spawns in its own grid cell. Removing it
    # collapses all 1024 hands onto the same world coords. Keep it.
    #
    # `reset_cube` is what we add: the cube is a floating-base freejoint
    # entity and needs its init_state.pos written explicitly on each
    # reset, otherwise mjlab leaves the cube wherever the scene compile
    # left it (observed at z=0.063 instead of the requested z=0.02).
    events = {
        "reset_base": EventTermCfg(
            func=velocity_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_cube": EventTermCfg(
            func=velocity_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=velocity_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }
    # train-the-handoff: spawn from Policy A's recorded terminal states. Added
    # LAST so it overrides reset_cube/reset_robot_joints (dict-insertion order).
    if cfg.skip_lift_phase and cfg.handoff_state_bank:
        events["reset_from_bank"] = EventTermCfg(
            func=mjlab_terms.reset_from_handoff_bank,
            mode="reset",
            params={"bank_path": str(cfg.handoff_state_bank)},
        )
    # normal-lift ONSET-grip injection: mid-episode (step-mode) overwrite of the object
    # pose+vel + robot qpos from A's real delivery bank, at the handoff onset. This keeps
    # the normal-lift obs schedule (unlike skip-lift reset_from_bank above) while putting
    # B into A's real delivered state -> the one combination that matches deploy.
    if (not cfg.skip_lift_phase) and cfg.handoff_onset_bank:
        onset = (int(cfg.handoff_onset_step) if cfg.handoff_onset_step is not None
                 else int(cfg.lift_phase_start_step))
        events["inject_onset_bank"] = EventTermCfg(
            func=mjlab_terms.inject_handoff_bank_at_onset,
            mode="step",
            params={"bank_path": str(cfg.handoff_onset_bank), "onset_step": onset,
                    "inject_velocity": bool(cfg.handoff_inject_velocity),
                    "inject_last_action": bool(cfg.handoff_inject_last_action)},
        )

    terminations = {
        "time_out": TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
    }
    if cfg.enable_lift_terminations:
        # GAE cuts the bootstrap on non-time_out terminals -> sharp
        # negative signal for unstable lifts.
        terminations["object_slip"] = TerminationTermCfg(
            func=mjlab_terms.terminate_object_slip,
            params=dict(
                lift_phase_start_step=int(cfg.lift_phase_start_step),
                xy_drift_threshold=float(cfg.term_object_slip_xy),
                object_name="cube",
            ),
        )
        terminations["object_orientation_slip"] = TerminationTermCfg(
            func=mjlab_terms.terminate_object_orientation_slip,
            params=dict(
                lift_phase_start_step=int(cfg.lift_phase_start_step),
                orientation_drift_threshold=float(cfg.term_object_slip_yaw),
                object_name="cube",
            ),
        )
        terminations["object_drop"] = TerminationTermCfg(
            func=mjlab_terms.terminate_object_drop,
            params=dict(
                lift_phase_start_step=int(cfg.lift_phase_start_step),
                drop_threshold=float(cfg.term_object_drop),
                object_name="cube",
            ),
        )
        if cfg.strict_tip_lost_termination:
            terminations["tip_lost"] = TerminationTermCfg(
                func=mjlab_terms.terminate_any_tip_lost,
                params=dict(
                    lift_phase_start_step=int(cfg.lift_phase_start_step),
                    sensor_name="fingertip_cube_contact",
                ),
            )
        else:
            terminations["tip_lost"] = TerminationTermCfg(
                func=mjlab_terms.terminate_tip_lost,
                params=dict(
                    lift_phase_start_step=int(cfg.lift_phase_start_step),
                    consecutive_steps=int(cfg.term_tip_lost_steps),
                    sensor_name="fingertip_cube_contact",
                ),
            )
        terminations["finger_slip"] = TerminationTermCfg(
            func=mjlab_terms.terminate_finger_slip,
            params=dict(
                lift_phase_start_step=int(cfg.lift_phase_start_step),
                finger_drift_threshold=float(cfg.term_finger_slip),
            ),
        )
        if cfg.terminate_low_tilt_velocity and cfg.enable_target_axis_reward:
            terminations["low_tilt_velocity"] = TerminationTermCfg(
                func=mjlab_terms.terminate_low_tilt_velocity,
                params=dict(
                    object_name="cube",
                    object_axis_local=cfg.target_axis_object_local,
                    target_axis_world=cfg.target_axis_world,
                    reorient_start_step=int(cfg.reorient_start_step),
                    window_steps=int(cfg.tilt_velocity_window),
                    min_progress=float(cfg.tilt_velocity_min_progress),
                ),
            )
        if cfg.enable_floor_proximity_termination:
            phase_start = (int(cfg.floor_proximity_phase_start_step)
                           if cfg.floor_proximity_phase_start_step is not None
                           else int(cfg.reorient_start_step))
            terminations["object_floor_proximity"] = TerminationTermCfg(
                func=mjlab_terms.terminate_object_floor_proximity,
                params=dict(
                    object_name="cube",
                    phase_start_step=phase_start,
                    min_z=float(cfg.object_min_z),
                ),
            )
    if cfg.enable_alignment_success_termination and cfg.enable_target_axis_reward:
        terminations["alignment_success"] = TerminationTermCfg(
            func=mjlab_terms.terminate_alignment_success,
            params=dict(
                object_name="cube",
                object_axis_local=cfg.target_axis_object_local,
                target_axis_world=cfg.target_axis_world,
                align_thresh=float(cfg.success_align_thresh),
                hold_steps=int(cfg.success_hold_steps),
                reorient_start_step=int(cfg.reorient_start_step),
            ),
        )

    # ---- sensors -------------------------------------------------------
    fingertip_cube_sensor = ContactSensorCfg(
        name="fingertip_cube_contact",
        primary=ContactMatch(
            mode="body",
            pattern=("thumb_tip", "index_tip", "middle_tip"),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
        fields=("found", "force"),  # force added for grip-strength (Phase 3 brace) reward
        reduce="none",
        num_slots=1,
        history_length=0,
    )
    # Palm plate <-> cylinder contact (the flat palm_mat geom lives on the
    # palm_pose body). Used by the Phase-3 bracing reward to detect/score the
    # cylinder's lower end pressing flat into the palm.
    palm_cube_sensor = ContactSensorCfg(
        name="palm_cube_contact",
        primary=ContactMatch(mode="body", pattern="palm_pose", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=0,
    )

    # ---- curriculum (jitter anneal) ------------------------------------
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    curriculum: dict = {}
    if cfg.dr_anneal_iters > 0:
        curriculum["dr_anneal"] = CurriculumTermCfg(
            func=mjlab_terms.anneal_cube_spawn_jitter,
            params=dict(
                x_max=x_j, y_max=y_j, yaw_max=yaw_j,
                x_center=x_c, y_center=y_c,
                anneal_iters=int(cfg.dr_anneal_iters),
                command_name="lift_height",
            ),
        )
    if cfg.skip_lift_phase and cfg.handoff_dr_curriculum_iters > 0:
        curriculum["handoff_dr_anneal"] = CurriculumTermCfg(
            func=mjlab_terms.anneal_spawn_tilt_z,
            params=dict(
                tilt_max=float(cfg.skip_lift_spawn_tilt_jitter),
                z_max=float(cfg.skip_lift_spawn_z_jitter),
                z_center=float(obj_init_xyz[2]),
                anneal_iters=int(cfg.handoff_dr_curriculum_iters),
                command_name="lift_height",
            ),
        )
    if cfg.tracking_anneal_iters > 0:
        track_names = (
            "track_finger_qpos", "track_object_pos",
            "track_object_quat", "track_finger_ctrl_anchor",
        )
        track_base_weights = tuple(
            float(rewards[n].weight) for n in track_names
        )
        curriculum["tracking_anneal"] = CurriculumTermCfg(
            func=mjlab_terms.anneal_tracking_weights,
            params=dict(
                term_names=track_names,
                base_weights=track_base_weights,
                final_scale=float(cfg.tracking_final_scale),
                anneal_iters=int(cfg.tracking_anneal_iters),
            ),
        )
    if (cfg.enable_target_axis_reward and cfg.target_axis_weight > 0
            and cfg.target_axis_alpha_curriculum_iters > 0):
        curriculum["target_axis_alpha_anneal"] = CurriculumTermCfg(
            func=mjlab_terms.anneal_target_axis_alpha,
            params=dict(
                alpha_start=float(cfg.target_axis_alpha_start),
                alpha_end=float(cfg.target_axis_alpha),
                anneal_iters=int(cfg.target_axis_alpha_curriculum_iters),
                reward_term_name="target_axis_alignment",
            ),
        )
    # ---- smoothness-weight ramp (Policy B v2: learn it, then make it smooth)
    smooth_names: list[str] = []
    smooth_base: list[float] = []
    smooth_final: list[float] = []
    if cfg.action_rate_weight_final is not None and "action_rate_l2" in rewards:
        smooth_names.append("action_rate_l2")
        smooth_base.append(float(rewards["action_rate_l2"].weight))
        smooth_final.append(float(cfg.action_rate_weight_final) * task_scale)
    if cfg.object_ang_acc_weight_final is not None and "object_ang_acc_l2" in rewards:
        smooth_names.append("object_ang_acc_l2")
        smooth_base.append(float(rewards["object_ang_acc_l2"].weight))
        smooth_final.append(float(cfg.object_ang_acc_weight_final) * task_scale)
    if cfg.smoothness_curriculum_iters > 0 and smooth_names:
        curriculum["smoothness_anneal"] = CurriculumTermCfg(
            func=mjlab_terms.anneal_smoothness_weights,
            params=dict(
                term_names=tuple(smooth_names),
                base_weights=tuple(smooth_base),
                final_weights=tuple(smooth_final),
                start_iter=int(cfg.smoothness_curriculum_start_iter),
                anneal_iters=int(cfg.smoothness_curriculum_iters),
            ),
        )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=cfg.num_envs,
            env_spacing=cfg.env_spacing,
            entities={"robot": hand_entity, "cube": cube_entity},
            sensors=(fingertip_cube_sensor, palm_cube_sensor),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="palm_pose",
            distance=cfg.viewer_distance,
            elevation=-15.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=64,
            njmax=400,
            mujoco=MujocoCfg(
                timestep=cfg.sim_timestep,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=cfg.decimation,
        episode_length_s=cfg.episode_length_s,
    )

"""Reward terms + curriculum weight-anneals for the morphohand mjlab env.

Split from mjlab_terms.py (CODEBASE_AUDIT.md step 3). These adapt our
pure-function reward ideas into mjlab's RewTerm signature:
`fn(env: ManagerBasedRlEnv, ...) -> torch.Tensor` returning (num_envs,).

Designed around the fact that mjlab's stock `staged_position_reward` is the
wrong signal for our morphology (it rewards palm getting close to the cube;
our hand grips with fingers from a static palm).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from morphohand.rl.terms_common import (
    _alignment_cos,
    _alignment_hold_counter,
    _contact_force_mag,
    _contact_gate,
    _get_finger_action,
    _get_finger_joint_ids,
    _get_finger_qpos,
    _get_ref_batch,
    _spawn_pose,
    _track,
)
from morphohand.rl.terms_obs import object_pose_rel_palm

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ----------------------------------------------------------------------
# CEM-reference tracking
# ----------------------------------------------------------------------

def track_finger_qpos(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str,
                       alpha: float = 20.0) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    finger_qpos = _get_finger_qpos(env)
    return _track(finger_qpos - ref["finger_qpos"], alpha)


def track_object_pos(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str,
                      alpha: float = 200.0) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    obj = env.scene["cube"]
    pos = obj.data.root_link_pose_w[:, :3]
    return _track(pos - ref["object_pos"], alpha)


def track_object_quat(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str,
                       alpha: float = 10.0) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    obj = env.scene["cube"]
    quat = obj.data.root_link_pose_w[:, 3:7]
    dot = torch.sum(quat * ref["object_quat"], dim=-1)
    geo_sq = torch.clamp(1.0 - dot * dot, min=0.0)
    return torch.exp(-alpha * geo_sq)


def track_finger_ctrl_anchor(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str,
                              alpha: float = 4.0) -> torch.Tensor:
    """Reward staying close to the CEM finger ctrl.

    The finger action is configured with default_offset = CEM best_finger_ctrl,
    so the raw policy action *is* the residual from the reference ctrl.
    Penalising its magnitude is equivalent to anchoring the absolute ctrl
    to the reference. (`ref["finger_ctrl"]` is not subtracted again — that
    would double-count the offset and saturate the reward to ~0.)
    """
    del run_dir, frozen_scene_xml  # ref isn't needed here; kept for cfg parity
    action = _get_finger_action(env)
    return _track(action, alpha)


# ----------------------------------------------------------------------
# Contact / grip force
# ----------------------------------------------------------------------

def fingertip_contact_mean(env: "ManagerBasedRlEnv",
                            sensor_name: str = "fingertip_cube_contact") -> torch.Tensor:
    """Mean fingertip-cube contact across the 3 tips, per env.

    Returns shape (num_envs,). Range [0, 1]: 1.0 means all three tips in
    contact with the cube; 0.0 means no contact.
    """
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (B, N=3)
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (found > 0).float().mean(dim=-1)


def fingertip_contact_min(env: "ManagerBasedRlEnv",
                           sensor_name: str = "fingertip_cube_contact",
                           phase_start_step: int = 0) -> torch.Tensor:
    """Worst-finger contact, per env. Discourages 2-finger grips.

    `phase_start_step` > 0 pays this only from that step on (see
    env_cfg.grip_phase_start_step); 0 = always on.

    Returns shape (num_envs,) in [0, 1].
    """
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    out = (found > 0).float().min(dim=-1).values
    if phase_start_step > 0:
        out = out * (env.episode_length_buf >= int(phase_start_step)).float()
    return out


def grip_force(env: "ManagerBasedRlEnv",
               sensor_name: str = "fingertip_cube_contact",
               max_force: float = 3.0,
               reduce: str = "mean",
               phase_start_step: int = 0) -> torch.Tensor:
    """Normalised fingertip grip force in [0, 1] — a "pinch-to-power" signal
    for the screwdriver bracing posture. Each fingertip's contact-force
    magnitude is clamped at `max_force` and normalised; `reduce` selects mean
    (overall grip) or min (worst finger). Shape (num_envs,).

    `phase_start_step` > 0 pays this only from that step on (see
    env_cfg.grip_phase_start_step); 0 = always on."""
    mag = _contact_force_mag(env, sensor_name)              # (B, n_tips)
    norm = (mag / float(max_force)).clamp(0.0, 1.0)
    out = norm.min(dim=-1).values if reduce == "min" else norm.mean(dim=-1)
    if phase_start_step > 0:
        out = out * (env.episode_length_buf >= int(phase_start_step)).float()
    return out


def grip_force_excess(env: "ManagerBasedRlEnv",
                      sensor_name: str = "fingertip_cube_contact",
                      thresh: float = 4.0,
                      scale: float = 4.0,
                      reduce: str = "mean") -> torch.Tensor:
    """Positive penalty magnitude for fingertip force ABOVE `thresh` Newtons,
    quadratic in the normalised excess ((force - thresh) / scale)**2. Reward
    weight should be NEGATIVE. `reduce` = 'mean' (overall over-grip) or 'max'
    (worst finger). This is the counter-lever to the learned "death-grip"
    (b32 clamps ~11 N): it leaves the `grip_force` REWARD (which saturates at
    grip_force_max) untouched below `thresh` — the two only overlap above
    `thresh`, where extra force earns nothing but now costs. Shape (num_envs,)."""
    mag = _contact_force_mag(env, sensor_name)                       # (B, n_tips)
    excess = ((mag - float(thresh)).clamp(min=0.0) / float(scale)).pow(2)
    return excess.amax(dim=-1) if reduce == "max" else excess.mean(dim=-1)


def grip_force_spread(env: "ManagerBasedRlEnv",
                      sensor_name: str = "fingertip_cube_contact",
                      scale: float = 4.0) -> torch.Tensor:
    """Penalty for an IMBALANCED grip: the per-finger force spread (max - min) over
    the 3 fingertips, normalised by `scale`. Reward weight should be NEGATIVE. A
    balanced tripod (all fingers sharing the load, like B4: ~7/10/10 N) scores low; a
    degenerate grip where one finger carries the load and another is idle (the
    user-observed lopsided grip — thumb ~2 N idle while index/middle clamp ~8 N) scores
    high. Targets the imbalance that the magnitude penalty (reduce=mean) is BLIND to.
    Shape (num_envs,)."""
    mag = _contact_force_mag(env, sensor_name)                       # (B, n_tips)
    spread = (mag.amax(dim=-1) - mag.amin(dim=-1)) / float(scale)
    return spread


# ----------------------------------------------------------------------
# Bracing (palm normal force + dense gap shaping)
# ----------------------------------------------------------------------

def palm_brace_force(env: "ManagerBasedRlEnv",
                     sensor_name: str = "palm_cube_contact",
                     object_name: str = "cube",
                     object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                     target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
                     align_thresh: float = 0.7,
                     reorient_start_step: int = 0,
                     max_force: float = 3.0) -> torch.Tensor:
    """Normalised palm<->cylinder contact force in [0, 1], GATED so it only
    pays once the cylinder is substantially reoriented (alignment cos >=
    `align_thresh`) and past `reorient_start_step`. Promotes the bracing
    posture — pressing the cylinder's lower end flat into the palm — without
    fighting the reorientation early. Shape (num_envs,)."""
    mag = _contact_force_mag(env, sensor_name).amax(dim=-1)  # (B,) strongest palm contact
    norm = (mag / float(max_force)).clamp(0.0, 1.0)
    cos = _alignment_cos(env, object_name, object_axis_local, target_axis_world)
    gate = (cos >= float(align_thresh)) & (env.episode_length_buf >= int(reorient_start_step))
    return norm * gate.float()


def palm_brace_distance(env: "ManagerBasedRlEnv",
                        object_name: str = "cube",
                        palm_body: str = "palm_pose",
                        object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                        target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
                        cylinder_half_len: float = 0.04,
                        scale: float = 0.04,
                        align_thresh: float = 0.7,
                        reorient_start_step: int = 0) -> torch.Tensor:
    """DENSE shaping toward bracing: exp(-gap/scale) in (0,1], where `gap` is the
    distance from the cylinder's NEARER end to the palm-plate origin, gated on
    alignment + reorient phase. The sparse `palm_brace_force` reward can never
    fire on its own — the gripped cylinder sits ~8 cm from the palm — so this
    provides the gradient to draw the end up into the palm once reoriented."""
    robot = env.scene["robot"]
    obj = env.scene[object_name]
    palm_id = robot.body_names.index(palm_body)
    palm = robot.data.body_link_pose_w[:, palm_id, :3]
    op = obj.data.root_link_pose_w[:, :3]
    qw, qx, qy, qz = obj.data.root_link_pose_w[:, 3:7].unbind(-1)
    ax, ay, az = object_axis_local
    bz = torch.stack([
        (1 - 2*(qy*qy + qz*qz))*ax + 2*(qx*qy - qw*qz)*ay + 2*(qx*qz + qw*qy)*az,
        2*(qx*qy + qw*qz)*ax + (1 - 2*(qx*qx + qz*qz))*ay + 2*(qy*qz - qw*qx)*az,
        2*(qx*qz - qw*qy)*ax + 2*(qy*qz + qw*qx)*ay + (1 - 2*(qx*qx + qy*qy))*az,
    ], dim=-1)
    gap_p = (op + float(cylinder_half_len)*bz - palm).norm(dim=-1)
    gap_m = (op - float(cylinder_half_len)*bz - palm).norm(dim=-1)
    gap = torch.minimum(gap_p, gap_m)
    rew = torch.exp(-gap / float(scale))
    cos = _alignment_cos(env, object_name, object_axis_local, target_axis_world)
    gate = (cos >= float(align_thresh)) & (env.episode_length_buf >= int(reorient_start_step))
    return rew * gate.float()


# ----------------------------------------------------------------------
# Lift / hold
# ----------------------------------------------------------------------

def object_lift_height(env: "ManagerBasedRlEnv",
                        object_name: str = "cube",
                        target_lift: float = 0.05) -> torch.Tensor:
    """Linear reward in clip(object_z - settle_z, 0, target_lift). Per-env
    settle_z refreshed every episode."""
    obj = env.scene[object_name]
    z = obj.data.root_link_pose_w[:, 2]
    settle_z = _spawn_pose(env, object_name)[:, 2]
    return (z - settle_z).clamp(min=0.0, max=target_lift)


def object_drop_indicator(env: "ManagerBasedRlEnv",
                           object_name: str = "cube",
                           drop_threshold: float = 0.02) -> torch.Tensor:
    """1.0 when object_z fell below spawn_z - drop_threshold, else 0.0.
    Per-env spawn_z refreshed every episode."""
    obj = env.scene[object_name]
    z = obj.data.root_link_pose_w[:, 2]
    settle_z = _spawn_pose(env, object_name)[:, 2]
    return (z < settle_z - drop_threshold).float()


def object_xy_drift(env: "ManagerBasedRlEnv",
                     object_name: str = "cube") -> torch.Tensor:
    """L2 drift of the object in the xy plane since episode start."""
    obj = env.scene[object_name]
    xy = obj.data.root_link_pose_w[:, :2]
    spawn_xy = _spawn_pose(env, object_name)[:, :2]
    return (xy - spawn_xy).norm(dim=-1)


def object_orientation_drift(env: "ManagerBasedRlEnv",
                              object_name: str = "cube") -> torch.Tensor:
    """Geodesic quat distance (rad) between current and spawn orientation.

    Penalizes the object rotating from its spawn orientation — what we want
    for a "lift only" trajectory. 0 = no rotation, pi = flipped 180 deg.
    """
    obj = env.scene[object_name]
    quat = obj.data.root_link_pose_w[:, 3:7]
    spawn_quat = _spawn_pose(env, object_name)[:, 3:7]
    dot = (quat * spawn_quat).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def finger_drift_from_grip(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """L2 distance between current finger qpos and the finger grip ctrl
    (CEM `best_finger_ctrl`, which is the default_offset on the finger
    action term). Penalizes the policy deviating from the trained grip
    pose, encouraging stable contact configuration."""
    finger_jids = _get_finger_joint_ids(env)
    robot = env.scene["robot"]
    qpos = robot.data.joint_pos[:, finger_jids]  # (B, 9)
    # Pull target from the action term's offset (per-env, set at build time)
    a_term = env.action_manager.get_term("finger_ctrl")
    target = a_term._target if hasattr(a_term, "_target") else None
    if target is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (qpos - target.unsqueeze(0)).norm(dim=-1)


def fingertip_to_object_distance(env: "ManagerBasedRlEnv",
                                   object_name: str = "cube",
                                   fingertip_body_names: tuple[str, ...] = (
                                       "thumb_tip", "index_tip", "middle_tip",
                                   )) -> torch.Tensor:
    """Sum of fingertip-to-object distances (m), per env.

    Used as a penalty term (smaller is better). When all three tips are
    on the cube surface, returns ~3 * cube_size + slack.
    """
    robot = env.scene["robot"]
    obj = env.scene[object_name]
    obj_pos = obj.data.root_link_pose_w[:, :3]  # (B, 3)
    body_ids = []
    for name in fingertip_body_names:
        bid = robot.body_names.index(name) if name in robot.body_names else -1
        if bid >= 0:
            body_ids.append(bid)
    if not body_ids:
        return torch.zeros(env.num_envs, device=env.device)
    tip_pos = robot.data.body_link_pose_w[:, body_ids, :3]  # (B, 3, 3)
    diff = tip_pos - obj_pos.unsqueeze(1)  # (B, 3, 3)
    dist = diff.norm(dim=-1)  # (B, 3)
    return dist.sum(dim=-1)


def object_ang_acc_l2(env: "ManagerBasedRlEnv",
                       object_name: str = "cube",
                       phase_start_step: int = 0) -> torch.Tensor:
    """Penalty proportional to L2 norm² of object angular-velocity *change*
    between consecutive policy steps (proxy for angular acceleration /
    jerkiness). Discourages high-frequency vibration of the cylinder
    that's a sim-only exploit.

    The cylinder is free to rotate — what we penalize is the OSCILLATION,
    not the rotation itself. Smooth rotation has small Δω; jittery
    rotation has large Δω.
    """
    obj = env.scene[object_name]
    cur = obj.data.root_link_ang_vel_w  # (B, 3) world-frame angular velocity
    if not hasattr(env, "_morphohand_prev_ang_vel"):
        env._morphohand_prev_ang_vel = cur.detach().clone()
    prev = env._morphohand_prev_ang_vel

    just_started = env.episode_length_buf <= 1
    if just_started.any():
        prev[just_started] = cur[just_started]

    delta = cur - prev
    env._morphohand_prev_ang_vel = cur.detach().clone()

    active = (env.episode_length_buf >= int(phase_start_step)).float()
    return (delta * delta).sum(dim=-1) * active


# ----------------------------------------------------------------------
# Reorientation (target-axis alignment family)
# ----------------------------------------------------------------------

def target_axis_alignment(env: "ManagerBasedRlEnv",
                           object_name: str = "cube",
                           object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                           target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
                           alpha: float = 4.0,
                           reorient_start_step: int = 0) -> torch.Tensor:
    """Reward for aligning the object's body-local axis with a world-frame
    target axis. Returns exp(-alpha * (1 - cos(theta))^2) where theta is
    the angle between the rotated object axis and the target.

    Defaults: cylinder's long axis is body-local +Z (Mujoco cylinder
    convention); target = world +Z (vertical). exp shaping gives a sharp
    reward near alignment (cos = 1) and a long tail far from it.

    Gated by `reorient_start_step` so the reward fires only after the
    scripted lift completes (don't push the policy to reorient mid-grasp).
    """
    cos_theta = _alignment_cos(env, object_name, object_axis_local, target_axis_world)
    reward = torch.exp(-alpha * (1.0 - cos_theta).pow(2))
    if reorient_start_step > 0:
        active = (env.episode_length_buf >= int(reorient_start_step)).float()
        reward = reward * active
    return reward


def target_axis_progress(env: "ManagerBasedRlEnv",
                          object_name: str = "cube",
                          object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                          target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
                          reorient_start_step: int = 50,
                          clamp_negative: bool = False) -> torch.Tensor:
    """Reward = current_cos - previous_cos. Dense gradient for any rotation
    *toward* the target axis, even when state-based reward is small. The
    per-env previous-step alignment is tracked in an attribute buffer.

    With clamp_negative=True, only positive progress is rewarded (no penalty
    for slipping backward). With False, signed delta — penalizes regression.

    Gated by `reorient_start_step` so the reward fires only after the
    scripted lift completes.
    """
    cur = _alignment_cos(env, object_name, object_axis_local, target_axis_world)
    if not hasattr(env, "_morphohand_prev_alignment"):
        env._morphohand_prev_alignment = cur.detach().clone()
    prev = env._morphohand_prev_alignment

    just_started = env.episode_length_buf <= 1
    if just_started.any():
        prev[just_started] = cur[just_started]

    delta = cur - prev
    env._morphohand_prev_alignment = cur.detach().clone()

    if clamp_negative:
        delta = delta.clamp(min=0.0)

    active = (env.episode_length_buf >= int(reorient_start_step)).float()
    return delta * active


def alignment_success_bonus(env: "ManagerBasedRlEnv",
                            object_name: str = "cube",
                            object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                            target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
                            align_thresh: float = 0.9,
                            hold_steps: int = 10,
                            reorient_start_step: int = 0) -> torch.Tensor:
    """One-shot reward of 1.0 on the single step the alignment-hold counter
    reaches `hold_steps` (i.e. the moment success is achieved). Pair with a
    positive weight to give an explicit terminal bonus for reaching and
    holding vertical. Fires once per episode."""
    counter = _alignment_hold_counter(
        env, "_morphohand_align_hold_rew", object_name,
        object_axis_local, target_axis_world, align_thresh, reorient_start_step)
    return (counter == int(hold_steps)).float()


def reorient_time_cost(env: "ManagerBasedRlEnv",
                       reorient_start_step: int = 0) -> torch.Tensor:
    """Constant 1.0 each policy step during the reorient phase (0 before).
    Pair with a small negative weight to pressure the policy to finish the
    reorientation quickly (shorter trajectories)."""
    return (env.episode_length_buf >= int(reorient_start_step)).float()


def alignment_speed_bonus(env: "ManagerBasedRlEnv",
                          object_name: str = "cube",
                          object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                          target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
                          align_thresh: float = 0.9,
                          reorient_start_step: int = 0) -> torch.Tensor:
    """One-shot reward proportional to the fraction of the episode remaining
    when the alignment cos *first* crosses `align_thresh`. Crossing early
    (lots of time left) pays more than crossing late → rewards quick
    reorientation. Fires once per episode."""
    cos = _alignment_cos(env, object_name, object_axis_local, target_axis_world)
    if not hasattr(env, "_morphohand_speed_bonus_fired"):
        env._morphohand_speed_bonus_fired = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool)
    fired = env._morphohand_speed_bonus_fired
    just_started = env.episode_length_buf <= 1
    if just_started.any():
        fired[just_started] = False
    in_phase = env.episode_length_buf >= int(reorient_start_step)
    crossing = (cos >= float(align_thresh)) & in_phase & (~fired)
    fired[crossing] = True
    max_steps = float(getattr(env, "max_episode_length", 0) or 0)
    if max_steps <= 0:
        max_steps = float(int(env.episode_length_buf.max().item()) + 1)
    remaining = (max_steps - env.episode_length_buf.to(torch.float32)).clamp(min=0.0) / max_steps
    return crossing.float() * remaining


# ----------------------------------------------------------------------
# De-centering (palm-frame lateral drift)
# ----------------------------------------------------------------------

def object_lateral_drift(env: "ManagerBasedRlEnv",
                          object_name: str = "cube",
                          palm_body: str = "palm_pose",
                          deadband: float = 0.01,
                          power: float = 2.0) -> torch.Tensor:
    """Penalty on the object's **palm-frame lateral (xy) displacement** from
    its spawn, with a deadband (free movement up to `deadband` m) and a power
    (>1 → quadratic past the deadband).

    Targets the v2 "slide the cylinder sideways to reorient" de-centering: the
    index/middle MCP+PIP flex inward while the thumb pushes outward, translating
    the object far to one side. The deadband leaves the small regrip
    translations rotation legitimately needs unpenalised, while the quadratic
    tail bites hard on the large slide. Palm frame (vs world xy) is robust to
    any palm motion. Shape (num_envs,)."""
    rel = object_pose_rel_palm(env, object_name, palm_body)  # (B, 7)
    xy = rel[:, :2]
    if not hasattr(env, "_morphohand_spawn_palm_xy"):
        env._morphohand_spawn_palm_xy = xy.detach().clone()
    spawn = env._morphohand_spawn_palm_xy
    just_started = env.episode_length_buf <= 1
    if just_started.any():
        spawn[just_started] = xy[just_started]
    d = (xy - spawn).norm(dim=-1)
    d = (d - float(deadband)).clamp(min=0.0)
    return d.pow(float(power))


# ----------------------------------------------------------------------
# Contact-gated stability rewards
# ----------------------------------------------------------------------
# These multiply the raw drift signal by a contact gate so the penalty
# only fires once at least `contact_gate_min` fraction of tips are touching
# the object. Rationale: during approach the fingers are clear of the cube
# and the cube shouldn't have moved at all (drift ~0 anyway). After
# first-contact the cube physically WILL be perturbed by the closing
# motion — penalising that legitimate perturbation pushes the policy to
# never touch the cube. Gating concentrates the credit on "once you HAVE
# the cube, hold it still".

def object_xy_drift_gated(env: "ManagerBasedRlEnv",
                            object_name: str = "cube",
                            contact_gate_min: float = 0.5,
                            sensor_name: str = "fingertip_cube_contact"
                            ) -> torch.Tensor:
    return object_xy_drift(env, object_name) * _contact_gate(env, sensor_name, contact_gate_min)


def object_lateral_drift_gated(env: "ManagerBasedRlEnv",
                                object_name: str = "cube",
                                palm_body: str = "palm_pose",
                                deadband: float = 0.01,
                                power: float = 2.0,
                                contact_gate_min: float = 0.5,
                                sensor_name: str = "fingertip_cube_contact"
                                ) -> torch.Tensor:
    return (object_lateral_drift(env, object_name, palm_body, deadband, power)
            * _contact_gate(env, sensor_name, contact_gate_min))


def object_orientation_drift_gated(env: "ManagerBasedRlEnv",
                                     object_name: str = "cube",
                                     contact_gate_min: float = 0.5,
                                     sensor_name: str = "fingertip_cube_contact"
                                     ) -> torch.Tensor:
    return object_orientation_drift(env, object_name) * _contact_gate(env, sensor_name, contact_gate_min)


def finger_drift_from_grip_gated(env: "ManagerBasedRlEnv",
                                   contact_gate_min: float = 0.5,
                                   sensor_name: str = "fingertip_cube_contact"
                                   ) -> torch.Tensor:
    return finger_drift_from_grip(env) * _contact_gate(env, sensor_name, contact_gate_min)


# ----------------------------------------------------------------------
# Branch-B (un-freeze Policy A): deliver B10's grip
# ----------------------------------------------------------------------

def handoff_target_proximity(env: "ManagerBasedRlEnv",
                             bank_path: str,
                             seam_lo: int = 33,
                             seam_hi: int = 37,
                             qpos_tol: float = 0.05,
                             scale_mult: float = 1.0) -> torch.Tensor:
    """Branch-B (un-freeze Policy A): dense reward in (0, 1] for delivering the
    GRIP that Policy B10 reorients from.

    Diagnostic (2026-06-04): at the residual-onset / handoff step (~35) B10's
    OBJECT state (pose + velocity, ~settled) is already ~identical to A's flat
    delivery; the only measured gap is the FINGER configuration (A's grasp vs
    B10's holding grip — up to 0.16 rad per finger joint). Even matching the
    object state and handing off early still drops the object, so the OOD is the
    grip. So this targets the finger qpos only; A's native grasp/lift/centering
    rewards keep the object pose good (clean separation of concerns).

    Reward = exp(-0.5 * mean_j ((q_j - mu_j) / (qpos_tol * scale_mult))^2) over
    the actuated FINGER joints (joint names not starting with 'palm'), where mu
    is the per-joint mean of B10's recorded step-`record_step` grip. GATED to the
    seam window [seam_lo, seam_hi] so it only shapes the DELIVERED grip, not the
    grasp/lift everywhere else. Shape (num_envs,)."""
    robot = env.scene["robot"]
    if not hasattr(env, "_handoff_target_stats"):
        d = np.load(bank_path)
        bank_names = [str(n) for n in d["joint_names"]]
        finger_bank_ids = [i for i, n in enumerate(bank_names) if not n.startswith("palm")]
        finger_names = [bank_names[i] for i in finger_bank_ids]
        # map those joint names onto THIS env's robot joint order
        env_names = list(robot.joint_names)
        env_ids = [env_names.index(n) for n in finger_names]
        mu = torch.as_tensor(d["robot_qpos"][:, finger_bank_ids].mean(0),
                             device=env.device, dtype=torch.float32)
        scale = float(qpos_tol) * float(scale_mult)
        env._handoff_target_stats = (
            torch.as_tensor(env_ids, device=env.device, dtype=torch.long), mu, scale)
    env_ids, mu, scale = env._handoff_target_stats
    q = robot.data.joint_pos[:, env_ids]                     # (B, n_finger)
    d2 = (((q - mu) / scale) ** 2).mean(dim=1)               # (B,)
    prox = torch.exp(-0.5 * d2)
    step = env.episode_length_buf
    gate = (step >= int(seam_lo)) & (step <= int(seam_hi))
    return prox * gate.float()


# ----------------------------------------------------------------------
# Curriculum terms (reward-weight / DR anneals)
# ----------------------------------------------------------------------

def anneal_smoothness_weights(env: "ManagerBasedRlEnv", env_ids,
                              term_names: tuple[str, ...],
                              base_weights: tuple[float, ...],
                              final_weights: tuple[float, ...],
                              start_iter: int,
                              anneal_iters: int,
                              num_steps_per_env: int = 24) -> float:
    """Linearly ramp the weights of named smoothness reward terms from
    `base_weights` to `final_weights`, held flat before `start_iter` and
    after `start_iter + anneal_iters`. Lets a warmstarted policy keep its
    learned rotation while smoothness penalties are dialed up over training
    ("learn it first, then make it smooth").
    """
    del env_ids
    if anneal_iters <= 0 or not term_names:
        return 1.0
    iters = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    progress = float(min(1.0, max(0.0, (iters - int(start_iter)) / float(anneal_iters))))
    for name, base, final in zip(term_names, base_weights, final_weights, strict=False):
        try:
            cfg = env.reward_manager.get_term_cfg(name)
        except (ValueError, AttributeError):
            continue
        cfg.weight = float(base) + (float(final) - float(base)) * progress
    return progress


def anneal_target_axis_alpha(env: "ManagerBasedRlEnv", env_ids,
                              alpha_start: float, alpha_end: float,
                              anneal_iters: int,
                              num_steps_per_env: int = 24,
                              reward_term_name: str = "target_axis_alignment") -> float:
    """Anneal the target_axis_alignment reward's alpha from `alpha_start` to
    `alpha_end` linearly over the first `anneal_iters` PPO iterations.
    Soft start (alpha_start ~ 0.5) → broad reward basin, gradient at large
    tilts. Sharp end (alpha_end ~ 4.0) → focused reward near target.
    """
    del env_ids
    if anneal_iters <= 0:
        return float(alpha_end)
    iters = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    progress = float(min(1.0, max(0.0, iters / float(anneal_iters))))
    alpha = float(alpha_start) + (float(alpha_end) - float(alpha_start)) * progress
    try:
        cfg = env.reward_manager.get_term_cfg(reward_term_name)
        cfg.params["alpha"] = alpha
    except (ValueError, AttributeError, KeyError):
        pass
    return alpha


def anneal_tracking_weights(env: "ManagerBasedRlEnv", env_ids,
                              term_names: tuple[str, ...],
                              base_weights: tuple[float, ...],
                              final_scale: float,
                              anneal_iters: int,
                              num_steps_per_env: int = 24) -> float:
    """Linearly scale the weights of named reward terms from `base_weights`
    at iter 0 down to `final_scale * base_weights` at `anneal_iters`.

    Use to phase out tracking-from-CEM rewards over training: early on
    tracking keeps the policy in the basin; later it's a misleading signal
    under DR (the CEM reference object_pos doesn't match the spawned cube
    position).
    """
    del env_ids
    if anneal_iters <= 0 or not term_names:
        return 1.0
    iters = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    progress = float(min(1.0, max(0.0, iters / float(anneal_iters))))
    scale = 1.0 + (float(final_scale) - 1.0) * progress
    for name, base in zip(term_names, base_weights, strict=False):
        try:
            cfg = env.reward_manager.get_term_cfg(name)
        except (ValueError, AttributeError):
            continue
        cfg.weight = float(base) * scale
    return scale


def anneal_cube_spawn_jitter(env: "ManagerBasedRlEnv", env_ids,
                              x_max: float, y_max: float, yaw_max: float,
                              anneal_iters: int,
                              x_center: float = 0.0, y_center: float = 0.0,
                              num_steps_per_env: int = 24,
                              command_name: str = "lift_height") -> float:
    """Linearly ramp the cube spawn jitter on `command_name`'s
    `object_pose_range` from 0 to ±(x_max, y_max, yaw_max) around
    (x_center, y_center) over the first `anneal_iters` PPO iterations.
    """
    del env_ids
    if anneal_iters <= 0:
        return 1.0
    iters = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    progress = float(min(1.0, max(0.0, iters / float(anneal_iters))))

    cmd = env.command_manager.get_term(command_name)
    cmd.cfg.object_pose_range.x = (x_center - x_max * progress, x_center + x_max * progress)
    cmd.cfg.object_pose_range.y = (y_center - y_max * progress, y_center + y_max * progress)
    cmd.cfg.object_pose_range.yaw = (-yaw_max * progress, yaw_max * progress)
    return progress


def anneal_spawn_tilt_z(env: "ManagerBasedRlEnv", env_ids,
                        tilt_max: float, z_max: float, z_center: float,
                        anneal_iters: int, num_steps_per_env: int = 24,
                        command_name: str = "lift_height") -> float:
    """Linearly ramp the skip-lift spawn TILT (roll/pitch) and HEIGHT jitter
    from 0 to ±(tilt_max, z_max) over the first `anneal_iters` PPO iters. The
    gradual ramp lets a warmstarted grip ADAPT to handoff-pose variation
    instead of being shocked at iter 0 (which collapsed the high-DR run:
    152 drops/iter). Mutates the LiftingCommand's spawn_tilt_range + z range."""
    del env_ids
    cmd = env.command_manager.get_term(command_name)
    if anneal_iters <= 0:
        cmd.cfg.spawn_tilt_range = (-tilt_max, tilt_max)
        cmd.cfg.object_pose_range.z = (z_center - z_max, z_center + z_max)
        return 1.0
    iters = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    progress = float(min(1.0, max(0.0, iters / float(anneal_iters))))
    cmd.cfg.spawn_tilt_range = (-tilt_max * progress, tilt_max * progress)
    cmd.cfg.object_pose_range.z = (z_center - z_max * progress, z_center + z_max * progress)
    return progress

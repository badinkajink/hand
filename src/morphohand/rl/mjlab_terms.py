"""Custom mjlab RewTerm functions for morphohand cube grasping.

These adapt our pure-function reward terms (see `morphohand.rl.reward`) into
mjlab's RewTerm signature: `fn(env: ManagerBasedRlEnv, ...) -> torch.Tensor`
returning a `(num_envs,)` tensor.

Designed around the fact that mjlab's stock `staged_position_reward` is the
wrong signal for our morphology (it rewards palm getting close to the cube;
our hand grips with fingers from a static palm).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from morphohand.rl.reference_trajectory import ReferenceTrajectory

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_FINGER_JOINT_NAMES: tuple[str, ...] = (
    "thumb_yaw", "thumb_mcp", "thumb_pip",
    "index_yaw", "index_mcp", "index_pip",
    "middle_yaw", "middle_mcp", "middle_pip",
)


def _track(diff: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.exp(-alpha * diff.pow(2).mean(dim=-1))


def _get_step_dt(env: "ManagerBasedRlEnv") -> float:
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    sim = getattr(env, "sim", None)
    sim_dt = float(getattr(sim, "dt", 0.002)) if sim is not None else 0.002
    decimation = float(getattr(env, "decimation", 1))
    return sim_dt * decimation


def _get_env_time(env: "ManagerBasedRlEnv") -> torch.Tensor:
    if hasattr(env, "progress_buf"):
        return env.progress_buf.to(dtype=torch.float32) * _get_step_dt(env)
    return torch.zeros(env.num_envs, device=env.device)


def _get_ref(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str
             ) -> ReferenceTrajectory:
    if not hasattr(env, "_morphohand_ref"):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(frozen_scene_xml))
        env._morphohand_ref = ReferenceTrajectory.from_run_dir(run_dir, model)
    return env._morphohand_ref


def _get_ref_batch(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str
                   ) -> dict[str, torch.Tensor]:
    ref = _get_ref(env, run_dir, frozen_scene_xml)
    ts = _get_env_time(env).detach().cpu().numpy()
    batch = ref.batch_at(ts)
    return {k: torch.as_tensor(v, device=env.device, dtype=torch.float32) for k, v in batch.items()}


def _get_finger_joint_ids(env: "ManagerBasedRlEnv") -> list[int]:
    if not hasattr(env, "_morphohand_finger_joint_ids"):
        robot = env.scene["robot"]
        names = list(robot.joint_names)
        env._morphohand_finger_joint_ids = [names.index(n) for n in _FINGER_JOINT_NAMES]
    return env._morphohand_finger_joint_ids


def _get_finger_qpos(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    ids = _get_finger_joint_ids(env)
    return robot.data.joint_pos[:, ids]


def _get_finger_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
    action = None
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        action = env.action_manager.action
    elif hasattr(env, "_last_actions"):
        action = env._last_actions
    if action is None:
        return torch.zeros((env.num_envs, len(_FINGER_JOINT_NAMES)), device=env.device)
    if action.dim() == 1:
        action = action.unsqueeze(0)
    if action.shape[-1] > len(_FINGER_JOINT_NAMES):
        action = action[:, :len(_FINGER_JOINT_NAMES)]
    return action


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


def ref_finger_qpos(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str
                    ) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    return ref["finger_qpos"]


def ref_object_pose(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str
                    ) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    return torch.cat([ref["object_pos"], ref["object_quat"]], dim=-1)


def object_pose_rel_palm(env: "ManagerBasedRlEnv",
                          object_name: str = "cube",
                          palm_body: str = "palm_pose") -> torch.Tensor:
    """Actual cube pose (pos+quat, 7-d) expressed in the palm frame.

    Adds a true observation of where the cube actually is — distinct from
    `ref_object_pose` which is a fixed CEM trajectory. Critical under
    domain randomization (cube spawn xy/yaw jitter): without this, the
    policy can't observe how the cube has shifted from its nominal pose.

    Returns shape (num_envs, 7): [px, py, pz, qw, qx, qy, qz].
    """
    robot = env.scene["robot"]
    obj = env.scene[object_name]
    palm_id = robot.body_names.index(palm_body)

    palm_pose_w = robot.data.body_link_pose_w[:, palm_id, :]  # (B, 7)
    obj_pose_w = obj.data.root_link_pose_w  # (B, 7)

    # pos in palm frame: rotate (obj_pos - palm_pos) by palm_quat^-1
    palm_pos = palm_pose_w[:, :3]
    palm_quat = palm_pose_w[:, 3:7]  # wxyz
    obj_pos = obj_pose_w[:, :3]
    obj_quat = obj_pose_w[:, 3:7]

    # Inverse quat: conjugate (negate xyz). wxyz format.
    palm_quat_inv = palm_quat.clone()
    palm_quat_inv[:, 1:] *= -1.0

    # Rotate (obj_pos - palm_pos) by palm_quat_inv.
    # quat rotation: q * v * q^-1, treating v as pure quat (0, vx, vy, vz).
    diff = obj_pos - palm_pos
    rel_pos = _quat_rotate(palm_quat_inv, diff)
    rel_quat = _quat_mul(palm_quat_inv, obj_quat)

    return torch.cat([rel_pos, rel_quat], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two batched wxyz quaternions, shapes (B, 4)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors `v` (B, 3) by batched wxyz quats `q` (B, 4)."""
    qw, qx, qy, qz = q.unbind(dim=-1)
    vx, vy, vz = v.unbind(dim=-1)
    # q * (0, v) * q^-1
    # Use the explicit formula: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    qvec = torch.stack([qx, qy, qz], dim=-1)
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + qw.unsqueeze(-1) * t + torch.cross(qvec, t, dim=-1)


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
                           sensor_name: str = "fingertip_cube_contact") -> torch.Tensor:
    """Worst-finger contact, per env. Discourages 2-finger grips.

    Returns shape (num_envs,) in [0, 1].
    """
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (found > 0).float().min(dim=-1).values


def reset_from_handoff_bank(env: "ManagerBasedRlEnv", env_ids, bank_path: str) -> None:
    """Reset event (train-the-handoff): spawn the object + hand from a randomly
    sampled state in Policy A's recorded terminal-state bank, so Policy B trains
    on the EXACT physically-valid grips A hands off (not synthetic spawn jitter).
    Writes object root pose (rel-pos + env origin) + velocity, and robot joint
    qpos. Pair with LiftingCommand object_pose_range=None so it isn't overwritten."""
    if not hasattr(env, "_handoff_bank"):
        import numpy as np
        d = np.load(bank_path)
        env._handoff_bank = (
            torch.as_tensor(d["obj_pose"], device=env.device, dtype=torch.float32),
            torch.as_tensor(d["obj_vel"], device=env.device, dtype=torch.float32),
            torch.as_tensor(d["robot_qpos"], device=env.device, dtype=torch.float32),
        )
    obj_pose, obj_vel, robot_qpos = env._handoff_bank
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    n = int(len(env_ids)); N = int(obj_pose.shape[0])
    idx = torch.randint(0, N, (n,), device=env.device)
    obj = env.scene["cube"]; robot = env.scene["robot"]
    pose = obj_pose[idx].clone()
    pose[:, :3] = pose[:, :3] + env.scene.env_origins[env_ids]
    obj.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    obj.write_root_link_velocity_to_sim(obj_vel[idx], env_ids=env_ids)
    robot.write_joint_position_to_sim(robot_qpos[idx], env_ids=env_ids)


def inject_handoff_bank_at_onset(env: "ManagerBasedRlEnv", env_ids, bank_path: str,
                                 onset_step: int, inject_velocity: bool = True,
                                 inject_last_action: bool = True) -> None:
    """Step-mode event (normal-lift ONSET-grip injection): ONCE per episode, on the
    step where `episode_length_buf == onset_step`, overwrite the object pose+vel and
    robot qpos with a randomly sampled state from Policy A's REAL delivery bank.

    Unlike `reset_from_handoff_bank` (reset-mode, skip-lift only), this fires
    MID-EPISODE in the NORMAL-lift env, so Policy B trains on A's actual delivered
    grip+pose (state in-distribution) AND under the normal-lift observation schedule
    (obs in-distribution) — the one combination that matches the continuous deploy and
    that neither branch-B nor adapt-B achieved. The teleport is just how training puts
    B into A's delivery state; at deploy B arrives there organically (no teleport), so
    we also zero the finger joint velocities (A's delivery is settled) to avoid a
    spurious velocity transient that B could otherwise learn to expect.

    Markov-COMPLETE injection (2026-06-09): since the env has no differenced/history
    observations and position actuators carry no `act` state, the ONLY non-physical
    quantity a teleport corrupts is the history-dependent `last_action` obs. So we
    inject A's REAL object velocity + REAL finger/palm qvel from the bank AND overwrite
    the action manager's `last_action` with A's delivered action — making the injected
    seam physically indistinguishable from A's organic deploy hand-off. (Earlier this
    zeroed velocities and left last_action as B's gated value: a STATIC, OOD seam.)

    mjlab calls "step" events every step with env_ids=None; this self-gates on the
    step, so the equality fires exactly once per episode (episode_length_buf is +1/step).
    Step-mode events fire BEFORE the obs is recomputed, so the last_action override lands
    in the very obs B reads for its first post-seam decision."""
    if not hasattr(env, "_handoff_onset_bank"):
        d = np.load(bank_path)
        N = int(d["obj_pose"].shape[0])
        # Back-compat: old banks lack robot_qvel / a_last -> fall back to zeros (the
        # prior static behavior) rather than crash, but warn so it's never silent.
        if "robot_qvel" not in d or "a_last" not in d:
            print(f"[onset-inject] WARN bank {bank_path} missing robot_qvel/a_last "
                  f"-> falling back to STATIC injection (zero finger vel, no last_action override). "
                  f"Re-record with the fixed rl_record_handoff_states.py for the complete-state seam.")
        qvel = (d["robot_qvel"] if "robot_qvel" in d
                else np.zeros_like(d["robot_qpos"]))
        a_last = (d["a_last"] if "a_last" in d
                  else np.zeros((N, env.action_manager.total_action_dim), dtype=np.float32))
        env._handoff_onset_bank = (
            torch.as_tensor(d["obj_pose"], device=env.device, dtype=torch.float32),
            torch.as_tensor(d["obj_vel"], device=env.device, dtype=torch.float32),
            torch.as_tensor(d["robot_qpos"], device=env.device, dtype=torch.float32),
            torch.as_tensor(qvel, device=env.device, dtype=torch.float32),
            torch.as_tensor(a_last, device=env.device, dtype=torch.float32),
        )
    obj_pose, obj_vel, robot_qpos, robot_qvel, a_last = env._handoff_onset_bank
    at_onset = (env.episode_length_buf == int(onset_step)).nonzero().flatten()
    if at_onset.numel() == 0:
        return
    n = int(at_onset.numel()); N = int(obj_pose.shape[0])
    idx = torch.randint(0, N, (n,), device=env.device)
    obj = env.scene["cube"]; robot = env.scene["robot"]
    pose = obj_pose[idx].clone()
    pose[:, :3] = pose[:, :3] + env.scene.env_origins[at_onset]
    obj.write_root_link_pose_to_sim(pose, env_ids=at_onset)
    robot.write_joint_position_to_sim(robot_qpos[idx], env_ids=at_onset)
    if inject_velocity:
        obj.write_root_link_velocity_to_sim(obj_vel[idx], env_ids=at_onset)   # REAL obj velocity
        robot.write_joint_velocity_to_sim(robot_qvel[idx], env_ids=at_onset)  # REAL finger/palm vel
    else:  # STATIC ablation: zero velocities (the pre-2026-06-09 behavior)
        obj.write_root_link_velocity_to_sim(torch.zeros_like(obj_vel[idx]), env_ids=at_onset)
        robot.write_joint_velocity_to_sim(torch.zeros_like(robot_qvel[idx]), env_ids=at_onset)
    # Override the history-dependent `last_action` obs so B's first post-seam decision
    # reads A's delivered action (as at deploy), not B's gated-off lift action.
    if inject_last_action:
        am = env.action_manager
        adim = a_last.shape[1]
        am._action[at_onset, :adim] = a_last[idx]
        am._prev_action[at_onset, :adim] = a_last[idx]


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


def _contact_force_mag(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Per-slot contact force magnitude for a contact sensor with a `force`
    field. Returns shape (num_envs, n_slots); the contact `force` field is a
    3-vector per slot, so we take its L2 norm. Zeros if unavailable."""
    sensor = env.scene.sensors[sensor_name]
    force = getattr(sensor.data, "force", None)
    if force is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    if force.dim() == 2:            # (B, 3) single slot
        return force.norm(dim=-1, keepdim=True)
    return force.norm(dim=-1)       # (B, n_slots, 3) -> (B, n_slots)


def grip_force(env: "ManagerBasedRlEnv",
               sensor_name: str = "fingertip_cube_contact",
               max_force: float = 3.0,
               reduce: str = "mean") -> torch.Tensor:
    """Normalised fingertip grip force in [0, 1] — a "pinch-to-power" signal
    for the screwdriver bracing posture. Each fingertip's contact-force
    magnitude is clamped at `max_force` and normalised; `reduce` selects mean
    (overall grip) or min (worst finger). Shape (num_envs,)."""
    mag = _contact_force_mag(env, sensor_name)              # (B, n_tips)
    norm = (mag / float(max_force)).clamp(0.0, 1.0)
    return norm.min(dim=-1).values if reduce == "min" else norm.mean(dim=-1)


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


def _spawn_pose(env: "ManagerBasedRlEnv", object_name: str = "cube"):
    """Per-env cube spawn pose, refreshed on episode reset.

    Returns dict of tensors {xy: (B, 2), z: (B,), quat: (B, 4)} taken at
    the FIRST step of each episode (episode_length_buf == 1).

    The previous implementation cached on first call across all envs and
    never refreshed — so under DR, where the spawn varies per episode,
    the "drift from spawn" was actually drift from the very first
    episode's pose, which is meaningless after the first reset.
    """
    obj = env.scene[object_name]
    pose = obj.data.root_link_pose_w  # (B, 7)
    if not hasattr(env, "_morphohand_spawn_pose"):
        env._morphohand_spawn_pose = pose.detach().clone()  # init buffer

    # Snapshot pose for envs that just started a new episode.
    # episode_length_buf is incremented after the env step; == 1 means we
    # just finished the first step after reset. Capture HERE (after one
    # step of settling so the cube isn't penetrating the floor).
    just_started = env.episode_length_buf <= 1
    if just_started.any():
        env._morphohand_spawn_pose[just_started] = pose[just_started].detach().clone()
    return env._morphohand_spawn_pose


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
    obj = env.scene[object_name]
    quat = obj.data.root_link_pose_w[:, 3:7]  # (B, 4) wxyz
    qw, qx, qy, qz = quat.unbind(dim=-1)
    ax, ay, az = object_axis_local
    # Rotate (ax, ay, az) by quat — full formula for body axis in world.
    # body_z_world = R @ (0,0,1) for the default axis; generalize below.
    # Using rotation matrix columns:
    # R[:,0] = (1 - 2(y²+z²), 2(xy+wz), 2(xz-wy))
    # R[:,1] = (2(xy-wz), 1 - 2(x²+z²), 2(yz+wx))
    # R[:,2] = (2(xz+wy), 2(yz-wx), 1 - 2(x²+y²))
    r00 = 1 - 2 * (qy * qy + qz * qz);  r01 = 2 * (qx * qy - qw * qz);  r02 = 2 * (qx * qz + qw * qy)
    r10 = 2 * (qx * qy + qw * qz);      r11 = 1 - 2 * (qx * qx + qz * qz); r12 = 2 * (qy * qz - qw * qx)
    r20 = 2 * (qx * qz - qw * qy);      r21 = 2 * (qy * qz + qw * qx);  r22 = 1 - 2 * (qx * qx + qy * qy)
    bx = r00 * ax + r01 * ay + r02 * az
    by = r10 * ax + r11 * ay + r12 * az
    bz = r20 * ax + r21 * ay + r22 * az
    tx, ty, tz = target_axis_world
    cos_theta = (bx * tx + by * ty + bz * tz).clamp(-1.0, 1.0)
    reward = torch.exp(-alpha * (1.0 - cos_theta).pow(2))
    if reorient_start_step > 0:
        active = (env.episode_length_buf >= int(reorient_start_step)).float()
        reward = reward * active
    return reward


def target_axis_misalignment(env: "ManagerBasedRlEnv",
                              object_name: str = "cube",
                              object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                              target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
                              ) -> torch.Tensor:
    """Raw angle (rad) between object's body-local axis and a world target
    axis. For observation use — gives the policy a vector to "drive to zero".
    Returns shape (num_envs,)."""
    obj = env.scene[object_name]
    quat = obj.data.root_link_pose_w[:, 3:7]
    qw, qx, qy, qz = quat.unbind(dim=-1)
    ax, ay, az = object_axis_local
    bx = (1 - 2 * (qy * qy + qz * qz)) * ax + 2 * (qx * qy - qw * qz) * ay + 2 * (qx * qz + qw * qy) * az
    by = 2 * (qx * qy + qw * qz) * ax + (1 - 2 * (qx * qx + qz * qz)) * ay + 2 * (qy * qz - qw * qx) * az
    bz = 2 * (qx * qz - qw * qy) * ax + 2 * (qy * qz + qw * qx) * ay + (1 - 2 * (qx * qx + qy * qy)) * az
    tx, ty, tz = target_axis_world
    cos_theta = (bx * tx + by * ty + bz * tz).clamp(-1.0, 1.0)
    return torch.acos(cos_theta).unsqueeze(-1)


def _alignment_cos(env: "ManagerBasedRlEnv", object_name: str,
                    object_axis_local: tuple[float, float, float],
                    target_axis_world: tuple[float, float, float]) -> torch.Tensor:
    """Shared helper: cos(theta) between object's body-local axis (rotated
    into world frame) and the world target axis. Returns shape (num_envs,).
    Used by both `target_axis_alignment` (state reward), `target_axis_progress`
    (delta reward), and the velocity-based termination."""
    obj = env.scene[object_name]
    quat = obj.data.root_link_pose_w[:, 3:7]
    qw, qx, qy, qz = quat.unbind(dim=-1)
    ax, ay, az = object_axis_local
    bx = (1 - 2 * (qy * qy + qz * qz)) * ax + 2 * (qx * qy - qw * qz) * ay + 2 * (qx * qz + qw * qy) * az
    by = 2 * (qx * qy + qw * qz) * ax + (1 - 2 * (qx * qx + qz * qz)) * ay + 2 * (qy * qz - qw * qx) * az
    bz = 2 * (qx * qz - qw * qy) * ax + 2 * (qy * qz + qw * qx) * ay + (1 - 2 * (qx * qx + qy * qy)) * az
    tx, ty, tz = target_axis_world
    return (bx * tx + by * ty + bz * tz).clamp(-1.0, 1.0)


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


def _alignment_hold_counter(env: "ManagerBasedRlEnv", attr: str,
                            object_name: str,
                            object_axis_local: tuple[float, float, float],
                            target_axis_world: tuple[float, float, float],
                            align_thresh: float,
                            reorient_start_step: int) -> torch.Tensor:
    """Per-env count of *consecutive* policy steps with alignment cos >=
    `align_thresh` during the reorient phase. Resets on episode start and
    whenever alignment drops below threshold. Stored under `attr` so the
    success reward and success termination can keep independent (but
    identical) counters without double-incrementing a shared one.
    """
    cos = _alignment_cos(env, object_name, object_axis_local, target_axis_world)
    if not hasattr(env, attr):
        setattr(env, attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    counter = getattr(env, attr)
    just_started = env.episode_length_buf <= 1
    if just_started.any():
        counter[just_started] = 0
    in_phase = env.episode_length_buf >= int(reorient_start_step)
    aligned = (cos >= float(align_thresh)) & in_phase
    counter[aligned] += 1
    counter[~aligned] = 0
    return counter


def terminate_alignment_success(env: "ManagerBasedRlEnv",
                                object_name: str = "cube",
                                object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                                target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
                                align_thresh: float = 0.9,
                                hold_steps: int = 10,
                                reorient_start_step: int = 0) -> torch.Tensor:
    """Terminate (success) when the object axis has been within `align_thresh`
    cos of the target for `hold_steps` consecutive policy steps. Ending the
    episode on success means earlier success → higher discounted return
    (rewards quick reorientation), and locks in the result (discourages
    slipping back down)."""
    counter = _alignment_hold_counter(
        env, "_morphohand_align_hold_term", object_name,
        object_axis_local, target_axis_world, align_thresh, reorient_start_step)
    return counter >= int(hold_steps)


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


def terminate_low_tilt_velocity(env: "ManagerBasedRlEnv",
                                 object_name: str = "cube",
                                 object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                                 target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
                                 reorient_start_step: int = 50,
                                 window_steps: int = 20,
                                 min_progress: float = 0.05) -> torch.Tensor:
    """Terminate envs whose alignment hasn't improved by at least
    `min_progress` over the last `window_steps` policy steps during the
    reorient phase. Kills "just hold the lift" local optima.

    Tracks per-env alignment from `window_steps` ago in a buffer. Fires
    only after `reorient_start_step + window_steps` (need history first).
    """
    cur = _alignment_cos(env, object_name, object_axis_local, target_axis_world)
    if not hasattr(env, "_morphohand_alignment_history"):
        # Ring buffer: (num_envs, window_steps)
        env._morphohand_alignment_history = torch.zeros(
            env.num_envs, int(window_steps), device=env.device
        )
        env._morphohand_alignment_history_idx = 0

    buf = env._morphohand_alignment_history
    idx = env._morphohand_alignment_history_idx
    old = buf[:, idx].clone()  # value from window_steps ago
    buf[:, idx] = cur.detach()
    env._morphohand_alignment_history_idx = (idx + 1) % int(window_steps)

    just_started = env.episode_length_buf <= 1
    if just_started.any():
        # On episode reset, fill the buffer with current value so we don't
        # falsely fire termination on the first window_steps after reset.
        buf[just_started] = cur[just_started].unsqueeze(-1)

    progress = cur - old
    # Fire only after we have history AND we're in reorient phase.
    in_phase = env.episode_length_buf >= int(reorient_start_step) + int(window_steps)
    insufficient = progress < float(min_progress)
    return in_phase & insufficient


def terminate_any_tip_lost(env: "ManagerBasedRlEnv",
                            lift_phase_start_step: int = 40,
                            sensor_name: str = "fingertip_cube_contact",
                            ) -> torch.Tensor:
    """Terminate if ANY single tip is off the object for one step during
    the lift/manipulation phase. Stricter version of `terminate_tip_lost`
    (no consecutive-step grace). Use for the in-hand reorient task where
    contact maintenance is a hard requirement."""
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    any_tip_lost = (found <= 0).any(dim=-1)
    in_phase = _in_lift_phase(env, lift_phase_start_step)
    return any_tip_lost & in_phase


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


def _contact_gate(env: "ManagerBasedRlEnv", sensor_name: str,
                   contact_gate_min: float) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    contact_mean = (found > 0).float().mean(dim=-1)
    return (contact_mean >= contact_gate_min).float()


def object_xy_drift_gated(env: "ManagerBasedRlEnv",
                            object_name: str = "cube",
                            contact_gate_min: float = 0.5,
                            sensor_name: str = "fingertip_cube_contact"
                            ) -> torch.Tensor:
    return object_xy_drift(env, object_name) * _contact_gate(env, sensor_name, contact_gate_min)


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
# Lift-phase early terminations
# ----------------------------------------------------------------------
# Engaged once `env.episode_length_buf >= lift_phase_start_step` (i.e.,
# after the scripted lift ramp has completed + a few steps of hold). These
# do NOT fire during approach or initial contact formation, so the policy
# isn't punished for legitimate transients. Returning True for an env
# causes mjlab to reset that env at end-of-step; GAE treats it as a
# terminal state (no bootstrap), which is the negative signal.


def _in_lift_phase(env: "ManagerBasedRlEnv", lift_phase_start_step: int
                    ) -> torch.Tensor:
    return env.episode_length_buf >= int(lift_phase_start_step)


def terminate_object_slip(env: "ManagerBasedRlEnv",
                            lift_phase_start_step: int = 40,
                            xy_drift_threshold: float = 0.015,
                            object_name: str = "cube") -> torch.Tensor:
    drift = object_xy_drift(env, object_name)
    return _in_lift_phase(env, lift_phase_start_step) & (drift > xy_drift_threshold)


def terminate_object_orientation_slip(env: "ManagerBasedRlEnv",
                                        lift_phase_start_step: int = 40,
                                        orientation_drift_threshold: float = 0.5,
                                        object_name: str = "cube"
                                        ) -> torch.Tensor:
    drift = object_orientation_drift(env, object_name)
    return _in_lift_phase(env, lift_phase_start_step) & (drift > orientation_drift_threshold)


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


def terminate_object_drop(env: "ManagerBasedRlEnv",
                            lift_phase_start_step: int = 40,
                            drop_threshold: float = 0.02,
                            object_name: str = "cube") -> torch.Tensor:
    obj = env.scene[object_name]
    z = obj.data.root_link_pose_w[:, 2]
    spawn_z = _spawn_pose(env, object_name)[:, 2]
    return _in_lift_phase(env, lift_phase_start_step) & (z < spawn_z - drop_threshold)


def terminate_object_floor_proximity(env: "ManagerBasedRlEnv",
                                       phase_start_step: int = 40,
                                       min_z: float = 0.05,
                                       object_name: str = "cube") -> torch.Tensor:
    """Terminate if object center z falls below `min_z` (world frame)
    during the post-lift phase. Used for reorient tasks to forbid
    floor-bracing strategies — the policy must hold the object high
    enough that no body extent can touch the ground."""
    obj = env.scene[object_name]
    z = obj.data.root_link_pose_w[:, 2]
    in_phase = env.episode_length_buf >= int(phase_start_step)
    return in_phase & (z < float(min_z))


def terminate_tip_lost(env: "ManagerBasedRlEnv",
                         lift_phase_start_step: int = 40,
                         consecutive_steps: int = 3,
                         sensor_name: str = "fingertip_cube_contact"
                         ) -> torch.Tensor:
    """Terminate if any tip is off the object for >= consecutive_steps
    consecutive policy steps during the lift phase.

    Per-env counter persists across steps; resets when (a) the env
    resets, or (b) contact is restored on all tips.
    """
    if not hasattr(env, "_morphohand_tip_lost_counter"):
        env._morphohand_tip_lost_counter = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
    counter = env._morphohand_tip_lost_counter

    just_started = env.episode_length_buf <= 1
    if just_started.any():
        counter[just_started] = 0

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    any_tip_lost = (found <= 0).any(dim=-1)
    in_phase = _in_lift_phase(env, lift_phase_start_step)

    fire = any_tip_lost & in_phase
    counter[fire] += 1
    counter[~fire] = 0
    return counter >= int(consecutive_steps)


def terminate_finger_slip(env: "ManagerBasedRlEnv",
                            lift_phase_start_step: int = 40,
                            finger_drift_threshold: float = 0.3
                            ) -> torch.Tensor:
    drift = finger_drift_from_grip(env)
    return _in_lift_phase(env, lift_phase_start_step) & (drift > finger_drift_threshold)


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

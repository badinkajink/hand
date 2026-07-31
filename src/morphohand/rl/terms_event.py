"""Event terms (reset/step-mode state writes) + termination terms for the
morphohand mjlab env. Split from mjlab_terms.py (CODEBASE_AUDIT.md step 3)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.event_manager import requires_model_fields

from morphohand.rl.terms_common import (
    _alignment_cos,
    _alignment_hold_counter,
    _in_lift_phase,
    _spawn_pose,
)
from morphohand.rl.terms_reward import (
    finger_drift_from_grip,
    object_orientation_drift,
    object_xy_drift,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ----------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------

@requires_model_fields("geom_solimp")
def randomize_geom_solimp(env: "ManagerBasedRlEnv", env_ids,
                          dmin_range: tuple[float, float] = (0.97, 0.985),
                          dmax_range: tuple[float, float] = (0.995, 0.999)) -> None:
    """Reset event: per-env contact-compliance DR (docs/rl/compliance_dr_plan.md).

    Samples ONE softness u ~ U(0,1) per env and lerps BOTH solimp dmin and dmax
    from it — a joint draw keeps the pair correlated and ordered (independent
    draws can pair a soft dmin with a hard dmax, a contact regime the
    compliance-robustness sweep never validated). Writes ALL geoms in the env's
    world, matching the whole-scene solimp edit the sweep evaluates against.
    solref and solimp width/midpoint/power are left at scene values.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    u = torch.rand(len(env_ids), 1, device=env.device)
    solimp = env.sim.model.geom_solimp  # (nworld, ngeom, 5): dmin dmax width mid power
    solimp[env_ids, :, 0] = dmin_range[0] + (dmin_range[1] - dmin_range[0]) * u
    solimp[env_ids, :, 1] = dmax_range[0] + (dmax_range[1] - dmax_range[0]) * u


def reset_from_handoff_bank(env: "ManagerBasedRlEnv", env_ids, bank_path: str) -> None:
    """Reset event (train-the-handoff): spawn the object + hand from a randomly
    sampled state in Policy A's recorded terminal-state bank, so Policy B trains
    on the EXACT physically-valid grips A hands off (not synthetic spawn jitter).
    Writes object root pose (rel-pos + env origin) + velocity, and robot joint
    qpos. Pair with LiftingCommand object_pose_range=None so it isn't overwritten."""
    if not hasattr(env, "_handoff_bank"):
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


# ----------------------------------------------------------------------
# Lift-phase early terminations
# ----------------------------------------------------------------------
# Engaged once `env.episode_length_buf >= lift_phase_start_step` (i.e.,
# after the scripted lift ramp has completed + a few steps of hold). These
# do NOT fire during approach or initial contact formation, so the policy
# isn't punished for legitimate transients. Returning True for an env
# causes mjlab to reset that env at end-of-step; GAE treats it as a
# terminal state (no bootstrap), which is the negative signal.

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
                         sensor_name: str = "fingertip_cube_contact",
                         min_tips_in_contact: int = 3,
                         ) -> torch.Tensor:
    """Terminate if fewer than `min_tips_in_contact` tips are on the object for
    >= consecutive_steps consecutive policy steps during the lift phase.

    `min_tips_in_contact` defaults to 3 (all of them), which is the historical
    behaviour and what every run before it existed used. Lower it ONLY where a
    finger is structurally unable to reach the object — on the perpendicular
    topology the thumb cannot touch the shaft at any pinch offset that still
    permits the gravity reorient (measured: thumb 0 N throughout the successful
    open-loop swing, and the offsets where it does contact kill the swing), so
    requiring 3 makes every episode terminate ~3 steps after the lift phase
    opens regardless of what the policy does. This is NOT a way to quiet a
    policy that merely drops the object: object_drop, object_slip and
    floor_proximity all still fire, and losing one of the two opposed fingers
    still ends the episode at min_tips_in_contact=2.

    Per-env counter persists across steps; resets when (a) the env
    resets, or (b) enough tips regain contact.
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
    n_in_contact = (found > 0).sum(dim=-1)
    any_tip_lost = n_in_contact < int(min_tips_in_contact)
    in_phase = _in_lift_phase(env, lift_phase_start_step)

    fire = any_tip_lost & in_phase
    counter[fire] += 1
    counter[~fire] = 0
    return counter >= int(consecutive_steps)


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


def terminate_finger_slip(env: "ManagerBasedRlEnv",
                            lift_phase_start_step: int = 40,
                            finger_drift_threshold: float = 0.3
                            ) -> torch.Tensor:
    drift = finger_drift_from_grip(env)
    return _in_lift_phase(env, lift_phase_start_step) & (drift > finger_drift_threshold)


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

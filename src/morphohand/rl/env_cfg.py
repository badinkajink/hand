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
    init_joint_pos = dict(zip(PALM_JOINT_NAMES, kf_state["palm_joint_pos"]))
    init_joint_pos.update(zip(FINGER_JOINT_NAMES, (float(v) for v in cfg.open_finger_qpos)))

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
    start_ctrl = tuple(float(v) for v in cfg.open_finger_qpos)
    actions = {
        "finger_ctrl": LerpFingerActionCfg(
            entity_name="robot",
            joint_names=tuple(FINGER_JOINT_NAMES),
            start_ctrl=start_ctrl,
            target_ctrl=tuple(float(v) for v in cfg.finger_default_ctrl),
            settle_sim_steps=cfg.finger_close_sim_steps,
            residual_scale=cfg.finger_residual_scale,
            easing=cfg.finger_close_easing,
        ),
        "palm_ctrl": ScriptedPalmActionCfg(
            entity_name="robot",
            joint_names=tuple(PALM_JOINT_NAMES),
            palm_default_ctrl=kf_state["palm_default_ctrl"],
            palm_pz_index=PALM_JOINT_NAMES.index("palm_pz"),
            settle_steps=cfg.settle_steps,
            lift_ramp_steps=cfg.lift_ramp_steps,
            lift_delta_z=cfg.lift_delta_z,
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
            func=mjlab_terms.fingertip_contact_min, weight=30.0 * task_scale,
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
            func=velocity_mdp.action_rate_l2, weight=-0.005 * task_scale,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=velocity_mdp.joint_pos_limits, weight=-2.0 * task_scale,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }

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
    from morphohand.rl.lifting_command import LiftingCommandWithBaseQuatCfg
    commands = {
        "lift_height": LiftingCommandWithBaseQuatCfg(
            entity_name="cube",
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            difficulty="dynamic",
            object_pose_range=manipulation_mdp.LiftingCommandCfg.ObjectPoseRangeCfg(
                x=(x_c - init_xj, x_c + init_xj),
                y=(y_c - init_yj, y_c + init_yj),
                z=(obj_init_xyz[2], obj_init_xyz[2]),  # keep cube on floor at its keyframe z
                yaw=(-init_yawj, init_yawj),
            ),
            base_quat=obj_init_quat,  # preserves flat orientation under yaw DR
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

    # ---- sensors -------------------------------------------------------
    fingertip_cube_sensor = ContactSensorCfg(
        name="fingertip_cube_contact",
        primary=ContactMatch(
            mode="body",
            pattern=("thumb_tip", "index_tip", "middle_tip"),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
        fields=("found",),
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

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=cfg.num_envs,
            env_spacing=cfg.env_spacing,
            entities={"robot": hand_entity, "cube": cube_entity},
            sensors=(fingertip_cube_sensor,),
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

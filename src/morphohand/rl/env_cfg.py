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

def make_hand_spec(frozen_scene_xml: Path):
    """Return an `mujoco.MjSpec` of the hand alone.

    Drops: the cube body, the floor geom, and all keyframes (whose qpos
    arrays still reference the cube freejoint columns). Adds a
    `palm_pose_site` to the palm_pose body so mjlab's
    `staged_position_reward` (site-based) has an anchor.
    """
    import mujoco
    spec = mujoco.MjSpec.from_file(str(frozen_scene_xml))

    for body in list(spec.worldbody.bodies):
        if body.name == "cube":
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
    """Recreate the cube body from `scene_cube_short_proximal.xml`.

    Defaults: 4x4x4 cm box, density 500 -> mass 0.016 kg. Friction matches
    the per-geom override on the original cube.
    """
    import mujoco
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube", pos=(0.0, 0.0, cube_size))
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


# ----------------------------------------------------------------------
# Cfg
# ----------------------------------------------------------------------

@dataclass
class MorphoHandEnvCfg:
    """Top-level config bag for the morphohand RL env.

    Translates to `ManagerBasedRlEnvCfg` via `to_mjlab_cfg(cfg)`. Kept as
    a plain dataclass so the env can be inspected/serialised without
    importing mjlab/torch.
    """
    frozen_scene_xml: Path
    keyframe_name: str = "open_short_manual"
    foundational_run_dir: Path | None = None
    initial_finger_ctrl: tuple[float, ...] = (
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
    object_size: float = 0.02
    object_mass: float = 0.016
    object_friction: tuple[float, float, float] = (2.4, 0.2, 0.02)
    lift_target_z_above_init: float = 0.05


# ----------------------------------------------------------------------
# Translation -> mjlab
# ----------------------------------------------------------------------

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

    # ---- entities ------------------------------------------------------
    init_joint_pos = {jn: 0.0 for jn in PALM_JOINT_NAMES}
    for jn, v in zip(FINGER_JOINT_NAMES, cfg.initial_finger_ctrl):
        init_joint_pos[jn] = float(v)

    hand_entity = EntityCfg(
        spec_fn=lambda: make_hand_spec(cfg.frozen_scene_xml),
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

    cube_entity = EntityCfg(
        spec_fn=lambda: make_cube_spec(
            cube_size=cfg.object_size,
            mass=cfg.object_mass,
            friction=cfg.object_friction,
        ),
    )

    # ---- actions: finger ctrls (9) + palm residual (6) ----------------
    actions = {
        "finger_ctrl": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(FINGER_JOINT_NAMES),  # joint names; see module docstring
            scale=1.0,
            use_default_offset=True,
        ),
        "palm_ctrl": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(PALM_JOINT_NAMES),
            scale=0.05,                     # small residual on palm
            use_default_offset=True,
        ),
    }

    # ---- observations --------------------------------------------------
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
        "actions": ObservationTermCfg(func=velocity_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    # ---- rewards: contact-shaped (replaces palm-based staged_position_reward,
    # ---- which is the wrong signal for our morphology — palm is static and
    # ---- fingers do the gripping).
    rewards = {
        "contact_mean": RewardTermCfg(
            func=mjlab_terms.fingertip_contact_mean, weight=2.0,
            params={"sensor_name": "fingertip_cube_contact"},
        ),
        "contact_min": RewardTermCfg(
            func=mjlab_terms.fingertip_contact_min, weight=3.0,
            params={"sensor_name": "fingertip_cube_contact"},
        ),
        "lift_height": RewardTermCfg(
            func=mjlab_terms.object_lift_height, weight=80.0,
            params={"object_name": "cube", "target_lift": cfg.lift_target_z_above_init},
        ),
        "object_drop": RewardTermCfg(
            func=mjlab_terms.object_drop_indicator, weight=-12.0,
            params={"object_name": "cube", "drop_threshold": 0.02},
        ),
        "object_xy_drift": RewardTermCfg(
            func=mjlab_terms.object_xy_drift, weight=-3.0,
            params={"object_name": "cube"},
        ),
        "fingertip_to_object": RewardTermCfg(
            func=mjlab_terms.fingertip_to_object_distance, weight=-0.5,
            params={"object_name": "cube"},
        ),
        "action_rate_l2": RewardTermCfg(func=velocity_mdp.action_rate_l2, weight=-0.005),
        "joint_pos_limits": RewardTermCfg(
            func=velocity_mdp.joint_pos_limits, weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }

    # ---- commands ------------------------------------------------------
    commands = {
        "lift_height": manipulation_mdp.LiftingCommandCfg(
            entity_name="cube",
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            difficulty="dynamic",
            object_pose_range=manipulation_mdp.LiftingCommandCfg.ObjectPoseRangeCfg(
                x=(-0.003, 0.003),
                y=(-0.003, 0.003),
                z=(cfg.lift_target_z_above_init, cfg.lift_target_z_above_init + 0.02),
                yaw=(-0.17, 0.17),
            ),
        ),
    }

    # ---- events --------------------------------------------------------
    events = {
        "reset_base": EventTermCfg(
            func=velocity_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
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
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="palm_pose",
            distance=0.4,
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

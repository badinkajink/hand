"""Translation of `MorphoHandEnvCfg` into a live mjlab `ManagerBasedRlEnvCfg`.

Split out of env_cfg.py (CODEBASE_AUDIT.md step 3): env_cfg.py keeps the plain
config dataclass (inspectable/serialisable without mjlab/torch); this module owns
the spec factories and the per-manager builders that `to_mjlab_cfg` assembles.
mjlab is imported lazily inside the functions so importing this module stays
possible on machines without the RL stack.

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

from dataclasses import dataclass
from pathlib import Path

from morphohand.rl.env_cfg import (
    FINGER_JOINT_NAMES,
    PALM_JOINT_NAMES,
    MorphoHandEnvCfg,
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


# ----------------------------------------------------------------------
# Keyframe readers
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Per-manager builders
# ----------------------------------------------------------------------

@dataclass
class _InitContext:
    """Keyframe-derived initial state shared by the entity/action/command builders."""
    palm_joint_pos: list
    palm_default_ctrl: list
    palm_pz_idx: int
    open_finger_qpos: tuple
    finger_init_pos: tuple
    init_joint_pos: dict
    obj_init_xyz: tuple
    obj_init_quat: tuple


def _init_context(cfg: MorphoHandEnvCfg) -> _InitContext:
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
    # the reset pose (finger_init_pos) and the LerpFinger interpolation start.
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

    # Object spawn: the keyframe qpos of the object body (where CEM tuned it).
    # Quat is required for flat-laying cylinders whose source body has a
    # non-identity rest quat — without it the extracted body resets to identity
    # and the cylinder spawns standing up.
    obj_init_xyz, obj_init_quat = _read_keyframe_object_pose(
        cfg.frozen_scene_xml, cfg.keyframe_name, cfg.object_body_name
    )
    if cfg.skip_lift_phase:
        obj_init_xyz = (obj_init_xyz[0], obj_init_xyz[1],
                        obj_init_xyz[2] + float(cfg.lift_delta_z)
                        + float(cfg.skip_lift_drop_offset))
    return _InitContext(palm_joint_pos, palm_default_ctrl, palm_pz_idx,
                        open_finger_qpos, finger_init_pos, init_joint_pos,
                        obj_init_xyz, obj_init_quat)


def _build_entities(cfg: MorphoHandEnvCfg, ctx: _InitContext):
    from mjlab.actuator import XmlPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

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
            joint_pos=ctx.init_joint_pos,
            joint_vel={".*": 0.0},
        ),
    )

    # Extract the object body from the frozen scene XML (works for any
    # object: cube, prism, screwdriver, ...). init_state.pos/rot default to
    # the keyframe qpos so the object spawns where CEM tuned it.
    # `reset_cube` event in the events dict writes default_root_state
    # to sim each reset, ensuring the init pose actually applies.
    cube_entity = EntityCfg(
        spec_fn=lambda: make_object_spec_from_frozen(
            cfg.frozen_scene_xml, cfg.object_body_name
        ),
        init_state=EntityCfg.InitialStateCfg(pos=ctx.obj_init_xyz, rot=ctx.obj_init_quat),
    )
    return hand_entity, cube_entity


def _build_actions(cfg: MorphoHandEnvCfg, ctx: _InitContext) -> dict:
    # Linear-interp finger closing + scripted palm lift: policy outputs 9 dims
    # as residuals on top of a time-varying setpoint that lerps from
    # `open_finger_qpos` toward `finger_default_ctrl` over
    # `finger_close_sim_steps`. action=0 produces a clean open->close motion.
    # After the lerp finishes there is a hold-at-grip phase (until
    # `settle_steps`) so the cube has time to seat, then the palm scripted
    # action ramps palm_pz by `lift_delta_z` over `lift_ramp_steps`.
    from morphohand.rl.actions import LerpFingerActionCfg, ScriptedPalmActionCfg

    if cfg.skip_lift_phase:
        # Fingers start AT the grip pose — no closing motion. lerp(start, target, alpha)
        # is constant = finger_default_ctrl.
        finger_start = tuple(float(v) for v in cfg.finger_default_ctrl)
        # Palm holds at lifted height (palm_default_ctrl already offset in ctx).
        scripted_settle = 0
        scripted_ramp = 1
        scripted_lift_delta = 0.0
    else:
        finger_start = tuple(float(v) for v in ctx.open_finger_qpos)
        scripted_settle = cfg.settle_steps
        scripted_ramp = cfg.lift_ramp_steps
        scripted_lift_delta = cfg.lift_delta_z
    return {
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
            palm_default_ctrl=tuple(ctx.palm_default_ctrl),
            palm_pz_index=ctx.palm_pz_idx,
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


def _build_observations(cfg: MorphoHandEnvCfg) -> dict:
    from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.manipulation import mdp as manipulation_mdp
    from mjlab.tasks.velocity import mdp as velocity_mdp
    from mjlab.utils.noise import UniformNoiseCfg as Unoise
    from morphohand.rl import mjlab_terms

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
    return {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }


def _build_rewards(cfg: MorphoHandEnvCfg) -> dict:
    # Contact-shaped rewards (replaces palm-based staged_position_reward, which
    # is the wrong signal for our morphology — palm is static and fingers do
    # the gripping).
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.velocity import mdp as velocity_mdp
    from morphohand.rl import mjlab_terms
    from morphohand.rl.reward import DEFAULT_REWARD_WEIGHTS

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
    # ---- object-relative fingertip IMITATION prior (morphology-transferable) ----
    if cfg.imitation_ref_npz and cfg.imitation_weight != 0.0:
        from morphohand.rl.imitation import track_fingertip_obj
        rewards["track_fingertip_obj"] = RewardTermCfg(
            func=track_fingertip_obj,
            weight=float(cfg.imitation_weight) * task_scale,
            params={
                "ref_path": str(cfg.imitation_ref_npz),
                "alpha": float(cfg.imitation_alpha),
                "object_name": "cube",
                "reorient_start_step": int(cfg.reorient_start_step),
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
    return rewards


def _spawn_jitter(cfg: MorphoHandEnvCfg) -> tuple[float, float, float, float, float]:
    """(x_jitter, y_jitter, yaw_jitter, x_center, y_center) with the
    `cube_spawn_xy_jitter` back-compat inheritance applied."""
    x_j = float(cfg.cube_spawn_x_jitter if cfg.cube_spawn_x_jitter is not None
                else cfg.cube_spawn_xy_jitter)
    y_j = float(cfg.cube_spawn_y_jitter if cfg.cube_spawn_y_jitter is not None
                else cfg.cube_spawn_xy_jitter)
    return x_j, y_j, float(cfg.cube_spawn_yaw_jitter), \
        float(cfg.cube_spawn_x_center), float(cfg.cube_spawn_y_center)


def _build_commands(cfg: MorphoHandEnvCfg, ctx: _InitContext) -> dict:
    # `object_pose_range` is the cube SPAWN range — `LiftingCommand`
    # writes the cube to a value sampled from this range each reset.
    # Separate x and y because the hand's reachable envelope is
    # asymmetric (see /tmp/sweep_reachable2.py).
    from mjlab.tasks.manipulation import mdp as manipulation_mdp
    from morphohand.rl.lifting_command import LiftingCommandWithBaseQuatCfg

    x_j, y_j, yaw_j, x_c, y_c = _spawn_jitter(cfg)
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
    return {
        "lift_height": LiftingCommandWithBaseQuatCfg(
            entity_name="cube",
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            difficulty="dynamic",
            object_pose_range=(None if cfg.handoff_state_bank else
                manipulation_mdp.LiftingCommandCfg.ObjectPoseRangeCfg(
                    x=(x_c - init_xj, x_c + init_xj),
                    y=(y_c - init_yj, y_c + init_yj),
                    z=(ctx.obj_init_xyz[2] - init_zj, ctx.obj_init_xyz[2] + init_zj),
                    yaw=(-init_yawj, init_yawj),
                )),
            base_quat=ctx.obj_init_quat,  # preserves flat orientation under yaw DR
            spawn_tilt_range=(-init_tilt, init_tilt),
        ),
    }


def _build_events(cfg: MorphoHandEnvCfg) -> dict:
    # `reset_base` (default asset="robot") writes the mocap pose for the
    # fixed-base robot — this is also what applies env_origin offsets so
    # every parallel env's hand spawns in its own grid cell. Removing it
    # collapses all 1024 hands onto the same world coords. Keep it.
    #
    # `reset_cube` is what we add: the cube is a floating-base freejoint
    # entity and needs its init_state.pos written explicitly on each
    # reset, otherwise mjlab leaves the cube wherever the scene compile
    # left it (observed at z=0.063 instead of the requested z=0.02).
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.velocity import mdp as velocity_mdp
    from morphohand.rl import mjlab_terms

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
    # contact-compliance DR: resample every geom's solimp per env on each reset
    # so the policy can't overfit one contact stiffness (the measured failure —
    # docs/rl/compliance_dr_plan.md). Eval envs leave this off: they pin the
    # stiffness via the frozen-scene xml.
    if cfg.compliance_dr:
        events["randomize_compliance"] = EventTermCfg(
            func=mjlab_terms.randomize_geom_solimp,
            mode="reset",
            params={"dmin_range": tuple(cfg.compliance_dr_dmin),
                    "dmax_range": tuple(cfg.compliance_dr_dmax)},
        )
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
    return events


def _build_terminations(cfg: MorphoHandEnvCfg) -> dict:
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.tasks.velocity import mdp as velocity_mdp
    from morphohand.rl import mjlab_terms

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
    return terminations


def _build_sensors(cfg: MorphoHandEnvCfg) -> tuple:
    from mjlab.sensor import ContactMatch, ContactSensorCfg

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
    return (fingertip_cube_sensor, palm_cube_sensor)


def _build_curriculum(cfg: MorphoHandEnvCfg, rewards: dict, ctx: _InitContext) -> dict:
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from morphohand.rl import mjlab_terms

    x_j, y_j, yaw_j, x_c, y_c = _spawn_jitter(cfg)
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
                z_center=float(ctx.obj_init_xyz[2]),
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
    task_scale = 0.0 if cfg.reward_mode == "tracking_only" else 1.0
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
    # anneal the imitation weight down ("learn the demo motion first, then let the task refine")
    if cfg.imitation_curriculum_iters > 0 and "track_fingertip_obj" in rewards:
        curriculum["imitation_anneal"] = CurriculumTermCfg(
            func=mjlab_terms.anneal_smoothness_weights,   # generic reward-weight annealer
            params=dict(
                term_names=("track_fingertip_obj",),
                base_weights=(float(cfg.imitation_weight) * task_scale,),
                final_weights=(float(cfg.imitation_weight_final) * task_scale,),
                start_iter=0,
                anneal_iters=int(cfg.imitation_curriculum_iters),
            ),
        )
    return curriculum


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

def to_mjlab_cfg(cfg: MorphoHandEnvCfg):
    """Build a live mjlab `ManagerBasedRlEnvCfg` from a `MorphoHandEnvCfg`.

    Imports mjlab lazily so this module is importable on machines without GPU.
    """
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.viewer import ViewerConfig

    ctx = _init_context(cfg)
    hand_entity, cube_entity = _build_entities(cfg, ctx)
    rewards = _build_rewards(cfg)

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=cfg.num_envs,
            env_spacing=cfg.env_spacing,
            entities={"robot": hand_entity, "cube": cube_entity},
            sensors=_build_sensors(cfg),
        ),
        observations=_build_observations(cfg),
        actions=_build_actions(cfg, ctx),
        commands=_build_commands(cfg, ctx),
        events=_build_events(cfg),
        rewards=rewards,
        terminations=_build_terminations(cfg),
        curriculum=_build_curriculum(cfg, rewards, ctx),
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

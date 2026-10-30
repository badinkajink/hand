"""real_v3 hand + a flat-lying cylinder, for CPU MuJoCo grasp training.

Built programmatically from `assets/mjcf/real_v3/real_hand.xml` (the STEP-derived hand with
MorphIt spheres + box pads) so the hand has exactly one source of truth:

  * the gantry XY slides are BAKED to a layout (moved into the mount body positions and the
    slide joints deleted) -- the steppers set the layout before a grasp, they do not take part
    in it. The zero-travel `len` shims go too. What is left is 9 finger hinges + 1 palm slide.
  * `palm_z`: a vertical slide on the palm, position-controlled. The lift is the arm's job and
    is scripted, as in the GPU stack (`actions.py` ramps palm_pz); the policy never sees it.
  * the floor collides (the real_v3 hand file leaves it visual-only).
  * `object`: a free cylinder lying along palm-frame y, centred between thumb and pair. Its
    radius / mass / friction are changed per env at runtime on that env's own MjModel copy
    (`set_object`), so one compiled model serves every size.
  * `<f>_pad_site` (emitted by the generator with the pad's sphere grid): the centre of each
    pad's CONTACT face, z axis = outward normal, size = the face's in-plane half-extents --
    the fingertip point the reward and the IK both reason about.

Palm frame: thumb at -x, index at +y, middle at -y, fingers hang toward -z (see
build_real_v3_from_step.py). The object axis is y, so the pinch is along x and the reorient
is a rotation about x.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HAND_XML = ROOT / "assets" / "mjcf" / "real_v3" / "real_hand.xml"
FINGERS = ("thumb", "index", "middle")
JOINTS = tuple(f"{f}_{j}" for f in FINGERS for j in ("yaw", "mcp", "pip"))
CYL_HALF_LEN = 0.05            # 100 mm long cylinders


@dataclass
class Layout:
    """Gantry slide offsets (m, palm frame) from each finger's workspace centre."""
    slides: dict = field(default_factory=lambda: {f: (0.0, 0.0) for f in FINGERS})


# D1 = sv1_w6689, the SR^2 paper's top-ranked design and the gantry the manta-hand proposal
# fixes (hand-main scripts/real_v1_transfer_ranking.py names it). Its `vector_m` in
# docs/experiments/20260831-real_v1-sobol8192/hardware_manifest.json is the slide offset of
# each mount from its workspace centre -- the quantity the steppers realise -- so it carries
# over to real_v3's (CAD-derived) workspace centres unchanged.
D1 = Layout(slides={"thumb": (-0.00432052, -0.033743434),
                    "index": (-0.012068113, -0.012924523),
                    "middle": (-0.010976371, 0.011970241)})

# Feetech SCS0009: stall ~2.3 kg*cm = 0.23 N*m at 6 V. real_v3's file still carries real_v1's
# kp 30 / +-10 N*m, ~40x the real servo. The grasp is judged on whether it survives a reorient,
# which is exactly where an over-strong finger would hide a bad grasp.
SERVO_TORQUE = 0.23      # N*m
SERVO_KP = 1.5           # N*m/rad: saturates at ~9 deg of tracking error, like the real servo
SERVO_KV = 0.02
# Joint damping caps the free speed at torque/damping. real_v3 inherits 0.5 N*m*s/rad, which
# at 0.23 N*m is 26 deg/s; the SCS0009 runs ~0.1 s/60 deg (~10 rad/s) unloaded.
SERVO_DAMPING = 0.02     # N*m*s/rad -> ~11 rad/s free speed

# Contact friction (2026-10-30). MuJoCo combines two geoms' friction by taking the max unless their
# `priority` differs, in which case the higher-priority geom's value is used. The real_v3 pad
# spheres carry friction 2.4, so before this fix every pad-object contact ran at 2.4 whatever the
# object's own value. Now the object (priority 1) sets pad-object friction, and the floor
# (priority 2) sets object-table friction.
# Test objects are PLA cylinders wrapped in foam grip tape (compresses ~0.2 mm under a firm press,
# so default contact softness covers it). 2.0 is nominal, not measured; 2.4 was the value the
# earlier real_v1 scenes carried. TPU 85A on bare PLA would be ~0.7 and fails 20-25 mm cylinders.
MU_PAD_OBJ = 2.0
MU_TABLE = 0.4           # PLA on the bench top, nominal (assumed)


# Real-hand software joint limits (deg), from the canonical driver on the CB1
# (~/hand/src/morphohand/driver/manta/host/manta_hand/servos.py FINGER_JOINTS, read 2026-10-30).
# The sim uses these instead of the CAD ranges so no plan, residual or judge target can ask the
# real hand for an angle its driver would refuse.
REAL_LIMITS_DEG = {
    "thumb_yaw": (-85.0, 85.0), "thumb_mcp": (-12.3, 89.06), "thumb_pip": (-18.0, 86.72),
    "index_yaw": (-85.0, 85.0), "index_mcp": (-15.0, 64.75), "index_pip": (-2.93, 92.0),
    "middle_yaw": (-85.0, 85.0), "middle_mcp": (-15.0, 69.73), "middle_pip": (-16.11, 77.05),
}


def build(layout: Layout | None = None, palm_z: float = 0.15, servo_torque: float = SERVO_TORQUE,
          servo_kp: float = SERVO_KP, post: bool = False) -> mujoco.MjModel:
    layout = layout or D1
    spec = mujoco.MjSpec.from_file(str(HAND_XML))
    spec.option.timestep = 0.002

    for key in list(spec.keys):
        spec.delete(key)
    for f in FINGERS:
        mount = spec.body(f"{f}_mount")
        dx, dy = layout.slides[f]
        mount.pos = [mount.pos[0] + dx, mount.pos[1] + dy, mount.pos[2]]
        for j in (f"{f}_x", f"{f}_y", f"{f}_len"):
            spec.delete(spec.joint(j))

    for j in JOINTS:
        spec.joint(j).damping = SERVO_DAMPING
        a = spec.actuator(f"a_{j}")
        a.set_to_position(kp=servo_kp, kv=SERVO_KV)
        a.forcerange, a.forcelimited = [-servo_torque, servo_torque], mujoco.mjtLimited.mjLIMITED_TRUE

    floor = spec.geom("floor")
    floor.contype, floor.conaffinity = 1, 1
    floor.friction = [MU_TABLE, 0.005, 0.0001]
    floor.priority = 2

    palm = spec.body("palm_pose")
    palm.pos = [0, 0, palm_z]
    palm.explicitinertial = True            # the base-tray visual mesh would make it ~1 kg of PLA
    palm.mass, palm.inertia, palm.ipos = 0.5, [1e-3, 1e-3, 1e-3], [0, 0, 0]
    pj = palm.add_joint(name="palm_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                        range=[-0.1, 0.25], damping=50.0)
    act = spec.add_actuator(name="a_palm_z", target="palm_z", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=4000.0, kv=0.0)
    act.ctrlrange, act.ctrllimited = [-0.1, 0.25], mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange, act.forcelimited = [-400, 400], mujoco.mjtLimited.mjLIMITED_TRUE
    del pj

    obj = spec.worldbody.add_body(name="object", pos=[0, 0, 0.02], quat=[0.70710678, 0.70710678, 0, 0])
    obj.add_freejoint(name="object_free")
    obj.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0125, CYL_HALF_LEN, 0],
                 rgba=[0.75, 0.75, 0.78, 1], friction=[MU_PAD_OBJ, 0.005, 0.0001], density=1000,
                 priority=1)

    if post:
        # the bench's support post (real_v1 scene_mutate.set_object_platform): a thin static
        # cylinder; size and position are set per rollout on the model copy
        spec.worldbody.add_geom(name="tool_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.008, 0.05, 0], pos=[0, 0, 0.05], rgba=[0.45, 0.45, 0.5, 1],
                                friction=[1.0, 0.1, 0.01])
    m = spec.compile()
    for j, (lo, hi) in REAL_LIMITS_DEG.items():
        m.jnt_range[m.joint(j).id] = np.radians([lo, hi])
    return m


def set_object(model: mujoco.MjModel, radius: float, mass: float, friction: float = MU_PAD_OBJ) -> None:
    """Resize / reweigh the cylinder on THIS model copy (per-env domain randomisation)."""
    g = model.geom("object_geom").id
    b = model.body("object").id
    model.geom_size[g, 0] = radius
    model.geom_rbound[g] = float(np.hypot(radius, CYL_HALF_LEN))
    # the collision pre-filter uses the compile-time local AABB (centre, half-extents); without
    # this a cylinder grown past its compiled radius is never tested against the fingers
    model.geom_aabb[g] = [0.0, 0.0, 0.0, radius, radius, CYL_HALF_LEN]
    # ... and so does the body's bounding-volume hierarchy (midphase). Left at the compiled
    # radius it silently culled every finger contact on cylinders larger than that (found
    # 2026-10-29: 40 mm cylinder, pad spheres 3 mm inside it, zero contacts).
    for k in range(model.body_bvhnum[b]):
        model.bvh_aabb[model.body_bvhadr[b] + k] = [0.0, 0.0, 0.0, radius, radius, CYL_HALF_LEN]
    model.geom_friction[g, 0] = friction      # pad-object sliding friction (object has priority)
    L = 2 * CYL_HALF_LEN
    i_axis = 0.5 * mass * radius**2                     # about the cylinder axis (local z)
    i_perp = mass * (3 * radius**2 + L**2) / 12
    model.body_mass[b] = mass
    model.body_inertia[b] = [i_perp, i_perp, i_axis]


def object_rest_pose(model: mujoco.MjModel, radius: float, x: float | None = None) -> np.ndarray:
    """Freejoint qpos (7) for the cylinder lying along y on the table, centred under the pinch."""
    if x is None:
        tx = model.body("thumb_mount").pos[0]
        px = 0.5 * (model.body("index_mount").pos[0] + model.body("middle_mount").pos[0])
        x = 0.5 * (tx + px)
    return np.array([x, 0.0, radius, 0.70710678, 0.70710678, 0.0, 0.0])


def indices(model: mujoco.MjModel) -> dict:
    """Address book for the env: qpos/qvel/ctrl slots, body/site/geom ids."""
    jq = np.array([model.jnt_qposadr[model.joint(j).id] for j in JOINTS])
    jv = np.array([model.jnt_dofadr[model.joint(j).id] for j in JOINTS])
    act = np.array([model.actuator(f"a_{j}").id for j in JOINTS])
    fj = model.joint("object_free").id
    finger_bodies = {f: {model.body(f"{f}_{b}").id for b in ("yaw_frame", "mcp_frame", "pip_frame", "tip")}
                     for f in FINGERS}
    return dict(
        joint_qpos=jq, joint_dof=jv, joint_act=act,
        palm_qpos=model.jnt_qposadr[model.joint("palm_z").id], palm_act=model.actuator("a_palm_z").id,
        obj_qpos=model.jnt_qposadr[fj], obj_dof=model.jnt_dofadr[fj], obj_body=model.body("object").id,
        obj_geom=model.geom("object_geom").id, floor_geom=model.geom("floor").id,
        pad_site=np.array([model.site(f"{f}_pad_site").id for f in FINGERS]),
        pad_geoms={f: np.array([g for g in range(model.ngeom) if model.geom(g).name.startswith(f"{f}_pad_s")])
                   for f in FINGERS},
        finger_bodies=finger_bodies, palm_body=model.body("palm_pose").id,
        jnt_range=model.jnt_range[[model.joint(j).id for j in JOINTS]].copy(),
    )

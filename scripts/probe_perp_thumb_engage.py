"""Scripted viability probe: can the opposed-pair hand be made to REORIENT ON THREE FINGERS?

The standing result on this topology (r4, r6, r7) is that the shaft turns by sliding through a
two-point pinch — rotation and retention are the same degree of freedom — and the thumb never
touches. Every previous attempt to fix that gave the thumb a reward and let PPO find the motion.
None found one. This asks the cheaper question first: with the pair pinching NEAR the shaft's
centre of mass and the thumb parked to the side, is there a hand-authored thumb trajectory that
engages the shaft after it has swung, and does engaging it help or eject?

Phases (all open-loop position targets; the pair and the thumb are both posed by fingertip IK,
never by joint offsets — on this topology "more flexion" swings a facing tip AWAY from the
shaft):

  settle -> close (pair IK onto the shaft flanks) -> lift (palm_pz ramp) -> hang (gravity
  swings the shaft) -> engage (thumb IK onto the shaft surface) -> press (hold and watch)

The engage target is stated in the OBJECT's own frame at engage onset — "put the thumb pad on
the shaft surface, s mm along the axis from the centre of mass, on the side facing the thumb" —
because by then the shaft has moved and rotated and any world-frame target authored beforehand
is aimed at where it used to be. `--thumb-track` re-solves it every step so the thumb follows a
shaft that is still moving.

The reorient objective is SIGNED: cos is the object body +Z (the shaft's long axis, world +X at
spawn) against world +Z, so +1 means the +X end came up.

Run:
  MUJOCO_GL=egl uv run python scripts/probe_perp_thumb_engage.py \
    --scene assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml \
    --start-keyframe open_manual --out-dir docs/experiments/20260819-perp_thumb_engage/lift_only \
    --no-engage
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mj_snap import VIEWS, _label, tile  # noqa: E402
from morphohand.tools.keyframe_ik import FINGERS, TIPS, ik_finger  # noqa: E402

OBJ = "screwdriver_medium"
# A finger loads the shaft through its distal links, not only its pad; attributing tip-body
# contacts alone reports 0 N while the hand is visibly gripping.
FINGER_BODIES = {"thumb": ("thumb_tip", "thumb_pip_frame", "thumb_len_frame"),
                 "index": ("index_tip", "index_pip_frame", "index_len_frame"),
                 "middle": ("middle_tip", "middle_pip_frame", "middle_len_frame")}
ACT = {"thumb": ("a_thumb_yaw", "a_thumb_mcp", "a_thumb_pip"),
       "index": ("a_index_yaw", "a_index_mcp", "a_index_pip"),
       "middle": ("a_middle_yaw", "a_middle_mcp", "a_middle_pip")}


def aid(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def obj_frame(model, data):
    """(centre, unit long axis, half-length) of the shaft in world coordinates."""
    R = data.body(OBJ).xmat.reshape(3, 3)
    gid = model.body(OBJ).geomadr[0] if hasattr(model.body(OBJ), "geomadr") else None
    half = float(model.geom_size[gid][1]) if gid is not None else 0.05
    return data.body(OBJ).xpos.copy(), R[:, 2].copy(), half


def obj_radius(model) -> float:
    return float(model.geom_size[model.body(OBJ).geomadr[0]][0])


def finger_forces(model, data) -> dict[str, float]:
    body_to_finger = {b: f for f, bodies in FINGER_BODIES.items() for b in bodies}
    out = {f: 0.0 for f in FINGER_BODIES}
    obj_id = model.body(OBJ).id
    for i in range(data.ncon):
        con = data.contact[i]
        b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
        if obj_id not in (b1, b2):
            continue
        other = b2 if b1 == obj_id else b1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        finger = body_to_finger.get(name)
        if finger is not None:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            out[finger] += float(np.linalg.norm(force[:3]))
    return out


def on_floor(model, data) -> bool:
    obj_id, world_id = model.body(OBJ).id, 0
    for i in range(data.ncon):
        con = data.contact[i]
        pair = {model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]}
        if pair == {obj_id, world_id}:
            return True
    return False


def surface_point(model, data, s: float, direction: np.ndarray, depth: float) -> np.ndarray:
    """A point on (or `depth` inside) the shaft surface: `s` metres along the axis from the
    centre of mass, offset radially along the component of `direction` normal to the axis."""
    centre, axis, _ = obj_frame(model, data)
    radial = direction - np.dot(direction, axis) * axis
    n = np.linalg.norm(radial)
    radial = radial / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    return centre + s * axis + (obj_radius(model) - depth) * radial


def chuck_target(model, data, s: float, azimuth: float, depth: float) -> np.ndarray:
    """A contact point on the shaft at axial coordinate `s`, at `azimuth` degrees around it.

    Azimuth is measured in the plane normal to the shaft axis, from the direction the thumb
    approaches from reversed (+X side = 0 deg) toward +Y. So 180 deg is the thumb's own flank
    and +-60 deg are the two flanks the opposed pair has to move ONTO for the three contact
    normals to positively span the plane. With the pair left where it naturally ends up (+-90
    deg) the two normals are collinear and nothing opposes the thumb's push but friction.
    """
    centre, axis, _ = obj_frame(model, data)
    u = np.array([1.0, 0.0, 0.0]) - float(np.dot([1.0, 0.0, 0.0], axis)) * axis
    n = np.linalg.norm(u)
    u = u / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
    v = np.cross(axis, u)
    phi = np.deg2rad(azimuth)
    return centre + s * axis + (obj_radius(model) - depth) * (np.cos(phi) * u + np.sin(phi) * v)


def axial_s(model, data, tip: str) -> float:
    """Where along the shaft axis a fingertip currently sits (m from the COM)."""
    centre, axis, _ = obj_frame(model, data)
    return float(np.dot(data.body(tip).xpos - centre, axis))


def axis_gap(model, data, tip: str) -> float:
    """Signed clearance (m) from a fingertip's surface to the shaft's surface, measured to the
    axis segment — negative means interpenetrating."""
    centre, axis, half = obj_frame(model, data)
    p = data.body(tip).xpos - centre
    t = float(np.clip(np.dot(p, axis), -half, half))
    return float(np.linalg.norm(p - t * axis) - obj_radius(model) - 0.006)


def palm_frame(data, world_pt: np.ndarray) -> np.ndarray:
    R = data.body("palm_pose").xmat.reshape(3, 3)
    return R.T @ (world_pt - data.body("palm_pose").xpos)


_SCRATCH: dict[int, mujoco.MjData] = {}


def solve_ctrl(model, data, finger: str, target: np.ndarray) -> tuple[np.ndarray, float]:
    """IK `finger` to `target` on a SCRATCH state, returning its three ctrl values.

    The IK writes qpos, so it runs on a copy: solving on the live state would teleport the hand
    mid-rollout instead of commanding the position actuators toward the pose.
    """
    scratch = _SCRATCH.get(id(model))
    if scratch is None:
        scratch = _SCRATCH[id(model)] = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)
    err = ik_finger(model, scratch, finger, target)
    vals = np.array([float(scratch.qpos[model.jnt_qposadr[model.joint(j).id]])
                     for j in FINGERS[finger]])
    return vals, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--start-keyframe", default="open_manual")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--label", default=None)

    ap.add_argument("--pinch-x", type=float, default=None,
                    help="world x of the pair's pinch (default: keep each tip's own x)")
    ap.add_argument("--pinch-depth", type=float, default=0.003,
                    help="how far INSIDE the shaft surface the pair tips are commanded (m)")
    ap.add_argument("--lift", type=float, default=0.14)
    ap.add_argument("--object-dx", type=float, default=0.0,
                    help="shift the shaft's spawn along x (m) -- the perturbation the pinch is "
                         "most sensitive to, since nothing constrains the shaft along its axis")
    ap.add_argument("--object-dy", type=float, default=0.0)

    ap.add_argument("--no-engage", action="store_true", help="lift and hang only; thumb parked")
    ap.add_argument("--thumb-s", type=float, default=-0.030,
                    help="engage point along the shaft axis from its COM (m); negative = the "
                         "end that swings DOWN, positive = the end that comes up")
    ap.add_argument("--thumb-dir", default="-1 0 0",
                    help="world direction the thumb approaches from (radial side of the shaft)")
    ap.add_argument("--thumb-depth", type=float, default=0.004)
    ap.add_argument("--thumb-track", action="store_true",
                    help="re-solve the engage target every step so the thumb follows the shaft")
    ap.add_argument("--chuck-tilt", type=float, default=0.0,
                    help="degrees to swing the PAIR's contacts off the +-90 deg flanks toward "
                         "the +x side during engage, so their normals can react the thumb's "
                         "push. 0 leaves the pair where the swing put it (two collinear "
                         "normals; the thumb then ejects the shaft).")
    ap.add_argument("--chuck-depth", type=float, default=0.003)
    ap.add_argument("--chuck-x", type=float, default=0.0,
                    help="metres to move BOTH pair pads toward +x of the shaft axis while "
                         "holding their +-y separation, so they push back along -x against the "
                         "thumb without loosening the pinch. This is the same intent as "
                         "--chuck-tilt but keeps the grasp CENTRED: tilting around the axis "
                         "shrinks the y half-separation to cos(tilt)*r, so the pads slide off "
                         "toward whichever flank the shaft drifts to.")
    ap.add_argument("--chuck-frame", choices=("object", "palm"), default="object",
                    help="which frame the PAIR's hold targets are stated in. 'object' re-solves "
                         "them from the shaft every retrack -- so the pads CHASE a shaft that is "
                         "sliding along the pinch axis and can never push it back; the trace "
                         "shows y walking out 40 mm with the far pad unloading to 4 N. 'palm' "
                         "pins them in the hand at engage time, so a shaft that drifts runs "
                         "into a pad that stayed put.")
    ap.add_argument("--pair-yaw-bias", type=float, default=0.0,
                    help="radians of YAW preload added to each pair finger during the hold, "
                         "signed so both pads push back along -x against the thumb. The pinch "
                         "clamps along y and cannot otherwise react the thumb at all; without "
                         "this the shaft slides toward whichever pad it is closest to and the "
                         "other unloads.")
    ap.add_argument("--track-every", type=int, default=0,
                    help="re-solve every finger's contact target every N steps through engage "
                         "and press. A target solved once goes stale: the shaft creeps a "
                         "fraction of a millimetre, the contacts unload, and it falls straight "
                         "through a cage that still looks closed.")

    ap.add_argument("--settle-steps", type=int, default=200)
    ap.add_argument("--close-steps", type=int, default=300)
    ap.add_argument("--lift-steps", type=int, default=400)
    ap.add_argument("--hang-steps", type=int, default=600)
    ap.add_argument("--engage-at-cos", type=float, default=0.0,
                    help="engage the thumb the first time the shaft reaches this alignment, "
                         "instead of after a fixed --hang-steps (0 = use the step count)")
    ap.add_argument("--engage-steps", type=int, default=300)
    ap.add_argument("--press-steps", type=int, default=600)

    ap.add_argument("--views", default="side,front")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--video-stride", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-chuck-pose", type=Path, default=None,
                    help="write the three fingertips expressed in the OBJECT's frame: the whole "
                         "trajectory, plus the mean over the hold window. The hold is a static "
                         "object-relative configuration, so that mean is the thing a policy has "
                         "to reproduce -- and unlike a time-indexed reference it does not care "
                         "that the learned rotation takes 4x longer than the gravity swing.")
    ap.add_argument("--save-npz", type=Path, default=None,
                    help="write the rollout as a reference trajectory (qpos/qvel/cube_z/"
                         "contacts/best_finger_ctrl) in the layout rl_train_cube.py's "
                         "--morphology-run expects, so the scripted maneuver can seed training")
    args = ap.parse_args()

    # The scene's offscreen framebuffer defaults to 640x480; a larger --width/--height has to
    # be declared before compile or the Renderer refuses it.
    spec = mujoco.MjSpec.from_file(str(args.scene))
    spec.visual.global_.offwidth = max(args.width, 640)
    spec.visual.global_.offheight = max(args.height, 480)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.start_keyframe).id)
    data.qpos[0] += args.object_dx
    data.qpos[1] += args.object_dy
    mujoco.mj_forward(model, data)

    ids = {name: aid(model, name) for names in ACT.values() for name in names}
    ids["a_palm_pz"] = aid(model, "a_palm_pz")
    start_ctrl = data.ctrl.copy()
    palm_pz0 = float(start_ctrl[ids["a_palm_pz"]])

    # --- the close pose: each pair tip driven onto its own flank of the shaft ---------------
    radius = obj_radius(model)
    centre0, axis0, _ = obj_frame(model, data)
    closed_ctrl = {}
    for finger, side in (("index", +1.0), ("middle", -1.0)):
        tip_now = data.body(TIPS[finger]).xpos
        s = float(np.dot(tip_now - centre0, axis0)) if args.pinch_x is None else \
            float(args.pinch_x - centre0[0])
        target = centre0 + s * axis0 + np.array([0.0, side * (radius - args.pinch_depth), 0.0])
        vals, err = solve_ctrl(model, data, finger, target)
        closed_ctrl[finger] = vals
        print(f"[close] {finger:6s} target {np.round(target, 4)}  IK residual {err*1000:5.2f} mm"
              f"  -> yaw/mcp/pip {np.round(vals, 3)}")

    p = [args.settle_steps, args.close_steps, args.lift_steps,
         args.hang_steps, args.engage_steps, args.press_steps]
    if args.no_engage:
        p[4] = 0
        p[5] = 0
    bounds = np.cumsum(p)
    names = ["settle", "close", "lift", "hang", "engage", "press"]
    total = int(bounds[-1])

    def phase_of(step: int) -> tuple[str, float]:
        for k, b in enumerate(bounds):
            if step < b:
                lo = 0 if k == 0 else bounds[k - 1]
                return names[k], min(1.0, (step - lo) / max(1, p[k]))
        return names[-1], 1.0

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    cams = []
    for v in views:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth, cam.elevation, cam.distance = VIEWS[v][0], VIEWS[v][1], 0.42
        cams.append(cam)

    grab_at = set(np.linspace(0, total - 1, args.frames).astype(int).tolist())
    strip: list[Image.Image] = []
    video: list[np.ndarray] = []
    trace: list[dict] = []
    engage_ctrl: dict[str, np.ndarray] = {}
    chuck_ctrl: dict[str, np.ndarray] = {}
    chuck_s: dict[str, float] = {}
    chuck_palm_tgt: dict[str, np.ndarray] = {}
    tips_obj: list[np.ndarray] = []
    tips_phase: list[str] = []
    ref_qpos: list[np.ndarray] = []
    ref_qvel: list[np.ndarray] = []
    ref_z: list[float] = []
    ref_con: list[float] = []
    engage_target: np.ndarray | None = None
    thumb_ctrl_at_engage: np.ndarray | None = None

    gated = args.engage_at_cos <= 0.0

    # Which way each pair finger has to YAW to push back against the thumb. On this topology the
    # yaw axis IS the finger's pointing axis, so yawing rolls the mcp/pip swing plane and, with
    # the finger flexed, carries the tip along x — the one axis the thumb loads and the pinch
    # (clamping along y) otherwise reacts only through friction. Measured off the model rather
    # than reasoned from the mount quats, which differ in sign between index and middle.
    yaw_sign = {}
    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos
    for finger in ("index", "middle"):
        adr = model.jnt_qposadr[model.joint(FINGERS[finger][0]).id]
        mujoco.mj_forward(model, probe)
        x0 = float(probe.body(TIPS[finger]).xpos[0])
        probe.qpos[adr] += 0.05
        mujoco.mj_forward(model, probe)
        dx = float(probe.body(TIPS[finger]).xpos[0]) - x0
        probe.qpos[adr] -= 0.05
        # the thumb pushes +x, so preload the pads toward -x
        yaw_sign[finger] = -1.0 if dx > 0 else 1.0
        print(f"[yaw] {finger:6s} d(tip x)/d(yaw) = {dx*1000:+.2f} mm/0.05 rad  "
              f"-> bias sign {yaw_sign[finger]:+.0f}")

    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for step in range(total):
            phase, t = phase_of(step)

            # Gate the engage on TASK PROGRESS, not on a step count. The swing's timing moves
            # with the pinch offset by hundreds of steps, so a fixed onset engages a shaft that
            # is still turning on one design and one that has already swung back on the next --
            # which reads as the maneuver being fragile when it is the trigger that is wrong.
            if not gated and phase == "hang" and float(data.body(OBJ).xmat.reshape(3, 3)[2, 2]) \
                    >= args.engage_at_cos:
                bounds[3:] -= (bounds[3] - (step + 1))
                gated = True
                print(f"[gate] cos >= {args.engage_at_cos} at step {step}; engaging")
                phase, t = phase_of(step)

            if phase in ("close", "lift", "hang"):
                tc = 1.0 if phase != "close" else t
                for finger in ("index", "middle"):
                    for k, name in enumerate(ACT[finger]):
                        data.ctrl[ids[name]] = (start_ctrl[ids[name]] * (1 - tc)
                                                + closed_ctrl[finger][k] * tc)
            if phase in ("lift", "hang", "engage", "press"):
                tl = 1.0 if phase != "lift" else t
                # RELATIVE to where the keyframe parks the palm. A shorter finger has to start
                # lower to reach the same grasp, and an absolute lift target would snap the
                # palm up by that offset the instant the lift phase begins.
                data.ctrl[ids["a_palm_pz"]] = palm_pz0 + args.lift * tl
            if phase in ("engage", "press"):
                if thumb_ctrl_at_engage is None:
                    thumb_ctrl_at_engage = np.array(
                        [start_ctrl[ids[n]] for n in ACT["thumb"]], dtype=float)
                retrack = args.track_every > 0 and step % args.track_every == 0
                if engage_target is None or args.thumb_track or retrack:
                    engage_target = surface_point(
                        model, data, args.thumb_s,
                        np.array([float(v) for v in args.thumb_dir.split()]), args.thumb_depth)
                    vals, err = solve_ctrl(model, data, "thumb", engage_target)
                    engage_ctrl["thumb"] = vals
                    if phase == "engage" and step % 100 == 0:
                        print(f"[engage] step {step} target {np.round(engage_target, 4)}"
                              f"  IK residual {err*1000:5.2f} mm  -> {np.round(vals, 3)}")
                te = 1.0 if phase != "engage" else t
                for k, name in enumerate(ACT["thumb"]):
                    data.ctrl[ids[name]] = (thumb_ctrl_at_engage[k] * (1 - te)
                                            + engage_ctrl["thumb"][k] * te)
                if args.chuck_tilt > 0.0:
                    if not chuck_ctrl or retrack:
                        for finger, side in (("index", +1.0), ("middle", -1.0)):
                            # Freeze each contact's axial coordinate at its engage-time value.
                            # Re-reading it from the tip lets the target walk along the shaft,
                            # and the fingers then drag the object off vertical instead of
                            # holding it.
                            s_f = chuck_s.setdefault(finger, axial_s(model, data, TIPS[finger]))
                            if args.chuck_x != 0.0:
                                centre, axis, _ = obj_frame(model, data)
                                u = np.array([1.0, 0.0, 0.0])
                                u = u - float(np.dot(u, axis)) * axis
                                u /= max(np.linalg.norm(u), 1e-9)
                                v = np.cross(axis, u)
                                tgt = (centre + s_f * axis + args.chuck_x * u
                                       + side * (obj_radius(model) - args.chuck_depth) * v)
                            else:
                                tgt = chuck_target(model, data, s_f,
                                                   side * (90.0 - args.chuck_tilt),
                                                   args.chuck_depth)
                            if args.chuck_frame == "palm":
                                R = data.body("palm_pose").xmat.reshape(3, 3)
                                o = data.body("palm_pose").xpos
                                if finger in chuck_palm_tgt:
                                    tgt = R @ chuck_palm_tgt[finger] + o
                                else:
                                    chuck_palm_tgt[finger] = R.T @ (tgt - o)
                            vals, err = solve_ctrl(model, data, finger, tgt)
                            if args.pair_yaw_bias != 0.0:
                                vals = vals.copy()
                                vals[0] += args.pair_yaw_bias * yaw_sign[finger]
                            first = finger not in chuck_ctrl
                            chuck_ctrl[finger] = vals
                            if first:
                                print(f"[chuck] {finger:6s} s={s_f*1000:+5.1f} mm  "
                                      f"target {np.round(tgt, 4)}  IK residual {err*1000:5.2f} mm")
                    for finger in ("index", "middle"):
                        for k, name in enumerate(ACT[finger]):
                            data.ctrl[ids[name]] = (closed_ctrl[finger][k] * (1 - te)
                                                    + chuck_ctrl[finger][k] * te)

            mujoco.mj_step(model, data)

            if args.save_chuck_pose is not None:
                R_o = data.body(OBJ).xmat.reshape(3, 3)
                c_o = data.body(OBJ).xpos
                tips_obj.append(np.stack([R_o.T @ (data.body(TIPS[f]).xpos - c_o)
                                          for f in ("thumb", "index", "middle")]))
                tips_phase.append(phase)
            if args.save_npz is not None:
                ref_qpos.append(data.qpos.copy())
                ref_qvel.append(data.qvel.copy())
                ref_z.append(float(data.body(OBJ).xpos[2]))
                ref_con.append(float(sum(1 for v in finger_forces(model, data).values()
                                         if v > 0.1)))

            R = data.body(OBJ).xmat.reshape(3, 3)
            cos = float(R[2, 2])
            if step % args.video_stride == 0 or step in grab_at:
                shots = []
                for cam in cams:
                    cam.lookat[:] = data.body(OBJ).xpos
                    renderer.update_scene(data, camera=cam)
                    shots.append(renderer.render())
                joined = np.concatenate(shots, axis=1)
                if step % args.video_stride == 0:
                    video.append(joined)
                if step in grab_at:
                    f = finger_forces(model, data)
                    strip.append(_label(Image.fromarray(joined),
                                        f"{phase} t={step} cos={cos:+.2f} "
                                        f"z={data.body(OBJ).xpos[2]:.3f} "
                                        f"N={f['thumb']:.1f}/{f['index']:.1f}/{f['middle']:.1f}"))

            if step % args.log_every == 0 or step == total - 1:
                forces = finger_forces(model, data)
                pos = data.body(OBJ).xpos.copy()
                trace.append({
                    "phase": phase, "step": step, "cos": cos,
                    "obj_z": float(pos[2]),
                    "obj_palm": [round(v, 5) for v in palm_frame(data, pos)],
                    "f_thumb": forces["thumb"], "f_index": forces["index"],
                    "f_middle": forces["middle"],
                    "gap_thumb": axis_gap(model, data, "thumb_tip"),
                    "on_floor": on_floor(model, data),
                })

    print(f"\n[probe] {args.scene.name}  start={args.start_keyframe}  lift={args.lift}  "
          f"engage={'off' if args.no_engage else f's={args.thumb_s} depth={args.thumb_depth}'}")
    print(f"{'phase':7s} {'step':>5s} {'cos':>7s} {'obj_z':>7s} {'thumbgap':>9s}   "
          f"forces N (thumb/index/middle)   obj in palm frame")
    for r in trace:
        print(f"{r['phase']:7s} {r['step']:5d} {r['cos']:+7.3f} {r['obj_z']:7.3f} "
              f"{r['gap_thumb']*1000:8.1f}m   "
              f"{r['f_thumb']:5.1f}/{r['f_index']:5.1f}/{r['f_middle']:5.1f}"
              f"   {r['obj_palm']}{'  ON-FLOOR' if r['on_floor'] else ''}")

    final = trace[-1]
    peak = max(r["cos"] for r in trace)
    grip = final["f_thumb"] + final["f_index"] + final["f_middle"]
    verdict = ("RELEASED-on-floor" if final["on_floor"] and grip < 0.1
               else "ON-FLOOR" if final["on_floor"]
               else "DROPPED" if final["obj_z"] <= 0.03 else "HELD")
    thumb_ever = max(r["f_thumb"] for r in trace)
    print(f"[result] peak cos {peak:+.3f}  final cos {final['cos']:+.3f}  "
          f"final z {final['obj_z']:.3f}  grip {grip:.2f} N  "
          f"thumb peak {thumb_ever:.2f} N  {verdict}")

    # The question is not whether it ever touched vertical -- the two-finger pinch does that
    # too, on its way past. It is how much of the hold window is spent vertical AND on the hand.
    hold = [r for r in trace if r["phase"] in ("hang", "press")
            and r["step"] >= (bounds[3] if args.no_engage else bounds[4])]
    if hold:
        up = [r for r in hold if r["cos"] >= 0.90 and not r["on_floor"]]
        share = 100.0 * len(up) / len(hold)
        fmean = {k: float(np.mean([r[f"f_{k}"] for r in hold])) for k in FINGER_BODIES}
        three = 100.0 * sum(1 for r in hold if min(r["f_thumb"], r["f_index"],
                                                   r["f_middle"]) > 0.5) / len(hold)
        print(f"[hold]   window {len(hold)} samples  cos>=0.90 and held {share:5.1f}%  "
              f"all three fingers loaded {three:5.1f}%  "
              f"mean N thumb/index/middle {fmean['thumb']:.1f}/{fmean['index']:.1f}/"
              f"{fmean['middle']:.1f}")
        summary = {"hold_share": share, "three_finger_share": three, "mean_force": fmean}
    else:
        summary = {}

    if args.save_chuck_pose is not None:
        traj = np.stack(tips_obj)                                  # (T, 3, 3)
        held = np.array([p == "press" for p in tips_phase])
        if not held.any():
            held = np.array([p in ("hang", "press") for p in tips_phase])
        hold = traj[held].mean(axis=0)
        spread = traj[held].std(axis=0).max()
        args.save_chuck_pose.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_chuck_pose, fingertip_obj=traj[::10], dt=0.02, t0=0.0,
                 hold_pose=hold, hold_spread=spread,
                 source=str(args.scene))
        print("[chuck-pose] object-frame fingertips over the hold (m), thumb/index/middle:")
        for name, row in zip(("thumb", "index", "middle"), hold):
            print(f"    {name:7s} {np.round(row, 4)}")
        print(f"[chuck-pose] max per-axis sd over the hold {spread*1000:.2f} mm "
              f"-> {args.save_chuck_pose}")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_npz, qpos=np.stack(ref_qpos), qvel=np.stack(ref_qvel),
                 cube_z=np.asarray(ref_z), contacts=np.asarray(ref_con),
                 best_finger_ctrl=np.concatenate([closed_ctrl.get(f, np.zeros(3))
                                                  for f in ("thumb", "index", "middle")]))
        print(f"[probe] reference trajectory -> {args.save_npz} ({len(ref_qpos)} steps)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tile(strip, 3).save(args.out_dir / "filmstrip.png")
    if video:
        import imageio.v3 as iio
        iio.imwrite(args.out_dir / "rollout.mp4", np.stack(video), fps=30)
    (args.out_dir / "trace.json").write_text(json.dumps({
        "label": args.label or args.out_dir.name,
        "scene": str(args.scene), "start_keyframe": args.start_keyframe,
        "lift": args.lift, "engage": (not args.no_engage),
        "thumb_s": args.thumb_s, "thumb_depth": args.thumb_depth,
        "thumb_track": args.thumb_track, "pinch_depth": args.pinch_depth,
        "chuck_tilt": args.chuck_tilt, "chuck_depth": args.chuck_depth,
        "peak_cos": peak, "final_cos": final["cos"], "final_z": final["obj_z"],
        "thumb_peak_force": thumb_ever, "verdict": verdict, **summary, "trace": trace,
    }, indent=2))
    print(f"[probe] {args.out_dir}/filmstrip.png  rollout.mp4  trace.json")


if __name__ == "__main__":
    main()

"""Scripted open-loop probe of the perpendicular-hand grasp/reorient mechanism.

Cheap CPU sanity check that runs BEFORE any GPU training: does the opposed index/middle pair
actually pinch the shaft, does the palm lift carry it, and does the shaft pitch toward vertical
once it is airborne? Prints a per-phase trace and writes a filmstrip PNG so the motion can be
inspected directly rather than inferred from numbers.

Phases (all open-loop position targets, no policy):
  settle -> close (squeeze the opposed pair) -> lift (palm_pz ramp) -> press (thumb pushes down)

The reorient target is SIGNED: reward wants the object's body +Z (the shaft's long axis, world
+X at spawn) pointing at world +Z. So a positive cos means the +X end came UP, which is what
both a +X-offset pinch (gravity pitches the heavier -X side down) and the thumb pressing at
negative X produce.

Run:
  MUJOCO_GL=egl uv run python scripts/probe_perp_mechanism.py \
    --scene assets/mjcf/scene_screwdriver_medium_perp.xml --out /tmp/perp_probe.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mj_snap import VIEWS, _label, tile  # noqa: E402

OBJ = "screwdriver_medium"
TIPS = ("thumb_tip", "index_tip", "middle_tip")
# A finger can load the shaft through its distal shaft, not just its pad; attributing only
# tip-body contacts reports 0 N while the hand is visibly gripping.
FINGER_BODIES = {"thumb": ("thumb_tip", "thumb_pip_frame", "thumb_len_frame"),
                 "index": ("index_tip", "index_pip_frame", "index_len_frame"),
                 "middle": ("middle_tip", "middle_pip_frame", "middle_len_frame")}


def act(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def obj_cos(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """cos between the shaft's body +Z and world +Z (the signed reorient objective)."""
    return float(data.body(OBJ).xmat.reshape(3, 3)[2, 2])


def tip_forces(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    """Total contact force each FINGER (pad + distal links) applies to the shaft."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--keyframe", default="open")
    ap.add_argument("--closed-keyframe", default="closed",
                    help="keyframe holding the SQUEEZING pose; the close phase ramps ctrl to it")
    ap.add_argument("--press-keyframe", default="press",
                    help="keyframe with the thumb driven INTO the shaft's top surface")
    ap.add_argument("--lift", type=float, default=0.10, help="palm_pz target (m)")
    ap.add_argument("--object-x", type=float, default=None,
                    help="respawn the shaft at this x (negative = pinch sits toward its +x end)")
    ap.add_argument("--settle-steps", type=int, default=200)
    ap.add_argument("--close-steps", type=int, default=400)
    ap.add_argument("--lift-steps", type=int, default=700)
    ap.add_argument("--press-steps", type=int, default=900)
    ap.add_argument("--view", default="side")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
    if args.object_x is not None:
        data.qpos[0] = args.object_x
    mujoco.mj_forward(model, data)

    ids = {n: act(model, n) for n in (
        "a_palm_pz", "a_thumb_yaw", "a_thumb_mcp", "a_thumb_pip",
        "a_index_mcp", "a_index_pip", "a_middle_mcp", "a_middle_pip")}
    base = data.ctrl.copy()
    closed = np.array(model.key(args.closed_keyframe).ctrl, dtype=float)
    press = np.array(model.key(args.press_keyframe).ctrl, dtype=float)

    total = args.settle_steps + args.close_steps + args.lift_steps + args.press_steps
    grab_at = set(np.linspace(0, total - 1, args.frames).astype(int).tolist())
    frames: list[Image.Image] = []
    trace: list[tuple[str, int, float, float, dict[str, float]]] = []

    az, el = VIEWS[args.view]
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = az, el, 0.45

    def phase_of(step: int) -> str:
        if step < args.settle_steps:
            return "settle"
        if step < args.settle_steps + args.close_steps:
            return "close"
        if step < args.settle_steps + args.close_steps + args.lift_steps:
            return "lift"
        return "press"

    with mujoco.Renderer(model, height=420, width=560) as renderer:
        for step in range(total):
            phase = phase_of(step)

            if phase in ("close", "lift", "press"):
                t = min(1.0, (step - args.settle_steps) / max(1, args.close_steps))
                # Facing fingers are near-radial to the shaft: "more flexion" curls the tip DOWN
                # past the object, not inward. The squeeze is an IK'd pose, not a flexion offset.
                for name in ("a_index_mcp", "a_index_pip", "a_middle_mcp", "a_middle_pip"):
                    data.ctrl[ids[name]] = base[ids[name]] * (1 - t) + closed[ids[name]] * t
            if phase in ("lift", "press"):
                t = min(1.0, (step - args.settle_steps - args.close_steps) / max(1, args.lift_steps))
                data.ctrl[ids["a_palm_pz"]] = args.lift * t
            if phase == "press":
                t = min(1.0, (step - args.settle_steps - args.close_steps - args.lift_steps)
                        / max(1, args.press_steps))
                # Same lesson as the pinch: "press down" is an IK'd tip target, not a flexion
                # offset -- adding mcp to this folded thumb swings its tip AWAY from the shaft.
                for name in ("a_thumb_mcp", "a_thumb_pip", "a_thumb_yaw"):
                    data.ctrl[ids[name]] = closed[ids[name]] * (1 - t) + press[ids[name]] * t

            mujoco.mj_step(model, data)

            if step in grab_at:
                cam.lookat[:] = data.body(OBJ).xpos
                renderer.update_scene(data, camera=cam)
                cos = obj_cos(model, data)
                frames.append(_label(Image.fromarray(renderer.render()),
                                     f"{phase} t={step} cos={cos:+.2f} z={data.body(OBJ).xpos[2]:.3f}"))
            if step % 100 == 0 or step == total - 1:
                trace.append((phase, step, obj_cos(model, data),
                              float(data.body(OBJ).xpos[2]), tip_forces(model, data)))

    print(f"[probe] {args.scene.name}  object_x={args.object_x}  lift={args.lift}  "
          f"closed_kf={args.closed_keyframe}  press_kf={args.press_keyframe}")
    print(f"{'phase':7s} {'step':>5s} {'cos':>7s} {'obj_z':>7s}   tip forces (N) thumb/index/middle")
    for phase, step, cos, z, forces in trace:
        f = "/".join(f"{forces[t]:5.1f}" for t in FINGER_BODIES)
        print(f"{phase:7s} {step:5d} {cos:+7.3f} {z:7.3f}   {f}")

    final_cos = trace[-1][2]
    peak_cos = max(t[2] for t in trace)
    final_z = trace[-1][3]
    print(f"[result] final cos {final_cos:+.3f}  peak cos {peak_cos:+.3f}  final obj z {final_z:.3f}  "
          f"{'HELD' if final_z > 0.03 else 'DROPPED'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tile(frames, 4).save(args.out)
    print(f"[probe] filmstrip -> {args.out}")


if __name__ == "__main__":
    main()

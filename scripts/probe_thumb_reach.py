"""Can the thumb REACH the shaft after the perp hand has reoriented it to vertical?

The perp reorient is done by gravity on the opposed index/middle pinch; the thumb reads ~0 N
throughout and contributes nothing (`docs/rl/perp_topology.md`). Its useful job is AFTER the
swing — stabilising the hanging shaft and closing the grasp so it can carry an axial load. That
job needs the shaft to land inside the thumb's reachable shell, which on the base perp design it
does not: the thumb mounts at x = -0.065 while the vertical shaft hangs at x = +0.035.

This runs the scripted settle -> close -> lift, then reports the geometry that decides whether a
candidate morphology can recruit the thumb at all:

  * the held state (alignment cos, object height) — so a design that lost the reorient is visible
  * where the shaft's axis ends up, in the PALM frame (the frame the morph x/y params live in)
  * the thumb's reach shell [D_min, D_max] and the distance to the nearest point on the shaft
  * a verdict: SHORT (target beyond D_max), INSIDE, or TOO-CLOSE (inside D_min, where the chain
    overshoots and the distal link hooks back — see the mujoco-eyes skill)

Reach is necessary, not sufficient: `probe_axial_load.py --press` is what says whether the thumb
actually carries load once it can touch.

Run:
  MUJOCO_GL=egl uv run python scripts/probe_thumb_reach.py --scene <scene.xml>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from probe_perp_mechanism import OBJ, obj_cos, tip_forces  # noqa: E402
from pose_open_keyframe import reach_shell  # noqa: E402

ACTS = ("a_palm_pz", "a_index_mcp", "a_index_pip", "a_middle_mcp", "a_middle_pip")


def build_hold(model, data, closed_key: str, lift: float,
               settle: int, close: int, lift_steps: int, hold_steps: int = 0) -> None:
    """settle -> close the opposed pair -> lift -> HOLD. No thumb press: we are measuring where
    the shaft ENDS UP under the gravity swing, before the thumb is asked to do anything.

    The hold phase is not optional padding — the swing is still in progress when the lift ramp
    ends (cos +0.62 at that instant vs +0.96 once settled), so measuring reach without it aims
    the thumb at a pose the shaft passes through rather than the one it rests in."""
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTS}
    base = data.ctrl.copy()
    closed = np.array(model.key(closed_key).ctrl, dtype=float)
    for step in range(settle + close + lift_steps + hold_steps):
        if step >= settle:
            t = min(1.0, (step - settle) / max(1, close))
            for n in ("a_index_mcp", "a_index_pip", "a_middle_mcp", "a_middle_pip"):
                data.ctrl[ids[n]] = base[ids[n]] * (1 - t) + closed[ids[n]] * t
        if step >= settle + close:
            t = min(1.0, (step - settle - close) / max(1, lift_steps))
            data.ctrl[ids["a_palm_pz"]] = lift * t
        mujoco.mj_step(model, data)


def shaft_axis(model, data) -> tuple[np.ndarray, np.ndarray, float]:
    """Object centre, unit long axis, and half-length (world frame)."""
    bid = model.body(OBJ).id
    centre = data.body(bid).xpos.copy()
    axis = data.body(bid).xmat.reshape(3, 3)[:, 2].copy()   # body +Z is the shaft's long axis
    half = 0.0
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == bid:
            half = max(half, float(model.geom_size[gi][1]) if model.geom_type[gi] ==
                       mujoco.mjtGeom.mjGEOM_CAPSULE else float(np.max(model.geom_size[gi])))
    return centre, axis / (np.linalg.norm(axis) + 1e-12), half


def dist_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--keyframe", default="open")
    ap.add_argument("--closed-keyframe", default="closed")
    ap.add_argument("--lift", type=float, default=0.14)
    ap.add_argument("--settle-steps", type=int, default=200)
    ap.add_argument("--close-steps", type=int, default=400)
    ap.add_argument("--lift-steps", type=int, default=900)
    ap.add_argument("--hold-steps", type=int, default=1400,
                    help="steps at full lift so the gravity swing SETTLES before measuring")
    ap.add_argument("--quiet", action="store_true", help="one summary line only")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
    mujoco.mj_forward(model, data)
    build_hold(model, data, args.closed_keyframe, args.lift,
               args.settle_steps, args.close_steps, args.lift_steps, args.hold_steps)

    cos = obj_cos(model, data)
    centre, axis, half = shaft_axis(model, data)
    a, b = centre - axis * half, centre + axis * half
    palm = data.body("palm_pose").xpos.copy()
    mount = data.body("thumb_mount").xpos.copy()
    tip = data.body("thumb_tip").xpos.copy()
    lo, hi = reach_shell(model, data, "thumb")
    d_axis, nearest = dist_to_segment(mount, a, b)
    d_tip, _ = dist_to_segment(tip, a, b)

    if d_axis > hi:
        verdict, gap = "SHORT", d_axis - hi
    elif d_axis < lo:
        verdict, gap = "TOO-CLOSE", lo - d_axis
    else:
        verdict, gap = "INSIDE", 0.0

    held = "HELD" if centre[2] > 0.05 else "DROPPED"
    if args.quiet:
        print(f"{args.scene.stem:52s} cos {cos:+.3f} z {centre[2]:.3f} {held:7s} "
              f"d_mount {d_axis:.4f} shell [{lo:.4f},{hi:.4f}] {verdict:9s} gap {gap*1000:+.1f}mm "
              f"d_tip {d_tip*1000:.1f}mm")
        return

    print(f"[thumb-reach] {args.scene.name}  lift={args.lift}")
    print(f"  held state          cos {cos:+.3f}   object z {centre[2]:.4f}   {held}")
    print(f"  grip N t/i/m        " + "/".join(
        f"{tip_forces(model, data)[k]:.1f}" for k in ("thumb", "index", "middle")))
    print(f"  shaft centre (world) {np.round(centre, 4)}   axis {np.round(axis, 3)}  "
          f"half-len {half:.4f}")
    print(f"  shaft ends (world)   {np.round(a, 4)} .. {np.round(b, 4)}")
    print(f"  palm  (world)        {np.round(palm, 4)}")
    print(f"  thumb mount (world)  {np.round(mount, 4)}   -> in palm frame "
          f"{np.round(mount - palm, 4)}")
    print(f"  thumb tip   (world)  {np.round(tip, 4)}   distance to shaft "
          f"{d_tip * 1000:.1f} mm")
    print(f"  thumb reach shell    [{lo:.4f}, {hi:.4f}] m")
    print(f"  mount -> nearest point on shaft {np.round(nearest, 4)}  = {d_axis:.4f} m")
    print(f"  VERDICT              {verdict}" + (f"  by {gap * 1000:.1f} mm" if gap else ""))
    if verdict == "SHORT":
        print(f"  -> needs {gap * 1000:.1f} mm more reach: thumb_len (+0..35 mm) and/or "
              f"thumb_x toward +x (mount range +-30 mm)")


if __name__ == "__main__":
    main()

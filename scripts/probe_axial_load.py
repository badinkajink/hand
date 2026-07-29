"""How much AXIAL (world-Z) force can the hand's grip on the screwdriver resist?

Holding a screwdriver is not the same as holding a mass. In use the driver is pushed down into
the screw and the screw pushes back UP the shaft, so the functional requirement is an axial load
capacity in BOTH directions, not just enough friction to carry 0.24 N of weight. This probe
measures that number so it can be optimised against instead of asserted.

Method: run the scripted grasp -> lift -> (optional thumb press) sequence to reach the held
state, then ramp an external force on the object along +Z and -Z until it escapes. "Escape" is
slip RELATIVE TO THE HAND (the palm is servoed, so a world-frame threshold would just measure
the palm actuator), declared when the object moves more than `--slip-tol` from the pose it held
at the moment the ramp started. Reports the force at escape for each direction.

Run:
  MUJOCO_GL=egl uv run python scripts/probe_axial_load.py \
    --scene assets/mjcf/generated/scene_perp_<...>.xml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from probe_perp_mechanism import OBJ, obj_cos, tip_forces  # noqa: E402

ACTS = ("a_palm_pz", "a_thumb_yaw", "a_thumb_mcp", "a_thumb_pip",
        "a_index_mcp", "a_index_pip", "a_middle_mcp", "a_middle_pip")


def build_hold(model, data, args) -> None:
    """Scripted settle -> close -> lift -> press, leaving the object held."""
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTS}
    base = data.ctrl.copy()
    closed = np.array(model.key(args.closed_keyframe).ctrl, dtype=float)
    press = np.array(model.key(args.press_keyframe).ctrl, dtype=float) if args.press else None

    for step in range(args.settle_steps + args.close_steps + args.lift_steps + args.press_steps):
        if step >= args.settle_steps:
            t = min(1.0, (step - args.settle_steps) / max(1, args.close_steps))
            for n in ("a_index_mcp", "a_index_pip", "a_middle_mcp", "a_middle_pip"):
                data.ctrl[ids[n]] = base[ids[n]] * (1 - t) + closed[ids[n]] * t
        if step >= args.settle_steps + args.close_steps:
            t = min(1.0, (step - args.settle_steps - args.close_steps) / max(1, args.lift_steps))
            data.ctrl[ids["a_palm_pz"]] = args.lift * t
        if press is not None and step >= args.settle_steps + args.close_steps + args.lift_steps:
            t = min(1.0, (step - args.settle_steps - args.close_steps - args.lift_steps)
                    / max(1, args.press_steps))
            for n in ("a_thumb_mcp", "a_thumb_pip", "a_thumb_yaw"):
                data.ctrl[ids[n]] = closed[ids[n]] * (1 - t) + press[ids[n]] * t
        mujoco.mj_step(model, data)


def ramp(model, data, args, sign: int) -> tuple[float, float]:
    """Ramp an axial force until the object escapes. Returns (force_at_escape, max_force)."""
    bid = model.body(OBJ).id
    # displacement is measured relative to the PALM, since the palm is servoed and moves
    palm_bid = model.body("palm_pose").id if mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "palm_pose") >= 0 else model.body("palm").id
    ref = (data.body(bid).xpos - data.body(palm_bid).xpos).copy()

    force = 0.0
    for step in range(args.ramp_steps):
        force = args.max_force * (step / max(1, args.ramp_steps - 1))
        data.xfrc_applied[bid, 2] = sign * force
        mujoco.mj_step(model, data)
        rel = data.body(bid).xpos - data.body(palm_bid).xpos
        if float(np.linalg.norm(rel - ref)) > args.slip_tol:
            data.xfrc_applied[bid, :] = 0.0
            return force, args.max_force
    data.xfrc_applied[bid, :] = 0.0
    return float("inf"), args.max_force


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--keyframe", default="open")
    ap.add_argument("--closed-keyframe", default="closed")
    ap.add_argument("--press-keyframe", default="press")
    ap.add_argument("--press", action="store_true", default=True,
                    help="run the thumb press phase before loading (default on)")
    ap.add_argument("--no-press", dest="press", action="store_false")
    ap.add_argument("--lift", type=float, default=0.16)
    ap.add_argument("--settle-steps", type=int, default=200)
    ap.add_argument("--close-steps", type=int, default=400)
    ap.add_argument("--lift-steps", type=int, default=900)
    ap.add_argument("--press-steps", type=int, default=1400)
    ap.add_argument("--ramp-steps", type=int, default=1500)
    ap.add_argument("--max-force", type=float, default=15.0, help="N at the end of the ramp")
    ap.add_argument("--slip-tol", type=float, default=0.01,
                    help="m of object motion RELATIVE TO THE PALM that counts as escape")
    args = ap.parse_args()

    results = {}
    for sign, label in ((+1, "up (+Z)"), (-1, "down (-Z)")):
        model = mujoco.MjModel.from_xml_path(str(args.scene))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
        mujoco.mj_forward(model, data)
        build_hold(model, data, args)
        held_cos, held_z = obj_cos(model, data), float(data.body(OBJ).xpos[2])
        forces = tip_forces(model, data)
        f_escape, f_max = ramp(model, data, args, sign)
        results[label] = (f_escape, f_max, held_cos, held_z, forces)

    print(f"[axial] {args.scene.name}  lift={args.lift}  press={args.press}  "
          f"slip_tol={args.slip_tol * 1000:.0f} mm")
    first = next(iter(results.values()))
    print(f"[axial] held state before loading: cos {first[2]:+.3f}  z {first[3]:.3f}  "
          f"grip N thumb/index/middle "
          + "/".join(f"{first[4][k]:.1f}" for k in ("thumb", "index", "middle")))
    print(f"{'direction':12s} {'escape force':>13s}")
    for label, (f_escape, f_max, *_ ) in results.items():
        shown = f">{f_max:.1f} N (held)" if f_escape == float("inf") else f"{f_escape:.2f} N"
        print(f"{label:12s} {shown:>13s}")
    weight = 9.81 * 0.0245  # medium screwdriver, ~24.5 g
    print(f"[axial] for scale, the shaft's own weight is {weight:.2f} N")


if __name__ == "__main__":
    main()

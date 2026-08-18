#!/usr/bin/env python3
"""What does a fingertip SHAPE buy, measured on the hand with no policy in the loop?

    uv run python scripts/probe_fingertip_mechanics.py --shapes all \
        --json-out docs/experiments/FINGERTIP_MECHANICS.json

WHY NOT JUST TRAIN ON EACH SHAPE. Per-design reorient quality in this program is seed-dominated
(per-draw sd 0.3-0.5, large enough to anti-order real designs), which is what made the 9-param
morphology sweep unable to resolve anything at n=4. A tip-shape sweep scored by RL outcome would
walk into the same wall at the same cost. So the shapes are ranked here by MECHANICS, which is
deterministic and takes seconds, and only the shapes that separate get a policy.

THE TRADE BEING MEASURED. The inline hand reorients the screwdriver by ROLLING it between the
pads, and it loses it by the shaft SLIDING along its own axis. Those pull the same lever in
opposite directions: a tip that resists sliding harder also resists the roll that does the
reorienting. So each shape gets both numbers on the same grip --

  axial capacity   N of force along the shaft axis before the tool escapes the hand   (hold)
  roll resistance  N*m of torque about the shaft axis before it turns 20 deg          (turn)

-- plus the ratio, which is the design's actual character: high axial per unit roll torque is a
tip that holds without fighting the reorient. A shape that merely grips harder in both raises
both numbers and buys nothing, and the ratio is what says so.

AND THE PAD SPEC. Each grip also reports penetration depth per newton. In MuJoCo soft contact,
penetration IS the model's stand-in for pad deflection, so that number is the compliance the
trained policy is actually relying on, expressed in millimetres the user can hold a durometer
against. It exists because the shipped policy's reorient collapses when contact is stiffened,
which is a hardware requirement wearing a solver setting's clothes.

Deterministic, CPU, no Warp. Reports every shape even if it fails, so a failure is visible
rather than absent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from probe_perp_mechanism import FINGER_BODIES, OBJ, obj_cos, tip_forces  # noqa: E402

from morphohand.sampling.morphology import FINGER_ACTUATOR_NAMES  # noqa: E402
from morphohand.studies.scene_mutate import TIP_SHAPES, Scene  # noqa: E402

MORPH = ROOT / "results/phase1/landscape/m05_ik_cem"


def pad_forces(model, data) -> dict[str, float]:
    """Per-finger contact force carried by the TIP bodies alone.

    Deliberately narrower than `tip_forces`, which also counts the phalanx links behind the pad.
    Two reasons. It is what a fingertip study is about; and it is exactly what the policies see
    -- the `fingertip_cube_contact` sensor matches bodies thumb_tip/index_tip/middle_tip only, so
    the deployed 9.5/5.6/5.5 N grip is pad force, and squeezing to a matching TOTAL would be a
    different grip. Measured directly: an early version targeted total force and settled into a
    pose carrying 20 N with 0% through the pads, which is not the hand b33 flies.
    """
    obj_id = model.body(OBJ).id
    out = {f: 0.0 for f in FINGER_BODIES}
    for i in range(data.ncon):
        con = data.contact[i]
        b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
        if obj_id not in (b1, b2):
            continue
        other = b2 if b1 == obj_id else b1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if not name.endswith("_tip"):
            continue
        f6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f6)
        out[name[: -len("_tip")]] += float(np.linalg.norm(f6[:3]))
    return out


# --------------------------------------------------------------------------- the held state
def build_hold(model, data, bfc, args, closure: float = 1.0) -> float:
    """settle -> close to `closure` x the CEM grip -> lift, leaving the tool held in the air.

    The closed pose is the CEM `best_finger_ctrl` scaled by `closure`, not a keyframe: that is
    the set-point both trained policies perturb, so a mechanics number taken from a different
    grip would not describe the hand the policies actually fly.

    WHY `closure` IS SEARCHED AND NOT FIXED (see `best_hold`). The CEM grip alone loads the pads
    at only ~0.4 N while the deployed a10->b33 handoff runs at ~20 N of pad force, so measuring
    capacity at the CEM pose describes a grip nobody uses. Two ways to squeeze harder were tried
    and both failed, informatively:

      - continuing along the CEM ray past s=1 does not squeeze. Pad force stayed near zero out to
        the ceiling because past the grasp pose more flexion curls each tip PAST the shaft and
        lands the phalanx behind it on the object -- the standing "flexion is not close" gotcha,
        met again here;
      - a greedy per-joint search for the direction that raises pad force drives fingers straight
        through the shaft (7.9 mm of penetration) and reports capacities for an interpenetrating
        pose.

    So closure is not extrapolated at all. It is line-searched over a bounded range and the pose
    that best LOADS THE PADS is kept, which is a pose the hand can actually reach.
    """
    fid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in FINGER_ACTUATOR_NAMES]
    pz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_palm_pz")
    base = data.ctrl.copy()
    open_ctrl = np.array([base[a] for a in fid], dtype=float)
    closed = np.array(bfc, dtype=float)

    for step in range(args.settle_steps + args.close_steps + args.lift_steps):
        if step >= args.settle_steps:
            t = min(1.0, (step - args.settle_steps) / max(1, args.close_steps))
            for i, a in enumerate(fid):
                data.ctrl[a] = open_ctrl[i] + closure * t * (closed[i] - open_ctrl[i])
        if step >= args.settle_steps + args.close_steps:
            t = min(1.0, (step - args.settle_steps - args.close_steps) / max(1, args.lift_steps))
            data.ctrl[pz] = args.lift * t
        mujoco.mj_step(model, data)
    return closure


def shaft_axis(data) -> np.ndarray:
    """Unit world vector along the tool's own axis (its body +z)."""
    return np.array(data.body(OBJ).xmat).reshape(3, 3)[:, 2]


def contact_report(model, data) -> dict:
    """Contacts on the tool, split by finger, with penetration depth and pad-vs-link attribution.

    `pad_frac` is the share of contact force carried by the TIP bodies rather than the phalanx
    behind them. A shape study is only meaningful where the pad is what touches; a low pad_frac
    means the link is doing the gripping and the shape barely matters at that pose.
    """
    body_to_finger = {b: f for f, bodies in FINGER_BODIES.items() for b in bodies}
    obj_id = model.body(OBJ).id
    n_per = {f: 0 for f in FINGER_BODIES}
    pad_f = link_f = 0.0
    depths = []
    for i in range(data.ncon):
        con = data.contact[i]
        b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
        if obj_id not in (b1, b2):
            continue
        other = b2 if b1 == obj_id else b1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        finger = body_to_finger.get(name)
        if finger is None:
            continue
        f6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f6)
        mag = float(np.linalg.norm(f6[:3]))
        n_per[finger] += 1
        depths.append(-float(con.dist))            # dist < 0 inside the surface
        if name.endswith("_tip"):
            pad_f += mag
        else:
            link_f += mag
    tot = pad_f + link_f
    return dict(
        n_contacts=int(sum(n_per.values())),
        n_per_finger=n_per,
        fingers_touching=int(sum(1 for v in n_per.values() if v > 0)),
        pad_frac=(round(pad_f / tot, 3) if tot > 0 else None),
        penetration_max_mm=(round(max(depths) * 1000, 3) if depths else 0.0),
        penetration_mean_mm=(round(float(np.mean(depths)) * 1000, 3) if depths else 0.0),
    )


# --------------------------------------------------------------------------- the two loads
def ramp_force(model, data, axis: np.ndarray, sign: int, *, max_force: float,
               steps: int, slip_tol: float) -> float:
    """Force along `axis` until the tool escapes the hand. inf = never escaped."""
    bid = model.body(OBJ).id
    palm = model.body("palm_pose").id
    ref = (data.body(bid).xpos - data.body(palm).xpos).copy()
    for step in range(steps):
        f = max_force * (step / max(1, steps - 1))
        data.xfrc_applied[bid, :3] = sign * f * axis
        mujoco.mj_step(model, data)
        if float(np.linalg.norm((data.body(bid).xpos - data.body(palm).xpos) - ref)) > slip_tol:
            data.xfrc_applied[bid, :] = 0.0
            return f
    data.xfrc_applied[bid, :] = 0.0
    return float("inf")


def ramp_torque(model, data, *, max_torque: float, steps: int, turn_tol: float) -> float:
    """Torque about the tool's own axis until it turns `turn_tol` rad. inf = never turned.

    Measured about the SHAFT axis specifically, because that is the rotation the reorient is
    made of. Resistance to any other rotation is a different (and less interesting) quantity.
    """
    bid = model.body(OBJ).id
    R0 = np.array(data.body(bid).xmat).reshape(3, 3)
    for step in range(steps):
        tau = max_torque * (step / max(1, steps - 1))
        data.xfrc_applied[bid, 3:] = tau * shaft_axis(data)
        mujoco.mj_step(model, data)
        R = np.array(data.body(bid).xmat).reshape(3, 3)
        dR = R0.T @ R
        angle = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        if angle > turn_tol:
            data.xfrc_applied[bid, :] = 0.0
            return tau
    data.xfrc_applied[bid, :] = 0.0
    return float("inf")


# --------------------------------------------------------------------------- one shape
def hold_at(scene: Path, bfc, args, closure: float):
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(args.keyframe).id)
    mujoco.mj_forward(m, d)
    build_hold(m, d, bfc, args, closure)
    return m, d


def best_hold(scene: Path, bfc, args) -> tuple[float, float]:
    """Line-search the closure scale for the pose that loads the PADS hardest.

    Each shape gets its own best closure rather than a shared one, because reach differs between
    shapes by design and a shared closure would compare a well-seated pad against a barely
    touching one. Poses that hold the tool below `--min-held-z`, or that bury a finger deeper
    than `--max-penetration`, are rejected outright -- a capacity measured through an
    interpenetrating link is not a measurement of the pad.
    """
    best = (1.0, -1.0)
    for closure in np.linspace(args.closure_min, args.closure_max, args.closure_points):
        m, d = hold_at(scene, bfc, args, float(closure))
        if float(d.body(OBJ).xpos[2]) < args.min_held_z:
            continue
        con = contact_report(m, d)
        if con["penetration_max_mm"] > args.max_penetration:
            continue
        f = sum(pad_forces(m, d).values())
        if f > best[1]:
            best = (float(closure), f)
    return best


def measure(scene: Path, bfc, args) -> dict:
    closure, _ = best_hold(scene, bfc, args)
    model, data = hold_at(scene, bfc, args, closure)
    grip = pad_forces(model, data)
    con = contact_report(model, data)
    pad_n = sum(grip.values())
    held_z, held_cos = float(data.body(OBJ).xpos[2]), obj_cos(model, data)
    out = dict(
        held_z=round(held_z, 4), held_cos=round(held_cos, 3), closure=round(closure, 3),
        grip_N={k: round(v, 2) for k, v in grip.items()},
        pad_N=round(pad_n, 2),
        finger_total_N=round(sum(tip_forces(model, data).values()), 2),   # pads + links
        # Penetration per newton IS the pad compliance the policy is relying on.
        pad_compliance_mm_per_N=(round(con["penetration_max_mm"] / pad_n, 4)
                                 if pad_n > 0.05 else None),
        **con,
    )
    if held_z < args.min_held_z:
        # A tool on the floor has no grip to load; loading it would report the floor's friction.
        out["status"] = "DROPPED_BEFORE_LOAD"
        return out
    if pad_n < args.min_pad_N:
        # No reachable closure put load on this pad. That is a verdict about the shape, not an
        # error, and it must be reported rather than papered over with a capacity number that
        # would really be describing the phalanx behind the pad.
        out["status"] = "PAD_NEVER_LOADED"
        return out
    out["status"] = "ok"

    axis = shaft_axis(data)
    for sign, tag in ((+1, "axial_plus_N"), (-1, "axial_minus_N")):
        m, d = hold_at(scene, bfc, args, closure)
        f = ramp_force(m, d, shaft_axis(d), sign, max_force=args.max_force,
                       steps=args.ramp_steps, slip_tol=args.slip_tol)
        out[tag] = (None if f == float("inf") else round(f, 3))
    m, d = hold_at(scene, bfc, args, closure)
    t = ramp_torque(m, d, max_torque=args.max_torque, steps=args.ramp_steps,
                    turn_tol=args.turn_tol)
    out["roll_torque_Nm"] = None if t == float("inf") else round(t, 5)

    ax = [v for v in (out["axial_plus_N"], out["axial_minus_N"]) if v is not None]
    axial = min(ax) if ax else args.max_force            # censored at the ramp ceiling
    roll = out["roll_torque_Nm"] or args.max_torque
    # Shapes cannot be compared at raw capacity, because the line search leaves each at its own
    # pad load. Normalising by that load gives the two dimensionless numbers that ARE the shape's
    # property: an effective axial friction coefficient, and a roll resistance in units of
    # (force x shaft radius). Their ratio is the design's character -- hold bought per unit of
    # resistance to the very rotation the reorient needs.
    r_shaft = 0.0125
    out["axial_per_N"] = round(axial / pad_n, 3)
    out["roll_per_Nm"] = round(roll / (pad_n * r_shaft), 4)
    out["hold_per_roll"] = round(out["axial_per_N"] / out["roll_per_Nm"], 2)
    out["axis_world"] = [round(float(v), 3) for v in axis]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph", type=Path, default=MORPH)
    ap.add_argument("--shapes", nargs="*", default=["all"])
    ap.add_argument("--radius", type=float, default=0.005)
    ap.add_argument("--half-length", type=float, default=0.006)
    ap.add_argument("--keyframe", default="open_ik")
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--settle-steps", type=int, default=200)
    ap.add_argument("--close-steps", type=int, default=500)
    ap.add_argument("--lift-steps", type=int, default=900)
    ap.add_argument("--closure-min", type=float, default=0.55)
    ap.add_argument("--closure-max", type=float, default=1.15)
    ap.add_argument("--closure-points", type=int, default=13,
                    help="line-search resolution for the closure that best loads the pads")
    ap.add_argument("--max-penetration", type=float, default=3.0,
                    help="mm; a pose burying a finger deeper than this is rejected as a pose, "
                         "not reported as a capacity")
    ap.add_argument("--min-pad-N", type=float, default=0.15,
                    help="below this the pad is not really loaded and the shape is reported as "
                         "PAD_NEVER_LOADED rather than given a meaningless capacity")
    ap.add_argument("--ramp-steps", type=int, default=1500)
    ap.add_argument("--max-force", type=float, default=15.0)
    ap.add_argument("--max-torque", type=float, default=0.08, help="N*m at the end of the ramp")
    ap.add_argument("--slip-tol", type=float, default=0.01, help="m relative to the palm")
    ap.add_argument("--turn-tol", type=float, default=0.35, help="rad (~20 deg) that counts as turned")
    ap.add_argument("--min-held-z", type=float, default=0.05)
    ap.add_argument("--solimp", nargs=2, type=float, default=None,
                    help="override contact stiffness (dmin dmax) to probe a stiffer pad")
    ap.add_argument("--tip-friction", type=float, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    shapes = list(TIP_SHAPES) if args.shapes == ["all"] else args.shapes
    bfc = np.load(args.morph / "best_rollout.npz")["best_finger_ctrl"].reshape(-1)
    work = ROOT / "results/phase1/fingertip/scenes"
    src = args.morph / "frozen_scene.xml"

    rows = {}
    for shape in shapes:
        sc = Scene(src).set_tip_shape(shape, r=args.radius, h=args.half_length)
        if args.solimp:
            sc.set_solimp(*args.solimp)
        if args.tip_friction is not None:
            sc.set_tip_friction(args.tip_friction)
        path = sc.write(work / f"probe_{shape}.xml")
        rows[shape] = measure(path, bfc, args)
        print(f"[{shape}] {rows[shape]['status']}  pad {rows[shape]['pad_N']:.2f} N")

    hdr = (f"{'shape':15s} {'fing':>4s} {'clos':>5s} {'pad N':>6s} {'pad%':>5s} "
           f"{'pen mm':>7s} {'mm/N':>6s} {'axial/N':>8s} {'roll/Nr':>8s} {'hold/roll':>10s}")
    print(f"\n# fingertip mechanics — lift {args.lift} m, escape = {args.slip_tol * 1000:.0f} mm "
          f"slip, turn = {args.turn_tol:.2f} rad")
    print("# 'clos' = closure scale chosen by the line search (1.0 = the CEM grip). Capacities "
          "are normalised by pad load:")
    print("#   axial/N = axial force to strip the tool per newton of pad force (an effective "
          "friction coefficient)")
    print("#   roll/Nr = roll torque to turn it, per (pad force x shaft radius) — resistance to "
          "the reorient itself")
    print(hdr)
    print("-" * len(hdr))
    for shape, r in rows.items():
        if r["status"] != "ok":
            print(f"{shape:15s} {r['status']}  (held_z {r['held_z']:.3f}, "
                  f"pad {r['pad_N']:.2f} N, links {r['finger_total_N']:.1f} N)")
            continue
        print(f"{shape:15s} {r['fingers_touching']:>4d} {r['closure']:>5.2f} "
              f"{r['pad_N']:>6.2f} {(r['pad_frac'] or 0) * 100:>4.0f}% "
              f"{r['penetration_max_mm']:>7.2f} {(r['pad_compliance_mm_per_N'] or 0):>6.3f} "
              f"{r['axial_per_N']:>8.2f} {r['roll_per_Nm']:>8.3f} {r['hold_per_roll']:>10.2f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "rows": rows}, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

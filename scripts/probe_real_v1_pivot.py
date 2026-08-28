"""Is the real_v1 grasp a rotational LOCK about the pinch axis?

WHY (2026-08-28). Four real_v1 designs carried the shaft through the A->B handoff cleanly and
NONE of them turned it: peak alignment cos 0.015-0.069 on hands whose morphologies differ by
60 mm of mount separation. A result that is invariant to the morphology is not about the
morphology. The suspect is the GRASP, and specifically the STRADDLE -- how far apart index and
middle sit ALONG the shaft.

`fit_real_v1_pose.py` picks the straddle by scoring candidates on a close->lift->hold rollout,
and its own docstring says what that selects for: "whether that tripod resists the shaft
LEVERING ABOUT THE PINCH AXIS depends on the straddle". Levering about the pinch axis is the
reorientation. The fitter was choosing, per design, the grasp that best prevents the task, and
it chose +-30 to +-40 mm on a shaft whose half-length is 50 mm.

Two things follow, and this probe measures both rather than arguing them:

  LOCK       Rotation about the pinch axis slides the pair contacts axially in OPPOSITE
             directions with a moment arm equal to the straddle, so friction resists with
             torque ~ 2*mu*N*straddle. At 14 N and 30 mm that is ~2 Nm against a 24.5 g shaft.
             Reported here as a BREAKAWAY TORQUE: the external torque about world X needed to
             move the shaft by 0.1 of alignment cosine in one second.
  GEARING    To reach vertical, each pair pad must travel straddle * pi/2 in Z, in opposite
             directions, while holding. At +-30 mm that is 47 mm on a finger with 78.66 mm of
             total flexing reach -- which is why `probe_real_v1_vertical_hold.py` found two of
             three designs could not even reach the terminal pose.

And what the reference policies did instead (docs/experiments/REORIENT_PRIMITIVE.txt): r4 on
perp holds TWO loaded contacts parked at the same axial station s = +42 mm with the thumb at
0.0 N, i.e. a pinch HINGE at a large offset from the centre of mass, and the shaft HANGS and
settles toward vertical. b33 on m05 balances it inverted with a ~20 mm offset. real_v1 grips at
offset ZERO with a wide straddle: no gravitational gradient to follow and a friction lock
against following one. So the probe sweeps both axes.

    MUJOCO_GL=egl uv run python scripts/probe_real_v1_pivot.py \
        --scene <generated scene>.xml --straddle 0.005,0.015,0.030 --offset 0,0.020,0.030
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fit_real_v1_pose as fp  # noqa: E402
from morphohand.tools.keyframe_ik import FINGERS, actuator_ctrl_from_qpos  # noqa: E402

TIPS = ("thumb_tip", "index_tip", "middle_tip")


def _finger_act(m) -> list[int]:
    names = [f"a_{j}" for joints in FINGERS.values() for j in joints]
    return [k for k in range(m.nu) if m.actuator(k).name in names]


def _contacts(m, d, obj: str) -> tuple[int, float]:
    n, tot = 0, 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b = {m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name}
        if obj in b and b & set(TIPS):
            mujoco.mj_contactForce(m, d, i, f6)
            n += 1
            tot += float(abs(f6[0]))
    return n, tot


def _cos(m, d, obj: str) -> float:
    """Alignment of the shaft's long axis with world +Z. MuJoCo cylinders are local +Z."""
    return float(abs(d.body(obj).xmat[8]))


def _snapshot(d):
    return (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy() if d.act.size else None)


def _restore(m, d, snap):
    d.qpos[:], d.qvel[:], d.ctrl[:] = snap[0], snap[1], snap[2]
    if snap[3] is not None:
        d.act[:] = snap[3]
    d.xfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)


def cell(scene: Path, obj: str, straddle: float, offset: float, lift: float,
         gap: float, squeeze: float, elevation: float,
         torques: tuple[float, ...], releases: tuple[float, ...]) -> dict | None:
    out = fp.fit(scene, gap, straddle, elevation, obj, -0.030, 0.060, 0.0025, verbose=False,
                 spreads=(straddle,), squeeze=squeeze, lift_probe=lift,
                 hold_min=-1.0, axial_offset=offset)
    if out is None:
        return None
    rep = out[0]
    px, py, pz = rep["palm"]["px"], rep["palm"]["py"], rep["palm"]["pz"]

    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    seed = fp._seed_key(m, "open")
    mujoco.mj_resetDataKeyframe(m, d, seed)
    mujoco.mj_forward(m, d)
    centre, radius, half = fp._object_geometry(m, d, obj)

    fp.solve(m, d, fp.tip_targets(centre, radius, gap, straddle, elevation, offset),
             px, py, pz, seed, iters=600)
    open_qpos = d.qpos.copy()
    surf_ctrl = np.array(actuator_ctrl_from_qpos(m, d))
    fp.solve(m, d, fp.tip_targets(centre, radius, gap - squeeze, straddle, elevation, offset),
             px, py, pz, seed, iters=600)
    grip_ctrl = np.array(actuator_ctrl_from_qpos(m, d))

    pz_a = next(k for k in range(m.nu) if m.actuator(k).name == "a_palm_pz")
    fing = _finger_act(m)
    bid = m.body(obj).id
    mass = float(m.body_mass[bid])

    # close -> lift -> settle
    d.qpos[:], d.qvel[:] = open_qpos, 0.0
    d.ctrl[:] = grip_ctrl
    mujoco.mj_forward(m, d)
    for _ in range(250):
        mujoco.mj_step(m, d)
    pz0 = float(grip_ctrl[pz_a])
    for k in range(200):
        d.ctrl[pz_a] = pz0 + lift * (k + 1) / 200
        mujoco.mj_step(m, d)
    for _ in range(300):
        mujoco.mj_step(m, d)

    held = float(d.body(obj).xpos[2] - centre[2])
    ncon, force = _contacts(m, d, obj)
    base_cos = _cos(m, d, obj)
    lifted = _snapshot(d)

    # BREAKAWAY: external torque about world X (the pinch axis) for 1.0 s.
    breakaway, torque_rows = None, []
    for tq in torques:
        _restore(m, d, lifted)
        d.xfrc_applied[bid, 3] = tq
        for _ in range(500):
            mujoco.mj_step(m, d)
        c, z = _cos(m, d, obj), float(d.body(obj).xpos[2])
        n2, f2 = _contacts(m, d, obj)
        torque_rows.append({"torque_Nm": tq, "cos": round(c, 3),
                            "z": round(z, 4), "contacts": n2, "force_N": round(f2, 2)})
        if breakaway is None and abs(c - base_cos) > 0.1:
            breakaway = tq
    d.xfrc_applied[:] = 0.0

    # HANG: gravity only, grip dialled between the squeeze (1.0) and the bare surface (0.0).
    hang_rows = []
    for r in releases:
        _restore(m, d, lifted)
        target = surf_ctrl + r * (grip_ctrl - surf_ctrl)
        for _ in range(1000):
            for k in fing:
                d.ctrl[k] = target[k]
            mujoco.mj_step(m, d)
        n2, f2 = _contacts(m, d, obj)
        hang_rows.append({"grip": r, "cos": round(_cos(m, d, obj), 3),
                          "z": round(float(d.body(obj).xpos[2]), 4),
                          "contacts": n2, "force_N": round(f2, 2)})

    return {
        "straddle_mm": straddle * 1000, "offset_mm": offset * 1000,
        "offset_frac": round(offset / half, 3),
        "palm_z": round(pz, 4), "grip_depth_mm": round(rep["grip_depth_mm"], 1),
        "tip_residual_mm": {k: round(v, 2) for k, v in rep["tip_residual_mm"].items()},
        "held_mm": round(held * 1000, 1), "lift_contacts": ncon, "lift_force_N": round(force, 2),
        "cos_after_lift": round(base_cos, 3),
        "gravity_torque_Nm": round(mass * 9.81 * offset, 5),
        "breakaway_Nm": breakaway, "torque_sweep": torque_rows, "hang_sweep": hang_rows,
        # what each pair pad must travel in Z to bring the shaft upright
        "pad_travel_mm": round(straddle * np.pi / 2 * 1000, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--object-body", default="screwdriver_medium")
    ap.add_argument("--straddle", default="0.005,0.010,0.020,0.030")
    ap.add_argument("--offset", default="0.0,0.020,0.030")
    ap.add_argument("--lift", type=float, default=0.05)
    ap.add_argument("--gap", type=float, default=0.001)
    ap.add_argument("--squeeze", type=float, default=0.004)
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--torques", default="0.002,0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--releases", default="1.0,0.5,0.25,0.0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    strad = tuple(float(v) for v in args.straddle.split(","))
    offs = tuple(float(v) for v in args.offset.split(","))
    tq = tuple(float(v) for v in args.torques.split(","))
    rel = tuple(float(v) for v in args.releases.split(","))

    rows = []
    print(f"{'strad':>6} {'off':>5} {'depth':>6} {'held':>7} {'con':>4} {'N':>6} "
          f"{'padTrav':>8} {'tauG':>8} {'breakaway':>10}   hang cos by grip "
          f"({', '.join(str(r) for r in rel)})")
    for off in offs:
        for st in strad:
            r = cell(args.scene, args.object_body, st, off, args.lift, args.gap,
                     args.squeeze, args.elevation, tq, rel)
            if r is None:
                print(f"{st*1000:6.0f} {off*1000:5.0f}   -- no pose --")
                continue
            rows.append(r)
            hb = " ".join(f"{h['cos']:.2f}{'*' if h['contacts'] == 0 else ''}"
                          for h in r["hang_sweep"])
            bk = "none" if r["breakaway_Nm"] is None else f"{r['breakaway_Nm']:.3f}"
            print(f"{st*1000:6.0f} {off*1000:5.0f} {r['grip_depth_mm']:6.1f} "
                  f"{r['held_mm']:+7.1f} {r['lift_contacts']:4d} {r['lift_force_N']:6.2f} "
                  f"{r['pad_travel_mm']:8.1f} {r['gravity_torque_Nm']:8.5f} {bk:>10}   {hb}")
    print("\n* = shaft no longer in contact (dropped); cos of a dropped shaft is meaningless.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""A held apex from the holder plan's apex search, and the holder's frame through the measured apexes (owner, 2026-10-27:
the first touch-offs found no contact 1.2 mm past the expected apex; the plywood is level). Each touch-off came down at an
offset (dx, dy) in I from the apex the plan expected; its onset past the expected contact, delta (mm, positive deeper),
puts the contact at z = z_plan - delta. Two pads' domes meet lower the farther the robot's apex stands from the held apex:
z = z0 - (dx - x0)^2 / (2 Rx) - (dy - y0)^2 / (2 Ry), with Ry the sum of the two pads' radii across the tip at the apex
(2 x 7.4^2 / 8.5 = 12.9 mm) and Rx along it (fitted when five or more touches are in, else 2 x 11^2 / 8.5 = 28.5 mm;
--flat-x drops the x terms: on the +y apex the touches 3 mm along x came 0.04 and 0.07 mm deeper than at the apex).
Least squares give the apex (x0, y0, z0 - z_plan) in I, the frame the program computed at homing (the session's T_BI).

--out FOLDER writes FOLDER/registration.json with the holder's frame T_BI through the measured apexes, for the plan
builder's --true-frame and the cell's renders. With one station measured, the frame is level, turned about the vertical
as the program's frame, and placed so that the model's apex of that station (fingertip_holder.held_apex) lands on the
measured one. With --frame, a frame written before from the other station, both apexes fix it: the turn about the
vertical and the tilt along the line between the stations by least squares, no tilt across that line. The program's
frame comes from seating the mount on the holder's key by hand, and the seatings of 2026-10-27 put it 4.3 deg from level
with its base 5 mm above the 2x4; the fixtures registered by touch-offs stand within 0.7 to 1.8 deg of level.

    .venv/bin/python scripts/ur5_tip_calibration_holder_apex_fit.py SESSION_FOLDER [--plan PLAN] [--out FOLDER]
        [--frame EARLIER_FOLDER/registration.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RY = 2 * 7.4 ** 2 / 8.5
RX0 = 2 * 11.0 ** 2 / 8.5


def fit(rows, flat_x=False):
    D = np.array([r[1:] for r in rows], float)
    z = -D[:, 2]                                                    # contact height relative to the plan's apex
    if flat_x:                                                      # no curvature along x: x0 stays the plan's
        A = np.c_[np.ones(len(z)), D[:, 1] / RY]
        y = z + D[:, 1] ** 2 / (2 * RY)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        y0 = c[1]
        return np.array([0.0, y0, c[0] + y0 ** 2 / (2 * RY)]), math.inf, float(np.sqrt(np.mean((A @ c - y) ** 2)))
    best = None
    for rx in ([RX0] if len(rows) < 5 else np.linspace(10.0, 120.0, 111)):
        A = np.c_[np.ones(len(z)), D[:, 0] / rx, D[:, 1] / RY]
        y = z + D[:, 0] ** 2 / (2 * rx) + D[:, 1] ** 2 / (2 * RY)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sqrt(np.mean((A @ c - y) ** 2)))
        if best is None or r < best[0]:
            best = (r, rx, c)
    r, rx, c = best
    x0, y0 = c[1], c[2]
    z0 = c[0] + x0 ** 2 / (2 * rx) + y0 ** 2 / (2 * RY)
    return np.array([x0, y0, z0]), rx, r


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rx_(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def model_apexes():
    sys.path.insert(0, str(ROOT / "scripts"))
    import fingertip_holder as FH
    import ur5_cb3_cell as CELL
    return {("+y" if CELL.TIPHOLD["stations"][i][0] > 0 else "-y"): FH.held_apex(CELL, i)
            for i in range(len(CELL.TIPHOLD["stations"]))}


def frame_one(T_prog, a_model, p_B):
    """Level, turned as the program's frame, through the measured apex."""
    R = rz(math.atan2(T_prog[1, 0], T_prog[0, 0]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p_B - R @ a_model
    return T


def frame_two(T_prog, a, p):
    """Turned about the vertical and tilted about x_I (along the line between the stations) to put both model apexes on
    the measured ones, by least squares; no tilt about y_I."""
    from scipy.optimize import least_squares
    A = np.array([a["+y"], a["-y"]])
    P = np.array([p["+y"], p["-y"]])

    def pose(v):
        R = rz(v[0]) @ rx_(v[1])
        t = (P - A @ R.T).mean(axis=0)
        return R, t

    def res(v):
        R, t = pose(v)
        return (A @ R.T + t - P).ravel()

    v = least_squares(res, [math.atan2(T_prog[1, 0], T_prog[0, 0]), 0.0]).x
    R, t = pose(v)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T, float(np.sqrt(np.mean(res(v) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--plan", default=None, help="the plan with the touch-offs' offsets (default: the session's plan.json)")
    ap.add_argument("--out", default=None, help="write the holder's frame through the measured apex(es) here")
    ap.add_argument("--frame", default=None, help="a registration.json written before by --out from the other station")
    ap.add_argument("--flat-x", action="store_true", help="the domes flat along x (the touches at x +-3 mm as deep as at "
                    "the apex, 2026-10-27 20:36): fit y0 and z0 only, x0 the plan's")
    a = ap.parse_args()
    sd = Path(a.session)
    touches = json.loads((sd / "touches.json").read_text())
    plan = json.loads(Path(a.plan or sd / "plan.json").read_text())
    acts = {x["id"]: x for x in plan["actions"]}
    station = plan.get("holder", {}).get("touch_station", "+y")
    rows = []
    for t in touches:
        x = acts.get(t["id"])
        if x is None or not t["id"].startswith("hold-touch"):
            continue
        off = x.get("offset", (0.0, 0.0))
        dx, dy = -float(off[0]), -float(off[1])                    # the contact point moves by -offset
        delta = float(t["s0"]) - float(t.get("expected_mm", x.get("standoff", 5.0)))
        rows.append((t["id"], dx, dy, delta))
        print(f"{t['id']}: offset in I ({dx:+.2f}, {dy:+.2f}) mm, onset {delta:+.3f} mm past the expected contact")
    if not rows:
        sys.exit("no hold-touch onsets in this session")
    d, rx, r = fit(rows, a.flat_x)
    print(f"{station} apex fit (Ry {RY:.1f} mm, Rx {rx:.1f} mm, rms {r:.3f} mm, {len(rows)} touches): the held apex stands "
          f"{d[0]:+.2f} mm along x_I, {d[1]:+.2f} mm along y_I and {d[2]:+.2f} mm in z from the plan's")
    if not a.out:
        print(f"rebuild with --z-offset {d[2]:+.2f} (and the seed shifted by ({d[0]:+.2f}, {d[1]:+.2f}) mm in I), or "
              "write the frame with --out")
        return
    T_prog = np.array(json.loads((sd / "registration.json").read_text())["T_BI"], float)
    apex_plan = np.array(plan["holder"]["held_apex_I"][station], float)
    p_B = T_prog[:3, :3] @ (apex_plan + d) + T_prog[:3, 3]
    am = model_apexes()
    measured = {station: p_B}
    if a.frame:
        prev = json.loads(Path(a.frame).read_text())
        for k, v in prev.get("measured_apex_B", {}).items():
            measured.setdefault(k, np.array(v, float))
    if len(measured) == 2:
        T, rms = frame_two(T_prog, am, measured)
        how = f"both apexes (residual {rms:.3f} mm rms)"
    else:
        T = frame_one(T_prog, am[station], p_B)
        how = f"the {station} apex, level"
    tilt = math.degrees(math.acos(min(1.0, T[2, 2])))
    yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pred = {k: (T[:3, :3] @ v + T[:3, 3]) for k, v in am.items()}
    in_prog = {k: (T_prog[:3, :3].T @ (v - T_prog[:3, 3])) for k, v in pred.items()}
    rec = dict(T_BI=np.round(T, 5).tolist(), T_BI_program=np.round(T_prog, 5).tolist(),
               measured_apex_B={k: np.round(v, 3).tolist() for k, v in measured.items()},
               apex_B={k: np.round(v, 3).tolist() for k, v in pred.items()},
               apex_in_program_I={k: np.round(v, 3).tolist() for k, v in in_prog.items()},
               notes=dict(made=time.strftime("%Y-%m-%d %H:%M"), how=how, session=str(sd), station=station,
                          fit=dict(offset_I=np.round(d, 3).tolist(), Rx=round(rx, 1), Ry=round(RY, 1), rms=round(r, 3),
                                   touches=[list(x) for x in rows]),
                          script="scripts/ur5_tip_calibration_holder_apex_fit.py"))
    (out / "registration.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"holder frame from {how}: origin {np.round(T[:3, 3], 2).tolist()} mm, yaw {yaw:.2f} deg, tilt {tilt:.2f} deg "
          f"(the program's: tilt {math.degrees(math.acos(min(1.0, T_prog[2, 2]))):.2f} deg) -> {out / 'registration.json'}")
    for k in sorted(pred):
        dp = in_prog[k] - np.array(plan["holder"]["held_apex_I"][k], float)
        print(f"  {k} apex in the program's I: {np.round(in_prog[k], 2).tolist()}, {np.round(dp, 2).tolist()} mm from the "
              "plan's")


if __name__ == "__main__":
    main()

"""Finger layouts of the SR2 hand in the simulator's palm frame P, in the hand driver's palm coordinates and in stepper
millimetres, and the check that a layout fits the gantries' travel (2026-11-15).

Layouts. The HOM studies give a layout as six coordinates d = [thumb x, thumb y, index x, index y, middle x, middle y]
in mm of P (hom_gpu.mount_xy), and a symmetric tripod as (x_sep, y_sep): the thumb at (-x_sep/2, 0), the index at
(x_sep/2, y_sep/2) and the middle at (x_sep/2, -y_sep/2) (fp_sr2_toy.layout2, hom_gpu.sym6). A mount is the centre of
the finger's yaw axis on the plane z_P = 0. P is the palm frame of assets/cad_v3/frames.json: x_P points from the thumb
toward the index and middle, y_P from the middle toward the index, and the origin lies midway between the travel
boxes' centres.

Driver coordinates. The driver (src/sr2hand/driver/manta/host/manta_hand) takes a mount in its own palm coordinates. A
finger's local coordinates are its palm coordinates less its origin (kinematics.FINGER_GEOMETRY: thumb (-50, 0), index
(50, 55), middle (50, -55) mm), and local (0, 0) puts both steppers at half their travel (kinematics._TRANSFORM and
FULL_EXTENSION_MM). The travel boxes of the cad_v3 CAD are anchored at the home stops and centred elsewhere
(frames.json fingers.<f>.box_centre_P_mm; docs/experiments/20261104-cad_v3_hand, section "Finger travel against the
driver's positions"): thumb (-56.18, -2.40), index (56.18, 58.40), middle (56.18, -58.40) mm. The local axes are P's
axes, because the CAD puts each stage's home at the corner the driver homes to (thumb bottom-left, index top-right,
middle bottom-right; docs/experiments/20261104-cad_v3_hand/analysis/gantry/gantry.json, identification_evidence). A
finger at driver coordinates u therefore stands at u + o in P, with o = box centre - driver origin (frames.json
box_minus_nominal_mm):

    o_thumb = (-6.18, -2.40), o_index = (6.18, 3.40), o_middle = (6.18, -3.40) mm
    driver = P - o,   P = driver + o,   gantry offset from the box centre = P - box centre = driver - origin

The gantry offset is the driver's local coordinate and the x_mm, y_mm of tool_chain.SR2Hand.finger_links. The
crosstalk analysis of 2026-11-05 used the same offsets (scripts/fingertip/sr2_crosstalk_hand.py SPACING), and its
stepper map STEPPER, fitted on the station's 14 logged layout moves, is the driver's palm_from_stepper.

Travel. A mount fits when its gantry offset lies in the box: |x| <= 30 and |y| <= 55 mm on the thumb, |x| and |y| <= 30
mm on the index and middle. The driver checks the same box (plan.mount_violations, FULL_EXTENSION_MM). Past the box's
far end the CAD's stages reach mechanical stops 4.80 mm further on the stages along y_P (firmware J0, J2, J4) and 2.58
mm further on those along x_P (J1, J3, J5) (frames.json stops_disp_mm against box_disp_mm). The driver does not
command that range.

    .venv/bin/python -m sr2hand.robot.sr2_layouts                         # the five layouts of the first trial series
    .venv/bin/python -m sr2hand.robot.sr2_layouts --xy 70,110 --d6=-35,0,35,55,35,-55 [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from sr2hand.robot.tool_chain import SR2Hand

ROOT = Path(__file__).resolve().parents[3]
DRIVER_HOST = ROOT / "src/sr2hand/driver/manta/host"
FINGERS = ("thumb", "index", "middle")
# The first real-trial series (docs/notes/20261105-object_seating_and_tracking/20261105-object_seating_and_tracking_plan.md,
# section "Reproduction of the simulated experiments"): the five layouts of the task-conflict study on the SCS0009 plant
# (docs/experiments/20261103-hom_task_conflict/20261104-conflict_sv25.json), each as (x_sep, y_sep) in mm and its index
# `call` on the 5 mm lattice of fp_sr2_toy.grid2d.
FIRST_SERIES = {"generalist": ((70.0, 110.0), 162), "twist": ((100.0, 70.0), 304), "roll": ((85.0, 170.0), 249),
                "pinpush": ((75.0, 55.0), 176), "press": ((70.0, 150.0), 170)}
STAGE_AXIS_A = {"x_stage": (1.0, 0.0, 0.0), "y_stage": (0.0, 1.0, 0.0)}    # frames.json: the stages move along x_A, y_A


def _driver():
    """The driver's kinematics and plan modules (its host package on sys.path, as sr2hand.control.hom_bench does)."""
    if str(DRIVER_HOST) not in sys.path:
        sys.path.insert(0, str(DRIVER_HOST))
    from manta_hand import kinematics as KN
    from manta_hand import plan as PL
    return KN, PL


# ------------------------------------------------------------------------------------------------ layouts

def sym6(x_sep, y_sep) -> np.ndarray:
    """The six coordinates (mm, P) of the symmetric tripod (x_sep, y_sep)."""
    return np.array([-x_sep / 2, 0.0, x_sep / 2, y_sep / 2, x_sep / 2, -y_sep / 2], float)


def seps(d6, tol=1e-6) -> tuple[float, float]:
    """(x_sep, y_sep) of a symmetric tripod given as six coordinates; ValueError for any other layout."""
    d = np.asarray(d6, float)
    xs, ys = float(d[2] - d[0]), float(d[3] - d[5])
    if np.abs(d - sym6(xs, ys)).max() > tol:
        raise ValueError(f"{d.tolist()} is not a symmetric tripod")
    return xs, ys


def mounts(d6) -> dict:
    """{finger: (x, y)} (mm, P) of six coordinates."""
    d = np.asarray(d6, float)
    return {f: d[2 * k:2 * k + 2].copy() for k, f in enumerate(FINGERS)}


def d6_of(m) -> np.ndarray:
    """Six coordinates of {finger: (x, y)}."""
    return np.concatenate([np.asarray(m[f], float)[:2] for f in FINGERS])


# ------------------------------------------------------------------------------------------------ the tables

@lru_cache(maxsize=None)
def tables() -> dict:
    """Per finger: the travel box's centre and half sizes in P (frames.json through tool_chain.SR2Hand), the driver's
    origin (kinematics.FINGER_GEOMETRY), their difference o, and each stage's direction in P, box and stops (mm of
    stage displacement from the CAD pose, frames.json box_disp_mm and stops_disp_mm). Raises if frames.json and the
    driver disagree on a box's size or with box_minus_nominal_mm."""
    hand = SR2Hand.load()
    KN, _PL = _driver()
    F = hand.frames
    R_PA = np.array(F["P"]["R_P_from_A"], float)
    out = {}
    for k, f in enumerate(FINGERS):
        centre, half = hand.box(f)
        geo = KN.FINGER_GEOMETRY[k]
        origin = np.array(geo["origin"], float)
        ff = F["fingers"][f]
        if not np.allclose(2.0 * half, geo["box"], atol=1e-6):
            raise ValueError(f"{f}: frames.json box {2 * half} mm against the driver's {geo['box']}")
        off = centre[:2] - origin
        if not np.allclose(off, ff["box_minus_nominal_mm"], atol=1e-3):
            raise ValueError(f"{f}: box centre - origin {off} against box_minus_nominal_mm {ff['box_minus_nominal_mm']}")
        stages = {}
        for st, axis_A in STAGE_AXIS_A.items():
            box = np.array(ff["box_disp_mm"][st], float)
            stages[st] = {"joint": ff["firmware"][st], "u": (R_PA @ np.array(axis_A))[:2], "centre": float(box.mean()),
                          "box": box, "stops": np.array(ff["stops_disp_mm"][st], float)}
        out[f] = {"centre": centre[:2].copy(), "half": half.copy(), "origin": origin, "offset": off, "stages": stages}
    return out


def offset(finger) -> np.ndarray:
    """o (mm): the travel box's centre in P less the driver's origin of the finger."""
    return tables()[finger]["offset"].copy()


# ------------------------------------------------------------------------------------------------ conversions

def driver_from_P(finger, xy) -> np.ndarray:
    """A mount in P (mm) -> the driver's palm coordinates (mm), as a HandPlan's mounts_palm_mm."""
    return np.asarray(xy, float)[:2] - offset(finger)


def P_from_driver(finger, xy) -> np.ndarray:
    """The driver's palm coordinates (mm) -> P (mm)."""
    return np.asarray(xy, float)[:2] + offset(finger)


def gantry_from_P(finger, xy) -> np.ndarray:
    """The gantry's offset (mm) from its travel box's centre along x_P and y_P: the driver's local coordinates."""
    return np.asarray(xy, float)[:2] - tables()[finger]["centre"]


def steppers_from_P(finger, xy) -> dict:
    """{firmware joint index: mm from home} that put the finger's mount at xy (mm, P): the driver's
    plan.stepper_mm of its local coordinates."""
    _KN, PL = _driver()
    return {int(j): float(v) for j, v in PL.stepper_mm(finger, *PL.local_from_palm(finger, *driver_from_P(finger, xy))).items()}


def P_from_steppers(finger, axes_mm) -> np.ndarray:
    """{firmware joint index: mm from home} -> the mount in P (mm): the driver's plan.palm_from_stepper, then + o."""
    _KN, PL = _driver()
    return P_from_driver(finger, PL.palm_from_stepper(finger, axes_mm))


def driver_layout(d6) -> dict:
    """{finger: [x, y]} (mm) in the driver's palm coordinates of six coordinates in P."""
    return {f: driver_from_P(f, xy).tolist() for f, xy in mounts(d6).items()}


# ------------------------------------------------------------------------------------------------ travel

def box_margins(finger, xy) -> dict:
    """Distance (mm) from a mount to each edge of its travel box, negative outside: x_lo, x_hi, y_lo, y_hi along x_P
    and y_P, and the smallest with its edge."""
    g, h = gantry_from_P(finger, xy), tables()[finger]["half"]
    m = {"x_lo": g[0] + h[0], "x_hi": h[0] - g[0], "y_lo": g[1] + h[1], "y_hi": h[1] - g[1]}
    edge = min(m, key=m.get)
    return {**{k: round(float(v), 4) for k, v in m.items()}, "min": round(float(m[edge]), 4), "edge": edge}


def stop_margins(finger, xy) -> dict:
    """Per stage: its firmware joint, its displacement from the CAD pose at this mount (mm), and the distance to the
    nearer end of the driver's box and to the nearer mechanical stop (mm, negative beyond it)."""
    t = tables()[finger]
    g = gantry_from_P(finger, xy)
    out = {}
    for st, s in t["stages"].items():
        disp = s["centre"] + float(s["u"] @ g)
        out[s["joint"]] = {"stage": st, "disp_mm": round(disp, 4),
                           "box_mm": round(float(min(disp - s["box"][0], s["box"][1] - disp)), 4),
                           "stop_mm": round(float(min(disp - s["stops"][0], s["stops"][1] - disp)), 4)}
    return out


def layout_report(d6) -> dict:
    """Every coordinate of a layout and its travel check: per finger the mount in P and in driver coordinates, the
    gantry offset, the steppers, the box and stop margins and the driver's own violations; `fits` when every mount lies
    in its box, and the smallest margin."""
    _KN, PL = _driver()
    fingers = {}
    for f, xy in mounts(d6).items():
        drv = driver_from_P(f, xy)
        fingers[f] = {"P_mm": [round(float(v), 4) for v in xy], "driver_mm": [round(float(v), 4) for v in drv],
                      "gantry_mm": [round(float(v), 4) for v in gantry_from_P(f, xy)],
                      "steppers_mm": {f"J{j}": round(v, 4) for j, v in sorted(steppers_from_P(f, xy).items())},
                      "box_margin_mm": box_margins(f, xy), "stop_margin_mm": stop_margins(f, xy),
                      "driver_violations": [str(v) for v in PL.mount_violations(f, *drv)]}
    worst = min(FINGERS, key=lambda f: fingers[f]["box_margin_mm"]["min"])
    out = {"d6_P_mm": [round(float(v), 4) for v in np.asarray(d6, float)], "fingers": fingers,
           "fits": all(fingers[f]["box_margin_mm"]["min"] >= -1e-9 for f in FINGERS),
           "min_margin_mm": fingers[worst]["box_margin_mm"]["min"],
           "min_margin_at": f"{worst} {fingers[worst]['box_margin_mm']['edge']}"}
    try:
        out["xy_mm"] = list(seps(d6))
    except ValueError:
        pass
    return out


def nearest_fit(d6) -> np.ndarray:
    """The nearest layout inside the travel boxes (mm, P). A symmetric tripod stays one: x_sep/2 and y_sep/2 are clipped
    to the range every finger's box allows, rounded inward to 0.01 mm. Any other layout has each mount moved to the
    nearest point of its box."""
    t = tables()
    try:
        xs, ys = seps(d6)
    except ValueError:
        return d6_of({f: t[f]["centre"] + np.clip(xy - t[f]["centre"], -t[f]["half"], t[f]["half"])
                      for f, xy in mounts(d6).items()})
    lo = {f: t[f]["centre"] - t[f]["half"] for f in FINGERS}
    hi = {f: t[f]["centre"] + t[f]["half"] for f in FINGERS}
    hx = (max(-hi["thumb"][0], lo["index"][0], lo["middle"][0]), min(-lo["thumb"][0], hi["index"][0], hi["middle"][0]))
    hy = (max(lo["index"][1], -hi["middle"][1]), min(hi["index"][1], -lo["middle"][1]))
    if not lo["thumb"][1] <= 0.0 <= hi["thumb"][1]:
        raise ValueError("the thumb's box does not reach y_P = 0: no symmetric tripod fits")
    out = []
    for h, (a, b) in ((xs / 2, hx), (ys / 2, hy)):
        out.append(2.0 * (math.ceil(a * 100.0) / 100.0 if h < a else math.floor(b * 100.0) / 100.0 if h > b else h))
    return sym6(*out)


# ------------------------------------------------------------------------------------------------ report

def _fmt(name, rep) -> list[str]:
    head = f"{name}  d6 {rep['d6_P_mm']} mm in P" + (f", (x_sep, y_sep) = {tuple(rep['xy_mm'])}" if "xy_mm" in rep else "")
    lines = [head, f"  {'finger':7s} {'P (mm)':>17s} {'driver (mm)':>17s} {'from centre':>17s}  {'steppers (mm)':>22s}"
                   f" {'box margin':>14s}  stop margins (mm)"]
    for f, r in rep["fingers"].items():
        st = "  ".join(f"{k} {v:.2f}" for k, v in r["steppers_mm"].items())
        bm = r["box_margin_mm"]
        sm = "  ".join(f"{j} {v['stop_mm']:.2f}" for j, v in r["stop_margin_mm"].items())
        lines.append(f"  {f:7s} {r['P_mm'][0]:8.2f} {r['P_mm'][1]:8.2f} {r['driver_mm'][0]:8.2f} {r['driver_mm'][1]:8.2f}"
                     f" {r['gantry_mm'][0]:8.2f} {r['gantry_mm'][1]:8.2f}  {st:>22s} {bm['min']:8.2f} {bm['edge']:>5s}"
                     f"  {sm}")
        lines += [f"    driver refuses: {v}" for v in r["driver_violations"]]
    lines.append(f"  fits the travel boxes: {'yes' if rep['fits'] else 'NO'}; smallest margin {rep['min_margin_mm']:.2f} mm"
                 f" ({rep['min_margin_at']})")
    if not rep["fits"]:
        lines.append(f"  nearest layout inside the boxes: {nearest_fit(rep['d6_P_mm']).round(2).tolist()} mm")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xy", action="append", default=[], help="a symmetric tripod x_sep,y_sep (mm); repeatable")
    ap.add_argument("--d6", action="append", default=[], help="six coordinates in P (mm), comma-separated; repeatable")
    ap.add_argument("--json", action="store_true", help="print the reports as JSON")
    a = ap.parse_args()
    todo = [(f"({x}, {y})", sym6(*map(float, (x, y)))) for x, y in (s.split(",") for s in a.xy)]
    todo += [(s, np.array([float(v) for v in s.split(",")])) for s in a.d6]
    if not todo:
        todo = [(f"{name} {xy}", sym6(*xy)) for name, (xy, _call) in FIRST_SERIES.items()]
    reps = {name: layout_report(d6) for name, d6 in todo}
    if a.json:
        print(json.dumps(reps, indent=1))
        return 0
    t = tables()
    print("box centre - driver origin (mm): " + ", ".join(f"{f} {t[f]['offset'].round(4).tolist()}" for f in FINGERS))
    for name, rep in reps.items():
        print("\n".join(_fmt(name, rep)))
    return 0 if all(r["fits"] for r in reps.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Finger-base spacing against object size on the real_v1 hand: the layout family, the reachable
(layout, object) cells, and the fixed-contact precision-manipulation workspace.

Each finger is a yaw-MCP-PIP chain below its gantry mount (scripts/build_real_v1_scenes.py): yaw about
palm +x at the mount, the MCP 20.75 mm below the mount and the PIP 20.75 mm below that, both flexing
about palm y (the thumb's axis mirrored, so positive flexion closes every finger toward the opposing
side), and the pad centre 26.61 mm past the PIP. Forward and inverse kinematics of that chain are closed
form here and checked against the compiled MuJoCo scene by `check`, so a whole
(layout x object x palm height x straddle) grid takes seconds on one core.

Joint limits are the servos' calibrated ranges (manta_hand.servos.FINGER_JOINTS through plan.JOINT_SIGN)
intersected with the scene ROM, less LIMIT_MARGIN; index MCP stops at 64.75 deg and index PIP at
-2.93 deg, which the scene does not model. `--sim-rom` uses the scene ROM alone (what
fit_real_v1_pose.py uses), for the comparison with that fitter.

Layouts are symmetric tripods: thumb mount (-x_sep/2, 0), index (+x_sep/2, +y_sep/2), middle
(+x_sep/2, -y_sep/2), palm frame, mm. The fitter re-centres the palm over the object, so the six gantry
coordinates of a symmetric tripod reduce to these two separations; the realisation written to the
manifest puts the thumb and the pair the same distance from their CAD-nominal mounts.

  uv run python scripts/hand_object_scale_kin.py check
  uv run python scripts/hand_object_scale_kin.py reach
  uv run python scripts/hand_object_scale_kin.py workspace
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/manta/host"))

OUT_DIR = ROOT / "docs/experiments/20261008-hand_object_scale"
BASE_SCENE = ROOT / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml"

# --- CAD geometry (build_real_v1_scenes.py), metres -----------------------------------------
J = 0.020750                  # yaw axis -> MCP axis and MCP axis -> PIP axis
P = 0.026610                  # PIP axis -> pad centre
R = 0.010550                  # phalange radius = pad sphere radius
CAP_YAW, CAP_MID, CAP_DIST = 0.022900, 0.022900, 0.016060   # capsule medial-axis lengths
PLATE_BOTTOM = 0.025 - 0.0015  # palm plate underside above the mounting plane
GAP = 0.001                   # pad surface to object surface at the fitted pose (fit_real_v1_pose default)
LIMIT_MARGIN = 0.05           # rad; a joint closer than this to its stop is parked (fit_real_v1_pose)
CLEAR_GATE = 0.005            # finger-to-finger PASS threshold of real_v1_trajectory_clearance.py
OBJ_CLEAR = 0.001             # non-pad finger geometry must stay this far off the object
FINGERS = ("thumb", "index", "middle")
FLEX = {"thumb": -1.0, "index": 1.0, "middle": 1.0}          # sign of the flexion axis' y component
SIM_ROM = np.radians([[-85.0, 85.0], [-15.0, 92.0], [-18.0, 92.0]])
CHAIN = J + P                 # MCP -> pad centre, fully extended


def limits(sim_rom_only: bool = False) -> dict[str, np.ndarray]:
    """Per finger, rows yaw/mcp/pip, columns lo/hi, radians: the scene ROM meet the servo range."""
    if sim_rom_only:
        return {f: SIM_ROM.copy() for f in FINGERS}
    from manta_hand.plan import FINGER_ID, JOINT_SIGN, SIM_JOINT_TO_SERVO
    from manta_hand.servos import FINGER_JOINTS
    out = {}
    for f in FINGERS:
        rows = []
        for k, j in enumerate(("yaw", "mcp", "pip")):
            _sid, _zero, (lo, hi) = FINGER_JOINTS[FINGER_ID[f]][SIM_JOINT_TO_SERVO[j]]
            sgn = JOINT_SIGN[(f, j)]
            slo, shi = sorted((lo / sgn, hi / sgn))
            rows.append([max(SIM_ROM[k, 0], math.radians(slo)), min(SIM_ROM[k, 1], math.radians(shi))])
        out[f] = np.array(rows)
    return out


# --- layouts ---------------------------------------------------------------------------------
def mounts(x_sep: float, y_sep: float) -> dict[str, np.ndarray]:
    """Palm-frame mount positions (m) of the symmetric tripod, origin at the pinch midpoint."""
    a, b = x_sep / 2.0, y_sep / 2.0
    return {"thumb": np.array([-a, 0.0, 0.0]), "index": np.array([a, b, 0.0]),
            "middle": np.array([a, -b, 0.0])}


def morph_vector(x_sep: float, y_sep: float) -> list[float]:
    """The 9-vector (thumb x,y,len, index x,y,len, middle x,y,len) of the realisation that moves the
    thumb and the pair by the same amount from the CAD-nominal mounts (thumb -50, pair +50/+-55 mm)."""
    dx = (0.100 - x_sep) / 2.0          # thumb +dx, pair -dx
    dy = y_sep / 2.0 - 0.055            # index +dy, middle -dy
    return [dx, 0.0, 0.0, -dx, dy, 0.0, -dx, -dy, 0.0]


def circumcentre_x(x_sep: float, y_sep: float) -> float:
    """x of the circle through the three mounts (thumb (-a,0), pair (a,+-b)): b^2 / 4a."""
    a, b = x_sep / 2.0, y_sep / 2.0
    return b * b / (4.0 * a)


def palm_radius(x_sep: float, y_sep: float) -> float:
    """Radius of the circle through the three mounts, the 'palm radius' of Borras and Dollar."""
    a = x_sep / 2.0
    return a + circumcentre_x(x_sep, y_sep)


def family(kind: str) -> list[dict]:
    """Layouts, mm. 'diag': the 1-D tripod scale, t in [-1, 1] (t in [0, 1] is
    real_v1_compact_design(t, t, t); t < 0 continues to the gantries' wide limit). 'xsep' / 'ysep':
    one separation varied, the other at its CAD-nominal value. 'grid': the 9 x 9 product."""
    rows = []
    if kind == "diag":
        for t in np.round(np.arange(-1.0, 1.0001, 0.125), 4):
            rows.append(dict(t=float(t), x_sep=100.0 - 60.0 * t, y_sep=110.0 - 60.0 * t))
    elif kind == "xsep":
        rows = [dict(x_sep=float(x), y_sep=110.0) for x in np.arange(40.0, 160.01, 7.5)]
    elif kind == "ysep":
        rows = [dict(x_sep=100.0, y_sep=float(y)) for y in np.arange(50.0, 170.01, 7.5)]
    elif kind == "grid":
        rows = [dict(x_sep=float(x), y_sep=float(y))
                for x in np.arange(40.0, 160.01, 15.0) for y in np.arange(50.0, 170.01, 15.0)]
    else:
        raise ValueError(kind)
    for r in rows:
        r["palm_radius"] = round(palm_radius(r["x_sep"] / 1000, r["y_sep"] / 1000) * 1000, 2)
        r["tag"] = f"x{r['x_sep']:05.1f}_y{r['y_sep']:05.1f}"
    return rows


# --- finger kinematics -------------------------------------------------------------------------
def _yaw(x, z, psi):
    """Local (pre-yaw) point (x, 0, z) rotated by the yaw about +x."""
    return np.stack([x, -z * np.sin(psi), z * np.cos(psi)], -1)


def fk(finger: str, q: np.ndarray) -> dict[str, np.ndarray]:
    """Points of one finger relative to its mount, palm frame, for joint angles q (..., 3) rad."""
    s = FLEX[finger]
    psi, t1, t2 = q[..., 0], q[..., 1], q[..., 2]
    z0 = np.zeros_like(psi)
    pip_x, pip_z = -s * J * np.sin(t1), -J - J * np.cos(t1)
    a12 = t1 + t2
    return {
        "yaw_a": _yaw(z0, z0, psi), "yaw_b": _yaw(z0, z0 - CAP_YAW, psi),
        "mid_a": _yaw(z0, z0 - J, psi),
        "mid_b": _yaw(-s * CAP_MID * np.sin(t1), -J - CAP_MID * np.cos(t1), psi),
        "dist_a": _yaw(pip_x, pip_z, psi),
        "dist_b": _yaw(pip_x - s * CAP_DIST * np.sin(a12), pip_z - CAP_DIST * np.cos(a12), psi),
        "pad": _yaw(pip_x - s * P * np.sin(a12), pip_z - P * np.cos(a12), psi),
        # palmar face normal of the distal link: the direction the pad moves under flexion
        "palmar": _yaw(-s * np.cos(a12), np.sin(a12), psi),
    }


def ik(finger: str, T: np.ndarray):
    """Both elbow branches for pad-centre targets T (..., 3) relative to the mount.

    Returns q (..., 2, 3) and the reach mask (...). The yaw puts the flexion plane through the target;
    MCP and PIP then solve a planar two-link problem with links J and P from the MCP axis."""
    s = FLEX[finger]
    X, Y, Z = T[..., 0], T[..., 1], T[..., 2]
    rho = np.hypot(Y, Z)
    psi = np.arctan2(Y, -Z)
    u, v = -s * X, rho - J
    c2 = (u * u + v * v - J * J - P * P) / (2.0 * J * P)
    ok = np.abs(c2) <= 1.0
    t2 = np.arccos(np.clip(c2, -1.0, 1.0))
    q = np.empty(T.shape[:-1] + (2, 3))
    for k, sg in enumerate((1.0, -1.0)):
        tt2 = sg * t2
        q[..., k, 0] = psi
        q[..., k, 1] = np.arctan2(u, v) - np.arctan2(P * np.sin(tt2), J + P * np.cos(tt2))
        q[..., k, 2] = tt2
    return q, ok


def pick_branch(q: np.ndarray, ok: np.ndarray, lim: np.ndarray, margin: float):
    """Per target, the branch inside the limits (with margin) with the larger smallest margin.

    Returns q (..., 3), feasible (...), the smallest margin (rad, ...) and per-joint margins (..., 3)."""
    mj = np.minimum(q - lim[:, 0], lim[:, 1] - q)          # (..., 2, 3)
    mmin = mj.min(-1)                                         # (..., 2)
    good = (mmin >= margin) & ok[..., None]
    score = np.where(good, mmin, -np.inf)
    k = np.argmax(score, -1)
    take = lambda a: np.take_along_axis(a, k[..., None, None], -2)[..., 0, :]  # noqa: E731
    qq, mm = take(q), take(mj)
    return qq, good.any(-1), mm.min(-1), mm


def jac_pad(finger: str, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """3x3 pad-position Jacobian (columns yaw, mcp, pip; m/rad), finite differences of fk."""
    base = fk(finger, q)["pad"]
    cols = []
    for j in range(3):
        dq = np.zeros(3)
        dq[j] = eps
        cols.append((fk(finger, q + dq)["pad"] - base) / eps)
    return np.stack(cols, -1)


def jac_cond(finger: str, q: np.ndarray, eps: float = 1e-6):
    """Condition number and smallest singular value (m/rad) of the pad Jacobian."""
    sv = np.linalg.svd(jac_pad(finger, q, eps), compute_uv=False)
    return sv[..., 0] / np.maximum(sv[..., -1], 1e-12), sv[..., -1]


# --- distances ---------------------------------------------------------------------------------
def seg_seg(p1, q1, p2, q2):
    """Closest distance between segments p1q1 and p2q2 (..., 3); degenerate segments allowed."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    tiny = 1e-14
    A, E = np.where(a > tiny, a, 1.0), np.where(e > tiny, e, 1.0)
    denom = a * e - b * b
    s = np.where(denom > tiny, np.clip((b * f - c * e) / np.where(denom > tiny, denom, 1.0), 0, 1), 0.0)
    t = (b * s + f) / E
    s = np.where(t < 0, np.clip(-c / A, 0, 1), np.where(t > 1, np.clip((b - c) / A, 0, 1), s))
    t = np.clip(t, 0, 1)
    # a point against a segment (Ericson 5.1.9): the second, then the first, degenerate
    s, t = np.where(e <= tiny, np.clip(-c / A, 0, 1), s), np.where(e <= tiny, 0.0, t)
    s, t = np.where(a <= tiny, 0.0, s), np.where(a <= tiny, np.clip(f / E, 0, 1), t)
    return np.linalg.norm(p1 + d1 * s[..., None] - (p2 + d2 * t[..., None]), axis=-1)


GEOMS = (("yaw_a", "yaw_b"), ("mid_a", "mid_b"), ("dist_a", "dist_b"), ("pad", "pad"))


def finger_clearance(pts: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """Smallest surface-to-surface distance between geoms of different fingers (all radius R)."""
    best = None
    for i, fa in enumerate(FINGERS):
        for fb in FINGERS[i + 1:]:
            for ga in GEOMS:
                for gb in GEOMS:
                    dd = seg_seg(pts[fa][ga[0]], pts[fa][ga[1]], pts[fb][gb[0]], pts[fb][gb[1]]) - 2 * R
                    best = dd if best is None else np.minimum(best, dd)
    return best


def _point_object(p, obj):
    """Signed distance from points p (..., 3) to the object's surface (negative inside)."""
    c = obj["centre"]
    if obj["shape"] == "sphere":
        return np.linalg.norm(p - c, axis=-1) - obj["r"]
    # cylinder, axis along obj["axis"] (unit), half-length hl
    ax = obj["axis"]
    w = p - c
    along = (w * ax).sum(-1)
    radial = np.linalg.norm(w - along[..., None] * ax, axis=-1) - obj["r"]
    dy = np.abs(along) - obj["hl"]
    inside = (dy <= 0) & (radial <= 0)
    out = np.hypot(np.maximum(dy, 0), np.maximum(radial, 0))
    return np.where(inside, np.maximum(dy, radial), out)


def object_clearance(pts_f: dict[str, np.ndarray], obj, n: int = 9) -> np.ndarray:
    """Smallest distance from a finger's non-pad capsules to the object surface, minus R."""
    best = None
    for a, b in GEOMS[:3]:
        for k in range(n):
            u = k / (n - 1)
            dd = _point_object(pts_f[a] * (1 - u) + pts_f[b] * u, obj) - R
            best = dd if best is None else np.minimum(best, dd)
    return best


# --- grasps --------------------------------------------------------------------------------------
def object_spec(shape: str, d: float, length: float = 0.100, centre=(0.0, 0.0, 0.0)) -> dict:
    o = {"shape": shape, "d": d, "r": d / 2.0, "centre": np.asarray(centre, float)}
    if shape == "cylinder":
        o.update(axis=np.array([0.0, 1.0, 0.0]), hl=length / 2.0, length=length)
    return o


def contact_targets(shape: str, x_sep: float, y_sep: float, r: float, spread: float = 0.0):
    """Pad-centre targets relative to the object centre, the object centre relative to the pinch
    midpoint (x, y), for an equator grasp. Cylinder (axis along y): thumb from -x at the midpoint,
    index and middle from +x at +-spread along the axis (fit_real_v1_pose.tip_targets, elevation 0).
    Sphere: centre under the mounts' circumcentre, each pad on the equator toward its own mount."""
    rr = r + R + GAP
    if shape == "cylinder":
        return {"thumb": np.array([-rr, 0.0, 0.0]), "index": np.array([rr, spread, 0.0]),
                "middle": np.array([rr, -spread, 0.0])}, np.zeros(3)
    c = np.array([circumcentre_x(x_sep, y_sep), 0.0, 0.0])
    out = {}
    for f, m in mounts(x_sep, y_sep).items():
        v = m - c
        v[2] = 0.0
        out[f] = rr * v / np.linalg.norm(v)
    return out, c


def grasp_scan(x_sep, y_sep, shape, d, lim, spreads=(None,), h_grid=None, length=0.100):
    """Every (straddle, palm height) of an equator grasp, with the gates. Palm height h is the mounting
    plane's height above the object centre (fit_real_v1_pose's grip depth)."""
    if h_grid is None:
        h_grid = np.arange(-0.040, 0.1201, 0.0005)
    M = mounts(x_sep, y_sep)
    r = d / 2.0
    rows = []
    for sp in spreads:
        rel, c_xy = contact_targets(shape, x_sep, y_sep, r, sp or 0.0)
        H = h_grid
        centre = np.stack([np.full_like(H, c_xy[0]), np.full_like(H, c_xy[1]), -H], -1)
        obj = object_spec(shape, d, length, centre=centre)
        q, feas, mmin, pts = {}, np.ones_like(H, bool), np.full_like(H, np.inf), {}
        ext, cond, smin, angle = {}, {}, {}, {}
        for f in FINGERS:
            T = centre + rel[f] - M[f]
            qq, okk = ik(f, T)
            qf, ff, mm, _ = pick_branch(qq, okk, lim[f], LIMIT_MARGIN)
            q[f], feas, mmin = qf, feas & ff, np.minimum(mmin, mm)
            pts[f] = fk(f, qf)
            # extension left: how much further the pad centre can move away from the MCP before the
            # finger is straight -- the reach margin
            mcp = pts[f]["mid_a"]
            ext[f] = CHAIN - np.linalg.norm(pts[f]["pad"] - mcp, axis=-1)
            cond[f], smin[f] = jac_cond(f, qf)
            # angle between the fingertip's palmar normal and the object's inward surface normal at the
            # contact: 0 = the pad face-on, 90 = the contact at the edge of the pad region (front half)
            n_in = centre - (pts[f]["pad"] + M[f])
            if shape == "cylinder":
                n_in[..., 1] = 0.0
            n_in = n_in / np.linalg.norm(n_in, axis=-1, keepdims=True)
            angle[f] = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n_in, pts[f]["palmar"]), -1, 1)))
        # every geom in the palm frame for the clearance checks
        world = {f: {k: v + M[f] for k, v in pts[f].items() if k != "palmar"} for f in FINGERS}
        ff_clear = finger_clearance(world)
        ob_clear = np.min([object_clearance(world[f], obj) for f in FINGERS], axis=0)
        plate = PLATE_BOTTOM + H - r
        face = np.max([angle[f] for f in FINGERS], axis=0) <= 90.0
        ok = feas & face & (ff_clear >= CLEAR_GATE) & (ob_clear >= OBJ_CLEAR) & (plate >= OBJ_CLEAR)
        rows.append(dict(spread=sp, h=H, ok=ok, reach=feas, face=face, ff_clear=ff_clear, ob_clear=ob_clear,
                         plate=plate, mmin=mmin, q=q, ext=ext, cond=cond, smin=smin, angle=angle))
    return rows


def summarise(rows, x_sep, y_sep, shape, d) -> dict:
    """The cell's verdict and its reference grasp: among feasible (straddle, h), the most centred posture
    (largest smallest joint margin), and the deepest palm (fit_real_v1_pose's choice)."""
    out = {"x_sep_mm": round(x_sep * 1000, 2), "y_sep_mm": round(y_sep * 1000, 2), "shape": shape,
           "d_mm": round(d * 1000, 2), "feasible": False}
    best, deep, dex, n_ok, n_reach = None, None, None, 0, 0
    bands = []
    for row in rows:
        idx = np.flatnonzero(row["ok"])
        n_ok += idx.size
        n_reach += int(row["reach"].sum())
        if idx.size:
            bands.append({"spread_mm": None if row["spread"] is None else round(row["spread"] * 1000, 1),
                          "h_lo_mm": round(float(row["h"][idx].min()) * 1000, 1),
                          "h_hi_mm": round(float(row["h"][idx].max()) * 1000, 1), "n": int(idx.size)})
            k = idx[np.argmax(row["mmin"][idx])]
            if best is None or row["mmin"][k] > best[0]["mmin"][best[1]]:
                best = (row, k)
            k2 = idx[np.argmax(row["h"][idx])]
            if deep is None or row["h"][k2] > deep[0]["h"][deep[1]] + 1e-9:
                deep = (row, k2)
            sm = np.min([row["smin"][f] for f in FINGERS], axis=0)
            k3 = idx[np.argmax(sm[idx])]
            if dex is None or sm[k3] > dex[2]:
                dex = (row, k3, sm[k3])
    out["n_feasible"] = n_ok
    out["n_reach_only"] = n_reach
    out["bands"] = bands
    if best is None:
        # why not: the first gate that empties the set
        why = []
        for row in rows:
            if not row["reach"].any():
                why.append("reach")
            elif not (row["reach"] & row["face"]).any():
                why.append("contact_angle")
            elif not (row["reach"] & (row["ff_clear"] >= CLEAR_GATE)).any():
                why.append("finger_clearance")
            elif not (row["reach"] & (row["ff_clear"] >= CLEAR_GATE) & (row["ob_clear"] >= OBJ_CLEAR)).any():
                why.append("link_on_object")
            else:
                why.append("palm_on_object")
        out["fail"] = max(set(why), key=why.count)
        return out
    out["feasible"] = True

    def pose(row, k):
        return {"spread_mm": None if row["spread"] is None else round(row["spread"] * 1000, 1),
                "h_mm": round(float(row["h"][k]) * 1000, 1),
                "q_deg": {f: [round(float(v), 2) for v in np.degrees(row["q"][f][k])] for f in FINGERS},
                "min_margin_deg": round(float(np.degrees(row["mmin"][k])), 2),
                "ext_mm": {f: round(float(row["ext"][f][k]) * 1000, 2) for f in FINGERS},
                "cond": {f: round(float(row["cond"][f][k]), 2) for f in FINGERS},
                "sigma_min_mm": {f: round(float(row["smin"][f][k]) * 1000, 2) for f in FINGERS},
                "contact_angle_deg": {f: round(float(row["angle"][f][k]), 1) for f in FINGERS},
                "ff_clear_mm": round(float(row["ff_clear"][k]) * 1000, 2),
                "ob_clear_mm": round(float(row["ob_clear"][k]) * 1000, 2),
                "plate_clear_mm": round(float(row["plate"][k]) * 1000, 2)}
    out["centred"] = pose(*best)
    out["deepest"] = pose(*deep)
    out["dexterous"] = pose(*dex[:2])
    return out


# --- fixed-contact precision-manipulation workspace ------------------------------------------------
def rotvec_grid(step_deg: float = 15.0, max_deg: float = 45.0) -> np.ndarray:
    """Rotation vectors on a cubic grid of `step_deg`, inside a ball of `max_deg` (123 at 15/45)."""
    k = int(round(max_deg / step_deg))
    ijk = np.array([(i, j, l) for i in range(-k, k + 1) for j in range(-k, k + 1) for l in range(-k, k + 1)], float)
    v = ijk * step_deg
    return np.radians(v[np.linalg.norm(v, axis=1) <= max_deg + 1e-9])


def rotmat(rv) -> np.ndarray:
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(rv, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


POS_STEP = 0.003
POS_GRID = np.stack(np.meshgrid(np.arange(-0.090, 0.0901, POS_STEP), np.arange(-0.090, 0.0901, POS_STEP),
                                np.arange(-0.150, 0.0201, POS_STEP), indexing="ij"), -1).reshape(-1, 3)


def feasible_poses(x_sep, y_sep, shape, d, rel, Rm, lim, centres=POS_GRID, length=0.100):
    """Mask over object centres (palm frame) at orientation Rm: with each pad centre fixed in the object frame at
    rel[f] (a spherical joint at the pad centre, Borras and Dollar's point contact), every finger reaches it within
    its joint limits, the object's surface normal at the contact lies in the front half of the fingertip (within
    90 deg of the palmar normal), no two fingers interpenetrate, no finger link enters the object, and the object
    stays below the palm plate."""
    M = mounts(x_sep, y_sep)
    r = d / 2.0
    T = {f: centres + Rm @ rel[f] - M[f] for f in FINGERS}
    ok = np.ones(len(centres), bool)
    for f in FINGERS:                                   # a pad farther than the straight chain is out of reach
        ok &= np.einsum("ij,ij->i", T[f], T[f]) <= (2 * J + P) ** 2 + 1e-12
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return ok
    q = {}
    good = np.ones(idx.size, bool)
    for f in FINGERS:
        qq, okk = ik(f, T[f][idx])
        q[f], ff, _, _ = pick_branch(qq, okk, lim[f], 0.0)
        good &= ff
    idx, q = idx[good], {f: v[good] for f, v in q.items()}
    if idx.size == 0:
        ok[:] = False
        return ok
    c = centres[idx]
    axis = Rm @ np.array([0.0, 1.0, 0.0])
    pts = {f: fk(f, q[f]) for f in FINGERS}
    good = np.ones(idx.size, bool)
    for f in FINGERS:                                   # contact normal in the front half of the tip
        pad = pts[f]["pad"] + M[f]
        n_in = c - pad
        if shape == "cylinder":
            n_in = n_in - (n_in @ axis)[:, None] * axis
        n_in /= np.linalg.norm(n_in, axis=1, keepdims=True)
        good &= np.einsum("ij,ij->i", n_in, pts[f]["palmar"]) >= 0.0
    world = {f: {k: v + M[f] for k, v in pts[f].items() if k != "palmar"} for f in FINGERS}
    good &= finger_clearance(world) >= 0.0
    obj = object_spec(shape, d, length, centre=c)
    if shape == "cylinder":
        obj["axis"] = axis
        top = c[:, 2] + obj["hl"] * abs(axis[2]) + r * math.sqrt(max(0.0, 1 - axis[2] ** 2))
    else:
        top = c[:, 2] + r
    good &= np.min([object_clearance(world[f], obj) for f in FINGERS], axis=0) >= 0.0
    good &= PLATE_BOTTOM - top >= 0.0
    ok[:] = False
    ok[idx[good]] = True
    return ok


def axis_range(x_sep, y_sep, shape, d, rel, lim, axis: int, step_deg: float = 5.0, max_deg: float = 90.0):
    """Largest |angle| reachable by rotating the object about palm axis `axis` (0 x, 1 y, 2 z) with its centre
    free on the position grid, contiguous from 0, both signs; and the reachable volume (cm^3) per angle."""
    vols = {}
    reach = {}
    for sgn in (1, -1):
        best = 0.0
        for a in np.arange(0.0, max_deg + 1e-9, step_deg):
            rv = np.zeros(3)
            rv[axis] = sgn * math.radians(a)
            n = int(feasible_poses(x_sep, y_sep, shape, d, rel, rotmat(rv), lim).sum())
            vols[round(sgn * a, 1)] = round(n * POS_STEP ** 3 * 1e6, 3)
            if n == 0:
                break
            best = a
        reach[sgn] = best
    return reach[1], reach[-1], dict(sorted(vols.items()))


def workspace_cell(x_sep, y_sep, shape, d, lim, spread=None, rv=None):
    rel, _ = contact_targets(shape, x_sep, y_sep, d / 2.0, spread or 0.0)
    rv = rotvec_grid() if rv is None else rv
    counts = np.array([int(feasible_poses(x_sep, y_sep, shape, d, rel, rotmat(v), lim).sum()) for v in rv])
    out = {"poses": int(counts.sum()), "n_orient_reached": int((counts > 0).sum()), "n_orient": len(rv),
           "vol_identity_cm3": round(counts[0 if np.linalg.norm(rv[0]) < 1e-12 else
                                            int(np.argmin(np.linalg.norm(rv, axis=1)))] * POS_STEP ** 3 * 1e6, 3)}
    for ax, name in ((0, "x"), (1, "y"), (2, "z")):
        p, m, vols = axis_range(x_sep, y_sep, shape, d, rel, lim, ax)
        out[f"rot_{name}_deg"] = [p, m]
        if ax == 0:
            out["vol_by_rot_x"] = vols
    return out


def cmd_workspace(args) -> int:
    """Step 3(i): the fixed-contact workspace per (layout, object) cell of the chosen families, rows appended
    (fsynced) to workspace.jsonl; resumable; --shard i/n splits the cell list."""
    lim = limits(False)
    R_ = json.loads(Path(args.reach).read_text())
    cells = {(c["tag"], c["shape"], c["d_mm"]): c for c in R_["cells"]}
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["tag"], r["shape"], r["d_mm"], r["spread_mm"]))
    todo = []
    seen = set()
    for kind in args.families.split(","):
        for lay in R_["families"][kind]:
            for shape, ds in OBJECT_SET.items():
                for dmm in ds:
                    c = cells.get((lay["tag"], shape, float(dmm)))
                    if not c or not c["feasible"]:
                        continue
                    sps = [None] if shape == "sphere" else [b["spread_mm"] for b in c["bands"]
                                                            if b["spread_mm"] in args.spreads]
                    for sp in sps:
                        key = (lay["tag"], shape, float(dmm), sp)
                        if key in seen:
                            continue
                        seen.add(key)
                        todo.append((lay, shape, dmm, sp))
    k, n = (int(v) for v in args.shard.split("/"))
    todo = [t for i, t in enumerate(todo) if i % n == k]
    print(f"{len(todo)} cells in shard {args.shard}, {sum(1 for t in todo if (t[0]['tag'], t[1], float(t[2]), t[3]) in done)} done")
    for lay, shape, dmm, sp in todo:
        if (lay["tag"], shape, float(dmm), sp) in done:
            continue
        t0 = time.time()
        w = workspace_cell(lay["x_sep"] / 1000, lay["y_sep"] / 1000, shape, dmm / 1000, lim,
                           None if sp is None else sp / 1000)
        row = {"tag": lay["tag"], "x_sep_mm": lay["x_sep"], "y_sep_mm": lay["y_sep"],
               "palm_radius_mm": lay["palm_radius"], "shape": shape, "d_mm": float(dmm), "spread_mm": sp,
               **w, "seconds": round(time.time() - t0, 1)}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"{lay['tag']} {shape:8} d {dmm:5.1f} sp {sp}: poses {w['poses']:7d} vol0 {w['vol_identity_cm3']:7.2f} "
              f"rot x {w['rot_x_deg']} y {w['rot_y_deg']} z {w['rot_z_deg']}  {row['seconds']} s", flush=True)
    return 0


# --- commands ------------------------------------------------------------------------------------
def cmd_check(args) -> int:
    """Analytic FK against the compiled MuJoCo scene at random joint angles on three layouts, and the
    analytic finger-finger clearance against mj_geomDistance."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(BASE_SCENE))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    worst = 0.0
    worst_clear = 0.0
    for (xs, ys) in ((0.100, 0.110), (0.040, 0.050), (0.160, 0.170)):
        vec = morph_vector(xs, ys)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for f, (dx, dy) in zip(FINGERS, ((vec[0], vec[1]), (vec[3], vec[4]), (vec[6], vec[7]))):
            d.qpos[m.jnt_qposadr[m.joint(f"{f}_x").id]] = dx
            d.qpos[m.jnt_qposadr[m.joint(f"{f}_y").id]] = dy
        for _ in range(200):
            q = {f: rng.uniform(SIM_ROM[:, 0], SIM_ROM[:, 1]) for f in FINGERS}
            for f in FINGERS:
                for j, v in zip(("yaw", "mcp", "pip"), q[f]):
                    d.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]] = v
            mujoco.mj_forward(m, d)
            palm = d.body("palm_pose").xpos
            pts = {}
            for f in FINGERS:
                mount = d.body(f"{f}_mount").xpos - palm
                p = fk(f, q[f])
                pts[f] = {k: v + mount for k, v in p.items()}
                for body, key in ((f"{f}_mcp_frame", "mid_a"), (f"{f}_pip_frame", "dist_a"),
                                  (f"{f}_tip", "pad")):
                    worst = max(worst, float(np.linalg.norm(d.body(body).xpos - palm - pts[f][key])))
                # IK round trip
                qq, ok = ik(f, p["pad"])
                rt = min(float(np.linalg.norm(fk(f, qq[k])["pad"] - p["pad"])) for k in range(2))
                worst = max(worst, rt)
            # clearance against mj_geomDistance between finger geoms (no palm, no object)
            own = {}
            for g in range(m.ngeom):
                b = m.geom_bodyid[g]
                while b > 0:
                    nm = m.body(b).name
                    hit = next((f for f in FINGERS if nm.startswith(f)), None)
                    if hit:
                        own[g] = hit
                        break
                    b = m.body_parentid[b]
            ref = np.inf
            for ga, fa in own.items():
                for gb, fb in own.items():
                    if fa < fb:
                        ref = min(ref, mujoco.mj_geomDistance(m, d, ga, gb, 0.25, None))
            mine = float(finger_clearance(pts))
            if ref < 0.2:
                worst_clear = max(worst_clear, abs(mine - ref))
    print(f"FK/IK worst position error {worst * 1e3:.4f} mm over 600 random poses on 3 layouts")
    print(f"finger-finger clearance worst |analytic - mj_geomDistance| {worst_clear * 1e3:.4f} mm")
    return 0 if worst < 1e-6 and worst_clear < 2e-4 else 1


def cylinder_spreads(length: float = 0.100):
    """fit_real_v1_pose's straddle candidates: down from 0.85 of the half-length in 5 mm steps to
    about 20 mm (42.5, 37.5, 32.5, 27.5, 22.5 mm on the 100 mm shaft)."""
    hi = min(0.85 * length / 2.0, length / 2.0 - 0.005)
    n = max(1, int(round((hi - 0.020) / 0.005)))
    return tuple(round(hi - k * 0.005, 4) for k in range(n + 1))


D_CYL = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22.5, 25, 27.5, 30, 32, 35, 40, 45, 50, 55, 60, 63, 70, 80, 90,
         100, 110, 120, 130, 140, 150)
D_SPH = D_CYL + (125, 160, 170, 180, 190, 200, 210, 220)
D_SPH = tuple(sorted(set(D_SPH)))


def cmd_reach(args) -> int:
    lim = limits(args.sim_rom)
    t0 = time.time()
    out = {"meta": {"date": "2026-10-08", "script": "scripts/hand_object_scale_kin.py reach",
                    "limits": "scene ROM" if args.sim_rom else "servo calibrated range meet scene ROM",
                    "limits_deg": {f: np.round(np.degrees(lim[f]), 2).tolist() for f in FINGERS},
                    "limit_margin_deg": round(math.degrees(LIMIT_MARGIN), 2),
                    "clear_gate_mm": CLEAR_GATE * 1000, "obj_clear_mm": OBJ_CLEAR * 1000,
                    "gap_mm": GAP * 1000, "cyl_length_mm": args.length * 1000,
                    "spreads_mm": [s * 1000 for s in cylinder_spreads(args.length)]},
           "families": {}, "cells": []}
    seen = {}
    for kind in args.families.split(","):
        fam = family(kind)
        out["families"][kind] = fam
        for lay in fam:
            if lay["tag"] in seen:
                continue
            seen[lay["tag"]] = lay
    for tag, lay in seen.items():
        xs, ys = lay["x_sep"] / 1000.0, lay["y_sep"] / 1000.0
        for shape, ds in (("cylinder", D_CYL), ("sphere", D_SPH)):
            spreads = cylinder_spreads(args.length) if shape == "cylinder" else (None,)
            for dmm in ds:
                rows = grasp_scan(xs, ys, shape, dmm / 1000.0, lim, spreads, length=args.length)
                cell = summarise(rows, xs, ys, shape, dmm / 1000.0)
                cell["tag"] = tag
                cell["palm_radius_mm"] = lay["palm_radius"]
                out["cells"].append(cell)
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    n = len(out["cells"])
    print(f"{n} cells over {len(seen)} layouts in {out['meta']['seconds']} s -> {path}")
    # the diagonal family's reachable diameter range, per shape
    for shape in ("cylinder", "sphere"):
        print(f"\n{shape}: reachable diameters (mm) per layout of the 1-D family")
        for lay in out["families"].get("diag", []):
            ok = [c["d_mm"] for c in out["cells"] if c["tag"] == lay["tag"] and c["shape"] == shape
                  and c["feasible"]]
            fails = {}
            for c in out["cells"]:
                if c["tag"] == lay["tag"] and c["shape"] == shape and not c["feasible"]:
                    fails[c["fail"]] = fails.get(c["fail"], 0) + 1
            rng_s = f"{min(ok):6.1f} .. {max(ok):6.1f}" if ok else "   none        "
            print(f"  t {lay['t']:+.3f}  x_sep {lay['x_sep']:6.1f}  y_sep {lay['y_sep']:6.1f}  "
                  f"rho {lay['palm_radius']:6.1f}  d {rng_s}  ({len(ok)} ok)  fails {fails}")
    return 0


# --- the object set ----------------------------------------------------------------------------------
OBJECT_SET = {"cylinder": (6, 10, 16, 25, 40, 50, 63, 80), "sphere": (10, 16, 25, 40, 50, 63, 80, 100, 125, 160)}
PLA, STEEL = 1.24, 7.85       # g/cm^3
TARGET_G = 25.0               # the simulated screwdriver's mass (25 mm x 100 mm at 500 kg/m^3: 24.5 g)
FLAG_G = 2.5                  # printed flag carrying the 40 mm tag36h11 (src/morphohand/bench/tags.py)
# Bench build per object: (body, ballast). Cylinders are 100 mm long. Body kinds: 'rod' steel rod,
# 'tube' PLA over a steel core of diameter c, 'solid' PLA at infill f, 'shell' PLA walls w (mm) at infill f.
# Spheres: 'ball' steel ball, 'shell' PLA (two hemispheres when f == 0); ballast a steel ball of diameter b
# glued at the centre, or extra grams g in the cylinder's core. Chosen to put each object at TARGET_G with
# the flag where the geometry allows it.
RECIPES = {
    ("cylinder", 6): dict(body="rod"),
    ("cylinder", 10): dict(body="tube", c=5.0),
    ("cylinder", 16): dict(body="solid", f=0.90),
    ("cylinder", 25): dict(body="shell", w=1.2, f=0.10, g=4.5),
    ("cylinder", 40): dict(body="shell", w=0.8, f=0.05),
    ("cylinder", 50): dict(body="shell", w=0.8, f=0.0, g=3.0),
    ("cylinder", 63): dict(body="shell", w=0.8, f=0.0),
    ("cylinder", 80): dict(body="shell", w=0.8, f=0.0),
    ("sphere", 10): dict(body="ball"),
    ("sphere", 16): dict(body="ball"),
    ("sphere", 25): dict(body="shell", w=1.2, f=0.0, b=17.0),
    ("sphere", 40): dict(body="shell", w=1.2, f=0.0, b=15.0),
    ("sphere", 50): dict(body="shell", w=1.2, f=0.0, b=14.0),
    ("sphere", 63): dict(body="shell", w=1.2, f=0.0, b=10.0),
    ("sphere", 80): dict(body="shell", w=0.8, f=0.0, b=9.0),
    ("sphere", 100): dict(body="shell", w=0.8, f=0.0),
    ("sphere", 125): dict(body="shell", w=0.8, f=0.0),
    ("sphere", 160): dict(body="shell", w=0.8, f=0.0),
}


def recipe_mass(shape: str, d: float, rec: dict, length: float = 100.0) -> tuple[float, str]:
    """Grams of the bench build (flag included) and a one-line description."""
    r = d / 2.0
    vol = lambda rr, ll: math.pi * rr * rr * ll / 1000.0          # noqa: E731 cylinder, cm^3
    ball = lambda dd: 4 / 3 * math.pi * (dd / 2) ** 3 / 1000.0     # noqa: E731
    if shape == "cylinder":
        k = rec["body"]
        if k == "rod":
            g, txt = STEEL * vol(r, length), f"steel rod {d:g} mm"
        elif k == "tube":
            c = rec["c"] / 2
            g = PLA * (vol(r, length) - vol(c, length)) + STEEL * vol(c, length)
            txt = f"PLA tube on a {rec['c']:g} mm steel rod"
        elif k == "solid":
            g, txt = PLA * rec["f"] * vol(r, length), f"PLA, {rec['f']:.0%} infill"
        else:
            w, f = rec["w"], rec["f"]
            shell = (math.pi * d * length * w + 2 * math.pi * r * r * w) / 1000.0
            g = PLA * (shell + f * vol(r - w, length - 2 * w))
            txt = f"PLA, {w:g} mm walls, {f:.0%} infill"
        if rec.get("g"):
            g += rec["g"]
            txt += f", {rec['g']:g} g steel at the centre"
        return g + FLAG_G, txt
    k = rec["body"]
    if k == "ball":
        g, txt = STEEL * ball(d), f"steel ball {d:g} mm"
    else:
        w, f = rec["w"], rec["f"]
        g = PLA * (4 * math.pi * r * r * w / 1000.0 + f * ball(d - 2 * w))
        txt = f"PLA, {w:g} mm walls" + (f", {f:.0%} infill" if f else ", hollow halves")
        if rec.get("b"):
            g += STEEL * ball(rec["b"])
            txt += f", {rec['b']:g} mm steel ball at the centre"
    return g + FLAG_G, txt


def cmd_objects(args) -> int:
    """family.json (layouts, their gantry realisation and checks) and objects.json (the printed set and
    which layouts reach each object), from reach.json."""
    from manta_hand.plan import mount_violations
    R_ = json.loads(Path(args.reach).read_text())
    cells = {(c["tag"], c["shape"], c["d_mm"]): c for c in R_["cells"]}
    fam_out = {"meta": {"date": "2026-10-08", "source": str(Path(args.reach).relative_to(ROOT)),
                        "realisation": "thumb and pair moved by equal and opposite x from the CAD-nominal "
                                       "mounts; index and middle mirrored in y"}, "families": {}}
    for kind, lays in R_["families"].items():
        rows = []
        for lay in lays:
            vec = morph_vector(lay["x_sep"] / 1000, lay["y_sep"] / 1000)
            m = {"thumb": (-50 + vec[0] * 1000, vec[1] * 1000), "index": (50 + vec[3] * 1000, 55 + vec[4] * 1000),
                 "middle": (50 + vec[6] * 1000, -55 + vec[7] * 1000)}
            bad = [str(v) for f, (x, y) in m.items() for v in mount_violations(f, x, y)]
            row = dict(lay, mounts_mm={f: [round(x, 2), round(y, 2)] for f, (x, y) in m.items()},
                       morph_vector=[round(v, 6) for v in vec], gantry_violations=bad)
            for shape, ds in OBJECT_SET.items():
                ok = []
                for dmm in ds:
                    c = cells.get((lay["tag"], shape, float(dmm)))
                    if c and c["feasible"]:
                        ok.append({"d_mm": dmm, "ff_clear_mm": c["centred"]["ff_clear_mm"],
                                   "min_margin_deg": c["centred"]["min_margin_deg"],
                                   "h_band_mm": [min(b["h_lo_mm"] for b in c["bands"]),
                                                 max(b["h_hi_mm"] for b in c["bands"])]})
                row[f"reach_{shape}"] = ok
            rows.append(row)
        fam_out["families"][kind] = rows
    (OUT_DIR / "family.json").write_text(json.dumps(fam_out, indent=1))

    objs = []
    for shape, ds in OBJECT_SET.items():
        for dmm in ds:
            g, recipe = recipe_mass(shape, dmm, RECIPES[(shape, dmm)])
            o = {"shape": shape, "d_mm": dmm, "length_mm": 100.0 if shape == "cylinder" else None,
                 "mass_g": round(g, 1), "weight_N": round(g * 9.81e-3, 3), "recipe": recipe,
                 "on_target": abs(g - TARGET_G) <= 0.1 * TARGET_G}
            for kind in ("diag", "xsep", "ysep", "grid"):
                lays = R_["families"].get(kind, [])
                hit = [lay for lay in lays if cells.get((lay["tag"], shape, float(dmm)), {}).get("feasible")]
                if kind == "diag":
                    o["diag_t"] = [lay["t"] for lay in hit]
                o[f"n_{kind}"] = [len(hit), len(lays)]
            objs.append(o)
    (OUT_DIR / "objects.json").write_text(json.dumps({"meta": {
        "date": "2026-10-08", "pla_g_cm3": PLA, "steel_g_cm3": STEEL, "target_g": TARGET_G, "flag_g": FLAG_G,
        "note": "cylinders lie along palm y, 100 mm long (the screwdriver class); grasps on the equator; "
                "masses include the tag flag"},
        "objects": objs}, indent=1))
    print(f"{'shape':9}{'d':>6}{'mass g':>8}  recipe{'':48}diag t reachable          grid")
    for o in objs:
        t = o["diag_t"]
        ts = f"{min(t):+.3f}..{max(t):+.3f} ({len(t)})" if t else "none"
        print(f"{o['shape']:9}{o['d_mm']:>6}{o['mass_g']:>8.1f}  {o['recipe']:54}{ts:26}"
              f"{o['n_grid'][0]}/{o['n_grid'][1]}")
    viol = sum(len(r["gantry_violations"]) for rows in fam_out["families"].values() for r in rows)
    print(f"gantry violations over all layouts: {viol}")
    return 0


def mujoco_pose_scene(shape: str, d: float, length: float = 0.100):
    """The base real_v1 scene with its object replaced by a cylinder (along world y) or a sphere of
    diameter d, compiled, plus a function that poses (layout, object centre, finger angles) in it."""
    import mujoco
    import xml.etree.ElementTree as ET
    root = ET.parse(BASE_SCENE).getroot()
    body = root.find(".//body[@name='screwdriver_medium']")
    g = body.find("geom")
    if shape == "cylinder":
        g.set("size", f"{d / 2:.6f} {length / 2:.6f}")
    else:
        g.set("type", "sphere")
        g.set("size", f"{d / 2:.6f}")
    for kf in root.findall("keyframe"):
        root.remove(kf)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    dd = mujoco.MjData(m)
    adr = lambda n: m.jnt_qposadr[m.joint(n).id]  # noqa: E731
    obj_q = m.jnt_qposadr[m.body("screwdriver_medium").jntadr[0]]
    palm0 = m.body("palm_pose").pos.copy()

    def pose(x_sep, y_sep, centre_palm, q):
        """centre_palm: object centre in the palm frame (m); q: finger -> (yaw, mcp, pip)."""
        dd.qpos[:] = 0.0
        vec = morph_vector(x_sep, y_sep)
        for f, (dx, dy) in zip(FINGERS, ((vec[0], vec[1]), (vec[3], vec[4]), (vec[6], vec[7]))):
            dd.qpos[adr(f"{f}_x")], dd.qpos[adr(f"{f}_y")] = dx, dy
            for j, v in zip(("yaw", "mcp", "pip"), q[f]):
                dd.qpos[adr(f"{f}_{j}")] = v
        # palm frame origin at the pinch midpoint: the realisation's midpoint is at palm x 0, y 0
        palm_world = np.array([0.0, 0.0, 0.30])
        for k, n in enumerate(("palm_px", "palm_py", "palm_pz")):
            dd.qpos[adr(n)] = palm_world[k] - palm0[k]
        ctr = palm_world + centre_palm
        dd.qpos[obj_q:obj_q + 3] = ctr
        dd.qpos[obj_q + 3:obj_q + 7] = [0.70711, 0.70711, 0, 0] if shape == "cylinder" else [1, 0, 0, 0]
        mujoco.mj_forward(m, dd)
        return dd
    return m, dd, pose


def cmd_verify(args) -> int:
    """Analytic object and palm clearances against mj_geomDistance on the compiled scene, at the centred
    grasp of every feasible cell of the 1-D family for the object set."""
    import mujoco
    R_ = json.loads(Path(args.reach).read_text())
    lim = limits(False)
    worst = {"pad_gap": 0.0, "links": 0.0, "plate": 0.0, "fingers": 0.0}
    n = 0
    for shape, ds in OBJECT_SET.items():
        for dmm in ds:
            m, dd, pose = mujoco_pose_scene(shape, dmm / 1000.0)
            og = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "screwdriver_medium"][0]
            plate = m.geom("palm_plate").id
            own = {}
            for g in range(m.ngeom):
                b = m.geom_bodyid[g]
                while b > 0:
                    nm = m.body(b).name
                    hit = next((f for f in FINGERS if nm.startswith(f)), None)
                    if hit:
                        own[g] = (hit, m.body(m.geom_bodyid[g]).name.endswith("_tip"))
                        break
                    b = m.body_parentid[b]
            for lay in R_["families"]["diag"]:
                c = next((c for c in R_["cells"] if c["tag"] == lay["tag"] and c["shape"] == shape
                          and c["d_mm"] == float(dmm)), None)
                if not c or not c["feasible"]:
                    continue
                xs, ys = lay["x_sep"] / 1000, lay["y_sep"] / 1000
                g0 = c["centred"]
                sp = (g0["spread_mm"] or 0.0) / 1000
                rel, cxy = contact_targets(shape, xs, ys, dmm / 2000, sp)
                h = g0["h_mm"] / 1000
                centre = np.array([cxy[0], cxy[1], -h])
                q = {f: np.radians(g0["q_deg"][f]) for f in FINGERS}
                pose(xs, ys, centre, q)
                pads = [mujoco.mj_geomDistance(m, dd, og, g, 0.5, None) for g, (f, tip) in own.items() if tip]
                links = min(mujoco.mj_geomDistance(m, dd, og, g, 0.5, None) for g, (f, tip) in own.items() if not tip)
                pl = mujoco.mj_geomDistance(m, dd, og, plate, 0.5, None)
                ff = min(mujoco.mj_geomDistance(m, dd, ga, gb, 0.5, None) for ga, (fa, _) in own.items()
                         for gb, (fb, _) in own.items() if fa < fb)
                worst["pad_gap"] = max(worst["pad_gap"], max(abs(p - GAP) for p in pads))
                worst["links"] = max(worst["links"], abs(min(links, 0.5) - min(g0["ob_clear_mm"] / 1000, 0.5)))
                worst["plate"] = max(worst["plate"], abs(pl - g0["plate_clear_mm"] / 1000) if pl < 0.4 else 0.0)
                worst["fingers"] = max(worst["fingers"], abs(ff - g0["ff_clear_mm"] / 1000))
                n += 1
    print(f"{n} grasps checked against mj_geomDistance; worst |analytic - MuJoCo| in mm: "
          + ", ".join(f"{k} {v * 1000:.3f}" for k, v in worst.items()))
    return 0 if max(worst.values()) < 1e-3 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    v = sub.add_parser("verify")
    v.add_argument("--reach", default=str(OUT_DIR / "reach.json"))
    r = sub.add_parser("reach")
    r.add_argument("--families", default="diag,xsep,ysep,grid")
    r.add_argument("--length", type=float, default=0.100, help="cylinder length, m")
    r.add_argument("--sim-rom", action="store_true", help="scene ROM instead of the servo ranges")
    r.add_argument("--out", default=str(OUT_DIR / "reach.json"))
    o = sub.add_parser("objects")
    o.add_argument("--reach", default=str(OUT_DIR / "reach.json"))
    w = sub.add_parser("workspace")
    w.add_argument("--reach", default=str(OUT_DIR / "reach.json"))
    w.add_argument("--families", default="diag")
    w.add_argument("--spreads", type=lambda s: [float(v) for v in s.split(",")], default=[22.5, 32.5, 42.5],
                   help="cylinder straddles (mm) to evaluate, among those with a feasible grasp")
    w.add_argument("--shard", default="0/1")
    w.add_argument("--out", default=str(OUT_DIR / "workspace.jsonl"))
    a = ap.parse_args()
    return {"check": cmd_check, "reach": cmd_reach, "objects": cmd_objects, "verify": cmd_verify,
            "workspace": cmd_workspace}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

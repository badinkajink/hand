#!/usr/bin/env python3
"""The gantries' travel from a sweep of a DIP-link tag (docs/experiments/20261115-gantry_travel/lab/gantry_travel.py
sweep; owner, 2026-11-15: "we theoretically should be able to track travel").

Per point the tag's pose comes from both cameras' corners at once (sr2hand.tracking.pose.solve_multi, the lenses and
camera poses named in the sweep's header), and goes into the palm frame P through the arm's TCP pose and the tool chain
(T_PB = (T_BE T_EP)^-1, sr2hand.robot.tool_chain.SR2Hand.T_E_P); without the arm's pose it stays in the arm's base
frame. The station's palm coordinates differ from P by a translation (sr2_layouts.driver_from_P), so a sweep along its
x moves the tag along x_P. Per axis the tag centre is fitted with a line against the commanded coordinate:

    scale           measured mm per commanded mm along the line (1 when the steps per mm are right)
    direction       the angle between the line and the stage's axis in P (x_P or y_P)
    linearity       the residuals along the line (sd and largest), straightness the residuals across it
    backlash        the difference along the line between the down and up passes at the same command (mean and sd)
    span            the measured travel between the lowest and highest commands against the commanded span
    rotation        the tag's largest turn from its first pose (the finger's joints should hold it still)

Camera stack (cv2, scipy): ~/miniconda3/bin/python on the workstation, or /usr/bin/python3 on corvus.

    ~/miniconda3/bin/python scripts/hardware/gantry_travel_fit.py SWEEP.jsonl [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

ROOT = Path(__file__).resolve().parents[2]

from sr2hand.geometry.rotations import log_so3, tf_inv  # noqa: E402
from sr2hand.tracking import live  # noqa: E402
from sr2hand.tracking import pose as tp  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402

AXIS_P = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0])}


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    hdr = rows[0]
    pts = [r for r in rows[1:] if "n" in r]
    return hdr, pts


def tag_pose(row, tag, tag_mm, cams):
    """(T_BT, info): the tag's pose in the arm's base frame from every camera that saw it, or (None, info)."""
    tm = TG.plate_map("tag36h11", tag_mm, [tag], [(0.0, 0.0, 0.0)])
    obs, cl, Tl = [], [], []
    for sn, (cam, T_BC) in cams.items():
        c = row["cams"].get(sn, {}).get(str(tag))
        if c is not None:
            obs.append({tag: np.asarray(c["corners"], float)})
            cl.append(cam)
            Tl.append(T_BC)
    if not obs:
        return None, {"n_cams": 0}
    T_CT, _ = tp.solve_body(obs[0], tm, cl[0], max_view_deg=None)
    if T_CT is None:
        return None, {"n_cams": 0}
    return tp.solve_multi(obs, tm, cl, Tl, Tl[0] @ T_CT, max_view_deg=None)


def fit_axis(c, p, passes, axis, frame):
    """The line of tag centres p (N, 3) mm against the commanded coordinate c (N,) mm, and its residuals."""
    cm = c.mean()
    A = np.c_[np.ones_like(c), c - cm]
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)
    p0, a = coef[0], coef[1]
    s = float(np.linalg.norm(a))
    u = a / s
    r = p - (p0 + np.outer(c - cm, a))
    along = r @ u
    across = np.linalg.norm(r - np.outer(along, u), axis=1)
    out = {"points": int(len(c)), "scale": round(s, 5), "frame": frame,
           "linearity_mm": {"sd": round(float(along.std()), 4), "max": round(float(np.abs(along).max()), 4)},
           "straightness_mm": {"rms": round(float(np.sqrt((across ** 2).mean())), 4), "max": round(float(across.max()), 4)},
           "span_mm": {"commanded": round(float(c.max() - c.min()), 3),
                       "measured": round(float((p[np.argmax(c)] - p[np.argmin(c)]) @ u), 3)},
           "direction_P": np.round(u, 5).tolist()}
    if frame == "P":
        nom = AXIS_P[axis]
        out["angle_to_stage_axis_deg"] = round(float(np.degrees(np.arccos(min(1.0, abs(u @ nom))))), 3)
    up = {round(float(ci), 3): pi for ci, pi, ps in zip(c, p, passes) if ps == "up"}
    dn = {round(float(ci), 3): pi for ci, pi, ps in zip(c, p, passes) if ps == "down"}
    both = sorted(set(up) & set(dn))
    if both:
        d = np.array([(dn[k] - up[k]) @ u for k in both])
        out["backlash_mm"] = {"mean": round(float(d.mean()), 4), "sd": round(float(d.std()), 4), "pairs": len(both)}
    for name in ("up", "down"):
        m = np.array([ps == name for ps in passes])
        if m.sum() >= 3:
            cc = c[m]
            co, *_ = np.linalg.lstsq(np.c_[np.ones(m.sum()), cc - cc.mean()], p[m], rcond=None)
            out[f"scale_{name}"] = round(float(np.linalg.norm(co[1])), 5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sweep")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    hdr, pts = load(a.sweep)
    cams = live.load_cameras(ROOT / hdr["lens"], ROOT / hdr["handeye"])
    tag, tag_mm = int(hdr["tag"]), float(hdr["tag_mm"])
    T_EP = None
    if any(r.get("T_BE") for r in pts):
        from sr2hand.robot.tool_chain import SR2Hand
        T_EP = SR2Hand.load().T_E_P()
    rec, missing = [], []
    for r in pts:
        T_BT, info = tag_pose(r, tag, tag_mm, cams)
        if T_BT is None:
            missing.append(r["n"])
            continue
        if T_EP is not None and r.get("T_BE"):
            T = tf_inv(np.asarray(r["T_BE"], float) @ T_EP) @ T_BT
            frame = "P"
        else:
            T, frame = T_BT, "B"
        rec.append({"n": r["n"], "axis": r["axis"], "pass": r.get("pass", "up"), "cmd": r["cmd_mm"], "T": T,
                    "frame": frame, "n_cams": info.get("n_cams"), "rms_px": info.get("rms_px")})
    res = {"sweep": a.sweep, "finger": hdr["finger"], "tag": tag, "points": len(pts), "seen": len(rec),
           "not_seen": missing, "axes": {}}
    if rec:
        R0 = rec[0]["T"][:3, :3]
        res["rotation_max_deg"] = round(max(float(np.degrees(np.linalg.norm(log_so3(R0.T @ q["T"][:3, :3]))))
                                            for q in rec), 3)
        res["cams_per_point"] = {str(k): sum(1 for q in rec if q["n_cams"] == k) for k in (1, 2)}
        res["rms_px_max"] = round(max(q["rms_px"] or 0.0 for q in rec), 3)
    for ax in sorted({q["axis"] for q in rec}):
        k = "xy".index(ax)
        sel = [q for q in rec if q["axis"] == ax]
        if len(sel) < 3:
            continue
        c = np.array([q["cmd"][k] for q in sel])
        p = np.array([q["T"][:3, 3] for q in sel])
        res["axes"][ax] = fit_axis(c, p, [q["pass"] for q in sel], ax, sel[0]["frame"])
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

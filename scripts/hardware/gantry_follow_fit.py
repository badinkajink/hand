#!/usr/bin/env python3
"""The gantries' travel from the follow logs of the owner's DIP-tag session (2026-11-15 16:53-17:13;
docs/experiments/20261115-gantry_travel/lab/gantry_follow.py move; owner: "move each stepper to its max extent, track
with apriltags").

Per move, the moving finger's tag is located before the command (the 1.5 s before it) and after the stage stopped (from
1 s after the station went idle, for 2 s). Each location is the mean over the window of:

    joint     both cameras' corners solved at once (sr2hand.tracking.pose.solve_multi), over frame pairs within half a
              frame (the cameras run unsynchronized at 15 fps); depth from the two views, so the printed tag's size
              barely enters
    per sn    that camera's own single-tag solve as logged (T_BT, IPPE then LM), whose distance, and so whose travel,
              scales with the tag size assumed (14.0 mm)

The displacement goes into P through the arm's TCP pose and the tool chain (T_PB = (T_BE T_EP)^-1). Against the
commanded displacement u (the station's palm axes are P's, a translation apart) it gives the travel along u, the scale
(travel / commanded), the axis's turn about the palm normal z_P (signed, counterclockwise positive) and the height change
along z_P. The tag's edge from two-camera triangulation of its corners in every joint pair checks the print's scale.

    ~/.claude/bin/resguard.sh run --mem 2G -- ~/miniconda3/bin/python scripts/hardware/gantry_follow_fit.py \\
        [--logs logs/20261115-gantry_travel/follow] [--json docs/experiments/20261115-gantry_travel/data/FIT.json]
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

from sr2hand.geometry.rotations import tf_inv  # noqa: E402
from sr2hand.robot.tool_chain import SR2Hand  # noqa: E402
from sr2hand.tracking import live  # noqa: E402
from sr2hand.tracking import pose as tp  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402

TAGS = {"thumb": 24, "index": 25, "middle": 26}
TAG_MM = 14.0
HALF_FRAME_S = 0.5 / 15.0
BEFORE_S, SETTLE_S, AFTER_S = 1.5, 1.0, 2.0
NOTES = {"165655": "stalled at about 30 mm: the limp fingertip caught the stepper rail",
         "170256": "counter resync: the carriage was home already",
         "170506": "tag lost behind the thumb from 16.3 s",
         "170813": "J4 (middle palm y): nothing moved",
         "171230": "J4 back: nothing moved"}


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    hdr = rows[0]
    frames = [r for r in rows if "sn" in r and "corners" in r]
    mv = next((r["move"] for r in rows if "move" in r), None)
    res = next((r for r in rows if "move_result" in r), None)
    return hdr, frames, mv, res


def triangulate(cams, pair):
    """The tag's four corners (B, mm) from two cameras' undistorted pixels, and its mean edge (mm)."""
    import cv2
    P, uv = [], []
    for sn, c in pair:
        cam, T_BC = cams[sn]
        T_CB = tf_inv(T_BC)
        P.append(T_CB[:3, :])
        pts = cv2.undistortPoints(np.asarray(c, float).reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)
        uv.append(pts.T)
    X = cv2.triangulatePoints(P[0], P[1], uv[0], uv[1])
    X = (X[:3] / X[3]).T
    e = np.linalg.norm(np.roll(X, -1, 0) - X, axis=1)
    return X, float(e.mean())


def window(frames, tag, t0, t1):
    return [f for f in frames if f["tag"] == tag and t0 <= f["t"] < t1]


def locate(frames, tag, t0, t1, cams, T_PB):
    """Per camera and joint: the tag's mean centre in P (mm) over [t0, t1), its sd, the frames, and the triangulated
    edge."""
    rows = window(frames, tag, t0, t1)
    out = {}
    by = {}
    for f in rows:
        by.setdefault(f["sn"], []).append(f)
    for sn, fs in by.items():
        p = np.array([(T_PB @ np.asarray(f["T_BT"], float))[:3, 3] for f in fs])
        out[sn] = {"p": p.mean(0), "sd": float(np.linalg.norm(p.std(0))), "n": len(p)}
    sns = sorted(by)
    if len(sns) == 2:
        a, b = (sorted(by[s], key=lambda f: f["t"]) for s in sns)
        tb = np.array([f["t"] for f in b])
        tm = TG.plate_map("tag36h11", TAG_MM, [tag], [(0.0, 0.0, 0.0)])
        P, E = [], []
        for fa in a:
            k = int(np.argmin(np.abs(tb - fa["t"])))
            if abs(tb[k] - fa["t"]) > HALF_FRAME_S:
                continue
            fb = b[k]
            obs = [{tag: np.asarray(fa["corners"], float)}, {tag: np.asarray(fb["corners"], float)}]
            T, info = tp.solve_multi(obs, tm, [cams[sns[0]][0], cams[sns[1]][0]], [cams[sns[0]][1], cams[sns[1]][1]],
                                     np.asarray(fa["T_BT"], float), max_view_deg=None)
            if T is None:
                continue
            P.append((T_PB @ T)[:3, 3])
            E.append(triangulate(cams, [(sns[0], fa["corners"]), (sns[1], fb["corners"])])[1])
        if P:
            P = np.array(P)
            out["joint"] = {"p": P.mean(0), "sd": float(np.linalg.norm(P.std(0))), "n": len(P),
                            "edge_mm": float(np.mean(E)), "edge_sd_mm": float(np.std(E))}
    return out


def track(frames, tag, t0, t1, cams, T_PB, p0, u):
    """Travel along u (mm) from p0 against time (s from t0), per camera, during [t0, t1)."""
    out = {}
    for f in window(frames, tag, t0, t1):
        p = (T_PB @ np.asarray(f["T_BT"], float))[:3, 3]
        out.setdefault(f["sn"], []).append((f["t"] - t0, float((p - p0[f["sn"]]) @ u)))
    return {sn: sorted(v) for sn, v in out.items()}


def fit_move(path, cams, T_EP):
    hdr, frames, mv, res = load(path)
    stamp = Path(path).name[9:15]
    T_PB = tf_inv(np.asarray(hdr["T_BE"], float) @ T_EP)
    f = mv["finger"]
    tag = TAGS[f]
    d_cmd = np.array([mv["to"]["x"] - mv["from"]["x"], mv["to"]["y"] - mv["from"]["y"], 0.0])
    L = float(np.linalg.norm(d_cmd))
    u = d_cmd / L
    z = np.array([0.0, 0.0, 1.0])
    w = np.cross(z, u)
    t, t_done = float(mv["t"]), float(res["t_done"])
    b = locate(frames, tag, t - BEFORE_S, t, cams, T_PB)
    e = locate(frames, tag, t_done + SETTLE_S, t_done + SETTLE_S + AFTER_S + 0.5, cams, T_PB)
    rec = {"stamp": stamp, "finger": f, "tag": tag, "from": mv["from"], "to": mv["to"],
           "axis": "x" if abs(d_cmd[0]) > 0 else "y", "cmd_mm": round(L, 3),
           "steppers": mv["resolved"]["targets"][f]["steppers"], "moved_s": round(t_done - t, 2),
           "note": NOTES.get(stamp, ""), "solves": {}}
    for k in sorted(set(b) & set(e)):
        d = e[k]["p"] - b[k]["p"]
        along = float(d @ u)
        rec["solves"][k] = {"along_mm": round(along, 3), "scale": round(along / L, 5),
                            "turn_deg": round(float(np.degrees(np.arctan2(d @ w, max(abs(along), 1e-9)))) *
                                              (1 if along >= 0 else -1), 3),
                            "dz_mm": round(float(d @ z), 3), "d_P_mm": np.round(d, 3).tolist(),
                            "sd_before_mm": round(b[k]["sd"], 3), "sd_after_mm": round(e[k]["sd"], 3),
                            "n": [b[k]["n"], e[k]["n"]]}
        if k == "joint":
            rec["solves"][k]["edge_mm"] = [round(b[k]["edge_mm"], 3), round(e[k]["edge_mm"], 3)]
    p0 = {sn: b[sn]["p"] for sn in b if sn != "joint"}
    rec["track"] = {sn: [[round(a, 3), round(c, 3)] for a, c in v]
                    for sn, v in track(frames, tag, t - BEFORE_S, t_done + SETTLE_S + AFTER_S, cams, T_PB, p0, u).items()
                    if sn in p0}
    rec["p_before_P_mm"] = {k: np.round(v["p"], 3).tolist() for k, v in b.items()}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", default=str(ROOT / "logs/20261115-gantry_travel/follow"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    files = sorted(Path(a.logs).glob("*_move.jsonl"))
    hdr = json.loads(files[0].read_text().splitlines()[0])
    cams = live.load_cameras(ROOT / hdr["lens"], ROOT / hdr["handeye"])
    T_EP = SR2Hand.load().T_E_P()
    out = {"logs": str(Path(a.logs).relative_to(ROOT)) if Path(a.logs).is_absolute() else a.logs,
           "tag_mm": TAG_MM, "lens": hdr["lens"], "handeye": hdr["handeye"], "moves": []}
    for p in files:
        rec = fit_move(p, cams, T_EP)
        out["moves"].append(rec)
        s = rec["solves"]
        line = ", ".join(f"{k[-4:] if k != 'joint' else 'joint'} {v['along_mm']:+.2f} mm (x{v['scale']:.4f}, "
                         f"turn {v['turn_deg']:+.2f} deg, dz {v['dz_mm']:+.2f})" for k, v in s.items())
        edge = s.get("joint", {}).get("edge_mm")
        print(f"{rec['stamp']} {rec['finger']:6s} {rec['axis']} {rec['cmd_mm']:5.1f} mm: {line}"
              + (f"; tag edge {edge[0]:.3f}/{edge[1]:.3f} mm" if edge else "")
              + (f"  [{rec['note']}]" if rec["note"] else ""), flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

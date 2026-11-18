"""Joint zero offsets and finger mounts from the DIP tags of 2026-11-18 (W2 of the identification queue; S1(b) with the
arm lowered, rows data/s1/20261118-{middle,index,thumb,thumb_home}_rows.jsonl), beside the thumb's rows of 2026-11-17
(data/s1/rows.jsonl).

The tag model is sim/20261117-s1_zero_fit.py's (lab/fingertip_tag_check.fk): the sticker's centre at (-s 8.10, 0,
-36.205) mm and its outward normal -s x in <finger>_pip_frame of sr2hand.robot.tool_chain.SR2Hand, at the measured joints
(simulator degrees) and the mount the lab tool put in P (the station's mount plus frames.json's box_minus_nominal_mm,
which is (-6.18, -2.40) for the thumb, (6.18, 3.40) for the index and (6.18, -3.40) for the middle). Residuals are the
centre's error (mm) and the normal's error (deg, weighted 1 mm per deg). Each model is fitted by least squares; the
leave-one-out error leaves out both runs of a commanded pose.

Models (per finger; the thumb's two mounts of 11-18 share one offset, its gantry step taken as commanded):
  built        no parameter
  mount        the mount's (x, y)
  zeros        the yaw, MCP and PIP zeros
  both         zeros and mount
  pip_only     the yaw and PIP zeros and the mount (MCP zero 0)
  mcp_only     the yaw and MCP zeros and the mount (PIP zero 0)
  station      the zeros, with the mount at the station's own (x, y) (box_minus_nominal_mm not added)

    .venv/bin/python sim/20261118-s1b_tag_fit.py [--out data/s1/20261118-s1b_tag_fit.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.robot.tool_chain import SR2Hand, ToolChain  # noqa: E402

H = SR2Hand.load()
TAG_C = (8.10, -36.205)
NOMINAL = {f: np.array(H.frames["fingers"][f]["box_minus_nominal_mm"], float) for f in ("thumb", "index", "middle")}
SETS = {"middle": ["20261118-middle_rows.jsonl"], "index": ["20261118-index_rows.jsonl"],
        "thumb": ["20261118-thumb_rows.jsonl", "20261118-thumb_home_rows.jsonl"], "thumb_1117": ["rows.jsonl"]}
MODELS = {"built": (), "mount": ("mx", "my"), "zeros": ("yaw", "mcp", "pip"),
          "both": ("yaw", "mcp", "pip", "mx", "my"), "pip_only": ("yaw", "pip", "mx", "my"),
          "mcp_only": ("yaw", "mcp", "mx", "my"), "station": ("yaw", "mcp", "pip")}


def fk(finger, mount, q_deg):
    c, _ = H.box(finger)
    links = H.finger_links(finger, mount[0] - c[0], mount[1] - c[1], *(math.radians(v) for v in q_deg))
    T = ToolChain(finger, links).transform(upto=f"{finger}_pip_frame")
    s = H.fingers[finger].face_sign
    return (T @ np.array([-s * TAG_C[0], 0.0, TAG_C[1], 1.0]))[:3], T[:3, :3] @ np.array([-s, 0.0, 0.0])


def load(name):
    rows = []
    for fn in SETS[name]:
        for line in open(STUDY / "data/s1" / fn):
            r = json.loads(line) if line.strip() else None
            if r and r.get("p_meas") is not None and r.get("seen", 0) >= 3:
                rows.append(r)
    return rows


def unpack(x, names, model, finger):
    p = dict(zip(names, x))
    dq = np.array([p.get("yaw", 0.0), p.get("mcp", 0.0), p.get("pip", 0.0)])
    dm = np.array([p.get("mx", 0.0), p.get("my", 0.0)])
    if model == "station":
        dm = -NOMINAL[finger]
    return dq, dm


def residuals(x, rows, model):
    names = MODELS[model]
    out = []
    for r in rows:
        dq, dm = unpack(x, names, model, r["finger"])
        p, n = fk(r["finger"], np.asarray(r["mounts_P"], float) + dm, np.asarray(r["q_meas"], float) + dq)
        nm = np.asarray(r["n_meas"], float)
        nm = nm / np.linalg.norm(nm)
        out.extend(list(p - np.asarray(r["p_meas"], float)) + list(np.degrees(np.cross(n, nm))))
    return np.array(out)


def fit(rows, model):
    k = len(MODELS[model])
    if k == 0:
        return np.zeros(0), np.zeros(0), residuals(np.zeros(0), rows, model)
    r = least_squares(residuals, np.zeros(k), args=(rows, model))
    s2 = float(r.fun @ r.fun) / max(1, len(r.fun) - k)
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(r.jac.T @ r.jac) * s2), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return r.x, se, r.fun


def rms(f, cols):
    f = f.reshape(-1, 6)[:, cols]
    return float(np.sqrt(np.mean(np.sum(f ** 2, axis=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s1/20261118-s1b_tag_fit.json"))
    a = ap.parse_args()
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "), "script": str(Path(__file__).relative_to(ROOT)),
           "models": {m: list(v) for m, v in MODELS.items()}, "units": "zeros in deg (added to the reading), mount in mm "
           "(added to the lab's mount in P), errors in mm (centre) and deg (normal)", "sets": {}}
    for name in SETS:
        rows = load(name)
        q = np.array([r["q_meas"] for r in rows])
        poses = sorted({tuple(r["q_cmd"]) for r in rows})
        S = {"runs": len(rows), "poses": len(poses), "q_min": q.min(0).round(1).tolist(),
             "q_max": q.max(0).round(1).tolist(), "mounts_P": sorted({tuple(r["mounts_P"]) for r in rows}),
             "models": {}}
        print(f"{name}: {len(rows)} runs at {len(poses)} poses; yaw {S['q_min'][0]}..{S['q_max'][0]}, "
              f"MCP {S['q_min'][1]}..{S['q_max'][1]}, PIP {S['q_min'][2]}..{S['q_max'][2]} deg")
        for model in MODELS:
            x, se, f = fit(rows, model)
            loo = []
            for pose in poses:
                tr = [r for r in rows if tuple(r["q_cmd"]) != pose]
                te = [r for r in rows if tuple(r["q_cmd"]) == pose]
                xi = fit(tr, model)[0]
                loo.extend(np.linalg.norm(residuals(xi, te, model).reshape(-1, 6)[:, :3], axis=1))
            names = MODELS[model]
            o = {"params": {n: round(float(v), 2) for n, v in zip(names, x)},
                 "se": {n: round(float(v), 2) for n, v in zip(names, se)},
                 "centre_rms_mm": round(rms(f, slice(0, 3)), 2), "normal_rms_deg": round(rms(f, slice(3, 6)), 2),
                 "mean_err_mm": (-f.reshape(-1, 6)[:, :3].mean(0)).round(2).tolist(),
                 "loo_centre_rms_mm": round(float(np.sqrt(np.mean(np.square(loo)))), 2)}
            if model == "station":
                o["mount_offset_mm"] = (-NOMINAL[rows[0]["finger"]]).round(2).tolist()
            S["models"][model] = o
            print(f"  {model:8s} centre {o['centre_rms_mm']:5.2f} mm (left out {o['loo_centre_rms_mm']:5.2f}), normal "
                  f"{o['normal_rms_deg']:5.2f} deg | " + ", ".join(f"{n} {o['params'][n]:+.2f}+-{o['se'][n]:.2f}"
                                                               for n in names), flush=True)
        if name == "thumb":
            sep = {}
            for tag, fn in (("stepped", SETS["thumb"][0]), ("home", SETS["thumb"][1])):
                rr = [r for r in rows if r["mounts_P"] == json.loads(open(STUDY / "data/s1" / fn).readline())["mounts_P"]]
                sep[tag] = fit(rr, "mount")[0].round(2).tolist()
            S["mount_fit_each_mount"] = sep
            print(f"  mount alone at each gantry place: stepped {sep['stepped']}, home {sep['home']} mm")
        out["sets"][name] = S
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

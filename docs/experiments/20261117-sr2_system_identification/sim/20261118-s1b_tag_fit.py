"""Joint zero offsets and finger mounts from the DIP tags of 2026-11-18 (W2 of the identification queue; S1(b) with the
arm lowered, rows data/s1/20261118-{middle,index,thumb,thumb_home}_rows.jsonl), beside the thumb's rows of 2026-11-17
(data/s1/rows.jsonl).

The tag model is sim/20261117-s1_zero_fit.py's (lab/fingertip_tag_check.fk): the sticker's centre at (-s 8.10, 0,
-36.205) mm and its outward normal -s x in <finger>_pip_frame of sr2hand.robot.tool_chain.SR2Hand, at the measured joints
(simulator degrees) and the mount the lab tool put in P (the station's mount plus frames.json's box_minus_nominal_mm,
which is (-6.18, -2.40) for the thumb, (6.18, 3.40) for the index and (6.18, -3.40) for the middle). Residuals are the
centre's error (mm) and the normal's error (deg, weighted 1 mm per deg). Each model is fitted by least squares; the
leave-one-out error leaves out both runs of a commanded pose.

Models per finger (the thumb's two mounts of 11-18 share one offset, its gantry step taken as commanded):
  built        no parameter
  mount        the mount's (x, y)
  zeros        the yaw, MCP and PIP zeros
  both         zeros and mount
  pip_only     the yaw and PIP zeros and the mount (MCP zero 0)
  mcp_only     the yaw and MCP zeros and the mount (PIP zero 0)
  station      the zeros, with the mount at the station's own (x, y) (box_minus_nominal_mm not added)
  both_mz      zeros and mount, and the mount's height (z)
  both_tagz    zeros and mount, and the tag moved along the finger (z of <finger>_pip_frame)
Shared fits over the three fingers of 11-18 (per finger zeros and mount (x, y), and one offset of P for all):
  none, dz (P's height), dz_tilt (and P tilted about x and y), dz_tagz (and the thumb's tag moved along its finger)

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
from sr2hand.geometry.rotations import exp_so3  # noqa: E402
from sr2hand.robot.tool_chain import SR2Hand, ToolChain  # noqa: E402

H = SR2Hand.load()
TAG_C = (8.10, -36.205)
FINGERS = ("thumb", "index", "middle")
NOMINAL = {f: np.array(H.frames["fingers"][f]["box_minus_nominal_mm"], float) for f in FINGERS}
SETS = {"middle": ["20261118-middle_rows.jsonl"], "index": ["20261118-index_rows.jsonl"],
        "thumb": ["20261118-thumb_rows.jsonl", "20261118-thumb_home_rows.jsonl"], "thumb_1117": ["rows.jsonl"]}
MODELS = {"built": (), "mount": ("mx", "my"), "zeros": ("yaw", "mcp", "pip"),
          "both": ("yaw", "mcp", "pip", "mx", "my"), "pip_only": ("yaw", "pip", "mx", "my"),
          "mcp_only": ("yaw", "mcp", "mx", "my"), "station": ("yaw", "mcp", "pip"),
          "both_mz": ("yaw", "mcp", "pip", "mx", "my", "mz"), "both_tagz": ("yaw", "mcp", "pip", "mx", "my", "tz")}
SHARED = {"none": (), "dz": ("dz",), "dz_tilt": ("dz", "wx", "wy"), "dz_tagz": ("dz", "tz")}
PER = ("yaw", "mcp", "pip", "mx", "my")


def fk(finger, mount, q_deg, tz=0.0):
    c, _ = H.box(finger)
    links = H.finger_links(finger, mount[0] - c[0], mount[1] - c[1], *(math.radians(v) for v in q_deg))
    T = ToolChain(finger, links).transform(upto=f"{finger}_pip_frame")
    s = H.fingers[finger].face_sign
    return (T @ np.array([-s * TAG_C[0], 0.0, TAG_C[1] + tz, 1.0]))[:3], T[:3, :3] @ np.array([-s, 0.0, 0.0])


def load(name):
    rows = []
    for fn in SETS[name]:
        for line in open(STUDY / "data/s1" / fn):
            r = json.loads(line) if line.strip() else None
            if r and r.get("p_meas") is not None and r.get("seen", 0) >= 3:
                rows.append(r)
    return rows


def predict(r, p, model=None):
    """The tag's centre and normal for row r with parameters p (dict; missing ones 0), in the measured frame."""
    f = r["finger"]
    dq = np.array([p.get("yaw", 0.0), p.get("mcp", 0.0), p.get("pip", 0.0)])
    dm = -NOMINAL[f] if model == "station" else np.array([p.get("mx", 0.0), p.get("my", 0.0)])
    tz = p.get("tz", 0.0) if f == "thumb" or model is not None else 0.0
    c, n = fk(f, np.asarray(r["mounts_P"], float) + dm, np.asarray(r["q_meas"], float) + dq, tz)
    R = exp_so3(np.radians([p.get("wx", 0.0), p.get("wy", 0.0), 0.0]))
    return R @ c + np.array([0.0, 0.0, p.get("dz", 0.0) + p.get("mz", 0.0)]), R @ n


def err(r, c, n):
    nm = np.asarray(r["n_meas"], float)
    nm = nm / np.linalg.norm(nm)
    return list(c - np.asarray(r["p_meas"], float)) + list(np.degrees(np.cross(n, nm)))


def residuals(x, rows, model):
    p = dict(zip(MODELS[model], x))
    return np.array([e for r in rows for e in err(r, *predict(r, p, model))])


def shared_params(x, shared):
    per = {f: dict(zip(PER, x[5 * i:5 * i + 5])) for i, f in enumerate(FINGERS)}
    return per, dict(zip(SHARED[shared], x[15:]))


def shared_residuals(x, rows, shared):
    per, sh = shared_params(x, shared)
    return np.array([e for r in rows for e in err(r, *predict(r, {**per[r["finger"]], **sh}))])


def lsq(fun, k, args):
    if k == 0:
        return np.zeros(0), np.zeros(0), fun(np.zeros(0), *args)
    r = least_squares(fun, np.zeros(k), args=args)
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
           "models": {m: list(v) for m, v in MODELS.items()}, "shared_models": {m: list(v) for m, v in SHARED.items()},
           "units": "zeros in deg (added to the reading); mount and tag offsets in mm (added to the lab's model in P); "
           "dz in mm, wx and wy in deg (applied to the model's prediction); errors in mm (centre) and deg (normal)",
           "sets": {}, "shared": {}}
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
            k = len(MODELS[model])
            x, se, f = lsq(residuals, k, (rows, model))
            loo = []
            for pose in poses:
                tr = [r for r in rows if tuple(r["q_cmd"]) != pose]
                te = [r for r in rows if tuple(r["q_cmd"]) == pose]
                xi = lsq(residuals, k, (tr, model))[0]
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
            print(f"  {model:9s} centre {o['centre_rms_mm']:5.2f} mm (left out {o['loo_centre_rms_mm']:5.2f}), normal "
                  f"{o['normal_rms_deg']:5.2f} deg | " + ", ".join(f"{n} {o['params'][n]:+.2f}+-{o['se'][n]:.2f}"
                                                               for n in names), flush=True)
        if name == "thumb":
            sep = {}
            for tag, fn in (("stepped", SETS["thumb"][0]), ("home", SETS["thumb"][1])):
                m0 = json.loads(open(STUDY / "data/s1" / fn).readline())["mounts_P"]
                sep[tag] = lsq(residuals, 2, ([r for r in rows if r["mounts_P"] == m0], "mount"))[0].round(2).tolist()
            S["mount_fit_each_mount"] = sep
            print(f"  mount alone at each gantry place: stepped {sep['stepped']}, home {sep['home']} mm")
        out["sets"][name] = S
    rows = load("thumb") + load("index") + load("middle")
    for shared in SHARED:
        k = 15 + len(SHARED[shared])
        x, se, f = lsq(shared_residuals, k, (rows, shared))
        per, sh = shared_params(x, shared)
        F6 = f.reshape(-1, 6)
        o = {"shared": {n: round(float(v), 2) for n, v in sh.items()},
             "shared_se": {n: round(float(v), 2) for n, v in zip(SHARED[shared], se[15:])},
             "per_finger": {g: {n: round(float(v), 2) for n, v in per[g].items()} for g in FINGERS},
             "centre_rms_mm": {}, "loo_centre_rms_mm": {}}
        for g in FINGERS:
            sel = np.array([r["finger"] == g for r in rows])
            o["centre_rms_mm"][g] = round(float(np.sqrt(np.mean(np.sum(F6[sel, :3] ** 2, 1)))), 2)
            loo = []
            for pose in sorted({tuple(r["q_cmd"]) for r in rows if r["finger"] == g}):
                tr = [r for r in rows if not (r["finger"] == g and tuple(r["q_cmd"]) == pose)]
                te = [r for r in rows if r["finger"] == g and tuple(r["q_cmd"]) == pose]
                xi = lsq(shared_residuals, k, (tr, shared))[0]
                loo.extend(np.linalg.norm(shared_residuals(xi, te, shared).reshape(-1, 6)[:, :3], axis=1))
            o["loo_centre_rms_mm"][g] = round(float(np.sqrt(np.mean(np.square(loo)))), 2)
        out["shared"][shared] = o
        print(f"shared {shared:8s}: " + ", ".join(f"{g} {o['centre_rms_mm'][g]:.2f} (left out "
                                                  f"{o['loo_centre_rms_mm'][g]:.2f})" for g in FINGERS)
              + " mm | " + ", ".join(f"{n} {v:+.2f}+-{o['shared_se'][n]:.2f}" for n, v in o["shared"].items()), flush=True)
        for g in FINGERS:
            print("    " + g.ljust(7) + ", ".join(f"{n} {v:+.2f}" for n, v in o["per_finger"][g].items()))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

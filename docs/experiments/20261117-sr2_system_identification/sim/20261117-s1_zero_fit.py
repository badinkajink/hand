"""Joint zero offsets (and the mount's position) of a finger from its DIP tag (W2 of the identification queue,
2026-11-17; S1(b) rows in docs/experiments/20261117-sr2_system_identification/data/s1/rows.jsonl).

The tag model is lab/fingertip_tag_check.fk's: the sticker's centre at (-s 8.10, 0, -36.205) mm and its outward normal
-s x in <finger>_pip_frame, on sr2hand.robot.tool_chain.SR2Hand's chain at the plan's mount in P (mounts_P), for the
measured joint angles (simulator degrees). Fitted by least squares over the poses the tag was seen at: residuals the
centre's error (mm) and the normal's error (deg, as mm at 1 mm per deg), for three models: the joint zeros alone, the
mount alone (x, y), and both. Reports the RMS before and after, the parameters with their standard errors, and the
leave-one-out RMS of the centre.

    .venv/bin/python s1_zero_fit.py ROWS.jsonl [--finger thumb] [--out JSON]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Programs/hand"
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.robot.tool_chain import SR2Hand, ToolChain  # noqa: E402

H = SR2Hand.load()
TAG_C = (8.10, -36.205)


def fk(finger, mount, q_deg):
    c, _ = H.box(finger)
    links = H.finger_links(finger, mount[0] - c[0], mount[1] - c[1], *(math.radians(v) for v in q_deg))
    T = ToolChain(finger, links).transform(upto=f"{finger}_pip_frame")
    s = H.fingers[finger].face_sign
    return (T @ np.array([-s * TAG_C[0], 0.0, TAG_C[1], 1.0]))[:3], T[:3, :3] @ np.array([-s, 0.0, 0.0])


def residuals(x, rows, finger, which):
    dq = np.zeros(3)
    dm = np.zeros(2)
    if which in ("zeros", "both"):
        dq = x[:3]
    if which == "mount":
        dm = x[:2]
    if which == "both":
        dm = x[3:5]
    out = []
    for r in rows:
        p, n = fk(finger, np.asarray(r["mounts_P"], float) + dm, np.asarray(r["q_meas"], float) + dq)
        pm, nm = np.asarray(r["p_meas"], float), np.asarray(r["n_meas"], float)
        nm = nm / np.linalg.norm(nm)
        e = np.cross(n, nm)                                     # small-angle normal error, rad
        out.extend(list(p - pm) + list(np.degrees(e)))
    return np.array(out)


def fit(rows, finger, which):
    from scipy.optimize import least_squares
    k = {"zeros": 3, "mount": 2, "both": 5}[which]
    r = least_squares(residuals, np.zeros(k), args=(rows, finger, which))
    J = r.jac
    dof = max(1, len(r.fun) - k)
    s2 = float(r.fun @ r.fun) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return r.x, se, r.fun


def rms_centre(f):
    f = f.reshape(-1, 6)
    return float(np.sqrt(np.mean(np.sum(f[:, :3] ** 2, axis=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--finger", default="thumb")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.rows) if x.strip()]
    rows = [r for r in rows if r["finger"] == a.finger and r.get("p_meas") is not None and r.get("seen", 0) > 0]
    print(f"{a.finger}: {len(rows)} poses seen; joints (yaw, mcp, pip) span "
          f"{np.ptp([r['q_meas'] for r in rows], axis=0).round(1).tolist()} deg", flush=True)
    f0 = residuals(np.zeros(0), rows, a.finger, "none")
    out = {"finger": a.finger, "poses": len(rows), "rms_centre_mm_before": round(rms_centre(f0), 3),
           "mean_centre_err_mm_before": f0.reshape(-1, 6)[:, :3].mean(0).round(3).tolist()}
    print(f"  before: centre RMS {out['rms_centre_mm_before']:.2f} mm, mean error {out['mean_centre_err_mm_before']} mm")
    for which in ("zeros", "mount", "both"):
        x, se, f = fit(rows, a.finger, which)
        loo = []
        for i in range(len(rows)):
            xi, _, _ = fit(rows[:i] + rows[i + 1:], a.finger, which)
            fi = residuals(xi, [rows[i]], a.finger, which)
            loo.append(float(np.linalg.norm(fi[:3])))
        names = {"zeros": ["d_yaw_deg", "d_mcp_deg", "d_pip_deg"], "mount": ["d_mount_x_mm", "d_mount_y_mm"],
                 "both": ["d_yaw_deg", "d_mcp_deg", "d_pip_deg", "d_mount_x_mm", "d_mount_y_mm"]}[which]
        out[which] = {"params": {n: round(float(v), 3) for n, v in zip(names, x)},
                      "se": {n: round(float(v), 3) for n, v in zip(names, se)},
                      "rms_centre_mm": round(rms_centre(f), 3),
                      "rms_normal_deg": round(float(np.sqrt(np.mean(np.sum(f.reshape(-1, 6)[:, 3:] ** 2, axis=1)))), 3),
                      "loo_rms_centre_mm": round(float(np.sqrt(np.mean(np.square(loo)))), 3)}
        o = out[which]
        print(f"  {which:6s}: centre RMS {o['rms_centre_mm']:.2f} mm (leave-one-out {o['loo_rms_centre_mm']:.2f}), "
              f"normal {o['rms_normal_deg']:.2f} deg | " +
              ", ".join(f"{n} {o['params'][n]:+.2f} +- {o['se'][n]:.2f}" for n in names), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

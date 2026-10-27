#!/usr/bin/env python3
"""Whether v2 ID 1's board shows the friction torque of the UR5 twists (owner, 2026-10-27: "you still think there's no
good way to detect twists/wrenches?"). The friction torque reverses with the direction of a twist, so at each angle
reached on the way out and on the way back, half the difference of the two holds' board changes contains it. The part
that the two holds' different forces explain (a linear map from the HEX-E's force to the board change, fitted on the
press and slide holds within 1.5 mm of the twist's point) is removed. What remains is compared across pairs at the same
point (cosine between the remaining vectors), with the force and with the HEX-E's half out-back torsion about the
normal. Prints the table; writes data/20261026-192158-v2id1/twist_board.json.

    .venv/bin/python scripts/ur5_tip_calibration_twist_board.py
"""
import json
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ur5_tip_calibration_cb1r_checks as K
H = K.load(K.EXP / "data/20261026-192158-v2id1")
T = K.sub(H, H["kind"] == "twist")
PS = K.sub(H, np.isin(H["kind"], ["press", "slide"]) & np.isfinite(H["d_board"]).all(1) & np.isfinite(H["f"]).all(1))
ids = T["id"].astype(str)
rows = []
for aid in sorted(set(ids)):
    r = np.where(ids == aid)[0]
    r = r[np.argsort(T["t"][r])]
    p0 = T["p"][r].mean(0)
    near = np.linalg.norm(PS["p"] - p0, axis=1) < 1.5
    X = np.c_[PS["f"][near], np.ones(near.sum())]
    A = np.linalg.lstsq(X, PS["d_board"][near], rcond=None)[0]          # (4, 15): b = f A[:3] + A[3]
    resid_fit = PS["d_board"][near] - X @ A
    deg = np.round(T["turn_deg"][r], 1)
    peak = deg.max()
    for a in np.unique(deg):
        k = r[deg == a]
        if len(k) != 2 or a == 0:
            continue
        db = (T["d_board"][k[0]] - T["d_board"][k[1]]) / 2
        df = (T["f"][k[0]] - T["f"][k[1]]) / 2
        pred = df @ A[:3]
        rows.append(dict(aid=aid, deg=float(a), peak=float(peak), fn=float(T["f_n"][k].mean()), db=db, res=db - pred,
                         dtn=float((T["tn"][k[0]] - T["tn"][k[1]]) / 2), n_near=int(near.sum()),
                         fit_rms=float(np.sqrt(np.mean(resid_fit ** 2))), p=p0))
print("pairs", len(rows))
nb = np.array([np.linalg.norm(q["db"]) for q in rows]); nr = np.array([np.linalg.norm(q["res"]) for q in rows])
fn = np.array([q["fn"] for q in rows]); dtn = np.array([q["dtn"] for q in rows])
for lo, hi in ((0, 1.5), (1.5, 3.5), (3.5, 9)):
    m = (fn >= lo) & (fn < hi)
    print(f"f_n {lo}-{hi}: n {m.sum()}, |db| median {np.median(nb[m]):.1f} uT, after the force difference {np.median(nr[m]):.1f} uT, HEX-E half out-back torsion median {np.median(dtn[m]):.2f} N mm (sd {dtn[m].std():.2f})")
# consistency: the residual's direction across pairs at the same target point
by_pt = collections.defaultdict(list)
for q in rows:
    by_pt[tuple(np.round(q["p"], 0))].append(q)
for pt, qs in by_pt.items():
    R = np.array([q["res"] for q in qs]); u = R / np.linalg.norm(R, axis=1, keepdims=True)
    C = u @ u.T; iu = np.triu_indices(len(qs), 1)
    mean_dir = R.mean(0); mean_dir /= np.linalg.norm(mean_dir)
    proj = R @ mean_dir
    fns = np.array([q["fn"] for q in qs]); dt = np.array([q["dtn"] for q in qs])
    print(f"point {pt}: {len(qs)} pairs, median cosine between pairs {np.median(C[iu]):.2f}, projection on the mean direction "
          f"{np.round(np.percentile(proj, [10, 50, 90]), 1)} uT; corr(projection, f_n) {np.corrcoef(proj, fns)[0, 1]:.2f}, "
          f"corr(projection, HEX-E torsion) {np.corrcoef(proj, dt)[0, 1]:.2f}; local fit rms {qs[0]['fit_rms']:.1f} uT on {qs[0]['n_near']} holds")
print("--- by action: time between out and back, residual projection on the point's mean direction")
for pt, qs in by_pt.items():
    if len(qs) < 6:
        continue
    R = np.array([q["res"] for q in qs]); md = R.mean(0); md /= np.linalg.norm(md)
    for q in qs:
        r = np.where(ids == q["aid"])[0]; r = r[np.argsort(T["t"][r])]
        deg = np.round(T["turn_deg"][r], 1); k = r[deg == q["deg"]]
        dt = T["t"][k[1]] - T["t"][k[0]]
        print(f"  {q['aid']:26s} {q['deg']:5.1f} deg  dt {dt:6.1f} s  f_n {q['fn']:.2f}  proj {q['res'] @ md:6.1f} uT  |res| {np.linalg.norm(q['res']):5.1f}  HEX-E {q['dtn']:5.2f} N mm")

out = [dict(aid=q["aid"], deg=q["deg"], f_n=q["fn"], board_half_diff_uT=float(np.linalg.norm(q["db"])),
            after_force_uT=float(np.linalg.norm(q["res"])), hexe_half_torsion_Nmm=q["dtn"], point=q["p"].tolist()) for q in rows]
(K.EXP / "data/20261026-192158-v2id1/twist_board.json").write_text(json.dumps(out, indent=1) + "\n")

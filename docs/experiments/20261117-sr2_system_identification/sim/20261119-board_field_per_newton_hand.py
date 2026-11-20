"""The boards' raw field change per newton on the hand's S3 presses against the same tip's on the UR5 calibration, at
the same place on the pad (2026-11-19, for the owner's 16:3x question on the contact area and the force reading).

The calibration tables say a tip's field per newton does not depend on the indenter (R 2 and R 5 spheres, R 12.5 rod
within about 10 %: sim/20261119-board_patch_size_calibration.py). If the hand's boards read about half the HEX-E's force
because of the flat plate, the raw field per newton on the hand would be lower than the calibration's at that place on
the pad; if the raw field per newton matches, the shortfall is in the estimator (the GP, its blend with the physical
model, the own-pose term).

Per pad-face press of 2026-11-18 (sim/20261118-s3_board_vs_hexe.py's presses and rests; the yaw presses' tip sides are
left out) and per step with a HEX-E force of 0.8 N and over: the board's change d from the rest (15 components, uT, as
tip_contacts.Tip.estimate forms it before the own-pose term), the own-pose term the trials subtract, the HEX-E's |F_xy|
(x_P, y_P; z drifts), the GP's contact point (pad frame) and the calibration's median |d_board| / f_n there (the tip's
R 5 sphere holds, near-normal, within 1.5 mm and 1-3 N; the tables of 20261105-viewer_inputs/correction). Ratio =
(|d| / |F_xy|) / calibration.

    /usr/bin/python3 sim/20261119-board_field_per_newton_hand.py [--out data/s3/20261119-board_field_per_newton_hand.json]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
spec = importlib.util.spec_from_file_location("bvh", HERE / "20261118-s3_board_vs_hexe.py")
BVH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BVH)
TC = BVH.TC
CORR = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261105-viewer_inputs/correction"
CAL = {1: CORR / "20261028-spherev3id1_twists", 2: CORR / "20261029-spherev3id2", 3: CORR / "20261030-spherev3id3"}


def cal_table(tip):
    z = np.load(CAL[tip] / "holds.npz", allow_pickle=True)
    ok = np.isfinite(z["d_board"]).all(1) & np.isfinite(z["p"]).all(1) & np.isfinite(z["f"]).all(1)
    f, n, p, B, k = z["f"][ok], z["n"][ok], z["p"][ok], z["d_board"][ok], z["kind"][ok].astype(str)
    fn = np.abs(np.sum(f * n, 1))
    ft = np.linalg.norm(f - np.sum(f * n, 1)[:, None] * n, axis=1)
    m = (ft < 0.15 * fn) & (fn >= 1.0) & (fn <= 3.0) & (k == "press")
    return p[m], np.linalg.norm(B[m], axis=1) / fn[m], B[m] / fn[m][:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-board_field_per_newton_hand.json"))
    ap.add_argument("--near", type=float, default=1.5)
    a = ap.parse_args()
    R_PH = np.asarray(json.loads((BVH.LOGS / f"{BVH.R_PH_FROM}.jsonl").read_text().splitlines()[0])["R_PH"], float)
    tabs = {t: cal_table(t) for t in CAL}
    out = []
    for name, finger, appr in BVH.presses():
        if "_yaw" in name:
            continue
        tip_id = TC.TIP_ID[finger]
        Cp, Cr, CB = tabs[tip_id]
        others = json.loads(next((STUDY / "data/s3").glob(f"{appr}_approach_*.json")).read_text())["joints"]
        head, rows, ts, qs = BVH.load(name)
        t_first = rows[0]["t"]
        feed = TC.MuxLog(BVH.mux_for(t_first), t_first - BVH.REST_BEFORE_S - BVH.REST_S - TC.KEEP_S - 1.0,
                         rows[-1]["t"] + 1.0)

        def joints(t):
            q = {f: dict(others[f]) for f in TC.FINGERS}
            q[finger] = {j: float(np.interp(t, ts, qs[:, i])) for i, j in enumerate(("yaw", "mcp", "pip"))}
            return BVH.servo33(q)

        tc = TC.TipContacts(feed, np.zeros(6), gp=True, own_pose=True)
        tip = tc.tips[finger]
        tip.threshold, tip.min_force = -1e9, -1e9
        rest_end = t_first - BVH.REST_BEFORE_S
        tc.rest(rest_end, BVH.REST_S, refit=True, joints_servo=joints(rest_end))
        steps = []
        for r in rows:
            F = np.asarray(r["dF_P"] if "dF_P" in r else R_PH @ np.asarray(r["dF_N"], float), float)
            Fxy = float(np.hypot(F[0], F[1]))
            ds, owns, ps, gws = [], [], [], []
            for t in r["t"] - 1.0 + (np.arange(BVH.N_EVAL) + 0.5) / BVH.N_EVAL:
                tt, rr = feed.window(finger, t, TC.KEEP_S)
                ok = TC.valid(rr)
                tt, rr = tt[ok], rr[ok]
                if len(tt) < 3:
                    continue
                sel = tt > tt[-1] - TC.WIN_S
                d = tip.fields(rr[sel], rr).mean(0) - tip.base
                q = TC.TipContacts._q(joints(t)).get(finger)
                own = (tip.own.predict(finger, q) - tip.own.predict(finger, tip.q_rest)) if (
                    tip.own is not None and tip.q_rest is not None and q is not None) else np.zeros(15)
                e = tip.estimate(tt, rr, q)
                ds.append(d)
                owns.append(own)
                if e.get("contact"):
                    ps.append(e["p"])
                    gws.append(e.get("w"))
            if not ds:
                continue
            d, own = np.mean(ds, 0), np.mean(owns, 0)
            p = np.mean(ps, 0) if ps else None
            cal = None
            if p is not None:
                near = np.linalg.norm(Cp - p, axis=1) < a.near
                if near.sum() >= 5:
                    cal = float(np.median(Cr[near]))
                    cos = float(d @ CB[near].mean(0) / (np.linalg.norm(d) * np.linalg.norm(CB[near].mean(0)) + 1e-12))
                else:
                    cos = None
            steps.append(dict(k=r["k"], F_xy=round(Fxy, 3), F_z=round(float(F[2]), 3), d_uT=round(float(np.linalg.norm(d)), 1),
                              own_uT=round(float(np.linalg.norm(own)), 1),
                              d_minus_own_uT=round(float(np.linalg.norm(d - own)), 1),
                              own_along_d=round(float(own @ d / (np.linalg.norm(d) + 1e-9)), 1),
                              p=None if p is None else np.round(p, 2).tolist(),
                              w=None if not gws or gws[0] is None else round(float(np.mean(gws)), 3),
                              cal_uT_per_N=None if cal is None else round(cal, 1),
                              cos_to_cal=None if cal is None else round(cos, 3)))
        hi = [s for s in steps if s["F_xy"] >= 0.8 and s["cal_uT_per_N"]]
        # the slope of |d| against F_xy over all steps (the rest's |d| is noise, a few uT)
        Fx = np.array([s["F_xy"] for s in steps])
        Dd = np.array([s["d_uT"] for s in steps])
        Do = np.array([s["d_minus_own_uT"] for s in steps])
        A = np.c_[Fx, np.ones(len(Fx))]
        sl = float(np.linalg.lstsq(A, Dd, rcond=None)[0][0]) if len(Fx) > 3 else None
        slo = float(np.linalg.lstsq(A, Do, rcond=None)[0][0]) if len(Fx) > 3 else None
        calm = float(np.median([s["cal_uT_per_N"] for s in hi])) if hi else None
        rec = dict(press=name, finger=finger, tip=tip_id, n_steps=len(steps), n_hi=len(hi),
                   slope_uT_per_N=None if sl is None else round(sl, 1),
                   slope_minus_own_uT_per_N=None if slo is None else round(slo, 1),
                   cal_uT_per_N=None if calm is None else round(calm, 1),
                   ratio=None if (sl is None or not calm) else round(sl / calm, 3),
                   ratio_minus_own=None if (slo is None or not calm) else round(slo / calm, 3),
                   p_med=None if not hi else np.round(np.median([s["p"] for s in hi], 0), 2).tolist(),
                   cos_med=None if not hi else round(float(np.median([s["cos_to_cal"] for s in hi])), 3),
                   own_share=None if not hi else round(float(np.median([s["own_along_d"] / max(s["d_uT"], 1e-9)
                                                                        for s in hi])), 3),
                   steps=steps)
        out.append(rec)
        print(f"{name:28s} {finger:6s} tip {tip_id}: |d| per N of F_xy {rec['slope_uT_per_N']} (minus own-pose "
              f"{rec['slope_minus_own_uT_per_N']}); calibration at the GP's point {rec['cal_uT_per_N']} uT/N -> ratio "
              f"{rec['ratio']} ({rec['ratio_minus_own']}); point {rec['p_med']}; cos to the calibration's pattern "
              f"{rec['cos_med']}; own-pose share along d {rec['own_share']}", flush=True)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "presses": out},
                                      separators=(",", ":")) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

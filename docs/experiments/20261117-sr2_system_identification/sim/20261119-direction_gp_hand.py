"""The direction GP against the trials' board GP, on held-out calibration sessions and on the hand's arm presses
(2026-11-19 19:0x; the fingertip sensing page's first open question: does a GP whose input is the field's direction
remove the board GP's pull toward the apex at low force, and the quarter of the force it loses on the hand?).

The board GP (tip_contacts' deployed GP) maps the field change d (15 components) to the point and the 3D force. The
direction GP (ur5_tip_calibration_hard_extrapolation.fit_gp_direction, October) maps d / |d| to the point and to the
force per unit field f / |d|, and multiplies back by |d|. Both are fitted here on the deployed GP's training recipe
(sim/20261119-board_contact_place_rig.py's training(): its 15 sessions, presses and slides, 1,500 rig holds of each of
IDs 4-6), the board GP at the deployed length and noise, the direction GP at the length and noise its marginal
likelihood picks on 3,000 of the holds.

  rig    each hand-tip session left out in turn: place error by force, the share of low-force errors that point
         toward the training mean place, and the force's size ratio
  hand   the arm presses of 2026-11-19 (50 ms means of the trials' fields, sim/20261119-board_estimate_stability.py's
         pipeline): each GP's place over the 2 s hold against the plate's geometry with the 11-18 tags' flexion
         (data/s3/20261119-arm_press_contact_place.json), its place at 0.3-0.6 N of the loading, and its force per
         HEX-E newton (a straight line over the windows above 0.3 N)

    OPENBLAS_NUM_THREADS=4 /usr/bin/python3 sim/20261119-direction_gp_hand.py [--out data/s3/20261119-direction_gp_hand.json]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "src/sr2hand/driver/reskin5x/host"), str(ROOT / "src"), str(ROOT / "scripts"),
                str(ROOT / "scripts/ur5")]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RIG = load_module("rig", HERE / "20261119-board_contact_place_rig.py")
STAB = load_module("stab", HERE / "20261119-board_estimate_stability.py")
import tip_contacts as TC  # noqa: E402
import ur5_tip_calibration_board_gp as BG  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

Z_MEAN = 10.45


class DirGP:
    def __init__(self, X, Y, ell=None, nz=None, n_pick=3000, seed=0):
        n = np.linalg.norm(X, axis=1)
        U, T = X / n[:, None], np.c_[Y[:, :3], Y[:, 3:] / n[:, None]]
        floor = np.full(15, 0.01)
        if ell is None:
            i = np.random.default_rng(seed).choice(len(U), min(n_pick, len(U)), replace=False)
            m = BG.fit(U[i], T[i], floor)
            ell, nz = m["ell"], m["nz"]
        self.ell, self.nz = float(ell), float(nz)
        self.m = RIG.fit_fixed(U, T, floor, self.ell, self.nz)

    def predict(self, X):
        n = np.linalg.norm(X, axis=1)
        Y = RIG.predict(self.m, X / n[:, None])
        return Y[:, :3], Y[:, 3:] * n[:, None]


def board_predict(m, X):
    Y = RIG.predict(m, X)
    return Y[:, :3], Y[:, 3:]


def rig_scores(h, P, F):
    z, f = h["p"][:, 2], np.linalg.norm(h["f"], axis=1)
    e = np.array([np.linalg.norm(RIG.on_pad(P[i])[0] - RIG.on_pad(h["p"][i])[0]) for i in range(len(P))])
    ratio = np.linalg.norm(F, axis=1) / np.maximum(f, 1e-6)
    toward = (P[:, 2] - z) * np.sign(Z_MEAN - z)
    out = {}
    for lo, hi in ((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99.0)):
        m = (f >= lo) & (f < hi)
        far = m & (np.abs(z - Z_MEAN) > 3)
        out[f"{lo}-{hi}"] = dict(n=int(m.sum()), place_mm=round(float(np.median(e[m])), 3) if m.any() else None,
                                 toward_share=round(float(np.mean(toward[far] > 0)), 3) if far.sum() > 5 else None,
                                 toward_mm=round(float(np.median(toward[far])), 3) if far.sum() > 5 else None,
                                 force_ratio=round(float(np.median(ratio[m])), 3) if m.any() else None)
    out["all"] = dict(place_mm=round(float(np.median(e)), 3), force_ratio=round(float(np.median(ratio)), 3))
    return out


def hand_windows():
    """Per arm press: the 50 ms means of the trials' field change and the HEX-E's force along the press."""
    out = []
    for stem, label in STAB.RUNS.items():
        L = [json.loads(x) for x in (STAB.LOGS / f"{stem}.jsonl").read_text().splitlines() if x.strip()]
        h = L[0]
        f = h["finger"]
        raws = {json.loads(x)["rep"]: json.loads(x) for x in (STAB.LOGS / f"{stem}_raw.jsonl").read_text().splitlines()
                if x.strip()}
        R_BH = np.asarray(h["T_BP_end"], float)[:3, :3] @ np.asarray(h["R_PH"], float)
        d_B = np.asarray(h["press_dir_B"], float)
        for rep in [x for x in L[1:] if "rep" in x and x.get("t_hold")]:
            t0, th0, t1 = rep["t_start"], rep["t_hold"], rep["t_end"]
            feed = TC.MuxLog(STAB.mux_for(t0), t0 - 2.0 - TC.KEEP_S, t1 + 1.0)
            tc = TC.TipContacts(feed, np.zeros(6), gp=True, own_pose=False)
            tip = tc.tips[f]
            tc.rest(t0 - 0.1, 0.9, refit=True)
            raw = raws[rep["rep"]]
            th = np.asarray(raw["hexe_t"], float)
            Fp = -((np.asarray(raw["hexe"], float)[:, :3] - np.asarray(rep["F0"], float)) @ R_BH.T) @ d_B
            T, Rr = feed.window(f, t1 + 0.5, t1 + 0.5 - (t0 - 1.5))
            ok = TC.valid(Rr)
            T, Rr = T[ok], Rr[ok]
            _, rT = feed.window(f, t0, TC.KEEP_S)
            D = tip.fields(Rr, rT) - tip.base
            edges = np.arange(t0, t1, TC.WIN_S)
            k = np.digitize(T, edges)
            idx = [i for i in range(1, len(edges)) if (k == i).sum() >= 3]
            M = np.array([D[k == i].mean(0) for i in idx])
            Mt = np.array([T[k == i].mean() for i in idx])
            out.append(dict(group=label, rep=rep["rep"], t_hold=th0, t=Mt, D=M, F=np.interp(Mt, th, Fp)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261119-direction_gp_hand.json"))
    a = ap.parse_args()
    dep = np.load(RIG.GP_FILE)
    ell, nz = float(dep["ell"]), float(dep["nz"])
    floor = PR.floors(0)[:15]
    t0 = time.time()
    res = {"what": __doc__.split("\n\n")[0].replace("\n", " "), "script": str(Path(__file__).resolve().relative_to(ROOT)),
           "rig": [], "hand": []}
    # the direction GP's length and noise, picked once on the full recipe's holds
    X, Y, tips, H = RIG.training(())
    dg_all = DirGP(X, Y)
    res["direction_gp"] = dict(length=dg_all.ell, noise=dg_all.nz, n=int(len(X)))
    print(f"direction GP: length {dg_all.ell:.3f}, noise {dg_all.nz} on {len(X)} holds  [{time.time() - t0:.0f} s]", flush=True)
    for s in RIG.SETS["tip1"] + RIG.SETS["tip2"] + RIG.SETS["tip3"] + RIG.SETS["rig3"]:
        X, Y, tips, H = RIG.training([s])
        mb = RIG.fit_fixed(X, Y, floor, ell, nz)
        dg = DirGP(X, Y, dg_all.ell, dg_all.nz)
        h = RIG.load(s, RIG.EVAL_KINDS)
        sb = rig_scores(h, *board_predict(mb, h["d"]))
        sd = rig_scores(h, *dg.predict(h["d"]))
        res["rig"].append(dict(session=s, board_gp=sb, direction_gp=sd))
        print(f"{s.split('/')[-1]:30s} place board {sb['all']['place_mm']} / direction {sd['all']['place_mm']} mm; "
              f"0.1-0.5 N: {sb['0.1-0.5']['place_mm']} / {sd['0.1-0.5']['place_mm']} mm, toward apex "
              f"{sb['0.1-0.5']['toward_share']} / {sd['0.1-0.5']['toward_share']}; force ratio {sb['all']['force_ratio']} / "
              f"{sd['all']['force_ratio']}  [{time.time() - t0:.0f} s]", flush=True)
        del mb, dg, X, Y, H
    # the hand: both GPs fitted on everything
    X, Y, tips, H = RIG.training(())
    mb = RIG.fit_fixed(X, Y, floor, ell, nz)
    arm = json.loads((STUDY / "data/s3/20261119-arm_press_contact_place.json").read_text())
    geo = {r["group"]: (r["z_geometry"], r["z_geometry_tags"]) for r in arm["rows"]}
    for w in hand_windows():
        row = dict(group=w["group"], rep=w["rep"], geometry_z=geo[w["group"]][0], geometry_tags_z=geo[w["group"]][1])
        hold = (w["t"] >= w["t_hold"] + 0.2) & (w["t"] <= w["t_hold"] + 2.0)
        for name, (P, F) in (("board_gp", board_predict(mb, w["D"])), ("direction_gp", dg_all.predict(w["D"]))):
            fn = np.linalg.norm(F, axis=1)
            on = w["F"] > 0.3
            slope = float(np.polyfit(w["F"][on], fn[on], 1)[0]) if on.sum() > 10 else None
            low = (w["F"] >= 0.3) & (w["F"] <= 0.6) & (w["t"] < w["t_hold"])
            row[name] = dict(hold_z=round(float(np.median(P[hold, 2])), 2),
                             hold_jitter_z=round(float(np.std(P[hold, 2])), 3),
                             low_force_z=round(float(np.median(P[low, 2])), 2) if low.any() else None,
                             force_per_hexe_N=None if slope is None else round(slope, 3),
                             path=[[round(float(x), 2), round(float(y), 2)] for x, y in
                                   zip(w["F"][w["t"] < w["t_hold"]][::2], P[w["t"] < w["t_hold"], 2][::2])])
        res["hand"].append(row)
        b, d = row["board_gp"], row["direction_gp"]
        print(f"{w['group']:14s} rep {w['rep']}: geometry {row['geometry_z']} (tags {row['geometry_tags_z']}); hold z board "
              f"{b['hold_z']} / direction {d['hold_z']} (jitter {b['hold_jitter_z']} / {d['hold_jitter_z']}); at 0.3-0.6 N "
              f"{b['low_force_z']} / {d['low_force_z']}; force per HEX-E N {b['force_per_hexe_N']} / "
              f"{d['force_per_hexe_N']}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=None, separators=(",", ":")))
    print("wrote", a.out, f"{Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

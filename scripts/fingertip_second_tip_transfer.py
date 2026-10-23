#!/usr/bin/env python3
"""The UR5 run sphere2's calibration, made on the robot fingertip (board 9c:d0, all five poles one way), applied to a
second fingertip on the workstation (board d9:94, poles (-1, -1, 1, 1, 1)), from the live viewer's stream log (owner,
2026-10-23: "now that we have this known calibration from the robot fingertip, can we see how well that applies to the
fingertip plugged in here? ... this has a different setup of polarities").

rest      the second tip's field at rest against its own fit, the robot tip's poles refitted and the robot tip's
          calibration as it stands
presses   holds found in the stream (change over --threshold uT for at least --min-s s, each against the rest just before
          it), each estimated by
            physical      the viewer's estimator on the second tip's own rest fit (poles, strength, offsets), untrained
            physical+gp   that estimate corrected by the GP learned on sphere2's presses; its inputs and outputs are in the
                          pad's units, so it carries over to a tip whose physical model is rebuilt (fingertip_hybrid_cv.py)
            gp raw, mlp raw   sphere2's predictors from the 15 board components
            robot cal     the viewer's estimator on the robot tip's calibration
          With --landmarks (json list in hold order: {"name", "p": [x, y, z] mm in the pad frame or null, "f_n": N or
          null}) each estimate is scored; without it the pairwise distances of the holds are printed.

    .venv/bin/python scripts/fingertip_second_tip_transfer.py --since 11:05 [--until 11:12] [--landmarks f.json] [--preview]
        --out docs/experiments/20261023-second_fingertip_transfer/<name>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

STREAM = ROOT / "logs/20261020-contact_estimation/stream/20261022-171928-stream.bin"
SPHERE2 = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261022-210852-sphere2"
BOARD = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json"
DT = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])


def clock(s):
    """'11:05' or '11:05:30' today, or seconds since the epoch."""
    if ":" not in s:
        return float(s)
    h, m, *sec = (int(v) for v in s.split(":"))
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, sec[0] if sec else 0, 0, 0, -1))


def read(stream, t0, t1):
    """Records in [t0, t1], each chip's field divided by its gain at its die temperature over the 3 s before."""
    mm = np.memmap(stream, dtype=DT, mode="r")
    i0, i1 = np.searchsorted(mm["t"], [t0 - 3.0, t1])
    t, r = np.array(mm["t"][i0:i1]), np.array(mm["r"][i0:i1]).reshape(-1, 5, 4).astype(float)
    ok = (~np.all(r[:, :, 1:] == 0, axis=2).any(axis=1)) & (np.abs(r) < 1e5).all(axis=(1, 2))
    t, r = t[ok], r[ok]
    kap = W.load_thermal()
    T = r[:, :, 0]
    c = np.cumsum(np.r_[np.zeros((1, 5)), T], axis=0)
    j0, j1 = np.searchsorted(t, t - 3.0), np.arange(1, len(T) + 1)
    Tm = (c[j1] - c[j0]) / (j1 - j0)[:, None]
    b = W.thermal_correct(r[:, :, 1:], Tm, kap if kap is not None else np.full(5, -0.0067)).reshape(-1, 15)
    keep = t >= t0
    return t[keep], b[keep]


def holds(t, b, threshold, min_s, rest_s=1.0):
    """Press holds: runs where the change from a slow rest exceeds `threshold` (uT, norm of 15) for `min_s`, merged
    across gaps under 0.3 s; the hold is the middle half of the run, its rest the mean of the `rest_s` before it."""
    rest = np.median(b[t < t[0] + 1.0], axis=0)
    mag, still = np.empty(len(t)), np.zeros(len(t), bool)
    k0 = 0
    for k in range(len(t)):
        mag[k] = np.linalg.norm(b[k] - rest)
        still[k] = mag[k] < 0.5 * threshold
        if t[k] - t[k0] >= 0.5:                                  # the rest follows slow drift through still stretches
            j = np.searchsorted(t, t[k] - 2.0)
            sel = still[j:k + 1]
            if sel.sum() > 20 and sel.mean() > 0.9:
                rest = np.median(b[j:k + 1][sel], axis=0)
            k0 = k
    on = (mag > threshold).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, on, 0]))
    merged = []
    for s, e in zip(edges[::2], edges[1::2]):
        if merged and t[s] - t[merged[-1][1] - 1] < 0.3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        ta, tb = t[s], t[e - 1]
        if tb - ta < min_s:
            continue
        sel = (t >= ta + 0.25 * (tb - ta)) & (t <= ta + 0.75 * (tb - ta))
        rs = (t >= ta - rest_s - 0.3) & (t <= ta - 0.3) & still
        if sel.sum() >= 10 and rs.sum() >= 10:
            out.append(dict(t0=float(ta), t1=float(tb), n_rec=int(sel.sum()), d=b[sel].mean(0) - b[rs].mean(0),
                            peak_uT=float(mag[s:e].max())))
    return out


def train_sphere2():
    """sphere2's predictors on all its press holds: the GP correction of the untrained physical estimate, and the raw
    GP and MLP (as in scripts/fingertip_hybrid_cv.py)."""
    H, _, cal0 = PR.load(SPHERE2)
    basis = W.load_basis(PR.BASIS)
    ph0 = PR.Physical(cal0, basis)
    idx = np.arange(len(H["id"]))
    Pv, Fv = ph0.predict(H, idx)
    ok = np.all(np.isfinite(Pv), axis=1) & np.isfinite(Fv)
    Xb, Y = PR.features(H, 0), np.c_[H["p"], H["f_n"]]
    gp_h = PR.GP().fit(np.c_[Pv, Fv][ok], np.c_[H["p"][ok] - Pv[ok], H["f_n"][ok] - Fv[ok]], np.array([0.2, 0.2, 0.2, 0.05]))
    gp_raw = PR.GP().fit(Xb, Y, PR.floors(0))
    mlp_raw = PR.MLP().fit(Xb, Y, PR.floors(0))
    return dict(cal0=cal0, basis=basis, gp_h=gp_h, gp_raw=gp_raw, mlp_raw=mlp_raw, n=int(len(idx)), n_ok=int(ok.sum()))


def preview(own, c_true=1.15, delta=(-0.33, -1.21, 0.72), noise_uT=2.0, seed=0):
    """The transfer study of scripts/fingertip_magnet_transfer_sim.py with the second tip's own rest fit as a set D:
    sphere2's true contact points and forces, the board changes synthesized for set A (the robot tip) and set D, the
    predictors trained on A's training presses and scored on the held-out presses of each set."""
    from dataclasses import replace
    import fingertip_magnet_transfer_sim as MT
    H, _, cal0 = PR.load(SPHERE2)
    basis = W.load_basis(PR.BASIS)
    rbf = PR.Physical(cal0, basis).rbf
    train, test = PR.split(H)
    tr, te = np.nonzero(np.isin(H["id"], train))[0], np.nonzero(np.isin(H["id"], test))[0]
    P, F, N, fn = H["p"], H["f"], H["n"], H["f_n"]
    rng = np.random.default_rng(seed)
    sets = {"A": cal0, "D": replace(own, theta0=cal0.theta0, perm=cal0.perm, s=cal0.s, g_uT=cal0.g_uT)}
    D, EST, PHYS = {}, {}, {}
    for k, cal in sets.items():
        cal = replace(cal, b_rest_uT=W.Model(cal).rest_uT().tolist())
        sets[k] = cal
        D[k] = MT.synth(cal, P, F, rbf, rng, c_true, np.asarray(delta), noise_uT)
        est = W.SurfaceEstimator(W.Model(cal), basis, floor_uT=noise_uT)
        EST[k] = est
        ph, pf = np.full((len(P), 3), np.nan), np.full(len(P), np.nan)
        for i in range(len(P)):
            e = est.estimate(D[k][i])
            if e.get("contact"):
                ph[i], pf[i] = e["p"], e["f_n"]
        PHYS[k] = (ph, pf)

    def score(Pp, Fp, idx):
        loc = MT.tangential(Pp, P[idx], N[idx])
        return dict(loc_mm=float(np.nanmedian(loc)), loc_p90_mm=float(np.nanpercentile(loc, 90)),
                    force_N=float(np.nanmedian(np.abs(Fp - fn[idx]))))

    snap = lambda Y: PR.snap(EST["A"], Y)                                         # noqa: E731
    fl = PR.floors(0)
    gp_raw = PR.GP().fit(D["A"][tr], np.c_[P[tr], fn[tr]], fl)
    ok = np.all(np.isfinite(PHYS["A"][0][tr]), axis=1)
    gp_res = PR.GP(center=True).fit(np.c_[PHYS["A"][0][tr][ok], PHYS["A"][1][tr][ok]],
                                    np.c_[P[tr][ok] - PHYS["A"][0][tr][ok], fn[tr][ok] - PHYS["A"][1][tr][ok]],
                                    np.array([0.2, 0.2, 0.2, 0.05]))
    out = {}
    for k in sets:
        ph, pf = PHYS[k]
        sc = {"physical": score(ph[te], pf[te], te)}
        Y = gp_raw.predict(D[k][te])
        sc["gp raw (trained on A)"] = score(snap(Y[:, :3]), Y[:, 3], te)
        X = np.c_[ph[te], pf[te]]
        okt = np.all(np.isfinite(X), axis=1)
        Yc = np.full((len(te), 4), np.nan)
        Yc[okt] = gp_res.predict(X[okt])
        sc["physical+gp (trained on A)"] = score(snap(np.where(np.isfinite(ph[te]), ph[te], 0.0) + Yc[:, :3]),
                                                 pf[te] + Yc[:, 3], te)
        Yk = PR.GP().fit(D[k][tr], np.c_[P[tr], fn[tr]], fl).predict(D[k][te])
        sc[f"gp raw (retrained on {k})"] = score(snap(Yk[:, :3]), Yk[:, 3], te)
        out[k] = dict(k=sets[k].k, poles=list(sets[k].poles), median_change_uT=float(np.median(np.linalg.norm(D[k], axis=1))),
                      scores=sc)
        print(f"set {k} (k {sets[k].k:.3f}, poles {sets[k].poles}, |d| median {out[k]['median_change_uT']:.1f} uT): "
              + "; ".join(f"{m} {v['loc_mm']:.2f} mm, {v['force_N']:.3f} N" for m, v in sc.items()), flush=True)
    return dict(c_true=c_true, delta_mm=list(delta), noise_uT=noise_uT, n_train=int(len(tr)), n_test=int(len(te)), sets=out)


def estimate(est, d, n_rec):
    est.set_noise_scale(1.0 / math.sqrt(max(n_rec, 1)))
    e = est.estimate(d)
    return (np.array(e["p"], float), float(e["f_n"])) if e.get("contact") else (np.full(3, np.nan), float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", required=True, help="clock time today (11:05) or epoch seconds; the pad still for 2 s")
    ap.add_argument("--until", default=None)
    ap.add_argument("--stream", default=str(STREAM))
    ap.add_argument("--threshold", type=float, default=30.0)
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--landmarks", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", action="store_true", help="also run the transfer study in simulation with this tip as set D")
    a = ap.parse_args()
    t0 = clock(a.since)
    t1 = clock(a.until) if a.until else time.time()
    t, b = read(a.stream, t0, t1)
    if len(t) < 300:
        raise SystemExit(f"{len(t)} records between {a.since} and {a.until or 'now'}: the board was not streaming")
    print(f"{len(t)} records from {time.strftime('%H:%M:%S', time.localtime(t[0]))} to "
          f"{time.strftime('%H:%M:%S', time.localtime(t[-1]))}")
    board = W.Calib.load(BOARD)
    rest = np.median(b[t < t[0] + 2.0], axis=0).reshape(5, 3)
    own = W.calibrate_frames_fixed(rest, board)
    robot = W.Calib.load(SPHERE2 / "calib_rest.json")
    forced = W.calibrate_frames_fixed(rest, board, poles=robot.poles)
    as_is = float(np.sqrt(np.mean(((W.Model(robot).rest_uT() - rest) / 1e3) ** 2)))
    res = dict(rest=dict(own=dict(poles=own.poles, k=own.k, offset_mm=[own.dx, own.dy, own.dz], rms_mT=own.rms_mT,
                                  rms_next_mT=own.rms_next_mT),
                         robot_poles_refit=dict(poles=forced.poles, k=forced.k, rms_mT=forced.rms_mT),
                         robot_as_is=dict(poles=robot.poles, k=robot.k, rms_mT=as_is)))
    print(f"rest: own fit {own.poles} k {own.k:.3f} rms {own.rms_mT:.3f} mT (next pattern {own.rms_next_mT:.3f}); robot "
          f"poles refitted {forced.rms_mT:.3f} mT; robot calibration as it stands {as_is:.3f} mT")
    if a.preview:
        res["preview"] = preview(own)
    hs = holds(t, b, a.threshold, a.min_s)
    print(f"{len(hs)} holds over {a.threshold:.0f} uT for {a.min_s:.1f} s")
    if not hs:
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o)) + "\n")
        return
    tr = train_sphere2()
    print(f"sphere2 predictors trained on {tr['n']} press holds ({tr['n_ok']} with a physical estimate)")
    est_own = W.SurfaceEstimator(W.Model(own), tr["basis"], floor_uT=2.0)
    est_rob = W.SurfaceEstimator(W.Model(robot), tr["basis"], floor_uT=2.0)
    D = np.array([h["d"] for h in hs])
    raw = {m: tr[k].predict(D) for m, k in (("gp raw", "gp_raw"), ("mlp raw", "mlp_raw"))}
    rows = []
    for j, h in enumerate(hs):
        P, F = estimate(est_own, h["d"], h["n_rec"])
        r = {"physical": (P, F)}
        if np.all(np.isfinite(P)):
            C = tr["gp_h"].predict(np.r_[P, F][None])[0]
            r["physical+gp"] = (PR.snap(est_own, (P + C[:3])[None])[0], F + C[3])
        else:
            r["physical+gp"] = (np.full(3, np.nan), float("nan"))
        for m, Yp in raw.items():
            r[m] = (PR.snap(est_own, Yp[j:j + 1, :3])[0], float(Yp[j, 3]))
        r["robot cal"] = estimate(est_rob, h["d"], h["n_rec"])
        rows.append(dict(t0=time.strftime("%H:%M:%S", time.localtime(h["t0"])), s=round(h["t1"] - h["t0"], 1),
                         change_uT=round(float(np.linalg.norm(h["d"])), 1),
                         est={m: dict(p=[round(float(v), 2) for v in p], f_n=round(f, 3)) for m, (p, f) in r.items()}))
    models = list(rows[0]["est"])
    print("hold  start     s    change uT  " + "  ".join(f"{m:>24s}" for m in models))
    for j, rw in enumerate(rows):
        cells = [f"({e['p'][0]:5.1f},{e['p'][1]:5.1f},{e['p'][2]:5.1f}) {e['f_n']:5.2f}N" for e in rw["est"].values()]
        print(f"{j:3d}   {rw['t0']}  {rw['s']:4.1f}  {rw['change_uT']:8.1f}  " + "  ".join(f"{c:>24s}" for c in cells))
    if a.landmarks:
        lm = json.loads(Path(a.landmarks).read_text())
        sc = {}
        for m in models:
            loc, fe = [], []
            for rw, L in zip(rows, lm):
                e = rw["est"][m]
                if L.get("p") is not None and all(np.isfinite(e["p"])):
                    loc.append(float(np.linalg.norm(np.array(e["p"]) - np.array(L["p"]))))
                if L.get("f_n") is not None and np.isfinite(e["f_n"]):
                    fe.append(abs(e["f_n"] - L["f_n"]))
            sc[m] = dict(loc_median_mm=float(np.median(loc)) if loc else None, force_median_N=float(np.median(fe)) if fe else None,
                         n_loc=len(loc), n_force=len(fe))
            print(f"{m:12s} location {sc[m]['loc_median_mm']} mm over {len(loc)}, force {sc[m]['force_median_N']} N over {len(fe)}")
        res["scores"] = sc
    res["holds"] = rows
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o)) + "\n")


if __name__ == "__main__":
    main()

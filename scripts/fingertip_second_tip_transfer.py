#!/usr/bin/env python3
"""The UR5 run sphere2's calibration, made on the robot fingertip (board 9c:d0, all five poles one way), applied to a
second fingertip on the workstation (board d9:94, poles (-1, -1, 1, 1, 1)), from the live viewer's stream log or a saved
recording (owner, 2026-10-23: "now that we have this known calibration from the robot fingertip, can we see how well that
applies to the fingertip plugged in here? ... this has a different setup of polarities").

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
          null}) each estimate is scored; without it each predictor's points are summarized against REF, the UR5
          protocol's reference press at the face centre.
--save-records  the raw records of the window (and the 3 s before it) as an .npz with t and r, which --stream reads
                back and wrench_viewer.py --replay plays
--trace         the change and the estimates in --bin-s bins over the window, for the page's figure
--step          the change of the rest between two windows, fitted by a uniform field and by a rigid offset of the pad

    .venv/bin/python scripts/fingertip_second_tip_transfer.py --since 12:27:20 [--until 12:28:25] [--stream rec.npz]
        [--landmarks f.json] [--preview] [--trace] [--step 12:14:04 12:15:09 12:16:34 12:19:44] [--save-records rec.npz]
        --out docs/experiments/20261023-second_fingertip_transfer/data/<name>.json
Clock times are on the day of the recording's first record when --stream is an .npz, today otherwise.
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
REF = np.array([8.475, 0.502, 10.360])   # the UR5 protocol's reference press: pad point 788, the face centre (mm, pad frame)


def clock(s, day=None):
    """'11:05' or '11:05:30' on the day of `day` (epoch seconds; today if None), or seconds since the epoch."""
    if ":" not in s:
        return float(s)
    h, m, *sec = (int(v) for v in s.split(":"))
    lt = time.localtime(time.time() if day is None else day)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, sec[0] if sec else 0, 0, 0, -1))


def records(stream, t0, t1):
    """Host times (n,) and raw records (n, 5, 4: t x y z per chip) in [t0, t1] of a stream log (.bin) or a saved
    recording (.npz with t and r)."""
    if str(stream).endswith(".npz"):
        z = np.load(stream)
        tt, rr = z["t"], z["r"]
    else:
        mm = np.memmap(stream, dtype=DT, mode="r")
        tt, rr = mm["t"], mm["r"]
    i0, i1 = np.searchsorted(tt, [t0, t1])
    return np.array(tt[i0:i1]), np.array(rr[i0:i1]).reshape(-1, 5, 4).astype(float)


def first_time(stream):
    return float(np.load(stream)["t"][0]) if str(stream).endswith(".npz") else None


def read(stream, t0, t1):
    """Records in [t0, t1], each chip's field divided by its gain at its die temperature over the 3 s before."""
    t, r = records(stream, t0 - 3.0, t1)
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


def segment(t, b, threshold):
    """The change from a slow rest at each record (uT, norm of 15), the records at rest, and the runs of records over
    `threshold` as index pairs, merged across gaps under 0.3 s."""
    rest = np.median(b[t < t[0] + 1.0], axis=0)
    mag, still = np.empty(len(t)), np.zeros(len(t), bool)
    k0 = 0
    for k in range(len(t)):
        mag[k] = np.linalg.norm(b[k] - rest)
        still[k] = mag[k] < 0.5 * threshold
        if t[k] - t[k0] >= 0.5:
            # a new rest whenever the field holds steady for 4 s (longer than a press), wherever it settles: a turn of
            # the board in the Earth's field or a shift of the pad in its seat moves the rest by tens of uT
            j = np.searchsorted(t, t[k] - 4.0)
            if t[k] - t[j] > 3.5 and np.all(b[j:k + 1].std(0) < 3.0):
                rest = b[j:k + 1].mean(0)
                still[j:k + 1] = True
            k0 = k
    on = (mag > threshold).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, on, 0]))
    merged = []
    for s, e in zip(edges[::2], edges[1::2]):
        if merged and t[s] - t[merged[-1][1] - 1] < 0.3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return mag, still, merged


def holds(t, b, threshold, min_s, rest_s=1.0):
    """Press holds: runs over `threshold` for at least `min_s`; the hold is the middle half of the run, its rest the
    mean of the `rest_s` before it."""
    mag, still, merged = segment(t, b, threshold)
    out = []
    for s, e in merged:
        ta, tb = t[s], t[e - 1]
        if tb - ta < min_s:
            continue
        sel = (t >= ta + 0.25 * (tb - ta)) & (t <= ta + 0.75 * (tb - ta))
        rs = (t >= ta - rest_s - 0.3) & (t <= ta - 0.3) & still
        if sel.sum() >= 10 and rs.sum() >= 10:
            out.append(dict(t0=float(ta), t1=float(tb), n_rec=int(sel.sum()), d=b[sel].mean(0) - b[rs].mean(0),
                            rest=b[rs].mean(0), peak_uT=float(mag[s:e].max())))
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
    return dict(cal0=cal0, basis=basis, gp_h=gp_h, gp_raw=gp_raw, mlp_raw=mlp_raw, n=int(len(idx)), n_ok=int(ok.sum()),
                n_presses=int(len(set(H["id"]))), Xb=Xb, Y=Y)


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


def corrected(est_own, gp_h, P, F):
    """The physical estimate (P, F) corrected by sphere2's GP, snapped onto the pad's surface."""
    if not np.all(np.isfinite(P)):
        return np.full(3, np.nan), float("nan")
    C = gp_h.predict(np.r_[P, F][None])[0]
    return PR.snap(est_own, (P + C[:3])[None])[0], F + C[3]


def rest_step(t, b, own, pre, post):
    """The change of the rest from the window `pre` to `post` (epoch seconds), fitted by a uniform field read by every
    chip (the board turning in the Earth's field), by a rigid offset of the pad on the board, and by both."""
    A = b[(t >= pre[0]) & (t < pre[1])].mean(0)
    B = b[(t >= post[0]) & (t < post[1])].mean(0)
    step = B - A
    M = W.Model(own)
    U = np.stack([M.R[:, :, a].ravel() for a in range(3)], 1)                # a uniform field of 1 uT, read by each chip

    def rest_at(dx, dy, dz):
        c = W.Calib(**{**own.__dict__, "dx": own.dx + dx, "dy": own.dy + dy, "dz": own.dz + dz})
        return W.Model(c).rest_uT().ravel()

    J = np.stack([(rest_at(*e) - rest_at(*(-e))) / 0.02 for e in np.eye(3) * 0.01], 1)   # uT per mm of pad offset
    g = np.linalg.lstsq(U, step, rcond=None)[0]
    o = np.linalg.lstsq(J, step, rcond=None)[0]
    x = np.linalg.lstsq(np.c_[U, J], step, rcond=None)[0]
    r = lambda v: [round(float(q), 2) for q in v]                                  # noqa: E731
    return dict(pre_s=[float(pre[0]), float(pre[1])], post_s=[float(post[0]), float(post[1])],
                step_uT=r(step), step_norm_uT=float(np.linalg.norm(step)),
                per_chip_uT=r(np.linalg.norm(step.reshape(5, 3), axis=1)),
                uniform=dict(g_uT=r(g), g_norm_uT=float(np.linalg.norm(g)), fit_uT=r(U @ g),
                             residual_uT=float(np.linalg.norm(step - U @ g))),
                offset=dict(o_um=r(o * 1e3), fit_uT=r(J @ o), residual_uT=float(np.linalg.norm(step - J @ o))),
                both=dict(g_uT=r(x[:3]), o_um=r(x[3:] * 1e3), residual_uT=float(np.linalg.norm(step - np.c_[U, J] @ x))))


def trace(t, b, threshold, bin_s, est_own, est_rob, gp_h, rest_s=1.0):
    """The change and the estimates in bins of `bin_s` over the window: inside a run over `threshold` against the rest
    before that run (as the holds), elsewhere against the slow rest (change only)."""
    mag, still, merged = segment(t, b, threshold)
    runs = []
    for s, e in merged:
        ta, tb = t[s], t[e - 1]
        rs = (t >= ta - rest_s - 0.3) & (t <= ta - 0.3) & still
        if rs.sum() >= 10:
            runs.append((ta, tb, b[rs].mean(0)))
    rows = []
    for a in np.arange(t[0], t[-1], bin_s):
        m = (t >= a) & (t < a + bin_s)
        if m.sum() < 5:
            continue
        tm = a + 0.5 * bin_s
        row = dict(t=round(float(tm - t[0]), 3), change_uT=round(float(np.median(mag[m])), 1))
        run = next((r for r in runs if r[0] <= tm <= r[1]), None)
        if run is not None:
            d = b[m].mean(0) - run[2]
            row["change_uT"] = round(float(np.linalg.norm(d)), 1)
            P, F = estimate(est_own, d, int(m.sum()))
            Pc, Fc = corrected(est_own, gp_h, P, F)
            Pr, Fr = estimate(est_rob, d, int(m.sum()))
            for k, (p, f) in (("physical", (P, F)), ("physical+gp", (Pc, Fc)), ("robot cal", (Pr, Fr))):
                if np.all(np.isfinite(p)):
                    row[k] = dict(p=[round(float(v), 2) for v in p], f_n=round(float(f), 3))
        rows.append(row)
    return dict(t0=float(t[0]), bin_s=bin_s, threshold_uT=threshold, rows=rows)


def summary(rows, Xb=None, D=None):
    """Each predictor's points over the holds against REF: the mean point, its distance from REF, the holds' largest
    distance from that mean, the force range and the number of distinct answers."""
    out = {}
    for m in rows[0]["est"]:
        P = np.array([r["est"][m]["p"] for r in rows], float)
        F = np.array([r["est"][m]["f_n"] for r in rows], float)
        ok = np.all(np.isfinite(P), axis=1)
        if not ok.any():
            continue
        mu = P[ok].mean(0)
        out[m] = dict(n=int(ok.sum()), mean_p=[round(float(v), 2) for v in mu],
                      mean_minus_ref=[round(float(v), 2) for v in mu - REF],
                      mean_from_ref_mm=float(np.linalg.norm(mu - REF)),
                      from_ref_median_mm=float(np.median(np.linalg.norm(P[ok] - REF, axis=1))),
                      spread_max_mm=float(np.linalg.norm(P[ok] - mu, axis=1).max()),
                      f_min=float(np.nanmin(F)), f_max=float(np.nanmax(F)), f_mean=float(np.nanmean(F)),
                      distinct=int(len({tuple(np.round(p, 2)) for p in P[ok]})))
    if Xb is not None and D is not None:
        # how far the presses' board changes lie from sphere2's training changes, against the training set's own spacing
        dd = np.linalg.norm(D[:, None, :] - Xb[None], axis=2).min(1)
        tt = np.linalg.norm(Xb[:, None, :] - Xb[None], axis=2)
        np.fill_diagonal(tt, np.inf)
        out["inputs"] = dict(nearest_training_uT=[round(float(v), 1) for v in dd],
                             training_nearest_median_uT=float(np.median(tt.min(1))),
                             training_norm_median_uT=float(np.median(np.linalg.norm(Xb, axis=1))),
                             press_norm_uT=[round(float(v), 1) for v in np.linalg.norm(D, axis=1)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", required=True, help="clock time (11:05) or epoch seconds; the pad still for 2 s")
    ap.add_argument("--until", default=None)
    ap.add_argument("--stream", default=str(STREAM), help="the viewer's stream log (.bin) or a saved recording (.npz)")
    ap.add_argument("--threshold", type=float, default=30.0)
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--landmarks", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", action="store_true", help="also run the transfer study in simulation with this tip as set D")
    ap.add_argument("--save-records", default=None, help="write the window's raw records to this .npz")
    ap.add_argument("--trace", action="store_true", help="add the change and the estimates in --bin-s bins")
    ap.add_argument("--bin-s", type=float, default=0.2)
    ap.add_argument("--step", nargs=4, default=None, metavar=("PRE0", "PRE1", "POST0", "POST1"),
                    help="fit the change of the rest between the windows [PRE0, PRE1] and [POST0, POST1]")
    a = ap.parse_args()
    day = first_time(a.stream)
    t0 = clock(a.since, day)
    t1 = clock(a.until, day) if a.until else time.time()
    if a.save_records:
        tr_, rr_ = records(a.stream, t0 - 3.0, t1)
        np.savez_compressed(a.save_records, t=tr_, r=rr_.astype(np.float32))
        print(f"{len(tr_)} raw records -> {a.save_records}")
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
    res = dict(window=dict(since=a.since, until=a.until, stream=str(Path(a.stream).name), t0=float(t[0]), t1=float(t[-1]),
                           n_records=int(len(t))),
               rest=dict(own=dict(poles=own.poles, k=own.k, offset_mm=[own.dx, own.dy, own.dz], rms_mT=own.rms_mT,
                                  rms_next_mT=own.rms_next_mT, rest_norm_mT=float(np.linalg.norm(rest) / 1e3),
                                  chip_norm_mT=[round(float(v), 3) for v in np.linalg.norm(rest, axis=1) / 1e3]),
                         robot_poles_refit=dict(poles=forced.poles, k=forced.k, rms_mT=forced.rms_mT),
                         robot_as_is=dict(poles=robot.poles, k=robot.k, rms_mT=as_is, rms_own_mT=robot.rms_mT,
                                          offset_mm=[robot.dx, robot.dy, robot.dz])))
    print(f"rest: own fit {own.poles} k {own.k:.3f} rms {own.rms_mT:.3f} mT (next pattern {own.rms_next_mT:.3f}); robot "
          f"poles refitted {forced.rms_mT:.3f} mT; robot calibration as it stands {as_is:.3f} mT")
    if a.step:
        w = [clock(s, day) for s in a.step]
        res["step"] = rest_step(t, b, own, w[:2], w[2:])
        st = res["step"]
        print(f"rest step {st['step_norm_uT']:.1f} uT: uniform field {st['uniform']['g_norm_uT']:.1f} uT leaves "
              f"{st['uniform']['residual_uT']:.1f}; pad offset {st['offset']['o_um']} um leaves {st['offset']['residual_uT']:.1f}")
    if a.preview:
        res["preview"] = preview(own)
    hs = holds(t, b, a.threshold, a.min_s)
    print(f"{len(hs)} holds over {a.threshold:.0f} uT for {a.min_s:.1f} s")
    if not hs:
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o)) + "\n")
        return
    tr = train_sphere2()
    res["sphere2"] = dict(holds=tr["n"], presses=tr["n_presses"], holds_with_physical=tr["n_ok"],
                          train_mean=[round(float(v), 3) for v in tr["Y"].mean(0)])
    print(f"sphere2 predictors trained on {tr['n']} press holds of {tr['n_presses']} presses ({tr['n_ok']} with a "
          "physical estimate)")
    est_own = W.SurfaceEstimator(W.Model(own), tr["basis"], floor_uT=2.0)
    est_rob = W.SurfaceEstimator(W.Model(robot), tr["basis"], floor_uT=2.0)
    D = np.array([h["d"] for h in hs])
    raw = {m: tr[k].predict(D) for m, k in (("gp raw", "gp_raw"), ("mlp raw", "mlp_raw"))}
    rows = []
    for j, h in enumerate(hs):
        P, F = estimate(est_own, h["d"], h["n_rec"])
        r = {"physical": (P, F), "physical+gp": corrected(est_own, tr["gp_h"], P, F)}
        for m, Yp in raw.items():
            r[m] = (PR.snap(est_own, Yp[j:j + 1, :3])[0], float(Yp[j, 3]))
        r["robot cal"] = estimate(est_rob, h["d"], h["n_rec"])
        rows.append(dict(t0=time.strftime("%H:%M:%S", time.localtime(h["t0"])), t0_s=round(h["t0"] - t[0], 2),
                         t1_s=round(h["t1"] - t[0], 2), s=round(h["t1"] - h["t0"], 1),
                         change_uT=round(float(np.linalg.norm(h["d"])), 1), peak_uT=round(h["peak_uT"], 1),
                         d_uT=[round(float(v), 2) for v in h["d"]],
                         est={m: dict(p=[round(float(v), 2) for v in p], f_n=round(f, 3)) for m, (p, f) in r.items()}))
    models = list(rows[0]["est"])
    print("hold  start     s    change uT  " + "  ".join(f"{m:>24s}" for m in models))
    for j, rw in enumerate(rows):
        cells = [f"({e['p'][0]:5.1f},{e['p'][1]:5.1f},{e['p'][2]:5.1f}) {e['f_n']:5.2f}N" for e in rw["est"].values()]
        print(f"{j:3d}   {rw['t0']}  {rw['s']:4.1f}  {rw['change_uT']:8.1f}  " + "  ".join(f"{c:>24s}" for c in cells))
    res["summary"] = summary(rows, tr["Xb"], D)
    for m, v in res["summary"].items():
        if m != "inputs":
            print(f"{m:12s} mean {v['mean_p']} mm, {v['mean_from_ref_mm']:.2f} mm from REF, spread {v['spread_max_mm']:.2f} mm, "
                  f"{v['f_min']:.2f}-{v['f_max']:.2f} N, {v['distinct']} distinct")
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
    if a.trace:
        res["trace"] = trace(t, b, a.threshold, a.bin_s, est_own, est_rob, tr["gp_h"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o)) + "\n")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()

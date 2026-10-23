#!/usr/bin/env python3
"""Rejection of fields from outside the pad (steel near the tip, another magnet) by the contact estimator: the test of
fingertip_wrench_model.ForeignField on the owner's touches of a steel mass and a steel ruler on the second fingertip
(2026-10-23), on that fingertip's presses of the morning and on the UR5 sessions' holds and streams (owner, 2026-10-23:
"can you determine some abstract rule to ignore these ferromagnetic presences? even if it works, it probably needs to
be a toggle"; "they were not presses--i just touched the object to the approximate center").

touches   the raw records of the window (--save writes them to data/), its steady plateaus (a change over 50 uT for at
          least 0.8 s, each against the 0.6 s before it), and for each plateau and each round-2 press of
          20261023-second_fingertip_transfer: the estimate as it stands, Lambda, and the shares of the change explained
          by a uniform field, by the far field and by the magnets' images in a steel plane at the face
holds     Lambda on every press and turn hold of the UR5 sessions sphere2 and rod1 (joint registration)
frames    the viewer's frames (50 ms window, re-zero at rest, the display rule) over a recording with the test off and on;
          for a UR5 stream also the frames inside each hold
episodes  the frames of a UR5 stream that the test rejected, in runs, with the action running and the HEX-E's force
approach  the six touches' changes scaled from 2 to 100 % with the chips' noise added: the force shown as it stands and
          the share rejected as the object comes nearer

    .venv/bin/python scripts/fingertip_foreign_field.py touches [--save]
    .venv/bin/python scripts/fingertip_foreign_field.py holds
    .venv/bin/python scripts/fingertip_foreign_field.py frames {touches,round1,round2,sphere2,rod1}
    .venv/bin/python scripts/fingertip_foreign_field.py episodes {sphere2,rod1}
    .venv/bin/python scripts/fingertip_foreign_field.py approach
Each writes docs/experiments/20261023-foreign_field_rejection/data/<name>.json (frames also <name>.npz).
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
import fingertip_second_tip_transfer as T  # noqa: E402
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

OUT = ROOT / "docs/experiments/20261023-foreign_field_rejection"
DATA = OUT / "data"
CAL = DATA / "20261023-105631-calib.json"             # the viewer's calibration of the second fingertip at 10:56:31
REC = DATA / "20261023-153800_touch_records.npz"      # the raw records of 15:38:00-15:44:15
WINDOW = ("15:38:00", "15:44:15")
SIX = ("15:43:15", "15:43:45")                        # the six touches the owner described
EARLY = ("15:38:30", "15:40:30")                      # the handling two minutes before them
TRANSFER = ROOT / "docs/experiments/20261023-second_fingertip_transfer/data"
UR5 = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
SESSIONS = {"sphere2": ("data/20261022-210852-sphere2-joint", "lab/20261022-210852-sphere2"),
            "rod1": ("data/20261023-104526-rod1-joint", "lab/20261023-104526-rod1")}
FACE_X = 8.475                                        # the top of the palmar face (mm, tip frame): pad point 788
VIEW_N = 0.05 * 140                                   # the viewer's 50 ms window at the 140 Hz it assumes
THRESHOLD, MIN_F = 25.0, 0.25                         # the viewer's display rule: explained chi-square, pressing force


def basis():
    return W.load_basis(PR.BASIS)


def estimator(cal, b=None):
    est = W.SurfaceEstimator(W.Model(cal), b if b is not None else basis(), floor_uT=2.0)
    return est, W.ForeignField(est)


def image(M, xs):
    """Change (uT, (15,)) from the magnets' images in a perfect steel plane x = xs, parallel to the face: each image sits
    at the magnet's mirror point with the same moment (the moments lie along the plane's normal)."""
    c = M.mc.copy()
    c[:, 0] = 2 * xs - c[:, 0]
    pts = (c[:, None, :] + M.q_off[None]).reshape(-1, 3)
    mom = (M.mu[:, None, :] * M.q_w[None, :, None]).reshape(-1, 3)
    return (np.einsum("cij,cj->ci", M.R, W.dipole_field(M.P, pts, mom)) * 1e6).ravel()


def share(A, d):
    """The share of |d|^2 explained by the columns of A (least squares)."""
    r = d - A @ np.linalg.lstsq(A, d, rcond=None)[0]
    return float(1 - (r @ r) / (d @ d))


def describe(est, ff, d, n_rec):
    """The estimate as it stands, Lambda and the explained shares of one change."""
    est.set_noise_scale(1.0 / math.sqrt(max(n_rec, 1)))
    e = est.estimate(d)
    M = est.model
    U = est.Nu[:, :3]
    best = max(((share(image(M, xs)[:, None], d), xs) for xs in np.arange(FACE_X, 20.01, 0.25)))
    out = dict(change_uT=round(float(np.linalg.norm(d)), 1), lam=round(ff.lam(d, e), 1),
               contact=bool(e.get("contact")), shown=bool(e.get("contact") and e["e"] > THRESHOLD and e["f_n"] >= MIN_F),
               share_uniform=round(share(U, d), 3), share_far=round(share(W.far_field_basis(M), d), 3),
               share_image_face=round(share(np.c_[image(M, FACE_X), U], d), 3),
               share_image_best=round(best[0], 3), image_best_xs=float(best[1]),
               share_contact=round(e["e"] / e["chi2_none"], 3) if e.get("contact") else 0.0, d_uT=np.round(d, 2).tolist())
    if e.get("contact"):
        out.update(p=np.round(e["p"], 2).tolist(), f_n=round(e["f_n"], 2), region=region(np.array(e["n"])))
    return out


def region(n):
    if n[0] > 0.7:
        return "palmar face"
    if n[2] > 0.6:
        return "tip end"
    if n[2] < -0.6:
        return "mount end"
    side = "+y side" if n[1] > 0 else "-y side"
    return side if n[0] < 0.3 else f"face edge, {side}"


def plateaus(t, b, thr=50.0, min_s=0.8, rest_s=0.6):
    """Steady stretches of a change over thr uT lasting at least min_s, each as its middle 40 % against the rest_s
    before it. The rest is the median of the first 2 s, taken again over every quiet 1.5 s."""
    from scipy.ndimage import median_filter
    bm = median_filter(b, size=(5, 1))                  # single-record glitches of the stream out
    rest = np.median(bm[t < t[0] + 2.0], axis=0)
    on = np.linalg.norm(bm - rest, axis=1) > thr
    out, i, n = [], 0, len(t)
    while i < n:
        if not on[i]:
            j = np.searchsorted(t, t[i] - 1.5)
            if i % 50 == 0 and t[i] - t[j] > 1.4 and np.all(np.linalg.norm(bm[j:i + 1] - bm[j:i + 1].mean(0), axis=1) < 12):
                rest = bm[j:i + 1].mean(0)
                on = np.linalg.norm(bm - rest, axis=1) > thr
            i += 1
            continue
        s = i
        while i < n and on[i]:
            i += 1
        ta, tb = t[s], t[i - 1]
        if tb - ta >= min_s:
            sel = (t >= ta + 0.3 * (tb - ta)) & (t <= ta + 0.7 * (tb - ta))
            rs = (t >= ta - rest_s - 0.2) & (t <= ta - 0.2)
            out.append(dict(t0=float(ta), t1=float(tb), d=b[sel].mean(0) - b[rs].mean(0)))
    return out


def hms(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def touches(a):
    day = T.first_time(str(REC)) if REC.exists() and not a.save else None
    t0, t1 = T.clock(WINDOW[0], day), T.clock(WINDOW[1], day)
    if a.save:
        tr, rr = T.records(T.STREAM, t0 - 3.0, t1)
        np.savez_compressed(REC, t=tr, r=rr.astype(np.float32))
        print(f"{len(tr)} raw records -> {REC.relative_to(ROOT)}")
    t, b = T.read(str(REC), t0, t1)
    cal = W.Calib.load(CAL)
    est, ff = estimator(cal)
    six = (T.clock(SIX[0], day), T.clock(SIX[1], day))
    early = (T.clock(EARLY[0], day), T.clock(EARLY[1], day))
    rows = []
    for p in plateaus(t, b):
        r = dict(t0=hms(p["t0"]), t0_s=round(p["t0"] - t0, 2), s=round(p["t1"] - p["t0"], 2),
                 group="six" if six[0] <= p["t0"] <= six[1] else "early" if early[0] <= p["t0"] <= early[1] else "other",
                 **describe(est, ff, p["d"], VIEW_N))
        rows.append(r)
        print(f"{r['group']:5s} {r['t0']} {r['s']:5.2f} s  |d| {r['change_uT']:6.1f}  Lambda {r['lam']:7.1f}  shown {r['shown']!s:5s} "
              f"f_n {r.get('f_n', 0):5.2f} {r.get('region', '-'):16s} far {r['share_far']:.2f} contact {r['share_contact']:.2f}")
    z2 = TRANSFER / "20261023-122717_round2_records.npz"
    tz = np.load(z2)["t"]
    t2, b2 = T.read(str(z2), float(tz[0]) + 3.0, float(tz[-1]))
    presses = []
    for p in plateaus(t2, b2, thr=30.0, min_s=1.0):
        r = dict(t0=hms(p["t0"]), s=round(p["t1"] - p["t0"], 2), **describe(est, ff, p["d"], VIEW_N))
        presses.append(r)
        print(f"press {r['t0']} {r['s']:5.2f} s  |d| {r['change_uT']:6.1f}  Lambda {r['lam']:7.1f}  f_n {r.get('f_n', 0):5.2f} "
              f"{r.get('region', '-'):16s} far {r['share_far']:.2f} contact {r['share_contact']:.2f}")
    M = est.model
    res = dict(window=dict(since=WINDOW[0], until=WINDOW[1], six=list(SIX), early=list(EARLY), records=REC.name,
                           t0=float(t[0]), n=int(len(t))),
               calib=dict(file=CAL.name, poles=list(cal.poles), k=cal.k, rms_mT=cal.rms_mT),
               geometry=dict(face_x_mm=FACE_X, magnet_x_mm=float(M.mc[:, 0].mean()), chip_x_mm=float(M.P[:, 0].mean()),
                             image_face_uT=round(float(np.linalg.norm(image(M, FACE_X))), 1),
                             image_face_chip_uT=np.round(np.linalg.norm(image(M, FACE_X).reshape(5, 3), axis=1), 1).tolist(),
                             rest_chip_mT=np.round(np.linalg.norm(np.array(cal.b_rest_uT), axis=1) / 1e3, 2).tolist()),
               plateaus=rows, presses=presses)
    write("touches", res)


def holds(a):
    b = basis()
    res = {}
    for name, (dfold, _) in SESSIONS.items():
        H, Tt, cal0 = PR.load(UR5 / dfold)
        est, ff = estimator(cal0, b)
        for kind, X in (("press", H), ("turn", Tt)):
            L, Fn, S = [], [], []
            for i in range(len(X["id"])):
                est.set_noise_scale(1.0 / math.sqrt(max(X["n_rec"][i], 1)))
                e = est.estimate(X["d_board"][i])
                L.append(ff.lam(X["d_board"][i], e))
                Fn.append(e.get("f_n", 0.0))
                S.append(bool(e.get("contact") and e["e"] > THRESHOLD and e["f_n"] >= MIN_F))
            L = np.array(L)
            res[f"{name}_{kind}"] = dict(n=int(len(L)), lam=np.round(L, 2).tolist(), f_n=np.round(X["f_n"], 3).tolist(),
                                         level_N=np.round(X["level_N"], 3).tolist(), region=X["region"].tolist(),
                                         kind=X["kind"].tolist(), change_uT=np.round(np.linalg.norm(X["d_board"], axis=1), 1).tolist(),
                                         est_f_n=np.round(Fn, 3).tolist(), shown=S, lam_max=float(L.max()),
                                         over_on=int((L > ff.on).sum()))
            print(f"{name:8s} {kind:5s} n {len(L):3d}  Lambda max {L.max():6.1f}  p99 {np.percentile(L, 99):6.1f}  over {ff.on:.0f}: {(L > ff.on).sum()}")
    write("holds", res)


def replay(chunks, cal, n_win=VIEW_N, fps=30.0, b=None):
    """The viewer's frames (wrench_viewer.App.frame) over (t, b) chunks: the mean of the last 50 ms against the base, the
    base taken again by RestZero, a contact shown when the estimate explains over THRESHOLD and presses at least MIN_F
    (0.6 of it while shown). Rows: t, |d|, Lambda (nan without a contact), f_n, shown with the test off, shown with it on."""
    est, ff = estimator(cal, b)
    est.set_noise_scale(1 / math.sqrt(n_win))
    rz = W.RestZero(hold_s=3.0)
    base, prev, rows = None, [False, False], []
    for t, bb in chunks:
        if base is None:
            base = np.median(bb[t < t[0] + 2.0], axis=0)
            start = t[0] + 2.0
        else:
            start = t[0]
        for tf in np.arange(start, t[-1], 1.0 / fps):
            i1 = np.searchsorted(t, tf)
            i0 = np.searchsorted(t, tf - 0.05)
            if i1 - i0 < 2:
                continue
            d = bb[i0:i1].mean(0) - base
            j0 = np.searchsorted(t, tf - rz.span_s)
            nb = rz.update(t[j0:i1], bb[j0:i1], base)
            if nb is not None:
                d, base = d + base - nb, nb
            e = est.estimate(d)
            ok = bool(e.get("contact") and e["e"] > THRESHOLD)
            lam = ff.lam(d, e) if ok else float("nan")
            row = [tf, float(np.linalg.norm(d)), lam, e.get("f_n", 0.0)]
            for k, rej in enumerate((False, ok and lam > ff.on)):
                lim = MIN_F * (0.6 if prev[k] else 1.0)
                prev[k] = bool(ok and e["f_n"] >= lim and not rej)
                row.append(float(prev[k]))
            rows.append(row)
    return np.array(rows)


def stream_chunks(path, t0, t1, chunk_s=600.0):
    for a in np.arange(t0, t1, chunk_s):
        t, b = T.read(str(path), a, min(a + chunk_s, t1))
        if len(t) > 100:
            yield t, b


def frames(a):
    if a.recording in SESSIONS:
        dfold, lab = SESSIONS[a.recording]
        H, Tt, cal = PR.load(UR5 / dfold)
        path = UR5 / lab / "board.bin"
        mm = np.memmap(path, dtype=T.DT, mode="r")
        t0, t1 = float(mm["t"][0]) + 3.0, float(mm["t"][-1])
        rate = len(mm) / (t1 - t0)
        del mm
        F = replay(stream_chunks(path, t0, t1), cal, fps=a.fps)
        hold = np.zeros(len(F), bool)
        for X in (H, Tt):
            for tc, n in zip(X["t"], X["n_rec"]):
                hold |= np.abs(F[:, 0] - tc) <= 0.5 * n / rate
    else:
        cal = W.Calib.load(CAL)
        if a.recording == "touches":
            path = REC
            day = T.first_time(str(path))
            t0, t1 = T.clock(WINDOW[0], day), T.clock(WINDOW[1], day)
        else:
            path = TRANSFER / {"round1": "20261023-121357_round1_records.npz", "round2": "20261023-122717_round2_records.npz"}[a.recording]
            tz = np.load(path)["t"]
            t0, t1 = float(tz[0]) + 3.0, float(tz[-1])
        t, b = T.read(str(path), t0, t1)
        F = replay([(t, b)], cal, fps=a.fps)
        hold = None
    off, on = F[:, 4] > 0, F[:, 5] > 0
    lam = F[:, 2]
    res = dict(recording=a.recording, source=str(Path(path).relative_to(ROOT)), fps=a.fps, frames=int(len(F)),
               shown_off=int(off.sum()), shown_on=int(on.sum()), rejected=int((off & ~on).sum()),
               lam_max=float(np.nanmax(lam)) if np.isfinite(lam).any() else None,
               lam_over_on=int(np.nansum(lam > 25.0)))
    if hold is not None:
        res.update(hold_frames=int(hold.sum()), hold_shown_off=int((off & hold).sum()), hold_shown_on=int((on & hold).sum()),
                   hold_rejected=int((off & ~on & hold).sum()), between_rejected=int((off & ~on & ~hold).sum()),
                   lam_hold_max=float(np.nanmax(np.where(hold, lam, np.nan))))
    if a.recording == "touches":
        R = json.loads((DATA / "touches.json").read_text())
        for g in ("six", "early"):
            m = np.zeros(len(F), bool)
            for p in R["plateaus"]:
                if p["group"] == g:
                    m |= (F[:, 0] >= R["window"]["t0"] + p["t0_s"] - 0.3) & (F[:, 0] <= R["window"]["t0"] + p["t0_s"] + p["s"] + 0.3)
            res[g] = dict(frames=int(m.sum()), shown_off=int((off & m).sum()), shown_on=int((on & m).sum()),
                          f_max_off=float(F[off & m, 3].max()) if (off & m).any() else 0.0,
                          f_max_on=float(F[on & m, 3].max()) if (on & m).any() else 0.0)
    print(json.dumps(res, indent=1))
    # frame times as offsets from the first in float32 (0.5 ms over 2.3 h): epoch seconds in float32 resolve 128 s
    np.savez_compressed(DATA / f"frames_{a.recording}.npz", t0=F[0, 0], dt=(F[:, 0] - F[0, 0]).astype(np.float32),
                        F=F[:, 1:].astype(np.float32), hold=hold if hold is not None else np.zeros(0, bool))
    write(f"frames_{a.recording}", res)


def episodes(a):
    """The frames of a UR5 stream that the test rejected, in runs (gaps under 0.3 s), each with the action that was running
    and the HEX-E's force against the rests before and after that action's press (ft.bin columns 3-5, N)."""
    dfold, lab = SESSIONS[a.recording]
    z = np.load(DATA / f"frames_{a.recording}.npz")
    t = float(z["t0"]) + z["dt"].astype(float)
    F = z["F"].astype(float)
    rej = np.flatnonzero((F[:, 3] > 0) & (F[:, 4] == 0))
    ev = [json.loads(line) for line in open(UR5 / lab / "events.jsonl")]
    starts = {e["id"]: e for e in ev if e["event"] == "start"}
    ends = {e["id"]: e for e in ev if e["event"] == "end"}
    ft = np.fromfile(UR5 / lab / "ft.bin", "<f8")
    ft = ft[: len(ft) // 9 * 9].reshape(-1, 9)

    def mean_f(w):
        m = (ft[:, 0] >= w[0]) & (ft[:, 0] <= w[1])
        return ft[m, 3:6].mean(0)

    runs = np.split(rej, np.flatnonzero(np.diff(t[rej]) > 0.3) + 1) if len(rej) else []
    out = []
    for g in runs:
        t0, t1 = float(t[g[0]]), float(t[g[-1]])
        act = [k for k, s in starts.items() if s["t"] <= t0 <= ends.get(k, {"t": 1e12})["t"]
               and (ends.get(k, {}).get("rec") or {}).get("t_rest1")]
        rec = ends[act[-1]]["rec"] if act else None
        row = dict(t0=hms(t0), s=round(t1 - t0, 2), frames=int(len(g)), change_uT=round(float(F[g, 0].max()), 1),
                   lam_max=round(float(np.nanmax(F[g, 1])), 1), f_shown_min=round(float(F[g, 2].min()), 2),
                   f_shown_max=round(float(F[g, 2].max()), 2))
        if rec is not None:
            fa, fb = mean_f(rec["t_rest1"]), mean_f(rec["t_rest0"])
            dfa = [float(np.linalg.norm(mean_f((t[k] - 0.05, t[k])) - fa)) for k in g]
            dfb = [float(np.linalg.norm(mean_f((t[k] - 0.05, t[k])) - fb)) for k in g]
            row.update(action=act[-1], kind=starts[act[-1]].get("kind"), to_rest_after_s=round(rec["t_rest1"][0] - t1, 1),
                       hexe_from_rest_after_N=[round(min(dfa), 2), round(max(dfa), 2)],
                       hexe_from_rest_before_N=[round(min(dfb), 2), round(max(dfb), 2)])
        out.append(row)
        print(row)
    write(f"episodes_{a.recording}", dict(recording=a.recording, runs=out, frames=int(len(rej))))


def approach(a):
    R = json.loads((DATA / "touches.json").read_text())
    six = [np.array(p["d_uT"]) for p in R["plateaus"] if p["group"] == "six"]
    est, ff = estimator(W.Calib.load(CAL))
    est.set_noise_scale(1 / math.sqrt(VIEW_N))
    rng = np.random.default_rng(0)
    rows = []
    for s in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
        shown, f, rej, L, mag = [], [], [], [], []
        for d in six:
            for _ in range(20):
                dn = s * d + rng.normal(0, est.sig_now)
                e = est.estimate(dn)
                sh = bool(e.get("contact") and e["e"] > THRESHOLD and e["f_n"] >= MIN_F)
                lam = ff.lam(dn, e) if e.get("contact") else -np.inf
                shown.append(sh)
                f.append(e["f_n"] if sh else 0.0)
                rej.append(sh and lam > ff.on)
                L.append(lam)
                mag.append(np.linalg.norm(s * d))
        shown, f, rej = np.array(shown), np.array(f), np.array(rej)
        rows.append(dict(scale=s, change_uT=round(float(np.mean(mag)), 1), shown_off=round(float(shown.mean()), 3),
                         f_mean_shown_off=round(float(f[shown].mean()), 2) if shown.any() else 0.0,
                         shown_on=round(float((shown & ~rej).mean()), 3),
                         f_mean_shown_on=round(float(f[shown & ~rej].mean()), 2) if (shown & ~rej).any() else 0.0,
                         lam_median=round(float(np.median(L)), 1)))
        print(rows[-1])
    write("approach", dict(rows=rows, n_per_scale=len(six) * 20))


def write(name, res):
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"{name}.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    print(f"-> {p.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("touches")
    s.add_argument("--save", action="store_true", help="copy the window's raw records from the viewer's stream log")
    sub.add_parser("holds")
    s = sub.add_parser("frames")
    s.add_argument("recording", choices=["touches", "round1", "round2", *SESSIONS])
    s.add_argument("--fps", type=float, default=30.0)
    s = sub.add_parser("episodes")
    s.add_argument("recording", choices=list(SESSIONS))
    sub.add_parser("approach")
    a = ap.parse_args()
    dict(touches=touches, holds=holds, frames=frames, episodes=episodes, approach=approach)[a.cmd](a)


if __name__ == "__main__":
    main()

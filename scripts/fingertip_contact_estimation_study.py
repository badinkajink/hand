"""Numbers of the model-based contact estimation study (2026-10-20), for its result page.

Reads the board captures in logs/20261020-reskin5x/ and logs/20261020-contact_estimation/ (copied into the study folder's
data/), the FE basis (scripts/fingertip_wrench_fe.py) and the measurement model (scripts/fingertip_wrench_model.py), and
writes docs/experiments/20261020-fingertip_contact_estimation/results.json and data/20261020-uncertainty_maps.npz:

  calibration  the rest fit of the pad as found: every hypothesis' RMS, the best fit, measured and modelled components
  synthetic    errors of the estimator on 300 simulated presses per configuration (magnets as found or N52 all one way;
               one sample or 50 ms of samples), against the model's 1-sigma
  no_contact   the estimator over 30 s of the board without a press (detection statistic, background estimate)
  maps         the model's 1-sigma of pressing force and position at 1 N over the face
  replay       the viewer's estimates of two synthetic presses replayed through its server
  surface      the whole-surface estimator (fingertip_wrench_model.SurfaceEstimator on the FE surface basis): errors on
               600 simulated presses per configuration by region of the surface, its 1-sigma at every pad point
               (data/20261020-surface_uncertainty.npz) and its behaviour without a press
  session      the owner's recording of 2026-10-20 19:39 (five presses by hand, the pad loose in its seat): the rests'
               offsets, the rigid pad motion that fits each, and what the estimator reads
  thermal      (subcommand thermal, merged into results.json) the chips' gain against their die temperature, fitted on
               the viewer's stream log of the night of 2026-10-20 (pad untouched from 20:18 to 09:15) and tested on the
               board's warm-up after it was plugged in again on 2026-10-21; written to data/20261021-thermal_gain.json,
               which the viewer reads; the force the viewer shows at rest with and without the correction; and the
               presses by hand replayed with and without fingertip_wrench_model.RestZero

    ~/miniconda3/bin/python scripts/fingertip_contact_estimation_study.py [thermal]
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_wrench_model as W  # noqa: E402

D = ROOT / "docs/experiments/20261020-fingertip_contact_estimation"
DATA = D / "data"
BASIS = DATA / "20261020-b_v2_patch_basis.npz"
SURF = DATA / "20261020-b_v2_surface_basis.npz"
SRC = {"rest": ROOT / "logs/20261020-reskin5x/20261020-pad_as_found.csv",
       "quiet": ROOT / "logs/20261020-contact_estimation/20261020-press_session_1.csv",
       "synthetic": ROOT / "logs/20261020-contact_estimation/20261020-synthetic_presses.csv",
       "session": ROOT / "logs/20261020-contact_estimation/20261020-193936-viewer_session.csv"}
WIN_S, RATE_HZ, THRESHOLD = 0.05, 140.0, 25.0
STREAM = ROOT / "logs/20261020-contact_estimation/stream"
NIGHT = STREAM / "20261020-200647-stream.bin"        # viewer started 20:06:47; presses to 20:17:50; unplugged 09:15:24
WARM = STREAM / "20261021-095907-stream.bin"         # board plugged in again 09:55; viewer started 09:59:07
REC_DT = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])
FLOOR_UT, MIN_FORCE_N = 2.0, 0.25                    # the viewer's defaults from 2026-10-21


def copy_inputs():
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, p in SRC.items():
        dst = DATA / p.name
        if p.exists() and (not dst.exists() or dst.stat().st_mtime < p.stat().st_mtime):
            shutil.copy2(p, dst)
        out[k] = dst
    return out


def rest_mean(path):
    return np.loadtxt(path).reshape(-1, 5, 4)[:, :, 1:].mean(0)


def synthetic(est, n=300, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        while True:
            yt, zt = rng.uniform(est.ys[0], est.ys[-1]), rng.uniform(est.zs[0], est.zs[-1])
            St, ok = est._interp([yt], [zt], est.S)
            if ok[0]:
                break
        f = np.array([rng.uniform(0.3, 3.0), rng.normal(0, 0.3), rng.normal(0, 0.3)])
        d = St[0] @ f + est.U @ rng.normal(0, 5, 3) + est.b0 * rng.normal(0, 1e-3) + rng.normal(0, 1, 15) * est.sig_now
        t0 = time.time()
        r = est.estimate(d)
        dt = time.time() - t0
        if r["contact"]:
            sd = np.sqrt(np.abs(np.diag(np.array(r["cov"]))))
            rows.append([r["y"] - yt, r["z"] - zt, *(np.array(r["f"]) - f), sd[3], sd[4], sd[0], sd[1], sd[2], f[0], dt,
                         r["e"]])
    E = np.array(rows)
    big = E[:, 10] > 1.0
    keys = ("y_mm", "z_mm", "f_press_N", "f_y_N", "f_z_N")
    return dict(n=n, found=len(E), frame_ms=float(1e3 * E[:, 11].mean()),
                detected_at_threshold=int((E[:, 12] > THRESHOLD).sum()),
                rms={k: float(np.sqrt(np.mean(E[:, i] ** 2))) for i, k in enumerate(keys)},
                rms_over_1N={k: float(np.sqrt(np.mean(E[big, i] ** 2))) for i, k in enumerate(keys)},
                sigma_median={k: float(np.median(E[:, 5 + i])) for i, k in enumerate(keys)})


def no_contact(est, path):
    d = np.loadtxt(path)
    t, r = d[:, 0] - d[0, 0], d[:, 1:].reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
    est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
    base = r[t < 1.0].mean(0)
    out = []
    for t1 in np.arange(1.0, t[-1], 1 / 30):
        sel = (t > t1 - WIN_S) & (t <= t1)
        if sel.sum() < 3:
            continue
        e = est.estimate(r[sel].mean(0) - base)
        out.append([t1, e.get("e", 0.0), e["chi2_none"], *(e.get("g_uT") or [0, 0, 0])])
    O = np.array(out)
    gz = np.convolve(O[:, 5], np.ones(15) / 15, mode="same")
    k = 15 + int(np.argmax(np.abs(gz[30:-15] - gz[15:-30])))      # largest change of the 0.5 s mean over 0.5 s
    step = float(O[k, 0])
    pre, post = O[:, 0] < step - 1, O[:, 0] > step + 1
    return dict(step_s=step, frames=len(O), e_median=float(np.median(O[:, 1])), e_p99=float(np.percentile(O[:, 1], 99)),
                e_max=float(O[:, 1].max()), false_contacts=int((O[:, 1] > THRESHOLD).sum()), threshold=THRESHOLD,
                chi2_median=float(np.median(O[:, 2])), dof=11, g_before=O[pre, 3:6].mean(0).tolist(),
                g_after=O[post, 3:6].mean(0).tolist(), series=np.round(O[:, :3], 2).tolist(),
                g_series=np.round(O[:, 3:6], 2).tolist())


def maps(est, f0=(1.0, 0.0, 0.0)):
    """1-sigma of (f_press, f_y, f_z, y, z) at every face point of the grid for a 1 N press."""
    f0 = np.array(f0)
    h = est.ys[1] - est.ys[0]
    sig = np.full(est.valid.shape + (5,), np.nan)
    for i in range(1, len(est.ys) - 1):
        for j in range(1, len(est.zs) - 1):
            if not (est.valid[i, j] and est.valid[i - 1, j] and est.valid[i + 1, j] and est.valid[i, j - 1] and est.valid[i, j + 1]):
                continue
            Sy = (est.Sp[i + 1, j] - est.Sp[i - 1, j]) / (2 * h)
            Sz = (est.Sp[i, j + 1] - est.Sp[i, j - 1]) / (2 * h)
            J = np.c_[est.Sp[i, j], Sy @ f0, Sz @ f0]
            try:
                sig[i, j] = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))
            except np.linalg.LinAlgError:
                pass
    return sig


def region_of(n):
    return np.where(n[:, 0] > 0.7, "face", np.where(n[:, 0] >= 0.3, "edge", "sides"))


def surface_points(basis, n, rng):
    """n points uniform by area on the pad points' triangulation, with their interpolated unit normals."""
    P, N, tri = basis["pts"], basis["normals"], basis["tris"]
    A = 0.5 * np.linalg.norm(np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]]), axis=1)
    k = rng.choice(len(tri), n, p=A / A.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    w = np.c_[1 - u - v, u, v]
    X = np.einsum("nk,nkc->nc", w, P[tri[k]])
    Nn = np.einsum("nk,nkc->nc", w, N[tri[k]])
    return X, Nn / np.linalg.norm(Nn, axis=1, keepdims=True)


def surface_synthetic(est, basis, n=600, seed=1, far_mm=1.5):
    """Errors of the whole-surface estimator on n presses at points uniform by area, press 0.3-3 N along -n, shear
    normal with 0.3 N sd along the tangents, background 5 uT, strength change 1e-3, the noise of the window."""
    rng = np.random.default_rng(seed)
    X, Nn = surface_points(basis, n, rng)
    T1, T2 = W.tangent_frames(Nn)
    rows = []
    t_frame = []
    for i in range(n):
        fn = rng.uniform(0.3, 3.0)
        ft = rng.normal(0, 0.3, 2)
        f = -fn * Nn[i] + ft[0] * T1[i] + ft[1] * T2[i]
        d = (est._S(X[i][None])[0] @ f + est.U @ rng.normal(0, 5, 3) + est.b0 * rng.normal(0, 1e-3)
             + rng.normal(0, 1, 15) * est.sig_now)
        t0 = time.time()
        r = est.estimate(d)
        t_frame.append(time.time() - t0)
        if not r["contact"]:
            rows.append([np.nan] * 6 + [Nn[i, 0]])
            continue
        cov = np.array(r["cov"])
        rows.append([np.linalg.norm(np.array(r["p"]) - X[i]), r["f_n"] - fn,
                     np.linalg.norm(np.array(r["f"]) - f), float(np.sqrt(max(Nn[i] @ cov[:3, :3] @ Nn[i], 0))),
                     float(np.sqrt(max(np.sqrt(abs(cov[3, 3] * cov[4, 4])), 0))), float(r["e"] > THRESHOLD), Nn[i, 0]])
    R = np.array(rows)
    reg = region_of(np.c_[R[:, 6], np.zeros((len(R), 2))])
    out = dict(n=n, frame_ms=float(1e3 * np.mean(t_frame)), far_mm=far_mm, regions={})
    for name in ("face", "edge", "sides", "all"):
        s = np.ones(len(R), bool) if name == "all" else reg == name
        found = s & ~np.isnan(R[:, 0])
        far = found & (R[:, 0] > far_mm)
        good = found & ~far
        out["regions"][name] = dict(
            n=int(s.sum()), found=int(found.sum()), far_pct=float(100 * far.sum() / max(found.sum(), 1)),
            shown_pct=float(100 * np.nansum(R[s, 5]) / max(s.sum(), 1)),
            pos_median_mm=float(np.median(R[good, 0])), pos_p90_mm=float(np.percentile(R[good, 0], 90)),
            press_abs_median_N=float(np.median(np.abs(R[good, 1]))), press_rms_N=float(np.sqrt(np.mean(R[good, 1] ** 2))),
            force_rms_N=float(np.sqrt(np.mean(R[good, 2] ** 2))), sd_press_median_N=float(np.median(R[good, 3])),
            sd_pos_median_mm=float(np.median(R[good, 4])),
            press_z_rms=float(np.sqrt(np.mean((R[good, 1] / R[good, 3]) ** 2))))
    return out


def surface_maps(est, basis, f0=1.0, h=0.2):
    """1-sigma of the press (N) and of the position (mm, geometric mean of the two tangents) for a press of f0 N along
    -n at every pad point."""
    P, Nn = basis["pts"], basis["normals"]
    T1, T2 = W.tangent_frames(Nn)
    sp, sx = np.full(len(P), np.nan), np.full(len(P), np.nan)
    for i in range(len(P)):
        f = -f0 * Nn[i]
        Sd = est._S(np.stack([P[i], P[i] + h * T1[i], P[i] - h * T1[i], P[i] + h * T2[i], P[i] - h * T2[i]]))
        J = np.c_[est.Linv @ Sd[0], est.Linv @ ((Sd[1] - Sd[2]) / (2 * h)) @ f, est.Linv @ ((Sd[3] - Sd[4]) / (2 * h)) @ f]
        try:
            cov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            continue
        sp[i] = np.sqrt(max(Nn[i] @ cov[:3, :3] @ Nn[i], 0))
        sx[i] = np.sqrt(np.sqrt(abs(cov[3, 3] * cov[4, 4])))
    return sp, sx


def surface_no_contact(est, path):
    d = np.loadtxt(path)
    t, r = d[:, 0] - d[0, 0], d[:, 1:].reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
    base = r[t < 1.0].mean(0)
    E = []
    for t1 in np.arange(1.0, t[-1], 1 / 30):
        sel = (t > t1 - WIN_S) & (t <= t1)
        if sel.sum() >= 3:
            e = est.estimate(r[sel].mean(0) - base)
            E.append([t1, e.get("e", 0.0) if e["contact"] else 0.0, e["chi2_none"], *e["g_uT"]])
    E = np.array(E)
    return dict(frames=len(E), e_median=float(np.median(E[:, 1])), e_p99=float(np.percentile(E[:, 1], 99)),
                e_max=float(E[:, 1].max()), false_contacts=int((E[:, 1] > THRESHOLD).sum()),
                chi2_median=float(np.median(E[:, 2])), series=np.round(E[:, :3], 2).tolist(),
                g_series=np.round(E[:, 3:6], 2).tolist())


SYN_PRESSES = (   # surface point (y, z) or a named point, normal press (N), shear along t1, t2 (N)
    ("palmar face", (0.0, 11.0), 2.0, (0.0, 0.0)),
    ("+y side", "side+", 1.5, (0.0, 0.0)),
    ("face edge, -y", "edge-", 2.0, (0.4, 0.0)),
    ("tip end", "tip", 1.0, (0.0, 0.0)),
    ("palmar face, shear along z", (1.5, 6.0), 1.5, (0.6, 0.0)),
)


def synthetic_recording(cal, basis, out, seed=3, rate=140.0):
    """A recording in the format of reskin5x_plot.py --out: the measured rest reading, then the presses of SYN_PRESSES
    (0.4 s up, 0.8 s held, 0.4 s down, 1.2 s apart), each the model's change S(p) f plus the measured noise per sample."""
    rng = np.random.default_rng(seed)
    est = W.SurfaceEstimator(W.Model(cal), basis)
    P, Nn = basis["pts"], basis["normals"]
    def nearest(ok, x):
        return int(np.argmin(np.where(ok, np.abs(P[:, 0] - x), np.inf)))
    pick = {"side+": nearest((np.abs(P[:, 2] - 11) < 1.5) & (Nn[:, 1] > 0.9), 0.5),
            "edge-": nearest((np.abs(P[:, 2] - 13) < 1.5) & (Nn[:, 1] < 0) & (Nn[:, 0] > 0.45) & (Nn[:, 0] < 0.65), 6.0),
            "tip": nearest((np.abs(P[:, 1]) < 1.0) & (Nn[:, 2] > 0.8), 3.0)}
    b0 = np.asarray(cal.b_rest_uT, float).ravel()
    t, rows, presses = 0.0, [], []
    temps = np.array([27.1, 25.7, 25.0, 28.3, 27.7])

    def emit(dv):
        nonlocal t
        rec = (b0 + dv + rng.normal(0, 1, 15) * est.sig).reshape(5, 3)
        rows.append(np.r_[t, np.c_[temps, rec].ravel()])
        t += 1 / rate
    for _ in range(int(3.0 * rate)):
        emit(np.zeros(15))
    for name, where, fn, (s1, s2) in SYN_PRESSES:
        i = pick[where] if isinstance(where, str) else int(np.argmin(np.where(Nn[:, 0] > 0.7, (P[:, 1] - where[0]) ** 2
                                                                         + (P[:, 2] - where[1]) ** 2, np.inf)))
        p, n = P[i], Nn[i]
        t1, t2 = (v[0] for v in W.tangent_frames(n[None]))
        f = -fn * n + s1 * t1 + s2 * t2
        S = est._S(p[None])[0]
        presses.append(dict(name=name, t=round(t + 0.8, 2), p=p.round(3).tolist(), n=n.round(3).tolist(), f=f.round(3).tolist(),
                            f_n=fn, f_t=[s1, s2]))
        for k in range(int(1.6 * rate)):
            a = k / rate
            w = min(a / 0.4, 1.0, max((1.6 - a) / 0.4, 0.0))
            emit(S @ f * w)
        for _ in range(int(1.2 * rate)):
            emit(np.zeros(15))
    np.savetxt(out, np.array(rows), fmt="%.4f", header="host_time " + " ".join(f"{q}{c}" for c in range(5) for q in "txyz"))
    return presses


def replay_check(path, presses, board, basis):
    """The viewer's pipeline on a recording: calibration on its first 2 s, baseline the same, the 50 ms window at each
    press's middle; estimate against truth."""
    d = np.loadtxt(path)
    t, r = d[:, 0] - d[0, 0], d[:, 1:].reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
    rest = r[t < 2.0].mean(0)
    est = W.SurfaceEstimator(W.Model(W.calibrate_frames_fixed(rest.reshape(5, 3), board)), basis)
    est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
    out = []
    for q in presses:
        sel = (t > q["t"] - WIN_S) & (t <= q["t"])
        e = est.estimate(r[sel].mean(0) - rest)
        cov = np.array(e["cov"])
        out.append(dict(name=q["name"], p=e["p"], f_n=e["f_n"], f_t=e["f_t"], e=e["e"],
                        pos_err_mm=float(np.linalg.norm(np.array(e["p"]) - np.array(q["p"]))),
                        sd_pos_mm=float(np.sqrt(np.sqrt(abs(cov[3, 3] * cov[4, 4])))),
                        sd_press_N=float(np.sqrt(max(np.array(e["n"]) @ cov[:3, :3] @ np.array(e["n"]), 0)))))
    return out


def temperature_range(path, block_s=1.0):
    """Range (max - min) over the recording of each chip's temperature averaged over block_s, in degrees C."""
    d = np.loadtxt(path)
    t, T = d[:, 0] - d[0, 0], d[:, 1::4]
    k = (t // block_s).astype(int)
    means = np.array([T[k == i].mean(0) for i in np.unique(k) if (k == i).sum() > 50])
    return [float(v) for v in np.ptp(means, axis=0)]


def session(path, board):
    """The rests of a recording by hand, their offsets from the first, the rigid pad motion that fits each, and the
    whole-surface estimator's reading of the presses and the rests."""
    sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
    import pad_shift_check as PS
    t, r, start = PS.load(path)
    segs = PS.rests(t, r, 0.4, 40.0)
    s0 = (t >= segs[0][0]) & (t < segs[0][1])
    ref = r[s0].mean(0)
    cal = W.calibrate_frames_fixed(ref.reshape(5, 3), board)
    model = W.Model(cal)
    A = PS.rigid_jacobian(model)
    est = W.SurfaceEstimator(model, W.load_basis(SURF))
    est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
    rng = np.random.default_rng(0)
    rand = []
    for _ in range(2000):
        v = rng.normal(size=15)
        x, *_ = np.linalg.lstsq(A, v, rcond=None)
        rand.append(np.linalg.norm(v - A @ x) / np.linalg.norm(v))
    rests_out = []
    for t0, t1 in segs:
        sel = (t >= t0) & (t < t1)
        Pv = r[sel].mean(0) - ref
        x, *_ = np.linalg.lstsq(A, Pv, rcond=None)
        e = est.estimate(Pv)
        rests_out.append(dict(t0=float(t0), t1=float(t1), offset_uT=float(np.linalg.norm(Pv)), components_uT=Pv.round(2).tolist(),
                              rigid_fit_uT=(A @ x).round(2).tolist(), translation_um=(1e3 * x[:3]).round(2).tolist(),
                              tilt_mrad=(1e3 * x[3:6]).round(3).tolist(),
                              residual_pct=float(100 * np.linalg.norm(Pv - A @ x) / max(np.linalg.norm(Pv), 1e-9)),
                              read_as=(dict(f_n=e["f_n"], f=e["f"], p=e["p"], e=e["e"]) if e["contact"] and e["e"] > THRESHOLD
                                       else None)))
    # 30 Hz trace of the reading change and the estimate
    tr, dvs = [], []
    for t1 in np.arange(segs[0][0] + WIN_S, t[-1], 1 / 30):
        sel = (t > t1 - WIN_S) & (t <= t1)
        if sel.sum() < 3:
            continue
        dv = r[sel].mean(0) - ref
        e = est.estimate(dv)
        shown = e["contact"] and e["e"] > THRESHOLD
        tr.append([t1, np.linalg.norm(dv), e["f_n"] if shown else 0.0,
                   np.linalg.norm(e["f_t"]) if shown else 0.0, *(e["p"] if shown else [np.nan] * 3)])
        dvs.append(dv)
    T, dvs = np.array(tr), np.array(dvs)
    # presses: stretches where the reading departs by more than 30 uT from the latest rest's level
    level = np.zeros_like(dvs)
    for q, (t0, t1) in zip(rests_out, segs):
        level[T[:, 0] >= t0] = np.array(q["components_uT"])
    dep = np.linalg.norm(dvs - level, axis=1)
    inside = np.zeros(len(T), bool)
    for t0, t1 in segs:
        inside |= (T[:, 0] >= t0) & (T[:, 0] < t1)
    act = (dep > 30) & ~inside
    peaks, k = [], 0
    while k < len(act):
        if act[k]:
            j = k
            while j < len(act) and act[j]:
                j += 1
            m = k + int(np.argmax(dep[k:j]))
            peaks.append(dict(t0=float(T[k, 0]), t1=float(T[j - 1, 0]), t=float(T[m, 0]), dB_uT=float(dep[m]),
                              f_n=float(T[m, 2]), f_t=float(T[m, 3]), p=T[m, 4:7].tolist()))
            k = j
        else:
            k += 1
    return dict(file=Path(path).name, start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)), span_s=float(t[-1]),
                calibration=dict(k=cal.k, rms_mT=cal.rms_mT), rests=rests_out, peaks=peaks,
                random_residual_pct=[float(100 * np.median(rand)), float(100 * np.percentile(rand, 1))],
                temperature_C=temperature_range(path),
                trace=np.round(T, 3).tolist())


def reduce_stream(path, block_s, t0_s=0.0, t1_s=None):
    """Block means of a viewer stream log: (start as epoch s, block centres (s from the start), T (n, 5) C, B (n, 15) uT,
    the largest range of a field component within each block (uT), records per block). Records with a chip reading
    zeros or values out of range (framing errors at start-up) are dropped."""
    a = np.memmap(path, REC_DT, mode="r")
    th = a["t"]
    start = float(th[0])
    i0 = int(np.searchsorted(th, start + t0_s))
    i1 = len(a) if t1_s is None else int(np.searchsorted(th, start + t1_s))
    nb = int(np.ceil((th[i1 - 1] - start - t0_s) / block_s))
    S, C = np.zeros((nb, 20)), np.zeros(nb)
    MX, MN = np.full((nb, 15), -np.inf), np.full((nb, 15), np.inf)
    for s in range(i0, i1, 2_000_000):
        c = a[s:min(s + 2_000_000, i1)]
        t, r = np.asarray(c["t"]), np.asarray(c["r"], float)
        rr = r.reshape(-1, 5, 4)
        ok = (np.isfinite(r).all(1) & (np.abs(r) < 1e5).all(1) & ~np.all(rr == 0, axis=2).any(axis=1)
              & (rr[:, :, 0] > -40).all(1) & (rr[:, :, 0] < 125).all(1))
        t, r = t[ok], r[ok]
        k = np.minimum(((t - start - t0_s) // block_s).astype(int), nb - 1)
        C += np.bincount(k, minlength=nb)
        for j in range(20):
            S[:, j] += np.bincount(k, weights=r[:, j], minlength=nb)
        f = r.reshape(-1, 5, 4)[:, :, 1:].reshape(-1, 15)
        np.maximum.at(MX, k, f)
        np.minimum.at(MN, k, f)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = (S / C[:, None]).reshape(-1, 5, 4)
    return start, t0_s + (np.arange(nb) + 0.5) * block_s, M[:, :, 0], M[:, :, 1:].reshape(-1, 15), np.max(MX - MN, axis=1), C


def _series(npz, path, block_s, t0_s=0.0, t1_s=None):
    """reduce_stream of a stream log, kept in data/ (the logs are not in git) and read from there when the log is gone."""
    if path.exists() and (not npz.exists() or npz.stat().st_mtime < path.stat().st_mtime):
        start, t, T, B, spread, n = reduce_stream(path, block_s, t0_s, t1_s)
        np.savez_compressed(npz, start=start, t=t, T=T, B=B, spread=spread, n=n, source=str(path.relative_to(ROOT)),
                            block_s=block_s)
    z = np.load(npz)
    return float(z["start"]), z["t"], z["T"], z["B"], z["spread"], z["n"]


def fit_kappa(T, B):
    """Each chip's gain coefficient kappa (1/C): the value that makes its reading divided by 1 + kappa (T - T_ref) the
    most constant over the series (least squares over its three components)."""
    from scipy.optimize import minimize_scalar
    out = []
    for i in range(5):
        b, x = B[:, 3 * i:3 * i + 3], T[:, i] - W.T_REF_C

        def cost(k):
            c = b / (1 + k * x)[:, None]
            return float(((c - c.mean(0)) ** 2).sum())
        out.append(minimize_scalar(cost, bounds=(-0.03, 0.03), method="bounded").x)
    return np.array(out)


def _shown(est, d, min_force=MIN_FORCE_N):
    """The force the viewer shows for reading change d (0 when it shows no contact)."""
    e = est.estimate(d)
    return e["f_n"] if e["contact"] and e["e"] > THRESHOLD and e["f_n"] >= min_force else 0.0


def thermal(board):
    """See the module docstring (thermal)."""
    sn = _series(DATA / "20261021-overnight_drift_60s.npz", NIGHT, 60.0)
    sw = _series(DATA / "20261021-warmup_1s.npz", WARM, 1.0, 0.0, 600.0)
    start_n, tn, Tn, Bn, spn, nn = sn
    start_w, tw, Tw, Bw, spw, nw = sw
    quiet = tn > 1800                                  # from 20:36, 18 min after the last press
    kap = fit_kappa(Tn[quiet], Bn[quiet])
    one = float(np.median(kap))
    # warm-up: the first 150 s, before the first touch at 150 s
    ok = np.isfinite(Bw).all(1) & (tw > 2) & (tw < 148)
    tw_, Tw_, Bw_ = tw[ok], Tw[ok], Bw[ok]
    ref = slice(0, 8)

    def drift(B):
        return B - B[ref].mean(0)
    variants = {"uncorrected": Bw_,
                "per chip, own temperature": W.thermal_correct(Bw_.reshape(-1, 5, 3), Tw_, kap).reshape(-1, 15),
                "one coefficient, own temperature": W.thermal_correct(Bw_.reshape(-1, 5, 3), Tw_, np.full(5, one)).reshape(-1, 15),
                "per chip, mean temperature": W.thermal_correct(Bw_.reshape(-1, 5, 3), np.repeat(Tw_.mean(1, keepdims=True), 5, 1),
                                                                kap).reshape(-1, 15)}
    warm = {k: dict(rms_uT=float(np.sqrt((drift(v)[-10:].mean(0) ** 2).mean())),
                    max_uT=float(np.abs(drift(v)[-10:].mean(0)).max()),
                    norm_uT=float(np.linalg.norm(drift(v)[-10:].mean(0)))) for k, v in variants.items()}
    # the night: the fit's residual and the correction applied to all of it
    Bq, Tq = Bn[quiet], Tn[quiet]
    Cq = W.thermal_correct(Bq.reshape(-1, 5, 3), Tq, kap).reshape(-1, 15)
    night = dict(T_range_C=np.ptp(Tq, 0).round(2).tolist(), raw_ptp_uT=np.ptp(Bq, 0).round(1).tolist(),
                 corrected_ptp_uT=np.ptp(Cq, 0).round(1).tolist(), raw_rms_uT=float(np.sqrt(Bq.var(0).mean())),
                 corrected_rms_uT=float(np.sqrt(Cq.var(0).mean())),
                 field_mT=(np.linalg.norm(Bq.mean(0).reshape(5, 3), axis=1) / 1e3).round(2).tolist())
    # the force the viewer shows at rest: estimator as the viewer runs it, baseline at the start of the stretch
    cal = W.calibrate_frames_fixed(W.thermal_correct(Bq[:5].mean(0).reshape(5, 3), Tq[:5].mean(0), kap), board)
    sb = W.load_basis(SURF)
    shown = {}
    for floor in (0.0, FLOOR_UT):
        est = W.SurfaceEstimator(W.Model(cal), sb, floor_uT=floor)
        est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
        for name, B, T in (("night", Bq, Tq), ("warm-up", Bw_, Tw_)):
            for corr in (False, True):
                X = W.thermal_correct(B.reshape(-1, 5, 3), T, kap).reshape(-1, 15) if corr else B
                base = X[:5].mean(0) if name == "night" else X[ref].mean(0)
                f = np.array([_shown(est, x - base, 0.0 if floor == 0 else MIN_FORCE_N) for x in X])
                shown[f"{name}, {'corrected' if corr else 'uncorrected'}, floor {floor:g} uT"] = dict(
                    share_pct=float(100 * (f > 0).mean()), max_N=float(f.max()), p95_N=float(np.percentile(f, 95)))
    # what the estimator's nuisances (a uniform field, a uniform gain change) leave of the drift
    mdl = W.Model(cal)
    Qn, _ = np.linalg.qr(np.c_[np.concatenate(list(mdl.R), 0), W.thermal_correct(Bq.mean(0).reshape(5, 3), Tq.mean(0),
                                                                                    kap).ravel()])

    def left(v):
        return float(np.linalg.norm(v - Qn @ (Qn.T @ v)))
    after = {}
    for corr in (False, True):
        Xn = W.thermal_correct(Bq.reshape(-1, 5, 3), Tq, kap).reshape(-1, 15) if corr else Bq
        Xw = variants["per chip, own temperature" if corr else "uncorrected"]
        after["corrected" if corr else "uncorrected"] = dict(
            night_max_uT=max(left(x - Xn[:5].mean(0)) for x in Xn),
            warmup_end_uT=left(Xw[-10:].mean(0) - Xw[ref].mean(0)))
    # the gain as N per C: a 1 C change of every chip's temperature, uncorrected, read by the estimator at rest
    est = W.SurfaceEstimator(W.Model(cal), sb)
    est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
    b25 = W.thermal_correct(Bq.mean(0).reshape(5, 3), Tq.mean(0), kap)
    e1 = est.estimate((b25 * (1 + kap[:, None]) - b25).ravel())
    out = dict(kappa_per_C=kap.round(5).tolist(), kappa_median=one, T_ref_C=W.T_REF_C,
               night=dict(file="data/20261021-overnight_drift_60s.npz", start=time.strftime("%Y-%m-%d %H:%M", time.localtime(start_n)),
                          hours=float(tn[-1] / 3600), fit_from_h=0.5, **night),
               warmup=dict(file="data/20261021-warmup_1s.npz", start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_w)),
                           span_s=float(tw_[-1] - tw_[0]), T_rise_C=(Tw_[-10:].mean(0) - Tw_[ref].mean(0)).round(2).tolist(),
                           variants=warm),
               shown=shown, one_C_reads_as=dict(f_n=e1.get("f_n"), e=e1.get("e"), contact=e1["contact"]),
               after_nuisances=after,
               floor_uT=FLOOR_UT, min_force_N=MIN_FORCE_N)
    W.THERMAL.write_text(json.dumps(dict(kappa_per_C=out["kappa_per_C"], T_ref_C=W.T_REF_C,
                                         fitted_on=f"{NIGHT.name}, from 20:36 (pad untouched), 60 s means",
                                         chips="streaming order 0x0C, 0x11, 0x12, 0x13, 0x10",
                                         script="scripts/fingertip_contact_estimation_study.py thermal",
                                         when=time.strftime("%Y-%m-%d %H:%M")), indent=1) + "\n")
    return out


def rezero_replay(board, kap, recordings, out_npz):
    """The viewer's pipeline at 30 Hz over recordings of presses by hand (temperature correction, the 50 ms window, the
    estimator with the noise floor and the shown-force threshold with its hysteresis), with and without RestZero.
    recordings: name -> (t, records, baseline window or None for the first rest). The traces (time, the 15-component
    reading, the baseline's change, the shown force) go to out_npz; the summary per recording and setting: the share of
    frames with a contact shown, the re-zeros and their offsets, and the re-zeros the reading came back from within 10 s
    (within 30 % of the re-zero's offset of the earlier baseline): a held press or a passing state taken into the
    baseline, whose end then shows as a contact until the next re-zero."""
    sb = W.load_basis(SURF)
    out, arrays = {}, {}
    for name, (t, r, win) in recordings.items():
        T, B = r[:, :, 0], r[:, :, 1:]
        if win is None:
            rest = PS_rests(t, B.reshape(-1, 15))
            r0 = rest[0] if rest else (t[0], t[0] + 1.0)
        else:
            r0 = win
        s0 = (t >= r0[0]) & (t < r0[1])
        Tcal = T[s0].mean(0)
        cal = W.calibrate_frames_fixed(W.thermal_correct(B[s0].mean(0), Tcal, kap), board)
        est = W.SurfaceEstimator(W.Model(cal), sb, floor_uT=FLOOR_UT)
        est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
        res = dict(baseline_window_s=[float(r0[0]), float(r0[1])])
        for auto in (False, True):
            base = W.thermal_correct(B[s0].mean(0), Tcal, kap).ravel()
            rz = W.RestZero()
            rows, X, zeros, shown_prev = [], [], [], False
            for t1 in np.arange(r0[1], t[-1], 1 / 30):
                i1 = np.searchsorted(t, t1, side="right")
                iw = np.searchsorted(t, t1 - WIN_S, side="right")
                if i1 - iw < 3:
                    continue
                Ts = T[np.searchsorted(t, t1 - 3.0):i1].mean(0)            # the viewer's 3 s of chip temperatures
                x = W.thermal_correct(B[iw:i1].mean(0), Ts, kap).ravel()
                if auto:
                    i_1 = np.searchsorted(t, t1 - rz.span_s)
                    nb = rz.update(t[i_1:i1], W.thermal_correct(B[i_1:i1], Ts, kap).reshape(-1, 15), base)
                    if nb is not None:
                        zeros.append((float(t1), base.copy(), nb.copy()))
                        base = nb
                e = est.estimate(x - base)
                lim = MIN_FORCE_N * (0.6 if shown_prev else 1.0)
                shown_prev = bool(e["contact"] and e["e"] > THRESHOLD and e["f_n"] >= lim)
                rows.append([t1, np.linalg.norm(x - base), e["f_n"] if shown_prev else 0.0])
                X.append(x)
            R, X = np.array(rows), np.array(X)
            back = []
            for tz, b_old, b_new in zeros:
                off = np.linalg.norm(b_new - b_old)
                nxt = (R[:, 0] > tz) & (R[:, 0] < tz + 10.0)
                if off > 10.0 and nxt.any() and np.min(np.linalg.norm(X[nxt] - b_old, axis=1)) < 0.3 * off:
                    back.append([tz, float(off)])
            key = "auto-zero" if auto else "baseline kept"
            res[key] = dict(shown_share_pct=float(100 * (R[:, 2] > 0).mean()), frames=int(len(R)),
                            rezeros=len(zeros), rezero_offsets_uT=[round(float(np.linalg.norm(b - a)), 1) for _, a, b in zeros],
                            rezeros_over_26uT=int(sum(np.linalg.norm(b - a) > 26 for _, a, b in zeros)),
                            came_back=back)
            tag = name.replace(":", "").replace(" ", "_") + ("_auto" if auto else "_kept")
            arrays[tag + "_trace"] = R.astype(np.float32)
            arrays[tag + "_zeros"] = np.array([[tz, np.linalg.norm(b - a)] for tz, a, b in zeros], np.float32).reshape(-1, 2)
        out[name] = res
    np.savez_compressed(out_npz, **arrays)
    return out


def detection_limit(board, b_rest, floors=(0.0, FLOOR_UT)):
    """The smallest press along the normal the viewer shows (explained chi-square above THRESHOLD and, with the floor,
    at least MIN_FORCE_N), median by region of the surface, for each noise floor: the force at which a press's explained
    energy reaches the threshold, from its energy at 1 N (the energy grows with the square of the force). b_rest: the
    rest field (5, 3) uT the model is calibrated on."""
    sb = W.load_basis(SURF)
    cal = W.calibrate_frames_fixed(b_rest, board)
    out = {}
    for floor in floors:
        est = W.SurfaceEstimator(W.Model(cal), sb, floor_uT=floor)
        est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
        y = -np.einsum("nik,nk->ni", est.Sw, est.Cn)          # whitened change for 1 N along -n
        lim = np.sqrt(THRESHOLD / np.einsum("ni,ni->n", y, y))
        if floor > 0:
            lim = np.maximum(lim, MIN_FORCE_N)
        reg = region_of(est.Cn)
        out[f"floor {floor:g} uT"] = {k: float(np.median(lim[reg == k])) for k in ("face", "edge", "sides")}
    return out


def floor_sweep(board, kap, b25, floors=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0)):
    """For each noise floor: the share of the night (60 s means) and of the warm-up (1 s means) shown as a contact with
    the temperature correction, of the warm-up without it, and the smallest press shown, median by region."""
    zn = np.load(DATA / "20261021-overnight_drift_60s.npz")
    q = zn["t"] > 1800
    Bn = W.thermal_correct(zn["B"][q].reshape(-1, 5, 3), zn["T"][q], kap).reshape(-1, 15)
    zw = np.load(DATA / "20261021-warmup_1s.npz")
    ok = np.isfinite(zw["B"]).all(1) & (zw["t"] > 2) & (zw["t"] < 148)
    Bw = W.thermal_correct(zw["B"][ok].reshape(-1, 5, 3), zw["T"][ok], kap).reshape(-1, 15)
    Bu = zw["B"][ok]
    sb = W.load_basis(SURF)
    cal = W.calibrate_frames_fixed(b25, board)
    out = []
    for floor in floors:
        est = W.SurfaceEstimator(W.Model(cal), sb, floor_uT=floor)
        est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
        row = dict(floor_uT=floor)
        for key, X in (("night", Bn), ("warmup", Bw), ("warmup_uncorrected", Bu)):
            base = X[:5].mean(0) if key == "night" else X[:8].mean(0)
            f = np.array([_shown(est, x - base) for x in X])
            row[key + "_shown_pct"], row[key + "_max_N"] = float(100 * (f > 0).mean()), float(f.max())
        y = -np.einsum("nik,nk->ni", est.Sw, est.Cn)
        lim = np.maximum(np.sqrt(THRESHOLD / np.einsum("ni,ni->n", y, y)), MIN_FORCE_N)
        reg = region_of(est.Cn)
        row["smallest_N"] = {k: float(np.median(lim[reg == k])) for k in ("face", "edge", "sides")}
        out.append(row)
    return out


def seat_offsets(t, r, board, kap, min_rest=3.0):
    """The rests of a recording (pad_shift_check.rests over the temperature-corrected reading) against its first rest,
    and the rigid motion of the magnet set and uniform field that fit each offset."""
    sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
    import pad_shift_check as PS
    B = W.thermal_correct(r[:, :, 1:], PS.trailing_mean(t, r[:, :, 0]), kap).reshape(-1, 15)
    segs = PS.rests(t, B, min_rest, 40.0)
    s0 = (t >= segs[0][0]) & (t < segs[0][1])
    ref = B[s0].mean(0)
    A = PS.rigid_jacobian(W.Model(W.calibrate_frames_fixed(ref.reshape(5, 3), board)))
    out = []
    for a, b in segs[1:]:
        sel = (t >= a) & (t < b)
        P = B[sel].mean(0) - ref
        x, *_ = np.linalg.lstsq(A, P, rcond=None)
        out.append(dict(t0=float(a), t1=float(b), offset_uT=float(np.linalg.norm(P)),
                        shift_um=float(1e3 * np.linalg.norm(x[:3])), tilt_mrad=float(1e3 * np.linalg.norm(x[3:6])),
                        residual_pct=float(100 * np.linalg.norm(P - A @ x) / max(np.linalg.norm(P), 1e-9))))
    return out


def rigid_vs_press(board, b_rest):
    """The share of a simulated 1 N press along the normal that a rigid motion of the magnet set and a uniform field
    (pad_shift_check.rigid_jacobian) leave unexplained, median over the candidate points and by region, against that of
    random 15-component patterns."""
    sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
    import pad_shift_check as PS
    model = W.Model(W.calibrate_frames_fixed(b_rest, board))
    est = W.SurfaceEstimator(model, W.load_basis(SURF))
    Q, _ = np.linalg.qr(PS.rigid_jacobian(model))
    y = -np.einsum("nik,nk->ni", est.S, est.Cn)
    res = np.linalg.norm(y - (y @ Q) @ Q.T, axis=1) / np.linalg.norm(y, axis=1)
    reg = region_of(est.Cn)
    v = np.random.default_rng(0).normal(size=(4000, 15))
    rr = np.linalg.norm(v - (v @ Q) @ Q.T, axis=1) / np.linalg.norm(v, axis=1)
    return dict(press_median_pct=float(100 * np.median(res)),
                by_region={k: float(100 * np.median(res[reg == k])) for k in ("face", "edge", "sides")},
                random_median_pct=float(100 * np.median(rr)))


def PS_rests(t, B):
    sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
    import pad_shift_check as PS
    return PS.rests(t, B, 0.4, 40.0)


def _load_records(path, t0_s=None, t1_s=None):
    """(time from the start (s), records (n, 5, 4)) of a CSV recording or a stream log, start-up garbage dropped."""
    if str(path).endswith(".bin"):
        a = np.memmap(path, REC_DT, mode="r")
        th = np.asarray(a["t"])
        i0 = 0 if t0_s is None else int(np.searchsorted(th, th[0] + t0_s))
        i1 = len(a) if t1_s is None else int(np.searchsorted(th, th[0] + t1_s))
        t, r = th[i0:i1] - th[0], np.asarray(a["r"][i0:i1], float).reshape(-1, 5, 4)
    else:
        d = np.loadtxt(path)
        t, r = d[:, 0] - d[0, 0], d[:, 1:].reshape(-1, 5, 4)
    ok = (np.isfinite(r).all((1, 2)) & (np.abs(r) < 1e5).all((1, 2)) & ~np.all(r == 0, axis=2).any(axis=1)
          & (r[:, :, 0] > -40).all(1) & (r[:, :, 0] < 125).all(1))
    return t[ok], r[ok]


def main_thermal():
    board = W.Calib.load(DATA / "20261020-calib_pad_as_found.json")
    th = thermal(board)
    kap = np.array(th["kappa_per_C"])
    recs = {"19:39 session": (*_load_records(DATA / "20261020-193936-viewer_session.csv"), None)}
    for name, path, a, b, fname in (("20:06 presses", NIGHT, 0.0, 1100.0, "20261020-200647_presses_records.npz"),
                                    ("09:59 touches", WARM, 2.0, 600.0, "20261021-095907_touches_records.npz")):
        npz = DATA / fname
        if path.exists() and not npz.exists():
            t, r = _load_records(path, a, b)
            np.savez_compressed(npz, t=t, r=r.astype(np.float32), source=str(path.relative_to(ROOT)))
        z = np.load(npz)
        # 09:59: the viewer's own first 2 s; 20:06: the rest at 8-10 s (in the viewer's first 2 s the pad sat 1.3 mT
        # away in its seat until a touch at 4-6 s)
        w = (float(z["t"][0]), float(z["t"][0]) + 2.0) if name.startswith("09") else (8.0, 10.0)
        recs[name] = (z["t"], z["r"].astype(float), w)
    th["replay"] = rezero_replay(board, kap, recs, DATA / "20261021-rezero_replay.npz")
    z = np.load(DATA / "20261021-overnight_drift_60s.npz")
    q = z["t"] > 1800
    b25 = W.thermal_correct(z["B"][q].reshape(-1, 5, 3), z["T"][q], kap).mean(0)
    th["detection_limit_N"] = detection_limit(board, b25)
    th["floor_sweep"] = floor_sweep(board, kap, b25)
    th["rigid_vs_press"] = rigid_vs_press(board, b25)
    th["seat_0959"] = seat_offsets(*recs["09:59 touches"][:2], board, kap)
    res = json.loads((D / "results.json").read_text())
    res["thermal"] = th
    (D / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", (D / "results.json").relative_to(ROOT), "and", W.THERMAL.relative_to(ROOT))
    print("  kappa (%/C):", np.round(100 * kap, 3).tolist())
    print("  night:", {k: v for k, v in th["night"].items() if k in ("T_range_C", "raw_rms_uT", "corrected_rms_uT")})
    for k, v in th["warmup"]["variants"].items():
        print(f"  warm-up {k:34s} norm {v['norm_uT']:6.1f} uT  max {v['max_uT']:5.1f} uT")
    for k, v in th["shown"].items():
        print(f"  shown at rest, {k:42s} {v['share_pct']:5.1f} % of the time, max {v['max_N']:.2f} N, p95 {v['p95_N']:.2f} N")
    print("  1 C uncorrected reads as", th["one_C_reads_as"])
    print("  left after the nuisances (uT):", th["after_nuisances"])
    print("  smallest press shown (N, median by region):", th["detection_limit_N"])
    print("  rigid fit of a 1 N press:", th["rigid_vs_press"])
    print("  rests of 09:59:", [(round(q['t0']), round(q['offset_uT'], 1), round(q['shift_um'], 1), round(q['tilt_mrad'], 2),
                                 round(q['residual_pct'])) for q in th["seat_0959"]])
    for name, rr in th["replay"].items():
        for k in ("baseline kept", "auto-zero"):
            v = rr[k]
            print(f"  replay {name:14s} {k:14s} shown {v['shown_share_pct']:5.1f} %  re-zeros {v['rezeros']} "
                  f"(over 26 uT: {v['rezeros_over_26uT']}), came back {v['came_back']}; offsets {v['rezero_offsets_uT'][:10]}")


def main():
    if sys.argv[1:] == ["thermal"]:
        return main_thermal()
    src = copy_inputs()
    basis = W.load_basis(BASIS)
    res = dict(when=time.strftime("%Y-%m-%d %H:%M"), script="scripts/fingertip_contact_estimation_study.py")
    b = rest_mean(src["rest"])
    cal, fits = W.calibrate(b, top=12, verbose=False)
    seen, per_perm = set(), []
    for f in fits:
        if f["perm"] not in seen:
            seen.add(f["perm"])
            per_perm.append(dict(perm=list(f["perm"]), poles=list(f["poles"]), s=f["s"], rms_mT=f["rms_mT"],
                                 theta0=f["theta0"], k=f["k"]))
    model = W.Model(cal)
    turned = [f for f in fits if tuple(f["perm"]) == tuple([0] + [{1: 3, 2: 4, 3: 1, 4: 2}[u] for u in cal.perm[1:]])]
    res["calibration"] = dict(best={k: v for k, v in cal.__dict__.items() if k != "b_rest_uT"}, per_assignment=per_perm[:8],
                              n_hypotheses=len(fits), turned_rms_mT=turned[0]["rms_mT"] if turned else None,
                              measured_uT=b.tolist(), model_uT=model.rest_uT().tolist(),
                              field_mT=[float(np.linalg.norm(v)) / 1e3 for v in b])
    cal.save(DATA / "20261020-calib_pad_as_found.json")
    n52 = W.Calib(**{**cal.__dict__, "k": 1.0, "poles": (1, 1, 1, 1, 1), "b_rest_uT": []})
    res["synthetic"] = []
    res["maps"] = {}
    sig_maps = {}
    for name, c in (("as found", cal), ("N52, poles one way", n52)):
        est = W.Estimator(W.Model(c), basis)
        for lbl, scale in (("one sample", 1.0), ("50 ms", 1 / np.sqrt(WIN_S * RATE_HZ))):
            est.set_noise_scale(scale)
            res["synthetic"].append(dict(magnets=name, window=lbl, **synthetic(est)))
        est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
        sm = maps(est)
        sig_maps[name] = sm
        ok = ~np.isnan(sm[..., 0])
        res["maps"][name] = dict(f_press_N=[float(np.nanpercentile(sm[..., 0], q)) for q in (10, 50, 90)],
                                 y_mm=[float(np.nanpercentile(sm[..., 3], q)) for q in (10, 50, 90)],
                                 z_mm=[float(np.nanpercentile(sm[..., 4], q)) for q in (10, 50, 90)], points=int(ok.sum()))
        if name == "as found":
            res["no_contact"] = no_contact(est, src["quiet"])
    np.savez_compressed(DATA / "20261020-uncertainty_maps.npz", ys=est.ys, zs=est.zs, valid=est.valid,
                        as_found=sig_maps["as found"], n52=sig_maps["N52, poles one way"],
                        magnets=W.magnets(model.row)[0], chips=W.chip_points())
    row = W.fe_row(cal.case)
    res["fe"] = dict(case=cal.case, n_dof=int(basis["n_dof"]), patch_centres=int(len(basis["pts"])),
                     load_cases=int(3 * len(basis["pts"])), display_nodes=int(len(basis["disp_nodes"])),
                     sigma_mm=float(basis["sigma_mm"]), E_star_MPa=float(row["E_best_MPa"]),
                     approach_1N_um=float(row["per_force"]["1"]["delta_um"]), magnets=row["magnets"],
                     depth_mm=row["depth_mm"], face_u_press_um_per_N=[float(np.percentile(basis["face_u"][:, 0], q)) for q in (10, 50, 90)])
    # whole surface
    sb = W.load_basis(SURF)
    nrm = sb["normals"]
    lu = sb["load_u"]
    res["fe_surface"] = dict(pads=int(len(sb["pts"])), area_mm2=float(sb["pad_area"].sum()), load_cases=int(3 * len(sb["pts"])),
                             surface_nodes=int(sb["n_surface_nodes"]), n_dof=int(sb["n_dof"]),
                             regions={k: int((region_of(nrm) == k).sum()) for k in ("face", "edge", "sides")},
                             normal_compliance_um_per_N=[float(np.percentile(np.einsum("pkc,pk,pc->p", lu, nrm, nrm), q))
                                                         for q in (5, 50, 95)])
    # agreement with the face basis on the face: the nearest face pad to each face-basis point, press motions
    P = sb["pts"]
    fc = np.where(nrm[:, 0] > 0.8)[0]
    rel, dist = [], []
    for i, (y, z) in enumerate(basis["pts"]):
        j = fc[np.argmin((P[fc, 1] - y) ** 2 + (P[fc, 2] - z) ** 2)]
        a_, b_ = basis["motions"][i, 0], -sb["motions"][j, 0]
        rel.append(np.linalg.norm(a_ - b_) / np.linalg.norm(a_))
        dist.append(np.hypot(P[j, 1] - y, P[j, 2] - z))
    res["fe_surface"]["face_agreement"] = dict(points=len(rel), rel_diff_median=float(np.median(rel)),
                                               rel_diff_p90=float(np.percentile(rel, 90)), match_mm_median=float(np.median(dist)))
    res["surface"] = {}
    maps_s = {}
    for name, c in (("as found", cal), ("N52, poles one way", n52)):
        est = W.SurfaceEstimator(W.Model(c), sb)
        est.set_noise_scale(1 / np.sqrt(WIN_S * RATE_HZ))
        res["surface"][name] = dict(candidates=int(len(est.C)), **surface_synthetic(est, sb))
        maps_s[name] = surface_maps(est, sb)
        reg = region_of(nrm)
        res["surface"][name]["maps"] = {k: dict(press_N=float(np.nanmedian(maps_s[name][0][reg == k])),
                                                pos_mm=float(np.nanmedian(maps_s[name][1][reg == k])))
                                        for k in ("face", "edge", "sides")}
        if name == "as found":
            res["surface"][name]["no_contact"] = surface_no_contact(est, src["quiet"])
    np.savez_compressed(DATA / "20261020-surface_uncertainty.npz", pts=sb["pts"], normals=nrm, tris=sb["tris"],
                        as_found_press=maps_s["as found"][0], as_found_pos=maps_s["as found"][1],
                        n52_press=maps_s["N52, poles one way"][0], n52_pos=maps_s["N52, poles one way"][1],
                        magnets=W.magnets(model.row)[0], chips=W.chip_points())
    res["session"] = session(src["session"], W.Calib.load(DATA / "20261020-calib_pad_as_found.json"))
    pr = synthetic_recording(cal, sb, DATA / "20261020-synthetic_surface_presses.csv")
    res["synthetic_recording"] = dict(file="data/20261020-synthetic_surface_presses.csv", presses=pr,
                                      estimates=replay_check(DATA / "20261020-synthetic_surface_presses.csv", pr,
                                                             W.Calib.load(DATA / "20261020-calib_pad_as_found.json"), sb))
    (D / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", (D / "results.json").relative_to(ROOT))
    for s in res["synthetic"]:
        print(f"  {s['magnets']:20s} {s['window']:10s} y {s['rms']['y_mm']:.2f} z {s['rms']['z_mm']:.2f} mm, "
              f"press {s['rms']['f_press_N']:.3f} N, shear {s['rms']['f_y_N']:.3f}/{s['rms']['f_z_N']:.3f} N, "
              f"detected {s['detected_at_threshold']}/{s['found']}")
    print("  no contact:", {k: v for k, v in res["no_contact"].items() if k not in ("series", "g_series")})
    print("  maps:", res["maps"])
    for name, s in res["surface"].items():
        for k, v in s["regions"].items():
            print(f"  surface {name:20s} {k:5s} n {v['n']:3d} far {v['far_pct']:4.1f} % pos {v['pos_median_mm']:.2f} mm "
                  f"press |err| {v['press_abs_median_N']:.3f} N (sd {v['sd_press_median_N']:.3f}, z {v['press_z_rms']:.2f})")
    print("  session rests:", [(round(q['t0'], 1), round(q['offset_uT'], 1), round(q['residual_pct'])) for q in res["session"]["rests"]])
    print("  session presses:", [(round(q['t'], 1), round(q['dB_uT']), round(q['f_n'], 2), np.round(q['p'], 1).tolist()) for q in res["session"]["peaks"]])


if __name__ == "__main__":
    main()

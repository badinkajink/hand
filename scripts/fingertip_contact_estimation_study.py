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

    ~/miniconda3/bin/python scripts/fingertip_contact_estimation_study.py
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


def main():
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

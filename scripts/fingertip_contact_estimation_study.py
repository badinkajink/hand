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
SRC = {"rest": ROOT / "logs/20261020-reskin5x/20261020-pad_as_found.csv",
       "quiet": ROOT / "logs/20261020-contact_estimation/20261020-press_session_1.csv",
       "synthetic": ROOT / "logs/20261020-contact_estimation/20261020-synthetic_presses.csv"}
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
    (D / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", (D / "results.json").relative_to(ROOT))
    for s in res["synthetic"]:
        print(f"  {s['magnets']:20s} {s['window']:10s} y {s['rms']['y_mm']:.2f} z {s['rms']['z_mm']:.2f} mm, "
              f"press {s['rms']['f_press_N']:.3f} N, shear {s['rms']['f_y_N']:.3f}/{s['rms']['f_z_N']:.3f} N, "
              f"detected {s['detected_at_threshold']}/{s['found']}")
    print("  no contact:", {k: v for k, v in res["no_contact"].items() if k not in ("series", "g_series")})
    print("  maps:", res["maps"])


if __name__ == "__main__":
    main()

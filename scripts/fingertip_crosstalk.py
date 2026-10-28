#!/usr/bin/env python3
"""Crosstalk between two magnetic fingertips: the field of one tip's magnets at the other tip's chips, the contact that the
viewer's estimator shows for it, and what is left once that field is predicted from the two tips' relative pose and
subtracted. Owner, 2026-10-27: "do you think we can use joint kinematics well to remove crosstalk? i think the broader
concern is that the crosstalk is very noisy and very high magnitude".

  budget   The receiving tip is v2 ID 1 (five N52 magnets; its fit at rest in UR5 session 20261026-192158-v2id1). The
           neighbour is the same five-magnet tip, or pad 05R4's four N52 magnets in a 5.0 mm alternating square, both at
           v2 ID 1's strength. Two poses: pads facing across a gap W (a pinch), and side by side with a gap g between the
           tips' flanks. Each row gives the field at the chips and the contact the viewer would show for it (explained
           chi-square over 25 and a press of at least 0.25 N), with and without the steel test (--foreign-gate). It also
           gives the field left after the steel test's far-field fit, and the contacts shown after subtracting the
           field predicted at a pose with an error. The pose error is drawn with a standard deviation per axis, in mm and
           deg, as from joint angles. The same pose errors are added to v2 ID 1's light and firm face presses, and the
           change in their estimated location and force is reported. Last, the Cramer-Rao bound on the press at the face's
           centre, with the neighbour's pose fitted from one board or from both, given the joint angles as a prior.
  sharpie  The owner's pinch of a sharpie between v1 ID 2 and ID 3 on 2026-10-27 12:18 (the multi-board viewer's
           streams). For each board: the change, its uniform part and the contact the viewer showed. Then a fit of the
           tips' relative pose that explains both boards together with a contact on each.

    .venv/bin/python scripts/fingertip_crosstalk.py budget
    .venv/bin/python scripts/fingertip_crosstalk.py sharpie
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_wrench_model as W  # noqa: E402
import tpu_tip_design as TD  # noqa: E402

EXP = ROOT / "docs/experiments/20261027-fingertip_crosstalk"
UR5 = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
V2 = UR5 / "20261026-192158-v2id1"
STREAM = ROOT / "logs/20261020-contact_estimation/stream"
WIN = 1 / np.sqrt(0.05 * 140)          # the viewer's noise scale: 50 ms of samples at 140 Hz (wrench_viewer.WIN_S)
SHOW_E, SHOW_N, LAMBDA_ON = 25.0, 0.25, 25.0   # the viewer's defaults: --threshold, --min-force, --foreign-lambda
FLOOR = 2.0                            # the viewer's default --floor (uT a component)
FOUR_POLES = (1, -1, -1, 1)            # pad 05R4: the square's diagonals alike (checkerboard)
FACING = Rotation.from_euler("z", 180, degrees=True).as_matrix()   # x and y reversed: the pads face each other


def five_row():
    return W.fe_row("b_v2")


def four_row():
    """Pad 05R4 (scripts/dip_b_four_compact_tip.py, four_y2p5_z5): a 5.0 mm square centred on U1, at the five-magnet
    tip's depth, ordered as there (proximal first, -y before +y)."""
    r = dict(W.fe_row("b_v2"))
    r["magnets"] = [[3.0, 2.0, sy * 2.5, TD.Z_C + sz * 2.5] for sz in (-1, 1) for sy in (-1, 1)]
    return r


def source(row, poles, k):
    """Quadrature points (n, 3) mm and moment vectors (n, 3) A m^2 of a tip's magnets in its own tip frame."""
    c, m = W.magnets(row)
    d, t = row["magnets"][0][:2]
    q, w = W.cylinder_quadrature(d, t)
    pts = (c[:, None, :] + q[None]).reshape(-1, 3)
    mom = ((k * np.asarray(poles, float) * m)[:, None] * w[None]).reshape(-1)
    return pts, mom[:, None] * np.array([[1.0, 0.0, 0.0]])


def reading(model, src, R, t):
    """Reading (uT, (15,), the receiving model's chip frames) of the magnets src placed by rotation R and translation t
    (mm) in the receiving tip's frame."""
    pts, mom = src
    B = W.dipole_field(model.P, pts @ R.T + t, mom @ R.T)
    return np.einsum("cij,cj->ci", model.R, B).ravel() * 1e6


def inverse(R, t):
    return R.T, -R.T @ t


def perturb(R, t, dt_mm, dr_deg):
    return Rotation.from_rotvec(np.radians(dr_deg)).as_matrix() @ R, t + dt_mm


def shown(est, ff, d):
    """(press shown without the steel test, with it, the estimate): 0 when the viewer shows no contact."""
    e = est.estimate(d)
    ok = bool(e.get("contact")) and e["e"] > SHOW_E and e["f_n"] >= SHOW_N
    f = e["f_n"] if ok else 0.0
    fg = f if (ok and ff.lam(d, e) <= LAMBDA_ON) else 0.0
    return f, fg, e


def far_residual(ff, d):
    x = np.linalg.lstsq(ff.A, d, rcond=None)[0]
    return float(np.linalg.norm(d - ff.A @ x))


def poses():
    out = [(f"facing, W {w:g} mm", FACING, np.array([2 * TD.X_FACE + w, 0.0, 0.0])) for w in (0, 5, 10, 15, 20, 30)]
    out += [(f"side by side, g {g:g} mm", np.eye(3), np.array([0.0, 2 * TD.Y_HALF + g, 0.0])) for g in (0, 2, 5, 10, 20)]
    return out


def tangential(v, n):
    return v - (v @ n) * n


def jac(fun, x0, h):
    return np.stack([(fun(x0 + h * e) - fun(x0 - h * e)) / (2 * h) for e in np.eye(len(x0))], 1)


def crb_fn(blocks, sd_pose):
    """Standard deviation (N) of the press along -x at the face's centre of the first board. blocks: per board (S (15, 3),
    J (15, 6) the reading per mm and per deg of the neighbour's pose, U (15, 3), b0 (15,), sig (15,)). Priors as the
    estimator's: force 3 N, uniform field 30 uT, strength 3e-3; pose sd_pose mm and deg per axis (None: no prior)."""
    nb = len(blocks)
    npar = 3 * nb + 6 + 4 * nb
    H = np.zeros((15 * nb, npar))
    P = np.zeros(npar)
    for i, (S, J, U, b0, sig) in enumerate(blocks):
        r = slice(15 * i, 15 * i + 15)
        H[r, 3 * i:3 * i + 3] = S / sig[:, None]
        H[r, 3 * nb:3 * nb + 6] = J / sig[:, None]
        o = 3 * nb + 6 + 4 * i
        H[r, o:o + 3] = U / sig[:, None]
        H[r, o + 3] = b0 / sig
        P[3 * i:3 * i + 3] = 1 / 3.0 ** 2
        P[o:o + 3] = 1 / 30.0 ** 2
        P[o + 3] = 1 / 3e-3 ** 2
    if sd_pose is None:
        P[3 * nb:3 * nb + 6] = 1e-12
    elif sd_pose == 0:
        P[3 * nb:3 * nb + 6] = 1e12
    else:
        P[3 * nb:3 * nb + 6] = 1 / sd_pose ** 2
    C = np.linalg.inv(H.T @ H + np.diag(P))
    return float(np.sqrt(C[0, 0]))


def budget(args):
    t0 = time.time()
    cal = W.Calib.load(V2 / "calib_rest.json")
    model = W.Model(cal)
    est = W.SurfaceEstimator(model, W.load_basis(BASIS), floor_uT=FLOOR)
    est.set_noise_scale(WIN)
    ff = W.ForeignField(est)
    srcs = {"five N52": source(five_row(), cal.poles, cal.k), "four N52": source(four_row(), FOUR_POLES, cal.k)}
    H = np.load(V2 / "holds.npz")
    sel = (H["kind"] == "press") & (H["region"] == "face")
    presses = {}
    rng = np.random.default_rng(1)
    for name, lo, hi in (("light", 0.3, 0.7), ("firm", 1.5, 2.5)):
        idx = np.where(sel & (H["f_n"] >= lo) & (H["f_n"] <= hi))[0]
        idx = rng.choice(idx, size=min(args.presses, len(idx)), replace=False)
        rows = []
        for i in idx:
            f0, _, e0 = shown(est, ff, H["d_board"][i])
            rows.append((i, e0))
        presses[name] = rows
    # the press's sensitivity at the face's centre, for the bound
    kc = int(np.argmin(np.linalg.norm(est.C - np.array([TD.X_FACE, 0.0, TD.Z_C]), axis=1)))
    sig = est.sig_now
    res = dict(receiver="v2 ID 1 (UR5 session 20261026-192158-v2id1)", k=cal.k, noise_scale=WIN, floor_uT=FLOOR, rows=[],
               presses={k: len(v) for k, v in presses.items()}, face_centre=est.C[kc].tolist())
    errs = [(0.1, 0.1), (0.3, 0.3), (1.0, 1.0)]
    for sname, src in srcs.items():
        for pname, R, t in poses():
            X = reading(model, src, R, t)
            f, fg, e = shown(est, ff, X)
            row = dict(source=sname, pose=pname, max_chip_uT=float(np.abs(X.reshape(5, 3)).max()),
                       norm_uT=float(np.linalg.norm(X)), after_far_uT=far_residual(ff, X), shown_N=f, shown_gate_N=fg,
                       shown_p=e.get("p"), kin={}, crb={})
            for st, sr in errs:
                fs, fgs, resid, dl, df = [], [], [], {"light": [], "firm": []}, {"light": [], "firm": []}
                for _ in range(args.samples):
                    R2, t2 = perturb(R, t, rng.normal(0, st, 3), rng.normal(0, sr, 3))
                    r = X - reading(model, src, R2, t2)
                    a, b, _ = shown(est, ff, r)
                    fs.append(a)
                    fgs.append(b)
                    resid.append(np.linalg.norm(r))
                for pk, rows in presses.items():
                    for i, e0 in rows:
                        R2, t2 = perturb(R, t, rng.normal(0, st, 3), rng.normal(0, sr, 3))
                        r = X - reading(model, src, R2, t2)
                        e1 = est.estimate(H["d_board"][i] + r)
                        if not (e0.get("contact") and e1.get("contact")):
                            continue
                        n = H["n"][i]
                        dl[pk].append(float(np.linalg.norm(tangential(np.array(e1["p"]) - H["p"][i], n)))
                                      - float(np.linalg.norm(tangential(np.array(e0["p"]) - H["p"][i], n))))
                        df[pk].append(abs(e1["f_n"] - H["f_n"][i]) - abs(e0["f_n"] - H["f_n"][i]))
                fs, fgs = np.array(fs), np.array(fgs)
                row["kin"][f"{st:g} mm, {sr:g} deg"] = dict(
                    residual_uT_median=float(np.median(resid)), shown_frac=float(np.mean(fs > 0)),
                    shown_N_p90=float(np.percentile(fs, 90)), shown_gate_frac=float(np.mean(fgs > 0)),
                    **{f"{pk}_loc_added_mm_median": float(np.median(v)) if v else None for pk, v in dl.items()},
                    **{f"{pk}_force_added_N_median": float(np.median(v)) if v else None for pk, v in df.items()})
            # Cramer-Rao bound on the press at the face's centre, the neighbour's pose fitted with a prior
            p0 = np.r_[np.zeros(3), np.zeros(3)]
            fA = lambda p: reading(model, src, *perturb(R, t, p[:3], p[3:]))
            JA = jac(fA, p0, 1e-3)
            blkA = (est.S[kc], JA, est.U, est.b0, sig)
            out = {}
            for sd in (0, 0.1, 0.3, 1.0, 3.0, None):
                key = "known" if sd == 0 else ("free" if sd is None else f"{sd:g}")
                out[key] = dict(one_board=crb_fn([blkA], sd))
                if sname == "five N52":
                    def fB(p):
                        Ri, ti = inverse(*perturb(R, t, p[:3], p[3:]))
                        return reading(model, srcs["five N52"], Ri, ti)
                    JB = jac(fB, p0, 1e-3)
                    out[key]["two_boards"] = crb_fn([blkA, (est.S[kc], JB, est.U, est.b0, sig)], sd)
            row["crb"] = out
            row["pose_gradient_uT_per_mm"] = float(np.linalg.norm(JA[:, :3], axis=0).max())
            row["pose_gradient_uT_per_deg"] = float(np.linalg.norm(JA[:, 3:], axis=0).max())
            res["rows"].append(row)
            k1 = row["kin"]["0.3 mm, 0.3 deg"]
            print(f"{sname:9s} {pname:22s} max chip {row['max_chip_uT']:7.0f} uT  |X| {row['norm_uT']:7.0f}  after far "
                  f"{row['after_far_uT']:6.0f}  shown {f:5.2f} N (gate {fg:5.2f})  grad {row['pose_gradient_uT_per_mm']:6.0f} uT/mm "
                  f"{row['pose_gradient_uT_per_deg']:5.0f} uT/deg | 0.3: resid {k1['residual_uT_median']:5.1f} shown "
                  f"{k1['shown_frac']:.2f} p90 {k1['shown_N_p90']:.2f} N | crb known {out['known']['one_board']:.3f} "
                  f"0.3 {out['0.3']['one_board']:.3f} free {out['free']['one_board']:.3f}"
                  + (f" two {out['free']['two_boards']:.3f}" if 'two_boards' in out['free'] else ""), flush=True)
    # v2 ID 1's own light presses for scale
    fl = H["f_n"][sel]
    dn = np.linalg.norm(H["d_board"][sel], axis=1)
    res["press_signal_uT"] = {f"{lo:g}-{hi:g} N": float(np.median(dn[(fl >= lo) & (fl < hi)]))
                              for lo, hi in ((0.2, 0.4), (0.4, 0.7), (0.7, 1.5), (1.5, 3.0))}
    res["seconds"] = time.time() - t0
    out = EXP / "data" / "budget.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("press signal |d| (uT, face presses):", res["press_signal_uT"])
    print("wrote", out)


# --- the sharpie pinch of 2026-10-27 -------------------------------------------------------------------------------

SHARPIE = dict(
    files={"ID 2": "20261027-044648-id_2-stream.bin", "ID 3": "20261027-044649-id_3-stream.bin"},
    calib={"ID 2": UR5 / "20261025-152920-cb2t/calib_rest.json",
           "ID 3": ROOT / "logs/20261020-contact_estimation/calib/20261025-112147-calib.json"},
    rest=("12:17:00", "12:18:00"),
    windows=[("12:18:21", "12:18:24"), ("12:18:33", "12:18:35"), ("12:18:39", "12:18:43"), ("12:18:44", "12:18:48"),
             ("12:18:49", "12:18:53"), ("12:18:54", "12:18:58"), ("12:19:01", "12:19:04"), ("12:19:04", "12:19:08")])


def stamp(hms):
    h, m, s = map(float, hms.split(":"))
    return (datetime.datetime(2026, 10, 27) + datetime.timedelta(hours=h, minutes=m, seconds=s)).timestamp()


def window_change(t, r, kappa, w, ref):
    s = (t >= stamp(w[0])) & (t < stamp(w[1]))
    T = r[s, :, 0].mean(0)
    return W.thermal_correct(r[s][:, :, 1:], T, kappa).reshape(-1, 15).mean(0) - ref, int(s.sum())


def chi2_after(est, d):
    e = est.estimate(d)
    r = d - est.S[e["index"]] @ np.asarray(e["f"]) if e.get("contact") else d
    rw = est.Linv @ r
    return float(rw @ rw), e


def sharpie(args):
    kappa = W.load_thermal()
    basis = W.load_basis(BASIS)
    tips = {}
    for name, fn in SHARPIE["files"].items():
        a = np.memmap(STREAM / fn, np.dtype([("t", "<f8"), ("r", "<f4", (20,))]), mode="r")
        i0, i1 = np.searchsorted(a["t"], stamp(SHARPIE["rest"][0]) - 5), np.searchsorted(a["t"], stamp("12:19:30"))
        t, r = np.array(a["t"][i0:i1]), np.array(a["r"][i0:i1]).reshape(-1, 5, 4).astype(float)
        s = (t >= stamp(SHARPIE["rest"][0])) & (t < stamp(SHARPIE["rest"][1]))
        ref = W.thermal_correct(r[s][:, :, 1:], r[s, :, 0].mean(0), kappa).reshape(-1, 15).mean(0)
        cal = W.Calib.load(SHARPIE["calib"][name])
        model = W.Model(cal)
        # the tips were picked up and turned in the Earth's field, so the uniform field is free (300 uT)
        est = W.SurfaceEstimator(model, basis, floor_uT=2.0, sigma_g_uT=300.0)
        tips[name] = dict(t=t, r=r, ref=ref, cal=cal, model=model, est=est,
                          src=source(W.fe_row(cal.case), cal.poles, cal.k))
    A, B = tips["ID 2"], tips["ID 3"]

    def objective(p, dA, dB):
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        t = p[3:]
        xA = reading(A["model"], B["src"], R, t)
        xB = reading(B["model"], A["src"], *inverse(R, t))
        return chi2_after(A["est"], dA - xA)[0] + chi2_after(B["est"], dB - xB)[0]

    starts = []
    for flip in ("z", "y"):
        R0 = Rotation.from_euler(flip, 180, degrees=True)
        for psi in (0, 90, 180, 270):
            R = (Rotation.from_euler("x", psi, degrees=True) * R0).as_matrix()
            apex = np.array([TD.X_FACE, 0.0, TD.Z_C])
            for w in (6, 12, 18, 26):
                t = np.array([TD.X_FACE + w, 0.0, TD.Z_C]) - R @ apex
                starts.append(np.r_[Rotation.from_matrix(R).as_rotvec(), t])
    for sy in (-1, 1):
        for R in (np.eye(3), Rotation.from_euler("x", 180, degrees=True).as_matrix()):
            starts.append(np.r_[Rotation.from_matrix(R).as_rotvec(), 0.0, sy * (2 * TD.Y_HALF + 3), 0.0])
    rng = np.random.default_rng(0)
    for _ in range(args.random_starts):
        starts.append(np.r_[Rotation.random(random_state=rng).as_rotvec(), rng.uniform(-40, 40, 3)])

    out = dict(windows=[], rest=SHARPIE["rest"], calib={k: str(v.relative_to(ROOT)) for k, v in SHARPIE["calib"].items()})
    seeds = None                       # the first window searches every start; later ones start from its best five
    for w in SHARPIE["windows"]:
        dA, nA = window_change(A["t"], A["r"], kappa, w, A["ref"])
        dB, nB = window_change(B["t"], B["r"], kappa, w, B["ref"])
        A["est"].set_noise_scale(1 / np.sqrt(nA))
        B["est"].set_noise_scale(1 / np.sqrt(nB))
        c0A, eA = chi2_after(A["est"], dA)
        c0B, eB = chi2_after(B["est"], dB)
        found = []
        for p0 in (starts if seeds is None else seeds):
            found.append(minimize(objective, p0, args=(dA, dB), method="Powell",
                                  options=dict(maxiter=3000, xtol=1e-2, ftol=1e-3)))
        found.sort(key=lambda q: q.fun)
        if seeds is None:
            seeds = [q.x for q in found[:5]]
            out["search"] = [dict(chi2=float(q.fun), rotvec=q.x[:3].tolist(), t_mm=q.x[3:].tolist()) for q in found[:10]]
        best = found[0]
        p = best.x
        R, t = Rotation.from_rotvec(p[:3]).as_matrix(), p[3:]
        xA, xB = reading(A["model"], B["src"], R, t), reading(B["model"], A["src"], *inverse(R, t))
        c1A, e1A = chi2_after(A["est"], dA - xA)
        c1B, e1B = chi2_after(B["est"], dB - xB)
        apexB = R @ np.array([TD.X_FACE, 0.0, TD.Z_C]) + t
        nB_in_A = R @ np.array([1.0, 0.0, 0.0])
        row = dict(window=w, n=[nA, nB], d_uT=[float(np.linalg.norm(dA)), float(np.linalg.norm(dB))],
                   chi2_without=[c0A, c0B], chi2_with=[c1A, c1B],
                   crosstalk_uT=[float(np.linalg.norm(xA)), float(np.linalg.norm(xB))],
                   crosstalk_after_uniform_uT=[float(np.linalg.norm(xA - A["est"].U @ np.linalg.lstsq(A["est"].U, xA, rcond=None)[0])),
                                               float(np.linalg.norm(xB - B["est"].U @ np.linalg.lstsq(B["est"].U, xB, rcond=None)[0]))],
                   rotvec=p[:3].tolist(), t_mm=t.tolist(), apex_B_in_A=apexB.tolist(),
                   apex_distance_mm=float(np.linalg.norm(apexB - np.array([TD.X_FACE, 0.0, TD.Z_C]))),
                   normal_angle_deg=float(np.degrees(np.arccos(np.clip(-nB_in_A[0], -1, 1)))),
                   without={k: dict(f_n=e.get("f_n"), p=e.get("p"), g=e.get("g_uT")) for k, e in (("ID 2", eA), ("ID 3", eB))},
                   with_crosstalk={k: dict(f_n=e.get("f_n"), p=e.get("p"), g=e.get("g_uT")) for k, e in (("ID 2", e1A), ("ID 3", e1B))})
        out["windows"].append(row)
        fmt = lambda e: f"{e.get('f_n', 0):5.2f} N at ({', '.join(f'{v:5.1f}' for v in e['p'])})" if e.get("contact") else "none"
        print(f"{w[0]}-{w[1]}: |d| {row['d_uT'][0]:.0f}/{row['d_uT'][1]:.0f} uT; chi2 {c0A:.0f}+{c0B:.0f} -> {c1A:.0f}+{c1B:.0f}; "
              f"crosstalk {row['crosstalk_uT'][0]:.0f}/{row['crosstalk_uT'][1]:.0f} uT "
              f"(non-uniform {row['crosstalk_after_uniform_uT'][0]:.0f}/{row['crosstalk_after_uniform_uT'][1]:.0f}); "
              f"B's apex at {np.round(apexB, 1).tolist()} ({row['apex_distance_mm']:.1f} mm, normals {row['normal_angle_deg']:.0f} deg "
              f"from facing)\n    ID 2 {fmt(eA)} -> {fmt(e1A)};  ID 3 {fmt(eB)} -> {fmt(e1B)}", flush=True)
    path = EXP / "data" / "sharpie.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("wrote", path)


def pull(args):
    """The magnetic force between two tips at the holder's test poses (the HEX-E reads it as a contact force): the
    gradient of the robot tip's magnets' energy in the held tip's field, by central differences of 1 um, at twist 0 and
    +-20 deg (facing, the turn about the common normal) and side by side at s 0. Both tips at v2 ID 1's strength k."""
    cal = W.Calib.load(V2 / "calib_rest.json")
    tips = {"five": (five_row(), cal.poles), "four": (four_row(), FOUR_POLES)}
    pairs = [("five", "four"), ("four", "four"), ("five", "five")]
    rows = []
    h = 1e-3                                                         # mm

    def energy(src_r, src_n, R, t):
        pts, mom = src_r
        pn, mn = src_n
        B = W.dipole_field(pts, pn @ R.T + t, mn @ R.T)              # T at the robot tip's quadrature points
        return float(np.sum(mom * B))                                # -U, J

    for robot, held in pairs:
        sr, sn = source(*tips[robot], cal.k), source(*tips[held], cal.k)
        cases = [("facing", w, tw) for w in (0.5, 1, 2, 3, 5, 10) for tw in (-20, 0, 20)]
        cases += [("side", g, 0) for g in (0, 5, 10, 13, 15, 20, 25)]
        for test, gap, tw in cases:
            if test == "facing":
                R = Rotation.from_euler("x", tw, degrees=True).as_matrix() @ FACING
                apex = np.array([TD.X_FACE, 0.0, TD.Z_C])            # the held apex gap mm off the robot apex, turned
                t = apex + np.array([gap, 0.0, 0.0]) - R @ apex          # about the common normal through the apexes
            else:
                R, t = np.eye(3), np.array([0.0, 2 * TD.Y_HALF + gap, 0.0])
            F = np.zeros(3)
            for e in range(3):
                d = np.zeros(3)
                d[e] = h
                F[e] = (energy(sr, sn, R, t - d) - energy(sr, sn, R, t + d)) / (2 * h * 1e-3)   # force on the robot tip, N
            rows.append(dict(robot=robot, held=held, test=test, gap_mm=gap, twist_deg=tw, F_N=F.round(5).tolist(),
                             F_abs_N=float(np.linalg.norm(F))))
            print(f"{robot}/{held} {test} gap {gap:g} tw {tw:+d}: |F| {np.linalg.norm(F) * 1e3:.2f} mN "
                  f"(along the normal {F[0] * 1e3:+.2f})", flush=True)
    path = EXP / "data" / "20261027-pull.json"
    path.write_text(json.dumps(dict(model="fingertip_crosstalk pull: v2 ID 1's k, quadrature dipoles", rows=rows),
                               indent=1) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--samples", type=int, default=100)
    b.add_argument("--presses", type=int, default=30)
    s = sub.add_parser("sharpie")
    s.add_argument("--random-starts", type=int, default=16)
    sub.add_parser("pull")
    a = ap.parse_args()
    {"budget": budget, "sharpie": sharpie, "pull": pull}[a.cmd](a)


if __name__ == "__main__":
    main()

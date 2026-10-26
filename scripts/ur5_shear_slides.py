#!/usr/bin/env python3
"""Shear against surface displacement in the UR5 slides of the printed fingertip: the give before slip, the shear at
which the pad slips and the friction coefficient, and the shear left when the turn returns to 0 deg.

A slide (scripts/ur5_tip_calibration_plan.py, stage "shear") presses the pad onto the 5 mm sphere to 0.5-3 N and then
turns the flange about the sphere's centre in eight directions 45 deg apart: to 1.146 and 2.292 deg (0.1 and 0.2 mm at
the sphere's surface; the shear-* actions also to 3.438 deg, 0.3 mm), and back to 0 deg before the next direction. The
turn leaves the sphere's centre where it is in the pad frame, so the indentation stays and the sphere's surface slides
over the pad's by R theta. The robot holds each pose; the normal force is free to change.

Every HEX-E sample of a slide (100 Hz, quantized at 0.1 N and 1 N mm) gives the force on the pad as the ground truth of a
hold does (ur5_tip_calibration_analysis.Session.ground_truth): the reading less the zero interpolated between the rests
before and after the action, less the change of the payload's weight, through the fitted map. The pad's pose includes
the HEX-E's deflection under the wrench (data sheet). Along each direction:

  d   displacement: the motion of the pad's contact point in the robot's base frame along the direction of its motion,
      from the direction's start, mm (R theta less the HEX-E's deflection);
  s   shear: the component of the force on the pad against that motion, N (friction drags the pad back);
  f_n normal force, N.

Per direction (one row): s, d and f_n at the start and at each turn's hold (0.4 s means); the give of each sub-move,
Delta d / Delta s in um/N; the slope of a straight line through the samples of the first sub-move (the give from the
direction's start); whether the shear stops growing (the last sub-move's stiffness below a quarter of the first's); the
ratio s / f_n at the largest turn (the friction coefficient when the pad slips, a lower bound when it does not); and at the
return to 0 deg the shear left along the direction and the whole tangential force.

The first direction of each slide starts from the press alone. Every later direction starts from the shear that the
previous return left.

    .venv/bin/python scripts/ur5_shear_slides.py rows [--sessions cb2t cb1r]     # rows and traces
    .venv/bin/python scripts/ur5_shear_slides.py figure                          # the figure from the rows
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
HOST = ROOT / "src/morphohand/driver/reskin5x/host"
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration as C  # noqa: E402
import ur5_tip_calibration_analysis as A  # noqa: E402
import ur5_tip_calibration_sim as SIM  # noqa: E402

LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
SESSIONS = {"cb2t": LAB / "20261025-152920-cb2t", "cb1r": LAB / "20261025-114653-cb1r"}
GRAVITY_FROM = LAB / "20261022-195420-first2"      # the gravity fit both sessions' ground truth takes (summary.json)
TIPS = {"cb2t": 2, "cb1r": 1}
OUT = ROOT / "docs/experiments/20261026-shear_friction_impratio/data"
MEDIA = ROOT / "docs/experiments/20261026-shear_friction_impratio/media"
R_SPHERE = 5.0                  # mm
FE_GIVE_UM_N = (48.0, 177.0)    # the FE pad's tangential compliance under the 12.5 mm rod (contact overview)
STOP_RATIO = 0.25               # the shear stops growing: last sub-move's stiffness under this share of the first's
BIN_S = 0.1                     # s, bins of the traces (about 10 HEX-E and 12 robot samples)
SLOPE_UM = 50.0                 # um of travel over which the slopes after the start and after the reversal are fitted


def epp(d, fn, k, mu, z0):
    """Shear of a spring k (N/mm) in series with a Coulomb slider of limit mu f_n, driven by the travel d (mm) from the
    stored shear z0 (N): the elastic-perfectly-plastic (play) operator."""
    z = np.empty(len(d))
    zc = z0
    for i in range(len(d)):
        if i:
            zc += k * (d[i] - d[i - 1])
        lim = mu * fn[i]
        zc = min(max(zc, -lim), lim)
        z[i] = zc
    return z


def fit_epp(d, s, fn):
    """Least squares of the measured shear s against s_off + epp(d, fn; k, mu, z0) over a direction's bins: the spring's
    give 1/k, the friction coefficient mu, the stored shear at the start z0 and the HEX-E's offset along the direction
    s_off. Whether the slider slipped on the way out (at the largest turn) and on the way back (at the return)."""
    from scipy.optimize import least_squares
    if len(d) < 20:
        return None
    fn = np.maximum(fn, 0.05)

    def res(p):
        k, mu = math.exp(p[0]), p[1]
        return s - p[3] - epp(d, fn, k, mu, mu * fn[0] * math.tanh(p[2]))

    best = None
    for k0 in (2.0, 5.0, 12.0):
        for mu0 in (0.25, 0.5, 0.9):
            for a0 in (-1.0, 0.0):
                r = least_squares(res, [math.log(k0), mu0, a0, 0.0], bounds=([math.log(0.3), 0.02, -4.0, -1.5],
                                                                              [math.log(200.0), 2.5, 4.0, 1.5]),
                                  diff_step=1e-3, max_nfev=400)
                if best is None or r.cost < best.cost:
                    best = r
    p = best.x
    k, mu = math.exp(p[0]), p[1]
    z0 = mu * fn[0] * math.tanh(p[2])
    z = epp(d, fn, k, mu, z0)
    lim = mu * fn
    i_top = int(np.argmax(d))
    slip_out = bool(np.any(np.abs(z[: i_top + 1] - lim[: i_top + 1]) < 1e-9))
    slip_back = bool(np.any(np.abs(z[i_top:] + lim[i_top:]) < 1e-9))
    return dict(k_N_mm=k, give_um_N=1e3 / k, mu=mu, z0_N=z0, s_off_N=float(p[3]),
                rms_N=float(np.sqrt(np.mean(res(p) ** 2))), slip_out=slip_out, slip_back=slip_back,
                z_top_N=float(z[i_top]), z_end_N=float(z[-1]), n_bins=int(len(d)))


def vplay(X, fn, k, mu, z0):
    """The vector form of epp: a 2D spring k (N/mm) in series with an isotropic Coulomb slider of radius mu f_n, driven by
    the 2D travel X (mm); past the limit the shear is returned radially to the circle."""
    Z = np.empty_like(X)
    z = np.array(z0, float)
    for i in range(len(X)):
        if i:
            z = z + k * (X[i] - X[i - 1])
        lim = mu * fn[i]
        m = math.hypot(z[0], z[1])
        if m > lim:
            z = z * (lim / m)
        Z[i] = z
    return Z


def fit_vec(X, G, fn, offset="const"):
    """Least squares of the measured 2D shear G against b + vplay(X): the spring's give, the friction coefficient from the
    magnitude of the shear at slip, the stored shear at the start, and an in-plane offset b per slide. offset: 'none'
    (b = 0), 'const' (b fixed) or 'prop' (b = beta f_n, the tangential part of a normal force read along a tilted normal)."""
    from scipy.optimize import least_squares
    fn = np.maximum(fn, 0.05)

    def bvec(p):
        if offset == "none":
            return np.zeros(2)
        return np.array(p[2:4]) * (fn[:, None] if offset == "prop" else 1.0)

    def res(p):
        Z = vplay(X, fn, math.exp(p[0]), p[1], p[4:6])
        return (G - bvec(p) - Z).ravel()

    best = None
    for k0 in (2.5, 6.0):
        for mu0 in (0.35, 0.6, 1.0):
            r = least_squares(res, [math.log(k0), mu0, 0.0, 0.0, 0.0, 0.0],
                              bounds=([math.log(0.3), 0.02, -2.0, -2.0, -2.0, -2.0], [math.log(200.0), 3.0, 2.0, 2.0, 2.0, 2.0]),
                              diff_step=1e-3, max_nfev=800)
            if best is None or r.cost < best.cost:
                best = r
    p = best.x
    k, mu = math.exp(p[0]), p[1]
    Z = vplay(X, fn, k, mu, p[4:6])
    at_lim = np.abs(np.hypot(Z[:, 0], Z[:, 1]) - mu * fn) < 1e-6
    b = bvec(p)
    bm = float(np.mean(np.hypot(*(b.T if b.ndim == 2 else b[:, None])))) if offset != "none" else 0.0
    return dict(offset=offset, k_N_mm=k, give_um_N=1e3 / k, mu=mu, b=[float(p[2]), float(p[3])] if offset != "none" else [0.0, 0.0],
                b_over_fn=bm / float(np.mean(fn)), rms_N=float(np.sqrt(np.mean(res(p) ** 2))), slip_share=float(at_lim.mean()),
                n_bins=int(len(X)))


def cmd_vector(a):
    """Per slide of cb2t, the 2D shear against the 2D travel over its eight directions in sequence, fitted with the vector
    spring and slider without an offset, with a constant offset and with an offset proportional to the normal force; and
    the largest |f_t| / f_n of each direction's holds."""
    rows = load_rows("cb2t")
    Z = np.load(OUT / "20261026-ur5_slide_traces_cb2t.npz")
    meta = json.loads(str(Z["meta"]))
    out = OUT / "20261026-ur5_slides_vector_cb2t.jsonl"
    with open(out, "w") as f:
        for aid in sorted({r["action"] for r in rows}):
            R = sorted([r for r in rows if r["action"] == aid], key=lambda r: r["dir_index"])
            idx = [i for i, m in enumerate(meta) if m["action"] == aid]
            ta = np.concatenate([Z[f"ta_{i}"] for i in idx]).astype(float)
            o = np.argsort(ta, kind="stable")
            keep = np.r_[True, np.diff(np.round(ta[o], 2)) > 0]
            sel = o[keep]
            X = np.c_[np.concatenate([Z[f"x1_{i}"] for i in idx]), np.concatenate([Z[f"x2_{i}"] for i in idx])][sel].astype(float)
            G = np.c_[np.concatenate([Z[f"g1_{i}"] for i in idx]), np.concatenate([Z[f"g2_{i}"] for i in idx])][sel].astype(float)
            fn = np.concatenate([Z[f"fn_{i}"] for i in idx])[sel].astype(float)
            fits = {m: fit_vec(X, G, fn, m) for m in ("none", "const", "prop")}
            mx = [max(h["f_t_N"] / h["f_n_N"] for h in (r["holds"] + ([r["back"]] if r["back"] else []))) for r in R]
            row = dict(action=aid, region=R[0]["region"], press_N=R[0]["press_N"], target_P=R[0]["target_P"],
                       fn_mean_N=float(np.mean(fn)), fits=fits, max_ft_over_fn_by_dir=mx,
                       epp_mu_by_dir=[r["epp"]["mu"] for r in R])
            f.write(json.dumps(row) + "\n")
            print(f"{aid:28s} {R[0]['press_N']:g} N: mu none {fits['none']['mu']:.3f} const {fits['const']['mu']:.3f} prop "
                  f"{fits['prop']['mu']:.3f}; |b|/fn {fits['const']['b_over_fn']:.3f}; rms {fits['none']['rms_N']:.3f} "
                  f"{fits['const']['rms_N']:.3f} {fits['prop']['rms_N']:.3f}; max|ft|/fn {np.median(mx):.3f}", flush=True)


def slope(B, m):
    """Least-squares slope of the shear against the travel over the bins m, N/mm (None for too few bins or travel)."""
    if m.sum() < 5 or np.ptp(B["d"][m]) < 0.02:
        return None
    return float(np.polyfit(B["d"][m], B["s"][m], 1)[0])


def load_lean(sd):
    """ur5_tip_calibration_analysis.load without the magnetometer stream (board.bin, 0.9 GB on cb2t)."""
    sd = Path(sd)
    ev = [json.loads(l) for l in open(sd / "events.jsonl") if l.strip()]
    ft = np.fromfile(sd / "ft.bin", "<f8")
    ft = ft[: len(ft) // 9 * 9].reshape(-1, 9)
    rb = np.fromfile(sd / "robot.bin", "<f8")
    rb = rb[: len(rb) // 15 * 15].reshape(-1, 15)
    plan = json.loads((sd / "plan.json").read_text())
    have = {a["id"] for a in plan["actions"]}
    for f in sorted(sd.glob("plan_resume_*.json")):
        for a in json.loads(f.read_text())["actions"]:
            if a["id"] not in have:
                plan["actions"].append(a)
                have.add(a["id"])
    return dict(plan=plan, reg=json.loads((sd / "registration.json").read_text()), ev=ev, ft=ft,
                t_b=np.zeros(0), r_b=np.zeros((0, 5, 4)), rb=rb, truth=None, sim=None)


class Slides:
    def __init__(self, sd):
        A.load = load_lean
        A.GRAVITY_FROM = str(GRAVITY_FROM)
        self.S = A.Session(sd)
        S = self.S
        self.ft, self.rb = S.d["ft"], S.d["rb"]
        self.T_ES = S.reg.T_ES
        cw = SIM.HEX_COMPLIANCE
        self.cr, self.ct = np.array(cw["r_rad_Nmm"]), np.array(cw["f_mm_N"])

    def samples(self, rec, T_rest, t0, t1):
        """Every HEX-E sample in [t0, t1]: time, the force on the pad in the base frame (N), the pad's pose T_BP (with
        the HEX-E's deflection) and the flange's pose."""
        S = self.S
        m = (self.ft[:, 0] >= t0) & (self.ft[:, 0] <= t1)
        t, w = self.ft[m, 0], self.ft[m, 3:9]
        if not len(t):
            return None
        j = np.clip(np.searchsorted(self.rb[:, 0], t), 1, len(self.rb) - 1)
        j = np.where(np.abs(self.rb[j - 1, 0] - t) < np.abs(self.rb[j, 0] - t), j - 1, j)
        ta, tb = float(np.mean(rec["t_rest0"])), float(np.mean(rec["t_rest1"])) if rec.get("t_rest1") else None
        za = S.zero(rec, ta)[0]
        zb = S.zero(rec, tb)[0] if tb is not None else za
        R0 = T_rest[:3, :3]
        g0 = R0.T @ np.array([0.0, 0.0, -1.0])
        out = dict(t=t, f_B=np.zeros((len(t), 3)), T_BP=np.zeros((len(t), 4, 4)), T_BE=np.zeros((len(t), 4, 4)),
                   tq_E=np.zeros((len(t), 3)))
        for k in range(len(t)):
            T_BE = C.ur_to_T(self.rb[j[k], 8:14])
            z = za + (zb - za) * ((t[k] - ta) / max(tb - ta, 1e-6) if tb is not None else 0.0)
            g1 = T_BE[:3, :3].T @ np.array([0.0, 0.0, -1.0])
            f_E = S.Kinv @ (w[k, :3] - z[:3] - S.A @ (g1 - g0))
            t_E = S.Ktinv @ (w[k, 3:] - z[3:] - S.Bt @ (g1 - g0)) * 1e3
            f_S, t_S = S.R_SE @ f_E, S.R_SE @ t_E
            T_BP = T_BE @ self.T_ES @ C.Tm(C.exp_so3(self.cr * t_S), self.ct * f_S) @ C.inv(self.T_ES) @ S.T_EP
            out["f_B"][k] = T_BE[:3, :3] @ f_E
            out["T_BP"][k] = T_BP
            out["T_BE"][k] = T_BE
            out["tq_E"][k] = t_E
        return out

    def slide(self, aid):
        """Rows (one per direction) and traces of one slide action."""
        S = self.S
        e = S.ends.get(aid)
        if not e or e.get("status") != "ok":
            return None
        rec, a = e["rec"], S.acts[aid]
        T_rest = np.array(rec["T_rest"])
        lv = rec["levels"][-1]
        gt0 = S.ground_truth(rec, lv["t"][0], lv["t"][1], T_rest)
        p_c, n_c = gt0["p"], gt0["n"]                   # contact point and normal in P at the press
        moves = rec["moves"]
        windows = [tuple(lv["t"])]
        for mv in moves:
            windows += [tuple(tu["t"]) for tu in mv["turns"]]
        rows, traces = [], []
        w_prev = tuple(lv["t"])
        # a fixed basis of the tangent plane for the whole slide: e1 along the first direction's travel, e2 = n x e1, and
        # the contact point's position at the press, for the 2D travel and shear of the vector fit
        sp = self.samples(rec, T_rest, lv["t"][0], lv["t"][1])
        xp_ = (np.einsum("kij,j->ki", sp["T_BP"][:, :3, :3], p_c) + sp["T_BP"][:, :3, 3]).mean(0)
        n0 = np.einsum("kij,j->ki", sp["T_BP"][:, :3, :3], n_c).mean(0)
        n0 /= np.linalg.norm(n0)
        mv0 = moves[0]
        e1 = np.cross(np.array(mv0["axis_B"], float), xp_ - np.array(mv0["centre_B"], float))
        e1 -= (e1 @ n0) * n0
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n0, e1)
        t_act = float(lv["t"][0])
        for k, mv in enumerate(moves):
            turns = mv["turns"]
            t_end = turns[-1]["t"][1]
            smp = self.samples(rec, T_rest, w_prev[0], t_end)
            if smp is None:
                w_prev = tuple(turns[-1]["t"])
                continue
            c_B = np.array(mv["centre_B"], float)
            ax_B = np.array(mv["axis_B"], float)
            ax_B /= np.linalg.norm(ax_B)
            x = np.einsum("kij,j->ki", smp["T_BP"][:, :3, :3], p_c) + smp["T_BP"][:, :3, 3]   # contact point, base
            n_B = np.einsum("kij,j->ki", smp["T_BP"][:, :3, :3], n_c)
            i0 = smp["t"] <= w_prev[1]
            x0 = x[i0].mean(0)
            v = np.cross(ax_B, x0 - c_B)
            v -= (v @ n_B[i0].mean(0)) * n_B[i0].mean(0)
            v /= np.linalg.norm(v)                          # the pad's direction of motion at the contact
            d = (x - x0) @ v
            s = -(smp["f_B"] @ v)
            fn = -np.einsum("ki,ki->k", smp["f_B"], n_B)
            ft_vec = smp["f_B"] + fn[:, None] * n_B
            ft_mag = np.linalg.norm(ft_vec, axis=1)
            # the turn angle from the flange's poses about the move's axis, against its pose at the start
            R0 = smp["T_BE"][i0][0][:3, :3]
            th = np.array([C.log_so3(T[:3, :3] @ R0.T) @ ax_B for T in smp["T_BE"]])
            hw = [w_prev] + [tuple(tu["t"]) for tu in turns]
            hold = []
            for (h0, h1), deg in zip(hw, [0.0] + [tu["deg"] for tu in turns]):
                mm = (smp["t"] >= h0) & (smp["t"] <= h1)
                hold.append(dict(deg=float(deg), d_mm=float(d[mm].mean()), s_N=float(s[mm].mean()), f_n_N=float(fn[mm].mean()),
                                 f_t_N=float(np.linalg.norm(ft_vec[mm].mean(0))), theta_deg=float(math.degrees(th[mm].mean())),
                                 n=int(mm.sum())))
            out = [h for h in hold[1:] if h["deg"] > 0]
            back = hold[-1] if hold[-1]["deg"] == 0.0 and len(hold) > 1 else None
            seq = [hold[0]] + out
            sub = []
            for h_a, h_b in zip(seq[:-1], seq[1:]):
                dd, ds = h_b["d_mm"] - h_a["d_mm"], h_b["s_N"] - h_a["s_N"]
                sub.append(dict(to_deg=h_b["deg"], dd_um=dd * 1e3, ds_N=ds, give_um_N=(dd * 1e3 / ds) if ds > 0.02 else None,
                                k_N_mm=ds / dd if abs(dd) > 1e-3 else None))
            k_first = sub[0]["k_N_mm"] if sub else None
            k_last = sub[-1]["k_N_mm"] if len(sub) > 1 else None
            stops = bool(k_first and k_first > 0 and k_last is not None and k_last < STOP_RATIO * k_first)
            top = out[-1] if out else None
            # 0.1 s bins of the trace, the stick-slip fit and the slopes after the start and after the reversal
            tb = smp["t"] - smp["t"][0]
            ib = np.floor(tb / BIN_S).astype(int)
            nb = np.bincount(ib)
            okb = nb > 0
            X2 = (x - xp_) @ np.c_[e1, e2]                       # mm, travel in the slide's fixed tangent basis
            G2 = -(smp["f_B"] @ np.c_[e1, e2])                  # N, the force on the pad against the travel
            B = {nm: (np.bincount(ib, w) / np.maximum(nb, 1))[okb]
                 for nm, w in (("t", tb), ("d", d), ("s", s), ("fn", fn), ("ft", ft_mag), ("x1", X2[:, 0]), ("x2", X2[:, 1]),
                               ("g1", G2[:, 0]), ("g2", G2[:, 1]), ("ta", smp["t"] - t_act))}
            t_rev = turns[-2]["t"][1] - smp["t"][0] if len(turns) > 1 else None
            fit = fit_epp(B["d"], B["s"], B["fn"])
            k_unl = slope(B, (B["t"] >= t_rev) & (B["d"] >= B["d"][B["t"] >= t_rev][0] - SLOPE_UM * 1e-3)) if t_rev else None
            t_go = w_prev[1] - smp["t"][0]
            k_load = slope(B, (B["t"] >= t_go) & (B["d"] <= SLOPE_UM * 1e-3) & (B["t"] <= (t_rev or 1e9)))
            rows.append(dict(
                action=aid, kind=a["kind"], region=a["target"].get("region"), target_P=a["target"]["p"],
                twist=float(a.get("twist") or 0.0), press_N=float(a.get("press_N") or 0.0), turn_v=a.get("turn_v"),
                dir_deg=float(mv.get("dir_deg", 0.0)), dir_index=k, from_rest=(k == 0), v_B=v.tolist(),
                v_P=(smp["T_BP"][i0][0][:3, :3].T @ v).tolist(), n_P=np.asarray(n_c).tolist(), p_P=np.asarray(p_c).tolist(),
                c_P=np.asarray(gt0["c_P"]).tolist(),
                start=hold[0], holds=out, back=back, sub=sub, give_first_um_N=sub[0]["give_um_N"] if sub else None,
                stops_growing=stops, epp=fit,
                k_load_N_mm=k_load, k_unload_N_mm=k_unl,
                s_over_fn_top=(top["s_N"] / top["f_n_N"]) if top else None,
                ft_over_fn_top=(top["f_t_N"] / top["f_n_N"]) if top else None,
                loop_mu=((top["s_N"] - back["s_N"]) / (top["f_n_N"] + back["f_n_N"])) if back and top else None,
                s_back_N=back["s_N"] if back else None, ft_back_N=back["f_t_N"] if back else None,
                z_back_N=(back["s_N"] - fit["s_off_N"]) if back and fit else None))
            traces.append(dict(action=aid, dir_deg=float(mv.get("dir_deg", 0.0)), dir_index=k, t_rev=t_rev, **B))
            w_prev = tuple(turns[-1]["t"])
        return rows, traces


def cmd_rows(a):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.sessions:
        t0 = time.time()
        sl = Slides(SESSIONS[name])
        ids = [aid for aid, act in sl.S.acts.items() if act.get("kind") == "slide" and aid in sl.S.ends]
        rows, traces = [], []
        for aid in ids:
            r = sl.slide(aid)
            if r is None:
                continue
            rows += r[0]
            traces += r[1]
        for r in rows:
            r["session"], r["tip"] = name, TIPS[name]
        p = OUT / f"20261026-ur5_slides_{name}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        np.savez_compressed(OUT / f"20261026-ur5_slide_traces_{name}.npz",
                            **{f"{k}_{i}": np.asarray(tr[k], np.float32) for i, tr in enumerate(traces)
                               for k in ("t", "d", "s", "fn", "ft", "x1", "x2", "g1", "g2", "ta")},
                            meta=json.dumps([dict(action=tr["action"], dir_deg=tr["dir_deg"], dir_index=tr["dir_index"],
                                                  t_rev=tr["t_rev"]) for tr in traces]))
        print(f"{name}: {len(ids)} slides, {len(rows)} directions -> {p.relative_to(ROOT)} ({time.time() - t0:.0f} s)",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--sessions", nargs="+", default=list(SESSIONS))
    sub.add_parser("figure")
    sub.add_parser("vector")
    a = ap.parse_args()
    if a.cmd == "rows":
        cmd_rows(a)
    elif a.cmd == "vector":
        cmd_vector(a)
    else:
        cmd_figure(a)


def mindlin_give_um_N(P, E_MPa, R=R_SPHERE, nu=0.45):
    """Mindlin's initial tangential compliance of a rigid sphere of radius R (mm) on an elastic half-space of modulus E
    under the normal force P (N): 1 / k_t, k_t = 8 G a / (2 - nu), a Hertz's contact radius; um/N."""
    Es = E_MPa / (1.0 - nu ** 2)
    a = (3.0 * P * R / (4.0 * Es)) ** (1.0 / 3.0)
    G = E_MPa / (2.0 * (1.0 + nu))
    return 1e3 * (2.0 - nu) / (8.0 * G * a)


def load_rows(name):
    return [r for r in (json.loads(l) for l in open(OUT / f"20261026-ur5_slides_{name}.jsonl")) if r.get("epp")]


def mean_fn(r):
    return float(np.mean([r["start"]["f_n_N"]] + [h["f_n_N"] for h in r["holds"]]))


def summary(rows):
    """Medians and quartiles by press level, and the power laws of the give and the friction coefficient."""
    def q(x):
        x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
        return dict(n=int(len(x)), median=float(np.median(x)), p25=float(np.percentile(x, 25)),
                    p75=float(np.percentile(x, 75))) if len(x) else dict(n=0)
    out = {"by_level": {}}
    for F in sorted({r["press_N"] for r in rows}):
        R = [r for r in rows if r["press_N"] == F]
        out["by_level"][f"{F:g}"] = dict(
            n=len(R), f_n_N=q([mean_fn(r) for r in R]), give_um_N=q([r["epp"]["give_um_N"] for r in R]),
            mu=q([r["epp"]["mu"] for r in R]),
            give_load50_um_N=q([1e3 / r["k_load_N_mm"] if r["k_load_N_mm"] and r["k_load_N_mm"] > 0 else None for r in R]),
            give_unload50_um_N=q([1e3 / r["k_unload_N_mm"] if r["k_unload_N_mm"] and r["k_unload_N_mm"] > 0 else None for r in R]),
            z_back_N=q([r["z_back_N"] for r in R]), ft_back_N=q([r["ft_back_N"] for r in R]),
            z_back_over_fn=q([r["z_back_N"] / r["back"]["f_n_N"] for r in R if r["back"]]),
            rms_N=q([r["epp"]["rms_N"] for r in R]),
            slip_both=int(sum(r["epp"]["slip_out"] and r["epp"]["slip_back"] for r in R)),
            slip_out_only=int(sum(r["epp"]["slip_out"] and not r["epp"]["slip_back"] for r in R)),
            from_rest=dict(n=int(sum(r["from_rest"] for r in R)),
                           give_um_N=q([r["epp"]["give_um_N"] for r in R if r["from_rest"]]),
                           mu=q([r["epp"]["mu"] for r in R if r["from_rest"]]))) 
    fn = np.array([mean_fn(r) for r in rows])
    g = np.array([r["epp"]["give_um_N"] for r in rows])
    mu = np.array([r["epp"]["mu"] for r in rows])
    bg, bm = np.polyfit(np.log(fn), np.log(g), 1), np.polyfit(np.log(fn), np.log(mu), 1)
    out["give_power_law"] = dict(exponent=float(bg[0]), at_1N_um_N=float(np.exp(bg[1])))
    out["mu_power_law"] = dict(exponent=float(bm[0]), at_1N=float(np.exp(bm[1])))
    out["mu_all"] = q(mu)
    out["by_direction"] = {f"{d:g}": dict(mu=q([r["epp"]["mu"] for r in rows if r["dir_deg"] == d]),
                                          give_um_N=q([r["epp"]["give_um_N"] for r in rows if r["dir_deg"] == d]))
                           for d in sorted({r["dir_deg"] for r in rows})}
    out["by_region"] = {g_: dict(mu=q([r["epp"]["mu"] for r in rows if r["region"] == g_]),
                                 give_um_N=q([r["epp"]["give_um_N"] for r in rows if r["region"] == g_]))
                        for g_ in sorted({r["region"] for r in rows})}
    out["by_turn_v"] = {str(tv): dict(mu=q([r["epp"]["mu"] for r in rows if str(r["turn_v"]) == str(tv)]),
                                      give_um_N=q([r["epp"]["give_um_N"] for r in rows if str(r["turn_v"]) == str(tv)]))
                        for tv in sorted({str(r["turn_v"]) for r in rows})}
    return out


def cmd_figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 11.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "mathtext.fontset": "dejavuserif"})
    rows = {nm: load_rows(nm) for nm in SESSIONS}
    S = {nm: summary(R) for nm, R in rows.items()}
    (OUT / "20261026-ur5_slides_summary.json").write_text(json.dumps(S, indent=1) + "\n")
    Z = np.load(OUT / "20261026-ur5_slide_traces_cb2t.npz")
    meta = json.loads(str(Z["meta"]))
    fig = plt.figure(figsize=(10.0, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.0))
    a1 = fig.add_subplot(gs[0, :])
    a2, a3 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    # (a) one face slide at 1 N, its eight directions: shear less the fitted offset against the travel
    aid = "slide-000-1N"
    R1 = {r["dir_deg"]: r for r in rows["cb2t"] if r["action"] == aid}
    hl = 90.0
    for i, m in enumerate(meta):
        if m["action"] != aid:
            continue
        r = R1[m["dir_deg"]]
        d, s, fn = Z[f"d_{i}"] * 1e3, Z[f"s_{i}"] - r["epp"]["s_off_N"], Z[f"fn_{i}"]
        if m["dir_deg"] != hl:
            a1.plot(d, s, "-", color="0.8", lw=0.8, zorder=1)
            continue
        a1.plot(d, s, "-", color="#2e5fa8", lw=1.6, zorder=3, label=f"Measured, direction {hl:.0f}\N{DEGREE SIGN}")
        e = r["epp"]
        z = epp(Z[f"d_{i}"].astype(float), np.maximum(fn.astype(float), 0.05), e["k_N_mm"], e["mu"], e["z0_N"])
        a1.plot(d, z, "--", color="black", lw=1.4, zorder=4,
                label=f"Spring and slider: {e['give_um_N']:.0f} \N{MICRO SIGN}m/N, \N{GREEK SMALL LETTER MU} {e['mu']:.2f}")
    a1.plot([], [], "-", color="0.8", lw=0.8, label="The other seven directions")
    a1.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    a1.set_xlabel("Travel of the pad along the turn (\N{MICRO SIGN}m)")
    a1.set_ylabel("Shear against the travel (N)")
    a1.set_title("(a) One face slide at 1 N, eight directions", loc="left")
    a1.legend(loc="upper left", frameon=False)
    # (b) give against the normal force
    colors = {"cb2t": "#2e5fa8", "cb1r": "#b06010"}
    labels = {"cb2t": "Tip 2 (cb2t)", "cb1r": "Tip 1 (cb1r)"}
    rng = np.random.default_rng(0)
    for nm, R in rows.items():
        fn = np.array([mean_fn(r) for r in R])
        g = np.array([r["epp"]["give_um_N"] for r in R])
        mu = np.array([r["epp"]["mu"] for r in R])
        a2.scatter(fn, g, s=9, alpha=0.35, color=colors[nm], lw=0, rasterized=True)
        for F, v in S[nm]["by_level"].items():
            if v["n"] < 8:
                continue
            x = v["f_n_N"]["median"]
            a2.errorbar([x], [v["give_um_N"]["median"]], yerr=[[v["give_um_N"]["median"] - v["give_um_N"]["p25"]],
                                                                [v["give_um_N"]["p75"] - v["give_um_N"]["median"]]],
                        fmt="o", ms=7, color=colors[nm], mec="white", mew=1.2, capsize=3, zorder=5)
        a2.plot([], [], "o", color=colors[nm], ms=7, label=labels[nm])
    P = np.linspace(0.3, 3.0, 100)
    a2.fill_between([0.0, 3.2], FE_GIVE_UM_N[0], FE_GIVE_UM_N[1], color="0.88", lw=0, zorder=0)
    a2.text(0.08, FE_GIVE_UM_N[0] + 8, "FE pad under the 12.5 mm rod", fontsize=11, va="bottom", color="0.3")
    for E, ls in ((2.0, "-"), (0.9, ":")):
        a2.plot(P, mindlin_give_um_N(P, E), ls, color="black", lw=1.2, label=f"Mindlin, 5 mm sphere, {E:g} MPa")
    a2.set_xlim(0.0, 3.2)
    a2.set_ylim(0, 700)
    a2.set_xlabel("Normal force during the slide (N)")
    a2.set_ylabel("Give before slip (\N{MICRO SIGN}m/N)")
    a2.set_title("(b) Give before slip", loc="left")
    a2.legend(loc="upper right", frameon=False, fontsize=10.5)
    V = [json.loads(l) for l in open(OUT / "20261026-ur5_slides_vector_cb2t.jsonl")] if (OUT / "20261026-ur5_slides_vector_cb2t.jsonl").exists() else []
    if V:
        fnv = np.array([v["fn_mean_N"] for v in V])
        muv = np.array([v["fits"]["const"]["mu"] for v in V])
        mx = np.array([np.median(v["max_ft_over_fn_by_dir"]) for v in V])
        a3.scatter(fnv, mx, s=16, marker="s", facecolor="none", edgecolor="0.55", lw=0.8, label="Largest shear over normal force of a direction")
        a3.scatter(fnv, muv, s=16, color=colors["cb2t"], alpha=0.6, lw=0, label="Vector fit with an offset, per slide")
        for F in sorted({v["press_N"] for v in V}):
            sel = np.array([v["press_N"] == F for v in V])
            if sel.sum() < 8:
                continue
            m_, lo, hi = np.median(muv[sel]), np.percentile(muv[sel], 25), np.percentile(muv[sel], 75)
            a3.errorbar([np.median(fnv[sel])], [m_], yerr=[[m_ - lo], [hi - m_]], fmt="o", ms=7, color=colors["cb2t"], mec="white",
                        mew=1.2, capsize=3, zorder=5)
        a3.legend(loc="upper right", frameon=False, fontsize=10.5)
    a3.axhspan(0.55, 1.15, color="0.88", lw=0, zorder=0)
    a3.text(1.95, 0.74, "RL friction range", fontsize=11, color="0.3")
    a3.axhline(1.0, color="black", lw=1.0, ls="--")
    a3.text(0.1, 1.02, "Scenes' \N{GREEK SMALL LETTER MU} = 1", fontsize=11, color="0.3", va="bottom")
    a3.set_xlim(0.0, 3.2)
    a3.set_ylim(0, 1.6)
    a3.set_xlabel("Normal force during the slide (N)")
    a3.set_ylabel("Friction coefficient \N{GREEK SMALL LETTER MU}")
    a3.set_title("(c) Friction coefficient", loc="left")
    for ax in (a1, a2, a3):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / "20261026-ur5_slides.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out.relative_to(ROOT), flush=True)
    for nm in SESSIONS:
        b = S[nm]["by_level"]
        print(nm, {F: (v["n"], round(v["give_um_N"]["median"]), round(v["mu"]["median"], 3)) for F, v in b.items()},
              "give ~ f_n^%.2f" % S[nm]["give_power_law"]["exponent"])


if __name__ == "__main__":
    main()

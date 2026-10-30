#!/usr/bin/env python3
"""Figures, film and numbers of the page on one contact-model modulus for the v3 fingertips (scripts/v3_tip_contact_model.py
writes the data; scripts/v3_tip_contact_model_page.py builds the page).

  force    normal force against registered indentation at the face: (a) the low 5 mm sphere, (b) the 12.5 mm rod, both v3
           tips' face holds, with the pads and Drake's field at their joint modulus, the FE at s = 1 and at its joint
           scale, and the scenes' 10 MPa pads
  ratio    measured over model force at the joint modulus by load level, low sphere and rod, both tips pooled
  patch    (a) the patch's pressure-weighted mean radius against normal force at the face under the sphere and the rod,
           three models; (b) the twists' torsion against normal force with the models' mu F r_bar
  film     the three models' contact patches at the face centre under the 5 mm sphere as the force rises from 0.25 to 8 N
  numbers  data/20261030-numbers.json: the measured face compliance at 1 N (power law), the models' depth at 1 N, the
           FE's force at s = 1 over the measured, the twists by level

    .venv/bin/python scripts/v3_tip_contact_model_figures.py [--only force ratio patch film numbers]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
D = ROOT / "docs/experiments/20261030-v3_tip_contact_model"
MEDIA, DATA = D / "media", D / "data"
FIT = DATA / "20261030-contact_model_fit.json"
NUMBERS = DATA / "20261030-numbers.json"
ORANGE, BLUE, GREY, BLACK = "#BE7514", "#4A7FC4", "#79858E", "#222222"
E_SCENES = 10.0
LOW, ROD = ("v3id1_low", "v3id2_low"), ("v3id1_rod", "v3id2_rod")


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "mathtext.fontset": "dejavuserif"})
    return plt


def save(fig, name):
    import matplotlib.pyplot as plt
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


def arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def load():
    return json.loads(FIT.read_text())


def face_holds(d, keys):
    F, Dp, T = [], [], []
    for k in keys:
        h = d["sessions_detail"][k]["holds"]
        m = np.array(h["region"]) == "face"
        F.append(np.array(h["f_n"])[m])
        Dp.append(np.array(h["depth"])[m])
        T.append(np.full(m.sum(), k))
    return np.concatenate(F), np.concatenate(Dp), np.concatenate(T)


def model_curves(d, keys):
    """Mean over the sessions of the median curves at the face targets: depth, pads and field per MPa, FE at s = 1 and
    at the joint scale, and r_bar of each model against depth."""
    C = [d["sessions_detail"][k]["curves"] for k in keys]
    dep = np.array(C[0]["depth_mm"])
    mean = lambda key: np.nanmean([arr(c[key]) for c in C], axis=0)  # noqa: E731
    rb = {m: np.nanmean([arr(c["rbar_mm"][m]) for c in C], axis=0) for m in ("pads", "field", "fe")}
    return dep, mean("pads_N_per_MPa"), mean("field_N_per_MPa"), mean("fe_s1_N"), mean("fe_fit_N"), rb


def level_medians(F, Dp):
    """Median force and depth of the holds at each nominal level, from the force's own clusters."""
    lv = np.array([0.25, 0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 8])
    out = []
    for L in lv:
        m = np.abs(F - L * 0.85) < 0.18 * L + 0.05
        if m.sum() >= 5:
            out.append((float(np.median(Dp[m])), float(np.median(F[m]))))
    return np.array(out)


def fig_force(d):
    plt = plt_setup()
    J = d["fits"]["joint"]
    Ep, Ef, sf = J["E_pads_MPa"], J["E_field_MPa"], J["s_fe"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), sharey=True)
    for ax, keys, title, xmax in ((axes[0], LOW, "(a) 5 mm sphere, face", 2.6), (axes[1], ROD, "(b) 12.5 mm rod, face", 1.8)):
        F, Dp, T = face_holds(d, keys)
        for k, shade in zip(keys, ("0.62", "0.30")):
            m = T == k
            ax.scatter(Dp[m], F[m], s=10, color=shade, alpha=0.55, lw=0, zorder=1, label=f"Measured, {d['sessions'][k]['tip']}")
        dep, Gp, Gf, F1, Fs, _ = model_curves(d, keys)
        ax.plot(dep, Ep * Gp, color=ORANGE, lw=2.6, zorder=4, label=f"Pads, {Ep:.2f} MPa (fitted)")
        ax.plot(dep, Ef * Gf, color=BLUE, lw=2.0, ls=(0, (6, 3)), zorder=4, label=f"Drake’s field, {Ef:.2f} MPa (fitted)")
        ax.plot(dep, Fs, color=BLACK, lw=2.0, zorder=3, label=f"FE, {12.0 / sf:.1f} MPa (one scale, fitted)")
        ax.plot(dep, F1, color=BLACK, lw=1.4, ls=":", zorder=3, label="FE, 12 MPa (as computed)")
        ax.plot(dep, E_SCENES * Gp, color=ORANGE, lw=1.4, ls=":", zorder=3, label="Pads, 10 MPa (scenes)")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 8.6)
        ax.set_xlabel("Registered indentation (mm)")
        ax.set_title(title)
        ax.grid(color="0.9", lw=0.6)
    axes[0].set_ylabel("Normal force (N)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[2:] + h[:2], lab[2:] + lab[:2], loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=2, frameon=False, fontsize=12, handlelength=2.6, markerscale=2.0)
    fig.tight_layout()
    save(fig, "20261030-force_indentation.png")


def fig_ratio(d):
    plt = plt_setup()
    S = d["sessions_detail"]
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.axhspan(0.8, 1.25, color="0.93", zorder=0)
    ax.axhline(1.0, color="0.5", lw=0.8, zorder=1)
    mk = dict(low="o", rod="s")
    off = dict(pads=-0.045, field=0.0, fe=0.045)
    for ind, keys in (("low", LOW), ("rod", ROD)):
        for model, col, lab in (("pads", ORANGE, "Pads"), ("field", BLUE, "Drake’s field"), ("fe", BLACK, "FE, one scale")):
            r_all, lv_all = [], []
            for k in keys:
                h = S[k]["holds"]
                f = np.array(h["f_n"])
                if model == "pads":
                    m = f / (d["fits"]["joint"]["E_pads_MPa"] * np.array(h["Gp"]))
                elif model == "field":
                    m = f / (d["fits"]["joint"]["E_field_MPa"] * np.array(h["Gf"]))
                else:
                    fe = np.array(h["fe_fit"])
                    m = np.where(fe > 0, f / np.where(fe > 0, fe, 1), np.nan)
                r_all.append(m)
                lv_all.append(np.array(h["level"]))
            r, lv = np.concatenate(r_all), np.concatenate(lv_all)
            Ls = [L for L in sorted(set(lv.tolist())) if (lv == L).sum() >= 8]
            x, med, lo, hi = [], [], [], []
            for L in Ls:
                v = r[(lv == L) & np.isfinite(r) & (r > 0)]
                x.append(L * np.exp(off[model] * 3))
                med.append(np.median(v))
                lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
            x, med, lo, hi = map(np.array, (x, med, lo, hi))
            ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt=mk[ind], color=col, mfc=col if ind == "low" else "white",
                        mec=col, ms=7, lw=1.2, capsize=2.5, ls="-" if ind == "low" else "--", zorder=3,
                        label=f"{lab}, {'5' if ind == 'low' else '12.5'} mm {'sphere' if ind == 'low' else 'rod'}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 3.5, 5, 8])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "3.5", "5", "8"])
    ax.set_yticks([0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.6, 2, 2.5])
    ax.set_yticklabels(["0.4", "0.5", "0.6", "0.8", "1", "1.25", "1.6", "2", "2.5"])
    ax.minorticks_off()
    ax.set_ylim(0.38, 2.7)
    ax.set_xlabel("Load level of the press (N)")
    ax.set_ylabel("Measured force over model force")
    ax.grid(color="0.9", lw=0.6, which="major")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=12)
    fig.tight_layout()
    save(fig, "20261030-force_ratio_by_level.png")


def fig_patch(d):
    plt = plt_setup()
    J = d["fits"]["joint"]
    Ep, Ef = J["E_pads_MPa"], J["E_field_MPa"]
    MU = {float(k): v for k, v in d["mu"].items()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.9))
    Fq = np.linspace(0.25, 8.0, 200)
    for keys, ls, lab in ((LOW, "-", "sphere"), (ROD, "--", "rod")):
        dep, Gp, Gf, F1, Fs, rb = model_curves(d, keys)
        for model, col, F in (("pads", ORANGE, Ep * Gp), ("field", BLUE, Ef * Gf), ("fe", BLACK, Fs)):
            Fm = np.maximum.accumulate(F)
            ok = np.r_[True, np.diff(Fm) > 1e-9] & np.isfinite(rb[model])
            if ok.sum() < 3:
                continue
            r_at = np.interp(Fq, Fm[ok], rb[model][ok], left=np.nan, right=np.nan)
            if model == "fe":
                r_at[Fq < 2.0] = np.nan            # the FE's surface nodes are 1 mm apart: one node carries a light press
            name = {"pads": "Pads", "field": "Drake’s field", "fe": "FE, one scale"}[model]
            a1.plot(Fq, r_at, color=col, lw=2.0, ls=ls, label=f"{name}, {'5' if lab == 'sphere' else '12.5'} mm {lab}")
    a1.set_xlabel("Normal force (N)")
    a1.set_ylabel("Mean radius of the patch, $\\bar r$ (mm)")
    a1.set_title("(a) Patch at the face, fitted moduli")
    a1.set_xlim(0, 8.2)
    a1.set_ylim(0, 2.2)
    a1.grid(color="0.9", lw=0.6)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=11)
    S = d["sessions_detail"]
    for k, mk_, col, lab in (("v3id1_low", "o", "0.35", "5 mm sphere, v3 ID 1"), ("v3id2_low", "o", "0.65", "5 mm sphere, v3 ID 2"),
                             ("v3id1_rod", "s", "0.35", "12.5 mm rod, v3 ID 1"), ("v3id2_rod", "s", "0.65", "12.5 mm rod, v3 ID 2")):
        T = S[k]["twists"]
        a2.scatter([t["f_n"] for t in T], [t["M_Nmm"] for t in T], marker=mk_, s=26, color=col, alpha=0.8, lw=0, label=lab, zorder=3)
    dep, Gp, Gf, F1, Fs, rb = model_curves(d, LOW)
    for model, col, F in (("pads", ORANGE, Ep * Gp), ("field", BLUE, Ef * Gf)):
        Fm = np.maximum.accumulate(F)
        ok = np.r_[True, np.diff(Fm) > 1e-9] & np.isfinite(rb[model])
        r_at = np.interp(Fq, Fm[ok], rb[model][ok], left=np.nan, right=np.nan)
        mu = np.where(Fq < 1.5, MU[1.0], MU[2.5])
        a2.plot(Fq, mu * Fq * r_at, color=col, lw=2.0,
                label=f"{'Pads' if model == 'pads' else 'Drake’s field'}, $\\mu F \\bar r$, sphere")
    a2.axhline(0, color="0.5", lw=0.8)
    a2.set_xlabel("Normal force (N)")
    a2.set_ylabel("Torsion about the normal (N mm)")
    a2.set_title("(b) Twists: half the out-and-back difference")
    a2.set_xlim(0, 5.6)
    a2.set_ylim(-4, 11)
    a2.grid(color="0.9", lw=0.6)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "20261030-patch_and_twists.png")


def film(d, n_frames=72, fps=18):
    """The three models' patches at the face centre of v3 ID 1 under the 5 mm sphere, at equal normal force from 0.25 to
    8 N and back, each at its fitted modulus: pads in contact (discs, area by force), the field's pressure on the sphere's
    surface, the FE's loaded nodes; the dashed circle has radius r_bar about the pressure centre."""
    import matplotlib.animation as animation
    import v3_tip_contact_model as M
    plt = plt_setup()
    J = d["fits"]["joint"]
    Ep, Ef, sf = J["E_pads_MPa"], J["E_field_MPa"], J["s_fe"]
    key = "v3id1_low"
    ctx = M.make_ctx(key)
    H, R = ctx["H"], ctx["R"]
    m = np.nonzero((H["kind"] == "press") & (H["region"] == "face") & (H["f_n"] > 0.2))[0]
    i = int(m[np.argmax(H["p"][m][:, 0])])                     # the face hold nearest the apex (largest x_P)
    c0, n = H["c_P"][i].astype(float), H["n"][i].astype(float)
    d0 = float(H["depth"][i])
    p0 = H["p"][i].astype(float) + d0 * n                       # the pad surface point under the sphere
    t1 = np.cross(n, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    uv = lambda X: np.c_[(X - p0) @ t1, (X - p0) @ t2]  # noqa: E731
    pads, field, fe = ctx["pads"], ctx["field"], ctx["fe"]
    fe.n_hold = n
    pl = field.planes(H["p"][i])
    at_depth = lambda dd: c0 + (d0 - dd) * n  # noqa: E731

    def solve(fun, F):
        lo, hi = 0.0, 3.2
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if fun(mid) < F:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    f_p = lambda dd: Ep * float(M.pads_contact(pads, at_depth(dd), n, R, None)[0] @ n)  # noqa: E731
    f_f = lambda dd: Ef * float(M.field_contact(field, at_depth(dd), n, R, None, dd + 0.6, pl)[0] @ n)  # noqa: E731
    f_e = lambda dd: float(-fe.contact(at_depth(dd), sf, R, None)[0] @ n)  # noqa: E731
    up = np.geomspace(0.25, 8.0, n_frames // 2)
    Fs = np.r_[up, up[::-1]]
    lat = uv(pads.S)
    near = np.linalg.norm(lat, axis=1) < 4.6
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.9))
    import matplotlib.cm as cm
    import matplotlib
    cmap = matplotlib.colormaps["magma_r"]
    frames = []
    for F in Fs:
        dp, df, de = solve(f_p, F), solve(f_f, F), solve(f_e, F)
        frames.append((F, dp, df, de))

    def draw(k):
        F, dp, df, de = frames[k]
        for ax in axes:
            ax.clear()
            ax.set_xlim(-4.2, 4.2)
            ax.set_ylim(-4.2, 4.2)
            ax.set_aspect("equal")
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_yticks([-4, -2, 0, 2, 4])
            ax.grid(color="0.92", lw=0.5)
        # pads
        a = axes[0]
        a.scatter(lat[near, 0], lat[near, 1], s=10, color="0.82", lw=0)
        c = at_depth(dp)
        v = c - pads.C
        dd_ = np.linalg.norm(v, axis=1)
        o = M.PD.RS + R - dd_
        mm = o > 0
        f = Ep * pads.k[mm] * o[mm]
        P = pads.C[mm] + M.PD.RS * v[mm] / dd_[mm, None]
        q = uv(P)
        a.scatter(q[:, 0], q[:, 1], s=40 + 900 * f / max(F, 0.25), c=f / np.maximum(f.max(), 1e-9), cmap=cmap, vmin=0, vmax=1,
                  edgecolors="black", linewidths=0.4)
        _, cen, npd, rbp = M.pads_contact(pads, c, n, R, None)
        if cen is not None:
            cc = uv(cen[None])[0]
            a.add_patch(plt.Circle(cc, rbp, fill=False, ls="--", color=ORANGE, lw=1.6))
        a.set_title(f"1 mm pads, {Ep:.2f} MPa\n{npd} pads, depth {dp:.2f} mm", fontsize=12)
        # field
        a = axes[1]
        X, U, A = M.indenter_samples(at_depth(df), n, R, None, df + 0.6)
        p = field.p(X, pl)
        ins = p > 0
        q = uv(X[ins])
        a.scatter(q[:, 0], q[:, 1], s=3, c=p[ins] / max(p.max(), 1e-12), cmap=cmap, vmin=0, vmax=1, lw=0)
        _, cen, af, rbf = M.field_contact(field, at_depth(df), n, R, None, df + 0.6, pl)
        if cen is not None:
            a.add_patch(plt.Circle(uv(cen[None])[0], rbf, fill=False, ls="--", color=BLUE, lw=1.6))
        a.set_title(f"Drake’s field, {Ef:.2f} MPa\n{af:.1f} mm², depth {df:.2f} mm", fontsize=12)
        # FE
        a = axes[2]
        ql = uv(fe.X)
        nr = np.linalg.norm(ql, axis=1) < 4.6
        a.scatter(ql[nr, 0], ql[nr, 1], s=10, color="0.82", lw=0)
        r = fe.X - at_depth(de)
        dist = np.linalg.norm(r, axis=1)
        gap = dist - R
        nh = r / dist[:, None]
        K = np.nonzero((gap < 0.6) & (np.einsum("ij,ij->i", nh, fe.N) < -0.2))[0]
        Fv, cen, nk, rbe = fe.contact(at_depth(de), sf, R, None)
        if len(K):
            from scipy.optimize import nnls
            idx = (3 * K[:, None] + np.arange(3)[None]).ravel()
            Ck = fe.C[np.ix_(idx, idx)] * sf
            Nk = np.zeros((3 * len(K), len(K)))
            for j, ii in enumerate(K):
                Nk[3 * j:3 * j + 3, j] = nh[ii]
            Mn = Nk.T @ Ck @ Nk
            Mn = 0.5 * (Mn + Mn.T) + 1e-9 * np.eye(len(K))
            Lc = np.linalg.cholesky(Mn)
            pp, _ = nnls(Lc.T, -np.linalg.solve(Lc, gap[K]))
            ld = pp > 1e-9
            q = uv(fe.X[K][ld])
            a.scatter(q[:, 0], q[:, 1], s=40 + 900 * pp[ld] / max(F, 0.25), c=pp[ld] / pp.max(), cmap=cmap, vmin=0, vmax=1,
                      edgecolors="black", linewidths=0.4)
        if cen is not None and np.isfinite(rbe) and rbe > 0:
            a.add_patch(plt.Circle(uv(cen[None])[0], rbe, fill=False, ls="--", color=BLACK, lw=1.6))
        a.set_title(f"FE, one scale ({12.0 / sf:.1f} MPa)\n{nk} nodes, depth {de:.2f} mm", fontsize=12)
        axes[0].set_ylabel("Along the finger (mm)")
        for ax in axes:
            ax.set_xlabel("Across the finger (mm)")
        fig.suptitle(f"Normal force {F:.2f} N under the 5 mm sphere at the face centre (v3 ID 1)", fontsize=13, y=0.99)
        fig.tight_layout()

    MEDIA.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    out = MEDIA / "20261030-patch_film.mp4"
    with writer.saving(fig, str(out), dpi=100):
        for k in range(len(frames)):
            draw(k)
            writer.grab_frame()
    kp = int(np.argmin(np.abs(np.array([f[0] for f in frames[:len(up)]]) - 2.0)))
    draw(kp)
    fig.savefig(MEDIA / "20261030-patch_film_poster.png", dpi=100, facecolor="white")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT), "and its poster", flush=True)
    return dict(target=str(H["id"][i]), p_P=p0.round(3).tolist(),
                at_N={f"{F:.2f}": dict(depth_pads=dp, depth_field=df, depth_fe=de) for F, dp, df, de in frames[:len(up)][::7]})


def numbers(d):
    """Measured face compliance at 1 N from depth = a + b F^p over each indenter's face holds; the curves' depth at 1 N;
    the FE at s = 1 over the measured force at the registered positions (face); the twists by level."""
    from scipy.optimize import curve_fit
    J = d["fits"]["joint"]
    Ep, Ef = J["E_pads_MPa"], J["E_field_MPa"]
    out = dict(joint=J, sessions={k: v for k, v in d["sessions"].items()}, spring=d.get("spring"))
    for ind, keys in (("low", LOW), ("rod", ROD)):
        F, Dp, _ = face_holds(d, keys)
        f = lambda x, a, b, p: a + b * np.power(x, p)  # noqa: E731
        (a, b, p), _ = curve_fit(f, F, Dp, p0=(0.1, 0.4, 0.7), maxfev=20000)
        dep, Gp, Gf, F1, Fs, rb = model_curves(d, keys)
        inv = lambda Fc: float(np.interp(1.0, np.maximum.accumulate(Fc), dep))  # noqa: E731
        slope = lambda Fc: float(np.interp(1.1, np.maximum.accumulate(Fc), dep) - np.interp(0.9, np.maximum.accumulate(Fc), dep)) / 0.2  # noqa: E731
        fe_s1 = np.concatenate([np.array(d["sessions_detail"][k]["holds"]["fe_s1"], float)[np.array(d["sessions_detail"][k]["holds"]["region"]) == "face"] for k in keys])
        fm = np.concatenate([np.array(d["sessions_detail"][k]["holds"]["f_n"])[np.array(d["sessions_detail"][k]["holds"]["region"]) == "face"] for k in keys])
        ok = fe_s1 > 0
        def rbar_at(Fc, rbm, F0):
            Fm = np.maximum.accumulate(Fc)
            ok = np.r_[True, np.diff(Fm) > 1e-9] & np.isfinite(rbm)
            return float(np.interp(F0, Fm[ok], rbm[ok], left=np.nan, right=np.nan))
        patch = {f"{F0:g}": dict(pads_fit=rbar_at(Ep * Gp, rb["pads"], F0), field_fit=rbar_at(Ef * Gf, rb["field"], F0),
                                 fe_fit=rbar_at(Fs, rb["fe"], F0) if F0 >= 2.0 else None,
                                 pads_10MPa=rbar_at(E_SCENES * Gp, rb["pads"], F0)) for F0 in (1.0, 2.0, 2.5, 5.0)}
        out[ind] = dict(face_holds=int(len(F)), fit=dict(a_mm=float(a), b_mm=float(b), p=float(p)), rbar_mm=patch,
                        measured_depth_1N_mm=float(a + b), measured_compliance_1N_mm_N=float(b * p),
                        depth_1N_mm=dict(pads_fit=inv(Ep * Gp), field_fit=inv(Ef * Gf), fe_s1=inv(F1), fe_fit=inv(Fs),
                                         pads_10MPa=inv(E_SCENES * Gp)),
                        compliance_1N_mm_N=dict(pads_fit=slope(Ep * Gp), pads_10MPa=slope(E_SCENES * Gp), fe_fit=slope(Fs)),
                        fe_s1_over_measured_face=float(np.median(fe_s1[ok] / fm[ok])))
    tw = {}
    for ind, keys in (("low", LOW), ("rod", ROD)):
        T = [t for k in keys for t in d["sessions_detail"][k]["twists"]]
        for L in sorted(set(t["level"] for t in T)):
            X = [t for t in T if t["level"] == L]
            M_ = np.array([t["M_Nmm"] for t in X])
            Fn = np.array([t["f_n"] for t in X])
            mu = d["mu"]["1"] if L < 2 else d["mu"]["2.5"]
            ap = arr([t["arm_pads_mm"] for t in X])
            af = arr([t["arm_field_mm"] for t in X])
            tw[f"{ind}_{L:g}"] = dict(n=len(X), f_n=float(np.median(Fn)), M_median=float(np.median(M_)),
                                      M_p25=float(np.percentile(M_, 25)), M_p75=float(np.percentile(M_, 75)),
                                      M_min=float(M_.min()), M_max=float(M_.max()),
                                      pads=float(np.nanmedian(mu * Fn * ap)), field=float(np.nanmedian(mu * Fn * af)),
                                      arm_meas=float(np.median(M_ / (mu * Fn))), arm_pads=float(np.nanmedian(ap)),
                                      arm_field=float(np.nanmedian(af)))
    out["twists"] = tw
    NUMBERS.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("wrote", NUMBERS.relative_to(ROOT))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    d = load()
    want = lambda k: a.only is None or k in a.only  # noqa: E731
    if want("numbers"):
        numbers(d)
    if want("force"):
        fig_force(d)
    if want("ratio"):
        fig_ratio(d)
    if want("patch"):
        fig_patch(d)
    if want("film"):
        info = film(d)
        N = json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}
        N["film"] = info
        NUMBERS.write_text(json.dumps(N, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()

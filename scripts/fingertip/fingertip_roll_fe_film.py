#!/usr/bin/env python3
"""Finite elements and magnetic field inside the fingertip during the UR5 rod roll of scripts/fingertip/viewer_roll_film.py
(owner, 2026-11-10: "ideally updated FEM and magnetic field simulation plots/videos if we dont arleady have them").

The FE model is the one of the estimator's surface basis for pad 05R4 of the v3 tips (v2_tip_fe_checks case b_05R4, as
scripts/fingertip/fingertip_wrench_fe.py --surface and fingertip_physical_model_fe.py use it: walls and skins of solid
85A, the homogenized 30 % gyroid core, four bonded 3 x 2 mm N52 magnets, the cut face and the mount face fixed). At
keyframes every --dt s of the roll, the rod of radius 12.5 mm is placed on the face at its recorded pose (the robot's
record in the pad frame P, which is the FE's tip frame) and pressed along -x until the contact's force along -x equals
the HEX-E's (frictionless contact on the face nodes, solved as a linear complementarity problem, tpu_tip_fem_tip.
force_controlled); the HEX-E's whole force vector is then applied on the contact nodes in proportion to their pressure.
Each keyframe keeps the displacement of the nodes in the section y = the contact's y, the face nodes' pressure, and the
magnets' rigid motions. The field model (fingertip_wrench_model.Model, with v3 ID 5's full rest calibration) turns the
motions into the five chips' change, which the film sets against the change the board measured (the viewer's frames).

    ~/.claude/bin/resguard.sh run --mem 5G --cpu 300 -- ~/miniconda3/envs/femtip/bin/python \\
        scripts/fingertip/fingertip_roll_fe_film.py solve [--dt 0.5]
    ~/miniconda3/bin/python scripts/fingertip/fingertip_roll_fe_film.py film
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

OUT = ROOT / "docs/experiments/20261110-fingertip_viewer_roll"
DATA, MEDIA = OUT / "data", OUT / "media"
WORK = ROOT / "logs/20261110-viewer_roll_film"
STEM = "20261110-v3id5_rod_roll"
KEYS = WORK / "fe_roll_keyframes.npz"
T_FROM, T_TO = 12.7, 83.7               # s of the window: the robot's samples of the roll


def rod_gaps(X, c, a, R):
    """Distance (mm) each point X (n, 3) must move along +x to reach the surface of the rod of radius R about the line
    through c along a (inf where the line x -> +inf misses the rod): the smaller root of |P (X + s e_x - c)| = R."""
    a = a / np.linalg.norm(a)
    P = np.eye(3) - np.outer(a, a)
    v = (X - c) @ P.T
    e = P @ np.array([1.0, 0.0, 0.0])
    A, B, C = e @ e, 2 * v @ e, (v * v).sum(1) - R * R
    disc = B * B - 4 * A * C
    s = np.full(len(X), np.inf)
    ok = disc >= 0
    s[ok] = (-B[ok] - np.sqrt(disc[ok])) / (2 * A)
    return s


def solve(a):
    import fe_contact_calibration_fe as FE
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as TM
    import v2_tip_fe_checks as V
    t0 = time.time()
    Tr = np.load(DATA / f"{STEM}_truth.npz")
    tR = Tr["t"] - float(Tr["t_lo"])
    R = float(Tr["rod_R_mm"])
    tk = np.arange(T_FROM, T_TO, a.dt)
    idx = np.clip(np.searchsorted(tR, tk), 0, len(tR) - 1)
    y_c = float(np.median(Tr["p"][:, 1]))
    c = V.CASES["b_05R4"]
    sh, G, mat_id, mats, lay, counts, recipe = V.mat_ids(c)
    jsec = int(np.argmin(np.abs(G.cy - y_c)))
    cls_sec = mat_id[:, jsec, :].copy()
    model = V.make_model(c, G, mat_id, mats)
    del mat_id
    mn = FE.magnet_nodes(model, lay)
    ys = G.ys[np.argmin(np.abs(G.ys - y_c))]
    sec = np.where(np.abs(model.X[:, 1] - ys) < 1e-9)[0]
    print(f"model {time.time() - t0:.0f} s: {len(model.free)} dof, section y {ys:.2f} mm ({len(sec)} nodes)", flush=True)
    nodes = model.face_nodes(zlim=(0.4, 21.41))
    X = model.X[nodes].copy()
    xt = sh.fe_surface_x(X[:, 1], X[:, 2])
    X[:, 0] = np.where(np.isfinite(xt), xt, X[:, 0])
    G0 = []
    for k in idx:                                      # gaps to the rod at each keyframe, from its first touch
        s = rod_gaps(X, Tr["c_P"][k], Tr["a_P"][k], R)
        G0.append(s - np.min(s[np.isfinite(s)]))
    G0 = np.array(G0)
    cand = np.where((G0 < a.gap).any(0))[0]
    W = model.compliance(nodes[cand], block=100)
    print(f"compliance of {len(cand)} face nodes in {time.time() - t0:.0f} s", flush=True)
    K = len(tk)
    sec_u = np.zeros((K, len(sec), 3), np.float32)
    mot = np.zeros((K, len(mn), 6))
    pres = np.zeros((K, len(cand)), np.float32)
    appr = np.zeros(K)
    Fk = Tr["f_P"][idx]
    for q in range(K):
        F = float(max(-Fk[q][0], 0.05))
        d, p, _ = T.force_controlled(W, G0[q, cand], F)
        act = np.where(p > 1e-12)[0]
        ids = nodes[cand[act]]
        Fv = (p[act] / p[act].sum())[:, None] * Fk[q][None]          # the HEX-E's force, spread as the pressure
        u = model.solve_full((ids, Fv))
        mot[q] = [np.r_[1e3 * tt, 1e3 * ww] for tt, ww in (TM.rigid_fit(model.X[i], u[i]) for i in mn)]
        sec_u[q] = 1e3 * u[sec]
        pres[q, act] = p[act]
        appr[q] = d
        if q % 20 == 0:
            print(f"  keyframe {q + 1}/{K} at {tk[q]:.1f} s: {F:.2f} N, approach {1e3 * d:.0f} um, {len(act)} contact "
                  f"nodes, magnets {np.round(np.linalg.norm(mot[q, :, :3], axis=1), 1)} um, {time.time() - t0:.0f} s",
                  flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(KEYS, t=tk, sec_X=model.X[sec], sec_u=sec_u, cls=cls_sec, xs=G.xs, zs=G.zs, y_sec=ys,
                        mc=np.array([model.X[i].mean(0) for i in mn]), mot=mot, cand_X=X[cand], pres=pres, approach=appr,
                        F=Fk, rod_c=Tr["c_P"][idx], rod_a=Tr["a_P"][idx], rod_R=R, truth_p=Tr["p"][idx],
                        n_dof=len(model.free), recipe=str(recipe))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    summary = dict(keyframes=K, dt_s=a.dt, n_dof=int(len(model.free)), contact_candidates=int(len(cand)),
                   approach_um=dict(median=float(np.median(1e3 * appr)), max=float(1e3 * appr.max())),
                   magnet_travel_um=dict(median=np.round(np.median(np.linalg.norm(mot[:, :, :3], axis=2), 0), 2).tolist(),
                                         max=np.round(np.linalg.norm(mot[:, :, :3], axis=2).max(0), 2).tolist()),
                   section_y_mm=float(ys), seconds=round(time.time() - t0), peak_rss_gb=round(peak, 2))
    json.dump(summary, open(DATA / f"{STEM}_fe_summary.json", "w"), indent=1)
    print(f"wrote {KEYS.relative_to(ROOT)}: {json.dumps(summary)}", flush=True)


MOD, MEAS = "#e8590c", "#111827"           # modelled, measured
POLE = {1: "#c0392b", -1: "#2563eb"}


def film_data(a):
    """Everything the frames draw, interpolated from the keyframes to the viewer film's frame times."""
    import fingertip_wrench_model as W
    Z = dict(np.load(KEYS))
    D = json.load(open(WORK / a.frames))
    rows = D["rows"]
    keep = [i for i, rw in enumerate(rows) if rw["t"] >= a.t_start]
    tt = np.array([rows[i]["t"] for i in keep])
    meas = np.array([D["msgs"]["physical"][i].get("db", [np.nan] * 15) for i in keep], float)
    tk = Z["t"]

    def lerp(arr):
        arr = np.asarray(arr, float)
        flat = arr.reshape(len(tk), -1)
        out = np.stack([np.interp(tt, tk, flat[:, j]) for j in range(flat.shape[1])], 1)
        return out.reshape((len(tt),) + arr.shape[1:])
    cal = W.Calib.load(DATA / f"{STEM.replace('_rod_roll', '')}_full_rest_calib.json")
    model = W.Model(cal)
    mot = lerp(Z["mot"])
    mod = np.array([model.motion_uT(m) for m in mot])                   # (n, 15) stream order, chip frames
    s_min = np.array([np.min(rod_gaps(Z["cand_X"], c, ax, float(Z["rod_R"]))) for c, ax in zip(Z["rod_c"], Z["rod_a"])])
    c_draw = Z["rod_c"] - ((s_min + Z["approach"])[:, None] * np.array([1.0, 0.0, 0.0])[None])
    T = np.load(DATA / f"{STEM}_truth.npz")
    tR = T["t"] - float(T["t_lo"])
    th = np.interp(tt, tR, T["theta"])
    return dict(Z=Z, tt=tt, meas=meas, mod=mod, mot=mot, sec_u=lerp(Z["sec_u"]), pres=lerp(Z["pres"]),
                appr=np.interp(tt, tk, Z["approach"]), F=lerp(Z["F"]), c_draw=lerp(c_draw), truth_p=lerp(Z["truth_p"]),
                model=model, cal=cal, theta=th, t_hold=float(T["plateaus"][0][0] - T["t_lo"]))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 15,
                         "mathtext.fontset": "dejavuserif"})
    return plt


def fe_section(ax, cax, S, j, fs=1.0, title=True):
    """(a) The section along the finger through the contact, the rod, the magnets' motion and the chips' change."""
    from matplotlib.patches import Circle, Rectangle
    import fingertip_wrench_model as W
    Z, model, cal = S["Z"], S["model"], S["cal"]
    xs, zs, cls = Z["xs"], Z["zs"], Z["cls"]
    X = Z["sec_X"]
    u = S["sec_u"][j]
    ix = np.searchsorted(xs, X[:, 0] - 1e-9)
    iz = np.searchsorted(zs, X[:, 2] - 1e-9)
    Un = np.full((len(xs), len(zs)), np.nan)
    Un[ix, iz] = np.linalg.norm(u, axis=1)
    Uc = 0.25 * (Un[:-1, :-1] + Un[1:, :-1] + Un[:-1, 1:] + Un[1:, 1:])
    Uc[cls < 0] = np.nan
    pc = ax.pcolormesh(zs, xs, np.ma.masked_invalid(Uc), cmap="magma_r", vmin=0, vmax=600.0, shading="flat",
                       rasterized=True)
    top = {}
    for (x, _, z), du in zip(X, u):
        if z not in top or x > top[z][0]:
            top[z] = (x, du)
    zz = np.array(sorted(top))
    xt = np.array([top[z][0] for z in zz])
    dut = np.array([top[z][1] for z in zz]) * 1e-3
    ax.plot(zz, xt, "-", color="0.6", lw=1.0)
    ax.plot(zz + dut[:, 2], xt + dut[:, 0], "-", color="black", lw=1.8)
    c = S["c_draw"][j]
    R = float(Z["rod_R"])
    ax.add_patch(Circle((c[2], c[0]), R, fc="#dfe3e8", ec="#4b5563", lw=1.6, alpha=0.95, zorder=3))
    ax.text(c[2], 13.2, "rod, R 12.5 mm", ha="center", fontsize=14 * fs, color="#374151", zorder=4)
    for jm, (cc, mm) in enumerate(zip(Z["mc"], S["mot"][j])):
        off = -0.45 if cc[1] < 0 else 0.45
        ax.add_patch(Rectangle((cc[2] - 1.5, cc[0] - 1.0), 3.0, 2.0, fc="none", ec=POLE[cal.poles[jm]], lw=1.3,
                               ls="-" if cc[1] < 0 else "--", zorder=4))
        ax.annotate("", xy=(cc[2] + off + 0.1 * mm[2], cc[0] + 0.1 * mm[0]), xytext=(cc[2] + off, cc[0]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color=POLE[cal.poles[jm]], shrinkA=0, shrinkB=0), zorder=5)
    P, Rc, uix = model.P, model.R, list(cal.perm)     # chips, stream order, tip frame
    for i in range(5):
        if abs(P[i, 1]) > 1.0:
            continue                                  # chips off this section: U3 and U5
        ax.add_patch(Rectangle((P[i, 2] - 1.5, P[i, 0] - 0.45), 3.0, 0.9, fc="#374151", ec="none", zorder=4))
        ax.text(P[i, 2], P[i, 0] - 1.25, W.U_ORDER[uix[i]], ha="center", va="top", fontsize=13 * fs)
        for vec, col in ((S["mod"][j], MOD), (S["meas"][j], MEAS)):
            B = Rc[i].T @ vec[3 * i:3 * i + 3]           # uT in the tip frame
            ax.annotate("", xy=(P[i, 2] + 0.08 * B[2], P[i, 0] + 0.08 * B[0]), xytext=(P[i, 2], P[i, 0]),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col, shrinkA=0, shrinkB=0), zorder=6)
    ax.set_xlim(-0.5, 22.0)
    ax.set_ylim(-8.6, 14.0)
    ax.set_aspect("equal")
    ax.set_xlabel("z, from the mount end toward the tip end (mm)", fontsize=15 * fs)
    ax.set_ylabel("x, toward the palm (mm)", fontsize=15 * fs)
    ax.tick_params(labelsize=13 * fs)
    if title:
        ax.set_title(f"(a) Section along the finger through the contact; the face sinks {1e3 * S['appr'][j]:.0f} µm",
                     fontsize=15, loc="left")
    cb = ax.figure.colorbar(pc, cax=cax)
    cb.set_label("Displacement (µm)", fontsize=15 * fs)
    cb.ax.tick_params(labelsize=13 * fs)


def fe_palm(a2, S, j, fs=1.0, title=True):
    """(b) From the palm: the contact's nodal forces, the magnets' motion and the chips' change."""
    from matplotlib.patches import Circle, Polygon, Rectangle
    from scipy.spatial import ConvexHull
    import fingertip_wrench_model as W
    Z, model, cal = S["Z"], S["model"], S["cal"]
    B0 = W.load_basis(W.basis_path("four_05R4"))
    yz = B0["pts"][:, 1:]
    a2.add_patch(Polygon(yz[ConvexHull(yz).vertices], closed=True, fc="#f7ecdb", ec="#8a6a3c", lw=1.2))
    cx, pr = Z["cand_X"], S["pres"][j]
    on = pr > 1e-4
    a2.scatter(cx[on, 1], cx[on, 2], c=pr[on], cmap="Greens", vmin=0, vmax=max(0.05, float(S["pres"].max()) * 0.6),
               s=10 * fs ** 2, marker="s", lw=0, zorder=2)
    for jm, (cc, mm) in enumerate(zip(Z["mc"], S["mot"][j])):
        a2.add_patch(Circle((cc[1], cc[2]), 1.5, fc=POLE[cal.poles[jm]], ec="black", lw=0.6, alpha=0.3, zorder=3))
        a2.annotate("", xy=(cc[1] + 0.1 * mm[1], cc[2] + 0.1 * mm[2]), xytext=(cc[1], cc[2]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color="black", shrinkA=0, shrinkB=0), zorder=5)
        a2.text(cc[1] + (1.8 if cc[1] > 0 else -1.8), cc[2], f"{mm[0]:+.0f}", fontsize=12 * fs,
                ha="left" if cc[1] > 0 else "right", va="center", color=POLE[cal.poles[jm]])
    P, Rc, uix = model.P, model.R, list(cal.perm)
    for i in range(5):
        a2.add_patch(Rectangle((P[i, 1] - 1.5, P[i, 2] - 1.5), 3.0, 3.0, fc="none", ec="#374151", lw=1.2, zorder=3))
        a2.text(P[i, 1], P[i, 2] + 2.0, W.U_ORDER[uix[i]], ha="center", va="bottom", fontsize=11 * fs, color="#374151")
        for vec, col in ((S["mod"][j], MOD), (S["meas"][j], MEAS)):
            B = Rc[i].T @ vec[3 * i:3 * i + 3]
            a2.annotate("", xy=(P[i, 1] + 0.06 * B[1], P[i, 2] + 0.06 * B[2]), xytext=(P[i, 1], P[i, 2]),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0, color=col, shrinkA=0, shrinkB=0), zorder=6)
    tp = S["truth_p"][j]
    a2.plot(tp[1], tp[2], "o", color="#00a33c", ms=9 * fs, mec="white", mew=1.2, zorder=7)
    a2.set_xlim(-11, 11)
    a2.set_ylim(-1.5, 23)
    a2.set_aspect("equal")
    a2.set_xlabel("y (mm)", fontsize=15 * fs)
    a2.set_ylabel("z (mm)", fontsize=15 * fs)
    a2.tick_params(labelsize=13 * fs)
    if title:
        a2.set_title("(b) Seen from the palm", fontsize=15, loc="left")


PALM_KEY = ("green: the contact's\nnodal forces, and the\nrobot's contact point\n\nmagnets: motion in\nthe face's plane 100x;\n"
            "numbers: motion\nalong x (µm)\n\nchips: field change\nin the face's plane,\n0.06 mm per µT")


def fe_bars(a3, S, j, fs=1.0, title=True):
    """(c) The 15 components of the board change, measured and modelled."""
    import fingertip_wrench_model as W
    uix = list(S["cal"].perm)
    order = [uix.index(u_) for u_ in range(5)]          # stream index of U1..U5
    lab, mo, me = [], [], []
    for u_, i in enumerate(order):
        for q, ch in enumerate("xyz"):
            lab.append(f"{W.U_ORDER[u_]}{ch}")
            mo.append(S["mod"][j][3 * i + q])
            me.append(S["meas"][j][3 * i + q])
    xb = np.arange(15)
    a3.bar(xb - 0.2, me, 0.4, color=MEAS, label="measured by the board")
    a3.bar(xb + 0.2, mo, 0.4, color=MOD, label="FE and field model")
    a3.axhline(0, color="k", lw=0.8)
    a3.set_xticks(xb)
    a3.set_xticklabels(lab, fontsize=10.5 * fs, rotation=90)
    a3.set_ylabel("Change (µT)", fontsize=15 * fs)
    a3.tick_params(axis="y", labelsize=13 * fs)
    lim = max(40.0, 1.1 * float(np.nanmax(np.abs(np.r_[S["mod"], S["meas"]]))))
    a3.set_ylim(-lim, lim)
    a3.spines[["top", "right"]].set_visible(False)
    a3.legend(loc="upper right", fontsize=12 * fs, frameon=False, ncol=1)
    me_, mo_ = np.array(me), np.array(mo)
    r = float(np.corrcoef(me_, mo_)[0, 1]) if np.all(np.isfinite(me_)) else np.nan
    txt = (f"correlation {r:.2f}, model/measured {np.linalg.norm(mo_) / max(np.linalg.norm(me_), 1e-9):.2f}")
    if title:
        a3.set_title(f"(c) Change at each chip axis: {txt}", fontsize=14.5, loc="left")
    return txt


def draw_frame(j, S, path):
    plt = _plt()
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    t = S["tt"][j]
    F = S["F"][j]
    fig.text(0.02, 0.965, "Inside the fingertip during the roll: finite elements and magnetic field", fontsize=26,
             weight="bold", va="top")
    fig.text(0.02, 0.918, "The rod's measured force on the FE model of pad 05R4 at the robot's contact, solved every 0.5 s. "
             "The field model turns the magnets' motion into\nthe change at the five chips (orange), against the change "
             "the board measured (black). v3 ID 5 on the UR5 rig, 4 November; 4x speed.",
             fontsize=15, va="top", color="#374151", linespacing=1.4)
    fig.text(0.98, 0.965, f"{t - S['t_hold']:5.1f} s   turn {S['theta'][j]:+.1f}°   {np.linalg.norm(F):.1f} N",
             fontsize=18, va="top", ha="right")
    fe_section(fig.add_axes([0.035, 0.115, 0.50, 0.69]), fig.add_axes([0.545, 0.2, 0.008, 0.45]), S, j)
    fig.text(0.035, 0.018, "In (a), the magnets' motion is drawn 100x (red north toward the palm, blue south; solid y < 0, "
             "dashed y > 0) and the chips' field change at 0.08 mm per µT.", fontsize=13, color="#374151")
    a2 = fig.add_axes([0.585, 0.40, 0.27, 0.42])
    fe_palm(a2, S, j)
    a2.text(1.04, 0.98, PALM_KEY, transform=a2.transAxes, fontsize=12.5, va="top", color="#374151", linespacing=1.3)
    fe_bars(fig.add_axes([0.615, 0.075, 0.37, 0.22]), S, j)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def draw_parts(j, S, paths):
    """The three panels of draw_frame as figures of their own, each stamped with the film time."""
    from PIL import Image
    import viewer_roll_film as V
    plt = _plt()
    st = f"{S['tt'][j] - S['t_hold']:5.1f} s   turn {S['theta'][j]:+.1f}°"

    def save(fig, path):
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        Image.fromarray(V.stamp(img, st)).save(path)
    fig = plt.figure(figsize=(15.0, 10.8), dpi=100)
    fe_section(fig.add_axes([0.07, 0.13, 0.80, 0.74]), fig.add_axes([0.885, 0.22, 0.013, 0.55]), S, j, fs=1.25,
               title=False)
    fig.text(0.07, 0.955, f"Finite elements: the section through the contact, the face {1e3 * S['appr'][j]:.0f} µm down",
             fontsize=23, va="top")
    fig.text(0.07, 0.045, "Magnets' motion drawn 100x (red north toward the palm, blue south; solid y < 0, dashed y > 0).\n"
             "Chips' field change at 0.08 mm per µT: model orange, board black.", fontsize=14, color="#374151",
             va="center", linespacing=1.4)
    save(fig, paths["fe_section"])
    fig = plt.figure(figsize=(12.0, 10.8), dpi=100)
    a2 = fig.add_axes([0.08, 0.08, 0.62, 0.80])
    fe_palm(a2, S, j, fs=1.3, title=False)
    fig.text(0.04, 0.955, "Magnets and chips, palm side", fontsize=23, va="top")
    fig.text(0.73, 0.85, PALM_KEY + "\n\norange: model\nblack: board", fontsize=16, va="top", color="#374151",
             linespacing=1.3)
    save(fig, paths["fe_palm"])
    fig = plt.figure(figsize=(16.0, 8.0), dpi=100)
    a3 = fig.add_axes([0.07, 0.17, 0.91, 0.66])
    txt = fe_bars(a3, S, j, fs=1.35, title=False)
    fig.text(0.07, 0.955, f"Change at each chip axis: {txt}", fontsize=23, va="top")
    save(fig, paths["fe_bars"])


def film(a):
    import multiprocessing as mp
    import subprocess
    from PIL import Image
    S = film_data(a)
    frames = WORK / "fe_film"
    frames.mkdir(parents=True, exist_ok=True)
    for f in frames.glob("f*.png"):
        f.unlink()
    idx = a.only if a.only else list(range(len(S["tt"])))
    global _S
    _S = S
    if a.parts:
        import viewer_roll_film as V
        names = ("fe_section", "fe_palm", "fe_bars")
        dirs = {n: WORK / "parts" / n for n in names}
        for dd in dirs.values():
            dd.mkdir(parents=True, exist_ok=True)
            for f in dd.glob("f*.png"):
                f.unlink()
        with mp.get_context("fork").Pool(a.jobs) as pool:
            pool.map(_draw_parts, [(j, {n: str(dirs[n] / f"f{j:05d}.png") for n in names}) for j in idx], chunksize=4)
        if a.only:
            return
        V.PARTS.mkdir(parents=True, exist_ok=True)
        for n, dd in dirs.items():
            dst = V.PARTS / f"{STEM}_{n}.mp4"
            V.encode(dd, dst)
            print(f"wrote {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
        return
    with mp.get_context("fork").Pool(a.jobs) as pool:
        pool.map(_draw, [(j, str(frames / f"f{j:05d}.png")) for j in idx], chunksize=4)
    if a.only:
        return
    last = len(S["tt"]) - 1
    for h in range(1, 38):
        (frames / f"f{last + h:05d}.png").write_bytes((frames / f"f{last:05d}.png").read_bytes())
    dst = MEDIA / f"{STEM}_fe_field_film.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", str(frames / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high", "-crf", "23", "-preset", "slow",
                    "-movflags", "+faststart", str(dst)], check=True)
    Image.open(frames / f"f{int(np.argmin(np.abs(S['tt'] - 31.3))):05d}.png").convert("RGB").save(
        dst.with_suffix(".jpg"), quality=90)
    corr = []
    for j in range(len(S["tt"])):
        me_, mo_ = S["meas"][j], S["mod"][j]
        if np.all(np.isfinite(me_)):
            corr.append((float(np.corrcoef(me_, mo_)[0, 1]), float(np.linalg.norm(mo_) / np.linalg.norm(me_)),
                         float(np.linalg.norm(me_)), float(np.linalg.norm(mo_))))
    corr = np.array(corr)
    s = json.load(open(DATA / f"{STEM}_fe_summary.json"))
    holds = {}
    for lab, tq in (("press, 0 deg", 14.25), ("+6 deg", 32.3), ("-6 deg", 66.25)):
        j = int(np.argmin(np.abs(S["tt"] - tq)))
        order = [list(S["cal"].perm).index(u_) for u_ in range(5)]
        holds[lab] = dict(t_window_s=float(S["tt"][j]),
                          measured=[round(float(S["meas"][j][3 * i + q]), 2) for i in order for q in range(3)],
                          modelled=[round(float(S["mod"][j][3 * i + q]), 2) for i in order for q in range(3)])
    s["holds_U_order"] = holds
    s["film"] = dict(frames=len(S["tt"]), correlation_median=float(np.median(corr[:, 0])),
                     correlation_range=[float(corr[:, 0].min()), float(corr[:, 0].max())],
                     size_ratio_median=float(np.median(corr[:, 1])), measured_norm_uT_median=float(np.median(corr[:, 2])),
                     modelled_norm_uT_median=float(np.median(corr[:, 3])))
    json.dump(s, open(DATA / f"{STEM}_fe_summary.json", "w"), indent=1)
    print(f"wrote {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e6:.1f} MB); {json.dumps(s['film'])}", flush=True)


_S = None


def _draw(args):
    j, path = args
    draw_frame(j, _S, path)


def _draw_parts(args):
    j, paths = args
    draw_parts(j, _S, paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    so = sub.add_parser("solve")
    so.add_argument("--dt", type=float, default=0.5, help="s between keyframes")
    so.add_argument("--gap", type=float, default=1.2, help="face nodes within this gap (mm) of the rod are candidates")
    fi = sub.add_parser("film")
    fi.add_argument("--frames", default="frames_step0.16_az_calib.json", help="the viewer film's frames (logs/)")
    fi.add_argument("--t-start", type=float, default=14.0)
    fi.add_argument("--jobs", type=int, default=4)
    fi.add_argument("--only", type=int, nargs="*", default=None)
    fi.add_argument("--parts", action="store_true", help="each panel as a film of its own, into media/20261110-synced_parts")
    a = ap.parse_args()
    dict(solve=solve, film=film)[a.cmd](a)


if __name__ == "__main__":
    main()

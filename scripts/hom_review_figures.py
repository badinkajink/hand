#!/usr/bin/env python3
"""Figures of the 2026-10-29 HOM review page (scripts/hom_review_page.py): small worked examples that explain the HOM
paper's equations (arXiv 2609.25619v2), and the chart of the fingertip-sensing variants (scripts/hom_tasks.py run
--variants rate50 notip0 notip tip tip3 tipbias tipfl tipflb).

  schematic  the pin-push as a hand-object mechanism: the pinch as a revolute joint, the middle finger as its actuator,
             the generalized contact frames
  planar     Eqs. (1)-(2) on a planar two-link finger beside a fixed cylinder: one relative-velocity command at a time
  brake      the pinch as a revolute joint with friction: the brake law against the simulators' measured angles
  filter     the complementary filter of Eqs. (4)-(6) on a contact point measured with the fingertips' noise
  sensing    held runs and tracking errors of the four tasks under the sensing variants

  .venv/bin/python scripts/hom_review_figures.py [schematic planar brake filter sensing]
Writes docs/experiments/20261029-hom_control_review/media/*.png.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261029-hom_control_review"
MEDIA = D / "media"
ROWS = D / "rows"

# CAD-model SR2 finger (scripts/hand_object_scale_kin.py, SCALE_HAND=cad_v1), mm
L1, L2, RHO = 30.80, 26.61, 10.55
ELL = 20.0                      # mm: the linear rows of Eq. (2) are divided by ell (src/morphohand/control/hom.py)
C_THUMB, C_INDEX, C_MIDDLE = "#1f4e9c", "#b03a2e", "#2e7d32"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "mathtext.fontset": "dejavuserif"})
    return plt


def save(fig, name):
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


# ------------------------------------------------------------------------------------------------ schematic
def fig_schematic():
    """Top and side views of the pin-push grasp of hom_tasks.py on a 25 mm cylinder (pinch 15 deg below the equator
    is drawn at the equator for clarity)."""
    plt = plt_setup()
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    r, L, yp, ym = 12.5, 100.0, 20.0, -32.5
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.0, 1.25]})
    # top view, palm frame (x across, y along the cylinder)
    a.add_patch(Rectangle((-r, -L / 2), 2 * r, L, fc="#d9d9d9", ec="k", lw=1.2))
    pads = {"thumb": (-(r + RHO), yp, C_THUMB), "index": (r + RHO, yp, C_INDEX), "middle": (r + RHO, ym, C_MIDDLE)}
    for f, (x, y, c) in pads.items():
        a.add_patch(Circle((x, y), RHO, fc="white", ec=c, lw=2.2))
        a.text(x, y - RHO - 2, f, ha="center", va="top", color=c)
        n = 1.0 if x < 0 else -1.0                       # the finger's GCF x axis points into the object
        cx = x + n * RHO
        a.annotate("", xy=(cx + n * 9, y), xytext=(cx, y), arrowprops=dict(arrowstyle="-|>", color="#c00000", lw=2))
        a.annotate("", xy=(cx, y + 9), xytext=(cx, y), arrowprops=dict(arrowstyle="-|>", color="#008000", lw=2))
    a.plot([-r - 26, r + 26], [yp, yp], ls="--", color="k", lw=1.2)
    a.text(-r - 27, yp - 3, "pinch\naxis", ha="right", va="top", fontsize=12)
    a.annotate("", xy=(r + RHO + 16, ym), xytext=(r + RHO + 16, yp),
               arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    a.text(r + RHO + 18, 0.5 * (yp + ym), "lever\n52.5 mm", fontsize=12, va="center")
    a.set_xlim(-62, 70)
    a.set_ylim(-60, 60)
    a.set_aspect("equal")
    a.set_xlabel("palm x (mm)")
    a.set_ylabel("palm y (mm)")
    a.set_title("(a) Top view: a pinch and a pusher")
    # side view along the pinch axis: the cylinder turns about the pinch point; the middle pad drives it by friction
    # s runs along the cylinder from the pinch point toward the middle finger (palm -y): the cylinder spans
    # s = yp - 50 ... yp + 50 mm and the middle finger's contact sits at s = yp - ym (the lever)
    for ang, alpha, ls in ((0.0, 1.0, "-"), (15.0, 0.55, "--"), (-15.0, 0.55, ":")):
        t = math.radians(ang)
        u = np.array([math.cos(t), math.sin(t)])
        nrm = np.array([-u[1], u[0]])
        p1, p2 = (yp - L / 2) * u, (yp + L / 2) * u
        corners = np.array([p1 + r * nrm, p2 + r * nrm, p2 - r * nrm, p1 - r * nrm, p1 + r * nrm])
        b.plot(corners[:, 0], corners[:, 1], color="k", alpha=alpha, ls=ls, lw=1.4)
        b.plot(*((yp - ym) * u), marker="o", ms=7, color=C_MIDDLE, alpha=alpha)
    b.add_patch(Circle((0, 0), 2.2, fc="k"))
    b.text(0, -24, "pinch point (thumb and index\nbehind and in front of it)", ha="center", va="top", fontsize=12)
    arc = FancyArrowPatch((52, -14), (52, 14), connectionstyle="arc3,rad=0.35", arrowstyle="<|-|>", lw=1.8,
                          color="k", mutation_scale=16)
    b.add_patch(arc)
    b.text(58, 0, r"$s$ (Eq. 3)", va="center", fontsize=13)
    b.annotate("middle pad drags\nthe cylinder round\nby friction", xy=(yp - ym, 0), xytext=(18, 26), fontsize=12,
               color=C_MIDDLE, arrowprops=dict(arrowstyle="-|>", color=C_MIDDLE))
    b.set_xlim(-45, 85)
    b.set_ylim(-46, 42)
    b.set_aspect("equal")
    b.set_xlabel("along the cylinder (mm)")
    b.set_ylabel("palm z (mm)")
    b.set_title(r"(b) Side view: 0 and $\pm$15$^\circ$ about the pinch axis")
    fig.tight_layout()
    save(fig, "20261029-pinpush_mechanism.png")


# ------------------------------------------------------------------------------------------------ planar Eq. (1)-(2)
def planar_fk(th):
    t1, t2 = th
    pip = L1 * np.array([math.sin(t1), -math.cos(t1)])
    pad = pip + L2 * np.array([math.sin(t1 + t2), -math.cos(t1 + t2)])
    return pip, pad


def planar_rows(th, C, R):
    """Eq. (1) for a planar finger beside a fixed disk: A (3 x 2) maps joint rates to the GCF's relative velocities
    (roll about the joint axis rad/s, separation mm/s, slide mm/s); also the gap (mm) and the GCF origin."""
    pip, pad = planar_fk(th)
    u = C - pad
    dist = float(np.linalg.norm(u))
    x1 = u / dist                                      # the finger's GCF normal, toward the object
    z1 = np.array([-x1[1], x1[0]])                     # tangent
    p1 = pad + RHO * x1                                # witness point on the pad
    # material velocity of the distal body at p1: joint 1 at the origin, joint 2 at the PIP; a positive joint rate
    # turns the finger counter-clockwise in the (x, z) plane of the figure: v = w (-r_z, r_x)
    def vel(r):
        return np.array([-r[1], r[0]])
    J = np.stack([vel(p1 - np.zeros(2)), vel(p1 - pip)], 1)        # (2, 2)
    A = np.vstack([np.ones(2), x1 @ J, z1 @ J])                    # roll, sep (closing > 0), slide
    return A, dist - RHO - R, p1, pad, pip


def planar_run(vref, th0, C, R, T=1.0, dt=0.01, w=(1.0, 1.0, 1.0), w_damp=1e-3, w_healthy=1e-5, aw=0.5):
    """Integrate Eq. (2) without joint limits (weighted damped least squares, healthy pose = start)."""
    th = np.array(th0, float)
    out = []
    W = np.diag(w)
    S = np.diag([1.0, 1.0 / ELL, 1.0 / ELL])          # linear rows divided by ell
    for k in range(int(round(T / dt))):
        A, gap, p1, pad, pip = planar_rows(th, C, R)
        v = np.array(vref, float)
        if gap < aw and v[1] > 0:
            v[1] = 0.0                                 # anti-windup
        As, vs = S @ A, S @ v
        H = As.T @ W @ As + (w_damp + w_healthy) * np.eye(2)
        g = As.T @ W @ vs + w_healthy * 1.0 * (np.asarray(th0) - th)
        u = np.linalg.solve(H, g)
        out.append({"t": k * dt, "th": th.copy(), "pad": pad, "pip": pip, "p1": p1, "gap": gap, "v_ach": A @ u,
                    "v_ref": v})
        th = th + dt * u
    return out


def fig_planar():
    plt = plt_setup()
    from matplotlib.patches import Circle
    R = 12.5
    th0 = np.radians([20.0, 40.0])
    _, pad0 = planar_fk(th0)
    al = math.radians(-20.0)
    C = pad0 + (RHO + R + 6.0) * np.array([math.cos(al), math.sin(al)])
    cases = [("separation +20 mm/s", (0.0, 20.0, 0.0)), ("slide +20 mm/s", (0.0, 0.0, 20.0)),
             ("roll +40 deg/s", (math.radians(40.0), 0.0, 0.0))]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
    stats = []
    for ax, (name, vref) in zip(axs, cases):
        run = planar_run(vref, th0, C, R, T=1.0)
        ax.add_patch(Circle(tuple(C), R, fc="#d9d9d9", ec="k", lw=1.2))
        n = len(run)
        for k in list(range(0, n, 20)) + [n - 1]:
            s = run[k]
            a_ = 0.25 + 0.75 * k / (n - 1)
            pts = np.array([[0, 0], s["pip"], s["pad"]])
            ax.plot(pts[:, 0], pts[:, 1], color="#555555", lw=2.5, alpha=a_, solid_capstyle="round")
            ax.add_patch(Circle(tuple(s["pad"]), RHO, fc="none", ec=C_INDEX, lw=1.6, alpha=a_))
            ax.plot(*s["p1"], "o", ms=3.5, color=C_INDEX, alpha=a_)
        tr = np.array([s["pad"] for s in run])
        ax.plot(tr[:, 0], tr[:, 1], color=C_INDEX, lw=1.2, ls="--")
        va = np.array([s["v_ach"] for s in run])
        mean = va.mean(0)
        stats.append((name, mean, run[-1]["gap"]))
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(-12, 82)
        ax.set_ylim(-78, 8)
        ax.set_xlabel("x (mm)")
        txt = (f"achieved, mean over 1 s:\nroll {math.degrees(mean[0]):+.1f} deg/s\n"
               f"separation {mean[1]:+.1f} mm/s\nslide {mean[2]:+.1f} mm/s\nend gap {run[-1]['gap']:.1f} mm")
        ax.text(-9, -76, txt, fontsize=11.5, va="bottom")
    axs[0].set_ylabel("z (mm)")
    fig.tight_layout()
    save(fig, "20261029-planar_eq1_eq2.png")
    return stats


# ------------------------------------------------------------------------------------------------ brake
def fig_brake():
    plt = plt_setup()
    m, g, d, mu = 0.0245, 9.81, 0.015, 1.0
    N = np.logspace(math.log10(0.1), math.log10(6.0), 400)
    rbar = 1.0e-3 * N ** 0.25
    c = 2 * mu * N * rbar / (m * g * d)
    phi = np.degrees(np.arccos(np.clip(c, 0, 1)))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(N, phi, color="k", lw=2, label=r"law: $\cos\varphi = 2\mu N\,\bar r(N)/(mgd)$")
    rig = [(1.20, 45, "Drake hydroelastic, rig"), (1.23, 45, "1 mm sphere pads, rig"), (1.20, 45, "condim 4, rig")]
    hand = [(1.26, 45), (1.28, 45), (1.33, 45), (0.40, 80), (0.41, 80), (0.45, 80)]
    ax.plot([x for x, _, _ in rig], [y for _, y, _ in rig], "s", ms=9, mfc="none", mec=C_THUMB, mew=2,
            label="two-pad rig (2026-10-01)")
    ax.plot([x for x, _ in hand], [y for _, y in hand], "o", ms=8, mfc="none", mec=C_INDEX, mew=2,
            label="SR2 hand, three contact models")
    ax.axvline(m * g / (2 * mu), color="#777777", ls=":", lw=1.5)
    ax.text(m * g / (2 * mu) * 1.06, 40, r"$N_{\min}=mg/2\mu$ = 0.12 N:" + "\nbelow it the tool\nslips out of the pinch",
            fontsize=12, color="#555555")
    ax.axhline(90, color=C_MIDDLE, ls="--", lw=1.5)
    ax.text(0.5, 92.5, "point contact: no torsional friction, the tool swings to hanging", color=C_MIDDLE, fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.set_xlabel(r"pinch force per pad, $N$ (N)")
    ax.set_ylabel(r"angle swung below horizontal, $\varphi$ (deg)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", fontsize=11.5, frameon=False, bbox_to_anchor=(0.02, 0.02))
    fig.tight_layout()
    save(fig, "20261029-pinch_brake_law.png")


# ------------------------------------------------------------------------------------------------ filter
def filt(meas, gref, dt, kmin=5.0, kmax=100.0, alpha=20.0):
    """Eqs. (4)-(6) in one dimension (mm along the surface): g += (gref + k (meas - g)) dt."""
    g = meas[0]
    out = []
    for z, gr in zip(meas, gref):
        k = kmin + (kmax - kmin) * (1.0 - math.exp(-3.0 * abs(gr) / alpha))
        g = g + (gr + k * (z - g)) * dt
        out.append(g)
    return np.array(out)


def fig_filter():
    plt = plt_setup()
    rng = np.random.default_rng(4)
    speed = math.radians(40.0) * RHO                      # contact point speed of a 40 deg/s roll on the pad, mm/s
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    res = {}
    for ax, (title, commanded) in zip(axs, (("(a) rolling commanded: the gain rises", True),
                                            ("(b) rolling not commanded: the gain stays at its minimum", False))):
        for dt, col, lab in ((0.01, C_THUMB, "100 Hz"), (0.02, C_INDEX, "50 Hz")):
            t = np.arange(0.0, 3.0, dt)
            truth = np.clip(t - 1.0, 0, 1.0) * speed           # rolls for 1 s from t = 1 s
            gref = np.where((t >= 1.0) & (t < 2.0), speed, 0.0) if commanded else np.zeros_like(t)
            sig = 0.4
            meas = truth + rng.normal(size=t.size) * sig
            est = filt(meas, gref, dt)
            if dt == 0.01:
                ax.plot(t, meas, ".", ms=3, color="#9a9a9a", label=r"measured, $\sigma$ 0.4 mm")
                ax.plot(t, truth, color="k", lw=2, label="true contact point")
            ax.plot(t, est, color=col, lw=1.8, label=f"filter at {lab}")
            res[(commanded, dt)] = float(np.sqrt(np.mean((est - truth) ** 2)))
        ax.set_title(title, fontsize=12.5)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0.5, 2.6)
    axs[0].set_ylabel("position along the pad (mm)")
    axs[0].legend(loc="upper left", frameon=False, fontsize=11.5)
    fig.tight_layout()
    save(fig, "20261029-contact_filter.png")
    # the steady-state error against measurement noise: the filter's variance at rest is k dt / (2 - k dt) sigma^2
    return res


# ------------------------------------------------------------------------------------------------ sensing variants
VAR_ORDER = ("rate50", "notip0", "notip", "tip", "tipbias", "tip3", "tipfl", "tipflb", "cal_notip", "cal_tip",
             "cal_tipfl")
VAR_LABEL = {"rate50": "exact", "notip0": "no tips,\nexact\npose", "notip": "no tips", "tip": "tips",
             "tipbias": "tips,\nbiased", "tip3": "tips,\n3x\nerrors", "tipfl": "tips,\nforce\nloop",
             "tipflb": "loop,\nbiased", "cal_notip": "cal.,\nno tips", "cal_tip": "cal.,\ntips",
             "cal_tipfl": "cal.,\ntips,\nloop"}


def sensing_rows(name="sensing"):
    """Rows of rows/<name>.jsonl: sensing (the CPU's 528 runs) or sensing_gpu (scripts/hom_gpu.py, 30 seeds), with the
    angles of rows scored before the integration fix rescaled (hom_rows.rescale_metrics)."""
    import hom_rows
    p = ROWS / f"{name}.jsonl"
    rows = {}
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[(r["task"], r["d_mm"], r["seed"], r["variant"])] = hom_rows.rescale_metrics(r)
    return list(rows.values())


def task_score(r):
    """The task's tracking error, as the results page reports it: pinch-angle RMSE (deg) for the cylinder tasks,
    angular-velocity RMSE / step for roll and twist."""
    if r["task"] in ("pinpush", "press"):
        return r.get("theta_rmse_deg")
    return r.get("w_rmse_ratio")


def fig_sensing():
    plt = plt_setup()
    name = "sensing_gpu" if (ROWS / "sensing_gpu.jsonl").exists() else "sensing"
    rows = [r for r in sensing_rows(name) if r.get("status") == "ok"]
    jit = np.random.default_rng(0)
    ref = {(r["task"], r["d_mm"], r["seed"]): r for r in rows if r["variant"] == "rate50"}
    panels = [("pinpush", "rmse", "pin-push: pinch-angle RMSE (deg)"), ("press", "rmse", "press: pinch-angle RMSE (deg)"),
              ("roll", "rmse", "roll: velocity RMSE per step"), ("twist", "rmse", "twist: velocity RMSE per step"),
              ("pinpush", "force", "pin-push: force error per pad (N)"), ("press", "force", "press: force error per pad (N)")]
    fig, axs = plt.subplots(3, 2, figsize=(17, 12.5), sharex=True)
    x = np.arange(len(VAR_ORDER))
    cap = {"roll": 2.5, "twist": 2.5}                    # velocity RMSE per step: a few calibration runs reach 12
    for ax, (task, kind, title) in zip(axs.flat, panels):
        for k, var in enumerate(VAR_ORDER):
            rr = [r for r in rows if r["task"] == task and r["variant"] == var]
            if not rr:
                continue
            if kind == "rmse":
                vals = [task_score(r) for r in rr if r.get("held") and task_score(r) is not None]
                held = sum(bool(r.get("held")) for r in rr)
                ax.text(k, 1.02, f"{held}/{len(rr)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
            else:
                vals = []
                for r in rr:
                    b = ref.get((task, r["d_mm"], r["seed"]))
                    if b is not None and r.get("F_mean") and b.get("F_mean"):
                        vals.append(float(np.mean([abs(r["F_mean"][f] - b["F_mean"][f]) for f in b["F_mean"]])))
            if vals:
                col = C_INDEX if var.startswith("cal") else C_THUMB
                many = len(vals) > 30
                xx = k + (jit.uniform(-0.28, 0.28, len(vals)) if many else np.zeros(len(vals)))
                top = cap.get(task) if kind == "rmse" else None
                if top is not None and max(vals) > top:
                    k_over = sum(v > top for v in vals)
                    ax.text(k, top * 0.985, f"{k_over} above", ha="center", va="top", fontsize=9.5, color="#555555")
                    keep = [v <= top for v in vals]
                    xx, vals_p = xx[keep], [v for v in vals if v <= top]
                else:
                    vals_p = vals
                ax.plot(xx, vals_p, "o", ms=3.5 if many else 6, mfc="none", mec=col, alpha=0.45 if many else 0.85)
                ax.plot([k - 0.34, k + 0.34], [np.median(vals)] * 2, color="k", lw=2.4)
        ax.set_title(title, pad=18 if kind == "rmse" else 6)
        ax.axvline(7.5, color="#888888", ls=":", lw=1.2)
        ax.grid(axis="y", color="#e2e2e2")
        ax.set_ylim(bottom=0)
    for ax in axs[2]:
        ax.set_xticks(x)
        ax.set_xticklabels([VAR_LABEL[v] for v in VAR_ORDER], fontsize=10)
    axs[0, 0].set_ylim(0, 26)
    axs[0, 1].set_ylim(0, 20)
    axs[1, 0].set_ylim(0, cap["roll"])
    axs[1, 1].set_ylim(0, cap["twist"])
    fig.tight_layout()
    save(fig, "20261029-sensing_variants.png")
    return {"rows": name, "n": len(rows)}


def main():
    which = sys.argv[1:] or ["schematic", "planar", "brake", "filter"]
    for w in which:
        out = {"schematic": fig_schematic, "planar": fig_planar, "brake": fig_brake, "filter": fig_filter,
               "sensing": fig_sensing}[w]()
        if out is not None:
            print(w, out)


if __name__ == "__main__":
    main()

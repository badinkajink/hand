#!/usr/bin/env python3
"""Walkthrough page of the HOM controller's code (src/morphohand/control/hom.py, surfaces.py; scripts/hom_tasks.py,
scripts/hom_gpu.py), a subpage of the HOM review page, 2026-10-30.

Owner, 2026-10-29: "fully explain how the code works, each component/substep, the physics/math intuition, combined
with examples with our own hand, matching the hom paper and our own new tasks"; "if it has to be yet another separate
page (a subpage linked out of from the main hom docpage), then so be it".

Reads docs/experiments/20261029-hom_control_review/rows/20261030-code_{twist,pinpush}.json and media/20261030-code_*
(scripts/hom_code_trace.py), rows/20261030-cpu_{twist_d40,pinpush_d25}.jsonl (hom_gpu.py cpu-check) and the GPU queue's
grid rows (docs/experiments/20261029-foundational_pose_review/rows/grid2d_v4b). Mathematics through scripts/texsvg.py.

    .venv/bin/python scripts/hom_code_page.py              # figures and page
    .venv/bin/python scripts/hom_code_page.py --numbers    # the numbers the prose quotes
"""
from __future__ import annotations

import base64
import html
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261029-hom_control_review"
MEDIA, ROWS = D / "media", D / "rows"
OUT = D / "20261030-hom_controller_code_walkthrough.html"
TEMPLATE = ROOT / "scripts/hom_review_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
GRID = ROOT / "docs/experiments/20261029-foundational_pose_review/rows/grid2d_v4b"
REVIEW = "20261029-hom_control_and_fingertip_sensing.html"
METHOD = "../20261021-hom_framework/20261021-hom_controller_method_and_implementation.html"
FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "MCP", "PIP")
COMP = ("spin", "roll y", "roll z", "sep", "slide y", "slide z")
FACE = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
ELL = (8.5, 7.4, 11.0, 22.0)                 # mm: the pad's depth, across, distal and proximal semi-axes
PAD_Z = (-10.55, 10.86)                      # mm: the pad's extent along the finger (fp_sr2_toy.PAD_Z)
R_APEX = 10.55                               # mm: apex from the finger's axis (hand_object_scale_kin.R, cad_v2)
T_TW, T_PP, T_SQ, T_AP = 2.10, 2.50, 1.41, 0.49   # s: the instants the tables quote
FIG, FILM, TAB = [0], [0], [0]


def esc(s):
    return html.escape(str(s), quote=True)


def fill(s, **kw):
    """Replace <<key>> tokens (the prose keeps LaTeX braces, so str.format is out)."""
    for k, v in kw.items():
        s = s.replace(f"<<{k}>>", str(v))
    left = re.findall(r"<<(\w+)>>", s)
    if left:
        raise KeyError(f"unfilled tokens {left}")
    return s


def data_uri(p, mime):
    return f"data:{mime};base64,{base64.b64encode(Path(p).read_bytes()).decode()}"


def img(path, caption, width=None):
    FIG[0] += 1
    w = f' style="max-width:{width}px"' if width else ""
    return (f'<figure><img src="{data_uri(path, "image/png")}" alt=""{w}>'
            f'<figcaption>Figure {FIG[0]}. {caption}</figcaption></figure>')


def film(path, caption, poster=None):
    FILM[0] += 1
    pa = f' poster="{data_uri(poster, "image/jpeg")}"' if poster and Path(poster).exists() else ""
    return (f'<figure><video controls muted loop playsinline preload="metadata"{pa} '
            f'src="{data_uri(path, "video/mp4")}"></video><figcaption>Film {FILM[0]}. {caption}</figcaption></figure>')


def table(head, body, caption, num_from=1):
    TAB[0] += 1
    h = "".join(f'<th class="{"n" if i >= num_from else ""}">{c}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"n" if i >= num_from else ""}">{c}</td>' for i, c in enumerate(r))
                + "</tr>" for r in body)
    return (f'<div class="tw"><table><caption>Table {TAB[0]}. {caption}</caption><thead><tr>{h}</tr></thead>'
            f'<tbody>{b}</tbody></table></div>')


def render(page_text, holder):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page_text)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    holder.append(len(items))
    return TEX_RE.sub(lambda m: next(svgs), page_text)


# ------------------------------------------------------------------------------------------------ data

def load(name):
    return json.loads((ROWS / f"20261030-code_{name}.json").read_text())


def at(tr, t):
    k = int(round(t / tr["hom"]["dt"]))
    p = tr["periods"][k]
    assert abs(p["t"] - t) < 1e-6, (p["t"], t)
    return p


def arr(x, shape=None):
    a = np.asarray(x, float)
    return a.reshape(shape) if shape else a


def series(tr, f, key, k=None):
    out = []
    for p in tr["periods"]:
        v = p["fingers"][f][key]
        out.append(np.nan if v is None else (v if k is None else v[k]))
    return np.asarray(out, float)


def eq2_enum(A, b, v_ref, w, theta, tr0, lo, hi, prm, healthy):
    """Eq. (2) for one finger by enumerating the 27 patterns of free and bound joint rates (hom_gpu.Batch.control)."""
    dt, ell = prm["dt"], prm["ell"]
    sw = np.sqrt(w) * np.array([1, 1, 1, 1 / ell, 1 / ell, 1 / ell])
    As, rs = sw[:, None] * A, sw * (v_ref - b)
    Hm = As.T @ As + (prm["w_damp"] + prm["w_healthy"]) * np.eye(3)
    g = As.T @ rs + prm["w_healthy"] * prm["k_healthy"] * (healthy - theta)
    lb = np.maximum(-prm["u_max"], (lo - tr0) / dt)
    ub = np.minimum(prm["u_max"], (hi - tr0) / dt)
    lb, ub = np.minimum(lb, ub - 1e-12), np.maximum(ub, lb + 1e-12)
    best, val = None, np.inf
    for pat in np.ndindex(3, 3, 3):
        free = [i for i in range(3) if pat[i] == 0]
        u = np.where(np.array(pat) == 1, lb, ub)
        if free:
            fixed = [i for i in range(3) if pat[i] != 0]
            rhs = g[free] - Hm[np.ix_(free, fixed)] @ u[fixed]
            u[free] = np.linalg.solve(Hm[np.ix_(free, free)], rhs)
        if np.all(u >= lb - 1e-9) and np.all(u <= ub + 1e-9):
            v = 0.5 * u @ Hm @ u - u @ g
            if v < val:
                best, val = u, v
    return np.clip(best, lb, ub)


def eq2_check(trs):
    """Eq. (2) re-solved from every logged period and finger with SciPy's BVLS (hom.solve_eq2, the CPU controller) and
    with the 27-pattern enumeration (the GPU controller): the largest difference of the joint rates, and the share of
    periods with a bound active."""
    sys.path.insert(0, str(ROOT / "src"))
    from morphohand.control import hom as H
    dmax, n, nb = 0.0, 0, 0
    for tr in trs:
        prm = tr["hom"]
        for p in tr["periods"]:
            for f in FINGERS:
                r = p["fingers"][f]
                A, b = arr(r["A"], (6, 3)), arr(r["b"])
                args = (arr(r["v_ref"]), arr(r["w"]), arr(r["theta"]), arr(r["tr0"]), arr(tr["limits"][f]["lo"]),
                        arr(tr["limits"][f]["hi"]))
                u1 = H.solve_eq2(A, b, args[0], args[1], args[2], args[3], args[4], args[5], prm["dt"],
                                 arr(tr["q_open"][f]), prm["w_damp"], prm["w_healthy"], prm["k_healthy"], prm["u_max"],
                                 scale=np.array([1.0, 1.0, 1.0] + [1.0 / prm["ell"]] * 3))
                u2 = eq2_enum(A, b, *args, prm, arr(tr["q_open"][f]))
                dmax = max(dmax, float(np.abs(u1 - u2).max()))
                lb = np.maximum(-prm["u_max"], (args[4] - args[3]) / prm["dt"])
                ub = np.minimum(prm["u_max"], (args[5] - args[3]) / prm["dt"])
                nb += int(np.any(u1 <= lb + 1e-9) or np.any(u1 >= ub - 1e-9))
                n += 1
    return dmax, n, nb


def cpu_gpu(task, d):
    """Held agreement and Spearman correlation of the task score between the CPU runner (rows/20261030-cpu_*) and the
    GPU queue's grid rows (seed 0), on the same layout and grasp; the GPU's ms per rollout from its timing lines."""
    import hom_landscape as L
    cpu = [json.loads(x) for x in (ROWS / f"20261030-cpu_{task}_d{d:g}.jsonl").read_text().splitlines() if x.strip()]
    key = lambda r: (tuple(round(v, 3) for v in r["d"]), r.get("grasp", 0), r["seed"])      # noqa: E731
    seeds = {}
    for x in (GRID / f"{task}_d{d:g}.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            if r.get("variant", "base") == "base":
                seeds.setdefault((tuple(round(v, 3) for v in r["d"]), r.get("grasp", 0)), {})[r["seed"]] = r
    gpu = {k + (0,): v[0] for k, v in seeds.items() if 0 in v}
    pairs = [(c, gpu[key(c)]) for c in cpu if key(c) in gpu]
    same = np.mean([bool(c["held"]) == bool(g["held"]) for c, g in pairs])
    sc = np.array([[L.rollout_score(task, c), L.rollout_score(task, g)] for c, g in pairs])
    from scipy.stats import spearmanr
    rho = float(spearmanr(sc[:, 0], sc[:, 1]).statistic)
    both = np.array([bool(c["held"]) and bool(g["held"]) for c, g in pairs])
    rho_both = float(spearmanr(sc[both, 0], sc[both, 1]).statistic)
    # the same grasps between the GPU's own seeds (seeds 1 and 2 start the object up to 1 mm and 2 deg off)
    grp = [seeds[key(c)[:2]] for c in cpu if all(s_ in seeds.get(key(c)[:2], {}) for s_ in (0, 1, 2))]
    s_same, s_rho = [], []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        s_same.append(np.mean([bool(g[a]["held"]) == bool(g[b]["held"]) for g in grp]))
        s_rho.append(spearmanr([L.rollout_score(task, g[a]) for g in grp], [L.rollout_score(task, g[b]) for g in grp]).statistic)
    f_mid = None
    if task == "pinpush":
        f_mid = (float(np.median([c["F_mean"]["middle"] for c, _ in pairs])),
                 float(np.median([g["F_mean"]["middle"] for _, g in pairs if g.get("F_mean")])))
    tl = [json.loads(x) for x in (GRID / f"{task}_d{d:g}_timing.jsonl").read_text().splitlines() if x.strip()]
    nw = sum(t["worlds"] for t in tl)
    ms = 1e3 * sum(t["run_s"] for t in tl) / nw
    ms_all = 1e3 * sum(t["build_s"] + t["run_s"] + t["score_s"] for t in tl) / nw
    cpu_s = float(np.median([c["wall_s"] for c, _ in pairs]))
    return {"n": len(pairs), "same": float(same), "rho": rho, "gpu_ms": ms, "gpu_ms_all": ms_all, "cpu_s": cpu_s,
            "batch": tl[0]["worlds"], "gpu": tl[0]["gpu"], "rho_both": rho_both, "n_both": int(both.sum()),
            "seed_same": (float(min(s_same)), float(max(s_same))), "seed_rho": (float(min(s_rho)), float(max(s_rho))),
            "f_mid": f_mid,
            "held_cpu": float(np.mean([bool(c["held"]) for c, _ in pairs])),
            "held_gpu": float(np.mean([bool(g["held"]) for _, g in pairs])), "sc": sc}


def port_check(task, d):
    """The controller captured as a CUDA graph (hom_gpu.Batch since 2026-10-30, 08:50) against the queue's eager one on
    the same worlds: the grid's validation set on seeds 3 to 5 run again into rows/grid2d_v4b/port_check. The share
    with the same held outcome, the rank correlation of the per-grasp mean scores and the ms per rollout of each."""
    import hom_landscape as L
    import hom_rows
    from scipy.stats import spearmanr

    def load(q):
        out = {}
        for x in (q / f"val_{task}_d{d:g}.jsonl").read_text().splitlines():
            r = hom_rows.rescale_metrics(json.loads(x)) if x.strip() else None
            if r and r.get("status", "ok") == "ok":
                out[(r["call"], r.get("grasp", 0), r["seed"])] = (bool(r.get("held")), L.rollout_score(task, r))
        return out
    a, b = load(GRID), load(GRID / "port_check")
    k = sorted(set(a) & set(b))
    g = {}
    for x in k:
        g.setdefault(x[:2], []).append((a[x][1], b[x][1]))
    m = np.array([np.mean(v, 0) for v in g.values()])
    ms = []
    for q in (GRID, GRID / "port_check"):
        tl = [json.loads(x) for x in (q / f"val_{task}_d{d:g}_timing.jsonl").read_text().splitlines() if x.strip()]
        ms.append(1e3 * sum(t["run_s"] for t in tl) / sum(t["worlds"] for t in tl))
    return {"n": len(k), "same": float(np.mean([a[x][0] == b[x][0] for x in k])),
            "rho": float(spearmanr(m[:, 0], m[:, 1]).statistic), "ms": ms}


# ------------------------------------------------------------------------------------------------ figures

def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 11,
                         "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10})
    return plt


FC = {"thumb": "#1f77b4", "index": "#d62728", "middle": "#2ca02c"}


def phase_bands(ax, tr, label=False):
    ph = [p["phase"] for p in tr["periods"]]
    t = [p["t"] for p in tr["periods"]]
    i0 = 0
    shade = False
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[i0]:
            t1 = t[i] if i < len(ph) else t[-1] + tr["hom"]["dt"]
            if shade:
                ax.axvspan(t[i0], t1, color="0.92", lw=0, zorder=0)
            if label and t1 - t[i0] >= 0.25:
                ax.text(0.5 * (t[i0] + t1), 1.02, ph[i0], transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=9, rotation=0)
            shade = not shade
            i0 = i


def support_tilt(margin=1.5):
    """Normal tilts (deg, toward the PIP and toward the tip) at which the pad's support point reaches its end, less
    the margin of fp_sr2_toy's pad-coverage constraint."""
    ax_, _ay, az, azp = ELL
    out = []
    for a_z, z_end in ((azp, PAD_Z[1] - margin), (az, -PAD_Z[0] - margin)):
        th = np.radians(np.linspace(0, 89, 8901))
        z = a_z ** 2 * np.sin(th) / np.sqrt(ax_ ** 2 * np.cos(th) ** 2 + a_z ** 2 * np.sin(th) ** 2)
        out.append(float(np.degrees(th[np.argmax(z >= z_end)])))
    return out


def fig_pad(N, out):
    """(a) The pad's section in the plane of the finger (y = 0 of the tip frame) with the closest-point construction for
    the sphere's centre and the tilt limits of the support point; (b) the pad seen from the object, with the contact
    points of the twist."""
    plt = plt_setup()
    tw = N["tw"]
    ax_, ay, az, azp = ELL
    c0 = R_APEX - ax_
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.2, 4.4), gridspec_kw={"width_ratios": [1.55, 1]})
    ph = np.linspace(0, 2 * np.pi, 721)
    x = c0 + ax_ * np.cos(ph)
    z = np.where(np.sin(ph) < 0, az, azp) * np.sin(ph)
    ax.plot(z, x, color="0.6", ls="--", lw=1.2, label="the pad's ellipsoid")
    pad = (x >= c0) & (z >= PAD_Z[0]) & (z <= PAD_Z[1])
    ax.plot(np.where(pad, z, np.nan), np.where(pad, x, np.nan), color="#e6731a", lw=4, solid_capstyle="butt",
            label="printed pad")
    p = at(tw, T_SQ)["fingers"]["thumb"]
    o = np.array(p["obj_tip_mm"])
    a1 = np.array(p["p1_tip_mm"])
    ax.plot([o[2], a1[2]], [o[0], a1[0]], color="k", lw=1)
    ax.scatter([o[2]], [o[0]], marker="+", s=80, color="k", zorder=6)
    ax.scatter([a1[2]], [a1[0]], s=40, color=FC["thumb"], edgecolor="k", zorder=6)
    ax.annotate("sphere centre", (o[2], o[0]), xytext=(6, -2), textcoords="offset points", fontsize=10, va="top")
    th = np.radians(np.linspace(-90, 90, 361))
    ax.plot(o[2] + 20 * np.sin(th), o[0] - 20 * np.cos(th), color="0.35", lw=1, label="40 mm sphere")
    for tilt, sgn in zip(support_tilt(), (1.0, -1.0)):
        u = np.array([math.cos(math.radians(tilt)), sgn * math.sin(math.radians(tilt))])     # (x, z) of the normal
        a = np.array([ax_, azp if sgn > 0 else az])
        q = c0 * np.array([1, 0]) + a * a * u / np.linalg.norm(a * u)
        ax.annotate("", (q[1] + 6 * u[1], q[0] + 6 * u[0]), (q[1], q[0]),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
        ax.text(q[1] + 6.5 * u[1], q[0] + 6.5 * u[0], f"{tilt:.0f} deg", fontsize=10,
                ha="left" if sgn > 0 else "right", va="bottom")
    ax.set_aspect("equal")
    ax.set_xlim(-24, 26)
    ax.set_ylim(-8, 32)
    ax.set_xlabel("z (mm): tip < 0 < PIP")
    ax.set_ylabel("x, palmar (mm)")
    ax.set_title("(a) section through the thumb's pad, 1.41 s", fontsize=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    zz = np.linspace(PAD_Z[0], PAD_Z[1], 400)
    yy = ay * np.sqrt(np.clip(1 - (zz / np.where(zz < 0, az, azp)) ** 2, 0, 1))
    bx.fill_betweenx(zz, -yy, yy, color="#e6731a", alpha=0.25, lw=0)
    bx.plot(yy, zz, color="#e6731a", lw=2)
    bx.plot(-yy, zz, color="#e6731a", lw=2)
    bx.plot([-yy[0], yy[0]], [zz[0], zz[0]], color="#e6731a", lw=2)
    bx.plot([-yy[-1], yy[-1]], [zz[-1], zz[-1]], color="#e6731a", lw=2)
    for f in FINGERS:
        pts = np.array([q["fingers"][f]["cop_tip_mm"] for q in tw["periods"]
                        if q["t"] >= 2.0 and q["fingers"][f]["cop_tip_mm"] is not None])
        bx.scatter(pts[:, 1], pts[:, 2], s=4, color=FC[f], alpha=0.3, lw=0)
        m = N["cop_tip"][f]
        bx.scatter([m[1]], [m[2]], s=60, color=FC[f], edgecolor="k", zorder=5, label=f)
    bx.set_aspect("equal")
    bx.set_xlim(-9, 9)
    bx.set_ylim(-12, 12)
    bx.set_xlabel("y, across (mm)")
    bx.set_ylabel("z (mm)")
    bx.set_title("(b) contacts on the pads, twist", fontsize=10)
    bx.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def fig_twist(N, out):
    plt = plt_setup()
    tw = N["tw"]
    t = np.array([p["t"] for p in tw["periods"]])
    fig, axs = plt.subplots(5, 1, figsize=(8.2, 11.5), sharex=True)
    for f in FINGERS:
        d = series(tw, f, "dist_mm")
        axs[0].plot(t, d, color=FC[f], lw=1.3, label=f)
        cop = np.array([p["fingers"][f]["src"] == "cop" for p in tw["periods"]])
        axs[0].plot(t[cop], np.full(cop.sum(), -1.4 - 0.25 * FINGERS.index(f)), "|", color=FC[f], ms=4)
        axs[1].plot(t, series(tw, f, "F"), color=FC[f], lw=1.2, label=f)
    axs[0].axhline(0.5, color="k", lw=0.8, ls=":")
    axs[0].text(t[-1], 0.6, "anti-windup 0.5 mm", ha="right", va="bottom", fontsize=9)
    axs[0].set_ylim(-2.3, 9)
    axs[0].set_ylabel("GCF distance (mm)")
    axs[0].legend(loc="upper right", frameon=False, ncol=3)
    fl = np.array([sum(p["F_link"].values()) for p in tw["periods"]])
    axs[1].plot(t, fl, color="0.4", lw=1.0, label="links (all fingers)")
    axs[1].axhline(tw["prm"]["f_tripod"], color="k", lw=0.8, ls="--")
    axs[1].set_ylabel("normal force (N)")
    axs[1].legend(loc="upper left", frameon=False, ncol=4)
    wz = np.array([p["w_dps"][2] for p in tw["periods"]])
    wr = np.array([(p.get("w_ref_dps") or [0, 0, 0])[2] for p in tw["periods"]])
    axs[2].plot(t, wz, color="k", lw=1.2, label="measured")
    axs[2].plot(t, wr, color="#e6731a", lw=1.5, ls="--", label="reference")
    axs[2].set_ylabel("sphere $\\omega_z$ (deg/s)")
    axs[2].legend(loc="lower left", frameon=False, ncol=2)
    f = "index"
    for k, j in enumerate(JOINTS):
        c = ("#444444", "#d62728", "#9467bd")[k]
        axs[3].plot(t, np.degrees(series(tw, f, "theta", k)), color=c, lw=1.3, label=f"{j} measured")
        axs[3].plot(t, np.degrees(series(tw, f, "tr", k)), color=c, lw=1.0, ls="--", label=f"{j} reference")
    axs[3].set_ylabel("index joints (deg)")
    axs[3].legend(loc="upper left", frameon=False, ncol=3, fontsize=9)
    for key, c, lab in (("off_gap", "#1f77b4", "gap step"), ("off_force", "#d62728", "force"),
                        ("off_grav", "#2ca02c", "gravity")):
        v = np.array([np.linalg.norm(np.degrees(p["fingers"][f][key])) for p in tw["periods"]])
        axs[4].plot(t, v, color=c, lw=1.3, label=lab)
    axs[4].set_ylabel("index offsets |.| (deg)")
    axs[4].set_xlabel("time (s)")
    axs[4].legend(loc="upper left", frameon=False, ncol=3)
    for k, a in enumerate(axs):
        phase_bands(a, tw, label=(k == 0))
        a.set_xlim(0, t[-1] + 0.01)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def fig_pinpush(N, out):
    plt = plt_setup()
    pp = N["pp"]
    t = np.array([p["t"] for p in pp["periods"]])
    fig, axs = plt.subplots(4, 1, figsize=(8.2, 9.6), sharex=True)
    psi = pp["pose_v"][3]
    Rz = np.array([[math.cos(psi), math.sin(psi), 0], [-math.sin(psi), math.cos(psi), 0], [0, 0, 1.0]])
    ax_ = np.array([Rz @ np.asarray(p["axis"]) for p in pp["periods"]])
    th = np.degrees(np.arctan2(ax_[:, 2], -ax_[:, 1]))
    th = th - th[int(np.argmax(t >= 2.0 - 1e-9))]                 # from the task's start, as hom_tasks.metrics
    ref = np.array([p.get("theta_ref_deg", np.nan) for p in pp["periods"]])
    axs[0].plot(t, th, color="k", lw=1.3, label="measured")
    axs[0].plot(t, ref, color="#e6731a", lw=1.5, ls="--", label="commanded")
    axs[0].set_ylabel("angle about\npinch axis (deg)")
    axs[0].legend(loc="lower left", frameon=False, ncol=2)
    for f in FINGERS:
        axs[1].plot(t, series(pp, f, "F"), color=FC[f], lw=1.3, label=f"{f} measured")
        axs[1].plot(t, [p["fingers"][f]["force"] for p in pp["periods"]], color=FC[f], lw=1.0, ls="--")
    axs[1].set_ylabel("normal force (N)")
    axs[1].legend(loc="center right", bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=3, fontsize=9)
    axs[1].text(0.99, 0.04, "dashed: commanded", transform=axs[1].transAxes, ha="right", fontsize=9)
    f = "middle"
    axs[2].plot(t, series(pp, f, "gap_mm"), color="#1f77b4", lw=1.3, label="gap closed by the target")
    axs[2].plot(t, series(pp, f, "dist_mm"), color="k", lw=1.0, label="GCF distance")
    axs[2].axhline(pp["hom"]["gap_max"] * 1e3, color="k", lw=0.8, ls=":")
    axs[2].set_ylabel("middle (mm)")
    axs[2].legend(loc="center left", frameon=False)
    for key, c, lab, ls in (("theta", "k", "measured", "-"), ("tr", "#1f77b4", "reference", "--"),
                            ("q_cmd", "#d62728", "servo target", "-")):
        axs[3].plot(t, np.degrees(series(pp, f, key, 1)), color=c, lw=1.2, ls=ls, label=lab)
    axs[3].set_ylabel("middle MCP (deg)")
    axs[3].set_xlabel("time (s)")
    axs[3].legend(loc="upper left", frameon=False, ncol=3)
    for k, a in enumerate(axs):
        phase_bands(a, pp, label=(k == 0))
        a.set_xlim(0, t[-1] + 0.01)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def fig_cpu_gpu(C, out):
    plt = plt_setup()
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.9))
    for a, (task, lab) in zip(axs, (("twist", "Twist, 40 mm sphere"), ("pinpush", "Pin-push, 25 mm cylinder"))):
        sc = C[task]["sc"]
        a.scatter(sc[:, 0], sc[:, 1], s=22, color="k", alpha=0.7, lw=0)
        a.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls="--")
        a.set_xlim(-0.03, 1.03)
        a.set_ylim(-0.03, 1.03)
        a.set_aspect("equal")
        a.set_title(f"{lab}\nn = {C[task]['n']}, Spearman {C[task]['rho']:.2f}")
        a.set_xlabel("score, CPU controller")
        a.set_ylabel("score, GPU controller")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def numbers():
    tw, pp = load("twist"), load("pinpush")
    N = {"tw": tw, "pp": pp}
    N["wall_ms"] = float(np.median([p["wall_ms"] for p in tw["periods"] + pp["periods"]]))
    # contact positions on the pad, tip frame, during the twist's squeeze (1.0-1.6 s)
    cz = {}
    for f in FINGERS:
        pts = np.array([p["fingers"][f]["cop_tip_mm"] for p in tw["periods"]
                        if 1.0 <= p["t"] < 1.6 and p["fingers"][f]["cop_tip_mm"] is not None])
        cz[f] = np.median(pts, 0)
    N["cop_tip"] = cz
    return N


# ------------------------------------------------------------------------------------------------ page

def deg(x, k=1):
    return ", ".join(f"{v:.{k}f}" for v in np.degrees(np.asarray(x, float)))


def mm(x, k=1):
    return ", ".join(f"{v:.{k}f}" for v in 1e3 * np.asarray(x, float))


def rel_twist(v, k=1):
    """A relative twist as (deg/s x3, mm/s x3) cells."""
    v = np.asarray(v, float)
    return [f"{a:.{k}f}" for a in np.r_[np.degrees(v[:3]), 1e3 * v[3:]]]


def sec_terms():
    terms = [
        ("HOM", "the hand-object mechanism framework of [1]: the fingers and the object are treated as one mechanism, "
                "and each finger is commanded by its motion relative to the object."),
        ("Layout", r"the palm-frame positions of the three finger mounts. The symmetric tripod puts the thumb at "
                   r"$(-x_{\mathrm{sep}}/2, 0)$ and the index and middle at $(x_{\mathrm{sep}}/2, \pm y_{\mathrm{sep}}/2)$. "
                   r"The example has $x_{\mathrm{sep}}$ = 95 mm and $y_{\mathrm{sep}}$ = 75 mm, cell 280 of the 2-D grid."),
        ("Foundational pose (FP)", "the object position and the three contacts that <code>scripts/fp_sr2_toy.py</code> "
                                   "solves for a layout, with the largest margin to every constraint; the examples start "
                                   "from it."),
        ("v4b pad", "the printed fingertip pad, half an ellipsoid. Its semi-axes are 8.5 mm deep, 7.4 mm across, 11 mm "
                    "toward the tip and 22 mm toward the PIP joint, and its apex lies 10.55 mm from the finger's axis."),
        (r"$\theta$, $\theta_{\mathrm{ref}}$, $u$", "a finger's measured joint angles (yaw, MCP, PIP), its integrated "
                                                    "joint reference, and the joint rates (rad/s) that Eq. (2) returns."),
        (r"GCF $\{L_1\}$, $\{L_0\}$", r"the generalized contact frames, with origins $p_1$ on the finger and $p_0$ on "
                                      r"the object. The contact normal $x_1$ points from the finger toward the object. "
                                      r"$y_1$ is the PIP axis projected onto the tangent plane, and $z_1 = x_1 \times "
                                      r"y_1$. $\{L_0\}$ has $x_0 = -x_1$ and $y_0 = y_1$."),
        ("Twist", r"$V = (\omega, v)$, an angular velocity followed by a linear one. An object twist is a body twist in "
                  r"the object frame $\{B_0\}$."),
        ("Relative contact twist", r"the twist of $\{L_1\}$ relative to the object at $p_1$, in $\{L_1\}$'s axes, with "
                                   r"components (spin, roll $y$, roll $z$, sep, slide $y$, slide $z$). Spin turns the "
                                   r"finger about the normal; the rolls turn it about the tangent axes; sep > 0 closes "
                                   r"the gap; the slides move the contact along the surface."),
        (r"$A$, $b$", r"the rows of Eq. (1): the relative contact twist is $A u + b$."),
        (r"$v_{\mathrm{ref}}$, $w_i$", r"the commanded relative contact twist and the weight of component $i$ in Eq. (2); "
                                       r"$w_i = 0$ leaves the component free."),
        (r"$\ell$", r"20 mm: the linear rows of Eq. (2) are divided by $\ell$, so 20 mm/s counts as 1 rad/s."),
        (r"$F$, $k_p$, $J_p$, $J_r$", r"a finger's commanded normal force (N); the servo gain, 4 N m/rad; the linear and "
                                      r"angular Jacobians of a point of the finger over its three joints."),
        ("Witness points; centre of pressure (CoP)", "the closest points of the finger's and the object's surfaces; the "
                                                     "force-weighted mean of a finger's pad contacts on the object."),
        (r"$\{M\}$, $s$", r"the pinch frame (origin midway between the pinch contacts on the object, $x$ from the "
                          r"thumb's contact to the index's) and the commanded angular speed about its $x$ axis, Eq. (3)."),
        (r"$\{C_0\}$, $\{C_1\}$, $k$", r"the filter's contact frames on the object and on the finger, and its gain (1/s), "
                                       r"Eqs. (4) to (6)."),
        ("Held", "a rollout ends with the object's centre less than 15 mm below its start and at least two fingers "
                 "each pressing with half the object's weight (0.12 N) or more."),
        ("Twist score", "the largest twist angle reached in the ramp while all three pads touch, divided by 90 deg; 0 "
                        "when the sphere is dropped."),
        ("Pin-push score", r"$1 - \mathrm{RMSE}/15^\circ$, with the RMSE of the angle about the pinch axis against "
                           r"the command; 0 when dropped. A stroke's gain is its achieved angle over its commanded one."),
        ("Rollout", "one simulated run of a task from the open pose: 7.8 s for the twist, 6.5 s for the pin-push."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms) + "</dl>"


def build():
    N = numbers()
    tw, pp = N["tw"], N["pp"]
    MEDIA.mkdir(parents=True, exist_ok=True)
    figs = {k: MEDIA / f"20261030-code_{k}.png" for k in ("pad", "twist_trace", "pinpush_trace", "cpu_gpu")}
    fig_pad(N, figs["pad"])
    fig_twist(N, figs["twist_trace"])
    fig_pinpush(N, figs["pinpush_trace"])
    C = {t: cpu_gpu(t, d) for t, d in (("twist", 40), ("pinpush", 25))}
    fig_cpu_gpu(C, figs["cpu_gpu"])
    dmax, n2, nb2 = eq2_check([tw, pp])
    res_tw, res_pp = tw["result"], pp["result"]
    p_tw, p_sq, p_ap, p_pp = at(tw, T_TW), at(tw, T_SQ), at(tw, T_AP), at(pp, T_PP)
    ix = p_tw["fingers"]["index"]
    step1 = [q for q in tw["periods"] if q["phase"] == "wz +"]
    slide1 = max(1e3 * math.hypot(q["fingers"][f]["v_plan"][4], q["fingers"][f]["v_plan"][5])
                 for q in step1 for f in FINGERS)
    spin1 = float(np.median([abs(math.degrees(q["fingers"][f]["v_plan"][0])) for q in step1 for f in FINGERS]))
    surf1 = float(np.median([1e3 * math.hypot(q["fingers"][f]["b"][4], q["fingers"][f]["b"][5])
                             for q in step1 for f in FINGERS]))
    stop = {}
    for f in FINGERS:
        for q in tw["periods"]:
            if q["phase"] == "close" and q["t"] > 0.2 and q["fingers"][f]["v_ref"][3] == 0.0:
                stop[f] = (q["t"], q["fingers"][f]["dist_mm"])
                break
    lead = {k: max(float(np.abs(np.degrees(arr(q["fingers"][f]["tr"]) - arr(q["fingers"][f]["theta"]))).max())
                   for q in tr["periods"] for f in FINGERS) for k, tr in (("tw", tw), ("pp", pp))}
    ramp = [q for q in tw["periods"] if q["phase"] == "twist ramp"]
    a1 = np.array([q["fingers"]["index"]["p1_tip_mm"] for q in ramp])
    a2 = np.array([q["fingers"]["index"]["act_tip_mm"] for q in ramp])
    jit_m, jit_f = (float(np.sqrt(np.mean(np.sum(np.diff(a, axis=0) ** 2, 1)))) for a in (a1, a2))
    tilt_p, tilt_d = support_tilt()
    nper = len(tw["periods"]) + len(pp["periods"])
    # the pinch frame and the pusher's lever in the pin-push: {M}'s x runs between the pinch contacts on the object
    mid = p_pp["fingers"]["middle"]
    Rm, pm = arr(p_pp["M"]["R"], (3, 3)), arr(p_pp["M"]["p"])
    pa, pb = arr(p_pp["fingers"]["thumb"]["p0"]), arr(p_pp["fingers"]["index"]["p0"])
    xm = (pb - pa) / np.linalg.norm(pb - pa)
    rr = arr(mid["p0"]) - 0.5 * (pa + pb)
    lever = float(np.linalg.norm(rr - (rr @ xm) * xm))
    v_surf = abs(p_pp["s"]) * lever
    A_m = arr(mid["A"], (6, 3))
    v_mid = float(np.linalg.norm(arr(mid["R1"], (3, 3)) @ (A_m[3:] @ arr(mid["u"]))))
    task_pp = [q for q in pp["periods"] if q["t"] >= 2.0]
    capped = float(np.mean([q["fingers"]["middle"]["gap_mm"] >= pp["hom"]["gap_max"] * 1e3 - 1e-6 for q in task_pp]))
    f_mid = float(np.median([q["fingers"]["middle"]["F"] for q in task_pp]))
    d_mcp = float(np.median([math.degrees(q["fingers"]["middle"]["q_cmd"][1] - q["fingers"]["middle"]["theta"][1])
                             for q in task_pp]))
    yaw = [math.degrees(math.atan2(q["axis"][0], -q["axis"][1])) for q in (at(pp, 0.79), at(pp, 1.41))]
    psi = pp["pose_v"][3]
    Rz = np.array([[math.cos(psi), math.sin(psi), 0], [-math.sin(psi), math.cos(psi), 0], [0, 0, 1.0]])
    pin = [math.degrees(math.atan2((Rz @ arr(q["axis"]))[2], -(Rz @ arr(q["axis"]))[1])) for q in (at(pp, 0.79), at(pp, 2.0))]
    sq_turn = pin[1] - pin[0]
    mcp = [math.degrees(at(pp, t)["fingers"]["middle"]["theta"][1]) for t in (0.79, 1.41)]
    mdl = tw["model"]
    st = {s_["step"]: s_ for s_ in res_tw["steps"]}
    sk = {s_["stroke"]: s_ for s_ in res_pp["strokes"]}

    # ---------------------------------------------------------------- 2. the period
    def ex(f, key, k=1, conv=deg):
        return conv(p_tw["fingers"][f][key], k)
    rows = [
        ["1. Measurement", "Sec. III-E (inputs)", "<code>Measurement</code>, filled by <code>hom_tasks.rollout</code> "
         "from <code>Plant.state</code>", f"$\\theta$ = ({ex('index', 'theta')}) deg; pad force {ix['F']:.2f} N"],
        ["2. Kinematic copy", "", "<code>Mirror.set_state</code>: joint angles and object pose, then "
         "<code>mj_kinematics</code>", "tip pose and joint axes at the measured state"],
        ["3. Contact frame", "Sec. III-B", "<code>HomController.gcf</code> &rarr; <code>Mirror.gcf_from_cop</code> or "
         "<code>gcf_witness</code>", f"from the CoP; distance {ix['dist_mm']:.2f} mm (pressed in)"],
        ["4. Command", "Sec. III-E (sliders)", "<code>Schedule.__call__</code> &rarr; <code>HandCommand</code>",
         f"$v_{{\\mathrm{{ref}}}}$ = 0; $w$ = (1, 0, 0, 1, 1, 1); $F$ = {ix['force']:.1f} N; sphere at 40 deg/s about the "
         f"vertical"],
        ["5. Object twist", "Eq. (3) for a pusher", "<code>step</code>: <code>cmd.object_twist</code>, plus "
         "<code>pusher_body_twist</code>", f"$V$ = ({deg(ix['V'][:3], 1)}) deg/s, body axes"],
        ["6. Anti-windup", "Sec. III-C", "<code>step</code>: $v_{\\mathrm{sep,ref}} = 0$ within 0.5 mm",
         "inactive: $v_{\\mathrm{sep,ref}}$ is already 0"],
        ["7. Relative twist rows", "Eq. (1)", "<code>HomController.eq1_rows</code>", "$A$ (6 x 3) and $b$, Table 3"],
        ["8. Joint rates", "Eq. (2)", "<code>solve_eq2</code> (SciPy BVLS)", f"$u$ = ({ex('index', 'u')}) deg/s"],
        ["9. Reference", "Sec. III-C", "<code>step</code>: $\\theta_{\\mathrm{ref}} \\mathrel{+}= u\\,\\Delta t$, "
         "clipped", f"$\\theta_{{\\mathrm{{ref}}}}$ = ({ex('index', 'tr', 2)}) deg"],
        ["10. Action point", "Eqs. (4)&ndash;(6)", "<code>ContactFilter.update</code>",
         f"gain $k$ = {ix['k'][1]:.0f} s$^{{-1}}$; point on the pad, normal $n_a$"],
        ["11. Servo targets", "Sec. III-C/D (impedance)", "<code>step</code>: three offsets, Eq. (7)",
         f"gap step ({ex('index', 'off_gap', 2)}), force ({ex('index', 'off_force', 2)}), gravity "
         f"({ex('index', 'off_grav', 2)}) deg; target ({ex('index', 'q_cmd', 2)}) deg"],
    ]
    s_period = fill(r"""<p>The controller runs every 10 ms (100 Hz). Between two of its periods the plant takes ten physics
steps of 1 ms toward the servo targets of the last period. One call of <code>HomController.step(meas, cmd)</code> turns the
measurement and the command into the next targets. Table 1 lists its substeps in the order of the code, with the
paper's section or equation, the function and the index finger's values in the twist at 2.10 s. At that instant the
sphere's first twist step is 0.10 s old.</p>"""
                    ) + table(["Substep", "Paper [1]", "Code", "Index finger, twist, 2.10 s"], rows,
                              "One control period of <code>HomController.step</code>. Angles in degrees; the joints "
                              "are (yaw, MCP, PIP).", num_from=9)
    s_period += fill(r"""<p>A call of <code>step</code> took a median <<ms>> ms on one core of the workstation, over the
<<nper>> periods of both rollouts. On the CPU the controller alone therefore runs at about <<rt>> times real time. Film 1
shows the twist with each finger's contact frame drawn; Figure 1 plots the quantities of Table 1 over the whole
rollout.</p>""", ms=f"{N['wall_ms']:.2f}", nper=nper, rt=f"{10.0 / N['wall_ms']:.0f}")
    s_period += film(MEDIA / "20261030-code_twist.mp4",
                     "Twist of the 40 mm sphere on the tripod with mounts 95 mm by 75 mm apart, v4b pads, seed 0. "
                     "Arrows at each contact: $x_1$ red (toward the object), $y_1$ green, $z_1$ blue; magenta, the "
                     "velocity Eq. (2) planned for the finger's point at the contact (1 mm per mm/s); yellow, the "
                     "sphere's reference angular velocity (1 mm per deg/s). Finger links and sphere drawn see-through; "
                     "the support post fades when it is removed at 1.6 s.",
                     MEDIA / "20261030-code_twist.jpg")
    s_period += img(figs["twist_trace"], fill(
        r"""The twist rollout. (a) GCF distance per finger; the ticks mark periods whose frame came from the
centre of pressure. (b) Pad normal force per finger, the commanded 2.5 N (dashed) and the force on all finger links.
(c) The sphere's angular velocity about the vertical against the reference. (d) The index finger's measured joint angles
and its reference $\theta_{\mathrm{ref}}$. (e) The sizes of the index finger's three servo offsets of Eq. (7). Grey bands
alternate with the phases named on top. The sphere was held. It turned <<a1>> of <<r1>> deg in the first step and <<a2>>
of <<r2>> deg in the second. In the ramp it turned <<ramp>> deg on three pads.""",
        a1=f"{st['wz +']['angle_deg']:.1f}", r1=f"{st['wz +']['ref_deg']:.1f}", a2=f"{abs(st['wz -']['angle_deg']):.1f}",
        r2=f"{abs(st['wz -']['ref_deg']):.1f}", ramp=f"{res_tw['ramp_three_pads_max_deg']:.1f}"))

    # ---------------------------------------------------------------- 3. the model
    cz = N["cop_tip"]
    s_model = fill(r"""<p>The hand is represented in the controller by a kinematic copy, the <code>Mirror</code>. It is
a MuJoCo model of the hand and the object, and each period its joint angles and object pose are set to the measured ones.
The copy is the scene's point-contact variant, with each pad as one convex mesh. The plant's contacts and velocities are
not inputs of the controller, so the same code drives a MuJoCo plant, a Drake plant or the bench. A finger is a <code>FingerSpec</code>:
its three joints, the tip body that touches the object, the tip's collision geom and the PIP joint, whose axis orients
the contact frame. The pad's surface and the object's are analytic <code>Surface</code>s from
<code>src/morphohand/control/surfaces.py</code>: a sphere, a cylinder with flat end caps, and for the v4b pad a
<code>HalfEllipsoid</code>.</p>
<p>In each tip frame the pad's ellipsoid is centred 2.05 mm from the finger's axis on the palmar side, so its apex lies
10.55 mm from the axis. The printed pad covers the ellipsoid from 10.55 mm toward the tip to 10.86 mm toward the PIP. The
dorsal half sits inside the cassette and is part of the surface only so that the queries stay smooth (Figure 2). The
closest point $x$ of an ellipsoid with semi-axes $a_i$ to a point $p$ is</p>
$$x_i = \frac{a_i^2\,p_i}{a_i^2 + t}, \qquad s(t) = \sum_{i=1}^{3} \left(\frac{a_i\,p_i}{a_i^2 + t}\right)^2 = 1,$$
<p>with $t$ the largest root. The root lies between $t_0 = -a_k^2 + a_k|p_k|$, where $s \ge 1$, and
$t_1 = -a_k^2 + \lVert a \circ p\rVert$, where $s \le 1$; $a_k$ is the smallest semi-axis. <code>ellipsoid_closest</code>
runs Newton's method from $t_0$ on</p>
$$\varphi(t) = s(t)^{-1/2} - 1,$$
<p>which is close to linear near the pole of $s$ at $-a_k^2$. Three to six steps reach the root to $10^{-13}$ for points
outside the pad; a step that would leave the bracket is a bisection. Newton's method on $s$ itself creeps near the pole
and needed a median 42 steps, which made a control period 5.8 ms long; on $\varphi$ a query takes 20 &micro;s and a
period 1.9 ms. <code>HalfEllipsoid.closest</code> queries both halves and keeps the point that lies on its own half. The
filter of Section 9 also needs the shape operator, the rate at which the normal turns along the surface:</p>
$$S = \frac{P\,\operatorname{diag}(a^{-2})\,P}{\lVert \operatorname{diag}(a^{-2})\,x\rVert}, \qquad P = I - n\,n^{\mathsf T}.$$
<p>In the twist's squeeze the thumb's contact sat <<zt>> mm toward the tip from the apex, on the pad's midline. The index
and middle touched <<zi>> mm toward the tip and <<yi>> mm to the side, on the sides of their pads that face each other
(Figure 2). The pad is flat toward the PIP. Tilting the contact normal <<tp>> deg toward the PIP moves the pad's support
point to 1.5 mm from the pad's end; toward the tip, the same takes <<td>> deg. The foundational poses are solved with that 1.5 mm
margin (<code>fp_sr2_toy.pad_slack</code>), and with the cassette, cap and DIP link clear of the object
(<code>rigid_clearance</code>).</p>""", zt=f"{-cz['thumb'][2]:.1f}", zi=f"{-cz['index'][2]:.1f}",
                   yi=f"{abs(cz['index'][1]):.1f}", tp=f"{tilt_p:.0f}", td=f"{tilt_d:.0f}")
    s_model += img(figs["pad"], fill(r"""The v4b pad in its tip frame. (a) The section through the thumb's pad in the
plane of the finger. Orange: the printed pad; dashed: the rest of its ellipsoid. The cross is the sphere's centre at 1.41
s, joined to its closest point on the pad (dot). The arrows are the normals at which the support point comes within
1.5 mm of the pad's ends: <<tp>> deg toward the PIP and <<td>> deg toward the tip. (b) The pad seen from the object,
in each finger's tip frame, with each finger's median contact point during the squeeze (large dots) and its contact
points during the task (small dots). In the ramp the contacts rolled up to 7 mm across the pads.""",
                                     tp=f"{tilt_p:.0f}", td=f"{tilt_d:.0f}"), width=820)

    # ---------------------------------------------------------------- 4. GCFs
    gc_rows = []
    for t_, p_ in ((T_AP, p_ap), (T_TW, p_tw)):
        for f in FINGERS:
            r = p_["fingers"][f]
            x1 = arr(r["R1"], (3, 3))[:, 0]
            gc_rows.append([f"{t_:.2f}", f, "centre of pressure" if r["src"] == "cop" else "witness points",
                            f"{r['dist_mm']:.2f}", ", ".join(f"{v:.2f}" for v in x1), f"{r['F']:.2f}"])
    s_gcf = r"""<p>The GCF $\{L_1\}$ fixes where a finger touches the object and the axes in which its motion is
commanded. <code>HomController.gcf</code> builds it in one of two ways.</p>
<ul>
<li>While the finger's pad force is under 0.02 N, <code>Mirror.gcf_witness</code> takes the closest points of the pad's
mesh and the object from MuJoCo (<code>mj_geomDistance</code>). They are $p_1$ and $p_0$, and
$x_1 = (p_0 - p_1)/\lVert p_0 - p_1\rVert$. When the bodies touch or overlap, $x_1$ is the object's inward normal at
$p_0$.</li>
<li>In contact, <code>Mirror.gcf_from_cop</code> projects the measured centre of pressure $c$ onto both analytic surfaces:
$p_1$ and $p_0$ are the projections, and the normal averages the two outward normals there,
$x_1 = (n_1 - n_0)/\lVert n_1 - n_0\rVert$. The signed distance $(p_0 - p_1)\cdot x_1$ is negative in penetration.</li>
</ul>
<p><code>frame_from_normal</code> then completes the frame: $y_1$ is the PIP axis with its component along $x_1$
removed, and $z_1 = x_1 \times y_1$. The PIP axis is used because it is the axis about which the finger curls, so roll
$y$ is the finger curling over the object. Table 2 gives the frames at two instants of the twist; the keyframes of
Figure 3 draw them.</p>""" + table(
        ["Time (s)", "Finger", "Source", "Distance (mm)", "$x_1$, world axes", "Pad force (N)"], gc_rows,
        "The contact frames of the twist while the fingers approach (0.49 s) and during the first twist step (2.10 s). "
        "The world axes are the palm's: $z$ up, the thumb on $-x$.", num_from=3)
    s_gcf += img(MEDIA / "20261030-code_twist_keyframes.png",
                 "Keyframes of the twist at 0.49, 1.41 and 2.10 s, from the side and from two directions below. "
                 "Colours as in Film 1.")

    # ---------------------------------------------------------------- 5. Eq. (1)
    A = arr(ix["A"], (6, 3))
    b = arr(ix["b"])
    eq1_rows = []
    for i, c in enumerate(COMP):
        sc = math.degrees(1.0) if i < 3 else 1e3
        unit = ("deg/s per deg/s" if i < 3 else "mm/s per rad/s")
        a_ = A[i] if i < 3 else A[i] * 1e3
        eq1_rows.append([c, unit] + [f"{v:.3f}" if i < 3 else f"{v:.1f}" for v in a_] + [f"{b[i] * sc:.1f}"])
    vplan = A @ arr(ix["u"]) + b
    s_eq1 = fill(r"""<p>In Eq. (1) of [1], a finger's motion relative to the object is a linear function of its joint
rates:</p>
$$V_{L_0L_1} = A\,u + b, \qquad A = \begin{bmatrix} R_1^{\mathsf T} J_r \\ R_1^{\mathsf T} J_p \end{bmatrix}, \qquad
b = -\begin{bmatrix} R_1^{\mathsf T}\,\omega_o \\ R_1^{\mathsf T}\left(v_o + \omega_o \times (p_1 - o)\right)\end{bmatrix},$$
<p>Here $J_p$ and $J_r$ are the Jacobians of the finger's point at $p_1$, in world axes over the three joints. The
columns of $R_1$ are the GCF's axes, $o$ is the object's origin, and $(\omega_o, v_o)$ is the object's reference twist in
world axes.
Column $j$ of $A$ is what joint $j$ alone does to the contact frame. The vector $b$ is minus the motion of the object's
material point under the contact, so a finger that moves with that point has zero slide. <code>eq1_rows</code> takes the
Jacobians from <code>mj_jac</code> on the Mirror. Table 3 gives the index finger's rows at 2.10 s. The yaw column is
mostly slide $y$ and sep: yawing swings the pad across the sphere. The MCP and PIP columns curl the finger: they roll the
pad about $y_1$ and move it along the normal and the tangent plane. Every column that spins the pad about its normal also
moves it, so the joints cannot spin the pad in place. The sphere's rotation of 40 deg/s about the vertical turns it
<<w0>> deg/s about the index's contact normal and moves its surface under the pad at <<v4>> mm/s along $y_1$; $b$ is
minus these.</p>""",
                 w0=f"{-math.degrees(b[0]):.1f}", v4=f"{-1e3 * b[4]:.1f}") + table(
        ["Component", "Unit of $A$", "Yaw", "MCP", "PIP", "$b$ (deg/s, mm/s)"], eq1_rows,
        "Eq. (1) for the index finger at 2.10 s of the twist: the columns of $A$ per joint, and $b$ for the sphere's "
        "reference rotation of 40 deg/s about the vertical.", num_from=2)
    s_eq1 += fill(r"""<p>With the joint rates of Section 6, $A u + b$ is the planned relative twist, (<<vp>>) in deg/s
and mm/s. It is returned as <code>v_plan</code> in the info of <code>step</code>.</p>""",
                  vp=", ".join(rel_twist(vplan)))

    # ---------------------------------------------------------------- 6. Eq. (2)
    e2 = []
    for i, c in enumerate(COMP):
        e2.append([c, f"{ix['w'][i]:.0f}"] + [rel_twist(p_tw["fingers"][f]["v_plan"])[i] for f in FINGERS])
    e2.append(["joint rates $u$ (deg/s)", ""] + [deg(p_tw["fingers"][f]["u"]) for f in FINGERS])
    s_eq2 = fill(r"""<p>Eq. (2) of [1] defines the joint rates as those whose relative contact twist is closest to the
command, component by component:</p>
$$u^\star = \arg\min_{u}\; \sum_{i=1}^{6} w_i\,\lambda_i^2 \left(v_{i,\mathrm{ref}} - (A u + b)_i\right)^2
+ w_{\mathrm{d}}\,\lVert u\rVert^2 + w_{\mathrm{h}}\,\lVert k_{\mathrm{h}}(\theta_{\mathrm{h}} - \theta) - u\rVert^2,$$
$$\max\!\left(-u_{\max},\ \frac{\theta_{\min} - \theta_{\mathrm{ref}}}{\Delta t}\right) \le u \le
\min\!\left(u_{\max},\ \frac{\theta_{\max} - \theta_{\mathrm{ref}}}{\Delta t}\right).$$
<p>Here $\lambda_i$ is 1 for the angular rows and $1/\ell$ for the linear ones. The term with $w_{\mathrm{d}} =
10^{-3}$ damps the rates. The term with $w_{\mathrm{h}} = 10^{-5}$ pulls the finger weakly toward its open pose
$\theta_{\mathrm{h}}$, with $k_{\mathrm{h}} = 1\ \mathrm{s}^{-1}$. The rate limit is $u_{\max}$ = 2 rad/s, and the bounds
keep $\theta_{\mathrm{ref}}$ inside the joint limits after the step. The weight of a component sets how much its error
costs. In the twist every pad may roll (both roll weights 0) and should neither slide nor spin (weight 1). Three joints
cannot meet four such rows, so their errors are traded against each other (Table 4). During the first step the planned
slides stayed under <<sl>> mm/s, while the sphere's surface under each pad moved at a median <<vs>> mm/s, so the pads
stick. The spin was given up: a median <<sp>> deg/s about each
normal, which the pads' torsional friction then has to absorb. The rolls take up the rest of the sphere's rotation.</p>""",
                 sl=f"{slide1:.1f}", sp=f"{spin1:.0f}", vs=f"{surf1:.0f}") + table(
        ["Component", "Weight", "Thumb", "Index", "Middle"], e2,
        "Eq. (2) at 2.10 s of the twist: the planned relative contact twist $A u^\\star + b$ per finger (deg/s for spin "
        "and rolls, mm/s for sep and slides) against a reference of zero, and the joint rates.", num_from=1)
    s_eq2 += fill(r"""<p><code>solve_eq2</code> stacks the weighted rows, the damping rows and the posture rows into one
least-squares problem and solves it under the bounds with SciPy's bounded-variable least squares (BVLS). The GPU
controller solves the same problem in closed form. Its cost is the quadratic
$\tfrac12 u^{\mathsf T} H u - g^{\mathsf T} u$ with $H = A^{\mathsf T} W A + (w_{\mathrm{d}} + w_{\mathrm{h}}) I$. Each of
the three rates is either free, at its lower bound or at its upper bound, which makes 27 patterns. For each pattern the
free rates solve a linear system, and the feasible pattern of lowest cost is the minimum, because the problem is strictly
convex. Re-solving all <<n>> finger-periods of both rollouts both ways gave rates that differ by at most <<d>> rad/s; a
bound was active in <<nb>> of them.</p>""", n=n2, d=f"{dmax:.0e}", nb=nb2)

    # ---------------------------------------------------------------- 7. reference
    s_ref = fill(r"""<p>The planned rates are integrated into the joint reference, which is the servo target before the
offsets of Section 10:</p>
$$\theta_{\mathrm{ref}} \leftarrow \operatorname{clip}\!\Big(\operatorname{clip}\big(\theta_{\mathrm{ref}} + u^\star\Delta t,\
\theta - L,\ \theta + L\big),\ \theta_{\min},\ \theta_{\max}\Big), \qquad L = 0.3\ \mathrm{rad}.$$
<p>A position servo is a torsional spring of stiffness $k_p$ between its target and its shaft, so a finger trails its
reference by the torque it needs divided by $k_p$. The lead limit $L$ stops a blocked finger from winding its reference
up. The largest lead in the two rollouts was <<l1>> deg (twist) and <<l2>> deg (pin-push), under $L$ = 17.2 deg.</p>
<p>The anti-windup of [1] ends an approach. From 0.10 s each finger is commanded $v_{\mathrm{sep}}$ = 20 mm/s toward the
object from 8 mm away. Once the GCF distance is under 0.5 mm, <code>step</code> sets $v_{\mathrm{sep,ref}}$ to zero, so
the finger stops at the surface. In the twist all three fingers stopped at <<ta>> s, <<d0>> to <<d1>> mm from the
sphere. The commanded forces turn on at 0.80 s.</p>""",
                 l1=f"{lead['tw']:.1f}", l2=f"{lead['pp']:.1f}", ta=f"{max(v[0] for v in stop.values()):.2f}",
                 d0=f"{min(v[1] for v in stop.values()):.2f}", d1=f"{max(v[1] for v in stop.values()):.2f}")

    # ---------------------------------------------------------------- 8. Eq. (3)
    m_rows = [
        ["{M} origin in the object frame", f"({mm(pm)}) mm"],
        ["{M}'s $x$ in the object frame", "(" + ", ".join(f"{v:.3f}" for v in Rm[:, 0]) + ")"],
        ["$s$", f"{math.degrees(p_pp['s']):.0f} deg/s"],
        ["Middle's $V$, Eq. (3): $\\omega$", f"({deg(mid['V'][:3])}) deg/s"],
        ["Middle's $V$, Eq. (3): $v$", f"({mm(mid['V'][3:])}) mm/s"],
        ["Middle's joint rates $u$", f"({deg(mid['u'])}) deg/s"],
        ["Middle's planned relative twist", "(" + ", ".join(rel_twist(mid["v_plan"])) + ")"],
        ["Middle's contact, distance to the pinch axis", f"{1e3 * lever:.0f} mm"],
    ]
    s_eq3 = fill(r"""<p>In the pin-push the thumb and the index pinch the 25 mm cylinder across palm $x$, and the middle
finger presses its side further along. The pinch is a revolute joint about the line through its two contacts. The
pinching fingers have no object twist to follow, and their weights free the spin and both rolls. The middle is the
pusher. When its command starts ($s \ne 0$), <code>pinch_frame</code> fixes $\{M\}$ to the object. Its origin is the
midpoint of the pinch contacts on the object, and its $x$ axis runs from the thumb's contact to the index's. By Eq. (3)
of [1], the speed $s$ about that axis becomes a body twist of the object,</p>
$$V_{PB_0} = \mathrm{Ad}_{T_{B_0M}}\,(s, 0, 0, 0, 0, 0)^{\mathsf T}, \qquad
\mathrm{Ad}_{T} = \begin{bmatrix} R & 0 \\ [p]\,R & R \end{bmatrix},$$
<p>with $(R, p)$ the pose of $\{M\}$ in the object frame and $[p]$ the cross-product matrix. <code>pusher_body_twist</code>
computes it and <code>step</code> adds it to the middle's $V$ before Eq. (1). The middle's weights are all 1 and its
reference is zero: it must move with the cylinder's surface as if the cylinder already turned at $s$. Its pad presses
along the pinch axis with 1.5 N, so the pad sticks and drags the cylinder round by friction. Table 5 gives the numbers 0.50
s into the up stroke. The middle's contact lies <<lev>> mm from the pinch axis, where the commanded rotation moves the
surface at <<vs>> mm/s; Eq. (2) planned the middle's point to move at <<vm>> mm/s. With all six weights at 1, the plan
kept the middle's slides within <<msl>> mm/s and gave up roll: <<mro>> deg/s about $y_1$. The cylinder turned <<u1>> of 15 deg
in the up stroke, <<u2>> of 30 deg down and <<u3>> of 15 deg back (pin-push score <<sc>>).</p>""",
                 lev=f"{1e3 * lever:.0f}", vs=f"{1e3 * v_surf:.0f}", vm=f"{1e3 * v_mid:.0f}",
                 msl=f"{1e3 * max(abs(mid['v_plan'][4]), abs(mid['v_plan'][5])):.1f}",
                 mro=f"{math.degrees(mid['v_plan'][1]):.1f}",
                 u1=f"{sk['up']['angle_deg']:.1f}", u2=f"{abs(sk['down']['angle_deg']):.1f}",
                 u3=f"{sk['back']['angle_deg']:.1f}", sc=f"{max(0.0, 1 - res_pp['theta_rmse_deg'] / 15):.2f}")
    s_eq3 += table(["Quantity", "Value"], m_rows, "Eq. (3) in the pin-push at 2.50 s (relative twists in deg/s and "
                   "mm/s, ordered spin, roll $y$, roll $z$, sep, slide $y$, slide $z$).", num_from=1)
    s_eq3 += film(MEDIA / "20261030-code_pinpush.mp4",
                  "Pin-push of the 25 mm cylinder on the same layout from its pinch pose, seed 0. The cylinder is "
                  "commanded up 15 deg, down 30 deg and back at 15 deg/s about the pinch axis (yellow). Other colours as "
                  "in Film 1.",
                  MEDIA / "20261030-code_pinpush.jpg")
    s_eq3 += img(MEDIA / "20261030-code_pinpush_keyframes.png",
                 "Keyframes of the pin-push at 0.49, 1.41 and 2.49 s, from the side and from two directions below.")
    s_eq3 += img(figs["pinpush_trace"], fill(r"""The pin-push rollout. (a) The cylinder's angle about the pinch axis,
from its value at the task's start, against the command; the squeeze had already turned it <<sq>> deg. (b) Pad normal forces against their commands (dashed). (c) The middle finger's GCF distance and the gap that its
servo target closes, cut at 6 mm (dotted). (d) The middle finger's MCP joint: measured, reference and servo target.""",
                                             sq=f"{sq_turn:.1f}"))

    # ---------------------------------------------------------------- 9. filter
    s_flt = fill(r"""<p>The servo targets of Section 10 press at an action point $p_a$ along a normal $n_a$. In [1] both
are estimated with a complementary filter. Frames $\{C_0\}$ on the object and $\{C_1\}$ on the finger move over their surfaces with
the geodesic velocities that the reference twist implies, corrected toward the measured contact origins:</p>
$$\dot g_j = \dot g_{j,\mathrm{ref}} + k_j\,E\,(p_{L_j} - p_{C_j}), \qquad
k_j = k_{\min} + (k_{\max} - k_{\min})\left(1 - e^{-3\lVert\dot g_{j,\mathrm{ref}}\rVert/\alpha}\right),$$
<p>These are Eqs. (5) and (6). $E$ projects onto the tangent plane; $k_{\min}$ = 5 s$^{-1}$, $k_{\max}$ = 100
s$^{-1}$ and $\alpha$ = 20 mm/s. <code>roll_slide_rates</code> finds $\dot g_{j,\mathrm{ref}}$ from the rolling and sliding
kinematics, $[S\,\mathrm{Ad}\,G_0(K_0)\ \ {-S}\,G_1(K_1)]\,\dot g = S\,V_{\mathrm{ref}}$. In it, $G(K)$ maps a geodesic
velocity to the frame's twist through the 2 x 2 shape operator $K$, and $S$ keeps the four tangential rows.
<code>Surface.geodesic_step</code> integrates Eq. (4): it follows the osculating circle of the surface for the step's arc
length and projects the end back onto the surface. The new $\{C_1\}$ gives the action point $p_a$ and its normal
$n_a$.</p>
<p>Every commanded roll and slide of our four tasks is zero. The reference geodesic velocities are then zero, and the
gain stays at $k_{\min}$. Each period moves the action point $k_{\min}\Delta t$ = 5 % of the way toward the measured
contact point, so the filter is a low-pass with a 0.2 s time constant. In the twist's ramp the index's measured contact point moved
<<jm>> mm per period (RMS) on the pad, and the action point <<jf>> mm.</p>""",
                 jm=f"{jit_m:.3f}", jf=f"{jit_f:.3f}")

    # ---------------------------------------------------------------- 10. impedance
    im_rows = []
    for f in FINGERS:
        r = p_sq["fingers"][f]
        for k, j in enumerate(JOINTS):
            im_rows.append([f if k == 0 else "", j,
                            f"{r['force']:.1f}" if k == 0 else "", f"{r['F']:.2f}" if k == 0 else "",
                            f"{r['gap_mm']:.2f}" if k == 0 else "",
                            f"{math.degrees(r['off_gap'][k]):.2f}", f"{math.degrees(r['off_force'][k]):.2f}",
                            f"{math.degrees(r['off_grav'][k]):.2f}",
                            f"{math.degrees(r['q_cmd'][k] - r['theta'][k]):.2f}"])
    s_imp = fill(r"""<p>The paper's impedance controller commands joint torques. Our servos take position targets only,
so <code>step</code> writes the force into the target as three offsets (Eq. 7 of the review page):</p>
$$\theta_{\mathrm{target}} = \theta_{\mathrm{ref}}
+ \underbrace{J_a^{\mathsf T}\big(J_a J_a^{\mathsf T} + \delta^2 I\big)^{-1} g\,n_a}_{\text{gap step}}
+ \underbrace{\frac{J_a^{\mathsf T}\,F\,n_a}{k_p}}_{\text{force}}
+ \underbrace{\frac{\tau_g}{k_p}}_{\text{gravity}},\qquad
g = \operatorname{clip}\!\big(d - n_a^{\mathsf T} J_a(\theta_{\mathrm{ref}} - \theta),\ 0,\ g_{\max}\big).$$
<p>$J_a$ is the linear Jacobian at the action point and $d$ the GCF distance. The damping is $\delta$ = 2 mm/rad and
the cut $g_{\max}$ = 6 mm. $\tau_g$ are the joint torques that hold the finger against its weight (<code>mj_rne</code> at
zero velocity). The gap $g$
is the distance from the reference's action point to the measured surface, linearised; it is positive when the reference
has fallen behind the surface. The gap step moves the target onto the surface. The force term then places it $F/k_p$
springs' worth beyond, so the servo presses with $F$ when its shaft stops at the surface. A 2.5 N press on a 50 mm lever
takes 0.125 N m of joint torque. At $k_p$ = 4 N m/rad, that is a target 1.8 deg beyond the shaft. Table 6 gives the offsets in the twist's
squeeze.</p>""") + table(
        ["Finger", "Joint", "$F$ (N)", "Pad force (N)", "$g$ (mm)", "Gap step", "Force", "Gravity",
         "Target &minus; measured"], im_rows,
        "Servo targets in the twist at 1.41 s: the commanded and measured pad force, the gap and the three offsets of "
        "Eq. (7) per joint, in degrees.", num_from=2)
    s_imp += fill(r"""<p>The force is open loop: the measured pad force follows from the target offset, the servo spring
and the geometry. In the squeeze the thumb pressed <<ft>> N and the index and middle <<fi>> and <<fm>> N, against 2.5 N
commanded.</p>
<p>The cut at $g_{\max}$ matters in the pin-push. When the forces came on at 0.80 s, the cylinder yawed <<yaw>> deg
away from the middle finger. The middle's MCP followed it from <<m0>> to <<m1>> deg, while its reference stayed where the
approach had left it. The reference then lay more than 6 mm behind the surface for <<cap>> % of the task periods. The gap step
was cut at 6 mm, and the target ended a median <<dm>> deg beyond the measured MCP angle. The middle pressed with a
median <<fmid>> N of its commanded 1.5 N (Figure 5).</p>""", ft=f"{p_sq['fingers']['thumb']['F']:.1f}", fi=f"{p_sq['fingers']['index']['F']:.1f}",
                  fm=f"{p_sq['fingers']['middle']['F']:.1f}", yaw=f"{yaw[0] - yaw[1]:.1f}", m0=f"{mcp[0]:.1f}",
                  m1=f"{mcp[1]:.1f}", cap=f"{100 * capped:.0f}", dm=f"{d_mcp:.2f}", fmid=f"{f_mid:.2f}")

    # ---------------------------------------------------------------- 11. plant
    s_plant = fill(r"""<p><code>hom_tasks.Plant</code> wraps the pads scene in MuJoCo. Each v4b pad is modelled as
<<npad>> spheres of <<rpad>> mm radius, with stiffnesses from pad builder v2. The cassette, cap and DIP link are packs of rigid
spheres. The object weighs <<mass>> g. Every joint is a position servo with $k_p$ = <<kp>> N m/rad, a torque limit of
<<tau>> N m and <<damp>> N m s/rad of damping, the bench plant of the SR2 hand. The friction coefficient of the pads is
<<mu>>, with an elliptic friction cone. <code>set_targets</code> writes the servo targets; <code>advance</code> steps the
physics at <<dt>> ms up to the next period. <code>contacts</code> sums each finger's pad contacts on the object into a
normal force and a centre of pressure, the quantities a fingertip sensor reports. The force of the finger's other bodies
on the object is summed separately. <code>state</code> returns these with the joint angles and the object pose, and the
controller receives nothing else. The support post under the object is removed at 1.6 s.</p>""",
                   npad=f"{mdl['pads_per_finger']:,}", rpad=f"{mdl['pad_r_mm']:.2f}", mass=f"{mdl['obj_mass_g']:.0f}",
                   kp=f"{mdl['kp']:.0f}", tau=f"{mdl['forcerange'][1]:.0f}", damp=f"{mdl['damping']:.2f}",
                   mu=f"{mdl['pad_friction'][0]:.1f}", dt=f"{1e3 * mdl['timestep']:.0f}")

    # ---------------------------------------------------------------- 12. tasks
    prm = tw["prm"]
    t_rows = [
        ["Open", "0&ndash;0.1", "the open pose, 8 mm off the object", "the same"],
        ["Close", "0.1&ndash;0.8", "approach at $v_{\\mathrm{sep}}$ = 20 mm/s until the anti-windup", "the same"],
        ["Squeeze", "0.8&ndash;1.6", f"force $F$ = {prm['f_tripod']:.1f} N per pad",
         f"force $F$ = {prm['f_pinch']:.0f} N for the index, {pp['periods'][100]['fingers']['thumb']['force']:.2f} N for the "
         f"thumb (it also opposes the middle), {prm['f_drive']:.1f} N for the middle"],
        ["Lift", "1.6&ndash;2.0", "the post is removed", "the same"],
        ["Task", "2.0&ndash;", f"sphere at $\\omega_z$ = +40 then &minus;40 deg/s for 0.5 s each, 0.3 s rest, a ramp at "
         f"{prm['ramp_dps']:.0f} deg/s for 3 s, 0.5 s hold", f"turns of +15, &minus;30 and +15 deg at "
         f"{prm['rot_dps']:.0f} deg/s by $s$, 0.5 s hold"],
        ["Weights", "", "(1, 0, 0, 1, 1, 1) for every finger", "pinch (0, 0, 0, 1, 1, 1); middle all 1"],
        ["Object twist", "", "about the vertical through the centre, body axes", "none for the pinch; Eq. (3) for the "
         "middle"],
    ]
    s_tasks = r"""<p><code>hom_tasks.Schedule</code> plays the role of the paper's teleoperation sliders: it returns the
command of every period as a function of time and the measured state. All tasks start with the paper's pick. Table 7
lists the two examples' schedules; the roll and the press, the other two tasks of the study, are described in the
module docstring of <code>scripts/hom_tasks.py</code> and on the review page.</p>""" + table(
        ["Phase", "Time (s)", "Twist", "Pin-push"], t_rows, "Commands of the two example tasks.", num_from=9)

    # ---------------------------------------------------------------- 13. GPU
    g_rows = [
        ["<code>Mirror.set_state</code>", "a second MuJoCo-Warp data (<code>Batch.mirror</code>), its kinematics "
         "replayed from a captured CUDA graph", "float32"],
        ["<code>gcf_witness</code> (<code>mj_geomDistance</code>, pad mesh)", "<code>_ell_witness</code>: the pad's "
         "point nearest the sphere's centre; for the cylinder, alternating projections between the solid pad and the "
         "cylinder, 4 per period from the last period's point", "analytic pad instead of its mesh"],
        ["<code>gcf_from_cop</code>", "the same formulas on tensors", ""],
        ["<code>ellipsoid_closest</code>, Newton on $\\varphi$", "<code>_ell_newton</code>: six steps of Newton on "
         "$\\varphi$ from $t_0$, a step leaving the bracket being a bisection", "under 1 &micro;m for points outside "
         "the pad"],
        ["<code>eq1_rows</code> (<code>mj_jac</code>)", "Jacobians from the joint axes and anchors: $J_r$ = axis, "
         "$J_p$ = axis &times; ($p_1$ &minus; anchor)", ""],
        ["<code>solve_eq2</code> (BVLS)", "the 27-pattern enumeration of Section 6", f"rates within {dmax:.0e} rad/s"],
        ["<code>ContactFilter</code>, Eqs. (4)&ndash;(6)", "the low-pass at $k_{\\min}$ on the pad's surface",
         "the same for zero commanded roll and slide"],
        ["<code>Mirror.gravity</code> (<code>mj_rne</code>)", "each joint's axis crossed with the moments of the "
         "weights of the bodies below it", ""],
        ["<code>Plant</code> (MuJoCo, float64)", "MuJoCo-Warp, float32, solver tolerance 10$^{-6}$, 20 iterations; "
         "pads and the rigid spheres under 2 mm collide with the object only", "physics"],
    ]
    cg = [[("Twist, 40 mm sphere" if t == "twist" else "Pin-push, 25 mm cylinder"), str(c["n"]),
           f"{100 * c['held_cpu']:.0f}", f"{100 * c['held_gpu']:.0f}", f"{100 * c['same']:.0f}",
           f"{100 * c['seed_same'][0]:.0f}&ndash;{100 * c['seed_same'][1]:.0f}", f"{c['rho']:.2f}",
           f"{c['seed_rho'][0]:.2f}&ndash;{c['seed_rho'][1]:.2f}", f"{c['cpu_s']:.1f}", f"{c['gpu_ms']:.0f}"]
          for t, c in C.items()]
    s_gpu = fill(r"""<p><code>scripts/hom_gpu.py</code> runs the same control period for thousands of worlds at once on
MuJoCo-Warp. All layouts of one object size share one model. They differ only in the positions of the three mount
bodies, so <code>body_pos</code> gets one row per world. The palm and the support post are mocap bodies. A world
is one (layout, grasp, seed, sensing variant). <code>Batch.control</code> is <code>HomController.step</code> written in
torch over tensors of shape (worlds, fingers, ...); Table 8 pairs its parts with the CPU code. The plant's ten physics
steps per period run as one captured CUDA graph.</p>""") + table(
        ["CPU (<code>hom.py</code>, <code>hom_tasks.py</code>)", "GPU (<code>hom_gpu.Batch</code>)", "Difference"],
        g_rows, "The batched controller's parts.", num_from=9)
    ct, cp = C["twist"], C["pinpush"]
    s_gpu += fill(r"""<p>The CPU runner (<code>hom_gpu.py cpu-check</code>) repeated <<n1>> twist and <<n2>> pin-push
grasps of the 2-D grid at seed 0. It used the GPU queue's layouts, grasps and open poses (Table 9, Figure 6). The held
outcome agreed on <<s1>> % of the twists and <<s2>> % of the pin-pushes. The task scores had rank correlations of <<r1>>
and <<r2>>. The GPU's own seeds 0, 1 and 2 start the object up to 1 mm and 2 deg apart. Between them the held outcome
agreed on <<ss1>> % of the twists and <<ss2>> % of the pin-pushes, with rank correlations of <<sr1>> and <<sr2>>. CPU
and GPU therefore differ no more than two starts of the same rollout do. The pin-push is the more sensitive task: one
pin-push rollout is a noisy measurement of a grasp, and the grid maps average three seeds. On an <<gpu>>, the GPU took
<<g1>> ms per twist rollout in batches of <<b>> worlds and <<g2>> ms per pin-push rollout in batches of <<b2>>. One CPU
core took <<c1>> and <<c2>> s.</p>""",
                  n1=ct["n"], n2=cp["n"], s1=f"{100 * ct['same']:.0f}", s2=f"{100 * cp['same']:.0f}",
                  r1=f"{ct['rho']:.2f}", r2=f"{cp['rho']:.2f}",
                  ss1=f"{100 * ct['seed_same'][0]:.0f} to {100 * ct['seed_same'][1]:.0f}",
                  ss2=f"{100 * cp['seed_same'][0]:.0f} to {100 * cp['seed_same'][1]:.0f}",
                  sr1=f"{ct['seed_rho'][0]:.2f} to {ct['seed_rho'][1]:.2f}",
                  sr2=f"{cp['seed_rho'][0]:.2f} to {cp['seed_rho'][1]:.2f}",
                  b=f"{ct['batch']:,}", b2=f"{cp['batch']:,}", gpu=ct["gpu"].replace("NVIDIA GeForce ", ""),
                  g1=f"{ct['gpu_ms']:.0f}", g2=f"{cp['gpu_ms']:.0f}", c1=f"{ct['cpu_s']:.1f}", c2=f"{cp['cpu_s']:.1f}")
    s_gpu += table(["Task", "Grasps", "Held, CPU (%)", "Held, GPU (%)", "Same outcome, CPU and GPU (%)",
                    "Same outcome, GPU seeds (%)", "Spearman, CPU and GPU", "Spearman, GPU seeds", "CPU (s/rollout)",
                    "GPU (ms/rollout)"], cg,
                   "CPU and GPU controllers on the same grid grasps at seed 0, and the GPU's seeds 0, 1 and 2 against "
                   "each other (range over the three pairs). Scores are the twist and pin-push scores.", num_from=1)
    s_gpu += img(figs["cpu_gpu"], fill("Task scores of the CPU and GPU controllers on the same grasps at seed 0 (0 when "
                                       "dropped). Where both held, the rank correlation is <<a>> (twist, <<na>> grasps) "
                                       "and <<b>> (pin-push, <<nb>>).", a=f"{ct['rho_both']:.2f}", na=ct["n_both"],
                                       b=f"{cp['rho_both']:.2f}", nb=cp["n_both"]), width=820)
    pt, pq = port_check("twist", 40), port_check("pinpush", 25)
    s_gpu += fill(r"""<p>From the second period of the task phase, <code>Batch.control_period</code> replays the
controller's period as one captured CUDA graph. The period has no host synchronisation: the fingers' roles enter as
float masks, the 3 x 3 systems are solved in closed form, the pad's closest point takes six Newton steps, and the
controller's state is updated in place. On the grid's validation worlds (seeds 3 to 5) it gave the same held outcome as
the earlier eager controller on <<st>> % of <<nt>> twists and <<sq>> % of <<nq>> pin-pushes. Two seeds of one controller
agree on <<ss1>> % and <<ss2>> %. A rollout took <<mt1>> ms instead of <<mt0>> in the twist and <<mq1>> ms instead of
<<mq0>> in the pin-push.</p>""", st=f"{100 * pt['same']:.1f}", nt=f"{pt['n']:,}", sq=f"{100 * pq['same']:.1f}",
                  nq=f"{pq['n']:,}", ss1=f"{100 * ct['seed_same'][0]:.0f} to {100 * ct['seed_same'][1]:.0f}",
                  ss2=f"{100 * cp['seed_same'][0]:.0f} to {100 * cp['seed_same'][1]:.0f}", mt0=f"{pt['ms'][0]:.0f}",
                  mt1=f"{pt['ms'][1]:.0f}", mq0=f"{pq['ms'][0]:.0f}", mq1=f"{pq['ms'][1]:.0f}")

    # ---------------------------------------------------------------- 14. map
    mp = [
        ["Sec. III-B, GCFs", "<code>Gcf</code>, <code>Mirror.gcf_witness</code>, <code>gcf_from_cop</code>",
         "three pads on the sphere", "two pinching pads and the pusher"],
        ["Eq. (1)", "<code>HomController.eq1_rows</code>", "the sphere's rotation sets $b$", "Eq. (3) sets $b$ for the "
         "middle"],
        ["Eq. (2)", "<code>solve_eq2</code>", "free rolls; no slide, no spin", "pinch: free spin and rolls; pusher: all 1"],
        ["Sec. III-B, sub-assemblies", "<code>FingerCommand.mode</code>, <code>HandCommand.pinch</code>",
         "three pinches", "pinch and pusher"],
        ["Eq. (3)", "<code>pusher_body_twist</code>, <code>pinch_frame</code>", "", "speed $s$ = &plusmn;15 deg/s"],
        ["Sec. III-C, anti-windup", "<code>step</code>, <code>aw_dist</code>", "the approach", "the approach"],
        ["Eqs. (4)&ndash;(6)", "<code>ContactFilter</code>, <code>roll_slide_rates</code>, <code>surface_jacobian</code>",
         "low-pass at $k_{\\min}$", "low-pass at $k_{\\min}$"],
        ["Impedance (Sec. III-C/D)", "<code>step</code>: <code>off_gap</code>, <code>off_force</code>, "
         "<code>off_grav</code>", "2.5 N per pad", "3, 4.15 and 1.5 N"],
    ]
    s_map = (f'<p>Table 10 maps the paper onto the code and the two tasks. The method page of 2026-10-21 '
             f'(<a href="{METHOD}">20261021-hom_controller_method_and_implementation.html</a>, Section 10) lists every '
             f'difference from the paper with its reason, and the <a href="{REVIEW}">review page</a> condenses them.</p>'
             + table(["Paper [1]", "Code", "Twist", "Pin-push"], mp, "Paper, code and the example tasks.", num_from=9))

    # ---------------------------------------------------------------- 15. future work, 16. files
    s_fut = fill(r"""<ul>
<li>In the example the pin-push middle finger pressed <<fmid>> N of its 1.5 N, while its reference stayed more than
$g_{\max}$ behind the surface. Over the 60 grid grasps of Table 9, the median of its mean force was <<fc>> N (CPU) and
<<fg>> N (GPU). Re-anchor a finger's reference to its measured angles when its gap exceeds $g_{\max}$ in contact
(<code>HomController.reanchor</code>). Then rerun the pin-push grid (<code>hom_gpu.py run --task pinpush --layouts
grid2d_size --d 25</code>) and compare the middle's mean force (<code>F_mean</code>) and the scores. A middle force that
stays under 1 N after the change would point to the cylinder's yaw as the limit.</li>
<li>After the running queues finish, replace the GPU's ellipsoid solver by Newton's method on $\varphi$ with 8 steps.
Then check the held outcome against the current rows of the 40 mm twist grid, and the time per rollout.</li>
<li>The filter runs only at $k_{\min}$ in these tasks. A task with commanded rolling, the paper's Exp. B on the sphere,
would exercise Eqs. (4) to (6) and $k_{\max}$.</li>
</ul>""", fmid=f"{f_mid:.2f}", fc=f"{C['pinpush']['f_mid'][0]:.2f}",
                 fg=f"{C['pinpush']['f_mid'][1]:.2f}")
    s_files = f"""<p>References:</p><ul><li>[1] S. Wang, J. Oh and N. S. Pollard, "Relative Contact Velocity-Controlled
Hand-Object Mechanism for Dexterous Tool Manipulation", arXiv 2609.25619v2, 2026.
<a href="https://arxiv.org/abs/2609.25619">arxiv.org/abs/2609.25619</a>.</li></ul>
<p>Files:</p><ul>
<li>This page: <code>{OUT.relative_to(ROOT)}</code>, built by <code>scripts/hom_code_page.py</code> (template
<code>scripts/hom_review_page.template.html</code>); the parent page: <a href="{REVIEW}">{REVIEW}</a>.</li>
<li>Controller: <code>src/morphohand/control/hom.py</code> (<code>HomController.step</code> reports every
intermediate in its info dict), <code>src/morphohand/control/surfaces.py</code>; tests
<code>tests/test_hom_control.py</code>.</li>
<li>Rollouts, films and keyframes: <code>~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python
scripts/hom_code_trace.py</code>, rows <code>{(ROWS / '20261030-code_twist.json').relative_to(ROOT)}</code> and
<code>20261030-code_pinpush.json</code>.</li>
<li>CPU against GPU: <code>logs/20261029-hom_sizes/cpu_check_code_page.sh</code> (rows
<code>{(ROWS / '20261030-cpu_twist_d40.jsonl').relative_to(ROOT)}</code>, <code>20261030-cpu_pinpush_d25.jsonl</code>)
against <code>{GRID.relative_to(ROOT)}/twist_d40.jsonl</code> and <code>pinpush_d25.jsonl</code>.</li>
<li>Tasks: <code>scripts/hom_tasks.py</code>; GPU controller <code>scripts/hom_gpu.py</code>; grasps
<code>scripts/fp_sr2_toy.py</code>; the v4b hand <code>scripts/build_cad_v2_finger.py</code>,
<code>assets/mjcf/experimental/20261030-cad_v2</code>.</li>
</ul>"""

    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("period", "One control period of the HOM controller", s_period),
        ("model", "The hand, the v4b pad and the object in the controller", s_model),
        ("gcf", "Generalized contact frames from witness points and the centre of pressure", s_gcf),
        ("eq1", "Relative contact twist, Eq. (1)", s_eq1),
        ("eq2", "Joint rates by weighted least squares, Eq. (2)", s_eq2),
        ("ref", "Joint reference, lead limit and anti-windup", s_ref),
        ("eq3", "Pinch frame and pusher twist in the pin-push, Eq. (3)", s_eq3),
        ("filter", "Contact-point filter, Eqs. (4) to (6)", s_flt),
        ("imp", "Servo targets from the impedance and gravity offsets", s_imp),
        ("plant", "The simulated plant and its measurements", s_plant),
        ("tasks", "Task schedules of the twist and the pin-push", s_tasks),
        ("gpu", "The batched controller on MuJoCo-Warp", s_gpu),
        ("map", "Paper, code and tasks", s_map),
        ("future", "Future work", s_fut),
        ("files", "References and files", s_files),
    ]
    lede = fill(r"""The HOM controller turns each finger's commanded motion relative to the object into three servo
targets, in eleven substeps every 10 ms. The example hand is the tripod with mounts 95 mm by 75 mm apart and the v4b
fingertip. It twisted a 40 mm sphere and pin-pushed a 25 mm cylinder, and held both. In the
first twist step the planned slides of the pads stayed under <<sl>> mm/s, and a spin of <<sp>> deg/s about each contact
normal was given up. A period takes <<ms>> ms on one CPU core, and a GPU rollout <<g1>> ms. On 60 grid grasps per task the
GPU controller reached the CPU's held outcome in <<s1>> % of twists and <<s2>> % of pin-pushes, as often as two of its
own seeds agree.""", sl=f"{slide1:.1f}", sp=f"{spin1:.0f}",
                ms=f"{N['wall_ms']:.1f}", s1=f"{100 * C['twist']['same']:.0f}", s2=f"{100 * C['pinpush']['same']:.0f}",
                g1=f"{C['twist']['gpu_ms']:.0f}")
    summary = fill(r"""Walkthrough of the HOM controller's code on the SR2 hand with the v4b fingertip: eleven substeps
per 10 ms period, each with the twist's and pin-push's numbers. A period takes <<ms>> ms on one CPU core; the GPU
controller agreed with the CPU on the held outcome for <<s1>> % and <<s2>> % of 60 grasps per task, as often as two of its
own seeds agree.""",
                   ms=f"{N['wall_ms']:.1f}", s1=f"{100 * C['twist']['same']:.0f}",
                   s2=f"{100 * C['pinpush']['same']:.0f}")
    byline = (f'Code walkthrough, 2026-10-30. Paper: arXiv 2609.25619v2 [1]. Parent page: <a href="{REVIEW}">HOM '
              'control and fingertip sensing</a>. Hand: the CAD-model SR2 hand with the v4b fingertip on the 1 N m '
              'plant. Builder: <code>scripts/hom_code_page.py</code>.')
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>'
                     for k, (sid, hd, body) in enumerate(sections, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "HOM controller code", "H1": "Implementation of the HOM controller on the SR2 hand with the "
                 "v4b fingertip", "EYEBROW": "Reorientation mechanisms and control", "LEDE": lede, "BYLINE": byline,
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    nf = []
    page = render(page, nf)
    page = docs_index.tag(page, topic="mechanisms", summary=summary, date="2026-10-30")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {nf[0]} formulas", flush=True)


def main():
    if "--numbers" in sys.argv:
        N = numbers()
        tw, pp = N["tw"], N["pp"]
        print("wall per period ms", N["wall_ms"])
        print("twist", tw["result"])
        print("pinpush", {k: v for k, v in pp["result"].items()})
        print("cop tip", {f: np.round(v, 2) for f, v in N["cop_tip"].items()})
        print("model", tw["model"])
        print("eq2 check", eq2_check([tw, pp]))
        for task, d in (("twist", 40), ("pinpush", 25)):
            try:
                c = cpu_gpu(task, d)
                print(task, {k: v for k, v in c.items() if k != "sc"})
            except FileNotFoundError as e:
                print(task, "no rows yet", e)
        return
    build()


if __name__ == "__main__":
    main()

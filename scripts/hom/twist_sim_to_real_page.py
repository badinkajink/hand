#!/usr/bin/env python3
"""Result page of 2026-11-20: the twist layout ranking's predictions against the SR2 hand's trials (owner, 11-20 11:5x:
"deliver me a wonderful chart showing whatever sim2real landscape/ranking correspondence we've determined").

    .venv/bin/python scripts/hom/twist_sim_to_real_page.py

Refresh the trials first (corvus's records and hom_series's rows, then the scorer):
    rsync -a corvus:workspace/hand/logs/20261116-hom_trials/'2026112*_corvus.jsonl' \\
        corvus:workspace/hand/logs/20261116-hom_trials/20261120-series_morning_seed20261120.jsonl \\
        docs/experiments/20261115-hom_bench_trials/runs_corvus/
    .venv/bin/python docs/experiments/20261119-twist_layout_ranking/sim/20261120-trials_vs_prediction.py

Data: the twist ranking's data/20261120-trials_vs_prediction.json (the scorer), data/20261119-twist_ranking_summary.json
(the predicted landscape), data/20261120-feasible_ranking.json and, once written, data/20261120-feasible_ranking_f1.json
(each layout's best grasp whose simulated twist keeps the fingers 10 mm apart, at 2.5 N and at 1.0 N), and the bench
analyses analysis/20261120-command_jitter_replay.json and 20261120-command_jitter_sim.json. Films: media/films.json
lists clips copied into media/ (label, file), linked from the page.
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
RANK = ROOT / "docs/experiments/20261119-twist_layout_ranking"
BENCH = ROOT / "docs/experiments/20261115-hom_bench_trials"
D = ROOT / "docs/experiments/20261120-twist_sim_to_real"
PAGE = D / "20261120-twist_ranking_sim_to_real.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
TEX_RE = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S)
TEX_CACHE = D / "media/texsvg_cache.json"
# the dataviz skill's reference categorical palette, slots 1-6 in order, and its blue ramp for magnitude
BLOCK = {"A1": ("#2a78d6", "o", "Block A1, ranked layouts"), "A4feas": ("#eb6834", "s", "Block A4feas, feasible layouts"),
         "anchor": ("#1baf7a", "D", "Anchor (70, 110)"), "b3E": ("#eda100", "^", "Stage E, lower force"),
         "b4cem": ("#e87ba4", "v", "CEM plans"), "other": ("#008300", "P", "Touching or dropping layouts")}
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID = "#222222", "#555555", "#d9d9d9"
Y_WIDE = 150.0                     # mm: index-to-middle spacing from which the hand lost the sphere (2026-11-20)


def j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def esc(s):
    return html.escape(str(s), quote=True)


def png_uri(figure, dpi=110):
    import matplotlib.pyplot as plt
    b = io.BytesIO()
    figure.savefig(b, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, uri, caption):
    return (f'<figure id="fig{n}"><img src="{uri}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def table(head, body, num=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def f(x, nd=3):
    return "&ndash;" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def xy_s(xy):
    return f"({xy[0]:g}, {xy[1]:g})"


def block_of(x):
    """The plan's block for colour and marker: the trials' block labels, or the plan's role."""
    bl = x.get("blocks") or []
    if x.get("role") == "anchor" or any("_anchor_" in (t.get("label") or "") for t in x.get("trials") or []):
        return "anchor"
    for b in ("A4feas", "b3E", "b4cem"):
        if b in bl:
            return b
    if any(b in ("b2touch", "A2bad", "A3drop") for b in bl):
        return "other"
    if x.get("role", "").startswith("cem"):
        return "b4cem"
    if x.get("prm"):
        return "b3E"
    return "A1"


def setting(x):
    p = x.get("prm") or {}
    if x.get("role", "").startswith("cem"):
        return x["role"].replace("cem ", "CEM, ")
    if p:
        return f"{p.get('f_tripod', 2.5):g} N, spin weight {p.get('w_spin_tripod', 1.0):g}"
    return "2.5 N (default)"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def label_points(ax, placed, span, fs=10, others=()):
    """Labels beside points, each at the first of four offsets whose box clears the boxes and points placed before (a
    greedy placement in data units; a character is about 0.0108 of a 0.8-wide axis at 10 pt)."""
    cw, h = 0.0108 * span / 0.8, 0.03 * span / 0.8
    boxes = [(px - 0.012, py - 0.012, px + 0.012, py + 0.012) for px, py in [q[:2] for q in placed] + list(others)]
    for px, py, lab in sorted(placed, key=lambda q: (-q[1], q[0])):
        w = cw * len(lab)
        for dx, dy, ha in ((0.012, 0.006, "left"), (0.012, -0.036, "left"), (-0.012, 0.006, "right"),
                           (-0.012, -0.036, "right")):
            x0 = px + dx if ha == "left" else px + dx - w
            b = (x0, py + dy, x0 + w, py + dy + h)
            if not any(b[0] < c[2] and c[0] < b[2] and b[1] < c[3] and c[1] < b[3] for c in boxes):
                break
        boxes.append(b)
        ax.text(px + dx, py + dy, lab, ha=ha, va="bottom", fontsize=fs, color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def fig_scatter(T):
    """Predicted against measured tracking per tested plan, with its trials."""
    plt = plt_setup()
    tested = [x for x in T["plans"] if x["n"] > 0 and x["predicted_both"] is not None]
    fg, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=0)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    seen = set()
    placed = []
    for x in tested:
        b = block_of(x)
        c, mk, lab = BLOCK[b]
        px = x["predicted_both"]
        sc = x["scores"]
        lost = any(t["ended"] != "end of schedule" for t in x["trials"] if not t.get("tilted_start"))
        ax.scatter([px] * len(sc), sc, s=22, color=c, alpha=0.45, lw=0, zorder=2)
        sd = x["measured_sd"] or 0.0
        ax.errorbar(px, x["measured_mean"], yerr=sd if x["n"] > 1 else None, fmt=mk, ms=9, color=c,
                    mfc="white" if lost else c, mec=c, mew=1.8, elinewidth=1.5, capsize=0, zorder=3)
        seen.add(b)
        if x.get("role") == "anchor" and not x.get("prm") and not x["role"].startswith("cem"):
            placed.append((px, x["measured_mean"], f"{xy_s(x['xy'])} g{x['k']}"))
    worst6 = sorted([x for x in tested if x.get("role") != "anchor"],
                    key=lambda x: -abs(x["measured_mean"] - x["predicted_both"]))[:6]
    for x in worst6:
        lab = f"{xy_s(x['xy'])}" + (" CEM" if x["role"].startswith("cem") else f" g{x['k']}"
                                         + (" 1.0 N" if x.get("prm") else ""))
        placed.append((x["predicted_both"], x["measured_mean"], lab))
    lo = 0.0
    hi = max(0.8, max([x["predicted_both"] for x in tested] + [max(x["scores"]) for x in tested if x["scores"]] + [0.0])
             + 0.05)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    label_points(ax, placed, hi - lo, others=[(x["predicted_both"], x["measured_mean"]) for x in tested])
    ax.set_xlabel("Predicted tracking score (simulation)")
    ax.set_ylabel("Measured tracking score on the hand")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=BLOCK[b][1], ls="", color=BLOCK[b][0], ms=9, label=BLOCK[b][2]) for b in BLOCK
          if b in seen]
    hs.append(Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, mew=1.8, ms=9,
                     label="A trial lost the sphere"))
    ax.legend(handles=hs, loc="upper left", frameon=False, fontsize=11)
    return png_uri(fg)


def split_stats(default):
    """Predicted against measured over the default-force layouts, all of them and split at Y_WIDE."""
    from scipy import stats as st
    out = {}
    for lab, sel in (("all", lambda x: True), ("narrow", lambda x: x["xy"][1] < Y_WIDE),
                     ("wide", lambda x: x["xy"][1] >= Y_WIDE)):
        s = [x for x in default if sel(x)]
        if not s:
            continue
        pr = np.array([x["predicted_both"] for x in s])
        me = np.array([x["measured_mean"] for x in s])
        o = {"n": len(s), "bias": float(np.mean(me - pr)), "mae": float(np.mean(np.abs(me - pr)))}
        if len(s) > 2:
            o.update({"rho": float(st.spearmanr(pr, me)[0]), "r": float(np.corrcoef(pr, me)[0, 1])})
        out[lab] = o
    return out


def fig_spacing(default):
    """Measured less predicted tracking against the index-to-middle spacing, per default-force layout."""
    plt = plt_setup()
    fg, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axhline(0, color=GRID, lw=1.5, zorder=0)
    ax.axvline(Y_WIDE, color=INK2, lw=1.0, ls="--", zorder=0)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    seen = set()
    for x in default:
        b = block_of(x)
        c, mk, lab = BLOCK[b]
        lost = any(t["ended"] != "end of schedule" for t in x["trials"] if not t.get("tilted_start"))
        ax.scatter([x["xy"][1]], [x["measured_mean"] - x["predicted_both"]], s=70, marker=mk,
                   facecolors=c if not lost else "white", edgecolors=c, lw=1.8, zorder=3)
        seen.add(b)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=BLOCK[b][1], ls="", color=BLOCK[b][0], ms=9, label=BLOCK[b][2]) for b in BLOCK
          if b in seen]
    hs.append(Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, mew=1.8, ms=9,
                     label="A trial lost the sphere"))
    ax.set_xlabel("Index-to-middle spacing y (mm)")
    ax.set_ylabel("Measured less predicted score")
    ax.legend(handles=hs, loc="lower left", frameon=False, fontsize=11)
    return png_uri(fg)


def fig_gain(T):
    """Per refined plan (lower force, CEM), its gain over its layout's default plan: predicted against measured."""
    plt = plt_setup()
    rv = [x for x in T["refined_vs_default"] if x["n"] > 0 and x["default_measured_mean"] is not None]
    fg, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.axhline(0, color=GRID, lw=1.2, zorder=0)
    ax.axvline(0, color=GRID, lw=1.2, zorder=0)
    ax.plot([-0.1, 0.8], [-0.1, 0.8], color=GRID, lw=1.5, zorder=0)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for x in rv:
        b = "b4cem" if x["role"].startswith("cem") else "b3E"
        if x["role"] == "anchor":
            b = "anchor"
        c, mk, _ = BLOCK[b]
        ax.scatter([x["predicted_both"] - x["default_predicted_both"]], [x["measured_mean"] - x["default_measured_mean"]],
                   s=80, marker=mk, color=c, edgecolors="white", lw=1.2, zorder=3)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=BLOCK["b3E"][1], ls="", color=BLOCK["b3E"][0], ms=9, label="HOM at 1.0 N"),
          Line2D([], [], marker=BLOCK["b4cem"][1], ls="", color=BLOCK["b4cem"][0], ms=9, label="CEM force schedule"),
          Line2D([], [], marker=BLOCK["anchor"][1], ls="", color=BLOCK["anchor"][0], ms=9,
                 label="Anchor (70, 110) at 1.0 N")]
    ax.legend(handles=hs, loc="lower right", frameon=False, fontsize=11)
    ax.set_xlabel("Predicted gain over the default plan")
    ax.set_ylabel("Measured gain over the default plan")
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(-0.1, 0.8)
    ax.set_aspect("equal")
    return png_uri(fg)


def fig_landscape(S, T, F25, F10):
    """The predicted landscape over the lattice of layouts, with the tested layouts; the feasible landscapes beside."""
    plt = plt_setup()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    panels = [("(a) Predicted, 2.5 N", {tuple(c["xy"]): c["score"] for c in S["candidates"] if c["fits"]}, set())]
    for lab, F in (("(b) Best feasible grasp, 2.5 N", F25), ("(c) Best feasible grasp, 1.0 N", F10)):
        if F:
            panels.append((lab, {tuple(L["xy"]): (L["seeds012"] if L["seeds012"] is not None else L["seed0"])
                                 for L in F["layouts"] if L["k"] is not None},
                           {tuple(L["xy"]) for L in F["layouts"] if L["k"] is None}))
    allxy = [tuple(c["xy"]) for c in S["candidates"]]
    tested = {}
    for x in T["plans"]:
        if x["n"] > 0 and not x.get("prm") and not x.get("role", "").startswith("cem"):
            tested[tuple(x["xy"])] = x["measured_mean"]
    vmax = max(0.65, max(max((v for v in vals.values() if v is not None), default=0) for _, vals, _ in panels))
    plt.rcParams.update({"font.size": 14})
    fg, axs = plt.subplots(1, len(panels), figsize=(4.7 * len(panels) + 1.2, 5.4), squeeze=False, sharey=True)
    for i, (ax, (lab, vals, infeasible)) in enumerate(zip(axs[0], panels)):
        ax.scatter([p[0] for p in allxy], [p[1] for p in allxy], s=30, marker="s", color="#eeeeee", lw=0, zorder=1)
        k = [p for p in allxy if p in vals and vals[p] is not None]
        im = ax.scatter([p[0] for p in k], [p[1] for p in k], c=[vals[p] for p in k], cmap=cmap, vmin=0, vmax=vmax,
                        s=30, marker="s", lw=0, zorder=2)
        if infeasible:
            ax.scatter([p[0] for p in infeasible], [p[1] for p in infeasible], s=16, marker="x", color="#9a9a9a",
                       lw=1.0, zorder=2)
        for p, m in tested.items():
            ax.scatter([p[0]], [p[1]], s=150, facecolors="none", edgecolors=INK, lw=1.6, zorder=3)
            ax.annotate(f(m, 2), p, xytext=(8, -4), textcoords="offset points", fontsize=12, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=INK2, lw=0.6), zorder=4)
        ax.set_title(lab, fontsize=14, color=INK, loc="left")
        ax.set_xlabel("Separation x (mm)")
        if i == 0:
            ax.set_ylabel("Separation y (mm)")
        ax.set_aspect("equal")
    cb = fg.colorbar(im, ax=axs[0].tolist(), shrink=0.8, pad=0.02)
    cb.set_label("Tracking score")
    handles = [Line2D([], [], marker="s", ls="", color="#eeeeee", ms=8, label="Does not fit the hand"),
               Line2D([], [], marker="x", ls="", color="#9a9a9a", ms=7, label="No feasible grasp"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=10, label="Tried on the hand (measured mean)")]
    fg.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.45, -0.06), fontsize=13)
    return png_uri(fg)


def fig_jitter(RP, SM):
    """The share of control periods with a commanded reversal: on the hand (replayed) and in simulation."""
    plt = plt_setup()
    rs = [r for r in RP["records"] if r.get("budget_nm") == 0.56 and r.get("smooth_jt")]
    hand = [("Hand: sent, budget 0.56 N m", np.median([r["recorded"]["rev_pct"] for r in rs]), BLOCK["A1"][0]),
            ("Hand: HOM's own targets, no budget", np.median([r["raw"]["rev_pct"] for r in rs]), BLOCK["A4feas"][0]),
            ("Hand: budget on low-passed joints", np.median([r["smooth_j"]["rev_pct"] for r in rs]), BLOCK["anchor"][0]),
            ("Hand: targets low-passed too", np.median([r["smooth_jt"]["rev_pct"] for r in rs]), BLOCK["b3E"][0])]
    by = {}
    for r in SM["runs"]:
        if r.get("cmd"):
            by.setdefault(r["variant"], []).append(r["cmd"]["rev_pct"])
    sim = [(lab, np.median(by[v]), c) for v, lab, c in
           (("budget", "Simulation: exact joint readings", BLOCK["b4cem"][0]),
            ("budget_noise", "Simulation: hand's flicker added", BLOCK["A1"][0]),
            ("none_noise", "Simulation: flicker, no budget", BLOCK["A4feas"][0]),
            ("budget_noise_j", "Simulation: budget on low-passed joints", BLOCK["anchor"][0]),
            ("budget_noise_jt", "Simulation: targets low-passed too", BLOCK["b3E"][0])) if v in by]
    rows = hand + sim
    fg, ax = plt.subplots(figsize=(8.6, 0.48 * len(rows) + 1.2))
    yy = np.arange(len(rows))[::-1]
    ax.barh(yy, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62, zorder=2)
    for y, r in zip(yy, rows):
        ax.text(r[1] + 0.8, y, f"{r[1]:.0f} %", va="center", fontsize=11, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axhline(len(sim) - 0.5, color=GRID, lw=1.2)
    ax.set_xlabel("Periods with a commanded reversal (median, %)")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    keys_s = {"Simulation: exact joint readings": "exact", "Simulation: hand's flicker added": "flicker",
              "Simulation: flicker, no budget": "flicker_none", "Simulation: budget on low-passed joints": "j",
              "Simulation: targets low-passed too": "jt"}
    return png_uri(fg), {"hand": dict(zip(("sent", "own", "j", "jt"), (r[1] for r in hand))),
                         "sim": {keys_s[r[0]]: r[1] for r in sim}, "n_records": len(rs),
                         "n_plans": len(by.get("budget_noise", []))}


def films():
    """The films of media/films.json, linked from the page folder's media/, with their posters inlined."""
    F = j(D / "media/films.json") or []
    out = []
    for x in F:
        poster = D / "media" / x.get("poster", "")
        pu = ("data:image/jpeg;base64," + base64.b64encode(poster.read_bytes()).decode()) if poster.is_file() else ""
        out.append(f'<figure><video controls muted loop playsinline preload="metadata" poster="{pu}" '
                   f'src="media/{esc(x["file"])}" style="width:100%"></video>'
                   f"<figcaption>{esc(x['label'])}</figcaption></figure>")
    return "".join(out)


def page():
    import docs_index
    import retro_style
    import texsvg
    T = j(RANK / "data/20261120-trials_vs_prediction.json")
    S = j(RANK / "data/20261119-twist_ranking_summary.json")
    F25 = j(RANK / "data/20261120-feasible_ranking.json")
    F10 = j(RANK / "data/20261120-feasible_ranking_f1.json")
    RP = j(BENCH / "analysis/20261120-command_jitter_replay.json")
    RM = j(BENCH / "analysis/20261120-recovery_metrics.json")
    FP = j(RANK / "data/20261120-friction_predictions.json")
    SM = j(BENCH / "analysis/20261120-command_jitter_sim.json")
    tested = [x for x in T["plans"] if x["n"] > 0]
    default = [x for x in tested if x["role"] in ("ranked", "anchor") and not x.get("prm")]
    rho = T["rho"]["both models"]
    n_trials = T["trials_scored"]
    tilted = T.get("tilted_starts") or []
    anchor = next((x for x in tested if tuple(x["xy"]) == (70.0, 110.0) and not x.get("prm")), None)
    worst = min(default, key=lambda x: x["measured_mean"] - x["predicted_both"]) if default else None
    D.mkdir(parents=True, exist_ok=True)
    sec = []

    sec.append(("terms", "Terms and metrics", r"""<ul>
<li><b>Layout</b>: the SR2 hand's finger mounts, given as the thumb-to-finger separation x and the index-to-middle
separation y in millimetres of the palm frame P (<code>sr2_layouts.sym6</code>). <b>Grasp</b> g: one of each layout's
ground-clear grasps of the 40 mm sphere, numbered as in the ranking.</li>
<li><b>Twist</b>: the task of the trials. HOM, the hand's controller, lifts the sphere and turns it about the palm normal
through the commanded schedule (a ramp of the turn from 3.3 s, held from 6.3 s).</li>
<li><b>Tracking score</b>: \(s = \max\left(0,\ 1 - e_{\mathrm{rms}} / 59.4^\circ\right)\), with \(e_{\mathrm{rms}}\) the
RMS angle between the sphere's tracked rotation and the commanded one from 2.0 s to the schedule's end. The command's own
RMS is 59.4&deg;, so a sphere held still scores 0. A trial or run that drops the sphere, or that corvus's drift guard
stops, scores 0.</li>
<li><b>Prediction</b>: the mean score over the simulation's two contact models (point contact and the 1 mm sphere pads)
at seeds 0-2. The plant is the calibrated one of 2026-11-17: 7.5 V servo gains, non-backdrivable gears, MCP stops and the
bench's torque budget. Every prediction was written before the first trial of its plan.</li>
<li><b>Measured score</b>: the mean over a plan's trials on the hand. A trial whose sphere started more than 5&deg; from the
plan's grasp is listed apart and left out (Table 2).</li>
<li><b>&rho;</b>: Spearman's rank correlation between predicted and measured scores over the layouts at the default force
(2.5 N), with a 95 % interval from resampling each layout's trials (5,000 draws).</li>
<li><b>Commanded reversal share</b>: the share of 20 ms control periods in which a joint's command changes direction
(steps over 0.1&deg;), over the nine joints.</li>
<li><b>Camera</b>: the tag tracker ran at 1920 &times; 1080 and 30 frames/s with a 16.6 ms exposure until 11:5x on
2026-11-20. From then it ran at 960 &times; 540 and 60 frames/s with a 5 ms exposure.</li>
</ul>"""))

    rows = []
    for x in sorted(tested, key=lambda x: -(x["predicted_both"] or 0)):
        tr = ", ".join(f"{f(t['track_held'], 2)}" + ("" if t["ended"] == "end of schedule" else f" ({esc(t['ended'])})")
                       for t in x["trials"] if not t.get("tilted_start"))
        rows.append([f"{xy_s(x['xy'])} g{x['k']}", esc(setting(x)), esc(", ".join(x.get("blocks") or [])),
                     f(x["predicted_both"]), f(x["measured_mean"]), f(x["measured_sd"]), str(x["n"]),
                     f(x.get("measured_mean_60")), tr])
    cap1 = (f"Predicted against measured tracking for the {len(tested)} plans tried on the hand ({n_trials} trials). "
            "Large marks: a plan's mean over its trials, with the standard deviation over trials as a bar. Small "
            "marks: single trials. Hollow marks: the sphere was lost in at least one trial. Labelled: the anchor at "
            "the default force and the six largest misses. The grey line is equality.")
    rho_txt = (f"Over the {rho['layouts']} layouts at the default force, &rho; = {f(rho.get('rho'), 2)} "
               f"(95 % interval {rho.get('rho_trials_resampled_95')}), and the measured scores lay "
               f"{f(abs(rho.get('mean_measured_minus_predicted') or 0), 3)} "
               f"{'below' if (rho.get('mean_measured_minus_predicted') or 0) < 0 else 'above'} the predictions on "
               "average." if rho.get("rho") is not None else
               f"The {rho['layouts']} layouts tried so far at the default force are too few for &rho;.")
    sp = split_stats(default)
    split_txt = ""
    if sp.get("narrow") and sp.get("wide") and "r" in sp["narrow"]:
        nw, wd = sp["narrow"], sp["wide"]
        split_txt = (f"<p>The misses depend on the index-to-middle spacing y. For the {nw['n']} layouts with y under "
                     f"{Y_WIDE:g} mm, Pearson's r between predicted and measured scores is {f(nw['r'], 2)}, "
                     f"&rho; is {f(nw.get('rho'), 2)} and the mean absolute error is {f(nw['mae'])}. With the "
                     f"{wd['n']} layouts at {Y_WIDE:g} mm or more, the hand lost the sphere, in drops or at corvus's "
                     f"drift guard. Their scores fell {f(abs(wd['bias']))} below the predictions on average (Figure 2). "
                     "No such fall with the spacing was predicted.</p>")
    cap1b = (f"Measured less predicted score per layout at the default force, against the index-to-middle spacing. "
             f"Hollow marks: the sphere was lost in at least one of the layout's trials. The dashed line is {Y_WIDE:g} mm.")
    sec.append(("scatter", "Predicted and measured tracking per layout",
                f"<p>{rho_txt}</p>" + fig(1, fig_scatter(T), cap1) + split_txt + fig(2, fig_spacing(default), cap1b)
                + table(["Layout, grasp", "HOM's force", "Block", "Predicted", "Measured", "SD", "Trials",
                         "Measured at 60 fps", "Trial scores"], rows, num=(3, 4, 5, 6, 7), n=1,
                        caption="Each plan tried on the hand. Trial scores in order; a trial that ended early names "
                                "its ending.")))

    rv = [x for x in T["refined_vs_default"] if x["n"] > 0 and x["default_measured_mean"] is not None]
    if rv:
        dp = np.array([x["predicted_both"] - x["default_predicted_both"] for x in rv])
        dm = np.array([x["measured_mean"] - x["default_measured_mean"] for x in rv])
        agree = int(np.sum(np.sign(dp) == np.sign(dm)))
        an = next((x for x in rv if x["role"] == "anchor"), None)
        gain_txt = (f"<p>The lower-force plans of stage E ran HOM at 1.0 N in place of 2.5 N. The CEM plans ran a force "
                    f"schedule refined in simulation. Each ran on a layout whose default plan also ran. On {agree} of these "
                    f"{len(rv)} plans the hand's score moved in the predicted direction. The mean gain was "
                    f"{f(float(dm.mean()))} on the hand against {f(float(dp.mean()))} predicted. The size of each plan's "
                    f"gain was not predicted: Pearson's r between predicted and measured gains is "
                    f"{f(float(np.corrcoef(dp, dm)[0, 1]), 2)} (Figure 3).")
        if an:
            gain_txt += (f" The anchor (70, 110) rose from {f(an['default_measured_mean'])} to {f(an['measured_mean'])} at "
                         f"1.0 N; {f(an['default_predicted_both'])} and {f(an['predicted_both'])} were predicted.")
        gain_txt += "</p>"
        sec.append(("gain", "The lower force and the CEM plans on the hand", gain_txt + fig(
            3, fig_gain(T), "The gain of each lower-force and CEM plan over its layout's default plan at 2.5 N: "
                            "predicted against measured. The grey diagonal is equality.")))
    # -------------------------------------------------- plant variants against the hand
    pv_rows = []
    names_pl = {"pads": "Pads, friction 1.0", "pads_mu0.6": "Pads, friction 0.6", "pads_stiff": "Pads, stiff",
                "point": "Point contact, friction 1.0", "point_mu0.6": "Point contact, friction 0.6",
                "point_5v": "Point contact, 5 V servos"}
    for m, v in sorted((T.get("rho_per_plant") or {}).items(), key=lambda kv: -(kv[1].get("rho") or -9)):
        pv_rows.append([esc(names_pl.get(m, m)), "stage D, default plans", str(v.get("layouts")), f(v.get("rho"), 2),
                        f(v.get("pearson_r"), 2), f(v.get("mean_abs_error")), f(v.get("mean_measured_minus_predicted"))])
    fr_best = None
    if FP:
        from scipy import stats as _st
        meas = {x["plan"]: x for x in T["plans"] if x["n"] > 0}
        for v in FP["variants"]:
            pts = [(p_["variants"][v]["both"], meas[p_["plan"]]["measured_mean"], meas[p_["plan"]]["xy"][1])
                   for p_ in FP["plans"] if p_["plan"] in meas and (p_["variants"].get(v) or {}).get("both") is not None]
            if len(pts) < 3:
                continue
            pr_ = np.array([q_[0] for q_ in pts])
            me_ = np.array([q_[1] for q_ in pts])
            wide = np.array([q_[2] >= Y_WIDE for q_ in pts])
            row = {"v": v, "n": len(pts), "rho": float(_st.spearmanr(pr_, me_)[0]),
                   "r": float(np.corrcoef(pr_, me_)[0, 1]), "mae": float(np.mean(np.abs(me_ - pr_))),
                   "bias": float(np.mean(me_ - pr_)),
                   "wide_bias": float(np.mean((me_ - pr_)[wide])) if wide.any() else None}
            if fr_best is None or row["mae"] < fr_best["mae"]:
                fr_best = row
            pv_rows.append([f"Both models, friction {FP['variants'][v].get('mu', 1.0):g}", "sweep, all plans but CEM",
                            str(row["n"]), f(row["rho"], 2), f(row["r"], 2), f(row["mae"]), f(row["bias"])])
    if pv_rows:
        txt = ("<p>The same plans were simulated under other plants. The six plants of the ranking's stage D were run for "
               "the layouts of the morning list at the default force. The friction sweep ran every plan but the CEM "
               "plans at five scene frictions. Each ran under both contact models at seeds 0-2.")
        if fr_best:
            txt += (f" Of the frictions, {FP['variants'][fr_best['v']].get('mu', 1.0):g} came closest to the hand, with "
                    f"a mean absolute error of {f(fr_best['mae'])} and &rho; = {f(fr_best['rho'], 2)}.")
            if fr_best.get("wide_bias") is not None:
                txt += (f" At that friction the layouts {Y_WIDE:g} mm wide or more still scored "
                        f"{f(abs(fr_best['wide_bias']))} {'below' if fr_best['wide_bias'] < 0 else 'above'} "
                        "their predictions.")
            txt += (" The UR5 rig measured 0.37-0.44 for the TPU pads on bare PLA (the v3 tips' contact model of "
                    "2026-10-30). The scenes' 1.0 stands for the grip tape of the bench tools, and the sphere has none.")
        txt += "</p>"
        sec.append(("plants", "Plant variants against the hand", txt + table(
            ["Plant", "Set", "Plans", "&rho;", "Pearson's r", "Mean absolute error", "Measured less predicted"],
            pv_rows, num=(2, 3, 4, 5, 6), n=2,
            caption="Each plant's predictions against the hand's measured scores over the plans tried.")))
    cap2 = ("The lattice of layouts: x is the thumb-to-finger separation and y the index-to-middle separation, in mm "
            "of the palm frame. (a) The ranking's predicted score at each fitting layout's best grasp, HOM at its "
            "default 2.5 N. (b, c) The score of each layout's best feasible grasp at 2.5 N and at 1.0 N. A grasp is "
            "feasible when its simulated twist keeps every pair of fingers 10 mm apart on the CAD collision spheres "
            "under both contact models, with the sphere held. The layouts tried on the hand at the default force are "
            "ringed and labelled with their measured mean score.")
    land = ("<p>The trials sample the predicted landscape across its range, from layouts whose fingers touch in "
            "simulation to the clear layouts of block A4feas.</p>")
    if F25 and F10:
        from scipy import stats as _st
        a = {tuple(L["xy"]): L for L in F25["layouts"]}
        b = {tuple(L["xy"]): L for L in F10["layouts"]}
        both = [xy for xy in a if xy in b and a[xy]["k"] is not None and b[xy]["k"] is not None]
        sa = np.array([a[xy]["seeds012"] if a[xy]["seeds012"] is not None else a[xy]["seed0"] for xy in both])
        sb = np.array([b[xy]["seeds012"] if b[xy]["seeds012"] is not None else b[xy]["seed0"] for xy in both])
        n25 = sum(1 for L in F25["layouts"] if L["k"] is not None)
        n10 = sum(1 for L in F10["layouts"] if L["k"] is not None)
        top = sorted(F10["layouts"], key=lambda L: -((L["seeds012"] if L["seeds012"] is not None else L["seed0"]) or 0))[:5]
        top25 = sorted(F25["layouts"], key=lambda L: -((L["seeds012"] if L["seeds012"] is not None else L["seed0"]) or 0))[:5]
        land += (f"<p>With HOM's force lowered to 1.0 N, {n10} of the {len(F10['layouts'])} fitting layouts have a "
                 f"feasible grasp, against {n25} at 2.5 N. Over the {len(both)} layouts feasible at both forces, the mean "
                 f"score rises from {f(float(sa.mean()))} to {f(float(sb.mean()))}. The two forces rank these layouts "
                 f"with &rho; = {f(float(_st.spearmanr(sa, sb)[0]), 2)}. The five best feasible layouts at 1.0 N are "
                 + ", ".join(f"{xy_s(L['xy'])}" for L in top)
                 + f", scoring {f(top[-1]['seeds012'], 2)} to {f(top[0]['seeds012'], 2)}. At 2.5 N they were "
                 + ", ".join(f"{xy_s(L['xy'])}" for L in top25)
                 + f", scoring {f(top25[-1]['seeds012'], 2)} to {f(top25[0]['seeds012'], 2)}.</p>")
    sec.append(("landscape", "The ranking's landscape and the tested layouts",
                land + fig(4, fig_landscape(S, T, F25, F10), cap2)))

    if RP and SM:
        uri, J = fig_jitter(RP, SM)
        h, s = J["hand"], J["sim"]
        sec.append(("jitter", "Commanded jitter on the hand", f"""<p>The hand's joint readings flicker by one or two encoder
steps of 0.29&deg;, about 0.5&deg; RMS about a 7-period mean. The torque budget sets each target as the measured joint
plus an offset, so the flicker passes into the commands. Over {J['n_records']} trials under the 0.56 N m budget, the
commands sent reversed in {h['sent']:.0f} % of periods (median). HOM's own targets on the
same readings, replayed offline without the budget, reversed in {h['own']:.0f} %. The
simulation's joint readings are exact, and it predicted {s.get('exact', float('nan')):.0f} %.
With the hand's flicker added, it reproduced {s.get('flicker', float('nan')):.0f} %.</p>
<p>Two server options were added on 2026-11-20 at 12:1x, off by default. With the first, the budget's offsets are
taken from joints low-passed at 0.1 s; with the second, HOM's targets are low-passed at 0.04 s. Replayed on the hand's
readings, the two together lower the share to
{h['jt']:.0f} %; in closed-loop simulation with the flicker, to
{s.get('jt', float('nan')):.0f} % at the same tracking score. Whether the trials use them is
recorded in each server record's header.</p>""" + fig(5, uri, "The share of control periods in which a joint's command "
                                                         "reversed. Upper bars: the hand's trials under the 0.56 N m "
                                                         "budget, as sent and replayed offline; lower bars: the "
                                                         f"simulation of {J['n_plans']} plans at seed 0.")))

    if tilted:
        rows2 = [[esc(t.get("label") or t["stamp"]), f(t["ready_tilt_deg"], 1), f(t["track_held"], 2), esc(t["ended"])]
                 for t in tilted]
        sec.append(("apart", "Trials set apart", "<p>A putdown stood a tilted sphere up by tilting the hand, and the "
                    "next trial's sphere then started tilted. Since 11:12 on 2026-11-20, hom_series levels the hand "
                    "before every recentre.</p>" + table(["Trial", "Tilt at start (deg)", "Score", "Ending"], rows2,
                                                       num=(1, 2), n=3, caption="Trials whose sphere started more "
                                                       "than 5&deg; from the plan's grasp.")))

    if RM:
        pdn = RM["putdowns"]
        lost = RM["sphere_lost_in_trial"]

        def pc(k):
            v = pdn.get(k) or {"count": 0, "of": RM["trials"]}
            return f"{v['count']} ({100 * v['count'] / max(1, v['of']):.0f} %)"
        rows3 = [["Trials", str(RM["trials"]), ""],
                 ["Sphere left the grasp in the trial (drift guard)", f"{lost['count']} ({100 * lost['share']:.0f} %)", ""],
                 ["Putdown, clean release", pc("clean"), ""],
                 ["Putdown stopped, finished by the lab session", pc("stopped, finished by the session"), ""],
                 ["Putdown, tilted release", pc("tilted release"), ""],
                 ["Pick-up of a lying sphere, all grasps", f"{RM['pickup_grasps']['count']} of {RM['pickup_grasps']['of']}",
                  "succeeded"],
                 ["Pick-up with a push first", f"{RM['pickup_with_push_first']['count']} of "
                  f"{RM['pickup_with_push_first']['of']}", "succeeded"],
                 ["Pick-up without a push", f"{RM['pickup_without_push']['count']} of {RM['pickup_without_push']['of']}",
                  "succeeded"],
                 ["Blind pinch (the tracker only before the descent)", f"{(RM.get('pickup_blind') or {}).get('count', 0)} of "
                  f"{(RM.get('pickup_blind') or {}).get('of', 0)}", "succeeded"],
                 ["Pick-ups that found no plan", str(RM["pickup_planner_refusals"]), ""]]
        sec.append(("bench", "Trial losses, putdowns and pick-ups on the bench", f"""<p>The series ran its trials one after
another without a person at the bench for most of the day. When a trial lost the sphere or a putdown stopped, the lab
session finished the putdown or picked the sphere up with the hand. A pick-up succeeded when the ball lay on bare table.
The cables across the table caught pushes and grasps, and the tracker saw a lying sphere with one or two tags.</p>"""
                    + table(["Event", "Count", ""], rows3, num=(1,), n=4,
                            caption=f"Bench events of 2026-11-20 up to {RM['generated'][11:]}, from hom_series's rows, "
                                    "its log and the pick-up logs.")))

    sec.append(("data", "Data and scripts", """<ul>
<li>Trials: the HOM server's records <code>docs/experiments/20261115-hom_bench_trials/runs/</code>, corvus's records and
hom_series's rows <code>runs_corvus/</code>; scored by
<code>docs/experiments/20261119-twist_layout_ranking/sim/20261120-trials_vs_prediction.py</code>.</li>
<li>Predictions: <code>docs/experiments/20261119-twist_layout_ranking/data/20261119-trial_plans.json</code>, written
before the trials; the landscape <code>data/20261119-twist_ranking_summary.json</code>; the feasible landscapes
<code>data/20261120-feasible_ranking.json</code> and <code>_f1.json</code>.</li>
<li>Jitter: <code>docs/experiments/20261115-hom_bench_trials/analysis/20261120-command_jitter_replay.py</code> and
<code>20261120-command_jitter_sim.py</code>.</li>
<li>Bench events: <code>docs/experiments/20261115-hom_bench_trials/analysis/20261120-recovery_metrics.py</code> and
<code>20261120-series_time_table.py</code>.</li>
<li>Decisions: <code>docs/notes/20261120-twist_ranking_decisions.md</code>.</li>
</ul>"""))
    sec.append(("future", "Future work", """<ul>
<li>Re-tune HOM's force at friction 0.4 (<code>inner_loop_mpc.py run --method hom --plant-mu 0.4</code> over the force
grid) and re-rank the 404 layouts there (<code>sim/20261120-feasible_grasps.py --mu 0.4 --prm ...</code>). The friction
calibration holds if the next trials of the new top layouts score within 0.1 of their predictions.</li>
<li>Measure the friction between a TPU pad and the printed sphere on the UR5 rig; the fit says about 0.4.</li>
<li>Run a block with the smoothed budget (<code>hom_server --budget-joint-tau 0.1 --target-tau 0.04</code>) against its
predictions in <code>data/20261120-smoothed_predictions.json</code>. The smoothing is borne out if the same layouts'
measured scores rise or their spread falls.</li>
<li>Route the cables away from the table's centre and count the pick-ups again; the pick-up on bare table is expected to
succeed every time.</li>
</ul>"""))

    allp = [x for x in tested if x["predicted_both"] is not None]
    if len(allp) > 2:
        from scipy import stats as _st
        pa = np.array([x["predicted_both"] for x in allp])
        ma = np.array([x["measured_mean"] for x in allp])
        lede = (f"<p>Over all {len(allp)} plans tried on the SR2 hand ({n_trials} trials), the measured tracking "
                f"correlated with the simulation's predictions at &rho; = {f(float(_st.spearmanr(pa, ma)[0]), 2)} "
                f"(Pearson's r {f(float(np.corrcoef(pa, ma)[0, 1]), 2)}). ")
        rvl = [x for x in T["refined_vs_default"] if x["n"] > 0 and x["default_measured_mean"] is not None]
        if rvl:
            g_m = float(np.mean([x["measured_mean"] - x["default_measured_mean"] for x in rvl]))
            g_p = float(np.mean([x["predicted_both"] - x["default_predicted_both"] for x in rvl]))
            lede += (f"With HOM's lower force and the CEM force schedules, the hand's score rose over each layout's "
                     f"default plan by {f(g_m)} on average, against {f(g_p)} predicted. ")
    elif default and rho.get("rho") is not None:
        lede = (f"<p>On the SR2 hand, the measured tracking of {rho['layouts']} layouts at HOM's default force "
                f"correlated with the simulation's predictions at &rho; = {f(rho['rho'], 2)}. ")
    else:
        lede = f"<p>{len(tested)} plans of the twist ranking have run on the SR2 hand so far ({n_trials} trials). "
    if FP:
        _best = None
        _meas = {x["plan"]: x["measured_mean"] for x in T["plans"] if x["n"] > 0}
        for v in FP["variants"]:
            pts = [(p_["variants"][v]["both"], _meas[p_["plan"]]) for p_ in FP["plans"]
                   if p_["plan"] in _meas and (p_["variants"].get(v) or {}).get("both") is not None]
            if len(pts) > 2:
                mae = float(np.mean([abs(b_ - a_) for a_, b_ in pts]))
                if _best is None or mae < _best[1]:
                    _best = (v, mae, len(pts), float(np.corrcoef([a_ for a_, _ in pts], [b_ for _, b_ in pts])[0, 1]))
        if _best:
            lede += (f"The simulation's scene friction of 1.0 stands for grip tape; the TPU pads on bare PLA measured "
                     f"about {FP['variants'][_best[0]].get('mu', 1.0):g}. At that friction, the predictions of {_best[2]} "
                     f"plans and the hand's scores correlated at Pearson's r {f(_best[3], 2)}, with a mean absolute error "
                     f"of {f(_best[1])} (Table 2). ")
    sp0 = split_stats(default) if default else {}
    if sp0.get("narrow") and sp0.get("wide") and "r" in sp0["narrow"]:
        lede += (f"At the default force and friction 1.0, the hand lost the sphere with the {sp0['wide']['n']} layouts "
                 f"whose index and middle fingers sit {Y_WIDE:g} mm apart or more. Their scores fell "
                 f"{f(abs(sp0['wide']['bias']))} below those predictions. ")
    elif anchor:
        lede += (f"The anchor layout (70, 110) scored {f(anchor['measured_mean'])} against its prediction of "
                 f"{f(anchor['predicted_both'])}. ")
    lede += ("HOM's commands chattered on the hand, because the torque budget passed the joint readings' flicker into "
             "them. The simulation's joint readings are exact, and smooth commands were predicted.</p>")
    summary = re.sub(r"<[^>]+>", "", lede).replace("&rho;", "rho").strip()
    toc = "".join(f'<li><a href="#{k}">{t_}</a></li>' for k, t_, _ in sec)
    body = "".join(f'<section id="{k}"><h2>{t_}</h2>{s_}</section>' for k, t_, s_ in sec)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(body)]
    TEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    svgs = iter(texsvg.render(items, cache_path=str(TEX_CACHE), scale=1.1))
    body = TEX_RE.sub(lambda m: next(svgs), body)
    title = "Simulated and measured tracking of the twist layout ranking on the SR2 hand"
    out = (TEMPLATE.read_text()
           .replace("{{TITLE}}", title)
           .replace("{{TEXCSS}}", texsvg.CSS)
           .replace("{{EYEBROW}}", "HOM on the SR2 hand · 2026-11-20")
           .replace("{{H1}}", title)
           .replace("{{LEDE}}", lede)
           .replace("{{BYLINE}}", "Workstation session hand-4a, 2026-11-20; trials by the lab session on corvus")
           .replace("{{FILM}}", films())
           .replace("{{TOC}}", toc)
           .replace("{{SECTIONS}}", body)
           .replace("{{FOOTER}}", "Built by scripts/hom/twist_sim_to_real_page.py from the twist ranking's data and the "
                                  "bench trials."))
    out = retro_style.apply(out)
    out = docs_index.tag(out, topic="design", summary=summary)
    PAGE.write_text(out)
    print(PAGE, f"{PAGE.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    page()

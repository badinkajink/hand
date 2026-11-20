#!/usr/bin/env python3
"""Result page of 2026-11-19: the sphere twist's layouts ranked on the calibrated SR2 simulation, the ranking's
agreement across contact models, seeds and plants, HOM's forces per layout, the short-window gap between the simulated
plant and the hand's twists, and the plans for testing the ranking on the hand (owner, 2026-11-19 17:2x).

    .venv/bin/python scripts/hom/twist_ranking_page.py

Data (docs/experiments/20261119-twist_layout_ranking/data/): 20261119-twist_ranking_summary.json and
20261119-trial_proposal.json (sim/20261119-twist_ranking_report.py over the rows of sim/20261119-twist_ranking_search.py),
20261119-force_refine.json (sim/20261119-twist_force_refine.py), 20261119-trial_plans.json (sim/20261119-trial_plans.py),
20261119-segment_gap.json (written here from logs/20261119-twist_layout_ranking/gap/gap_rows.jsonl of
sim/20261119-segment_gap.py when that file exists on this machine).
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/experiments/20261119-twist_layout_ranking"
DATA = D / "data"
PAGE = D / "20261119-sphere_twist_layout_ranking.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
GAP_ROWS = ROOT / "logs/20261119-twist_layout_ranking/gap/gap_rows.jsonl"
GAP = DATA / "20261119-segment_gap.json"
TEX_RE = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S)
TEX_CACHE = D / "media/texsvg_cache.json"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#3a9e5f", "#8a5cc2"     # the reference palette's categorical slots
PLANT_NAMES = {"pads_stiff": "stiff pads (impratio 100)", "point_mu0.6": "point contact, friction 0.6",
               "pads_mu0.6": "pads, friction 0.6", "point_5v": "point contact, servos at 5 V"}


def j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def esc(s):
    return html.escape(str(s), quote=True)


def png_uri(figure, dpi=110):
    import matplotlib.pyplot as plt
    b = io.BytesIO()
    figure.savefig(b, format="png", dpi=dpi, bbox_inches="tight")
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


def f(x, nd=2):
    return "&ndash;" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def xy_s(xy):
    return f"({xy[0]:g}, {xy[1]:g})"


# ------------------------------------------------------------------ data
def gap_data():
    """The segment gap's summary per plant and the windows of the plant used for the ranking (pads, 7.5 V, friction
    1.0), from the rows when they are on this machine; written to data/ so the page rebuilds elsewhere."""
    if GAP_ROWS.exists():
        sys.path.insert(0, str(D / "sim"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("segment_gap", D / "sim/20261119-segment_gap.py")
        SG = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(SG)
        summ = SG.summary(GAP_ROWS)
        win = []
        for x in GAP_ROWS.read_text().splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            if "error" in r or r["contact"] not in ("pads", "point") or r["volts"] != "7v5" or r["mu"] != 1.0:
                continue
            for s in r["segments"]:
                if not s["diverged"]:
                    win.append([r["contact"], r["stamp"], s["t0"], s["phase"], s["twist_hand"], s["twist_sim"],
                                s["rot_err"], s["settle_mm"]])
        GAP.write_text(json.dumps({"what": "Short-window gap of the 20 g0 twists of 2026-11-17 (sim/20261119-segment_gap.py)",
                                   "summary": summ, "windows_cols": ["contact", "trial", "t0", "phase", "twist_hand",
                                                                      "twist_sim", "rot_err", "settle_mm"],
                                   "windows": win}, indent=0) + "\n")
    return j(GAP)


# ------------------------------------------------------------------ figures
def fig_landscape(S, prop):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(40, 161, 5.0)
    ys = np.arange(50, 171, 5.0)
    fg, axs = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    picks = {tuple(p["xy"]): p for p in prop["pick"]} if prop else {}
    for ax, m, name in zip(axs, ("point", "pads"), ("point contact", "sphere pads")):
        Z = np.full((len(ys), len(xs)), np.nan)
        for L in S["per_layout"]:
            if m in L:
                i, k = int(round((L["xy"][1] - 50) / 5)), int(round((L["xy"][0] - 40) / 5))
                Z[i, k] = L[m]["score"]
        im = ax.imshow(Z, origin="lower", extent=(37.5, 162.5, 47.5, 172.5), cmap="viridis", vmin=0,
                       vmax=np.nanmax(Z) if np.isfinite(Z).any() else 1, aspect="equal")
        for (x, y), p in picks.items():
            ax.plot(x, y, "o", mfc="none", mec="w", mew=2.0, ms=11)
            ax.text(x + 3, y + 2, str(p["rank_among_deployable"]), color="w", fontsize=12, weight="bold")
        ax.plot(70, 110, "s", mfc="none", mec=C2, mew=2.0, ms=11)
        ax.set_title(f"Best grasp's tracking, {name}", fontsize=15)
        ax.set_xlabel("x_sep (mm)", fontsize=14)
        ax.tick_params(labelsize=13)
    axs[0].set_ylabel("y_sep (mm)", fontsize=14)
    cb = fg.colorbar(im, ax=axs, shrink=0.85, pad=0.02)
    cb.set_label("tracking, mean of seeds 0-2", fontsize=14)
    cb.ax.tick_params(labelsize=13)
    return png_uri(fg)


def fig_agreement(S):
    import matplotlib.pyplot as plt
    pl = {L["call"]: L for L in S["per_layout"] if "point" in L and "pads" in L}
    fg, axs = plt.subplots(1, 2, figsize=(11.0, 4.8))
    a = np.array([[pl[c]["point"]["score"], pl[c]["pads"]["score"]] for c in pl])
    if len(a):
        axs[0].plot(a[:, 0], a[:, 1], ".", color=C1, ms=6)
        lim = max(1e-3, float(a.max()) * 1.05)
        axs[0].plot([0, lim], [0, lim], "-", color="0.6", lw=1)
    axs[0].set_xlabel("tracking, point contact", fontsize=14)
    axs[0].set_ylabel("tracking, sphere pads", fontsize=14)
    axs[0].set_title("Each layout under both contact models", fontsize=15)
    for m, c in (("point", C1), ("pads", C2)):
        b = np.array([[L[m]["score"], L[m]["fresh"]] for L in S["per_layout"] if m in L and L[m].get("fresh") is not None])
        if len(b):
            axs[1].plot(b[:, 0], b[:, 1], "o", color=c, ms=5, label=m)
    lo, hi = axs[1].get_xlim()
    axs[1].plot([lo, hi], [lo, hi], "-", color="0.6", lw=1)
    axs[1].set_xlabel("tracking, seeds 0-2 (selection)", fontsize=14)
    axs[1].set_ylabel("tracking, seeds 10-12 (fresh)", fontsize=14)
    axs[1].set_title("The best layouts at fresh seeds", fontsize=15)
    axs[1].legend(fontsize=13)
    for ax in axs:
        ax.tick_params(labelsize=13)
    return png_uri(fg)


def fig_gap(G):
    import matplotlib.pyplot as plt
    W = [w for w in G["windows"] if w[0] == "pads"]
    fg, axs = plt.subplots(1, 2, figsize=(11.0, 5.2))
    for ph, c in (("grasp", C3), ("task", C1)):
        a = np.array([[w[4], w[5]] for w in W if w[3] == ph])
        if len(a):
            axs[0].plot(a[:, 0], a[:, 1], ".", color=c, ms=6, label=f"{ph} windows")
    lo = min(axs[0].get_xlim()[0], axs[0].get_ylim()[0])
    hi = max(axs[0].get_xlim()[1], axs[0].get_ylim()[1])
    axs[0].plot([lo, hi], [lo, hi], "-", color="0.6", lw=1)
    axs[0].set_xlabel("twist over the window, hand (°)", fontsize=14)
    axs[0].set_ylabel("twist over the window, simulation (°)", fontsize=14)
    axs[0].set_title("0.5 s windows from the hand's state, pads", fontsize=15)
    axs[0].legend(fontsize=13)
    rows = sorted(((v["task"]["twist_err_rms"], k) for k, v in G["summary"].items()
                   if v.get("task") and v["task"]["windows"] >= 200), reverse=True)
    axs[1].barh([k for _, k in rows], [r for r, _ in rows], color=C1)
    axs[1].set_xlabel("twist error over task windows, RMS (°)", fontsize=14)
    axs[1].set_title("Per plant", fontsize=15)
    for ax in axs:
        ax.tick_params(labelsize=13)
    return png_uri(fg)


def fig_forces(FR):
    import matplotlib.pyplot as plt
    fg, ax = plt.subplots(figsize=(9.0, 5.0))
    cols = [C1, C2, C3, C4, "#c23a3a", "#3ab0b0", "#b08a3a", "#666666", "#d6a02a", "#5a7a2a", "#2a2a8a", "#aa44aa"]
    for i, L in enumerate(FR["layouts"]):
        w = L["best_prm"]["w_spin_tripod"]
        pts = []
        for k, v in L["grid"].items():
            p = json.loads(k)
            if abs(p["w_spin_tripod"] - w) < 1e-9:
                pts.append((p["f_tripod"], v))
        pts.sort()
        if pts:
            a = np.array(pts)
            ax.plot(a[:, 0], a[:, 1], "-o", color=cols[i % len(cols)], ms=4, lw=1.5,
                    label=f"{xy_s(L['xy'])} g{L['k']}, w {w:g}")
    ax.axvline(2.5, color="0.5", lw=1, ls="--")
    ax.set_xlabel("HOM's tripod force f_tripod (N)", fontsize=14)
    ax.set_ylabel("tracking, both models, seeds 0-2", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12, ncol=2, loc="best")
    return png_uri(fg)


# ------------------------------------------------------------------ page
def page():
    sys.path.insert(0, str(ROOT / "scripts/docs"))
    import docs_index
    import retro_style
    import texsvg
    S = j(DATA / "20261119-twist_ranking_summary.json")
    if S is None:
        sys.exit("no summary yet: run sim/20261119-twist_ranking_report.py")
    prop = j(DATA / "20261119-trial_proposal.json")
    FR = j(DATA / "20261119-force_refine.json")
    TP = j(DATA / "20261119-trial_plans.json")
    G = gap_data()
    rho = S["rho"]

    def r_(k):
        v = rho.get(k)
        return v["rho"] if isinstance(v, dict) else None

    def rn(k):
        v = rho.get(k)
        return v["n"] if isinstance(v, dict) else None
    top = {m: S["top10"][m][0] for m in ("point", "pads") if S["top10"].get(m)}
    wc = S.get("fresh_seeds") or {}
    gsum = G["summary"] if G else {}
    g_pads = (gsum.get("pads 7v5 mu 1") or {}).get("task") or {}
    n_dep = prop["deployable"] if prop else None
    picks = prop["pick"] if prop else []
    n_plans = len(TP["plans"]) if TP else 0
    names = {"point": "point contact", "pads": "the sphere pads"}
    best_s = " and ".join(f"{xy_s(top[m]['xy'])} under {names[m]} (tracking {top[m]['score']:.2f})" for m in top)
    lede = (f"On the calibrated simulation of the SR2 hand, the sphere twist's best layout is {best_s}, out of "
            f"{S['layouts']} layouts with ground-clear grasps. "
            f"The two contact models rank the layouts with Spearman's ρ {f(r_('point vs pads (seeds 0-2)'))}. ")
    if wc:
        d = np.mean([wc[m]["change_mean"] for m in wc])
        lede += f"At fresh seeds the {wc[next(iter(wc))]['layouts']} best layouts per model changed by {d:+.3f}. "
    if g_pads:
        lede += (f"Started from the hand's own state in 0.5 s windows, the simulated plant turned the sphere "
                 f"{g_pads['twist_hand_mean'] + g_pads['twist_err_mean']:.1f}° per window against the hand's "
                 f"{g_pads['twist_hand_mean']:.1f}°. ")
    if picks:
        lede += (f"{len(picks)} of the {n_dep} deployable layouts and the layout of the 11-17 trials have bench plans "
                 f"({n_plans} plans), with the predictions written before any trial.")
    m0 = next(iter(top))
    summary = (f"Sphere twist ranked over {S['layouts']} layouts on the calibrated SR2 simulation: best "
               f"{xy_s(top[m0]['xy'])} ({m0}, tracking {top[m0]['score']:.2f}); point vs pads ρ "
               f"{f(r_('point vs pads (seeds 0-2)'))}; plans for {len(picks) + 1} layouts on the hand.")

    sec = []
    sec.append(("terms", "Terms and metrics", r"""
<dl class="terms">
<dt>Layout</dt><dd>The places of the three fingers' mounts on the palm, as a symmetric tripod: the thumb on the palm's
\(x_P\) axis, the index and middle mirrored about it (sr2_layouts.sym6). \(x_{\mathrm{sep}}\) is the distance along
\(x_P\) from the thumb's mount to the line through the index's and the middle's (mm), \(y_{\mathrm{sep}}\) the distance
between the index's and the middle's mounts (mm). The lattice runs over 40-160 mm and 50-170 mm in 5 mm steps, 625
layouts (the pose toy's grid2d).</dd>
<dt>Ground-clear grasp</dt><dd>One of up to 16 tripod grasps per layout from the pose toy (rows grid2dgr): pad contact
points on the 40 mm sphere whose fingers stay clear of a ground at the sphere's lowest point. 482 layouts have them; the
other 143 have none. A grasp is named by its index k.</dd>
<dt>Tracking</dt><dd>The twist's score on the hand and in simulation (sr2hand.control.trial_score): one less the RMS
angle between the sphere's orientation and the commanded one over the RMS commanded rotation, over the task window
(2.0 s to the end), at least 0; 0 when the sphere dropped:
\[ s = \max\!\left(0,\; 1 - \frac{\mathrm{RMS}_t\,\angle(R_t,\, R^{\mathrm{ref}}_t)}{\mathrm{RMS}_t\,\angle(R_0,\, R^{\mathrm{ref}}_t)}\right) \]
1 means the commanded rotation was followed, 0 no better than holding the sphere still.</dd>
<dt>Layout score</dt><dd>The tracking of the layout's best grasp, the mean over seeds 0, 1 and 2, among the grasps run
at all three seeds. A seed sets the sphere's placement: seed 0 places it as planned, any other seed shifts it by up to 1
mm in \(x\) and \(y\) and turns it by up to 2° about \(z\) (hom_tasks.Plant.reset).</dd>
<dt>Fresh seeds</dt><dd>Seeds 10, 11 and 12, never used to choose a grasp or a layout; the change from seeds 0-2 to
them is the selection's optimism.</dd>
<dt>Spearman's ρ</dt><dd>The correlation of the ranks of the same layouts under two scores (1: same order).</dd>
<dt>Window gap</dt><dd>The plant's error over 0.5 s windows of a recorded trial, each started from the hand's measured
joints and tracked sphere: the simulated twist about \(z_P\) over the window less the tracked one (°).</dd>
</dl>"""))

    tbl1 = table(["Part", "Setting"], [
        ["servos", "7.5 V: kp 0.825 N m/rad, damping 0.011 N m s/rad, torque limit 0.27 N m; index and middle yaw gain "
                   "0.525 N m/rad"],
        ["gears", "the load cannot backdrive a joint; forward friction 0.6 |τ| (rev_clutch2)"],
        ["MCP stops", "index −11.4°, middle −10.0° (measured on 2026-11-17)"],
        ["object", "40 mm sphere on its stem and block, a table under the block until the lift"],
        ["contact", "soft tangential contact (impratio 1), friction 1.0, point contact or the sphere pads"],
        ["controller", "HOM as the bench server runs it: forces converted at 0.55 N m/rad under the torque budget 0.56 "
                       "N m, from the layout's own open pose (pads 8 mm off the sphere), contacts not given to HOM"],
    ], caption="The simulation of the ranking (sim/20261117-g13_open_pose_replan.py's closed-loop run, unchanged).", n=1)
    sec.append(("sim", "Simulation and runs", f"""
<p>The ranking uses the simulation whose blind predictions came nearest the hand on 2026-11-17. For the 20 twists of
grasp 0 at (70, 110) it predicted tracking 0.437 with point contact, against the hand's mean of 0.447 over the night's
tips-off trials, and 0.406 with the pads (identification queue, "W3, A/B"). Its parts are in Table 1.</p>
{tbl1}
<p>Stage A ran every grasp of every layout at seed 0 under both contact models ({S['runs']['A']} runs). Stage B ran each
layout's three best grasps at seeds 1 and 2 ({S['runs']['B']} runs). Stage C ran the 60 best layouts' two best grasps at
the fresh seeds ({S['runs']['C']} runs), and stage D the 150 best layouts' two best grasps on four other plants
({S['runs']['D']} runs). Of these, {sum(S['runs']['errors'].values())} runs ended in an error of HOM's least squares (the
SVD did not converge) and score 0.</p>"""))

    t2 = []
    for m in ("point", "pads"):
        for i, r in enumerate(S["top10"].get(m, [])):
            t2.append([m if i == 0 else "", str(i + 1), xy_s(r["xy"]), str(r["k"]), f(r["score"], 3), f(r["fresh"], 3)])
    sec.append(("landscape", "The twist landscape", f"""
<p>Figure 1 shows each layout's score under both contact models; Table 2 lists the ten best.</p>
{fig(1, fig_landscape(S, prop), "Layout score over the lattice, under point contact (left) and the sphere pads "
     "(right); grey: no ground-clear grasp. White circles: the layouts proposed for the hand, numbered by rank "
     "among the deployable layouts (Section 7); orange square: (70, 110), run on the hand on 2026-11-17.")}
{table(["Model", "Rank", "Layout (mm)", "Grasp k", "Tracking, seeds 0-2", "Tracking, seeds 10-12"], t2,
       num=(1, 3, 4, 5), caption="The ten best layouts per contact model.", n=2)}"""))

    t3 = [[esc(k), f(v["rho"], 3) if isinstance(v, dict) else esc(v), str(v["n"]) if isinstance(v, dict) else ""]
          for k, v in rho.items()]
    wc_rows = [[m, str(v["layouts"]), f(v["seeds012_mean"], 3), f(v["seeds101112_mean"], 3), f(v["change_mean"], 3),
                f(v["rho_seeds012_vs_fresh"], 3)] for m, v in wc.items()]
    sec.append(("agreement", "Agreement of the ranking", f"""
<p>The contact models, the number of seeds and the plant each change the scores; Table 3 gives how far each changes the
order, Figure 2 the first two, and Table 4 the change at fresh seeds. The GPU grid of the landscapes (rows
grid2dff_v4b_sv25r) differs from this simulation in several ways at once: the bare ball without stem or table, the 5 V
servo of 0.72 N m/rad with gears the load can backdrive, HOM converting at 0.72, impratio 100, and the 16 film-grid
grasps in place of the ground-clear ones.</p>
{fig(2, fig_agreement(S), "Left: each layout's score under point contact against the sphere pads. Right: the best "
     "layouts' scores at the fresh seeds against seeds 0-2; points under the grey line lost at the fresh seeds.")}
{table(["Comparison", "Spearman's ρ", "Layouts"], t3, num=(1, 2), caption="Rank agreement of layout scores. Stage D's "
       "plants are compared with the mean of the point and pads scores over its 150 layouts.", n=3)}
{table(["Model", "Layouts", "Seeds 0-2", "Seeds 10-12", "Change", "ρ"], wc_rows, num=(1, 2, 3, 4, 5),
       caption="Mean tracking of the best layouts' chosen grasps at the selection's seeds and at fresh seeds.", n=4)}"""))

    sr = S.get("search") or {}
    if sr:
        e, rr = sr["ea"]["runs_to_within_0.05"], sr["random"]["runs_to_within_0.05"]
        sec.append(("search", "Search by evolution and by sampling", f"""
<p>Yang et al.'s elitist evolutionary search (hom_landscape.ea_search, population 64) and uniform sampling were replayed
on stage A's point-contact scores, a layout costing its 16 grasps. The full grid cost {sr['ea']['grid_runs']} runs. The
search found a layout within 0.05 of the grid's best in {e['found']} of {e['of']} runs, after a median of
{e['median']:.0f} runs (quartiles {e['q25']:.0f} and {e['q75']:.0f}). Uniform sampling found one in {rr['found']} of
{rr['of']}, after a median of {rr['median']:.0f} (quartiles {rr['q25']:.0f} and {rr['q75']:.0f}).</p>"""))

    if FR and FR.get("layouts"):
        t5 = [[xy_s(L["xy"]), str(L["k"]), f"{L['best_prm']['f_tripod']:g}", f"{L['best_prm']['w_spin_tripod']:g}",
               f(L["default_seeds012"], 3), f(L["best_seeds012"], 3), f(L["best_seeds101112"], 3),
               f(L.get("default_mu06"), 3), f(L.get("best_mu06"), 3)] for L in FR["layouts"]]
        sec.append(("forces", "HOM's forces per layout", f"""
<p>On the hand the twist stalls where the index and middle yaws and the thumb MCP reach the torque budget's cap; the
grip's normal forces leave the yaws too little torque. HOM's tripod force f_tripod (default 2.5 N) and the weight of
the spin rows in its Eq. (2), w_spin (default 1.0), set that balance. Each proposed layout's grasp ran on a grid of
the two at seeds 0-2 under both models; the best setting then ran at the fresh seeds, and with the default at friction
0.6 (Table 5, Figure 3).</p>
{fig(3, fig_forces(FR), "Tracking against HOM's tripod force per proposed layout, at the spin weight chosen for it; "
     "dashed: the default 2.5 N.")}
{table(["Layout (mm)", "Grasp k", "f_tripod (N)", "w_spin", "Default, seeds 0-2", "Chosen, seeds 0-2",
        "Chosen, seeds 10-12", "Default, μ 0.6", "Chosen, μ 0.6"], t5, num=(1, 2, 3, 4, 5, 6, 7, 8),
       caption="HOM's forces chosen per layout. μ 0.6: point contact at friction 0.6, seeds 0-2.", n=5)}"""))

    if G:
        slip = [w for w in G["windows"] if w[0] == "pads" and w[3] == "task" and w[4] > 15.0]
        n_slip = len(slip)
        slip_sim = float(np.mean([w[5] for w in slip])) if slip else float("nan")
        big = [w for w in G["windows"] if w[0] == "pads" and w[4] > 25.0]
        big_trials = len({w[1] for w in big})
        big_t = (min(w[2] for w in big), max(w[2] + 0.5 for w in big)) if big else (float("nan"), float("nan"))
        big_hand = (min(w[4] for w in big), max(w[4] for w in big)) if big else (float("nan"), float("nan"))
        big_sim = (min(w[5] for w in big), max(w[5] for w in big)) if big else (float("nan"), float("nan"))
        t6 = []
        for k, v in sorted(gsum.items(), key=lambda kv: (kv[1].get("task") or {}).get("twist_err_rms") or 99):
            t, g = v.get("task") or {}, v.get("grasp") or {}
            if not t:
                continue
            t6.append([esc(k), str(t["windows"]), f(t["twist_err_rms"]), f(t["twist_err_mean"]), f(t["rot_err_rms"]),
                       f(t["pos_err_rms"]), f(t["joint_rms_mean"]), f(g.get("rot_err_rms")), f(t["settle_mm_median"])])
        sec.append(("gap", "Short-window gap against the hand's twists", f"""
<p>A full open-loop replay accumulates its error, so each of the 20 g0 twists of 2026-11-17 was cut into 0.5 s windows
every 0.25 s. Each window starts the plant from the hand's measured joints and tracked sphere, holds the targets in
force for 40 ms, and replays the hand's targets (sim/20261119-segment_gap.py). Table 6 compares 24 plants. In the task
windows the hand turned the sphere {g_pads.get('twist_hand_mean', float('nan')):.1f}° per window on average and the
pads plant {g_pads.get('twist_hand_mean', 0) + g_pads.get('twist_err_mean', 0):.1f}°. The twist error's RMS is
{g_pads.get('twist_err_rms', float('nan')):.1f}° with the pads and within 1° of that for every plant at friction 1.0
(Figure 4). In the {n_slip} windows where the hand turned the sphere more than 15°, the pads plant turned it
{slip_sim:.1f}° on average. The largest come at one point of the schedule: in {big_trials} of the 20 trials a window
between {big_t[0]:.1f} and {big_t[1]:.1f} s, midway through the twist's ramp (3.4-6.2 s), turned the sphere by
{big_hand[0]:.0f}-{big_hand[1]:.0f}° on the hand and by {big_sim[0]:.0f} to {big_sim[1]:.0f}° in simulation. The sphere
held while the commanded rotation built up and then slipped at once, as the films of 2026-11-18 show at 5.0-5.3 s. Starting a window moved the tracked sphere by a median of
{g_pads.get('settle_mm_median', float('nan')):.1f} mm before it began: the tracked sphere and the model's fingertips
disagree by that much. The pads with noslip are left out of Figure 4: from these starts most of their windows threw the sphere
out of the grasp or ended in an error.</p>
{fig(4, fig_gap(G), "Left: the sphere's twist over each 0.5 s window, simulated (pads, 7.5 V, friction 1.0) against "
     "tracked, from the hand's state at the window's start. Right: the RMS of the twist error over the task windows "
     "per plant (contact model, servo set, friction).")}
{table(["Plant", "Task windows", "Twist error RMS (°)", "Twist error mean (°)", "Rotation error RMS (°)",
        "Position error RMS (mm)", "Joint error (°)", "Grasp windows: rotation (°)", "Start shift (mm)"], t6,
       num=tuple(range(1, 9)), caption="Window gap per plant. 7v5: the servos at 7.5 V; 7v5y: with the index and "
       "middle yaws as strong as the other joints; ns: MuJoCo's noslip; μ: friction. Joint error: RMS over the nine "
       "joints at the window's end. Start shift: median motion of the sphere while the window's start resolved.", n=6)}"""))

    if TP:
        t7 = []
        for p in TP["plans"]:
            pr = p["prediction"]
            t7.append([esc(Path(p["plan"]).name.replace("_plan.json", "").replace("20261119-rank_", "")), p["role"],
                       str(p.get("rank_among_deployable") or ""), xy_s(p["xy"]), str(p["k"]),
                       esc(json.dumps(p.get("prm"))) if p.get("prm") else "default",
                       f(pr["point"]["track_seeds012"], 3), f(pr["pads"]["track_seeds012"], 3),
                       f(pr["point"]["twist_end_deg"], 1), "yes" if p.get("fits") else "no"])
        sec.append(("plans", "Layouts and plans for the hand on 2026-11-20", f"""
<p>A layout is deployable when its mounts fit the gantries' travel boxes, its index and middle stay at least 15 mm
apart in every simulated run of its grasp (on 2026-11-17 the hand's fingers came about 10 mm closer than the
simulation's), and the sphere stayed in the grasp in every run under both models; {n_dep} layouts are. The proposal
takes the four best and six at evenly spaced ranks down to the lowest scoring deployable layout, so that the trials span
the predicted range, and (70, 110) grasp 0, run 20 times on the hand on 2026-11-17. Each plan was built as the plans of
2026-11-16 (logs/20261116-hom_setup/film_plan.py, which reproduces the g0 plan exactly), with the open pose's index and
middle MCPs at −11.0 and −9.5° as in the lab's plans of 2026-11-17. Where stage E's forces scored at least 0.05 more at
the fresh seeds and no less at friction 0.6, a second plan carries them. The predictions in Table 7 were written into
each plan's meta.ranking before any trial ran.</p>
{table(["Plan", "Role", "Rank", "Layout (mm)", "Grasp k", "HOM forces", "Predicted, point", "Predicted, pads",
        "Twist at the end (°)", "Fits"], t7, num=(2, 4, 6, 7, 8),
       caption="Plans in docs/experiments/20261115-hom_bench_trials/plans/ (20261119-rank_*). Predicted: tracking, mean "
               "of seeds 0-2.", n=7)}"""))

    sec.append(("future", "Future work", """
<ul>
<li>Run the plans of Section 7 on the hand, three trials each, in a random order of layouts, with the fingertip boards
recorded and both cameras filming; score them with trial_score and compute Spearman's ρ between the predicted and the
measured tracking over the layouts. A ρ near 0 over ten layouts falsifies the claim that this simulation ranks twist
layouts for this hand.</li>
<li>Fit the offset between the tracked sphere and the model's fingertips per trial at the grasp, and rerun
sim/20261119-segment_gap.py with it, so that a window starts without the 2-4 mm shift.</li>
<li>Add a slip model of the gear train or the pads (a static friction above the sliding one, or the joints' give past
the potentiometers) to the plant and compare its window gap with Table 6.</li>
<li>Film the best, a middle and the worst proposed layout's simulated twist beside the hand's trial of the same
plan.</li>
</ul>"""))

    toc = "".join(f'<li><a href="#{k}">{t}</a></li>' for k, t, _ in sec)
    body = "".join(f'<section id="{k}"><h2>{t}</h2>{s}</section>' for k, t, s in sec)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(body)]
    TEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    svgs = iter(texsvg.render(items, cache_path=str(TEX_CACHE), scale=1.1))
    body = TEX_RE.sub(lambda m: next(svgs), body)
    title = "Sphere-twist layout ranking on the calibrated SR2 simulation"
    out = (TEMPLATE.read_text()
           .replace("{{TITLE}}", title)
           .replace("{{TEXCSS}}", texsvg.CSS)
           .replace("{{EYEBROW}}", "HOM on the SR2 hand · 2026-11-19")
           .replace("{{H1}}", title)
           .replace("{{LEDE}}", lede)
           .replace("{{BYLINE}}", "Workstation session hand-4a, overnight 2026-11-19 to 11-20; calibration data from the "
                                  "corvus lab session")
           .replace("{{FILM}}", "")
           .replace("{{TOC}}", toc)
           .replace("{{SECTIONS}}", body)
           .replace("{{FOOTER}}", "Built by scripts/hom/twist_ranking_page.py from "
                                  "docs/experiments/20261119-twist_layout_ranking/data/."))
    out = retro_style.apply(out)
    out = docs_index.tag(out, topic="design", summary=summary)
    PAGE.write_text(out)
    print(PAGE, f"{PAGE.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="another data folder (a test build)")
    ap.add_argument("--out", default=None, help="another page path (a test build)")
    a = ap.parse_args()
    if a.data:
        DATA = Path(a.data)
        GAP = DATA / "20261119-segment_gap.json"
    if a.out:
        PAGE = Path(a.out)
    page()

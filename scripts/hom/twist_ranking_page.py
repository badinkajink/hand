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


def film(n, rel, caption):
    return (f'<figure id="fig{n}"><video controls muted loop playsinline preload="metadata" src="{rel}" '
            f'style="width:100%"></video><figcaption>Figure {n}. {caption} (<a href="{rel}">{rel}</a>)</figcaption>'
            f"</figure>")


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
    cm = matplotlib.colormaps["viridis"].copy()
    cm.set_bad("0.82")                                       # the lattice's layouts without a ground-clear grasp
    for ax, m, name in zip(axs, ("point", "pads"), ("point contact", "sphere pads")):
        Z = np.full((len(ys), len(xs)), np.nan)
        for L in S["per_layout"]:
            if m in L:
                i, k = int(round((L["xy"][1] - 50) / 5)), int(round((L["xy"][0] - 40) / 5))
                Z[i, k] = L[m]["score"]
        im = ax.imshow(Z, origin="lower", extent=(37.5, 162.5, 47.5, 172.5), cmap=cm, vmin=0,
                       vmax=np.nanmax(Z) if np.isfinite(Z).any() else 1, aspect="equal")
        for (x, y), p in picks.items():
            ax.plot(x, y, "o", mfc="none", mec="w", mew=2.0, ms=11)
            ax.text(x + 3, y + 2, str(p["rank_among_deployable"]), color="w", fontsize=12, weight="bold")
        ax.plot(70, 110, "s", mfc="none", mec=C2, mew=2.0, ms=11)
        ax.set_title(f"Best grasp's tracking, {name}", fontsize=15)
        ax.set_xlabel(r"$x_{\mathrm{sep}}$ (mm)", fontsize=14)
        ax.tick_params(labelsize=13)
    axs[0].set_ylabel(r"$y_{\mathrm{sep}}$ (mm)", fontsize=14)
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


def fig_trials(TV):
    import matplotlib.pyplot as plt
    fg, ax = plt.subplots(figsize=(8.0, 6.0))
    X = [x for x in TV["plans"] if x["role"] in ("ranked", "anchor") and not x.get("prm") and x["n"]]
    for x in X:
        c = C2 if x["role"] == "anchor" else C1
        ax.plot([x["predicted_both"]] * x["n"], x["scores"], ".", color=c, alpha=0.45, ms=7)
        ax.errorbar(x["predicted_both"], x["measured_mean"], yerr=x["measured_sd"] or 0, fmt="o", color=c, ms=8,
                    capsize=3)
        ax.annotate(f"{x['xy'][0]:g}, {x['xy'][1]:g}", (x["predicted_both"], x["measured_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=12)
    R_ = [x for x in TV["plans"] if (x["role"] not in ("ranked", "anchor") or x.get("prm")) and x["n"]]
    for x in R_:
        ax.plot(x["predicted_both"], x["measured_mean"], "D", mfc="none", mec=C3, mew=1.6, ms=8)
    hi = max([0.7] + [x["predicted_both"] or 0 for x in X] + [max(x["scores"]) for x in X])
    ax.plot([0, hi], [0, hi], "-", color="0.6", lw=1)
    ax.set_xlabel("predicted tracking (both models, seeds 0-2)", fontsize=14)
    ax.set_ylabel("tracking on the hand", fontsize=14)
    ax.tick_params(labelsize=13)
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
    plant_rho = [r_(f"{p} vs mean of point and pads (stage D layouts)") for p in PLANT_NAMES]
    plant_rho = [x for x in plant_rho if x is not None]
    spread = S.get("stage_d_spread") or {}
    # figure and table numbers in the order they appear; the optional sections take theirs only when present
    FILM = D / "media/20261119-simulated_twist_three_layouts.mp4"
    has_film = FILM.exists() and bool(picks)
    FR_ok = bool(FR and FR.get("layouts"))
    CEM = j(DATA / "20261119-cem_refine.json")
    CEM_ok = bool(CEM and CEM.get("picks"))
    nf = iter(range(1, 20))
    F_LAND = next(nf)
    F_FILM = next(nf) if has_film else None
    F_AGREE = next(nf)
    F_FORCE = next(nf) if FR_ok else None
    F_GAP = next(nf) if G else None
    TV = j(DATA / "20261120-trials_vs_prediction.json")
    TV_ok = bool(TV and TV.get("trials_scored"))
    F_TRIALS = next(nf) if TV_ok else None
    nt = iter(range(1, 20))
    T_SIM, T_TOP, T_RHO, T_FRESH = next(nt), next(nt), next(nt), next(nt)
    T_FORCE = next(nt) if FR_ok else None
    T_CEM = next(nt) if CEM_ok else None
    T_GAP = next(nt) if G else None
    T_PLANS = next(nt) if TP else None
    T_TRIALS = next(nt) if TV_ok else None
    PLANS_REF = '<a href="#plans">the section on the layouts and plans for the hand</a>'
    lede = ""
    if TV_ok:
        rb = TV["rho"]["both models"]
        nl = rb.get("layouts")
        ci = rb.get("rho_trials_resampled_95")
        lede += (f"On the hand, {TV['trials_scored']} twist trials of the proposed plans were scored against the "
                 f"simulation's predictions, written before the trials. Over the {nl} layouts with default forces, "
                 f"Spearman's ρ between the predicted and the measured tracking is {f(rb.get('rho'))}"
                 + (f" (95 % interval over resampled trials {ci[0]:.2f} to {ci[1]:.2f})" if ci else "") + ". ")
    if picks:
        p0 = picks[0]
        lede += (f"The best deployable layout for the sphere twist on the calibrated SR2 simulation is "
                 f"{xy_s(p0['xy'])} with grasp {p0['k']}. Its predicted tracking is {p0['score']:.2f}, and {n_dep} of "
                 f"{prop['of']} layouts are deployable on the hand. ")
    lede += (f"Over all {S['layouts']} layouts with ground-clear grasps, Spearman's ρ between the orders under the two "
             f"contact models is {f(r_('point vs pads (seeds 0-2)'))}. ")
    if plant_rho and spread:
        lede += (f"The {spread['layouts']} best layouts differ little (standard deviation {spread['sd']:.3f}). Among "
                 f"them, ρ is {f(r_('point vs pads (stage D layouts)'))} between the two contact models and "
                 f"{min(plant_rho):.2f} to {max(plant_rho):.2f} between four other plants and their mean. ")
    if g_pads:
        lede += (f"Started from the hand's state in 0.5 s windows, the simulated hand turned the sphere "
                 f"{g_pads['twist_hand_mean'] + g_pads['twist_err_mean']:.1f}° per window; the real hand turned it "
                 f"{g_pads['twist_hand_mean']:.1f}°. ")
    if picks and n_plans:
        lede += (f"The {n_plans} bench plans cover {len(picks)} deployable layouts over the predicted range and the "
                 f"layout of the 11-17 trials; their predictions were written before any trial.")
    elif picks:
        lede += (f"{len(picks)} deployable layouts over the predicted range and the layout of the 11-17 trials are "
                 f"proposed for the hand.")
    m0 = next(iter(top))
    summary = (f"Sphere twist ranked over {S['layouts']} layouts on the calibrated SR2 simulation: best "
               f"{xy_s(top[m0]['xy'])} ({m0}, tracking {top[m0]['score']:.2f}); point vs pads ρ "
               f"{f(r_('point vs pads (seeds 0-2)'))}; plans for {len(picks) + 1} layouts on the hand.")

    sec = []
    sec.append(("terms", "Terms and metrics", r"""
<dl class="terms">
<dt>Layout</dt><dd>The places of the three fingers' mounts on the palm, as a symmetric tripod: the thumb on the palm's
\(x_P\) axis, the index and middle mirrored about it (sr2_layouts.sym6). \(x_{\mathrm{sep}}\) is the distance along
\(x_P\) from the thumb's mount to the line through the index's and the middle's mounts (mm). \(y_{\mathrm{sep}}\) is the
distance between the index's and the middle's mounts (mm). The lattice of the pose toy (grid2d) runs over 40-160 mm and
50-170 mm in 5 mm steps, 625 layouts.</dd>
<dt>Ground-clear grasp</dt><dd>One of up to 16 tripod grasps per layout from the pose toy (rows grid2dgr). Its contacts
lie on the 40 mm sphere, and its fingers stay clear of a ground plane at the sphere's lowest point. 482 layouts have such
grasps; the other 143 have none. A grasp is named by its index k.</dd>
<dt>Tracking</dt><dd>The twist's score on the hand and in simulation (sr2hand.control.trial_score), over the task window
from 2.0 s to the end. The RMS angle between the sphere's orientation and the commanded one is divided by the RMS of the
commanded rotation, and the ratio is subtracted from 1:
\[ s = \max\!\left(0,\; 1 - \frac{\mathrm{RMS}_t\,\angle(R_t,\, R^{\mathrm{ref}}_t)}{\mathrm{RMS}_t\,\angle(R_0,\, R^{\mathrm{ref}}_t)}\right) \]
The score is 0 when the sphere dropped. A score of 1 means the commanded rotation was followed, and 0 no better than
holding the sphere still.</dd>
<dt>Layout score</dt><dd>The tracking of the layout's best grasp, as the mean over seeds 0, 1 and 2, among the grasps run
at all three seeds. A seed sets the sphere's placement (hom_tasks.Plant.reset). Seed 0 places it as planned; any other
seed shifts it by up to 1 mm in \(x\) and \(y\) and turns it by up to 2° about \(z\).</dd>
<dt>Fresh seeds</dt><dd>Seeds 10, 11 and 12, which were never used to choose a grasp or a layout. The change in score
from seeds 0-2 to them is the selection's optimism.</dd>
<dt>Spearman's ρ</dt><dd>The correlation of the ranks of the same layouts under two scores (1: same order).</dd>
<dt>Window gap</dt><dd>The plant's error over 0.5 s windows of a recorded trial. Each window starts from the hand's
measured joints and tracked sphere. The gap is the simulated twist about \(z_P\) over the window less the tracked one
(°).</dd>
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
    ], caption="The simulation of the ranking (sim/20261117-g13_open_pose_replan.py's closed-loop run, unchanged).", n=T_SIM)
    sec.append(("sim", "Simulation and runs", f"""
<p>The ranking was run on the simulation whose blind predictions came nearest the hand on 2026-11-17. For the 20
twists of grasp 0 at (70, 110), it predicted a tracking of 0.437 with point contact and 0.406 with the pads. The hand's
mean over that night's tips-off trials was 0.447 (identification queue, "W3, A/B"). Its parts are in Table
{T_SIM}.</p>
{tbl1}
<p>In stage A, every grasp of every layout was run at seed 0 under both contact models ({S['runs']['A']} runs). Stage B
added each layout's three best grasps at seeds 1 and 2 ({S['runs']['B']} runs). In stage C, the two best grasps of the
60 best layouts were run at the fresh seeds ({S['runs']['C']} runs). Stage D ran the two best grasps of the 150 best
layouts on four other plants ({S['runs']['D']} runs). Of all these runs, {sum(S['runs']['errors'].values())} ended in an
error of HOM's least squares (the SVD did not converge) and scored 0.</p>"""))

    t2 = []
    for m in ("point", "pads"):
        for i, r in enumerate(S["top10"].get(m, [])):
            t2.append([m if i == 0 else "", str(i + 1), xy_s(r["xy"]), str(r["k"]), f(r["score"], 3), f(r["fresh"], 3)])
    film_html, film_p = "", ""
    if has_film:
        film_p = (f" Figure {F_FILM} shows the simulated twist at three of the proposed layouts: the best, a middle one "
                  f"and the lowest ranked.")
        film_html = film(F_FILM, "media/" + FILM.name, "The simulated twist (pads, seed 0) at the best, a middle and the "
                         "lowest-ranked of the proposed layouts, from the ranking's run with the frames drawn "
                         "(sim/20261119-twist_ranking_film.py). The red and blue segments are the sphere's body axes; "
                         "they turn with the twist.")
    excl = (prop or {}).get("excluded") or {}
    cands = S.get("candidates") or []
    excl_p = ""
    if excl and cands:
        b = cands[0]
        ex = {k.split()[0]: v for k, v in excl.items()}           # "do", "sphere", "index-middle"
        why = []
        if not b["fits"]:
            why.append("its mounts do not fit the gantries' travel boxes")
        if not b["held_all"]:
            why.append("the sphere dropped or HOM failed in one of its runs")
        if (b["im_gap_min_mm"] or 0) < 15:
            g = b["im_gap_min_mm"] or 0
            why.append(f"its index and middle touched in one of its {b['runs']} runs" if g <= 0.05 else
                       f"its index and middle came within {g:.1f} mm of each other in one of its {b['runs']} runs")
        if why:
            excl_p = (f" The best layout over both models, {xy_s(b['xy'])} with grasp {b['k']} at {b['score']:.2f}, is "
                      f"not deployable: {' and '.join(why)}. Of the {prop['of']} layouts whose best grasp ran under both "
                      f"models, {ex.get('do', 0)} do not fit the travel boxes. At {ex.get('sphere', 0)}, the sphere "
                      f"dropped or HOM failed in one of the best grasp's runs. At {ex.get('index-middle', 0)}, the index "
                      f"and middle came within 15 mm; a layout can fail more than one test ({PLANS_REF}).")
    sec.append(("landscape", "The twist landscape", f"""
<p>Figure {F_LAND} shows each layout's score under both contact models; Table {T_TOP} lists the ten best.{excl_p}
{film_p}</p>
{fig(F_LAND, fig_landscape(S, prop), "Layout score over the lattice, under point contact (left) and the sphere pads "
     "(right); grey: no ground-clear grasp. White circles: the layouts proposed for the hand, numbered by rank "
     "among the deployable layouts; orange square: (70, 110), run on the hand on 2026-11-17.")}
{table(["Model", "Rank", "Layout (mm)", "Grasp k", "Tracking, seeds 0-2", "Tracking, seeds 10-12"], t2,
       num=(1, 3, 4, 5), caption="The ten best layouts per contact model.", n=T_TOP)}{film_html}"""))

    def rho_label(k):
        for p, nm in PLANT_NAMES.items():
            if k.startswith(p + " vs"):
                return f"{nm[0].upper() + nm[1:]} vs the mean of point contact and the pads ({spread.get('layouts', 150)} best)"
        return {"point vs pads (seeds 0-2)": "Point contact vs the pads",
                "point vs pads (stage D layouts)": f"Point contact vs the pads ({spread.get('layouts', 150)} best)",
                "point: seed 0 vs seeds 0-2": "Point contact: seed 0 vs seeds 0-2",
                "pads: seed 0 vs seeds 0-2": "Pads: seed 0 vs seeds 0-2",
                "point vs grid sv25r": "Point contact vs the GPU grid sv25r",
                "pads vs grid sv25r": "Pads vs the GPU grid sv25r"}.get(k, k)
    t3 = [[esc(rho_label(k)), f(v["rho"], 3) if isinstance(v, dict) else esc(v), str(v["n"]) if isinstance(v, dict) else ""]
          for k, v in rho.items()]
    wc_rows = [[m, str(v["layouts"]), f(v["seeds012_mean"], 3), f(v["seeds101112_mean"], 3), f(v["change_mean"], 3),
                f(v["rho_seeds012_vs_fresh"], 3)] for m, v in wc.items()]
    agree_p = (f"Over all {rn('point vs pads (seeds 0-2)')} layouts, ρ between the orders under point contact and under "
               f"the pads is {f(r_('point vs pads (seeds 0-2)'))}. Between seed 0 alone and seeds 0-2, it is "
               f"{f(r_('point: seed 0 vs seeds 0-2'))} under point contact. ")
    if plant_rho and spread:
        agree_p += (f"The {spread['layouts']} best layouts lie close together: their mean over both models is "
                    f"{spread['mean']:.3f}, with a standard deviation of {spread['sd']:.3f}. Among them, ρ between the two "
                    f"models falls to {f(r_('point vs pads (stage D layouts)'))}. Each of stage D's four plants changes "
                    f"one part: the pads at impratio 100, friction 0.6 under either contact model, or the servos at 5 V. "
                    f"Between each plant's order and that of the two models' mean, ρ is {min(plant_rho):.2f} to "
                    f"{max(plant_rho):.2f}. ")
    PPRED = j(DATA / "20261119-plant_predictions.json")
    if PPRED and PPRED.get("layouts"):
        import itertools
        from sr2hand.numerics.stats import spearman
        P_ = [L["predicted"] for L in PPRED["layouts"]]
        ms_ = [m for m in P_[0] if all(p.get(m) is not None for p in P_)]
        pr_ = {(a_, b_): spearman([p[a_] for p in P_], [p[b_] for p in P_]) for a_, b_ in itertools.combinations(ms_, 2)}
        if pr_:
            agree_p += (f"Over the {len(P_)} layouts proposed for the hand, which span the predicted range, ρ between "
                        f"any two of the {len(ms_)} plants' orders is {min(pr_.values()):.2f} to "
                        f"{max(pr_.values()):.2f}; it is {pr_.get(('point', 'pads'), float('nan')):.2f} between point "
                        f"contact and the pads (data/20261119-plant_predictions.json). ")
    if "point" in wc and "pads" in wc:
        agree_p += (f"At the fresh seeds, the 60 best layouts scored {-wc['point']['change_mean']:.3f} less under point "
                    f"contact and {-wc['pads']['change_mean']:.3f} less under the pads, on average (Table {T_FRESH}). "
                    f"The standard deviations of these changes were {wc['point']['change_sd']:.3f} and "
                    f"{wc['pads']['change_sd']:.3f}. ")
    agree_p += (f"The ranking differs from the GPU landscape of the servo plant (ρ {f(r_('point vs grid sv25r'))} "
                f"and {f(r_('pads vs grid sv25r'))}). That grid (rows grid2dff_v4b_sv25r) differs from this simulation in "
                f"five ways at once. Its sphere is a bare ball without stem or table, and its 5 V servos of 0.72 N m/rad "
                f"have gears that the load can backdrive. HOM converts its forces at 0.72, the contacts run at impratio "
                f"100, and the 16 grasps come from the film grid in place of the ground-clear set.")
    sec.append(("agreement", "Agreement of the ranking", f"""
<p>The rank agreement of the layout scores between contact models, numbers of seeds and plants is in Table {T_RHO}, and
the first two are plotted in Figure {F_AGREE}. {agree_p}</p>
{fig(F_AGREE, fig_agreement(S), "Left: each layout's score under point contact against the sphere pads. Right: the "
     "best layouts' scores at the fresh seeds against seeds 0-2; points under the grey line lost at the fresh seeds.")}
{table(["Comparison", "Spearman's ρ", "Layouts"], t3, num=(1, 2), caption="Rank agreement of layout scores. Stage D's "
       "plants are compared with the mean of the point and pads scores over its layouts, the best by that mean.",
       n=T_RHO)}
{table(["Model", "Layouts", "Seeds 0-2", "Seeds 10-12", "Change", "ρ"], wc_rows, num=(1, 2, 3, 4, 5),
       caption="Mean tracking of the best layouts' chosen grasps at the selection's seeds and at fresh seeds.",
       n=T_FRESH)}"""))

    sr = S.get("search") or {}
    if sr:
        e, rr = sr["ea"]["runs_to_within_0.05"], sr["random"]["runs_to_within_0.05"]
        sec.append(("search", "Search by evolution and by sampling", f"""
<p>Yang et al.'s elitist evolutionary search (hom_landscape.ea_search, population 64) and uniform sampling were replayed
{e['of']} times each on stage A's point-contact scores, at a cost of 16 runs per layout. The full grid cost
{sr['ea']['grid_runs']} runs. A layout within 0.05 of the grid's best was reached by the evolutionary search in
{e['found']} of {e['of']} replays, after a median of {e['median']:.0f} runs (quartiles {e['q25']:.0f} and
{e['q75']:.0f}). With uniform sampling, one was reached in {rr['found']} of {rr['of']} replays, after a median of
{rr['median']:.0f} runs (quartiles {rr['q25']:.0f} and {rr['q75']:.0f}).</p>"""))

    if FR and FR.get("layouts"):
        t5 = [[xy_s(L["xy"]), str(L["k"]), f"{L['best_prm']['f_tripod']:g}", f"{L['best_prm']['w_spin_tripod']:g}",
               f(L["default_seeds012"], 3), f(L["best_seeds012"], 3), f(L["best_seeds101112"], 3),
               f(L.get("default_mu06"), 3), f(L.get("best_mu06"), 3)] for L in FR["layouts"]]
        FL = FR["layouts"]

        def mean_of(key):
            v = [L.get(key) for L in FL if L.get(key) is not None]
            return float(np.mean(v)) if v else float("nan")
        n_better = sum(1 for L in FL if L.get("best_seeds101112") is not None and L.get("default_seeds012") is not None
                       and L["best_seeds101112"] >= L["default_seeds012"] + 0.05
                       and (L.get("best_mu06") or 0) >= (L.get("default_mu06") or 0))
        forces_res = (f" Over the {len(FL)} layouts, the chosen forces raised the mean tracking from "
                      f"{mean_of('default_seeds012'):.3f} to {mean_of('best_seeds012'):.3f} at seeds 0-2. At the fresh "
                      f"seeds the chosen forces scored {mean_of('best_seeds101112'):.3f}. At friction 0.6, the default "
                      f"scored {mean_of('default_mu06'):.3f} and the chosen forces {mean_of('best_mu06'):.3f}. At "
                      f"{n_better} layouts the chosen forces gained at least 0.05 at the fresh seeds and lost nothing at "
                      f"friction 0.6; these have a second plan.")
        sec.append(("forces", "HOM's forces per layout", f"""
<p>On the hand, the twist stalls when the index and middle yaws and the thumb MCP reach the torque budget's cap. The
grip's normal forces then leave the yaws too little torque to turn the sphere. That balance is set by HOM's tripod force
f_tripod (default 2.5 N) and by the weight w_spin of the spin rows in HOM's least squares (default 1.0). Each proposed
layout's grasp was run on a grid of the two, at seeds 0-2 under both models. The best setting was then run at the fresh
seeds, and with the default at friction 0.6 (Table {T_FORCE}, Figure {F_FORCE}).{forces_res}</p>
{fig(F_FORCE, fig_forces(FR), "Tracking against HOM's tripod force per proposed layout, at the spin weight chosen for it; "
     "dashed: the default 2.5 N.")}
{table(["Layout (mm)", "Grasp k", "f_tripod (N)", "w_spin", "Default, seeds 0-2", "Chosen, seeds 0-2",
        "Chosen, seeds 10-12", "Default, μ 0.6", "Chosen, μ 0.6"], t5, num=(1, 2, 3, 4, 5, 6, 7, 8),
       caption="HOM's forces chosen per layout. μ 0.6: point contact at friction 0.6, seeds 0-2.", n=T_FORCE)}"""))

    CEM = j(DATA / "20261119-cem_refine.json")
    if CEM and CEM.get("picks"):
        t8 = [[c["level"], xy_s(c["xy"]), str(c["k"]), f(c["start_seeds012"], 3), f(c["best_seeds012"], 3),
               f(c["best_seeds101112"], 3), f(c["best_mu06"], 3), f(c["lat_start_deg"], 1), f(c["lat_deg"], 1),
               esc(", ".join(f"{k2[1]:.2g}" for k2 in c["prm"]["f_tripod_knots"]) if c["prm"].get("f_tripod_knots")
                   else f"{c['prm'].get('f_tripod', 2.5):.3g}")] for c in CEM["picks"]]
        CP = CEM["picks"]
        ok_c = [c for c in CP if c.get("best_seeds101112") is not None and c.get("start_seeds012") is not None
                and c["best_seeds101112"] >= c["start_seeds012"] + 0.05
                and (c.get("best_mu06") or 0) >= c["best_seeds101112"] - 0.05]
        cem_res = ""
        for lev in sorted({c["level"] for c in CP}):
            cc = [c for c in CP if c["level"] == lev and c.get("best_seeds101112") is not None]
            if cc:
                cem_res += (f" At level {lev}, the {len(cc)} picks scored {np.mean([c['best_seeds101112'] for c in cc]):.3f} "
                            f"on average at the fresh seeds; their starts scored "
                            f"{np.mean([c['start_seeds012'] for c in cc]):.3f} at seeds 0-2.")
        cem_res += (f" {len(ok_c)} of the {len(CP)} picks gained at least 0.05 at the fresh seeds and lost at most 0.05 "
                    f"at friction 0.6; these have a plan.")
        sec.append(("cem", "Grasp and HOM refined together", f"""
<p>The cross-entropy method (CEM) of the refinement study of 2026-11-03 was run on this simulation for each proposed
layout (sim/20261119-twist_cem_refine.py). Its level full searches the grasp's variables, the tripod's force and the
spin, roll and slide weights of HOM's least squares together. The grasps are held to the ground-clear constraints, so
the contacts stay near the equator. A second level, knots, replaces the constant force by a schedule, after Sunyu's
suggestion of impedance force targets that change over the task. The force is set at 1.6, 3.4, 4.9 and 6.4 s and
interpolated linearly between them (hom_tasks.force_at). Each of {CEM['iters']} iterations draws {CEM['pop']}
candidates, each scored by its mean tracking over seeds 0-2 under both contact models. Table {T_CEM} gives the start
(the layout's grasp with HOM's defaults), the pick at the search's seeds and at the fresh seeds, and the pick at
friction 0.6.{cem_res}</p>
{table(["Level", "Layout (mm)", "From grasp k", "Start", "Pick, seeds 0-2", "Pick, seeds 10-12", "Pick, μ 0.6",
        "Latitude, start (°)", "Latitude, pick (°)", "Force (N)"], t8, num=tuple(range(2, 10)),
       caption="CEM picks per layout. Tracking means over both contact models; μ 0.6: point contact at friction 0.6, "
               "seeds 0-2. Latitude: the contacts' angle below the sphere's equator (v[6]). Force: f_tripod, or its "
               "four knots.", n=T_CEM)}"""))

    if G:
        slip = [w for w in G["windows"] if w[0] == "pads" and w[3] == "task" and w[4] > 15.0]
        n_slip = len(slip)
        slip_sim = float(np.mean([w[5] for w in slip])) if slip else float("nan")
        big = [w for w in G["windows"] if w[0] == "pads" and w[3] == "task" and w[4] > 25.0]
        nan2 = (float("nan"), float("nan"))
        big_trials = len({w[1] for w in big})
        big_t = (min(w[2] for w in big), max(w[2] for w in big)) if big else nan2
        big_hand = (min(w[4] for w in big), max(w[4] for w in big)) if big else nan2
        big_sim = (min(w[5] for w in big), max(w[5] for w in big)) if big else nan2
        big_sim_med = float(np.median([w[5] for w in big])) if big else float("nan")
        t6 = []
        for k, v in sorted(gsum.items(), key=lambda kv: (kv[1].get("task") or {}).get("twist_err_rms") or 99):
            t, g = v.get("task") or {}, v.get("grasp") or {}
            if not t:
                continue
            t6.append([esc(k), str(t["windows"]), f(t["twist_err_rms"]), f(t["twist_err_mean"]), f(t["rot_err_rms"]),
                       f(t["pos_err_rms"]), f(t["joint_rms_mean"]), f(g.get("rot_err_rms")), f(t["settle_mm_median"])])
        sec.append(("gap", "Short-window gap against the hand's twists", f"""
<p>A full open-loop replay accumulates its error, so each of the 20 g0 twists of 2026-11-17 was cut into 0.5 s windows
every 0.25 s (sim/20261119-segment_gap.py). Each window starts the plant from the hand's measured joints and tracked
sphere. The first targets are held for 40 ms, and then the hand's targets are replayed. Table {T_GAP} compares
{len(t6)} plants. In the task windows, the hand turned the sphere {g_pads.get('twist_hand_mean', float('nan')):.1f}° per
window on average, and the simulated hand with the pads
{g_pads.get('twist_hand_mean', 0) + g_pads.get('twist_err_mean', 0):.1f}°. The RMS of the twist error is
{g_pads.get('twist_err_rms', float('nan')):.1f}° with the pads, and within 1° of that for every plant at friction 1.0
(Figure {F_GAP}). In the {n_slip} windows where the hand turned the sphere more than 15°, the simulated hand turned it
{slip_sim:.1f}° on average.</p>
<p>The largest windows come midway through the twist's ramp (3.4-6.2 s). In {big_trials} of the 20 trials, a window
starting between {big_t[0]:.2f} and {big_t[1]:.2f} s turned the sphere by {big_hand[0]:.0f}-{big_hand[1]:.0f}° on the
hand. In simulation the same windows turned it by {big_sim[0]:.0f} to {big_sim[1]:.0f}°, with a median of
{big_sim_med:.0f}°. On the hand the sphere held while the commanded rotation built up, and then slipped at once; the
films of 2026-11-18 show this at 5.0-5.3 s. During the 40 ms before each window, the simulated sphere moved by a median
of {g_pads.get('settle_mm_median', float('nan')):.1f} mm: the tracked sphere and the model's fingertips disagree by about
that much. The pads with noslip are left out of Figure {F_GAP}. From these starts, most of their windows threw the sphere
out of the grasp or ended in an error.</p>
{fig(F_GAP, fig_gap(G), "Left: the sphere's twist over each 0.5 s window, simulated (pads, 7.5 V, friction 1.0) against "
     "tracked, from the hand's state at the window's start. Right: the RMS of the twist error over the task windows "
     "per plant (contact model, servo set, friction).")}
{table(["Plant", "Task windows", "Twist error RMS (°)", "Twist error mean (°)", "Rotation error RMS (°)",
        "Position error RMS (mm)", "Joint error (°)", "Grasp windows: rotation (°)", "Start shift (mm)"], t6,
       num=tuple(range(1, 9)), caption="Window gap per plant. 7v5: the servos at 7.5 V; 7v5y: with the index and "
       "middle yaws as strong as the other joints; ns: MuJoCo's noslip; μ: friction. Joint error: RMS over the nine "
       "joints at the window's end. Start shift: median motion of the sphere while the window's start resolved.", n=T_GAP)}"""))

    if TP:
        t7 = []
        for p in TP["plans"]:
            pr = p["prediction"]
            if "point" in pr:
                cells = [f(pr["point"]["track_seeds012"], 3), f(pr["pads"]["track_seeds012"], 3),
                         f(pr["point"]["twist_end_deg"], 1)]
            else:                                          # a CEM pick: the mean over both models
                cells = [f(pr["both_models_seeds012"], 3) + " (both)", f(pr["both_models_seeds101112"], 3) + " (fresh)",
                         "&ndash;"]
            forces = "default"
            if p.get("prm"):
                q = p["prm"]
                forces = (f"f {' / '.join(f'{k2[1]:.2g}' for k2 in q['f_tripod_knots'])} N at "
                          f"{', '.join(f'{k2[0]:g}' for k2 in q['f_tripod_knots'])} s" if q.get("f_tripod_knots")
                          else f"f {q.get('f_tripod', 2.5):.3g} N")
                forces += "; w " + ", ".join(f"{q[w]:.2g}" for w in ("w_spin_tripod", "w_roll_tripod", "w_slide_tripod")
                                             if w in q)
            t7.append([esc(Path(p["plan"]).name.replace("_plan.json", "").replace("20261119-rank_", "")), p["role"],
                       str(p.get("rank_among_deployable") or ""), xy_s(p["xy"]), str(p["k"]), esc(forces)] + cells +
                      ["yes" if p.get("fits") else "no"])
        sec.append(("plans", "Layouts and plans for the hand on 2026-11-20", f"""
<p>A layout is deployable when it passes three tests. Its mounts fit the gantries' travel boxes. Its index and middle
stay at least 15 mm apart in every simulated run of its grasp; on 2026-11-17 the hand's fingers came about 10 mm closer
than the simulation's. The sphere stayed in the grasp in every run under both models. Of {prop['of'] if prop else ''}
layouts, {n_dep} pass. The proposal takes the four best deployable layouts and six at evenly spaced ranks down to the
lowest scoring one, so that the trials span the predicted range. It adds (70, 110) with grasp 0, run 20 times on the
hand on 2026-11-17. Each plan was built as the plans of 2026-11-16 were (logs/20261116-hom_setup/film_plan.py, which
reproduces the g0 plan exactly). The open pose's index and middle MCPs were raised to −11.0 and −9.5° where lower, as in
the lab's plans of 2026-11-17. A second plan has stage E's forces where they scored at least 0.05 more at the fresh
seeds and no less at friction 0.6. The predictions in Table {T_PLANS} were written into each plan's meta.ranking before
any trial ran.</p>
{table(["Plan", "Role", "Rank", "Layout (mm)", "Grasp k", "HOM forces", "Predicted (a)", "Predicted (b)",
        "Twist at the end (°)", "Fits"], t7, num=(2, 4, 6, 7, 8),
       caption="Plans in docs/experiments/20261115-hom_bench_trials/plans/ (20261119-rank_*). Predicted tracking: for the "
               "ranked and anchor plans (a) point contact and (b) the pads, mean of seeds 0-2; for the CEM plans (a) the "
               "mean over both models at seeds 0-2 and (b) at the fresh seeds 10-12. HOM forces: the tripod's force f (or "
               "its knots and their times) and the spin, roll and slide weights w where changed.", n=T_PLANS)}"""))

    if TV_ok:
        rr = TV["rho"]
        t9 = []
        for x in sorted(TV["plans"], key=lambda x: -(x["predicted_both"] or 0)):
            if not x["n"]:
                continue
            t9.append([esc(x["plan"].replace("_plan.json", "").replace("20261119-rank_", "")), x["role"],
                       xy_s(x["xy"]), str(x["k"]), f(x["predicted_both"], 3), f(x["predicted_point"], 3),
                       f(x["predicted_pads"], 3), str(x["n"]), f(x["measured_mean"], 3), f(x["measured_sd"], 3),
                       f(x["measured_partial_mean"], 3)])

        def rho_s(key, label):
            v = rr.get(key) or {}
            if v.get("rho") is None:
                return ""
            ci = v.get("rho_trials_resampled_95")
            return (f" {label}, ρ is {v['rho']:.2f} over {v['layouts']} layouts (p {v['p_two_sided']:.3f}"
                    + (f"; 95 % interval over resampled trials {ci[0]:.2f} to {ci[1]:.2f}" if ci else "") + ").")
        tri_p = (rho_s("both models", "Against the mean of both contact models") + rho_s("point", "Against point "
                 "contact alone") + rho_s("pads", "Against the pads alone")
                 + rho_s("both models, partial trials scored", "With trials that ended early scored over the periods "
                         "that ran"))
        vb = rr.get("both models") or {}
        if vb.get("mean_measured_minus_predicted") is not None:
            tri_p += (f" The hand's tracking differed from the prediction by {vb['mean_measured_minus_predicted']:+.3f} "
                      f"on average, and by {vb['mean_abs_error']:.3f} in absolute value.")
        sec.append(("trials", "Trials on the hand", f"""
<p>Each plan of Table {T_PLANS} was run on the hand by the lab's series (lab/hom_series.py), three trials in a row per
plan and the layouts in a seeded random order. Each trial was scored as the trials of 2026-11-16 and 11-17 were
(scripts/hardware/hom_trial_tracking.py), and scored 0 when it did not end by the schedule's end, as a simulated run
scores 0 when the sphere drops (sim/20261120-trials_vs_prediction.py). {TV['trials_scored']} trials were scored.{tri_p}
Figure {F_TRIALS} plots the measured tracking against the prediction, and Table {T_TRIALS} lists the plans.</p>
{fig(F_TRIALS, fig_trials(TV), "Tracking on the hand against the simulation's prediction per plan. Blue: the ranked "
     "layouts; orange: (70, 110), run on 2026-11-17; small dots: single trials; circles and bars: their mean and "
     "standard deviation; green diamonds: plans with refined forces or a CEM pick. Grey: equal tracking.")}
{table(["Plan", "Role", "Layout (mm)", "Grasp k", "Predicted, both", "Predicted, point", "Predicted, pads", "Trials",
        "Measured", "SD", "Measured, partial"], t9, num=(3, 4, 5, 6, 7, 8, 9, 10),
       caption="Trials per plan. Measured: mean tracking, 0 for a trial that did not end by the schedule's end; "
               "partial: such trials scored over the periods that ran.", n=T_TRIALS)}"""))

    gap_ref = f"Table {T_GAP}" if T_GAP else "the window gap of sim/20261119-segment_gap.py"
    plans_t = f"Table {T_PLANS}" if T_PLANS else PLANS_REF
    sec.append(("future", "Future work", f"""
<ul>
<li>Run each plan of {plans_t} three times on the hand, in a random order of layouts, with the fingertip boards
recorded and both cameras filming. Score the trials with trial_score and compute Spearman's ρ between the predicted
and the measured tracking over the layouts. A ρ near 0 over ten layouts falsifies the claim that this simulation ranks
twist layouts for this hand.</li>
<li>Run stage D's four plants on the proposed layouts outside its {spread.get('layouts', 150)} and score each plant's
predictions against the trials. A plant whose ρ with the hand exceeds that of the plant used here replaces it in the
next search.</li>
<li>Fit the offset between the tracked sphere and the model's fingertips at the grasp of each trial. Rerun
sim/20261119-segment_gap.py with it, so that a window starts without the 2-4 mm shift.</li>
<li>Add a slip model to the plant: a static friction above the sliding one in the gear train or the pads, or the joints'
give past the potentiometers. Compare its window gap with {gap_ref}.</li>
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

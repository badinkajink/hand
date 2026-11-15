#!/usr/bin/env python3
"""Builds docs/experiments/20261112-inner_loop_benchmark/20261113-sampling_mpc_vs_hom.html: sampling MPC (predictive
sampling, MPPI, CEM, iCEM, DIAL-MPC's annealing, SPIDER and its worst case) against HOM on the inner-loop benchmark's
twist of the 40 mm sphere, on the hand-design talk's plant (owner, 2026-11-12: "let's try MPC"; 2026-11-13: "Are there
videos / comparisons / a docpage hosting this for me to analyze?", HOM-space controls, MPC without the sphere-packed
model, and "any reasonably fast and simple way to improve HOM performance?").

Data: rows/mpc_<method>_twist[_mu<mu>][_<tag>].jsonl of scripts/hom/inner_loop_mpc.py (one row per plant world: J,
held, the usability metrics of scripts/hom/inner_loop_usability.py, the replanning time); rows/mpc_timing_twist.jsonl;
the films media/20261113-film_mpc_twist_*.mp4 and their rows/film_mpc_twist_*.json. Rows with the suffix _work10 are
the runs of 11-12 and 11-13 on the work plant (4 N m/rad, 1 N m, 10 MPa pads; scripts/hom/inner_loop_plant.py).

    .venv/bin/python scripts/hom/inner_loop_mpc_page.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
from sr2hand.numerics.stats import spearman  # noqa: E402

D = ROOT / "docs/experiments/20261112-inner_loop_benchmark"
MEDIA, ROWS = D / "media", D / "rows"
OUT = D / "20261113-sampling_mpc_vs_hom.html"
TEX_CACHE = D / "texsvg_cache.json"
BENCH = "20261112-inner_loop_benchmark.html"
REVIEW = "../20261029-hom_control_review/20261029-hom_control_and_fingertip_sensing.html"
fill, table, esc = P.fill, P.table, P.esc
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
VARIANTS = (("rate50", "exact, 50 Hz"), ("notip", "tag tracker, no fingertip sensing"),
            ("sense", "tag tracker, contact points and forces with Gaussian noise"),
            ("tip", "tag tracker and the magnetic fingertips' model"),
            ("cal_tip", "the same with calibration errors (joints 1 deg, object frame 1 mm)"))
PERIOD_S = 0.01
# (file stem after mpc_, label); the stems are inner_loop_mpc.py's rows names on the servo plant
MAIN = [("hom_twist", "HOM"), ("ps_twist_r10h10k2i1", "Predictive sampling, real-time budget"),
        ("ps_twist", "Predictive sampling"), ("mppi_twist", "MPPI"), ("cem_twist", "CEM"), ("icem_twist", "iCEM"),
        ("anneal_twist", "DIAL annealing"), ("spider_twist", "SPIDER"), ("spider_wc_twist", "SPIDER, worst case")]
REFS = [
    '[1] T. Howell, N. Gileadi, S. Tunyasuvunakool, K. Zakka, T. Erez and Y. Tassa, "Predictive Sampling: Real-time '
    'Behaviour Synthesis with MuJoCo", arXiv 2212.00541, 2022.',
    '[2] G. Williams, A. Aldrich and E. Theodorou, "Model Predictive Path Integral Control using Covariance Variable '
    'Importance Sampling", arXiv 1509.01149, 2015.',
    '[3] C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolinek and G. Martius, "Sample-efficient '
    'Cross-Entropy Method for Real-time Planning" (iCEM), CoRL 2020.',
    '[4] H. Xue, C. Pan, Z. Yi, G. Qu and G. Shi, "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control '
    'via Diffusion-Style Annealing" (DIAL-MPC), ICRA 2025.',
    '[5] C. Pan et al., "SPIDER: Scalable Physics-Informed Dexterous Retargeting", 2025; optimizer '
    'spider/optimizers/sampling.py.',
    '[6] W. Li and E. Todorov, "Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems", '
    'ICINCO 2004.',
    '[7] Y. Tassa, T. Erez and E. Todorov, "Synthesis and Stabilization of Complex Behaviors through Online Trajectory '
    'Optimization", IROS 2012.',
    '[8] C. Mastalli et al., "Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control" (FDDP), '
    'ICRA 2020.',
]


# ------------------------------------------------------------------------------------------------ data
def load(stem):
    p = ROWS / f"mpc_{stem}.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def key(r):
    return (r["x_sep"], r["y_sep"], r["grasp"], r["seed"])


def stats(rows, hom=None):
    """Mean J, held share, the share above HOM on the same pair and seed, and medians of the usability metrics over the
    held rollouts."""
    if not rows:
        return None
    H = [r for r in rows if r["held"]]
    med = (lambda k: float(np.median([r[k] for r in H if r.get(k) is not None]))
           if any(r.get(k) is not None for r in H) else float("nan"))       # drift_max_mm: spheres only
    out = {"n": len(rows), "J": float(np.mean([r["J"] for r in rows])),
           "J_se": float(np.std([r["J"] for r in rows], ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0,
           "held": float(np.mean([bool(r["held"]) for r in rows])),
           "err": float(np.mean([r["track_err_rms_deg"] for r in H])) if H else float("nan"),
           "contact": float(np.mean([r["contact_all_frac"] for r in H])) if H else float("nan")}
    for k in ("grip_mean_N", "link_mean_N", "pad_share", "sat_frac", "limit_frac", "cmd_rate_dps", "reversals_ps",
              "drift_max_mm"):
        out[k] = med(k)
    if hom:
        both = [r for r in rows if key(r) in hom]
        out["above"] = float(np.mean([r["J"] > hom[key(r)]["J"] for r in both])) if both else float("nan")
    walls = [r["replan_s_mean"] for r in rows if r.get("replan_s_mean")]
    out["replan_s"] = float(np.mean(walls)) if walls else None
    out["rollouts"] = rows[0].get("rollouts_per_replan")
    out["plant"] = rows[0].get("plant")
    return out


def timing_rows():
    """The timing rows of every task (mpc_timing_<task>.jsonl) on the SCS0009 plant."""
    out = []
    for p in sorted(ROWS.glob("mpc_timing_*.jsonl")):
        out += [r for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip()) if r.get("plant") == SERVO]
    return out


def one_hand(T, n, iters, H, model="pads"):
    """GPU seconds per replanning of one hand alone (timing rows, the last of each setting)."""
    m = [r for r in T if r["plant_worlds"] == 1 and r["n"] == n and r.get("iters") == iters and r["H"] == H
         and r.get("model", "pads") == model and r.get("plan_dt") is None]
    return m[-1]["replan_s"] if m else None


def f2(x, d=2):
    return "&#8212;" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def paired(ra, rb, n_boot=4000, seed=0):
    """J(a) - J(b) on the same worlds: the mean over the pairs (seeds averaged within a pair), its 95 % bootstrap
    interval over the pairs and the number of pairs where a is higher."""
    A, B = {key(r): r["J"] for r in ra}, {key(r): r["J"] for r in rb}
    d = {}
    for k in set(A) & set(B):
        d.setdefault(k[:3], []).append(A[k] - B[k])
    if len(d) < 4:
        return None
    v = np.array([np.mean(x) for x in d.values()])
    bs = np.random.default_rng(seed).choice(v, (n_boot, len(v))).mean(1)
    return {"d": float(v.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n": len(v), "wins": int((v > 0).sum())}


def sg(x):
    return f"{x:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000")


def fd(p):
    """A paired difference as text: +0.035 (-0.01 to +0.08), 12 of 20."""
    if not p:
        return "&#8212;"
    return f"{sg(p['d'])} ({sg(p['lo'])} to {sg(p['hi'])}), {p['wins']} of {p['n']}"


def fds(p):
    """A paired difference for a sentence: +0.035 (-0.010 to +0.080)."""
    return f"{sg(p['d'])} ({sg(p['lo'])} to {sg(p['hi'])})" if p else "&#8212;"


# contact models of the plant and of a planner: (rows tag, label); the pads are the benchmark's plant
PLANTS = [("pt", "Point contact"), ("pt_t0.5", "Point contact, 0.5 mm torsion"),
          ("pt_t1", "Point contact, 1 mm torsion"), ("pt_t2", "Point contact, 2 mm torsion"), ("pads", "Sphere pads")]
PLANT_SHORT = {"pt": "point", "pt_t0.5": "0.5 mm", "pt_t1": "1 mm", "pt_t2": "2 mm", "pads": "pads"}
# planner configurations: (iterations, horizon, knots, replanning interval in periods, planner step s or None, samples)
CFG_FULL, CFG_PSRT = (4, 30, 3, 10, None, 64), (1, 10, 2, 10, None, 64)
RT_CANDS = [(1, 10, 2, 10, None, 64), (2, 10, 2, 10, 0.002, 64), (1, 20, 3, 10, 0.002, 64), (3, 10, 2, 10, 0.002, 32),
            (3, 10, 2, 10, 0.002, 16), (2, 20, 3, 20, 0.002, 64), (2, 10, 2, 10, 0.002, 32), (2, 10, 2, 10, 0.002, 16),
            (3, 10, 2, 10, 0.002, 12)]      # logs/20261112-mpc/pick_rt.py's candidates
RT_TABLE = [("ps", CFG_PSRT), ("ps", (1, 20, 3, 10, 0.002, 64)), ("anneal", CFG_FULL), ("anneal", (2, 10, 2, 10, None, 64))] \
    + [("anneal", c) for c in RT_CANDS]


def cfg_tag(c):
    """The rows-name tag of inner_loop_mpc.py run for a configuration: planner step, samples, then the schedule."""
    it, h, k, r, dt, n = c
    return ((f"_pdt{1e3 * dt:g}" if dt else "") + (f"_n{n}" if n != 64 else "")
            + (f"_r{r}h{h}k{k}i{it}" if (r, h, k, it) != (10, 30, 3, 4) else ""))


def stem_of(method, cfg, plant="pads", model="pads", delay="", sh=""):
    """inner_loop_mpc.py's rows stem: mpc_<method>_twist[_plant_<plant>][<delay>][_<model>][<options>]<cfg_tag>, the
    options those of the reversal runs (_extrap, _lam<x>, _sig<x>, _rw<x>, _acc<x>, _rev<x>, _db<x>, in that order)."""
    return (f"{method}_twist" + ("" if plant == "pads" else f"_plant_{plant}") + delay
            + ("" if model == "pads" else f"_{model}") + sh + cfg_tag(cfg))


def hom_stem(plant):
    return "hom_twist" + ("" if plant == "pads" else f"_plant_{plant}")


def one_hand_cfg(T, c, model="pads"):
    """GPU seconds of one hand's replanning for a configuration (the newest timing row; rows of 11-13 morning carry
    no knots or planner step and count as 1 ms)."""
    it, h, k, r, dt, n = c
    m = [t for t in T if t["plant_worlds"] == 1 and t["n"] == n and t.get("iters") == it and t["H"] == h
         and t.get("knots", k) == k and t.get("replan", 10) == r and t.get("plan_dt") == dt
         and t.get("model", "pads") == model]
    return m[-1]["replan_s"] if m else None


def rt_pick(R, T):
    """The real-time DIAL configuration of the contact runs: the candidate whose point-contact plant rows exist (the
    lane's pick), else the real-time candidate with the highest mean J."""
    for c in RT_CANDS:
        if R.get(stem_of("anneal", c, plant="pt")):
            return c
    ok = [(np.mean([r["J"] for r in R[stem_of("anneal", c)]]), c) for c in RT_CANDS
          if len(R.get(stem_of("anneal", c), [])) == 40 and (one_hand_cfg(T, c) or 1e9) <= c[3] * PERIOD_S]
    return max(ok)[1] if ok else None


def cfg_text(c):
    it, h, k, r, dt, n = c
    return (f"{it} &#215; {n} over {h * PERIOD_S:g} s, {k} knots, every {r * PERIOD_S:g} s"
            + (f", {1e3 * dt:g} ms step" if dt else ""))


# ------------------------------------------------------------------------------------------------ figures
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
    return plt


FIG_LABEL = {"hom_twist": "HOM", "ps_twist_r10h10k2i1": "PS, real time", "ps_twist": "PS", "mppi_twist": "MPPI",
             "cem_twist": "CEM", "icem_twist": "iCEM", "anneal_twist": "DIAL", "spider_twist": "SPIDER",
             "spider_wc_twist": "SPIDER, worst case", "hs_anneal_twist": "DIAL, HOM space"}


def fig_overview(S, out):
    """J, grip and target reversals per controller, and the J of each pair under HOM and DIAL annealing."""
    plt = plt_setup()
    stems = [stem for stem, _lab in MAIN if S.get(stem)] + [k for k in ("hs_anneal_twist",) if S.get(k)]
    labs = [FIG_LABEL.get(k, k) for k in stems]
    st = [S[k] for k in stems]
    y = np.arange(len(labs))[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(8.6, 0.36 * len(labs) + 1.5), sharey=True)
    col = ["#555555" if k == "hom_twist" else "#8a5a2b" if k.startswith("hs_") else "#3a6ea5" for k in stems]
    ax[0].barh(y, [s["J"] for s in st], xerr=[s["J_se"] for s in st], color=col, capsize=3)
    ax[0].set_xlabel("Mean J")
    ax[1].barh(y, [s["grip_mean_N"] for s in st], color=col)
    ax[1].set_xlabel("Grip (N)")
    ax[2].barh(y, [s["reversals_ps"] for s in st], color=col)
    ax[2].set_xlabel("Reversals (1/s)")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labs)
    for a in ax:
        a.grid(axis="x", alpha=0.3)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def fig_pairs(rows_h, rows_m, label, out):
    plt = plt_setup()
    hom = {key(r): r for r in rows_h}
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for by, mk, nm in (("hom", "o", "pairs chosen by HOM's J"), ("uhas", "s", "pairs chosen by UHAS's J")):
        xs = [hom[key(r)]["J"] for r in rows_m if r["by"] == by and key(r) in hom]
        ys = [r["J"] for r in rows_m if r["by"] == by and key(r) in hom]
        ax.scatter(xs, ys, marker=mk, s=36, alpha=0.8, label=nm)
    ax.plot([0, 1], [0, 1], color="#888888", lw=1)
    ax.set_xlim(-0.02, 0.97)
    ax.set_ylim(-0.02, 0.97)
    ax.set_xlabel("J under HOM")
    ax.set_ylabel(f"J under {label}")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def contact_lines(S, rt):
    """The lines of the contact-model figure: (label, colour, line style, {plant: stats})."""
    lines = [("HOM", "#555555", "-", {p: S.get(hom_stem(p)) for p, _ in PLANTS})]
    for meth, cfg, lab, col in (("ps", CFG_PSRT, "PS, real time", "#c06a1a"), ("anneal", rt, "DIAL, real time",
                                                                                   "#3a6ea5"),
                                ("anneal", CFG_FULL, "DIAL, 4\u00d764", "#1b3a63")):
        if cfg is None:
            continue
        for model, ls, ml in (("pads", "-", "pad model"), ("pt", "--", "point-contact model"), ("own", ":",
                                                                                                 "plant's own model")):
            pts = {p: S.get(stem_of(meth, cfg, plant=p, model=p if model == "own" else model)) for p, _ in PLANTS
                   if not (model == "own" and p in ("pads", "pt"))}
            if any(pts.values()):
                lines.append((f"{lab}, {ml}", col, ls, pts))
    return lines


def fig_contact(S, R, rt, out):
    """Left: mean J against the plant's contact model. Right: J with the pad model minus J with the point-contact model
    (paired over the 20 pairs, 95 % bootstrap interval), for each planner."""
    plt = plt_setup()
    xs = {p: i for i, (p, _) in enumerate(PLANTS)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 6.2))
    left = [("HOM", "#555555", "-", "o", {p: S.get(hom_stem(p)) for p, _ in PLANTS})]
    for meth, cfg, lab, col in (("ps", CFG_PSRT, "PS, real time", "#c06a1a"), ("anneal", rt, "DIAL, real time", "#3a6ea5"),
                                ("anneal", CFG_FULL, "DIAL, 4\u00d764", "#1b3a63")):
        if cfg is None:
            continue
        left.append((f"{lab}, pad model", col, "-", "o", {p: S.get(stem_of(meth, cfg, plant=p)) for p, _ in PLANTS}))
        own = {p: S.get(stem_of(meth, cfg, plant=p, model="pt" if p == "pt" else p if p != "pads" else "pads"))
               for p, _ in PLANTS}
        left.append((f"{lab}, plant's own model", col, ":", "^", own))
    for k, (lab, col, ls, mk, pts) in enumerate(left):
        P_ = sorted((xs[p], s_["J"]) for p, s_ in pts.items() if s_)
        if not P_:
            continue
        a1.plot([x + 0.03 * (k - 3) for x, _ in P_], [y for _, y in P_], color=col, marker=mk, ms=5, lw=1.6,
                ls=ls if len(P_) == len(PLANTS) else "none", label=lab)     # no line across plants not run
    a1.set_ylabel("Mean J over 40 runs")
    a1.set_title("Mean J", fontsize=13)
    for meth, cfg, lab, col in (("ps", CFG_PSRT, "PS, real time", "#c06a1a"), ("anneal", rt, "DIAL, real time",
                                                                             "#3a6ea5"),
                                ("anneal", CFG_FULL, "DIAL, 4\u00d764", "#1b3a63")):
        if cfg is None:
            continue
        pts = []
        for p, _ in PLANTS:
            d = paired(R.get(stem_of(meth, cfg, plant=p), []), R.get(stem_of(meth, cfg, plant=p, model="pt"), []))
            if d:
                pts.append((xs[p], d))
        if not pts:
            continue
        off = {"PS, real time": -0.08, "DIAL, real time": 0.0}.get(lab, 0.08)
        a2.errorbar([x + off for x, _ in pts], [d["d"] for _, d in pts],
                    yerr=[[d["d"] - d["lo"] for _, d in pts], [d["hi"] - d["d"] for _, d in pts]], color=col, marker="s",
                    ms=5, capsize=3, lw=1.6, ls="-" if len(pts) == len(PLANTS) else "none",
                    label=f"{lab}: pad minus point-contact model")
    a2.axhline(0.0, color="#888888", lw=1)
    a2.set_ylabel("Difference in J")
    a2.set_title("Pad minus point-contact model", fontsize=13)
    for a in (a1, a2):
        a.set_xticks(range(len(PLANTS)))
        a.set_xticklabels([PLANT_SHORT[p] for p, _ in PLANTS])
        a.set_xlabel("Plant's fingertip contact")
        a.grid(alpha=0.3)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.27, 1, 1))
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


# the reversal options (owner, 2026-11-13): (method, configuration, option tag, label, kind)
CFG_REV = (3, 10, 2, 10, 0.002, 16)
CFG_PICK = (2, 10, 2, 10, 0.002, 32)     # the real-time pick of logs/20261112-mpc/pick_rt.py
REV_RUNS = [("anneal", CFG_REV, "", "None (3 &#215; 16, 2 ms step)", "&#8212;"),
            ("anneal", CFG_REV, "_lam0.3", "Temperature 0.3", "sampling"),
            ("anneal", CFG_REV, "_lam1", "Temperature 1.0", "sampling"),
            ("anneal", CFG_REV, "_extrap", "Warm start continues the slope", "sampling"),
            ("anneal", CFG_REV, "_extrap_lam1", "Slope continued, temperature 1.0", "sampling"),
            ("anneal", CFG_REV, "_sig0.04", "Sampling noise 0.04 rad", "sampling"),
            ("anneal", CFG_REV, "_rw4", "Knot-rate weight 4", "cost"),
            ("anneal", CFG_REV, "_acc1", "Rate-change cost, weight 1", "cost"),
            ("anneal", CFG_REV, "_acc4", "Rate-change cost, weight 4", "cost"),
            ("anneal", CFG_REV, "_rev10", "Reversal cost, weight 10", "cost"),
            ("anneal", CFG_REV, "_db3", "Dead band of 3 deg on the error", "cost"),
            ("anneal", CFG_REV, "_extrap_lam1_acc1", "Slope continued, temperature 1.0, rate-change cost 1", "both"),
            ("anneal", CFG_PICK, "", "Real-time pick, 2 &#215; 32: none", "&#8212;"),
            ("anneal", CFG_PICK, "_acc1", "Real-time pick: rate-change cost 1", "cost"),
            ("anneal", CFG_PICK, "_acc1_db3", "Real-time pick: rate-change cost 1, dead band 3 deg", "cost"),
            ("anneal", CFG_PICK, "_acc2_db3", "Real-time pick: rate-change cost 2, dead band 3 deg", "cost"),
            ("ps", CFG_PSRT, "", "PS real time: none", "&#8212;"),
            ("ps", CFG_PSRT, "_extrap", "PS real time: slope continued", "sampling"),
            ("anneal", CFG_FULL, "", "DIAL 4 &#215; 64: none", "&#8212;"),
            ("anneal", CFG_FULL, "_lam1", "DIAL 4 &#215; 64: temperature 1.0", "sampling"),
            ("anneal", CFG_FULL, "_acc1", "DIAL 4 &#215; 64: rate-change cost 1", "cost")]
REV_REF = [("hom_twist", "HOM"), ("ps_twist_r10h10k2i1", "PS, real time"), ("anneal_twist", "DIAL"),
           ("icem_twist", "iCEM"), ("spider_twist", "SPIDER"), ("mppi_twist", "MPPI"), ("cem_twist", "CEM"),
           ("ps_twist", "PS")]
REV_LABEL_AT = {14: (-4, 7)}      # option number -> label offset (points) where a label would cover a marker


def fig_reversals(S, out):
    """Mean J against target reversals: the benchmark's controllers, and the reversal options on the real-time DIAL."""
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for stem, lab in REV_REF:
        s = S.get(stem)
        if s:
            ax.plot(s["reversals_ps"], s["J"], "o", color="#555555" if stem == "hom_twist" else "#999999", ms=7)
            ax.annotate(lab, (s["reversals_ps"], s["J"]), textcoords="offset points",
                        xytext={"PS": (-22, -4), "MPPI": (6, -4)}.get(lab, (5, 4)), fontsize=10, color="#555555")
    bases = {c: S.get(stem_of("anneal", c)) for c in (CFG_REV, CFG_PICK)}
    col = {"sampling": "#3a6ea5", "cost": "#c06a1a", "both": "#7a3a8a", "&#8212;": "#1b3a63"}
    k = 0
    for meth, cfg, sh, lab, kind in REV_RUNS:
        s = S.get(stem_of(meth, cfg, sh=sh))
        if not s or cfg not in bases:
            continue
        k += 1
        ax.plot(s["reversals_ps"], s["J"], "s" if kind == "cost" else "D" if kind == "both" else "^", color=col[kind],
                ms=7, mfc="white" if cfg == CFG_PICK else col[kind], mew=1.5)
        ax.annotate(str(k), (s["reversals_ps"], s["J"]), textcoords="offset points",
                    xytext=REV_LABEL_AT.get(k, (5, -10)), fontsize=10, color=col[kind])
        base = bases[cfg]
        if base and sh:
            ax.annotate("", (s["reversals_ps"], s["J"]), (base["reversals_ps"], base["J"]),
                        arrowprops=dict(arrowstyle="->", color=col[kind], alpha=0.35, lw=1))
    ax.set_xlabel("Target reversals per joint per second (median over held runs)")
    ax.set_ylabel("Mean J over 40 runs")
    ax.plot([], [], "^", color=col["sampling"], label="sampling options")
    ax.plot([], [], "s", color=col["cost"], label="cost options")
    ax.plot([], [], "D", color=col["both"], label="both")
    ax.plot([], [], "s", color="#555555", mfc="white", mew=1.5, label="open: on the 2 × 32 pick")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def rt_rows(S, T):
    """The configurations of the real-time table and figure that have rows and a timing: (method, config, stats, s)."""
    out = []
    for meth, cfg in RT_TABLE:
        s, v = S.get(stem_of(meth, cfg)), one_hand_cfg(T, cfg)
        if s and v is not None:
            out.append((meth, cfg, s, v))
    return out


def fig_realtime(S, T, out):
    """Mean J against the GPU time per second of control of one hand, for predictive sampling and DIAL annealing."""
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    pts = []
    for num, (meth, cfg, s, v) in enumerate(rt_rows(S, T), 1):
        g = v / (cfg[3] * PERIOD_S)
        col, mk = ("#3a6ea5", "o") if meth == "anneal" else ("#c06a1a", "s")
        ax.errorbar([g], [s["J"]], yerr=[s["J_se"]], fmt=mk, capsize=2, color=col, ms=6)
        pts.append((num, g, s["J"], col, mk))
    box = (0.86, 1.09, 0.32, 0.45)                 # the cluster at the real-time line, drawn again in the inset
    inside = [x for x in pts if box[0] <= x[1] <= box[1] and box[2] <= x[2] <= box[3]]
    for num, g, j, col, _mk in pts:
        if (num, g, j, col, _mk) not in inside:
            ax.annotate(str(num), (g, j), textcoords="offset points", xytext=(5, 2), fontsize=11, color=col)
    if len(inside) > 2:
        ins = ax.inset_axes([0.42, 0.10, 0.34, 0.42])
        for num, g, j, col, mk in inside:
            ins.plot([g], [j], mk, color=col, ms=6)
            ins.annotate(str(num), (g, j), textcoords="offset points", xytext=(5, 1), fontsize=11, color=col)
        ins.axvline(1.0, color="#aa3333", lw=1.2)
        ins.set_xlim(box[0], box[1])
        ins.set_ylim(box[2], box[3])
        ins.tick_params(labelsize=10)
        ins.grid(alpha=0.3)
        ax.indicate_inset_zoom(ins, edgecolor="#888888")
    hs = S.get("hom_twist")
    if hs:
        ax.axhline(hs["J"], color="#555555", ls="--", lw=1.2)
        ax.text(0.97, hs["J"], " HOM", transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=11,
                color="#555555")
    ax.axvline(1.0, color="#aa3333", lw=1.2)
    ax.text(1.0, 0.02, " real time", transform=ax.get_xaxis_transform(), color="#aa3333", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("GPU time per second of control, one hand")
    ax.set_ylabel("Mean J over 40 runs")
    ax.plot([], [], "s", color="#c06a1a", label="Predictive sampling")
    ax.plot([], [], "o", color="#3a6ea5", label="DIAL annealing")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ sections
def sec_terms():
    rows = [
        ["J", "tracking score of one run: $J = \\max(0, 1 - e/e_{\\mathrm{ref}}) \\cdot c$, with $e$ the RMS angle "
         "between the sphere's orientation and the commanded one over the task, $e_{\\mathrm{ref}}$ the RMS commanded "
         "rotation (59.5 deg on these pairs) and $c$ the share of the task with all three pads above 0.05 N; 0 when the "
         "sphere fell", "0 to 1; 1 follows the command on three pads"],
        ["Held", "the sphere did not fall 15 mm below its start before the end", "share of runs"],
        ["Above HOM", "share of runs whose J exceeds HOM's on the same pair and seed", "share of 40 runs"],
        ["Tracking error", "$e$ above, mean over the held runs", "deg"],
        ["Three-pad share", "$c$ above, mean over the held runs", "0 to 1"],
        ["Grip", "total normal force of the three fingers on the sphere, mean over the task's periods; median over the "
         "held runs", "N"],
        ["Rigid-part force", "the part of the grip through the links, cassette and cap, not the pads; same averaging",
         "N"],
        ["Saturation", "share of joint-periods with the servo's torque at 95 % of its limit (0.18 N m) or more", "0 to 1"],
        ["Target reversals", "sign changes of a servo target's rate where the rate exceeds 1 deg/s, per joint per second",
         "1/s"],
        ["Target rate", "RMS rate of the servo targets over the joints and periods", "deg/s"],
        ["Replanning time", "wall time of one replanning, every 0.1 s of control unless stated; for 16 hands planned "
         "together (the runs) or one hand alone (the timing runs)", "s"],
        ["GPU time per second of control", "replanning time of one hand alone over the replanning interval; 1 or less "
         "runs in real time", "s/s"],
        ["Point contact", "one contact between a tip's convex mesh and the sphere, with no torque about its normal",
         "&#8212;"],
        ["Torsional friction coefficient $\\mu_t$", "the largest torque about a point contact's normal per newton of "
         "normal force (MuJoCo's condim 4)", "mm"],
        ["Paired difference", "J of one controller minus J of another on the same worlds, seeds averaged within a pair; "
         "the mean over the 20 pairs with its 95 % interval from 4,000 bootstrap resamples of the pairs", "J"],
        ["Rank correlation $\\rho$", "Spearman's rank correlation between two controllers' J on the 20 pairs, seeds "
         "averaged within a pair (<code>sr2hand.numerics.stats.spearman</code>)", "-1 to 1; 1 is the same order"],
    ]
    return table(["Term", "Definition", "Unit or range"], rows, "Terms and metrics. Every metric is computed on the task "
                 "phase, from HOM's handover at 2.4 s to the end of the schedule.", num_from=9)


def sec_setup(S):
    return fill(r"""<p>The task is the inner-loop benchmark's twist of the 40 mm sphere (<a href="<<b>>">benchmark page</a>).
HOM closes the hand, squeezes and lifts. From 2.4 s the controller under test drives the nine servo targets. The
schedule commands 40 deg/s about the vertical for 0.5 s and then back for 0.5 s. A 0.3 s rest and a 30 deg/s ramp
for 3 s follow. The plant is the hand-design talk's. Each servo is the SCS0009 model of 0.72 N m/rad with a 0.18 N m
limit, and the v4b tips are sphere pads at 2.5 MPa with friction 1.0. MuJoCo-Warp 3.6 steps the physics every 1 ms,
and the controller runs every 10 ms. Twenty pairs of a layout and a grasp were run, each at seeds 0 and 1. They are the
ten test layouts with HOM's best J and the ten with UHAS's best J (<code>rows/mpc_pairs_twist.json</code>).</p>
<p>The runs of 12 November and of 13 November before 10:30 were made on another plant by mistake. The scripts set no
plant, so the default was used: the work plant of 4 N m/rad and 1 N m per joint, with 10 MPa pads. That plant is 5.6
times as strong as the SCS0009. Those rows are kept under the suffix <code>_work10</code> and summarized in Section <<w>>. Since
then <code>scripts/hom/inner_loop_plant.py</code> sets the plant before any physics is loaded, and every row records it.</p>""",
                b=BENCH, w="WORKSEC")


def sec_methods():
    t = r"""<p>Every planner rolls out the plant's own MuJoCo-Warp model, except the point-contact planner of its own section. Every 0.1 s
the full state of each plant world is copied into its planner worlds: positions, velocities, servo targets and the
solver's warm start. A plan consists of the nine servo targets at three knots over 0.3 s, linear between them. Its
first knot is pinned to the target applied now, so a plan has 18 free numbers. The plant applies the first 0.1 s. In
each replanning, 4 iterations of 64 samples are rolled out per hand, 256 rollouts. The real-time budget is 1 iteration
of 64 rollouts over 0.1 s with two knots. The cost of a rollout is summed over its periods:</p>
$$c = \sum_t \Big[\Big(\frac{e_t}{5^\circ}\Big)^2 + \Big(\frac{d_t}{3\,\mathrm{mm}}\Big)^2 + 2\,n^{\mathrm{off}}_t
+ 100\,\mathbb{1}_{\mathrm{fell}} + 0.5 \sum_j \mathrm{clip}\Big(\frac{k_p |u_j - q_j|/\tau_{\max} - 0.8}{0.2}, 0, 3\Big)
+ 0.25 \sum_j \mathrm{clip}\Big(\frac{3^\circ - \delta_j}{3^\circ}, 0, 1\Big)\Big]
+ \sum_{\mathrm{knots}} \sum_j \Big(\frac{\dot u_j}{20^\circ/\mathrm{s}}\Big)^2$$
<p>Here $e_t$ is the angle to the commanded orientation and $d_t$ the sphere's distance from its start. The number of
pads under 0.05 N is $n^{\mathrm{off}}_t$. A servo's target and angle are $u_j$ and $q_j$, its joint's distance to the
range limit is $\delta_j$, and $\dot u_j$ is the target's rate between knots. The planners differ in how the plan is
updated from the sampled costs:</p>"""
    rows = [["Predictive sampling [1]", "the cheapest of the nominal plan and its noisy copies"],
            ["MPPI [2]", "weights $\\exp(-(c - c_{\\min})/\\lambda)$ with $\\lambda = 0.03\\,\\mathrm{std}(c)$"],
            ["CEM", "the cheapest 15 % set the mean and the spread; the mean is executed"],
            ["iCEM [3]", "CEM with correlated knot noise (0.7); the two best kept into the next replanning"],
            ["DIAL annealing [4]", "noise shrinking with the iteration and towards the first knot; an MPPI step per "
             "iteration"],
            ["SPIDER [5]", "noise from 0.5 to 1.0 of its scale along the horizon, times $0.1^{i/I}$ in iteration $i$; a "
             "softmax at temperature 0.3 over the best 10 %"],
            ["SPIDER, worst case [5]", "SPIDER with each sample's cost the worst over frictions 0.4, 0.7 and 1.0, a "
             "third of the samples in each"]]
    t += table(["Planner", "Update"], rows, "The sampling planners (<code>scripts/hom/inner_loop_mpc.py</code>).",
               num_from=9)
    t += r"""<p>HOM is the controller of the talk (<code>scripts/hom/hom_gpu.py</code>, batched in torch on the GPU).
Its twist command is the schedule's angular velocity alone. The section on simple changes to HOM adds feedback on
the sphere's measured orientation.</p>"""
    return t


def sec_results(S, figs):
    hs = S.get("hom_twist")
    if not hs:
        return ""
    body = []
    for stem, lab in MAIN:
        s = S.get(stem)
        if not s:
            continue
        body.append([lab, f2(s["J"], 3), f"{100 * s['held']:.0f}", "&#8212;" if stem == "hom_twist" else
                     f"{100 * s['above']:.0f}", f2(s["err"], 1), f2(s["contact"]), f2(s["grip_mean_N"], 1),
                     f2(s["link_mean_N"]), f"{100 * s['sat_frac']:.0f}", f2(s["reversals_ps"]),
                     f2(s["cmd_rate_dps"], 1)])
    best = max((S[stem]["J"], lab) for stem, lab in MAIN if S.get(stem) and stem != "hom_twist") if len(body) > 1 \
        else None
    t = ""
    if best:
        t += fill(r"""<p>On the SCS0009 plant HOM's mean J was <<h>>, with the sphere kept in <<hh>> % of the runs.
The best planner was <<bl>>, at <<bj>>. Every controller is listed in Table <<tn>>; the medians of the usability
metrics are taken over the held runs.</p>""", h=f"{hs['J']:.3f}", hh=f"{100 * hs['held']:.0f}", bl=best[1], bj=f"{best[0]:.3f}",
                  tn=P.TAB[0] + 1)
    t += table(["Controller", "Mean J", "Held (%)", "Above HOM (%)", "Tracking error (deg)", "Three-pad share",
                "Grip (N)", "Rigid-part force (N)", "Saturation (%)", "Target reversals (1/s)", "Target rate (deg/s)"],
               body, "Twenty pairs at seeds 0 and 1 on the SCS0009 plant at 2.5 MPa, plant friction 1.0; the planners "
               "plan with the plant's own model.")
    if figs.get("overview"):
        t += P.img(figs["overview"], "Mean J (bars: one standard error over the 40 runs), median grip and target "
                   "reversals per joint per second on the SCS0009 plant. HOM in grey, the planners over servo targets "
                   "in blue, DIAL annealing in HOM's space in brown. PS: predictive sampling; DIAL: DIAL-MPC's "
                   "annealing.")
    if figs.get("pairs"):
        t += P.img(figs["pairs"], "J of each run under HOM and under DIAL annealing on the same pair and seed. Above "
                   "the diagonal the planner tracked better.")
    return t


def pair_j(rows, seed=None):
    """J per pair (layout, grasp): the mean over its seeds, or one seed's."""
    d = {}
    for r in rows:
        if seed is None or r["seed"] == seed:
            d.setdefault(key(r)[:3], []).append(r["J"])
    return {k: float(np.mean(v)) for k, v in d.items()}


def sec_rank(S, R, T):
    """HOM against the planners on mean J and on the order of the pairs."""
    runs = [("hom_twist", "HOM", None), ("ps_twist_r10h10k2i1", "Predictive sampling, real time", CFG_PSRT),
            (stem_of("anneal", CFG_PICK), "DIAL, real-time pick (2 &#215; 32)", CFG_PICK),
            (stem_of("anneal", CFG_PICK, sh="_acc1"), "DIAL, real-time pick, rate-change cost 1", CFG_PICK),
            ("anneal_twist", "DIAL, 4 &#215; 64", CFG_FULL), ("icem_twist", "iCEM", CFG_FULL),
            ("hs_anneal_twist", "DIAL in HOM's space", None)]
    if not all(R.get(stem) for stem in ("hom_twist", "anneal_twist", stem_of("anneal", CFG_PICK), "icem_twist")):
        return ""
    keys = sorted(pair_j(R["hom_twist"]))
    vec = (lambda stem, seed=None: [pair_j(R[stem], seed).get(k, np.nan) for k in keys])
    rho = (lambda a, b: spearman(vec(a), vec(b)))
    rows, dd = [], {}
    for stem, lab, cfg in runs:
        s = S.get(stem)
        if not s:
            continue
        dd[stem] = paired(R[stem], R["hom_twist"]) if stem != "hom_twist" else None
        v = one_hand_cfg(T, cfg) if cfg else None
        rows.append([lab, f2(s["J"], 3), fd(dd[stem]) if dd[stem] else "&#8212;",
                     "&#8212;" if stem == "hom_twist" else f2(rho(stem, "hom_twist")),
                     "&#8212;" if stem == "anneal_twist" else f2(rho(stem, "anneal_twist")),
                     f2(spearman(vec(stem, 0), vec(stem, 1))),
                     "0.20, one CPU core" if stem == "hom_twist" else "&#8212;" if not v else
                     f2(v / (cfg[3] * PERIOD_S), 2 if v < cfg[3] * PERIOD_S * 10 else 1)])
    h, dl = np.array(vec("hom_twist")), np.array(vec("anneal_twist"))
    top_h, top_d = set(np.argsort(-h)[:5]), set(np.argsort(-dl)[:5])
    t = fill(r"""<p>On mean J the planners that run in real time were within the noise of HOM. Against HOM, the paired
difference was <<dp>> for the real-time pick. It was <<dps>> for real-time predictive sampling. Planning one hand with
DIAL at 4 &#215; 64 samples or with iCEM takes <<v>> s of GPU per second of control. Their J was above HOM's by <<dd>>
and <<di>>.</p>
<p>In the design search a controller scores each layout and grasp, so the order it gives the pairs matters there as
much as its mean J. The 20 pairs were ranked under HOM nearly as under DIAL at 4 &#215; 64, with $\rho$ = <<rd>>
(Table <<tn>>). DIAL's order was followed as closely by the real-time pick, at <<rp>>, and by iCEM, at <<ri>>. The
order changed little between seeds: $\rho$ between seeds 0 and 1 was <<sh>> for HOM and <<sd>> for DIAL. DIAL's five
best pairs included <<ov>> of HOM's five best. On HOM's best pair DIAL reached <<jhb>>. DIAL's own best pair scored
<<jdb>> under DIAL and <<jhd>> under HOM.</p>
<p>The 20 pairs are the ten test layouts with HOM's best J and the ten with UHAS's best J (Section 2). Under DIAL at
4 &#215; 64 their J spans <<lo>> to <<hi>>.</p>""",
             dp=fds(dd[stem_of("anneal", CFG_PICK)]), dps=fds(dd.get("ps_twist_r10h10k2i1")),
             v=f"{one_hand_cfg(T, CFG_FULL) / (CFG_FULL[3] * PERIOD_S):.1f}", dd=fds(dd["anneal_twist"]),
             di=fds(dd["icem_twist"]), rd=f2(rho("hom_twist", "anneal_twist")), tn=P.TAB[0] + 1,
             rp=f2(rho(stem_of("anneal", CFG_PICK), "anneal_twist")), ri=f2(rho("icem_twist", "anneal_twist")),
             sh=f2(spearman(vec("hom_twist", 0), vec("hom_twist", 1))),
             sd=f2(spearman(vec("anneal_twist", 0), vec("anneal_twist", 1))),
             ov=("none", "one", "two", "three", "four", "all five")[len(top_h & top_d)],
             jhb=f"{dl[np.argmax(h)]:.3f}", jdb=f"{dl.max():.3f}", jhd=f"{h[np.argmax(dl)]:.3f}",
             lo=f"{dl.min():.2f}", hi=f"{dl.max():.2f}")
    return t + table(["Controller", "Mean J", "J minus HOM's", "&#961; with HOM", "&#961; with DIAL 4 &#215; 64",
                      "&#961; between seeds 0 and 1", "Compute per second of control (s/s)"], rows,
                     "Mean J and the order of the 20 pairs under each controller, seeds 0 and 1 on the SCS0009 plant. "
                     "Compute: GPU time for one hand alone, for HOM the time of one CPU core.", num_from=1)


FILM_NAME = {"hom": "HOM", "ps_rt": "predictive sampling at the real-time budget", "anneal": "DIAL annealing",
             "hs_anneal": "DIAL annealing in HOM's space", "uhas": "UHAS"}
FILM_STEM = {"hom": "hom_twist", "ps_rt": "ps_twist_r10h10k2i1", "anneal": "anneal_twist", "hs_anneal": "hs_anneal_twist"}
TABLE_ROWS = {}


def sec_films():
    out = ""
    for p in sorted(MEDIA.glob("20261113-film_mpc_twist_*.mp4")):
        info_p = ROWS / (p.stem.replace("20261113-", "") + ".json")
        if not info_p.exists():
            continue
        info = json.loads(info_p.read_text())
        pr = info["pair"]
        cl = info["clips"]
        plant = info.get("plant_model", "pads")
        parts = []
        k = (pr["x"], pr["y"], pr["grasp"], info["seed"])
        for spec, c in cl.items():
            nm, _, arg = spec.partition(":")
            if nm == "hom":
                stem = hom_stem(plant)
            elif nm in ("ps_rt", "anneal"):
                stem = stem_of("ps" if nm == "ps_rt" else nm, CFG_PSRT if nm == "ps_rt" else CFG_FULL, plant=plant,
                               model=arg or "pads")
            else:
                stem = FILM_STEM.get(nm, "")
            tab = TABLE_ROWS.get(stem, {}).get(k)
            nm_ = FILM_NAME.get(nm, esc(spec))
            if nm in ("ps_rt", "anneal", "hs_anneal") and (arg or plant != "pads"):
                nm_ += {"pads": " with the pad model", "pt": " with the point-contact model"}.get(
                    arg or "pads", f" with the {esc(arg)} model")
            parts.append(f"{nm_[0].upper() + nm_[1:]}: J {c['J']:.2f}"
                         + (f" (in the table's run {tab['J']:.2f})" if tab and nm != "hom" else "")
                         + f", grip {c['grip_mean_N']:.1f} N, {c['reversals_ps']:.1f} reversals per joint per s")
        cap = (f"Layout ({pr['x']:g}, {pr['y']:g}) mm, grasp {pr['grasp']}, seed {info['seed']}, the plant's fingertips "
               f"{'sphere pads' if plant == 'pads' else dict(PLANTS)[plant].lower()}, plant friction "
               f"{info['plant_mu']:g}; the task phase from 2.4 s. " + ". ".join(parts) + ". A planner's run in the "
               "film draws other samples than its run in the table. Each frame gives each finger's pad force and the "
               "force through its rigid parts on the sphere (N); the links are drawn see-through and the sphere's "
               "axes as rods.")
        out += P.film(p, cap)
    return out


def sec_compute(T, S):
    if not T:
        return ""
    rows = []
    for lab, n, it, H, model in (("Planner, 256 rollouts over 0.3 s (4 x 64)", 64, 4, 30, "pads"),
                                 ("Planner, 64 rollouts over 0.15 s", 64, 1, 15, "pads"),
                                 ("Planner, real-time budget: 64 rollouts over 0.1 s", 64, 1, 10, "pads"),
                                 ("Planner with point contact, 4 x 64 over 0.3 s", 64, 4, 30, "pt")):
        v = one_hand(T, n, it, H, model)
        if v is not None:
            rows.append([lab, f2(v, 3), f2(v / (10 * PERIOD_S), 2)])
    rows.append(["HOM, one CPU core (CAD v1 hand)", "0.0020 per 10 ms period", "0.20"])
    t = table(["Controller, one hand", "Replanning time (s)", "GPU time per second of control"], rows,
              "Compute per second of control for one hand on the RTX 4070 Ti SUPER (rows/mpc_timing_twist.jsonl); HOM's "
              "controller timed on one core of this workstation over 1,560 periods (median 2.01 ms, p99 2.09 ms).",
              num_from=1)
    return t + r"""<p>A planner world costs about as much to simulate as a plant world. A plan of 256 rollouts of 30
periods therefore costs about 7,700 plant periods per 10 control periods. HOM costs one controller evaluation per period
and no rollouts.</p>"""


def sec_feedback(S):
    rows = []
    hs = S.get("hom_twist")
    for g in (1, 2, 4, 8):
        s = S.get(f"hom_twist_att{g}")
        if s:
            rows.append([f"{g}", f2(s["J"], 3), f"{100 * s['held']:.0f}", f"{100 * s['above']:.0f}", f2(s["err"], 1),
                         f2(s["contact"]), f2(s["grip_mean_N"], 1), f"{100 * s['sat_frac']:.0f}",
                         f2(s["reversals_ps"])])
    if not rows or not hs:
        return ""
    rows.insert(0, ["0 (HOM)", f2(hs["J"], 3), f"{100 * hs['held']:.0f}", "&#8212;", f2(hs["err"], 1),
                    f2(hs["contact"]), f2(hs["grip_mean_N"], 1), f"{100 * hs['sat_frac']:.0f}",
                    f2(hs["reversals_ps"])])
    t = r"""<p>HOM's twist command is the schedule's angular velocity $\omega_{\mathrm{ref}}$, expressed in the
measured object frame. Nothing in it reacts to the angle the sphere has actually turned, so slip and lag accumulate. The
commanded orientation $R_c$ is the schedule's angular velocity integrated from the task's start, as in J. The feedback
adds a proportional term on the rotation from the measured orientation $R$ to $R_c$:</p>
$$\omega = \omega_{\mathrm{ref}} + \mathrm{sat}_{1\,\mathrm{rad/s}}\big(k\,\log(R_c R^{\top})^{\vee}\big)$$
<p>Eq. (1) and Eq. (2) of HOM are unchanged; only the object twist they are given changes
(<code>uhas_sr2.ATT_GAIN</code>, <code>inner_loop_mpc.py run --method hom --att-gain k</code>).</p>"""
    t += table(["Gain k (1/s)", "Mean J", "Held (%)", "Above HOM (%)", "Tracking error (deg)", "Three-pad share",
                "Grip (N)", "Saturation (%)", "Target reversals (1/s)"], rows,
               "HOM with feedback on the measured orientation, on the same 40 runs.")
    fr = []
    for f in (1.5, 2.0, 2.5, 3.5):
        s_ = hs if f == 2.5 else S.get(f"hom_twist_ft{f:g}")
        if s_:
            fr.append([f"{f:g}" + (" (HOM)" if f == 2.5 else ""), f2(s_["J"], 3), f"{100 * s_['held']:.0f}",
                       f2(s_["err"], 1), f2(s_["contact"]), f2(s_["grip_mean_N"], 1), f"{100 * s_['sat_frac']:.0f}",
                       f2(s_["reversals_ps"])])
    if len(fr) > 1:
        t += r"""<p>HOM's other free quantity in the twist is the pad force it commands per finger, 2.5 N
(<code>hom_tasks.PRM</code>, <code>f_tripod</code>). On the SCS0009 plant HOM's servos are at their torque limit in
16 % of the joint-periods, so a lower force may leave room for the twist.</p>"""
        t += table(["Commanded force per finger (N)", "Mean J", "Held (%)", "Tracking error (deg)", "Three-pad share",
                    "Grip (N)", "Saturation (%)", "Target reversals (1/s)"], fr,
                   "HOM with another commanded pad force, on the same 40 runs.")
    return t


def sec_contact(S, R, T, rt, figs):
    """The plant's fingertip contact and the planner's model of it, varied separately."""
    if not S.get(hom_stem("pt")) and not S.get(stem_of("ps", CFG_PSRT, model="pt")):
        return sec_pt(S)
    t = r"""<p>The plant's fingertips and the planner's model of them were varied separately. Table <<tm>> lists the
contact models of the v4b tip: the sphere pads, point contact, and point contact with torsion at three coefficients.
The sphere pads are the benchmark's plant. Point contact is the tip's convex mesh, which touches the sphere at one point
and resists no torque about the contact normal. With MuJoCo's condim 4 a point contact also resists a torque about its
normal, up to $\mu_t$ times its normal force. On the two-pad rig of 1 October the pads' torque about the pinch axis
was $\mu N r$, with an arm $r = 1.00\,\mathrm{mm}\,(N/1\,\mathrm{N})^{0.25}$ at 10 MPa that scales as $E^{-1/4}$. For
2.5 MPa pads at friction 1 that is a $\mu_t$ of about 1.4 mm at 1 N and 1.8 mm at 2.5 N. That rig pinched a 25 mm
cylinder between two spherical tips, so the value transfers to the v4b tip and the 40 mm sphere only roughly. The
coefficients 0.5, 1 and 2 mm bracket it.</p>
<p>HOM computes its commands from the contact geometry and rolls out no model, so only the plant changes for it. The
planners rolled out the pad model or the point-contact model, and on the torsion plants also the plant's own model.
Predictive sampling ran at the real-time budget on every plant. DIAL annealing ran on every plant at the real-time
configuration with the highest mean J on the pads, from the section on annealed sampling at the real-time budget. DIAL
annealing at the benchmark's budget of 4 &#215; 64 rollouts ran on the pads and the point-contact plants.</p>"""
    t = fill(t, tm=P.TAB[0] + 1)
    t += table(["Contact model", "Contact with the sphere", "Torque about the normal", "Used as"],
               [["Sphere pads", "1,018 spheres of 0.75 mm radius per tip, each a soft contact, 2.5 MPa",
                 "from the friction over the patch", "plant; planner's model"],
                ["Point contact", "the tip's convex mesh, one contact", "none", "plant; planner's model"],
                ["Point contact with torsion", "the same, condim 4, $\\mu_t$ = 0.5, 1 or 2 mm",
                 "up to $\\mu_t$ times the normal force", "plant; the planner's model on that plant"]],
               "Contact models of the v4b tip (<code>inner_loop_mpc.py</code> <code>--plant-model</code> and "
               "<code>--model</code>: <code>pads</code>, <code>pt</code>, <code>pt_t&lt;mm&gt;</code>). Every model "
               "has friction 1.0 against the sphere.")
    cols = [("hom", None, "pads")]
    for meth, cfg in (("ps", CFG_PSRT), ("anneal", rt), ("anneal", CFG_FULL)):
        if cfg is not None:
            cols += [(meth, cfg, "pads"), (meth, cfg, "pt"), (meth, cfg, "own")]
    rows = []
    for p, lab in PLANTS:
        row = [lab]
        for meth, cfg, model in cols:
            if meth == "hom":
                s = S.get(hom_stem(p))
            elif model == "own" and p in ("pads", "pt"):
                s = None
            else:
                s = S.get(stem_of(meth, cfg, plant=p, model=p if model == "own" else model))
            row.append(f2(s["J"], 3) if s else "&#8212;")
        rows.append(row)
    names = {"ps": "PS real time", "anneal": "DIAL"}
    hdr = ["Plant"] + ["HOM"] + [f"{names[m]}{' 4 &#215; 64' if c == CFG_FULL else (' real time' if m == 'anneal' else '')}"
                                f", {'pad' if mo == 'pads' else 'point' if mo == 'pt' else 'own'} model"
                                for m, c, mo in cols[1:]]
    keep = [i for i in range(len(hdr)) if i == 0 or any(r_[i] != "&#8212;" for r_ in rows)]
    tj = P.TAB[0] + 1
    t += table([hdr[i] for i in keep], [[r_[i] for i in keep] for r_ in rows],
               "Mean J over the 40 runs on each plant. A planner's own model is the plant's: on the pads plant the pad "
               "model, on the point-contact plant the point-contact model.", num_from=1)
    if figs.get("contact"):
        t += P.img(figs["contact"], "Left: mean J over the 40 runs against the plant's fingertip contact; solid lines "
                   "plan with the pad model, dotted with the plant's own model, HOM in grey. Right: J with the pad model "
                   "minus J with the point-contact model on the same worlds, the mean over the 20 pairs with its 95 % "
                   "bootstrap interval; above zero the pad model planned better.")
    # paired differences: the pad model against the point-contact model on each plant, and each plant against the pads
    rows_p, plan = [], []
    others = [(p, lab_p) for p, lab_p in PLANTS if p != "pads"]
    for meth, cfg in (("ps", CFG_PSRT), ("anneal", rt), ("anneal", CFG_FULL)):
        if cfg is None:
            continue
        lab = f"{names[meth]}{' 4 &#215; 64' if cfg == CFG_FULL else (' real time' if meth == 'anneal' else '')}"
        plan.append((meth, cfg, lab))
        for model in ("pads", "pt", "own"):
            cells = []
            for p, _l in others:
                m_p = p if model == "own" else model
                m_0 = "pads" if model == "own" else model
                d = paired(R.get(stem_of(meth, cfg, plant=p, model=m_p), []), R.get(stem_of(meth, cfg, model=m_0), []))
                cells.append(fd(d))
            if any(c != "&#8212;" for c in cells):
                ml = {"pads": "pad model", "pt": "point-contact model", "own": "the plant's own model"}[model]
                rows_p.append([f"{lab}, {ml}"] + cells)
    rows_p.insert(0, ["HOM"] + [fd(paired(R.get(hom_stem(p), []), R.get("hom_twist", []))) for p, _l in others])
    rows_m = [[lab_p] + [fd(paired(R.get(stem_of(m, c, plant=p), []), R.get(stem_of(m, c, plant=p, model="pt"), [])))
                         for m, c, _l in plan] for p, lab_p in PLANTS]
    rows_m = [r_ for r_ in rows_m if any(x != "&#8212;" for x in r_[1:])]
    tm_ = P.TAB[0] + 1
    if rows_m:
        t += table(["Plant"] + [lab for _m, _c, lab in plan], rows_m,
                   "J with the pad model minus J with the point-contact model, on each plant: the mean over the 20 "
                   "pairs (seeds averaged within a pair), its 95 % bootstrap interval, and the pairs where the pad "
                   "model was higher.")
    tp = P.TAB[0] + 1
    if rows_p:
        t += table(["Controller"] + [f"{lab_p}" for _p, lab_p in others], rows_p,
                   "The plant's effect on each controller: J on the plant of the column minus J on the pads plant, with "
                   "the same statistics. With the plant's own model the planner on the pads plant used the pad model.")
    t += sec_contact_text(S, R, rt, tj, tm_, tp)
    return t


def sec_contact_text(S, R, rt, tj, tm_, tp):
    """Sentences on the contact runs from the numbers that exist."""
    out, p1 = [], []
    order = ["pads", "pt_t2", "pt_t1", "pt_t0.5", "pt"]
    h = [S.get(hom_stem(p)) for p in order]
    if all(h):
        dh = paired(R.get(hom_stem("pt"), []), R.get("hom_twist", []))
        p1.append(fill("HOM's mean J fell at each step from the pads to point contact (Table <<tj>>). It was <<a>> on "
                       "the pads, <<b>>, <<c>> and <<d>> at $\\mu_t$ of 2, 1 and 0.5 mm, and <<e>> on point contact. On "
                       "point contact it was <<dd>> below its J on the pads (Table <<tp>>).",
                       **dict(zip("abcde", (f"{s['J']:.3f}" for s in h))), tj=tj, tp=tp,
                       dd=f"{-dh['d']:.3f} ({-dh['hi']:.3f} to {-dh['lo']:.3f})" if dh else "&#8212;"))
    o = [S.get(stem_of("anneal", rt, plant=p, model=p)) for p in order] if rt is not None else [None]
    if all(o):
        d = {p: paired(R.get(stem_of("anneal", rt, plant=p, model=p), []), R.get(stem_of("anneal", rt), []))
             for p in order[1:]}
        p1.append(fill("With the plant's own model the real-time DIAL reached <<j0>> on the pads, <<j1>>, <<j2>> and "
                       "<<j3>> at 2, 1 and 0.5 mm, and <<j4>> on point contact. Down to 1 mm it was within <<k>> of its "
                       "J on the pads. It lost <<a>> at 0.5 mm and <<b>> on point contact.",
                       **{f"j{i}": f"{x['J']:.3f}" for i, x in enumerate(o)},
                       k=f"{max(abs(d[p]['d']) for p in ('pt_t2', 'pt_t1')):.3f}",
                       a=f"{-d['pt_t0.5']['d']:.3f}", b=f"{-d['pt']['d']:.3f}"))
    fo, fp = S.get(stem_of("anneal", CFG_FULL, plant="pt", model="pt")), S.get("anneal_twist")
    if fo and fp:
        dfo = paired(R.get(stem_of("anneal", CFG_FULL, plant="pt", model="pt"), []), R.get("anneal_twist", []))
        p1.append(fill("DIAL at 4 &#215; 64 with the point-contact model reached <<a>> on point contact, against <<b>> "
                       "with the pad model on the pads, a paired difference of <<d>>.", a=f"{fo['J']:.3f}",
                       b=f"{fp['J']:.3f}", d=fds(dfo)))
    if p1:
        out.append("<p>" + " ".join(p1) + "</p>")
    sent = []
    for meth, cfg, lab in (("anneal", CFG_FULL, "DIAL annealing at 4 &#215; 64"), ("anneal", rt, "the real-time DIAL"),
                           ("ps", CFG_PSRT, "real-time predictive sampling")):
        if cfg is None:
            continue
        a = paired(R.get(stem_of(meth, cfg), []), R.get(stem_of(meth, cfg, model="pt"), []))
        rest = [paired(R.get(stem_of(meth, cfg, plant=p), []), R.get(stem_of(meth, cfg, plant=p, model="pt"), []))
                for p in order[1:]]
        rest = [x for x in rest if x]
        if not (a and rest):
            continue
        if len(rest) == 1:
            sent.append(fill("For <<l>> the pad model changed J by <<a>> on the pads plant and by <<b>> on the "
                             "point-contact plant.", l=lab, a=fds(a), b=fds(rest[0])))
        else:
            sent.append(fill("For <<l>> the pad model changed J by <<a>> on the pads plant and by <<lo>> to <<hi>> on "
                             "the other four plants.", l=lab, a=fds(a), lo=f"{min(x['d'] for x in rest):+.3f}",
                             hi=f"{max(x['d'] for x in rest):+.3f}"))
    if sent:
        out.append("<p>" + " ".join(sent) + fill(" Table <<tm>> gives each plant's difference.</p>", tm=tm_))
    return "".join(out)


def sec_realtime(S, T, R, rt, figs):
    """DIAL annealing at budgets that run in real time for one hand."""
    full = one_hand_cfg(T, CFG_FULL)
    if not any(S.get(stem_of("anneal", c)) for c in RT_CANDS):
        return ""
    v16 = one_hand(T, 16, 4, 30)
    t = fill(r"""<p>DIAL annealing at the benchmark's budget needed <<f>> s of GPU for one hand per 0.1 s of control.
A planner runs in real time when one hand's replanning takes no longer than the replanning interval. A period of
rollouts, ten physics steps of 1 ms, took <<p16>> ms for one hand at 16 samples and <<p64>> ms at 64. At a 0.1 s
interval the iterations times the horizon must therefore stay near ten periods. Two changes double that budget. In
one, the planner's model steps its physics every 2 ms, five steps per period, while the plant keeps 1 ms. In the
other, the plan is replaced every 0.2 s and its horizon is at least 0.2 s. Table <<tn>> lists the configurations that
were run.</p>""",
             f=f2(full, 2), p16=f2(1e3 * v16 / 120 if v16 else None, 1), p64=f2(1e3 * full / 120 if full else None, 1),
             tn=P.TAB[0] + 1)
    rows = []
    for num, (meth, cfg, s, v) in enumerate(rt_rows(S, T), 1):
        it, h, k, r, dt, n = cfg
        g = v / (r * PERIOD_S)
        rows.append([f"{num}. " + ("PS" if meth == "ps" else "DIAL"), f"{it} &#215; {n}", f"{h * PERIOD_S:g}", str(k),
                     f"{r * PERIOD_S:g}", f"{1e3 * (dt or 0.001):g}", f2(v, 3), f2(g, 2), f2(s["J"], 3),
                     f"{100 * s['held']:.0f}", f2(s["reversals_ps"])])
    t += table(["Planner", "Iterations &#215; samples", "Horizon (s)", "Knots", "Replanning interval (s)",
                "Model's physics step (ms)", "Replanning, one hand (s)", "GPU time per second of control", "Mean J",
                "Held (%)", "Target reversals (1/s)"], rows,
               "Predictive sampling (PS) and DIAL annealing at several budgets, 40 runs each on the pads plant with the "
               "pad model. One hand's replanning time is from <code>inner_loop_mpc.py timing --chunk 1</code> on the RTX "
               "4070 Ti SUPER; a GPU time per second of control of 1 or less runs in real time.", num_from=1)
    if figs.get("realtime"):
        t += P.img(figs["realtime"], "Mean J against one hand's GPU time per second of control (log scale; bars: one "
                   "standard error). The numbers are the rows of the table above; left of the red line a configuration "
                   "runs in real time.")
    ok = [(S[stem_of("anneal", c)]["J"], c) for c in RT_CANDS if S.get(stem_of("anneal", c))
          and (one_hand_cfg(T, c) or 1e9) <= c[3] * PERIOD_S]
    sf, sp = S.get("anneal_twist"), S.get(stem_of("ps", CFG_PSRT))
    if ok and sf:
        bj, bc = max(ok)
        t += fill("<p>The best real-time DIAL configuration, <<c>>, reached <<j>>. DIAL at the benchmark's budget "
                  "reached <<f>> and real-time predictive sampling <<p>>.</p>", c=cfg_text(bc), j=f"{bj:.3f}",
                  f=f"{sf['J']:.3f}", p=f"{sp['J']:.3f}" if sp else "&#8212;")
    if rt is not None:
        lat, said = [], []
        for meth, cfg, lab, slab in (("ps", CFG_PSRT, "PS, real time", "Real-time predictive sampling"),
                                     ("anneal", rt, "DIAL, real time", "The real-time DIAL"),
                                     ("anneal", CFG_FULL, "DIAL, 4 &#215; 64", "DIAL at 4 &#215; 64")):
            cells = [lab]
            ss = [S.get(stem_of(meth, cfg, delay=dl)) for dl in ("", "_d4n", "_d14n")]
            for s in ss:
                cells.append(f"{s['J']:.3f} ({100 * s['held']:.0f})" if s else "&#8212;")
            lat.append(cells)
            if ss[0] and ss[1] and ss[2]:
                said.append(fill("<<l>> reached <<a>> from the exact state, <<b>> from a 40 ms old state and <<c>> "
                                 "from a 140 ms old state. From the latter the sphere fell in <<f>> of the 40 runs.",
                                 l=slab,
                                 a=f"{ss[0]['J']:.3f}", b=f"{ss[1]['J']:.3f}", c=f"{ss[2]['J']:.3f}",
                                 f=round(40 * (1 - ss[2]["held"]))))
            elif ss[0] and ss[1]:
                said.append(fill("<<l>> reached <<a>> from the exact state and <<b>> from a 40 ms old state.", l=slab,
                                 a=f"{ss[0]['J']:.3f}",
                                 b=f"{ss[1]['J']:.3f}"))
        if any(c != "&#8212;" for r_ in lat for c in r_[2:]):
            t += r"""<p>On the bench the planner starts from the tag tracker's pose, 40 ms old with 0.5 mm and 1 deg of
noise. Its plan also reaches the servos one replanning time after that pose. The runs below plan from the plant's
state 4 or 14 periods earlier, with the tracker's noise on the sphere's pose. The plan is applied when it is computed,
without a forward prediction of the state.</p>"""
            t += table(["Planner", "Exact state", "40 ms old, tracker noise", "140 ms old, tracker noise"], lat,
                       "Mean J of the planners from delayed, noisy states, with the share of held runs in % "
                       "(<code>--plan-delay</code>, <code>--plan-noise</code>); 140 ms is the tracker's 40 ms and one "
                       "replanning interval.", num_from=1)
            if said:
                t += "<p>" + " ".join(said) + "</p>"
    return t


def sec_reversals(S, R, figs):
    """Why the planners' targets reverse, and the sampling and cost options tested against it."""
    if not any(S.get(stem_of(m, c, sh=sh)) for m, c, sh, _l, _k in REV_RUNS if sh):
        return ""
    hs, sd, si = S.get("hom_twist"), S.get("anneal_twist"), S.get("icem_twist")
    t = fill(r"""<p>HOM's targets reversed <<h>> times per joint per second, DIAL annealing's <<d>> and iCEM's <<i>>.
A planner's targets are linear between two replannings, so they can reverse only at a replanning, at most ten times
per second. Across the planners the reversals rose with the tracking accuracy (Figure <<f>>).</p>
<p>Two parts of the planners were tested as causes. The first is the temperature of the MPPI step of MPPI and DIAL
annealing, which weights the samples by $\exp(-(c - c_{\min})/\lambda)$ with $\lambda = 0.03\,\mathrm{std}(c)$. For
normally distributed costs that gives an effective sample size of 1.1, at 16 and at 64 samples, and each iteration
moves the plan by about 0.96 of one sample's noise. At $\lambda = 1.0\,\mathrm{std}(c)$ the effective sample size is
7.8 at 16 samples and 27 at 64. A larger effective sample size was expected to average out the noise in finger motions
that change the cost little.</p>
<p>The second is the warm start. Before a replanning the plan is shifted by the replanning interval, and past its last
knot it holds the last knot. When the horizon equals the interval, as in every real-time configuration with two knots,
the shifted plan is a constant target, so each replanning starts from a zero rate. The option <code>--warm extrap</code>
continues the last segment's slope past the horizon.</p>
<p>The options in Table <<tn>> were run on DIAL annealing at 3 &#215; 16 samples with a 2 ms model step, which
needs 0.101 s of GPU per 0.1 s for one hand. Two were also run on the benchmark's DIAL and one on real-time predictive
sampling. The rate-change cost and the dead band were then run on the real-time pick of 2 &#215; 32 samples, which
needs 0.082 s. The cost options add, per joint $j$, with $\dot u_{j,0}$ the executed target's rate at the replanning,
$\dot u_{j,k}$ the plan's rate in its $k$-th knot gap and $\dot u_s = 20$ deg/s:</p>
$$c_{\mathrm{acc}} = w_a \sum_j \sum_{k \ge 1} \Big(\frac{\dot u_{j,k} - \dot u_{j,k-1}}{\dot u_s}\Big)^2, \qquad
c_{\mathrm{rev}} = w_r \sum_j \frac{\max(0, -\dot u_{j,1}\,\dot u_{j,0})}{\dot u_s^2}$$
<p>The dead band replaces the rotation error $e_t$ in the cost by $\max(0, e_t - 3^\circ)$. J still scores the full
error.</p>""", h=f2(hs["reversals_ps"]) if hs else "&#8212;", d=f2(sd["reversals_ps"]) if sd else "&#8212;",
             i=f2(si["reversals_ps"]) if si else "&#8212;", f=P.FIG[0] + 1, tn=P.TAB[0] + 1)
    if figs.get("reversals"):
        t = t.replace("</p>", "</p>" + P.img(figs["reversals"], "Mean J against target reversals per joint per second. "
                                             "Grey: the benchmark's controllers. Coloured: the options on DIAL "
                                             "annealing at 3 x 16 samples (filled) and on the real-time pick of 2 x 32 "
                                             "samples (open). They are numbered as in the table; each arrow starts at "
                                             "the same configuration without options."), 1)
    rows, k = [], 0
    base_of = {CFG_REV: stem_of("anneal", CFG_REV), CFG_PICK: stem_of("anneal", CFG_PICK),
               CFG_PSRT: stem_of("ps", CFG_PSRT), CFG_FULL: "anneal_twist"}
    for meth, cfg, sh, lab, kind in REV_RUNS:
        s = S.get(stem_of(meth, cfg, sh=sh))
        if not s:
            continue
        num = ""
        if cfg in (CFG_REV, CFG_PICK):
            k += 1
            num = f"{k}. "
        d = paired(R.get(stem_of(meth, cfg, sh=sh), []), R.get(base_of[cfg], [])) if sh else None
        rows.append([num + lab, kind, f2(s["J"], 3), fd(d) if d else "&#8212;", f"{100 * s['held']:.0f}",
                     f2(s["err"], 1), f2(s["reversals_ps"]), f2(s["cmd_rate_dps"], 1), f2(s["grip_mean_N"], 1)])
    t += table(["Option", "Changes the", "Mean J", "J minus J without the option", "Held (%)", "Tracking error (deg)",
                "Target reversals (1/s)", "Target rate (deg/s)", "Grip (N)"], rows,
               "The reversal options, 40 runs each on the SCS0009 plant with the pad model (<code>inner_loop_mpc.py "
               "run</code> <code>--lam-rel</code>, <code>--warm extrap</code>, <code>--sigma</code>, <code>--rate-w"
               "</code>, <code>--acc-w</code>, <code>--rev-w</code>, <code>--err-db</code>). Temperature: $\\lambda$ "
               "over the costs' standard deviation (0.03 without options). The paired difference is taken against the "
               "same planner without options.", num_from=2)
    t += rev_text(S, R)
    return t


def rev_text(S, R):
    """The reversal options' outcomes, from the rows."""
    def g(meth, cfg, sh=""):
        return S.get(stem_of(meth, cfg, sh=sh))

    def d(meth, cfg, sh, base):
        p = paired(R.get(stem_of(meth, cfg, sh=sh), []), R.get(base, []))
        return f"{p['d']:+.3f} ({p['lo']:+.3f} to {p['hi']:+.3f})" if p else "&#8212;"

    b, l3, l1, ex, sg = (g("anneal", CFG_REV, x) for x in ("", "_lam0.3", "_lam1", "_extrap", "_sig0.04"))
    a1, a4, rw, rv, db = (g("anneal", CFG_REV, x) for x in ("_acc1", "_acc4", "_rw4", "_rev10", "_db3"))
    pe, ps_ = g("ps", CFG_PSRT, "_extrap"), g("ps", CFG_PSRT)
    fl, fa, ff = S.get("anneal_twist_lam1"), S.get("anneal_twist_acc1"), S.get("anneal_twist")
    if not all((b, l3, l1, ex, sg, a1, a4, rw, rv, db)):
        return ""
    base = stem_of("anneal", CFG_REV)
    out = [fill(r"""<p>No sampling option lowered the reversals of the real-time DIAL. At temperatures 0.3 and 1.0 its
targets reversed <<l3>> and <<l1>> times per joint per second, against <<b>> at 0.03. Steps of independent random sign
would reverse at half the replannings, 5 times per second. With half the sampling noise the targets reversed <<sg>>
times. The continued slope raised the target rate from <<rb>> to <<re>> deg/s, and the sphere fell in <<fe>> of the 40
runs against <<fb>>. Its J changed by <<de>>.""", l3=f2(l3["reversals_ps"]), l1=f2(l1["reversals_ps"]),
                b=f2(b["reversals_ps"]), sg=f2(sg["reversals_ps"]), rb=f2(b["cmd_rate_dps"], 1),
                re=f2(ex["cmd_rate_dps"], 1), fe=round(40 * (1 - ex["held"])), fb=round(40 * (1 - b["held"])),
                de=d("anneal", CFG_REV, "_extrap", base))]
    if pe and ps_:
        out.append(fill(" On real-time predictive sampling the continued slope lowered J from <<a>> to <<b>>, and the "
                        "sphere fell in <<f>> of 40 runs.", a=f"{ps_['J']:.3f}", b=f"{pe['J']:.3f}",
                        f=round(40 * (1 - pe["held"]))))
    if fl and ff:
        out.append(fill(" The benchmark's DIAL plans over 0.3 s, so its warm start keeps the previous plan's rates. On "
                        "it, temperature 1.0 lowered the reversals from <<a>> to <<b>> and changed J by <<c>>.",
                        a=f2(ff["reversals_ps"]), b=f2(fl["reversals_ps"]), c=d("anneal", CFG_FULL, "_lam1",
                                                                                 "anneal_twist")))
    out.append("</p>")
    out.append(fill(r"""<p>Every cost option lowered the reversals. The rate-change cost at weight 1 lowered them from
<<b>> to <<a1>>, and J changed by <<da1>>. The dead band of 3 deg lowered them to <<db>>, and J changed by <<ddb>>. The
knot-rate weight 4, the rate-change weight 4 and the reversal cost lowered them to <<lo>> to <<hi>>, below HOM's, and
lowered J by <<jlo>> to <<jhi>>. With the knot-rate weight 4 the tracking error rose from <<eb>> to <<erw>> deg.""",
                    b=f2(b["reversals_ps"]), a1=f2(a1["reversals_ps"]), da1=d("anneal", CFG_REV, "_acc1", base),
                    db=f2(db["reversals_ps"]), ddb=d("anneal", CFG_REV, "_db3", base),
                    lo=f2(min(x["reversals_ps"] for x in (rw, a4, rv))),
                    hi=f2(max(x["reversals_ps"] for x in (rw, a4, rv))),
                    jlo=f"{min(b['J'] - x['J'] for x in (rw, a4, rv)):.3f}",
                    jhi=f"{max(b['J'] - x['J'] for x in (rw, a4, rv)):.3f}", eb=f2(b["err"], 1), erw=f2(rw["err"], 1)))
    if fa and ff:
        out.append(fill(" On the benchmark's DIAL the rate-change cost at weight 1 lowered the reversals from <<a>> to "
                        "<<b>>, and J changed by <<c>>.", a=f2(ff["reversals_ps"]), b=f2(fa["reversals_ps"]),
                        c=d("anneal", CFG_FULL, "_acc1", "anneal_twist")))
    out.append("</p>")
    pb, pa, pd, p2 = (g("anneal", CFG_PICK, x) for x in ("", "_acc1", "_acc1_db3", "_acc2_db3"))
    hs = S.get("hom_twist")
    if pb and pa and pd and p2 and hs:
        pbase = stem_of("anneal", CFG_PICK)
        out.append(fill(r"""<p>On the real-time pick of 2 &#215; 32 samples, the rate-change cost at weight 1 lowered
the reversals from <<b>> to <<a>>. HOM's were <<h>>. J changed by <<da>>. With the dead band added they fell to <<ad>>, and J
changed by <<dd>>. At weight 2 with the dead band they fell to <<a2>>, and J changed by <<d2>>. With the rate-change cost
at weight 1 the pick reached J <<j>>. HOM reached <<jh>> on the same runs, a paired difference of <<dh>>.</p>""",
                        b=f2(pb["reversals_ps"]), a=f2(pa["reversals_ps"]), h=f2(hs["reversals_ps"]),
                        da=d("anneal", CFG_PICK, "_acc1", pbase), ad=f2(pd["reversals_ps"]),
                        dd=d("anneal", CFG_PICK, "_acc1_db3", pbase), a2=f2(p2["reversals_ps"]),
                        d2=d("anneal", CFG_PICK, "_acc2_db3", pbase), j=f"{pa['J']:.3f}", jh=f"{hs['J']:.3f}",
                        dh=d("anneal", CFG_PICK, "_acc1", "hom_twist")))
    return "".join(out)


def sec_pt(S):
    s, sp = S.get("anneal_twist_pt"), S.get("anneal_twist")
    if not s:
        return ""
    rows = [["DIAL annealing, pads model", f2(sp["J"], 3) if sp else "&#8212;", f"{100 * sp['held']:.0f}" if sp else
             "&#8212;", f2(sp["grip_mean_N"], 1) if sp else "&#8212;", f2(sp["reversals_ps"]) if sp else "&#8212;",
             f2(sp["replan_s"], 2) if sp else "&#8212;"],
            ["DIAL annealing, point-contact model", f2(s["J"], 3), f"{100 * s['held']:.0f}", f2(s["grip_mean_N"], 1),
             f2(s["reversals_ps"]), f2(s["replan_s"], 2)]]
    t = r"""<p>The point-contact model is the same hand with each v4b tip as one convex mesh, which touches the sphere at
one point (the scene cache's <code>pt</code> model). The plant's tip is 1,018 sphere pads. The planner rolls out
the point-contact model, and the plant keeps the pads.</p>"""
    t += table(["Planner", "Mean J", "Held (%)", "Grip (N)", "Target reversals (1/s)", "Replanning, 16 hands (s)"],
               rows, "DIAL annealing with its own model of the fingertips, on the same 40 runs.")
    return t


def sec_hs(S):
    rows = []
    for stem, lab in (("hom_twist", "HOM"), ("anneal_twist", "DIAL annealing, servo targets (18 numbers per plan)"),
                      ("hs_anneal_twist", "DIAL annealing in HOM's space (24 numbers per plan), noise 0.3"),
                      ("hs_anneal_twist_hs0.1", "DIAL annealing in HOM's space, noise 0.1"),
                      ("hs_ps_twist", "Predictive sampling in HOM's space"), ("hs_icem_twist", "iCEM in HOM's space")):
        s_ = S.get(stem)
        if s_:
            rows.append([lab, f2(s_["J"], 3), f"{100 * s_['held']:.0f}", "&#8212;" if stem == "hom_twist" else
                         f"{100 * s_['above']:.0f}", f2(s_["err"], 1), f2(s_["grip_mean_N"], 1),
                         f2(s_["reversals_ps"]), f2(s_["cmd_rate_dps"], 1),
                         "&#8212;" if s_["replan_s"] is None else f2(s_["replan_s"], 2),
                         "&#8212;" if not s_.get("rollouts") else f"{s_['rollouts']:.0f}"])
    if not any(S.get(k) for k in ("hs_anneal_twist", "hs_anneal_twist_hs0.1", "hs_ps_twist", "hs_icem_twist")):
        return ""
    t = r"""<p>A HOM-space plan sets the two rolling and the two sliding rates of each contact relative to the
sphere, 12 numbers per knot. The separation and the spin about the normal are left to HOM. Each period HOM solves its
Eq. (2) as usual and then solves it a second time. In the second solve every row's reference is taken from the first
solution, and the plan's rates are added to the rolling and sliding rows. Those rows get weight 1, and the separation
row 10 times its weight. A finger has three joints, so without the heavier separation row the added rates pulled the
pads off the sphere. With
the plan at zero the controller is HOM; on HOM's best pair a run with noise 10<sup>-9</sup> gave HOM's J, 0.772.
Each sample's rollout runs HOM in its own world, from a copy of the plant's physics state and HOM's state
(<code>inner_loop_mpc.HomRollouts</code>). One unit of noise is 1 rad/s of rolling or 20 mm/s of sliding.</p>"""
    return t + table(["Controller", "Mean J", "Held (%)", "Above HOM (%)", "Tracking error (deg)", "Grip (N)",
                      "Target reversals (1/s)", "Target rate (deg/s)", "Replanning, 16 hands (s)", "Rollouts per "
                      "replanning"], rows, "Sampling over HOM's relative twists against sampling over the servo targets, "
                     "on the same 40 runs.")


def sec_sensing(S):
    rows = []
    hs = S.get("hom_twist")
    if hs:
        rows.append(["Exact, 100 Hz", f2(hs["J"], 3), f"{100 * hs['held']:.0f}", f2(hs["err"], 1),
                     f2(hs["contact"]), f2(hs["grip_mean_N"], 1), f2(hs["reversals_ps"])])
    for v, desc in VARIANTS:
        s_ = S.get(f"hom_twist_{v}")
        if s_:
            rows.append([desc[0].upper() + desc[1:], f2(s_["J"], 3), f"{100 * s_['held']:.0f}", f2(s_["err"], 1),
                         f2(s_["contact"]), f2(s_["grip_mean_N"], 1), f2(s_["reversals_ps"])])
    pl = []
    for stem, lab in (("ps_twist_r10h10k2i1", "Predictive sampling, real-time budget"), ("anneal_twist", "DIAL annealing")):
        ex, nz = S.get(stem), S.get(stem.replace("_twist", "_twist_d4n") if stem == "anneal_twist" else
                                    "ps_twist_d4n_r10h10k2i1")
        if ex and nz:
            pl.append([lab, f2(ex["J"], 3), f2(nz["J"], 3), f"{100 * nz['held']:.0f}", f2(nz["err"], 1),
                       f2(nz["contact"])])
    if len(rows) < 2 and not pl:
        return ""
    t = fill(r"""<p>HOM reads the joint angles, the sphere's pose and each fingertip's contact point and force. Every
row above reads them exactly. Here they come through the sensing models of the <a href="<<r>>">HOM control review</a>
of 29 October, at the bench's 50 Hz. The tag tracker's pose has 0.5 mm and 1 deg of noise and arrives 40 ms late. The
magnetic fingertips' model adds their measured errors of contact point and force. The planners and UHAS still read the
plant's exact state.</p>""", r=REVIEW)
    if len(rows) > 1:
        t += table(["Measurement", "Mean J", "Held (%)", "Tracking error (deg)", "Three-pad share", "Grip (N)",
                    "Target reversals (1/s)"], rows,
                   "HOM under the sensing variants of <code>hom_tasks.VARIANTS</code>, on the same 40 runs. HOM does "
                   "not filter the measured pose, so the tracker's noise reaches its servo targets.")
    if pl:
        t += r"""<p>A planner copies the plant's state into its rollouts. With the tag tracker the copy is 40 ms old, and
the sphere's pose in it has the tracker's noise. Here each plan starts from the plant's state 4 periods earlier, with
0.5 mm and 1 deg of noise on the sphere's pose. The plan is applied from now (<code>--plan-delay 4 --plan-noise</code>).</p>"""
        t += table(["Planner", "Mean J, exact state", "Mean J, measured state", "Held (%)", "Tracking error (deg)",
                    "Three-pad share"], pl, "Planners from a 40 ms old, noisy state, on the same 40 runs.")
    return t


PF_RUNS = (("hom_twist_notip", "Tag tracker 50 Hz, frames used as they come"),
           ("hom_twist_notip_pf0.05", "Tag tracker 50 Hz, filter 0.05 s"),
           ("hom_twist_notip_pf0.1", "Tag tracker 50 Hz, filter 0.1 s"),
           ("hom_twist_notip_pf0.2", "Tag tracker 50 Hz, filter 0.2 s"),
           ("hom_twist_notip_pf0.5", "Tag tracker 50 Hz, filter 0.5 s"),
           ("hom_twist_notip_pfinf", "Tag tracker until the task, then open loop"),
           ("hom_twist_notip_holde3", "Tag tracker 17 Hz, each frame held"),
           ("hom_twist_notip_pf0.2e3", "Tag tracker 17 Hz, filter 0.2 s"),
           ("hom_twist_tip", "Tag tracker and fingertip model, frames as they come"),
           ("hom_twist_tip_pf0.1", "Tag tracker and fingertip model, filter 0.1 s"),
           ("hom_twist_tip_pf0.2", "Tag tracker and fingertip model, filter 0.2 s"),
           ("hom_twist_rate50", "Exact pose, 50 Hz"),
           ("hom_twist_rate50_pf0.2", "Exact pose, 50 Hz, filter 0.2 s"))


def sec_filter(S):
    rows = []
    for stem, lab in PF_RUNS:
        s_ = S.get(stem)
        if s_:
            rows.append([lab, f2(s_["J"], 3), f"{100 * s_['held']:.0f}", f2(s_["err"], 1), f2(s_["contact"]),
                         f2(s_["reversals_ps"]), f2(s_["cmd_rate_dps"], 1)])
    if not any(S.get(stem) for stem, _l in PF_RUNS if "_pf" in stem or "hold" in stem):
        return ""
    t = r"""<p>HOM feeds the measured pose of the sphere into its contact geometry every period, so the tracker's
noise reaches the servo targets. A complementary filter keeps a belief of the pose instead. Each period it rotates the
estimate by the schedule's angular velocity. A tracker frame is compared with the estimate that was kept for the
frame's own time, 40 ms earlier. The current estimate then moves by $\Delta t_{\mathrm{frame}}/\tau$ of the difference,
in position and in rotation (<code>hom_gpu.POSE_FILTER</code>). With $\tau$ infinite the estimate is never
corrected after the task's start: open loop. A 17 Hz tracker sends a frame every third period.</p>"""
    return t + table(["Measurement of the sphere's pose", "Mean J", "Held (%)", "Tracking error (deg)",
                      "Three-pad share", "Target reversals (1/s)", "Target rate (deg/s)"], rows,
                     "HOM with the tag tracker's pose (0.5 mm, 1 deg, 40 ms late) through the complementary filter, on "
                     "the same 40 runs at 50 Hz.")


# UHAS retrained on the SCS0009 plant (lane steps sv_uhas_*, logs/20261112-mpc/uhas_train_eval.sh): (rows tag, run, label)
UHAS_SV = [("sv_published_s0", "20261113-2115_twist_s0", "UHAS, published reward"),
           ("sv_limits_tau0.02_cap0_s0", "20261113-2347_twist_limits_tau0.02_cap0_s0", "UHAS with the limit terms")]


def usability(run):
    """HOM's and the policy's rows on the test split's worlds (inner_loop_usability.py), by controller."""
    p = ROWS / f"usability_twist_{run}.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return {c: [r for r in rows if r["controller"] == c] for c in ("hom", "uhas")}


def sec_uhas(S, R):
    """UHAS trained on the SCS0009 plant: the test split's worlds beside HOM, and the 20 pairs."""
    U = [(tag, run, lab, usability(run)) for tag, run, lab in UHAS_SV]
    U = [u for u in U if u[3]["uhas"]]
    if not U:
        return ""
    rows = []
    hom_u = U[0][3]["hom"]
    for lab, rs in [("HOM", hom_u)] + [(lab, u["uhas"]) for _t, _r, lab, u in U]:
        s = stats(rs)
        rows.append([lab, str(s["n"]), f2(s["J"], 3), f"{100 * s['held']:.0f}", f2(s["err"], 1),
                     f2(s["link_mean_N"]), f2(s["pad_share"]), f"{100 * s['sat_frac']:.0f}", f2(s["reversals_ps"], 1),
                     f"{s['cmd_rate_dps']:,.0f}"])
    t = fill(r"""<p>UHAS was retrained on the SCS0009 plant for 120 min with 2,048 worlds of the training split at seed 0
(<code>uhas_sr2.py train --task twist</code>). One run used the published reward. The other added the limit terms of
<code>uhas_sr2.py</code>. A hinge penalizes each servo's torque demand above 0.8 of its limit, here at weight 0.02 and
without a cap. A second term penalizes a joint within 3 deg of its range limit, and a third, $0.04\,|a -
a_{\mathrm{prev}}|^2$, the change of the action. HOM and each policy then drove the same <<n>> worlds of the test split at seed 0
(<code>inner_loop_usability.py</code>, Table <<tn>>).</p>""", n=f"{len(hom_u):,}", tn=P.TAB[0] + 1)
    t += table(["Controller", "Worlds", "Mean J", "Held (%)", "Tracking error (deg)", "Rigid-part force (N)",
                "Pad share", "Saturation (%)", "Target reversals (1/s)", "Target rate (deg/s)"], rows,
               "HOM and UHAS on the test split's worlds of the SCS0009 plant, seed 0 "
               "(<code>rows/usability_twist_&lt;run&gt;.jsonl</code>). HOM's row is from the first policy's file; the "
               "second file gives the same within 0.001 in J.", num_from=1)
    sh, su = stats(hom_u), stats(U[0][3]["uhas"])
    t += fill(r"""<p>On the test split UHAS with the published reward reached a mean J of <<ju>>, HOM <<jh>>. Its tracking
error was <<eu>> deg against HOM's <<eh>> deg, and the force through the rigid parts <<lu>> N against <<lh>> N. Its
targets moved at <<ru>> deg/s RMS, about <<step>> deg per 10 ms period, and reversed <<vu>> times per joint per second.
The servos were at their torque limit in <<su>> % of the joint-periods.""", ju=f"{su['J']:.3f}", jh=f"{sh['J']:.3f}",
              eu=f2(su["err"], 1), eh=f2(sh["err"], 1), lu=f2(su["link_mean_N"]), lh=f2(sh["link_mean_N"]),
              ru=f"{su['cmd_rate_dps']:,.0f}", step=f"{su['cmd_rate_dps'] * PERIOD_S:.0f}",
              vu=f2(su["reversals_ps"], 1), su=f"{100 * su['sat_frac']:.0f}")
    if len(U) > 1:
        sl = stats(U[1][3]["uhas"])
        t += fill(" With the limit terms the saturation fell to <<sat>> %, the reversals rose to <<v>> per joint per "
                  "second, and J fell to <<j>>.", sat=f"{100 * sl['sat_frac']:.0f}", v=f2(sl["reversals_ps"], 1),
                  j=f"{sl['J']:.3f}")
    t += "</p>"
    pr = []
    for tag, _run, lab, _u in U:
        s0, s4 = S.get(f"uhas_twist_{tag}"), S.get(f"uhas_twist_mu0.4_{tag}")
        d = paired(R.get(f"uhas_twist_{tag}", []), R.get("hom_twist", []))
        if s0:
            pr.append([lab, f2(s0["J"], 3), f"{100 * s0['held']:.0f}", fd(d), f2(s0["reversals_ps"], 1),
                       f2(s4["J"], 3) if s4 else "&#8212;"])
    if pr:
        hs, h4 = S.get("hom_twist"), S.get("hom_twist_mu0.4")
        pr.insert(0, ["HOM", f2(hs["J"], 3), f"{100 * hs['held']:.0f}", "&#8212;", f2(hs["reversals_ps"], 1),
                      f2(h4["J"], 3) if h4 else "&#8212;"])
        t += table(["Controller", "Mean J", "Held (%)", "J minus HOM's", "Target reversals (1/s)",
                    "Mean J at plant friction 0.4"], pr,
                   "The 20 pairs at seeds 0 and 1 on the SCS0009 plant. Ten of the pairs were chosen by the J of UHAS "
                   "trained on the work plant.", num_from=1)
    return t


def sec_friction(S, R):
    rows = []
    runs = (("hom_twist_mu0.4", "HOM"), ("ps_twist_mu0.4_r10h10k2i1", "Predictive sampling, real-time budget"),
            ("anneal_twist_mu0.4", "DIAL annealing"), ("spider_twist_mu0.4", "SPIDER"),
            ("spider_wc_twist_mu0.4", "SPIDER, worst case")) + tuple(
        (f"uhas_twist_mu0.4_{tag}", lab + ", trained at friction 1.0") for tag, _r, lab in UHAS_SV)
    for stem, lab in runs:
        s = S.get(stem)
        if s:
            d = paired(R.get(stem, []), R.get("hom_twist_mu0.4", [])) if stem != "hom_twist_mu0.4" else None
            rows.append([lab, f2(s["J"], 3), fd(d), f"{100 * s['held']:.0f}", f2(s["err"], 1),
                         f2(s["grip_mean_N"], 1)])
    if not rows:
        return ""
    t = (r"""<p>The plant's friction is set to 0.4, the value measured for TPU on bare PLA on the UR5. The planners
keep friction 1.0 in their model; the worst case of SPIDER samples 0.4, 0.7 and 1.0.</p>"""
         + table(["Controller", "Mean J", "J minus HOM's", "Held (%)", "Tracking error (deg)", "Grip (N)"], rows,
                 "Plant friction 0.4, the same 40 runs."))
    h4, w4 = S.get("hom_twist_mu0.4"), S.get("spider_wc_twist_mu0.4")
    if h4 and w4:
        dw = paired(R.get("spider_wc_twist_mu0.4", []), R.get("hom_twist_mu0.4", []))
        rest = [paired(R.get(stem, []), R.get("hom_twist_mu0.4", [])) for stem, _l in runs
                if stem not in ("hom_twist_mu0.4", "spider_wc_twist_mu0.4") and R.get(stem)]
        only = dw and dw["d"] > 0 and all(x and x["d"] <= 0 for x in rest)
        t += fill("<p>At friction 0.4 HOM kept the sphere in <<h>> % of the runs and reached <<j>>. The worst-case "
                  "SPIDER<<o>> reached <<w>>, a paired difference with HOM of <<d>>.</p>",
                  o=", the only controller above HOM," if only else "", h=f"{100 * h4['held']:.0f}",
                  j=f"{h4['J']:.3f}", w=f"{w4['J']:.3f}", d=fds(dw))
    return t


def sec_work(SW):
    rows = []
    for stem, lab in MAIN + [("uhas_twist", "UHAS, trained on the work plant")]:
        s = SW.get(stem)
        if s:
            rows.append([lab, f2(s["J"], 3), f"{100 * s['held']:.0f}", f2(s["grip_mean_N"], 1),
                         f"{100 * s['sat_frac']:.0f}", f2(s["reversals_ps"])])
    if not rows:
        return ""
    t = r"""<p>The first runs were made on the work plant: 4 N m/rad and a 1 N m limit per joint, 5.6 times the SCS0009's
torque, with 10 MPa pads. On it, UHAS gripped the sphere with a median 57 N and saturated its servos in 91 % of the
joint-periods. Neither is possible with 0.18 N m joints. The planners' grip of 16 to 23 N on that plant also exceeds what
the SCS0009 can press.</p>"""
    return t + table(["Controller", "Mean J", "Held (%)", "Grip (N)", "Saturation (%)", "Target reversals (1/s)"], rows,
                     "The same 40 runs on the work plant (rows <code>*_work10.jsonl</code>, 2026-11-12 21:08 to "
                     "2026-11-13 09:42).")


# ------------------------------------------------------------------------------------------------ four tasks, 11-14
TASKS4 = ("twist", "roll", "pinpush", "press")
TNAME = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
CFG_RT7 = (2, 20, 3, 20, 0.002, 48)
CFG_RT30 = (2, 30, 3, 20, 0.002, 16)      # the best real-time DIAL setting on the four tasks' tuning pairs (11-14)
CFG_FULL2 = (4, 30, 3, 10, 0.002, 64)
CFG_IL2 = (2, 30, 3, 10, 0.002, 64)       # iLQR (inner_loop_ilqr.py): 2 iterations over 0.3 s, steps of 50 ms
CFG_IL4 = (4, 30, 3, 10, 0.002, 64)       # the same at 4 iterations (the row tag of the default schedule: none)
# the configurations run on the tuning pairs (logs/20261114-mpc_tasks/tune_rt.sh, tune_rt2.sh): (method, cfg, label)
TUNE = [("hom", None, "HOM"),
        ("anneal", CFG_PICK, "DIAL 2 &#215; 32, 0.1 s ahead (the twist's real-time pick)"),
        ("anneal", (1, 20, 3, 10, 0.002, 32), "DIAL 1 &#215; 32, 0.2 s ahead"),
        ("anneal", (1, 20, 3, 10, 0.002, 48), "DIAL 1 &#215; 48, 0.2 s ahead"),
        ("anneal", (2, 20, 3, 20, 0.002, 32), "DIAL 2 &#215; 32, 0.2 s ahead, every 0.2 s"),
        ("anneal", CFG_RT7, "DIAL 2 &#215; 48, 0.2 s ahead, every 0.2 s"),
        ("grad", CFG_PICK, "Gradient descent 2 &#215; 32, 0.1 s ahead"),
        ("anneal", (1, 30, 3, 10, 0.002, 16), "DIAL 1 &#215; 16, 0.3 s ahead"),
        ("anneal", (2, 30, 3, 20, 0.002, 16), "DIAL 2 &#215; 16, 0.3 s ahead, every 0.2 s"),
        ("anneal", (2, 30, 3, 20, 0.002, 24), "DIAL 2 &#215; 24, 0.3 s ahead, every 0.2 s"),
        ("anneal", (1, 30, 3, 15, 0.002, 32), "DIAL 1 &#215; 32, 0.3 s ahead, every 0.15 s"),
        ("anneal", (2, 30, 3, 30, 0.002, 32), "DIAL 2 &#215; 32, 0.3 s ahead, every 0.3 s"),
        ("anneal", (3, 20, 3, 10, 0.002, 32), "DIAL 3 &#215; 32, 0.2 s ahead"),
        ("anneal", (4, 30, 3, 10, 0.002, 16), "DIAL 4 &#215; 16, 0.3 s ahead"),
        ("anneal", (4, 30, 3, 10, 0.002, 32), "DIAL 4 &#215; 32, 0.3 s ahead"),
        ("anneal", (2, 30, 3, 10, 0.002, 64), "DIAL 2 &#215; 64, 0.3 s ahead"),
        ("anneal", CFG_FULL2, "DIAL 4 &#215; 64, 0.3 s ahead, 2 ms step"),
        ("anneal", CFG_FULL, "DIAL 4 &#215; 64, 0.3 s ahead, 1 ms step (Section 4)"),
        ("grad", CFG_FULL2, "Gradient descent 4 &#215; 64, 0.3 s ahead"),
        ("ilqr", CFG_IL2, "iLQR, 2 iterations, 0.3 s ahead"),
        ("ilqr", CFG_IL4, "iLQR, 4 iterations, 0.3 s ahead")]


def il_rt(R, c):
    """One hand's replanning time of iLQR over the replanning interval, from its run at --chunk 1 on two twist tuning
    pairs (rows ..._chunk1), or None."""
    v = [r["replan_s_mean"] for r in R.get(tstem("ilqr", "twist", "_tune", c) + "_chunk1") or [] if r.get("replan_s_mean")]
    return float(np.mean(v)) / (c[3] * PERIOD_S) if v else None


def tstem(method, task, pairs, cfg=None, sh=""):
    """inner_loop_mpc.py's rows stem of a task and a pair set: mpc_<method>_<task><pairs>[<options>]<cfg_tag>."""
    return f"{method}_{task}{pairs}{sh}" + (cfg_tag(cfg) if cfg else "")


def mean_j(rows):
    return float(np.mean([r["J"] for r in rows])) if rows else None


def rt_ratio(T, c):
    """GPU time of one hand's replanning over the replanning interval, or None when it was not timed."""
    v = one_hand_cfg(T, c) if c else None
    return None if v is None else v / (c[3] * PERIOD_S)


def rank_ref(R, task):
    """The planner that the ranking test compares HOM with on a task: full DIAL on the 100 pairs when its rows exist,
    else the real-time pick. (stem, label)."""
    for c, lab in ((CFG_FULL2, "DIAL 4 &#215; 64"), (CFG_FULL, "DIAL 4 &#215; 64"),
                   (CFG_PICK, "DIAL 2 &#215; 32 (real time)")):
        st = tstem("anneal", task, "_rank", c)
        if len(R.get(st, [])) >= 100:
            return st, lab
    return None, None


RANK_RUNS = [("hom", None, "HOM"), ("anneal", CFG_PICK, "DIAL 2 &#215; 32, 0.1 s ahead, real time"),
             ("anneal", CFG_RT30, "DIAL 2 &#215; 16, 0.3 s ahead every 0.2 s, real time"),
             ("grad", CFG_PICK, "Gradient descent 2 &#215; 32, real time"),
             ("anneal", CFG_FULL2, "DIAL 4 &#215; 64, 2 ms step")]


def half(rows, task, by):
    """The rows of a _rank run on one half of its pairs: by 'hom_best' (HOM's 50 best layouts) or 'random'."""
    prs = {(p["x"], p["y"], p["grasp"]) for p in json.loads(
        (ROWS / f"mpc_pairs_{task}_rank.json").read_text())["pairs"] if p["by"] == by}
    return [r for r in rows if key(r)[:3] in prs]


def ilqr_text(R, j):
    """iLQR on the tuning pairs (inner_loop_ilqr.py, rows mpc_ilqr_<task>_tune_pdt2_r10h30k3i2): the method and its J
    against DIAL 2 x 64 over 0.3 s, the planner nearest its rollouts per replanning, and HOM; empty while rows are
    missing."""
    il = {k: R.get(tstem("ilqr", k, "_tune", CFG_IL2)) for k in TASKS4}
    il4 = {k: R.get(tstem("ilqr", k, "_tune", CFG_IL4)) for k in TASKS4}
    if not all(il.values()):
        return ""
    rows = [r for k in TASKS4 for r in il[k]]
    st = [r["ilqr_stats"] for r in rows if r.get("ilqr_stats")]
    rt = il_rt(R, CFG_IL2)
    d64 = (2, 30, 3, 10, 0.002, 64)
    return fill(r"""<p>iLQR [6] was run on the tuning pairs as well (<code>scripts/hom/inner_loop_ilqr.py</code>). Its
plan is the nine targets, each held for 50 ms, over the 0.3 s horizon. Each iteration linearizes the planner's model about
the plan by forward differences, as MJPC's iLQG does on the CPU. From each step's start one step is run with each of the 30
state coordinates and each of the 9 targets perturbed, 40 one-step rollouts per step. The backward pass treats the cost's
rotation and position terms as residuals (Gauss-Newton) and its rate term exactly, and regularizes $Q_{uu}$ as in [7]. The
pad and drop terms have no gradient and are scored only in the line search. Eight step sizes are rolled out in parallel,
$$u_t = \bar u_t + \alpha\, k_t + K_t\,(z_t \ominus \bar z_t), \qquad \alpha = 2^{-i},\ i = 0, \dots, 7,$$
where $z_t$ is the state with the last target and $\ominus$ the difference in the tangent space. The cheapest under the
full cost replaces the plan if it costs less. Two iterations take <<ro>> rollouts of the horizon per replanning, against
128 for DIAL 2 &#215; 64 over 0.3 s. iLQR scored <<ij>> on the twist, the roll, the pin-push and the press. DIAL
2 &#215; 64 scored <<dj>> and HOM <<hj>>. <<i4>> The line search lowered the plan's cost in <<acc>> % of the iterations and took
the full step in <<fs>> %. <<rts>> FDDP [8] differs from iLQR in its treatment of gaps in an infeasible initial guess; the
plan here starts from a rollout, which has none.</p>""",
                ro=f"{np.mean([r['rollouts_per_replan'] for r in rows]):.0f}",
                ij=", ".join(f"{mean_j(il[k]):.3f}" for k in TASKS4),
                dj=", ".join(f2(j("anneal", k, d64), 3) for k in TASKS4),
                hj=", ".join(f2(j("hom", k, None), 3) for k in TASKS4),
                acc=f"{100 * np.mean([x['accept'] for x in st]):.0f}" if st else "&#8212;",
                fs=f"{100 * np.mean([x['full_step'] for x in st]):.0f}" if st else "&#8212;",
                rts=(f"One hand's replanning took {rt:.1f} times the replanning interval." if rt is not None else ""),
                i4=(f"Four iterations, at {np.mean([r['rollouts_per_replan'] for k in TASKS4 for r in il4[k]]):.0f} "
                    "rollouts per replanning, scored " + ", ".join(f"{mean_j(il4[k]):.3f}" for k in TASKS4) + "."
                    if all(il4.values()) else ""))


def tasks_text(R, T, tn0):
    """The four-task section's text from the tuning and ranking rows (tn0: the number of the table before it)."""
    j = (lambda m, t, c, ps="_tune": mean_j(R.get(tstem(m, t, ps, c))))
    if j("hom", "roll", None) is None or j("anneal", "roll", CFG_PICK) is None:
        return ""
    h30 = [(1, 30, 3, 10, 0.002, 16), (2, 30, 3, 20, 0.002, 16), (4, 30, 3, 10, 0.002, 16), (4, 30, 3, 10, 0.002, 32),
           (2, 30, 3, 10, 0.002, 64), CFG_FULL2, CFG_FULL]
    short = [(1, 20, 3, 10, 0.002, 32), (1, 20, 3, 10, 0.002, 48), (2, 20, 3, 20, 0.002, 32), CFG_RT7, CFG_PICK]
    full_roll = [x for x in (j("anneal", "roll", c) for c in h30[2:]) if x is not None]
    short_roll = [x for x in (j("anneal", "roll", c) for c in short) if x is not None]
    t = fill(r"""<p>The planners were run on the roll, the pin-push and the press as well as the twist. In each task the
cost's rotation term is the error of J: the angle to the commanded orientation for the sphere, and for the cylinder the
angle between its axis and the commanded axis. The position term follows the sphere's centre, which moves along the
thumb's pad in the roll, and for the cylinder the pinch between thumb and index. Settings were compared on six tuning
layouts per task, outside the ranking test's pairs (Table <<t1>>).</p>
<p>The twist's real-time setting, two iterations of 32 samples over 0.1 s, failed on the roll and the pin-push.
It scored <<cr>> on the roll and <<cp>> on the pin-push, against HOM's <<hr>> and <<hp>>. On the roll the
horizon decided the outcome. Every DIAL setting that looked 0.3 s ahead scored <<fr>> there, from 16 to 64 samples
per iteration. Every setting that looked 0.2 s ahead or less scored at most <<sr>>. Within the real-time budget a
0.3 s horizon leaves room for two iterations of 16 samples every 0.2 s. That setting scored <<rr>> on the roll. The
gradient planner, at the twist setting's budget, scored <<gr>> on the roll and <<gp>> on the pin-push.</p>""",
             t1=tn0 + 1, cr=f"{j('anneal', 'roll', CFG_PICK):.3f}", cp=f"{j('anneal', 'pinpush', CFG_PICK):.3f}",
             hr=f"{j('hom', 'roll', None):.3f}", hp=f"{j('hom', 'pinpush', None):.3f}",
             fr=f"{min(full_roll):.2f} to {max(full_roll):.2f}" if full_roll else "&#8212;",
             sr=f"{max(short_roll):.3f}" if short_roll else "&#8212;",
             rr=f2(j("anneal", "roll", CFG_RT30), 3), gr=f2(j("grad", "roll", CFG_PICK), 3),
             gp=f2(j("grad", "pinpush", CFG_PICK), 3))
    t += ilqr_text(R, j)
    hom_r = {k: R.get(tstem("hom", k, "_rank")) for k in TASKS4}
    rt30 = {k: R.get(tstem("anneal", k, "_rank", CFG_RT30)) for k in TASKS4}
    full = {k: R.get(tstem("anneal", k, "_rank", CFG_FULL2)) for k in TASKS4}
    if all(hom_r.values()) and all(rt30.values()):
        dr = {k: paired(half(rt30[k], k, "random"), half(hom_r[k], k, "random")) for k in TASKS4}
        db = {k: paired(half(rt30[k], k, "hom_best"), half(hom_r[k], k, "hom_best")) for k in TASKS4}
        t += fill(r"""<p>On each task's 100 pairs HOM scored <<hj>> (Table <<t2>>). The real-time DIAL setting with the
0.3 s horizon scored <<rj>>. On HOM's 50 best layouts it trailed HOM by <<db>>. On the 50 layouts drawn at random
the paired difference from HOM ranged from <<dr>> (Table <<t4>>). The best half was chosen at seeds that these runs
share, which favours HOM on it.</p>""",
                  hj=", ".join(f"{mean_j(hom_r[k]):.3f}" for k in TASKS4) + " on the twist, the roll, the pin-push "
                     "and the press",
                  rj=", ".join(f"{mean_j(rt30[k]):.3f}" for k in TASKS4),
                  db=f"{min(-v['d'] for v in db.values()):.3f} to {max(-v['d'] for v in db.values()):.3f}",
                  dr=f"{min(v['d'] for v in dr.values()):+.3f} to {max(v['d'] for v in dr.values()):+.3f}",
                  t2=tn0 + 2, t4=tn0 + 4)
        if all(full.values()):
            dfu = {k: paired(full[k], hom_r[k]) for k in TASKS4}
            t += fill(r"""<p>DIAL with four iterations of 64 samples over 0.3 s needs 6.3 times real time. It scored
<<fj>>. Its paired difference from HOM was <<dd>> (Table <<t3>>), and it scored above HOM on <<wn>> of the 100
pairs.</p>""",
                      fj=and_list([f"{mean_j(full[k]):.3f}" for k in TASKS4]),
                      dd=and_list([f"{dfu[k]['d']:+.3f}" for k in TASKS4]),
                      wn=and_list([str(dfu[k]["wins"]) for k in TASKS4]), t3=tn0 + 3)
    return t


def sec_tasks(R, T):
    """HOM and the planners on the twist, the roll, the pin-push and the press: the configuration sweep on the tuning
    pairs, then mean J and the paired difference from HOM on the 100 pairs of each task."""
    tune = {(m, c): {t: R.get(tstem(m, t, "_tune", c)) for t in TASKS4} for m, c, _l in TUNE}
    if not any(v for d in tune.values() for v in d.values()):
        return ""
    tn0 = P.TAB[0]
    rows = []
    for m, c, lab in TUNE:
        js = [mean_j(tune[(m, c)][t]) for t in TASKS4]
        if all(j is None for j in js):
            continue
        rr = il_rt(R, c) if m == "ilqr" else rt_ratio(T, c)
        rows.append([lab, "0.20 (one CPU core)" if m == "hom" else f2(rr) if rr is not None else "&#8212;"]
                    + [f2(j, 3) if j is not None else "&#8212;" for j in js]
                    + [f2(np.mean(js), 3) if all(j is not None for j in js) else "&#8212;"])
    t1 = table(["Controller", "Time per second of control (s/s)"] + [TNAME[t] for t in TASKS4] + ["Mean"], rows,
               "Mean J on the tuning pairs: six layouts per task outside the ranking test's pairs, with HOM's J above "
               "0.05, each with its best grasp under HOM, seed 0. Time: GPU seconds of one hand's replanning per second "
               "of control, measured alone; at 1 or less the planner runs in real time. HOM's is one CPU core's.",
               num_from=1)
    rows, rowsd, have = [], [], False
    hom_r = {t: R.get(tstem("hom", t, "_rank")) for t in TASKS4}
    for m, c, lab in RANK_RUNS:
        cells, cellsd, any_ = [], [], False
        for t in TASKS4:
            rr = R.get(tstem(m, t, "_rank", c))
            if not rr:
                cells.append("&#8212;")
                cellsd.append("&#8212;")
                continue
            any_ = True
            cells.append(f"{mean_j(rr):.3f} ({len(rr)})")
            cellsd.append("&#8212;" if m == "hom" or not hom_r[t] else fd(paired(rr, hom_r[t])))
        if any_:
            have = True
            rows.append([lab] + cells)
            if m != "hom":
                rowsd.append([lab] + cellsd)
    out = tasks_text(R, T, tn0) + t1
    if have:
        out += table(["Controller"] + [TNAME[t] for t in TASKS4], rows,
                     "Mean J on each task's 100 pairs (HOM's 50 best layouts and 50 drawn at random, each with HOM's best "
                     "grasp), seeds 0 and 1; in brackets the number of runs.", num_from=1)
        if rowsd:
            out += table(["Controller"] + [TNAME[t] for t in TASKS4], rowsd,
                         "J minus HOM's on the same pairs and seeds: the mean over the pairs, its 95 % bootstrap interval "
                         "and the number of pairs where the planner was higher.", num_from=1)
            rowsr = []
            for m, c, lab in RANK_RUNS[1:]:
                cells = []
                for t in TASKS4:
                    rr, hr = R.get(tstem(m, t, "_rank", c)), hom_r[t]
                    cells.append(fd(paired(half(rr, t, "random"), half(hr, t, "random"))) if rr and hr else "&#8212;")
                if any(x != "&#8212;" for x in cells):
                    rowsr.append([lab] + cells)
            out += table(["Controller"] + [TNAME[t] for t in TASKS4], rowsr,
                         "J minus HOM's on the 50 pairs drawn at random. HOM's 50 best pairs were chosen at seeds 0-2 of "
                         "its grid, which include the seeds of these runs, so on that half the comparison favours HOM; "
                         "the random half has no such selection.", num_from=1)
    return out + task_films(R)


# the controllers of the four tasks' films (inner_loop_mpc.py film --methods), by their spec
FILM_SPEC = {"hom": "HOM", "anneal@2,10,2,10,2,32": "DIAL 2 &#215; 32, 0.1 s ahead, real time",
             "anneal@2,30,3,20,2,16": "DIAL 2 &#215; 16, 0.3 s ahead, real time",
             "grad@2,10,2,10,2,32": "gradient planner 2 &#215; 32, real time",
             "anneal@4,30,3,10,2,64": "DIAL 4 &#215; 64, 6.3 times real time"}


def task_films(R, date="20261115"):
    """The films of the four tasks (inner_loop_mpc.py film --task T --pairs _rank --pair k): HOM and the planners on one
    pair and seed, with each clip's J and grip from the film's rows/film_mpc_<task>_rank_p<k>_s<seed>.json."""
    out = ""
    for t in TASKS4:
        st = tstem("anneal", t, "_rank", CFG_FULL2)
        dbest = None
        if len(R.get(st) or load(st)) >= 150:
            dj = pair_j(R.get(st) or load(st))
            dbest = max(dj, key=dj.get)
        for p in sorted(MEDIA.glob(f"{date}-film_mpc_{t}_rank_p*_s*.mp4")):
            info_p = ROWS / (p.stem.replace(f"{date}-", "") + ".json")
            if not info_p.exists():
                continue
            info = json.loads(info_p.read_text())
            pr = info["pair"]
            k = (pr["x"], pr["y"], pr["grasp"])
            who = ["HOM's best layout by its grid" if p.stem.endswith("_p0_s0") else
                   {"hom_best": "one of HOM's 50 best layouts", "random": "a layout drawn at random"}.get(pr.get("by"), "")]
            if k == dbest:
                who.append("full DIAL's best of the 100 pairs")
            parts = []
            for spec, c in info["clips"].items():
                nm = FILM_SPEC.get(spec, esc(spec))
                parts.append(f"{nm[0].upper() + nm[1:]}: J {c['J']:.2f}, grip {c['grip_mean_N']:.1f} N"
                             + ("" if c.get("held") else ", dropped"))
            cap = (f"{TNAME[t]}: layout ({pr['x']:g}, {pr['y']:g}) mm, grasp {pr['grasp']}, seed {info['seed']}; "
                   + ", ".join(x for x in who if x) + ". The task phase from 2.4 s. " + ". ".join(parts) + ". A "
                   "planner's run in the film draws other samples than its runs in the tables. Each frame gives each "
                   "finger's pad force and the force through its rigid parts (N); the links are drawn see-through.")
            out += P.film(p, cap)
    return out


def rank_numbers(R, task, ref):
    """HOM's order of a task's 100 pairs against the reference planner's: Spearman's rho over all pairs, over HOM's 50
    best and over the 50 random ones, the overlap of the ten best, HOM's first choice under the planner, and each
    controller's agreement between seeds 0 and 1."""
    h_rows = R.get(tstem("hom", task, "_rank"))
    if not h_rows or not R.get(ref):
        return None
    prs = {(p["x"], p["y"], p["grasp"]): p["by"] for p in json.loads(
        (ROWS / f"mpc_pairs_{task}_rank.json").read_text())["pairs"]}
    H, Dp = pair_j(h_rows), pair_j(R[ref])
    keys = [k for k in prs if k in H and k in Dp]
    h, d = np.array([H[k] for k in keys]), np.array([Dp[k] for k in keys])
    best = np.array([prs[k] == "hom_best" for k in keys])
    top_h, top_d = set(np.argsort(-h)[:10]), set(np.argsort(-d)[:10])
    i_h = int(np.argmax(h))
    seeds, seeds_best = {}, {}
    for name, rows in (("hom", h_rows), ("ref", R[ref])):
        a, b = pair_j(rows, 0), pair_j(rows, 1)
        kk = [k for k in keys if k in a and k in b]
        seeds[name] = spearman([a[k] for k in kk], [b[k] for k in kk]) if len(kk) > 5 else None
        kb = [k for k in kk if prs[k] == "hom_best"]
        seeds_best[name] = spearman([a[k] for k in kb], [b[k] for k in kb]) if len(kb) > 5 else None
    rel = {k: (2.0 * v / (1.0 + v) if v is not None and v > 0 else None) for k, v in seeds.items()}
    rho = spearman(h, d)
    fresh = pair_j(R.get(tstem("hom", task, "_rank") + "_s1011") or load(tstem("hom", task, "_rank") + "_s1011"))
    hf = np.array([fresh.get(k, np.nan) for k in keys])
    return {"n": len(keys), "rho": rho, "rho_best": spearman(h[best], d[best]),
            "rho_rand": spearman(h[~best], d[~best]), "top10": len(top_h & top_d),
            "first_d": float(d[i_h]), "first_rank": int((d > d[i_h]).sum()) + 1, "best_d": float(d.max()),
            "mean_h_best": float(h[best].mean()), "mean_d_best": float(d[best].mean()),
            "mean_h_rand": float(h[~best].mean()), "mean_d_rand": float(d[~best].mean()),
            "seed_hom": seeds["hom"], "seed_ref": seeds["ref"], "rel_hom": rel["hom"], "rel_ref": rel["ref"],
            "seed_hom_best": seeds_best["hom"], "seed_ref_best": seeds_best["ref"],
            "rho_dis": rho / np.sqrt(rel["hom"] * rel["ref"]) if rel["hom"] and rel["ref"] else None,
            "fresh_best": float(np.nanmean(hf[best])) if np.isfinite(hf).sum() >= 50 else None,
            "fresh_rand": float(np.nanmean(hf[~best])) if np.isfinite(hf).sum() >= 50 else None,
            "rho_fresh": spearman(hf, d) if np.isfinite(hf).sum() >= 50 else None,
            "h": h.tolist(), "d": d.tolist(), "best": best.tolist()}


def fig_rank100(N, out):
    plt = plt_setup()
    tasks = [t for t in TASKS4 if N.get(t)]
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 9.2), squeeze=False)
    for a, t in zip(ax.flat, tasks):
        n = N[t]
        h, d, b = np.array(n["h"]), np.array(n["d"]), np.array(n["best"])
        a.scatter(h[~b], d[~b], s=22, color="#9ca3af", label="50 at random", lw=0)
        a.scatter(h[b], d[b], s=26, color="#2563eb", label="HOM's 50 best", lw=0)
        m = max(h.max(), d.max(), 0.05) * 1.05
        a.plot([0, m], [0, m], color="#6b7280", lw=1.0, ls=(0, (4, 3)))
        a.set_xlim(-0.02, m)
        a.set_ylim(-0.02, m)
        a.set_title(f"{TNAME[t]}: rho {n['rho']:.2f}", loc="left")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    for a in ax[1]:
        a.set_xlabel("J under HOM")
    for a in ax[:, 0]:
        a.set_ylabel("J under the planner")
    ax[0][0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def confirm_numbers(task):
    """The fresh-seed check of the ranking test (seeds 10 and 11 on each controller's ten best _rank pairs by its J at
    seeds 0 and 1): each set's mean J under both controllers at seeds 0-1 (the _rank rows) and 10-11, the difference of
    the two sets' means by bootstrap over pairs, the episodes that held the object, and full DIAL's J at 10-11 on HOM's
    first choice, on DIAL's own first choice and on the fused choice (DIAL's best of HOM's ten at seeds 0-1)."""
    pf = ROWS / f"mpc_pairs_{task}_top.json"
    hr, dr = load(tstem("hom", task, "_top")), load(tstem("anneal", task, "_top", CFG_FULL2))
    if not pf.exists() or len(hr) < 20 or len(dr) < 20:
        return None
    prs = json.loads(pf.read_text())["pairs"]
    fk = (lambda p: (p["x"], p["y"], p["grasp"]))
    ph = [p for p in prs if p["by"] in ("hom_top", "both")]
    pd = [p for p in prs if p["by"] in ("planner_top", "both")]
    kh, kd = [fk(p) for p in ph], [fk(p) for p in pd]
    first_h = max(ph, key=lambda p: p["J_hom"])
    first_d = max(pd, key=lambda p: p["J_planner"])
    fused = max(ph, key=lambda p: p["J_planner"])
    H, Dp = pair_j(hr), pair_j(dr)
    rng = np.random.default_rng(0)

    def diff(M):
        a, b = np.array([M[k] for k in kh if k in M]), np.array([M[k] for k in kd if k in M])
        bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(4000)]
        return {"d": float(a.mean() - b.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}

    def held(rows, ks):
        h = [bool(r["held"]) for r in rows if (r["x_sep"], r["y_sep"], r["grasp"]) in ks]
        return len(h) - sum(h), len(h)
    ks = set(kh) | set(kd)
    m = (lambda xs: float(np.mean(xs)))
    return {"n_h": len(kh), "n_d": len(kd), "both": sum(p["by"] == "both" for p in prs),
            "dial_on_hom": m([Dp[k] for k in kh]), "dial_on_dial": m([Dp[k] for k in kd]),
            "hom_on_hom": m([H[k] for k in kh]), "hom_on_dial": m([H[k] for k in kd]),
            "dial_on_hom_s01": m([p["J_planner"] for p in ph]), "dial_on_dial_s01": m([p["J_planner"] for p in pd]),
            "hom_on_hom_s01": m([p["J_hom"] for p in ph]), "hom_on_dial_s01": m([p["J_hom"] for p in pd]),
            "d_dial": diff(Dp), "d_hom": diff(H),
            "drop_hom_s01": held(load(tstem("hom", task, "_rank")), ks), "drop_hom": held(hr, ks),
            "drop_dial_s01": held(load(tstem("anneal", task, "_rank", CFG_FULL2)), ks), "drop_dial": held(dr, ks),
            "first_h_dial": Dp[fk(first_h)], "first_d_dial": Dp[fk(first_d)], "fused_dial": Dp[fk(fused)],
            "fused_dial_s01": fused["J_planner"], "fused_is": ("planner" if fk(fused) == fk(first_d) else
                                                               "hom" if fk(fused) == fk(first_h) else "other")}


def sec_confirm():
    C = {t: confirm_numbers(t) for t in TASKS4}
    C = {t: c for t, c in C.items() if c}
    if not C:
        return ""
    ar = (lambda a, b: f"{a:.3f} &#8594; {b:.3f}")
    rows = [[TNAME[t], f"{c['n_h']} / {c['n_d']} ({c['both']} shared)",
             ar(c["dial_on_hom_s01"], c["dial_on_hom"]), ar(c["dial_on_dial_s01"], c["dial_on_dial"]), fds(c["d_dial"]),
             ar(c["hom_on_hom_s01"], c["hom_on_hom"]), ar(c["hom_on_dial_s01"], c["hom_on_dial"]),
             f"{c['first_h_dial']:.3f} / {c['fused_dial']:.3f} / {c['first_d_dial']:.3f}"] for t, c in C.items()]
    tn = P.TAB[0] + 1
    fall_h = {t: c["dial_on_hom_s01"] - c["dial_on_hom"] for t, c in C.items()}
    fall_d = {t: c["dial_on_dial_s01"] - c["dial_on_dial"] for t, c in C.items()}
    tw = max(fall_d, key=fall_d.get)
    zero = [t for t, c in C.items() if c["d_dial"]["lo"] <= 0 <= c["d_dial"]["hi"]]
    low = [t for t, c in C.items() if c["d_dial"]["hi"] < 0]
    high = [t for t, c in C.items() if c["d_dial"]["lo"] > 0]
    s_ = []
    if zero:
        s_.append(f"On {tasks_list(zero)} the 95 % interval of the difference between the two sets included zero.")
    for t in low + high:
        s_.append(f"On the {TNAME[t].lower()} the difference was {fds(C[t]['d_dial'])}.")
    hl = [t for t, c in C.items() if c["d_hom"]["lo"] > 0]
    own = (f" HOM's ten still scored above DIAL's ten under HOM on {tasks_list(hl)}, by "
           + and_list([fds(C[t]["d_hom"]) for t in hl]) + "." if hl else "")
    cyl = [t for t in ("pinpush", "press") if t in C]
    drops = " ".join(
        f"On the {TNAME[t].lower()} HOM dropped the cylinder in {C[t]['drop_hom'][0]} of the {C[t]['drop_hom'][1]} runs "
        f"at seeds 10 and 11 and in {C[t]['drop_hom_s01'][0]} at seeds 0 and 1. Full DIAL dropped it in "
        + ("none of the same runs." if C[t]["drop_dial"][0] == C[t]["drop_dial_s01"][0] == 0 else
           f"{C[t]['drop_dial'][0]} and {C[t]['drop_dial_s01'][0]} of the same runs.") for t in cyl)
    gap = {t: c["first_d_dial"] - c["first_h_dial"] for t, c in C.items()}
    crit = (f"At seeds 10 and 11 HOM's first choice scored " + and_list([f"{gap[t]:.3f}" for t in C])
            + f" below DIAL's first choice under full DIAL, on the {and_list([TNAME[t].lower() for t in C])}. The second "
            f"outcome set in advance, a shortfall of more than 0.1, held on {tasks_list([t for t in C if gap[t] > 0.1])}.")
    same = [t for t, c in C.items() if c["fused_is"] == "planner"]
    other = [t for t in C if t not in same]
    fu = (f"Among HOM's ten, full DIAL's best pair at seeds 0 and 1 was its own first choice of the 100 on "
          f"{tasks_list(same)}. " if same else "")
    fu += " ".join(f"On the {TNAME[t].lower()} that pair scored {C[t]['fused_dial']:.3f} at the new seeds, and DIAL's "
                   f"own first choice {C[t]['first_d_dial']:.3f}." for t in other)
    text = fill(r"""<p>Each controller's ten best of the 100 pairs, by its J at seeds 0 and 1, were run again with both
controllers at seeds 10 and 11 (Table <<tn>>). These seeds chose no pair. Under full DIAL, HOM's ten lost
<<fh_lo>> to <<fh_hi>> of their mean J at the new seeds. DIAL's own ten lost up to <<fd_hi>>, on the <<tw>>; they had
been chosen on DIAL's J at seeds 0 and 1. <<cmp>></p>
<p>Under HOM both sets lost J at the new seeds, most on the cylinder tasks. <<drops>><<own>></p>
<p><<crit>> In the search of the <a href="../20261114-hom_inner_loop/20261114-hom_inner_loop_stability_variants_fusion.html">HOM
page</a>, full DIAL re-scores HOM's best layouts and chooses among them. <<fu>></p>""",
                tn=tn, fh_lo=f"{min(fall_h.values()):.3f}", fh_hi=f"{max(fall_h.values()):.3f}",
                fd_hi=f"{fall_d[tw]:.3f}", tw=TNAME[tw].lower(), cmp=" ".join(s_), drops=drops, own=own, fu=fu,
                crit=crit)
    return text + table(["Task", "Pairs, HOM's / DIAL's ten", "HOM's ten under full DIAL, seeds 0-1 &#8594; 10-11",
                         "DIAL's ten under full DIAL, seeds 0-1 &#8594; 10-11", "Difference at seeds 10-11 (95 % interval)",
                         "HOM's ten under HOM, seeds 0-1 &#8594; 10-11", "DIAL's ten under HOM, seeds 0-1 &#8594; 10-11",
                         "Full DIAL's J at seeds 10-11 on HOM's first choice / the fused choice / DIAL's first choice"],
                        rows,
                        "The ranking test at fresh seeds: each controller's ten best of the 100 pairs (by its J at seeds 0 "
                        "and 1) run again under both controllers at seeds 10 and 11; mean J of each set. Difference: "
                        "HOM's ten minus DIAL's ten under full DIAL, with the 95 % interval of 4,000 bootstrap resamples of "
                        "each set's pairs. Fused choice: the pair of HOM's ten with full DIAL's best J at seeds 0 and 1.",
                        num_from=1)


def and_list(xs):
    """'a, b and c'."""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def tasks_list(ts):
    """'the twist and the press', 'all four tasks' or 'no task'."""
    if not ts:
        return "no task"
    if len(ts) == len(TASKS4):
        return "all four tasks"
    nm = ["the " + TNAME[t].lower() for t in ts]
    return nm[0] if len(nm) == 1 else ", ".join(nm[:-1]) + " and " + nm[-1]


def sec_rank100(R):
    """The ranking test of 2026-11-14 (owner: "you can run the further test on running realtime DIAL"): HOM's order of
    each task's 100 pairs against a planner's; the figure is drawn here."""
    N, refs = {}, {}
    for t in TASKS4:
        st, lab = rank_ref(R, t)
        if st:
            N[t] = rank_numbers(R, t, st)
            refs[t] = lab
    N = {t: v for t, v in N.items() if v}
    if not N:
        return ""
    fig = fig_rank100(N, MEDIA / "20261114-mpc_rank100.png")
    tn, fn = P.TAB[0] + 1, P.FIG[0] + 1
    ts = list(N)
    lst = (lambda k, d=2: and_list([f"{N[t][k]:.{d}f}" for t in ts]))
    gap = {t: N[t]["best_d"] - N[t]["first_d"] for t in ts}
    text = fill(r"""<p>In the ranking test HOM's order of the pairs was compared with a planner's, to see whether HOM can stand
in for the planner when layouts are compared. For each task the 100 pairs were ordered by their J under HOM and under
<<pl>> (Table <<tn>>, Figure <<fn>>). Spearman's $\rho$ was <<rho>> on the <<tl>>. Over HOM's 50 best layouts alone it
was <<rb>>, and over the 50 random ones <<rr>>. Of HOM's ten best pairs, <<ov>> were among the planner's ten best.</p>
<p>Part of the disagreement is each controller's own noise. Between seeds 0 and 1 HOM's scores agreed with $\rho$
<<sh>>, the planner's with <<sd>>. Divided by the square root of the two reliabilities, $\rho$ becomes <<rc>>. Within
HOM's 50 best, HOM's scores at the two seeds agreed with $\rho$ <<shb>>, and the planner's with <<sdb>>.</p>
<p>On HOM's first choice the planner scored <<fd_>>. The planner's own best scored <<bd>>, so HOM's choice fell short
of the planner's best by <<gp>>. Two outcomes had been set in advance that would show that HOM cannot stand in for the
planner. One is a $\rho$ below 0.6 over HOM's 50 best. The other is HOM's first choice more than 0.1 below the
planner's best, in the planner's J. The first held on <<f1>>, the second on <<f2_>>.</p>""",
                pl=refs[ts[0]], tn=tn, fn=fn, rho=lst("rho"), tl=and_list([TNAME[t].lower() for t in ts]),
                rb=lst("rho_best"), rr=lst("rho_rand"), ov=and_list([str(N[t]["top10"]) for t in ts]),
                sh=lst("seed_hom"), sd=lst("seed_ref"),
                rc=and_list([f2(N[t]["rho_dis"]) for t in ts]), fd_=lst("first_d", 3), bd=lst("best_d", 3),
                gp=and_list([f"{gap[t]:.3f}" for t in ts]),
                shb=lst("seed_hom_best"), sdb=lst("seed_ref_best"),
                f1=tasks_list([t for t in ts if N[t]["rho_best"] < 0.6]),
                f2_=tasks_list([t for t in ts if gap[t] > 0.1]))
    rows = [[TNAME[t], refs[t], str(n["n"]), f2(n["rho"]), f2(n["rho_dis"]), f2(n["rho_best"]), f2(n["rho_rand"]),
             f"{n['top10']} of 10", f"{n['first_d']:.3f} (rank {n['first_rank']})", f"{n['best_d']:.3f}",
             f2(n["seed_hom"]), f2(n["seed_ref"])]
            for t, n in N.items()]
    out = table(["Task", "Planner", "Pairs", "&#961;, all", "&#961; corrected for seed noise", "&#961;, HOM's 50 best",
                 "&#961;, 50 random", "Ten best shared", "HOM's first choice under the planner", "Planner's best",
                 "&#961; seeds 0/1, HOM", "&#961; seeds 0/1, planner"], rows,
                "The order of each task's 100 pairs under HOM and under the planner: Spearman's &#961; over the pairs' J "
                "(seeds averaged within a pair), over all pairs and over each half; the same &#961; divided by the square "
                "root of the two controllers' seed reliabilities (Spearman-Brown for two seeds, 2r/(1+r), with r the "
                "&#961; between seeds 0 and 1); how many of HOM's ten best pairs are among the planner's ten best; the "
                "planner's J on HOM's best pair and its rank among the planner's J; and each controller's &#961; between "
                "seeds 0 and 1.", num_from=1)
    out = text + out
    out += P.img(fig, "Each task's 100 pairs: J under HOM (x) against J under the planner of Table "
                 f"{P.TAB[0]} (y), seeds averaged. Blue: HOM's 50 best layouts; grey: 50 drawn at random. Dashed: equal J.")
    return out + sec_confirm()


def sec_future():
    items = [
        "Full DIAL on every layout of the press's grid with HOM's best grasp, at seeds 0 and 1. The 369 layouts take "
        "about 1.6 hours at 7.8 s of GPU per rollout; <code>inner_loop_mpc.py pairs --rule</code> needs a grid rule. If "
        "more than five of the planner's ten best layouts lie outside HOM's 50 best, HOM's grid misses the planner's best "
        "press layouts.",
        "UHAS with a target-rate term in its reward (<code>uhas_sr2.py train --limits</code>, "
        "<code>REW_LIMITS['rate']</code> from -0.04 upwards). The term should bring its reversals to the planners' 1 to "
        "3 per joint per second. A test-split J at or below HOM's 0.085 at that rate would show that UHAS's lead over "
        "HOM needs the saturated targets.",
        "Planning from a state predicted over the latency, with the plan applied at the times it was computed for. The "
        "runs from a 140 ms old state apply the plan when it arrives, without that prediction "
        "(<code>measured</code> in <code>inner_loop_mpc.py</code>).",
        "The pads' torsional friction coefficient measured in this scene, to place the pads on the torsion axis of the "
        "contact figure. It is the torque about each tip's normal over its normal force while the sphere spins. The rig of "
        "1 October gives 1.4 to 1.8 mm; a value outside 1 to 2 mm in this scene would falsify that transfer to the v4b "
        "tip.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def sec_files():
    items = ["<code>scripts/hom/inner_loop_mpc.py</code>: planners, HOM and UHAS on the pairs; <code>timing</code>; "
             "<code>film</code>",
             "<code>scripts/hom/inner_loop_ilqr.py</code>: iLQR (method <code>ilqr</code> of the script above)",
             "<code>scripts/hom/inner_loop_plant.py</code>: the plant (SCS0009, 2.5 MPa scene cache)",
             "<code>scripts/hom/uhas_sr2.py</code>: <code>UBatch</code>, HOM's angle feedback (<code>ATT_GAIN</code>)",
             "<code>scripts/hom/inner_loop_mpc_page.py</code>: this page",
             "<code>docs/experiments/20261112-inner_loop_benchmark/rows/mpc_*.jsonl</code>: one row per run"]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def lede(S, T, R_=None):
    hs = S.get("hom_twist")
    if not hs:
        return "Results pending.", "Results pending."
    R_ = R_ or {}
    parts = [fill("On the SCS0009 plant, HOM's mean J over 40 runs of the sphere twist was <<h>>. It needs 2.0 ms of "
                  "one CPU core per 10 ms period.", h=f"{hs['J']:.3f}")]
    done = [(m, c, lab) for m, c, lab in RANK_RUNS
            if all(len(R_.get(tstem(m, t, "_rank", c)) or []) >= 150 for t in TASKS4)]
    if done and done[0][2] == "HOM" and done[-1] == RANK_RUNS[-1]:
        mj = {lab: float(np.mean([mean_j(R_[tstem(m, t, "_rank", c)]) for t in TASKS4])) for m, c, lab in done}
        rt = [v_ for lab, v_ in mj.items() if "real time" in lab] or [float("nan")]
        rho = [rank_numbers(R_, t, rank_ref(R_, t)[0])["rho"] for t in TASKS4]
        parts.insert(0, fill("On the SR2 hand's four tasks, with 100 layout-grasp pairs each, HOM's mean J was <<h>>. DIAL "
                             "at 4 x 64 samples, at 6.3 times real time, reached <<f>>, and the three real-time planners "
                             "<<lo>> to <<hi>>. HOM ordered the pairs as full DIAL did with Spearman's rho <<r>> on the "
                             "twist, roll, pin-push and press.", h=f"{mj['HOM']:.3f}",
                             f=f"{mj[RANK_RUNS[-1][2]]:.3f}", lo=f"{min(rt):.3f}", hi=f"{max(rt):.3f}",
                             r=", ".join(f2(x) for x in rho)))
    ok = [(S[stem_of("anneal", c)]["J"], c) for c in RT_CANDS if S.get(stem_of("anneal", c))
          and (one_hand_cfg(T, c) or 1e9) <= c[3] * PERIOD_S]
    if ok:
        bj, bc = max(ok)
        dh = paired(R_.get(stem_of("anneal", bc), []), R_.get("hom_twist", []))
        parts.append(fill("DIAL annealing at <<n>> samples<<st>> runs in real time. It reached <<j>>, and the "
                          "paired difference from HOM was <<d>>.", n=f"{bc[0]} x {bc[5]}", j=f"{bj:.3f}", d=fds(dh),
                          st=f" with a {1e3 * bc[4]:g} ms model step" if bc[4] else ""))
        b, a1 = S.get(stem_of("anneal", bc)), S.get(stem_of("anneal", bc, sh="_acc1"))
        samp = [S.get(stem_of(m, c, sh=sh)) for m, c, sh, _l, k in REV_RUNS if k == "sampling" and c == CFG_REV]
        b16 = S.get(stem_of("anneal", CFG_REV))
        if b and a1:
            none = b16 and all(x and x["reversals_ps"] >= b16["reversals_ps"] for x in samp)
            parts.append(fill("With a cost on the change of the target rate its targets reversed <<a>> times per joint "
                              "per second, fewer than HOM's <<h>>, at J <<j>>."
                              + (" No change to the sampling lowered the reversals." if none else ""),
                              a=f2(a1["reversals_ps"]), h=f2(hs["reversals_ps"]), j=f"{a1['J']:.3f}"))
    h1, h2 = S.get(hom_stem("pt")), S.get(hom_stem("pt_t2"))
    if h1:
        parts.append(fill("With point contact in place of the pads HOM's J fell to <<b>>"
                          + (", and torsional friction of 2 mm on the point contacts raised it to <<c>>." if h2 else "."),
                          b=f"{h1['J']:.3f}", c=f"{h2['J']:.3f}" if h2 else ""))
    U = usability(UHAS_SV[0][1])
    if U["uhas"]:
        su, sh_ = stats(U["uhas"]), stats(U["hom"])
        parts.append(fill("UHAS retrained on this plant reached <<u>> on the test split, HOM <<h>>, with its servo "
                          "targets reversing <<r>> times per joint per second.", u=f"{su['J']:.3f}",
                          h=f"{sh_['J']:.3f}", r=f"{su['reversals_ps']:.0f}"))
    text = " ".join(parts)
    return text, re.sub(r"\s+", " ", text)


def main():
    P.FIG[0] = P.TAB[0] = P.FILM[0] = 0
    stems = [s for s, _ in MAIN] + [f"hom_twist_att{g}" for g in (1, 2, 4, 8)] + [
        "hs_anneal_twist", "hs_anneal_twist_hs0.1", "hs_ps_twist", "hs_icem_twist", "hom_twist_ft1.5", "hom_twist_ft2",
        "hom_twist_ft3.5", "anneal_twist_d4n", "ps_twist_d4n_r10h10k2i1"] + [f"hom_twist_{v}" for v, _d in VARIANTS] + [
        stem for stem, _l in PF_RUNS] + [
        "anneal_twist_pt", "hom_twist_mu0.4", "ps_twist_mu0.4_r10h10k2i1", "anneal_twist_mu0.4", "spider_twist_mu0.4",
        "spider_wc_twist_mu0.4"] + [f"uhas_twist{mu}_{tag}" for tag, _r, _l in UHAS_SV for mu in ("", "_mu0.4")]
    for p, _ in PLANTS:                             # the contact runs: every plant, the pad, point and own models
        stems.append(hom_stem(p))
        for meth, cfg in [("ps", CFG_PSRT), ("anneal", CFG_FULL)] + [("anneal", c) for c in RT_CANDS]:
            stems += [stem_of(meth, cfg, plant=p, model=m) for m in ("pads", "pt", p)]
    for meth, cfg in RT_TABLE:                      # the real-time configurations, and from delayed noisy states
        stems += [stem_of(meth, cfg, delay=dl) for dl in ("", "_d4n", "_d14n")]
    stems += [stem_of(m, c, sh=sh) for m, c, sh, _l, _k in REV_RUNS] + [s_ for s_, _l in REV_REF]
    for t in TASKS4:
        stems += [tstem(m, t, "_tune", c) for m, c, _l in TUNE] + [tstem(m, t, "_rank", c) for m, c, _l in RANK_RUNS]
    stems.append(tstem("ilqr", "twist", "_tune", CFG_IL2) + "_chunk1")
    stems = list(dict.fromkeys(stems))
    R = {s: load(s) for s in stems}
    hom = {key(r): r for r in R["hom_twist"]}
    hom04 = {key(r): r for r in R["hom_twist_mu0.4"]}
    S = {s: stats(v, hom04 if "mu0.4" in s else hom) for s, v in R.items() if v}
    TABLE_ROWS.update({s_: {key(r): r for r in v} for s_, v in R.items() if v})
    SW = {}
    hw = {key(r): r for r in load("hom_twist_work10")}
    for stem, _lab in MAIN + [("uhas_twist", "")]:
        v = load(stem + "_work10") if stem != "ps_twist_r10h10k2i1" else load("ps_twist_r10h10k2i1_work10")
        if v:
            SW[stem] = stats(v, hw)
    figs = {}
    MEDIA.mkdir(parents=True, exist_ok=True)
    if len(S) > 1:
        figs["overview"] = fig_overview(S, MEDIA / "20261113-mpc_overview.png")
    if R.get("anneal_twist") and R["hom_twist"]:
        figs["pairs"] = fig_pairs(R["hom_twist"], R["anneal_twist"], "DIAL annealing", MEDIA / "20261113-mpc_pairs.png")
    T = timing_rows()
    rt = rt_pick(R, T)
    if S.get(hom_stem("pt")) or S.get(stem_of("ps", CFG_PSRT, model="pt")):
        figs["contact"] = fig_contact(S, R, rt, MEDIA / "20261113-mpc_contact_models.png")
    if any(S.get(stem_of("anneal", c)) for c in RT_CANDS):
        figs["realtime"] = fig_realtime(S, T, MEDIA / "20261113-mpc_realtime.png")
    if any(S.get(stem_of(m, c, sh=sh)) for m, c, sh, _l, _k in REV_RUNS if sh):
        figs["reversals"] = fig_reversals(S, MEDIA / "20261113-mpc_reversals.png")
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("setup", "Plant, task and pairs", sec_setup(S)),
                ("methods", "The controllers", sec_methods()),
                ("results", "HOM and the planners on the SCS0009 plant", sec_results(S, figs)),
                ("rank", "Mean J and ranking of the pairs under HOM and the planners", sec_rank(S, R, T)),
                ("tasks", "HOM and the planners on four tasks", sec_tasks(R, T)),
                ("rank100", "Ranking of 100 pairs per task under HOM and a planner", sec_rank100(R)),
                ("films", "Films", sec_films()),
                ("uhas", "UHAS retrained on the SCS0009 plant", sec_uhas(S, R)),
                ("feedback", "Two simple changes to HOM", sec_feedback(S)),
                ("hs", "Sampling in HOM's space", sec_hs(S)),
                ("compute", "Compute per second of control", sec_compute(T, S)),
                ("realtime", "Annealed sampling at the real-time budget", sec_realtime(S, T, R, rt, figs)),
                ("reversals", "Target reversals of the sampling planners", sec_reversals(S, R, figs)),
                ("contact", "Fingertip contact model of the plant and of the planner", sec_contact(S, R, T, rt, figs)),
                ("sensing", "Measured state in place of the simulator's", sec_sensing(S)),
                ("filter", "A complementary filter on the sphere's pose", sec_filter(S)),
                ("friction", "Plant friction 0.4", sec_friction(S, R)),
                ("work", "The same runs on the work plant", sec_work(SW)),
                ("future", "Future work", sec_future()),
                ("refs", "References and files", '<ol class="refs">' + "".join(
                    f"<li>{r[r.index(']') + 2:]}</li>" for r in REFS) + "</ol>" + sec_files())]
    sections = [x for x in sections if x[2]]
    num = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sections = [(sid, hd, b.replace("WORKSEC", str(num.get("work", "&#8212;")))) for sid, hd, b in sections]
    lede_t, summary = lede(S, T, R)
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|s\b|min\b|h\b)", "\\1&nbsp;\\2", s_))
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{unit(b)}</section>'
                     for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("Sampling MPC against HOM on the inner-loop benchmark's sphere twist (2026-11-13) and on four tasks "
              "(2026-11-14), SCS0009 plant at 2.5 MPa, "
              f'MuJoCo-Warp 3.6. The benchmark: <a href="{BENCH}">{BENCH}</a>. Builder: '
              "<code>scripts/hom/inner_loop_mpc_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "Sampling MPC and HOM", "H1": "Sampling MPC and HOM on the SR2 hand's four tasks",
                 "EYEBROW": "Inner-loop benchmark, 2026-11-13", "LEDE": unit(lede_t), "BYLINE": byline,
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-13")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

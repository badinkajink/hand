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
    med = (lambda k: float(np.median([r[k] for r in H])) if H else float("nan"))
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
    p = ROWS / "mpc_timing_twist.jsonl"
    if not p.exists():
        return []
    return [r for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip()) if r.get("plant") == SERVO]


def one_hand(T, n, iters, H, model="pads"):
    """GPU seconds per replanning of one hand alone (timing rows, the last of each setting)."""
    m = [r for r in T if r["plant_worlds"] == 1 and r["n"] == n and r.get("iters") == iters and r["H"] == H
         and r.get("model", "pads") == model]
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


def fd(p):
    """A paired difference as text: +0.035 (-0.01 to +0.08), 12 of 20."""
    if not p:
        return "&#8212;"
    return f"{p['d']:+.3f} ({p['lo']:+.3f} to {p['hi']:+.3f}), {p['wins']} of {p['n']}"


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


def stem_of(method, cfg, plant="pads", model="pads", delay=""):
    """inner_loop_mpc.py's rows stem: mpc_<method>_twist[_plant_<plant>][<delay>][_<model>]<cfg_tag>."""
    return (f"{method}_twist" + ("" if plant == "pads" else f"_plant_{plant}") + delay
            + ("" if model == "pads" else f"_{model}") + cfg_tag(cfg))


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


def fig_contact(S, rt, out):
    """Mean J against the plant's contact model for HOM and the planners with the pad and the point-contact model."""
    plt = plt_setup()
    lines = contact_lines(S, rt)
    xs = {p: i for i, (p, _) in enumerate(PLANTS)}
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    k = 0
    for lab, col, ls, pts in lines:
        P_ = [(xs[p], s["J"], s["J_se"]) for p, s in pts.items() if s]
        if not P_:
            continue
        P_.sort()
        dx = 0.035 * (k - len(lines) / 2)
        k += 1
        ax.errorbar([x + dx for x, _, _ in P_], [y for _, y, _ in P_], yerr=[e for _, _, e in P_], color=col, ls=ls,
                    marker="o" if ls == "-" else "s" if ls == "--" else "^", ms=5, capsize=2, lw=1.6, label=lab)
    ax.set_xticks(range(len(PLANTS)))
    ax.set_xticklabels([PLANT_SHORT[p] for p, _ in PLANTS])
    ax.set_xlabel("Plant's fingertip contact (torsional friction coefficient of the point contact)")
    ax.set_ylabel("Mean J over 40 runs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def fig_realtime(S, T, out):
    """Mean J against the GPU time per second of control of one hand, for predictive sampling and DIAL annealing."""
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for meth, cfg in RT_TABLE:
        s, v = S.get(stem_of(meth, cfg)), one_hand_cfg(T, cfg)
        if not s or v is None:
            continue
        g = v / (cfg[3] * PERIOD_S)
        ax.errorbar([g], [s["J"]], yerr=[s["J_se"]], fmt="o" if meth == "anneal" else "s", capsize=2,
                    color="#3a6ea5" if meth == "anneal" else "#c06a1a", ms=6)
        it, h, k, r, dt, n = cfg
        ax.annotate(f"{it}x{n}, {h * 10} ms" + (f", {1e3 * dt:g} ms step" if dt else "") + (", every 0.2 s" if r == 20
                                                                                          else ""),
                    (g, s["J"]), textcoords="offset points", xytext=(6, 4), fontsize=10)
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
    ax.legend(frameon=False, loc="lower right")
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
        t += P.img(figs["contact"], "Mean J against the plant's fingertip contact (bars: one standard error over the 40 "
                   "runs). Solid lines plan with the pad model, dashed with the point-contact model, dotted with the "
                   "plant's own torsion model; HOM in grey.")
    # paired differences: the pad model against the point-contact model on each plant, and the plants against each other
    rows_p, plan = [], []
    for meth, cfg in (("ps", CFG_PSRT), ("anneal", rt), ("anneal", CFG_FULL)):
        if cfg is None:
            continue
        lab = f"{names[meth]}{' 4 &#215; 64' if cfg == CFG_FULL else (' real time' if meth == 'anneal' else '')}"
        plan.append((meth, cfg, lab))
        for model in ("pads", "pt"):
            d = paired(R.get(stem_of(meth, cfg, model=model), []), R.get(stem_of(meth, cfg, plant="pt", model=model),
                                                                         []))
            if d:
                rows_p.append([f"{lab}, {'pad' if model == 'pads' else 'point-contact'} model", fd(d)])
    dh = paired(R.get("hom_twist", []), R.get(hom_stem("pt"), []))
    if dh:
        rows_p.insert(0, ["HOM", fd(dh)])
    rows_m = [[lab_p] + [fd(paired(R.get(stem_of(m, c, plant=p), []), R.get(stem_of(m, c, plant=p, model="pt"), [])))
                         for m, c, _l in plan] for p, lab_p in PLANTS]
    rows_m = [r_ for r_ in rows_m if any(x != "&#8212;" for x in r_[1:])]
    tm_ = P.TAB[0] + 1
    if rows_m:
        t += table(["Plant"] + [lab for _m, _c, lab in plan], rows_m,
                   "J with the pad model minus J with the point-contact model, on each plant: the mean over the 20 "
                   "pairs (seeds averaged within a pair), its 95 % bootstrap interval, and the pairs where the pad "
                   "model was higher.")
    if rows_p:
        t += table(["Controller", "J on the pads plant minus J on the point-contact plant"], rows_p,
                   "The plant's effect on each controller, with the same statistics.")
    t += sec_contact_text(S, R, rt, tj, tm_)
    return t


def sec_contact_text(S, R, rt, tj, tm_):
    """Sentences on the contact runs from the numbers that exist."""
    out = []
    h0, h1 = S.get("hom_twist"), S.get(hom_stem("pt"))
    if h0 and h1:
        out.append(fill("HOM's mean J was <<a>> on the pads and <<b>> on point contact (Table <<tj>>).",
                        a=f"{h0['J']:.3f}", b=f"{h1['J']:.3f}", tj=tj))
        ht = [(PLANT_SHORT[p], S[hom_stem(p)]["J"]) for p in ("pt_t0.5", "pt_t1", "pt_t2") if S.get(hom_stem(p))]
        if ht:
            out.append("With torsion it was " + ", ".join(f"{j:.3f} at {lab}" for lab, j in ht) + ".")
    for meth, cfg, lab in (("anneal", CFG_FULL, "DIAL annealing at 4 &#215; 64"), ("anneal", rt, "real-time DIAL"),
                           ("ps", CFG_PSRT, "real-time predictive sampling")):
        if cfg is None:
            continue
        a = paired(R.get(stem_of(meth, cfg), []), R.get(stem_of(meth, cfg, model="pt"), []))
        b = paired(R.get(stem_of(meth, cfg, plant="pt"), []), R.get(stem_of(meth, cfg, plant="pt", model="pt"), []))
        if a and b:
            out.append(fill("For <<l>> the pad model changed J by <<a>> on the pads plant and by <<b>> on the "
                            "point-contact plant (Table <<tm>>).", l=lab, a=f"{a['d']:+.3f}", b=f"{b['d']:+.3f}",
                            tm=tm_))
    return f"<p>{' '.join(out)}</p>" if out else ""


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
    for meth, cfg in RT_TABLE:
        s = S.get(stem_of(meth, cfg))
        if not s:
            continue
        it, h, k, r, dt, n = cfg
        v = one_hand_cfg(T, cfg)
        g = v / (r * PERIOD_S) if v is not None else None
        rows.append(["PS" if meth == "ps" else "DIAL", f"{it} &#215; {n}", f"{h * PERIOD_S:g}", str(k),
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
                   "standard error). Labels give iterations x samples, the horizon, the model's physics step if not 1 ms "
                   "and the replanning interval if not 0.1 s.")
    ok = [(S[stem_of("anneal", c)]["J"], c) for c in RT_CANDS if S.get(stem_of("anneal", c))
          and (one_hand_cfg(T, c) or 1e9) <= c[3] * PERIOD_S]
    sf, sp = S.get("anneal_twist"), S.get(stem_of("ps", CFG_PSRT))
    if ok and sf:
        bj, bc = max(ok)
        t += fill("<p>The best real-time DIAL configuration, <<c>>, reached <<j>>. DIAL at the benchmark's budget "
                  "reached <<f>> and real-time predictive sampling <<p>>.</p>", c=cfg_text(bc), j=f"{bj:.3f}",
                  f=f"{sf['J']:.3f}", p=f"{sp['J']:.3f}" if sp else "&#8212;")
    if rt is not None:
        lat = []
        for meth, cfg, lab in (("ps", CFG_PSRT, "PS, real time"), ("anneal", rt, "DIAL, real time")):
            cells = [lab]
            for dl in ("", "_d4n", "_d14n"):
                s = S.get(stem_of(meth, cfg, delay=dl))
                cells.append(f2(s["J"], 3) if s else "&#8212;")
            lat.append(cells)
        if any(c != "&#8212;" for r_ in lat for c in r_[2:]):
            t += r"""<p>On the bench the planner starts from the tag tracker's pose, 40 ms old with 0.5 mm and 1 deg of
noise. Its plan also reaches the servos one replanning time after that pose. The runs below plan from the plant's
state 4 or 14 periods earlier, with the tracker's noise on the sphere's pose. The plan is applied when it is computed,
without a forward prediction of the state.</p>"""
            t += table(["Planner", "Exact state", "40 ms old, tracker noise", "140 ms old, tracker noise"], lat,
                       "Mean J of the real-time planners from delayed, noisy states (<code>--plan-delay</code>, "
                       "<code>--plan-noise</code>); 140 ms is the tracker's 40 ms and one replanning interval.",
                       num_from=1)
    return t


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


def sec_friction(S):
    rows = []
    for stem, lab in (("hom_twist_mu0.4", "HOM"), ("ps_twist_mu0.4_r10h10k2i1", "Predictive sampling, real-time budget"),
                      ("anneal_twist_mu0.4", "DIAL annealing"), ("spider_twist_mu0.4", "SPIDER"),
                      ("spider_wc_twist_mu0.4", "SPIDER, worst case")):
        s = S.get(stem)
        if s:
            rows.append([lab, f2(s["J"], 3), f"{100 * s['held']:.0f}", f2(s["err"], 1), f2(s["grip_mean_N"], 1)])
    if not rows:
        return ""
    return (r"""<p>The plant's friction is set to 0.4, the value measured for TPU on bare PLA on the UR5. The planners
keep friction 1.0 in their model; the worst case of SPIDER samples 0.4, 0.7 and 1.0.</p>"""
            + table(["Controller", "Mean J", "Held (%)", "Tracking error (deg)", "Grip (N)"], rows,
                    "Plant friction 0.4, the same 40 runs."))


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


def sec_future():
    items = [
        "UHAS retrained on the SCS0009 plant (published reward, and with the limit terms), 120 min each, then its "
        "evaluation, usability and the 20 pairs (<code>scripts/hom/uhas_sr2.py train</code>; lane "
        "<code>logs/20261112-mpc/lane.txt</code>).",
        "The friction-0.4 block, MPPI, CEM and predictive sampling at the full budget on the SCS0009 plant (same lane).",
        "Planning from a state predicted over the latency, with the plan applied at the times it was computed for. The "
        "runs from a 140 ms old state apply the plan when it arrives, without that prediction "
        "(<code>measured</code> in <code>inner_loop_mpc.py</code>).",
        "The pads' torsional friction coefficient measured in this scene: the torque about each tip's normal over its "
        "normal force while the sphere spins, to place the pads on the torsion axis of the contact figure. The rig of "
        "1 October gives 1.4 to 1.8 mm; a value outside 1 to 2 mm in this scene would falsify that transfer to the v4b "
        "tip.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def sec_files():
    items = ["<code>scripts/hom/inner_loop_mpc.py</code>: planners, HOM and UHAS on the pairs; <code>timing</code>; "
             "<code>film</code>",
             "<code>scripts/hom/inner_loop_plant.py</code>: the plant (SCS0009, 2.5 MPa scene cache)",
             "<code>scripts/hom/uhas_sr2.py</code>: <code>UBatch</code>, HOM's angle feedback (<code>ATT_GAIN</code>)",
             "<code>scripts/hom/inner_loop_mpc_page.py</code>: this page",
             "<code>docs/experiments/20261112-inner_loop_benchmark/rows/mpc_*.jsonl</code>: one row per run"]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def lede(S, T, R_=None):
    hs = S.get("hom_twist")
    if not hs:
        return "Results pending.", "Results pending."
    parts = [fill("On the SCS0009 plant, HOM's mean J over 40 runs of the sphere twist was <<h>> with <<g>> N of grip. "
                  "It needs 2.0 ms of one CPU core per 10 ms period.", h=f"{hs['J']:.3f}", g=f"{hs['grip_mean_N']:.1f}")]
    pl = [(S[stem]["J"], lab, S[stem]) for stem, lab in MAIN if S.get(stem) and stem not in
          ("hom_twist", "ps_twist_r10h10k2i1")]
    v = one_hand(T, 64, 4, 30)
    if pl:
        bj, bl, bs = max(pl, key=lambda x: x[0])
        parts.append(fill("The best sampling planner, <<bl>>, reached <<bj>> with <<g>> N of grip. Its targets "
                          "reversed <<r>> times per joint per second, HOM's <<hr>> times.", bl=bl, bj=f"{bj:.3f}",
                          g=f"{bs['grip_mean_N']:.1f}", r=f"{bs['reversals_ps']:.1f}", hr=f"{hs['reversals_ps']:.1f}"))
        if v:
            parts.append(fill("For one hand it needs <<v>> s of GPU per second of control.", v=f"{v / 0.1:.1f}"))
    rt = S.get("ps_twist_r10h10k2i1")
    if rt:
        parts.append(fill("At a budget that runs in real time, predictive sampling reached <<r>>.", r=f"{rt['J']:.3f}"))
    ok = [(S[stem_of("anneal", c)]["J"], c) for c in RT_CANDS if S.get(stem_of("anneal", c))
          and (one_hand_cfg(T, c) or 1e9) <= c[3] * PERIOD_S]
    if ok:
        bj, bc = max(ok)
        parts.append(fill("DIAL annealing reached <<j>> in real time with <<c>>.", j=f"{bj:.3f}",
                          c=cfg_text(bc).replace("&#215;", "x")))
    h1 = S.get(hom_stem("pt"))
    a = paired(R_.get("anneal_twist", []), R_.get("anneal_twist_pt", [])) if R_ else None
    b = paired(R_.get(stem_of("anneal", CFG_FULL, plant="pt"), []),
               R_.get(stem_of("anneal", CFG_FULL, plant="pt", model="pt"), [])) if R_ else None
    if h1:
        parts.append(fill("With point-contact fingertips in the plant HOM's J was <<b>>.", b=f"{h1['J']:.3f}"))
    if a and b:
        parts.append(fill("DIAL's J with the pad model minus its J with the point-contact model was <<a>> on the pads "
                          "plant and <<b>> on the point-contact plant.", a=f"{a['d']:+.3f}", b=f"{b['d']:+.3f}"))
    h2 = S.get("hs_anneal_twist")
    if h2:
        parts.append(fill("Sampling HOM's rolling and sliding rates in place of the servo targets gave <<j>> with <<r>> "
                          "reversals per second.", j=f"{h2['J']:.3f}", r=f"{h2['reversals_ps']:.1f}"))
    fb = [(S[f"hom_twist_att{g}"]["J"], g) for g in (1, 2, 4, 8) if S.get(f"hom_twist_att{g}")]
    if fb:
        bj, bg = max(fb)
        parts.append(fill("Feedback on the measured angle lowered HOM's tracking error but also its three-pad share; "
                          "its best J was <<j>> at <<g>>/s.", j=f"{bj:.3f}", g=f"{bg:g}"))
    ft = [(S[f"hom_twist_ft{f:g}"]["J"], f) for f in (1.5, 2.0, 3.5) if S.get(f"hom_twist_ft{f:g}")]
    if ft:
        bj, bf = max(ft)
        if bj > hs["J"]:
            parts.append(fill("A commanded pad force of <<f>> N per finger in place of 2.5 N raised HOM's J to <<j>>.",
                              f=f"{bf:g}", j=f"{bj:.3f}"))
    text = " ".join(parts)
    return text, re.sub(r"\s+", " ", text)


def main():
    P.FIG[0] = P.TAB[0] = P.FILM[0] = 0
    stems = [s for s, _ in MAIN] + [f"hom_twist_att{g}" for g in (1, 2, 4, 8)] + [
        "hs_anneal_twist", "hs_anneal_twist_hs0.1", "hs_ps_twist", "hs_icem_twist", "hom_twist_ft1.5", "hom_twist_ft2",
        "hom_twist_ft3.5", "anneal_twist_d4n", "ps_twist_d4n_r10h10k2i1"] + [f"hom_twist_{v}" for v, _d in VARIANTS] + [
        stem for stem, _l in PF_RUNS] + [
        "anneal_twist_pt", "hom_twist_mu0.4", "ps_twist_mu0.4_r10h10k2i1", "anneal_twist_mu0.4", "spider_twist_mu0.4",
        "spider_wc_twist_mu0.4"]
    for p, _ in PLANTS:                             # the contact runs: every plant, the pad, point and own models
        stems.append(hom_stem(p))
        for meth, cfg in [("ps", CFG_PSRT), ("anneal", CFG_FULL)] + [("anneal", c) for c in RT_CANDS]:
            stems += [stem_of(meth, cfg, plant=p, model=m) for m in ("pads", "pt", p)]
    for meth, cfg in RT_TABLE:                      # the real-time configurations, and from delayed noisy states
        stems += [stem_of(meth, cfg, delay=dl) for dl in ("", "_d4n", "_d14n")]
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
        figs["contact"] = fig_contact(S, rt, MEDIA / "20261113-mpc_contact_models.png")
    if any(S.get(stem_of("anneal", c)) for c in RT_CANDS):
        figs["realtime"] = fig_realtime(S, T, MEDIA / "20261113-mpc_realtime.png")
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("setup", "Plant, task and pairs", sec_setup(S)),
                ("methods", "The controllers", sec_methods()),
                ("results", "HOM and the planners on the SCS0009 plant", sec_results(S, figs)),
                ("films", "Films", sec_films()),
                ("feedback", "Two simple changes to HOM", sec_feedback(S)),
                ("hs", "Sampling in HOM's space", sec_hs(S)),
                ("compute", "Compute per second of control", sec_compute(T, S)),
                ("realtime", "Annealed sampling at the real-time budget", sec_realtime(S, T, R, rt, figs)),
                ("contact", "Fingertip contact model of the plant and of the planner", sec_contact(S, R, T, rt, figs)),
                ("sensing", "Measured state in place of the simulator's", sec_sensing(S)),
                ("filter", "A complementary filter on the sphere's pose", sec_filter(S)),
                ("friction", "Plant friction 0.4", sec_friction(S)),
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
    byline = ("Sampling MPC against HOM on the inner-loop benchmark's sphere twist, 2026-11-13, SCS0009 plant at 2.5 MPa, "
              f'MuJoCo-Warp 3.6. The benchmark: <a href="{BENCH}">{BENCH}</a>. Builder: '
              "<code>scripts/hom/inner_loop_mpc_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "Sampling MPC and HOM", "H1": "Sampling MPC and HOM on the SR2 sphere twist",
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

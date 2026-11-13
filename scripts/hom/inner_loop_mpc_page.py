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
        ["Replanning time", "wall time of one replanning, every 0.1 s of control; for 16 hands planned together "
         "(the runs) or one hand alone (the timing runs)", "s"],
        ["GPU time per second of control", "replanning time of one hand alone over 0.1 s; 1 is real time", "s/s"],
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
        parts = []
        k = (pr["x"], pr["y"], pr["grasp"], info["seed"])
        for spec, c in cl.items():
            nm = spec.split(":")[0]
            tab = TABLE_ROWS.get(FILM_STEM.get(nm, ""), {}).get(k)
            parts.append(f"{FILM_NAME.get(nm, esc(spec))}: J {c['J']:.2f}"
                         + (f" (in the table's run {tab['J']:.2f})" if tab and nm != "hom" else "")
                         + f", grip {c['grip_mean_N']:.1f} N, {c['reversals_ps']:.1f} reversals per joint per s")
        cap = (f"Layout ({pr['x']:g}, {pr['y']:g}) mm, grasp {pr['grasp']}, seed {info['seed']}, plant friction "
               f"{info['plant_mu']:g}; the task phase from 2.4 s. " + "; ".join(parts) + ". A planner's run in the "
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


def sec_pt(S):
    s, sp = S.get("anneal_twist_pt"), S.get("anneal_twist")
    if not s:
        return ""
    rows = [["DIAL annealing, pads model", f2(sp["J"], 3) if sp else "&#8212;", f"{100 * sp['held']:.0f}" if sp else
             "&#8212;", f2(sp["grip_mean_N"], 1) if sp else "&#8212;", f2(sp["reversals_ps"]) if sp else "&#8212;",
             f2(sp["replan_s"], 2) if sp else "&#8212;"],
            ["DIAL annealing, point-contact model", f2(s["J"], 3), f"{100 * s['held']:.0f}", f2(s["grip_mean_N"], 1),
             f2(s["reversals_ps"]), f2(s["replan_s"], 2)]]
    t = r"""<p>The point-contact model is the same hand with each v4b tip as one convex mesh that touches the sphere at
one point, where the plant's tip is 1,018 sphere pads (the scene cache's <code>pt</code> model). The planner rolls out
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
    t = r"""<p>A HOM-space plan sets, per finger and knot, the two rolling and the two sliding rates of the contact
relative to the sphere: 12 numbers per knot, the separation and the spin about the normal left to HOM. Each period HOM
solves its Eq. (2) as usual. It then solves it again with every row's reference taken from its own solution, the plan's
rates added to the rolling and sliding rows, those rows at weight 1 and the separation row at 10 times its weight. A
finger has three joints, so without the heavier separation row the added rates pulled the pads off the sphere. With
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
        rows.append(["Exact, 100 Hz (Table 3)", f2(hs["J"], 3), f"{100 * hs['held']:.0f}", f2(hs["err"], 1),
                     f2(hs["contact"]), f2(hs["grip_mean_N"], 1)])
    for v, desc in VARIANTS:
        s_ = S.get(f"hom_twist_{v}")
        if s_:
            rows.append([desc[0].upper() + desc[1:], f2(s_["J"], 3), f"{100 * s_['held']:.0f}", f2(s_["err"], 1),
                         f2(s_["contact"]), f2(s_["grip_mean_N"], 1)])
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
        t += table(["Measurement", "Mean J", "Held (%)", "Tracking error (deg)", "Three-pad share", "Grip (N)"], rows,
                   "HOM under the sensing variants of <code>hom_tasks.VARIANTS</code>, on the same 40 runs.")
    if pl:
        t += r"""<p>A planner copies the plant's state into its rollouts. With the tag tracker the copy is 40 ms old, and
the sphere's pose in it carries the tracker's noise. Here each plan starts from the plant's state 4 periods earlier, with
0.5 mm and 1 deg of noise on the sphere's pose, and is applied from now (<code>--plan-delay 4 --plan-noise</code>).</p>"""
        t += table(["Planner", "Mean J, exact state", "Mean J, measured state", "Held (%)", "Tracking error (deg)",
                    "Three-pad share"], pl, "Planners from a 40 ms old, noisy state, on the same 40 runs.")
    return t


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
        "HOM-space sampling: per finger the two rolling and two sliding rates of Eq. (2)'s reference relative twist, "
        "12 numbers per knot, with v<sub>sep</sub> and the spin left to HOM (the 30 October planner, "
        "<code>scripts/hom/hom_mpc_gpu.py</code>, sampled 2 numbers per knot). The claim that sampling in HOM's space "
        "is cheaper is falsified if it needs more than 256 rollouts per 0.1 s to reach DIAL annealing's J.",
        "Noisy state: HOM and the planners with the tag tracker's noise (0.5 mm, 1 deg, 40 ms late) and the magnetic "
        "fingertips' contact estimates (<code>hom_tasks.VARIANTS</code> sense and tip), at 100 Hz; and without the "
        "tags, the sphere's pose estimated in the hand from the contacts.",
        "UHAS retrained on the SCS0009 plant (published reward, and with the limit terms), 120 min each, then its "
        "evaluation, usability and the 20 pairs (<code>scripts/hom/uhas_sr2.py train</code>).",
        "The friction-0.4 block and MPPI, CEM, iCEM, SPIDER and its worst case on the SCS0009 plant (lane "
        "<code>logs/20261112-mpc/lane.txt</code>).",
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


def lede(S, T):
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
    h2 = S.get("hs_anneal_twist")
    if h2:
        parts.append(fill("Sampling HOM's rolling and sliding rates in place of the servo targets gave <<j>> with <<r>> "
                          "reversals per second.", j=f"{h2['J']:.3f}", r=f"{h2['reversals_ps']:.1f}"))
    fb = [(S[f"hom_twist_att{g}"]["J"], g) for g in (1, 2, 4, 8) if S.get(f"hom_twist_att{g}")]
    if fb:
        bj, bg = max(fb)
        parts.append(fill("Feedback on the measured angle lowered HOM's tracking error but also its three-pad share; "
                          "its best J was <<j>> at <<g>>/s.", j=f"{bj:.3f}", g=f"{bg:g}"))
    text = " ".join(parts)
    return text, re.sub(r"\s+", " ", text)


def main():
    P.FIG[0] = P.TAB[0] = P.FILM[0] = 0
    stems = [s for s, _ in MAIN] + [f"hom_twist_att{g}" for g in (1, 2, 4, 8)] + [
        "hs_anneal_twist", "hs_anneal_twist_hs0.1", "hs_ps_twist", "hs_icem_twist", "hom_twist_ft1.5", "hom_twist_ft2",
        "hom_twist_ft3.5", "anneal_twist_d4n", "ps_twist_d4n_r10h10k2i1"] + [f"hom_twist_{v}" for v, _d in VARIANTS] + [
        "anneal_twist_pt", "hom_twist_mu0.4", "ps_twist_mu0.4_r10h10k2i1", "anneal_twist_mu0.4", "spider_twist_mu0.4",
        "spider_wc_twist_mu0.4"]
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
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("setup", "Plant, task and pairs", sec_setup(S)),
                ("methods", "The controllers", sec_methods()),
                ("results", "HOM and the planners on the SCS0009 plant", sec_results(S, figs)),
                ("films", "Films", sec_films()),
                ("feedback", "Two simple changes to HOM", sec_feedback(S)),
                ("hs", "Sampling in HOM's space", sec_hs(S)),
                ("pt", "A planner with point-contact fingertips", sec_pt(S)),
                ("sensing", "Measured state in place of the simulator's", sec_sensing(S)),
                ("compute", "Compute per second of control", sec_compute(T, S)),
                ("friction", "Plant friction 0.4", sec_friction(S)),
                ("work", "The same runs on the work plant", sec_work(SW)),
                ("future", "Future work", sec_future()),
                ("refs", "References and files", '<ol class="refs">' + "".join(
                    f"<li>{r[r.index(']') + 2:]}</li>" for r in REFS) + "</ol>" + sec_files())]
    sections = [x for x in sections if x[2]]
    num = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sections = [(sid, hd, b.replace("WORKSEC", str(num.get("work", "&#8212;")))) for sid, hd, b in sections]
    lede_t, summary = lede(S, T)
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

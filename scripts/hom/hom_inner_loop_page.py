#!/usr/bin/env python3
"""Builds docs/experiments/20261114-hom_inner_loop/20261114-hom_inner_loop_stability_variants_fusion.html: HOM as the
inner-loop controller of the SR2 hand search, against sampling MPC (owner, 2026-11-14: "i would not like to stop using
HOM, it is quite nice to have such a fast controller. if theres anything else you can think of to improve it, feel free
to try. is there a better algorithm to use than DLS? can we fuse methods? ... did we try gradient based methods?"; "i
wonder if there are emergent trends in the hyperparameters such as impedance forces, twist weights"; "ideally we have
further evidence that HOM is the right choice ... stability across metrics, tasks, solutions? ... explore this").

Data:
  - the CEM refinements of HOM's forces and contact modes under J of 2026-11-11 (scripts/hom/hom_grasp_refine.py,
    levels hom and shared, docs/experiments/20261103-hom_score_refine/rows/sv25_track/), reduced to each layout's pick
    in data/20261111-hom_picks_track.json (refreshed with --picks);
  - the runs of scripts/hom/inner_loop_mpc.py on the benchmark's plant (rows of
    docs/experiments/20261112-inner_loop_benchmark/rows/): HOM, real-time and full DIAL on the four tasks' pair sets
    _tune, _rank and _rank20, HOM's variants (--preview, --prm-shared, --hom-param, --variant floop) and the planners
    fused with HOM (res_anneal, hs_anneal);
  - the grids grid2dff_v4b_sv25r of HOM on every layout (seeds 0-2).

    .venv/bin/python scripts/hom/hom_inner_loop_page.py [--picks]
"""
from __future__ import annotations

import argparse
import collections
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
import inner_loop_mpc_page as MP  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
from sr2hand.numerics.stats import spearman  # noqa: E402

D = ROOT / "docs/experiments/20261114-hom_inner_loop"
MEDIA, DATA = D / "media", D / "data"
OUT = D / "20261114-hom_inner_loop_stability_variants_fusion.html"
TEX_CACHE = D / "texsvg_cache.json"
MPC_PAGE = "../20261112-inner_loop_benchmark/20261113-sampling_mpc_vs_hom.html"
# HOM's GPU time per twist rollout in the CEM workload (the talk's compute chart, codesign_section_figures.py)
FIGJSON = ROOT / "docs/experiments/20261110-codesign_formulation/media/20261110-figures.json"
al = MP.and_list
REFINE = ROOT / "docs/experiments/20261103-hom_score_refine"
REFINE_PAGE = "../20261103-hom_score_refine/20261103-hom_score_refine.html"
TRACK = REFINE / "rows/sv25_track"
TASKS = ("twist", "roll", "pinpush", "press")
D_MM = {"twist": 40, "roll": 40, "pinpush": 25, "press": 25}
TASK_NAME = {"twist": "twist", "roll": "roll", "pinpush": "pin-push", "press": "press"}
# HOM's defaults (hom_tasks.PRM, mode_weights) for the forces and contact modes that level hom searched
DEFAULT = {"f_tripod": 2.5, "f_pinch": 3.0, "f_drive": 1.5, "f_press": 0.5, "w_spin_tripod": 1.0, "w_roll_tripod": 0.0,
           "w_slide_tripod": 1.0, "w_spin_pinch": 0.0, "w_roll_pinch": 0.0, "w_slide_pinch": 1.0, "w_spin_push": 1.0,
           "w_roll_push": 1.0, "w_slide_push": 1.0}


def esc(s):
    return P.esc(s)


# ------------------------------------------------------------------------------------------------ the 11-11 picks
def picks_from_rows():
    """Each layout's pick of level hom (the candidate with the best mean J over seeds 0-2 of the search) with its
    forces and mode weights, the layout's coordinates and the default's J (candidate 0 of iteration 0, HOM's defaults),
    for the tasks that level hom ran; and each task's shared pick (level shared)."""
    import hom_landscape as L
    import hom_rows
    out = {"source": str(TRACK.relative_to(ROOT)), "tasks": {}}
    for task in TASKS:
        p = TRACK / "hom" / f"{task}_d{D_MM[task]}.jsonl"
        sh = TRACK / "shared" / f"{task}_d{D_MM[task]}_pick.json"
        rec = {"shared": json.loads(sh.read_text())["pick"]["prm"] if sh.exists() else None}
        if p.exists():
            cand, prm_of, d_of = collections.defaultdict(list), {}, {}
            for line in p.read_text().splitlines():
                r = hom_rows.rescale_metrics(json.loads(line))
                k = (r["call"], r["it"], r["cand"])
                cand[k].append(L.track_score(task, r) if r.get("status", "ok") == "ok" else 0.0)
                prm_of[k], d_of[r["call"]] = r["prm"], r["d"]
            best, base = {}, {}
            for (call, it, c), v in cand.items():
                m = float(np.mean(v))
                if call not in best or m > best[call][0]:
                    best[call] = (m, prm_of[(call, it, c)])
                if it == 0 and c == 0:
                    base[call] = m
            calls = sorted(best)
            rec.update({"calls": calls, "J_pick": [best[c][0] for c in calls], "J_default": [base[c] for c in calls],
                        "x_sep": [round(d_of[c][2] - d_of[c][0], 3) for c in calls],
                        "y_sep": [round(d_of[c][3] - d_of[c][5], 3) for c in calls],
                        "prm": {k: [best[c][1][k] for c in calls] for k in sorted(best[calls[0]][1])}})
        out["tasks"][task] = rec
    lv = json.loads((TRACK / "20261111-tuning_levels.json").read_text())["tasks"]
    for task in TASKS:
        t = lv.get(task, {})
        out["tasks"][task]["val"] = {k: {"median": t[k]["median"], "gain": t[k]["gain"]["median"],
                                         "up": t[k]["gain"]["up"], "down": t[k]["gain"]["down"],
                                         "rho_s1": t[k]["rho_s1_val"]}
                                     for k in ("hom_val", "shared_val", "both_val") if k in t}
        out["tasks"][task]["val"]["s1_median"] = t.get("median_s1_val")
    return out


def picks(refresh=False):
    path = DATA / "20261111-hom_picks_track.json"
    if refresh or not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(picks_from_rows(), indent=1))
    return json.loads(path.read_text())


FORCE_KEYS = (("twist", "f_tripod", "twist, pad force"), ("roll", "f_tripod", "roll, pad force"),
              ("press", "f_pinch", "press, pinch force"), ("press", "f_press", "press, pusher force"))
MODE_KEYS = (("twist", "w_spin_tripod", "twist, spin"), ("twist", "w_slide_tripod", "twist, slide"),
             ("roll", "w_spin_tripod", "roll, spin"), ("roll", "w_slide_tripod", "roll, slide"),
             ("press", "w_slide_pinch", "press, pinch slide"), ("press", "w_spin_push", "press, pusher spin"),
             ("press", "w_roll_push", "press, pusher roll"), ("press", "w_slide_push", "press, pusher slide"))


def fig_trends(PK, out):
    """(a) The picked forces of the 30 best layouts per task as a share of HOM's default, the shared pick marked; (b) the
    roll's picked pad force against x_sep, coloured by y_sep; (c) the picked mode weights against the defaults."""
    plt = MP.plt_setup()
    fig = plt.figure(figsize=(10.5, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    rng = np.random.default_rng(0)
    for i, (task, k, lab) in enumerate(FORCE_KEYS):
        T = PK["tasks"][task]
        v = np.array(T["prm"][k]) / DEFAULT[k]
        ax[0].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color="#2563eb", alpha=0.75, lw=0)
        ax[0].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="#1e3a8a", lw=2.5)
        if T.get("shared") and k in T["shared"]:
            ax[0].scatter([i + 0.32], [T["shared"][k] / DEFAULT[k]], marker="D", s=60, color="#d97706", zorder=3)
    ax[0].axhline(1.0, color="#6b7280", lw=1.0, ls=(0, (4, 3)))
    ax[0].set_xticks(range(len(FORCE_KEYS)))
    ax[0].set_xticklabels([lab.replace(", ", ",\n") for _t, _k, lab in FORCE_KEYS], fontsize=12, rotation=25, ha="right")
    ax[0].set_ylabel("Picked force / HOM's default")
    ax[0].set_title("(a) Picked forces", loc="left")
    T = PK["tasks"]["roll"]
    sc = ax[1].scatter(T["x_sep"], T["prm"]["f_tripod"], c=T["y_sep"], cmap="viridis", s=42, lw=0)
    ax[1].axhline(DEFAULT["f_tripod"], color="#6b7280", lw=1.0, ls=(0, (4, 3)))
    cb = fig.colorbar(sc, ax=ax[1], pad=0.02)
    cb.set_label("y_sep (mm)")
    ax[1].set_xlabel("x_sep (mm)")
    ax[1].set_ylabel("Picked pad force (N)")
    r1 = spearman(T["x_sep"], T["prm"]["f_tripod"])
    r2 = spearman(T["y_sep"], T["prm"]["f_tripod"])
    ax[1].set_title(f"(b) Roll: rho {r1:+.2f} with x_sep,\n{r2:+.2f} with y_sep", loc="left")
    for i, (task, k, lab) in enumerate(MODE_KEYS):
        v = np.array(PK["tasks"][task]["prm"][k])
        ax[2].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=18, color="#2563eb", alpha=0.7, lw=0)
        ax[2].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="#1e3a8a", lw=2.5)
        ax[2].scatter([i], [DEFAULT[k]], marker="_", s=380, color="#dc2626", lw=2.5, zorder=3)
    ax[2].set_xticks(range(len(MODE_KEYS)))
    ax[2].set_xticklabels([lab for _t, _k, lab in MODE_KEYS], fontsize=12, rotation=30, ha="right")
    ax[2].set_ylim(-0.05, 1.08)
    ax[2].set_ylabel("Picked weight of Eq. (2)'s row")
    ax[2].set_title("(c) Picked contact-mode weights", loc="left")
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def trend_numbers(PK):
    """Medians, quartiles and rank correlations of each picked parameter with the layout's coordinates and the default's
    J, per task."""
    out = {}
    for task, T in PK["tasks"].items():
        if "prm" not in T:
            continue
        d0, xs, ys = np.array(T["J_default"]), T["x_sep"], T["y_sep"]
        for k, v in T["prm"].items():
            v = np.array(v)
            out[(task, k)] = {"med": float(np.median(v)), "q1": float(np.quantile(v, 0.25)),
                              "q3": float(np.quantile(v, 0.75)), "rx": spearman(v, xs), "ry": spearman(v, ys),
                              "rj": spearman(v, d0), "rg": spearman(v, np.array(T["J_pick"]) - d0)}
    return out


# ------------------------------------------------------------------------------------------------ HOM's grids
def grid_stability_from_rows():
    """Per task, HOM's grid (grid2dff_v4b_sv25r, every layout with each grasp of its set at seeds 0-2): each layout's
    best grasp by the mean of the seeds; Spearman's rho between the layouts' J at seed 0 and at seed 1 and the mean
    |J0 - J1|; and the landscape's smoothness, the mean |dJ| between lattice neighbours 5 mm apart over the mean |dJ|
    between 20,000 random pairs of layouts."""
    import hom_landscape as L
    import hom_rows
    import inner_loop_benchmark_page as BP
    out = {}
    for task in TASKS:
        per = collections.defaultdict(dict)
        for line in (BP.GRID / f"{task}_d{D_MM[task]:g}.jsonl").read_text().splitlines():
            r = hom_rows.rescale_metrics(json.loads(line))
            if r.get("status", "ok") != "ok":
                continue
            dd = r["d"]
            per[(round(dd[2] - dd[0], 3), round(dd[3] - dd[5], 3), int(r.get("grasp", 0)))][r["seed"]] = \
                L.track_score(task, r)
        lay = {}
        for (x, y, _g), sd in per.items():
            m = float(np.mean(list(sd.values())))
            if (x, y) not in lay or m > lay[(x, y)][0]:
                lay[(x, y)] = (m, sd)
        ks = sorted(lay)
        J = np.array([lay[k][0] for k in ks])
        s0 = np.array([lay[k][1].get(0, np.nan) for k in ks])
        s1 = np.array([lay[k][1].get(1, np.nan) for k in ks])
        ok = np.isfinite(s0) & np.isfinite(s1)
        idx = {k: i for i, k in enumerate(ks)}
        nb = [abs(J[i] - J[idx[(x + dx, y + dy)]]) for (x, y), i in idx.items() for dx, dy in ((5, 0), (0, 5))
              if (x + dx, y + dy) in idx]
        rng = np.random.default_rng(0)
        a, b = rng.integers(0, len(J), 20000), rng.integers(0, len(J), 20000)
        out[task] = {"layouts": len(ks), "rho_seeds": spearman(s0[ok], s1[ok]),
                     "mean_abs_seed": float(np.mean(np.abs(s0[ok] - s1[ok]))), "neighbour": float(np.mean(nb)),
                     "random": float(np.mean(np.abs(J[a] - J[b]))), "pairs_neighbour": len(nb), "sd": float(J.std())}
    return out


def grid_stability(refresh=False):
    path = DATA / "20261114-hom_grid_stability.json"
    if refresh or not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(grid_stability_from_rows(), indent=1))
    return json.loads(path.read_text())


# ------------------------------------------------------------------------------------------------ the runs
fill, table = P.fill, P.table
f2, fd, fds, paired, key, load = MP.f2, MP.fd, MP.fds, MP.paired, MP.key, MP.load
TN = MP.TNAME
CFG_PICK, CFG_RT7, CFG_FULL2 = MP.CFG_PICK, MP.CFG_RT7, MP.CFG_FULL2
# HOM's variants on each task's 20 pairs (_rank20): (rows option tag, label, tasks)
VARIANTS = [("_pv50", "Command read 50 ms ahead", TASKS), ("_pv100", "Command read 100 ms ahead", TASKS),
            ("_prmsh", "Forces and contact modes of the task's shared pick (11-11)", TASKS),
            ("_hpwdamp0.0001", "Eq. (2) damping 0.0001 (default 0.001)", TASKS),
            ("_hpwdamp0.01", "Eq. (2) damping 0.01", TASKS), ("_hpwdamp0.1", "Eq. (2) damping 0.1", TASKS),
            ("_hpumax1", "Joint-rate bound 1 rad/s (default 2)", TASKS), ("_hpumax4", "Joint-rate bound 4 rad/s", TASKS),
            ("_hplead0.15", "Reference lead 0.15 rad (default 0.3)", TASKS),
            ("_floop", "Force loop on the separation, 10 mm/s per N", ("twist", "roll")),
            ("_att2", "Orientation feedback on all three axes, 2/s", ("twist", "roll")),
            ("_att1ax", "Orientation feedback about the commanded axis, 1/s", ("twist", "roll")),
            ("_att4ax", "Orientation feedback about the commanded axis, 4/s", ("twist", "roll"))]
FUSION = [("res_anneal", "HOM with a DIAL residual in joint space, 2 &#215; 32"),
          ("hs_anneal", "DIAL in HOM's space, 2 &#215; 32")]


def rows_of(R, stem):
    if stem not in R:
        R[stem] = load(stem)
    return R[stem]


def hom_rank(R, task):
    return rows_of(R, f"hom_{task}_rank")


def sub20(R, task, rows):
    """The rows of the _rank20 pairs among a _rank run's rows."""
    prs = {(p["x"], p["y"], p["grasp"]) for p in json.loads(
        (MP.ROWS / f"mpc_pairs_{task}_rank20.json").read_text())["pairs"]}
    return [r for r in rows if key(r)[:3] in prs]


def med(rows, k):
    v = [r[k] for r in rows if r.get("held") and r.get(k) is not None]
    return float(np.median(v)) if v else float("nan")


# ------------------------------------------------------------------------------------------------ sections
def sec_terms():
    rows = [
        ["J", "tracking score of one run: $J = \\max(0, 1 - e/e_{\\mathrm{ref}}) \\cdot c$. Here $e$ is the RMS angle "
         "between the object's orientation and the commanded one over the task (for the cylinders, between the axis and "
         "the commanded axis), $e_{\\mathrm{ref}}$ the RMS commanded rotation and $c$ the three-pad share; 0 when the "
         "object fell", "0 to 1; 1 follows the command on three pads"],
        ["Tracking factor", "$\\max(0, 1 - e/e_{\\mathrm{ref}})$, the first factor of J", "0 to 1"],
        ["Three-pad share $c$", "share of the task with all three pads above 0.05 N", "0 to 1"],
        ["Pair", "a finger layout (thumb-to-pair distance $x_{\\mathrm{sep}}$, pair spacing $y_{\\mathrm{sep}}$) with one "
         "grasp of its set; HOM's best grasp on the layout's grid row", "&#8212;"],
        ["Pair sets", "per task: <i>rank</i>, the 50 layouts with HOM's best J and 50 drawn at random from the rest; "
         "<i>rank20</i>, every fifth pair of each half; <i>tune</i>, six other layouts with HOM's J above 0.05, for "
         "choosing a planner's settings", "100, 20, 6 pairs"],
        ["Paired difference", "J of one controller minus J of another on the same pairs and seeds, seeds averaged within a "
         "pair; the mean over the pairs with its 95 % interval from 4,000 bootstrap resamples of the pairs", "J"],
        ["Rank correlation $\\rho$", "Spearman's rank correlation of two sets of J over the same pairs "
         "(<code>sr2hand.numerics.stats.spearman</code>)", "-1 to 1"],
        ["Smoothness ratio", "mean |J difference| between lattice neighbours 5 mm apart over the mean |J difference| "
         "between random pairs of layouts; 1 is no spatial order", "0 to about 1"],
        ["Target reversals", "sign changes of a servo target's rate above 1 deg/s, per joint per second", "1/s"],
        ["Grip", "total normal force of the three fingers on the object, mean over the task; median over held runs", "N"],
        ["Real time", "one hand's replanning takes no longer than the replanning interval on the RTX 4070 Ti SUPER; "
         "time per second of control is that GPU time over the interval", "s/s"],
        ["Horizon", "how far ahead a planner's rollouts look", "s"],
    ]
    return table(["Term", "Definition", "Unit or range"], rows, "Terms and metrics. Every metric is computed over the "
                 "task phase, from HOM's handover at 2.4 s to the end of the schedule.", num_from=1)


def sec_setup():
    return fill(r"""<p>The plant is the one of the hand-design talk. Its servos follow the SCS0009 model of 0.72 N m/rad
with a 0.18 N m limit. The v4b tips are sphere pads at 2.5 MPa, and the scene friction is 1.0. MuJoCo-Warp 3.6 steps
it at 1 ms, and the controller runs every 10 ms.
The object is picked up and lifted under HOM. From 2.4 s the nine servo targets are set by the controller under test.
The four
tasks are HOM's demonstration tasks (<a href="<<mp>>">sampling MPC page</a>, Section on four tasks). The twist turns
the 40 mm sphere about the vertical by &#177;20&#176; and then ramps it at 30&#176;/s for 3 s. The roll rolls the same
sphere along the thumb's pad by &#177;20&#176; twice. The pin-push turns a 25 mm cylinder in a thumb-index pinch by
&#177;15&#176; with the middle finger; the press lifts it 20&#176; and lowers it again.</p>
<p>The planners are those of the sampling MPC page. DIAL annealing [1] rolls out the plant's own model on the GPU, a
batch of samples per hand. Its plan is the nine servo targets at two or three knots, linear between them. In the
gradient planner (method <code>grad</code>) the same rollouts are spent on finite differences. A least-squares gradient
is fitted to central differences at 0.02 rad, along each free knot coordinate and along random directions. Then as many
step sizes are tried in a line search. The scheme is MJPC's gradient planner [2]; MuJoCo-Warp 3.6 has no derivatives. In the fused planners
HOM runs in every rollout and in the plant. A joint-space offset is added to HOM's servo targets in
<code>res_anneal</code>, and rolls and slides are added to Eq. (2)'s reference in <code>hs_anneal</code>. The zero
plan of both is HOM.</p>""",
                mp=MPC_PAGE)


def tasks_table(R, T):
    """Per task: HOM, the best real-time DIAL setting of the tuning pairs, the gradient planner and full DIAL on the
    100 pairs: mean J, the paired difference from HOM."""
    runs = [("anneal", CFG_PICK, "DIAL 2 &#215; 32, 0.1 s ahead, real time (the twist's pick)"),
            ("anneal", MP.CFG_RT30, "DIAL 2 &#215; 16, 0.3 s ahead every 0.2 s, real time"),
            ("grad", CFG_PICK, "Gradient descent 2 &#215; 32, real time"),
            ("anneal", CFG_FULL2, "DIAL 4 &#215; 64, 6.3 times real time")]
    rows, have = [], False
    hrow = ["HOM, 0.2 s of one CPU core per second"]
    for t in TASKS:
        h = hom_rank(R, t)
        hrow.append(f"{MP.mean_j(h):.3f}" if h else "&#8212;")
    rows.append(hrow)
    for m, c, lab in runs:
        row = [lab]
        for t in TASKS:
            rr, h = rows_of(R, MP.tstem(m, t, "_rank", c)), hom_rank(R, t)
            if rr and h:
                have = True
                p_ = paired(rr, h)
                row.append(f"{MP.mean_j(rr):.3f}<br>{fd(p_)}" if p_ else f"{MP.mean_j(rr):.3f}")
            else:
                row.append("&#8212;")
        rows.append(row)
    return have, rows


def gpu_ratio(R, stem):
    """GPU time per twist rollout of a planner's run (16 hands per batch) over HOM's in the CEM workload, as in the
    talk's cost chart (codesign_section_figures.planner_cost)."""
    rows = [r for r in rows_of(R, stem) if r.get("chunk_wall_s")]
    hom_s = json.loads(FIGJSON.read_text())["compute_track"]["cem"]["ms_per_rollout"]["twist"] / 1e3
    return float(np.mean([r["chunk_wall_s"] / r["chunk_worlds"] for r in rows])) / hom_s if rows else None


RT_RUNS = (("c1", "anneal", "CFG_PICK"), ("rt30", "anneal", "CFG_RT30"), ("grad", "grad", "CFG_PICK"))


def sec_tasks(R, T):
    have, rows = tasks_table(R, T)
    if not have:
        return ""
    cfg = {"CFG_PICK": CFG_PICK, "CFG_RT30": MP.CFG_RT30}
    pd = {k: {t: paired(rows_of(R, MP.tstem(m, t, "_rank", cfg[c])), hom_rank(R, t)) for t in TASKS}
          for k, m, c in RT_RUNS}
    pf = {t: paired(rows_of(R, MP.tstem("anneal", t, "_rank", CFG_FULL2)), hom_rank(R, t)) for t in TASKS}
    if not all(pf.values()) or not all(all(v.values()) for v in pd.values()):
        return table(["Controller"] + [TN[t] for t in TASKS], rows, "Mean J on each task's 100 pairs at seeds 0 "
                     "and 1, and below it the paired difference from HOM.", num_from=1)
    below = (lambda k: ["the " + TN[t].lower() for t in TASKS if pd[k][t]["hi"] < 0])
    above = (lambda k: ["the " + TN[t].lower() for t in TASKS if pd[k][t]["lo"] > 0])
    within = (lambda k: [t for t in TASKS if pd[k][t]["lo"] <= 0 <= pd[k][t]["hi"]])
    lst = (lambda xs: "no task" if not xs else al(xs))

    def says(k, name):
        out = f"{name} scored below HOM on {lst(below(k))}"
        if above(k):
            out += f" and above it on {lst(above(k))}"
        w = within(k)
        if w:
            out += (f". On {lst(['the ' + TN[t].lower() for t in w])} its interval included zero ("
                    + al([f"{pd[k][t]['d']:+.3f}" for t in w]) + ")")
        return out + "."
    rts = [gpu_ratio(R, MP.tstem(m, "twist", "_rank", cfg[c])) for _k, m, c in RT_RUNS]
    t_ = fill(r"""<p>HOM's mean J over each task's 100 pairs was <<h>> on the twist, the roll, the pin-push and the press
(Table <<tn>>). Full DIAL at 4 &#215; 64 samples scored higher on every task, by <<df>>. On the twist one of its
rollouts took <<rf>> times HOM's GPU time in the search's batches. The three real-time planners took <<rr>> times
HOM's time. By the 95 % interval of the paired difference, <<b1>> <<b2>> <<b3>></p>""",
              tn=P.TAB[0] + 1, h=al([f"{MP.mean_j(hom_rank(R, t)):.3f}" for t in TASKS]),
              df=al([f"{pf[t]['d']:.3f}" for t in TASKS]),
              rf=f"{gpu_ratio(R, MP.tstem('anneal', 'twist', '_rank', CFG_FULL2)):.0f}",
              rr=f"{min(rts):.0f} to {max(rts):.0f}", b1=says("c1", "real-time DIAL with the twist's settings"),
              b2=says("rt30", "DIAL with a 0.3 s horizon"), b3=says("grad", "The gradient planner"))
    return t_ + table(["Controller"] + [TN[t] for t in TASKS], rows,
                 "Mean J on each task's 100 pairs at seeds 0 and 1, and below it the paired difference from HOM with its "
                 "95 % interval and the number of pairs where the planner was higher. Full numbers and the settings "
                 f'swept on the tuning pairs are on the <a href="{MPC_PAGE}">sampling MPC page</a>.', num_from=1) \
        + MP.task_films(R)


STAB_RUNS = [("hom", None, "HOM"), ("anneal", CFG_PICK, "DIAL 2 &#215; 32, 0.1 s ahead"),
             ("anneal", MP.CFG_RT30, "DIAL 2 &#215; 16, 0.3 s ahead"), ("grad", CFG_PICK, "Gradient 2 &#215; 32"),
             ("anneal", CFG_FULL2, "DIAL 4 &#215; 64")]


def seed_rho(rows):
    """Spearman's rho between the pairs' J at seed 0 and at seed 1."""
    a, b = MP.pair_j(rows, 0), MP.pair_j(rows, 1)
    ks = sorted(set(a) & set(b))
    return spearman([a[k] for k in ks], [b[k] for k in ks]) if len(ks) > 5 else None


def fresh_numbers(R, task):
    """HOM on the 100 pairs at seeds 0-1 (the grasps were chosen at seeds 0-2) and at seeds 10-11: share of runs that
    dropped the object, mean J."""
    out = {}
    for nm, rows in (("s01", hom_rank(R, task)), ("s1011", rows_of(R, f"hom_{task}_rank_s1011"))):
        if len(rows) < 150:
            return None
        out[nm] = {"drop": 1.0 - float(np.mean([bool(r.get("held")) for r in rows])), "J": MP.mean_j(rows)}
    return out


def sec_stability(R, GS):
    FR = {t: fresh_numbers(R, t) for t in TASKS}
    sr = {lab: [seed_rho(rows_of(R, MP.tstem(m, t, "_rank", c))) for t in TASKS] for m, c, lab in STAB_RUNS}
    txt = ""
    if all(FR.values()) and all(None not in v for v in sr.values()):
        pc = (lambda x: f"{100 * x:.0f} %")
        rtl = [lab for _m, _c, lab in STAB_RUNS[1:-1]]
        n_lower = sum(1 for t in range(4) for lab in rtl if sr[lab][t] < sr["HOM"][t])
        txt = fill(r"""<p>At each seed the object's start is shifted by up to 1 mm and turned by up to 2&#176; about the
vertical. The offsets are the same for every layout, and seed 0 is the nominal pose. Over the 100 pairs HOM's scores at seeds 0 and 1 were ordered
alike with $\rho$ = <<h>> on the twist, roll, pin-push and press (Table <<tn>>). Real-time DIAL's were ordered alike
with <<c1>>, and the gradient planner's with <<g>>. In <<nl>> of the 12 comparisons a real-time planner's $\rho$ was
below HOM's. Full DIAL's was <<f>>. On HOM's grids the layouts' scores at seeds 0 and 1 had $\rho$ <<gr>>. Lattice
neighbours 5 mm apart differed in J by <<sm>> times the difference between random layouts.</p>
<p>The grasps of the 100 pairs were chosen on HOM's grid at seeds 0-2. At seeds 10 and 11, two other placements, HOM
dropped the object in <<dn>> of the runs on the four tasks, against <<do>> at seeds 0 and 1. Most of the rise was on the
cylinders: on the press from <<po>> to <<pn>>, on the pin-push from <<ppo>> to <<ppn>>. HOM's mean J at the new seeds
was <<jn>>, against <<jo>>.</p>""",
                   tn=P.TAB[0] + 1, h=al([f2(x) for x in sr["HOM"]]),
                   c1=al([f2(x) for x in sr[STAB_RUNS[1][2]]]), g=al([f2(x) for x in sr[STAB_RUNS[3][2]]]),
                   nl=n_lower, f=al([f2(x) for x in sr[STAB_RUNS[-1][2]]]),
                   gr=al([f2(GS[t]["rho_seeds"]) for t in TASKS]),
                   sm=al([f2(GS[t]["neighbour"] / GS[t]["random"]) for t in TASKS]),
                   dn=pc(np.mean([FR[t]["s1011"]["drop"] for t in TASKS])),
                   do=pc(np.mean([FR[t]["s01"]["drop"] for t in TASKS])),
                   po=pc(FR["press"]["s01"]["drop"]), pn=pc(FR["press"]["s1011"]["drop"]),
                   ppo=pc(FR["pinpush"]["s01"]["drop"]), ppn=pc(FR["pinpush"]["s1011"]["drop"]),
                   jn=al([f"{FR[t]['s1011']['J']:.3f}" for t in TASKS]),
                   jo=al([f"{FR[t]['s01']['J']:.3f}" for t in TASKS]))
    rows = []
    for t in TASKS:
        g = GS[t]
        rows.append([TN[t], f2(g["rho_seeds"]), f2(g["neighbour"] / g["random"])]
                    + [f2(seed_rho(rows_of(R, MP.tstem(m, t, "_rank", c)))) if len(rows_of(R, MP.tstem(m, t, "_rank", c)))
                       >= 150 else "&#8212;" for m, c, _l in STAB_RUNS])
    t_ = table(["Task", "HOM's grid: &#961; seeds 0/1", "HOM's grid: smoothness ratio"]
               + [f"100 pairs, &#961; seeds 0/1: {lab}" for _m, _c, lab in STAB_RUNS], rows,
               "How well each controller's scores repeat between seeds 0 and 1: Spearman's &#961; between the pairs' J at "
               "the two seeds. HOM's grid: every reached layout with its best grasp by HOM's mean over seeds 0-2. 100 "
               "pairs: the ranking test's pairs of the task. Smoothness ratio: mean |J difference| between lattice "
               "neighbours over that between random pairs of layouts.", num_from=1)
    return txt + t_


def e_ref_of(rows):
    """The task's RMS commanded rotation (deg), the same for every world: e / (1 - J / c) of a held run with J > 0."""
    v = [r["track_err_rms_deg"] / (1.0 - r["J"] / r["contact_all_frac"]) for r in rows
         if r.get("held") and r["J"] > 0 and r.get("contact_all_frac") and r["J"] < r["contact_all_frac"]]
    return float(np.median(v)) if v else None


def metric_vectors(rows, keys, e_ref):
    """Per pair (seeds averaged): J, the tracking factor max(0, 1 - e/e_ref) and the three-pad share (0 for a drop)."""
    out = {"J": {}, "track": {}, "c": {}}
    for r in rows:
        k = key(r)[:3]
        tf = max(0.0, 1.0 - r["track_err_rms_deg"] / e_ref) if r.get("held") and r.get("track_err_rms_deg") is not None \
            else 0.0
        for m, v in (("J", r["J"]), ("track", tf), ("c", r.get("contact_all_frac", 0.0) if r.get("held") else 0.0)):
            out[m].setdefault(k, []).append(v)
    return {m: np.array([np.mean(d[k]) if k in d else np.nan for k in keys]) for m, d in out.items()}


def metric_agreement(R, task):
    """HOM's and full DIAL's order of the 100 pairs under J and under each of its factors, and within each controller
    the agreement of J's order with each factor's."""
    st, _lab = MP.rank_ref(R, task)
    h = hom_rank(R, task)
    if not st or not h:
        return None
    d = rows_of(R, st)
    keys = sorted({key(r)[:3] for r in h} & {key(r)[:3] for r in d})
    e_ref = e_ref_of(h + d)
    H, Dm = metric_vectors(h, keys, e_ref), metric_vectors(d, keys, e_ref)
    return {"n": len(keys), "e_ref": e_ref,
            "between": {m: spearman(H[m], Dm[m]) for m in ("J", "track", "c")},
            "hom_within": {m: spearman(H["J"], H[m]) for m in ("track", "c")},
            "dial_within": {m: spearman(Dm["J"], Dm[m]) for m in ("track", "c")},
            "means": {"hom": {m: float(np.nanmean(H[m])) for m in H}, "dial": {m: float(np.nanmean(Dm[m])) for m in Dm}}}


def sec_metrics(R):
    rows, A = [], {}
    for t in TASKS:
        a = metric_agreement(R, t)
        if not a:
            continue
        A[t] = a
        rows.append([TN[t], str(a["n"]), f2(a["between"]["J"]), f2(a["between"]["track"]), f2(a["between"]["c"]),
                     f"{a['means']['hom']['track']:.2f} / {a['means']['dial']['track']:.2f}",
                     f"{a['means']['hom']['c']:.2f} / {a['means']['dial']['c']:.2f}"])
    if not rows:
        return ""
    txt = ""
    if len(A) == len(TASKS):
        lo_c = [t for t in TASKS if A[t]["means"]["dial"]["c"] <= A[t]["means"]["hom"]["c"]]
        hi_c = [TN[t].lower() for t in TASKS if t not in lo_c]
        lst = (lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1])
        txt = fill(r"""<p>J is the product of the tracking factor and the three-pad share. Under HOM and full DIAL the pairs'
three-pad shares were ordered alike, with $\rho$ = <<c>> on the twist, roll, pin-push and press (Table <<tn>>). Their
tracking factors were ordered alike with <<tr>>. Full DIAL's mean tracking factor was <<trd>>, against HOM's <<trh>>.
Its three-pad share was higher on the <<hc>><<lc>>.</p>""",
                   tn=P.TAB[0] + 1, c=al([f2(A[t]["between"]["c"]) for t in TASKS]),
                   tr=al([f2(A[t]["between"]["track"]) for t in TASKS]),
                   trd=al([f"{A[t]['means']['dial']['track']:.2f}" for t in TASKS]),
                   trh=al([f"{A[t]['means']['hom']['track']:.2f}" for t in TASKS]), hc=lst(hi_c) if hi_c else "",
                   lc=("" if not lo_c else "; on the " + lst([TN[t].lower() for t in lo_c]) + " it was "
                       + ", ".join(f"{A[t]['means']['dial']['c']:.2f}" for t in lo_c) + " against HOM's "
                       + ", ".join(f"{A[t]['means']['hom']['c']:.2f}" for t in lo_c)))
    return txt + table(["Task", "Pairs", "&#961; HOM/DIAL, J", "&#961;, tracking factor", "&#961;, three-pad share",
                  "Mean tracking factor, HOM / DIAL", "Mean three-pad share, HOM / DIAL"], rows,
                 "The order of each task's 100 pairs under HOM and under full DIAL, by J and by each of its two factors, "
                 "seeds averaged; and each factor's mean under the two controllers.", num_from=1)


def screen_numbers(R, task):
    """HOM's grid as a screen for full DIAL on a task's 100 pairs: the best full-DIAL J among the first k layouts in the
    grid's order (the pair file's J_by, HOM's mean over seeds 0-2) and the expected best among k layouts drawn at random
    from the rest of the grid (the random half, 20,000 orders); each half's mean J under every controller with the
    bootstrap interval of the difference; the layouts of HOM's and the planner's ten best."""
    st, _lab = MP.rank_ref(R, task)
    if not st:
        return None
    prs = json.loads((MP.ROWS / f"mpc_pairs_{task}_rank.json").read_text())["pairs"]
    D, F = MP.pair_j(rows_of(R, st)), MP.pair_j(rows_of(R, f"hom_{task}_rank_s1011"))
    keys = [(p["x"], p["y"], p["grasp"]) for p in prs]
    if any(k not in D for k in keys):
        return None
    best = np.array([p["by"] == "hom_best" for p in prs])
    jby, d = np.array([p["J_by"] for p in prs]), np.array([D[k] for k in keys])
    order = np.argsort(-jby, kind="stable")
    scr = np.maximum.accumulate(d[order][:int(best.sum())])
    rv, rng = d[~best], np.random.default_rng(0)
    perm = np.array([np.maximum.accumulate(rv[rng.permutation(len(rv))]) for _ in range(20000)])
    rnd, lo, hi = perm.mean(0), np.quantile(perm, 0.1, 0), np.quantile(perm, 0.9, 0)
    halves = {}
    for m, c, lab in STAB_RUNS[:1] + [("hom", None, "HOM, seeds 10 and 11")] + STAB_RUNS[1:]:
        rr = rows_of(R, MP.tstem(m, task, "_rank", c) + ("_s1011" if lab.endswith("11") else ""))
        if len(rr) < 150:
            continue
        J_ = MP.pair_j(rr)
        v = np.array([J_.get(k, np.nan) for k in keys])
        a_, b_ = v[best], v[~best]
        bs = [rng.choice(a_, len(a_)).mean() - rng.choice(b_, len(b_)).mean() for _ in range(4000)]
        halves[lab] = (float(a_.mean()), float(b_.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    top_h, top_d = [int(i) for i in order[:10]], [int(i) for i in np.argsort(-d, kind="stable")[:10]]
    fresh_d = np.array([F.get(keys[i], np.nan) for i in top_d])
    need = {k0: next((k for k in range(1, len(rnd) + 1) if rnd[k - 1] >= scr[k0 - 1]), None) for k0 in (1, 3, 10)}
    return {"keys": keys, "best": best.tolist(), "d": d.tolist(), "scr": scr.tolist(), "rnd": rnd.tolist(),
            "lo": lo.tolist(), "hi": hi.tolist(), "d_max": float(d.max()), "halves": halves, "top_h": top_h,
            "top_d": top_d, "shared": len(set(top_h) & set(top_d)), "need": need,
            "top_d_from_best": int(best[top_d].sum()), "fresh_d_mean": float(np.nanmean(fresh_d)),
            "fresh_d_zero": int(np.sum(fresh_d == 0))}


def fig_screen(N, out):
    """Best full-DIAL J found after k planner runs, in HOM's grid order and on random layouts of the grid."""
    plt = MP.plt_setup()
    tasks = [t for t in TASKS if N.get(t)]
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 8.6), sharey=True)
    for a, t in zip(ax.flat, tasks):
        n = N[t]
        k = np.arange(1, len(n["scr"]) + 1)
        a.fill_between(np.arange(1, len(n["rnd"]) + 1), n["lo"], n["hi"], color="#d1d5db", lw=0,
                       label="random layouts, 10-90 %")
        a.plot(np.arange(1, len(n["rnd"]) + 1), n["rnd"], color="#6b7280", lw=2.0, label="random layouts, mean")
        a.step(k, n["scr"], where="post", color="#2563eb", lw=2.4, label="HOM's order")
        a.axhline(n["d_max"], color="#d97706", lw=1.2, ls=(0, (4, 3)), label="planner's best of the 100")
        a.set_xscale("log")
        a.set_xlim(1, len(k))
        a.set_ylim(0, 1.0)
        a.set_title(TN[t], loc="left")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    for a in ax[1]:
        a.set_xlabel("Layouts run under full DIAL")
    for a in ax[:, 0]:
        a.set_ylabel("Best J under full DIAL so far")
    h_, l_ = ax[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, loc="lower center", ncol=2, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_layouts(N, out):
    """The 100 pairs in the layout plane; HOM's ten best by its grid and full DIAL's ten best marked."""
    plt = MP.plt_setup()
    tasks = [t for t in TASKS if N.get(t)]
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 9.0), sharex=True, sharey=True)
    for a, t in zip(ax.flat, tasks):
        n = N[t]
        xy = np.array([k[:2] for k in n["keys"]])
        a.scatter(xy[:, 0], xy[:, 1], s=14, color="#d1d5db", lw=0, label="other pairs")
        a.scatter(xy[n["top_h"], 0], xy[n["top_h"], 1], s=95, facecolors="none", edgecolors="#2563eb", lw=1.8,
                  label="HOM's ten best")
        a.scatter(xy[n["top_d"], 0], xy[n["top_d"], 1], s=40, marker="^", color="#d97706", lw=0,
                  label="full DIAL's ten best")
        a.set_title(TN[t], loc="left")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    for a in ax[1]:
        a.set_xlabel("x_sep (mm)")
    for a in ax[:, 0]:
        a.set_ylabel("y_sep (mm)")
    h_, l_ = ax[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, loc="lower center", ncol=3, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def sec_screen(R):
    N = {t: screen_numbers(R, t) for t in TASKS}
    if not all(N.values()):
        return ""
    GS = grid_stability()
    full = "DIAL 4 &#215; 64"
    hv = {t: N[t]["halves"][full] for t in TASKS}
    lay = {t: GS[t]["layouts"] for t in TASKS}
    scr = {t: N[t]["scr"] for t in TASKS}
    rnd = {t: N[t]["rnd"] for t in TASKS}
    med = {(t, w): np.median(np.array([N[t]["keys"][i][:2] for i in N[t][w]]), 0) for t in TASKS
           for w in ("top_h", "top_d")}
    yh = {t: [N[t]["keys"][i][1] for i in N[t]["top_h"]] for t in TASKS}
    HR = {t: MP.pair_j(hom_rank(R, t)) for t in TASKS}
    rho_grid = {t: spearman([p_["J_by"] for p_ in json.loads((MP.ROWS / f"mpc_pairs_{t}_rank.json").read_text())["pairs"]],
                            [HR[t][k] for k in N[t]["keys"]]) for t in TASKS}
    mm = (lambda v: f"{v:g}")
    wide_pp = sum(1 for y in yh["pinpush"] if y >= 150)
    zero_pp = [N["pinpush"]["d"][i] for i in N["pinpush"]["top_h"] if N["pinpush"]["keys"][i][1] >= 150]
    t_ = fill(r"""<p>HOM's grids were run on <<l_t>> layouts for the twist and the roll, <<l_pp>> for the pin-push and <<l_p>>
for the press. Each layout was run with every grasp of its set at seeds 0-2. HOM's 50 best layouts scored higher
under full DIAL than the 50 drawn at random from the rest, on every task (Table <<tn>>). The difference was <<g_t>> on the
twist, <<g_r>> on the roll, <<g_pp>> on the pin-push and <<g_p>> on the press.</p>
<p>A search with the planner alone would run it on layouts of the grid in no particular order. Figure <<fn>> compares
that with running it in the order of HOM's grid. That order and the order of HOM's own runs at seeds 0 and 1 agreed with
$\rho$ = <<rg>>. On the twist HOM's second layout scored <<s2_t>> under full DIAL, and 50 random
layouts reached <<r50_t>> on average. On the roll HOM's first layout scored <<s1_r>>, a score that random layouts reached after
<<n1_r>> runs on average. On the pin-push HOM's first layout scored <<s1_pp>> under the planner, and its tenth reached
<<s10_pp>>, the planner's best of the 100. On the press HOM's order was ahead of random layouts for the first
<<k_lead_p>> runs. After ten runs it had reached <<s10_p>> against <<r10_p>> for random layouts. After 20 it had reached
<<s20_p>> against <<r20_p>>.</p>
<p>Figure <<fn2>> shows where the two controllers' ten best layouts lie. On the twist the medians of the two sets were
<<dmed_t>> mm apart or less, and <<sh_t>> layouts are shared. On the roll, the pin-push and the press the planner's ten
best lie at larger thumb distances. Their median $x_{\mathrm{sep}}$ was <<xd>> mm, against HOM's <<xh>> mm. On the
press they also lie at wider pair spacings, a median $y_{\mathrm{sep}}$ of <<yd_p>> mm against HOM's <<yh_p>> mm. Under
HOM at seeds 10 and 11 the planner's ten best press layouts scored <<fr_p>> on average, and <<fz_p>> of them scored 0.
On the pin-push <<wide_pp>> of HOM's ten best lie at $y_{\mathrm{sep}}$ of 150 mm or more, where the planner scored
<<zpp>>.</p>""",
              l_t=lay["twist"], l_pp=lay["pinpush"], l_p=lay["press"], tn=P.TAB[0] + 1, fn=P.FIG[0] + 1,
              fn2=P.FIG[0] + 2,
              g_t=f"{hv['twist'][0] - hv['twist'][1]:.3f}", g_r=f"{hv['roll'][0] - hv['roll'][1]:.3f}",
              g_pp=f"{hv['pinpush'][0] - hv['pinpush'][1]:.3f}", g_p=f"{hv['press'][0] - hv['press'][1]:.3f}",
              s2_t=f"{scr['twist'][1]:.3f}", r50_t=f"{rnd['twist'][-1]:.3f}", s1_r=f"{scr['roll'][0]:.3f}",
              n1_r=N["roll"]["need"][1] or "more than 50", s1_pp=f"{scr['pinpush'][0]:.3f}",
              s10_pp=f"{scr['pinpush'][9]:.3f}",
              k_lead_p=next((k for k in range(1, 51) if scr["press"][k - 1] < rnd["press"][k - 1]), 51) - 1,
              s10_p=f"{scr['press'][9]:.3f}", r10_p=f"{rnd['press'][9]:.3f}", s20_p=f"{scr['press'][19]:.3f}",
              r20_p=f"{rnd['press'][19]:.3f}", sh_t=N["twist"]["shared"],
              rg=al([f2(rho_grid[t]) for t in TASKS]),
              dmed_t=mm(float(np.abs(med[("twist", "top_h")] - med[("twist", "top_d")]).max())),
              xd=al([mm(med[(t, "top_d")][0]) for t in ("roll", "pinpush", "press")]),
              xh=al([mm(med[(t, "top_h")][0]) for t in ("roll", "pinpush", "press")]),
              yh_p=mm(med[("press", "top_h")][1]), yd_p=mm(med[("press", "top_d")][1]),
              fr_p=f"{N['press']['fresh_d_mean']:.3f}",
              fz_p=N["press"]["fresh_d_zero"], wide_pp=wide_pp,
              zpp=(f"{min(zero_pp):.3f} to {max(zero_pp):.3f}" if zero_pp else "&#8212;"))
    rows = []
    for t in TASKS:
        for lab, (a_, b_, l_, h_) in N[t]["halves"].items():
            rows.append([TN[t] if lab == "HOM" else "", lab.replace("seeds 10 and 11", "at seeds 10 and 11"), f"{a_:.3f}",
                         f"{b_:.3f}", f"{a_ - b_:+.3f} ({l_:+.3f} to {h_:+.3f})"])
    t_ += table(["Task", "Controller", "HOM's 50 best layouts", "50 random layouts", "Difference (95 % interval)"], rows,
                "Mean J of each half of the 100 pairs under each controller, seeds 0 and 1 unless marked. HOM's 50 best: the layouts "
                "of HOM's grid with the best J over seeds 0-2, each with its best grasp. 50 random: drawn from the rest "
                "of the grid, each with HOM's best grasp. Interval: 4,000 bootstrap resamples of the pairs.", num_from=2)
    t_ += P.img(fig_screen(N, MEDIA / "20261115-hom_screen_curves.png"),
                "Best J under full DIAL after running the planner on k layouts. Blue: the layouts in the order of HOM's "
                "grid (each with HOM's best grasp). Grey: k layouts drawn at random from the rest of the grid, the mean "
                "and the 10-90 % band of 20,000 orders. Dashed: the planner's best of the 100 pairs. Seeds 0 and 1.")
    t_ += P.img(fig_layouts(N, MEDIA / "20261115-hom_dial_top_layouts.png"),
                "The 100 pairs of each task in the plane of the layout coordinates: thumb-to-pair distance x_sep and "
                "pair spacing y_sep. Circles: HOM's ten best by its grid; triangles: full DIAL's ten best, seeds 0 and "
                "1. Pairs that share a layout overlap.")
    return t_


def sec_variants(R):
    rows, any_ = [], False
    for tag, lab, tasks in VARIANTS:
        row = [lab]
        for t in TASKS:
            v = rows_of(R, f"hom_{t}_rank20{tag}") if t in tasks else []
            h = hom_rank(R, t)
            if v and h:
                any_ = True
                p_ = paired(v, h)
                row.append(f"{MP.mean_j(v):.3f}<br>{fd(p_)}")
            else:
                row.append("&#8212;")
        rows.append(row)
    if not any_:
        return ""
    base = ["HOM, defaults"] + [f"{MP.mean_j(sub20(R, t, hom_rank(R, t))):.3f}" if hom_rank(R, t) else "&#8212;"
                                for t in TASKS]
    return table(["HOM variant"] + [TN[t] for t in TASKS], [base] + rows,
                 "HOM's variants on each task's 20 pairs at seeds 0 and 1: mean J, and below it the paired difference "
                 "from HOM's defaults on the same pairs and seeds.", num_from=1)


def search_fusion(R):
    """HOM's grid with a planner's re-scoring of its ten best layouts per task: the best full-DIAL J reached, and the
    re-scoring's GPU time against HOM's in the search workload of the talk's cost chart (three tasks, 1,423 layouts)."""
    N = {t: screen_numbers(R, t) for t in TASKS}
    if not all(N.values()):
        return ""
    cs = json.loads(FIGJSON.read_text())["cost_search_track"]
    hom_s = cs["rows_s"]["MuJoCo-Warp, 1 mm sphere pads,"]
    st = MP.tstem("anneal", "twist", "_rank", CFG_FULL2)
    per = float(np.mean([r["chunk_wall_s"] / r["chunk_worlds"] for r in rows_of(R, st) if r.get("chunk_wall_s")]))
    extra = 3 * 10 * 2 * per
    gap = {t: N[t]["d_max"] - N[t]["scr"][9] for t in TASKS}
    return fill(r"""<p>The methods can also be fused at the level of the search. HOM scores every layout of the grid,
and a planner re-scores HOM's ten best layouts of each task. On the 100 pairs of Section <<sn>>, the planner's best J
on HOM's ten was compared with its best on all 100. It fell short by <<gp>> on the twist, roll, pin-push and press. On the
search of the talk's cost chart, 1,423 layouts of the twist, the roll and the press, HOM takes <<hh>> hours of GPU
time. Re-scoring ten layouts per task at seeds 0 and 1 under full DIAL takes <<em>> minutes more, <<pc>> % of
HOM's time.</p>""", sn="SCREENSEC", gp=al([f"{gap[t]:.3f}" for t in TASKS]), hh=f"{hom_s / 3600:.1f}",
                em=f"{extra / 60:.0f}", pc=f"{100 * extra / hom_s:.1f}")


def sec_fusion(R):
    rows, any_ = [], False
    txt = search_fusion(R)
    for m, lab in FUSION:
        row = [lab]
        for t in TASKS:
            v = rows_of(R, MP.tstem(m, t, "_rank20", CFG_PICK))
            h = hom_rank(R, t)
            if v and h:
                any_ = True
                p_ = paired(v, h)
                row.append(f"{MP.mean_j(v):.3f}<br>{fd(p_)}<br>{np.mean([r['replan_s_mean'] for r in v]):.2f} s")
            else:
                row.append("&#8212;")
        rows.append(row)
    if not any_:
        return txt
    return txt + table(["Fused planner"] + [TN[t] for t in TASKS], rows,
                 "Planners fused with HOM on each task's 20 pairs at seeds 0 and 1: mean J; the paired difference from "
                 "HOM; the replanning time for 16 hands planned together.", num_from=1)


def sec_trends(PK, figs):
    TRN = trend_numbers(PK)
    HS, DS = MP.stats(load("hom_twist")), MP.stats(load("anneal_twist"))
    v = (lambda t, k: TRN[(t, k)])
    val = {t: PK["tasks"][t].get("val", {}) for t in TASKS}
    t_ = fill(r"""<p>On 11 November HOM's commanded forces and the weights of Eq. (2)'s rows were refined by CEM for each of the 30
layouts with the best J on the twist, the roll and the press. The grasp was held fixed (level hom of the
<a href="<<rp>>">score-levels page</a>). One set for all layouts of each task was searched as well (level shared).
Each layout's pick is the candidate with the best mean J over seeds 0-2.</p>
<p>The picked forces were below HOM's defaults on every task (Figure <<fn>>a). The median pad force of the twist was
<<ft>> N (quartiles <<ft1>>&#8211;<<ft3>>) against 2.5 N. The roll's was <<fr>> N. The press's pinch force was
<<fp>> N against 3.0 N, and its pusher force stayed near the default, at <<fq>> N against 0.5 N. The shared picks
were close to these medians: <<st>> N on the twist and <<sp>> N for the press's pinch. On the twist's 20 benchmark
pairs, HOM's servos were at 95 % of their 0.18 N m limit or more in <<sat>> % of the joint-periods. DIAL at
4 &#215; 64 samples, which scored higher, gripped with <<gd>> N against HOM's <<gh>> N (sampling MPC page).</p>
<p>The weights changed less. On the twist the spin and slide rows were relaxed to medians of <<ws>> and <<wl>>
(defaults 1), and rolling stayed free (<<wr>>). The press's pusher roll was more than halved, to <<wpr>> against 1
(Figure <<fn>>c). Only the roll's force varied with the layout: $\rho$ = <<rx>> with $x_{\mathrm{sep}}$ and <<ry>> with
$y_{\mathrm{sep}}$ (Figure <<fn>>b). On the twist and the press the largest $|\rho|$ of a force or weight with
either coordinate was <<rmt>> and <<rmp>>.</p>
<p>At seeds 20-24 the median J of the 30 layouts rose by <<gt>> (twist), <<gr>> (roll) and <<gp>> (press) with
the per-layout picks. With the shared set it rose by <<st_g>>, <<sr_g>> and <<sp_g>>, and by <<spp_g>> on the
pin-push. Under the refined J the 30 layouts kept HOM's order with $\rho$ = <<rt>>, <<rr>> and <<rpp>>.</p>""",
              rp=REFINE_PAGE, fn=P.FIG[0] + 1,
              ft=f"{v('twist', 'f_tripod')['med']:.2f}", ft1=f"{v('twist', 'f_tripod')['q1']:.2f}",
              ft3=f"{v('twist', 'f_tripod')['q3']:.2f}", fr=f"{v('roll', 'f_tripod')['med']:.2f}",
              fp=f"{v('press', 'f_pinch')['med']:.2f}", fq=f"{v('press', 'f_press')['med']:.2f}",
              st=f"{PK['tasks']['twist']['shared']['f_tripod']:.2f}", sp=f"{PK['tasks']['press']['shared']['f_pinch']:.2f}",
              ws=f"{v('twist', 'w_spin_tripod')['med']:.2f}", wl=f"{v('twist', 'w_slide_tripod')['med']:.2f}",
              wr=f"{v('twist', 'w_roll_tripod')['med']:.2f}", wpr=f"{v('press', 'w_roll_push')['med']:.2f}",
              rx=f"{v('roll', 'f_tripod')['rx']:+.2f}", ry=f"{v('roll', 'f_tripod')['ry']:+.2f}",
              gt=sg_(val["twist"]["hom_val"]["gain"]), gr=sg_(val["roll"]["hom_val"]["gain"]),
              gp=sg_(val["press"]["hom_val"]["gain"]), st_g=sg_(val["twist"]["shared_val"]["gain"]),
              sr_g=sg_(val["roll"]["shared_val"]["gain"]), sp_g=sg_(val["press"]["shared_val"]["gain"]),
              spp_g=sg_(val["pinpush"]["shared_val"]["gain"]), rt=f2(val["twist"]["hom_val"]["rho_s1"]),
              rr=f2(val["roll"]["hom_val"]["rho_s1"]), rpp=f2(val["press"]["hom_val"]["rho_s1"]),
              rmt=f"{max(max(abs(x['rx']), abs(x['ry'])) for (t, _k), x in TRN.items() if t == 'twist'):.2f}",
              rmp=f"{max(max(abs(x['rx']), abs(x['ry'])) for (t, _k), x in TRN.items() if t == 'press'):.2f}",
              sat=f"{100 * HS['sat_frac']:.0f}", gh=f"{HS['grip_mean_N']:.1f}", gd=f"{DS['grip_mean_N']:.1f}")
    t_ += P.img(figs["trends"], "HOM's forces and contact-mode weights picked by CEM for the 30 layouts with the best J "
                "per task, 11 November, under J. (a) Each layout's picked force over HOM's default (blue dots, the "
                "median as a bar) and the shared pick (orange diamond). (b) The roll's picked pad force against "
                "x_sep, coloured by y_sep; dashed: the default 2.5 N. (c) The picked weights of Eq. (2)'s rows (blue, "
                "the median as a bar) and HOM's defaults (red).")
    return t_


def sg_(x):
    return f"{x:+.3f}"


def sec_future():
    items = [
        "Full DIAL on every layout of the press's grid with HOM's best grasp, at seeds 0 and 1: 369 layouts at about "
        "7.8 s of GPU per rollout, 1.6 hours. If more than five of the planner's ten best layouts lie outside HOM's 50 "
        "best, the planner's best press layouts are missed by HOM's grid. The press would then need a planner in the "
        "search.",
        "Grasps chosen at five placements (seeds 0-4) in place of three (<code>hom_gpu.py run --seeds 0 1 2 3 4</code> "
        "on the grid). HOM would then run at seeds 10 and 11 on the 100 pairs of the pin-push and the press. A drop rate "
        "above 20 % there would show that the cylinder grasps fail at new placements however they are chosen.",
        "The planner's landscape on a patch of the lattice. The 49 layouts within 15 mm of the twist's best would be run "
        "under full DIAL at seeds 0 and 1, for a smoothness ratio to set against HOM's 0.39. A ratio above 0.6 would "
        "mean that the planner's scores are less ordered in space than HOM's.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def sec_files():
    items = ["<code>scripts/hom/inner_loop_mpc.py</code>: HOM, its variants (<code>--preview</code>, "
             "<code>--prm-shared</code>, <code>--hom-param</code>), the planners, the fused planners (<code>res_*</code>, "
             "<code>hs_*</code>) and the gradient planner (<code>grad</code>) on any task's pair set",
             "<code>scripts/hom/uhas_sr2.py</code>: <code>UBatch</code>, <code>PREVIEW_S</code>, the residual "
             "<code>q_add</code>",
             "<code>scripts/hom/hom_inner_loop_page.py</code>: this page",
             f"<code>{(DATA / '20261111-hom_picks_track.json').relative_to(ROOT)}</code>, "
             f"<code>{(DATA / '20261114-hom_grid_stability.json').relative_to(ROOT)}</code>: the reduced data",
             "<code>docs/experiments/20261112-inner_loop_benchmark/rows/mpc_*_{twist,roll,pinpush,press}_*.jsonl</code>: "
             "one row per run"]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


REFS = ["[1] Xue, Pan, Yi, Qu and Shi, Full-order sampling-based MPC for torque-level locomotion control via diffusion-"
        "style annealing (DIAL-MPC), ICRA 2025.",
        "[2] Howell, Gileadi, Tunyasuvunakool, Zakka, Erez and Tassa, Predictive sampling: real-time behaviour "
        "synthesis with MuJoCo, arXiv 2212.00541, 2022."]


def lede(R, PK, GS):
    """The outcome with its numbers, from whatever runs exist; (lede html, docs-summary)."""
    js = {}
    for name, m, c in (("hom", "hom", None), ("c1", "anneal", CFG_PICK), ("rt30", "anneal", MP.CFG_RT30),
                       ("grad", "grad", CFG_PICK), ("full", "anneal", CFG_FULL2)):
        js[name] = {t: MP.mean_j(rows_of(R, MP.tstem(m, t, "_rank", c))) if len(rows_of(R, MP.tstem(m, t, "_rank", c)))
                    >= 150 else None for t in TASKS}
    mean4 = (lambda k: float(np.mean(list(js[k].values()))) if None not in js[k].values() else None)
    parts = []
    if mean4("hom") is not None and mean4("full") is not None:
        rf = gpu_ratio(R, MP.tstem("anneal", "twist", "_rank", CFG_FULL2))
        parts.append(f"Over 100 layout-grasp pairs on each of the SR2 hand's four tasks, HOM's mean J was "
                     f"{mean4('hom'):.3f}. Full DIAL at 4 &#215; 64 samples scored {mean4('full'):.3f}, at {rf:.0f} "
                     "times HOM's GPU time per rollout.")
        rt = [mean4(k) for k in ("c1", "rt30", "grad") if mean4(k) is not None]
        rr = [gpu_ratio(R, MP.tstem(m, "twist", "_rank", c)) for m, c in (("anneal", CFG_PICK), ("anneal", MP.CFG_RT30),
                                                                          ("grad", CFG_PICK))]
        if len(rt) == 3:
            parts.append(f"In real time, at {min(rr):.0f} to {max(rr):.0f} times HOM's GPU time, DIAL and a "
                         f"finite-difference gradient planner scored {min(rt):.3f} to {max(rt):.3f}.")
    N = {t: screen_numbers(R, t) for t in TASKS}
    if all(N.values()):
        g = [N[t]["halves"]["DIAL 4 &#215; 64"][0] - N[t]["halves"]["DIAL 4 &#215; 64"][1] for t in TASKS]
        gap = {t: N[t]["d_max"] - N[t]["scr"][9] for t in TASKS}
        ok = [TN[t].lower() for t in TASKS if gap[t] <= 0.05]
        bad = [t for t in TASKS if gap[t] > 0.05]
        parts.append(f"Under full DIAL, HOM's 50 best layouts scored {min(g):.2f} to {max(g):.2f} higher than 50 random "
                     "ones.")
        parts.append(f"Run on HOM's ten best layouts, the planner came within {max(gap[t] for t in TASKS if t not in bad):.3f}"
                     f" of its best of the 100 on the {al(ok)}."
                     + ("" if not bad else " On the " + " and ".join(TN[t].lower() for t in bad) + " it fell short by "
                        + " and ".join(f"{gap[t]:.3f}" for t in bad) + ", and the two controllers' best layouts lie "
                        "apart."))
    best = None
    for tag, lab, tasks in VARIANTS:
        d = [paired(rows_of(R, f"hom_{t}_rank20{tag}"), hom_rank(R, t)) for t in tasks if hom_rank(R, t)]
        d = [x for x in d if x]
        if len(d) == len(tasks) and (best is None or np.mean([x["d"] for x in d]) > best[0]):
            best = (float(np.mean([x["d"] for x in d])), lab, d)
    if best:
        parts.append(f"Of HOM's variants, the best on average was {best[1][0].lower() + best[1][1:]}, at "
                     f"{best[0]:+.3f} in J over the defaults.")
    TRN = trend_numbers(PK)
    parts.append(f"The forces picked for HOM by CEM were below its defaults on every task: a median "
                 f"{TRN[('twist', 'f_tripod')]['med']:.2f} N against 2.5 N on the twist.")
    if not parts:
        parts.append("HOM against sampling MPC on the SR2 hand's four tasks; the planner runs are under way.")
    lede_t = "<p>" + " ".join(parts) + "</p>"
    return lede_t, re.sub(r"<[^>]+>", "", " ".join(parts[:2])).replace("&#215;", "x").replace("&#961;", "rho")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--picks", action="store_true", help="reduce the 11-11 rows and the grids to data/ again")
    a = ap.parse_args()
    P.FIG[0] = P.TAB[0] = P.FILM[0] = 0
    PK, GS = picks(a.picks), grid_stability(a.picks)
    MEDIA.mkdir(parents=True, exist_ok=True)
    figs = {"trends": fig_trends(PK, MEDIA / "20261114-hom_picks_trends.png")}
    R = {}
    T = MP.timing_rows()
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("setup", "Plant, tasks and controllers", sec_setup()),
                ("tasks", "HOM and the planners on four tasks", sec_tasks(R, T)),
                ("screen", "HOM's best layouts under full DIAL", sec_screen(R)),
                ("stability", "Stability of HOM's scores across seeds, layouts and controllers", sec_stability(R, GS)),
                ("metrics", "Where HOM loses J against the planner", sec_metrics(R)),
                ("variants", "Variants of HOM", sec_variants(R)),
                ("trends", "Forces and contact modes picked for HOM per layout", sec_trends(PK, figs)),
                ("fusion", "Planners fused with HOM", sec_fusion(R)),
                ("future", "Future work", sec_future()),
                ("refs", "References and files", '<ol class="refs">' + "".join(
                    f"<li>{r[r.index(']') + 2:]}</li>" for r in REFS) + "</ol>" + sec_files())]
    sections = [x for x in sections if x[2]]
    num = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sections = [(sid, hd, b.replace("SCREENSEC", str(num.get("screen", "&#8212;")))) for sid, hd, b in sections]
    lede_t, summary = lede(R, PK, GS)
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|s\b|min\b|h\b)", "\\1&nbsp;\\2", s_))
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{unit(b)}</section>'
                     for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("HOM against sampling MPC on the SR2 hand's four tasks, 2026-11-14, SCS0009 plant at 2.5 MPa, MuJoCo-Warp "
              f'3.6. The planner runs: <a href="{MPC_PAGE}">sampling MPC page</a>. Builder: '
              "<code>scripts/hom/hom_inner_loop_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "HOM as the inner loop", "H1": "Stability, variants and planner fusion of HOM as the inner "
                 "loop of the SR2 hand search", "EYEBROW": "Inner-loop controller, 2026-11-14", "LEDE": unit(lede_t),
                 "BYLINE": byline, "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-14")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    docs/experiments/20261112-inner_loop_benchmark/rows/): HOM, real-time and full annealed sampling on the four tasks' pair sets
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
# the variants in a sentence (lower case; cap() at a sentence's start)
LNAME = {"_pv50": "a 50 ms preview of the command", "_pv100": "a 100 ms preview of the command",
         "_prmsh": "forces and contact modes picked by CEM for the task",
         "_hpwdamp0.0001": "a damping of 0.0001", "_hpwdamp0.01": "a damping of 0.01", "_hpwdamp0.1": "a damping of 0.1",
         "_hpumax1": "a rate bound of 1 rad/s", "_hpumax4": "a rate bound of 4 rad/s",
         "_hplead0.15": "a reference lead of 0.15 rad", "_floop": "a force loop on the separation",
         "_att2": "orientation feedback on all three axes", "_att1ax": "orientation feedback at 1/s about the commanded axis",
         "_att4ax": "orientation feedback at 4/s about the commanded axis"}
NUMW = dict(enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen".split()))
FUSION = [("res_anneal", "Annealed-sampling residual on HOM's servo targets, 2 &#215; 32"),
          ("hs_anneal", "Annealed sampling in HOM's space, 2 &#215; 32")]
CFG_IL2, CFG_IL4 = MP.CFG_IL2, MP.CFG_IL4          # iLQR at 2 and 4 iterations over 0.3 s (inner_loop_ilqr.py)
CFG_D64 = (2, 30, 3, 10, 0.002, 64)                # DIAL 2 x 64 over 0.3 s, the sampling planner nearest iLQR's budget


def cap(s):
    return s[:1].upper() + s[1:]


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
        ["Unclipped tracking ratio", "$1 - e/e_{\\mathrm{ref}}$ of a run that held the object, without the floor at 0; "
         "a run that dropped it ranks below every held run", "below 1"],
        ["Power", "the probability that the 95 % interval of a paired difference excludes zero when the true difference "
         "is 0.05, at the spread of the measured differences", "0 to 1"],
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
                 "task phase, from HOM's handover at 2.0 s to the end of the schedule.", num_from=1)


def sec_setup():
    return fill(r"""<p>The plant is the one of the hand-design talk. Its servos follow the SCS0009 model of 0.72 N m/rad
with a 0.18 N m limit. The v4b tips are sphere pads at 2.5 MPa, and the scene friction is 1.0. MuJoCo-Warp 3.6 steps
it at 1 ms, and the controller runs every 10 ms.
The object is picked up and lifted under HOM. From 2.0 s the nine servo targets are set by the controller under test.
The four
tasks are HOM's demonstration tasks (<a href="<<mp>>">sampling MPC page</a>, Section on four tasks). The twist turns
the 40 mm sphere about the vertical by &#177;20&#176; and then ramps it at 30&#176;/s for 3 s. The roll rolls the same
sphere along the thumb's pad by &#177;20&#176; twice. The pin-push turns a 25 mm cylinder in a thumb-index pinch by
&#177;15&#176; with the middle finger; the press lifts it 20&#176; and lowers it again.</p>
<p>The planners are those of the sampling MPC page. Annealed sampling, the planner of DIAL-MPC [1], rolls out the
plant's own model on the GPU, a
batch of samples per hand. Its plan is the nine servo targets at two or three knots, linear between them. In the
gradient planner (method <code>grad</code>) the same rollouts are spent on finite differences. A least-squares gradient
is fitted to central differences at 0.02 rad, along each free knot coordinate and along random directions. Then as many
step sizes are tried in a line search. The scheme is MJPC's gradient planner [2]; MuJoCo-Warp 3.6 has no derivatives. In the fused planners
HOM runs in every rollout and in the plant. In the annealed-sampling residual on HOM's servo targets (<code>res_anneal</code>) a
joint-space plan from annealed sampling is added to HOM's targets. In annealed sampling in HOM's space (<code>hs_anneal</code>) rolling and
sliding rates from annealed sampling are added to the rows of Eq. (2) (sampling MPC page, Section on HOM's space). The zero plan of
both is HOM.</p>""",
                mp=MPC_PAGE)


def tasks_table(R, T):
    """Per task: HOM, the best real-time annealed sampling setting of the tuning pairs, the gradient planner and full annealed sampling on the
    100 pairs: mean J, the paired difference from HOM."""
    runs = [("anneal", CFG_PICK, "Annealed sampling 2 &#215; 32, 0.1 s ahead, real time (the twist's pick)"),
            ("anneal", MP.CFG_RT30, "Annealed sampling 2 &#215; 16, 0.3 s ahead every 0.2 s, real time"),
            ("grad", CFG_PICK, "Gradient descent 2 &#215; 32, real time"),
            ("anneal", CFG_FULL2, "Annealed sampling 4 &#215; 64, 6.3 times real time")]
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
(Table <<tn>>). Full annealed sampling at 4 &#215; 64 samples scored higher on every task, by <<df>>. On the twist one of its
rollouts took <<rf>> times HOM's GPU time in the search's batches. The three real-time planners took <<rr>> times
HOM's time. By the 95 % interval of the paired difference, <<b1>> <<b2>> <<b3>></p>""",
              tn=P.TAB[0] + 1, h=al([f"{MP.mean_j(hom_rank(R, t)):.3f}" for t in TASKS]),
              df=al([f"{pf[t]['d']:.3f}" for t in TASKS]),
              rf=f"{gpu_ratio(R, MP.tstem('anneal', 'twist', '_rank', CFG_FULL2)):.0f}",
              rr=f"{min(rts):.0f} to {max(rts):.0f}", b1=says("c1", "real-time annealed sampling with the twist's settings"),
              b2=says("rt30", "Annealed sampling with a 0.3 s horizon"), b3=says("grad", "The gradient planner"))
    return t_ + table(["Controller"] + [TN[t] for t in TASKS], rows,
                 "Mean J on each task's 100 pairs at seeds 0 and 1, and below it the paired difference from HOM with its "
                 "95 % interval and the number of pairs where the planner was higher. Full numbers and the settings "
                 f'swept on the tuning pairs are on the <a href="{MPC_PAGE}">sampling MPC page</a>.', num_from=1) \
        + MP.task_films(R)


STAB_RUNS = [("hom", None, "HOM"), ("anneal", CFG_PICK, "Annealed sampling 2 &#215; 32, 0.1 s ahead"),
             ("anneal", MP.CFG_RT30, "Annealed sampling 2 &#215; 16, 0.3 s ahead"), ("grad", CFG_PICK, "Gradient 2 &#215; 32"),
             ("anneal", CFG_FULL2, "Annealed sampling 4 &#215; 64")]


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


def five_numbers(R, task):
    """HOM at seeds 10 and 11 on the 100 pairs with each layout's grasp chosen at five placements, seeds 0-4 (rows
    hom_<task>_rank5, pairs mpc_pairs_<task>_rank5.json), against the grasps chosen at seeds 0-2 (hom_<task>_rank_s1011):
    drops and mean J on all pairs and on the layouts whose grasp changed; None while rows are missing."""
    pf = MP.ROWS / f"mpc_pairs_{task}_rank5.json"
    r5, r3 = rows_of(R, f"hom_{task}_rank5"), rows_of(R, f"hom_{task}_rank_s1011")
    if not pf.exists() or len(r5) < 150 or len(r3) < 150:
        return None
    prs = json.loads(pf.read_text())["pairs"]
    ch = {(p["x"], p["y"]) for p in prs if p["grasp"] != p["grasp_s3"]}

    def st(rows, S=None):
        rr = [r for r in rows if S is None or (r["x_sep"], r["y_sep"]) in S]
        return {"drop": sum(1 for r in rr if not r.get("held")), "n": len(rr), "J": MP.mean_j(rr)}
    return {"pairs": len(prs), "changed": len(ch), "five": st(r5), "three": st(r3), "five_ch": st(r5, ch),
            "three_ch": st(r3, ch)}


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
alike with $\rho$ = <<h>> on the twist, roll, pin-push and press (Table <<tn>>). Real-time annealed sampling's were ordered alike
with <<c1>>, and the gradient planner's with <<g>>. In <<nl>> of the 12 comparisons a real-time planner's $\rho$ was
below HOM's. Full annealed sampling's was <<f>>. On HOM's grids the layouts' scores at seeds 0 and 1 had $\rho$ <<gr>>. Lattice
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
    F5 = {t: five_numbers(R, t) for t in ("pinpush", "press")}
    if all(F5.values()):
        pc = (lambda d: f"{100 * d['drop'] / d['n']:.1f} %")
        txt += fill(r"""<p>For the pin-push and the press the grasps of the 100 pairs were chosen again at five placements,
seeds 0-4 (<code>inner_loop_mpc.py pairs --rule rank5</code>). The best grasp changed on <<c1>> of the pin-push's layouts
and <<c2>> of the press's. At seeds 10 and 11 HOM then dropped the cylinder in <<d1>> and <<d2>> of the runs, against
<<o1>> and <<o2>> with the grasps chosen at three placements. On the layouts whose grasp changed it dropped it in
<<x1>> of <<n1>> and <<x2>> of <<n2>> runs, against <<y1>> and <<y2>> before.<<only>> HOM's mean J rose from <<j1>> to
<<k1>> on the pin-push and from <<j2>> to <<k2>> on the press.</p>""",
                    only=(" Every drop with the new grasps was on a layout whose grasp stayed the same." if
                          F5["pinpush"]["five_ch"]["drop"] == 0 and F5["press"]["five_ch"]["drop"] == 0 else ""),
                    c1=F5["pinpush"]["changed"], c2=F5["press"]["changed"], d1=pc(F5["pinpush"]["five"]),
                    d2=pc(F5["press"]["five"]), o1=pc(F5["pinpush"]["three"]), o2=pc(F5["press"]["three"]),
                    x1=F5["pinpush"]["five_ch"]["drop"] or "none",
                    n1=F5["pinpush"]["five_ch"]["n"],
                    x2=F5["press"]["five_ch"]["drop"] or "none",
                    n2=F5["press"]["five_ch"]["n"], y1=F5["pinpush"]["three_ch"]["drop"],
                    y2=F5["press"]["three_ch"]["drop"], j1=f"{F5['pinpush']['three']['J']:.3f}",
                    k1=f"{F5['pinpush']['five']['J']:.3f}", j2=f"{F5['press']['three']['J']:.3f}",
                    k2=f"{F5['press']['five']['J']:.3f}")
    C = {t: MP.confirm_numbers(t) for t in TASKS}
    if all(C.values()):
        fh = [c["dial_on_hom_s01"] - c["dial_on_hom"] for c in C.values()]
        fd = {t: c["dial_on_dial_s01"] - c["dial_on_dial"] for t, c in C.items()}
        txt += fill(r"""<p>In the <a href="<<href>>">ranking test of the sampling MPC page</a>, each controller's ten best pairs were
chosen by its J at seeds 0 and 1. Both sets were run again under both controllers at seeds 10 and 11. Under full annealed sampling, HOM's
ten lost <<lo>> to <<hi>> of their mean J at the new seeds. The planner's own ten lost up to <<fd>>, on the <<tw>>, since
they had been chosen on the planner's J at seeds 0 and 1.</p>""",
                    href="../20261112-inner_loop_benchmark/20261113-sampling_mpc_vs_hom.html#rank100",
                    lo=f"{min(fh):.3f}", hi=f"{max(fh):.3f}", fd=f"{max(fd.values()):.3f}",
                    tw=TN[max(fd, key=fd.get)].lower())
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


PATCH_TASKS = ("twist", "press")


def patch_numbers(R, task):
    """HOM and full annealed sampling on the neighbourhood of HOM's best layout (inner_loop_mpc.py pairs --rule patch: every layout
    within 15 mm in x_sep and y_sep, each with HOM's best grasp), seeds 0 and 1: per controller the layouts' J (seeds
    averaged), the mean |J difference| between lattice neighbours 5 mm apart and between all pairs of the patch's
    layouts, their ratio, the seed agreement and the best layout; and the two controllers' rho over the patch."""
    pf = MP.ROWS / f"mpc_pairs_{task}_patch.json"
    if not pf.exists():
        return None
    keys = [(p_["x"], p_["y"], p_["grasp"]) for p_ in json.loads(pf.read_text())["pairs"]]
    out = {"keys": keys}
    for name, stem in (("hom", MP.tstem("hom", task, "_patch")), ("full", MP.tstem("anneal", task, "_patch", CFG_FULL2))):
        rows = rows_of(R, stem)
        if len(rows) < 2 * len(keys):
            return None
        J, J0, J1 = MP.pair_j(rows), MP.pair_j(rows, 0), MP.pair_j(rows, 1)
        v = np.array([J[k] for k in keys])
        idx = {k[:2]: i for i, k in enumerate(keys)}
        nb = [abs(v[i] - v[idx[(x + dx, y + dy)]]) for (x, y), i in idx.items() for dx, dy in ((5, 0), (0, 5))
              if (x + dx, y + dy) in idx]
        rp = [abs(v[i] - v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
        out[name] = {"J": v.tolist(), "mean": float(v.mean()), "nb": float(np.mean(nb)), "rand": float(np.mean(rp)),
                     "ratio": float(np.mean(nb) / np.mean(rp)) if np.mean(rp) > 0 else None,
                     "seed_rho": spearman([J0[k] for k in keys], [J1[k] for k in keys]),
                     "best": keys[int(np.argmax(v))], "best_j": float(v.max())}
    out["rho"] = spearman(out["hom"]["J"], out["full"]["J"])
    out["centre"] = max(keys, key=lambda k: next(p_["J_by"] for p_ in json.loads(pf.read_text())["pairs"]
                                                 if (p_["x"], p_["y"], p_["grasp"]) == k))
    return out


def fig_patch(N, out):
    """J of each layout of the two patches under HOM and under full annealed sampling, one colour scale per task."""
    plt = MP.plt_setup()
    fig, ax = plt.subplots(len(N), 2, figsize=(9.5, 4.5 * len(N)), squeeze=False)
    for r, (t, n) in enumerate(N.items()):
        xs = sorted({k[0] for k in n["keys"]})
        ys = sorted({k[1] for k in n["keys"]})
        vmax = max(max(n["hom"]["J"]), max(n["full"]["J"]), 0.05)
        for c, (name, lab) in enumerate((("hom", "HOM"), ("full", "Planner"))):
            G = np.full((len(ys), len(xs)), np.nan)
            for k, v in zip(n["keys"], n[name]["J"]):
                G[ys.index(k[1]), xs.index(k[0])] = v
            a = ax[r][c]
            im = a.imshow(G, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                          extent=[xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5])
            a.plot(*n["centre"][:2], marker="x", color="white", ms=12, mew=2.5)
            a.plot(*n[name]["best"][:2], marker="o", mfc="none", mec="#dc2626", ms=16, mew=2.5)
            a.set_title(f"{lab}: smoothness {f2(n[name]['ratio'])}", loc="left")
            a.set_xlabel("x_sep (mm)")
            if c == 0:
                a.set_ylabel(f"{TN[t]}: y_sep (mm)")
        fig.colorbar(im, ax=list(ax[r]), shrink=0.85, label="J, seeds 0 and 1")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def repeat_numbers(R, task):
    """The same layout, grasp and seed run twice, in the patch's batches and in the ranking test's: per controller the
    number of layouts and the mean and largest |J difference| between the two runs. HOM draws no random numbers; the
    planner draws its samples from one generator per batch (inner_loop_mpc.MPC, seeded by the batch's first seed and its
    position), so a layout's samples change with the batch it runs in."""
    out = {}
    for name, a_, b_ in (("hom", MP.tstem("hom", task, "_patch"), MP.tstem("hom", task, "_rank")),
                         ("full", MP.tstem("anneal", task, "_patch", CFG_FULL2),
                          MP.tstem("anneal", task, "_rank", CFG_FULL2))):
        A, B = ({(r["x_sep"], r["y_sep"], r["grasp"], r["seed"]): r["J"] for r in rows_of(R, st)} for st in (a_, b_))
        ks = sorted(set(A) & set(B))
        if not ks:
            return None
        d = np.array([abs(A[k] - B[k]) for k in ks])
        out[name] = {"n": len({k[:3] for k in ks}), "runs": len(ks), "mean": float(d.mean()), "max": float(d.max())}
    return out


def sec_patch(R):
    N = {t: patch_numbers(R, t) for t in PATCH_TASKS}
    if not all(N.values()):
        return ""
    tw, pr = N["twist"], N["press"]
    dist = (lambda n: float(np.hypot(n["full"]["best"][0] - n["centre"][0], n["full"]["best"][1] - n["centre"][1])))
    t_ = fill(r"""<p>Both controllers ran every layout of the grid within 15 mm of HOM's best twist and press layouts in
$x_{\mathrm{sep}}$ and in $y_{\mathrm{sep}}$, <<n_t>> on the twist and <<n_p>> on the press, each with HOM's best grasp,
at seeds 0 and 1 (Figure <<fn>>). Smoothness is measured by the mean |J difference| between lattice neighbours 5 mm
apart, divided by the mean |J difference| between all pairs of the patch's layouts. A ratio of 1 means that a
neighbour's J predicts a layout's J no better than any other layout's J does. On the twist the ratio was <<rh_t>> under
HOM and <<rd_t>> under full annealed sampling. On the press it was <<rh_p>> and <<rd_p>>.</p>
<p>Within the patches the two controllers ordered the layouts with $\rho$ = <<rho_t>> on the twist and <<rho_p>> on the
press. HOM's best layout of each patch was its grid best. Full annealed sampling's best lay <<dd_t>> mm from it on the twist, at J
<<bj_t>>, and <<dd_p>> mm on the press, at J <<bj_p>>.</p>""",
              n_t=len(tw["keys"]), n_p=len(pr["keys"]), fn=P.FIG[0] + 1, rh_t=f2(tw["hom"]["ratio"]),
              rd_t=f2(tw["full"]["ratio"]), rh_p=f2(pr["hom"]["ratio"]), rd_p=f2(pr["full"]["ratio"]),
              rho_t=f2(tw["rho"]), rho_p=f2(pr["rho"]), dd_t=f"{dist(tw):.0f}", dd_p=f"{dist(pr):.0f}",
              bj_t=f"{tw['full']['best_j']:.3f}", bj_p=f"{pr['full']['best_j']:.3f}")
    assert all(n["hom"]["best"] == n["centre"] for n in N.values())
    Q = {t: repeat_numbers(R, t) for t in PATCH_TASKS}
    if all(Q.values()):
        qt, qp = Q["twist"], Q["press"]
        t_ += fill(r"""<p>Of the patches' layouts, <<n_t>> on the twist and <<n_p>> on the press had also run in the ranking
test, at the same seeds and grasps but in other batches. Between the two runs of a layout and seed, HOM's J differed by a
mean of <<h_t>> on the twist and <<h_p>> on the press. Full annealed sampling's differed by <<d_t>> and <<d_p>>, and by up to
<<dm>>. HOM draws no random numbers. The planner draws its samples from one random generator per batch, so a layout's
samples change with the batch it runs in.</p>""",
                   n_t=qt["hom"]["n"], n_p=qp["hom"]["n"], h_t=f"{qt['hom']['mean']:.4f}", h_p=f"{qp['hom']['mean']:.4f}",
                   d_t=f"{qt['full']['mean']:.3f}", d_p=f"{qp['full']['mean']:.3f}",
                   dm=f"{max(qt['full']['max'], qp['full']['max']):.2f}")
    t_ += P.img(fig_patch(N, MEDIA / "20261115-hom_dial_patch.png"),
                "J of every layout within 15 mm of HOM's best twist and press layouts (white cross), under HOM (left) "
                "and under the planner, full annealed sampling (right), each with HOM's best grasp, seeds 0 and 1 averaged; one colour scale per "
                "task. Red circle: the controller's best layout of the patch. White: no layout of the grid. "
                "Smoothness: the mean |J difference| between neighbours 5 mm apart over that between all pairs of the "
                "patch's layouts.")
    return t_


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
    """HOM's and full annealed sampling's order of the 100 pairs under J and under each of its factors, and within each controller
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


def power_numbers(R, task):
    """J's floor at 0 and the power of the 100-pair comparison of full annealed sampling with HOM on a task: runs at J = 0 per
    controller (dropped; held with e >= e_ref; held, closer, but never with three pads in contact; all runs), the order
    agreement under J and under the unclipped tracking ratio, the
    95 % half-width of the paired difference and the pairs that detect 0.05 with 80 % power."""
    st, _lab = MP.rank_ref(R, task)
    h = hom_rank(R, task)
    if not st or not h:
        return None
    d = rows_of(R, st)
    e_ref = e_ref_of(h + d)

    def unc(r):
        return 1.0 - r["track_err_rms_deg"] / e_ref if r.get("held") and r.get("track_err_rms_deg") is not None \
            else -3.0
    zero = {nm: (sum(r["J"] == 0 for r in rows), sum(r["J"] == 0 and not r.get("held") for r in rows),
                 sum(r["J"] == 0 and bool(r.get("held")) and r["track_err_rms_deg"] >= e_ref for r in rows),
                 sum(r["J"] == 0 and bool(r.get("held")) and r["track_err_rms_deg"] < e_ref for r in rows), len(rows))
            for nm, rows in (("hom", h), ("dial", d))}
    keys = sorted({key(r)[:3] for r in h} & {key(r)[:3] for r in d})

    def per_pair(rows, f):
        v = {}
        for r in rows:
            v.setdefault(key(r)[:3], []).append(f(r))
        return np.array([np.mean(v[k]) for k in keys])
    jh, jd = per_pair(h, lambda r: r["J"]), per_pair(d, lambda r: r["J"])
    dd = jd - jh
    sd = float(dd.std(ddof=1))
    return {"e_ref": e_ref, "zero": zero, "rho_J": spearman(jh, jd),
            "rho_unc": spearman(per_pair(h, unc), per_pair(d, unc)), "diff": float(dd.mean()),
            "half": 1.96 * sd / np.sqrt(len(dd)), "n80": (1.959964 + 0.841621) ** 2 * sd ** 2 / 0.05 ** 2, "n": len(dd)}


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
        txt = fill(r"""<p>J is the product of the tracking factor and the three-pad share. Under HOM and full annealed sampling the pairs'
three-pad shares were ordered alike, with $\rho$ = <<c>> on the twist, roll, pin-push and press (Table <<tn>>). Their
tracking factors were ordered alike with <<tr>>. Full annealed sampling's mean tracking factor was <<trd>>, against HOM's <<trh>>.
Its three-pad share was higher on the <<hc>><<lc>>.</p>""",
                   tn=P.TAB[0] + 1, c=al([f2(A[t]["between"]["c"]) for t in TASKS]),
                   tr=al([f2(A[t]["between"]["track"]) for t in TASKS]),
                   trd=al([f"{A[t]['means']['dial']['track']:.2f}" for t in TASKS]),
                   trh=al([f"{A[t]['means']['hom']['track']:.2f}" for t in TASKS]), hc=lst(hi_c) if hi_c else "",
                   lc=("" if not lo_c else "; on the " + lst([TN[t].lower() for t in lo_c]) + " it was "
                       + ", ".join(f"{A[t]['means']['dial']['c']:.2f}" for t in lo_c) + " against HOM's "
                       + ", ".join(f"{A[t]['means']['hom']['c']:.2f}" for t in lo_c)))
    txt += table(["Task", "Pairs", "&#961; HOM/annealed sampling, J", "&#961;, tracking factor", "&#961;, three-pad share",
                  "Mean tracking factor, HOM / annealed sampling", "Mean three-pad share, HOM / annealed sampling"], rows,
                 "The order of each task's 100 pairs under HOM and under full annealed sampling, by J and by each of its two factors, "
                 "seeds averaged; and each factor's mean under the two controllers.", num_from=1)
    W = {t: power_numbers(R, t) for t in TASKS}
    if not all(W.values()):
        return txt
    zh = {t: W[t]["zero"]["hom"] for t in TASKS}
    zd = {t: W[t]["zero"]["dial"] for t in TASKS}
    pct = (lambda z: 100.0 * z[0] / z[4])
    txt += fill(r"""<p>J is 0 for a run that dropped the object, that tracked worse than holding it still
($e \ge e_{\mathrm{ref}}$), or that never had all three pads in contact. Such runs were <<zh0>> to <<zh1>> % of HOM's and
<<zd0>> to <<zd1>> % of full annealed sampling's (Table <<tn>>). On the roll, whose commanded rotation has an RMS of <<er>> deg, <<rh>> of HOM's <<rn>> runs held the sphere with
$e \ge e_{\mathrm{ref}}$. With the tracking ratio left unclipped, the two controllers' orders of the pairs agreed with
$\rho$ = <<ru>>, against <<rj>> under J. Over 100 pairs the paired difference of full annealed sampling and HOM had a 95 % half-width of
<<h0>> to <<h1>>. At the measured spread, <<n0>> to <<n1>> pairs detect a difference of 0.05 with a power of 0.8.</p>""",
                tn=P.TAB[0] + 1, zh0=f"{min(pct(z) for z in zh.values()):.0f}", zh1=f"{max(pct(z) for z in zh.values()):.0f}",
                zd0=f"{min(pct(z) for z in zd.values()):.0f}", zd1=f"{max(pct(z) for z in zd.values()):.0f}",
                er=f"{W['roll']['e_ref']:.1f}", rh=zh["roll"][2], rn=zh["roll"][4],
                ru=al([f2(W[t]["rho_unc"]) for t in TASKS]), rj=al([f2(W[t]["rho_J"]) for t in TASKS]),
                h0=f"{min(W[t]['half'] for t in TASKS):.3f}", h1=f"{max(W[t]['half'] for t in TASKS):.3f}",
                n0=f"{min(W[t]['n80'] for t in TASKS):.0f}", n1=f"{max(W[t]['n80'] for t in TASKS):.0f}")
    prow = [[TN[t], f"{zh[t][0]} ({zh[t][1]}, {zh[t][2]}, {zh[t][3]})", f"{zd[t][0]} ({zd[t][1]}, {zd[t][2]}, {zd[t][3]})",
             f2(W[t]["rho_J"]), f2(W[t]["rho_unc"]), f"{W[t]['diff']:+.3f} &#177; {W[t]['half']:.3f}",
             f"{W[t]['n80']:.0f}"] for t in TASKS]
    return txt + table(["Task", "Runs at J = 0, HOM", "Runs at J = 0, full annealed sampling",
                        "&#961; HOM/annealed sampling, J", "&#961;, unclipped tracking ratio", "Full annealed sampling &#8722; HOM, 95 % interval",
                        "Pairs for power 0.8 at 0.05"], prow,
                       "J's floor at 0 and the power of the comparison on each task's 100 pairs, seeds 0 and 1 (200 runs "
                       "per controller). Runs at J = 0 with, in brackets, those that dropped the object, those that held "
                       "it with e &#8805; e_ref and those that held it closer but never had three pads in contact. Pairs "
                       "for power 0.8: (1.96 + 0.84)&#178; s&#178; / 0.05&#178;, s the standard deviation of the pairs' "
                       "differences.", num_from=1)


def screen_numbers(R, task):
    """HOM's grid as a screen for full annealed sampling on a task's 100 pairs: the best full-DIAL J among the first k layouts in the
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
                       label="Random layouts, 10-90 %")
        a.plot(np.arange(1, len(n["rnd"]) + 1), n["rnd"], color="#6b7280", lw=2.0, label="Random layouts, mean")
        a.step(k, n["scr"], where="post", color="#2563eb", lw=2.4, label="HOM's order")
        a.axhline(n["d_max"], color="#d97706", lw=1.2, ls=(0, (4, 3)), label="Planner's best of the 100")
        a.set_xscale("log")
        a.set_xlim(1, len(k))
        a.set_ylim(0, 1.0)
        a.set_title(TN[t], loc="left")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    for a in ax[1]:
        a.set_xlabel("Layouts run")
    for a in ax[:, 0]:
        a.set_ylabel("Best J so far")
    h_, l_ = ax[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, loc="lower center", ncol=2, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_layouts(N, out):
    """The 100 pairs in the layout plane; HOM's ten best by its grid and full annealed sampling's ten best marked."""
    plt = MP.plt_setup()
    tasks = [t for t in TASKS if N.get(t)]
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 9.0), sharex=True, sharey=True)
    for a, t in zip(ax.flat, tasks):
        n = N[t]
        xy = np.array([k[:2] for k in n["keys"]])
        a.scatter(xy[:, 0], xy[:, 1], s=14, color="#d1d5db", lw=0, label="Other pairs")
        a.scatter(xy[n["top_h"], 0], xy[n["top_h"], 1], s=95, facecolors="none", edgecolors="#2563eb", lw=1.8,
                  label="HOM's ten best")
        a.scatter(xy[n["top_d"], 0], xy[n["top_d"], 1], s=40, marker="^", color="#d97706", lw=0,
                  label="Full annealed sampling's ten best")
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
    full = "Annealed sampling 4 &#215; 64"
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
under full annealed sampling than the 50 drawn at random from the rest, on every task (Table <<tn>>). The difference was <<g_t>> on the
twist, <<g_r>> on the roll, <<g_pp>> on the pin-push and <<g_p>> on the press.</p>
<p>A search with the planner alone would run it on layouts of the grid in no particular order. Figure <<fn>> compares
that with running it in the order of HOM's grid. That order and the order of HOM's own runs at seeds 0 and 1 agreed with
$\rho$ = <<rg>>. On the twist HOM's second layout scored <<s2_t>> under full annealed sampling, and 50 random
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
                "Best J under full annealed sampling after running the planner on k layouts. Blue: the layouts in the order of HOM's "
                "grid (each with HOM's best grasp). Grey: k layouts drawn at random from the rest of the grid, the mean "
                "and the 10-90 % band of 20,000 orders. Dashed: the planner's best of the 100 pairs. Seeds 0 and 1.")
    t_ += P.img(fig_layouts(N, MEDIA / "20261115-hom_dial_top_layouts.png"),
                "The 100 pairs of each task in the plane of the layout coordinates: thumb-to-pair distance x_sep and "
                "pair spacing y_sep. Circles: HOM's ten best by its grid; triangles: full annealed sampling's ten best, seeds 0 and "
                "1. Pairs that share a layout overlap.")
    return t_


def grid_numbers(R, task):
    """HOM and full annealed sampling on every layout of a task's grid, each with HOM's best grasp by the grid (pairs _all, seeds 0
    and 1), and each controller's ten best of it at seeds 10 and 11 (pairs _alltop): the agreement of the two orders,
    the planner's ten best against HOM's 50 best, the best full-DIAL J after k layouts in the grid's order and in random
    orders, the fresh-seed scores and the repeat of the 100 ranking pairs inside the grid. None while rows are missing."""
    pf = MP.ROWS / f"mpc_pairs_{task}_all.json"
    if not pf.exists():
        return None
    prs = json.loads(pf.read_text())["pairs"]
    keys = [(p["x"], p["y"], p["grasp"]) for p in prs]
    hr, dr = rows_of(R, f"hom_{task}_all"), rows_of(R, MP.tstem("anneal", task, "_all", CFG_FULL2))
    H, D = MP.pair_j(hr), MP.pair_j(dr)
    if any(k not in H or k not in D for k in keys) or len(dr) < 2 * len(keys):
        return None
    h, d = np.array([H[k] for k in keys]), np.array([D[k] for k in keys])
    jby = np.array([p["J_by"] for p in prs])
    order = sorted(range(len(keys)), key=lambda i: (-jby[i], keys[i][:2]))          # the grid's order (rank_pairs)
    best50 = {(p["x"], p["y"], p["grasp"]) for p in json.loads((MP.ROWS / f"mpc_pairs_{task}_rank.json").read_text())[
        "pairs"] if p["by"] == "hom_best"}
    top_d, top_g = [int(i) for i in np.argsort(-d, kind="stable")[:10]], order[:10]
    scr = np.maximum.accumulate(d[order])
    rng = np.random.default_rng(0)
    perm = np.array([np.maximum.accumulate(d[rng.permutation(len(d))]) for _ in range(20000)])
    reach = (perm >= d.max() - 1e-12).argmax(1) + 1
    seed_rho = {nm: spearman([MP.pair_j(rr, 0)[k] for k in keys], [MP.pair_j(rr, 1)[k] for k in keys])
                for nm, rr in (("hom", hr), ("dial", dr))}
    rep = {}
    for nm, a_, b_ in (("hom", hr, hom_rank(R, task)), ("dial", dr, rows_of(R, MP.tstem("anneal", task, "_rank",
                                                                                       CFG_FULL2)))):
        A_, B_ = {key(r): r["J"] for r in a_}, {key(r): r["J"] for r in b_}
        ks = sorted(set(A_) & set(B_))
        rep[nm] = (float(np.mean([abs(A_[k] - B_[k]) for k in ks])), len(ks)) if ks else (float("nan"), 0)
    out = {"task": task, "n": len(keys), "keys": keys, "h": h.tolist(), "d": d.tolist(), "jby": jby.tolist(),
           "order": order, "top_d": top_d, "top_g": top_g, "rho_hd": spearman(h, d), "rho_gd": spearman(jby, d),
           "seed_rho": seed_rho, "out50": sum(keys[i] not in best50 for i in top_d),
           "in10": len(set(top_d) & set(top_g)), "scr": scr.tolist(), "rnd": perm.mean(0).tolist(),
           "lo": np.quantile(perm, 0.1, 0).tolist(), "hi": np.quantile(perm, 0.9, 0).tolist(), "d_max": float(d.max()),
           "k_reach": int(np.argmax(scr >= d.max() - 1e-12)) + 1, "k_reach_rnd": float(reach.mean()),
           "med": {w: np.median(np.array([keys[i][:2] for i in ix]), 0).tolist() for w, ix in (("d", top_d),
                                                                                                ("g", top_g))},
           "mean_h": float(h.mean()), "mean_d": float(d.mean()), "rep": rep, "first_g_rank": int((d > d[order[0]]).sum()) + 1,
           "drop_all": {nm: (sum(1 for r in rr if not r.get("held")), len(rr)) for nm, rr in (("hom", hr), ("dial", dr))}}
    gr = sorted(order.index(i) + 1 for i in top_d)               # the grid's ranks of the planner's ten
    out["in100"] = [r for r in gr if r <= 100]
    tp = MP.ROWS / f"mpc_pairs_{task}_alltop.json"
    hf_r, df_r = rows_of(R, f"hom_{task}_alltop"), rows_of(R, MP.tstem("anneal", task, "_alltop", CFG_FULL2))
    if tp.exists() and hf_r and df_r:
        T_ = json.loads(tp.read_text())["pairs"]
        Hf, Df = MP.pair_j(hf_r), MP.pair_j(df_r)
        sets = {"d": [keys[i] for i in top_d], "g": [keys[i] for i in top_g]}
        if all(k in Hf and k in Df for s_ in sets.values() for k in s_) and len(df_r) >= 2 * len(T_):
            drop_h = {w: (sum(1 for r in hf_r if (r["x_sep"], r["y_sep"], r["grasp"]) in set(s_) and not r.get("held")),
                          2 * len(s_)) for w, s_ in sets.items()}
            df = {w: np.array([Df[k] for k in s_]) for w, s_ in sets.items()}
            bs = [rng.choice(df["d"], 10).mean() - rng.choice(df["g"], 10).mean() for _ in range(4000)]
            fused = max(sets["g"], key=lambda k: D[k])                # the planner's best of the grid's ten, seeds 0-1
            first = keys[top_d[0]]
            cf = {(nm, w): float(np.mean([r["contact_all_frac"] for r in rr if (r["x_sep"], r["y_sep"], r["grasp"])
                                          in set(s_)])) for nm, rr in (("hom", hf_r), ("dial", df_r))
                  for w, s_ in sets.items()}
            out["fresh"] = {"contact": cf, "s01": {w: float(np.mean([D[k] for k in s_])) for w, s_ in sets.items()},
                            "dial": {w: float(v.mean()) for w, v in df.items()},
                            "hom": {w: float(np.mean([Hf[k] for k in s_])) for w, s_ in sets.items()},
                            "drop_h": drop_h, "diff": (float(df["d"].mean() - df["g"].mean()),
                                                       float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
                            "fused": fused, "fused_dial": Df[fused], "first": first, "first_dial": Df[first],
                            "fused_is_first": fused == first}
    return out


def press_grid_numbers(R):
    return grid_numbers(R, "press")


def grids_numbers(R):
    """grid_numbers of every task whose grid has run, in TASKS order."""
    return {t: N for t in TASKS if (N := grid_numbers(R, t))}


def fig_grids(Ns, out):
    """Per task (rows): the grid in the layout plane coloured by J under HOM and under full annealed sampling, the grid's ten best and
    the planner's ten best marked; the best J under full annealed sampling after k layouts in the grid's order and in random orders."""
    plt = MP.plt_setup()
    tasks = list(Ns)
    fig, ax = plt.subplots(len(tasks), 3, figsize=(10.4, 3.3 * len(tasks) + 0.7), squeeze=False, layout="constrained")
    for i, t in enumerate(tasks):
        N = Ns[t]
        xy = np.array([k[:2] for k in N["keys"]])
        h, d = np.array(N["h"]), np.array(N["d"])
        vmax = float(max(h.max(), d.max()))
        for j, (v, lab) in enumerate(((h, "HOM"), (d, "planner"))):
            a = ax[i, j]
            sc = a.scatter(xy[:, 0], xy[:, 1], c=v, cmap="viridis", vmin=0, vmax=vmax, s=16, marker="s", lw=0)
            a.scatter(xy[N["top_g"], 0], xy[N["top_g"], 1], s=80, facecolors="none", edgecolors="#2563eb", lw=1.8,
                      label="Grid's ten best (HOM)")
            a.scatter(xy[N["top_d"], 0], xy[N["top_d"], 1], s=48, marker="^", color="#f97316", edgecolors="black",
                      lw=0.5, label="Planner's ten best")
            a.set_title(f"{TN[t]}, {lab}")
            a.set_xlim(35, 165)
            a.set_ylim(50, 175)
            a.set_aspect("equal")
            a.set_ylabel("$y_{sep}$ (mm)" if j == 0 else "")
            if i == len(tasks) - 1:
                a.set_xlabel("$x_{sep}$ (mm)")
        fig.colorbar(sc, ax=[ax[i, 0], ax[i, 1]], shrink=0.9, pad=0.02, label="J")
        a = ax[i, 2]
        k = np.arange(1, N["n"] + 1)
        a.fill_between(k, N["lo"], N["hi"], color="#d1d5db", lw=0, label="Random order, 10-90 %")
        a.plot(k, N["rnd"], color="#6b7280", lw=1.8, label="Random order, mean")
        a.step(k, N["scr"], where="post", color="#2563eb", lw=2.2, label="HOM's grid order")
        a.axhline(N["d_max"], color="#f97316", lw=1.1, ls=(0, (4, 3)), label="Best of the grid")
        a.set_xscale("log")
        a.set_xlim(1, N["n"])
        a.set_ylim(0, 1.0)
        a.set_ylabel("Best J so far")
        if i == len(tasks) - 1:
            a.set_xlabel("Layouts run")
        a.set_title(f"{TN[t]}, search")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    h1, l1 = ax[0, 0].get_legend_handles_labels()
    h2, l2 = ax[0, 2].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, frameon=False, loc="outside lower center", ncol=3)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def grid_films(N, task):
    """The films of HOM and full annealed sampling at seed 10 on full annealed sampling's first choice of a task's grid and on the grid's first
    choice (logs/20261115-press_grid/films.sh, logs/20261115-uhas_sv/grid_films.sh): [(path, caption)]."""
    first_d, first_g = tuple(N["keys"][N["top_d"][0]]), tuple(N["keys"][N["order"][0]])
    out = []
    for p in sorted(MP.MEDIA.glob(f"20261115-film_mpc_{task}_alltop_p*_s10.mp4")):
        info_p = MP.ROWS / (p.stem.replace("20261115-", "") + ".json")
        if not info_p.exists():
            continue
        info = json.loads(info_p.read_text())
        pr = info["pair"]
        k_ = (pr["x"], pr["y"], pr["grasp"])
        who = ("full annealed sampling's first choice of the grid at seeds 0 and 1" if k_ == first_d else "")
        if k_ == first_g:
            who = (who + ", and " if who else "") + "the grid's first choice, by HOM's J over seeds 0-2"
        if not who:
            continue
        parts = []
        for spec, c in info["clips"].items():
            nm = MP.FILM_SPEC.get(spec, esc(spec))
            parts.append(f"{cap(nm)}: J {c['J']:.2f}, grip {c['grip_mean_N']:.1f} N" + ("" if c.get("held") else ", dropped"))
        out.append((p, f"{TN[task]}: layout ({pr['x']:g}, {pr['y']:g}) mm, grasp {pr['grasp']}, seed {info['seed']}; "
                       f"{who}. " + ". ".join(parts) + ". A planner's run in the film draws other samples than its runs "
                       "in the tables. Each frame gives each finger's pad force and the force through its rigid parts "
                       "(N); the links are drawn see-through."))
    return out


def sec_grids(R):
    Ns = grids_numbers(R)
    if not Ns:
        return ""
    ts = list(Ns)
    names = MP.tasks_list(ts)
    sizes = al([f"{Ns[t]['n']}" + (" layouts" if i == 0 else "") + f" for the {TN[t].lower()}" for i, t in enumerate(ts)])
    miss = [t for t in ts if Ns[t]["out50"] > 5]
    keep = [t for t in ts if Ns[t]["out50"] <= 5]
    rho = [Ns[t]["rho_hd"] for t in ts]
    txt = fill(r"""<p>Full annealed sampling and HOM ran on every layout of the grids of <<names>>, each layout with HOM's best grasp by
the grid, at seeds 0 and 1 (Table <<tn>>, Figure <<fn>>). The grids have <<sizes>>. Over each grid the two
controllers' J agreed with $\rho$ = <<r0>> to <<r1>>. Full annealed sampling's mean J over a grid was <<md0>> to <<md1>>, HOM's
<<mh0>> to <<mh1>>.</p>""",
               names=names, sizes=sizes, tn=P.TAB[0] + 1, fn=P.FIG[0] + 1, r0=f2(min(rho)), r1=f2(max(rho)),
               md0=f"{min(Ns[t]['mean_d'] for t in ts):.3f}", md1=f"{max(Ns[t]['mean_d'] for t in ts):.3f}",
               mh0=f"{min(Ns[t]['mean_h'] for t in ts):.3f}", mh1=f"{max(Ns[t]['mean_h'] for t in ts):.3f}")
    cnt = (lambda L_: al([NUMW[Ns[t]["out50"]] for t in L_]))
    txt += fill(r"""<p>The criterion was set before the press's run and applied unchanged to the other grids. HOM's grid
misses the planner's best layouts of a task if more than five of the planner's ten best lie outside HOM's 50 best. By
that criterion the grid missed them on <<miss>>, where <<mc>> of the ten lay outside.<<keep>></p>""",
                miss=MP.tasks_list(miss) if miss else "no task", mc=cnt(miss) if miss else "none",
                keep=(f" It did not miss them on {MP.tasks_list(keep)}, where {cnt(keep)} lay outside." if keep else ""))
    first1 = [t for t in ts if Ns[t]["first_g_rank"] == 1]
    if first1:
        txt = txt[:-4] + fill(" On <<t>> the grid's first layout was also full annealed sampling's best of the grid.</p>",
                              t=MP.tasks_list(first1))
    fr = {t: Ns[t]["fresh"] for t in ts if Ns[t].get("fresh")}
    if fr:
        up = [t for t in fr if fr[t]["diff"][1] > 0]
        flat = [t for t in fr if t not in up]
        dd = (lambda t, on=True: f"{fr[t]['diff'][0]:+.3f}" + (f" on the {TN[t].lower()}" if on else "")
              + f" ({fr[t]['diff'][1]:+.3f} to {fr[t]['diff'][2]:+.3f})")
        s = [r"""<p>Full annealed sampling's ten best layouts at seeds 0 and 1 and the grid's ten best ran again at seeds 10 and 11
(Table """ + str(P.TAB[0] + 2) + "; 95 % intervals in brackets)."]
        if up:
            s.append(f"Under full annealed sampling the planner's ten scored more than the grid's ten on {MP.tasks_list(up)}, by "
                     f"{al([dd(t) for t in up])}.")
        if flat:
            s.append("The difference was " + al([dd(t) for t in flat]) + ".")
        for i, t in enumerate(flat):
            s.append(f"On the {TN[t].lower()} the planner's ten fell from {fr[t]['s01']['d']:.3f} at seeds 0 and 1 to "
                     f"{fr[t]['dial']['d']:.3f}, and the grid's ten scored {fr[t]['dial']['g']:.3f}.")
        dh = {t: fr[t]["drop_h"] for t in fr}
        s.append("Under HOM the planner's ten dropped the object in " + al([str(dh[t]["d"][0]) for t in fr])
                 + " of 20 runs on the " + al([TN[t].lower() for t in fr]) + ". The grid's ten dropped it in "
                 + al([str(dh[t]["g"][0]) for t in fr]) + ".</p>")
        txt += " ".join(s)
    rows = []
    for t in ts:
        N = Ns[t]
        rows.append([TN[t], str(N["n"]), f2(N["rho_hd"]), f"{f2(N['seed_rho']['hom'])} / {f2(N['seed_rho']['dial'])}",
                     f"{N['mean_h']:.3f}", f"{N['mean_d']:.3f}", str(N["out50"]), str(N["first_g_rank"]),
                     f"{N['k_reach']} / {N['k_reach_rnd']:.0f}"])
    txt += table(["Task", "Layouts", "&#961;, HOM and planner", "&#961;, seeds 0 and 1: HOM / planner",
                  "Mean J, HOM", "Mean J, planner", "Planner's ten outside HOM's 50 best",
                  "Rank of the grid's first layout under the planner", "Layouts to the planner's best: grid order / random"],
                 rows, "HOM and the planner, full annealed sampling (4 &#215; 64), on every layout of each grid, each layout "
                 "with HOM's best grasp by the grid, seeds 0 and 1. HOM's 50 best: the ranking test's 50 layouts with "
                 "HOM's best J on the grid. Layouts to the planner's best: the layouts run before its best of the grid is "
                 "reached, in the grid's order and as the mean over 20,000 random orders.", num_from=1)
    if fr:
        rows = []
        for t in fr:
            F = fr[t]
            rows.append([TN[t], f"{F['dial']['d']:.3f}", f"{F['dial']['g']:.3f}", fds(dict(
                d=F["diff"][0], lo=F["diff"][1], hi=F["diff"][2])), f"{F['hom']['d']:.3f}", f"{F['hom']['g']:.3f}",
                f"{F['drop_h']['d'][0]} / {F['drop_h']['g'][0]}",
                f"{100 * F['contact'][('hom', 'd')]:.0f} / {100 * F['contact'][('hom', 'g')]:.0f}"])
        txt += table(["Task", "Planner: its ten", "Planner: the grid's ten", "Difference (95 % interval)",
                      "HOM: the planner's ten", "HOM: the grid's ten", "HOM's drops: planner's ten / grid's ten",
                      "Three pads in contact under HOM (% of the task): planner's ten / grid's ten"], rows,
                     "The planner's ten best layouts of each grid (full annealed sampling, at seeds 0 and 1) and the "
                     "grid's ten best (by HOM's J over seeds 0-2), run again at seeds 10 and 11: mean J under each "
                     "controller, drops out of 20 runs and the share of the task with all three pads in contact.",
                     num_from=1)
    rep_h = max(Ns[t]["rep"]["hom"][0] for t in ts)
    rep_d = [Ns[t]["rep"]["dial"][0] for t in ts]
    txt += fill(r"""<p>The 100 pairs of Section SCREENSEC are part of each grid, and their runs at seeds 0 and 1 were
repeated in it. Between the two runs a pair's J at the same seed differed by a mean of <<rh>> or less under HOM and by
<<rd0>> to <<rd1>> under full annealed sampling.</p>""", rh=f"{rep_h:.3f}", rd0=f"{min(rep_d):.3f}", rd1=f"{max(rep_d):.3f}")
    txt += P.img(fig_grids(Ns, MEDIA / "20261115-hom_grids.png"),
                 "HOM and full annealed sampling on every layout of each task's grid, each layout with HOM's best grasp by the grid, "
                 "seeds 0 and 1, one task per row. Left and middle: J of each layout under HOM and under the planner, full "
                 "annealed sampling, on one colour scale per task. Circles: the grid's ten best by HOM's J over seeds 0-2; "
                 "triangles: the planner's ten best. Right: the best J under the planner after running it on k layouts. "
                 "Blue: in the grid's order. Grey: in 20,000 random orders, the mean and the 10-90 % band. Dashed: the "
                 "best of the grid.")
    for t in ts:
        for p, c in grid_films(Ns[t], t):
            txt += P.film(p, c)
    return txt


def variant_effects(R):
    """Each variant's paired difference from HOM's defaults on its tasks' 20 pairs: [(tag, label, tasks, {task:
    paired})], or None while a variant's rows are missing."""
    eff = []
    for tag, lab, tasks in VARIANTS:
        d = {t: paired(rows_of(R, f"hom_{t}_rank20{tag}"), hom_rank(R, t)) for t in tasks if hom_rank(R, t)}
        if len(d) < len(tasks) or not all(d.values()):
            return None
        eff.append((tag, lab, tasks, d))
    return eff


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
    txt = fill(r"""<p>HOM solves Eq. (2) for the finger joints' rates: a least-squares problem with a damping term and a
bound on each joint's rate. It is solved exactly. For each finger the 27 patterns of its three joints at the lower bound,
free or at the upper bound are enumerated (<code>hom_gpu.py</code>). Another solver of the same problem returns the same
rates, so the variants change the problem. They change Eq. (2)'s damping, rate bound or reference lead. They add a
preview of the command, other forces and contact modes, a force loop or feedback on the object's orientation. Each was
run on the 20
pairs of each task at seeds 0 and 1 (Table <<tn>>).</p>""", tn=P.TAB[0] + 1)
    eff = variant_effects(R)
    if eff:
        ups = sorted([e for e in eff if any(e[3][t]["lo"] > 0 for t in e[2])],
                     key=lambda e: -max(e[3][t]["d"] for t in e[2] if e[3][t]["lo"] > 0))
        downs = [e for e in eff if e not in ups and all(e[3][t]["hi"] < 0 for t in e[2])]
        rest = [e for e in eff if e not in ups and e not in downs]
        s = ["A rise or fall below is one whose 95 % interval of the paired difference excludes zero."]
        if not ups:
            s.append("No variant raised J on any task.")
        for tag, lab, tasks, d in ups:
            up_t = [t for t in tasks if d[t]["lo"] > 0]
            oth = [d[t]["d"] for t in tasks if t not in up_t]
            s.append(f"{cap(LNAME[tag])} raised J by {al([f'{d[t]['d']:.3f}' for t in up_t])} on "
                     f"{MP.tasks_list(up_t)}.")
            if oth:
                s.append(f"On the other {NUMW[len(oth)]} tasks the change was {min(oth):+.3f} to {max(oth):+.3f}."
                         if len(oth) > 1 else f"On the other task the change was {oth[0]:+.3f}.")
        for tag, lab, tasks, d in downs:
            v = [-d[t]["d"] for t in tasks]
            s.append(f"{cap(LNAME[tag])} lowered J on {'every task' if len(tasks) == len(TASKS) else MP.tasks_list(tasks)}, "
                     f"by {min(v):.3f} to {max(v):.3f}.")
        if rest:
            s.append(f"The other {NUMW[len(rest)]} variants changed J by "
                     f"{max(abs(d[t]['d']) for _t, _l, tasks, d in rest for t in tasks):.3f} or less on every task.")
        txt += "<p>" + " ".join(s) + "</p>"
    return txt + table(["HOM variant"] + [TN[t] for t in TASKS], [base] + rows,
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
    eq = [t for t in TASKS if gap[t] < 5e-4]
    lo = [t for t in TASKS if t not in eq]
    gp = "On the 100 pairs of Section SCREENSEC, the planner's best J on HOM's ten "
    if eq:
        gp += f"equalled its best on all 100 on {MP.tasks_list(eq)}."
        if lo:
            gp += f" On {MP.tasks_list(lo)} it was {al([f'{gap[t]:.3f}' for t in lo])} lower."
    else:
        gp += f"was {al([f'{gap[t]:.3f}' for t in lo])} below its best on all 100 on {MP.tasks_list(lo)}."
    txt = fill(r"""<p>The methods can also be fused at the level of the search. HOM scores every layout of the grid,
and a planner re-scores HOM's ten best layouts of each task. <<gp>> On the search of the talk's cost chart, 1,423
layouts of the twist, the roll and the press, HOM takes <<hh>> hours of GPU time. Re-scoring ten layouts per task at
seeds 0 and 1 under full annealed sampling takes <<em>> minutes more, <<pc>> % of HOM's time.</p>""", gp=gp, hh=f"{hom_s / 3600:.1f}",
               em=f"{extra / 60:.0f}", pc=f"{100 * extra / hom_s:.1f}")
    C = {t: MP.confirm_numbers(t) for t in TASKS}
    if all(C.values()):
        prs = {t: json.loads((MP.ROWS / f"mpc_pairs_{t}_rank.json").read_text())["pairs"] for t in TASKS}
        g10 = {t: {(p_["x"], p_["y"], p_["grasp"]) for p_ in sorted(prs[t], key=lambda p_: -p_["J_by"])[:10]} for t in TASKS}
        r10 = {t: {(p_["x"], p_["y"], p_["grasp"]) for p_ in json.loads((MP.ROWS / f"mpc_pairs_{t}_top.json").read_text())[
            "pairs"] if p_["by"] in ("hom_top", "both")} for t in TASKS}
        same = [t for t in TASKS if C[t]["fused_is"] == "planner"]
        other = [t for t in TASKS if t not in same]
        oth = ""
        if other:
            oth = (f" On {MP.tasks_list(other)} it was another layout, which scored "
                   f"{al([f'{C[t]['first_d_dial'] - C[t]['fused_dial']:.3f}' for t in other])} below the planner's first "
                   "choice at seeds 10 and 11.")
            oth += "".join(f" On the {TN[t].lower()} HOM's own first choice scored "
                           f"{C[t]['first_h_dial'] - C[t]['fused_dial']:.3f} more than that layout."
                           for t in other if C[t]["first_h_dial"] - C[t]["fused_dial"] > 0.005)
        txt += fill(r"""<p>The fresh-seed check of the <a href="<<href>>">ranking test</a> chose HOM's ten by HOM's own
runs at seeds 0 and 1. They share <<sh>> layouts with the grid's ten on the twist, roll, pin-push and press. Among them
the planner's best at seeds 0 and 1 was its own first choice of the 100 on <<same>>.<<oth>></p>""",
                    href="../20261112-inner_loop_benchmark/20261113-sampling_mpc_vs_hom.html#rank100",
                    sh=al([str(len(g10[t] & r10[t])) for t in TASKS]), same=MP.tasks_list(same), oth=oth)
    return txt


def sec_fusion(R):
    rows, any_ = [], False
    txt = search_fusion(R)
    dd = {m: {t: paired(rows_of(R, MP.tstem(m, t, "_rank20", CFG_PICK)), hom_rank(R, t)) for t in TASKS} for m, _l in FUSION}
    if all(all(v.values()) for v in dd.values()):
        def says(m, name):
            up = [f"the {TN[t].lower()} ({dd[m][t]['d']:+.3f})" for t in TASKS if dd[m][t]["lo"] > 0]
            dn = [f"the {TN[t].lower()} ({dd[m][t]['d']:+.3f})" for t in TASKS if dd[m][t]["hi"] < 0]
            return (f"{name} scored above HOM on {al(up) if up else 'no task'}"
                    + (f" and below it on {al(dn)}" if dn else "") + ".")
        txt += fill(r"""<p>Two planners were fused with HOM at the real-time pick's budget, 2 &#215; 32 samples over 0.1 s.
They ran on the 20 pairs of each task at seeds 0 and 1 (Table <<tn>>). By the 95 % interval of the paired difference, <<a>>
<<b>></p>""", tn=P.TAB[0] + 1,
                    a=says("res_anneal", "the annealed-sampling residual on HOM's servo targets"),
                    b=says("hs_anneal", "Annealed sampling in HOM's space"))
    # the residual against DIAL 2 x 32 alone and full DIAL on the same 20 pairs, and the GPU time per twist rollout
    alone = {t: sub20(R, t, rows_of(R, MP.tstem("anneal", t, "_rank", CFG_PICK))) for t in TASKS}
    full = {t: sub20(R, t, rows_of(R, MP.tstem("anneal", t, "_rank", CFG_FULL2))) for t in TASKS}
    res = {t: rows_of(R, MP.tstem("res_anneal", t, "_rank20", CFG_PICK)) for t in TASKS}
    if all(alone[t] and full[t] and res[t] for t in TASKS):
        pf = {t: paired(res[t], full[t]) for t in TASKS}
        pc = {t: paired(res[t], alone[t]) for t in TASKS}
        below = [t for t in TASKS if pf[t]["hi"] < 0]
        zero = [t for t in TASKS if t not in below and pf[t]["lo"] <= 0]
        hi_c = [t for t in TASKS if pc[t]["lo"] > 0]
        s = []
        if below:
            s.append("Against full annealed sampling on the same pairs, the residual scored "
                     + al([f"{-pf[t]['d']:.3f} lower on the {TN[t].lower()}" for t in below]) + ".")
        if zero:
            s.append(f"On {MP.tasks_list(zero)} the interval of that difference included zero "
                     f"({al([f'{pf[t]['d']:+.3f}' for t in zero])}).")
        if hi_c:
            v = [pc[t]["d"] for t in hi_c]
            s.append(f"Against annealed sampling 2 &#215; 32 alone it scored {min(v):.3f} to {max(v):.3f} higher on "
                     f"{MP.tasks_list(hi_c)}.")
        gr = [gpu_ratio(R, MP.tstem(m, "twist", ps, c)) for m, ps, c in (
            ("res_anneal", "_rank20", CFG_PICK), ("anneal", "_rank", CFG_PICK), ("anneal", "_rank", CFG_FULL2))]
        if all(gr):
            s.append(f"Per twist rollout the residual took {gr[0]:.0f} times HOM's GPU time, annealed sampling 2 &#215; 32 alone "
                     f"{gr[1]:.0f} times and full annealed sampling {gr[2]:.0f} times.")
        one = one_hand_replan(R)
        if one:
            rt = one["res"] <= one["budget"]
            s.append(f"For one hand alone a replanning of the residual took {one['res']:.3f} s, against the replanning "
                     f"interval of {one['budget']:.1f} s. One of annealed sampling 2 &#215; 32 alone took {one['alone']:.3f} s (twist, "
                     f"{one['n']} replannings each)."
                     + (" The residual is therefore a real-time controller." if rt else
                        f" The residual therefore runs at {one['res'] / one['budget']:.1f} times real time."))
        txt += "<p>" + " ".join(s) + "</p>"

    def cell(v, h):
        rs = [r["replan_s_mean"] for r in v if r.get("replan_s_mean")]
        return f"{MP.mean_j(v):.3f}<br>{fd(paired(v, h))}" + (f"<br>{np.mean(rs):.2f} s" if rs else "")
    lines = ([("HOM", None), ("Annealed sampling 2 &#215; 32 alone, real time", ("anneal", "_rank", CFG_PICK))]
             + [(lab, (m, "_rank20", CFG_PICK)) for m, lab in FUSION]
             + [("Full annealed sampling 4 &#215; 64, 6.3 times real time", ("anneal", "_rank", CFG_FULL2))])
    for lab, spec in lines:
        row = [lab]
        for t in TASKS:
            h = hom_rank(R, t)
            if spec is None:
                row.append(f"{MP.mean_j(sub20(R, t, h)):.3f}" if h else "&#8212;")
                continue
            m, ps, c = spec
            v = rows_of(R, MP.tstem(m, t, ps, c))
            v = sub20(R, t, v) if ps == "_rank" and v else v
            if v and h:
                any_ = any_ or m != "anneal"
                row.append(cell(v, h))
            else:
                row.append("&#8212;")
        rows.append(row)
    if not any_:
        return txt
    return txt + table(["Controller"] + [TN[t] for t in TASKS], rows,
                 "Planners fused with HOM on each task's 20 pairs at seeds 0 and 1, with HOM, annealed sampling 2 &#215; 32 alone and "
                 "full annealed sampling on the same pairs and seeds: mean J; the paired difference from HOM; the replanning time "
                 "for 16 hands planned together.", num_from=1)


def ilqr_numbers(R):
    """iLQR (inner_loop_ilqr.py) on the 20 pairs of each task at seeds 0 and 1 and on the tuning pairs at 2 and 4
    iterations: mean J against HOM, full annealed sampling and the annealed-sampling residual on the same pairs, the paired differences, the line
    search's record, the targets' rate and reversals at the 50 ms step, the repeat of the tuning runs (rows _first), the
    GPU time; None while the 20-pair rows are missing."""
    r20 = {t: rows_of(R, MP.tstem("ilqr", t, "_rank20", CFG_IL2)) for t in TASKS}
    tune = {c: {t: rows_of(R, MP.tstem("ilqr", t, "_tune", c)) for t in TASKS} for c in (CFG_IL2, CFG_IL4)}
    if not all(len(v) >= 40 for v in r20.values()) or not all(tune[CFG_IL2].values()):
        return None
    h20 = {t: sub20(R, t, hom_rank(R, t)) for t in TASKS}
    f20 = {t: sub20(R, t, rows_of(R, MP.tstem("anneal", t, "_rank", CFG_FULL2))) for t in TASKS}
    s20 = {t: rows_of(R, MP.tstem("res_anneal", t, "_rank20", CFG_PICK)) for t in TASKS}
    ref = {nm: {t: MP.mean_j(rows_of(R, MP.tstem(m, t, "_tune", c))) for t in TASKS}
           for nm, m, c in (("hom", "hom", None), ("d64", "anneal", CFG_D64), ("full", "anneal", CFG_FULL2))}
    rows20, full20 = [r for t in TASKS for r in r20[t]], [r for t in TASKS for r in f20[t]]
    st = [r["ilqr_stats"] for r in rows20 if r.get("ilqr_stats")]
    rep, rep_max, flip = {}, 0.0, 0
    for t in TASKS:
        a_ = {key(r): r for r in tune[CFG_IL2][t]}
        b_ = {key(r): r for r in rows_of(R, MP.tstem("ilqr", t, "_tune", CFG_IL2) + "_first")}
        ks = set(a_) & set(b_)
        if ks:
            rep[t] = float(np.mean([abs(a_[k]["J"] - b_[k]["J"]) for k in ks]))
            rep_max = max(rep_max, max(abs(a_[k]["J"] - b_[k]["J"]) for k in ks))
            flip += sum(bool(a_[k].get("held")) != bool(b_[k].get("held")) for k in ks)
    one = [r["replan_s_mean"] for r in rows_of(R, MP.tstem("ilqr", "twist", "_tune", CFG_IL2) + "_chunk1")
           if r.get("replan_s_mean")]
    fac = {}                                    # J's two factors: three-pad share and tracking factor (held runs)
    for t in TASKS:
        e_ref = e_ref_of(r20[t] + f20[t] + h20[t])
        fac[t] = {nm: (float(np.mean([r["contact_all_frac"] for r in rr])),
                       float(np.mean([max(0.0, 1.0 - r["track_err_rms_deg"] / e_ref) for r in rr if r.get("held")])))
                  for nm, rr in (("il", r20[t]), ("full", f20[t]))}
    return {"fac": fac, "j20": {t: MP.mean_j(r20[t]) for t in TASKS}, "h20": {t: MP.mean_j(h20[t]) for t in TASKS},
            "f20": {t: MP.mean_j(f20[t]) for t in TASKS}, "s20": {t: MP.mean_j(s20[t]) for t in TASKS},
            "vs_hom": {t: paired(r20[t], h20[t]) for t in TASKS}, "vs_full": {t: paired(r20[t], f20[t]) for t in TASKS},
            "vs_res": {t: paired(r20[t], s20[t]) for t in TASKS},
            "tune": {c: {t: MP.mean_j(v) for t, v in d.items()} for c, d in tune.items()}, "ref": ref,
            "ro": {c: float(np.mean([r["rollouts_per_replan"] for t in TASKS for r in d[t]])) if all(d.values())
                   else None for c, d in tune.items()},
            "accept": float(np.mean([x["accept"] for x in st])), "full_step": float(np.mean([x["full_step"] for x in st])),
            "rate": float(np.mean([r["cmd_rate_dps"] for r in rows20])),
            "rev": float(np.mean([r["reversals_ps"] for r in rows20])),
            "rate_full": float(np.mean([r["cmd_rate_dps"] for r in full20])),
            "rev_full": float(np.mean([r["reversals_ps"] for r in full20])),
            "drop": (sum(not r.get("held") for r in rows20), len(rows20)),
            "drop_full": (sum(not r.get("held") for r in full20), len(full20)),
            "gpu": gpu_ratio(R, MP.tstem("ilqr", "twist", "_rank20", CFG_IL2)),
            "gpu_res": gpu_ratio(R, MP.tstem("res_anneal", "twist", "_rank20", CFG_PICK)),
            "gpu_full": gpu_ratio(R, MP.tstem("anneal", "twist", "_rank", CFG_FULL2)),
            "one": float(np.mean(one)) if one else None,
            "rep": rep if len(rep) == len(TASKS) else None, "rep_max": rep_max, "rep_flip": flip,
            "pad": ilqr_pad_numbers(R, r20, f20, h20)}


def ilqr_pad_numbers(R, r20, f20, h20):
    """iLQR with the smooth pad term (inner_loop_mpc.py run --method ilqr --il-pad) on the same 20 pairs: mean J, the
    paired differences from iLQR without it, full annealed sampling and HOM, the three-pad share, drops, the GPU time; None while its
    rows are missing."""
    p20 = {t: rows_of(R, MP.tstem("ilqr", t, "_rank20", CFG_IL2, sh="_pad")) for t in TASKS}
    if not all(len(v) >= 40 for v in p20.values()):
        return None
    share = (lambda rr: float(np.mean([r["contact_all_frac"] for r in rr])))
    return {"j20": {t: MP.mean_j(p20[t]) for t in TASKS}, "vs_il": {t: paired(p20[t], r20[t]) for t in TASKS},
            "vs_full": {t: paired(p20[t], f20[t]) for t in TASKS}, "vs_hom": {t: paired(p20[t], h20[t]) for t in TASKS},
            "share": {t: {"pad": share(p20[t]), "il": share(r20[t]), "full": share(f20[t])} for t in TASKS},
            "drop": (sum(not r.get("held") for t in TASKS for r in p20[t]), sum(len(p20[t]) for t in TASKS)),
            "gpu": gpu_ratio(R, MP.tstem("ilqr", "twist", "_rank20", CFG_IL2, sh="_pad"))}


def sec_ilqr(R):
    N = ilqr_numbers(R)
    if not N:
        return ""
    m4 = (lambda d: float(np.mean([d[t] for t in TASKS])))
    lst = (lambda d: al([f"{d[t]:.3f}" for t in TASKS]))

    def where(pd, sign):
        ts = [t for t in TASKS if (pd[t]["lo"] > 0 if sign > 0 else pd[t]["hi"] < 0)]
        return MP.tasks_list(ts) if ts else "no task"
    t4 = N["tune"][CFG_IL4]
    txt = fill(r"""<p>iLQR [<<ref>>] scored <<j>> on the 20 pairs of the twist, the roll, the pin-push and the press at seeds 0
and 1, a mean of <<jm>> (Table <<tn>>). HOM scored <<h>> on the same pairs (mean <<hm>>) and full annealed sampling <<f>> (mean <<fm>>).
By the 95 % interval of the paired difference, iLQR scored above HOM on <<ah>> and below it on <<bh>>. It scored below full
annealed sampling on <<bf>> and above it on <<af>>. It dropped the object in <<dr>> of <<dn>> runs, full annealed sampling in <<drf>>. Per twist
rollout iLQR took <<g>> times HOM's GPU time at 16 hands per batch, the annealed-sampling residual on HOM <<gr>> times and full annealed sampling
<<gf>> times.</p>
<p>J's two factors show where iLQR fell short. Its three-pad share was <<pd0>> to <<pd1>> below full annealed sampling's on the four
tasks, <<pi>> against <<pf>> on the twist. Its tracking factor differed from full annealed sampling's by <<td0>> to <<td1>>. The pad
term has no gradient and enters only iLQR's line search.</p>
<p>The plan is the nine servo targets, each held for 50 ms, over the 0.3 s horizon (<code>scripts/hom/inner_loop_ilqr.py</code>;
the method is set out on the <a href="<<mp>>">sampling MPC page</a>). Each iteration linearizes the planner's MuJoCo-Warp
model about the plan by forward differences, 40 one-step rollouts per step, and takes a Gauss-Newton backward pass.
Eight step sizes are then rolled out in parallel and scored with the full cost, and the cheapest is kept if it costs less
than the plan. Two iterations take <<ro2>> rollouts of the horizon per replanning; full annealed sampling takes 256. DDP adds the
dynamics' second derivatives to the backward pass. FDDP [<<ref2>>] treats an infeasible initial guess, and each plan here
starts from a rollout of the shifted plan.</p>
<p>On the six tuning pairs of each task, two iterations scored <<t2>> and four iterations <<t4>>. Annealed sampling 2 &#215; 64 over
0.3 s, at 128 rollouts per replanning, scored <<d64>> there, and full annealed sampling <<tf>>. The line search lowered the plan's cost
in <<acc>> % of the iterations and took the full step in <<fs>> %. iLQR draws no random numbers, but the GPU's sums
are not repeatable bit for bit. Run twice on the tuning pairs, a pair's J differed by a mean of <<rp>>. The largest
difference was <<rpx>>.
<<rpf>> For one hand alone a replanning took <<one>> s against
the 0.1 s interval. At the 50 ms step iLQR's targets moved at <<rt>> deg/s RMS and reversed <<rv>> times per joint per
second. Full annealed sampling's targets, counted per 10 ms period, moved at <<rtf>> deg/s and reversed <<rvf>> times on the same
pairs.</p>""",
               ref=REF_ILQR, ref2=REF_FDDP,
               pd0=f"{min(N['fac'][t]['full'][0] - N['fac'][t]['il'][0] for t in TASKS):.2f}",
               pd1=f"{max(N['fac'][t]['full'][0] - N['fac'][t]['il'][0] for t in TASKS):.2f}",
               pi=f"{N['fac']['twist']['il'][0]:.2f}", pf=f"{N['fac']['twist']['full'][0]:.2f}",
               td0=f"{min(N['fac'][t]['il'][1] - N['fac'][t]['full'][1] for t in TASKS):+.2f}",
               td1=f"{max(N['fac'][t]['il'][1] - N['fac'][t]['full'][1] for t in TASKS):+.2f}", j=lst(N["j20"]), jm=f"{m4(N['j20']):.3f}", tn=P.TAB[0] + 1,
               h=lst(N["h20"]), hm=f"{m4(N['h20']):.3f}", f=lst(N["f20"]), fm=f"{m4(N['f20']):.3f}",
               ah=where(N["vs_hom"], 1), bh=where(N["vs_hom"], -1), bf=where(N["vs_full"], -1),
               af=where(N["vs_full"], 1), dr=N["drop"][0], dn=N["drop"][1], drf=N["drop_full"][0],
               g=f"{N['gpu']:.0f}" if N["gpu"] else "&#8212;", gr=f"{N['gpu_res']:.0f}" if N["gpu_res"] else "&#8212;",
               gf=f"{N['gpu_full']:.0f}" if N["gpu_full"] else "&#8212;", mp=MPC_PAGE,
               ro2=f"{N['ro'][CFG_IL2]:.0f}", t2=lst(N["tune"][CFG_IL2]),
               t4=lst(t4) if all(v is not None for v in t4.values()) else "&#8212;",
               d64=lst(N["ref"]["d64"]), tf=lst(N["ref"]["full"]), acc=f"{100 * N['accept']:.0f}",
               fs=f"{100 * N['full_step']:.0f}",
               rp=(al([f"{N['rep'][t]:.3f} on the {TN[t].lower()}" for t in TASKS]) if N["rep"] else "&#8212;"),
               rpx=f"{N['rep_max']:.3f}",
               rpf=("" if not N["rep_flip"] else f"In {NUMW[N['rep_flip']]} of the pairs the object was dropped in one "
                    "run and held in the other."),
               one=f"{N['one']:.2f}" if N["one"] else "&#8212;", rt=f"{N['rate']:.1f}", rv=f"{N['rev']:.2f}",
               rtf=f"{N['rate_full']:.1f}", rvf=f"{N['rev_full']:.2f}")
    PD = N["pad"]
    rows = [[TN[t], f"{N['j20'][t]:.3f}", fd(N["vs_hom"][t]), fd(N["vs_full"][t]), fd(N["vs_res"][t])]
            + ([f"{PD['j20'][t]:.3f}", fd(PD["vs_il"][t]), fd(PD["vs_full"][t])] if PD else []) for t in TASKS]
    txt += table(["Task", "iLQR, mean J", "iLQR &#8722; HOM", "iLQR &#8722; full annealed sampling",
                  "iLQR &#8722; annealed-sampling residual on HOM"] + (["iLQR with the pad term, mean J",
                                                         "Pad term &#8722; iLQR without it",
                                                         "Pad term &#8722; full annealed sampling"] if PD else []), rows,
                 "iLQR at two iterations over 0.3 s on each task's 20 pairs, seeds 0 and 1: mean J and the paired "
                 "difference from HOM, full annealed sampling (4 &#215; 64) and the annealed-sampling residual on HOM's servo targets "
                 "(2 &#215; 32) on the same pairs and seeds, with its 95 % interval and the number of pairs where "
                 "the first was higher. The last three columns: iLQR with the smooth pad term.", num_from=1)
    if PD:
        sh = PD["share"]
        rest = [t for t in TASKS if t != "twist"]
        txt += fill(r"""<p>A smooth pad term was then added to iLQR's model (<code>inner_loop_mpc.py run --method ilqr
--il-pad</code>). Each pad's normal force $F$ adds the residual $\sqrt{w}\,\max(0,\ 1 - F/0.25\ \mathrm{N})$. Its square at
$F = 0$ equals the cost's charge for a pad below 0.05 N, and its Jacobian comes from the same finite differences. On the
twist J rose by <<dt>>, and the three-pad share from <<s0>> to <<s1>>; full annealed sampling's share was <<sf>>. The rest of the gap
to full annealed sampling on the twist was <<gf>>. On the <<rest>> the term changed J by <<r0>> to <<r1>>, and each 95 % interval
included zero. With the term iLQR's mean J over the four tasks was <<m>>, against full annealed sampling's <<fm>>. It dropped the
object in <<dr>> of <<dn>> runs.</p>""",
                    dt=fds(PD["vs_il"]["twist"]), s0=f"{sh['twist']['il']:.2f}", s1=f"{sh['twist']['pad']:.2f}",
                    sf=f"{sh['twist']['full']:.2f}", gf=fds(PD["vs_full"]["twist"]),
                    rest=al([TN[t].lower() for t in rest]),
                    r0=f"{min(PD['vs_il'][t]['d'] for t in rest):+.3f}", r1=f"{max(PD['vs_il'][t]['d'] for t in rest):+.3f}",
                    m=f"{m4(PD['j20']):.3f}", fm=f"{m4(N['f20']):.3f}", dr=PD["drop"][0], dn=PD["drop"][1])
    return txt


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
pairs, HOM's servos were at 95 % of their 0.18 N m limit or more in <<sat>> % of the joint-periods. Annealed sampling at
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


def one_hand_replan(R):
    """The replanning time of the annealed-sampling residual and of annealed sampling 2 x 32 alone for one hand (runs at --chunk 1 on two twist
    pairs, rows ..._chunk1): mean s, or None while either is missing."""
    v = {nm: [r["replan_s_mean"] for r in rows_of(R, MP.tstem(m, "twist", "_rank20", CFG_PICK) + "_chunk1")
              if r.get("replan_s_mean")] for nm, m in (("res", "res_anneal"), ("alone", "anneal"))}
    if not all(v.values()):
        return None
    n = {nm: sum(r.get("replans", 0) for r in rows_of(R, MP.tstem(m, "twist", "_rank20", CFG_PICK) + "_chunk1"))
         for nm, m in (("res", "res_anneal"), ("alone", "anneal"))}
    return {"res": float(np.mean(v["res"])), "alone": float(np.mean(v["alone"])), "budget": 10 * MP.PERIOD_S,
            "n": min(n.values())}


def sec_future(R):
    items = []
    one = one_hand_replan(R)
    if one is None:
        items.append(
            "The annealed-sampling residual on HOM's servo targets for one hand alone: <code>inner_loop_mpc.py run --method "
            "res_anneal --chunk 1</code>. If one hand's replanning at 2 &#215; 32 samples takes 0.1 s or less, the "
            "residual is a real-time controller that scored above HOM on three tasks. If it takes longer, the residual "
            "at 2 &#215; 16 samples would be timed and run on the 20 pairs of each task.")
    elif one["res"] <= one["budget"]:
        items.append(
            "The annealed-sampling residual on HOM's servo targets on the 100 pairs of each task at seeds 10 and 11 "
            "(<code>inner_loop_mpc.py run --method res_anneal --pairs _rank --seeds 10 11 --iters 2 --horizon 10 "
            "--knots 2 --n 32 --plan-dt 0.002</code>). If it stays above HOM by its 95 % interval on three tasks, it is "
            "the real-time controller for the bench.")
    else:
        items.append(
            f"The annealed-sampling residual at 2 &#215; 16 samples: one hand's replanning at 2 &#215; 32 took {one['res']:.2f} s "
            "against the 0.1 s interval. The smaller residual would be timed at <code>--chunk 1</code> and run on the "
            "20 pairs of each task; if it keeps J above HOM on three tasks in real time, it is the bench's controller.")
    Ns = grids_numbers(R)
    if Ns:
        items.append(
            "A two-stage search on each grid. Full annealed sampling re-scores HOM's 50 best layouts at seeds 0 and 1, and its first "
            "choice among them runs at seeds 10 and 11. The first stage's scores are in the grid runs above; the pair "
            "rule for the second is still to be written in <code>inner_loop_mpc.py pairs</code>. If that pick scores "
            "within 0.05 of full annealed sampling's first choice of the whole grid at seeds 10 and 11 on every task, the planner "
            "is needed only on HOM's 50 best layouts.")
    IL_ = ilqr_numbers(R)
    if IL_ and IL_.get("pad"):
        pdg = IL_["pad"]["gpu"]
        items.append(
            "iLQR with the pad term on the 100 pairs of each task at seeds 0 and 1 (<code>inner_loop_mpc.py run "
            "--method ilqr --il-pad --pairs _rank</code>)."
            + (f" It costs {pdg:.0f} times HOM's GPU time per twist rollout." if pdg else "")
            + " If its order of the 100 pairs agrees with full annealed sampling's at a higher &#961; than HOM's on three tasks, "
            "it can replace full annealed sampling in the search's second stage.")
    F5 = {t: five_numbers(R, t) for t in ("pinpush", "press")}
    if all(F5.values()):
        items.append(
            "Grasps chosen by their worst J over seeds 0-4 in place of the mean, then HOM at seeds 10 and 11 on the "
            "pin-push's and the press's 100 pairs. The minimum rule for <code>inner_loop_mpc.py pairs --rule rank5</code> "
            "is still to be written. With the mean over five placements HOM still dropped the cylinder in "
            f"{100 * F5['pinpush']['five']['drop'] / F5['pinpush']['five']['n']:.1f} % and "
            f"{100 * F5['press']['five']['drop'] / F5['press']['five']['n']:.1f} % of those runs. If the drop rate "
            "stays above 20 % under the worst-seed rule, the cylinder grasps of HOM's grid fail at new placements "
            "whichever rule picks them.")
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def sec_files():
    items = ["<code>scripts/hom/inner_loop_mpc.py</code>: HOM, its variants (<code>--preview</code>, "
             "<code>--prm-shared</code>, <code>--hom-param</code>), the planners, the fused planners (<code>res_*</code>, "
             "<code>hs_*</code>), the gradient planner (<code>grad</code>) and iLQR (<code>ilqr</code>) on any task's "
             "pair set",
             "<code>scripts/hom/inner_loop_mpc.py pairs</code>: <code>--rule all</code> and <code>--rule top --src "
             "_all</code> for the four grids, <code>--rule rank5</code> for the grasps chosen at five placements",
             "<code>scripts/hom/inner_loop_ilqr.py</code>: iLQR, its finite-difference linearization in batch, the "
             "Joseph-form backward pass and the parallel line search",
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
        "synthesis with MuJoCo, arXiv 2212.00541, 2022.",
        "[3] Li and Todorov, Iterative linear quadratic regulator design for nonlinear biological movement systems, "
        "ICINCO 2004.",
        "[4] Mastalli et al., Crocoddyl: an efficient and versatile framework for multi-contact optimal control (FDDP), "
        "ICRA 2020."]
REF_ILQR, REF_FDDP = 3, 4


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
                     f"{mean4('hom'):.3f}. Full annealed sampling at 4 &#215; 64 samples scored {mean4('full'):.3f}, at {rf:.0f} "
                     "times HOM's GPU time per rollout.")
        rt = [mean4(k) for k in ("c1", "rt30", "grad") if mean4(k) is not None]
        rr = [gpu_ratio(R, MP.tstem(m, "twist", "_rank", c)) for m, c in (("anneal", CFG_PICK), ("anneal", MP.CFG_RT30),
                                                                          ("grad", CFG_PICK))]
        if len(rt) == 3:
            parts.append(f"In real time, at {min(rr):.0f} to {max(rr):.0f} times HOM's GPU time, annealed sampling and a "
                         f"finite-difference gradient planner scored {min(rt):.3f} to {max(rt):.3f}.")
    N = {t: screen_numbers(R, t) for t in TASKS}
    if all(N.values()):
        g = [N[t]["halves"]["Annealed sampling 4 &#215; 64"][0] - N[t]["halves"]["Annealed sampling 4 &#215; 64"][1] for t in TASKS]
        parts.append(f"Under full annealed sampling, HOM's 50 best layouts scored {min(g):.2f} to {max(g):.2f} higher than 50 random "
                     "ones.")
    C = {t: MP.confirm_numbers(t) for t in TASKS}
    if all(C.values()):
        fh = max(c["dial_on_hom_s01"] - c["dial_on_hom"] for c in C.values())
        fd = max(c["dial_on_dial_s01"] - c["dial_on_dial"] for c in C.values())
        pr = C["press"]
        parts.append(f"At two fresh seeds, HOM's ten best pairs lost at most {fh:.3f} of their J under full annealed sampling, and the "
                     f"planner's own ten up to {fd:.3f}.")
        parts.append(f"At those seeds HOM dropped the press's cylinder in {pr['drop_hom'][0]} of {pr['drop_hom'][1]} runs, "
                     f"against {pr['drop_hom_s01'][0]} at seeds 0 and 1; full annealed sampling dropped it in "
                     + ("none." if pr["drop_dial"][0] == 0 else f"{pr['drop_dial'][0]}."))
    Ns = grids_numbers(R)
    if Ns:
        ts = list(Ns)
        rh = [Ns[t]["rho_hd"] for t in ts]
        miss = [t for t in ts if Ns[t]["out50"] > 5]
        parts.append(f"On every layout of the grids of {MP.tasks_list(ts)} the two controllers' J agreed with &#961; = "
                     f"{min(rh):.2f} to {max(rh):.2f}.")
        if miss:
            parts.append(f"HOM's grid missed full annealed sampling's best layouts on {MP.tasks_list(miss)}, where "
                         f"{al([NUMW[Ns[t]['out50']] for t in miss])} of the planner's ten best lay outside HOM's 50 "
                         "best.")
        fr = {t: Ns[t]["fresh"] for t in ts if Ns[t].get("fresh")}
        up = [t for t in fr if fr[t]["diff"][1] > 0]
        if up:
            by = al([f"{fr[t]['diff'][0]:.3f}" for t in up])
            same = [t for t in fr if t not in up]
            parts.append(f"At fresh seeds the planner's ten scored more than the grid's ten under full annealed sampling on "
                         f"{MP.tasks_list(up)}, by {by}.")
            if same:
                parts.append(f"On {MP.tasks_list(same)} the 95 % interval of the difference included zero.")
    Q = {t: repeat_numbers(R, t) for t in PATCH_TASKS}
    if all(Q.values()):
        parts.append(f"Run twice at the same seeds, a layout's J differed by a mean of "
                     f"{max(q['hom']['mean'] for q in Q.values()):.3f} or less under HOM and by "
                     f"{min(q['full']['mean'] for q in Q.values()):.2f} to {max(q['full']['mean'] for q in Q.values()):.2f} "
                     "under full annealed sampling.")
    second = []
    eff = variant_effects(R)
    if eff:
        up = sorted(((d[t]["d"], tag, t) for tag, _l, tasks, d in eff for t in tasks if d[t]["lo"] > 0), reverse=True)
        if up:
            second.append(f"Of {len(eff)} variants of HOM, {NUMW[len({u[1] for u in up})]} raised J on a task with a "
                          "95 % interval above zero.")
            more = [f"{LNAME[tag]} by {x:.3f} on the {TN[t].lower()}" for x, tag, t in up[1:]]
            second.append(f"{cap(LNAME[up[0][1]])} raised it by {up[0][0]:.3f} on the {TN[up[0][2]].lower()}"
                          + (", and " + al(more) if more else "") + ".")
        else:
            second.append(f"None of {len(eff)} variants of HOM raised J on a task with a 95 % interval above zero.")
    dr = {t: paired(rows_of(R, MP.tstem("res_anneal", t, "_rank20", CFG_PICK)), hom_rank(R, t)) for t in TASKS}
    grr = gpu_ratio(R, MP.tstem("res_anneal", "twist", "_rank20", CFG_PICK))
    if all(dr.values()) and grr:
        up_t = [t for t in TASKS if dr[t]["lo"] > 0]
        if up_t:
            second.append(f"An annealed-sampling residual on HOM's servo targets, at {grr:.0f} times HOM's GPU time per rollout, "
                          f"raised J by {al([f'{dr[t]['d']:.3f}' for t in up_t])} on {MP.tasks_list(up_t)}.")
    IL_ = ilqr_numbers(R)
    if IL_:
        m4 = (lambda d: float(np.mean([d[t] for t in TASKS])))
        second.append(f"iLQR with finite-difference linearization scored {m4(IL_['j20']):.3f} on the 20 pairs of each "
                      f"task, against HOM's {m4(IL_['h20']):.3f} and full annealed sampling's {m4(IL_['f20']):.3f}"
                      + (f", at {IL_['gpu']:.0f} times HOM's GPU time per rollout." if IL_["gpu"] else "."))
        if IL_.get("pad"):
            second.append(f"With a smooth pad term in its model iLQR scored {m4(IL_['pad']['j20']):.3f}, and "
                          f"{IL_['pad']['vs_il']['twist']['d']:.3f} more on the twist.")
    F5 = {t: five_numbers(R, t) for t in ("pinpush", "press")}
    if all(F5.values()):
        pc = (lambda d: f"{100 * d['drop'] / d['n']:.1f} %")
        second.append(f"Grasps chosen at five placements lowered HOM's drops at new placements from "
                      f"{pc(F5['pinpush']['three'])} to {pc(F5['pinpush']['five'])} on the pin-push and from "
                      f"{pc(F5['press']['three'])} to {pc(F5['press']['five'])} on the press.")
    TRN = trend_numbers(PK)
    second.append(f"The forces picked for HOM by CEM were below its defaults on every task: a median "
                  f"{TRN[('twist', 'f_tripod')]['med']:.2f} N against 2.5 N on the twist.")
    if not parts:
        parts.append("HOM against sampling MPC on the SR2 hand's four tasks; the planner runs are under way.")
    lede_t = "<p>" + " ".join(parts) + "</p>" + "<p>" + " ".join(second) + "</p>"
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
                ("screen", "HOM's best layouts under full annealed sampling", sec_screen(R)),
                ("grids", "Full annealed sampling on every layout of the four grids", sec_grids(R)),
                ("stability", "Stability of HOM's scores across seeds, layouts and controllers", sec_stability(R, GS)),
                ("patch", "Score landscapes of HOM and full annealed sampling around HOM's best layouts", sec_patch(R)),
                ("metrics", "Where HOM loses J against the planner", sec_metrics(R)),
                ("variants", "Variants of HOM", sec_variants(R)),
                ("trends", "Forces and contact modes picked for HOM per layout", sec_trends(PK, figs)),
                ("fusion", "Planners fused with HOM", sec_fusion(R)),
                ("ilqr", "iLQR with finite-difference linearization on the four tasks", sec_ilqr(R)),
                ("future", "Future work", sec_future(R)),
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

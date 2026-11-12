#!/usr/bin/env python3
"""Builds docs/experiments/20261112-inner_loop_benchmark/20261112-inner_loop_benchmark.html: published inner-loop
evaluators against HOM on the SR2 layout lattice, each judged by the J of the hand it would pick and by its wall time per
layout on this workstation (owner, 2026-11-12: "evaluate some of the other inner loop controls we tried? Karma, uhas?";
"show these other methods ... are both worse and slower than what we have cooked up with gpu parallelized HOM. We can let
the other methods have the sphere packed model and warp").

Data: the HOM grids under J (rows/grid2dff_v4b_sv25r of the foundational-pose folder, through hom_search_qd.lookup) and
their timing rows; KaRMA's rows (scripts/hom/karma_sr2.py: rows/karma_sph40.jsonl, karma_sph25.jsonl); UHAS's
training progress and evaluation rows (scripts/hom/uhas_sr2.py: rows/uhas_<task>_<run>_progress.jsonl and
rows/uhas_<task>_<run>_test.jsonl); the film of scripts/hom/uhas_sr2_film.py.

    .venv/bin/python scripts/hom/inner_loop_benchmark_page.py
"""
from __future__ import annotations

import glob
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
OUT = D / "20261112-inner_loop_benchmark.html"
TEX_CACHE = D / "texsvg_cache.json"
GRID = P.ROWS / "grid2dff_v4b_sv25r"
SEARCH = "../20261029-foundational_pose_review/20261109-hom_search_methods.html"
TRACK = "../20261029-foundational_pose_review/20261109-hom_tracking_score.html"
fill, table, esc = P.fill, P.table, P.esc
TASKS = (("twist", 40.0, "sph40"), ("roll", 40.0, "sph40"), ("pinpush", 25.0, "sph25"), ("press", 25.0, "sph25"))
TNAME = {"twist": "twist", "roll": "roll", "pinpush": "pin-push", "press": "press"}
OBJ = {"twist": "40 mm sphere", "roll": "40 mm sphere", "pinpush": "25 mm cylinder", "press": "25 mm cylinder"}
REFS = [
    '[1] M. F. Peticco and P. Agrawal, "A Kinematic Metric for Fine Manipulation Ability in Robotic Hands" (KaRMA), '
    'IEEE/RSJ IROS 2026, arXiv 2605.15548; code github.com/mfpeticco/karma-hand-metric, commit c8d3698.',
    '[2] L. F. Casas, R. Teal, K. Shah, A. Tadepalli, W. Jin and Y. Xiang, "Cross-Embodiment Robot Manipulation via a '
    'Unified Hand Action Space" (UHAS), arXiv 2607.03570, 2026; code irvlutd.github.io/UHAS.',
    '[3] C. Schwarke, M. Mittal, N. Rudin, D. Hoeller and M. Hutter, "RSL-RL: A Learning Library for Robotics '
    'Research", arXiv 2509.10771, 2025.',
    '[4] B. Yang, L. Jiang, W. Wu, M. Cheng and K. Gong, "Evolving Robot Morphology with Manipulation Learning Inspired '
    'by Human Hand Evolution", IEEE Transactions on Evolutionary Computation, 2026.',
]


# ------------------------------------------------------------------------------------------------ data
def j_lookup(task, size):
    """{(x_sep, y_sep): (J of the best grasp, rollouts)} of the HOM grid (hom_search_qd.lookup)."""
    import hom_search_qd as Q
    return Q.lookup(task, size)[0]


def j_rows(task, size):
    """The HOM grid's J per (layout, grasp): mean over seeds 0-2."""
    import hom_landscape as L
    import hom_rows
    out = {}
    for x in (GRID / f"{task}_d{size:g}.jsonl").read_text().splitlines():
        r = hom_rows.rescale_metrics(json.loads(x))
        if r.get("status", "ok") != "ok":
            continue
        d = r["d"]
        k = (round(d[2] - d[0], 3), round(d[3] - d[5], 3), int(r.get("grasp", 0)))
        out.setdefault(k, []).append((L.track_score(task, r), bool(r.get("held"))))
    return out


def karma(variant):
    p = ROWS / f"karma_{variant}.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return {(r["x_sep"], r["y_sep"]): r for r in rows if "error" not in r}


def hom_timing():
    """GPU seconds per layout of each HOM grid (build, run and score of every batch over the layouts run)."""
    out = {}
    for task, size, _v in TASKS:
        T = [json.loads(x) for x in (GRID / f"{task}_d{size:g}_timing.jsonl").read_text().splitlines()]
        calls = {json.loads(x)["call"] for x in (GRID / f"{task}_d{size:g}.jsonl").read_text().splitlines()}
        tot = sum(t["build_s"] + t["run_s"] + t["score_s"] for t in T)
        out[task] = {"s_per_layout": tot / len(calls), "layouts": len(calls), "rollouts": sum(t["worlds"] for t in T),
                     "minutes": tot / 60.0}
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def karma_stats():
    """Per task: KaRMA-T and KaRMA-R against J over the layouts KaRMA ran, and the J of each score's best layout."""
    out = {}
    for task, size, var in TASKS:
        K = karma(var)
        if len(K) < 10:
            continue
        look = j_lookup(task, size)
        ks = [k for k in K if k in look and look[k][1] > 0]
        J = np.array([look[k][0] for k in ks])
        rec = {"layouts": len(ks), "grid_best": max(f for f, n in look.values() if n > 0),
               "j_best_layout": list(ks[int(np.argmax(J))]), "j_best": float(J.max())}
        for name in ("karma_t", "karma_r"):
            v = np.array([K[k][name] for k in ks])
            rho, p = spearman(v, J)
            top = ks[int(np.argmax(v))]
            n10 = max(1, int(round(0.1 * len(ks))))
            topk = set(np.argsort(-v)[:n10])
            topj = set(np.argsort(-J)[:n10])
            rec[name] = {"rho": rho, "p": p, "best_layout": list(top), "best_J": float(look[top][0]),
                         "tenth_shared": len(topk & topj) / n10, "J_of_top_tenth": float(J[list(topk)].mean())}
        rec["secs_median"] = float(np.median([K[k]["secs"] for k in ks]))
        out[task] = rec
    return out


def uhas_runs():
    """The UHAS evaluation rows of each run: rows/uhas_<task>_<run>_test.jsonl."""
    out = {}
    for p in sorted(glob.glob(str(ROWS / "uhas_*_test.jsonl"))):
        rows = [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
        if rows:
            out[Path(p).stem] = rows
    return out


def uhas_stats(rows):
    """UHAS against HOM on the evaluated layouts: per layout the best grasp's mean J over seeds 0-2 under each."""
    task = rows[0]["task"]
    size = float(rows[0]["d_mm"])
    hom = j_rows(task, size)
    per = {}
    for r in rows:
        d = r["d"]
        k = (round(d[2] - d[0], 3), round(d[3] - d[5], 3), int(r.get("grasp", 0)))
        per.setdefault(k, []).append((r["J"], bool(r.get("held"))))
    lay = {}
    for (x, y, g), v in per.items():
        u = float(np.mean([j for j, _h in v]))
        h = float(np.mean([j for j, _h in hom.get((x, y, g), [])])) if (x, y, g) in hom else float("nan")
        L_ = lay.setdefault((x, y), {"uhas": [], "hom": []})
        L_["uhas"].append(u)
        L_["hom"].append(h)
    best = {k: (max(v["uhas"]), np.nanmax(v["hom"])) for k, v in lay.items()}
    ks = sorted(best)
    U = np.array([best[k][0] for k in ks])
    H = np.array([best[k][1] for k in ks])
    held_u = float(np.mean([h for v in per.values() for _j, h in v]))
    held_h = float(np.mean([h for k in per for _j, h in hom.get(k, [])]))
    rho = spearman(U, H) if np.std(U) > 0 else (float("nan"), float("nan"))
    return {"task": task, "layouts": len(ks), "rollouts": len(rows), "uhas_mean": float(U.mean()),
            "hom_mean": float(H.mean()), "uhas_best": float(U.max()), "hom_best": float(H.max()),
            "uhas_pick": list(ks[int(np.argmax(U))]), "hom_of_uhas_pick": float(H[int(np.argmax(U))]),
            "held_uhas": held_u, "held_hom": held_h, "rho": rho[0], "U": U.tolist(), "H": H.tolist(),
            "keys": [list(k) for k in ks]}


# ------------------------------------------------------------------------------------------------ figures
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 14,
                         "mathtext.fontset": "stix"})
    return plt


def fig_karma(St):
    """Per task: J and KaRMA-T over the layouts KaRMA ran (the 10 mm sub-lattice and J's ten best), with the best
    layout of each."""
    plt = plt_setup()
    tasks = [t for t, _s, _v in TASKS if t in St]
    fig, axs = plt.subplots(2, len(tasks), figsize=(4.0 * len(tasks), 7.6), squeeze=False)
    for j, task in enumerate(tasks):
        size, var = [(s, v) for t, s, v in TASKS if t == task][0]
        K, look = karma(var), j_lookup(task, size)
        ks = [k for k in K if k in look and look[k][1] > 0]
        X = np.array([k[0] for k in ks])
        Y = np.array([k[1] for k in ks])
        for i, (vals, lab, cm) in enumerate(((np.array([look[k][0] for k in ks]), "J (HOM)", "viridis"),
                                             (np.array([K[k]["karma_t"] for k in ks]), "KaRMA-T", "magma"))):
            ax = axs[i, j]
            sc = ax.scatter(X, Y, c=vals, cmap=cm, s=46, marker="s", edgecolors="none")
            b = int(np.argmax(vals))
            ax.plot(X[b], Y[b], "*", ms=17, mfc="white", mec="k")
            ax.set_xlim(35, 165)
            ax.set_ylim(45, 175)
            ax.set_aspect("equal")
            ax.set_xlabel("$x_{sep}$ (mm)")
            if j == 0:
                ax.set_ylabel("$y_{sep}$ (mm)")
            ax.set_title(f"{TNAME[task].capitalize()}, {OBJ[task]}: {lab}", fontsize=13)
            fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = MEDIA / "20261112-karma_maps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


VARIANTS = {"handover": ("UHAS from HOM's handover", "#0f766e"), "open": ("UHAS from the open hand", "#b45309")}


def variant_of(name):
    return "open" if "_open_" in name else "handover"


def fig_uhas(runs):
    """UHAS's J proxy against training time, and UHAS's J against HOM's on the test layouts, for each variant."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8))
    lim = 0.05
    for v, (S, prog) in runs.items():
        lab, col = VARIANTS[v]
        if prog:
            axs[0].plot([r["minutes"] for r in prog], [r.get("J_proxy") or 0.0 for r in prog], "-o", color=col, ms=3,
                        label=lab)
        axs[1].scatter(S["H"], S["U"], s=18, color=col, label=lab)
        lim = max(lim, max(S["H"] + S["U"]) * 1.05)
    axs[0].set_xlabel("Training time on the GPU (min)")
    axs[0].set_ylabel("J proxy of finished world episodes")
    axs[0].set_title("Training on the twist", fontsize=13)
    axs[0].legend(loc="upper left", fontsize=12)
    axs[0].grid(color="#eeeeee")
    ax = axs[1]
    ax.plot([0, lim], [0, lim], ":", color="#999999")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel("J under HOM (best grasp, seeds 0-2)")
    ax.set_ylabel("J under UHAS (best grasp, seeds 0-2)")
    ax.set_title("Test layouts of the twist", fontsize=13)
    ax.grid(color="#eeeeee")
    fig.tight_layout()
    path = MEDIA / "20261112-uhas_vs_hom.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------ sections
def sec_terms():
    terms = [
        ("Layout, lattice", "the SR2 hand's symmetric tripod in two coordinates: $x_{sep}$, from the thumb's mount to the "
         "pair's, and $y_{sep}$, from the index's mount to the middle's. The lattice has 625 layouts in steps of 5 mm."),
        ("Task", "the twist and the roll of a 40 mm sphere, and the pin-push and the press of a 25 mm cylinder. Each "
         "commands a rotation of the object (<a href=\"" + TRACK + "\">tracking-score page</a>)."),
        ("J, the tracking score", "one minus the ratio of two RMS angles over the task: the angle between the object's "
         "orientation and the commanded one, and the commanded rotation. It is at least 0 and is multiplied by the share "
         "of the task with all three pads above 0.05 N. It is 0 when the object drops 15 mm "
         "(<code>hom_landscape.track_score</code>). A layout's J is its best grasp's mean over seeds 0-2. J for every "
         "layout comes from the HOM grids, on the hand-design talk's settings: the SCS0009 plant at 0.72 N&#8201;m/rad, "
         "2.5 MPa pads and MuJoCo-Warp 3.6."),
        ("Inner loop", "what turns one layout into a score in co-design: a controller with its simulation (HOM, UHAS), or "
         "a metric of the layout alone (KaRMA). The layouts are chosen by the outer loop (<a href=\"" + SEARCH + "\">search "
         "methods page</a>)."),
        ("HOM", "the hand's model-based in-hand manipulation controller of the talk, run on every world of a MuJoCo-Warp "
         "batch; no training."),
        ("KaRMA", "a kinematic metric of a pinch on a sphere between the thumb and the index [1]. KaRMA-T is the volume "
         "of sphere-centre positions reached by rolling without letting go, normalized by the hand's size. KaRMA-R is "
         "the share of 228 orientation bins reached."),
        ("UHAS", "a learned controller for many hands [2]: one policy acting through a canonical sphere whose deformation is "
         "mapped to each hand's joints by cascade inverse kinematics, trained by PPO."),
        ("Single-hand UHAS", "the same policy and training on one layout alone, the paper's single-hand baseline; on our "
         "lattice it is a policy trained per layout, the usual reinforcement-learning inner loop."),
        ("Spearman's rho", "the rank correlation of two scores over the same layouts, from -1 to 1."),
        ("J of a pick", "the HOM grid's J of the layout ranked first by an evaluator, the J a designer would get by "
         "choosing with it."),
        ("Wall time per layout", "on this workstation: one CPU core of 20, and one RTX 4070 Ti SUPER of 16 GB."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_methods():
    return r"""<p>Every evaluator scores the same layouts of the lattice for the same tasks. HOM's scores are the HOM
grids of the talk: every layout with up to 16 grasps from the pose solver at seeds 0-2, 48 rollouts a layout.</p>
<p>KaRMA ran as published [1], its code unmodified, in its own conda environment
(<code>scripts/hom/karma_sr2.py</code>). The SR2 hand at each layout was exported as a URDF: each finger's three joints
with their axes and ranges, and a fixed fingertip link at the pad's distal end. Its test sphere was set to the tasks'
objects, 40 mm for the twist and the roll and 25 mm for the pin-push and the press. Its finger radius was set to 15.1 mm,
the pad's largest distance from the distal link's axis, and its scaling to the hand's size was switched off. KaRMA ran on
a 10 mm sub-lattice and on the ten best layouts by J of each task.</p>
<p>UHAS was ported to the HOM grids' MuJoCo-Warp scenes, with the same sphere-pad fingertips, plant, objects, grasps and
commanded rotations (<code>scripts/hom/uhas_sr2_cik.py</code>, <code>scripts/hom/uhas_sr2.py</code>). The sphere, its
deformation by two driving vectors per finger and the cascade inverse kinematics follow the paper and its code. The
torch kinematics of the SR2 finger agree with MuJoCo's to 0.0001 mm. HOM drives the approach, the closing, the squeeze
and the lift, the first 2.0 s, as in the grids. From there the policy drives the fingers at HOM's 10 ms period. Its
observations are UHAS's, without noise, plus the commanded angular velocity, because the goal moves. Its reward is
UHAS's, with the commanded rotation as the goal and J's three-pad factor added. PPO ran with UHAS's settings in RSL-RL
[3]. After a drop, a world restarts from a pool of HOM handovers in which the object was held, as UHAS's environment
restarts a fallen cube. Training used 418 layouts and their grasps; evaluation used the other 109, which include the five
best by J.</p>"""


def sec_karma(St, HT):
    if not St:
        return ""
    rows = []
    for task, _s, _v in TASKS:
        if task not in St:
            continue
        r = St[task]
        rows.append([TNAME[task].capitalize(), r["layouts"], f"{r['grid_best']:.2f}",
                     f"{r['karma_t']['rho']:.2f}", f"{r['karma_r']['rho']:.2f}",
                     f"({r['karma_t']['best_layout'][0]:g}, {r['karma_t']['best_layout'][1]:g})",
                     f"{r['karma_t']['best_J']:.2f}", f"{r['karma_r']['best_J']:.2f}",
                     f"{100 * r['karma_t']['tenth_shared']:.0f}"])
    tw = St.get("twist")
    t = ""
    if tw:
        t += fill(r"""<p>On the twist the layouts were ranked by KaRMA-T much as by J (Spearman's rho <<rt>> over <<n>>
layouts), with the best layout elsewhere. KaRMA-T's first choice, (<<bx>>, <<by>>) mm, scores J <<bj>> on the twist; the
grid's best scores <<gb>> (Table <<tab>>, Figure <<fig>>). Of the tenth of layouts ranked highest by KaRMA-T, <<sh>> %
are in J's highest tenth. KaRMA-R, the share of orientations reached, has a lower rank correlation with J, rho
<<rr>>.</p>""",
                  rt=f"{tw['karma_t']['rho']:.2f}", n=tw["layouts"], bx=f"{tw['karma_t']['best_layout'][0]:g}",
                  by=f"{tw['karma_t']['best_layout'][1]:g}", bj=f"{tw['karma_t']['best_J']:.2f}",
                  gb=f"{tw['grid_best']:.2f}", tab=P.TAB[0] + 1, fig=P.FIG[0] + 1,
                  sh=f"{100 * tw['karma_t']['tenth_shared']:.0f}", rr=f"{tw['karma_r']['rho']:.2f}")
    t += table(["Task", "Layouts", "Grid's best J", "rho, KaRMA-T", "rho, KaRMA-R", "KaRMA-T's first choice (mm)",
                "Its J", "J of KaRMA-R's first", "Top tenth shared (%)"], rows,
               "KaRMA against J over the layouts KaRMA ran (a 10 mm sub-lattice and J's ten best of each task). rho: "
               "Spearman's rank correlation with J. Top tenth shared: the share of KaRMA-T's highest tenth that is also "
               "in J's highest tenth.", num_from=1)
    t += P.img(fig_karma(St), "J of each layout under HOM (top) and KaRMA-T (bottom), on the layouts KaRMA ran; the "
               "star marks each score's best layout. <code>scripts/hom/karma_sr2.py</code>.")
    secs = [St[k]["secs_median"] for k in St]
    t += fill(r"""<p>A KaRMA run lasted a median of <<secs>> s per layout on one CPU core, with eight runs at a time
beside a training run on the GPU. HOM scored a layout in <<h0>> to <<h1>> s of the GPU: 48 rollouts with the batch's
build and scoring. Its grasps come from the pose solver beforehand, at 2.3 s per layout on one core for the sphere and
4.5 s for the cylinder.</p>""",
              secs=(f"{min(secs):.0f}" if round(min(secs)) == round(max(secs)) else f"{min(secs):.0f} to {max(secs):.0f}"),
              h0=f"{min(v['s_per_layout'] for v in HT.values()):.1f}",
              h1=f"{max(v['s_per_layout'] for v in HT.values()):.1f}")
    return t


def uhas_progress(run_name):
    p = ROWS / f"uhas_progress_{run_name}.jsonl"
    src = ROOT / "results/uhas_sr2" / run_name / "progress.jsonl"
    if src.exists():                                   # results/ is outside git: keep a copy beside the page
        p.write_text(src.read_text())
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def uhas_eval_seconds(run_name):
    """Wall time of the evaluation (its log's last line: rollouts and seconds)."""
    m = None
    for log in glob.glob(str(ROOT / "logs/20261112-uhas_sr2/eval_*.log")):
        for line in Path(log).read_text().splitlines():
            if run_name in line and "rollouts in" in line:
                m = re.search(r"(\d+) rollouts in (\d+) s", line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def run_of(name):
    return name[len("uhas_twist_"):-len("_test")] if name.startswith("uhas_twist_") else name


def sec_uhas(UR):
    main = {k: v for k, v in UR.items() if "_x" not in k}
    if not main:
        return ""
    runs, rows = {}, []
    for name, rr in sorted(main.items()):
        v = variant_of(name)
        S = uhas_stats(rr)
        prog = uhas_progress(run_of(name))
        runs[v] = (S, prog)
        last = prog[-1] if prog else {}
        rows.append([VARIANTS[v][0], f"{last.get('minutes', 0):.0f}", f"{last.get('transitions', 0) / 1e6:.0f}",
                     f"{100 * S['held_uhas']:.0f}", f"{S['uhas_mean']:.3f}", f"{S['uhas_best']:.2f}",
                     f"({S['uhas_pick'][0]:g}, {S['uhas_pick'][1]:g})", f"{S['hom_of_uhas_pick']:.2f}"])
    S0 = next(iter(runs.values()))[0]
    rows.append(["HOM", "none", "", f"{100 * S0['held_hom']:.0f}", f"{S0['hom_mean']:.3f}", f"{S0['hom_best']:.2f}",
                 "", ""])
    S, prog = runs.get("handover") or next(iter(runs.values()))
    last = prog[-1] if prog else {}
    t = fill(r"""<p>UHAS trained for <<mins>> min on the GPU from HOM's handover: <<its>> PPO iterations of 2,048 worlds and
16 steps, <<tr>> M transitions, <<pct>> % of the paper's 15,000 iterations of 8,192 environments. On the <<nl>> test
layouts it held the object to the end in <<hu>> % of <<nr>> rollouts; HOM held it in <<hh>> %. Its mean J over the
layouts, each at its best grasp, was <<mu>>; HOM's was <<mh>> (Table <<tab>>, Figure <<fig>>). The layout UHAS ranks
first scores <<hp>> under HOM, where HOM's best test layout scores <<bh>>.</p>""",
             mins=f"{last.get('minutes', 0):.0f}", its=f"{last.get('iteration', 0) + 1:,}",
             tr=f"{last.get('transitions', 0) / 1e6:.0f}",
             pct=f"{100 * last.get('transitions', 0) / (15000 * 8192 * 16):.1f}", nl=S["layouts"],
             hu=f"{100 * S['held_uhas']:.0f}", nr=f"{S['rollouts']:,}", hh=f"{100 * S['held_hom']:.0f}",
             mu=f"{S['uhas_mean']:.3f}", mh=f"{S['hom_mean']:.3f}", tab=P.TAB[0] + 1, fig=P.FIG[0] + 1,
             hp=f"{S['hom_of_uhas_pick']:.2f}", bh=f"{S['hom_best']:.2f}")
    t += table(["Controller", "Training (min)", "Transitions (M)", "Held to the end (%)", "Mean J", "Best layout's J",
                "First choice (mm)", "Its J under HOM"], rows,
               "The test layouts of the twist under each controller: every layout with its grasps at seeds 0-2. Mean J: "
               "over the layouts, each at its best grasp. Held: share of rollouts with the object held to the end.",
               num_from=1)
    t += P.img(fig_uhas(runs), "Left: the J proxy of the world episodes finished between checkpoints (1 - RMS error / "
               "RMS commanded rotation, times the three-pad share, 0 if dropped), against training time. Right: each test "
               "layout's J under UHAS against under HOM, each at its best grasp, mean of seeds 0-2.")
    for v, tag in (("handover", ""), ("open", "_open")):
        info = D / "rows" / f"film_hom_uhas_twist{tag}.json"
        if v in runs and info.exists():
            fi = json.loads(info.read_text())
            t += P.film(MEDIA / f"20261112-film_hom_uhas_twist{tag}.mp4",
                        f"The twist on the test set's best layout, ({fi['layout'][0]:g}, {fi['layout'][1]:g}) mm, "
                        f"grasp {fi['grasp']}, seed {fi['seed']}: HOM (left, J {fi['hom']['J']:.2f}) and "
                        f"{VARIANTS[v][0]} ({fi['checkpoint']}, right, J {fi['uhas']['J']:.2f}), in the same "
                        "MuJoCo-Warp world. <code>scripts/hom/uhas_sr2_film.py</code>.")
    return t


def sec_express():
    """How near UHAS's sphere actions come to HOM's handover grasps (uhas_sr2.py express)."""
    fits = {}
    for n in ("8x128", "16x256", "32x512"):
        q = ROWS / f"uhas_express_twist_{n}.json"
        if q.exists():
            fits[n] = json.loads(q.read_text())
    if "32x512" not in fits:
        return ""
    f = fits["32x512"]
    e, j = f["contact_err_mm"], f["joint_err_deg_median"]
    rows = [[n.replace("x", " &#215; "), f"{v['contact_err_mm']['median']:.1f}", f"{v['contact_err_mm']['p90']:.1f}",
             f"{100 * v['share_within_2mm']:.0f}", f"{100 * v['share_within_5mm']:.0f}"] for n, v in fits.items()]
    t = fill(r"""<p>UHAS's actions reach only part of the grasps HOM hands over. For <<h>> handovers of the twist on
training layouts in which the object was held, the sphere action nearest each grasp was found by the cross-entropy
method on the pads' contact points. At the largest search the contact points lay a median <<m>> mm from HOM's (90th
percentile <<p>> mm); <<s2>> % of grasps came within 2 mm and <<s5>> % within 5 mm (Table <<tab>>). The pip joint was
off by a median <<pip>> deg and the mcp by <<mcp>> deg: UHAS's rule fixes each link against the deformed sphere, which
ties the pip to the mcp. The median stopped falling between the two larger searches, so it is a limit of the action
space. A policy acting from the handover therefore changes about half of the grasps by more than 5 mm in its first
period.</p>""", h=f["held_at_handover"], m=f"{e['median']:.1f}", p=f"{e['p90']:.1f}",
             s2=f"{100 * f['share_within_2mm']:.0f}", s5=f"{100 * f['share_within_5mm']:.0f}", tab=P.TAB[0] + 1,
             pip=f"{j['pip']:.0f}", mcp=f"{j['mcp']:.0f}")
    t += table(["Search (iterations &#215; candidates)", "Median (mm)", "90th percentile (mm)", "Within 2 mm (%)",
                "Within 5 mm (%)"], rows,
               f"Distance from HOM's handover contact points to those of the nearest sphere action, over "
               f"{f['held_at_handover']} held handovers of the twist (<code>uhas_sr2.py express</code>).", num_from=1)
    return t


def sec_single(UR):
    one = {k: v for k, v in UR.items() if "_x" in k}
    if not one:
        return ""
    name, rows = sorted(one.items())[-1]
    S = uhas_stats(rows)
    m = re.search(r"_x([\d.]+)_y([\d.]+)_", name)
    return fill(r"""<p>Trained on the layout (<<x>>, <<y>>) mm alone for 30 min, UHAS reached J <<u>> there at its best
grasp, and held the object in <<hu>> % of the 48 rollouts; HOM's J there is <<h>>, with the object held in <<hh>> %.
Trained this way on every layout, the lattice's 527 layouts of one task would take <<days>> days of the GPU.</p>""",
                x=m.group(1), y=m.group(2), u=f"{S['uhas_best']:.2f}", hu=f"{100 * S['held_uhas']:.0f}",
                h=f"{S['hom_best']:.2f}", hh=f"{100 * S['held_hom']:.0f}", days=f"{527 * 30 / 60 / 24:.0f}")


# ------------------------------------------------------------------------------------------------ page
def render_tex(page):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    return P.TEX_RE.sub(lambda m: next(svgs), page)


def sec_cost(St, HT, UR):
    """Wall time per layout and for one task's lattice, by evaluator."""
    tw = HT["twist"]
    rows = [["HOM (this study)", "none", "2.3 s, one core", f"{tw['s_per_layout']:.1f} s", f"{tw['minutes']:.0f} min"]]
    if St:
        km = float(np.median([St[k]["secs_median"] for k in St]))
        rows.append(["KaRMA [1]", "none", "its own seed search", f"{km:.0f} s, one core",
                     f"{527 * km / 8 / 3600:.1f} h on 8 cores"])
    main = {k: v for k, v in UR.items() if "_x" not in k and variant_of(k) == "handover"}
    if main:
        name = sorted(main)[-1]
        run = run_of(name)
        prog = uhas_progress(run)
        ev = uhas_eval_seconds(run)
        mins = prog[-1]["minutes"] if prog else float("nan")
        per = (ev[1] / (ev[0] / 48.0)) if ev else float("nan")
        rows.append(["UHAS [2]", f"{mins:.0f} min, once", "HOM's, 2.3 s, one core", f"{per:.1f} s",
                     f"{mins:.0f} min + {527 * per / 60:.0f} min"])
    one = {k: v for k, v in UR.items() if "_x" in k}
    if one:
        rows.append(["Single-hand UHAS", "30 min per layout", "HOM's, 2.3 s, one core", "30 min",
                     f"{527 * 30 / 60 / 24:.0f} days"])
    return table(["Evaluator", "Training", "Grasps", "Score of one layout", "The twist's 527 layouts"], rows,
                 "Wall time on this workstation (one RTX 4070 Ti SUPER, 20 CPU cores) for the twist on the 40 mm sphere. "
                 "HOM: the grid's batches (build, run and scoring) over its layouts, 48 rollouts each. UHAS: its training, "
                 "and its evaluation's batches over the test layouts, 48 rollouts each.", num_from=1)


def sec_limits():
    return r"""<ul>
<li>UHAS trained once per setting, on the twist only, within a fixed budget of GPU time. Its paper trained 15,000
iterations of 8,192 environments, about 2 billion transitions, on another task and simulator.</li>
<li>UHAS's reward was adapted to a moving goal: the paper's bonus per goal reached has no counterpart, and J's three-pad
factor was added. UHAS's observations were exact; HOM's measurements had the grids' noise model.</li>
<li>KaRMA scores a pinch on a sphere between two fingers. The pin-push and the press move a cylinder, and the twist and
the roll use all three fingers.</li>
<li>KaRMA ran on 145 and 128 layouts: the 10 mm sub-lattice and J's ten best per task. The lattices have 527 and 437
reached layouts.</li>
</ul>"""


def sec_future():
    return r"""<ul>
<li>UHAS on the roll, the pin-push and the press, and a second training seed on the twist (<code>uhas_sr2.py train
--task roll</code>). The ranking of HOM above UHAS fails on a task where UHAS's mean test J exceeds HOM's.</li>
<li>UHAS started from HOM's demonstrations: the policy's sphere actions fitted to HOM's joint commands on the training
layouts, then PPO as above. It fails to change the conclusion if its test J stays below HOM's at the same budget.</li>
<li>KaRMA on every layout of the lattice (<code>karma_sr2.py batch --variant sph40</code>, about 3 h on 8 cores), and
its published scaled variant (<code>--variant scaled</code>). The conclusion fails if KaRMA's first choice comes within
0.05 of the grid's best J on any task.</li>
</ul>"""


def sec_files():
    return r"""<ul>
<li>KaRMA: <code>scripts/hom/karma_sr2.py</code>; rows <code>rows/karma_sph40.jsonl</code>,
<code>rows/karma_sph25.jsonl</code>; queue <code>logs/20261112-karma_sr2/queue.sh</code>; the code in
<code>external/karma</code> (commit c8d3698, environment <code>karma-hand-metric</code>).</li>
<li>UHAS: <code>scripts/hom/uhas_sr2_cik.py</code> (sphere controller), <code>scripts/hom/uhas_sr2.py</code> (environment,
training, evaluation), <code>scripts/hom/uhas_sr2_film.py</code>; rows <code>rows/uhas_*_test.jsonl</code>,
<code>rows/uhas_progress_*.jsonl</code>; runs <code>results/uhas_sr2/</code> on the workstation; queues
<code>logs/20261112-uhas_sr2/</code>.</li>
<li>HOM: the grids <code>docs/experiments/20261029-foundational_pose_review/rows/grid2dff_v4b_sv25r/</code>.</li>
<li>This page: <code>scripts/hom/inner_loop_benchmark_page.py</code>.</li>
</ul>"""


def lede(St, HT, UR):
    tw = St.get("twist")
    parts = []
    hs = HT["twist"]["s_per_layout"]
    if tw:
        parts.append(fill("""HOM scored a layout of the SR2 lattice in <<h>> s of the GPU; KaRMA took <<k>> s on one CPU core
and UHAS <<u>>. KaRMA's first choice on the twist scores J <<kj>> under HOM, where the best layout scores <<gb>>.""",
                          h=f"{hs:.1f}", k=f"{tw['secs_median']:.0f}", u="a training run first",
                          kj=f"{tw['karma_t']['best_J']:.2f}", gb=f"{tw['grid_best']:.2f}"))
    main = {k: v for k, v in UR.items() if "_x" not in k and variant_of(k) == "handover"}
    if main:
        S = uhas_stats(sorted(main.items())[-1][1])
        parts.append(fill("""On the 109 test layouts of the twist, UHAS's mean J was <<u>> and HOM's <<h>>.""",
                          u=f"{S['uhas_mean']:.3f}", h=f"{S['hom_mean']:.3f}"))
    text = " ".join(parts) or "Results pending."
    return text, re.sub(r"\s+", " ", text)


def main():
    St, HT = karma_stats(), hom_timing()
    UR = uhas_runs()
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("methods", "The evaluators and what each was given", sec_methods()),
                ("karma", "KaRMA on the layout lattice", sec_karma(St, HT)),
                ("express", "UHAS's actions and HOM's grasps", sec_express()),
                ("uhas", "UHAS on the twist", sec_uhas(UR)),
                ("single", "UHAS trained on one layout", sec_single(UR)),
                ("cost", "Wall time per layout", sec_cost(St, HT, UR)),
                ("limits", "Limitations", sec_limits()),
                ("future", "Future work", sec_future()),
                ("refs", "References and files", '<ol class="refs">' + "".join(
                    f"<li>{r[r.index(']') + 2:]}</li>" for r in REFS) + "</ol>" + sec_files())]
    sections = [x for x in sections if x[2]]
    lede_t, summary = lede(St, HT, UR)
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|s\b|min\b|h\b)", "\\1&nbsp;\\2", s_))
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{unit(b)}</section>'
                     for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("KaRMA and UHAS on the SR2 lattice, 2026-11-12, against the HOM grids of the hand-design talk (SCS0009 plant, "
              "2.5 MPa pads, MuJoCo-Warp 3.6; <code>rows/grid2dff_v4b_sv25r</code>). The outer-loop methods are on the "
              f'<a href="{SEARCH}">search methods page</a>, Section 12. Builder: '
              "<code>scripts/hom/inner_loop_benchmark_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "SR2 inner-loop evaluators", "H1": "Inner-loop evaluators for SR2 hand co-design: HOM, KaRMA "
                 "and UHAS", "EYEBROW": "Co-design benchmark, 2026-11-12", "LEDE": unit(lede_t), "BYLINE": byline,
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    page = render_tex(page)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-12")
    page = retro_style.apply(page)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

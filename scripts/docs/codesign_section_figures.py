#!/usr/bin/env python3
"""Slide figures for the three level sections of the hand-design talk (owner, 2026-11-10 evening: the morphology,
control and dynamics sections should have the co-design section's "clarity, visual elegance, and incrementalism"). Each
figure makes one point at slide size, in the typeface (Nimbus Sans, the drawings' Helvetica) and colours of the TikZ
drawings. Writes docs/experiments/20261110-codesign_formulation/media/:

  20261110-slide_landscape.png     every hand on the lattice, press on the 25 mm cylinder, the best one starred and
                                   the three hands of the grasp film lettered A, B and C (_best.png: the star only)
  20261110-slide_tasks.png         the four tasks' landscapes, each as a share of its own best, best hands starred
  20261110-slide_one_hand.png      the same four landscapes with the one hand that reaches 0.84 of every task's best
  20261110-slide_three_hands.png   and with the three hands that reach 0.95
  20261110-slide_size_trend.png    the best thumb-to-pair distance against object diameter, with Borras and Dollar's
                                   slope of 1 (the workspace rule) in purple
  20261110-slide_cost_{drake,all}.png   time to simulate one landscape: Drake (estimated), the pads on a CPU core and
                                   on one GPU

Data: the 625-hand grids grid2dff_v4b_sv25 (SCS0009 plant, 2.5 MPa pads, best of 16 grasps, mean of seeds 0-2), the
task-conflict study's validated covers (seeds 20-24, docs/experiments/20261103-hom_task_conflict/20261104-conflict_sv25.json)
and the live evolutionary search at 9-10 sizes per task on the same plant (rows/ea_live_sv25/summary.json).

    .venv/bin/python scripts/docs/codesign_section_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401

OUT = ROOT / "docs/experiments/20261110-codesign_formulation/media"
FPR = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
CONFLICT = ROOT / "docs/experiments/20261103-hom_task_conflict/20261104-conflict_sv25.json"
INK, GREY, LINE, NEW = "#1f2937", "#9ca3af", "#d1d5db", "#7c3aed"
STAR = "#facc15"
CMAP = "YlGnBu"
TASKS = [("twist", 40.0, "Twist", "40 mm sphere", "#b91c1c"), ("roll", 40.0, "Roll", "40 mm sphere", "#0e7490"),
         ("pinpush", 25.0, "Pin-push", "25 mm cylinder", "#a16207"), ("press", 25.0, "Press", "25 mm cylinder", "#4d7c0f")]
COVER_MARKS = ["o", "s", "D"]
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15, "axes.edgecolor": GREY,
                     "axes.titlepad": 10, "mathtext.fontset": "cm"})


VARIANT = ""            # "track" (--track): every figure from the tracking score, files *_track.png (owner, 11-11: v2 deck)
GRID_TRACK = "grid2dff_v4b_sv25r"                   # the grids rerun with the tracking fields (all four tasks)
EA_TRACK = "ea_live_sv25_track"                     # the size study's live EA with the tracking score as the fitness


def grid_track(task, size):
    """The best-of-16 tracking score per layout of the rerun grid (mean of seeds 0-2), on grid()'s axes."""
    import hom_landscape as L
    import hom_rows
    L.GRID_SRC = "grid2dff"
    poses = L.grid_poses(task, size)
    sc = {}
    for x in (L.ROWS / GRID_TRACK / f"{task}_d{size:g}.jsonl").read_text().splitlines():
        if x.strip():
            r = hom_rows.rescale_metrics(json.loads(x))
            if r.get("status", "ok") == "ok":
                sc.setdefault(int(r["call"]), {}).setdefault(int(r.get("grasp", 0)), []).append(L.track_score(task, r))
    xs = np.unique([float(p[0]) for p in poses.values()])
    ys = np.unique([float(p[1]) for p in poses.values()])
    B = np.full((len(ys), len(xs)), np.nan)
    for call, (x, y, rec) in poses.items():
        if rec and rec.get("ok") and sc.get(call):
            B[int(np.searchsorted(ys, y)), int(np.searchsorted(xs, x))] = max(np.mean(v) for v in sc[call].values())
    return xs, ys, B


def grid(task, size):
    import hom_landscape as L
    if VARIANT == "track":
        return grid_track(task, size)
    L.GRID_SRC, L.GRID_ROWS = "grid2dff", L.ROWS / "grid2dff_v4b_sv25"
    xs, ys, M = L.grid_maps(task, size)
    return np.asarray(xs, float), np.asarray(ys, float), np.asarray(M["best"], float)


def save(fig, name):
    name = name.replace(".png", "_track.png") if VARIANT == "track" else name
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    return OUT / name


def plane(ax, xs, ys, B, vmax, ylabel=True):
    ext = [xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5]
    im = ax.imshow(np.ma.masked_invalid(B), origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=vmax, aspect="equal")
    ax.set_facecolor("#f3f4f6")
    ax.set_xlabel(r"thumb to pair $s_1$ (mm)")
    if ylabel:
        ax.set_ylabel(r"index to middle $s_2$ (mm)")
    ax.set_xticks([50, 100, 150])
    ax.set_yticks([50, 100, 150])
    for s in ax.spines.values():
        s.set_color(LINE)
    return im


def star(ax, x, y, size=520):
    ax.scatter([x], [y], marker="*", s=size, c=STAR, edgecolors=INK, linewidths=1.6, zorder=5)


FILMED = (("A", (70.0, 150.0)), ("B", (70.0, 110.0)), ("C", (125.0, 55.0)))   # film_three_hands, left to right
FILMED_TRACK = (("A", (70.0, 65.0)), ("B", (70.0, 110.0)), ("C", (125.0, 55.0)))  # its tracking-score version


def landscape(filmed=True, name="20261110-slide_landscape.png"):
    """Every hand on the lattice for one task and size; the best starred. filmed: letter the hands of the grasp film (owner,
    11-10 night, asked what A, B and C denote: the A stood off its star, over another hand's cell)."""
    from matplotlib import patheffects
    xs, ys, B = grid("press", 25.0)
    j, i = np.unravel_index(np.nanargmax(B), B.shape)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    im = plane(ax, xs, ys, B, 0.85)
    star(ax, xs[i], ys[j])
    for lab, (x, y) in (FILMED_TRACK if VARIANT == "track" else FILMED) if filmed else ():
        if (x, y) == (xs[i], ys[j]):                           # the best hand: its letter sits on the star's top point
            ax.text(x, y + 4.6, lab, ha="center", va="bottom", fontsize=15, weight="bold", color=INK, zorder=6,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="white")])
        else:
            ax.scatter([x], [y], s=330, facecolors="white", edgecolors=INK, linewidths=1.8, zorder=5)
            ax.text(x, y, lab, ha="center", va="center", fontsize=13, weight="bold", color=INK, zorder=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("press score")
    cb.outline.set_edgecolor(LINE)
    ax.set_title("Press, 25 mm cylinder: 625 hands", loc="left")
    return save(fig, name), dict(best=[float(xs[i]), float(ys[j])], score=float(B[j, i]),
                                 reached=int(np.isfinite(B).sum()),
                                 filmed={lab: [*xy, round(float(B[list(ys).index(xy[1]), list(xs).index(xy[0])]), 4)]
                                         for lab, xy in (FILMED_TRACK if VARIANT == "track" else FILMED)})


def tasks(marks=None, name="20261110-slide_tasks.png"):
    """The four tasks' landscapes, each as a share of its own best; marks = [(xy, tasks served)] adds the chosen hands."""
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.6), gridspec_kw=dict(wspace=0.08))
    info = {}
    for k, (ax, (task, size, name_, obj, _c)) in enumerate(zip(axes, TASKS)):
        xs, ys, B = grid(task, size)
        top = np.nanmax(B)
        im = plane(ax, xs, ys, B / top, 1.0, ylabel=k == 0)
        if k:
            ax.set_yticklabels([])
        j, i = np.unravel_index(np.nanargmax(B), B.shape)
        if marks is None:
            star(ax, xs[i], ys[j])
        else:
            for (xy, served), mk in zip(marks, COVER_MARKS):   # each panel shows the hand that serves its task
                if task in served:
                    ax.scatter([xy[0]], [xy[1]], marker=mk, s=260, facecolors=NEW, edgecolors="white", linewidths=2.0,
                               zorder=5)
        ax.set_title(f"{name_}, {obj}", loc="left", fontsize=17)
        info[task] = dict(best=[float(xs[i]), float(ys[j])], score=round(float(top), 3))
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
    cb.set_label("share of the task's best")
    cb.outline.set_edgecolor(LINE)
    return save(fig, name), info


def covers_track(levels=(0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    """The task-conflict covers from the tracking-score grids (seeds 0-2): the one hand with the highest share of every
    task's best (alpha*), and the fewest hands that reach alpha of every task's best (K*(alpha), an exact search over up
    to four hands among the layouts that reach alpha on at least one task)."""
    import itertools
    R = {}
    for task, size, *_ in TASKS:
        xs, ys, B = grid(task, size)
        top = np.nanmax(B)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if np.isfinite(B[j, i]):
                    R.setdefault((float(x), float(y)), {})[task] = float(B[j, i] / top)
    names = [t for t, *_ in TASKS]
    share = {xy: min(r.get(t, 0.0) for t in names) for xy, r in R.items()}
    one_xy = max(share, key=share.get)
    tops = {t: float(np.nanmax(grid(t, sz)[2])) for t, sz, *_ in TASKS}
    one_sc = {t: R[one_xy].get(t, 0.0) * tops[t] for t in names}
    out = {"alpha_star": round(share[one_xy], 3), "one": [list(one_xy), names], "K": {},
           "one_scores": {t: round(v, 3) for t, v in one_sc.items()}, "one_mean": round(float(np.mean(list(one_sc.values()))), 3),
           "best_mean": round(float(np.mean(list(tops.values()))), 3)}
    covers_at = {}
    for a in levels:
        serve = {xy: tuple(t for t in names if r.get(t, 0.0) >= a) for xy, r in R.items()}
        cand = [xy for xy, sv in serve.items() if sv]
        best = None
        for k in range(1, 5):
            for combo in itertools.combinations(cand, k):
                if set().union(*(serve[c] for c in combo)) >= set(names):
                    score = min(max(R[c].get(t, 0.0) for c in combo) for t in names)
                    if best is None or score > best[0]:
                        best = (score, combo)
            if best:
                break
        out["K"][f"{a:.2f}"] = len(best[1]) if best else None
        covers_at[a] = [(list(c), [t for t in names if R[c].get(t, 0.0) >= a]) for c in best[1]] if best else None
    three_a = max((a for a in levels if covers_at[a] and len(covers_at[a]) == 3), default=None)
    out["three_alpha"] = three_a
    out["three"] = covers_at.get(three_a)
    tasks([(out["one"][0], names)], "20261110-slide_one_hand.png")
    if out["three"]:
        tasks(out["three"], "20261110-slide_three_hands.png")
    return out


def covers():
    """The hands of the task-conflict study's validated covers: one hand (0.84 of every task's best), three (0.95)."""
    if VARIANT == "track":
        return covers_track()
    d = json.loads(CONFLICT.read_text())["val"]["coverage"]["val"]
    lv = {round(x["alpha"], 2): x for x in d["levels"]}
    one = [(c["x"], c["tasks"]) for c in lv[0.84]["cover"]]
    three = [(c["x"], c["tasks"]) for c in lv[0.95]["cover"]]
    p1, _ = tasks(one, "20261110-slide_one_hand.png")
    p3, _ = tasks(three, "20261110-slide_three_hands.png")
    return dict(one=one, three=three, alpha_star=d["alpha_star"])


def ea_track_summary(near=0.05):
    """hom_ea.cmd_report's per-size outcome of the live EA under the tracking score (rows EA_TRACK): the best layout and
    the layouts within `near` of it (count, centroid, range), each layout's score its best grasp's mean over seeds 0-2.
    Computed here so that hom_gpu (and CUDA) is not imported beside a GPU queue."""
    import hom_landscape as L
    import hom_rows
    xs, ys = np.arange(40.0, 160.0 + 1e-9, 5.0), np.arange(50.0, 170.0 + 1e-9, 5.0)
    inv = {k: (float(x), float(y)) for k, (x, y) in enumerate((x, y) for x in xs for y in ys)}
    out = {}
    for p in sorted((FPR / EA_TRACK).glob("*_d*_s0.jsonl")):
        task, d = p.stem.split("_d")[0], float(p.stem.split("_d")[1].split("_s")[0])
        sc = {}
        for x in p.read_text().splitlines():
            if x.strip():
                r = hom_rows.rescale_metrics(json.loads(x))
                sc.setdefault(int(r["call"]), {}).setdefault(int(r.get("grasp", 0)), []).append(L.track_score(task, r))
        fit = {c: max(float(np.mean(v)) for v in g.values()) for c, g in sc.items() if g}
        if not fit:
            continue
        best = max(fit, key=fit.get)
        nr = np.array([inv[c] for c, f in fit.items() if f >= fit[best] - near])
        out[f"{task}_d{d:g}"] = {"task": task, "d_mm": d, "ea": {
            "best_xy": list(inv[best]), "best": round(fit[best], 4), "layouts": len(fit), "near_n": len(nr),
            "near_centroid": nr.mean(0).round(1).tolist(), "near_lo": nr.min(0).tolist(), "near_hi": nr.max(0).tolist()}}
    return out


def size_trend():
    """The best thumb-to-pair distance (centroid of the hands within 0.05 of each size's best, with their range)
    against object diameter, four tasks, least-squares lines; Borras and Dollar's slope of 1 through each panel's
    centre in purple."""
    S = ea_track_summary() if VARIANT == "track" else json.loads((FPR / "ea_live_sv25/summary.json").read_text())

    def series(task):
        pts = sorted((v["d_mm"], v["ea"]) for v in S.values() if isinstance(v, dict) and v.get("task") == task
                     and v.get("ea"))
        d = np.array([p[0] for p in pts])
        c = np.array([p[1]["near_centroid"][0] for p in pts])
        lo = np.array([p[1]["near_lo"][0] for p in pts])
        hi = np.array([p[1]["near_hi"][0] for p in pts])
        k, b = np.polyfit(d, c, 1)
        return d, c, lo, hi, k, b

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True, gridspec_kw=dict(wspace=0.06))
    slopes = {}
    for ax, title, group in ((axes[0], "Spheres", TASKS[:2]), (axes[1], "Cylinders", TASKS[2:])):
        cen = []
        for n, (task, _s, name_, _o, col) in enumerate(group):
            d, c, lo, hi, k, b = series(task)
            slopes[task] = round(float(k), 2)
            off = (-0.5, 0.5)[n]
            ax.errorbar(d + off, c, yerr=[c - lo, hi - c], fmt="none", ecolor=col, alpha=0.35, lw=1.6, capsize=0)
            ax.plot(d + off, c, "o", color=col, ms=9, label=f"{name_}: {k:.2f} mm per mm")
            xx = np.linspace(d.min() - 2, d.max() + 2, 50)
            ax.plot(xx, k * xx + b, "-", color=col, lw=2.4)
            cen.append((d.mean(), c.mean()))
        x0, y0 = np.mean(cen, axis=0)
        lo_, hi_ = ax.get_xlim()
        xx = np.linspace(lo_ + 1, hi_ - 1, 20)
        ax.plot(xx, y0 + (xx - x0), ls=(0, (5, 4)), color=NEW, lw=2.4, label="slope 1: Borràs and Dollar")
        ax.set_title(title, loc="left")
        ax.set_xlabel("object diameter $d$ (mm)")
        ax.grid(color="#eef0f3", lw=0.8)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.legend(loc="upper left", frameon=False, handlelength=1.8)
    axes[0].set_ylabel(r"best thumb to pair $s_1^\star$ (mm)")
    axes[0].set_ylim(30, 180)
    return save(fig, "20261110-slide_size_trend.png"), slopes


RL_RUNS = ROOT / "results/rl"


def rl_run_times():
    """Wall time of every PPO training run under results/rl (the lift and reorientation policies of 2026-05-27 to 10-11):
    from each run's first TensorBoard event file (its name carries the start, in s since the epoch) to the last write of
    its event files. Kept in figures.json (rl_runs), since results/ is not in git."""
    import glob
    import re as _re
    w = []
    for d in sorted(RL_RUNS.glob("*/")):
        ev = glob.glob(str(d / "**/events.out.tfevents.*"), recursive=True)
        if ev:
            t0 = min(int(_re.search(r"tfevents\.(\d+)", e).group(1)) for e in ev)
            w.append((max(Path(e).stat().st_mtime for e in ev) - t0) / 60.0)
    w = np.array([x for x in w if 0.5 <= x <= 24 * 60])           # past a day: resumed or touched later
    q = lambda p_: round(float(np.percentile(w, p_)), 1)           # noqa: E731
    return {"runs": int(len(w)), "min_p10": q(10), "min_p25": q(25), "min_median": q(50), "min_p75": q(75),
            "gpu_hours": round(float(w.sum()) / 60.0), "best_case_min": 10.0,
            "source": "results/rl/*/tensorboard event files, 0.5 min to 24 h"}


STEEL = ROOT / "docs/experiments/20261113-steel_solver/data/20261113-steel_solver.json"
MPC_ROWS = ROOT / "docs/experiments/20261112-inner_loop_benchmark/rows"
NEWTON = "nt_mjc_hydro"          # the Steel page's Newton row: SolverMuJoCo with Newton's SDF hydroelastic contact


def planner_cost(rec):
    """GPU time of one twist rollout under sampling MPC over HOM's in the CEM workload (owner, 2026-11-14: "the updated
    timing charts should elucidate why we stick to HOM"): the planner's batch runs of the inner-loop benchmark
    (scripts/hom/inner_loop_mpc.py, 16 hands per batch, each hand's samples as worlds), wall time of a batch over its
    hands, against HOM's GPU time per twist rollout in the compute chart (rec cem ms_per_rollout). The runs are the
    twist's 100-pair ranking runs: DIAL 2 x 32 is the real-time pick (0.082 s of GPU per 0.1 s for one hand), DIAL 4 x 64
    at a 2 ms planner step (6.3 times real time on one hand) the ranking test's full DIAL. Until 2026-11-15 the full row
    was the 1 ms planner step (mpc_anneal_twist, 401 times HOM)."""
    out = {}
    hom_s = rec["cem"]["ms_per_rollout"]["twist"] / 1e3
    for k, stem in (("rt", "mpc_anneal_twist_rank_pdt2_n32_r10h10k2i2"), ("full", "mpc_anneal_twist_rank_pdt2")):
        p = MPC_ROWS / f"{stem}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        per = float(np.mean([r["chunk_wall_s"] / r["chunk_worlds"] for r in rows]))
        out[k] = {"s_per_rollout": round(per, 3), "ratio": round(per / hom_s, 1), "stem": stem,
                  "hands_per_batch": int(rows[0]["chunk_worlds"])}
    return out


def human(sec):
    return (f"{sec / 86400 / 365:.1f} years" if sec > 400 * 86400 else f"{sec / 86400:.0f} days" if sec > 3 * 86400
            else f"{sec / 3600:.0f} hours" if sec > 3 * 3600 else f"{sec / 60:.0f} minutes")


def cost_chart(rows, title, name, show=None, split=None):
    """One bar per row (label, seconds, colour) on a log time axis from 1 hour to 1 year, the time written at the bar's
    end. Rows from index split on sit below a dashed line; only the first show rows are drawn (the axes and labels of
    the rest stay, for a build-up). Both cost charts use it, so they share their scale and layout. Returns {first line of
    the label: seconds}."""
    show = len(rows) if show is None else show
    fig, ax = plt.subplots(figsize=(13.0, 1.6 + 0.74 * len(rows)))
    gap = (lambda k: 0.0 if split is None or k < split else -0.35)
    ys = [len(rows) - 1 - k + gap(k) for k in range(len(rows))]
    out = {}
    for k, (lab, sec, col) in enumerate(rows):
        vis = k < show
        ax.barh(ys[k], sec, color=col if vis else "none", height=0.58)
        ax.text(sec * 1.25, ys[k], human(sec) if vis else "", va="center", fontsize=19, color=INK, weight="bold")
        out[lab.split("\n")[0]] = round(sec)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=14.5)
    if split is not None:
        ax.axhline(0.5 * (ys[split - 1] + ys[split]), color=LINE, lw=1.0, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xticks([3600, 86400, 30 * 86400, 365 * 86400])
    ax.set_xticklabels(["1 hour", "1 day", "30 days", "1 year"], fontsize=14)
    ax.set_xlim(3000, 5e8 if max(r[1] for r in rows) < 4e8 else 3e9)
    ax.set_ylim(min(ys) - 0.65, len(rows) - 0.45)
    ax.set_title(title, loc="left", fontsize=17)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(LINE)
    ax.grid(axis="x", color="#eef0f3")
    save(fig, name)
    return out


def cost_of_search():
    """The time to CEM-refine grasp and HOM on every layout of the twist, the roll and the press (1,423 layouts x 768
    rollouts, the workload of the compute chart, codesign_formulation_figures.py compute), in two charts of one layout
    (owner, 2026-11-16: "we replaced newton with steel, but steel performs worse/slower than newton's own hydroelastic sdf
    solver"; "the planner comparisons should also go on a different chart, albeit near identical in structure"; "put the
    RL comparison on this separate chart too, that way we isolate the contact model throughput comparison vs the control
    throughput comparison").

    Simulators, each with HOM (20261110-slide_cost_drake.png: Drake alone, 20261110-slide_cost_all.png: all four).
    MuJoCo-Warp: the compute chart's estimate. One CPU core: the CPU runner's time per rollout with every contact
    (score-levels page, Table 18: 3.31, 1.72 and 2.09 s). Drake: those times the step-cost ratio on the hand, 2,082 / 106
    us (contact overview, Table 1). Newton's SDF hydroelastic contact under its MuJoCo solver: the MuJoCo-Warp time times
    the ratio of the two step costs measured on the twist at 1,024 worlds (Steel page, 2026-11-13: 5.00 and 2.76 us per
    world-step). Until 2026-11-16 that row was Steel's compliant hydroelastic contact (9.17 us, 3.32 times the pads).

    Controllers, each on MuJoCo-Warp with the pads on one GPU (20261116-slide_cost_control.png): HOM; in its place the
    gradient planner and annealed sampling at the real-time budget, annealed sampling on HOM's servo targets (the
    residual), iLQR and full annealed sampling, at their GPU time per twist rollout over HOM's in the search's batches
    (planner_cost; figures.json controllers_track cost, residual_cost, ilqr cost); one PPO policy per layout on the same
    GPU at the median of our trainings (rl_run_times) and at 10 min, the best case."""
    path = OUT / "20261110-figures.json"
    fi = json.loads(path.read_text())
    rec = fi["compute_track" if VARIANT == "track" else "compute"]
    lay = rec["layouts"]
    n_lay, n_roll = sum(lay.values()), rec["cem"]["rollouts"]
    rl = rl_run_times() if RL_RUNS.exists() else fi["rl_runs"]
    fi["rl_runs"] = rl
    thr = json.loads((ROOT / "docs/experiments/20261103-hom_score_refine/compute/20261104-throughput.json").read_text())
    cpu_roll = {t: float(thr["cpu"][t]["plant_p1"]["wall_s"]) for t in lay}
    cpu = sum(lay[t] * 768 * cpu_roll[t] for t in lay)
    mjw = float(rec["cem"]["s"])
    st = json.loads(STEEL.read_text())
    sp = st["speed"]
    us = {k: sp[k]["1024"]["us_per_world_step"] for k in ("mjw_pads", NEWTON, "steel_hydro")}
    newton = us[NEWTON] / us["mjw_pads"]
    work = f"{n_lay:,} layouts \u00d7 768 rollouts = {n_roll / 1e6:.2f} M rollouts"
    sims = [("Drake hydroelastic, one CPU core\n(estimated from its step cost)", cpu * 2082.0 / 106.0, "#6b7280"),
            ("MuJoCo, 1 mm sphere pads,\none CPU core", cpu, "#f59e0b"),
            ("Newton hydroelastic (SDF), one GPU\n(its step cost on the twist, 1,024 worlds)", mjw * newton, "#7a4fa6"),
            ("MuJoCo-Warp, 1 mm sphere pads,\none GPU", mjw, "#d97706")]
    title = f"Every layout of three tasks with HOM: {work},\non four simulators"
    cost_chart(sims, title, "20261110-slide_cost_drake.png", show=1)
    out = cost_chart(sims, title, "20261110-slide_cost_all.png")
    mpc = planner_cost(rec)
    ct = fi.get("controllers_track", {})
    cc = ct.get("cost", {})
    ctl = [("HOM", mjw, "#d97706")]
    for lab, ratio, col in (
            ("Gradient descent, 2 \u00d7 32 rollouts,\nin place of HOM (real time)", cc.get("Gradient 2 \u00d7 32, real time"),
             "#7c3aed"),
            ("Annealed sampling, 2 \u00d7 32 samples,\nin place of HOM (real time)", mpc.get("rt", {}).get("ratio"), "#be185d"),
            ("Annealed sampling on HOM\u2019s targets,\n2 \u00d7 32 samples (a residual)", ct.get("residual_cost"), "#db2777"),
            ("iLQR, 2 iterations over 0.3 s,\nin place of HOM", ct.get("ilqr", {}).get("cost"), "#5b21b6"),
            ("Full annealed sampling, 4 \u00d7 64 samples,\nin place of HOM", mpc.get("full", {}).get("ratio"), "#9d174d")):
        if ratio:
            ctl.append((lab, mjw * float(ratio), col))
    n_plan = len(ctl)
    ctl += [(f"RL: a policy per layout, {rl['min_median']:.0f} min each\n(the median of our {rl['runs']} trainings)",
             n_lay * 60.0 * rl["min_median"], "#0f766e"),
            (f"RL: a policy per layout,\n{rl['best_case_min']:.0f} min each at best", n_lay * 60.0 * rl["best_case_min"],
             "#14b8a6")]
    out.update(cost_chart(ctl, f"The same {n_roll / 1e6:.2f} M rollouts on MuJoCo-Warp with the pads, one GPU,\n"
                               "under HOM, a planner in its place, or one trained policy per layout",
                          "20261116-slide_cost_control.png", split=n_plan))
    res = {"rows_s": out, "layouts": n_lay, "rollouts": n_roll, "cpu_s_per_rollout": cpu_roll,
           "newton_ratio": round(float(newton), 3), "steel_ratio": round(float(us["steel_hydro"] / us["mjw_pads"]), 3),
           "us_per_world_step_1024": {k: round(float(v), 3) for k, v in us.items()},
           "hom_on_twist": {k: {"mean_J": round(float(st["hom"][k]["mean_J"]), 4), "held": st["hom"][k]["held"],
                                "n": st["hom"][k]["n"]} for k in ("mjc_pads", NEWTON, "steel_hydro") if k in st["hom"]},
           "drake_ratio": round(2082.0 / 106.0, 2), "mpc": mpc,
           "sim_rows": [r[0].split("\n")[0] for r in sims], "control_rows": [r[0].split("\n")[0] for r in ctl]}
    fi["cost_search" + ("_track" if VARIANT == "track" else "")] = res
    path.write_text(json.dumps(fi, indent=1, default=float))
    return res


BENCH = [("hom_twist", "HOM", "#d97706"),
         ("ps_twist_r10h10k2i1", "Predictive sampling,\nreal time", "#f472b6"),
         ("anneal_twist", "Annealed sampling,\n4 \u00d7 64 samples", "#9d174d"),
         ("spider_twist", "SPIDER", "#be185d"),
         ("spider_wc_twist", "SPIDER, worst case\nover friction", "#db2777"),
         ("uhas_twist_sv_published_s0", "UHAS policy (RL)", "#0f766e"),
         ("uhas_twist_sv_limits_tau0.02_cap0_s0", "UHAS policy (RL),\nwith limit terms", "#14b8a6")]


def mu04(stem):
    """The stem of the same controller's runs at plant friction 0.4."""
    if stem.startswith("uhas_twist_"):
        return stem.replace("uhas_twist_", "uhas_twist_mu0.4_")
    if stem == "ps_twist_r10h10k2i1":
        return "ps_twist_mu0.4_r10h10k2i1"
    return stem + "_mu0.4"


def controller_benchmark():
    """The controllers of the inner-loop benchmark's twist on four measures (owner, 2026-11-16: "an additional chart(s)
    that neatly compares the performance of these controllers, in raw J, and then maybe also in reversals / other
    sim2real / qualitative benchmarks that help differentiate the methods"). The same 20 pairs of layout and grasp at
    seeds 0 and 1, 40 runs per controller, on the SCS0009 plant with 2.5 MPa pads (scripts/hom/inner_loop_mpc.py, sampling
    MPC page): (a) mean J and its standard error; (b) mean J with the plant's friction at 0.4, the value measured for TPU
    on bare PLA, while every controller's model keeps 1.0 (SPIDER's worst case samples 0.4, 0.7 and 1.0; UHAS was trained
    at 1.0); (c) the median rate of target reversals per joint per second over the held runs (inner_loop_usability:
    sign changes of a servo target's rate above 1 deg/s); (d) the median share of joint-periods with the servo torque at
    95 % of its limit or more. Writes 20261116-slide_controller_benchmark.png and figures.json controller_benchmark."""
    import inner_loop_mpc_page as MP
    S = {}
    for stem, lab, col in BENCH:
        a, b = MP.load(stem), MP.load(mu04(stem))
        if a and b:
            S[stem] = (lab, col, MP.stats(a), MP.stats(b))
    if not S:
        return None
    rows = [S[s] for s, _l, _c in BENCH if s in S]
    fig, ax = plt.subplots(1, 4, figsize=(17.0, 5.6), sharey=True,
                           gridspec_kw={"width_ratios": [1.0, 1.0, 1.15, 1.0], "wspace": 0.12})
    y = np.arange(len(rows))[::-1]
    cols = [r[1] for r in rows]

    def bars(a, v, txt, err=None, pad=0.015):
        """Bars with their values written past the bar's end and its error bar (pad in the axis's units; a log axis
        places the text at 1.18 times the value)."""
        a.barh(y, v, color=cols, height=0.62, xerr=err, error_kw={"ecolor": INK, "lw": 1.2, "capsize": 3})
        for k, (yi, vi, ti) in enumerate(zip(y, v, txt)):
            x = vi * 1.18 if a.get_xscale() == "log" else vi + (err[k] if err else 0.0) + pad
            a.text(x, yi, ti, va="center", fontsize=14, color=INK)

    ax[0].set_xlim(0, 0.62)
    bars(ax[0], [r[2]["J"] for r in rows], [f"{r[2]['J']:.2f}" for r in rows], err=[r[2]["J_se"] for r in rows])
    ax[0].set_title("Mean J, plant friction 1.0", loc="left", fontsize=16)
    ax[1].set_xlim(0, 0.62)
    bars(ax[1], [r[3]["J"] for r in rows], [f"{r[3]['J']:.2f}" for r in rows], err=[r[3]["J_se"] for r in rows])
    ax[1].set_title("Mean J, plant friction 0.4", loc="left", fontsize=16)
    ax[2].set_xscale("log")
    ax[2].set_xlim(0.2, 300)
    bars(ax[2], [r[2]["reversals_ps"] for r in rows], [f"{r[2]['reversals_ps']:.1f}" for r in rows])
    ax[2].set_title("Target reversals per joint per s", loc="left", fontsize=16)
    ax[3].set_xlim(0, 100)
    bars(ax[3], [100 * r[2]["sat_frac"] for r in rows], [f"{100 * r[2]['sat_frac']:.0f} %" for r in rows], pad=2.5)
    ax[3].set_title("Joint time at the torque limit", loc="left", fontsize=16)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([r[0] for r in rows], fontsize=14.5)
    for a in ax:
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
        for sp_ in ("left", "bottom"):
            a.spines[sp_].set_color(LINE)
        a.grid(axis="x", color="#eef0f3")
        a.tick_params(axis="x", labelsize=13)
        a.set_axisbelow(True)
    ax[2].set_xticks([1, 10, 100])
    ax[2].set_xticklabels(["1", "10", "100"])
    ax[3].set_xticks([0, 50, 100])
    ax[3].set_xticklabels(["0", "50", "100 %"])
    out = {"image": str(save(fig, "20261116-slide_controller_benchmark.png").relative_to(ROOT)),
           "runs": {s: {"label": lab.replace("\n", " "), "n": a["n"], "J": a["J"], "J_se": a["J_se"], "held": a["held"],
                        "J_mu04": b["J"], "held_mu04": b["held"], "reversals_ps": a["reversals_ps"],
                        "cmd_rate_dps": a["cmd_rate_dps"], "sat_frac": a["sat_frac"], "grip_mean_N": a["grip_mean_N"],
                        "link_mean_N": a["link_mean_N"], "contact": a["contact"], "replan_s": a["replan_s"]}
                    for s, (lab, _c, a, b) in S.items()}}
    path = OUT / "20261110-figures.json"
    fi = json.loads(path.read_text())
    fi["controller_benchmark" + ("_track" if VARIANT == "track" else "")] = out
    path.write_text(json.dumps(fi, indent=1, default=float))
    return out


def cost_per_landscape():
    """Time to simulate one landscape (625 hands x 16 grasps x 3 seeds = 30,000 rollouts): Drake estimated from its step
    cost on the hand, the pads on one CPU core and in GPU batches as measured, and Newton's SDF hydroelastic contact on the
    same GPU estimated from its throughput (owner, 11-11). Two images on one canvas: Drake alone, then all four. Step costs
    on the hand from the contact overview's table (Drake 2,082 us, pads 106 us per 1 ms step); the rollout times from the
    formulation page's figures.json (CPU 5.0 s; GPU median per rollout in a batch of 1536). Newton: the pads' GPU time
    times the ratio of the two throughputs at 4,096 worlds on the overview's SR2 pinch benchmark (Figure 12,
    docs/experiments/20261005-gpu_scaling: MuJoCo-Warp pads against Newton hydroelastic with contact reduction)."""
    fi = json.loads((OUT / "20261110-figures.json").read_text())
    gpu = float(fi.get("gpu", {}).get("s_per_rollout_median", 0.0365))
    n = 625 * 16 * 3
    cpu = 5.0
    drake = cpu * 2082.0 / 106.0
    import contact_overview_page as CO
    load = (lambda name: [json.loads(x) for x in (Path(CO.GPU) / name).read_text().splitlines() if x.strip()])
    curves = CO.gpu_curves(load("gpu_scaling.jsonl"), load("newton_scaling.jsonl"))
    ratio = curves["mjw_pads1"][4096] / curves["newton_hydro_mc"][4096]
    rows = [("Drake hydroelastic, one CPU core\n(estimated from its step cost)", n * drake, "#6b7280"),
            ("MuJoCo, 1 mm sphere pads,\none CPU core", n * cpu, "#f59e0b"),
            ("Newton hydroelastic (SDF), one GPU\n(estimated from its throughput)", n * gpu * ratio, "#7a4fa6"),
            ("MuJoCo-Warp, 1 mm sphere pads,\none GPU (batches of 1536)", n * gpu, "#d97706")]

    def human(sec):
        return (f"{sec / 86400:.0f} days" if sec > 3 * 86400 else f"{sec / 3600:.0f} hours" if sec > 3 * 3600
                else f"{sec / 60:.0f} minutes")
    out = {}
    for name, show in (("20261110-slide_cost_drake.png", 1), ("20261110-slide_cost_all.png", len(rows))):
        fig, ax = plt.subplots(figsize=(13.0, 5.6))
        for k, (lab, sec, col) in enumerate(rows):
            y = len(rows) - 1 - k
            vis = k < show
            ax.barh(y, sec, color=col if vis else "none", height=0.58)
            ax.text(sec * 1.25, y, human(sec) if vis else "", va="center", fontsize=19, color=INK, weight="bold")
            out[lab.split(",")[0]] = sec
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=15)
        ax.set_xscale("log")
        ax.set_xlim(300, 3e7)
        ax.set_xticks([60, 3600, 86400, 30 * 86400])
        ax.set_xticklabels(["1 minute", "1 hour", "1 day", "30 days"], fontsize=14)
        ax.set_xlim(300, 3e7)
        ax.set_title("One landscape: 625 hands \u00d7 16 grasps \u00d7 3 seeds = 30,000 rollouts", loc="left", fontsize=18)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(LINE)
        ax.grid(axis="x", color="#eef0f3")
        save(fig, name)
    return {k: round(v) for k, v in out.items()}


def fidelity_drake():
    """Median |deviation| from Drake over the contact rig's metrics, per contact model: the numbers of the contact
    overview's Figure 11 as bars (owner, 11-11: the chart of deviation against step cost was hard to read)."""
    import statistics
    import contact_overview_page as CO
    T = {name: CO.bed(fn) for name, fn in (("pull", "pull_slip"), ("twist", "twist_slip"), ("roll", "roll"),
                                             ("shake", "shake"), ("brake", "brake"), ("creep", "creep"))}
    dev = CO.deviations(CO.metrics(T))
    rows = [("mj_point3", "MuJoCo point contact", "#6b7280"), ("newton_hydro_mc", "Newton hydroelastic (SDF)", "#7a4fa6"),
            ("newton_pads1", "Newton, 1 mm sphere pads", "#f59e0b"), ("mj_pads1", "MuJoCo, 1 mm sphere pads", "#f59e0b"),
            ("mjw_pads1", "MuJoCo-Warp, 1 mm sphere pads", "#d97706")]
    med = {k: statistics.median(dev[k]) for k, _l, _c in rows}
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for y, (k, lab, col) in enumerate(rows[::-1]):
        ax.barh(y, med[k], color=col, height=0.6)
        ax.text(med[k] + 2.0, y, f"{med[k]:.1f} %", va="center", fontsize=17, color=INK, weight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _k, lab, _c in rows[::-1]], fontsize=15)
    ax.set_xlim(0, 115)
    ax.set_xlabel("median deviation from Drake (%)")
    ax.set_title("Against Drake's hydroelastic contact", loc="left", fontsize=18)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(LINE)
    save(fig, "20261111-slide_fidelity_drake.png")
    return {k: round(med[k], 1) for k, _l, _c in rows} | {"n_metrics": len(dev["mjw_pads1"])}


def fidelity_tip():
    """The real fingertip: UR5 presses of three printed v3 tips under the 12.5 mm rod against the pads before the
    calibration (10 MPa), the pads at the modulus fitted to the presses and Drake's field at its fitted modulus (the v3
    contact-model page, Figure 1 (b), with its FE curves left out)."""
    import v3_tip_contact_model_figures as VF
    d = VF.load()
    J = d["fits"]["joint"]
    Ep, Ef = J["E_pads_MPa"], J["E_field_MPa"]
    F, Dp, _ = VF.face_holds(d, VF.ROD)
    dep, Gp, Gf, *_ = VF.model_curves(d, VF.ROD)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.scatter(Dp, F, s=14, color=GREY, alpha=0.45, lw=0, zorder=1, label=f"measured: three printed tips, {len(F):,} holds")
    ax.plot(dep, VF.E_SCENES * Gp, color="#d97706", lw=2.2, ls=":", zorder=3, label="pads before calibration, 10 MPa")
    ax.plot(dep, Ep * Gp, color="#d97706", lw=3.4, zorder=4, label=f"pads fitted to the presses, {Ep:.2f} MPa")
    ax.plot(dep, Ef * Gf, color="#4a7fc4", lw=2.4, ls=(0, (6, 3)), zorder=4, label=f"Drake's field, {Ef:.2f} MPa")
    ax.set_xlim(0, 1.8)
    ax.set_ylim(0, 8.6)
    ax.set_xlabel("indentation (mm)")
    ax.set_ylabel("normal force (N)")
    ax.set_title("Against the real fingertip (12.5 mm rod, UR5)", loc="left", fontsize=18)
    ax.legend(fontsize=13.5, frameon=False, loc="lower right", markerscale=2.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(LINE)
    ax.grid(color="#eef0f3")
    save(fig, "20261111-slide_fidelity_tip.png")
    return {"E_pads_MPa": round(Ep, 2), "E_field_MPa": round(Ef, 2), "holds_rod": int(len(F))}


def controllers():
    """HOM against sampling MPC on the four tasks for the v2 deck (owner, 2026-11-14: "include some discussing regarding
    comparison to other methods"; "ideally we have further evidence that HOM is the right choice"): (a) mean J on each
    task's 100 pairs (HOM's 50 best layouts and 50 at random, seeds 0 and 1) under HOM, DIAL and the gradient planner at
    the real-time budget, and DIAL at 4 x 64 samples, each labelled with its GPU time per twist rollout over HOM's in the
    search's batches; (b) HOM's grid as a screen for full DIAL: the best full-DIAL J among HOM's ten best layouts, among
    ten layouts drawn at random from the rest of the grid (mean and 10-90 % of 20,000 draws), and DIAL's best of the 100
    (hom_inner_loop_page.screen_numbers). Rows of scripts/hom/inner_loop_mpc.py; the numbers go to figures.json
    (controllers_track) for the deck's notes."""
    import hom_inner_loop_page as HP
    import inner_loop_mpc_page as MP
    R, out = {}, {"tasks": {}, "cost": {}}
    runs = [("hom", None, "HOM", "#d97706"), ("anneal", MP.CFG_PICK, "Annealed sampling 2 × 32, real time", "#be185d"),
            ("anneal", MP.CFG_RT30, "Annealed sampling 2 × 16, 0.3 s ahead, real time", "#f472b6"),
            ("grad", MP.CFG_PICK, "Gradient 2 × 32, real time", "#7c3aed"),
            ("anneal", MP.CFG_FULL2, "Full annealed sampling 4 × 64, 6.3× real time", "#9d174d")]
    tasks = [t for t in MP.TASKS4 if MP.load(MP.tstem("hom", t, "_rank"))]
    if not tasks:
        return None
    hom_s = json.loads((OUT / "20261110-figures.json").read_text())["compute_track"]["cem"]["ms_per_rollout"]["twist"] / 1e3
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw={"width_ratios": [1.6, 1.0]})
    w = 0.8 / len(runs)
    for i, (m, c, lab, col) in enumerate(runs):
        ys, es = [], []
        for t in tasks:
            rows = HP.rows_of(R, MP.tstem(m, t, "_rank", c))
            pj = MP.pair_j(rows) if rows else {}
            v = np.array(list(pj.values())) if len(pj) >= 50 else np.array([np.nan])
            ys.append(float(np.mean(v)))
            es.append(float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
            out["tasks"].setdefault(t, {})[lab] = {"J": ys[-1], "pairs": len(pj)}
        tw = [r for r in HP.rows_of(R, MP.tstem(m, "twist", "_rank", c)) if r.get("chunk_wall_s")]
        ratio = 1.0 if m == "hom" else (float(np.mean([r["chunk_wall_s"] / r["chunk_worlds"] for r in tw])) / hom_s
                                         if tw else None)
        out["cost"][lab] = ratio
        leg = lab + (f", {ratio:.0f}× HOM's GPU time" if ratio and m != "hom" else "")
        ax[0].bar(np.arange(len(tasks)) + (i - (len(runs) - 1) / 2) * w, ys, w * 0.92, yerr=es, color=col, label=leg,
                  error_kw={"lw": 1.2, "ecolor": INK})
    ax[0].set_xticks(range(len(tasks)))
    ax[0].set_xticklabels([MP.TNAME[t] for t in tasks], fontsize=16)
    ax[0].set_ylabel("Mean J, 100 pairs", fontsize=15)
    ax[0].set_ylim(0, 0.75)
    ax[0].legend(fontsize=12.5, frameon=False, loc="upper right")
    ax[0].set_title("(a) Score per task", loc="left", fontsize=17)
    for t in tasks:
        st, _l = MP.rank_ref(R_load(MP, t), t)
        n = MP.rank_numbers(R_load(MP, t), t, st) if st else None
        out["tasks"][t]["rank"] = {k: n[k] for k in ("rho", "rho_best", "rho_rand", "top10", "first_d", "first_rank",
                                                     "best_d", "seed_hom", "seed_ref")} if n else None
        sc = HP.screen_numbers(R, t)
        out["tasks"][t]["screen"] = ({"d_max": sc["d_max"], "hom10": sc["scr"][9], "rand10": sc["rnd"][9],
                                      "rand10_lo": sc["lo"][9], "rand10_hi": sc["hi"][9], "shared10": sc["shared"],
                                      "halves_full": sc["halves"].get("Annealed sampling 4 &#215; 64"),
                                      **{f"{a}_med_{w}": float(np.median([sc["keys"][i][j] for i in sc[f"top_{w[0]}"]]))
                                         for j, a in enumerate(("xsep", "ysep")) for w in ("hom", "dial")}} if sc else None)
        c = MP.confirm_numbers(t)          # each controller's ten best again at seeds 10 and 11
        out["tasks"][t]["confirm"] = ({k: c[k] for k in (
            "dial_on_hom_s01", "dial_on_hom", "dial_on_dial_s01", "dial_on_dial", "hom_on_hom_s01", "hom_on_hom",
            "hom_on_dial_s01", "hom_on_dial", "d_dial", "first_h_dial", "first_d_dial", "fused_dial", "drop_hom_s01",
            "drop_hom", "drop_dial_s01", "drop_dial")} if c else None)
        out["tasks"][t]["repeat"] = HP.repeat_numbers(R, t) if t in HP.PATCH_TASKS else None
        rr = HP.rows_of(R, MP.tstem("res_anneal", t, "_rank20", MP.CFG_PICK))     # DIAL's residual on HOM's targets
        p_ = MP.paired(rr, HP.hom_rank(R, t)) if rr else None
        out["tasks"][t]["residual"] = ({"J": MP.mean_j(rr), **{k: p_[k] for k in ("d", "lo", "hi")}} if p_ else None)
    out["residual_cost"] = HP.gpu_ratio(R, MP.tstem("res_anneal", "twist", "_rank20", MP.CFG_PICK))
    G = HP.press_grid_numbers(R)        # full DIAL and HOM on all 369 press layouts (HOM page, section press grid)
    if G and G.get("fresh"):
        F = G["fresh"]
        out["press_grid"] = {"n": G["n"], "rho": G["rho_hd"], "out50": G["out50"], "in100": len(G["in100"]),
                             "ysep_med_dial": G["med"]["d"][1], "ysep_med_grid": G["med"]["g"][1],
                             "k_reach": G["k_reach"], "k_reach_rnd": G["k_reach_rnd"],
                             "fresh_dial": F["dial"], "fresh_hom": F["hom"], "fresh_diff": F["diff"],
                             "contact_hom": {w: F["contact"][("hom", w)] for w in ("d", "g")}}
    Gs = HP.grids_numbers(R)            # full annealed sampling and HOM on every layout of each task's grid (HOM page)
    out["grids"] = {t: {"n": G_["n"], "rho": G_["rho_hd"], "out50": G_["out50"], "in100": len(G_["in100"]),
                        "first_rank": G_["first_g_rank"], "k_reach": G_["k_reach"], "k_reach_rnd": G_["k_reach_rnd"],
                        "mean_h": G_["mean_h"], "mean_d": G_["mean_d"],
                        **({"fresh_dial": G_["fresh"]["dial"], "fresh_hom": G_["fresh"]["hom"],
                            "fresh_diff": G_["fresh"]["diff"], "drop_h": G_["fresh"]["drop_h"]} if G_.get("fresh") else {})}
                    for t, G_ in Gs.items()}
    F5 = {t: HP.five_numbers(R, t) for t in ("pinpush", "press")}   # grasps chosen at five placements (HOM page)
    out["five"] = F5 if all(F5.values()) else None
    IL_ = HP.ilqr_numbers(R)            # iLQR on the 20 pairs of each task (HOM page, section iLQR)
    if IL_:
        out["ilqr"] = {"J": IL_["j20"], "hom": IL_["h20"], "full": IL_["f20"], "cost": IL_["gpu"],
                       "one_hand_s": IL_["one"],
                       "vs_hom": {t: {k: v[k] for k in ("d", "lo", "hi")} for t, v in IL_["vs_hom"].items()},
                       "vs_full": {t: {k: v[k] for k in ("d", "lo", "hi")} for t, v in IL_["vs_full"].items()}}
        if IL_.get("pad"):
            pd_ = IL_["pad"]
            out["ilqr"]["pad"] = {"J": pd_["j20"], "cost": pd_["gpu"],
                                  "vs_il": {t: {k: v[k] for k in ("d", "lo", "hi")} for t, v in pd_["vs_il"].items()},
                                  "vs_full": {t: {k: v[k] for k in ("d", "lo", "hi")} for t, v in pd_["vs_full"].items()},
                                  "share": pd_["share"]}
    x = np.arange(len(tasks))
    S_ = [out["tasks"][t]["screen"] or {} for t in tasks]
    g = (lambda k: np.array([s_.get(k, np.nan) for s_ in S_]))
    ax[1].bar(x - 0.18, g("hom10"), 0.34, color="#d97706", label="HOM's ten best layouts")
    ax[1].bar(x + 0.18, g("rand10"), 0.34, color=GREY, yerr=[g("rand10") - g("rand10_lo"), g("rand10_hi") - g("rand10")],
              error_kw={"lw": 1.2, "ecolor": INK}, label="Ten random layouts (10-90 %)")
    ax[1].scatter(x, g("d_max"), marker="_", s=900, lw=3, color="#9d174d", zorder=3, label="Full annealed sampling's best of the 100")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([MP.TNAME[t] for t in tasks], fontsize=16)
    ax[1].set_ylim(0, 1.15)
    ax[1].set_ylabel("Best J under full annealed sampling", fontsize=15)
    ax[1].legend(fontsize=12.5, frameon=False, loc="upper right", ncol=1)
    ax[1].set_title("(b) HOM's picks under the planner", loc="left", fontsize=17)
    for a in ax:
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
        for sp_ in ("left", "bottom"):
            a.spines[sp_].set_color(LINE)
        a.tick_params(labelsize=14)
    fig.tight_layout()
    out["image"] = str(save(fig, "20261114-slide_controllers.png").relative_to(ROOT))
    path = OUT / "20261110-figures.json"
    fi = json.loads(path.read_text())
    fi["controllers" + ("_track" if VARIANT == "track" else "")] = out
    path.write_text(json.dumps(fi, indent=1, default=float))
    return out


def R_load(MP, task):
    """Rows of a task's 100-pair runs, keyed by stem, for inner_loop_mpc_page.rank_ref and rank_numbers."""
    R = {}
    for m, c, _l in MP.RANK_RUNS:
        st = MP.tstem(m, task, "_rank", c)
        R[st] = MP.load(st)
    R[MP.tstem("hom", task, "_rank")] = MP.load(MP.tstem("hom", task, "_rank"))
    return R


def main_track():
    """The figures of the v2 deck: the same drawings from the tracking score (files *_track.png)."""
    global VARIANT
    VARIANT = "track"
    info = {}
    _, info["landscape"] = landscape()
    landscape(filmed=False, name="20261110-slide_landscape_best.png")
    _, info["tasks"] = tasks()
    info["covers"] = covers()
    S = ea_track_summary()
    n = {t: sum(1 for v in S.values() if v["task"] == t) for t, *_ in TASKS}
    if min(n.values()) >= 8:                              # every task's sizes of the size study have run
        _, info["slopes"] = size_trend()
    info["ea_sizes"] = n
    path = OUT / "20261110-figures.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old["section_figures_track"] = info
    path.write_text(json.dumps(old, indent=1, default=float))
    print(json.dumps(info, default=float)[:1500], flush=True)
    return 0


def main():
    global VARIANT
    if "controllers" in sys.argv[1:]:   # HOM against sampling MPC on four tasks (owner, 11-14), v2 only
        VARIANT = "track"
        print(json.dumps(controllers(), default=float)[:2000], flush=True)
        return 0
    if "cost" in sys.argv[1:]:          # only the search-time charts (owner, 11-11), for the deck or with --track for v2
        VARIANT = "track" if "--track" in sys.argv else ""
        print(json.dumps(cost_of_search(), default=float), flush=True)
        return 0
    if "benchmark" in sys.argv[1:]:     # the controllers on the twist benchmark's four measures (owner, 11-16), v2 only
        VARIANT = "track"
        print(json.dumps(controller_benchmark(), default=float)[:3000], flush=True)
        return 0
    if "--track" in sys.argv:
        return main_track()
    OUT.mkdir(parents=True, exist_ok=True)
    info = {}
    _, info["landscape"] = landscape()
    landscape(filmed=False, name="20261110-slide_landscape_best.png")   # the EA slides: the star only
    _, info["tasks"] = tasks()
    info["covers"] = covers()
    _, info["slopes"] = size_trend()
    info["cost_per_landscape_s"] = cost_per_landscape()
    info["fidelity_drake"] = fidelity_drake()
    info["fidelity_tip"] = fidelity_tip()
    path = OUT / "20261110-figures.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old["section_figures"] = info
    path.write_text(json.dumps(old, indent=1, default=float))
    print(json.dumps(info, default=float)[:1200], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

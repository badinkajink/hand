#!/usr/bin/env python3
"""Builds docs/experiments/20261103-hom_task_conflict/20261104-hom_task_conflict.html: one SR2 finger layout for the four
HOM tasks against one layout per task, on performance alone (owner, 2026-11-03 20:3x: "how can we accommodate tasks
with conflicting morphology objectives? can we linearly interpolate? if not, is there some rule we can establish (pareto
front) for when blending, however we do it, is optimal, vs when we really should switch morphologies"; 22:3x: "we
shouldnt consider time, only performance. for desired performance, we should be able to mathematically formulate the
blend vs switch decision, given the current layout").

Data: the scopes <date>-conflict_<plant>.json of scripts/hom_task_conflict.py (scope --val: choices at seeds 0-2 scored
at seeds 20-24, smoothed variant), the size grid's 20261103-conflict_sizes.json, the refined shared layouts
<date>-conflict_refined_sv25.json with hom_grasp_refine.py's report under rows_refine/sv25/, the films' rows
media/<date>-films.json and the compute ledger of scripts/hom_compute.py. The design page's helpers and template
(scripts/fp_review_page.py) draw the tables and figures; the films reuse scripts/hom_score_refine_page.py's clip command.

    .venv/bin/python scripts/hom_task_conflict_page.py figs     # the figures from the JSON files
    .venv/bin/python scripts/hom_task_conflict_page.py films    # CPU clips through resguard (after the GPU queues)
    .venv/bin/python scripts/hom_task_conflict_page.py          # the page
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SCALE_HAND", "cad_v2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import hom_task_conflict as C  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

fill, table, img, film, esc = P.fill, P.table, P.img, P.film, P.esc
D = C.OUT
MEDIA = D / "media"
OUT = D / "20261104-hom_task_conflict.html"
TEX_CACHE = D / "texsvg_cache.json"
FILMS = MEDIA / "20261104-films.json"
SIZES = D / "20261103-conflict_sizes.json"
REFINE_PAGE = "../20261103-hom_score_refine/20261103-hom_score_refine.html"
PLANT_PAGE = "../20261029-foundational_pose_review/20261102-hom_plant_modulus.html"
DESIGN = "../20261029-foundational_pose_review/20261029-foundational_pose_hand_design.html"
TRENDS = "../20261029-foundational_pose_review/20261030-hom_layout_trends.html"
TASKS = tuple(C.TASKS)
NAME = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
OBJ = {"twist": "40 mm sphere", "roll": "40 mm sphere", "pinpush": "25 mm cylinder", "press": "25 mm cylinder"}
PLANTS = ("sv25", "k21", "k21mu0p4", "work10")
PLANT_NAME = {"sv25": "SCS0009, friction 1.0", "k21": "STS3032, friction 1.0", "k21mu0p4": "STS3032, friction 0.4",
              "work10": "Work plant, friction 1.0"}
PLANT_COLOR = {"sv25": "#1f4e9c", "k21": "#b03a2e", "k21mu0p4": "#d4881e", "work10": "#5b8c3a"}
TASK_COLOR = {"twist": "#1f4e9c", "roll": "#b03a2e", "pinpush": "#5b8c3a", "press": "#7b3f9e"}
TASK_MARK = {"twist": "o", "roll": "s", "pinpush": "^", "press": "D"}
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten", 11: "eleven"}
ALG = [0]


def conflict(plant):
    """The newest scope of a plant, or None."""
    ps = sorted(D.glob(f"*-conflict_{plant}.json"))
    return json.loads(ps[-1].read_text()) if ps else None


def refined(plant="sv25"):
    ps = sorted(D.glob(f"*-conflict_refined_{plant}.json"))
    return json.loads(ps[-1].read_text()) if ps else None


def refine_report(plant="sv25"):
    ps = sorted((C.REFINE / plant).glob("*-score_levels.json"))
    return json.loads(ps[-1].read_text()) if ps else None


def f2(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"


def xy(x):
    return f"({x[0]:g}, {x[1]:g})"


def andj(items):
    items = [str(i) for i in items]
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


# ----------------------------------------------------------------------------------------------- figures

def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                         "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "mathtext.fontset": "dejavuserif"})
    return plt


def save(fig, name):
    import matplotlib.pyplot as plt
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", MEDIA / name, flush=True)


def lattice(J, key):
    """(xs, ys, Z (T, ny, nx)) of a scope's matrix on the 25 x 25 lattice, NaN where the task is not reached."""
    M = J["val"]["matrix"]
    x0, y0, step = C.LATTICE
    xs = np.arange(x0, C.LATTICE_HI[0] + 1e-9, step)
    ys = np.arange(y0, C.LATTICE_HI[1] + 1e-9, step)
    Z = np.full((len(TASKS), len(ys), len(xs)), np.nan)
    for (x, y), reach, f in zip(M["x"], M["reach"], M[key]):
        i, j = int(round((x - x0) / step)), int(round((y - y0) / step))
        for t in range(len(TASKS)):
            if reach[t]:
                Z[t, j, i] = f[t]
    return xs, ys, Z


def fig_maps(J, plant):
    """Each task's validated score over the lattice, with the four specialists and the J_1 layout marked."""
    plt = plt_setup()
    xs, ys, Z = lattice(J, "val")
    c = J["val"]["chosen"]["val"]
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 9.2), constrained_layout=True, sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    for t, (ax, task) in enumerate(zip(axs.ravel(), TASKS)):
        im = ax.pcolormesh(xs, ys, np.ma.masked_invalid(Z[t]), cmap=cmap, vmin=0, vmax=1, shading="nearest")
        for s in c["specialists"]:
            ax.plot(*s["x"], TASK_MARK[s["task"]], ms=11, mfc=TASK_COLOR[s["task"]], mec="white", mew=1.6)
        ax.plot(*c["J"][0]["layouts"][0]["x"], "*", ms=20, mfc="white", mec="black", mew=1.4)
        ax.set_title(f"{NAME[task]}, {OBJ[task]}")
        ax.set_aspect("equal")
    for ax in axs[1]:
        ax.set_xlabel(r"$x_{sep}$ (mm)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$y_{sep}$ (mm)")
    cb = fig.colorbar(im, ax=axs, shrink=0.75, pad=0.02)
    cb.set_label("Validated score, seeds 20-24")
    handles = [plt.Line2D([], [], ls="", marker=TASK_MARK[t], ms=10, mfc=TASK_COLOR[t], mec="white",
                          label=f"Best for the {NAME[t].lower()}") for t in TASKS]
    handles.append(plt.Line2D([], [], ls="", marker="*", ms=16, mfc="white", mec="black", label="Best single layout"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    save(fig, f"20261104-maps_{plant}.png")


def fig_curves(Js):
    """Left: J_K / J_1 against K per plant (chosen at seeds 0-2, scored at 20-24). Middle: generalists against alpha at
    seeds 20-24, each line ending at the last level with one. Right: K* against alpha at seeds 20-24. A plant without
    validation is drawn dashed from seeds 0-2."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    handles = []
    for pl, J in Js.items():
        v = J.get("val")
        col, ls = PLANT_COLOR[pl], ("-" if v else "--")
        if v:
            js = [r["score"] for r in v["chosen"]["val"]["J"]]
            lev = v["coverage"]["val"]["levels"]
        else:
            js = [r["J"] for r in J["J"]]
            lev = J["coverage"]["levels"]
        axs[0].plot(range(1, len(js) + 1), np.array(js) / js[0], ls, marker="o", color=col)
        al = np.array([r["alpha"] for r in lev])
        ng = np.array([r["n_generalists"] for r in lev], float)
        axs[1].semilogy(al[ng > 0], ng[ng > 0], ls, color=col)
        axs[2].step(al, [r["K_star"] for r in lev], ls, where="post", color=col)
        handles.append(plt.Line2D([], [], ls=ls, color=col, label=PLANT_NAME[pl] + ("" if v else ", seeds 0-2 only")))
    axs[0].set_xlabel("Layouts $K$")
    axs[0].set_ylabel("$J_K / J_1$")
    axs[0].set_xticks([1, 2, 3, 4])
    axs[1].set_xlabel(r"Desired level $\alpha$")
    axs[1].set_ylabel("Generalists $N_G$")
    axs[2].set_xlabel(r"Desired level $\alpha$")
    axs[2].set_ylabel("Fewest layouts $K^*$")
    axs[2].set_yticks([1, 2, 3, 4])
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.17))
    save(fig, "20261104-curves.png")


def fig_pairs(J, plant):
    """The six task pairs at seeds 20-24: every layout, the front, the straight segment between the two specialists
    (chosen at seeds 0-2) and the J_1 layout."""
    plt = plt_setup()
    M = J["val"]["matrix"]
    F = np.array(M["val"])
    c = J["val"]["chosen"]["val"]
    calls = M["calls"]
    j1 = calls.index(c["J"][0]["layouts"][0]["call"])
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 9.0), constrained_layout=True)
    for ax, pr in zip(axs.ravel(), c["pairs"]):
        i, j = (TASKS.index(t) for t in pr["tasks"])
        ax.plot(F[:, i], F[:, j], ".", color="#9a9a9a", ms=5, zorder=1)
        fr = C.front(F[:, i], F[:, j])
        fx, fy = F[fr, i], F[fr, j]
        ax.step(fx, fy, "-", color="black", lw=1.4, where="post", zorder=2)
        seg = pr["segment"]
        sx = [s["score"][0] for s in seg]
        sy = [s["score"][1] for s in seg]
        ax.plot(sx, sy, "-", color="#d4881e", lw=0.9, alpha=0.8, zorder=3)
        ax.scatter(sx, sy, c=[s["l"] for s in seg], cmap="cividis", vmin=0, vmax=1, s=38, zorder=4,
                   edgecolor="black", lw=0.5)
        for key, task in (("a", pr["tasks"][0]), ("b", pr["tasks"][1])):
            k = calls.index(pr[key]["call"])
            ax.plot(F[k, i], F[k, j], TASK_MARK[task], ms=12, mfc=TASK_COLOR[task], mec="white", mew=1.6, zorder=5)
        ax.plot(F[j1, i], F[j1, j], "*", ms=18, mfc="white", mec="black", mew=1.3, zorder=6)
        ax.set_xlabel(f"{NAME[pr['tasks'][0]]} score")
        ax.set_ylabel(f"{NAME[pr['tasks'][1]]} score")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], ls="", marker=".", ms=8, color="#9a9a9a", label="Layouts"),
               plt.Line2D([], [], color="black", lw=1.4, label="Front"),
               plt.Line2D([], [], color="#d4881e", lw=0.9, marker="o", mfc="#7f7f7f", label="Straight segment"),
               plt.Line2D([], [], ls="", marker="*", ms=14, mfc="white", mec="black", label="Best single layout")]
    handles += [plt.Line2D([], [], ls="", marker=TASK_MARK[t], ms=10, mfc=TASK_COLOR[t], mec="white",
                           label=f"Best for the {NAME[t].lower()}") for t in TASKS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.09))
    save(fig, f"20261104-pairs_{plant}.png")


def fig_sizes(S):
    """The size grid: generalists and K* against alpha, with the largest mutually incompatible set; alpha*_1 over windows
    of consecutive sizes per task."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    lev = S["coverage"]["levels"]
    al = [r["alpha"] for r in lev]
    axs[0].step(al, [r["K_star"] for r in lev], "-", where="post", color="#1f4e9c", lw=2, label="Fewest layouts $K^*$")
    axs[0].step(al, [len(r["mutually_incompatible"]) for r in lev], "--", where="post", color="#b03a2e",
                label="Largest mutually incompatible set")
    axs[0].set_xlabel(r"Desired level $\alpha$")
    axs[0].set_ylabel("Layouts")
    axs[0].legend(frameon=False, loc="upper left")
    axs[0].grid(alpha=0.3)
    for task, by in S["breadth"].items():
        n = sorted(int(k) for k in by)
        axs[1].plot(n, [min(w["alpha1"] for w in by[str(k)]) for k in n], "-", color=TASK_COLOR[task],
                    marker=TASK_MARK[task], label=NAME[task])
    axs[1].set_xlabel("Consecutive object sizes")
    axs[1].set_ylabel(r"One layout's level $\alpha^*_1$")
    axs[1].set_xticks([1, 2, 3, 4])
    axs[1].set_ylim(0.5, 1.02)
    axs[1].legend(frameon=False, loc="lower left")
    axs[1].grid(alpha=0.3)
    save(fig, "20261104-sizes.png")


def fig_refined(Rf):
    """Per task, each shared layout's validated S1, S2 and S3, the layouts in a fixed order."""
    plt = plt_setup()
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 8.6), constrained_layout=True, sharey=True)
    lays = Rf["layouts"]
    n = len(lays)
    w = 0.26
    cols = {"S1": "#9a9a9a", "S2": "#1f4e9c", "S3": "#b03a2e"}
    for t, (ax, task) in enumerate(zip(axs.ravel(), TASKS)):
        for k, lv in enumerate(C.LEVELS):
            v = np.array(Rf["levels"][lv]["val"])[:, t]
            ax.bar(np.arange(n) + (k - 1) * w, v, w, color=cols[lv], label=lv)
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels([f"{L_['x'][0]:g}, {L_['x'][1]:g}" for L_ in lays], rotation=60, ha="right", fontsize=13)
        ax.set_title(f"{NAME[task]}, {OBJ[task]}")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
    for ax in axs[:, 0]:
        ax.set_ylabel("Validated score")
    for ax in axs[1]:
        ax.set_xlabel(r"Layout ($x_{sep}$, $y_{sep}$) in mm")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    save(fig, "20261104-refined_sv25.png")


def cmd_figs():
    Js = {pl: J for pl in PLANTS if (J := conflict(pl))}
    if "sv25" in Js and "val" in Js["sv25"]:
        fig_maps(Js["sv25"], "sv25")
        fig_pairs(Js["sv25"], "sv25")
    fig_curves(Js)
    fig_sizes(json.loads(SIZES.read_text()))
    if (Rf := refined()):
        fig_refined(Rf)
    return 0


# ----------------------------------------------------------------------------------------------- films

def val_of(J, call, t):
    M = J["val"]["matrix"]
    return M["val"][M["calls"].index(call)][t]


def film_specs():
    """Per task on the SCS0009: the task's specialist and the best single layout (J_1), each with its S1 grasp at HOM's
    defaults and, once the refined runs exist, its S3 pick (grasp, forces and contact modes), at seed 20 (a validation
    seed)."""
    import hom_grasp_refine as R
    R.ROWS_ROOT = C.REFINE
    J = conflict("sv25")
    c = J["val"]["chosen"]["val"]
    j1 = c["J"][0]["layouts"][0]["call"]
    rep = refine_report()
    specs = []
    for t, task in enumerate(TASKS):
        sc = next(s["call"] for s in c["specialists"] if s["task"] == task)
        lays = {L_["call"]: L_ for L_ in R.layouts(task, "sv25")}
        val = {e["call"]: e for e in (rep["tasks"][task]["layouts"] if rep and task in rep["tasks"] else [])}
        p3 = R.picks(task, "sv25", "full") if rep else {}
        for lv in ("S1", "S3"):
            for call in (sc, j1):
                L_ = lays[call]
                if lv == "S1":
                    v, prm, gpu = L_["v1"], {}, val_of(J, call, t)
                elif call in p3 and val.get(call, {}).get("s3_val") is not None:
                    v, prm, gpu = p3[call]["v"], p3[call]["prm"], val[call]["s3_val"]
                else:
                    continue
                specs.append({"set": "conflict", "plant": "sv25", "task": task, "call": call, "xy": L_["xy"],
                              "d6": L_["d"], "role": "specialist" if call == sc else "J_1", "pick": lv, "v": v,
                              "prm": prm, "gpu": gpu, "seed": 20,
                              "top": f"{NAME[task]}, {L_['xy'][0]:g} x {L_['xy'][1]:g} mm, {lv}",
                              "out": str(MEDIA / f"20261104-clip_{task}_{call}_{lv}.mp4")})
    return specs


def cmd_films():
    """CPU clips through resguard (scripts/hom_score_refine_page.py clip, the plant's environment set here), then one film
    per task: S1 above S3, the specialist left of the best single layout."""
    specs = film_specs()
    for s in specs:
        if Path(s["out"]).with_suffix(".json").exists():
            continue
        import hom_grasp_refine as R
        hp, scenes, tag = R.PLANTS[s["plant"]]
        env = dict(os.environ, HOS_PLANT=hp, HOS_SCENES=scenes, HOM_ROWS_TAG=tag, CUDA_VISIBLE_DEVICES="",
                   MUJOCO_GL="egl", NUMPY_MADVISE_HUGEPAGE="0")
        cmd = [str(Path.home() / ".claude/bin/resguard.sh"), "run", "--mem", "3G", "--cpu", "200", "--",
               str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/hom_score_refine_page.py"), "clip", json.dumps(s)]
        t0 = time.time()
        while True:
            rc = subprocess.call(cmd, env=env, cwd=ROOT)
            if rc != 3:
                break
            print(time.strftime("%T"), "refused by resguard; retry in 120 s", flush=True)
            time.sleep(120)
        if rc:
            raise SystemExit(f"clip {s['out']} failed (exit {rc})")
        info = json.loads(Path(s["out"]).with_suffix(".json").read_text())
        info["wall_s"] = round(time.time() - t0, 1)
        Path(s["out"]).with_suffix(".json").write_text(json.dumps(info))
    import hom_plant_page as HP
    for task in TASKS:
        cl = [s["out"] for s in specs if s["task"] == task]
        if cl:
            HP.compose(cl, MEDIA / f"20261104-film_{task}.mp4", cols=2)
    res = [dict(s, **json.loads(Path(s["out"]).with_suffix(".json").read_text())) for s in specs]
    FILMS.write_text(json.dumps(res, indent=1))
    print("wrote", FILMS.relative_to(ROOT))
    return 0


# ----------------------------------------------------------------------------------------------- page sections

def sec_terms():
    terms = [
        ("Layout", "the palm-frame mounts of the SR2 hand's three fingers, here a symmetric tripod in two coordinates. "
         "The thumb's mount lies $x_{sep}$ from the pair's, and the index's lies $y_{sep}$ from the middle's. The "
         "lattice has 625 layouts in steps of 5 mm [1]. Its $x_{sep}$ runs from 40 to 160 mm and its $y_{sep}$ from "
         "50 to 170 mm. A layout is written ($x_{sep}$, $y_{sep}$) in mm. The hand's gantries can move the mounts between tasks."),
        ("HOM tasks", "the four tasks of the design page under HOM control [2]. The twist turns a 40 mm sphere about "
         "the vertical, and the roll rolls it along the thumb's pad. In the pin-push and the press the middle finger "
         "turns a 25 mm cylinder about the pinch of thumb and index. A layout reaches a task when a feasible grasp for "
         "it is found by the pose toy."),
        ("Task score $f_t(x)$", "the score of layout $x$ on task $t$, in [0, 1]. It is 0 when the object is not held "
         "at the end: it fell more than 15 mm, or fewer than two fingers press on it. The twist "
         "scores its three-pad angle over 90 deg, capped at 1. The roll scores $1/(1+e_\\omega)$, with $e_\\omega$ the "
         "RMS error of the rolling rate over the commanded rate. The pin-push and the press score one less the RMS "
         "error of the cylinder's angle over the stroke of 15 or 20 deg. A layout that does not reach a task scores 0 "
         "on it."),
        ("S1, S2, S3", "three scores of a layout on a task. S1 is scored with HOM's defaults and the best of up to 16 "
         "grasps, as in the grids [3]. S2 adds a search over the grasp, and S3 a search over the grasp, the commanded "
         "forces and the contact modes together [4]. Unless a section names S2 or S3, $f_t(x)$ is S1."),
        ("Search and validation seeds", "a seed sets the object's start offset, up to 1 mm and 2 deg. The grids scored "
         "every grasp at seeds 0-2, and the best grasp of each layout was chosen there. Here each layout's chosen "
         "grasp ran again at seeds 20-24, which no grid or search used. A validated score is the mean at seeds 20-24."),
        ("Specialist $x_t^*$", "task $t$'s best layout, $x_t^* = \\arg\\max_x f_t(x)$, with ties going to the layout "
         "with the higher mean score on the other tasks. Its score is the task's best, $f_t^* = f_t(x_t^*)$."),
        ("Desired level $\\alpha$", "the fraction of each task's best score that a layout must reach, the same for "
         "every task: the threshold of task $t$ is $\\alpha f_t^*$."),
        ("Capable tasks $C_\\alpha(x)$", "the tasks on which layout $x$ reaches the desired level."),
        ("Generalist", "a layout capable of every task at level $\\alpha$. $N_G(\\alpha)$ counts them."),
        ("Fewest layouts $K^*(\\alpha)$", "the smallest number of layouts whose capable tasks together are all the "
         "tasks, an exact minimum set cover. With $K^*$ layouts each task runs on a layout capable of it."),
        ("Level of $K$ layouts $\\alpha^*_K$", "the highest level at which $K$ layouts cover every task, found on "
         "levels in steps of 0.01. $\\alpha^*_1$ is the best generalist's worst task, as a fraction of that task's "
         "best, and is given exactly."),
        ("Level of a set of tasks $\\lambda(A)$", "the highest level that one layout reaches on every task of the set "
         "$A$, $\\lambda(A) = \\max_x \\min_{t \\in A} f_t(x)/f_t^*$. For all four tasks it is $\\alpha^*_1$."),
        ("Incompatible tasks", "two tasks that no layout does both of at level $\\alpha$. Tasks that are pairwise "
         "incompatible need one layout each, so the largest such set is a lower bound on $K^*(\\alpha)$."),
        ("Best $K$ layouts $J_K$", "the highest mean task score over the four tasks with $K$ layouts, each task run on "
         "the better of them for it. $J_1$ is one layout for all tasks and $J_4$ one layout per task."),
        ("Pair front", "for two tasks, the layouts that no other layout beats on both scores."),
        ("Straight segment", "the lattice layouts nearest to the points of the straight line between two tasks' "
         "specialists in ($x_{sep}$, $y_{sep}$), at 21 evenly spaced points."),
        ("Dominance margin", "the largest amount by which some layout of the pair front beats a given layout on both "
         "tasks; 0 for a layout on the front."),
        ("Blend gain", "for two tasks and a weight $w$ on the first, the best layout's $w f_a + (1-w) f_b$ less that of "
         "the better of the two specialists. It is positive when a third layout beats both specialists on the mix."),
        ("Switch gain", "at $w = 0.5$, the mean of the two specialists' own scores less the best single layout's "
         "mean: what one layout per task adds for that pair."),
        ("Smoothed score", "a layout's score averaged over its 3 x 3 block of lattice neighbours (5 mm steps), 0 where "
         "the layout does not reach the task. It stands for seed noise and for a gantry that places the fingers "
         "within one lattice step of the commanded layout."),
        ("Chosen at seeds 0-2, scored at seeds 20-24", "a specialist, a set of layouts or a generalist is picked on "
         "the scores at seeds 0-2 and its validated scores are reported. Each task's best on the validation side is "
         "then the validated score of the specialist picked at seeds 0-2."),
        ("Spearman's rho", "the rank correlation of two scores over the same layouts; 1 means the same order."),
        ("Plants", "the finger servo model and the scene. The SCS0009 model (0.72 N m/rad, 0.18 N m) with pads at "
         "2.5 MPa is the built hand's. The STS3032 model (2.1 N m/rad, 0.515 N m) runs at scene friction 1.0 and at "
         "0.4, the friction of TPU on bare PLA. The work plant of 10-30 has 5.6 times the SCS0009's gain and torque, "
         "with pads at 10 MPa [3]."),
        ("Batch and run time", "the GPU's seconds per rollout. Run time is the rollout loop of one batch; batch time "
         "adds the batch's model build and its scoring."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def alg(caption, body):
    ALG[0] += 1
    return (f'<figure class="alg"><figcaption><b>Algorithm {ALG[0]}.</b> {caption}</figcaption>'
            f"\\begin{{algorithmic}}[1]\n{body}\n\\end{{algorithmic}}</figure>")


def sec_method(Js):
    J = Js["sv25"]
    a = alg("The decision at the current layout $x_0$ for a desired level $\\alpha$ (<code>hom_task_conflict.decide"
            "</code>).", r"""\Require scores \(f_t(x)\) of every layout on every task \(t \in T\), level \(\alpha\), current layout \(x_0\)
\State \(C \gets \{t \in T : f_t(x_0) \ge \alpha f_t^*\}\)
\If{\(C = T\)}
  \State \Return stay at \(x_0\)
\ElsIf{\(G_\alpha \neq \emptyset\)}
  \State \Return move once to a generalist \(x \in G_\alpha\)
\Else
  \State \(S \gets\) a minimum set of layouts whose capable tasks cover \(T \setminus C\)
  \State \Return keep \(x_0\) for the tasks in \(C\) and add the \(|S|\) layouts of \(S\)
\EndIf""")
    return fill("""<p>The grids of the plant page scored every lattice layout's S1 on the four tasks at seeds 0-2 [3]. On
the SCS0009 grid, <<n>> layouts reach at least one task and <<n_all>> reach all four. With equal weights, the best
$K$ layouts give</p>
$$J_K = \\max_{\\{T_1, \\dots, T_K\\}} \\sum_{k=1}^{K} \\max_x \\frac{1}{|T|} \\sum_{t \\in T_k} f_t(x),$$
<p>a maximum over the partitions of the task set $T$ into $K$ groups, each group on its best layout. $J_1$ is one
layout for all four tasks, and $J_4 = \\frac{1}{|T|} \\sum_t f_t^*$ gives each task its specialist. The ratio
$J_4/J_1$ measures what switching layouts adds to the mean score.</p>
<p>A mean can hide a task done badly, so the decision is stated per task, at a desired level $\\alpha$ of each task's
best:</p>
$$C_\\alpha(x) = \\{t \\in T : f_t(x) \\ge \\alpha f_t^*\\}, \\qquad G_\\alpha = \\{x : C_\\alpha(x) = T\\}, \\qquad
N_G(\\alpha) = |G_\\alpha|,$$
$$K^*(\\alpha) = \\min\\Big\\{|S| : \\bigcup_{x \\in S} C_\\alpha(x) = T\\Big\\}, \\qquad
\\alpha^*_K = \\max\\{\\alpha : K^*(\\alpha) \\le K\\}, \\qquad \\alpha^*_1 = \\max_x \\min_t \\frac{f_t(x)}{f_t^*}.$$
<p>One layout does every task at level $\\alpha$ exactly when $\\alpha \\le \\alpha^*_1$. Above it the hand must
switch, and it needs $K^*(\\alpha)$ layouts. If a set $I$ of tasks is pairwise incompatible at $\\alpha$, then
$K^*(\\alpha) \\ge |I|$. The minimum cover is found exactly, by a breadth-first search over the subsets of tasks
reached by the layouts whose capable tasks no other layout's include. Algorithm 1 gives the decision at the current
layout.</p>""", n=J["n_layouts"], n_all=J["n_all_tasks"]) + a + """
<p>For two tasks $a$ and $b$ the blend is examined on weighted mixes. The best layout of the mix
$w f_a + (1-w) f_b$ is compared with the two specialists for 21 weights $w$. The straight segment between the
specialists $x_a^*$ and $x_b^*$ is the linear interpolation of the layout. A segment layout $x$ has the dominance
margin</p>
$$m(x) = \\max\\Big(0,\\; \\max_{y \\in \\mathcal{F}} \\min\\big(f_a(y) - f_a(x),\\, f_b(y) - f_b(x)\\big)\\Big),$$
<p>with $\\mathcal{F}$ the pair front. A segment that interpolates the scores keeps its margins near 0.</p>
<p>Each specialist's score is the largest of up to 561 means over three seeds, so it is biased upward. Every reached
layout therefore ran again with its chosen grasp at seeds 20-24 (<code>hom_task_conflict.py validate</code>; 1,860
layout-task pairs and 9,300 rollouts per plant). Choices made at seeds 0-2 are scored at seeds 20-24, and the same
quantities are also computed from seeds 20-24 alone. The smoothed variant replaces each score by its 3 x 3 block mean
before the choice. The work plant's grid has seeds 0-2 only.</p>"""


def neighbours(J, key="val"):
    """Per task: the median and the 95th percentile of |f(x) - f(y)| over lattice neighbours 5 mm apart that both
    reach the task."""
    M = J["val"]["matrix"]
    X, R, F = np.array(M["x"]), np.array(M["reach"], bool), np.array(M[key])
    idx = {tuple(x): i for i, x in enumerate(X.tolist())}
    out = {}
    for t, task in enumerate(TASKS):
        d = [abs(F[i, t] - F[j, t]) for i, (x, y) in enumerate(X) for dx, dy in ((5, 0), (0, 5))
             if (j := idx.get((x + dx, y + dy))) is not None and R[i, t] and R[j, t]]
        out[task] = (float(np.median(d)), float(np.percentile(d, 95)))
    return out


def subset_levels(J):
    """Per subset A of two or more tasks: the layout x_A chosen at seeds 0-2 for the highest min_{t in A} f_t(x) / f_t*,
    that level at seeds 0-2, its validated level (each task's best the validated score of its specialist chosen at seeds
    0-2), and the level chosen and scored at seeds 20-24 alone."""
    import itertools
    M = J["val"]["matrix"]
    F0, F1, X = np.array(M["search"]), np.array(M["val"]), np.array(M["x"])
    T = len(TASKS)
    spec = [C.specialist(F0, t) for t in range(T)]
    b0, bv, b1 = F0.max(0), F1[spec, np.arange(T)], F1.max(0)
    out = []
    for r in range(2, T + 1):
        for A in itertools.combinations(range(T), r):
            A = list(A)
            g0 = (F0[:, A] / b0[A]).min(1)
            x = int(np.argmax(g0))
            g1 = (F1[:, A] / b1[A]).min(1)
            out.append({"tasks": [TASKS[a] for a in A], "x": X[x].tolist(), "search": float(g0[x]),
                        "val": float((F1[x, A] / bv[A]).min()), "val_only": float(g1.max()),
                        "x_val_only": X[int(np.argmax(g1))].tolist()})
    return out


def subset_table(J):
    rows = [[", ".join(NAME[t].lower() if k else NAME[t] for k, t in enumerate(e["tasks"])), xy(e["x"]),
             f2(e["search"]), f2(e["val"]), f2(e["val_only"])] for e in subset_levels(J)]
    return table(["Tasks $A$", "Layout chosen at seeds 0-2", "$\\lambda(A)$ at seeds 0-2", "Same layout at seeds 20-24",
                  "Chosen and scored at seeds 20-24"], rows,
                 "The level $\\lambda(A)$ of each set of tasks on the SCS0009: the highest level that one layout "
                 "reaches on every task of the set. The tasks of $A$ can share one layout at a desired level "
                 "$\\alpha$ exactly when $\\lambda(A) \\ge \\alpha$.", num_from=2)


def payoff_table(J):
    c = J["val"]["chosen"]["val"]
    best = c["best_score"]
    rows = []
    for s in c["specialists"]:
        sc = s["score"]
        rows.append([f"Best for the {NAME[s['task']].lower()}, {xy(s['x'])}", *[f2(v) for v in sc], f2(np.mean(sc)),
                     f2(min(v / b for v, b in zip(sc, best)))])
    g = c["J"][0]["layouts"][0]
    rows.append([f"Best single layout, {xy(g['x'])}", *[f2(v) for v in g["score"]], f2(np.mean(g["score"])),
                 f2(min(v / b for v, b in zip(g["score"], best)))])
    o = J["val"]["chosen"]["val_only"]
    same = ([s["call"] for s in c["specialists"]] + [g["call"]]
            == [s["call"] for s in o["specialists"]] + [o["J"][0]["layouts"][0]["call"]])
    return table(["Layout chosen at seeds 0-2", *[NAME[t] for t in TASKS], "Mean", "Worst task over its best"], rows,
                 "Validated scores (seeds 20-24) of each task's specialist and of the best single layout on the "
                 "SCS0009. " + ("The same five layouts are chosen at seeds 20-24 alone. " if same else
                                "At seeds 20-24 alone other layouts are chosen. ")
                 + "The last column is the lowest of $f_t(x) / f_t^*$ over the four tasks.")


def decision_table(J):
    ds = J["val"]["decisions"]["val"]
    starts = list(dict.fromkeys(d["x0"] for d in ds))
    rows = []
    for al in C.SHOW:
        cells = []
        for s0 in starts:
            d = next(d for d in ds if d["x0"] == s0 and abs(d["alpha"] - al) < 1e-9)
            cells.append("Stay" if d["action"] == "stay" else "Move once" if d["action"] == "move to a generalist"
                         else f"Add {d['extra_layouts']}")
        n = ds[[d["alpha"] for d in ds].index(al)]["n_generalists"]
        rows.append([f"{al:.2f}", str(n), *cells])
    head = ["Level $\\alpha$", "Generalists"] + [s0.replace("best for ", "From the best for the ").replace(
        "best generalist", "From the best single layout").replace("pinpush", "pin-push") for s0 in starts]
    return table(head, rows, "The decision of Algorithm 1 on the SCS0009 at seeds 20-24, from each specialist and "
                 "from the best single layout. 'Add k': keep the current layout for its capable tasks and add k "
                 "layouts.", num_from=1)


def pair_table(J):
    v, s = J["val"]["chosen"]["val"]["pairs"], J["val"]["chosen"]["smooth"]["pairs"]
    rows = []
    for p, q in zip(v, s):
        rows.append([f"{NAME[p['tasks'][0]]}, {NAME[p['tasks'][1]].lower()}", f"{p['distance_mm']:.0f}",
                     f2(p["rho"]["score"]), f2(p["blend_gain_max"]["score"]), f2(p["switch_gain_half"]["score"]),
                     f2(p["segment_margin_max"]["score"]), f2(q["segment_margin_max"]["score"])])
    return table(["Tasks", "Specialists apart (mm)", "Rho", "Largest blend gain", "Switch gain at w = 0.5",
                  "Largest segment margin", "Same, smoothed"], rows,
                 "Pairs of tasks on the SCS0009, chosen at seeds 0-2 and scored at seeds 20-24. Rho is over the "
                 "layouts that score above 0 on both tasks. The last column chooses and scores on the smoothed "
                 "scores.")


def sec_sv25(Js):
    J = Js["sv25"]
    v = J["val"]
    c, co, sm = v["chosen"]["val"], v["coverage"]["val"], v["chosen"]["smooth"]
    js = [r["score"] for r in c["J"]]
    jsrch = [r["J"] for r in J["J"]]
    lev = {r["alpha"]: r for r in co["levels"]}
    kept = {r["alpha"]: r for r in c["levels"]}
    g = c["J"][0]["layouts"][0]
    nz = v["noise"]
    nb_ = neighbours(J)
    rhos = [nz[t]["rho"] for t in TASKS]
    seg = [p["segment_margin_max"]["score"] for p in c["pairs"]]
    segs = [p["segment_margin_max"]["score"] for p in sm["pairs"]]
    worst = max(c["pairs"], key=lambda p: p["segment_margin_max"]["score"])
    n0, f0 = P.TAB[0], P.FIG[0]
    t1 = payoff_table(J)
    fmaps = img(MEDIA / "20261104-maps_sv25.png", "Validated score (seeds 20-24) of every lattice layout on each task "
                "on the SCS0009. Grey: the task's pose is not reached. Markers: each task's specialist and the best "
                "single layout, all chosen at seeds 0-2.")
    inc = lev[0.9]["mutually_incompatible"]
    p1 = f"""<p>On the SCS0009 the best single layout is {xy(g['x'])}. Its mean validated score over the four tasks
is {js[0]:.3f}, against {js[-1]:.3f} with each task on its own specialist (Table {n0 + 1}): $J_4/J_1$ =
{js[-1] / js[0]:.2f}. Two layouts give {js[1]:.3f} and three give {js[2]:.3f}. At seeds 0-2 the same choices score
{jsrch[0]:.3f} and {jsrch[-1]:.3f}, a ratio of {jsrch[-1] / jsrch[0]:.2f}. The specialists of the four tasks lie
{min(p['distance_mm'] for p in c['pairs']):.0f} to {max(p['distance_mm'] for p in c['pairs']):.0f} mm apart (Figure
{f0 + 1}). Between layouts 5 mm apart a task's score changes by a median {min(nb_[t][0] for t in TASKS):.2f}
({NAME[min(TASKS, key=lambda t: nb_[t][0])].lower()}) to {max(nb_[t][0] for t in TASKS):.2f}
({NAME[max(TASKS, key=lambda t: nb_[t][0])].lower()}). In one pair of neighbours in twenty it changes by more than
{min(nb_[t][1] for t in TASKS):.2f} to {max(nb_[t][1] for t in TASKS):.2f}.</p>"""
    p2 = f"""<p>At seeds 20-24 alone, {lev[0.6]['n_generalists']} layouts are generalists at level 0.6,
{lev[0.7]['n_generalists']} at 0.7 and {lev[0.8]['n_generalists']} at 0.8. None is at 0.85. The best single layout is
also the best generalist: its worst task reaches {co['best_generalist_worst']:.2f} of that task's best, so
$\\alpha^*_1$ = {co['alpha_star']['1']:.2f}. Two layouts reach $\\alpha^*_2$ = {co['alpha_star']['2']:.2f} and three
reach $\\alpha^*_3$ = {co['alpha_star']['3']:.2f}. At level 0.9 three layouts are needed, and the
{andj([NAME[t].lower() for t in inc])} are mutually incompatible there. Of the
{kept[0.6]['n_generalists_choose']} generalists at level 0.6 found at seeds 0-2, {kept[0.6]['n_generalists_kept']}
reach that level again at seeds 20-24. At 0.7, {kept[0.7]['n_generalists_kept']} of
{kept[0.7]['n_generalists_choose']} do, and at 0.8, {kept[0.8]['n_generalists_kept']} of
{kept[0.8]['n_generalists_choose']}.</p>"""
    sl = {tuple(e["tasks"]): e for e in subset_levels(J)}
    pairs_ = sorted((e for e in sl.values() if len(e["tasks"]) == 2), key=lambda e: -e["val_only"])
    hi_, lo_ = pairs_[0], pairs_[-1]
    tsub = subset_table(J)
    p2b = f"""<p>The same rule applies to any set of tasks $A$. Its level is
$\\lambda(A) = \\max_x \\min_{{t \\in A}} f_t(x)/f_t^*$, and the tasks of $A$ can share a layout at level
$\\alpha$ exactly when $\\lambda(A) \\ge \\alpha$ (Table {n0 + 2}). At seeds 20-24 the pairs reach
{pairs_[-1]['val_only']:.2f} to {pairs_[0]['val_only']:.2f}. The {NAME[hi_['tasks'][0]].lower()} and the
{NAME[hi_['tasks'][1]].lower()} share a layout best ({hi_['val_only']:.2f}), and the
{NAME[lo_['tasks'][0]].lower()} and the {NAME[lo_['tasks'][1]].lower()} worst ({lo_['val_only']:.2f}). Every set of
three tasks and the four together lie at {min(e['val_only'] for e in sl.values() if len(e['tasks']) > 2):.2f} to
{max(e['val_only'] for e in sl.values() if len(e['tasks']) > 2):.2f}.</p>"""
    t2 = decision_table(J)
    p3 = f"""<p>Table {n0 + 3} applies Algorithm 1. Up to level 0.8 each specialist moves once to a generalist, and the
best single layout stays. Above level {co['best_generalist_worst']:.2f} no generalist exists, and each start adds the
layouts of a minimum cover of the tasks it misses.</p>"""
    t3 = pair_table(J)
    fpairs = img(MEDIA / "20261104-pairs_sv25.png", "The six pairs of tasks on the SCS0009 at seeds 20-24. Grey: every "
                 "layout; black: the pair front; star: the best single layout. The straight segment runs from the first "
                 "task's specialist (dark) to the second's (yellow). The specialists and the segment are chosen at "
                 "seeds 0-2.")
    p4 = f"""<p>The straight segment between two specialists passes through layouts far below the front (Figure
{f0 + 2}, Table {n0 + 4}). Its largest dominance margin is {min(seg):.2f} to {max(seg):.2f} over the six pairs, and
{max(seg):.2f} between the {NAME[worst['tasks'][0]].lower()} and the {NAME[worst['tasks'][1]].lower()}. A third layout
beats both specialists on some mix of each pair but one: the largest blend gain is up to
{max(p['blend_gain_max']['score'] for p in c['pairs']):.2f}. One layout per task adds
{min(p['switch_gain_half']['score'] for p in c['pairs']):.2f} to
{max(p['switch_gain_half']['score'] for p in c['pairs']):.2f} to a pair's mean at $w = 0.5$.</p>"""
    roll = next((s for s in sm["specialists"] if s["task"] == "roll"), None)
    p5 = f"""<p>The layouts' order repeats at other seeds. Between seeds 0-2 and 20-24 Spearman's rho is {min(rhos):.2f} to
{max(rhos):.2f} per task, and the median change of a layout's score is {min(nz[t]['abs_diff_median'] for t in TASKS):.3f}
to {max(nz[t]['abs_diff_median'] for t in TASKS):.3f}. The differences between neighbours come from the layout and its
own best grasp. On the smoothed scores the segment margins fall to {min(segs):.2f} to {max(segs):.2f}, and
$\\alpha^*_1$ is {sm['max_min']['alpha_score']:.2f} at seeds 20-24. Smoothing also lowers each task's best, and the
mean of the best smoothed single layout is {sm['J'][0]['score']:.2f}.""" + (
        f""" The roll's best layout on the smoothed scores, {xy(roll['x'])}, scores {roll['choose'][1]:.2f} smoothed at
seeds 0-2 and {val_of(J, roll['call'], 1):.2f} at its own grasp at seeds 20-24. That grasp holds the sphere at seeds
0-2 and drops it at all five validation seeds.""" if roll and val_of(J, roll["call"], 1) < 0.05 else "") + "</p>"
    return p1 + t1 + fmaps + p2 + p2b + tsub + t2 + p3 + t3 + fpairs + p4 + p5


def plant_rows(Js):
    rows = []
    for pl in PLANTS:
        J = Js.get(pl)
        if not J:
            continue
        if "val" in J:
            c, co = J["val"]["chosen"]["val"], J["val"]["coverage"]["val"]
            js = [r["score"] for r in c["J"]]
            lev = {r["alpha"]: r for r in co["levels"]}
            seeds = "20-24"
        else:
            js = [r["J"] for r in J["J"]]
            co = J["coverage"]
            lev = {r["alpha"]: r for r in co["levels"]}
            seeds = "0-2"
        a = co["alpha_star"]
        rows.append([PLANT_NAME[pl], seeds, f2(js[0]), f2(js[-1]), f2(js[-1] / js[0]), f2(a["1"]), f2(a["2"]),
                     f2(a["3"]), str(lev[0.7]["n_generalists"]), str(lev[0.8]["n_generalists"]),
                     str(lev[0.9]["K_star"]), str(len(lev[0.9]["mutually_incompatible"]))])
    return rows


def sec_plants(Js):
    n0, f0 = P.TAB[0], P.FIG[0]
    rows = plant_rows(Js)
    t = table(["Plant", "Seeds", "$J_1$", "$J_4$", "$J_4/J_1$", "$\\alpha^*_1$", "$\\alpha^*_2$", "$\\alpha^*_3$",
               "$N_G(0.7)$", "$N_G(0.8)$", "$K^*(0.9)$", "Incompatible set at 0.9"], rows,
              "Coverage on four plants. $J_K$ is chosen at seeds 0-2 and scored at the seeds given; the levels, "
              "generalists and covers are computed at those seeds. The work plant's grid has seeds 0-2 only.", num_from=1)
    fc = img(MEDIA / "20261104-curves.png", "Left: $J_K/J_1$ against $K$, chosen at seeds 0-2 and scored at seeds "
             "20-24. Middle: generalists against the desired level, each line ending at the last level with one. "
             "Right: the fewest layouts $K^*$. Solid lines at seeds 20-24; the work plant (dashed) at seeds 0-2.")
    gain = [float(r[4]) for r in rows]
    txt = [f"""<p>On every plant one layout per task adds {100 * min(gain) - 100:.0f} to {100 * max(gain) - 100:.0f} %
to the mean score, and no layout reaches 0.9 of each task's best (Table {n0 + 1}, Figure {f0 + 1}). The plants differ
in how many layouts come close.</p>"""]
    k21, mu4 = Js.get("k21"), Js.get("k21mu0p4")
    if k21 and "val" in k21:
        co = k21["val"]["coverage"]["val"]
        lev = {r["alpha"]: r for r in co["levels"]}
        sl = subset_levels(k21)
        pr = [e["val_only"] for e in sl if len(e["tasks"]) == 2]
        tr = [e["val_only"] for e in sl if len(e["tasks"]) == 3]
        txt.append(f"""<p>On the STS3032 at friction 1.0, {lev[0.7]['n_generalists']} layouts are generalists at
level 0.7 and {lev[0.8]['n_generalists']} at 0.8, and the best reaches {co['best_generalist_worst']:.2f}. Every pair of
tasks shares a layout at level 0.9, with $\\lambda$ of {min(pr):.2f} to {max(pr):.2f}. The sets of three reach only
{min(tr):.2f} to {max(tr):.2f}, so the four tasks need {WORDS[lev[0.9]['K_star']]} layouts at 0.9 without an
incompatible pair.</p>""")
    if mu4 and "val" in mu4:
        co = mu4["val"]["coverage"]["val"]
        lev = {r["alpha"]: r for r in co["levels"]}
        c = mu4["val"]["chosen"]["val"]
        zero = sum(1 for s_ in c["specialists"] for v in s_["score"] if v < 0.005)
        txt.append(f"""<p>At friction 0.4 the best generalist reaches {co['best_generalist_worst']:.2f}, and
{lev[0.7]['n_generalists']} layouts are generalists at 0.7. The specialists there are narrow. On {zero} of their 12
scores on the other three tasks the object is not held at any seed, against {sum(1 for s_ in Js['sv25']['val']['chosen']['val']['specialists']
for v in s_['score'] if v < 0.005)} on the SCS0009. One layout per task adds {100 * float(rows[2][4]) - 100:.0f} % to the
mean, the most of the four plants.</p>""")
    return "".join(txt) + t + fc


def sec_sizes(S):
    lev = {r["alpha"]: r for r in S["coverage"]["levels"]}
    a = S["coverage"]["alpha_star"]
    b = {t: min(w["alpha1"] for w in by["4"]) for t, by in S["breadth"].items()}
    gap = {al: lev[al]["K_star"] - len(lev[al]["mutually_incompatible"]) for al in sorted(lev)}
    agree = min(al for al in gap if all(gap[b] == 0 for b in gap if b >= al))
    f0 = P.FIG[0]
    fs = img(MEDIA / "20261104-sizes.png", "The size grid: 16 tasks on 8,182 Sobol layouts of the six gantry "
             "coordinates, work plant, seeds 0-2. Left: the fewest layouts $K^*$ and the largest set of mutually "
             "incompatible tasks against the desired level. Right: one layout's level $\\alpha^*_1$ over runs of "
             "consecutive object sizes of one task (the lowest over the runs of that length).")
    return f"""<p>A wider task set needs more layouts. The size grid of the layout trends page [5] scored
{S['n_layouts']:,} Sobol layouts of the six gantry coordinates on 16 tasks. The twist and the roll ran on spheres of 30,
40, 50 and 63 mm. The pin-push and the press ran on cylinders of 10, 16, 25 and 40 mm. The grid ran on the work plant at
seeds 0-2 with one grasp per layout, the foundational pose. Over the 16 tasks one layout reaches
$\\alpha^*_1$ = {S['coverage']['best_generalist_worst']:.2f}, three reach {a['3']:.2f} and seven reach {a['7']:.2f}
(Figure {f0 + 1}). At level 0.9
the fewest layouts are {lev[0.9]['K_star']}, and at 0.95 they are {lev[0.95]['K_star']}. Both equal the largest set
of mutually incompatible tasks. From level {agree:.2f} up the two agree at every level, and below it $K^*$ exceeds the
bound by at most {max(gap.values())} layouts. At 0.7, for example, $K^*$ is {lev[0.7]['K_star']} and the largest
incompatible set has {len(lev[0.7]['mutually_incompatible'])} tasks.</p>
<p>Within one task type, one layout over the four sizes reaches {b['twist']:.2f} of each size's best on the twist
and {b['roll']:.2f} on the roll. It reaches {b['pinpush']:.2f} on the pin-push and {b['press']:.2f} on the press.</p>""" + fs


def sec_films():
    if not FILMS.exists():
        return "<p>The films are made after the GPU queues (<code>hom_task_conflict_page.py films</code>).</p>"
    info = json.loads(FILMS.read_text())
    agree = sum(abs(s["cpu"] - s["gpu"]) <= 0.1 for s in info)
    m0 = P.FILM[0]
    has3 = any(s["pick"] == "S3" for s in info)
    out = [f"""<p>Films {m0 + 1} to {m0 + 4} run each task's specialist (left) beside the best single layout (right) on
the SCS0009 at seed 20, a validation seed. """ + ("The top row has the S1 grasp at HOM's defaults, and the bottom row "
                                                  "the S3 pick of Section SEC[refined]. " if has3 else "")
           + """They run on the CPU plant, where every contact meets the object. On the GPU only the pads and the rigid
spheres under 2 mm do. On """ + (f"all {len(info)}" if agree == len(info) else f"{agree} of the {len(info)}")
           + """ clips the CPU score at seed 20 lies within 0.1 of the GPU's validated score at seeds 20-24.</p>"""]
    for task in TASKS:
        fs = [s for s in info if s["task"] == task]
        if not fs:
            continue
        sp = next(s for s in fs if s["role"] == "specialist")
        g = next(s for s in fs if s["role"] == "J_1")
        cap = (f"{NAME[task]} on the {OBJ[task]} on the SCS0009, CPU plant, seed 20: the {NAME[task].lower()}'s "
               f"specialist {xy(sp['xy'])} left and the best single layout {xy(g['xy'])} right"
               + (", S1 above S3" if has3 else ", S1 grasps") + ". Under each clip: its CPU score and the GPU's "
               "validated score.")
        out.append(film(MEDIA / f"20261104-film_{task}.mp4", cap))
    return "".join(out)


def timing_lines(root):
    out = {}
    for p in sorted(root.rglob("*_timing.jsonl")):
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        out[p] = [r for r in rows if "worlds" in r and "run_s" in r]
    return out


def step_walls():
    """{queue step: (wall s, peak GB)} from /usr/bin/time's line at the end of each step's log."""
    out = {}
    for q in sorted((ROOT / "logs/20261103-hom_task_conflict").glob("Q_*.log")):
        m = re.search(r"peak (\d+) kB, ([\d.]+) s", q.read_text(errors="replace"))
        if m:
            out[q.stem[2:]] = (float(m.group(2)), int(m.group(1)) / 1e6)
    return out


def sec_compute():
    parts = []
    walls = step_walls()
    for pl in ("sv25", "k21", "k21mu0p4"):
        ls = [r for p, rows in timing_lines(C.ROWS / pl).items() for r in rows]
        if ls:
            parts.append((f"Validation, {PLANT_NAME[pl]}", ls, [walls.get(f"val_{pl}", (0, 0))[0]]))
    for lv, name in (("grasp", "S2 searches"), ("full", "S3 searches"), ("val", "S1-S3 picks at seeds 20-24")):
        ls = [r for p, rows in timing_lines(C.REFINE / "sv25" / lv).items() for r in rows]
        if ls:
            parts.append((f"Shared layouts, {name}", ls, [w for k, (w, _m) in walls.items()
                                                          if k.startswith(f"refine_{lv}_")]))
    if not parts:
        return "<p>No GPU runs yet.</p>"
    rows, tot = [], {"n": 0, "b": 0, "bs": 0.0, "w": 0.0}
    for name, ls, wl in parts:
        n = sum(r["worlds"] for r in ls)
        bs = sum(r["build_s"] + r["run_s"] + r["score_s"] for r in ls)
        w = sum(wl)
        rows.append([name, f"{n:,}", str(len(ls)), f"{bs / 3600:.2f}", f"{w / 3600:.2f}", f"{1000 * bs / n:.0f}",
                     f"{1000 * w / n:.0f}" if w else "n/a"])
        tot = {"n": tot["n"] + n, "b": tot["b"] + len(ls), "bs": tot["bs"] + bs, "w": tot["w"] + w}
    rows.append(["All", f"{tot['n']:,}", str(tot["b"]), f"{tot['bs'] / 3600:.2f}", f"{tot['w'] / 3600:.2f}",
                 f"{1000 * tot['bs'] / tot['n']:.0f}", f"{1000 * tot['w'] / tot['n']:.0f}"])
    n0 = P.TAB[0]
    t = table(["Runs", "Rollouts", "GPU batches", "Batch time (h)", "Queue steps (h)", "Batch time per rollout (ms)",
               "Step time per rollout (ms)"], rows,
              "The study's GPU work, from the timing line of every batch and the wall clock of every queue step "
              "(<code>logs/20261103-hom_task_conflict/</code>).")
    Tp = sorted((ROOT / "docs/experiments/20261103-hom_score_refine/compute").glob("*-throughput.json"))
    cpu = ""
    if Tp:
        T_ = json.loads(Tp[-1].read_text())["plants"]["sv25"]["cpu"]
        per = {t_: T_[t_]["filter_p1"]["wall_s"] for t_ in TASKS if T_.get(t_, {}).get("filter_p1")}
        if per:
            cpu = (f" On one CPU core the CPU runner takes {min(per.values()):.1f} to {max(per.values()):.1f} s per "
                   f"rollout on these tasks [4], so the same rollouts would take about "
                   f"{tot['n'] * np.mean(list(per.values())) / 3600:.0f} core-hours.")
    return f"""<p>The study ran {tot['n']:,} HOM rollouts on the GPU in {tot['b']} batches, with
{tot['bs'] / 3600:.2f} h of batch time in {tot['w'] / 3600:.2f} h of queue steps (Table {n0 + 1}).{cpu} Every batch
wrote its timing line beside its rows, and <code>scripts/hom_compute.py ledger</code> includes them from 2026-11-04.
</p>""" + t


def sec_limits():
    return """<ul>
<li>The layouts are the symmetric tripods of the 5 mm lattice, at one object size per task. The six-coordinate layouts
and the other sizes enter only through the size grid, which ran on the work plant at seeds 0-2 with one grasp per
layout.</li>
<li>The desired level is a fraction of each task's best on this lattice. Thresholds from the tasks' own needs, such as
a turn angle the bench requires, were not set.</li>
<li>$J_K$ weights the four tasks equally.</li>
<li>Choices use three seeds and validation five. The GPU's run-to-run spread is about 0.01 per layout and task on
average; one roll layout differed by 0.35 between two batches of the same rollouts.</li>
<li>The twist's score saturates at 1.0, a 90 deg turn on three pads, and several layouts tie there.</li>
<li>The smoothed score averages the neighbours' own grasps. A gantry's placement error was not simulated with the
commanded layout's grasp.</li>
<li>S2 and S3 ran on ten shared layouts on the SCS0009 only.</li>
<li>All results are simulation; no layout was run on the hand.</li>
</ul>"""


def sec_files():
    return """<p>Files: the study <code>scripts/hom_task_conflict.py</code> (commands <code>scope</code>,
<code>validate</code>, <code>refine</code> and <code>refined</code>); this
page's builder <code>scripts/hom_task_conflict_page.py</code>; the JSON files, queues, rows and runbook in
<code>docs/experiments/20261103-hom_task_conflict/</code> (README.md); the logs in
<code>logs/20261103-hom_task_conflict/</code>; the refinement <code>scripts/hom_grasp_refine.py</code>.</p>"""


REFS = [
    f'[1] <a href="{DESIGN}">HOM design landscapes of the SR2 hand</a> (2026-10-29).',
    '[2] The HOM paper, <a href="https://arxiv.org/abs/2609.25619">arXiv 2609.25619v2</a>, 2026.',
    f'[3] <a href="{PLANT_PAGE}">SR2 hand layouts under HOM control across finger servos, pad moduli and scene '
    'friction</a> (2026-11-02).',
    f'[4] <a href="{REFINE_PAGE}">Grasp, force and contact-mode refinement of SR2 hand layouts under HOM control</a> '
    '(2026-11-03).',
    f'[5] <a href="{TRENDS}">Layout trends with object size</a> (2026-10-30).',
    '[6] J. Xu, W. Wan, K. Koyama, Y. Domae and K. Harada, "Selecting and designing grippers for an assembly task in a '
    'structured approach", Advanced Robotics 35(6), 381-397, 2021, <a href="https://arxiv.org/abs/2003.04087">arXiv '
    '2003.04087</a>.',
    '[7] M. Honarpardaz, M. Tarkian, J. Olvander and X. Feng, "Finger design automation for industrial robot grippers: '
    'a review", Robotics and Autonomous Systems 87, 104-119, 2017, DOI 10.1016/j.robot.2016.10.003.',
    '[8] M. Honarpardaz, J. Olvander and M. Tarkian, "Fast finger design automation for industrial robots", Robotics '
    'and Autonomous Systems 113, 120-131, 2019, DOI 10.1016/j.robot.2018.12.011.',
    '[9] S. Wang, J. Oh and N. S. Pollard, "The foundational pose as a selection mechanism for the design of '
    'tool-wielding multi-finger robotic hands", <a href="https://arxiv.org/abs/2409.14158">arXiv 2409.14158</a>.',
    '[10] B. Yang, L. Jiang, W. Wu, M. Cheng and K. Gong, "Evolving robot morphology with manipulation learning inspired '
    'by human hand evolution", IEEE Transactions on Evolutionary Computation, 2026, DOI 10.1109/TEVC.2026.3696420.',
    '[11] J. Xu, A. Spielberg, A. Zhao, D. Rus and W. Matusik, "Multi-objective graph heuristic search for terrestrial '
    'robot design", ICRA 2021, 9863-9869, <a href="https://arxiv.org/abs/2107.05858">arXiv 2107.05858</a>.',
    '[12] J. Whitman, M. Travers and H. Choset, "Modular mobile robot design selection with deep reinforcement '
    'learning", NeurIPS 2020 Workshop on Machine Learning for Engineering Modeling, Simulation and Design.',
    '[13] J. Hu, J. Whitman, M. Travers and H. Choset, "Modular robot design optimization with generative adversarial '
    'networks", ICRA 2022, 4282-4288, DOI 10.1109/ICRA46639.2022.9812091.',
    '[14] E. Sihite, A. Kalantari, R. Nemovi, A. Ramezani and M. Gharib, "Multi-Modal Mobility Morphobot (M4) with '
    'appendage repurposing for locomotion plasticity enhancement", Nature Communications 14, 3323, 2023, '
    'DOI 10.1038/s41467-023-39018-y.',
    '[15] J.-B. Mouret and G. Maguire, "Quality diversity for multi-task optimization", GECCO 2020, 121-129, '
    '<a href="https://arxiv.org/abs/2003.04407">arXiv 2003.04407</a>.',
    '[16] D. J. Hejna, P. Abbeel and L. Pinto, "Task-agnostic morphology evolution", ICLR 2021, '
    '<a href="https://arxiv.org/abs/2102.13100">arXiv 2102.13100</a>.',
    '[17] A. Gupta, S. Savarese, S. Ganguli and L. Fei-Fei, "Embodied intelligence via learning and evolution", Nature '
    'Communications 12, 5721, 2021, DOI 10.1038/s41467-021-25874-z.',
    '[18] C. Rueffler, T. J. M. Van Dooren and J. A. J. Metz, "The evolution of resource specialization through '
    'frequency-dependent and frequency-independent mechanisms", American Naturalist 167(1), 81-93, 2006, '
    'DOI 10.1086/498275.',
    '[19] R. Levins, "Theory of fitness in a heterogeneous environment. I. The fitness set and adaptive function", '
    'American Naturalist 96, 361-373, 1962, DOI 10.1086/282245 (cited through [18]).',
    '[20] O. Shoval, H. Sheftel, G. Shinar, Y. Hart, O. Ramote, A. Mayo, E. Dekel, K. Kavanagh and U. Alon, '
    '"Evolutionary trade-offs, Pareto optimality, and the geometry of phenotype space", Science 336, 1157-1160, 2012, '
    'DOI 10.1126/science.1217405.',
    '[21] T. Standley, A. R. Zamir, D. Chen, L. Guibas, J. Malik and S. Savarese, "Which tasks should be learned '
    'together in multi-task learning?", ICML 2020, <a href="https://arxiv.org/abs/1905.07553">arXiv 1905.07553</a>.',
    '[22] E. Kussell and S. Leibler, "Phenotypic diversity, population growth, and information in fluctuating '
    'environments", Science 309, 2075-2078, 2005, DOI 10.1126/science.1114383.',
    '[23] M. C. Donaldson-Matasci, C. T. Bergstrom and M. Lachmann, "The fitness value of information", Oikos 119(2), '
    '219-230, 2010, DOI 10.1111/j.1600-0706.2009.17781.x.',
    '[24] A. N. Kolmogorov and V. M. Tikhomirov, "epsilon-entropy and epsilon-capacity of sets in function spaces", '
    'Uspekhi Mat. Nauk 14(2), 3-86, 1959; English translation, AMS Translations (2) 17, 277-364, 1961.',
]


def render(page):
    blocks = re.findall(r"\\begin\{algorithmic\}.*?\\end\{algorithmic\}", page, re.S)
    page = re.sub(r"\\begin\{algorithmic\}.*?\\end\{algorithmic\}", "\x00ALG\x00", page, flags=re.S)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = texsvg.render(items + [(b, "block") for b in blocks], cache_path=TEX_CACHE)
    it = iter(svgs[:len(items)])
    page = P.TEX_RE.sub(lambda m: next(it), page)
    ib = iter(svgs[len(items):])
    return re.sub("\x00ALG\x00", lambda m: next(ib), page), len(items), len(blocks)


def refined_rows(Rf):
    rows = []
    for lv in C.LEVELS:
        L_ = Rf["levels"][lv]
        js = [r["score"] for r in L_["J"]]
        cv = L_["coverage_val"]
        lev = {r["alpha"]: r for r in cv["levels"]}
        th = L_["at_s1_thresholds"]["n_generalists"]
        rows.append([lv, f2(js[0]), f2(js[-1]), f2(js[-1] / js[0]), f2(cv["best_generalist_worst"]),
                     str(lev[0.8]["n_generalists"]), str(lev[0.9]["K_star"]), th["0.7"], th["0.8"], th["0.9"]])
    return rows


def sec_refined(Rf, Js):
    if not Rf:
        return "<p>The refinement of the shared layouts has not finished.</p>"
    n0, f0 = P.TAB[0], P.FIG[0]
    lays = Rf["layouts"]
    calls = [L_["call"] for L_ in lays]
    Lv = Rf["levels"]
    V = {lv: np.array(Lv[lv]["val"]) for lv in C.LEVELS}
    J = {lv: [r["score"] for r in Lv[lv]["J"]] for lv in C.LEVELS}
    g = Lv["S3"]["J"][0]["layouts"][0]
    gi = calls.index(g["call"])
    spec = {lv: {s_["task"]: calls.index(s_["call"]) for s_ in Lv[lv]["specialists"]} for lv in C.LEVELS}
    cov = {lv: Lv[lv]["coverage_val"] for lv in C.LEVELS}
    lev = {lv: {r["alpha"]: r for r in cov[lv]["levels"]} for lv in C.LEVELS}
    th = {lv: Lv[lv]["at_s1_thresholds"]["n_generalists"] for lv in C.LEVELS}
    t = table(["Level", "$J_1$", "$J_4$", "$J_4/J_1$", "$\\alpha^*_1$", "$N_G(0.8)$", "$K^*(0.9)$",
               "Generalists at S1's thresholds, 0.7", "0.8", "0.9"], refined_rows(Rf),
              f"The {len(lays)} shared layouts on the SCS0009 at S1, S2 and S3. $J_K$ is chosen at seeds 0-2 and "
              "scored at seeds 20-24. $\\alpha^*_1$, $N_G$ and $K^*$ use each level's own best among the shared "
              "layouts at seeds 20-24. The last three columns count the layouts that reach $\\alpha$ times the S1 "
              "specialist's validated score on every task.", num_from=1)
    fr = img(MEDIA / "20261104-refined_sv25.png", "Validated scores (seeds 20-24) of the shared layouts at S1, S2 and "
             "S3 on the SCS0009, per task. An empty slot is a score of 0: the object is not held at any seed, or the "
             "layout does not reach the task.")
    per = []
    for t_i, task in enumerate(TASKS):
        a1, a3 = V["S1"][gi, t_i], V["S3"][gi, t_i]
        b1, b3 = V["S1"][spec["S1"][task], t_i], V["S3"][spec["S3"][task], t_i]
        per.append((task, a1, a3, b1, b3))
    behind = min(per, key=lambda r: r[2] / r[4])
    cover = [c_ for c_ in lev["S3"][0.9]["cover"]]
    ti = TASKS.index("twist")
    up = sorted(range(len(lays)), key=lambda k: V["S1"][k, ti] - V["S3"][k, ti])[:2]
    drops = [(lays[k]["x"], TASKS[t_i], a, b, la, lb) for k in range(len(lays)) for t_i in range(len(TASKS))
             for a, b, la, lb in ((V["S1"][k, t_i], V["S2"][k, t_i], "S1", "S2"), (V["S2"][k, t_i], V["S3"][k, t_i], "S2", "S3"))
             if b < a - 0.05]
    def by(role):
        return [L_ for L_ in lays if role in L_["roles"]]

    rb = {t_: by(f"refined best of {t_}") for t_ in TASKS}
    rb_new = [t_ for t_ in TASKS if rb[t_] and not any(r.startswith("specialist") for r in rb[t_][0]["roles"])]
    rb_old = [t_ for t_ in TASKS if rb[t_] and t_ not in rb_new]
    third = [L_ for L_ in by("cover of alpha*_3") if len(L_["roles"]) == 1]
    names = (f"""They are the four specialists and the best single layout {xy(by('J_1')[0]['x'])}. Two more are"""
             + (f" the third layout {xy(third[0]['x'])} of the cover at $\\alpha^*_3$ and" if third else "")
             + f""" the best single layout on smoothed scores, {xy(by('J_1 smoothed')[0]['x'])}. The other
{WORDS[len(rb_new)]} are the best refined layouts of the {andj([NAME[t_].lower() for t_ in rb_new])} in [4]:
{andj([xy(rb[t_][0]['x']) for t_ in rb_new])}"""
             + (f"; the {andj([NAME[t_].lower() for t_ in rb_old])}'s is its specialist" if rb_old else "") + ".")
    rep = refine_report()
    n_search = 2 * sum(len(v["layouts"]) for v in rep["tasks"].values()) if rep else 0
    p1 = f"""<p>Ten layouts were refined on every task they reach. {names} Each ran the two searches of the
refinement page [4]. S2 searched the grasp with 24 candidates for 8 iterations. S3 then searched the grasp, the
commanded forces and the contact modes together, with 32 candidates for 8 iterations. Each search scored its start as its first candidate, and
the S1, S2 and S3 picks ran again at seeds 20-24.</p>"""
    p2 = f"""<p>Refining each task's grasp, forces and contact modes on one shared layout raises its mean but leaves the
gap to the refined specialists (Table {n0 + 1}, Figure {f0 + 1}). The best single layout stays {xy(g['x'])}. Its mean
validated score rises from {J['S1'][0]:.3f} at S1 to {J['S2'][0]:.3f} at S2 and {J['S3'][0]:.3f} at S3. The
specialists rise as well, from {J['S1'][-1]:.3f} to {J['S2'][-1]:.3f} and {J['S3'][-1]:.3f}, so $J_4/J_1$ goes from
{J['S1'][-1] / J['S1'][0]:.2f} to {J['S3'][-1] / J['S3'][0]:.2f}. On the twist and the press the shared layout
reaches its specialists: {per[0][2]:.2f} and {per[3][2]:.2f} at S3, against {per[0][4]:.2f} and {per[3][4]:.2f}. On the
{NAME[behind[0]].lower()} it falls further behind, from {behind[1]:.2f} against {behind[3]:.2f} at S1 to
{behind[2]:.2f} against {behind[4]:.2f} at S3.</p>"""
    p3 = f"""<p>Measured against each level's own best, the best generalist's worst task reaches
{cov['S1']['best_generalist_worst']:.2f} at S1 and {cov['S3']['best_generalist_worst']:.2f} at S3. At level 0.9 S3
needs {WORDS[len(cover)]} layouts and S1 {WORDS[lev['S1'][0.9]['K_star']]}. At S3 these are
{andj([f"{xy(c_['x'])} for the {andj([NAME[t_].lower() for t_ in c_['tasks']])}" for c_ in cover])}. At the
thresholds of S1's specialists the refinement adds generalists: {th['S1']['0.7']} at level 0.7 becomes
{th['S3']['0.7']}, {th['S1']['0.8']} at 0.8 becomes {th['S3']['0.8']}, and none reaches 0.9.</p>"""
    p4 = f"""<p>The twist changes most. The layout {xy(lays[up[0]]['x'])} scores {V['S1'][up[0], ti]:.2f} on it at S1 and
{V['S3'][up[0], ti]:.2f} at S3, and {xy(lays[up[1]]['x'])} goes from {V['S1'][up[1], ti]:.2f} to
{V['S3'][up[1], ti]:.2f}. """ + (f"""{WORDS[len(drops)].capitalize()} of the {n_search} searches picked a candidate that
scores more than 0.05 below its start at seeds 20-24. """
        + " ".join(f"On the {NAME[tk].lower()}, the {lb} pick of {xy(x_)} scores {b:.2f} against {a:.2f} at {la}."
                   for x_, tk, a, b, la, lb in drops) + "</p>" if drops else "</p>")
    return p1 + t + fr + p2 + p3 + p4


def sec_related(Js):
    seg = max(p["segment_margin_max"]["score"] for p in Js["sv25"]["val"]["chosen"]["val"]["pairs"])
    return f"""<p>The fewest layouts that cover a task set is a set cover. Xu et al. [6] chose the fewest sampled gripper
designs that grasp every part of an assembly. Their 0-1 program has one covering constraint per part and was solved by
branch and bound. $K^*(\\alpha)$ is that program with a graded score and a desired level in place of a grasp test.
The blend appears in industry as multi-function fingers, which grasp several parts and save tool changes [7]. In the
load tests of [8], fingers designed for one part resisted disturbances better than fingers designed for two, which
give each part less contact area.</p>
<p>Studies of design for several tasks find trade-offs between tasks. Wang et al. [9] sampled 100,480 hand designs and
computed one Pareto front over three tool poses and three metrics. In a t-SNE map their designs fall into isolated
islands. Hands evolved from one population by Yang et al. [10] diverge when the task set changes. Xu et al. [11]
searched robot designs for Pareto sets over pairs of locomotion and design objectives and one triple, and wrote that
"rarely will a single design be best at all tasks". Other work chooses a design per task. A network ranks modular
designs for a terrain map [12], and a GAN maps a terrain to several equally good designs [13]. The M4 robot [14]
switches between eight modes by using its appendages as wheels, thrusters and legs. Multi-task MAP-Elites [15] keeps
the best solution for each of thousands of tasks. TAME [16] evolves morphologies without a task or reward and reaches
the multi-task performance of task-supervised evolution. In DERL [17], morphologies evolved in more complex
environments learn new tasks faster.</p>
<p>In ecology the question is old. Rueffler et al. [18] summarize Levins's fitness sets [19]: when the set of
attainable pairs of performance is convex, a generalist is favoured, and when it is concave, a specialist. With
frequency-dependent selection a population can also split into two forms at the generalist [18]. Shoval et al. [20]
showed that Pareto-optimal phenotypes lie on the segment, or in the polygon, spanned by the single-task optima. Their
result assumes that each task has one optimum and that performance falls with distance from it. The HOM landscapes
have many peaks at the 5 mm scale. On the SCS0009 a layout of the straight segment between two specialists is
beaten on both tasks by up to {seg:.2f} (Section SEC[sv25]). Standley et al. [21] chose networks that together solve all tasks under a budget, a
problem they showed NP-hard by reduction from set cover; $K^*$ has the same form with layouts for networks. When the
next task is unknown, random switching can be favoured over sensing if the environment changes rarely [22]. A cue is
worth at most its mutual information with the environment [23]. Finally, $\\log_2 K^*(\\alpha)$ has the form of the
$\\varepsilon$-entropy of Kolmogorov and Tikhomirov [24], the logarithm of a covering number. Their bound of the
covering number by the largest set of points more than $2\\varepsilon$ apart has the form of the bound of $K^*$ by
mutually incompatible tasks. We found no study that counts generalists and the fewest layouts against a desired level for a
robot hand.</p>"""


def sec_future(Js, S, Rf):
    lev = {r["alpha"]: r for r in S["coverage"]["levels"]}
    return f"""<ul>
<li>Thresholds from the tasks' own needs. Set each task's threshold $\\tau_t$ from what the bench requires, such as
the three-pad angle that starts a screw, and recompute $G$ and $K^*$ (a <code>--tau</code> option for
<code>hom_task_conflict.py scope</code>). If a generalist exists at those thresholds, the four tasks need no layout
change on the hand.</li>
<li>Placement error by rollouts. Run the specialists' and the best single layout's own grasps at layouts displaced by
2.5 and 5 mm in $x_{{sep}}$ and $y_{{sep}}$ (a displacement option for <code>hom_task_conflict.py validate</code>). The
smoothed scores stand for this error only if the displaced scores fall as far as the 3 x 3 block means.</li>
<li>The size grid on the SCS0009 with the best of 16 grasps, at seeds 20-24. On the work plant $K^*$ exceeds the
largest mutually incompatible set by at most two layouts, and at level 0.9 both are {lev[0.9]['K_star']}. If the gap is
larger on the hand's plant, the incompatible set is no rule of thumb for the number of layouts.</li>
<li>Per-task refinement over all generalists. Run S3 (<code>hom_task_conflict.py refine</code> with a larger
<code>N_SHARED</code>) on the generalists at level 0.6 of Section SEC[sv25]. If one of them reaches 0.9 of the refined
specialists on all four tasks, task-specific control removes the switch at that level.</li>
<li>Generalists and specialists at scale, the owner's long-term question. Count $N_G(\\alpha)$ and $K^*(\\alpha)$
on the Sobol layouts against the number of tasks and the span of their sizes. If $\\log_2 K^*(\\alpha)$ grows
linearly with the span, each added size range costs a fixed factor in layouts.</li>
</ul>"""


def lede_and_summary(Js, S, Rf):
    J = Js["sv25"]
    c, co = J["val"]["chosen"]["val"], J["val"]["coverage"]["val"]
    js = [r["score"] for r in c["J"]]
    g = c["J"][0]["layouts"][0]
    lev = {r["alpha"]: r for r in co["levels"]}
    seg = max(p["segment_margin_max"]["score"] for p in c["pairs"])
    a1 = co["best_generalist_worst"]
    sl = {r["alpha"]: r for r in S["coverage"]["levels"]}
    lede = [f"On the SCS0009 plant no SR2 finger layout reaches 0.9 of each HOM task's best score on all four tasks, "
            f"and {WORDS[lev[0.9]['K_star']]} layouts are needed at that level. The best single layout, {xy(g['x'])} mm, "
            f"has a mean validated score of {js[0]:.2f}, against {js[-1]:.2f} with one layout per task. Its worst task "
            f"reaches {a1:.2f} of that task's best. One layout is therefore enough for the four tasks up to a desired "
            f"level of {a1:.2f}, and above that level the fewest layouts are found by an exact set cover. Straight "
            f"lines between two tasks' best layouts pass through layouts that another layout beats on both tasks by up "
            f"to {seg:.2f}."]
    summary = (f"On the SCS0009 one SR2 layout reaches at most {a1:.2f} of each HOM task's best (mean {js[0]:.2f} "
               f"against {js[-1]:.2f} with one layout per task, seeds 20-24), and {WORDS[lev[0.9]['K_star']]} layouts "
               f"are needed at 0.9.")
    if Rf:
        L3 = Rf["levels"]["S3"]
        j3 = [r["score"] for r in L3["J"]]
        k3 = {r["alpha"]: r for r in L3["coverage_val"]["levels"]}[0.9]["K_star"]
        lede.append(f"Refining each task's grasp, forces and contact modes on ten shared layouts raises the best single "
                    f"layout's mean to {j3[0]:.2f} and the specialists' to {j3[-1]:.2f}. {WORDS[k3].capitalize()} "
                    f"layouts then suffice at 0.9.")
        summary += (f" Per-task grasp, force and contact-mode refinement raises both ({j3[0]:.2f} against "
                    f"{j3[-1]:.2f}) and needs {WORDS[k3]} layouts at 0.9.")
    lede.append(f"Over 16 tasks of four object sizes each, one layout reaches {S['coverage']['best_generalist_worst']:.2f} "
                f"of each task's best, and {WORDS[sl[0.9]['K_star']]} layouts are needed at 0.9.")
    return " ".join(lede), summary


def cmd_page():
    Js = {pl: J for pl in PLANTS if (J := conflict(pl))}
    if "val" not in Js.get("sv25", {}):
        raise SystemExit("run scripts/hom_task_conflict.py scope --plant sv25 --val first")
    S = json.loads(SIZES.read_text())
    Rf = refined()
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("method", "Score matrices, coverage and the decision at a layout", sec_method(Js)),
        ("sv25", "One layout or several on the SCS0009", sec_sv25(Js)),
        ("plants", "Other servos and scene friction", sec_plants(Js)),
        ("refined", "Task-specific grasps, forces and contact modes on shared layouts", sec_refined(Rf, Js)),
        ("sizes", "Object sizes in the six gantry coordinates", sec_sizes(S)),
        ("related", "Related work", sec_related(Js)),
        ("films", "Films of the specialists and the best single layout", sec_films()),
        ("compute", "Compute", sec_compute()),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future(Js, S, Rf)),
        ("refs", "References and files", '<ol class="refs">' + "".join(f"<li>{r[r.index(']') + 2:]}</li>" for r in REFS)
         + "</ol>" + sec_files()),
    ]
    num = {sid: k for k, (sid, _hd, _b) in enumerate(sections, 1)}
    sub = (lambda s: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), s))
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|MPa|N\b|ms\b|s\b|h\b)", "\\1&nbsp;\\2", s_))
    sections = [(sid, hd, unit(sub(body))) for sid, hd, body in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>'
                     for k, (sid, hd, body) in enumerate(sections, 1))
    lede, summary = lede_and_summary(Js, S, Rf)
    byline = ("Simulation, 2026-11-03 and 04. Hand: the CAD-model SR2 hand with the v4b fingertip "
              "(<code>SCALE_HAND=cad_v2</code>); HOM on MuJoCo-Warp, the films on the CPU plant. Follows the plant "
              "page [3] and the refinement page [4]. Builder: <code>scripts/hom_task_conflict_page.py</code>.")
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "HOM task conflict", "H1": "One finger layout or several for the HOM tasks of the SR2 hand",
                 "EYEBROW": "HOM layout study, 2026-11-04", "LEDE": unit(sub(lede)), "BYLINE": byline, "TOC": toc,
                 "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    page, n, nb = render(page)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-04")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {n} formulas, {nb} algorithms", flush=True)
    return 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "figs":
        return cmd_figs()
    if a and a[0] == "films":
        return cmd_films()
    return cmd_page()


if __name__ == "__main__":
    sys.exit(main())

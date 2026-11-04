#!/usr/bin/env python3
"""Builds docs/experiments/20261103-hom_score_refine/20261103-hom_score_refine.html: each SR2 layout scored three ways
under HOM control (S1 the best of its sampled grasps at HOM's defaults; S2 the grasp refined on the HOM score; S3 the
grasp, the commanded forces and the contact modes refined together), on the SCS0009 plant and on the STS3032 at scene
friction 1.0 and 0.4 (owner, 2026-11-03: "can't we score and refine a grasp based off how well the grasp
(+morphology) executes the HOM control?"; "is it possible to co-design HOM?"; "let's send it on these tasks ... make
sure things are documented well!").

Data: the reports rows/<plant>/<date>-score_levels.json of scripts/hom_grasp_refine.py, its _gens.jsonl files (the
CEM's progress) and the films' rows media/<date>-films.json. The design page's helpers and template
(scripts/fp_review_page.py) draw the tables and figures.

    .venv/bin/python scripts/hom_score_refine_page.py figs     # the figures from the reports
    .venv/bin/python scripts/hom_score_refine_page.py films    # CPU clips through resguard (after the GPU queue)
    .venv/bin/python scripts/hom_score_refine_page.py          # the page
    .venv/bin/python scripts/hom_score_refine_page.py --smoke  # the page from rows_smoke/ (a layout check, not results)
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
import hom_grasp_refine as R  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

fill, table, img, film, esc = P.fill, P.table, P.img, P.film, P.esc
D = R.OUT
MEDIA = D / "media"
OUT = D / "20261103-hom_score_refine.html"
TEX_CACHE = D / "texsvg_cache.json"
FILMS = MEDIA / "20261103-films.json"
PLANT_PAGE = "../20261029-foundational_pose_review/20261102-hom_plant_modulus.html"
DESIGN = "../20261029-foundational_pose_review/20261029-foundational_pose_hand_design.html"
TRENDS = "../20261029-foundational_pose_review/20261030-hom_layout_trends.html"
TASKS = tuple(R.TASKS)
NAME = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
OBJ = {"twist": "40 mm sphere", "roll": "40 mm sphere", "pinpush": "25 mm cylinder", "press": "25 mm cylinder"}
PLANT_NAME = {"sv25": "SCS0009, 2.5 MPa", "k21": "STS3032 at kp 2.1, friction 1.0",
              "k21mu0p4": "STS3032 at kp 2.1, friction 0.4", "work10": "work plant, 10 MPa"}
PRM_LABEL = {"f_tripod": "tripod force (N)", "f_pinch": "pinch force (N)", "f_drive": "drive force (N)",
             "f_press": "press force (N)"}
C1, C2, C3 = "#7f7f7f", "#1f4e9c", "#b03a2e"
SMOKE = "--smoke" in sys.argv


def report(plant):
    """The newest report of a plant, or None."""
    base = (R.OUT / ("rows_smoke" if SMOKE else "rows")) / plant
    ps = sorted(base.glob("*-score_levels.json"))
    return json.loads(ps[-1].read_text()) if ps else None


def gens(plant, level, task):
    base = (R.OUT / ("rows_smoke" if SMOKE else "rows")) / plant / level
    return R.read_rows(base / f"{task}_d{R.TASKS[task]:g}_gens.jsonl")


def vals(lays, key, group=None):
    return [e[key] for e in lays if e.get(key) is not None and (group is None or e["group"] == group)]


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def f2(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"


def sg(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:+.2f}"


def paired(lays, a, b, group=None):
    """Differences b - a over the layouts that have both, and the shares above +0.05 and below -0.05."""
    d = np.array([e[b] - e[a] for e in lays if e.get(a) is not None and e.get(b) is not None
                  and (group is None or e["group"] == group)], float)
    if not len(d):
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "up": 0, "down": 0}
    return {"n": len(d), "median": float(np.median(d)), "mean": float(d.mean()), "up": int((d > 0.05).sum()),
            "down": int((d < -0.05).sum())}


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


def fig_levels(plant, J):
    """Per task: each layout's S1, S2 and S3 at the validation seeds, the layouts in order of S1."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), constrained_layout=True, sharey=True)
    for ax, task in zip(axs, TASKS):
        lays = [e for e in J["tasks"].get(task, {}).get("layouts", []) if e.get("s1_val") is not None]
        lays.sort(key=lambda e: (e["s1_val"], e["s1"]))
        x = np.arange(len(lays))
        for key, c, m, lab in (("s1_val", C1, "o", "S1"), ("s2_val", C2, "^", "S2"), ("s3_val", C3, "s", "S3")):
            xs = [i for i, e in zip(x, lays) if e.get(key) is not None]
            ys = [e[key] for e in lays if e.get(key) is not None]
            ax.plot(xs, ys, m, color=c, ms=5, mfc="none" if key == "s1_val" else c, label=lab, alpha=0.9)
        ax.set_title(f"{NAME[task]}, {OBJ[task]}")
        ax.set_xlabel("Layouts in order of S1")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Score at seeds 20-24")
    axs[0].legend(loc="upper left", frameon=False)
    save(fig, f"20261103-levels_{plant}.png")


def fig_cem(plant):
    """Per task: the median over layouts of the best candidate so far (seeds 0-2) by iteration, for both levels; S1's
    median at iteration -1."""
    plt = plt_setup()
    J = report(plant)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.7), constrained_layout=True, sharey=True)
    for ax, task in zip(axs, TASKS):
        if J and task in J["tasks"]:
            s1 = med(vals(J["tasks"][task]["layouts"], "s1"))
            ax.axhline(s1, color=C1, ls=":", label="S1")
        for level, c, lab in (("grasp", C2, "S2 search"), ("full", C3, "S3 search")):
            G = gens(plant, level, task)
            if not G:
                continue
            its = sorted({g["it"] for g in G})
            m = [med([g["best"] for g in G if g["it"] == k]) for k in its]
            q1 = [np.percentile([g["best"] for g in G if g["it"] == k], 25) for k in its]
            q3 = [np.percentile([g["best"] for g in G if g["it"] == k], 75) for k in its]
            ax.plot(its, m, "-o", color=c, ms=4, label=lab)
            ax.fill_between(its, q1, q3, color=c, alpha=0.15)
        ax.set_title(f"{NAME[task]}")
        ax.set_xlabel("CEM iteration")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Best so far, seeds 0-2")
    axs[0].legend(loc="lower right", frameon=False)
    save(fig, f"20261103-cem_{plant}.png")


def fig_prm(plant, J):
    """S3's forces (relative to HOM's defaults) and mode weights over the layouts, per task."""
    import hom_tasks as T
    plt = plt_setup()
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, task in zip(axs, TASKS):
        lays = [e for e in J["tasks"].get(task, {}).get("layouts", []) if e.get("prm3")]
        keys = list(R.FORCES[task]) + [f"w_{c}_{r}" for r in R.ROLES[task] for c in R.COMPS]
        for i, k in enumerate(keys):
            v = np.array([e["prm3"][k] for e in lays if k in e["prm3"]], float)
            if k.startswith("f_"):
                v = v / T.PRM[k]
            ax.plot(i + 0.12 * rng.standard_normal(len(v)), v, "o", color=C3, ms=4, alpha=0.6)
            ax.plot([i - 0.3, i + 0.3], [1.0 if k.startswith("f_") else T.PRM[k]] * 2, color="k", lw=1.5)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.replace("w_", "").replace("_", " ") if k.startswith("w_") else k[2:] + " / default"
                            for k in keys], rotation=60, ha="right")
        ax.set_title(f"{NAME[task]}")
        ax.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel("Force / default, or weight")
    save(fig, f"20261103-prm_{plant}.png")


def cmd_figs():
    for plant in ("sv25", "k21", "k21mu0p4"):
        J = report(plant)
        if not J:
            continue
        fig_levels(plant, J)
        fig_cem(plant)
        fig_prm(plant, J)
    return 0


# ----------------------------------------------------------------------------------------------- films

def film_specs():
    """Per task on the SCS0009: the layout with the highest S3 at the validation seeds, its S1, S2 and S3 picks at seed
    20 (a validation seed)."""
    J = report("sv25")
    specs = []
    for task in TASKS:
        lays = [e for e in J["tasks"].get(task, {}).get("layouts", []) if e.get("s3_val") is not None]
        if not lays:
            continue
        e = max(lays, key=lambda e: (e["s3_val"], e.get("s3_val_ramp_deg", 0.0)))
        L_ = next(x for x in R.layouts(task, "sv25") if x["call"] == e["call"])
        p2, p3 = R.picks(task, "sv25", "grasp"), R.picks(task, "sv25", "full")
        for name, v, prm, gpu in (("S1", L_["v1"], {}, e["s1_val"]), ("S2", p2[e["call"]]["v"], {}, e["s2_val"]),
                                  ("S3", p3[e["call"]]["v"], p3[e["call"]]["prm"], e["s3_val"])):
            specs.append({"task": task, "call": e["call"], "xy": e["xy"], "d6": L_["d"], "pick": name, "v": v,
                          "prm": prm, "gpu": gpu, "seed": R.VAL_SEEDS[0],
                          "top": f"{NAME[task]}, layout {e['xy'][0]:g} x {e['xy'][1]:g} mm, {name}",
                          "out": str(MEDIA / f"20261103-clip_{task}_{name}.mp4")})
    return specs


def cmd_films():
    specs = film_specs()
    hp, scenes, tag = R.PLANTS["sv25"]
    for s in specs:
        if Path(s["out"]).with_suffix(".json").exists():
            continue
        env = dict(os.environ, HOS_PLANT=hp, HOS_SCENES=scenes, HOM_ROWS_TAG=tag, CUDA_VISIBLE_DEVICES="",
                   MUJOCO_GL="egl", NUMPY_MADVISE_HUGEPAGE="0")
        cmd = [str(Path.home() / ".claude/bin/resguard.sh"), "run", "--mem", "3G", "--cpu", "200", "--",
               str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/hom_score_refine_page.py"), "clip", json.dumps(s)]
        while True:
            rc = subprocess.call(cmd, env=env, cwd=ROOT)
            if rc != 3:
                break
            print(time.strftime("%T"), "refused by resguard; retry in 120 s", flush=True)
            time.sleep(120)
        if rc:
            raise SystemExit(f"clip {s['out']} failed (exit {rc})")
    import hom_plant_page as HP
    for task in TASKS:
        cl = [s["out"] for s in specs if s["task"] == task]
        if cl:
            HP.compose(cl, MEDIA / f"20261103-film_{task}.mp4", cols=len(cl))
    res = [dict(s, **json.loads(Path(s["out"]).with_suffix(".json").read_text())) for s in specs]
    FILMS.write_text(json.dumps(res, indent=1))
    print("wrote", FILMS.relative_to(ROOT))
    return 0


def cmd_clip(spec):
    """One CPU rollout of a pick filmed in this process (the parent sets the plant's environment): the label on top,
    the scores below."""
    from dataclasses import replace
    import mujoco
    import fp_sr2_toy as FP
    import hom_contact_rig as HR
    import hom_films as HF
    import hom_gpu as HG
    import hom_landscape as L
    s = json.loads(spec)
    T, H, S = HG.T, HG.H, HG.S
    task = s["task"]
    FP.set_sizes(**({"sph_d_mm": R.TASKS[task]} if R.POSE[task] == "tripod" else {"cyl_d_mm": R.TASKS[task]}))
    setup = R.setup_of(HG, task, s["d6"], np.asarray(s["v"], float))
    mxy = HG.mount_xy(s["d6"])
    scene = HG.Scene("sphere" if R.POSE[task] == "tripod" else "cylinder", R.TASKS[task])
    bm, q_open = scene.prepare(mxy, setup)
    plant = T.Plant(scene.pads_path, setup)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for m in (plant.m, m_pt):
        for f in HG.FINGERS:
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    fl = HF.Film(plant, setup)
    var = "hold" if task in ("pinpush", "press") else "base"
    over, noise = T.VARIANTS[var]
    try:
        res, _tr, frames = T.rollout(task, setup, plant, mirror, q_open, s["seed"], {**T.PRM, **s["prm"]},
                                     replace(H.HomParams(dt=T.DT_C), **over), fl, noise,
                                     T.PLANT_VARIANTS.get(var, {}).get("hold_s", 0.0))
    finally:
        fl.close()
    cpu = L.rollout_score(task, res)
    frames = [HR.annotate_bottom(HR.annotate(f, s["top"], size=22),
                                 f"score {cpu:.2f} here; {s['gpu']:.2f} on the GPU, seeds 20-24", size=20)
              for f in frames]
    HR.write_mp4(frames, s["out"], fps=25)
    brief = dict(HF.res_brief(task, res), cpu=round(cpu, 3), frames=len(frames))
    Path(s["out"]).with_suffix(".json").write_text(json.dumps(brief, default=float))
    print("wrote", s["out"], brief, flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- page

def sec_terms():
    terms = [
        ("Layout", "the palm-frame mounts of the SR2 hand's three fingers, here a symmetric tripod in two coordinates. "
         "The thumb's mount lies $x_{sep}$ from the pair's, and the index's lies $y_{sep}$ from the middle's. The "
         "lattice steps by 5 mm, from 40 to 160 mm in $x_{sep}$ and from 50 to 170 mm in $y_{sep}$ [1]."),
        ("Grasp", "the object's pose and the three contacts on it, written as the pose toy's variables $v$. On the 40 mm "
         "sphere these are seven numbers: the centre (x, y, depth), each finger's azimuth offset and one latitude. On "
         "the 25 mm cylinder they are eight: the centre, the yaw, the positions of the pinch and the middle along the "
         "axis, and their two latitudes."),
        ("Feasible grasp", "a grasp that passes the pose toy's constraints. Each finger reaches its contact inside the "
         "servo limits less 3 deg, and its distal link faces the object. The pads touch the object, and the fingers' "
         "collision spheres stay 2 mm apart. The pinch has a lever of at least 40 mm."),
        ("Grasp set", "up to 16 feasible grasps per layout. The first is the foundational pose, solved for the largest "
         "smallest singular value of the pad Jacobians. The others come from a 600-step random walk over the feasible "
         "set, taken in farthest-point order."),
        ("HOM", "the hand-object controller of [2], reimplemented in <code>src/morphohand/control/hom.py</code> and "
         "batched on the GPU in <code>scripts/hom_gpu.py</code>. By default it commands 2.5 N per pad in the tripod "
         "and 3 N per pinching pad. The pin-push's middle finger drives with 1.5 N and the press's with 0.5 N."),
        ("Contact mode", "the weights of a finger's relative contact motion in Eq. (2) of [2]: spin, roll and slide. A "
         "weight of 1 keeps that motion at its reference, and 0 leaves it free; the separation row keeps weight 1. By "
         "default the tripod's fingers roll freely, as in the paper's Experiment B. In the pinch tasks the pinching "
         "fingers spin and roll freely and the pusher keeps all three, as in Experiment C."),
        ("Task score", "a rollout's score in [0, 1], 0 when the object drops. The twist scores its largest three-pad "
         "angle over 90 deg, capped at 1. The roll scores $1/(1 + e_\\omega)$, with $e_\\omega$ the angular-velocity "
         "error over the command. The pin-push and the press score $1 - e_\\theta/A$, with $e_\\theta$ the angle "
         "error about the pinch axis and $A$ the stroke of 15 or 20 deg. A layout's score is a mean over seeds."),
        ("S1, S2, S3", "three scores of one layout. S1 uses HOM's defaults and the best grasp of the grasp set; it is "
         "the score of the grids [3]. S2 uses HOM's defaults and the grasp refined by CEM from S1's grasp. S3 refines "
         "the grasp, the commanded forces and the contact modes together, from S2's grasp."),
        ("CEM", "the cross-entropy method. Candidates are drawn from a Gaussian, the best quarter are kept as elites, "
         "and the Gaussian moves toward them (Section SEC[method])."),
        ("Search and validation seeds", "a seed sets the object's start offset, up to 1 mm and 2 deg; seed 0 has none. "
         "The searches score every candidate at seeds 0-2, as the grids scored S1. The picks then run at seeds 20-24, "
         "which no grid or search used."),
        ("Spearman's rho", "the rank correlation of two scores over the same layouts; 1 means the same order."),
        ("Plants", "the finger servo model and the scene. The SCS0009 model (0.72 N m/rad, 0.18 N m) with pads at "
         "2.5 MPa is the hand's. The STS3032 model is tuned to 2.1 N m/rad and 0.515 N m. It runs at scene friction 1.0 "
         "and at 0.4, the friction of TPU on bare PLA. Section 2 of [3] defines both."),
    ]
    return "<dl>" + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_method(Js):
    J = Js.get("sv25") or next(iter(Js.values()))
    rows = [["Grasp $v$", "the pose toy's bounds", "0.25 (S2) or 0.15 (S3) of the feasible width",
             "0.02 of the feasible width"],
            ["Forces (S3)", "tripod 1-5 N; pinch 1.5-6 N; drive 0.5-3 N; press 0.2-2 N", "0.3 in $\\ln f$", "0.03"],
            ["Mode weights (S3)", "$w = u^2$, $u \\in [0, 1]$", "0.3 in $u$", "0.03"]]
    t = table(["Variables", "Range", "Starting spread", "Floor of the spread"], rows,
              "The CEM's variables. The feasible width of a grasp variable is the span over which it stays feasible "
              "with the others at the foundational pose (<code>fp_sr2_toy.grasp_set</code>).", num_from=9)
    return fill("""<p>Each sampled layout is scored three ways. S1 needs no new rollouts: it comes from the grids of
the plant page [3]. S2 and S3 come from a search per layout by the cross-entropy method. Iteration $k$ draws $n$
candidates $x$ from $\\mathcal{N}(\\mu_k, \\mathrm{diag}\\,\\sigma_k^2)$ inside the variables' ranges. A candidate whose
grasp fails the pose toy's constraints is drawn again, up to 40 times. Every candidate runs at seeds 0-2. The best
quarter by mean score are the elites $E$. Ties are broken by the twist's uncapped three-pad angle, or else by the
fingers in contact at the end. The update is</p>
$$\\mu_{k+1} = 0.8\\,\\bar{x}_E + 0.2\\,\\mu_k, \\qquad \\sigma_{k+1} = \\max\\left(0.8\\,s_E + 0.2\\,\\sigma_k,\\;
\\sigma_{\\min}\\right),$$
<p>with $\\bar{x}_E$ and $s_E$ the elites' mean and standard deviation. S2 searches the grasp alone with HOM's defaults,
from S1's grasp. It draws <<n2>> candidates per iteration for <<it>> iterations. S3 searches the grasp, the forces and
the mode weights together, from S2's best grasp and HOM's defaults. It draws <<n3>> candidates per iteration. The pick
of each level is the best candidate seen. That pick is the largest of about <<nc>> noisy means, so its search score is
biased upward. Each layout's three picks therefore run again at seeds 20-24, and Sections SEC[sv25] and SEC[sts]
compare those scores.</p>
<p>On the SCS0009 the layouts are the 30 best by S1 and 30 drawn at random from the others whose pose is reached, per
task. On the STS3032 they are the 20 best by S1 at friction 1.0 and 10 at random, run at both frictions. All rollouts
run on the GPU evaluator of the grids, with the forces and modes set per world. A check on 2026-11-03 compared default
worlds with the grid's rows: they agree within the evaluator's run-to-run spread. The CPU runner and the GPU also agree
on a world with changed forces and modes (README.md beside this page).</p>""",
                n2=J["pop"]["grasp"], n3=J["pop"]["full"], it=J["iters"],
                nc=J["pop"]["grasp"] * J["iters"]) + t


def level_table(J, caption):
    body = []
    for task in TASKS:
        if task not in J["tasks"]:
            continue
        lays = J["tasks"][task]["layouts"]
        for group, gl in (("top", "30 best by S1"), ("random", "random")):
            n = len([e for e in lays if e["group"] == group])
            if not n:
                continue
            d2, d3 = paired(lays, "s1_val", "s2_val", group), paired(lays, "s1_val", "s3_val", group)
            body.append([f"{NAME[task]}, {gl.replace('30', str(n))}", f2(med(vals(lays, "s1_val", group))),
                         f2(med(vals(lays, "s2_val", group))), f2(med(vals(lays, "s3_val", group))),
                         sg(d2["median"]), sg(d3["median"]), f"{d3['up']} / {d3['down']} of {d3['n']}"])
    return table(["Task, layouts", "S1", "S2", "S3", "S2 - S1", "S3 - S1", "S3 up / down by 0.05"], body, caption)


def rank_table(J, caption):
    body = []
    for task in TASKS:
        if task not in J["tasks"]:
            continue
        t = J["tasks"][task]
        b = {k: t.get(f"best_{k}") for k in ("s1_val", "s2_val", "s3_val")}
        bx = lambda k: f"{b[k]['xy'][0]:g} x {b[k]['xy'][1]:g} ({b[k]['score']:.2f})" if b[k] else "n/a"  # noqa: E731
        body.append([NAME[task], f2(t.get("rho_s1_val_s2_val")), f2(t.get("rho_s1_val_s3_val")),
                     f2(t.get("rho_s1_s1_val")), bx("s1_val"), bx("s3_val")])
    return table(["Task", "rho S1, S2", "rho S1, S3", "rho S1 at search, validation seeds", "Best layout by S1",
                  "Best layout by S3"], body, caption)


def sec_sv25(Js):
    J = Js.get("sv25")
    if not J:
        return "<p>The SCS0009 runs have not finished.</p>"
    t1 = level_table(J, "Median scores at the validation seeds (20-24) on the SCS0009 with pads at 2.5 MPa, and the "
                        "median paired change per layout. The last column counts the layouts whose S3 is more than "
                        "0.05 above or below their S1.")
    t2 = rank_table(J, "Rank agreement over each task's layouts at the validation seeds, and the best layout (x_sep x "
                       "y_sep, mm, with its score) under S1 and under S3.")
    fig1 = img(MEDIA / "20261103-levels_sv25.png", "Each layout's S1 (open circles), S2 (triangles) and S3 (squares) at "
               "the validation seeds, the layouts in order of S1. SCS0009, pads at 2.5 MPa.") \
        if (MEDIA / "20261103-levels_sv25.png").exists() else ""
    fig2 = img(MEDIA / "20261103-cem_sv25.png", "The searches' progress: the median over layouts of the best candidate "
               "so far at seeds 0-2 by iteration, with the quartiles shaded, against S1's median (dotted).") \
        if (MEDIA / "20261103-cem_sv25.png").exists() else ""
    ch = []
    for task in TASKS:
        if task in J["tasks"]:
            lays = J["tasks"][task]["layouts"]
            ch.append((NAME[task].lower(), paired(lays, "s1_val", "s2_val", "top")["median"],
                       paired(lays, "s1_val", "s3_val", "top")["median"]))
    text = ""
    if ch:
        text = ("<p>On the 30 best layouts of each task, refining the grasp changed the median validated score by "
                + ", ".join(f"{sg(a)} ({t})" for t, a, _b in ch) + ". Refining the forces and contact modes as well "
                "changed it by " + ", ".join(f"{sg(b)} ({t})" for t, _a, b in ch) + ". These are medians of each "
                "layout's own change. Table <<t1>> gives the medians for both groups of layouts, and Table <<t2>> the "
                "rank agreement.</p>")
    return fill(text, t1=P.TAB[0] - 1, t2=P.TAB[0]) + t1 + t2 + fig1 + fig2


def sec_prm(Js):
    J = Js.get("sv25")
    if not J:
        return ""
    import hom_tasks as T
    body = []
    for task in TASKS:
        lays = [e for e in J["tasks"].get(task, {}).get("layouts", []) if e.get("prm3")]
        if not lays:
            continue
        cells = []
        for k in list(R.FORCES[task]) + [f"w_{c}_{r}" for r in R.ROLES[task] for c in R.COMPS]:
            v = np.array([e["prm3"][k] for e in lays if k in e["prm3"]], float)
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            cells.append(f"{k.replace('w_', '').replace('_', ' ')}: {q2:.2f} ({q1:.2f}-{q3:.2f}; default {T.PRM[k]:g})")
        body.append([NAME[task], "<br>".join(cells)])
    t = table(["Task", "S3's forces (N) and mode weights: median (quartiles; HOM's default)"], body,
              "The forces and mode weights of the S3 picks over each task's layouts on the SCS0009.", num_from=9)
    fg = img(MEDIA / "20261103-prm_sv25.png", "The S3 picks' forces over HOM's defaults and their mode weights, one "
             "dot per layout; the black bars mark the defaults.") if (MEDIA / "20261103-prm_sv25.png").exists() else ""
    return "<p>Table <<t>> lists what the S3 search chose.</p>".replace("<<t>>", str(P.TAB[0])) + t + fg


def sec_sts(Js):
    A, B = Js.get("k21"), Js.get("k21mu0p4")
    if not (A and B):
        return "<p>The STS3032 runs have not finished.</p>"
    body = []
    for task in TASKS:
        if task not in A["tasks"] or task not in B["tasks"]:
            continue
        la, lb = A["tasks"][task]["layouts"], B["tasks"][task]["layouts"]
        body.append([NAME[task], f2(med(vals(la, "s1_val"))), f2(med(vals(la, "s3_val"))),
                     f2(med(vals(lb, "s1_val"))), f2(med(vals(lb, "s3_val_from_k21"))), f2(med(vals(lb, "s3_val"))),
                     f2(med(vals(la, "s3_val_from_k21mu0p4")))])
    t = table(["Task", "S1 at 1.0", "S3 at 1.0", "S1 at 0.4", "S3 tuned at 1.0, run at 0.4", "S3 tuned at 0.4",
               "S3 tuned at 0.4, run at 1.0"], body,
              "Median scores at the validation seeds on the STS3032 at kp 2.1, scene friction 1.0 and 0.4, over the "
              "same 30 layouts per task. The fifth column against the sixth is the gain of retuning the grasp, forces "
              "and modes once the friction is known.")
    return ("<p>On the STS3032 the same layouts were refined at friction 1.0 and at 0.4, and each friction's S3 picks "
            "ran at the other friction (Table <<t>>).</p>").replace("<<t>>", str(P.TAB[0])) + t


def sec_films():
    if not FILMS.exists():
        return "<p>The films are made after the GPU queue (<code>hom_score_refine_page.py films</code>).</p>"
    info = json.loads(FILMS.read_text())
    out = []
    for task in TASKS:
        fs = [s for s in info if s["task"] == task]
        if not fs:
            continue
        cap = (f"{NAME[task]} on the {OBJ[task]}, layout {fs[0]['xy'][0]:g} x {fs[0]['xy'][1]:g} mm on the SCS0009, "
               "the S1, S2 and S3 picks from left to right at seed 20, CPU plant: "
               + "; ".join(f"{s['pick']} {s['cpu']:.2f} here, {s['gpu']:.2f} on the GPU at seeds 20-24" for s in fs) + ".")
        out.append(film(MEDIA / f"20261103-film_{task}.mp4", cap))
    return "".join(out)


def sec_limits():
    return """<ul>
<li>The layouts are the symmetric tripods of the 5 mm lattice, at one object size per task. The size study [4] and the
six-coordinate layouts were not refined.</li>
<li>The twist's score saturates at 1.0, a 90 deg turn on three pads. Among saturated candidates the searches ranked by
the uncapped angle, and the page reports that angle beside the score.</li>
<li>The picks are the best of hundreds of noisy means. Only the validation seeds, five draws of the object's start,
compare the levels. The GPU evaluator's run-to-run spread is about 0.01 per rollout.</li>
<li>S3 tunes forces and modes per layout. A controller tuned once for all layouts was not searched.</li>
</ul>"""


def sec_future():
    return """<ul>
<li>One controller for all layouts. Tune HOM's forces, modes and gains once on a set of layouts, with per-world
<code>HomParams</code> in <code>hom_gpu</code>. Then rescore the grid and compare its ranking with S1's and S3's. The
fixed reference holds if Spearman's rho with S1 stays above 0.8.</li>
<li>The grasp in the evolutionary search's genome. A child inherits its parent's refined grasp, re-solved on its own
layout. Compare it with the best-of-16 search of <code>scripts/hom_ea.py</code> at equal rollouts.</li>
<li>A worst-case score: each candidate's lowest score over the friction band 0.4-1.0, with its picks run at each
friction.</li>
<li>The refined picks on the CPU plant with all contacts (<code>scripts/hom_top_cpu_check.py</code>), then on the
bench.</li>
</ul>"""


def sec_files():
    return """<p>Files: the search <code>scripts/hom_grasp_refine.py</code>; this page's builder
<code>scripts/hom_score_refine_page.py</code>; the runbook, queue and rows in
<code>docs/experiments/20261103-hom_score_refine/</code> (README.md); the logs in
<code>logs/20261103-hom_score_refine/</code>.</p>"""


REFS = [
    f'[1] <a href="{DESIGN}">HOM design landscapes of the SR2 hand</a> (2026-10-29).',
    '[2] The HOM paper, <a href="https://arxiv.org/abs/2609.25619">arXiv 2609.25619v2</a>, 2026.',
    f'[3] <a href="{PLANT_PAGE}">SR2 hand layouts under HOM control across finger servos, pad moduli and scene '
    'friction</a> (2026-11-02).',
    f'[4] <a href="{TRENDS}">Layout trends with object size</a> (2026-10-30).',
]


def lede_and_summary(Js):
    J = Js.get("sv25")
    if not J:
        return "The searches have not finished.", "Grasp and HOM parameter refinement of SR2 layouts: runs pending."
    parts = []
    for task in TASKS:
        if task in J["tasks"]:
            lays = J["tasks"][task]["layouts"]
            parts.append((task, med(vals(lays, "s1_val", "top")), med(vals(lays, "s2_val", "top")),
                          med(vals(lays, "s3_val", "top")), J["tasks"][task].get("rho_s1_val_s3_val")))
    lede = ("Each SR2 layout was scored three ways under HOM control. S1 takes the best of 16 sampled grasps at "
            "HOM's defaults. S2 refines the grasp on the HOM score, and S3 refines the grasp, forces and contact modes "
            "together. " + " ".join(f"In the {NAME[t].lower()} the median of the 30 best SCS0009 layouts went from "
                                  f"{f2(a)} to {f2(b)} and {f2(c)}." for t, a, b, c, _ in parts)
            + " Over 60 layouts per task, Spearman's rho between S1 and S3 is "
            + ", ".join(f"{f2(r)} ({NAME[t].lower()})" for t, _a, _b, _c, r in parts) + ".")
    summary = ("Grasp refinement on the HOM score (S2) and grasp, force and contact-mode refinement (S3) against the "
               "best of 16 grasps (S1) on SR2 layouts: median validated scores " + "; ".join(
                   f"{NAME[t].lower()} {f2(a)}/{f2(b)}/{f2(c)}" for t, a, b, c, _ in parts) + ".")
    return lede, summary


def render(page):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    return P.TEX_RE.sub(lambda m: next(svgs), page), len(items)


def cmd_page():
    Js = {p: report(p) for p in ("sv25", "k21", "k21mu0p4")}
    Js = {k: v for k, v in Js.items() if v}
    if not Js:
        raise SystemExit("no report yet: run scripts/hom_grasp_refine.py report")
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("method", "Scores, search and validation", sec_method(Js)),
        ("sv25", "Refinement on the SCS0009 plant", sec_sv25(Js)),
        ("prm", "Forces and contact modes chosen by the search", sec_prm(Js)),
        ("sts", "Refinement at two frictions on the STS3032", sec_sts(Js)),
        ("films", "Films of the picks", sec_films()),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future()),
        ("refs", "References and files", '<ol class="refs">' + "".join(f"<li>{r[r.index(']') + 2:]}</li>" for r in REFS)
         + "</ol>" + sec_files()),
    ]
    num = {sid: k for k, (sid, _hd, _b) in enumerate(sections, 1)}
    sub = (lambda s: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), s))
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|MPa|N\b|ms\b|s\b)", "\\1&nbsp;\\2", s_))
    sections = [(sid, hd, unit(sub(body))) for sid, hd, body in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>'
                     for k, (sid, hd, body) in enumerate(sections, 1))
    lede, summary = lede_and_summary(Js)
    byline = ("Simulation, 2026-11-03 and 04. Hand: the CAD-model SR2 hand with the v4b fingertip "
              "(<code>SCALE_HAND=cad_v2</code>); HOM on MuJoCo-Warp. Follows the plant and modulus page [3]. Builder: "
              "<code>scripts/hom_score_refine_page.py</code>." + (" SMOKE TEST DATA, NOT RESULTS." if SMOKE else ""))
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "HOM score levels", "H1": "Grasp, force and contact-mode refinement of SR2 hand layouts "
                 "under HOM control", "EYEBROW": "HOM layout study, 2026-11-03", "LEDE": unit(sub(lede)),
                 "BYLINE": byline, "TOC": toc, "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    page, n = render(page)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-03")
    page = retro_style.apply(page)
    out = Path(os.environ.get("SMOKE_OUT", "/tmp/20261103-hom_score_refine_smoke.html")) if SMOKE else OUT
    out.write_text(page)
    print(f"{out}: {out.stat().st_size / 1e6:.2f} MB, {n} formulas", flush=True)
    return 0


def main():
    a = [x for x in sys.argv[1:] if x != "--smoke"]
    if a and a[0] == "figs":
        return cmd_figs()
    if a and a[0] == "films":
        return cmd_films()
    if a and a[0] == "clip":
        return cmd_clip(a[1])
    return cmd_page()


if __name__ == "__main__":
    sys.exit(main())

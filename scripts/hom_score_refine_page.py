#!/usr/bin/env python3
"""Builds docs/experiments/20261103-hom_score_refine/20261103-hom_score_refine.html: each SR2 layout scored three ways
under HOM control (S1 the best of its sampled grasps at HOM's defaults; S2 the grasp refined on the HOM score; S3 the
grasp, the commanded forces and the contact modes refined together), on the SCS0009 plant and on the STS3032 at scene
friction 1.0 and 0.4 (owner, 2026-11-03: "can't we score and refine a grasp based off how well the grasp
(+morphology) executes the HOM control?"; "is it possible to co-design HOM?"; "let's send it on these tasks ... make
sure things are documented well!").

Data: the reports rows/<plant>/<date>-score_levels.json of scripts/hom_grasp_refine.py, its _gens.jsonl files (the
CEM's progress), the films' rows media/<date>-films.json, the friction check media/20261103-friction_check.json (figs:
drops, contact-normal elevations and the static friction need of the STS3032 picks, from the validation rows) and
compute/ (scripts/hom_compute.py). The results quote the elitist picks S2e and S3e. The worst-case section (2026-11-04)
reads rows/band_k21mu/<date>-band_levels.json (hom_grasp_refine.py report --band), its timing lines and the queue logs
of logs/20261104-hom_worst_case/. The design page's helpers and template (scripts/fp_review_page.py) draw the tables and
figures.

    .venv/bin/python scripts/hom_score_refine_page.py figs     # the figures and the friction check from the reports
    .venv/bin/python scripts/hom_score_refine_page.py films    # CPU clips through resguard (after the GPU queue)
    .venv/bin/python scripts/hom_score_refine_page.py band-figs    # the worst-case section's figure (2026-11-04)
    .venv/bin/python scripts/hom_score_refine_page.py band-films   # ... and its film
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


def grid4(figsize, **kw):
    """Four task panels in a 2 x 2 grid, sized for the page's 828 px column (text at least 13 px as displayed)."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, **kw)
    return fig, list(axs.ravel())


def labels(axs, xlabel, ylabel):
    """The x label under the bottom row and the y label left of the left column."""
    for ax in axs[2:]:
        ax.set_xlabel(xlabel)
    for ax in axs[::2]:
        ax.set_ylabel(ylabel)


def save(fig, name):
    import matplotlib.pyplot as plt
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", MEDIA / name, flush=True)


def fig_levels(plant, J):
    """Per task: each layout's S1 and its elitist picks S2e and S3e at the validation seeds, the layouts in order of
    S1."""
    plt = plt_setup()
    fig, axs = grid4((9.5, 7.0), sharey=True)
    for ax, task in zip(axs, TASKS):
        lays = [e for e in J["tasks"].get(task, {}).get("layouts", []) if e.get("s1_val") is not None]
        lays.sort(key=lambda e: (e["s1_val"], e["s1"]))
        x = np.arange(len(lays))
        for key, c, m, lab in (("s1_val", C1, "o", "S1"), ("s2e_val", C2, "^", "S2e"), ("s3e_val", C3, "s", "S3e")):
            xs = [i for i, e in zip(x, lays) if e.get(key) is not None]
            ys = [e[key] for e in lays if e.get(key) is not None]
            ax.plot(xs, ys, m, color=c, ms=5, mfc="none" if key == "s1_val" else c, label=lab, alpha=0.9)
        ax.set_title(f"{NAME[task]}, {OBJ[task]}")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    labels(axs, "Layouts in order of S1", "Score at seeds 20-24")
    axs[0].legend(loc="upper left", frameon=False)
    save(fig, f"20261103-levels_{plant}.png")


def fig_cem(plant):
    """Per task: the median over layouts of the best candidate so far (seeds 0-2) by iteration, for both levels; S1's
    median at iteration -1."""
    plt = plt_setup()
    J = report(plant)
    fig, axs = grid4((9.5, 6.8), sharey=True)
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
        ax.grid(alpha=0.3)
    labels(axs, "CEM iteration", "Best so far, seeds 0-2")
    axs[0].legend(loc="lower right", frameon=False)
    save(fig, f"20261103-cem_{plant}.png")


def fig_prm(plant, J):
    """S3's forces (relative to HOM's defaults) and mode weights over the layouts, per task."""
    import hom_tasks as T
    plt = plt_setup()
    fig, axs = grid4((9.5, 8.4))
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
        ax.set_xticklabels([f"{k[2:].capitalize()} force" if k.startswith("f_") else
                            "{}, {}".format(*k[2:].split("_")).capitalize() for k in keys], rotation=60, ha="right")
        ax.set_title(f"{NAME[task]}")
        ax.grid(alpha=0.3, axis="y")
    labels(axs, "", "Force over default, or mode weight")
    save(fig, f"20261103-prm_{plant}.png")


def fig_friction(A, B):
    """Per task on the STS3032: each layout's score at friction 0.4 with the picks tuned at 1.0 (x) and with the picks
    tuned at 0.4 (y), for S1 and S3e; the diagonal marks equal scores."""
    plt = plt_setup()
    fig, axs = grid4((8.6, 8.6), sharex=True, sharey=True)
    for ax, task in zip(axs, TASKS):
        lb = B["tasks"].get(task, {}).get("layouts", [])
        for kx, ky, c, m, lab in (("s1_val_from_k21", "s1_val", C1, "o", "S1"),
                                  ("s3e_val_from_k21", "s3e_val", C3, "s", "S3e")):
            pts = [(e[kx], e[ky]) for e in lb if e.get(kx) is not None and e.get(ky) is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], m, color=c, ms=6, alpha=0.8,
                    mfc="none" if lab == "S1" else c, label=lab)
        ax.plot([0, 1], [0, 1], color="k", lw=1)
        ax.set_title(NAME[task])
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    labels(axs, "Tuned at 1.0, run at 0.4", "Tuned at 0.4, run at 0.4")
    axs[0].legend(loc="lower right", frameon=False)
    save(fig, "20261103-friction_k21.png")


def cmd_figs():
    for plant in ("sv25", "k21", "k21mu0p4"):
        J = report(plant)
        if not J:
            continue
        fig_levels(plant, J)
        fig_cem(plant)
        fig_prm(plant, J)
    A, B = report("k21"), report("k21mu0p4")
    if A and B:
        fig_friction(A, B)
        friction_check()
    fig_throughput()
    return 0


# ----------------------------------------------------------------------------------------------- films

FRICTION_FILM = {"task": "twist", "call": 362}       # layout 110 x 110 mm, the plant page's one-layout twist design


def pick_of(task, plant, call, name):
    """(v, prm) of a layout's pick S1, S2 or S3 on a plant."""
    if name == "S1":
        return next(x for x in R.layouts(task, plant) if x["call"] == call)["v1"], {}
    p = R.picks(task, plant, "grasp" if name == "S2" else "full")[call]
    return p["v"], (p["prm"] if name == "S3" else {})


def film_specs():
    """Per task on the SCS0009: the layout with the highest S3e at the validation seeds, its S1, S2e and S3e picks at
    seed 20 (a validation seed). Then the friction film on the STS3032: one twist layout's S3e pick tuned at friction
    1.0, run at 1.0 and at 0.4, beside its S3e pick tuned at 0.4."""
    J = report("sv25")
    specs = []
    for task in TASKS:
        lays = [e for e in J["tasks"].get(task, {}).get("layouts", []) if e.get("s3e_val") is not None]
        if not lays:
            continue
        e = max(lays, key=lambda e: (e["s3e_val"], e.get(f"{e['s3e_pick'].lower()}_val_ramp_deg", 0.0)))
        L_ = next(x for x in R.layouts(task, "sv25") if x["call"] == e["call"])
        for name, raw in (("S1", "S1"), ("S2e", e["s2e_pick"]), ("S3e", e["s3e_pick"])):
            v, prm = pick_of(task, "sv25", e["call"], raw)
            specs.append({"set": "levels", "plant": "sv25", "task": task, "call": e["call"], "xy": e["xy"],
                          "d6": L_["d"], "pick": name, "raw": raw, "v": v, "prm": prm, "gpu": e[f"{name.lower()}_val"],
                          "seed": R.VAL_SEEDS[0],
                          "top": f"{NAME[task]}, layout {e['xy'][0]:g} x {e['xy'][1]:g} mm, {name}",
                          "out": str(MEDIA / f"20261103-clip_{task}_{name}.mp4")})
    A, B = report("k21"), report("k21mu0p4")
    if A and B:
        task, call = FRICTION_FILM["task"], FRICTION_FILM["call"]
        ea = next(e for e in A["tasks"][task]["layouts"] if e["call"] == call)
        eb = next(e for e in B["tasks"][task]["layouts"] if e["call"] == call)
        L_ = next(x for x in R.layouts(task, "k21") if x["call"] == call)
        va, pa = pick_of(task, "k21", call, ea["s3e_pick"])
        vb, pb = pick_of(task, "k21mu0p4", call, eb["s3e_pick"])
        for tag, plant, v, prm, gpu, lab in (
                ("10at10", "k21", va, pa, ea["s3e_val"], "tuned at friction 1.0, run at 1.0"),
                ("10at04", "k21mu0p4", va, pa, eb["s3e_val_from_k21"], "tuned at friction 1.0, run at 0.4"),
                ("04at04", "k21mu0p4", vb, pb, eb["s3e_val"], "tuned at friction 0.4, run at 0.4")):
            specs.append({"set": "friction", "plant": plant, "task": task, "call": call, "xy": ea["xy"], "d6": L_["d"],
                          "pick": "S3e", "cond": lab, "v": v, "prm": prm, "gpu": gpu, "seed": R.VAL_SEEDS[0],
                          "top": f"S3e {lab.replace('friction ', '')}",
                          "out": str(MEDIA / f"20261103-clip_friction_{task}_{tag}.mp4")})
    return specs


def cmd_films():
    specs = film_specs()
    for s in specs:
        if Path(s["out"]).with_suffix(".json").exists():
            continue
        hp, scenes, tag = R.PLANTS[s["plant"]]
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
        cl = [s["out"] for s in specs if s["task"] == task and s["set"] == "levels"]
        if cl:
            HP.compose(cl, MEDIA / f"20261103-film_{task}.mp4", cols=len(cl))
    cl = [s["out"] for s in specs if s["set"] == "friction"]
    if cl:
        HP.compose(cl, MEDIA / "20261103-film_friction_twist.mp4", cols=len(cl))
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
    if s.get("filter"):     # the GPU's contact filter, as scripts/hom_compute.py cpu-worker --variant filter applies it
        m = plant.m
        fine = re.compile(r"_(dip_link|cassette|cap)_s\d+$")
        filt = np.array([("_pad" in (m.geom(g).name or "")) or bool(fine.search(m.geom(g).name or "")
                                                                    and m.geom_size[g][0] < 0.002) for g in range(m.ngeom)])
        tool = np.flatnonzero(m.geom_bodyid == plant.tb)
        m.geom_contype[filt], m.geom_conaffinity[filt] = 2, 0
        m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
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
    frames = [HR.annotate_bottom(HR.annotate(f, s["top"], size=24), f"score {cpu:.2f}, GPU {s['gpu']:.2f}", size=23)
              for f in frames]
    HR.write_mp4(frames, s["out"], fps=25)
    brief = dict(HF.res_brief(task, res), cpu=round(cpu, 3), frames=len(frames))
    Path(s["out"]).with_suffix(".json").write_text(json.dumps(brief, default=float))
    print("wrote", s["out"], brief, flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- numbers from the rows

FRICTION_CHECK = MEDIA / "20261103-friction_check.json"
SPHERE_KG = 0.025                                   # the 40 mm sphere of the scenes


def angle_of(e, k):
    """The twist's three-pad angle (deg) at the validation seeds of a layout's pick k: s1, s2e or s3e."""
    name = "s1" if k == "s1" else e[f"{k}_pick"].lower()
    return e.get(f"{name}_val_ramp_deg") or 0.0


def best_layout(lays, k, task):
    """The layout with the highest validated score k (s1_val, s2e_val, s3e_val); the twist's ties at 1.0 go to the
    larger three-pad angle."""
    ok = [e for e in lays if e.get(k) is not None]
    if not ok:
        return None
    return max(ok, key=lambda e: (e[k], angle_of(e, k.removesuffix("_val")) if task == "twist" else 0.0))


def xy(e):
    return f"({e['xy'][0]:g}, {e['xy'][1]:g})"


def andj(items):
    """'a, b, c and d'."""
    items = [str(x) for x in items]
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


def fx(v):
    """Two decimals, or three below 0.01."""
    return f"{v:.3f}" if abs(v) < 0.01 else f"{v:.2f}"


WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four"}


def mu_need(n_out, f, weight):
    """The smallest friction coefficient at which tangential forces t_i, perpendicular to the inward normals n_i,
    balance the commanded normal forces f and the weight: min mu subject to sum(f n_i + t_i) = weight z and
    |t_i| <= mu f (scipy's SLSQP from three starts)."""
    from scipy.optimize import minimize
    import fp_sr2_toy as FP
    N = np.array([-np.asarray(n_out[k], float) for k in FP.FINGERS])
    target = np.array([0.0, 0.0, weight]) - f * N.sum(0)
    tt = lambda x: x[:9].reshape(3, 3)  # noqa: E731
    cons = [{"type": "eq", "fun": lambda x: tt(x).sum(0) - target},
            {"type": "eq", "fun": lambda x: np.einsum("ij,ij->i", tt(x), N)},
            {"type": "ineq", "fun": lambda x: (x[9] * f) ** 2 - (tt(x) ** 2).sum(1)}]
    best = None
    for s0 in (0.5, 1.0, 2.0):
        r = minimize(lambda x: x[9], np.r_[np.zeros(9), s0], constraints=cons, method="SLSQP",
                     options={"maxiter": 500, "ftol": 1e-10})
        if r.success and (best is None or r.x[9] < best):
            best = float(r.x[9])
    return best


def friction_check():
    """The STS3032 picks tuned at friction 1.0 and at 0.4 against the friction cone at 0.4, from the rows: the drops at
    each friction, the elevation of the pose toy's inward contact normals above the horizontal (the hand is palm-down),
    the friction each tripod grasp needs for static balance at its commanded forces (mu_need), and Spearman's rho of
    the pinch's latitude with the drops at 0.4. Written by figs to media/20261103-friction_check.json."""
    import math
    import fp_sr2_toy as FP
    import hom_tasks as T
    from scipy.stats import spearmanr
    cone = math.degrees(math.atan(0.4))
    out = {"cone_deg": round(cone, 1), "sphere_kg": SPHERE_KG, "tasks": {}}
    for task in TASKS:
        FP.set_sizes(**({"sph_d_mm": R.TASKS[task]} if R.POSE[task] == "tripod" else {"cyl_d_mm": R.TASKS[task]}))
        d = f"{task}_d{R.TASKS[task]:g}.jsonl"
        rr = {("1.0", "0.4"): R.read_rows(R.ROWS_ROOT / "k21mu0p4" / "val_from_k21" / d),
              ("1.0", "1.0"): R.read_rows(R.ROWS_ROOT / "k21" / "val" / d),
              ("0.4", "0.4"): R.read_rows(R.ROWS_ROOT / "k21mu0p4" / "val" / d),
              ("0.4", "1.0"): R.read_rows(R.ROWS_ROOT / "k21" / "val_from_k21mu0p4" / d)}
        t = {}
        for tuned, plant in (("1.0", "k21"), ("0.4", "k21mu0p4")):
            lays = {x["call"]: x for x in R.layouts(task, plant)}
            el = R.elitist(task, plant)
            p2, p3 = R.picks(task, plant, "grasp"), R.picks(task, plant, "full")
            for k in ("S1", "S3e"):
                elev, need, drops, n_rows, no_finger, forces = {}, [], {"0.4": [], "1.0": []}, {"0.4": 0, "1.0": 0}, 0, []
                for c, L_ in lays.items():
                    kk = "S1" if k == "S1" else el[c]["S3e"]
                    v = np.asarray(L_["v1"] if kk == "S1" else (p2 if kk == "S2" else p3)[c]["v"], float)
                    prm = p3[c]["prm"] if kk == "S3" else {}
                    if R.POSE[task] == "tripod":
                        e_ = {"tripod": math.degrees(v[6])}
                        f = prm.get("f_tripod", T.PRM["f_tripod"])
                        _o, _tg, n, _pc, _ax = FP.geometry("tripod", v, FP.mounts6(L_["d"]))
                        need.append(mu_need(n, f, SPHERE_KG * 9.81))
                        forces.append(f)
                    else:
                        e_ = {"pinch": math.degrees(v[6]), "middle": math.degrees(v[7])}
                        forces.append(prm.get("f_pinch", T.PRM["f_pinch"]))
                    for g, x in e_.items():
                        elev.setdefault(g, []).append(x)
                    for at in ("0.4", "1.0"):
                        mine = [r for r in rr[(tuned, at)] if r["call"] == c and r["pick"] == kk]
                        drops[at].append(sum(bool(r.get("dropped")) for r in mine))
                        n_rows[at] += len(mine)
                        if at == "0.4":
                            no_finger += sum(1 for r in mine if not r.get("dropped") and not r.get("held")
                                             and not int(r.get("fingers_end", 0)))
                rec = {"layouts": len(lays), "drops": {at: int(sum(x)) for at, x in drops.items()}, "rollouts": n_rows,
                       "no_finger_04": no_finger, "force_median": round(float(np.median(forces)), 3),
                       "elev": {g: {"median": round(float(np.median(x)), 1), "min": round(float(np.min(x)), 1),
                                    "max": round(float(np.max(x)), 1), "outside": int(sum(abs(y) > cone for y in x))}
                                for g, x in elev.items()}}
                if need:
                    ok = [x for x in need if x is not None]
                    rec["mu_need"] = {"median": round(float(np.median(ok)), 3), "min": round(float(np.min(ok)), 3),
                                      "max": round(float(np.max(ok)), 3), "above_04": int(sum(x > 0.4 for x in ok)),
                                      "n": len(ok)}
                if "pinch" in elev and np.ptp(drops["0.4"]) > 0:
                    rec["rho_pinch_drops_04"] = round(float(spearmanr(elev["pinch"], drops["0.4"]).statistic), 2)
                t[f"{k}_{tuned}"] = rec
        out["tasks"][task] = t
    FRICTION_CHECK.write_text(json.dumps(out, indent=1))
    print("wrote", FRICTION_CHECK.relative_to(ROOT), flush=True)


# ----------------------------------------------------------------------------------------------- page

def sec_terms():
    terms = [
        ("Layout", "the palm-frame mounts of the SR2 hand's three fingers, here a symmetric tripod in two coordinates. "
         "The thumb's mount lies $x_{sep}$ from the pair's, and the index's lies $y_{sep}$ from the middle's. The "
         "lattice steps by 5 mm, from 40 to 160 mm in $x_{sep}$ and from 50 to 170 mm in $y_{sep}$ [1]. A layout is "
         "written ($x_{sep}$, $y_{sep}$) in mm."),
        ("Grasp", "the object's pose and the three contacts on it, written as the pose toy's variables $v$. On the 40 mm "
         "sphere these are seven numbers: the centre (x, y, depth), each finger's azimuth offset and one latitude. On "
         "the 25 mm cylinder they are eight: the centre, the yaw, the positions of the pinch and the middle along the "
         "axis, and their two latitudes. A latitude is the contact's angle below the object's equator."),
        ("Feasible grasp", "a grasp that passes the pose toy's constraints. Each finger reaches its contact inside the "
         "servo limits less 3 deg, and its distal link faces the object. The pads touch the object, and the fingers' "
         "collision spheres stay 2 mm apart. The pinch has a lever of at least 40 mm."),
        ("Grasp set", "up to 16 feasible grasps per layout. The first is the foundational pose, solved for the largest "
         "smallest singular value of the pad Jacobians. The others come from a 600-step random walk over the feasible "
         "set, taken in farthest-point order."),
        ("HOM", "the hand-object controller of [2], reimplemented in <code>src/morphohand/control/hom.py</code> and "
         "batched on the GPU in <code>scripts/hom_gpu.py</code>. By default it commands 2.5 N per pad in the tripod "
         "and 3 N per pinching pad. The pin-push's middle finger drives with 1.5 N and the press's with 0.5 N."),
        ("Contact mode", "the weights of a finger's relative contact motion in Eq. (2) of [2]: spin, roll and slide. "
         "With a weight of 1 that motion follows its reference, and with 0 it is free; the separation row has weight 1 "
         "throughout. By default the tripod's fingers roll freely, as in the paper's Experiment B. In the pinch tasks "
         "the pinching fingers spin and roll freely and the pusher's three weights are 1, as in Experiment C."),
        ("Task score", "a rollout's score in [0, 1], 0 when the object drops. The twist scores its three-pad angle over "
         "90 deg, capped at 1. The roll scores $1/(1 + e_\\omega)$, with $e_\\omega$ the angular-velocity error over "
         "the command. The pin-push and the press score $1 - e_\\theta/A$, with $e_\\theta$ the angle error about the "
         "pinch axis and $A$ the stroke of 15 or 20 deg. A layout's score is a mean over seeds."),
        ("Three-pad angle", "the twist's largest turn of the sphere with all three pads in contact, in deg, uncapped."),
        ("Drop", "a rollout whose object ends more than 15 mm below its starting height. It scores 0."),
        ("S1, S2, S3", "three scores of one layout. S1 is scored with HOM's defaults and the best grasp of the grasp "
         "set, as in the grids [3]. S2 is scored with HOM's defaults and the best grasp that a search found from S1's "
         "grasp. S3 is the score of the best candidate of a search over the grasp, the commanded forces and the contact "
         "modes together, started from S2's grasp."),
        ("Elitist pick", "the better of a search's start and its best candidate, by the mean score at seeds 0-2. S2e "
         "is the better of S1's grasp and S2's pick. S3e is the best of S1, S2 and S3. Ties go to the larger "
         "tie-break value (Section SEC[method]) and then to the earlier level. The results quote S2e and S3e; S2 and "
         "S3 name the searches' own picks."),
        ("Paired change", "a layout's validated score under one level less its validated score under another. Tables "
         "give its median over the layouts."),
        ("Best layout", "the layout with the highest validated score under a level. Twist scores tie at 1.0, and the "
         "ties are broken by the three-pad angle."),
        ("CEM", "the cross-entropy method. Candidates are drawn from a Gaussian, the best quarter are kept as elites, "
         "and the Gaussian moves toward them (Section SEC[method])."),
        ("Search and validation seeds", "a seed sets the object's start offset, up to 1 mm and 2 deg; seed 0 has none. "
         "The searches score every candidate at seeds 0-2, as the grids scored S1. The picks then run at seeds 20-24, "
         "which no grid or search used. A validated score is the mean at seeds 20-24."),
        ("Spearman's rho", "the rank correlation of two scores over the same layouts; 1 means the same order."),
        ("Contact-normal elevation", "the angle of a contact's inward normal above the horizontal, from the pose toy's "
         "grasp. The hand is palm-down, so a positive elevation pushes the object up toward the palm."),
        ("Friction cone", "the contact forces within $\\arctan\\mu$ of the inward normal, with $\\mu$ the scene "
         "friction: 21.8 deg at 0.4 and 45 deg at 1.0. A pad whose force leaves it slides."),
        ("Static friction need", "the smallest friction coefficient $\\mu_{need}$ at which tangential pad forces can "
         "balance a grasp's commanded normal forces and the object's weight (Section SEC[sts])."),
        ("Run, batch and step time", "three clocks of the GPU work, each divided by the rollouts it covers. Run time is "
         "the rollout loop of one batch (the GPU's physics and the batched controller, with the host's loop). Batch "
         "time adds the batch's model build and its scoring. Step time is a whole queue step's wall clock, with the "
         "CPU's candidate draws, open poses and the start of a process per batch."),
        ("Speedup", "the CPU runner's wall seconds per rollout on one core, divided by the GPU's seconds per rollout "
         "at one of those clocks. It is the number of CPU cores that one GPU equals if the CPU scaled linearly."),
        ("Real-time factor", "simulated seconds per wall-clock second of one rollout."),
        ("Plants", "the finger servo model and the scene. The SCS0009 model (0.72 N m/rad, 0.18 N m) with pads at "
         "2.5 MPa is the hand's. The STS3032 model is tuned to 2.1 N m/rad and 0.515 N m. It runs at scene friction 1.0 "
         "and at 0.4, the friction of TPU on bare PLA, and in Section SEC[band] also at 0.7. Section 2 of [3] defines "
         "both."),
        ("Worst-case score", "a candidate's lowest three-seed mean score over the friction band, the STS3032 at scene "
         "friction 0.4, 0.7 and 1.0 (Section SEC[band])."),
        ("S1w, S2w, S3w", "S1, S2 and S3 with the worst-case score in place of the mean score at one friction. Their "
         "searches scored the start as the first candidate, so S2w and S3w are elitist by construction."),
        ("Lowest score over the band", "a pick's lowest validated score over the three frictions of the band."),
        ("Contact filter", "the rule for which hand geoms meet the object. On the GPU only the pads and the rigid "
         "spheres under 2 mm meet it. On the CPU plant every geom meets it, unless the GPU's contact filter is applied "
         "(<code>hom_compute.py</code>'s variant <code>filter</code>)."),
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
quarter by mean score are the elites $E$. Ties are broken by the twist's three-pad angle, or else by the fingers in
contact at the end. The update is</p>
$$\\mu_{k+1} = 0.8\\,\\bar{x}_E + 0.2\\,\\mu_k, \\qquad \\sigma_{k+1} = \\max\\left(0.8\\,s_E + 0.2\\,\\sigma_k,\\;
\\sigma_{\\min}\\right),$$
<p>with $\\bar{x}_E$ and $s_E$ the elites' mean and standard deviation. S2 searches the grasp alone with HOM's defaults,
from S1's grasp. It draws <<n2>> candidates per iteration for <<it>> iterations. S3 searches the grasp, the forces and
the mode weights together, from S2's pick and HOM's defaults. It draws <<n3>> candidates per iteration. The pick of each
search is the best candidate seen. That pick is the largest of about <<nc>> noisy means, so its search score is biased
upward. Each layout's three picks therefore run again at seeds 20-24, and those validated scores are compared.</p>
<p>A search scores draws around its start and never the start itself, so its pick can score below the start. The S3
search started from the S2 search's own pick, also where that pick scored below S1's grasp. The elitist picks S2e and
S3e keep the start where it scored higher (Section SEC[terms]). Searches run after 2026-11-04 score the start as the
first candidate of the first iteration.</p>
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
            d2, d3 = paired(lays, "s1_val", "s2e_val", group), paired(lays, "s2e_val", "s3e_val", group)
            d31 = paired(lays, "s1_val", "s3e_val", group)
            body.append([f"{NAME[task]}, {gl.replace('30', str(n))}", f2(med(vals(lays, "s1_val", group))),
                         f2(med(vals(lays, "s2e_val", group))), f2(med(vals(lays, "s3e_val", group))),
                         sg(d2["median"]), sg(d3["median"]), sg(d31["median"]),
                         f"{d31['up']} / {d31['down']} of {d31['n']}"])
    return table(["Task, layouts", "S1", "S2e", "S3e", "S2e - S1", "S3e - S2e", "S3e - S1", "S3e up / down by 0.05"],
                 body, caption)


def rank_table(J, caption):
    body = []
    for task in TASKS:
        if task not in J["tasks"]:
            continue
        t = J["tasks"][task]
        lays = t["layouts"]
        cells = []
        for k in ("s1_val", "s3e_val"):
            b = best_layout(lays, k, task)
            cells.append(f"{xy(b)}: {b[k]:.2f}" + (f", {angle_of(b, k[:-4]):.0f} deg" if task == "twist" else ""))
        body.append([NAME[task], f2(t.get("rho_s1_val_s2e_val")), f2(t.get("rho_s1_val_s3e_val")),
                     f2(t.get("rho_s1_s1_val")), *cells])
    return table(["Task", "Rho S1, S2e", "Rho S1, S3e", "Rho S1 at search and validation seeds", "Best layout by S1",
                  "Best layout by S3e"], body, caption)


def elitist_table(J, caption):
    body = []
    for task in TASKS:
        if task not in J["tasks"]:
            continue
        t = J["tasks"][task]
        bg, bf = t["below_start_grasp"], t["below_start_full"]
        cell = (lambda b: f"{b['n']} of {b['of']}" + (f" (-{fx(-b['worst'])})" if b["worst"] is not None else ""))
        body.append([NAME[task], cell(bg), cell(bf), f2(t["median_s2_val"]), f2(t["median_s2e_val"]),
                     f2(t["median_s3_val"]), f2(t["median_s3e_val"])])
    return table(["Task", "Grasp searches below their start", "Full searches below their start", "S2", "S2e", "S3",
                  "S3e"], body, caption)


def sec_sv25(Js):
    J = Js.get("sv25")
    if not J:
        return "<p>The SCS0009 runs have not finished.</p>"
    n0 = P.TAB[0]
    t1 = level_table(J, "Median validated scores (seeds 20-24) on the SCS0009 with pads at 2.5 MPa, and the median "
                        "paired changes per layout. The last column counts the layouts whose S3e is more than 0.05 "
                        "above or below their S1.")
    t2 = rank_table(J, "Rank agreement over each task's 60 layouts at the validation seeds, and the best layout "
                       "($x_{sep}$, $y_{sep}$) with its validated score under S1 and under S3e; on the twist also its "
                       "three-pad angle. The third column compares S1 at seeds 0-2 with S1 at seeds 20-24.")
    t3 = elitist_table(J, "Searches whose pick scored below their start at seeds 0-2, of the 60 layouts per task, "
                          "with the largest shortfall; and the median validated scores over the 60 layouts of the "
                          "searches' own picks (S2, S3) and the elitist picks (S2e, S3e).")
    f0 = P.FIG[0]
    fig1 = img(MEDIA / "20261103-levels_sv25.png", "Each layout's S1 (open circles), S2e (triangles) and S3e "
               "(squares) at the validation seeds, the layouts in order of S1. SCS0009, pads at 2.5 MPa.") \
        if (MEDIA / "20261103-levels_sv25.png").exists() else ""
    fig2 = img(MEDIA / "20261103-cem_sv25.png", "The searches' progress: the median over the 60 layouts of the best "
               "candidate so far at seeds 0-2, by iteration, with the quartiles shaded, against S1's median "
               "(dotted).") if (MEDIA / "20261103-cem_sv25.png").exists() else ""
    tk = {task: J["tasks"][task]["layouts"] for task in TASKS if task in J["tasks"]}
    top = {task: {k: paired(L_, a, b, "top") for k, (a, b) in
                  {"g": ("s1_val", "s2e_val"), "f": ("s2e_val", "s3e_val"), "a": ("s1_val", "s3e_val")}.items()}
           for task, L_ in tk.items()}
    rnd = {task: paired(L_, "s1_val", "s3e_val", "random") for task, L_ in tk.items()}
    lo = min(tk, key=lambda t_: top[t_]["a"]["median"])
    hi = max(tk, key=lambda t_: top[t_]["a"]["median"])
    T_ = lambda k: andj(f"{fx(top[t_][k]['median'])} on the {NAME[t_].lower()}" for t_ in tk)  # noqa: E731
    falls = {t_: [e["s3e_val"] - e["s1_val"] for e in L_ if e["group"] == "top"] for t_, L_ in tk.items()}
    tw = [e for e in tk.get("twist", []) if e["group"] == "top"]
    sat = {k: sum(e[f"{k}_val"] >= 0.9999 for e in tw) for k in ("s1", "s2e", "s3e")}
    ang = {k: med([angle_of(e, k) for e in tw]) for k in ("s1", "s2e", "s3e")}
    rho = {t_: (J["tasks"][t_]["rho_s1_val_s3e_val"], J["tasks"][t_]["rho_s1_val_s2e_val"]) for t_ in tk}
    moved = [(t_, best_layout(L_, "s1_val", t_), best_layout(L_, "s3e_val", t_)) for t_, L_ in tk.items()]
    bg = sum(J["tasks"][t_]["below_start_grasp"]["n"] for t_ in tk)
    bf = sum(J["tasks"][t_]["below_start_full"]["n"] for t_ in tk)
    nl = sum(J["tasks"][t_]["below_start_grasp"]["of"] for t_ in tk)
    worst = sorted(((J["tasks"][t_][f"below_start_{lv}"]["worst"], t_, lv) for t_ in tk for lv in ("grasp", "full")
                    if J["tasks"][t_][f"below_start_{lv}"]["worst"] is not None))[:2]
    same = [t_ for t_, a, b in moved if a["call"] == b["call"]]
    mv = [f"from {xy(a)} to {xy(b)} on the {NAME[t_].lower()}" for t_, a, b in moved if a["call"] != b["call"]]
    paras = [
        f"""<p>Refining the grasp, the commanded forces and the contact modes raised the validated scores of each
task's 30 best layouts (Table {n0 + 1}). The median gain over S1 ranges from {top[lo]['a']['median']:.2f} on the
{NAME[lo].lower()} to {top[hi]['a']['median']:.2f} on the {NAME[hi].lower()}. The grasp search alone gave a median {T_('g')}. The forces and
modes added {T_('f')}. On the press the median rose from {med(vals(tk['press'], 's1_val', 'top')):.2f} under S1 to
{med(vals(tk['press'], 's2e_val', 'top')):.2f} under S2e and {med(vals(tk['press'], 's3e_val', 'top')):.2f} under
S3e.</p>""",
        f"""<p>The 30 random layouts start lower and gain more. Their median S3e - S1 is
{andj(f"{rnd[t_]['median']:.2f} on the {NAME[t_].lower()}" for t_ in tk)}. On the 30 best layouts, S3e falls more than
0.05 below S1 on {andj(f"{WORDS.get(sum(x < -0.05 for x in falls[t_]))} of the {NAME[t_].lower()}" for t_ in tk)}.
The largest fall is {-min(falls['twist']):.2f} on the twist and {-min(falls['roll']):.2f} on the roll. These picks
scored at least as high as S1 at seeds 0-2.</p>""",
        f"""<p>On the twist, {sat['s1']} of the 30 best layouts score 1.0 under S1, {sat['s2e']} under S2e and
{sat['s3e']} under S3e. The median three-pad angle of the 30 rises from {ang['s1']:.0f} deg under S1 to {ang['s2e']:.0f} deg
under S2e and {ang['s3e']:.0f} deg under S3e (Figure {f0 + 1}).</p>""",
        f"""<p>The refined scores reorder the layouts. Over the 60 layouts of each task, Spearman's rho between S1 and
S3e is {min(r[0] for r in rho.values()):.2f} to {max(r[0] for r in rho.values()):.2f} (Table {n0 + 2}). Between S1 and
S2e it is {min(r[1] for r in rho.values()):.2f} to {max(r[1] for r in rho.values()):.2f}. Under S3e the best layout moves on {WORDS[len(mv)]} of the four tasks. It
moves {andj(mv)}. """ + (f"The {andj(NAME[t_].lower() for t_ in same)}'s best layout stays at "
       f"{andj(xy(a) for t_, a, b in moved if t_ in same)}." if same else "") + "</p>",
        f"""<p>At seeds 0-2, {bg} of {nl} grasp searches and {bf} of {nl} full searches picked a candidate below their
start (Table {n0 + 3}). The largest shortfalls are {-worst[0][0]:.2f} on the {NAME[worst[0][1]].lower()} and
{-worst[1][0]:.2f} on the {NAME[worst[1][1]].lower()}, """ + (f"both in the {worst[0][2]} searches" if worst[0][2] == worst[1][2]
                                                    else f"in the {worst[0][2]} and {worst[1][2]} searches") + f""". The elitist picks change no median of
the 30 best layouts. Over all 60 layouts they move the twist's median S3 from {J['tasks']['twist']['median_s3_val']:.2f}
to {J['tasks']['twist']['median_s3e_val']:.2f}.</p>""",
        f"""<p>The best candidates rise for about four of the eight iterations (Figure {f0 + 2}). On the press the
first iteration of the S3 search already scores above the last of the S2 search.</p>""",
    ]
    out = paras[0] + t1 + paras[1] + paras[2] + fig1 + paras[3] + t2 + paras[4] + t3 + paras[5] + fig2
    return out


def sec_prm(Js):
    J = Js.get("sv25")
    if not J:
        return ""
    import hom_tasks as T
    body, q = [], {}
    for task in TASKS:
        lays = [e for e in J["tasks"].get(task, {}).get("layouts", []) if e.get("prm3")]
        if not lays:
            continue
        cells = []
        for k in list(R.FORCES[task]) + [f"w_{c}_{r}" for r in R.ROLES[task] for c in R.COMPS]:
            v = np.array([e["prm3"][k] for e in lays if k in e["prm3"]], float)
            q[(task, k)] = np.percentile(v, [25, 50, 75])
            lab = (f"{k[2:].capitalize()} force (N)" if k.startswith("f_")
                   else "{}, {}".format(*k[2:].split("_")).capitalize())
            cells.append(f"{lab}: {q[(task, k)][1]:.2f} ({q[(task, k)][0]:.2f}-{q[(task, k)][2]:.2f}; "
                         f"default {T.PRM[k]:g})")
        body.append([NAME[task], "<br>".join(cells)])
    n0, f0 = P.TAB[0], P.FIG[0]
    t = table(["Task", "S3's forces (N) and mode weights: median (quartiles; HOM's default)"], body,
              "The forces and mode weights of the S3 search's picks over each task's 60 layouts on the SCS0009.",
              num_from=9)
    fg = img(MEDIA / "20261103-prm_sv25.png", "The S3 picks' forces over HOM's defaults and their mode weights, one "
             "dot per layout; the black bars mark the defaults.") if (MEDIA / "20261103-prm_sv25.png").exists() else ""
    m = lambda t_, k: q[(t_, k)][1]  # noqa: E731
    fixed = [m(t_, f"w_{c}_{r}") for t_ in TASKS for r in R.ROLES[t_] for c in R.COMPS if T.PRM[f"w_{c}_{r}"] == 1.0]
    free = [q[(t_, f"w_{c}_{r}")][2] for t_ in TASKS for r in R.ROLES[t_] for c in R.COMPS
            if T.PRM[f"w_{c}_{r}"] == 0.0]
    text = f"""<p>In the S3 picks the tripod's force stayed near HOM's 2.5 N: a median {m('twist', 'f_tripod'):.2f} N on
the twist and {m('roll', 'f_tripod'):.2f} N on the roll (Table {n0 + 1}, Figure {f0 + 1}). The pinch force on the
cylinder fell from 3 N to a median {m('pinpush', 'f_pinch'):.2f} N on the pin-push and {m('press', 'f_pinch'):.2f} N on
the press. The largest gain from S2e to S3e is the <<gmax>>'s. The pin-push's drive stayed at
{m('pinpush', 'f_drive'):.2f} N, and the press force went from 0.5 N to {m('press', 'f_press'):.2f} N.</p>
<p>The modes that HOM leaves free by default stayed nearly free: their weights stayed below {max(free):.2f} in three
quarters of the layouts. The weights that are 1 by default fell to medians of {min(fixed):.2f} to {max(fixed):.2f}. The
pusher's roll weight fell most, to {m('press', 'w_roll_push'):.2f} on the press and {m('pinpush', 'w_roll_push'):.2f}
on the pin-push. With a lower weight, that motion departs further from its reference.</p>"""
    gmax = max((t_ for t_ in TASKS if t_ in J["tasks"]),
               key=lambda t_: paired(J["tasks"][t_]["layouts"], "s2e_val", "s3e_val", "top")["median"])
    return text.replace("<<gmax>>", NAME[gmax].lower()) + t + fg


def sec_sts(Js):
    A, B = Js.get("k21"), Js.get("k21mu0p4")
    if not (A and B):
        return "<p>The STS3032 runs have not finished.</p>"
    body = []
    tk = [t_ for t_ in TASKS if t_ in A["tasks"] and t_ in B["tasks"]]
    la = {t_: A["tasks"][t_]["layouts"] for t_ in tk}
    lb = {t_: B["tasks"][t_]["layouts"] for t_ in tk}
    for task in tk:
        g = paired(lb[task], "s3e_val_from_k21", "s3e_val")
        body.append([NAME[task], f2(med(vals(la[task], "s1_val"))), f2(med(vals(la[task], "s3e_val"))),
                     f2(med(vals(lb[task], "s1_val"))), f2(med(vals(lb[task], "s3e_val_from_k21"))),
                     f2(med(vals(lb[task], "s3e_val"))), sg(g["median"]),
                     f2(med(vals(la[task], "s3e_val_from_k21mu0p4")))])
    n0, f0, m0 = P.TAB[0], P.FIG[0], P.FILM[0]
    t = table(["Task", "S1 at 1.0", "S3e at 1.0", "S1 at 0.4", "S3e tuned at 1.0, run at 0.4", "S3e tuned at 0.4",
               "Gain of retuning at 0.4", "S3e tuned at 0.4, run at 1.0"], body,
              "Median validated scores on the STS3032 at kp 2.1, scene friction 1.0 and 0.4, over the same 30 layouts "
              "per task. The gain of retuning is the median paired change from the fifth column to the sixth.")
    fg = img(MEDIA / "20261103-friction_k21.png", "Each layout's validated score at friction 0.4 with the picks tuned "
             "at 1.0 (horizontal) and with the picks tuned at 0.4 (vertical), for S1 (open circles) and S3e (squares). "
             "Points above the diagonal gain from retuning.") if (MEDIA / "20261103-friction_k21.png").exists() else ""
    gain = {t_: paired(lb[t_], "s3e_val_from_k21", "s3e_val")["median"] for t_ in tk}
    gain1 = {t_: paired(lb[t_], "s1_val_from_k21", "s1_val")["median"] for t_ in tk}
    back = {t_: paired(la[t_], "s3e_val_from_k21mu0p4", "s3e_val")["median"] for t_ in tk}
    up10 = {t_: paired(la[t_], "s1_val", "s3e_val")["median"] for t_ in tk}
    zero = [t_ for t_ in tk if med(vals(lb[t_], "s3e_val_from_k21")) < 0.005]
    others = [t_ for t_ in tk if t_ not in zero]
    lst = lambda d: andj(f"{d[t_]:.2f} on the {NAME[t_].lower()}" for t_ in tk)  # noqa: E731
    paras = [f"""<p>On the STS3032 at friction 1.0, refinement gains less than on the SCS0009. Over the 30 layouts the
median S3e - S1 is {lst(up10)} (Table {n0 + 1}).</p>"""]
    if FRICTION_CHECK.exists():
        F = json.loads(FRICTION_CHECK.read_text())["tasks"]
        dr = lambda k, tu, at: [F[t_][f"{k}_{tu}"]["drops"][at] for t_ in tk]  # noqa: E731
        per = lambda xs: andj(f"{x} on the {NAME[t_].lower()}" for x, t_ in zip(xs, tk))  # noqa: E731
        nr = F[tk[0]]["S3e_1.0"]["rollouts"]["0.4"]
        paras.append(f"""<p>At friction 0.4 the picks tuned at 1.0 fail. Their S3e picks score a median 0.00 on the
{andj(NAME[t_].lower() for t_ in zero)}, and {andj(f"{med(vals(lb[t_], 's3e_val_from_k21')):.2f} on the "
                                                f"{NAME[t_].lower()}" for t_ in others)}. Of {nr} rollouts per task,
they drop the object in {per(dr('S3e', '1.0', '0.4'))}. On the roll, {F['roll']['S3e_1.0']['no_finger_04']} more
rollouts end with no finger on the sphere. The picks tuned at 0.4 drop it in {per(dr('S3e', '0.4', '0.4'))}. At
friction 1.0 no pick drops the object in more than
{max(max(dr(k, tu, '1.0')) for k in ('S1', 'S3e') for tu in ('1.0', '0.4'))} of {nr} rollouts.</p>""")
    paras.append(f"""<p>Retuning at friction 0.4 gains a median {lst(gain)} (Figure {f0 + 1}). For S1, choosing among the same 16 sampled grasps at 0.4 gains {lst(gain1)}. Run at friction
1.0, the S3e picks tuned at 0.4 score below the picks tuned at 1.0. The median loss is {lst(back)}.</p>""")
    if FRICTION_CHECK.exists():
        F = json.loads(FRICTION_CHECK.read_text())
        G = F["tasks"]
        el = [G[t_][f"{k}_{tu}"]["elev"][g] for t_ in tk for k in ("S1", "S3e") for tu in ("1.0", "0.4")
              for g in G[t_][f"{k}_{tu}"]["elev"] if g != "middle"]
        mid = [G[t_][f"{k}_{tu}"]["elev"]["middle"] for t_ in ("press",) if t_ in tk for k in ("S1", "S3e")
               for tu in ("1.0", "0.4")]
        mu = {(t_, tu): G[t_][f"S1_{tu}"]["mu_need"]["median"] for t_ in ("twist", "roll") for tu in ("1.0", "0.4")}
        paras.append(f"""<p>The planned contact normals do not separate the grasps that drop at friction 0.4 from
those that stay. In both sets the pose toy's inward normals point {min(x['min'] for x in el):.0f} to
{max(x['max'] for x in el):.0f} deg above the horizontal, inside the {F['cone_deg']:.1f} deg cone of friction 0.4. Only
the press's middle finger, at {min(x['min'] for x in mid):.0f} to {max(x['max'] for x in mid):.0f} deg by the bounds of
its pose, lies outside it, in both sets. The friction needed for static balance with the commanded normal forces
$f_i$ and the {1000 * F['sphere_kg']:.0f} g sphere is</p>
$$\\mu_{{need}} = \\min\\Big\\{{\\mu : \\sum_i \\left(f_i \\hat n_i + t_i\\right) = m g\\,\\hat z,\\;\\; t_i \\perp \\hat n_i,\\;\\;
\\lVert t_i \\rVert \\le \\mu f_i\\Big\\}},$$
<p>with $\\hat n_i$ the inward normals. For the twist's S1 grasps its median is {mu[('twist', '1.0')]:.2f} for those
tuned at 1.0 and {mu[('twist', '0.4')]:.2f} for those tuned at 0.4; for the roll's it is {mu[('roll', '1.0')]:.2f} and
{mu[('roll', '0.4')]:.2f}. On the cylinder tasks the drops of the friction-1.0 S1 grasps at 0.4 rise with the pinch's
latitude: Spearman's rho is {G['pinpush']['S1_1.0']['rho_pinch_drops_04']:.2f} on the pin-push and
{G['press']['S1_1.0']['rho_pinch_drops_04']:.2f} on the press. The S1 grasps tuned at 0.4 pinch at a median
{G['pinpush']['S1_0.4']['elev']['pinch']['median']:.0f} deg on the pin-push, against
{G['pinpush']['S1_1.0']['elev']['pinch']['median']:.0f} deg for those tuned at 1.0.</p>""")
    fm = ""
    if FILMS.exists():
        info = [s for s in json.loads(FILMS.read_text()) if s.get("set") == "friction"]
        if len(info) == 3:
            a, b, c = info
            va, vb = np.asarray(a["v"], float), np.asarray(c["v"], float)
            fa = a["prm"].get("f_tripod", 2.5)
            fb = c["prm"].get("f_tripod", 2.5)
            paras.append(f"""<p>Film {m0 + 1} runs layout {a['xy'][0]:g} x {a['xy'][1]:g} mm on the CPU plant at seed
20. Its S3e pick tuned at friction 1.0 turns the sphere at 1.0 (score {a['cpu']:.2f}). At 0.4 the same pick loses the
sphere after the commanded forces turn on at 0.8 s. The sphere tilts at 1.0 s, rises out of the pads and has left the
hand by 1.3 s, before the support post is removed at 1.6 s (score {b['cpu']:.2f}). The pick tuned at 0.4 keeps the sphere and turns it (score
{c['cpu']:.2f} here and {c['gpu']:.2f} on the GPU at seeds 20-24). The pick tuned at 1.0 touches the sphere
{np.degrees(va[6]):.1f} deg below its equator with {fa:.2f} N per pad. The pick tuned at 0.4 touches it
{np.degrees(vb[6]):.1f} deg below with {fb:.2f} N per pad and a spin weight of {c['prm'].get('w_spin_tripod', 1.0):.2f}.
</p>""")
            fm = film(MEDIA / "20261103-film_friction_twist.mp4",
                      f"The twist on the STS3032, layout {a['xy'][0]:g} x {a['xy'][1]:g} mm, seed 20, CPU plant. Left: "
                      "the S3e pick tuned at friction 1.0, run at 1.0. Centre: the same pick at friction 0.4. Right: "
                      "the S3e pick tuned at 0.4, run at 0.4. Under each clip: its score and the pick's GPU score at "
                      "seeds 20-24.")
    return paras[0] + t + "".join(paras[1:3]) + fg + "".join(paras[3:]) + fm


# ----------------------------------------------------------------------------------------------- worst-case score

BAND = "band_k21mu"                                 # hom_grasp_refine.BANDS: the STS3032 at friction 0.4, 0.7, 1.0
FRIC = {"k21mu0p4": "0.4", "k21mu0p7": "0.7", "k21": "1.0"}
WC_LOGS = ROOT / "logs/20261104-hom_worst_case"
BAND_FILMS = MEDIA / "20261104-band_films.json"
BAND_FILM = {"task": "twist", "call": 362}          # layout 110 x 110 mm, as the friction film of Section sts
BAND_STEPS = D / "rows" / BAND / "20261104-queue_steps.json"   # wall clock and peak RSS of each step of WC_LOGS/queue.sh
BAND_CPU = D / "rows" / BAND / "20261104-cpu_check.jsonl"      # WC_LOGS/cpu_check.py: the picks on the CPU plant, seed 20


def band_report():
    """The newest worst-case report (hom_grasp_refine.py report --band), or None."""
    base = (R.OUT / ("rows_smoke" if SMOKE else "rows")) / BAND
    ps = sorted(base.glob("*-band_levels.json"))
    return json.loads(ps[-1].read_text()) if ps else None


def band_sets(W):
    """The three pick sets: (source, pick, label), the worst case first; the friction-1.0 picks are the nominal ones."""
    lo, hi = W["plants"][0], W["plants"][-1]
    return [(BAND, "S3", "Worst case, S3w"), (hi, "S3e", "Tuned at 1.0, S3e"), (lo, "S3e", "Tuned at 0.4, S3e")]


def fig_band(W):
    """Per task: the validated score of the three pick sets at each scene friction, median over the layouts with the
    interquartile range."""
    plt = plt_setup()
    fig, axs = grid4((9.5, 7.4), sharex=True, sharey=True)
    fr = [float(FRIC[p]) for p in W["plants"]]
    for ax, task in zip(axs, TASKS):
        lays = W["tasks"].get(task, {}).get("layouts", [])
        for (src, pick, lab), c, m in zip(band_sets(W), (C3, C2, C1), ("s", "o", "^")):
            ys = [[e["val"][src][pick][p] for e in lays if e["val"][src][pick][p] is not None] for p in W["plants"]]
            if not all(ys):
                continue
            q = np.array([np.percentile(y, (25, 50, 75)) for y in ys])
            ax.plot(fr, q[:, 1], "-" + m, color=c, ms=7, lw=2, label=lab)
            ax.fill_between(fr, q[:, 0], q[:, 2], color=c, alpha=0.12)
        ax.set_title(f"{NAME[task]}, {OBJ[task]}")
        ax.set_xticks(fr)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    labels(axs, "Scene friction", "Score at seeds 20-24")
    axs[1].legend(loc="lower right", frameon=False)
    save(fig, "20261104-band_k21mu.png")


def band_compute():
    """The worst-case job's GPU work (timing lines under rows/band_k21mu/, the queue steps' logs) and the CPU runner's
    time per rollout at friction 0.7 (scripts/hom_compute.py cpu-bench --plant k21mu0p7)."""
    import hom_compute as HC
    lines = {p: [] for p in FRIC}
    for q in sorted((D / "rows" / BAND).rglob("*_timing.jsonl")):
        p = q.parent.name
        if p in lines:
            lines[p] += HC.batch_lines(q)
    allb = [x for ls in lines.values() for x in ls]
    if not allb:
        return None
    steps = {}
    for q in sorted(WC_LOGS.glob("Q_*.log")):     # the queue's logs are not in git: their wall clocks go to BAND_STEPS
        m = re.search(r"peak (\d+) kB, ([\d.]+) s", q.read_text(errors="replace"))
        if m:
            steps[q.stem[2:]] = {"wall_s": float(m.group(2)), "peak_rss_gb": round(int(m.group(1)) / 1e6, 2)}
    if steps:
        BAND_STEPS.write_text(json.dumps(steps, indent=1))
    elif BAND_STEPS.exists():
        steps = json.loads(BAND_STEPS.read_text())
    out = {"all": HC.summarize(allb), "plants": {p: HC.summarize(ls) for p, ls in lines.items() if ls},
           "steps": len(steps), "wall_s": sum(s["wall_s"] for s in steps.values())}
    sess = HC.cpu_sessions(R.read_rows(D / "compute" / "cpu_bench.jsonl"))
    cpu = {t_: HC.cpu_summary(*sess[k]) for t_ in TASKS for k in [("k21mu0p7", t_, "filter", 1)] if k in sess}
    run07 = {t_: [1000 * r["run_s"] / r["worlds"] for r in lines["k21mu0p7"] if r["task"] == t_ and r["worlds"] >= 1024]
             for t_ in TASKS}
    out["cpu07"] = {t_: {"cpu_s": c["wall_s"], "gpu_run_ms": float(np.median(run07[t_])) if run07[t_] else None,
                         "load": c["load_before"][0]} for t_, c in cpu.items()}
    return out


def cpu_gains(lo, hi):
    """The CPU plant with every contact, seed 20: per task the median change of the lower of a pick's scores at 0.4 and
    1.0, S3w less the picks tuned at 1.0, split into the cylinder tasks and the sphere tasks."""
    if not BAND_CPU.exists():
        return [], []
    rows = [r for r in R.read_rows(BAND_CPU) if r["variant"] == "plant"]
    idx = {(r["task"], r["plant"], r["call"], r["set"]): r["cpu"] for r in rows}
    g = {}
    for t_ in TASKS:
        calls = sorted({r["call"] for r in rows if r["task"] == t_})
        if calls:
            low = lambda s: np.array([min(idx[(t_, p, c, s)] for p in (lo, hi)) for c in calls])  # noqa: E731
            g[t_] = float(np.median(low("S3w") - low("S3e10")))
    return ([g[t_] for t_ in g if OBJ[t_].endswith("cylinder")], [g[t_] for t_ in g if OBJ[t_].endswith("sphere")])


def cpu_para(tk, lo, hi):
    """The picks on the CPU plant at seed 20 (BAND_CPU) against the GPU's rollouts of the same seed: the lowest of each
    pick's scores at friction 0.4 and 1.0, S3w less the picks tuned at 1.0, as a paragraph and a table."""
    rows = R.read_rows(BAND_CPU)
    idx = {(r["task"], r["plant"], r["variant"], r["call"], r["set"]): r for r in rows}

    def change(task, var, src):
        calls = sorted({r["call"] for r in rows if r["task"] == task and r["variant"] == var})
        if not calls:
            return None
        low = lambda s: np.array([min(idx[(task, p, var, c, s)][src] for p in (lo, hi)) for c in calls])  # noqa: E731
        return float(np.median(low("S3w") - low("S3e10")))

    def zeros(task, s, src):
        return sum(r[src] < 0.005 for r in rows if (r["task"], r["plant"], r["variant"], r["set"]) == (task, lo, "plant", s))

    ch = {t_: {"gpu": change(t_, "plant", "gpu"), "cpu": change(t_, "plant", "cpu"), "filter": change(t_, "filter", "cpu")}
          for t_ in tk}
    n0 = P.TAB[0]
    agree = [t_ for t_ in tk if abs(ch[t_]["cpu"] - ch[t_]["gpu"]) <= 0.05]
    differ = [t_ for t_ in tk if t_ not in agree]
    nm = lambda t_: f"the {NAME[t_].lower()}"  # noqa: E731
    p = [f"""<p>Every layout's three picks also ran on the CPU plant at seed 20, at friction 0.4 and 1.0: {len(rows)}
rollouts (Table {n0 + 1}). Each was compared with the GPU's rollout of the same pick and seed."""]
    if agree:
        p.append(f"""On {andj(nm(t_) for t_ in agree)} the lowest of a pick's two scores changes under S3w, against the
picks tuned at 1.0, by a median of {andj(f"{ch[t_]['cpu']:+.2f}" for t_ in agree)} on the CPU with every contact. On the
GPU it changes by {andj(f"{ch[t_]['gpu']:+.2f}" for t_ in agree)}.""")
    for t_ in differ:
        p.append(f"""On {nm(t_)} it changes by {ch[t_]['cpu']:+.2f} on the CPU with every contact and by
{ch[t_]['gpu']:+.2f} on the GPU. At 0.4 on the CPU with every contact, S3w scores 0 on {zeros(t_, 'S3w', 'cpu')} of the
30 layouts and the picks tuned at 1.0 on {zeros(t_, 'S3e10', 'cpu')}. On the GPU they score 0 on
{zeros(t_, 'S3w', 'gpu')} and {zeros(t_, 'S3e10', 'gpu')}.""")
        if ch[t_]["filter"] is not None:
            p.append(f"With the GPU's contact filter on the CPU, the change on {nm(t_)} is {ch[t_]['filter']:+.2f}.")
    body = [[NAME[t_], sg(ch[t_]["gpu"]), sg(ch[t_]["cpu"]), sg(ch[t_]["filter"]) if ch[t_]["filter"] is not None else "",
             f"{zeros(t_, 'S3w', 'gpu')}, {zeros(t_, 'S3e10', 'gpu')}", f"{zeros(t_, 'S3w', 'cpu')}, {zeros(t_, 'S3e10', 'cpu')}"]
            for t_ in tk]
    tab = table(["Task", "Lowest, S3w less the 1.0 picks: GPU", "CPU, every contact", "CPU, GPU's contact filter",
                 "Scoring 0 at 0.4 on the GPU: S3w, 1.0 picks", "On the CPU, every contact: S3w, 1.0 picks"], body,
                "The picks at seed 20 on the GPU and on the CPU plant, at friction 0.4 and 1.0, over the 30 layouts per "
                "task. Lowest: the lower of a pick's two scores; the columns give the median paired change of S3w "
                "against the S3e picks tuned at 1.0. The twist also ran on the CPU with the GPU's contact filter.",
                num_from=1)
    return " ".join(p) + "</p>" + tab


def sec_band(W):
    if not W:
        return "<p>The worst-case runs have not finished.</p>"
    tk = [t_ for t_ in TASKS if t_ in W["tasks"]]
    pl = W["plants"]
    lo, hi = pl[0], pl[-1]
    sets = band_sets(W)
    T_ = {t_: W["tasks"][t_] for t_ in tk}
    Ls = {t_: T_[t_]["layouts"] for t_ in tk}
    M = {t_: T_[t_]["median_val"] for t_ in tk}
    dm = lambda t_, k: T_[t_]["diff"][k]["median"]  # noqa: E731
    zero = lambda t_, src, pick, p: sum(e["val"][src][pick][p] < 0.005 for e in Ls[t_])  # noqa: E731
    lsg = lambda xs: andj(f"{x:+.2f} on the {NAME[t_].lower()}" for x, t_ in zip(xs, tk))  # noqa: E731
    lst = lambda xs: andj(f"{x:.2f} on the {NAME[t_].lower()}" for x, t_ in zip(xs, tk))  # noqa: E731
    rng = lambda xs, f="{:d}": (f"{f.format(min(xs))} to {f.format(max(xs))}"  # noqa: E731
                                if f.format(min(xs)) != f.format(max(xs)) else f.format(min(xs)))
    n0, f0, m0 = P.TAB[0], P.FIG[0], P.FILM[0]

    out = [f"""<p>Each layout's lowest validated score over scene friction 0.4, 0.7 and 1.0 is higher under the
worst-case picks S3w than under the picks tuned at 1.0. The median paired change is
{lsg([dm(t_, f'S3|{hi}|min') for t_ in tk])} (Table {n0 + 2}). Against the picks tuned at 0.4 it is
{lsg([dm(t_, f'S3|{lo}|min') for t_ in tk])}. At friction 1.0 the median change against the picks tuned there is
{lsg([dm(t_, f'S3|{hi}|{hi}') for t_ in tk])}.</p>"""]
    zh = [zero(t_, hi, "S3e", lo) for t_ in tk]
    out.append(f"""<p>The S3e picks tuned at friction 1.0 score 0 at friction 0.4 on {rng(zh)} of the 30 layouts per task
(Section SEC[sts]). The bench's friction lies between 0.37, TPU on bare PLA, and 1.0 or more with grip tape. Picks for
that range were therefore searched with a worst-case score. Each candidate runs at seeds 0-2 on the STS3032 at scene
friction 0.4, 0.7 and 1.0, and its score is its lowest three-seed mean:</p>
$$W(x) = \\min_{{\\mu \\in \\{{0.4,\\,0.7,\\,1.0\\}}}}\\; \\frac{{1}}{{3}} \\sum_{{s=0}}^{{2}} J(x;\\, \\mu,\\, s),$$
<p>with $x$ the candidate's grasp, forces and modes, $\\mu$ the scene friction and $J$ a rollout's task score. Ties go
to the larger tie-break value at the friction of the minimum. S1w is the grasp of each layout's grasp set with the
highest $W$ on the three grids. S2w and S3w come from the two searches of Section SEC[method] with $W$ in place of the
mean score: the grasp search starts from S1w's grasp, and the search over the grasp, forces and modes from S2w's. Both
ran on the 30 layouts per task of Section SEC[sts], with {W.get('iters_run', 6)} iterations each. The picks then ran at
seeds 20-24 at each friction, as did the S3e picks tuned at 1.0 and at 0.4 (Table {n0 + 1}, Figure {f0 + 1}).</p>""")
    body = []
    for task in tk:
        for k, (src, pick, lab) in enumerate(sets):
            body.append([NAME[task] if k == 0 else "", lab] + [f2(M[task][f"{src}|{pick}|{p}"]) for p in [*pl, "min"]])
    ta = table(["Task", "Picks", "At 0.4", "At 0.7", "At 1.0", "Lowest of the three"], body,
               "Median validated scores (seeds 20-24) of the three pick sets on the STS3032 at kp 2.1 over the same 30 "
               "layouts per task, at each scene friction, and the median of each layout's lowest score over the three "
               "frictions.")
    out.append(ta)

    up = [T_[t_]["diff"][f"S3|{hi}|min"]["up"] for t_ in tk]
    down = [T_[t_]["diff"][f"S3|{hi}|min"]["down"] for t_ in tk]
    zw, zl = [zero(t_, BAND, "S3", lo) for t_ in tk], [zero(t_, lo, "S3e", lo) for t_ in tk]
    p2 = (f"<p>Against the picks tuned at 1.0, the lowest score rises by more than 0.05 on {rng(up)} of the 30 layouts "
          f"per task and falls by more than 0.05 on at most {max(down)}. At 0.4 the picks tuned at 1.0 score 0 on {rng(zh)} "
          f"layouts per task, S3w on {rng(zw)} and the picks tuned at 0.4 on {rng(zl)}.")
    if "twist" in tk:
        p2 += (f" The twist's picks tuned at 0.4 score a median {M['twist'][f'{lo}|S3e|{lo}']:.2f} at 0.4 and "
               f"{M['twist'][f'{lo}|S3e|{hi}']:.2f} at 1.0. At 1.0 the median change of S3w against them is "
               f"{dm('twist', f'S3|{lo}|{hi}'):+.2f}.")
    out.append(p2 + "</p>")
    body2 = [[NAME[task]] + [sg(dm(task, k)) for k in (f"S3|{hi}|min", f"S3|{lo}|min", f"S3|{hi}|{hi}", f"S3|{hi}|{lo}",
                                                         f"S3|{lo}|{lo}")]
             + [f"{zero(task, hi, 'S3e', lo)}, {zero(task, BAND, 'S3', lo)}, {zero(task, lo, 'S3e', lo)}"] for task in tk]
    tb = table(["Task", "Lowest, less the 1.0 picks", "Lowest, less the 0.4 picks", "At 1.0, less the 1.0 picks",
                "At 0.4, less the 1.0 picks", "At 0.4, less the 0.4 picks", "Scoring 0 at 0.4: 1.0 picks, S3w, 0.4 picks"],
               body2, "S3w against the S3e picks tuned at friction 1.0 (the 1.0 picks) and at 0.4 (the 0.4 picks): median "
               "paired changes over the 30 layouts per task at seeds 20-24. Lowest: each layout's lowest score over the "
               "three frictions. The last column counts the layouts whose validated score at friction 0.4 is 0.",
               num_from=1)
    fg = img(MEDIA / "20261104-band_k21mu.png", "Validated score at seeds 20-24 against the scene friction for the three "
             "pick sets: the median over the 30 layouts per task, with the interquartile range shaded.") \
        if (MEDIA / "20261104-band_k21mu.png").exists() else ""
    out += [tb, fg]

    wc = {t_: {p: sum(min(pl, key=lambda q, e=e: e["val"][BAND]["S3"][q]) == p for e in Ls[t_]) for p in pl} for t_ in tk}
    out.append(f"""<p>Of the 30 worst-case picks per task, the lowest validated score lies at friction 0.4 for
{andj(f"{wc[t_][lo]} on the {NAME[t_].lower()}" for t_ in tk)}. It lies at 1.0 for {andj(str(wc[t_][hi]) for t_ in tk)},
and at 0.7 for {rng([wc[t_][pl[1]] for t_ in tk])}.</p>""")

    rk = {}
    for t_ in tk:
        nom = np.array([e["val"][hi]["S3e"][hi] for e in Ls[t_]])
        low = np.array([e["val"][BAND]["S3"]["min"] for e in Ls[t_]])
        b = int(np.argmax(low))
        rk[t_] = {"rho": R.spearman(nom, low), "best": Ls[t_][b], "top": float(nom.max()),
                  "tied": int((nom >= nom.max() - 1e-9).sum()), "short": float(nom.max() - nom[b])}
    moved = [t_ for t_ in tk if rk[t_]["short"] > 1e-9]
    p5 = (f"<p>The layouts are reordered by the lowest score over the band. Spearman's rho between each layout's lowest "
          f"score under S3w and its score at 1.0 under the picks tuned at 1.0 is {lst([rk[t_]['rho'] for t_ in tk])}.")
    if moved:
        p5 += (f" Ranked by the lowest score, the best layout changes on {andj('the ' + NAME[t_].lower() for t_ in moved)}. "
               f"Under the picks tuned at 1.0, the new best layouts score {rng([rk[t_]['short'] for t_ in moved], '{:.2f}')} "
               f"below the old ones at 1.0.")
    for t_ in tk:
        if t_ not in moved:
            x, y = rk[t_]["best"]["xy"]
            p5 += (f" On the {NAME[t_].lower()} the best layout by the lowest score, ({x:g}, {y:g}), is one of the "
                   f"{rk[t_]['tied']} that score {rk[t_]['top']:.2f} at 1.0.")
    out.append(p5 + "</p>")

    below = sum(e["s2"] < e["s2_start"] - 1e-9 or e["s3"] < e["s3_start"] - 1e-9 for t_ in tk for e in Ls[t_])
    s2 = [T_[t_]["s2_minus_start"]["median"] for t_ in tk]
    s3 = [T_[t_]["s3_minus_start"]["median"] for t_ in tk]
    out.append(f"""<p>At the search seeds 0-2 the grasp search's picks score a median {rng(s2, '{:+.2f}')} above their
starts per task, and the search over the grasp, forces and modes adds {rng(s3, '{:+.2f}')} (Table {n0 + 3}).
{"No pick scores below its start." if not below else f"{below} picks score below their starts."} At seeds 20-24 the
median lowest score over the band is {lst([M[t_][f'{BAND}|S1|min'] for t_ in tk])} for S1w. For S1 chosen at 1.0 it
is {andj(f2(M[t_][f'{hi}|S1|min']) for t_ in tk)}.</p>""")
    body3 = [[NAME[t_], f2(T_[t_]["median_s1"]), f2(T_[t_]["median_s2_start"]), f2(T_[t_]["median_s2"]),
              f2(T_[t_]["median_s3_start"]), f2(T_[t_]["median_s3"])] for t_ in tk]
    tc = table(["Task", "S1w, grids", "Grasp search: start", "S2w", "Full search: start", "S3w"], body3,
               "Median worst-case scores $W$ at the search seeds 0-2 over the 30 layouts per task. Each search scored its "
               "start as its first candidate; S1w's score comes from the grids.", num_from=1)
    out.append(tc)

    if BAND_FILMS.exists():
        info = json.loads(BAND_FILMS.read_text())
        a = info[0]
        act = lambda s: (f"turns the sphere, scoring {s['cpu']:.2f}" if s["held"]  # noqa: E731
                         else "drops the sphere") + f" (GPU {s['gpu']:.2f} at seeds 20-24)"
        out.append(f"""<p>Film {m0 + 1} runs twist layout ({a['xy'][0]:g}, {a['xy'][1]:g}) on the CPU plant at seed
20. At friction 0.4 the S3e pick tuned at 1.0 {act(info[0])}. With the GPU's contact filter, where only the pads and
the rigid spheres under 2 mm meet the object, the worst-case pick S3w {act(info[1])}. With every contact it
{act(info[2])}. At 1.0 with every contact it {act(info[3])}.</p>""")
        out.append(film(MEDIA / f"20261104-film_band_{a['task']}.mp4",
                  f"The {NAME[a['task']].lower()} on the STS3032, layout ({a['xy'][0]:g}, {a['xy'][1]:g}), seed 20, CPU "
                  "plant. " + " ".join(f"{k}: {s['cond']}." for k, s in
                                       zip(("Top left", "Top right", "Bottom left", "Bottom right"), info))
                  + " Under each clip: its score and the pick's GPU score at seeds 20-24."))
    if BAND_CPU.exists():
        out.append(cpu_para(tk, lo, hi))

    C = band_compute()
    if C:
        a_ = C["all"]
        out.append(f"""<p>The worst-case runs took {a_['rollouts']:,} GPU rollouts in {a_['batches']:,} batches, with
{a_['batch_s'] / 3600:.2f} h of batch time in {C['wall_s'] / 3600:.2f} h of queue steps. Per rollout that is
{a_['ms_per_rollout_run']:.0f} ms of run time, {a_['ms_per_rollout_batch']:.0f} ms of batch time and
{1000 * C['wall_s'] / a_['rollouts']:.0f} ms of step time.</p>""")
        c7 = {t_: v for t_, v in C["cpu07"].items() if v["gpu_run_ms"]}
        if c7:
            sp = [1000 * v["cpu_s"] / v["gpu_run_ms"] for v in c7.values()]
            out.append(f"""<p>The STS3032 at friction 0.7 is a plant new to this study. On it the CPU runner takes
{andj(f"{v['cpu_s']:.1f} s on the {NAME[t_].lower()}" for t_, v in c7.items())} per rollout on one core, with the GPU's
contact filter. The GPU's full batches take {andj(f"{v['gpu_run_ms']:.0f} ms" for v in c7.values())} per rollout, so one
GPU equals {min(sp):.0f} to {max(sp):.0f} cores on the rollout loop.</p>""")
    return "".join(out)


def band_film_specs(W):
    """One twist layout at seed 20: its S3e pick tuned at friction 1.0 run at 0.4 (the clip of the friction film), its
    worst-case pick S3w run at 0.4 with the GPU's contact filter and with every contact, and S3w run at 1.0."""
    task, call = BAND_FILM["task"], BAND_FILM["call"]
    e = next(x for x in W["tasks"][task]["layouts"] if x["call"] == call)
    A = report("k21")
    ea = next(x for x in A["tasks"][task]["layouts"] if x["call"] == call)
    d6 = next(x for x in R.layouts(task, "k21") if x["call"] == call)["d"]
    va, pa = pick_of(task, "k21", call, ea["s3e_pick"])
    base = {"set": "band", "task": task, "call": call, "xy": e["xy"], "d6": d6, "seed": R.VAL_SEEDS[0]}
    return [dict(base, plant="k21mu0p4", pick="S3e", v=va, prm=pa, gpu=e["val"]["k21"]["S3e"]["k21mu0p4"],
                 cond="S3e pick tuned at friction 1.0, run at 0.4", top="S3e tuned at 1.0, run at 0.4",
                 out=str(MEDIA / f"20261103-clip_friction_{task}_10at04.mp4")),
            dict(base, plant="k21mu0p4", pick="S3w", v=e["v"]["S3"], prm=e["prm3"], gpu=e["val"][BAND]["S3"]["k21mu0p4"],
                 filter=True, cond="worst-case pick S3w, run at 0.4 with the GPU's contact filter",
                 top="S3w at 0.4, GPU's contact filter", out=str(MEDIA / f"20261104-clip_band_{task}_w04f.mp4")),
            dict(base, plant="k21mu0p4", pick="S3w", v=e["v"]["S3"], prm=e["prm3"], gpu=e["val"][BAND]["S3"]["k21mu0p4"],
                 cond="worst-case pick S3w, run at 0.4 with every contact", top="S3w at 0.4, every contact",
                 out=str(MEDIA / f"20261104-clip_band_{task}_w04.mp4")),
            dict(base, plant="k21", pick="S3w", v=e["v"]["S3"], prm=e["prm3"], gpu=e["val"][BAND]["S3"]["k21"],
                 cond="worst-case pick S3w, run at 1.0 with every contact", top="S3w at 1.0, every contact",
                 out=str(MEDIA / f"20261104-clip_band_{task}_w10.mp4"))]


def cmd_band(what):
    """band-figs: the worst-case figure; band-films: its film (CPU clips through resguard, after the GPU queue)."""
    W = band_report()
    if not W:
        raise SystemExit("no worst-case report: hom_grasp_refine.py report --band")
    if what == "band-figs":
        fig_band(W)
        return 0
    specs = band_film_specs(W)
    for s in specs:
        if Path(s["out"]).with_suffix(".json").exists():
            continue
        hp, scenes, tag = R.PLANTS[s["plant"]]
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
    HP.compose([s["out"] for s in specs], MEDIA / f"20261104-film_band_{specs[0]['task']}.mp4", cols=2)
    res = [dict(s, **json.loads(Path(s["out"]).with_suffix(".json").read_text())) for s in specs]
    BAND_FILMS.write_text(json.dumps(res, indent=1))
    print("wrote", BAND_FILMS.relative_to(ROOT))
    return 0


def sec_films():
    if not FILMS.exists():
        return "<p>The films are made after the GPU queue (<code>hom_score_refine_page.py films</code>).</p>"
    info = [s for s in json.loads(FILMS.read_text()) if s.get("set", "levels") == "levels"]
    agree = sum(abs(s["cpu"] - s["gpu"]) <= 0.05 for s in info)
    tw = {s["pick"]: s for s in info if s["task"] == "twist"}
    m0 = P.FILM[0]
    out = [f"""<p>Films {m0 + 1} to {m0 + 4} run each task's best layout under S3e on the SCS0009 at seed 20, with its
S1, S2e and S3e picks side by side. They run on the CPU plant, where every contact meets the object; on the GPU only
the pads and the rigid spheres under 2 mm do. On {agree} of the {len(info)} clips the CPU score lies within 0.05 of
the pick's GPU score at seeds 20-24. The twist's S2e and S3e picks score {tw['S2e']['cpu']:.2f} and
{tw['S3e']['cpu']:.2f} on the CPU at seed 20, against """ + (f"{tw['S2e']['gpu']:.2f} for both"
                                                       if f"{tw['S2e']['gpu']:.2f}" == f"{tw['S3e']['gpu']:.2f}"
                                                       else f"{tw['S2e']['gpu']:.2f} and {tw['S3e']['gpu']:.2f}")
           + """ on the GPU at seeds 20-24.</p>""" if {"S2e", "S3e"} <= set(tw) else ""]
    for task in TASKS:
        fs = [s for s in info if s["task"] == task]
        if not fs:
            continue
        cap = (f"{NAME[task]} on the {OBJ[task]}, layout {fs[0]['xy'][0]:g} x {fs[0]['xy'][1]:g} mm on the SCS0009: "
               "the S1, S2e and S3e picks from left to right, CPU plant, seed 20. Under each clip: its score and the "
               "pick's GPU score at seeds 20-24.")
        out.append(film(MEDIA / f"20261103-film_{task}.mp4", cap))
    return "".join(out)


def newest(pattern):
    ps = sorted((D / "compute").glob(pattern))
    return json.loads(ps[-1].read_text()) if ps else None


def sec_compute():
    """The queue's compute per plant, the GPU's time per rollout against the CPU runner's, and the speedups
    (scripts/hom_compute.py ledger, gpu-bench, cpu-bench and report)."""
    Lg, Tp = newest("*-ledger.json"), newest("*-throughput.json")
    if not (Lg and Tp and Tp.get("plants")):
        return "<p>The compute ledger is written by <code>scripts/hom_compute.py</code> (not run yet).</p>"
    base = str((D / "rows").relative_to(ROOT))
    plants = [pl for pl in ("sv25", "k21", "k21mu0p4") if pl in Tp["plants"]]
    per = {}
    for pl in plants:
        ds = [v for k, v in Lg["per_dir"].items() if k.startswith(f"{base}/{pl}/")]
        st_ = [v for k, v in Lg["queue_steps"].items() if k.startswith(f"{pl}_") and v.get("wall_s")]
        per[pl] = {"rollouts": sum(d["rollouts"] for d in ds), "batches": sum(d["batches"] for d in ds),
                   "batch_s": sum(d["batch_s"] for d in ds), "wall_s": sum(s["wall_s"] for s in st_)}
    rows_a = [[PLANT_NAME[pl], f"{per[pl]['rollouts']:,}", f"{per[pl]['batches']:,}", f"{per[pl]['batch_s'] / 3600:.2f}",
               f"{per[pl]['wall_s'] / 3600:.2f}", f"{1000 * per[pl]['batch_s'] / per[pl]['rollouts']:.1f}",
               f"{1000 * per[pl]['wall_s'] / per[pl]['rollouts']:.1f}"] for pl in plants if per[pl]["rollouts"]]
    tot = {k: sum(per[pl][k] for pl in plants) for k in ("rollouts", "batches", "batch_s", "wall_s")}
    rows_a.append(["All three", f"{tot['rollouts']:,}", f"{tot['batches']:,}", f"{tot['batch_s'] / 3600:.2f}",
                   f"{tot['wall_s'] / 3600:.2f}", f"{1000 * tot['batch_s'] / tot['rollouts']:.1f}",
                   f"{1000 * tot['wall_s'] / tot['rollouts']:.1f}"])
    n0, f0 = P.TAB[0], P.FIG[0]
    ta = table(["Plant", "Rollouts", "GPU batches", "Batch time (h)", "Queue steps (h)", "Batch time per rollout (ms)",
                "Step time per rollout (ms)"], rows_a,
               "The night's GPU work per plant, from the timing line of every batch and the wall clock of every queue "
               "step (searches, validation and the cross-friction runs).")
    body, sp_run, sp_step = [], [], []
    for pl in plants:
        for task in TASKS:
            q = Tp["plants"][pl]["queue"].get(task) or {}
            q = q.get("grasp") or q.get("full") or {}
            c = Tp["plants"][pl]["cpu"].get(task) or {}
            sp = (Tp["plants"][pl]["speedup"].get(task) or {}).get("filter_p1") or {}
            if sp.get("vs_gpu_run"):
                sp_run.append(sp["vs_gpu_run"])
            if sp.get("vs_gpu_step"):
                sp_step.append(sp["vs_gpu_step"])
            body.append([PLANT_NAME[pl], NAME[task], f2(q.get("ms_per_rollout_run_full_batches")),
                         f2(q.get("ms_per_rollout_batch")), f2(q.get("step_ms_per_rollout")),
                         f2(c["plant_p1"]["wall_s"]) if c.get("plant_p1") else "n/a",
                         f2(c["filter_p1"]["wall_s"]) if c.get("filter_p1") else "n/a",
                         f"{sp['vs_gpu_run']:.0f} / {sp['vs_gpu_batch']:.0f} / {sp['vs_gpu_step']:.0f}"
                         if sp.get("vs_gpu_step") else "n/a"])
    tb = table(["Plant", "Task", "GPU run (ms)", "GPU batch (ms)", "Queue step (ms)", "CPU, every contact (s)",
                "CPU, GPU's filter (s)", "Speedup: run / batch / step"], body,
               "Time per rollout. GPU columns: the S2 search's full batches (2,048 sphere or 1,536 cylinder worlds) and "
               "its whole queue steps. CPU columns: the CPU runner on one core with the scene's solver settings, with "
               "every contact as the CPU checks run it and with the GPU's contact filter, on the best S1 layouts at "
               "seed 0. The speedup uses the CPU with the GPU's filter. With the GPU's solver settings (20 iterations) "
               "the CPU dropped every twist and roll sphere, so those settings are no CPU baseline.", num_from=2)
    fg = ""
    if (MEDIA / "20261103-throughput.png").exists():
        fg = img(MEDIA / "20261103-throughput.png", "GPU time per rollout on the SCS0009 against the worlds per batch, "
                 "as run and batch time (<code>scripts/hom_compute.py gpu-bench</code>). The horizontal lines give the "
                 "CPU runner's time per rollout on one core; the two lie on top of each other.")
    gb = Tp["gpu_bench"].get("twist") or []
    gb = sorted(gb, key=lambda r: r["worlds"])
    sv = Tp["plants"]["sv25"]["cpu"]
    slow = [sv[t_]["plant_p4"]["per_core_slowdown"] for t_ in TASKS if sv.get(t_, {}).get("plant_p4", {}).get("per_core_slowdown")]
    prev = [(sv[t_]["plant_p1"]["previous_sessions"][0]["wall_s"], sv[t_]["plant_p1"]["wall_s"]) for t_ in TASKS
            if sv.get(t_, {}).get("plant_p1", {}).get("previous_sessions")]
    load = [sv[t_]["plant_p1"]["load_before"][0] for t_ in TASKS if sv.get(t_, {}).get("plant_p1")]
    paras = [f"""<p>Every GPU batch writes its own timing line beside its rows, and <code>scripts/hom_compute.py</code>
sums them into a ledger. The night's queue ran {tot['rollouts']:,} rollouts in {tot['batches']:,} GPU batches, with
{tot['batch_s'] / 3600:.1f} h of batch time in {tot['wall_s'] / 3600:.1f} h of queue steps (Table {n0 + 1}). All HOM GPU
runs since 2026-10-29 come to {Lg['total_hom_gpu']['rollouts']:,} rollouts in {Lg['total_hom_gpu']['batches']:,} batches
and {Lg['total_hom_gpu']['batch_hours']:.1f} h.</p>"""]
    if sp_run and sp_step:
        paras.append(f"""<p>One GPU equals {min(sp_run):.0f} to {max(sp_run):.0f} CPU cores on the rollout loop and
{min(sp_step):.0f} to {max(sp_step):.0f} cores over whole queue steps (Table {n0 + 2}). The CPU runner with every
contact takes about as long as with the GPU's contact filter.</p>""")
    if len(gb) >= 2:
        r1k = next((r for r in gb if r["worlds"] == 1024), None)
        paras.append(f"""<p>On the GPU the time per rollout falls with the batch's size: on the twist from
{gb[0]['run_ms']:.0f} ms at {gb[0]['worlds']} worlds to {gb[-1]['run_ms']:.0f} ms at {gb[-1]['worlds']:,} (Figure
{f0 + 1})."""
                     + (f" A batch of 1,024 worlds takes {r1k['run_ms'] / gb[-1]['run_ms']:.2f} times the full batch's "
                        f"time per rollout." if r1k else "") + "</p>")
    if slow:
        paras.append(f"""<p>With four CPU processes at once, a rollout takes {min(slow):.2f} to {max(slow):.2f} times its
time alone, so four cores run {4 / max(slow):.1f} to {4 / min(slow):.1f} times as many rollouts as one. The CPU times
were measured on 11-04 on the otherwise idle workstation (load {med(load):.1f} on 20 cores)."""
                     + (f" Those of 11-03, beside the running GPU queue at a load of about 5, are within "
                        f"{100 * max(abs(a - b) / b for a, b in prev):.0f} % of them." if prev else "") + "</p>")
    return paras[0] + ta + "".join(paras[1:]) + tb + fg


def fig_throughput():
    """GPU ms per rollout against worlds per batch, per task, with the CPU's per-rollout times as lines."""
    Tp = newest("*-throughput.json")
    if not Tp or not any(Tp["gpu_bench"].values()):
        return
    plt = plt_setup()
    fig, axs = grid4((9.5, 7.0), sharey=True)
    for ax, task in zip(axs, TASKS):
        g = sorted(Tp["gpu_bench"].get(task, []), key=lambda r: r["worlds"])
        if g:
            ax.loglog([r["worlds"] for r in g], [r["run_ms"] for r in g], "-o", color=C2, label="GPU run")
            ax.loglog([r["worlds"] for r in g], [r["batch_ms"] for r in g], "--s", color=C3, label="GPU batch")
        for key, ls, lab in (("plant_p1", ":", "CPU, every contact"), ("filter_p1", "-.", "CPU, GPU's filter")):
            c = (Tp["cpu"].get(task) or {}).get(key)
            if c:
                ax.axhline(1000 * c["wall_s"], color="k", ls=ls, label=lab)
        ax.set_title(NAME[task])
        ax.grid(alpha=0.3, which="both")
    labels(axs, "Worlds per batch", "ms per rollout")
    axs[0].legend(frameon=False)
    save(fig, "20261103-throughput.png")


def sec_limits():
    return """<ul>
<li>The layouts are the symmetric tripods of the 5 mm lattice, at one object size per task. The six-coordinate layouts
and other object sizes were not refined.</li>
<li>Each pick is chosen on three seeds and validated on five. The GPU evaluator's run-to-run spread is about 0.01 per
rollout. On marginal twist candidates, whose sphere is held at some seeds and dropped at others, a repeat of an
identical batch changed a three-seed mean by up to 0.22 (median 0.001).</li>
<li>The friction band has three points, 0.4, 0.7 and 1.0, on one plant, the STS3032 at kp 2.1. The UR5's 0.37 for
TPU on bare PLA lies below it.</li>
<li>The twist's score saturates at 1.0, a 90 deg turn on three pads. Among saturated candidates the searches ranked by
the three-pad angle, and Section SEC[sv25] reports that angle.</li>
<li>S3 tunes the forces and modes per layout. A controller tuned once for all layouts was not searched.</li>
<li>The rows record no contact normals or forces. The friction check of Section SEC[sts] uses the planned contacts of
the pose toy and static balance.</li>
<li>On the GPU only the pads and the rigid spheres under 2 mm meet the object. The films run on the CPU plant with every
contact, where some refined twist picks score lower than on the GPU (Section SEC[films]). On the 40 mm sphere at
friction 0.4 the CPU plant with every contact also scores the worst-case picks and the picks tuned at 1.0 differently
from the GPU (Section SEC[band]).</li>
</ul>"""


def band_future():
    """The worst-case runs' items for Future work: a two-friction band, and the band with every contact on the sphere."""
    W = band_report()
    if not W:
        return ""
    pl = W["plants"]
    n07 = [sum(min(pl, key=lambda q, e=e: e["val"][BAND]["S3"][q]) == pl[1] for e in W["tasks"][t_]["layouts"])
           for t_ in TASKS if t_ in W["tasks"]]
    return f"""<li>A two-friction band: the worst-case search at friction 0.4 and 1.0 alone (an entry of <code>BANDS</code>
in <code>scripts/hom_grasp_refine.py</code>), at two-thirds of the rollouts. Friction 0.7 gave the lowest validated
score of {min(n07)} to {max(n07)} of the 30 worst-case picks per task. Two frictions suffice if their picks' median
lowest score over the three frictions lies within 0.02 of S3w's on every task.</li>
<li>The band on the 40 mm sphere with every contact meeting the object: the twist's and the roll's worst-case search
with the rigid geoms in the GPU's contact filter (<code>scripts/hom_gpu.py</code>), its picks then run on the CPU plant
at seeds 20-24 (<code>logs/20261104-hom_worst_case/cpu_check.py</code>). The GPU's sphere results at friction 0.4 stand
if the new picks' lowest score on the CPU lies within 0.05 of S3w's on the GPU.</li>"""


def sec_future(Js):
    drops = ""
    if FRICTION_CHECK.exists():
        F = json.loads(FRICTION_CHECK.read_text())["tasks"]["twist"]["S1_1.0"]
        drops = f" If the drops stay near {F['drops']['0.4']} of {F['rollouts']['0.4']} at every force, the force is not the cause."
    return f"""<ul>
<li>Find what drops the friction-1.0 grasps at friction 0.4. Log each pad's contact normal and normal force from the
squeeze to the drop in the GPU rows (<code>scripts/hom_gpu.py</code>). Rerun the twist's friction-1.0 S1 grasps at 0.4
with the tripod force at 1.5, 2.5 and 4 N (a world's <code>prm</code>).{drops}</li>
{band_future()}
<li>One controller for all layouts (job <code>hand-20261104-hom_shared_controller</code>): one set of forces and modes
tuned on the 60 SCS0009 layouts of a task, then the grid rescored. Per-layout tuning is needed if the shared set's
median gain over S1 on the 30 best layouts is below half of S3e's.</li>
<li>The grasp in the evolutionary search's genome (job <code>hand-20261104-hom_grasp_genome_ea</code>): a child
inherits its parent's refined grasp. It is worth its rollouts if it beats the best-of-16 search of
<code>scripts/hom_ea.py</code> at equal rollouts by more than the GPU's spread of 0.01.</li>
<li>The S3e picks on the CPU plant with every contact (<code>scripts/hom_top_cpu_check.py</code>), then on the hand. A
pick transfers to the CPU plant if its CPU score at seeds 20-24 lies within 0.1 of its GPU score.</li>
</ul>"""


def sec_files():
    return """<p>Files: the search <code>scripts/hom_grasp_refine.py</code>; this page's builder
<code>scripts/hom_score_refine_page.py</code>; the compute ledger and benchmarks <code>scripts/hom_compute.py</code>
(<code>compute/</code> beside this page); the runbook, queue and rows in
<code>docs/experiments/20261103-hom_score_refine/</code> (README.md); the friction check
<code>media/20261103-friction_check.json</code>; the logs in <code>logs/20261103-hom_score_refine/</code>. The worst-case
runs: <code>hom_grasp_refine.py --band</code>, the queue <code>queue_band.sh</code> beside this page, the rows and report
in <code>rows/band_k21mu/</code>, the CPU check <code>logs/20261104-hom_worst_case/cpu_check.py</code> and the logs in
<code>logs/20261104-hom_worst_case/</code>.</p>"""


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
    tk = [t_ for t_ in TASKS if t_ in J["tasks"]]
    L = {t_: J["tasks"][t_]["layouts"] for t_ in tk}
    a = {t_: paired(L[t_], "s1_val", "s3e_val", "top")["median"] for t_ in tk}
    g = {t_: paired(L[t_], "s1_val", "s2e_val", "top")["median"] for t_ in tk}
    rho = [J["tasks"][t_]["rho_s1_val_s3e_val"] for t_ in tk]
    moved = sum(best_layout(L[t_], "s1_val", t_)["call"] != best_layout(L[t_], "s3e_val", t_)["call"] for t_ in tk)
    lo, hi = min(a, key=a.get), max(a, key=a.get)
    pinch = med([e["prm3"]["f_pinch"] for e in L["press"] if e.get("prm3")]) if "press" in L else float("nan")
    lede = [f"Each SR2 layout's grasp, HOM's commanded forces and its contact modes were refined together on the HOM "
            f"score. On the SCS0009 plant this raised the validated score of each task's 30 best layouts. The median gain over "
            f"the best of 16 sampled grasps ranges from {a[lo]:.2f} on the {NAME[lo].lower()} to {a[hi]:.2f} on the "
            f"{NAME[hi].lower()}. Refining "
            f"the grasp alone gave {min(g.values()):.2f} to {max(g.values()):.2f}."]
    if hi == "press":
        lede.append(f"On the press the pinch force was lowered from 3 N to a median {pinch:.1f} N.")
    lede.append(f"The refined scores reorder the layouts. Over 60 layouts per task, Spearman's rho between the sampled "
                f"and the refined score is {min(rho):.2f} to {max(rho):.2f}. The best layout moves on {WORDS[moved]} of "
                f"the four tasks.")
    summary = (f"Refining the grasp, HOM's forces and its contact modes per SR2 layout raised the median validated "
               f"score of the 30 best SCS0009 layouts by {a[lo]:.2f} ({NAME[lo].lower()}) to {a[hi]:.2f} "
               f"({NAME[hi].lower()}) over the best of 16 grasps and reordered the layouts (rho {min(rho):.2f} to "
               f"{max(rho):.2f}).")
    A, B = Js.get("k21"), Js.get("k21mu0p4")
    if A and B:
        tb = [t_ for t_ in TASKS if t_ in B["tasks"]]
        zero = sum(med(vals(B["tasks"][t_]["layouts"], "s3e_val_from_k21")) < 0.005 for t_ in tb)
        gn = [paired(B["tasks"][t_]["layouts"], "s3e_val_from_k21", "s3e_val")["median"] for t_ in tb]
        lede.append(f"On the STS3032, the picks tuned at friction 1.0 score a median 0 at friction 0.4 on {WORDS[zero]} of "
                    f"the four tasks. Retuning at 0.4 gains a median {min(gn):.2f} to {max(gn):.2f} per task.")
        summary += (f" On the STS3032, picks tuned at friction 1.0 score 0 at friction 0.4 on {WORDS[zero]} tasks; retuning "
                    f"at 0.4 gains {min(gn):.2f} to {max(gn):.2f}.")
    W = band_report()
    if W:
        tw = [t_ for t_ in TASKS if t_ in W["tasks"]]
        lo, hi = W["plants"][0], W["plants"][-1]
        dw = [W["tasks"][t_]["diff"][f"S3|{hi}|min"]["median"] for t_ in tw]
        dl = [W["tasks"][t_]["diff"][f"S3|{lo}|min"]["median"] for t_ in tw]
        lede.append(f"Picks searched on the lowest score over friction 0.4, 0.7 and 1.0 raise it by a median "
                    f"{min(dw):.2f} to {max(dw):.2f} per task above the picks tuned at 1.0. The picks tuned at 0.4 come "
                    f"within {max(dl):.2f} of them.")
        cyl, sph = cpu_gains(lo, hi)
        if cyl and sph:
            lede.append(f"On the CPU plant with every contact the gain stays on the two cylinder tasks and falls to "
                        f"{max(sph):.2f} or less on the two sphere tasks.")
        summary += (f" Picks searched on the lowest score over friction 0.4-1.0 raise it by {min(dw):.2f} to "
                    f"{max(dw):.2f} over the 1.0 picks.")
    return " ".join(lede), summary


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
        ("band", "Worst-case score over three frictions on the STS3032", sec_band(band_report())),
        ("films", "Films of the picks", sec_films()),
        ("compute", "Compute and throughput", sec_compute()),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future(Js)),
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
              "(<code>SCALE_HAND=cad_v2</code>); HOM on MuJoCo-Warp, the films on the CPU plant. Follows the plant and "
              "modulus page [3]. Builder: "
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
    if a and a[0] in ("band-figs", "band-films"):
        return cmd_band(a[0])
    return cmd_page()


if __name__ == "__main__":
    sys.exit(main())

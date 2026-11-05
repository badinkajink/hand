#!/usr/bin/env python3
"""Builds docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus.html: how the HOM design landscapes
of the SR2 hand change with the finger servo, the pad modulus and the scene friction (2026-11-02 and 03). Owner, 11-02:
"characterize the change in landscape between the idealized 4nmrad plant, the current scs0009 calibration, and the ideal
sts3032 calibration ... there seems to be no transfer between layouts, controls optimized for one plant to another. is that
really true?"; "seems like we need to queue up modulus studies across a range as well"; "run the friction and moduli sweep
for the sts3032".

Data: the grids rows/grid2dff_v4b* (runs and tags in logs/20261102-hom_servo_plant/README.md); their comparisons
rows/20261102-plant_transfer.json, -modulus_transfer.json, -plant_modulus_factorial.json, -sts_friction_transfer.json and
-sts_modulus_transfer.json (scripts/hom/hom_plant_transfer.py); rows/cpu_best/20261102-plant_compare.json
(scripts/hom/hom_plant_compare.py) and top_*_sv25.jsonl (hom_top_cpu_check.py); rows/ea_live/summary.json and
rows/ea_live_sv25/summary.json (scripts/hom/hom_ea.py report). The design page's helpers and template
(scripts/hom/fp_review_page.py) draw the tables and figures.

    .venv/bin/python scripts/hom/hom_plant_page.py maps    # rows/20261102-plant_landscapes.json and the map figures
    .venv/bin/python scripts/hom/hom_plant_page.py films   # CPU clips through resguard, the films, rows/20261102-plant_films.json
    .venv/bin/python scripts/hom/hom_plant_page.py         # the page
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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_figures as FR  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

fill, table, img, film, esc = P.fill, P.table, P.img, P.film, P.esc
D = ROOT / "docs/experiments/20261029-foundational_pose_review"
MEDIA, ROWS = D / "media", D / "rows"
OUT = D / "20261102-hom_plant_modulus.html"
TRENDS = "20261030-hom_layout_trends.html"
DESIGN = "20261029-foundational_pose_hand_design.html"
V3TIP = "../20261030-v3_tip_contact_model/20261030-v3_fingertip_contact_models.html"
LAND = ROWS / "20261102-plant_landscapes.json"
FILMS = ROWS / "20261102-plant_films.json"
REFS = [P.REFS[0], P.REFS[1]]                          # [1] foundational pose, [2] HOM
TASKS = ("twist", "roll", "pinpush", "press")
TABNO = {}                                             # table numbers of the sections, for the later sections' references
DEFAULT = {"twist": 40, "roll": 40, "pinpush": 25, "press": 25}
NAME = {"twist": "Twist", "roll": "Roll", "pinpush": "Pin-push", "press": "Press"}
OBJ = {"twist": "40 mm sphere", "roll": "40 mm sphere", "pinpush": "25 mm cylinder", "press": "25 mm cylinder"}
SVP = "logs/20261102-hom_servo_plant"
STS_PLANT = "kp2.1_kv0_fr0.515_fl0_dp0.042"
# key: (label, rows directory, (HOS_PLANT, HOS_SCENES, HOM_ROWS_TAG) of its runs; None: the defaults of 10-30)
GRIDS = {
    "work10": ("work plant, 10 MPa", "grid2dff_v4b", None),
    "work25": ("work plant, 2.5 MPa", "grid2dff_v4b_w25", ("work", f"{SVP}/scenes", "_w25")),
    "scs125": ("SCS0009, 1.25 MPa", "grid2dff_v4b_svE1p25", ("servo", f"{SVP}/scenes_E1.25", "_svE1p25")),
    "scs25": ("SCS0009, 2.5 MPa", "grid2dff_v4b_sv25", ("servo", f"{SVP}/scenes", "_sv25")),
    "scs5": ("SCS0009, 5 MPa", "grid2dff_v4b_svE5", ("servo", f"{SVP}/scenes_E5", "_svE5")),
    "scs10": ("SCS0009, 10 MPa", "grid2dff_v4b_svE10", ("servo", f"{SVP}/scenes_E10", "_svE10")),
    "sts125": ("STS3032, 1.25 MPa", "grid2dff_v4b_k21E1p25", (STS_PLANT, f"{SVP}/scenes_E1.25", "_k21E1p25")),
    "sts25": ("STS3032, 2.5 MPa", "grid2dff_v4b_k21", (STS_PLANT, f"{SVP}/scenes", "_k21")),
    "sts5": ("STS3032, 5 MPa", "grid2dff_v4b_k21E5", (STS_PLANT, f"{SVP}/scenes_E5", "_k21E5")),
    "sts10": ("STS3032, 10 MPa", "grid2dff_v4b_k21E10", (STS_PLANT, f"{SVP}/scenes_E10", "_k21E10")),
    "mu04": ("STS3032, friction 0.4", "grid2dff_v4b_k21mu0p4", (STS_PLANT, f"{SVP}/scenes_mu0.4", "_k21mu0p4")),
    "mu07": ("STS3032, friction 0.7", "grid2dff_v4b_k21mu0p7", (STS_PLANT, f"{SVP}/scenes_mu0.7", "_k21mu0p7")),
    "mu24": ("STS3032, friction 2.4", "grid2dff_v4b_k21mu2p4", (STS_PLANT, f"{SVP}/scenes_mu2.4", "_k21mu2p4")),
}
MAPS = {"plants": ("work10", "scs25", "sts25"), "friction": ("mu04", "mu07", "sts25", "mu24"),
        "modulus": ("scs125", "scs10", "sts125", "sts10")}


def load(name):
    p = ROWS / name
    return json.loads(p.read_text()) if p.exists() else None


def call_of(x, y):
    return int((x - 40) // 5 * 25 + (y - 50) // 5)    # the lattice, x_sep major (hom_plant_transfer.xy)


# ----------------------------------------------------------------------------------------------- maps

def cmd_maps():
    """Each grid's best-of-16 score and best grasp per layout, cached; then the map figures."""
    import hom_plant_transfer as TR
    out = {}
    for k, (_label, src, _env) in GRIDS.items():
        out[k] = {}
        for task in TASKS:
            g = TR.grasp_means(task, ROWS / src / f"{task}_d{DEFAULT[task]}.jsonl")
            out[k][task] = {"best": {str(c): round(max(v.values()), 4) for c, v in g.items()},
                            "grasp": {str(c): max(v, key=v.get) for c, v in g.items()}}
        print(k, {t: len(out[k][t]["best"]) for t in TASKS}, flush=True)
    LAND.write_text(json.dumps(out))
    print("wrote", LAND.relative_to(ROOT))
    for name, keys in MAPS.items():
        fig_maps(out, keys, f"20261102-plant_maps_{name}.png",
                 {"sts25": "STS3032, friction 1.0"} if name == "friction" else {})
    return 0


def fig_maps(Lc, keys, name, labels):
    """Best-of-16 maps over (x_sep, y_sep), a row per grid and a column per task; grey where no grasp was solved, the best
    layout starred."""
    plt = FR.plt_setup()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14})
    fig, axs = plt.subplots(len(keys), 4, figsize=(11, 2.45 * len(keys) + 0.6), constrained_layout=True, sharex=True,
                            sharey=True)
    ext = [37.5, 162.5, 47.5, 172.5]
    for i, k in enumerate(keys):
        for j, task in enumerate(TASKS):
            ax = axs[i, j]
            best = {int(c): v for c, v in Lc[k][task]["best"].items()}
            Z = np.full((25, 25), np.nan)
            for c, v in best.items():
                Z[c % 25, c // 25] = v
            ax.imshow(np.where(np.isfinite(Z), np.nan, 0.0), origin="lower", extent=ext, cmap="Greys", vmin=-1, vmax=1,
                      aspect="equal")
            im = ax.imshow(Z, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1, aspect="equal")
            c = max(best, key=best.get)
            ax.plot(40 + 5 * (c // 25), 50 + 5 * (c % 25), "r*", ms=14, mec="k")
            ax.set_xticks([50, 100, 150])
            ax.set_yticks([60, 100, 140])
            if i == 0:
                ax.set_title(f"{NAME[task]}\n{OBJ[task]}")
            if i == len(keys) - 1:
                ax.set_xlabel("$x_{sep}$ (mm)")
            if j == 0:
                ax.set_ylabel(labels.get(k, GRIDS[k][0]).replace(", ", ",\n") + "\n$y_{sep}$ (mm)")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="Best-of-16 score")
    FR.save(fig, name)


# ---------------------------------------------------------------------------------------------- films

def film_specs():
    """The clips of the two films: the three-servo design of the twist and the press (one layout and grasp, rows of
    20261102-plant_transfer.json) on each servo; the twist's best layout and grasp at friction 1.0 at friction 0.4, 1.0
    and 2.4."""
    Lc = json.loads(LAND.read_text())
    tp, tm = load("20261102-plant_transfer.json"), load("20261102-sts_friction_transfer.json")
    specs = {"plants": [], "friction": []}
    three = "|work plant, 10 MPa|SCS0009, 2.5 MPa|STS3032 tuned, 2.5 MPa"
    for task in ("twist", "press"):
        r = tp["robust"][f"{task}_d{DEFAULT[task]}{three}"]["one_layout_and_grasp"]
        c = call_of(*r["xy"])
        for k in ("work10", "scs25", "sts25"):
            specs["plants"].append(dict(grid=k, task=task, call=c, xy=r["xy"], grasp=int(r["grasp"]),
                                        top=f"{NAME[task]}, {GRIDS[k][0]}"))
    b = tm["pairs"]["twist_d40|STS3032, friction 1|STS3032, friction 2.4"]["best_a"]
    g = int(Lc["sts25"]["twist"]["grasp"][str(b["call"])])
    for k in ("mu04", "sts25", "mu24"):
        specs["friction"].append(dict(grid=k, task="twist", call=int(b["call"]), xy=b["xy"], grasp=g,
                                      top=GRIDS[k][0].replace("2.5 MPa", "friction 1.0")))
    for name, cl in specs.items():
        for s in cl:
            s["gpu"] = gpu_mean(s["grid"], s["task"], s["call"], s["grasp"])
            s["out"] = str(MEDIA / f"20261102-clip_{name}_{s['task']}_{s['grid']}.mp4")
    return specs


def gpu_mean(k, task, call, grasp):
    import hom_plant_transfer as TR
    g = TR.grasp_means(task, ROWS / GRIDS[k][1] / f"{task}_d{DEFAULT[task]}.jsonl")
    return round(g[call][grasp], 3)


def cmd_films():
    specs = film_specs()
    for name, cl in specs.items():
        for s in cl:
            if Path(s["out"]).with_suffix(".json").exists():
                continue
            env = {k: v for k, v in os.environ.items() if k not in ("HOS_PLANT", "HOS_SCENES", "HOM_ROWS_TAG")}
            e = GRIDS[s["grid"]][2]
            if e:
                env.update(HOS_PLANT=e[0], HOS_SCENES=e[1], HOM_ROWS_TAG=e[2])
            env.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", NUMPY_MADVISE_HUGEPAGE="0")
            cmd = [str(Path.home() / ".claude/bin/resguard.sh"), "run", "--mem", "3G", "--cpu", "200", "--",
                   str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/hom/hom_plant_page.py"), "clip", json.dumps(s)]
            while True:
                rc = subprocess.call(cmd, env=env, cwd=ROOT)
                if rc != 3:
                    break
                print(time.strftime("%T"), "refused by resguard; retry in 120 s", flush=True)
                time.sleep(120)
            if rc:
                raise SystemExit(f"clip {s['out']} failed (exit {rc})")
        compose([s["out"] for s in cl], MEDIA / f"20261102-film_{name}.mp4", cols=3)
    res = {name: [dict(s, **json.loads(Path(s["out"]).with_suffix(".json").read_text())) for s in cl]
           for name, cl in specs.items()}
    FILMS.write_text(json.dumps(res, indent=1))
    print("wrote", FILMS.relative_to(ROOT))
    return 0


def cmd_clip(spec):
    """One CPU rollout filmed in this process (the parent sets HOS_PLANT, HOS_SCENES, HOM_ROWS_TAG): the grid's label on
    top, the scores below."""
    s = json.loads(spec)
    import hom_contact_rig as HR
    import hom_films as HF
    var = "hold" if s["task"] in ("pinpush", "press") else "base"
    frames, res = HF.clip(s["task"], "grid2dff_size", float(DEFAULT[s["task"]]), s["call"], s["grasp"], "", var=var)
    cpu = HF.score(s["task"], res)
    frames = [HR.annotate_bottom(HR.annotate(f, s["top"], size=24), f"score {cpu:.2f} here, {s['gpu']:.2f} on the GPU",
                                 size=22) for f in frames]
    HR.write_mp4(frames, s["out"], fps=25)
    brief = dict(HF.res_brief(s["task"], res), cpu=round(cpu, 3), frames=len(frames), plant=res.get("plant"))
    Path(s["out"]).with_suffix(".json").write_text(json.dumps(brief, default=float))
    print("wrote", s["out"], brief, flush=True)
    return 0


def compose(paths, out, cols):
    """Clips in rows of `cols`, each holding its last frame until the longest ends; a poster at 55 % of the film."""
    import imageio.v2 as imageio
    from PIL import Image
    n = max(json.loads(Path(p).with_suffix(".json").read_text())["frames"] for p in paths)
    its = [iter(imageio.get_reader(p)) for p in paths]
    last = [None] * len(paths)
    w = imageio.get_writer(str(out), fps=25, macro_block_size=1)
    for t in range(n):
        for i, it in enumerate(its):
            last[i] = next(it, last[i])
        rows = [np.concatenate(last[r * cols:(r + 1) * cols], 1) for r in range(len(paths) // cols)]
        frame = np.concatenate(rows, 0)
        w.append_data(frame)
        if t == int(0.55 * n):
            Image.fromarray(frame).convert("RGB").save(out.with_suffix(".jpg"), quality=85)
    w.close()
    print("wrote", out.relative_to(ROOT), f"({n} frames)", flush=True)


# ----------------------------------------------------------------------------------------------- page

def f2(v):
    return f"{v:.2f}"


def andj(vals):
    vals = [f2(v) if not isinstance(v, str) else v for v in vals]
    return ", ".join(vals[:-1]) + " and " + vals[-1] if len(vals) > 1 else "".join(vals)


def pair(T, task, a, b):
    """The comparison of grids a and b for a task at its default object, keyed from a's side: rho, kept, same, the
    share above 0.5 on each, and each grid's best layout with its score on both grids."""
    k = f"{task}_d{DEFAULT[task]}|{a}|{b}"
    if k in T["pairs"]:
        r, fl = T["pairs"][k], False
    else:
        r, fl = T["pairs"][f"{task}_d{DEFAULT[task]}|{b}|{a}"], True
    A, B = ("b", "a") if fl else ("a", "b")
    best = lambda x, y: {"xy": r[f"best_{x}"]["xy"], "own": r[f"best_{x}"][f"on_{x}"], "other": r[f"best_{x}"][f"on_{y}"]}
    return {"spearman": r["spearman"], "top_tenth_kept": r["top_tenth_kept"], "same_best_grasp": r["same_best_grasp"],
            "above_a": r[f"above_half_{A}"], "above_b": r[f"above_half_{B}"], "best_a": best(A, B), "best_b": best(B, A)}


def robust(T, task, names):
    return T["robust"][f"{task}_d{DEFAULT[task]}|" + "|".join(names)]


def span(vals):
    lo, hi = min(vals), max(vals)
    return f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f} to {hi:.2f}"


def data():
    N = {k: load(f) for k, f in (("plants", "20261102-plant_transfer.json"), ("mod", "20261102-modulus_transfer.json"),
                                  ("fac", "20261102-plant_modulus_factorial.json"),
                                  ("mu", "20261102-sts_friction_transfer.json"),
                                  ("E", "20261102-sts_modulus_transfer.json"))}
    N["compare"] = load("cpu_best/20261102-plant_compare.json")
    N["ea"] = {src: load(f"{src}/summary.json") for src in ("ea_live", "ea_live_sv25")}
    N["films"] = load(FILMS.name)
    N["fresh"] = load("20261103-fresh_seeds.json")
    return N


def rhos(T, a, others):
    """{other: [Spearman per task]} against grid a."""
    return {o: [pair(T, t, a, o)["spearman"] for t in TASKS] for o in others}


def sec_terms():
    terms = [
        ("Layout", "the palm-frame mounts of the SR2 hand's three fingers, here the symmetric tripod in two coordinates. "
         "The thumb's mount lies $x_{sep}$ from the pair's, and the index's lies $y_{sep}$ from the middle's. The lattice "
         "has 625 layouts in steps of 5 mm. Its $x_{sep}$ runs from 40 to 160 mm and its $y_{sep}$ from 50 to 170 mm."),
        ("HOM tasks", "the four tasks of the design page under HOM control [2]. The twist turns a 40 mm sphere about the "
         "vertical, and the roll rolls it along the thumb's pad. In the pin-push and the press the middle finger turns a "
         "cylinder 25 mm across about the pinch of thumb and index."),
        ("Task score", "a rollout's score in [0, 1], 0 when the object is dropped. Twist: the largest turn with all three "
         "pads in contact, over 90 deg. Roll: $1/(1+e)$, where $e$ is the RMS error of the rolling rate over the commanded "
         "rate. Pin-push and press: one less the RMS error of the cylinder's angle over the stroke of 15 and 20 deg."),
        ("Grasp, best of 16", "up to 16 grasps of a layout are solved by the pose toy [1]. A grasp's score is its mean "
         "task score over seeds 0 to 2 on the GPU evaluator (HOM in batches on MuJoCo-Warp). A layout's best of 16 is its "
         "highest grasp score."),
        ("Grid", "the best of 16 of every layout on the lattice for one task at its default object, with one plant, one "
         "pad modulus and one scene friction."),
        ("Plant", "the simulated finger servo: a position servo with gain $k_p$ (N m/rad), a torque limit (N m) and a "
         "joint damping (N m s/rad), with the spec string <code>kp_kv_fr_fl_dp</code>. Section SEC[setup] lists the three "
         "plants."),
        ("Pad modulus $E$", "the Young's modulus (MPa) behind the 1 mm sphere pads of the v4b fingertip. Pad $i$ gets the "
         "stiffness $E a_i / h_i$ from its area $a_i$ and depth $h_i$. The scenes run 2.5 MPa, the fit to the UR5 presses."),
        ("Scene friction $\\mu$", "the friction coefficient of the pads and the object in the scene. It is 1.0 in the grids "
         "of 10-30 and in the reference here."),
        ("Spearman", "Spearman's rank correlation $\\rho$ of two grids' best-of-16 scores over the layouts both reached. "
         "1 means the same order of layouts."),
        ("Best tenth kept", "the share of one grid's best tenth of layouts that is in the other grid's best tenth."),
        ("Same best grasp", "the share of layouts whose best grasp is the same on both grids."),
        ("Above 0.5", "the share of a grid's layouts whose best of 16 is above 0.5."),
        ("Own, one layout, one layout and grasp", "three ways to choose a design for several grids. Own: each grid's best "
         "layout and grasp. One layout: the layout whose worst grid scores highest, with its grasp re-picked on each grid. "
         "One layout and grasp: the layout and grasp whose worst grid scores highest."),
        ("Loss", "a grid's own best score less the score of the chosen design on that grid."),
        ("Near set, slope", "the layouts that an evolutionary search ran within 0.05 of its best, at each object size. "
         "The slope is the least-squares slope of their mean $x_{sep}$ against the object's diameter, in mm per mm."),
        ("CPU check", "a layout's best grasp run once on MuJoCo on the CPU, seed 0, with every contact. On the GPU only "
         "the pads and the rigid spheres under 2 mm meet the object."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_setup(N):
    t1 = P.TAB[0] + 1
    plants = [
        ["Work plant", "4", "1", "0.08", "<code>kp4_kv0_fr1_fl0_dp0.08</code>", "fits the joint readbacks of the bench runs"],
        ["SCS0009", "0.72", "0.18", "0.0144", "<code>kp0.72_kv0_fr0.18_fl0_dp0.0144</code>",
         "the hand's servo: 0.23 N m stall at 6 V reached at 18 deg of error; the limit is the 80 % overload trip"],
        ["STS3032", "2.1", "0.515", "0.042", f"<code>{STS_PLANT}</code>",
         "the candidate servo: 0.515 N m at 7 V; full drive at 14 deg of error needs a raised position gain"],
    ]
    t = table(["Plant", "Gain $k_p$ (N m/rad)", "Torque limit (N m)", "Damping (N m s/rad)", "Spec", "Basis"], plants,
              "The three plants. The damping scales with the gain.", num_from=1)
    grids = [[GRIDS[k][0], f"<code>rows/{GRIDS[k][1]}</code>", f"<code>{GRIDS[k][2][1]}</code>" if GRIDS[k][2]
              else "<code>logs/20261030-hand_object_scale_cad_v2/scenes</code>"] for k in GRIDS]
    t2 = table(["Grid", "Rows", "Scene cache"], grids, "The 13 grids. Each runs the 625 layouts with up to 16 grasps and "
               "seeds 0 to 2 for each task at its default object, on the GPU evaluator. The scene caches were built by "
               "<code>scripts/hom/hom_modulus_scenes.py</code> at the grid's modulus and friction.", num_from=3)
    return fill("""
<p>The grids of 10-30 (<a href="<<trends>>">layout trends page</a>) ran on the work plant with pads at 10 MPa and friction
1.0. The work plant fits the hand's joint readbacks. Its gain and its torque limit are 5.6 times those of the SCS0009, the
servo of the built hand. Table <<t1>> lists it with the SCS0009 and with an STS3032, the stronger candidate. The STS3032's
gain of 2.1 N m/rad is an estimate. From its datasheet, by the construction used for the SCS0009, its gain would be 1.40
N m/rad at 6 V and 1.64 at 7 V. With those gains the 10-30 grids' best grasps mostly failed the twist and the press on the
CPU plant, and with 2.1 they recovered (<code>smoke_sts3032.sh</code> in the queue README).</p>
<p>The pad modulus was fitted to 3,568 holds of three v3 fingertips on the UR5 at 2.47 MPa. The fits per tip span 2.22 to
2.73 MPa (<a href="<<v3>>">v3 fingertip page</a>). With one modulus the measured force is 1.54 times the pads' force at
0.25 N and 0.49 times at 8 N. Moduli from 1.25 to 10 MPa were therefore run. The scenes' friction of 1.0 stands for the
grip tape of the bench tools. On the UR5, TPU on bare PLA was measured at 0.37 to 0.44, and the deploy scenes use 2.4. The
STS3032 was run at friction 0.4, 0.7 and 2.4 besides 1.0. Table <<t2>> lists the 13 grids.</p>
""", trends=TRENDS, v3=V3TIP, t1=t1, t2=t1 + 1) + t + t2


def rho_table(T, ref, rows, caption):
    body = []
    for label, other in rows:
        ps = [pair(T, t, ref, other) for t in TASKS]
        body.append([label] + [f"{p['spearman']:.2f} / {p['top_tenth_kept']:.2f} / {p['same_best_grasp']:.2f}"
                               for p in ps])
    return table(["Grid against " + ref.replace("STS3032 tuned", "STS3032")] + [NAME[t] for t in TASKS], body,
                 caption + " Each cell: Spearman / best tenth kept / same best grasp.")


def sec_plants(N):
    tb, fg = P.TAB[0] + 1, P.FIG[0] + 1
    T = N["plants"]
    W, S, K = "work plant, 10 MPa", "SCS0009, 2.5 MPa", "STS3032 tuned, 2.5 MPa"
    t = table(["Pair of plants"] + [NAME[x] for x in TASKS],
              [[a.replace("STS3032 tuned", "STS3032") + " and " + b.replace("STS3032 tuned", "STS3032")]
               + [f"{pair(T, x, a, b)['spearman']:.2f} / {pair(T, x, a, b)['top_tenth_kept']:.2f} / "
                  f"{pair(T, x, a, b)['same_best_grasp']:.2f}" for x in TASKS] for a, b in ((W, S), (W, K), (S, K))],
              "Landscapes on three plants. Each cell: Spearman / best tenth kept / same best grasp.")
    ws = [pair(T, x, W, S) for x in TASKS]
    wk = [pair(T, x, W, K)["spearman"] for x in TASKS]
    sk = [pair(T, x, S, K)["spearman"] for x in TASKS]
    pw = pair(T, "press", W, S)
    return fill("""
<p>The plant reorders the layouts. Between the work plant and the SCS0009 the Spearman correlation is <<ws>> for the
twist, the roll, the pin-push and the press (Table <<tb>>, Figure <<fg>>). The STS3032 at 2.1 N m/rad lies between them,
closer to the work plant. Its Spearman correlation is <<wk>> against the work plant. Against the SCS0009 it is <<sk>>. The
landscapes are ordered by the gain: 0.72, 2.1 and 4 N m/rad.</p>
<p>The best grasp changes more than the layout. Between the work plant and the SCS0009 the same grasp is best for <<same>>
of the layouts. The work plant's best press layout scores <<pa>> on the SCS0009 with its best grasp there. The SCS0009's
own best press layout scores <<pb>>. The weak servo changes the press most: <<pk>> of the work plant's best tenth stays in
the SCS0009's best tenth.</p>
""", ws=andj([p["spearman"] for p in ws]), wk=andj(wk), sk=andj(sk), same=span([p["same_best_grasp"] for p in ws]),
                pa=f2(pw["best_a"]["other"]), pb=f2(pw["best_b"]["own"]), pk=f2(pw["top_tenth_kept"]),
                tb=tb, fg=fg) + t + img(
        MEDIA / "20261102-plant_maps_plants.png",
        "Best-of-16 maps of the four tasks on the three plants at 2.5 MPa (work plant: 10 MPa), friction 1.0. Grey: no "
        "grasp solved. Star: the best layout. <code>hom_plant_page.py maps</code>.")


def sec_modulus(N):
    tb, fg = P.TAB[0] + 1, P.FIG[0] + 1
    M, E, F = N["mod"], N["E"], N["fac"]
    s25, k25 = "SCS0009, 2.5 MPa", "STS3032, 2.5 MPa"
    t = table(["Grid against 2.5 MPa"] + [NAME[x] for x in TASKS],
              [[f"{servo}, {e:g} MPa"] + [f"{pair(T, x, ref, f'{servo}, {e:g} MPa')['spearman']:.2f} / "
                                          f"{pair(T, x, ref, f'{servo}, {e:g} MPa')['top_tenth_kept']:.2f}" for x in TASKS]
               for servo, T, ref in (("SCS0009", M, s25), ("STS3032", E, k25)) for e in (1.25, 5, 10)]
              + [["work plant, 10 MPa"] + [f"{pair(F, x, 'work plant, 2.5 MPa', 'work plant, 10 MPa')['spearman']:.2f} / "
                                           f"{pair(F, x, 'work plant, 2.5 MPa', 'work plant, 10 MPa')['top_tenth_kept']:.2f}"
                                           for x in TASKS]],
              "Pad modulus on one plant, against the same plant at 2.5 MPa, friction 1.0. Each cell: Spearman / best tenth "
              "kept.")
    sc = [pair(M, x, s25, "SCS0009, 10 MPa")["spearman"] for x in TASKS]
    st = [pair(E, x, k25, f"STS3032, {e:g} MPa")["spearman"] for x in TASKS for e in (1.25, 5, 10)]
    wk = [pair(F, x, "work plant, 2.5 MPa", "work plant, 10 MPa")["spearman"] for x in TASKS]
    between = [pair(F, x, "work plant, 2.5 MPa", s25)["spearman"] for x in TASKS]
    return fill("""
<p>On one plant the pad modulus moves the landscape little. From 2.5 to 10 MPa the Spearman correlation is <<wk>> on the
work plant (Table <<tb>>). On the SCS0009 it is <<sc>>. On the STS3032 it is <<st>> over 1.25 to 10 MPa (Figure <<fg>>).
At one modulus the plants differ more. At 2.5 MPa the Spearman correlation between the work plant and the SCS0009 is
<<bt>>.</p>
""", wk=andj(wk), sc=andj(sc), st=span(st), bt=andj(between), tb=tb, fg=fg) + t + img(
        MEDIA / "20261102-plant_maps_modulus.png",
        "Best-of-16 maps on the SCS0009 and the STS3032 with pads at 1.25 and 10 MPa, friction 1.0; Figure "
        f"{P.FIG[0]} has both at 2.5 MPa. <code>hom_plant_page.py maps</code>.")


def sec_friction(N):
    tb, fg, fm = P.TAB[0] + 1, P.FIG[0] + 1, P.FILM[0] + 1
    T = N["mu"]
    ref = "STS3032, friction 1"
    oth = {mu: f"STS3032, friction {mu:g}" for mu in (0.4, 0.7, 2.4)}
    rows = [[f"friction {mu:g}"] + [f"{pair(T, x, ref, o)['spearman']:.2f} / {pair(T, x, ref, o)['top_tenth_kept']:.2f}"
                                    for x in TASKS] for mu, o in oth.items()]
    rows.append(["above 0.5 at friction 0.4 / 1.0"] + [f"{pair(T, x, oth[0.4], ref)['above_a']:.2f} / "
                                                       f"{pair(T, x, oth[0.4], ref)['above_b']:.2f}" for x in TASKS])
    t = table(["STS3032 grid against friction 1.0"] + [NAME[x] for x in TASKS], rows,
              "Scene friction on the STS3032 at 2.5 MPa. Each cell: Spearman / best tenth kept; the last row gives the "
              "share of layouts above 0.5 at friction 0.4 and 1.0.")
    TABNO["friction"] = P.TAB[0]
    r04 = [pair(T, x, ref, oth[0.4])["spearman"] for x in TASKS]
    r07 = [pair(T, x, ref, oth[0.7])["spearman"] for x in TASKS]
    r24 = [pair(T, x, ref, oth[2.4])["spearman"] for x in TASKS]
    tw = pair(T, "twist", ref, oth[0.4])["best_a"]
    pr = pair(T, "press", ref, oth[2.4])
    F = (N["films"] or {}).get("friction")
    fs = ""
    if F:
        a, b, c = F
        fs = (" In Film <<fm>> that layout and its best grasp at friction 1.0 run on the CPU plant at the three frictions. "
              f"At 0.4 the sphere slides up out of the grasp as the fingers close and is dropped (score {a['cpu']:.2f}). "
              f"At 1.0 it turns through {b['ramp_three_pads_max_deg']:.0f} deg with all three pads in contact "
              f"({b['cpu']:.2f}). At 2.4 that turn ends at {c['ramp_three_pads_max_deg']:.0f} deg ({c['cpu']:.2f}). The "
              "GPU scores of that grasp are the same.")
    txt = fill("""
<p>The scene friction reorders the layouts more than the pad modulus does. Against friction 1.0 the Spearman correlation
is <<r04>> at friction 0.4 for the twist, the roll, the pin-push and the press (Table <<tb>>, Figure <<fg>>). It is <<r07>>
at friction 0.7. At friction 2.4 it is <<r24>>. At friction 0.4 fewer layouts score above 0.5 in the twist, the pin-push
and the press. At friction 0.4 the twist's best layout of friction 1.0, (<<x>>, <<y>>), scores <<on04>> with its best
grasp there.<<fs>> At friction 2.4 the press's best score falls from <<p1>> to <<p2>>.</p>
""", r04=andj(r04), r07=andj(r07), r24=andj(r24), x=tw["xy"][0], y=tw["xy"][1], on04=f2(tw["other"]), fs=fs, fm=fm,
               p1=f2(pr["best_a"]["own"]), p2=f2(pr["best_b"]["own"]), tb=tb, fg=fg)
    out = txt + t + img(MEDIA / "20261102-plant_maps_friction.png",
                        "Best-of-16 maps on the STS3032 at 2.5 MPa with scene friction 0.4, 0.7, 1.0 and 2.4. "
                        "<code>hom_plant_page.py maps</code>.")
    if F:
        out += film(MEDIA / "20261102-film_friction.mp4",
                    f"The twist on the STS3032 at 2.5 MPa, layout ({F[0]['xy'][0]}, {F[0]['xy'][1]}) and grasp "
                    f"{F[0]['grasp']}, at scene friction 0.4, 1.0 and 2.4; CPU plant, seed 0, every contact. Bottom: the "
                    "run's score and the grasp's GPU score. <code>hom_plant_page.py films</code>.")
    return out


SUBSETS = [  # (data key, grid names, label)
    ("plants", ("work plant, 10 MPa", "SCS0009, 2.5 MPa", "STS3032 tuned, 2.5 MPa"), "three plants"),
    ("mod", ("SCS0009, 1.25 MPa", "SCS0009, 2.5 MPa", "SCS0009, 5 MPa"), "SCS0009, 1.25 to 5 MPa"),
    ("E", ("STS3032, 1.25 MPa", "STS3032, 2.5 MPa", "STS3032, 5 MPa"), "STS3032, 1.25 to 5 MPa"),
    ("mu", ("STS3032, friction 0.4", "STS3032, friction 0.7", "STS3032, friction 1"), "STS3032, friction 0.4 to 1.0"),
]


def one_design(N, key, names, task):
    r = robust(N[key], task, names)
    own = [r["own"][n]["score"] for n in names]
    ol = [r["one_layout"]["score"][n] for n in names]
    og = [r["one_layout_and_grasp"]["score"][n] for n in names]
    return r, own, ol, og


def sec_design(N):
    tb, fm = P.TAB[0] + 1, P.FILM[0] + 1
    body, worst = [], {}
    for key, names, label in SUBSETS:
        for task in TASKS:
            r, own, ol, og = one_design(N, key, names, task)
            lo, lg = max(a - b for a, b in zip(own, ol)), max(a - b for a, b in zip(own, og))
            worst[(key, task)] = (lo, lg)
            gr = sorted(set(r["one_layout"]["grasp"].values()))
            body.append([label if task == "twist" else "", NAME[task], " / ".join(f2(v) for v in own),
                         f"({r['one_layout']['xy'][0]}, {r['one_layout']['xy'][1]}), grasp "
                         f"{', '.join(str(g) for g in gr)}: " + " / ".join(f2(v) for v in ol), f2(lo),
                         f"({r['one_layout_and_grasp']['xy'][0]}, {r['one_layout_and_grasp']['xy'][1]}), grasp "
                         f"{r['one_layout_and_grasp']['grasp']}: " + " / ".join(f2(v) for v in og), f2(lg)])
    t = table(["Grids", "Task", "Own", "One layout", "Loss", "One layout and grasp", "Loss"], body,
              "One design for several grids, from the grids' three-seed means. Scores in the order of the grids: work "
              "plant, SCS0009, STS3032; or 1.25, 2.5, 5 MPa; or friction 0.4, 0.7, 1.0. One layout: the grasps it takes on "
              "the grids. Loss: the largest over the grids.",
              num_from=2)
    TABNO["design"] = P.TAB[0]
    four = {}
    for key, names in (("mod", ("SCS0009, 1.25 MPa", "SCS0009, 2.5 MPa", "SCS0009, 5 MPa", "SCS0009, 10 MPa")),
                       ("E", ("STS3032, 1.25 MPa", "STS3032, 2.5 MPa", "STS3032, 5 MPa", "STS3032, 10 MPa")),
                       ("mu", ("STS3032, friction 0.4", "STS3032, friction 0.7", "STS3032, friction 1",
                               "STS3032, friction 2.4"))):
        four[key] = [max(a - b for a, b in zip(*one_design(N, key, names, x)[1::2])) for x in TASKS]
    F = (N["films"] or {}).get("plants")
    fs = ""
    if F:
        tw, pr = F[:3], F[3:]
        fs = (f" Film <<fm>> runs the twist's design ({tw[0]['xy'][0]}, {tw[0]['xy'][1]}), grasp {tw[0]['grasp']}, and the "
              f"press's ({pr[0]['xy'][0]}, {pr[0]['xy'][1]}), grasp {pr[0]['grasp']}, on the three plants on the CPU. The "
              f"twist scores {andj([c['cpu'] for c in tw])} there, and the press {andj([c['cpu'] for c in pr])}.")
    pl = [worst[("plants", x)] for x in TASKS]
    txt = fill("""
<p>A design chosen for several grids loses little against each grid's own best (Table <<tb>>). One layout for the three
plants, with the grasp re-picked on each, loses at most <<pl>> in the twist, the roll, the pin-push and the press. With one
grasp as well it loses at most <<plg>>. Over 1.25 to 5 MPa one layout and one grasp lose at most <<scs>> on the SCS0009.
On the STS3032 they lose at most <<sts>>. Over friction 0.4 to 1.0 they lose at most <<mu>>. With the grids at 10 MPa
added, the largest losses are <<s10>> on the SCS0009 and <<k10>> on the STS3032. With friction 2.4 added they are
<<m24>>.<<fs>></p>
""", pl=andj([a for a, _ in pl]), plg=andj([b for _, b in pl]), scs=andj([worst[("mod", x)][1] for x in TASKS]),
               sts=andj([worst[("E", x)][1] for x in TASKS]), mu=andj([worst[("mu", x)][1] for x in TASKS]),
               s10=andj(four["mod"]), k10=andj(four["E"]), m24=andj(four["mu"]), fs=fs, fm=fm, tb=tb)
    out = txt + t
    if F:
        out += film(MEDIA / "20261102-film_plants.mp4",
                    "One design on three plants. Top: the twist; bottom: the press. Columns: the work plant at 10 MPa, the "
                    "SCS0009 and the STS3032 at 2.5 MPa. CPU plant, seed 0, every contact. Under each clip: the run's "
                    "score and the grasp's GPU score. <code>hom_plant_page.py films</code>.")
    return out + sec_fresh(N)


def fresh_stats(N):
    """Per entry of the fresh-seed check: the one-design loss at seeds 0-2 and at seeds 3-7 (largest over the grids)."""
    out = {}
    for e in (N.get("fresh") or {}).get("entries", []):
        ns, t3, fr = list(e["picks"]), e["three_seed"], e["fresh"]
        k = "one_layout_and_grasp"
        if any(fr[n].get(k) is None or t3[n].get(k) is None for n in ns):
            continue
        out[e["key"]] = (max(t3[n]["own"] - t3[n][k] for n in ns), max(fr[n]["own"] - fr[n][k] for n in ns), e)
    return out


def sec_fresh(N):
    st = fresh_stats(N)
    if not st:
        return ""
    grew = sorted(b - a for a, b, _ in st.values() if b - a > 0.05)
    body = []
    for key, names, label in SUBSETS:
        for task in TASKS:
            k = f"{task}_d{40 if task in ('twist', 'roll') else 25}|" + "|".join(names)
            if k not in st:
                continue
            a, b, e = st[k]
            p = e["picks"][names[0]]["one_layout_and_grasp"]
            xy = (40 + 5 * (p[0] // 25), 50 + 5 * (p[0] % 25))
            body.append([label if task == "twist" else "", NAME[task], f"({xy[0]}, {xy[1]}), grasp {p[1]}",
                         " / ".join(f2(e["three_seed"][n]["one_layout_and_grasp"]) for n in names),
                         " / ".join(f2(e["fresh"][n]["one_layout_and_grasp"]) for n in names),
                         " / ".join(f2(e["fresh"][n]["own"]) for n in names), f2(a), f2(b)])
    t = table(["Grids", "Task", "One layout and grasp", "Seeds 0-2", "Seeds 3-7", "Own best, seeds 3-7", "Loss, 0-2",
               "Loss, 3-7"], body,
              "The one-design picks of Table " + str(TABNO["design"]) + " rerun at seeds 3 to 7 on each grid's plant and "
              "scene cache, with a noise seed no grid used (<code>hom_fresh_seeds.py</code>). Scores in the order of the "
              "grids. Own best: each grid's own pick, rerun the same way. Loss: the largest over the grids, against the "
              "own picks at the same seeds.", num_from=3)
    txt = fill("""
<p>The picks were chosen and scored on seeds 0 to 2, so each score is the best of many draws. Rerun at seeds 3 to 7
(Table <<tf>>), the one-layout-and-grasp design keeps its loss within 0.05 of its three-seed loss in <<n_ok>> of the
<<n>> grid subsets of the transfer files. In the other <<n_bad>> its loss grows by <<lo>> to <<hi>>, because the pick
scored near its three-seed value on some grids and much lower on one. The grids' own picks fall as well: of the
<<n_own>> own picks, <<n_own_drop>> lose more than 0.1 at fresh seeds. Over friction 0.4 to 1.0 on the STS3032 the
fresh-seed losses are <<mu>>.</p>
""", tf=P.TAB[0], n=len(st), n_ok=len(st) - len(grew), n_bad=len(grew), lo=f2(grew[0]) if grew else "-",
               hi=f2(grew[-1]) if grew else "-", n_own=OWN_N(N), n_own_drop=OWN_DROP(N),
               mu=andj([st[f"{x}_d{40 if x in ('twist', 'roll') else 25}|" + "|".join(SUBSETS[3][1])][1] for x in TASKS]))
    return txt + t


def _own_pairs(N):
    seen = {}
    for e in (N.get("fresh") or {}).get("entries", []):
        for n in e["picks"]:
            seen[(e["task"], e["src"][n], tuple(e["picks"][n]["own"]))] = (e["three_seed"][n]["own"], e["fresh"][n]["own"])
    return seen


def OWN_N(N):
    return len(_own_pairs(N))


def OWN_DROP(N):
    return sum(1 for a, b in _own_pairs(N).values() if b is not None and a - b > 0.1)


def slopes(S):
    out = {}
    for task in TASKS:
        pts = sorted(((v["d_mm"], v.get("grid") or v.get("ea")) for v in S.values()
                      if isinstance(v, dict) and v.get("task") == task and (v.get("grid") or v.get("ea"))),
                     key=lambda p: p[0])
        d = np.array([p[0] for p in pts])
        out[task] = float(np.polyfit(d, [p[1]["near_centroid"][0] for p in pts], 1)[0])
    return out


def cpu_corrected():
    """Median over sizes of the CPU check (every contact) of each corrected search's best layout, and of its GPU score."""
    out = {}
    for task in TASKS:
        cpu, gpu = [], []
        for p in sorted((ROWS / "cpu_best").glob(f"top_{task}_d*_ff*_sv25.jsonl")):
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            rows = [r for r in rows if not r.get("gpu_filter")]
            if rows:
                b = max(rows, key=lambda r: r["gpu_best"])
                cpu.append(b["score"])
                gpu.append(b["gpu_best"])
        out[task] = (float(np.median(cpu)), float(np.median(gpu)), len(cpu))
    return out


def sec_sizes(N):
    tb = P.TAB[0] + 1
    s0, s1 = slopes(N["ea"]["ea_live"]), slopes(N["ea"]["ea_live_sv25"])
    C = cpu_corrected()
    pc = N["compare"]["searches"]
    old = {}
    for task in TASKS:
        v = [s["plants"]["servo, 2.5 MPa"]["all"]["best"] for k, s in pc.items() if k.startswith(task + "_")]
        w = [s["plants"]["work, 2.5 MPa"]["all"]["best"] for k, s in pc.items() if k.startswith(task + "_")]
        old[task] = (sum(1 for x in v if x < 0.005), len(v), float(np.median(w)), float(np.median(v)))
    t = table(["Task", "Slope, work plant 10 MPa", "Slope, SCS0009 2.5 MPa", "CPU check, SCS0009", "GPU, SCS0009",
               "10-30 designs at 0.00 on the SCS0009"],
              [[NAME[x], f2(s0[x]), f2(s1[x]), f2(C[x][0]), f2(C[x][1]), f"{old[x][0]} of {old[x][1]} sizes"]
               for x in TASKS],
              "Size trends and CPU checks. Slope: mm of $x_{sep}$ per mm of diameter. CPU check and GPU: medians over the "
              "sizes of the best layout of each search on the SCS0009 at 2.5 MPa. Last column: sizes at which the 10 best "
              "layouts of the 10-30 searches, with their 10-30 grasps, all score 0.00 on the SCS0009 (CPU, every contact).")
    TABNO["sizes"] = P.TAB[0]
    return fill("""
<p>The evolutionary searches at every object size were rerun on the SCS0009 at 2.5 MPa (<code>rows/ea_live_sv25</code>).
The near sets' $x_{sep}$ grows with the diameter at <<s1>> mm per mm in the twist, the roll, the pin-push and the press
(Table <<tb>>). On the work plant at 10 MPa the slopes were <<s0>>. The rerun searches' best layouts were checked on the
CPU. The medians over the sizes are <<cc>> there, and <<cg>> on the GPU. The 10-30 designs were also run with their 10-30
grasps on the SCS0009. In the press all 10 best layouts score 0.00 at <<pz>> of <<pn>> sizes.</p>
""", s1=andj([s1[x] for x in TASKS]), s0=andj([s0[x] for x in TASKS]), cc=andj([C[x][0] for x in TASKS]),
                cg=andj([C[x][1] for x in TASKS]), pz=old["press"][0], pn=old["press"][1], tb=tb) + t


def sec_limits(N):
    return """
<p>The one-design picks are chosen on three seeds. Rerun at five other seeds, some lose much more than at the seeds they
were chosen on (Section SEC[design]). The STS3032's gain of 2.1 N m/rad is an estimate that has not been measured. The scene
friction is one coefficient for the pads and the object, and the friction and modulus sweeps ran at the two default objects
only.</p>"""


def sec_future(N):
    return fill("""<ul>
<li>Choose a single design in two stages: rank the layouts and grasps by the worst grid's three-seed score, rerun the best
20 at ten more seeds on every grid, and choose by the worst grid's mean over all of them. Check the choice at five further
seeds. Two-stage choice fails if its loss there exceeds its selection loss by more than 0.05 as often as the one-stage
picks of Table <<td>> do.</li>
<li>Measure the STS3032's position gain on the bench, as the SCS0009's 0.0186 deg per load unit was measured, and rerun the
four grids at the measured gain. A gain of 1.64 N m/rad or less would bring back the twist and press failures of the smoke
test.</li>
<li>Measure the friction of the v4b pad on the bench tools' grip tape and rerun the STS3032 grids at that value. A value of
0.7 or less would reorder the best layouts by the amounts of Table <<tf>>.</li>
<li>Rerun the evolutionary searches at every object size on the STS3032 (<code>hom_ea.py</code> with
<code>HOS_PLANT=kp2.1_kv0_fr0.515_fl0_dp0.042</code>) and compare its slopes with those of Table <<ts>>.</li>
</ul>""", td=TABNO["design"], tf=TABNO["friction"], ts=TABNO["sizes"])


def sec_files():
    return """<p>Files: the builder <code>scripts/hom/hom_plant_page.py</code>; the comparisons
<code>scripts/hom/hom_plant_transfer.py</code> (<code>--set plants|modulus|factorial|sts_modulus|sts_friction</code>); the
scene caches <code>scripts/hom/hom_modulus_scenes.py</code>; the queues and logs in <code>logs/20261102-hom_servo_plant/</code>
(README.md); rows in <code>docs/experiments/20261029-foundational_pose_review/rows/</code>.</p>"""


def lede_and_summary(N):
    T, M, E, U = N["plants"], N["mod"], N["E"], N["mu"]
    ws = [pair(T, x, "work plant, 10 MPa", "SCS0009, 2.5 MPa")["spearman"] for x in TASKS]
    md = ([pair(M, x, "SCS0009, 2.5 MPa", f"SCS0009, {e:g} MPa")["spearman"] for x in TASKS for e in (1.25, 5, 10)]
          + [pair(E, x, "STS3032, 2.5 MPa", f"STS3032, {e:g} MPa")["spearman"] for x in TASKS for e in (1.25, 5, 10)])
    mu = [pair(U, x, "STS3032, friction 1", "STS3032, friction 0.4")["spearman"] for x in TASKS]
    lpl = max(max(a - b for a, b in zip(*one_design(N, "plants", SUBSETS[0][1], x)[1:3])) for x in TASKS)
    lmu = max(max(a - b for a, b in zip(*one_design(N, "mu", SUBSETS[3][1], x)[1::2])) for x in TASKS)
    lede = (f"The finger servo and the scene friction reorder the SR2 hand's HOM design landscapes, and the pad modulus "
            f"barely does. Between the work plant of 10-30 and the SCS0009, the Spearman correlation of the layouts' scores "
            f"is {span(ws)} over the four tasks. Over pad moduli of 1.25 to 10 MPa it is {span(md)} against 2.5 MPa. On the "
            f"STS3032, friction 0.4 against 1.0 gives {span(mu)}. One layout for the three plants loses at most {lpl:.2f} "
            f"against each plant's best, with the grasp re-picked on each. One layout and grasp for friction 0.4 to 1.0 "
            f"loses at most {lmu:.2f}.")
    st = fresh_stats(N)
    if st:
        grew = sorted(b - a for a, b, _ in st.values() if b - a > 0.05)
        lede += (f" Rerun at five fresh seeds, {len(st) - len(grew)} of {len(st)} such picks keep their loss within 0.05; "
                 f"{len(grew)} lose {grew[0]:.2f} to {grew[-1]:.2f} more." if grew else "")
    summary = (f"The finger servo and the scene friction reorder the HOM layouts (Spearman {span(ws)} between the work plant "
               f"and the SCS0009, {span(mu)} between friction 0.4 and 1.0); pad moduli of 1.25-10 MPa barely do "
               f"({span(md)}). One layout for three servos loses at most {lpl:.2f}.")
    return lede, summary


def assemble(sections, title, h1, eyebrow, lede, byline, summary):
    num = {sid: k for k, (sid, _hd, _b) in enumerate(sections, 1)}
    sub = (lambda s: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), s))
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|MPa|N\b|ms\b|s\b)", "\\1&nbsp;\\2", s_))
    sections = [(sid, hd, unit(sub(body))) for sid, hd, body in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>'
                     for k, (sid, hd, body) in enumerate(sections, 1))
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": title, "H1": h1, "EYEBROW": eyebrow, "LEDE": unit(sub(lede)), "BYLINE": byline, "TOC": toc,
                 "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = P.render(page, n)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-02")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)


def cmd_page():
    N = data()
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("setup", "Plants, pad moduli and scene frictions", sec_setup(N)),
        ("plants", "Landscapes on three finger servos", sec_plants(N)),
        ("modulus", "Pad modulus on one servo", sec_modulus(N)),
        ("friction", "Scene friction on the STS3032", sec_friction(N)),
        ("design", "One design for several servos, moduli or frictions", sec_design(N)),
        ("sizes", "Size trends and CPU checks on the SCS0009", sec_sizes(N)),
        ("limits", "Limitations", sec_limits(N)),
        ("future", "Future work", sec_future(N)),
        ("refs", "References and files", '<ol class="refs">' + "".join(f"<li>{r[r.index(']') + 2:]}</li>" for r in REFS)
         + "</ol>" + sec_files()),
    ]
    lede, summary = lede_and_summary(N)
    byline = ("Simulation, 2026-11-02 and 03. Hand: the CAD-model SR2 hand with the v4b fingertip "
              "(<code>SCALE_HAND=cad_v2</code>); HOM on MuJoCo-Warp and on the CPU plant. Follows "
              f"<a href=\"{TRENDS}\">the layout trends page</a>. Builder: <code>scripts/hom/hom_plant_page.py</code>.")
    assemble(sections, "SR2 layouts across servos", "SR2 hand layouts under HOM control across finger servos, pad "
             "moduli and scene friction", "HOM layout study, 2026-11-02", lede, byline, summary)
    return 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "maps":
        return cmd_maps()
    if a and a[0] == "films":
        return cmd_films()
    if a and a[0] == "clip":
        return cmd_clip(a[1])
    return cmd_page()


if __name__ == "__main__":
    sys.exit(main())

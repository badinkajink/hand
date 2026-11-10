#!/usr/bin/env python3
"""Builds docs/experiments/20261029-foundational_pose_review/20261110-hom_servo_stiffness_mujoco4.html: the SR2 hand's HOM
landscapes at the default objects on the SCS0009 stiffness measured on 2026-11-06 (0.55 N m/rad, then with its 0.59 deg
dead band) against the model's 0.72 N m/rad, and on MuJoCo 4.0 against 3.6 (post-reset steps 1 and 2; owner, 11-09: "we
previously re-did sysid from 0.72 nmrad to 0.55nmrad, but i dont want to rerun everything, so we can stick to 0.72 for
now"; "part of me also wants to just send it on new mujoco").

Data: the grids of GRIDS (hom_gpu.py run, logs/20261110-hom_post_reset/gpu_queue_step*.sh), ranked by
hom_landscape.track_score with rollout_score beside it; the cached maps rows/20261110-stiffness_landscapes.json.

    .venv/bin/python scripts/hom/hom_stiffness_page.py maps    # the cache and the map figures
    .venv/bin/python scripts/hom/hom_stiffness_page.py         # the page
"""
from __future__ import annotations

import json
import os
import re
import sys
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

table, img, film, esc = P.table, P.img, P.film, P.esc
D = ROOT / "docs/experiments/20261029-foundational_pose_review"
MEDIA, ROWS = D / "media", D / "rows"
OUT = D / "20261110-hom_servo_stiffness_mujoco4.html"
LAND = ROWS / "20261110-stiffness_landscapes.json"
PLANTPAGE = "20261102-hom_plant_modulus.html"
TRACK = "20261109-hom_tracking_score.html"
TRENDS = "20261030-hom_layout_trends.html"
SYSID = "../20261106-sr2_servo_sysid/20261106-sr2_servo_identification.html"
MJ4 = "../20261108-mujoco4_port/20261108-mujoco4_contact_and_stack.html"
TASKS = ("twist", "roll", "press")
DEFAULT = {"twist": 40, "roll": 40, "press": 25}
NAME = {"twist": "Twist", "roll": "Roll", "press": "Press"}
OBJ = {"twist": "40 mm sphere", "roll": "40 mm sphere", "press": "25 mm cylinder"}
# key: (label, rows directory, MuJoCo version, run date); every grid on the 2.5 MPa scenes of 11-02
GRIDS = {
    "sv72": ("0.72 N m/rad, MuJoCo 3.6, 11-02", "grid2dff_v4b_sv25", "3.6.0", "2026-11-02"),
    "sv72r": ("0.72 N m/rad, MuJoCo 3.6, 11-10", "grid2dff_v4b_sv25r", "3.6.0", "2026-11-10"),
    "sv55": ("0.55 N m/rad, MuJoCo 3.6", "grid2dff_v4b_sv55", "3.6.0", "2026-11-10"),
    "sv55db": ("0.55 N m/rad and dead band, MuJoCo 3.6", "grid2dff_v4b_sv55db", "3.6.0", "2026-11-10"),
    "sv72r2": ("0.72 N m/rad, MuJoCo 3.6, 11-10, 200 layouts again", "grid2dff_v4b_sv25r2", "3.6.0", "2026-11-10"),
    "mj4": ("0.55 N m/rad and dead band, MuJoCo 4.0", "grid2dff_v4b_sv55db_mj4", "4.0.0", "2026-11-10"),
}
SCORES = ("track", "rollout")


def load(name):
    p = ROWS / name
    return json.loads(p.read_text()) if p.exists() else None


def cmd_maps():
    """Each grid's best-of-16 per layout by both scores and its best grasp, cached; then the map figures."""
    import hom_plant_transfer as TR
    out = {}
    for k, (_label, src, _v, _d) in GRIDS.items():
        out[k] = {}
        for task in TASKS:
            p = ROWS / src / f"{task}_d{DEFAULT[task]}.jsonl"
            out[k][task] = {}
            if not p.exists():
                continue
            for sc in SCORES:
                g = TR.grasp_means(task, p, sc)
                if g:
                    out[k][task][sc] = {"best": {str(c): round(max(v.values()), 4) for c, v in g.items()},
                                        "grasp": {str(c): max(v, key=v.get) for c, v in g.items()}}
        print(k, {t: {s: len(v["best"]) for s, v in out[k][t].items()} for t in TASKS}, flush=True)
    LAND.write_text(json.dumps(out))
    print("wrote", LAND.relative_to(ROOT))
    keys = [k for k in GRIDS if k not in ("sv72", "sv72r2", "mj4") and all("track" in out[k][t] for t in TASKS)]
    fig_maps(out, keys, "20261110-stiffness_maps.png")
    if all("track" in out["mj4"].get(t, {}) for t in TASKS):
        fig_maps(out, ["sv55db", "mj4"], "20261110-mj4_maps.png")
    return 0


WARN = ROWS / "20261110-mj4_warnings.json"


def cmd_warnings():
    """MuJoCo Warp 4.0's iteration-limit warnings per grid, counted in the queue's compressed logs."""
    import gzip
    out = {}
    for task in TASKS:
        p = ROOT / f"logs/20261110-hom_post_reset/grid_mj4_sv55db_mj4_{task}.log.gz"
        n_ls = n_it = 0
        with gzip.open(p, "rt", errors="ignore") as fh:
            for line in fh:
                n_ls += line.startswith("linesearch iterations limit reached")
                n_it += line.startswith("solver iterations limit reached")
        out[task] = {"linesearch": n_ls, "solver": n_it}
    WARN.write_text(json.dumps(out, indent=1) + "\n")
    print(out)
    return 0


def fig_maps(Lc, keys, name):
    """Best-of-16 tracking-score maps over (x_sep, y_sep): a row per grid, a column per task; grey where no grasp was
    solved, the best layout starred."""
    plt = FR.plt_setup()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14})
    fig, axs = plt.subplots(len(keys), 3, figsize=(8.6, 2.6 * len(keys) + 0.7), constrained_layout=True, sharex=True,
                            sharey=True, squeeze=False)
    ext = [37.5, 162.5, 47.5, 172.5]
    im = None
    for i, k in enumerate(keys):
        for j, task in enumerate(TASKS):
            ax = axs[i, j]
            best = {int(c): v for c, v in Lc[k][task]["track"]["best"].items()}
            Z = np.full((25, 25), np.nan)
            for c, v in best.items():
                Z[c % 25, c // 25] = v
            ax.imshow(np.where(np.isfinite(Z), np.nan, 0.0), origin="lower", extent=ext, cmap="Greys", vmin=-1, vmax=1,
                      aspect="equal")
            im = ax.imshow(Z, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=0.8, aspect="equal")
            c = max(best, key=best.get)
            ax.plot(40 + 5 * (c // 25), 50 + 5 * (c % 25), "r*", ms=14, mec="k")
            ax.set_xticks([50, 100, 150])
            ax.set_yticks([60, 100, 140])
            if i == 0:
                ax.set_title(f"{NAME[task]}\n{OBJ[task]}")
            if i == len(keys) - 1:
                ax.set_xlabel("$x_{sep}$ (mm)")
            if j == 0:
                ax.set_ylabel(GRIDS[k][0].replace(", MuJoCo", ",\nMuJoCo") + "\n$y_{sep}$ (mm)", fontsize=12)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="Best-of-16 tracking score")
    FR.save(fig, name)


def xy(c):
    return (40 + 5 * (int(c) // 25), 50 + 5 * (int(c) % 25))


def pair(Lc, a, b, task, score):
    """Spearman's rho of two grids' best-of-16 over the layouts both reached, the share of a's best tenth in b's, and
    each grid's best layout with its score on both (hom_plant_transfer.compare's measures on the cached maps)."""
    from sr2hand.numerics.stats import spearman
    A, B = Lc[a][task].get(score, {}).get("best"), Lc[b][task].get(score, {}).get("best")
    if not A or not B:
        return None
    calls = sorted(set(A) & set(B), key=int)
    x, y = np.array([A[c] for c in calls]), np.array([B[c] for c in calls])
    k = max(1, len(calls) // 10)
    ta = {calls[i] for i in np.argsort(-x)[:k]}
    tb = {calls[i] for i in np.argsort(-y)[:k]}
    ba, bb = calls[int(np.argmax(x))], calls[int(np.argmax(y))]
    return {"n": len(calls), "rho": float(spearman(x, y)), "top": len(ta & tb) / k,
            "best_a": (xy(ba), A[ba], B[ba]), "best_b": (xy(bb), B[bb], A[bb]),
            "median_a": float(np.median(x)), "median_b": float(np.median(y))}


def scores_rho(Lc, k, task):
    """Spearman's rho between a grid's best-of-16 by the tracking score and by the earlier score."""
    from sr2hand.numerics.stats import spearman
    T_, R_ = Lc[k][task].get("track", {}).get("best"), Lc[k][task].get("rollout", {}).get("best")
    if not T_ or not R_:
        return None
    calls = sorted(set(T_) & set(R_), key=int)
    return float(spearman(np.array([T_[c] for c in calls]), np.array([R_[c] for c in calls])))


def cmd_numbers():
    """The comparisons printed from the cache, for writing the page."""
    Lc = load(LAND.name)
    for a, b in (("sv72", "sv72r"), ("sv72", "sv55"), ("sv72r", "sv55"), ("sv55", "sv55db"), ("sv72r", "sv55db")):
        if a not in Lc or b not in Lc:
            continue
        for task in TASKS:
            for sc in SCORES:
                r = pair(Lc, a, b, task, sc)
                if r:
                    print(f"{a:6s} {b:6s} {task:6s} {sc:7s} n {r['n']:3d} rho {r['rho']:.3f} top {r['top']:.2f} best_a "
                          f"{r['best_a'][0]} {r['best_a'][1]:.2f}->{r['best_a'][2]:.2f} best_b {r['best_b'][0]} "
                          f"{r['best_b'][1]:.2f} (on a {r['best_b'][2]:.2f}) med {r['median_a']:.2f}/{r['median_b']:.2f}")
    for k in Lc:
        print(k, {t: (None if scores_rho(Lc, k, t) is None else round(scores_rho(Lc, k, t), 3)) for t in TASKS})
    return 0


# ----------------------------------------------------------------------------------------------- films

FILMS = ROWS / "20261110-stiffness_films.json"
# Mesa's software renderer through EGL, so that the clips stay off the GPU while its queue runs
SOFT_GL = {"MUJOCO_GL": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
           "EGL_PLATFORM": "surfaceless", "LP_NUM_THREADS": "2"}
CLIPS = {"model": ("sv72r", "servo", None), "measured": ("sv55db", "kp0.55_kv0_fr0.18_fl0_dp0.011", 0.59)}


def film_specs():
    """Per task and servo: that grid's best layout by the tracking score with its grasp, run on its own servo."""
    Lc = load(LAND.name)
    out = []
    for which, (k, plant, db) in CLIPS.items():
        for task in TASKS:
            b = Lc[k][task]["track"]
            c = max(b["best"], key=b["best"].get)
            out.append({"which": which, "grid": k, "plant": plant, "deadband": db, "task": task, "call": int(c),
                        "grasp": int(b["grasp"][c]), "gpu": b["best"][c], "xy": list(xy(c)),
                        "out": str(MEDIA / f"20261110-clip_stiffness_{which}_{task}.mp4")})
    return out


def cmd_films():
    """The clips through resguard, each in a process with its servo's environment, then the film: the model servo's
    picks in the top row, the measured servo's below."""
    import subprocess
    import time
    import hom_plant_page as PP
    specs = film_specs()
    for s_ in specs:
        if Path(s_["out"]).with_suffix(".json").exists():
            continue
        env = dict(os.environ, HOS_PLANT=s_["plant"], HOS_SCENES="logs/20261102-hom_servo_plant/scenes",
                   HOM_ROWS_TAG="_" + s_["grid"], NUMPY_MADVISE_HUGEPAGE="0", OMP_NUM_THREADS="1", **SOFT_GL)
        env.pop("HOS_DEADBAND_DEG", None)
        cmd = [str(Path.home() / ".claude/bin/resguard.sh"), "run", "--mem", "3G", "--cpu", "300", "--",
               str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/hom/hom_stiffness_page.py"), "clip", json.dumps(s_)]
        while True:
            rc = subprocess.call(cmd, env=env, cwd=ROOT)
            if rc != 3:
                break
            print(time.strftime("%T"), "refused by resguard; retry in 120 s", flush=True)
            time.sleep(120)
        if rc:
            raise SystemExit(f"clip {s_['out']} failed (exit {rc})")
    PP.compose([s_["out"] for s_ in specs], MEDIA / "20261110-film_stiffness.mp4", cols=3)
    FILMS.write_text(json.dumps([dict(s_, **json.loads(Path(s_["out"]).with_suffix(".json").read_text()))
                                 for s_ in specs], indent=1))
    print("wrote", FILMS.relative_to(ROOT))
    return 0


def cmd_clip(spec):
    """One CPU rollout filmed in this process on the servo of the spec (hom_torque.run: position servos, or the dead
    band on motors), the label on top."""
    import hom_contact_rig as HR
    import hom_torque as Q
    s_ = json.loads(spec)
    size = float(DEFAULT[s_["task"]])
    plant, wn = ("deadband", s_["deadband"]) if s_["deadband"] else ("position", Q.WN["servo"])
    row, frames = Q.run(s_["task"], size, s_["call"], s_["grasp"], 0, plant, wn, render=True)
    name = "measured servo" if s_["deadband"] else "model servo"
    label = f"{NAME[s_['task']]}, {name}: {lay(s_['xy'])}, tracking {row['track']:.2f} (GPU {s_['gpu']:.2f})"
    frames = [HR.annotate(f, label, size=13) for f in frames]
    HR.write_mp4(frames, s_["out"], fps=25)
    Path(s_["out"]).with_suffix(".json").write_text(json.dumps({"frames": len(frames), "track": row["track"],
                                                                "old": row["old"], "held": row.get("held")}))
    print("wrote", s_["out"], row["track"], flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- page

def f2(v):
    return "&#8211;" if v is None else f"{v:.2f}"


def lay(t):
    return f"({t[0]:.0f}, {t[1]:.0f})"


def grid_rollouts(k, task):
    p = ROWS / GRIDS[k][1] / f"{task}_d{DEFAULT[task]}.jsonl"
    return sum(1 for x in p.read_text().splitlines() if x.strip()) if p.exists() else 0


def sec_terms():
    terms = [
        ("Layout, lattice", "the SR2 hand's symmetric tripod in two coordinates: $x_{sep}$ from the thumb's mount to the "
         "index-middle pair's, and $y_{sep}$ from the index's mount to the middle's. The lattice has 625 layouts in "
         "steps of 5 mm, $x_{sep}$ from 40 to 160 mm and $y_{sep}$ from 50 to 170 mm. A layout is written "
         "($x_{sep}$, $y_{sep}$) in mm."),
        ("Tasks", "the HOM twist and roll of a 40 mm sphere and the press of a cylinder 25 mm across "
         f"(<a href=\"{TRENDS}\">trends page</a>). The pin-push is left out (owner, 2026-11-09: \"effective pin-push "
         "trajectories are effectively press trajectories\")."),
        ("Grid", "every layout of the lattice with every grasp of its grasp set, at seeds 0 to 2. A grasp set has up to "
         "16 grasps, solved by the pose toy with the plant's finger clearance. The grids ran on the GPU evaluator: "
         "MuJoCo Warp with the v4b fingertip's sphere pads at 2.5 MPa. In it only the pads and the small rigid "
         "spheres meet the object."),
        ("Best of 16", "a layout's highest mean score over seeds 0 to 2 among its grasps, by the score named; 0 where "
         "its pose is not reached."),
        ("Tracking score", f"the score of the <a href=\"{TRACK}\">tracking page</a>. It is the share of the "
         "commanded rotation that the object followed, times the share of the task on three loaded pads, and 0 when "
         "the object is dropped. Layouts are ranked by it (owner, 2026-11-09)."),
        ("Earlier score", "the landscapes' score before 2026-11-09 (<code>hom_landscape.rollout_score</code>), quoted "
         "beside the tracking score."),
        ("Model servo", "the SCS0009 plant of the earlier pages, a position actuator per joint. Its gain $k_p$ = "
         "0.72 N m/rad follows from the datasheet's 0.23 N m at 18 deg of error. The force limit is 0.18 N m, and the "
         "joint damping is 0.02 s times $k_p$, 0.0144 N m s/rad."),
        ("Measured servo", f"the SCS0009 as identified on the hand on 2026-11-06 (<a href=\"{SYSID}\">servo "
         "identification page</a>): $k_p$ = 0.55 N m/rad and a damping of 0.011 N m s/rad, then also its dead band. "
         "The servo applies no torque while the error is inside the band of 0.59 deg. Outside it, the torque is $k_p$ "
         "times the error. In the GPU evaluator the band is applied at every 1 ms physics step "
         "(<code>hom_gpu.servo_deadband</code>)."),
        ("Rerun", "a grid run again with the same plant, scenes and seeds. The GPU's sums are not deterministic, so a "
         "rerun differs from the first run by the evaluator's run-to-run spread."),
        ("Spearman's rho", "the rank correlation of two grids' best-of-16 scores over the layouts that both reach; 1 "
         "is the same order of layouts."),
        ("Best tenth kept", "the share of one grid's best tenth of layouts that is in the other grid's best tenth."),
    ]
    return "<dl>" + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"


def sec_grids(Lc):
    body = []
    for k, (label, src, ver, date) in GRIDS.items():
        n = sum(grid_rollouts(k, t) for t in TASKS)
        if not n:
            continue
        body.append([label, ver, date, f"<code>rows/{src}</code>", f"{n:,}"])
    return ("<p>Every grid ran the three tasks at the default objects on the same scenes, the 2.5 MPa cache of "
            f"2026-11-02 (<a href=\"{PLANTPAGE}\">plant and modulus page</a>). The grids differ only in the servo "
            "model and the simulator's version (Table 1). The grid of 11-02 at 0.72 N m/rad has no tracking fields, so it was run "
            "again on 11-10 with them.</p>"
            + table(["Grid", "MuJoCo", "Run", "Rows", "Rollouts"], body,
                    "The grids at the default objects: the twist and the roll of the 40 mm sphere and the press of the "
                    "25 mm cylinder, every layout and grasp at seeds 0-2.", num_from=4))


def sec_stiffness(Lc):
    body = []
    rows = (("sv72", "sv72r", "0.72 N m/rad, rerun"), ("sv72r", "sv55", "0.55 against 0.72 N m/rad"))
    for a, b, name in rows:
        for task in TASKS:
            r_t, r_r = pair(Lc, a, b, task, "track"), pair(Lc, a, b, task, "rollout")
            body.append([name if task == TASKS[0] else "", NAME[task],
                         f2(r_t["rho"]) if r_t else "&#8211;", f2(r_t["top"]) if r_t else "&#8211;",
                         f2(r_r["rho"]), f2(r_r["top"])])
    P_ = {t: pair(Lc, "sv72r", "sv55", t, "track") for t in TASKS}
    R_ = {t: pair(Lc, "sv72r", "sv55", t, "rollout") for t in TASKS}
    N_ = {t: pair(Lc, "sv72", "sv72r", t, "rollout") for t in TASKS}
    dist = [float(np.hypot(*np.subtract(P_[t]["best_a"][0], P_[t]["best_b"][0]))) for t in TASKS]
    body2 = [[NAME[t], f"{lay(P_[t]['best_a'][0])}: {P_[t]['best_a'][1]:.2f}", f2(P_[t]["best_a"][2]),
              f"{lay(P_[t]['best_b'][0])}: {P_[t]['best_b'][1]:.2f}", f2(P_[t]["best_b"][2]),
              f"{P_[t]['median_a']:.2f} / {P_[t]['median_b']:.2f}"] for t in TASKS]
    t = (f"<p>The measured stiffness reorders the layouts by more than the GPU's run-to-run spread (Table 2). By the "
         f"earlier score, the two runs of the 0.72 N m/rad grids rank the layouts with rho "
         f"{min(N_[x]['rho'] for x in TASKS):.2f} to {max(N_[x]['rho'] for x in TASKS):.2f} and keep "
         f"{min(N_[x]['top'] for x in TASKS):.2f} of each other's best tenth. At 0.55 N m/rad rho falls to "
         f"{min(R_[x]['rho'] for x in TASKS):.2f} to {max(R_[x]['rho'] for x in TASKS):.2f}, and {min(R_[x]['top'] for x in TASKS):.2f} to "
         f"{max(R_[x]['top'] for x in TASKS):.2f} of the best tenth is kept. By the tracking score rho is "
         f"{P_['twist']['rho']:.2f} on the twist, {P_['roll']['rho']:.2f} on the roll and {P_['press']['rho']:.2f} on the "
         "press.</p>"
         "<p>Each task's best layout changes. The best twist layout of the 0.72 N m/rad grid scores "
         f"{P_['twist']['best_a'][1]:.2f} there and {P_['twist']['best_a'][2]:.2f} at 0.55 N m/rad. Its best roll "
         f"layout falls from {P_['roll']['best_a'][1]:.2f} to {P_['roll']['best_a'][2]:.2f} (Table 3). The best layouts of "
         f"the 0.55 N m/rad grid lie {min(dist):.0f} to {max(dist):.0f} mm from those of the 0.72 N m/rad grid.</p>")
    t += table(["Pair of grids", "Task", "Spearman, tracking", "Best tenth kept, tracking", "Spearman, earlier",
                "Best tenth kept, earlier"], body,
               "The 0.72 N m/rad grids of 11-02 and 11-10 against each other, and the 0.55 against the 0.72 N m/rad grid "
               "of 11-10, over the layouts both reach. The grid of 11-02 has no tracking score.", num_from=2)
    t += table(["Task", "Best at 0.72: tracking score", "Its score at 0.55", "Best at 0.55: tracking score",
                "Its score at 0.72", "Median, 0.72 / 0.55"], body2,
               "Each grid's best layout by the tracking score and its score on the other grid; best of 16 at seeds 0-2.",
               num_from=2)
    t += img(MEDIA / "20261110-stiffness_maps.png", "Best-of-16 tracking score over the lattice on the grids of 11-10. "
             "Grey: the pose is not reached. Star: the best layout. <code>hom_stiffness_page.py maps</code>.")
    return t


def sec_deadband(Lc):
    """The measured servo's dead band on top of its stiffness, and the tracking score's rerun spread."""
    if not all(Lc.get("sv55db", {}).get(t, {}).get("track") for t in TASKS):
        return ""
    body = []
    for a, b, name in (("sv72r", "sv72r2", "0.72 N m/rad, rerun of 200 layouts"), ("sv55", "sv55db", "Dead band added at 0.55"),
                       ("sv72r", "sv55db", "Measured servo against 0.72 N m/rad")):
        for task in TASKS:
            r_t, r_r = pair(Lc, a, b, task, "track"), pair(Lc, a, b, task, "rollout")
            if not r_t:
                continue
            body.append([name if task == TASKS[0] else "", NAME[task], f"{r_t['n']}", f2(r_t["rho"]), f2(r_t["top"]),
                         f2(r_r["rho"]), f2(r_r["top"])])
    D_ = {t: pair(Lc, "sv55", "sv55db", t, "track") for t in TASKS}
    M_ = {t: pair(Lc, "sv72r", "sv55db", t, "track") for t in TASKS}
    Z_ = {t: pair(Lc, "sv72r", "sv72r2", t, "track") for t in TASKS}
    body2 = [[NAME[t], f"{lay(M_[t]['best_a'][0])}: {M_[t]['best_a'][1]:.2f}", f2(M_[t]["best_a"][2]),
              f"{lay(M_[t]['best_b'][0])}: {M_[t]['best_b'][1]:.2f}", f2(M_[t]["best_b"][2]),
              f"{np.hypot(*np.subtract(M_[t]['best_a'][0], M_[t]['best_b'][0])):.0f}"] for t in TASKS]
    loss = [M_[t]["best_a"][1] - M_[t]["best_a"][2] for t in TASKS]
    t = (f"<p>The servo's dead band was added to the 0.55 N m/rad plant, as the stiffness alone had moved the rankings "
         f"(Table 5). The tracking score's own spread was measured by running 200 layouts of the 0.72 N m/rad grid "
         f"again: rho is {min(Z_[x]['rho'] for x in TASKS if Z_[x]):.2f} to "
         f"{max(Z_[x]['rho'] for x in TASKS if Z_[x]):.2f} between the two runs. Against the 0.55 N m/rad grid without the band, rho is {D_['twist']['rho']:.2f}, "
         f"{D_['roll']['rho']:.2f} and {D_['press']['rho']:.2f} on the twist, the roll and the press. Against the model "
         f"servo it is {M_['twist']['rho']:.2f}, {M_['roll']['rho']:.2f} and {M_['press']['rho']:.2f}.</p>"
         "<p>With the measured servo, the best layouts score "
         f"{M_['twist']['best_b'][1]:.2f} on the twist, {M_['roll']['best_b'][1]:.2f} on the roll and "
         f"{M_['press']['best_b'][1]:.2f} on the press (Table 6). With the model servo they score "
         f"{M_['twist']['best_a'][1]:.2f}, {M_['roll']['best_a'][1]:.2f} and {M_['press']['best_a'][1]:.2f}. On the "
         f"measured servo, the model servo's best layouts score {min(loss):.2f} to {max(loss):.2f} lower; the roll's "
         f"scores {M_['roll']['best_a'][2]:.2f}.</p>")
    t += table(["Pair of grids", "Task", "Layouts", "Spearman, tracking", "Best tenth kept, tracking",
                "Spearman, earlier", "Best tenth kept, earlier"], body,
               "The tracking score's rerun spread on 200 layouts, the dead band against the stiffness alone, and the "
               "measured servo against the model servo.", num_from=2)
    t += table(["Task", "Best with the model servo: tracking score", "Its score with the measured servo",
                "Best with the measured servo: tracking score", "Its score with the model servo", "Distance (mm)"],
               body2, "Each servo's best layout by the tracking score and its score with the other servo; best of 16 "
               "at seeds 0-2. The measured servo is 0.55 N m/rad with the 0.59 deg dead band.", num_from=2)
    return t


def sec_scores(Lc):
    body = [[GRIDS[k][0], *[f2(scores_rho(Lc, k, t)) for t in TASKS]] for k in GRIDS if k != "sv72r2"
            if k in Lc and all(scores_rho(Lc, k, t) is not None for t in TASKS)]
    v = [scores_rho(Lc, k, t) for k in GRIDS if k in Lc and k != "sv72r2" for t in TASKS if scores_rho(Lc, k, t) is not None]
    return (f"<p>On every grid the tracking score ranks the layouts differently from the earlier score: Spearman's rho "
            f"between the two is {min(v):.2f} to {max(v):.2f} (Table 4). The tracking page set 0.9 as the level above "
            "which a task's earlier landscapes would stand; no task reaches it. The roll differs most: a sphere held "
            "still scored 0.50 on its earlier score.</p>"
            + table(["Grid", *[NAME[t] for t in TASKS]], body,
                    "Spearman's rho between a grid's best-of-16 tracking scores and its best-of-16 earlier scores.",
                    num_from=1))


def run_ms(k, task):
    """A grid's GPU time per rollout in ms: its batches' run time over their worlds (rows' _timing.jsonl)."""
    p = ROWS / GRIDS[k][1] / f"{task}_d{DEFAULT[task]}_timing.jsonl"
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return 1000.0 * sum(r["run_s"] for r in rs) / sum(r["worlds"] for r in rs) if rs else None


def sec_mj4(Lc):
    """The measured-servo grids on MuJoCo 4.0 against MuJoCo 3.6 (post-reset step 2)."""
    if not all(Lc.get("mj4", {}).get(t, {}).get("track") for t in TASKS):
        return ""
    body, R_, Q_ = [], {}, {}
    for task in TASKS:
        R_[task], Q_[task] = pair(Lc, "sv55db", "mj4", task, "track"), pair(Lc, "sv55db", "mj4", task, "rollout")
        r_t, r_r = R_[task], Q_[task]
        a, b = run_ms("sv55db", task), run_ms("mj4", task)
        body.append([NAME[task], f"{r_t['n']}", f2(r_t["rho"]), f2(r_t["top"]), f2(r_r["rho"]), f2(r_r["top"]),
                     f"{lay(r_t['best_a'][0])}: {r_t['best_a'][1]:.2f}", f2(r_t["best_a"][2]),
                     f"{lay(r_t['best_b'][0])}: {r_t['best_b'][1]:.2f}", f2(r_t["best_b"][2]),
                     "&#8211;" if a is None or b is None else f"{a:.0f} / {b:.0f}"])
    sp = [run_ms("sv55db", t) / run_ms("mj4", t) for t in TASKS if run_ms("sv55db", t) and run_ms("mj4", t)]
    t = (f"<p>The measured-servo grids ran again on MuJoCo 4.0 and MuJoCo Warp 4.0 (<a href=\"{MJ4}\">MuJoCo 4.0 port "
         "page</a>), with the same scenes, plant, seeds and batches. Ranked by the tracking score, Spearman's rho "
         f"against MuJoCo 3.6 is {R_['twist']['rho']:.2f} on the twist, {R_['roll']['rho']:.2f} on the roll and "
         f"{R_['press']['rho']:.2f} on the press (Table 8). By the earlier score it is {Q_['twist']['rho']:.2f}, "
         f"{Q_['roll']['rho']:.2f} and {Q_['press']['rho']:.2f}.")
    keep = [x for x in TASKS if R_[x]["best_a"][0] == R_[x]["best_b"][0]]
    if keep:
        t += (" " + ("Every task keeps" if len(keep) == len(TASKS) else "The " + " and the ".join(keep) + " keep")
              + " its best layout by the tracking score, at " + ", ".join(lay(R_[x]["best_a"][0]) for x in keep[:-1])
              + (" and " if len(keep) > 1 else "") + f"{lay(R_[keep[-1]]['best_a'][0])} mm.")
    M_ = [pair(Lc, "sv72r", "sv55db", x, "track")["rho"] for x in TASKS]
    Z_ = [pair(Lc, "sv72r", "sv72r2", x, "track")["rho"] for x in TASKS]
    t += (f" The version moves the rankings by more than the rerun spread on MuJoCo 3.6, {min(Z_):.2f} to "
          f"{max(Z_):.2f}. It moves them by less than the measured servo did, {min(M_):.2f} to {max(M_):.2f} "
          "(Section SEC[deadband]).</p>")
    if sp:
        t += (f"<p>A rollout took {min(sp):.2f} to {max(sp):.2f} times as long on MuJoCo 3.6 as on 4.0, on the same "
              "GPU and batches.")
        W = load(WARN.name)
        if W:
            ls, it = [W[t_]["linesearch"] for t_ in TASKS], [W[t_]["solver"] for t_ in TASKS]
            t += (f" MuJoCo Warp 4.0 printed {min(ls) / 1e6:.1f} to {max(ls) / 1e6:.1f} million warnings per grid that "
                  f"the line search reached its limit of 10 iterations. It printed {min(it):,} to {max(it):,} that the "
                  "solver reached its 20 iterations. Both limits are the evaluator's settings on both versions; MuJoCo Warp "
                  "3.6 prints no such warning.")
        t += "</p>"
    t += img(MEDIA / "20261110-mj4_maps.png", "Best-of-16 tracking score over the lattice with the measured servo on "
             "MuJoCo 3.6 (top) and 4.0 (bottom). Grey: the pose is not reached. Star: the best layout.")
    t += table(["Task", "Layouts", "Spearman, tracking", "Best tenth kept, tracking", "Spearman, earlier",
                "Best tenth kept, earlier", "Best on 3.6: tracking score", "Its score on 4.0", "Best on 4.0: tracking score",
                "Its score on 3.6", "GPU time per rollout, 3.6 / 4.0 (ms)"], body,
               "The measured servo (0.55 N m/rad, 0.59 deg dead band) at 2.5 MPa on MuJoCo 3.6 and on MuJoCo 4.0: rank "
               "correlation of the best-of-16 scores over the layouts both reach, and each version's best layout by "
               "the tracking score with its score on the other.", num_from=1)
    return t


EA_SRC = {"model": ("ea_live_sv25", "0.72 N m/rad, MuJoCo 3.6"),
          "mj4": ("ea_live_sv55db_mj4", "0.55 N m/rad and dead band, MuJoCo 4.0")}
EA_SIZES = {"twist": (25, 30, 35, 45, 50, 56, 63, 71, 80), "roll": (25, 30, 35, 45, 50, 56, 63, 71, 80),
            "press": (8, 10, 13, 16, 20, 32, 40, 50)}


def ea_outcome(which, task, d, score):
    """The best layout of one live EA run by the score, its score, and the centroid of the layouts within 0.05."""
    import hom_plant_transfer as TR
    p = ROWS / EA_SRC[which][0] / f"{task}_d{d:g}_s0.jsonl"
    g = TR.grasp_means(task, p, score)
    if not g:
        return None
    best = {c: max(v.values()) for c, v in g.items()}
    c = max(best, key=best.get)
    near = np.array([xy(k) for k, v in best.items() if v >= best[c] - 0.05], float)
    return {"xy": xy(c), "best": best[c], "centroid": near.mean(0).tolist(), "layouts": len(best)}


def slope(which, task, score):
    """mm of x_sep per mm of diameter: the least-squares slope of the near set's centroid over the sizes."""
    pts = [(d, o["centroid"][0]) for d in EA_SIZES[task] if (o := ea_outcome(which, task, d, score))]
    if len(pts) < 3:
        return None
    d, x = np.array(pts).T
    return float(np.polyfit(d, x, 1)[0])


def sec_ea4():
    """The size study's live EA on the measured servo and MuJoCo 4.0 against the stored runs (post-reset step 2)."""
    if not (ROWS / EA_SRC["mj4"][0]).exists():
        return ""
    body = []
    for task in TASKS:
        for d in EA_SIZES[task]:
            a, b = ea_outcome("model", task, d, "rollout"), ea_outcome("mj4", task, d, "rollout")
            c = ea_outcome("mj4", task, d, "track")
            if not b:
                continue
            body.append([NAME[task] if d == EA_SIZES[task][0] else "", f"{d}",
                         f"{lay(a['xy'])}: {a['best']:.2f}" if a else "&#8211;", f"{lay(b['xy'])}: {b['best']:.2f}",
                         f"{lay(c['xy'])}: {c['best']:.2f}" if c else "&#8211;",
                         f"{np.hypot(*np.subtract(a['xy'], b['xy'])):.0f}" if a else "&#8211;"])
    sl = {t: (slope("model", t, "rollout"), slope("mj4", t, "rollout")) for t in TASKS}
    t = ("<p>The size study's live EA ran at the same object sizes on the measured servo and MuJoCo 4.0 (Table 9). As "
         "in the size study, each run had a population of 64, about 6,000 rollouts, seed 0 and the earlier score as its "
         "fitness. The stored runs of the size "
         "study used the model servo on MuJoCo 3.6, so the two columns differ in the servo and in the simulator. "
         "Section SEC[mj4] separates the simulator's part on the default objects.</p>")
    ok = [x for x in TASKS if all(v is not None for v in sl[x])]
    if ok:
        t += ("<p>The centroid of the layouts within 0.05 of the best moves with the object's diameter by "
              + "; ".join(f"{sl[x][0]:.2f} mm/mm on the model servo and {sl[x][1]:.2f} mm/mm on the measured servo "
                          f"for the {x}" for x in ok) + ".</p>")
    t += table(["Task", "Diameter (mm)", "Best, 0.72 N m/rad, MuJoCo 3.6: earlier score",
                "Best, measured servo, MuJoCo 4.0: earlier score", "Best, measured servo, MuJoCo 4.0: tracking score",
                "Distance between the first two (mm)"], body,
               "The live EA at each object size: best layout and score by the earlier score on the stored runs and on the "
               "measured servo with MuJoCo 4.0, and by the tracking score on the latter. Every run at 2.5 MPa.",
               num_from=1)
    return t


def sec_films():
    F = load(FILMS.name)
    if not F:
        return ""
    by = {(f["which"], f["task"]): f for f in F}
    body = [[NAME[t], *[f"{lay(by[(w, t)]['xy'])}, grasp {by[(w, t)]['grasp']}: {by[(w, t)]['track']:.2f} "
                        f"({by[(w, t)]['gpu']:.2f})" for w in CLIPS]] for t in TASKS]
    return ("<p>Each servo's best layout per task was run again on the CPU plant at seed 0 and filmed (Table 7). The "
            "CPU plant has every contact; the GPU evaluator lets the pads and the small rigid spheres meet the object "
            "only.</p>"
            + table(["Task", "Model servo (0.72 N m/rad)", "Measured servo (0.55 N m/rad, dead band)"], body,
                    "The filmed layouts: tracking score of the CPU rollout at seed 0, and in brackets the best-of-16 "
                    "mean over seeds 0-2 on the GPU.", num_from=1)
            + film(MEDIA / "20261110-film_stiffness.mp4", "Each servo's best layout by the tracking score, seed 0, CPU "
                   "plant: the model servo in the top row, the measured servo with its dead band below."))


def sec_limits():
    return ("<p>Each task ran at one object size, the twist and the roll of a 40 mm sphere and the press of a 25 mm "
            "cylinder, at seeds 0 to 2. The dead band and the stiffness measured on the hand on 2026-11-06 were "
            "applied to all nine joints alike. The servos' friction, 23 to 27 mN m at the yaw, was left out. "
            "The rerun spread of the tracking score was measured on 200 of the 625 layouts, at 0.72 N m/rad only.</p>")


def sec_future():
    return ("<ul>"
            "<li>Add the yaw servos' measured friction to the plant (the <code>fl</code> field of "
            "<code>HOS_PLANT</code>) and run the three grids again. If Spearman's rho of the tracking score against the "
            "measured-servo grid stays above 0.95 on every task, the friction can stay out of the plant.</li>"
            "<li>Run the size study's live EA on the measured servo with the tracking score as its fitness "
            "(<code>hom_ea.py run --score track</code> with <code>HOS_DEADBAND_DEG=0.59</code>). If its best layouts "
            "lie within the lattice's 5 mm of those found with the model servo, the design page's size slopes "
            "stand.</li>"
            "</ul>")


def sec_files():
    return ("<ul>"
            "<li><code>scripts/hom/hom_gpu.py</code>: the dead band (<code>HOS_DEADBAND_DEG</code>, kernel "
            "<code>servo_deadband</code>); rows name <code>mujoco</code> and <code>mujoco_warp</code></li>"
            "<li><code>scripts/hom/hom_plant_transfer.py --set stiffness --score track</code></li>"
            "<li>Builder <code>scripts/hom/hom_stiffness_page.py</code>: <code>maps</code> (the cache "
            "<code>rows/20261110-stiffness_landscapes.json</code> and <code>media/20261110-stiffness_maps.png</code>), "
            "<code>numbers</code>, <code>films</code> (<code>media/20261110-film_stiffness.mp4</code>, "
            "<code>rows/20261110-stiffness_films.json</code>), and the page</li>"
            "<li>Rows, on the workstation only (over 2 MB each): <code>rows/grid2dff_v4b_sv25r/</code>, "
            "<code>rows/grid2dff_v4b_sv25r2/</code>, <code>rows/grid2dff_v4b_sv55/</code>, "
            "<code>rows/grid2dff_v4b_sv55db/</code></li>"
            "<li>Queues: <code>logs/20261110-hom_post_reset/gpu_queue_step1.sh</code>, "
            "<code>gpu_queue_step1b2.sh</code></li>"
            "</ul>")


def lede(Lc):
    Z_ = [pair(Lc, "sv72r", "sv72r2", t, "track")["rho"] for t in TASKS]
    S_ = {t: pair(Lc, "sv72r", "sv55", t, "track") for t in TASKS}
    D_ = {t: pair(Lc, "sv55", "sv55db", t, "track") for t in TASKS}
    M_ = {t: pair(Lc, "sv72r", "sv55db", t, "track") for t in TASKS}
    loss = [M_[t]["best_a"][1] - M_[t]["best_a"][2] for t in TASKS]
    text = (
        "The SCS0009's measured stiffness and dead band reorder the HOM landscapes of the SR2 hand by more than the GPU "
        "evaluator's rerun spread. Between two runs of the 0.72 N m/rad grid, Spearman's rho of the tracking score is "
        f"{min(Z_):.2f} to {max(Z_):.2f} over 200 layouts. Between the measured 0.55 and the model's 0.72 N m/rad, "
        f"rho is {S_['twist']['rho']:.2f} on the twist, {S_['roll']['rho']:.2f} on the roll and "
        f"{S_['press']['rho']:.2f} on the press. With the servo's 0.59 deg dead band added, rho against 0.55 N m/rad "
        f"alone is {D_['twist']['rho']:.2f}, {D_['roll']['rho']:.2f} and {D_['press']['rho']:.2f}. On the measured "
        f"servo, the model servo's best layouts score {min(loss):.2f} to {max(loss):.2f} lower. The measured servo's "
        "own best layouts reach the model's best scores, "
        f"{M_['twist']['best_b'][1]:.2f}, {M_['roll']['best_b'][1]:.2f} and {M_['press']['best_b'][1]:.2f}. The "
        "next grids run on the measured servo with its dead band.")
    if all(Lc.get("mj4", {}).get(t, {}).get("track") for t in TASKS):
        R4 = [pair(Lc, "sv55db", "mj4", t, "track") for t in TASKS]
        same = all(r["best_a"][0] == r["best_b"][0] for r in R4)
        text = text.replace(" The next grids run on the measured servo with its dead band.", "")
        text += (f" On MuJoCo 4.0 the measured servo's landscapes keep rho {min(r['rho'] for r in R4):.2f} to "
                 f"{max(r['rho'] for r in R4):.2f} against 3.6" + (", with the same best layouts." if same else "."))
    summary = (f"The SCS0009's measured 0.55 N m/rad and 0.59 deg dead band reorder the HOM landscapes (tracking-score "
               f"rho {min(m['rho'] for m in M_.values()):.2f} to {max(m['rho'] for m in M_.values()):.2f} against the "
               f"0.72 N m/rad model, rerun spread {min(Z_):.2f} to {max(Z_):.2f}); the model's best layouts lose "
               f"{min(loss):.2f} to {max(loss):.2f} of their score on the measured servo.")
    return text, summary


def main():
    Lc = load(LAND.name)
    mj4 = all(Lc.get("mj4", {}).get(t, {}).get("track") for t in TASKS)
    sections = [("terms", "Terms and metrics", sec_terms()),
                ("grids", "Grids at the default objects", sec_grids(Lc)),
                ("stiffness", "Landscapes at the measured stiffness against the model's", sec_stiffness(Lc)),
                ("scores", "Tracking score against the earlier score on each grid", sec_scores(Lc)),
                ("deadband", "Landscapes with the servo's dead band", sec_deadband(Lc)),
                ("films", "Films of each servo's best layouts", sec_films()),
                ("mj4", "Landscapes of the measured servo on MuJoCo 4.0", sec_mj4(Lc)),
                ("ea4", "Live EA across object sizes on the measured servo and MuJoCo 4.0", sec_ea4()),
                ("limits", "Limitations", sec_limits()),
                ("future", "Future work", sec_future()),
                ("files", "Files", sec_files())]
    sections = [x for x in sections if x[2]]
    num = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sections = [(sid, hd, re.sub(r"SEC\[(\w+)\]", lambda m_: str(num[m_.group(1)]), b)) for sid, hd, b in sections]
    lede_t, summary = lede(Lc)
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N m|N\b|s\b)", "\\1&nbsp;\\2", s_))
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(sections, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{unit(b)}</section>'
                     for k, (sid, hd, b) in enumerate(sections, 1))
    byline = ("GPU grids on the SR2 hand's CAD model with the v4b fingertip at 2.5 MPa, 2026-11-10. Follows <a href=\""
              + TRACK + "\">the tracking score page</a> and <a href=\"" + PLANTPAGE + "\">the plant and modulus "
              "page</a>. Builder: <code>scripts/hom/hom_stiffness_page.py</code>.")
    h1 = ("HOM landscapes of the SR2 hand on the measured SCS0009 servo" + (" and on MuJoCo 4.0" if mj4 else ""))
    page = P.TEMPLATE.read_text()
    for k, v in {"TITLE": "HOM servo stiffness" + (" and MuJoCo 4" if mj4 else ""), "H1": h1,
                 "EYEBROW": "HOM layout study, 2026-11-10", "LEDE": unit(lede_t), "BYLINE": byline, "TOC": toc,
                 "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS + texsvg.BLOCK_CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    n = []
    page = P.render(page, n)
    page = docs_index.tag(page, topic="design", summary=summary, date="2026-11-10")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {n[0]} formulas", flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["maps"]:
        sys.exit(cmd_maps())
    if sys.argv[1:2] == ["numbers"]:
        sys.exit(cmd_numbers())
    if sys.argv[1:2] == ["warnings"]:
        sys.exit(cmd_warnings())
    if sys.argv[1:2] == ["films"]:
        sys.exit(cmd_films())
    if sys.argv[1:2] == ["clip"]:
        sys.exit(cmd_clip(sys.argv[2]))
    sys.exit(main())
